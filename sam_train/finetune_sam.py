# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Optional

from sam_train.sam.image_encoder import ImageEncoderViT
from sam_train.sam.lora_image_encoder import LoRA_Encoder
from sam_train.sam.mask_decoder import MaskDecoder
from sam_train.sam.prompt_encoder import PromptEncoder

from sam_train.utils.amg import build_all_layer_point_grids, batched_mask_to_box, bs_batch_iterator, calculate_stability_score, batch_iterator
import numpy as np
from detectron2.layers import cat, batched_nms
from sam_train.utils.transforms import ResizeLongestSide
from sam_train.modeling.sam_criterion import SetCriterion
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess


    
class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: 'ImageEncoderViT|LoRA_Encoder',
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,

        metadata,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 1,
        points_per_side: Optional[int] = 64, #32, #32,
        pred_iou_thresh: float = 0, #0.88,


        stability_score_thresh: float = 0, #0.95, #0.95,
        stability_score_offset: float = 1.0,

        test_nms_thresh: float = 0.9, #0.7,

        #train
        num_queries: int = 100,
        num_classes: int = 80,
        train_num_points: int = 12544,
        oversample_ratio: int = 3.0,
        importance_sample_ratio: int =  0.75,
        sem_seg_postprocessing_before_inference: bool = False,
        deep_supervision: bool = False,

        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        dice_weight: float = 1.0,
        mask_weight: float = 20.0,
        iou_weight:float = 1.0,



        # inference
        semantic_on: bool = False,
        panoptic_on: bool = False,
        instance_on: bool = True,
        test_topk_per_image: int = 100,

        # panseg
        object_mask_threshold: float = 0.8,
        overlap_threshold: float = 0.8,
        topk_candidates_test: int = 100,
        score_thresh_test: float =  0.05,

        size_divisibility: int = 1024, # 32

    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        # for k in self.image_encoder.parameters():
        #     k.requires_grad = False

        for k in self.prompt_encoder.parameters():
            k.requires_grad = False

        for n, k in self.mask_decoder.named_parameters():
            # if 'output_upscaling' not in n and 'output_hypernetworks_mlps.0' not in n:
            #     k.requires_grad = False
            k.requires_grad = False
        
        self.mask_decoder.mask_tokens.weight.requires_grad = True


        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.image_encoder.img_size = self.image_encoder.lora_vit.img_size

        self.transform = ResizeLongestSide(h = 0, w = 0, target_length=self.image_encoder.img_size)


        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.sem_seg_postprocess_before_inference = (
                sem_seg_postprocessing_before_inference
                or panoptic_on
                or instance_on
            )

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        #暂时放在这里
        self.num_classes = num_classes
        
      
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold
        self.num_queries = num_queries
        self.metadata = metadata

        self.topk_candidates_test = topk_candidates_test
        self.test_nms_thresh = test_nms_thresh
        self.score_thresh_test = score_thresh_test

        if points_per_side is not None:
              self.point_grids = build_all_layer_point_grids(
                  points_per_side,
                  crop_n_layers,
                  crop_n_points_downscale_factor,
              )
        
        self.size_divisibility = size_divisibility
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset

        # weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        # losses = ["labels", "masks"]

        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight, "loss_iou":iou_weight}
        losses = ["masks", "ious"]
        
        self.criterion = SetCriterion(
            num_classes,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=train_num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
        )
        self.image_encoder.img_size

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, 
                batched_input: List[Dict[str, Any]],
                ):
        processed_results = self.sam_forward(batched_input, multimask_output=False)
        return processed_results


    def preprocess_state_dict(self, state_dict):
        lis_keys = list(state_dict.keys())
        lis_model_keys = [n for n, _  in self.named_parameters()]
        if 'image_encoder.lora_vit.pos_embed' in lis_keys:
            pass
        else:
            # 处理模型权重 增加 lora_vit
            for k in lis_keys:
                if 'image_encoder' in k:
                    state_dict[k.replace('image_encoder', 'image_encoder.lora_vit')] = state_dict.pop(k)
                    k = k.replace('image_encoder', 'image_encoder.lora_vit')

                if 'attn.qkv' in k and k not in lis_model_keys:
                    state_dict[k.replace('qkv', 'qkv.qkv')] = state_dict.pop(k)
        
        return state_dict
    
    # @torch.no_grad()
    def sam_forward(
        self,
        batched_inputs: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.training:
            #训练阶段 预处理
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            # images = [self.preprocess(x) for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility) 
        else:
            # inference 和sam保持一致
            images = [self.preprocess(x) for x in images]
            images = ImageList.from_tensors(images, 1024)
            images.image_sizes = [(x["image"].size()[-2], x["image"].size()[-1]) for x in batched_inputs]


        image_embeddings = self.image_encoder(images.tensor)  #bs, 256, 64, 64

        if not self.training:
            ##评估的时候，自动产生点
            # 实现sam的默认设置 仅对原图crop一次（完整原图） 仅含一层
            crop_layer_idx = 0
            cropped_im_size = [(x['height'], x["width"]) for x in batched_inputs]
            # Get points for this crop
            points_scale = np.array(cropped_im_size)[:, None, ::-1]  # bs, 1, 2
            points_for_image = self.point_grids[crop_layer_idx][None,] * points_scale  #batchsize, 1024, 2

            # 尝试实现并行计算
            # 所有的模型都需要修改 以接受多一个维度 batchsize
            # all batchsize & all points
            transformed_points = self.transform.apply_multi_coords(points_for_image, cropped_im_size)
            in_points = torch.as_tensor(transformed_points, device=image_embeddings.device)  #transform后的点 batchsize, 100, 2
            in_labels = torch.ones(in_points.shape[:2], dtype=torch.int, device=in_points.device) #前景点 batchsize, 100

            points = (in_points[:, :, None, :], in_labels[:, :, None])

            # Embed prompts  point embedding 已实现
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
        
            output_masks = []
            output_iou = []
            # 效仿sam 叠成batch计算 _process_batch
            batch = 64
            for (sparse_embedding, dense_embedding) in bs_batch_iterator(batch, sparse_embeddings, dense_embeddings):
                low_res_masks, iou_predictions, mask_tokens_out = self.mask_decoder(
                    image_embeddings=image_embeddings,   #bs, embed_dim, h, w
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embedding,
                    dense_prompt_embeddings=dense_embedding,
                    multimask_output=multimask_output,
                )
                output_masks.append(low_res_masks)
                output_iou.append(iou_predictions)


            low_res_masks = torch.cat(output_masks, dim=1)
            iou_predictions = torch.cat(output_iou, dim=1)

            low_res_masks = low_res_masks.flatten(1, 2)      #bs, num_points * (1 / 3), 256, 256
            iou_predictions = iou_predictions.flatten(1, 2)  # bs, num_points * (1 / 3)

            self.num_queries = mask_tokens_out.size()[1]


            mask_pred_results = low_res_masks
            # 将predicted_iou替换分类置信度
            mask_cls_results = iou_predictions

            # 后处理
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs,  images.image_sizes
            ):
                # 逐图后处理
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                
                masks = []
                iou_preds = []

                # _process_batch
                batch = 64
                for (batch_mask_pred_result, batch_mask_cls_result) in batch_iterator(batch, mask_pred_result, mask_cls_result):
                    # # 原图尺寸 mask
                    mask = self.postprocess_masks(batch_mask_pred_result, input_size=image_size, original_size=(height, width))  
                    # 后处理
                    # Filter by predicted IoU
                    if self.pred_iou_thresh > 0.0:
                        keep_mask = batch_mask_cls_result > self.pred_iou_thresh
                        mask = mask[keep_mask]
                        batch_mask_cls_result = batch_mask_cls_result[keep_mask]

                    # Calculate stability score
                    stability_score = calculate_stability_score(
                        mask, self.mask_threshold, self.stability_score_offset
                    )
                    if self.stability_score_thresh > 0.0:
                        keep_mask = stability_score >= self.stability_score_thresh
                        mask = mask[keep_mask]
                        batch_mask_cls_result = batch_mask_cls_result[keep_mask]


                    masks.append(mask > self.mask_threshold)  # bool
                    iou_preds.append(batch_mask_cls_result)

                mask_pred_result = torch.cat(masks, dim=0)  # num_masks, h, w
                # 将predicted_iou替换分类置信度
                mask_cls_result = torch.cat(iou_preds, dim=0) # num_masks, iou_scores

                assert self.sem_seg_postprocess_before_inference
               
                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results 
        else:
            # 训练代码
           
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            
            #实例间等概率采点
            #每个实例采点的个数

            num_ins_samples = 1
            max_ins_num = max([len(tg['labels']) for tg in targets])
            sample_points = -torch.ones(len(targets), num_ins_samples * max_ins_num, 3).type_as(images.tensor)  # bs, num_ins_samples, 3
           
            for bs_i, t in enumerate(targets):
                mask_sample_points = torch.nonzero(t['masks'])  #  采样点
                for i in range(len(t['labels'])):
                    ins_sample_points = mask_sample_points[mask_sample_points[:, 0] == i]
                    ins_sample = ins_sample_points[torch.randperm(len(ins_sample_points))][:num_ins_samples]
                    sample_points[bs_i, i*num_ins_samples:(i+1)*num_ins_samples] = ins_sample
                
                if len(t['labels']) == 0:
                    # print('error!')
                    # 补充 label 以及 masks 避免访问越界
                    t['labels'] = -torch.ones(1).type_as(sample_points).long()  # 1
                    t['masks'] = torch.zeros((1, self.image_encoder.img_size, self.image_encoder.img_size), dtype=torch.uint8, device=sample_points.device)    # 1, 1024, 1024
                

            sample_points = sample_points[:, :, [0, 2, 1]]
            #########################################################################################            
            # sample_point: bs, num_prompts, 3 (indices, x, y)

            point_coords = sample_points[:, :, None, 1:]         # bs, num_prompts, num_points_per_prompt, 2  

            
            # 记录point对应的GT索引
            for i in range(len(targets)):
                targets[i]['indices'] = sample_points[i, :, 0].long()

            # point_coords: bs, num_prompts, num_points_per_prompt, 2
            # point_labels: bs, num_prompts, num_points_per_prompt  (1 for pos; 0 for neg)

            ######################################################################################
            # ori_img_size = [(x["height"], x["width"]) for x in batched_inputs]  for test
            # point_coords = torch.from_numpy(np.array([[500, 375], [200, 300]])[None, None]).repeat(len(image_embeddings), 16, 1, 1).numpy() # 原图位置坐标  for test
            ######################################################################################

            point_labels = np.ones((*point_coords.shape[:-1], ))


            # Transform input prompts
            # point 坐标 ---> point embedding
            coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
            if point_coords is not None:
                assert (
                    point_labels is not None
                ), "point_labels must be supplied if point_coords is supplied."
                # point_coords = self.transform.apply_multi_coords(point_coords, ori_img_size)    # 无需修正坐标 输入即为增强后的原图位置坐标  for test
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            
            ##########未实现
            # if box is not None:
            #     box = self.transform.apply_boxes(box, self.original_size)
            #     box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            #     box_torch = box_torch[None, :]
            # if mask_input is not None:
            #     mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            #     mask_input_torch = mask_input_torch[None, :, :, :]
            ######################################
            points = (coords_torch, labels_torch)

            # Embed prompts  point embedding 已实现
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )

            # mask decoder
            low_res_masks, iou_predictions, mask_tokens_out = self.mask_decoder(
                    image_embeddings=image_embeddings,   #bs, embed_dim, h, w
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
            )

            # mask prediction
            outputs = {}
            outputs['pred_masks'] = low_res_masks.squeeze(2)   # bs, num_prompts, h,w 
            outputs['pred_iou'] = iou_predictions.squeeze(2)   # bs, num_prompts

            # loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
    
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets
    

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks.unsqueeze(0),
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)[0]
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    
    ######未实现
    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

     ######未实现
    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info


    #修改为类别无关的评估方式
    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        ###后处理NMS操作
        boxes = batched_mask_to_box(mask_pred)
        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            boxes.float(),
            mask_cls,
            torch.zeros(len(boxes)).to(mask_cls.device),  # categories
            iou_threshold=self.test_nms_thresh,
        )

        pred_boxes = boxes[keep_by_nms[:self.test_topk_per_image]]
        mask_pred = mask_pred[keep_by_nms[:self.test_topk_per_image]]
        pred_scores = mask_cls[keep_by_nms[:self.test_topk_per_image]]
        labels_per_image = torch.ones(len(pred_boxes))


        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        # result.pred_masks = (mask_pred > 0).float()
        result.pred_masks = mask_pred 
        result.pred_boxes = Boxes(pred_boxes) #Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        # mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = pred_scores  #scores_per_image #* mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
