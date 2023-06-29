import copy
import logging
import numpy as np
from typing import List, Union, Optional
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances
from panopticapi.utils import rgb2id
from detectron2.config import configurable
import pycocotools.mask as mask_util
from detectron2.structures import Boxes, BoxMode
from detectron2.data.detection_utils import transform_keypoint_annotations
from pycocotools import mask as coco_mask


class COCOPanopticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        segmentation_format: str = "L",
        caption_key: str = "coco_captions",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.augmentations = T.AugmentationList(augmentations)
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}] Full TransformGens used in training: {self.augmentations}"
        )

        self.img_format = image_format
        self.seg_format = segmentation_format
        self.cap_key = caption_key
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), format=self.seg_format
            )
            if self.seg_format == "L":
                sem_seg_gt = sem_seg_gt.squeeze(2)
        else:
            sem_seg_gt = None

        # image, transforms = T.apply_augmentations(self.augmentations, image)
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
        segments_info = dataset_dict["segments_info"]

        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        pan_seg_gt = rgb2id(pan_seg_gt)
        dataset_dict["pan_seg_gt"] = torch.from_numpy(np.ascontiguousarray(pan_seg_gt))

        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()

        if self.cap_key in dataset_dict:
            dataset_dict["captions"] = dataset_dict.pop(self.cap_key)

        dataset_dict["instances"] = instances

        return dataset_dict


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        
        # Need to filter empty instances first (due to augmentation)
        instances = utils.filter_empty_instances(instances)
        # Generate masks from polygon
        h, w = instances.image_size
        # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
        if hasattr(instances, 'gt_masks'):
            gt_masks = instances.gt_masks
            if isinstance(gt_masks, BitMasks):
                # pass
                #dtype=torch.uint8
                gt_masks = gt_masks.tensor.to(dtype=torch.uint8)
            else:
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks

        dataset_dict["instances"] = instances

    # def transform_instance_annotations(
    #     self, annotation, transforms, image_size, *, keypoint_hflip_indices=None
    # ):
    #     """
    #     Apply transforms to box, segmentation and keypoints annotations of a single instance.

    #     It will use `transforms.apply_box` for the box, and
    #     `transforms.apply_coords` for segmentation polygons & keypoints.
    #     If you need anything more specially designed for each data structure,
    #     you'll need to implement your own version of this function or the transforms.

    #     Args:
    #         annotation (dict): dict of instance annotations for a single instance.
    #             It will be modified in-place.
    #         transforms (TransformList or list[Transform]):
    #         image_size (tuple): the height, width of the transformed image
    #         keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    #     Returns:
    #         dict:
    #             the same input dict with fields "bbox", "segmentation", "keypoints"
    #             transformed according to `transforms`.
    #             The "bbox_mode" field will be set to XYXY_ABS.
    #     """
    #     if isinstance(transforms, (tuple, list)):
    #         transforms = T.TransformList(transforms)
    #     # bbox is 1d (per-instance bounding box)
    #     bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    #     # clip transformed bbox to image size
    #     bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    #     annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    #     annotation["bbox_mode"] = BoxMode.XYXY_ABS

    #     if "segmentation" in annotation:
    #         # each instance contains 1 or more polygons
    #         segm = annotation["segmentation"]
    #         if isinstance(segm, list):
    #             # polygons
    #             polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
    #             annotation["segmentation"] = [
    #                 p.reshape(-1) for p in transforms.apply_polygons(polygons)
    #             ]
    #         elif isinstance(segm, dict):
    #             # RLE
    #             mask = mask_util.decode(segm)
    #             mask = transforms.apply_segmentation(mask)
    #             assert tuple(mask.shape[:2]) == image_size
    #             annotation["segmentation"] = mask
    #         else:
    #             raise ValueError(
    #                 "Cannot transform segmentation of type '{}'!"
    #                 "Supported types are: polygons as list[list[float] or ndarray],"
    #                 " COCO-style RLE as a dict.".format(type(segm))
    #             )

    #     if "keypoints" in annotation:
    #         keypoints = transform_keypoint_annotations(
    #             annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
    #         )
    #         annotation["keypoints"] = keypoints

    #     return annotation

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format) #RGB
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks
