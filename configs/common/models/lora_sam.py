from detectron2.config import LazyCall as L
from functools import partial

import torch
import sys
sys.path.append('/data/sda/lyk/code/finetune_sam/')
from detectron2.data import MetadataCatalog
from sam_train.finetune_sam import Sam
from sam_train.sam.image_encoder import ImageEncoderViT
from sam_train.sam.lora_image_encoder import LoRA_Encoder
from sam_train.sam.prompt_encoder import PromptEncoder
from sam_train.sam.transformer import TwoWayTransformer
from sam_train.sam.mask_decoder import MaskDecoder


prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size


#vit_base
# encoder_embed_dim=768
# encoder_depth=12
# encoder_num_heads=12
# encoder_global_attn_indexes=[2, 5, 8, 11]

#vit large
# encoder_embed_dim=1024
# encoder_depth=24
# encoder_num_heads=16
# encoder_global_attn_indexes=[5, 11, 17, 23]

#vit huge
encoder_embed_dim=1280
encoder_depth=32
encoder_num_heads=16
encoder_global_attn_indexes=[7, 15, 23, 31]

model = L(Sam)(
        image_encoder=L(LoRA_Encoder)(
            vit_model = L(ImageEncoderViT)(
                    depth=encoder_depth,
                    embed_dim=encoder_embed_dim,
                    img_size=image_size,
                    mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=encoder_num_heads,
                    patch_size=vit_patch_size,
                    qkv_bias=True,
                    use_rel_pos=True,
                    global_attn_indexes=encoder_global_attn_indexes,
                    window_size=14,
                    out_chans=prompt_embed_dim,
                ),
            r = 4,
            topk = 1, # last layer
        ),

        prompt_encoder=L(PromptEncoder)(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=L(MaskDecoder)(
            num_multimask_outputs=3,
            transformer=L(TwoWayTransformer)(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        metadata=L(MetadataCatalog.get)(name="coco_2017_val"),
    )
