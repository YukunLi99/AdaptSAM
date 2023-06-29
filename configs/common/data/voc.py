from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
# from detectron2.data import DatasetMapper
from sam_train.data.dataset_mapper import DatasetMapper
# from sam_train.data.dataset_mapper import VocDatasetMapper

from sam_train.data import (
    COCOPanopticDatasetMapper,
    build_d2_test_dataloader,
    build_d2_train_dataloader,
    get_openseg_labels,
)
from sam_train.evaluation.d2_evaluator import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    SemSegEvaluator,
)
from sam_train.modeling.wrapper.pano_wrapper import OpenPanopticInference
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    LVISEvaluator,
)
from sam_train.utils.transforms import ResizeLongestEdge
# from sam_train.evaluation.voc.pascal_voc_evaluation  import PascalVOCDetectionEvaluator

dataloader = OmegaConf.create()

dataloader.train = L(build_d2_train_dataloader)(
    dataset=L(get_detection_dataset_dicts)(
        # names=("voc_2007_trainval", 'voc_2012_trainval'), filter_empty=True
        names=("voc_2007_trainval_ins", "voc_2012_trainval_ins"), filter_empty=True
    ),
    mapper=L(DatasetMapper)(
        is_train=True,
        # COCO LSJ aug
        augmentations=[
            L(T.RandomFlip)(horizontal=True),
            L(T.ResizeScale)(
                min_scale=0.1,
                max_scale=2.0,
                target_height=1024,
                target_width=1024,
            ),
            L(T.FixedSizeCrop)(crop_size=(1024, 1024)),
            # L(T.ResizeShortestEdge)(short_edge_length=1024, sample_style="choice", max_size=2560),

            # L(ResizeLongestEdge)(target_length = 1024)
        ],
        image_format="RGB",
        use_instance_mask = True,
        instance_mask_format = "bitmask",
    ),
    total_batch_size=2,
    num_workers=4,
)

dataloader.test = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(
        names="voc_2007_test_ins",
        filter_empty=False,
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            # L(T.ResizeShortestEdge)(short_edge_length=1024, sample_style="choice", max_size=2560),
            L(ResizeLongestEdge)(target_length = 1024)
        ],
        image_format="${...train.mapper.image_format}",
    ),
    local_batch_size=1,
    num_workers=1,
)

dataloader.evaluator = [
    L(COCOEvaluator)(
        dataset_name="${...test.dataset.names}",
        tasks=("segm",),
        output_dir = './output_voc'
    ),
]

# dataloader.wrapper = L(OpenPanopticInference)(
#     labels=L(get_openseg_labels)(dataset="coco_panoptic", prompt_engineered=True),
#     metadata=L(MetadataCatalog.get)(name="${...test.dataset.names}"),
# )
