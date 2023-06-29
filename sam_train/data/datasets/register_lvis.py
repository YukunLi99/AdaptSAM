import os

#from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.lvis import register_lvis_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.lvis import get_lvis_instances_meta

_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))

def _get_metadata():
    #register时, load_coco_json 会自动更新下述两个变量
    return { 
        "thing_dataset_id_to_contiguous_id": {1 : 0},  #类别对应关系  thing-1对应下标0
        "thing_classes": ['thing']}    #类别名称

# _PREDEFINED_SPLITS_COCO = {
#     "agnostic_coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
# }

# for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO.items():
#     register_coco_instances(
#         key,
#         _get_metadata(),
#         os.path.join(_root, json_file) if "://" not in json_file else json_file,
#         os.path.join(_root, image_root),
#     )


_PREDEFINED_SPLITS_LVISE = {
    "agnostic_lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVISE.items():
    register_lvis_instances(
        key,
        _get_metadata(),
        os.path.join(_root, json_file) if "://" not in json_file else json_file,
        os.path.join(_root, image_root),
    )


_PREDEFINED_SPLITS_COCO = {
    "agnostic_coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO.items():
    register_coco_instances(
        key,
        _get_metadata(),
        os.path.join(_root, json_file) if "://" not in json_file else json_file,
        os.path.join(_root, image_root),
    )



_PREDEFINED_SPLITS_VOC = {
    "agnostic_voc_2007_test_ins": ("VOC2007/JPEGImages", "VOC2007/voc_2007_instance_test.json"),
    "agnostic_voc_2007_trainval_ins": ("VOC2007/JPEGImages", "VOC2007/voc_2007_instance_trainval.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_VOC.items():
    register_coco_instances(
        key,
        _get_metadata(),
        os.path.join(_root, json_file) if "://" not in json_file else json_file,
        os.path.join(_root, image_root),
    )
