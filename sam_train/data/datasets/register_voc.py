import os
from detectron2.data.datasets.register_coco import register_coco_instances

VOC_CLASS_NAMES = [
    {'id': 1, 'name': 'aeroplane'},
    {'id': 2, 'name': 'bicycle'},
    {'id': 3, 'name': 'bird'},
    {'id': 4, 'name': 'boat'},
    {'id': 5, 'name': 'bottle'},
    {'id': 6, 'name': 'bus'},
    {'id': 7, 'name': 'car'},
    {'id': 8, 'name': 'cat'},
    {'id': 9, 'name': 'chair'},
    {'id': 10, 'name': 'cow'},
    {'id': 11, 'name': 'diningtable'},
    {'id': 12, 'name': 'dog'},
    {'id': 13, 'name': 'horse'},
    {'id': 14, 'name': 'motorbike'},
    {'id': 15, 'name': 'person'},
    {'id': 16, 'name': 'pottedplant'},
    {'id': 17, 'name': 'sheep'},
    {'id': 18, 'name': 'sofa'},
    {'id': 19, 'name': 'train'},
    {'id': 20, 'name': 'tvmonitor'}
]

def _get_metadata():
    id_to_name = {x['id']: x['name'] for x in VOC_CLASS_NAMES}

    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_VOC_INS = {
    "voc_2007_trainval_ins": ("VOC2007/JPEGImages", "VOC2007/voc_2007_instance_trainval.json"),
    "voc_2012_trainval_ins": ("VOC2012/JPEGImages", "VOC2012/voc_2012_instance_trainval.json"),
    "voc_2007_test_ins": ("VOC2007/JPEGImages", "VOC2007/voc_2007_instance_test.json"),
}


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))

for key, (image_root, json_file) in _VOC_INS.items():
    register_coco_instances(
        key,
        _get_metadata(),
        os.path.join(_root, json_file) if "://" not in json_file else json_file,
        os.path.join(_root, image_root),
    )
