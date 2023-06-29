# Adapt SAM: adapt segment anything model based on detectron2

AdaptSAM is a library built on top of [Detectron2](https://github.com/facebookresearch/detectron2) that allows adapting the Segment-Anything Model (SAM) for custom COCO-format datasets. It supports point prompt training and leverages LORA technology for customizable adaptations of [SAM](https://github.com/luca-medeiros/lang-segment-anything).

## Getting Started with SAM Detectron2

### Command line-based Training & Evaluation

We provide a script `tools/lazyconfig_train_net.py` that trains all configurations of SAM.

To train a model with `tools/lazyconfig_train_net.py`, first prepare the datasets following the instructions in
[datasets/README.md](https://github.com/facebookresearch/detectron2/tree/b2948fb7abe0604db8b9ec25898871e656d0b210/datasets) and then run, for single-node (8-GPUs) NVIDIA-based training:

```bash
(node0)$ ./tools/lazyconfig_train_net.py --config-file configs/finetune/finetune_lora_sam_coco.py --num-gpus 8 
```

To evaluate a trained SAM model's performance, run on single node

```
(node0)$ ./tools/lazyconfig_train_net.py --config-file configs/finetune/finetune_lora_sam_coco.py --num-gpus 8 --eval-only --init-from /path/to/checkpoint
```

To evaluate a original SAM model's performance, run on single node
```
(node0)$ ./tools/lazyconfig_train_net.py --config-file configs/eval/eval_sam_coco.py --num-gpus 8 --eval-only
```

## Installation

Our environment requirements are consistent with ODISE, for installation, please refer to [ODISE](GitHub - NVlabs/ODISE: Official PyTorch implementation of ODISE: Open-Vocabulary Panoptic Segmentati)

## Features
- Compatible with all features of Detectron2 framework
- Supports custom COCO-format datasets
- Supports point prompt training
- Supports LORA technology
## Prepare Datasets
Dataset preparation for AdaptSAM follows [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md) and [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md). 

## Results

| method        | datasets | $AP_{100}$ | $AR_{100}$ | $AR_{s}$ | $AR_{m}$ | $AR_{l}$ |
| ------------- | -------- | ---------- | ---------- | -------- | -------- | -------- |
| SAM-Base      | voc      | 1.3        | 34         | 33.8     | 28.4     | 37.5     |
| AdaptSAM-Base | voc      | 5.8        | 52         | 22.6     | 38.1     | 65.8     |
| SAM-Huge      | coco     | 1.8        | 32.7       | 23.5     | 37.8     | 41.8     |
| AdaptSAM-Huge | coco     | 10.2       | 49.7       | 28.4     | 59.5     | 73.2     |

## Acknowledgement

Code is largely based on [Detectron2](https://github.com/facebookresearch/detectron2), [SAM](https://github.com/luca-medeiros/lang-segment-anything), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [ODISE](GitHub - NVlabs/ODISE: Official PyTorch implementation of ODISE: Open-Vocabulary Panoptic Segmentati)

## License

This project is licensed same as SAM model.