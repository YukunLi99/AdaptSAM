from detectron2.config import LazyCall as L
# from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ..common.models.lora_sam import model
from ..common.data.coco import dataloader
from ..common.train_d2 import train
from ..common.optim_d2 import AdamW as optimizer
from ..common.schedule_d2 import lr_multiplier_1x as lr_multiplier  # scheduler
from detectron2.solver import WarmupParamScheduler
from sam_train.evaluation.lvis.AgnosticLvisEval import AgnosticLVISEvaluator
from sam_train.evaluation.coco.AgnosticCOCOEval import AgnosticCOCOEvaluator

train.max_iter = 10000  #1×
train.checkpointer.period = 500
train.init_checkpoint = "weight/sam_vit_h_4b8939.pth"
train.eval_period = 10

train.amp.enabled = False  #使用 FP-16
train.output_dir = 'output_lora_coco'

optimizer.lr = 1e-4
optimizer.weight_decay = 0.05

scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        # note that scheduler is scale-invariant. This is equivalent to
        # milestones=[6, 8, 9]
        # milestones=[12000, 16000],
        milestones=[8000, 9000],   # 2500 for batchsize 16, 5000 for batchsize 8
        #milestones=[2000, 2500],   # 2500 for batchsize 16, 5000 for batchsize 8
    )

lr_multiplier = L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=100 / train.max_iter,
        warmup_method="linear",
        warmup_factor=0.001,
    )

#class-agnostic eval
dataloader.evaluator = [
    L(AgnosticCOCOEvaluator)(
        dataset_name="agnostic_coco_2017_val",
        tasks=("segm",),
        output_dir = 'output_lora_coco',
        max_dets_per_image = 100,
    ),
]
dataloader.train.total_batch_size = 16