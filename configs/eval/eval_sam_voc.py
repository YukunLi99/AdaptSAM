from detectron2.config import LazyCall as L
# from detectron2.solver import WarmupParamScheduler
# from fvcore.common.param_scheduler import MultiStepParamScheduler

from ..common.models.sam import model
from ..common.data.voc import dataloader
from ..common.train_d2 import train
from ..common.optim_d2 import AdamW as optimizer
from ..common.schedule_d2 import lr_multiplier_1x as lr_multiplier  # scheduler
from sam_train.evaluation.lvis.AgnosticLvisEval import AgnosticLVISEvaluator
from sam_train.evaluation.coco.AgnosticCOCOEval import AgnosticCOCOEvaluator

train.max_iter = 90000  #1×
train.checkpointer.period = 5000
train.init_checkpoint = "weight/sam_vit_h_4b8939.pth"

train.amp.enabled = False  #使用 FP-16
train.output_dir = './output_voc_eval'

optimizer.lr = 1e-4
optimizer.weight_decay = 0.05

#class-agnostic eval
dataloader.evaluator = [
    L(AgnosticCOCOEvaluator)(
        dataset_name="agnostic_voc_2007_test_ins",
        tasks=("segm",),
        output_dir = './output_voc_eval',
        max_dets_per_image = 100,
    ),
]
dataloader.train.total_batch_size = 16