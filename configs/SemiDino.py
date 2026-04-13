"""
=============================
SemiDino 最小可运行配置（教学版）
=============================

核心原则说明：
1. 直接复用 MMDetection 官方 DINO 配置，不重写 DINO 网络结构；
2. 仅增加一个“最小桥接类”来修复 DINO 在 teacher/student 双构建时的配置污染问题；
3. 其余半监督流程（三分支数据流、MeanTeacher、TeacherStudentValLoop）均复用官方实现。
"""

import copy

from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.semi_base import SemiBaseDetector
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class SemiDinoDetector(SemiBaseDetector):
    """最小桥接类：仅修复 DINO 双次 build 的配置污染。

    为什么需要它：
    - `SemiBaseDetector` 会使用同一个 detector cfg 构建 student 和 teacher 两次；
    - DINO 初始化时会原地修改部分 bbox_head 配置；
    - 如果第二次还用同一个 cfg 对象，会触发 DINO 断言。

    解决方式：
    - 对 detector cfg 分别 `deepcopy` 后再构建 student / teacher；
    - 除初始化外，其余行为全部继承官方 `SemiBaseDetector`。
    """

    def __init__(
        self,
        detector: ConfigType,
        semi_train_cfg: OptConfigType = None,
        semi_test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        BaseDetector.__init__(self, data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.student = MODELS.build(copy.deepcopy(detector))
        self.teacher = MODELS.build(copy.deepcopy(detector))
        self.semi_train_cfg = semi_train_cfg
        self.semi_test_cfg = semi_test_cfg
        if self.semi_train_cfg.get('freeze_teacher', True):
            self.freeze(self.teacher)


# =========================
# 一、集中可改参数区
# =========================

# 路径参数（遵循 data/ANNSPATH.md）
DATA_ROOT = 'data/coco/'
LABELED_ANN_FILE = 'semi_anns/instances_train2017.1@10.json'
UNLABELED_ANN_FILE = 'semi_anns/instances_train2017.1@10-unlabeled.json'
VAL_ANN_FILE = 'annotations/instances_val2017.json'

# 结构/实验关键参数
PATCH_SIZE = 16  # 预留项：当前 R50-DINO 不直接使用 patch size
FREEZE_TEACHER = True
SUP_WEIGHT = 1.0
UNSUP_WEIGHT = 1.0
CLS_PSEUDO_THR = 0.7
MIN_PSEUDO_BBOX_WH = (1e-2, 1e-2)

BATCH_SIZE = 4
NUM_WORKERS = 4
SOURCE_RATIO = [1, 1]
MAX_ITERS = 90000
VAL_INTERVAL = 5000
CHECKPOINT_INTERVAL = 5000
LOG_INTERVAL = 50

BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_NORM = 0.1


# =========================
# 二、复用官方配置
# =========================

_base_ = [
    '../thirdparty/mmdetection-3.3.0/configs/dino/dino-4scale_r50_8xb2-12e_coco.py',
    '../thirdparty/mmdetection-3.3.0/configs/_base_/datasets/semi_coco_detection.py',
]

default_scope = 'mmdet'

# 直接复用官方 DINO 模型定义，而不是在本文件“重写一个 DINO”。
detector = _base_.model

model = dict(
    _delete_=True,
    type='SemiDinoDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor,
    ),
    semi_train_cfg=dict(
        freeze_teacher=FREEZE_TEACHER,
        sup_weight=SUP_WEIGHT,
        unsup_weight=UNSUP_WEIGHT,
        cls_pseudo_thr=CLS_PSEUDO_THR,
        min_pseudo_bbox_wh=MIN_PSEUDO_BBOX_WH,
    ),
    semi_test_cfg=dict(
        predict_on='teacher',
        forward_on='teacher',
        extract_feat_on='teacher',
    ),
)


# =========================
# 三、数据：复用官方 semi_coco_detection 再做最小覆盖
# =========================

labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

labeled_dataset.data_root = DATA_ROOT
labeled_dataset.ann_file = LABELED_ANN_FILE
labeled_dataset.data_prefix = dict(img='train2017/')

unlabeled_dataset.data_root = DATA_ROOT
unlabeled_dataset.ann_file = UNLABELED_ANN_FILE
unlabeled_dataset.data_prefix = dict(img='train2017/')

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=dict(type='GroupMultiSourceSampler', batch_size=BATCH_SIZE, source_ratio=SOURCE_RATIO),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]),
)

val_dataloader = dict(
    dataset=dict(
        data_root=DATA_ROOT,
        ann_file=VAL_ANN_FILE,
        data_prefix=dict(img='val2017/'),
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=DATA_ROOT + VAL_ANN_FILE,
    metric='bbox',
)
test_evaluator = val_evaluator


# =========================
# 四、训练策略
# =========================

train_cfg = dict(type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=BASE_LR, weight_decay=WEIGHT_DECAY),
    clip_grad=dict(max_norm=MAX_NORM, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=MAX_ITERS, by_epoch=False, milestones=[60000, 80000], gamma=0.1),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=LOG_INTERVAL),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=CHECKPOINT_INTERVAL, max_keep_ckpts=2),
)

custom_hooks = [dict(type='MeanTeacherHook')]
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
auto_scale_lr = dict(enable=False, base_batch_size=16)

work_dir = './work_dirs/semi_dino_mvp'
