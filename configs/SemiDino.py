"""
=============================
SemiDino 最小可运行配置（高复用版）
=============================

本文件定位：
1. 以“最大限度复用官方实现”为原则，直接继承官方 DINO 与官方半监督数据配置；
2. 只做 SemiBaseDetector 包装与必要超参数覆盖；
3. 保留一个集中参数区，方便研究迭代与新手修改。

官方复用：
- DINO 基础配置：`configs/dino/dino-4scale_r50_8xb2-12e_coco.py`
- 半监督数据组织：`configs/_base_/datasets/semi_coco_detection.py`
- 半监督框架：`SemiBaseDetector` + `MeanTeacherHook` + `TeacherStudentValLoop`

本项目新增：
- 将官方 DINO 作为 teacher/student 的基础 detector 进行封装；
- 依据 `ANNSPATH.md` 约定预置半监督标注路径；
- 暴露关键半监督参数（阈值、权重、采样比例、训练步数等）。
"""

# =========================
# 一、集中可改参数区（建议优先改这里）
# =========================
DATA_ROOT = 'data/coco/'
LABELED_ANN_FILE = 'semi_anns/instances_train2017.1@10.json'
UNLABELED_ANN_FILE = 'semi_anns/instances_train2017.1@10-unlabeled.json'
VAL_ANN_FILE = 'annotations/instances_val2017.json'
TRAIN_IMAGE_PREFIX = 'train2017/'
VAL_IMAGE_PREFIX = 'val2017/'

# teacher/student 与伪标签关键参数
FREEZE_TEACHER = True
SUP_WEIGHT = 1.0
UNSUP_WEIGHT = 1.0
CLS_PSEUDO_THR = 0.7
MIN_PSEUDO_BBOX_WH = (1e-2, 1e-2)

# 训练与采样参数
BATCH_SIZE = 4
NUM_WORKERS = 4
SOURCE_RATIO = [1, 1]
MAX_ITERS = 90000
VAL_INTERVAL = 5000
CHECKPOINT_INTERVAL = 5000
LOG_INTERVAL = 50

# 输出目录
WORK_DIR = './work_dirs/semi_dino_mvp'

# =========================
# 二、继承官方配置
# =========================
_base_ = [
    '../thirdparty/mmdetection-3.3.0/configs/dino/dino-4scale_r50_8xb2-12e_coco.py',
    '../thirdparty/mmdetection-3.3.0/configs/_base_/datasets/semi_coco_detection.py',
]

# =========================
# 三、模型：SemiBaseDetector 包装官方 DINO
# =========================
# 这里直接拿官方 DINO 配置作为 detector，不手写 DINO 内部结构。
detector = _base_.model

model = dict(
    _delete_=True,
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=FREEZE_TEACHER,
        sup_weight=SUP_WEIGHT,
        unsup_weight=UNSUP_WEIGHT,
        cls_pseudo_thr=CLS_PSEUDO_THR,
        min_pseudo_bbox_wh=MIN_PSEUDO_BBOX_WH),
    semi_test_cfg=dict(
        predict_on='teacher',
        forward_on='teacher',
        extract_feat_on='teacher'),
)

# =========================
# 四、数据集与数据加载器（基于官方 semi_coco_detection.py 覆盖）
# =========================
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

labeled_dataset.data_root = DATA_ROOT
labeled_dataset.ann_file = LABELED_ANN_FILE
labeled_dataset.data_prefix = dict(img=TRAIN_IMAGE_PREFIX)

unlabeled_dataset.data_root = DATA_ROOT
unlabeled_dataset.ann_file = UNLABELED_ANN_FILE
unlabeled_dataset.data_prefix = dict(img=TRAIN_IMAGE_PREFIX)

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=dict(type='GroupMultiSourceSampler', batch_size=BATCH_SIZE, source_ratio=SOURCE_RATIO),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]),
)

val_dataloader = dict(
    dataset=dict(data_root=DATA_ROOT, ann_file=VAL_ANN_FILE, data_prefix=dict(img=VAL_IMAGE_PREFIX)))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=DATA_ROOT + VAL_ANN_FILE)
test_evaluator = val_evaluator

# =========================
# 五、训练/验证循环与 hook
# =========================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [dict(type='MeanTeacherHook')]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=LOG_INTERVAL),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=CHECKPOINT_INTERVAL, max_keep_ckpts=2),
)

log_processor = dict(type='LogProcessor', by_epoch=False, window_size=50)
work_dir = WORK_DIR
