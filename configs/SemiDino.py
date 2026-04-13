"""
SemiDino：在 MMDetection 3.3.0 中用“官方 DINO + 官方半监督数据基座”拼装的半监督配置。

==============================
整体运行逻辑（严格复用官方实现）
==============================
1) 先继承官方 DINO 检测器配置：
   - 来源：configs/dino/dino-4scale_r50_8xb2-12e_coco.py
   - 该文件中的 `model` 本身就是官方 DINO 完整实现。
2) 再继承官方半监督数据配置：
   - 来源：configs/_base_/datasets/semi_coco_detection.py
   - 该文件提供官方 MultiBranch pipeline、ConcatDataset、
     GroupMultiSourceSampler 等半监督数据流。
3) 以官方 `SemiBaseDetector` 封装“官方 DINO 检测器”：
   - teacher / student 都由同一个 DINO 配置构建；
   - `SemiBaseDetector` 内部会深拷贝 detector 配置，规避二次 build 修改问题。
4) 训练与验证流程继续使用官方半监督范式：
   - IterBasedTrainLoop + TeacherStudentValLoop + MeanTeacherHook。

说明：
- 本文件不重新实现 DINO 结构，不自定义新检测器类；
- 只做“参数入口 + 官方配置拼接 + 必要覆盖”。
"""

# =========================
# 0) 便捷修改区（实验常改参数）
# =========================

# 数据根目录。
DATA_ROOT = 'data/coco/'

# 标注文件（当前示例沿用 1% 划分）。
LABELED_ANN_FILE = 'semi_anns/instances_train2017.1@1.json'
UNLABELED_ANN_FILE = 'semi_anns/instances_train2017.1@1-unlabeled.json'
VAL_ANN_FILE = 'annotations/instances_val2017.json'

# 图像目录。
LABELED_IMG_PREFIX = 'train2017/'
UNLABELED_IMG_PREFIX = 'unlabeled2017/'
VAL_IMG_PREFIX = 'val2017/'

# 训练总迭代与验证间隔。
MAX_ITERS = 180000
VAL_INTERVAL = 5000

# DINO 预处理 pad 对齐粒度（保留为可调入口）。
PATCH_SIZE_DIVISOR = 1

# dataloader 相关。
BATCH_SIZE = 4
NUM_WORKERS = 4
SOURCE_RATIO = [1, 4]  # [有标注:无标注]

# 模型类别数与权重初始化。
NUM_CLASSES = 80
BACKBONE_CKPT = 'torchvision://resnet50'

# 半监督损失与阈值。
SUP_WEIGHT = 1.0
UNSUP_WEIGHT = 1.0
PSEUDO_SCORE_THR = 0.5
MIN_PSEUDO_BBOX_WH = (1e-2, 1e-2)

# 输出目录。
WORK_DIR = './work_dirs/semi_dino_r50_4scale_coco'
VIS_BACKEND_SAVE_DIR = './work_dirs/semi_dino_r50_4scale_coco/vis_data'

from copy import deepcopy
from pathlib import Path

from mmengine.config import Config

# =========================
# 1) 继承官方配置（关键修复）
# =========================
#
# 说明：
# - 不能把 “完整 dino 配置” 和 “semi_coco_detection 数据基座” 同时放进 _base_；
# - 因为 dino 配置里已经继承了 coco 数据集基座，semi_coco_detection 里也定义了
#   dataset_type / data_root / train_dataloader / val_dataloader 等同名顶层键；
# - mmengine 会在“多个 base”阶段直接报 Duplicate key 错误。
#
# 修复策略：
# - _base_ 仅保留 dino 官方配置（避免 base 间重复键）；
# - semi 数据配置通过 Config.fromfile 单独读取，再深拷贝需要的官方字段复用；
# - 所有第三方配置路径都基于“当前配置文件所在目录”拼绝对路径，避免
#   Windows 下因运行目录不同导致 FileNotFoundError。
_THIS_DIR = Path(__file__).resolve().parent
_MMDET_CONFIG_ROOT = _THIS_DIR.parent / 'thirdparty' / 'mmdetection-3.3.0' / 'configs'
_SEMI_DATASET_CONFIG_PATH = _MMDET_CONFIG_ROOT / '_base_' / 'datasets' / 'semi_coco_detection.py'

# 注意：_base_ 会被 mmengine 预解析，不能依赖后续 Python 变量；
# 这里必须写“字面量路径字符串”。使用相对路径可同时满足：
# 1) 写法短，不受绝对目录影响；
# 2) mmengine 会按当前配置文件位置解析，相比运行目录更稳定。
_base_ = ['../thirdparty/mmdetection-3.3.0/configs/dino/dino-4scale_r50_8xb2-12e_coco.py']

semi_dataset_cfg = Config.fromfile(str(_SEMI_DATASET_CONFIG_PATH))

# =========================
# 2) 直接复用官方 DINO（来自 dino base config）
# =========================

# 这里不手写 DINO 结构，直接拿官方 dino 配置作为 detector。
detector = _base_.model

# 仅覆盖必要参数入口。
detector.bbox_head.num_classes = NUM_CLASSES
detector.backbone.init_cfg.checkpoint = BACKBONE_CKPT
detector.data_preprocessor.pad_size_divisor = PATCH_SIZE_DIVISOR

# =========================
# 3) 用官方 SemiBaseDetector 封装官方 DINO
# =========================

model = dict(
    _delete_=True,
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=SUP_WEIGHT,
        unsup_weight=UNSUP_WEIGHT,
        cls_pseudo_thr=PSEUDO_SCORE_THR,
        min_pseudo_bbox_wh=MIN_PSEUDO_BBOX_WH),
    semi_test_cfg=dict(predict_on='teacher'))

# =========================
# 4) 复用官方 semi 数据配置并覆盖路径
# =========================

backend_args = None

labeled_dataset = deepcopy(semi_dataset_cfg.labeled_dataset)
unlabeled_dataset = deepcopy(semi_dataset_cfg.unlabeled_dataset)

labeled_dataset.data_root = DATA_ROOT
labeled_dataset.ann_file = LABELED_ANN_FILE
labeled_dataset.data_prefix = dict(img=LABELED_IMG_PREFIX)
labeled_dataset.backend_args = backend_args

unlabeled_dataset.data_root = DATA_ROOT
unlabeled_dataset.ann_file = UNLABELED_ANN_FILE
unlabeled_dataset.data_prefix = dict(img=UNLABELED_IMG_PREFIX)
unlabeled_dataset.backend_args = backend_args

train_dataloader = dict(
    _delete_=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=dict(type='GroupMultiSourceSampler', batch_size=BATCH_SIZE, source_ratio=SOURCE_RATIO),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    _delete_=True,
    dataset=dict(
        data_root=DATA_ROOT,
        ann_file=VAL_ANN_FILE,
        data_prefix=dict(img=VAL_IMG_PREFIX),
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=DATA_ROOT + VAL_ANN_FILE)
test_evaluator = val_evaluator

# =========================
# 5) 官方半监督训练 loop / hook
# =========================

train_cfg = dict(type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=MAX_ITERS,
        by_epoch=False,
        milestones=[int(MAX_ITERS * 0.8), int(MAX_ITERS * 0.9)],
        gamma=0.1)
]

custom_hooks = [dict(type='MeanTeacherHook')]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))
log_processor = dict(by_epoch=False)

# 可视化与输出目录。
vis_backends = [dict(type='LocalVisBackend', save_dir=VIS_BACKEND_SAVE_DIR)]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
work_dir = WORK_DIR

# 与官方 DINO 基线保持一致的自动 LR 缩放基准。
auto_scale_lr = dict(base_batch_size=16)
