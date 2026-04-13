"""
SemiDino：在 MMDetection 3.3.0 中用“官方 DINO + 官方半监督训练组件”拼装的半监督配置。

==============================
整体运行逻辑（路径无关版）
==============================
1) 仅继承官方 DINO 配置：
   - 来源：configs/dino/dino-4scale_r50_8xb2-12e_coco.py
   - 该 base 提供官方 DINO 检测器与其训练默认项。
2) 半监督数据流不再在配置执行期调用 Config.fromfile：
   - 直接在本文件中按官方 semi_coco_detection.py 结构定义；
   - 这样可彻底规避 __file__ / 工作目录 / 相对路径解释差异等问题。
3) 使用官方 SemiBaseDetector 封装官方 DINO：
   - teacher/student 均由官方机制构建；
   - 保持与官方半监督训练栈一致（MeanTeacherHook、TeacherStudentValLoop 等）。

说明：
- 本文件重点是“稳定可运行 + 复用官方组件 + 路径零陷阱”；
- 不新增自定义检测器类，只做必要参数覆盖。
"""

from copy import deepcopy

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

# =========================
# 1) 仅继承官方 DINO（避免 base 重复键）
# =========================
_base_ = ['../thirdparty/mmdetection-3.3.0/configs/dino/dino-4scale_r50_8xb2-12e_coco.py']

# =========================
# 2) 直接复用官方 DINO（来自 dino base config）
# =========================

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
# 4) 官方 semi_coco_detection 数据流（内联版，避免路径问题）
# =========================

backend_args = None
dataset_type = 'CocoDataset'

# 以下增强空间与 pipeline 结构对齐官方 semi_coco_detection.py。
color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

scale = [(1333, 400), (1333, 1200)]
branch_field = ['sup', 'unsup_teacher', 'unsup_student']

sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='MultiBranch', branch_field=branch_field, sup=dict(type='PackDetInputs')),
]

weak_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),
]

strong_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),
]

unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

labeled_dataset = dict(
    type=dataset_type,
    data_root=DATA_ROOT,
    ann_file=LABELED_ANN_FILE,
    data_prefix=dict(img=LABELED_IMG_PREFIX),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=backend_args,
)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=DATA_ROOT,
    ann_file=UNLABELED_ANN_FILE,
    data_prefix=dict(img=UNLABELED_IMG_PREFIX),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceSampler', batch_size=BATCH_SIZE, source_ratio=SOURCE_RATIO),
    dataset=dict(type='ConcatDataset', datasets=[deepcopy(labeled_dataset), deepcopy(unlabeled_dataset)]),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        ann_file=VAL_ANN_FILE,
        data_prefix=dict(img=VAL_IMG_PREFIX),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=DATA_ROOT + VAL_ANN_FILE,
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# =========================
# 5) 训练 loop 与调度器：尽量复用官方 DINO 基线
# =========================
#
# 说明：
# - train_cfg / test_cfg / param_scheduler 不在此处重写，直接沿用 dino base；
# - 仅将 val_loop 改为半监督需要的 TeacherStudentValLoop；
# - 这样可以最大程度保持“按官方 DINO 配置训练”。
val_cfg = dict(type='TeacherStudentValLoop')

custom_hooks = [dict(type='MeanTeacherHook')]

# 可视化与输出目录。
vis_backends = [dict(type='LocalVisBackend', save_dir=VIS_BACKEND_SAVE_DIR)]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
work_dir = WORK_DIR

# 与官方 DINO 基线保持一致的自动 LR 缩放基准。
auto_scale_lr = dict(base_batch_size=16)
