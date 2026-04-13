"""
SemiDino：基于 MMDetection 3.x 官方 SemiBaseDetector + DINO 的半监督目标检测配置。

==============================
整体运行逻辑（建议先读）
==============================
1. 本配置先构造一个“纯官方 DINO 检测器配置字典”（变量 `detector`）。
2. 再用官方 `SemiBaseDetector` 将该 `detector` 包裹成 teacher-student 结构：
   - student：接收有标注样本（sup）做监督训练；
   - teacher：接收无标注样本的弱增强分支（unsup_teacher）产出伪标签；
   - student：再接收无标注样本的强增强分支（unsup_student）学习伪标签。
3. 数据侧使用官方 `ConcatDataset + GroupMultiSourceSampler`：
   - source 0：有标注数据；
   - source 1：无标注数据；
   - `source_ratio` 控制每个 batch 中两者配比。
4. 训练 loop 使用官方 `IterBasedTrainLoop`，验证 loop 使用
   `TeacherStudentValLoop`，并通过官方 `MeanTeacherHook` 用 EMA 更新 teacher。

==============================
关键组件说明
==============================
- SemiBaseDetector：官方半监督通用检测器封装。
- DINO：官方端到端检测器，作为 teacher/student 的底层 detector。
- MultiBranch / MultiBranchDataPreprocessor：官方多分支数据与预处理模块。
- GroupMultiSourceSampler：官方半监督多源采样器。
- MeanTeacherHook：官方 teacher EMA 更新钩子。

注意：
- 在 mmdet 3.3.0 中，`SemiBaseDetector` 内部对 `detector` 使用了深拷贝后再构建
  teacher/student，能规避 “DINO 配置对象二次 build 被就地修改” 的问题。
- 因此本文件尽量复用官方实现，仅做必要配置组装。
"""

# =========================
# 0) 便捷修改区（实验常改参数）
# =========================

# ---- 数据路径与标注路径 ----
# data_root：数据集根目录。
DATA_ROOT = 'data/coco/'
# 有标注训练集标注文件。
LABELED_ANN_FILE = 'semi_anns/instances_train2017.1@1.json'
# 无标注训练集标注文件（需包含 categories 字段）。
UNLABELED_ANN_FILE = 'semi_anns/instances_train2017.1@1-unlabeled.json'
# 验证集标注文件。
VAL_ANN_FILE = 'annotations/instances_val2017.json'

# 图像目录前缀。
LABELED_IMG_PREFIX = 'train2017/'
UNLABELED_IMG_PREFIX = 'unlabeled2017/'
VAL_IMG_PREFIX = 'val2017/'

# ---- 训练轮数 / 迭代数 ----
# 半监督常用迭代制；如需按 epoch 训练，可切换 train_cfg。
MAX_ITERS = 180000
VAL_INTERVAL = 5000

# ---- patch size / pad divisor ----
# 对 DINO 而言更常见的是 pad_size_divisor。
# 这里保留为便捷开关：若后续 backbone 迁移到 ViT，可把它视作 patch 对齐粒度。
PATCH_SIZE_DIVISOR = 1

# ---- 批大小与采样比例 ----
BATCH_SIZE = 4
NUM_WORKERS = 4
# source_ratio=[有标注:无标注]。
SOURCE_RATIO = [1, 4]

# ---- 类别与 checkpoint ----
NUM_CLASSES = 80
# 预训练权重（用于 student/teacher 初始化）。
BACKBONE_CKPT = 'torchvision://resnet50'

# ---- 训练输出路径 ----
# 工作目录：日志、权重、可视化结果等都会放到该目录。
WORK_DIR = './work_dirs/semi_dino_r50_4scale_coco'
# 可视化后端本地保存目录。
VIS_BACKEND_SAVE_DIR = './work_dirs/semi_dino_r50_4scale_coco/vis_data'

# ---- 半监督核心权重与阈值 ----
SUP_WEIGHT = 1.0
UNSUP_WEIGHT = 1.0
PSEUDO_SCORE_THR = 0.5
MIN_PSEUDO_BBOX_WH = (1e-2, 1e-2)

# =========================
# 1) 基础配置继承
# =========================

_base_ = ['../thirdparty/mmdetection-3.3.0/configs/_base_/default_runtime.py']
backend_args = None

# =========================
# 2) 数据增强配置（官方 semi_det 教程范式）
# =========================

# 颜色增强空间。
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

# 几何增强空间。
geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

# 多尺度 resize 范围（短边, 长边上限）。
semi_train_scales = [(1333, 400), (1333, 1200)]

# MultiBranch 的三个分支字段：
# sup：有标注监督分支；
# unsup_teacher：无标注弱增强给 teacher；
# unsup_student：无标注强增强给 student。
branch_field = ['sup', 'unsup_teacher', 'unsup_student']

# 有标注数据 pipeline。
sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=semi_train_scales, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

# 无标注弱增强（teacher 产伪标签）。
weak_pipeline = [
    dict(type='RandomResize', scale=semi_train_scales, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# 无标注强增强（student 学伪标签）。
strong_pipeline = [
    dict(type='RandomResize', scale=semi_train_scales, keep_ratio=True),
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
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# 无标注主 pipeline：先读取图像与空标注，再分流成 teacher/student 两视图。
unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline)
]

# 验证/测试 pipeline。
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

# =========================
# 3) 构建 DINO 检测器（官方配置平移）
# =========================

detector = dict(
    type='DINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=PATCH_SIZE_DIVISOR),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=BACKBONE_CKPT)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048,
                         ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048,
                         ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=NUM_CLASSES,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ])),
    test_cfg=dict(max_per_img=300),
)

# =========================
# 4) 半监督模型封装（官方 SemiBaseDetector）
# =========================

model = dict(
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=SUP_WEIGHT,
        unsup_weight=UNSUP_WEIGHT,
        cls_pseudo_thr=PSEUDO_SCORE_THR,
        min_pseudo_bbox_wh=MIN_PSEUDO_BBOX_WH),
    semi_test_cfg=dict(predict_on='teacher'))

# =========================
# 5) 数据集与 DataLoader
# =========================

dataset_type = 'CocoDataset'

labeled_dataset = dict(
    type=dataset_type,
    data_root=DATA_ROOT,
    ann_file=LABELED_ANN_FILE,
    data_prefix=dict(img=LABELED_IMG_PREFIX),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=backend_args)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=DATA_ROOT,
    ann_file=UNLABELED_ANN_FILE,
    data_prefix=dict(img=UNLABELED_IMG_PREFIX),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=BATCH_SIZE,
        source_ratio=SOURCE_RATIO),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
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
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=DATA_ROOT + VAL_ANN_FILE,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# =========================
# 6) 训练策略、优化器、Hook
# =========================

train_cfg = dict(type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=MAX_ITERS, by_epoch=False,
         milestones=[int(MAX_ITERS * 0.8), int(MAX_ITERS * 0.9)], gamma=0.1),
]

# 关键：EMA 更新 teacher。
custom_hooks = [dict(type='MeanTeacherHook')]

# 默认 hooks：按迭代保存 checkpoint，便于半监督长训中断续训。
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50),
)

# 日志处理器按 iteration 统计。
log_processor = dict(by_epoch=False)

# 可视化配置：本地保存可视化结果到指定目录。
vis_backends = [dict(type='LocalVisBackend', save_dir=VIS_BACKEND_SAVE_DIR)]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 工作目录。
work_dir = WORK_DIR

# 自动学习率缩放：基准 batch size 设为 16（按 DINO 官方配置习惯）。
auto_scale_lr = dict(base_batch_size=16)
