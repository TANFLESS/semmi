"""
=============================
SemiDino 最小可运行配置（教学版）
=============================

本文件作用：
1. 在 MMDetection 3.x 中，以官方 `SemiBaseDetector` 包裹官方 `DINO`；
2. 通过官方半监督多分支数据流（sup / unsup_teacher / unsup_student）跑通训练闭环；
3. 保持“可直接改关键参数”的实验配置风格，便于后续逐步迁移到更接近 Semi-DETR 的实现。

本文件在项目流水线中的位置：
- 由根目录 `train.py` 读取并送入 MMEngine Runner；
- Runner 根据本文件构建 model / dataloader / hook / loop；
- 训练、验证、推理均由官方流程执行。

依赖关系：
- 复用 MMDetection 官方模块：SemiBaseDetector、DINO、MultiBranchDataPreprocessor、
  GroupMultiSourceSampler、TeacherStudentValLoop、MeanTeacherHook、CocoMetric。
- 不新增自定义 detector 类（当前阶段先保证最小可运行闭环）。

产出：
- 一个可用于半监督目标检测实验的完整 config（含模型、数据、训练策略、评估策略）。
"""

# =========================
# 一、集中可改关键参数（强烈建议优先在这里调参）
# =========================

# ---- 路径相关（遵循 data/ANNSPATH.md）----
DATA_ROOT = 'data/coco/'
LABELED_ANN_FILE = 'semi_anns/instances_train2017.1@10.json'
UNLABELED_ANN_FILE = 'semi_anns/instances_train2017.1@10-unlabeled.json'
VAL_ANN_FILE = 'annotations/instances_val2017.json'
TRAIN_IMAGE_PREFIX = 'train2017/'
VAL_IMAGE_PREFIX = 'val2017/'

# ---- 主干/结构相关 ----
# 说明：DINO(R50) 本质上不是 ViT patch 化主干，但为了后续研究可切换到 ViT 风格 backbone，
# 这里预留 PATCH_SIZE 作为统一实验入口参数（当前版本不直接参与计算）。
PATCH_SIZE = 16
BACKBONE_DEPTH = 50
BACKBONE_FROZEN_STAGES = 1
NUM_QUERIES = 900
NUM_CLASSES = 80
DINO_NUM_FEATURE_LEVELS = 4

# ---- neck / encoder / decoder 关键开关 ----
USE_NECK = True
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
WITH_BOX_REFINE = True
AS_TWO_STAGE = True

# ---- teacher / student 半监督关键超参数 ----
FREEZE_TEACHER = True
SUP_WEIGHT = 1.0
UNSUP_WEIGHT = 1.0
CLS_PSEUDO_THR = 0.7
MIN_PSEUDO_BBOX_WH = (1e-2, 1e-2)

# ---- 训练/验证阶段关键配置 ----
BATCH_SIZE = 4
NUM_WORKERS = 4
SOURCE_RATIO = [1, 1]  # [labeled, unlabeled]
MAX_ITERS = 90000
VAL_INTERVAL = 5000
CHECKPOINT_INTERVAL = 5000
LOG_INTERVAL = 50

# ---- 优化器相关 ----
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_NORM = 0.1

# =========================
# 二、基础运行时配置
# =========================

_base_ = [
    '../thirdparty/mmdetection-3.3.0/configs/_base_/default_runtime.py'
]

default_scope = 'mmdet'
backend_args = None

# =========================
# 三、模型定义：SemiBaseDetector + DINO
# =========================

# 下面的 detector 是“官方 DINO 结构”，我们不重写其内部模块。
detector = dict(
    type='DINO',
    num_queries=NUM_QUERIES,
    with_box_refine=WITH_BOX_REFINE,
    as_two_stage=AS_TWO_STAGE,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=BACKBONE_DEPTH,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=BACKBONE_FROZEN_STAGES,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=DINO_NUM_FEATURE_LEVELS) if USE_NECK else None,
    encoder=dict(
        num_layers=ENCODER_LAYERS,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=DINO_NUM_FEATURE_LEVELS, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=DECODER_LAYERS,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=DINO_NUM_FEATURE_LEVELS, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=NUM_CLASSES,
        sync_cls_avg_factor=True,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
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
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300),
)

# 这里复用官方 SemiBaseDetector，把上面的 DINO 同时作为 student 与 teacher。
model = dict(
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
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
# 四、数据 pipeline（官方半监督三分支组织）
# =========================

# branch_field 决定 MultiBranch 会输出哪些分支。
branch_field = ['sup', 'unsup_teacher', 'unsup_student']

# 有监督分支：直接给 student 做监督训练。
sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoiceResize', scales=[(480, 1333), (640, 1333), (800, 1333)], keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=MIN_PSEUDO_BBOX_WH),
    dict(type='MultiBranch', branch_field=branch_field, sup=dict(type='PackDetInputs')),
]

# 无监督 teacher 分支：弱增强，主要用于稳定地产生伪标签。
weak_pipeline = [
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoiceResize', scales=[(480, 1333), (640, 1333), (800, 1333)], keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'flip', 'flip_direction', 'homography_matrix')),
]

# 无监督 student 分支：强增强，提升伪标签训练鲁棒性。
strong_pipeline = [
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoiceResize', scales=[(480, 1333), (640, 1333), (800, 1333)], keep_ratio=True),
    dict(type='RandAugment', aug_space=[[dict(type='ColorTransform')], [dict(type='Contrast')]], aug_num=1),
    dict(type='RandomErasing', n_patches=(1, 3), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=MIN_PSEUDO_BBOX_WH),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'flip', 'flip_direction', 'homography_matrix')),
]

# 无监督入口：先加载图像与空标注，再在 MultiBranch 中复制并分别走弱/强增强。
unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(type='MultiBranch', branch_field=branch_field, unsup_teacher=weak_pipeline, unsup_student=strong_pipeline),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# =========================
# 五、数据集与 DataLoader
# =========================

labeled_dataset = dict(
    type='CocoDataset',
    data_root=DATA_ROOT,
    ann_file=LABELED_ANN_FILE,
    data_prefix=dict(img=TRAIN_IMAGE_PREFIX),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=backend_args,
)

unlabeled_dataset = dict(
    type='CocoDataset',
    data_root=DATA_ROOT,
    ann_file=UNLABELED_ANN_FILE,
    data_prefix=dict(img=TRAIN_IMAGE_PREFIX),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args,
)

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceSampler', batch_size=BATCH_SIZE, source_ratio=SOURCE_RATIO),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_ROOT,
        ann_file=VAL_ANN_FILE,
        data_prefix=dict(img=VAL_IMAGE_PREFIX),
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
# 六、训练/验证/测试循环
# =========================

train_cfg = dict(type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# =========================
# 七、优化器、学习率计划、Hook
# =========================

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
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=LOG_INTERVAL),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=CHECKPOINT_INTERVAL, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

custom_hooks = [
    # MeanTeacherHook 是半监督 teacher/student 协同的关键：
    # 每个迭代使用 student 参数的 EMA 更新 teacher。
    dict(type='MeanTeacherHook'),
]

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# 自动学习率缩放基准（可按总 batch 调整）
auto_scale_lr = dict(enable=False, base_batch_size=16)

# 默认输出目录（可被 train.py 覆盖）
work_dir = './work_dirs/semi_dino_mvp'
