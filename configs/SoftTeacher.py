"""
SoftTeacher 阶段（Phase 1）最小可运行骨架实现。

==============================
一、这个文件解决什么问题
==============================
本文件实现一个“教师-学生（Teacher-Student）”半监督检测训练骨架，目标是先把
流程跑通（Make it run），而不是一次性做到算法细节最优。

核心链路如下：
1. 监督分支：学生模型直接使用有标注数据计算监督损失。
2. 无监督分支：
   - 教师模型在弱增强图像上预测伪标签；
   - 根据置信度阈值过滤伪标签；
   - 将过滤后的伪标签作为强增强图像的“临时标注”；
   - 学生模型据此计算无监督损失。
3. 最终损失：监督损失 + 无监督损失（乘权重）。

==============================
二、与 MMDetection 3.x 的关系
==============================
- 官方实现（期望调用）：
  - 真正的检测器 forward / loss / predict（例如 RTMDet、DINO、Faster R-CNN 等）
  - 数据样本结构（如 DetDataSample）
- 本文件自定义内容：
  - SoftTeacher 训练流程编排
  - 伪标签筛选、损失字典前缀处理、权重缩放等“半监督胶水逻辑”

说明：本文件为了在当前仓库“先跑通结构”而写成框架无关风格，
后续接入 MMEngine/MMDet 时，只需要把 student/teacher 对象替换为官方模型实例即可。

==============================
三、你最常改的关键参数（集中在这里）
==============================
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional


# ===========================
# 关键参数区（建议优先改这里）
# ===========================
DEFAULT_PSEUDO_SCORE_THR: float = 0.7
"""伪标签置信度阈值。值越高，伪标签更干净但数量更少。"""

DEFAULT_UNSUP_WEIGHT: float = 1.0
"""无监督损失总权重。训练不稳定时可先减小到 0.25 或 0.5。"""

DEFAULT_SUP_WEIGHT: float = 1.0
"""监督损失总权重。通常保持 1.0。"""

DEFAULT_TEACHER_MOMENTUM: float = 0.999
"""EMA 更新教师参数的动量。越接近 1 越平滑。"""

DEFAULT_MIN_PSEUDO_BOXES: int = 0
"""每张图至少保留多少伪框。0 代表不强制。"""


@dataclass
class SoftTeacherConfig:
    """SoftTeacher 关键超参数配置。

    该 dataclass 的设计目的：
    - 让训练关键参数显式、可序列化、可打印；
    - 在后续接入配置系统（如 mmengine Config）时也能直接映射。
    """

    pseudo_score_thr: float = DEFAULT_PSEUDO_SCORE_THR
    unsup_weight: float = DEFAULT_UNSUP_WEIGHT
    sup_weight: float = DEFAULT_SUP_WEIGHT
    teacher_momentum: float = DEFAULT_TEACHER_MOMENTUM
    min_pseudo_boxes: int = DEFAULT_MIN_PSEUDO_BOXES


class SoftTeacher:
    """SoftTeacher 训练流程编排器。

    参数说明：
    - student_detector：学生检测器（需要提供 `loss(...)` 方法）。
    - teacher_detector：教师检测器（需要提供 `predict(...)` 方法）。
    - cfg：SoftTeacher 超参数。

    约定输入格式（便于 train.py 后续统一）：
    - multi_batch_inputs: {
        'sup': Any,
        'unsup_teacher': Any,
        'unsup_student': Any,
      }
    - multi_batch_data_samples: {
        'sup': Any,
        'unsup_teacher': Any,
        'unsup_student': Any,
      }

    说明：Any 是为了兼容你后续接入 MMDet 的 DetDataSample。
    """

    def __init__(
        self,
        student_detector: Any,
        teacher_detector: Any,
        cfg: Optional[SoftTeacherConfig] = None,
    ) -> None:
        self.student_detector = student_detector
        self.teacher_detector = teacher_detector
        self.cfg = cfg or SoftTeacherConfig()

    # ---------------------------
    # 对外主入口：一个训练 step 的损失
    # ---------------------------
    def loss(
        self,
        multi_batch_inputs: MutableMapping[str, Any],
        multi_batch_data_samples: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        """计算总损失。

        返回值：
        - dict，包含监督、无监督、总损失等键。
        """

        # 1) 监督损失：学生直接看带标签数据。
        sup_losses = self._compute_sup_loss(
            batch_inputs=multi_batch_inputs.get('sup'),
            batch_data_samples=multi_batch_data_samples.get('sup'),
        )

        # 2) 无监督损失：教师产伪标签，学生学习伪标签。
        unsup_losses = self._compute_unsup_loss(
            teacher_batch_inputs=multi_batch_inputs.get('unsup_teacher'),
            teacher_batch_data_samples=multi_batch_data_samples.get('unsup_teacher'),
            student_batch_inputs=multi_batch_inputs.get('unsup_student'),
            student_batch_data_samples=multi_batch_data_samples.get('unsup_student'),
        )

        # 3) 合并并加权。
        merged_losses: Dict[str, Any] = {}
        merged_losses.update(self._scale_loss_dict(sup_losses, self.cfg.sup_weight, 'sup_'))
        merged_losses.update(self._scale_loss_dict(unsup_losses, self.cfg.unsup_weight, 'unsup_'))
        return merged_losses

    # ---------------------------
    # 内部：监督分支
    # ---------------------------
    def _compute_sup_loss(self, batch_inputs: Any, batch_data_samples: Any) -> Dict[str, Any]:
        """学生模型监督分支损失。

        这里直接调用 student_detector.loss，保持与官方检测器一致的调用方式。
        """
        return self.student_detector.loss(batch_inputs, batch_data_samples)

    # ---------------------------
    # 内部：无监督分支
    # ---------------------------
    def _compute_unsup_loss(
        self,
        teacher_batch_inputs: Any,
        teacher_batch_data_samples: Any,
        student_batch_inputs: Any,
        student_batch_data_samples: Any,
    ) -> Dict[str, Any]:
        """无监督损失计算。

        流程：
        1. 教师预测伪标签。
        2. 伪标签按阈值过滤。
        3. 伪标签写入学生无监督样本（通常写到 gt_instances）。
        4. 学生用强增强图像 + 伪标签计算损失。
        """

        teacher_predictions = self.teacher_detector.predict(
            teacher_batch_inputs,
            teacher_batch_data_samples,
        )

        filtered_predictions = self._filter_pseudo_instances(teacher_predictions)

        # 深拷贝，避免破坏 dataloader 原始样本。
        student_samples_with_pseudo = deepcopy(student_batch_data_samples)
        self._attach_pseudo_to_student_samples(
            student_samples=student_samples_with_pseudo,
            pseudo_predictions=filtered_predictions,
        )

        return self.student_detector.loss(student_batch_inputs, student_samples_with_pseudo)

    # ---------------------------
    # 内部：伪标签过滤
    # ---------------------------
    def _filter_pseudo_instances(self, teacher_predictions: Any) -> Any:
        """按 score 阈值过滤教师预测结果。

        这里采用“约定优先”策略：
        - 若元素是 dict，且包含 `scores`，则按 scores 过滤；
        - 否则直接原样返回（保持最大兼容）。

        这样做是为了让你能逐步替换成 MMDet 的 InstanceData 精细操作。
        """

        if not isinstance(teacher_predictions, list):
            return teacher_predictions

        filtered: List[Any] = []
        for pred in teacher_predictions:
            if not isinstance(pred, dict) or 'scores' not in pred:
                filtered.append(pred)
                continue

            scores = pred['scores']
            keep_indices = [
                idx for idx, s in enumerate(scores)
                if float(s) >= self.cfg.pseudo_score_thr
            ]

            # 保底策略：若过滤后一个都没有，可按 min_pseudo_boxes 强制保留前 k 个。
            if not keep_indices and self.cfg.min_pseudo_boxes > 0:
                keep_indices = list(range(min(self.cfg.min_pseudo_boxes, len(scores))))

            filtered_pred = {}
            for key, value in pred.items():
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                    filtered_pred[key] = [value[i] for i in keep_indices]
                else:
                    filtered_pred[key] = value
            filtered.append(filtered_pred)
        return filtered

    # ---------------------------
    # 内部：把伪标签写入学生样本
    # ---------------------------
    def _attach_pseudo_to_student_samples(self, student_samples: Any, pseudo_predictions: Any) -> None:
        """将教师伪标签挂到学生样本上。

        当前使用通用字典协议：
        - student_samples 是 list[dict] 时，写入 `pseudo_instances`。

        你后续接入 MMDet 时，可把这里改成：
        - sample.gt_instances = pseudo_instance_data
        """

        if not isinstance(student_samples, list) or not isinstance(pseudo_predictions, list):
            return

        pair_count = min(len(student_samples), len(pseudo_predictions))
        for i in range(pair_count):
            sample = student_samples[i]
            if isinstance(sample, dict):
                sample['pseudo_instances'] = pseudo_predictions[i]

    # ---------------------------
    # 工具函数：损失字典加权+加前缀
    # ---------------------------
    @staticmethod
    def _scale_loss_dict(loss_dict: Dict[str, Any], weight: float, prefix: str) -> Dict[str, Any]:
        """给损失字典统一加前缀，并对数值项乘权重。"""

        output: Dict[str, Any] = {}
        for key, value in loss_dict.items():
            new_key = f'{prefix}{key}'
            if isinstance(value, (int, float)):
                output[new_key] = value * weight
            else:
                # 对 tensor 等对象，假设支持乘法。
                output[new_key] = value * weight
        return output
