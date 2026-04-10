"""
SemiDino 阶段（Phase 3）最小可运行骨架实现。

==============================
一、文件目标
==============================
在 Phase 2 的 `SemiBaseDetector` 基类之上，接入 DINO 风格检测器，形成
“可训练流程通、可后续逐步细化”的半监督 DINO 方案。

本文件强调“结构正确 + 接口清晰”：
- 学生 DINO：负责监督损失和无监督损失反向传播。
- 教师 DINO：只负责预测伪标签，不参与梯度更新。
- 基类流程：负责把两条分支组合为统一 loss 输出。

==============================
二、官方/自定义边界
==============================
- 官方能力（建议后续替换为 MMDet 3.3.0 实例）：
  - DINO detector 的 `loss(...)` 和 `predict(...)`。
- 本文件自定义：
  - 伪标签筛选、样本挂载、分支衔接、日志命名。

==============================
三、关键可改参数区
==============================
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from configs.SemiBaseDetector import SemiBaseDetector, SemiTrainWeightConfig


# ===========================
# 关键参数区（建议优先改这里）
# ===========================
DEFAULT_DINO_PSEUDO_THR: float = 0.6
"""DINO 伪标签置信度阈值。"""

DEFAULT_DINO_MIN_PSEUDO: int = 0
"""每张图最少保留伪框数量。"""


@dataclass
class SemiDinoConfig:
    """SemiDino 专属超参数。"""

    pseudo_thr: float = DEFAULT_DINO_PSEUDO_THR
    min_pseudo_boxes: int = DEFAULT_DINO_MIN_PSEUDO


class SemiDinoDetector(SemiBaseDetector):
    """半监督 DINO 检测器骨架。

    参数：
    - student_dino：学生检测器对象（需提供 loss）。
    - teacher_dino：教师检测器对象（需提供 predict）。
    - weight_cfg：监督/无监督分支权重。
    - dino_cfg：SemiDino 专属参数。
    """

    def __init__(
        self,
        student_dino: Any,
        teacher_dino: Any,
        weight_cfg: SemiTrainWeightConfig | None = None,
        dino_cfg: SemiDinoConfig | None = None,
    ) -> None:
        super().__init__(weight_cfg=weight_cfg)
        self.student_dino = student_dino
        self.teacher_dino = teacher_dino
        self.dino_cfg = dino_cfg or SemiDinoConfig()

    # ---------------------------
    # 覆写基类：监督分支
    # ---------------------------
    def compute_sup_loss(self, batch_inputs: Any, batch_data_samples: Any) -> Dict[str, Any]:
        """监督分支直接调用学生 DINO 的 loss。"""
        return self.student_dino.loss(batch_inputs, batch_data_samples)

    # ---------------------------
    # 覆写基类：无监督分支
    # ---------------------------
    def compute_unsup_loss(
        self,
        teacher_batch_inputs: Any,
        teacher_batch_data_samples: Any,
        student_batch_inputs: Any,
        student_batch_data_samples: Any,
    ) -> Dict[str, Any]:
        """无监督分支：教师预测伪标签，学生学习伪标签。"""

        # 第 1 步：教师 DINO 在“弱增强”样本上做推理。
        teacher_preds = self.teacher_dino.predict(
            teacher_batch_inputs,
            teacher_batch_data_samples,
        )

        # 第 2 步：伪标签过滤（置信度筛选 + 可选保底数量）。
        teacher_preds = self._filter_teacher_predictions(teacher_preds)

        # 第 3 步：把伪标签写回“强增强”学生样本。
        student_samples = deepcopy(student_batch_data_samples)
        self._inject_pseudo_to_student(student_samples, teacher_preds)

        # 第 4 步：学生 DINO 计算无监督损失。
        return self.student_dino.loss(student_batch_inputs, student_samples)

    # ---------------------------
    # 伪标签过滤
    # ---------------------------
    def _filter_teacher_predictions(self, teacher_preds: Any) -> Any:
        """对教师预测按阈值过滤。

        兼容协议：
        - 若元素是 dict 且包含 scores，则按 scores 过滤。
        - 否则保持原样，避免提前绑定具体数据结构。
        """

        if not isinstance(teacher_preds, list):
            return teacher_preds

        filtered: List[Any] = []
        for pred in teacher_preds:
            if not isinstance(pred, dict) or 'scores' not in pred:
                filtered.append(pred)
                continue

            scores = pred['scores']
            keep = [i for i, s in enumerate(scores) if float(s) >= self.dino_cfg.pseudo_thr]
            if not keep and self.dino_cfg.min_pseudo_boxes > 0:
                keep = list(range(min(self.dino_cfg.min_pseudo_boxes, len(scores))))

            new_pred: Dict[str, Any] = {}
            for k, v in pred.items():
                if isinstance(v, Iterable) and not isinstance(v, (str, bytes, dict)):
                    new_pred[k] = [v[i] for i in keep]
                else:
                    new_pred[k] = v
            filtered.append(new_pred)

        return filtered

    # ---------------------------
    # 伪标签挂载
    # ---------------------------
    @staticmethod
    def _inject_pseudo_to_student(student_samples: Any, pseudo_preds: Any) -> None:
        """将伪标签写入学生样本。

        当前写入键为 `pseudo_instances`。
        后续对接 MMDet 时，可以改成写到 `gt_instances` 并补齐结构类型。
        """

        if not isinstance(student_samples, list) or not isinstance(pseudo_preds, list):
            return

        n = min(len(student_samples), len(pseudo_preds))
        for i in range(n):
            sample = student_samples[i]
            if isinstance(sample, dict):
                sample['pseudo_instances'] = pseudo_preds[i]
