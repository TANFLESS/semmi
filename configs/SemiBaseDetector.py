"""
SemiBaseDetector 阶段（Phase 2）最小可运行基类实现。

==============================
一、文件目标
==============================
本文件实现一个“半监督检测基类”，把常见训练流程抽象成统一模板：
1. 监督分支（有标注数据）
2. 无监督分支（伪标签数据）
3. 总损失聚合（可独立配置监督/无监督权重）

这样设计的价值是：
- SoftTeacher、Semi-DETR、后续任何 teacher-student 检测器都能复用；
- 你只需要在子类里覆写关键钩子，不必重复写训练胶水逻辑。

==============================
二、哪些是官方能力，哪些是本文件自定义
==============================
- 官方能力（后续应调用）：
  - 具体 detector 的 `loss/predict/forward`。
- 本文件自定义：
  - 半监督训练流程模板（Template Method）。
  - 损失命名、分支加权、输入结构约定。

==============================
三、关键可调参数（优先改这里）
==============================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, MutableMapping


# ===========================
# 关键参数区（建议优先改这里）
# ===========================
DEFAULT_SUP_LOSS_WEIGHT: float = 1.0
"""监督分支总权重。"""

DEFAULT_UNSUP_LOSS_WEIGHT: float = 1.0
"""无监督分支总权重。"""


@dataclass
class SemiTrainWeightConfig:
    """半监督训练中的分支权重配置。"""

    sup_weight: float = DEFAULT_SUP_LOSS_WEIGHT
    unsup_weight: float = DEFAULT_UNSUP_LOSS_WEIGHT


class SemiBaseDetector:
    """半监督检测通用基类。

    子类需要至少实现：
    - compute_sup_loss(...)
    - compute_unsup_loss(...)

    约定输入结构：
    - multi_batch_inputs: {'sup': ..., 'unsup_teacher': ..., 'unsup_student': ...}
    - multi_batch_data_samples: 同上

    这样的约定来自 MMDet 半监督数据组织方式，便于后续直连 dataloader。
    """

    def __init__(self, weight_cfg: SemiTrainWeightConfig | None = None) -> None:
        self.weight_cfg = weight_cfg or SemiTrainWeightConfig()

    # ---------------------------
    # 对外主入口
    # ---------------------------
    def loss(
        self,
        multi_batch_inputs: MutableMapping[str, Any],
        multi_batch_data_samples: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        """统一 loss 入口。

        统一入口的目的：
        - 与 MMEngine 的 runner 调用方式兼容；
        - 上层训练脚本不需要知道具体 detector 的内部细节。
        """

        sup_losses = self.compute_sup_loss(
            batch_inputs=multi_batch_inputs.get('sup'),
            batch_data_samples=multi_batch_data_samples.get('sup'),
        )
        unsup_losses = self.compute_unsup_loss(
            teacher_batch_inputs=multi_batch_inputs.get('unsup_teacher'),
            teacher_batch_data_samples=multi_batch_data_samples.get('unsup_teacher'),
            student_batch_inputs=multi_batch_inputs.get('unsup_student'),
            student_batch_data_samples=multi_batch_data_samples.get('unsup_student'),
        )

        merged = {}
        merged.update(self._prefix_and_weight(sup_losses, 'sup_', self.weight_cfg.sup_weight))
        merged.update(self._prefix_and_weight(unsup_losses, 'unsup_', self.weight_cfg.unsup_weight))
        return merged

    # ---------------------------
    # 子类必须覆写的两个核心钩子
    # ---------------------------
    def compute_sup_loss(self, batch_inputs: Any, batch_data_samples: Any) -> Dict[str, Any]:
        """监督损失计算（子类实现）。"""
        raise NotImplementedError('请在子类中实现 compute_sup_loss。')

    def compute_unsup_loss(
        self,
        teacher_batch_inputs: Any,
        teacher_batch_data_samples: Any,
        student_batch_inputs: Any,
        student_batch_data_samples: Any,
    ) -> Dict[str, Any]:
        """无监督损失计算（子类实现）。"""
        raise NotImplementedError('请在子类中实现 compute_unsup_loss。')

    # ---------------------------
    # 可复用工具函数
    # ---------------------------
    @staticmethod
    def _prefix_and_weight(loss_dict: Dict[str, Any], prefix: str, weight: float) -> Dict[str, Any]:
        """统一给损失命名加前缀并加权。

        变量解释：
        - loss_dict：某一分支的原始损失字典。
        - prefix：`sup_` 或 `unsup_`，用于日志可读性。
        - weight：分支整体缩放系数。
        """
        out: Dict[str, Any] = {}
        for k, v in loss_dict.items():
            key = f'{prefix}{k}'
            if isinstance(v, (int, float)):
                out[key] = v * weight
            else:
                out[key] = v * weight
        return out
