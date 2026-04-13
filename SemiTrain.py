"""
SemiTrain.py

作用：
- 作为半监督训练入口脚本；
- 读取配置文件；
- 构建 Runner；
- 启动训练。

设计原则：
- 保持“非常简单”，仅做训练启动胶水层逻辑；
- 训练细节（模型、数据、优化器、Hook）全部放在配置文件中管理；
- 尽量复用 MMDetection / MMEngine 官方训练流程。
"""

from pathlib import Path
import sys

from mmengine.config import Config
from mmengine.runner import Runner


# 默认配置路径：可按需改成其他配置文件。
DEFAULT_CONFIG_PATH = 'configs/SemiDino.py'


def _ensure_local_mmdet_importable() -> None:
    """确保 thirdparty 中的 mmdet 源码可被导入。

    背景：
    - 你的环境里可能没有通过 pip 安装 mmdet；
    - 但仓库内已有 thirdparty/mmdetection-3.3.0 源码；
    - 训练脚本应优先复用仓库内源码，避免依赖外部安装状态。

    做法：
    - 若当前已能 import mmdet，则不做任何处理；
    - 若不能 import，则把 thirdparty/mmdetection-3.3.0 加入 sys.path。
    """
    try:
        import mmdet  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parent
    local_mmdet_root = repo_root / 'thirdparty' / 'mmdetection-3.3.0'
    if local_mmdet_root.exists():
        sys.path.insert(0, str(local_mmdet_root))


def main() -> None:
    """训练主函数。

    逻辑：
    1) 从配置文件读取完整配置；
    2) 使用 Runner.from_cfg 根据配置实例化训练器；
    3) 调用 train() 开始训练流程。
    """
    # 先确保可导入 mmdet（优先使用 thirdparty 源码路径）。
    _ensure_local_mmdet_importable()

    # 延迟导入：确保上一步路径注入后能够稳定导入 mmdet。
    from mmdet.utils import register_all_modules

    # 显式注册 MMDetection 全部模块并初始化 default scope，
    # 避免 Runner 构建 visualizer 时出现
    # “DetLocalVisualizer 未注册 / scope=None” 报错。
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(DEFAULT_CONFIG_PATH)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
