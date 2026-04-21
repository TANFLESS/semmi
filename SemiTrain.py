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
- 尽量复用 MMDetection / MMEngine 官方训练流程；
- 兼容单卡与多卡（如 6 卡 3090）统一启动方式。
"""

import argparse
import importlib.util
from pathlib import Path
import sys

from mmengine.config import Config
from mmengine.runner import Runner


# 默认配置路径：可按需改成其他配置文件。
DEFAULT_CONFIG_PATH = 'configs/SemiDino.py'


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    说明：
    - `--launcher` 控制是否启用分布式训练；
    - 单卡场景保持默认 `none`；
    - 多卡场景（如 6 卡 3090）建议用 torchrun + `--launcher pytorch`。
    """
    parser = argparse.ArgumentParser(
        description='Semi-DINO 训练启动脚本',
        epilog=(
            '启动示例：\n'
            '  单卡: python SemiTrain.py --launcher none\n'
            '  六卡: torchrun --nproc_per_node=6 SemiTrain.py --launcher pytorch\n'
            '可选覆盖: --work-dir <输出目录> --config <配置路径>'
        ),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_PATH,
        help='配置文件路径（默认：configs/SemiDino.py）')
    parser.add_argument(
        '--work-dir',
        default='',
        help='可选：覆盖配置中的 work_dir')
    parser.add_argument(
        '--launcher',
        default='none',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        help='分布式启动器类型，多卡请使用 pytorch')
    parser.add_argument(
        '--local-rank',
        '--local_rank',
        type=int,
        default=0,
        help='分布式 local rank（torchrun 自动注入）')
    return parser.parse_args()


def _ensure_local_mmdet_importable() -> str:
    """确保 thirdparty 中的 mmdet 源码可被导入。

    背景：
    - 你的环境里可能没有通过 pip 安装 mmdet；
    - 但仓库内已有 thirdparty/mmdetection-3.3.0 源码；
    - 训练脚本应优先复用仓库内源码，避免依赖外部安装状态。

    做法：
    - 若当前已能 import mmdet，则直接使用当前环境中的 mmdet；
    - 若不能 import，则把 thirdparty/mmdetection-3.3.0 加入 sys.path；
    - 若两者都不可用，则抛出明确错误，避免后续在深层调用处才报错。

    返回：
    - 'site-packages'：表示使用环境中已安装的 mmdet；
    - 'thirdparty'：表示使用仓库内 thirdparty 源码。
    """
    if importlib.util.find_spec('mmdet') is not None:
        return 'site-packages'

    repo_root = Path(__file__).resolve().parent
    local_mmdet_root = repo_root / 'thirdparty' / 'mmdetection-3.3.0'
    if local_mmdet_root.exists():
        sys.path.insert(0, str(local_mmdet_root))
        if importlib.util.find_spec('mmdet') is not None:
            return 'thirdparty'

    raise ModuleNotFoundError(
        '无法导入 mmdet：既没有检测到已安装的 mmdet，也没有在 '
        f'{local_mmdet_root} 找到可用源码。'
    )


def main() -> None:
    """训练主函数。

    逻辑：
    1) 从配置文件读取完整配置；
    2) 使用 Runner.from_cfg 根据配置实例化训练器；
    3) 调用 train() 开始训练流程。
    """
    args = _parse_args()

    # 先确保可导入 mmdet（优先使用 thirdparty 源码路径）。
    mmdet_source = _ensure_local_mmdet_importable()

    # 延迟导入：确保上一步路径注入后能够稳定导入 mmdet。
    from mmdet.utils import register_all_modules

    # 显式注册 MMDetection 全部模块并初始化 default scope，
    # 避免 Runner 构建 visualizer 时出现
    # “DetLocalVisualizer 未注册 / scope=None” 报错。
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.work_dir:
        cfg.work_dir = args.work_dir

    print(f'[SemiTrain] launcher={cfg.launcher}, mmdet_source={mmdet_source}, work_dir={cfg.work_dir}')

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
