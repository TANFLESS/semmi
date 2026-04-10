"""
=============================
项目训练入口（MMDetection 3.x 教学版）
=============================

本文件负责什么：
- 作为本项目统一训练启动脚本，读取 `configs/SemiDino.py`；
- 在启动前集中覆盖“实验中最常改”的路径与运行参数；
- 使用 MMDetection / MMEngine 官方 Runner 风格启动训练。

它在流水线中的位置：
- 用户运行 `python train.py ...`；
- 脚本加载配置 -> 注入路径/目录/运行参数 -> 构建 Runner -> `runner.train()`。

依赖组件：
- MMEngine: Config, Runner
- MMDetection: 默认注册表（通过 import mmdet 触发）

官方复用与本项目自定义：
- 官方复用：Runner 训练入口、配置系统、work_dir/log/checkpoint/hook 机制。
- 本项目自定义：
  1) 顶部参数区（方便新手改路径与实验目录）；
  2) 自动读取 `data/ANNSPATH.md` 以覆盖半监督标注路径；
  3) 更清晰的 CLI 参数组织与说明。
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from typing import Dict, Tuple

# =========================
# 一、顶部集中可改参数区
# =========================

# 配置文件路径：默认使用当前项目的 SemiDino 最小闭环配置。
DEFAULT_CONFIG_PATH = 'configs/SemiDino.py'

# 项目根目录下的半监督标注索引文件，默认读取前两行：
# 第 1 行 = labeled ann，相对项目根目录；
# 第 2 行 = unlabeled ann，相对项目根目录。
DEFAULT_SEMI_ANNS_PATH = 'data/ANNSPATH.md'

# 数据集根目录（COCO 风格）。
DEFAULT_DATA_ROOT = 'data/coco/'

# 输出目录相关。
DEFAULT_WORK_DIR = './work_dirs/semi_dino_mvp'
DEFAULT_LOG_DIR = './work_dirs/semi_dino_mvp/logs'
DEFAULT_VIS_DIR = './work_dirs/semi_dino_mvp/vis'
DEFAULT_CKPT_DIR = './work_dirs/semi_dino_mvp/checkpoints'

# 运行参数相关。
DEFAULT_DEVICE = 'cuda'
DEFAULT_LAUNCHER = 'none'  # none / pytorch / slurm / mpi
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4

# 预训练与恢复。
DEFAULT_LOAD_FROM = ''
DEFAULT_RESUME = False


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    该函数在训练开始前调用，目的是把最常见的实验控制项暴露出来，
    包括配置路径、数据路径、输出目录、分布式 launcher、batch size 等。
    """
    parser = argparse.ArgumentParser(description='SemiDino 训练入口（MMDetection 3.x）')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='配置文件路径')
    parser.add_argument('--semi-anns-path', default=DEFAULT_SEMI_ANNS_PATH, help='半监督标注索引文件路径（ANNSPATH.md）')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT, help='数据集根目录')

    parser.add_argument('--work-dir', default=DEFAULT_WORK_DIR, help='训练输出目录（Runner work_dir）')
    parser.add_argument('--log-dir', default=DEFAULT_LOG_DIR, help='日志目录（环境变量方式提供给 logger/hook）')
    parser.add_argument('--vis-dir', default=DEFAULT_VIS_DIR, help='可视化结果目录（环境变量方式提供）')
    parser.add_argument('--ckpt-dir', default=DEFAULT_CKPT_DIR, help='checkpoint 目录（环境变量方式提供）')

    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='训练 batch size')
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS, help='DataLoader worker 数')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='运行设备，如 cuda / cpu')
    parser.add_argument('--launcher', default=DEFAULT_LAUNCHER, choices=['none', 'pytorch', 'slurm', 'mpi'], help='分布式启动方式')

    parser.add_argument('--load-from', default=DEFAULT_LOAD_FROM, help='预训练权重路径')
    parser.add_argument('--resume', action='store_true', default=DEFAULT_RESUME, help='是否从最近 checkpoint 恢复')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=None,
        help='额外配置覆盖，格式如 key=value，例如 train_cfg.max_iters=100')

    return parser.parse_args()




def _parse_cfg_options(cfg_options: list[str] | None) -> Dict[str, object]:
    """把命令行 `key=value` 列表解析为字典。"""
    import ast

    parsed: Dict[str, object] = {}
    if not cfg_options:
        return parsed

    for item in cfg_options:
        if '=' not in item:
            raise ValueError(f'--cfg-options 参数格式错误: {item}，应为 key=value')
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed_value = ast.literal_eval(value)
        except Exception:
            parsed_value = value
        parsed[key] = parsed_value
    return parsed


def _read_semi_ann_paths(anns_path_file: str) -> Tuple[str, str]:
    """读取半监督标注路径。

    调用阶段：配置加载后、正式构建 Runner 前。

    输入：
    - anns_path_file: `data/ANNSPATH.md` 文件路径。

    输出：
    - (labeled_ann_rel, unlabeled_ann_rel): 两个相对项目根目录的标注文件路径。

    说明：
    - 这里遵循项目约定：第一行是 labeled，第二行是 unlabeled。
    - 返回的是“相对路径”，后续会转换成相对于 `data_root` 的路径再写回 cfg。
    """
    if not osp.exists(anns_path_file):
        raise FileNotFoundError(f'未找到半监督标注索引文件: {anns_path_file}')

    with open(anns_path_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f'{anns_path_file} 至少需要两行（labeled / unlabeled），当前仅有 {len(lines)} 行。')

    return lines[0], lines[1]


def _to_ann_file_under_data_root(full_rel_path: str, data_root: str) -> str:
    """把“相对项目根目录路径”转换为“相对 data_root 的 ann_file”。

    例如：
    - full_rel_path = data/coco/semi_anns/instances_train2017.1@10.json
    - data_root = data/coco/
    - 返回 semi_anns/instances_train2017.1@10.json

    这样可以与 MMDetection 的 `dataset.data_root + ann_file` 组合逻辑保持一致。
    """
    normalized_full = osp.normpath(full_rel_path)
    normalized_root = osp.normpath(data_root)

    if not normalized_full.startswith(normalized_root):
        raise ValueError(
            f'标注路径 {full_rel_path} 不在 data_root={data_root} 下，'
            '请检查 ANNSPATH.md 与 --data-root 是否一致。')

    rel = osp.relpath(normalized_full, normalized_root)
    return rel.replace('\\', '/')


def _ensure_dirs(paths: Dict[str, str]) -> None:
    """创建必要输出目录，避免日志/可视化/检查点写入时报错。"""
    for p in paths.values():
        os.makedirs(p, exist_ok=True)


def main() -> None:
    """训练主函数。

    调用顺序：
    1) 解析参数；
    2) 导入 mmdet/mmengine 并读取 cfg；
    3) 覆盖路径与运行参数；
    4) 构建 Runner 并开始训练。
    """
    args = parse_args()

    # 把本项目内置的 mmdetection 源码加入导入路径，避免环境中版本不一致。
    repo_mmdet_path = osp.abspath('thirdparty/mmdetection-3.3.0')
    if repo_mmdet_path not in sys.path:
        sys.path.insert(0, repo_mmdet_path)

    # 导入顺序说明：
    # - 先 import mmdet，可触发注册机制，保证 registry 内组件可用；
    # - 再从 mmengine 读取 Config/Runner。
    import mmdet  # noqa: F401
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)

    # 从 ANNSPATH.md 读取半监督标注路径，并写回 cfg。
    labeled_full_rel, unlabeled_full_rel = _read_semi_ann_paths(args.semi_anns_path)
    labeled_ann_file = _to_ann_file_under_data_root(labeled_full_rel, args.data_root)
    unlabeled_ann_file = _to_ann_file_under_data_root(unlabeled_full_rel, args.data_root)

    # 覆盖数据路径和 batch 参数。
    cfg.DATA_ROOT = args.data_root
    cfg.LABELED_ANN_FILE = labeled_ann_file
    cfg.UNLABELED_ANN_FILE = unlabeled_ann_file

    cfg.labeled_dataset.data_root = args.data_root
    cfg.labeled_dataset.ann_file = labeled_ann_file
    cfg.unlabeled_dataset.data_root = args.data_root
    cfg.unlabeled_dataset.ann_file = unlabeled_ann_file

    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.num_workers = args.num_workers
    cfg.train_dataloader.sampler.batch_size = args.batch_size

    # 覆盖验证集路径。
    cfg.val_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.data_root = args.data_root

    # 训练输出目录。
    cfg.work_dir = args.work_dir

    # 预训练与恢复设置。
    if args.load_from:
        cfg.load_from = args.load_from
    cfg.resume = bool(args.resume)

    # 启动设备与 launcher。
    cfg.launcher = args.launcher
    cfg.device = args.device

    # 提供日志/可视化/checkpoint 目录给下游组件（通过环境变量透传）。
    os.environ['SEMMI_LOG_DIR'] = args.log_dir
    os.environ['SEMMI_VIS_DIR'] = args.vis_dir
    os.environ['SEMMI_CKPT_DIR'] = args.ckpt_dir
    _ensure_dirs({'work_dir': args.work_dir, 'log_dir': args.log_dir, 'vis_dir': args.vis_dir, 'ckpt_dir': args.ckpt_dir})

    # 允许额外命令行覆盖（优先级最高）。
    parsed_cfg_options = _parse_cfg_options(args.cfg_options)
    if parsed_cfg_options:
        cfg.merge_from_dict(parsed_cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    # 这里不做复杂异常吞噬，方便新手直接看到完整报错栈并定位问题。
    main()
