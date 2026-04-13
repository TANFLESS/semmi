"""
=============================
项目训练入口（MMDetection 3.x 教学版）
=============================

本文件负责：
- 读取 `configs/SemiDino.py` 并构建 Runner；
- 提供集中可改参数区（配置、路径、输出目录、设备等）；
- 尽量少写重复覆盖逻辑，优先改配置里的集中参数。
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
DEFAULT_CONFIG_PATH = 'configs/SemiDino.py'
DEFAULT_SEMI_ANNS_PATH = 'data/ANNSPATH.md'
DEFAULT_DATA_ROOT = 'data/coco/'
DEFAULT_WORK_DIR = './work_dirs/semi_dino_mvp'
DEFAULT_LOG_DIR = './work_dirs/semi_dino_mvp/logs'
DEFAULT_VIS_DIR = './work_dirs/semi_dino_mvp/vis'
DEFAULT_CKPT_DIR = './work_dirs/semi_dino_mvp/checkpoints'
DEFAULT_DEVICE = 'cuda'
DEFAULT_LAUNCHER = 'none'
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_LOAD_FROM = ''
DEFAULT_RESUME = False


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='SemiDino 训练入口（MMDetection 3.x）')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='配置文件路径')
    parser.add_argument('--semi-anns-path', default=DEFAULT_SEMI_ANNS_PATH, help='半监督标注索引文件路径（ANNSPATH.md）')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT, help='数据集根目录')
    parser.add_argument('--work-dir', default=DEFAULT_WORK_DIR, help='训练输出目录（Runner work_dir）')
    parser.add_argument('--log-dir', default=DEFAULT_LOG_DIR, help='日志目录')
    parser.add_argument('--vis-dir', default=DEFAULT_VIS_DIR, help='可视化目录')
    parser.add_argument('--ckpt-dir', default=DEFAULT_CKPT_DIR, help='checkpoint 目录')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='训练 batch size')
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS, help='DataLoader worker 数')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='运行设备，例如 cuda / cpu')
    parser.add_argument('--launcher', default=DEFAULT_LAUNCHER, choices=['none', 'pytorch', 'slurm', 'mpi'], help='分布式启动方式')
    parser.add_argument('--load-from', default=DEFAULT_LOAD_FROM, help='预训练权重路径')
    parser.add_argument('--resume', action='store_true', default=DEFAULT_RESUME, help='是否从最近 checkpoint 恢复')
    parser.add_argument('--cfg-options', nargs='+', default=None, help='额外配置覆盖，格式 key=value')
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
    """读取半监督标注路径（前两行分别为 labeled 与 unlabeled）。"""
    if not osp.exists(anns_path_file):
        raise FileNotFoundError(f'未找到半监督标注索引文件: {anns_path_file}')
    with open(anns_path_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        raise ValueError(f'{anns_path_file} 至少需要两行（labeled / unlabeled），当前仅有 {len(lines)} 行。')
    return lines[0], lines[1]


def _to_ann_file_under_data_root(full_rel_path: str, data_root: str) -> str:
    """将“相对项目根目录路径”转换为“相对 data_root 的 ann_file”。"""
    normalized_full = osp.normpath(full_rel_path)
    normalized_root = osp.normpath(data_root)
    if not normalized_full.startswith(normalized_root):
        raise ValueError(f'标注路径 {full_rel_path} 不在 data_root={data_root} 下，请检查 ANNSPATH.md。')
    return osp.relpath(normalized_full, normalized_root).replace('\\', '/')


def _ensure_dirs(paths: Dict[str, str]) -> None:
    for p in paths.values():
        os.makedirs(p, exist_ok=True)


def _apply_runtime_overrides(cfg, args: argparse.Namespace, labeled_ann_file: str, unlabeled_ann_file: str) -> None:
    """统一写入运行期覆盖项，减少 train.py 与配置文件重复。"""
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

    cfg.val_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.data_root = args.data_root

    cfg.work_dir = args.work_dir
    if args.load_from:
        cfg.load_from = args.load_from
    cfg.resume = bool(args.resume)
    cfg.launcher = args.launcher
    cfg.device = args.device


def main() -> None:
    args = parse_args()

    repo_mmdet_path = osp.abspath('thirdparty/mmdetection-3.3.0')
    if repo_mmdet_path not in sys.path:
        sys.path.insert(0, repo_mmdet_path)

    import mmdet  # noqa: F401
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)

    labeled_full_rel, unlabeled_full_rel = _read_semi_ann_paths(args.semi_anns_path)
    labeled_ann_file = _to_ann_file_under_data_root(labeled_full_rel, args.data_root)
    unlabeled_ann_file = _to_ann_file_under_data_root(unlabeled_full_rel, args.data_root)

    _apply_runtime_overrides(cfg, args, labeled_ann_file, unlabeled_ann_file)

    os.environ['SEMMI_LOG_DIR'] = args.log_dir
    os.environ['SEMMI_VIS_DIR'] = args.vis_dir
    os.environ['SEMMI_CKPT_DIR'] = args.ckpt_dir
    _ensure_dirs({'work_dir': args.work_dir, 'log_dir': args.log_dir, 'vis_dir': args.vis_dir, 'ckpt_dir': args.ckpt_dir})

    parsed_cfg_options = _parse_cfg_options(args.cfg_options)
    if parsed_cfg_options:
        cfg.merge_from_dict(parsed_cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
