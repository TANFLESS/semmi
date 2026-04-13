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

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


# 默认配置路径：可按需改成其他配置文件。
DEFAULT_CONFIG_PATH = 'configs/SemiDino.py'


def main() -> None:
    """训练主函数。

    逻辑：
    1) 从配置文件读取完整配置；
    2) 使用 Runner.from_cfg 根据配置实例化训练器；
    3) 调用 train() 开始训练流程。
    """
    # 显式注册 MMDetection 全部模块并初始化 default scope。
    # 目的：避免 Runner 构建 visualizer 时出现
    # “DetLocalVisualizer 未注册 / scope=None” 的报错。
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(DEFAULT_CONFIG_PATH)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
