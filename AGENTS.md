# 工作记录

## 第一次工作（2026-04-13）

### 用户原始提示词（完整记录）
我正从事半监督目标检测的相关研究。目前半监督目标检测相关研究中，可靠的SOTA还是semi-detr，但是semi-detr使用mm 2.0版本实现了，我目前打算在mm 3.0当中，再一次实现semi-detr。\n目前的计划是先使用SemiBaseDetector包裹一个DINO构建一个能跑通全流程的基本模型。\n代码要参考官方教程（https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/semi_det.html），尽可能的使用官方实现，只修改必要的地方，复用mm官方的SemiBaseDetector和DINO，注意DINO构建两次会更改配置文件的问题。\nthirdpart文件夹用来放参考代码，目前里面放的是mm detection 3.3.0的源代码和semi-detr的源代码。\nSemiTrain.py是启动训练的脚本代码应该在的位置，这个训练脚本应该非常简单，导入config然后开始训练就完事了。\nconfigs文件夹下的SemiDino.py用来放模型的主体代码。SemiDino的代码顶端应该有一些方便修改配置和关键参数的地方，包括但不限于patch size、轮数、训练集的路径、测试集的路径、检查点和可视化数据和训练数据的保存路径等等等等。\n你的代码当中应该有非常详细的注释解释代码的运行逻辑，详尽到每个变量每个函数都是干什么用的，代码开头应该有代码整体的运行逻辑以及完成整个网络需要哪些组件的说明。\n你应该将我的提示词和你的汇报完整的添加到AGENTS.md当中以供以后参考，这是我们的第一次交流，记录为第一次工作。\n你所有的注释和写入md文件的内容都应该是中文。

### 本次实现汇报（完整记录）
1. 已在 `configs/SemiDino.py` 中完成基于 **MMDetection 3.3.0 官方组件** 的半监督 DINO 配置：
   - 使用官方 `SemiBaseDetector` 封装官方 `DINO`；
   - 使用官方 `MultiBranchDataPreprocessor`、`GroupMultiSourceSampler`、`MeanTeacherHook`、`TeacherStudentValLoop`；
   - 训练流程采用 sup / unsup_teacher / unsup_student 三分支；
   - 配置中包含数据集路径、迭代轮数、patch/pad 粒度、批大小、采样比、阈值、权重、checkpoint 保存、可视化输出、work_dir 等可调参数入口。

2. 已在 `SemiTrain.py` 中实现“极简训练启动脚本”：
   - 读取 `configs/SemiDino.py`；
   - `Runner.from_cfg(cfg)` 构建训练器；
   - `runner.train()` 启动训练。

3. 关于“DINO 构建两次会更改配置文件”问题：
   - 该问题在官方 mmdet 3.3.0 的 `SemiBaseDetector` 中已经通过对 `detector` 配置做深拷贝后分别构建 teacher/student 来规避；
   - 本次实现未重写该机制，仅复用官方逻辑。

4. 代码注释与文档要求：
   - 所有新增注释均为中文；
   - 配置文件开头添加了整体运行逻辑、关键组件说明、实现注意事项；
   - 关键变量与关键段落均提供中文用途说明。

5. 本次目标定位：
   - 先搭建一个“能跑通全流程”的官方化 baseline（SemiBaseDetector + DINO）；
   - 为后续对齐/迁移 Semi-DETR 细节（损失项、伪标签策略、query 策略等）提供稳定起点。

## 第二次工作（2026-04-13）

### 用户原始提示词（完整记录）
我强调了要尽量复用官方已有的组件，你没有调用mm官方实现了的DINO，也没有调用mm官方的semi config，改正这个问题。\n你应该将我的提示词和你的汇报完整的添加到AGENTS.md当中以供以后参考，这是我们的第二次交流，记录为第二次工作。

### 本次实现汇报（完整记录）
1. 已按你的要求将 `configs/SemiDino.py` 改为“**优先复用 MMDetection 3.3.0 官方配置**”的实现方式：
   - 通过 `_base_` 直接继承官方 `dino/dino-4scale_r50_8xb2-12e_coco.py`，不再手写 DINO 结构细节；
   - 通过 `_base_` 直接继承官方 `configs/_base_/datasets/semi_coco_detection.py`，不再手写半监督数据增强主结构；
   - 仅对必要参数做覆盖（类别数、预训练权重、pad 对齐粒度、数据路径、迭代轮数、阈值与损失权重等）。

2. 半监督模型封装保持官方路径：
   - 继续使用官方 `SemiBaseDetector` 封装 `detector = _base_.model`（即官方 DINO）；
   - 使用官方 `MultiBranchDataPreprocessor`；
   - 使用官方 `IterBasedTrainLoop`、`TeacherStudentValLoop`、`MeanTeacherHook`。

3. 关于你关心的“DINO 构建两次修改配置”问题，本次仍完全依赖官方机制：
   - 不自定义 teacher/student 构建逻辑；
   - 由官方 `SemiBaseDetector` 内部深拷贝流程处理该问题。

4. `AGENTS.md` 已按你的要求追加本次（第二次）完整“用户提示词 + 实现汇报”记录，便于后续持续迭代追踪。

## 第三次工作（2026-04-13）

### 用户原始提示词（完整记录）
Traceback (most recent call last):\n  File "C:\Users\tanfl\Desktop\codes\semmi\SemiTrain.py", line 38, in <module>\n    main()\n  File "C:\Users\tanfl\Desktop\codes\semmi\SemiTrain.py", line 32, in main\n    cfg = Config.fromfile(DEFAULT_CONFIG_PATH)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "C:\Users\tanfl\.conda\envs\mm\Lib\site-packages\mmengine\config\config.py", line 461, in fromfile\n    cfg_dict, cfg_text, env_variables = Config._file2dict(\n                                        ^^^^^^^^^^^^^^^^^^\n  File "C:\Users\tanfl\.conda\envs\mm\Lib\site-packages\mmengine\config\config.py", line 947, in _file2dict\n    raise e\n  File "C:\Users\tanfl\.conda\envs\mm\Lib\site-packages\mmengine\config\config.py", line 899, in _file2dict\n    raise KeyError(\nKeyError: "Duplicate key is not allowed among bases. Duplicate keys: {'test_pipeline', 'dataset_type', 'test_evaluator', 'val_evaluator', 'val_dataloader', 'train_dataloader', 'data_root', 'test_dataloader', 'backend_args'}"


解释并解决这个报错，记录为第三次工作。
\n

### 本次实现汇报（完整记录）
1. 已定位报错根因：`configs/SemiDino.py` 将“完整 DINO 配置”和“semi_coco_detection 数据基座”同时放入 `_base_`，两者都定义了 `dataset_type`、`data_root`、`train_dataloader`、`val_dataloader` 等同名顶层键，触发 mmengine 的 base 合并重复键校验。

2. 已完成修复并保持“尽量复用官方实现”：
   - `_base_` 仅保留官方 DINO 配置：`dino-4scale_r50_8xb2-12e_coco.py`；
   - 通过 `Config.fromfile(...)` 单独读取官方 `semi_coco_detection.py`；
   - 使用 `deepcopy` 复用其 `labeled_dataset` / `unlabeled_dataset`，再按实验路径做必要覆盖。

3. 为避免与 DINO base 中已有 dataloader 发生残留合并冲突：
   - 在 `train_dataloader` 与 `val_dataloader` 覆盖时增加 `_delete_=True`，确保完整替换。

4. 修复后效果：
   - 消除 `Duplicate key is not allowed among bases` 错误来源；
   - 继续保持“检测器走官方 DINO、半监督数据流复用官方 semi config”的实现路线，满足后续对齐 Semi-DETR 的基线要求。
