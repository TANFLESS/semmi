# SemiDino 最小可运行闭环实现计划（MMDetection 3.x）

## 0. 文档定位与阅读方式

本文档是当前阶段的“先计划、后编码”总设计说明，目标读者默认是**刚接触半监督目标检测（SSOD）与 MMDetection 3.x 的新手**。因此我会把关键调用链、数据结构、组件职责、可复用点与最小实现边界写清楚，再据此推进 `configs/SemiDino.py` 与 `train.py`。

---

## 1. 整体目标是什么

当前目标不是一次性复刻 Semi-DETR 的全部高级技巧，而是先构建一个能够在 MMDetection 3.x 下跑通的最小闭环：

1. 使用官方 `SemiBaseDetector` 作为 teacher/student 双分支框架；
2. 使用官方 `DINO` 作为 teacher 与 student 的基础 detector；
3. 跑通监督分支（有标注）+ 无监督分支（伪标签）+ teacher/student 协同；
4. 跑通训练（含反向传播）、验证、推理；
5. 保持配置化与可扩展性，便于后续逐步靠近 Semi-DETR 思路。

**最核心原则：能复用官方就不重写。**

---

## 2. 从输入到反向传播的完整调用链分析（问题 1）

### 2.1 训练入口层

- 训练入口由 `train.py` 读取 `configs/SemiDino.py`，再通过 `Runner.from_cfg(cfg).train()` 启动。
- 数据加载器输出的是多源拼接后的 batch：
  - `sup` 分支：有标注图像；
  - `unsup_teacher` 分支：无标注图像弱增强视图；
  - `unsup_student` 分支：同一无标注图像强增强视图。

### 2.2 模型层总入口

- `model.type='SemiBaseDetector'` 时，训练阶段调用 `SemiBaseDetector.loss(...)`。
- `loss` 内部流程：
  1. `loss_by_gt_instances(sup)`：student 走标准 DINO 监督 loss；
  2. `get_pseudo_instances(unsup_teacher)`：teacher 前向预测伪标签；
  3. `project_pseudo_instances(...)`：把 teacher 视角伪框映射到 student 视角；
  4. `loss_by_pseudo_instances(unsup_student)`：student 以伪标签作为 GT 计算无监督 loss；
  5. loss 字典按 `sup_`、`unsup_` 前缀汇总并加权。

### 2.3 反向传播层

- `SemiBaseDetector` 默认 `freeze_teacher=True`，teacher 参数不参与梯度。
- 仅 student 参数参与反向传播。
- `MeanTeacherHook` 每次迭代后用 EMA 将 student 权重同步给 teacher。

---

## 3. 监督分支与无监督分支如何组织（问题 2）

### 3.1 官方推荐组织方式

沿用 MMDetection 3.x 官方半监督范式：

1. 数据集采用 `ConcatDataset([labeled_dataset, unlabeled_dataset])`；
2. 采样器用 `GroupMultiSourceSampler`，通过 `source_ratio` 控制有/无标注采样比例；
3. pipeline 中使用 `MultiBranch` 把样本分发到 `sup / unsup_teacher / unsup_student` 三个字段；
4. `MultiBranchDataPreprocessor` 统一处理多分支 batch 张量。

### 3.2 本项目采用策略

- 保持与官方 SoftTeacher 的结构一致，只将基础 detector 改为 DINO。
- 这样可以最大化复用 `SemiBaseDetector` 的通用逻辑，降低自定义复杂度。

---

## 4. teacher 与 student 的职责（问题 3）

1. **student（可训练）**
   - 接收 `sup` 样本计算标准监督损失；
   - 接收 `unsup_student` 样本并结合伪标签计算无监督损失；
   - 参数通过优化器更新。

2. **teacher（EMA 模型）**
   - 接收 `unsup_teacher` 弱增强样本产出伪标签；
   - 默认在验证与推理阶段可作为主预测分支（`semi_test_cfg.predict_on='teacher'`）；
   - 参数由 `MeanTeacherHook` 从 student 平滑更新。

---

## 5. 需要保持一致的数据结构与预测结构（问题 4）

必须遵循 MMDetection 的 `DetDataSample` 与 `InstanceData` 约定：

1. 输入元信息至少保留：`img_shape`、`ori_shape`、`scale_factor`、`flip`、`homography_matrix` 等；
2. teacher 输出伪标签写入 `data_sample.gt_instances`；
3. `gt_instances` 中关键字段包括 `bboxes`、`labels`、`scores`（伪标签筛选要用）；
4. 推理/验证输出维持官方 `pred_instances` 格式，避免破坏 evaluator。

---

## 6. DINO 已提供能力与最小桥接逻辑（问题 5、6）

### 6.1 DINO 已提供能力（官方复用）

- Backbone/Neck/Encoder/Decoder 与 detection head；
- 标准监督训练 loss；
- 标准推理接口（`predict`）；
- 与 MMEngine Runner 兼容的完整训练/验证链路。

### 6.2 接入半监督的最小桥接

本阶段**不新增模型类**，直接依赖 `SemiBaseDetector` 现成方法：

- `get_pseudo_instances`：teacher 调 `predict` 产出伪标签；
- `project_pseudo_instances`：利用 `homography_matrix` 跨增强视角映射框；
- `loss_by_pseudo_instances`：student 把伪标签当 GT 训练并按阈值过滤。

结论：最小版本只需“配置正确”，无需重写 DINO 模块。

---

## 7. 推理与验证是否可直接复用（问题 7）

可以。

1. `SemiBaseDetector.predict` 支持按 `semi_test_cfg.predict_on` 切 teacher 或 student；
2. 验证 loop 采用 `TeacherStudentValLoop`（官方）；
3. 测试 loop 采用 `TestLoop`（官方）；
4. evaluator 仍用 `CocoMetric`，不需要额外适配。

---

## 8. loss 汇总、权重控制、伪标签筛选放在哪一层（问题 8）

1. **loss 汇总层次**：`SemiBaseDetector.loss` 内部统一聚合；
2. **监督/无监督总权重**：`semi_train_cfg.sup_weight`、`semi_train_cfg.unsup_weight`；
3. **伪标签分类阈值**：`semi_train_cfg.cls_pseudo_thr`；
4. **伪框最小尺寸过滤**：`semi_train_cfg.min_pseudo_bbox_wh`；
5. **teacher EMA 策略**：`custom_hooks=[MeanTeacherHook]` 管理。

---

## 9. 配置解决 vs 新增代码解决（问题 9）

### 9.1 通过配置解决（当前阶段优先）

- 模型装配：`SemiBaseDetector + DINO`；
- 三分支数据 pipeline；
- 有/无标注路径、标注文件、采样比例；
- 训练循环、优化器、调度器、hook；
- 推理使用 teacher 或 student。

### 9.2 需要新增代码的场景（当前暂不做）

- DINO 专用更复杂伪标签筛选策略（如 class-aware/box-aware 动态阈值）；
- 额外一致性损失（decoder token level / feature level）；
- 与 Semi-DETR 高级机制相关的专用 head/assigner 改造。

这些先列入后续扩展，不阻塞最小闭环。

---

## 10. 最小可运行版本边界（问题 10）

本次 MVP（最小可运行版本）以“闭环可跑通”为验收边界：

1. 模型可构建成功（`SemiBaseDetector` 包裹 `DINO`）；
2. dataloader 能产出三分支 batch；
3. 单次 train iter 能完成前向与反向；
4. val/test 能调用官方 loop 完整执行；
5. 路径与关键超参数集中可配置；
6. 代码注释和文档中文、详细、面向新手。

不在本次边界内：Semi-DETR 所有高级策略的一次性复刻。

---

## 11. 实施步骤（必须按顺序）

### 步骤 A：先完善计划（当前文档）

- 完成调用链、复用分析、扩展边界说明。

### 步骤 B：实现 `configs/SemiDino.py`

- 顶部集中放置关键参数（patch size、backbone、teacher/student 超参数、损失权重、伪标签阈值、路径等）；
- 组装 `SemiBaseDetector + DINO`；
- 配置三分支 pipeline、dataloader、loop、hook；
- 使用 `data/ANNSPATH.md` 对应的半监督标注文件。

### 步骤 C：实现 `train.py`

- 顶部集中放置可改参数（配置路径、数据路径、日志、可视化、checkpoint、resume、预训练、device/launcher 等）；
- 按官方风格用 `Config.fromfile` + `Runner.from_cfg` 启动；
- 支持 CLI 覆盖与输出目录定制。

### 步骤 D：基础校验

- 语法检查；
- 配置解析检查（不依赖完整数据集即可进行静态验证）。

---

## 12. 每步完成后的验证方法

1. **配置文件层验证**：
   - `python -m py_compile configs/SemiDino.py train.py`。
2. **训练入口层验证**：
   - `python train.py --help`（验证参数结构与脚本可运行性）。
3. **配置装载验证**：
   - 在脚本中打印或检查关键 cfg 字段（模型类型、ann_file、work_dir 等）。
4. **后续真实训练验证（依赖数据）**：
   - 小迭代 smoke test（如 max_iters=10）确认 train/val/backward 全链路通。

---

## 13. 官方复用点与自定义点清单

### 13.1 官方复用

- `SemiBaseDetector`
- `DINO`
- `MultiBranchDataPreprocessor`
- `GroupMultiSourceSampler`
- `TeacherStudentValLoop`
- `MeanTeacherHook`
- `CocoMetric`

### 13.2 本项目新增/覆盖（当前最小版本）

- 新增 `configs/SemiDino.py`：用于把官方模块按本项目目标组合；
- 新增 `train.py`：提供路径友好、实验友好的启动入口；
- 暂不新增自定义 detector 类，减少维护成本。

---

## 14. 后续扩展项（本次先不实现）

1. DINO 专用伪标签质量建模（动态阈值/类别自适应）；
2. teacher-student 多层特征一致性正则；
3. 更贴近 Semi-DETR 的 query 级监督策略；
4. 更完善的实验管理（自动记录阈值与伪标签统计）。

这些扩展都应先更新 `PLAN.md`，再实施代码。
