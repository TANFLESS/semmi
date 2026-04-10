# semmi

## 当前详细计划（先规划，再编码）

> 说明：本次任务对应三阶段目标（SoftTeacher、SemiBaseDetector、SemiDino）。
> 下面先给出完整执行步骤，再开始写代码。

### 步骤 1：梳理边界与输入输出（Phase 1~3 通用）
1. 明确三个文件的职责边界：
   - `configs/SoftTeacher.py`：实现可运行的教师-学生伪标签训练骨架。
   - `configs/SemiBaseDetector.py`：实现可复用的半监督基类，统一监督/无监督损失接口。
   - `configs/SemiDino.py`：在基类上接入 DINO 风格检测器，形成 Semi-DETR 迁移目标骨架。
2. 明确运行输入：
   - 来自 dataloader 的 `multi_batch_inputs` / `multi_batch_data_samples`（监督和无监督分支）。
3. 明确运行输出：
   - `dict[str, float/tensor]` 风格损失字典，键名前缀区分 `sup_` 与 `unsup_`。

### 步骤 2：先完成 Phase 1（SoftTeacher 可运行骨架）
1. 在文件头提供“关键可改参数区”，包括：
   - 伪标签阈值、教师 EMA 动量、无监督损失权重、最小候选框数量等。
2. 实现核心流程函数：
   - 教师推理 -> 伪标签过滤 -> 弱增广到强增广映射（先用占位实现）-> 学生无监督训练。
3. 保持与 MMDetection 3.x 设计兼容的接口风格（`loss` 主入口）。

### 步骤 3：完成 Phase 2（SemiBaseDetector 基类）
1. 抽象出统一流程：
   - `compute_sup_loss`（监督）
   - `compute_unsup_loss`（无监督）
2. 实现总损失聚合和分支权重控制。
3. 预留可覆写钩子，便于后续 detector 替换。

### 步骤 4：完成 Phase 3（SemiDino 接入）
1. 通过组合方式接入“学生 DINO / 教师 DINO”。
2. 复用 `SemiBaseDetector` 的通用流程，覆写与 DINO 相关的分支实现。
3. 输出对齐训练入口所需的 `loss` 接口。

### 步骤 5：自检与最小验证
1. 运行静态语法检查（`python -m py_compile`）。
2. 确认仅修改目标文件，不触及 `thirdparty/`。
3. 记录当前已完成项、阻塞项和下一步。

---

## 本次执行记录

### 已完成
- 已在 `configs/SoftTeacher.py` 完成 SoftTeacher 三段式训练骨架（教师伪标签 + 学生监督/无监督）。
- 已在 `configs/SemiBaseDetector.py` 完成半监督通用基类，统一监督/无监督损失聚合。
- 已在 `configs/SemiDino.py` 完成 DINO 版半监督检测器骨架，并与基类接口对齐。
- 三个文件均补充了大量中文注释（含文件级总览、关键变量说明、函数粒度解释）。

### 当前阻塞
- 当前仓库尚未接入完整的 MMDetection/MMEngine 运行环境与真实数据流，
  因此本次仅做“结构可运行骨架 + 语法可通过”的最小实现，未执行端到端训练。

### 下一步
1. 在 `train.py` 中补全配置解析、Runner 构建与训练启动流程。
2. 将 `SoftTeacher` 与 `SemiDino` 注册到项目的统一 registry。
3. 接入真实 COCO 半监督标注路径（按 `data/ANNSPATH.md`），验证单卡最小迭代可跑通。
4. 逐步替换占位实现（如框映射）为与 MM3.x 对齐的正式版本。

### MM2.x -> MM3.x 差异决策记录
- 决策 1：优先采用“统一 `loss` 主入口 + 分支函数拆分”的 MM3.x 风格，
  而不是沿用 MM2.x 里更分散的训练步骤拼接方式。
- 决策 2：教师-学生协同采用“组合 + 基类钩子”方式，以减少对旧工程层级结构的依赖。
- 决策 3：先保证流程可实例化和可调试，再逐步细化算子级一致性。
