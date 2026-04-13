# SemiDino 最小可运行闭环实现计划（MMDetection 3.x）

## 0. 文档定位与本轮修复目标

本文档用于约束“先计划、后编码”。当前这轮修改聚焦三件事：

1. **严格复用 MMDetection 官方 DINO 配置**，不再在 `configs/SemiDino.py` 内重复手写大段 DINO 结构；
2. **消除 `configs/SemiDino.py` 与 `train.py` 的重复覆盖逻辑**，保留必要的入口参数集中区；
3. **修复 `SemiBaseDetector + DINO` 在 teacher/student 二次构建时触发的断言错误**。

---

## 1. 用户问题复盘与根因

### 1.1 “为什么不复用官方 DINO？”

你指出的问题是正确的：之前版本虽然 `type='DINO'`，但在配置里手写了大量官方 DINO 结构，读起来像“自己造了一个 DINO”。这会降低可维护性，也违背“最大限度复用官方组件”的项目原则。

### 1.2 报错根因分析

报错信息（`share_pred_layer / num_pred_layer / as_two_stage` 断言）来自 DINO 初始化阶段。根因是：

- `SemiBaseDetector` 需要基于同一份 `detector` 配置分别构建 `student` 与 `teacher`；
- 官方 `DINO.__init__` 会**就地修改**传入的 `bbox_head` 配置（写入上述字段）；
- 第二次构建时，配置已被第一次构建污染，于是命中断言。

这不是“DINO 不能用于半监督”，而是“构建两次时要确保配置对象互不污染”。

---

## 2. 本轮实现策略（先复用、后最小修补）

### 2.1 配置层：改为“继承官方配置 + 最小覆盖”

`configs/SemiDino.py` 改为：

- 直接继承：
  - `configs/dino/dino-4scale_r50_8xb2-12e_coco.py`（官方 DINO）
  - `configs/_base_/datasets/semi_coco_detection.py`（官方半监督数据组织）
- 仅覆盖本项目必须项：
  - `model = SemiBaseDetector(detector=官方DINO)`
  - `semi_train_cfg / semi_test_cfg`
  - 标注路径、采样比例、迭代策略、hook、日志间隔。

这样可以把“官方能力”与“项目自定义桥接”边界清晰分开。

### 2.2 代码层：修复二次构建污染

对 `SemiBaseDetector` 做**最小必要修补**：

- 构建 `student` 与 `teacher` 时分别使用 `copy.deepcopy(detector)`；
- 避免 DINO 在第一次构建时对配置对象的就地修改影响第二次构建。

### 2.3 入口层：减少重复覆盖

`train.py` 保留“集中可改参数区”，但减少重复字段写入，统一通过一个函数批量覆盖：

- 数据路径（labeled / unlabeled / val ann）；
- dataloader 的 `batch_size / num_workers / source_ratio`；
- `work_dir / launcher / device / load_from / resume`。

---

## 3. 关键调用链（保持不变）

1. `Runner.from_cfg(cfg).train()`；
2. `SemiBaseDetector.loss`：
   - `loss_by_gt_instances(sup)`；
   - `get_pseudo_instances(unsup_teacher)`；
   - `project_pseudo_instances(...)`；
   - `loss_by_pseudo_instances(unsup_student)`；
3. `MeanTeacherHook` 进行 EMA 更新 teacher；
4. `TeacherStudentValLoop` 做验证，`predict_on='teacher'`。

---

## 4. 可复用与自定义边界

### 4.1 复用（本轮强调）

- 官方 `DINO`（完整结构）
- 官方 `SemiBaseDetector`
- 官方 `semi_coco_detection.py` 三分支数据管线
- 官方 `TeacherStudentValLoop` / `MeanTeacherHook` / `CocoMetric`

### 4.2 本项目最小自定义

- `configs/SemiDino.py`：参数集中区 + 最小覆盖
- `train.py`：路径和实验目录友好入口
- `semi_base.py`：为 DINO 双构建增加深拷贝保护（最小补丁）

---

## 5. 本轮验证方法

1. 语法检查：`python -m py_compile configs/SemiDino.py train.py thirdparty/mmdetection-3.3.0/mmdet/models/detectors/semi_base.py`；
2. 配置装载检查：`python train.py --help`（验证参数解析）；
3. 断言问题验证：重新构建 `Runner.from_cfg(cfg)`，确认不再触发 DINO 二次构建断言。

---

## 6. 后续扩展（不阻塞本轮）

1. DINO 专用动态伪标签阈值；
2. query 级一致性损失；
3. 更贴近 Semi-DETR 的额外监督头。

以上都需先更新 `PLAN.md` 再实现。
