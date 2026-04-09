# AGENTS.md

## Project goal

This repository is for research on semi-supervised object detection.

The current objective is to re-implement Semi-DETR on the MMDetection 3.x stack.
Semi-DETR is a strong and reliable baseline in semi-supervised object detection, but the original implementation is based on the MM 2.0 generation. The target here is to build an MM 3.0-compatible reimplementation with a clear, incremental migration path.

## Repository layout

- `configs/`
  - Stores model-related core code and configuration content used by the project.

- `data/`
  - Stores the COCO dataset locally.
  - The dataset is too large and is not tracked in GitHub.
  - `ANNSPATH.md` records the relative paths (from repository root) for the semi-supervised annotation files required for training.

- `thirdpart/`
  - Stores reference code.
  - Currently includes:
    - MMDetection 3.3.0 source code
    - Semi-DETR source code

- `train.py`
  - Expected entry point for training.

- `PLAN.md`
  - Used to record plans, implementation steps, progress, and notes.

## Working rules

- Prioritize a runnable implementation before optimization or cleanup.
- Treat MMDetection 3.3.0 as the primary framework reference.
- Treat the original Semi-DETR code as the algorithm reference.
- Prefer adapting Semi-DETR logic into MM 3.x conventions instead of trying to preserve MM 2.x structure verbatim.
- Do not commit datasets or large generated artifacts into GitHub.
- Always use repository-root-relative paths when referring to dataset annotations, and cross-check them against `ANNSPATH.md`.
- Keep changes incremental and easy to verify.
- Prefer modifying this project’s own code rather than editing files under `thirdpart/`, unless explicitly needed for reference experiments.
- Keep `train.py` as the canonical training entry whenever possible.

## Execution plan

The implementation should proceed in three phases:

1. Build a runnable SoftTeacher.
2. Build a runnable `SemiBaseDetector`.
3. Replace the detector inside `SemiBaseDetector` with DINO.

## Phase details

### Phase 1: runnable SoftTeacher

Goal:
- Build a SoftTeacher-style semi-supervised detection pipeline that can run on MM 3.x.

Minimum expectations:
- Model construction works.
- Teacher-student data flow works.
- Training can be launched from `train.py`.
- The pipeline can consume the COCO data and semi-supervised annotations defined in `ANNSPATH.md`.

### Phase 2: runnable `SemiBaseDetector`

Goal:
- Reconstruct a runnable `SemiBaseDetector` abstraction on MM 3.x.

Minimum expectations:
- Semi-supervised detector base logic is implemented clearly.
- Supervised / unsupervised branches are correctly organized.
- Training forward path and loss computation run correctly.
- The implementation is usable as the base class for later detector replacement.

### Phase 3: replace detector with DINO

Goal:
- Use DINO as the detector inside `SemiBaseDetector` to reproduce the Semi-DETR-style method on MM 3.x.

Minimum expectations:
- DINO integrates cleanly into the semi-supervised framework.
- Interfaces between detector, teacher-student logic, and training loop are consistent.
- Training can run end-to-end without obvious structural mismatch.

## Documentation requirements

- Before large refactors or multi-file changes, update `PLAN.md` with the current plan.
- After each major subtask, update `PLAN.md` with:
  - what was completed
  - current blockers
  - next step
- If an implementation decision is made because MM 2.x and MM 3.x APIs differ, record that decision briefly in `PLAN.md`.

## Implementation priorities

When there is ambiguity, follow this priority order:

1. Make it run.
2. Keep it compatible with MMDetection 3.3.0 design patterns.
3. Preserve the key Semi-DETR algorithm logic.
4. Improve structure and readability.
5. Optimize performance only after the pipeline is stable.

## Definition of done

A phase is considered done only if all of the following are true:

- The relevant module can be instantiated successfully.
- The training entry path through `train.py` is connected.
- Required dataset and annotation paths are clearly resolved from repository root.
- The core training logic for that phase runs without structural errors.
- The current status and remaining work are written into `PLAN.md`.

## Notes for Codex

- Do not assume dataset files are present in Git; they are local under `data/`.
- Do not hardcode machine-specific absolute paths if a repository-relative path can be used.
- Use `thirdpart/` for comparison and reference, not as the final implementation location.
- When porting from MM 2.x to MM 3.x, prefer understanding the intent of the original code instead of line-by-line translation.
- Keep the implementation path aligned with the three-phase plan above.