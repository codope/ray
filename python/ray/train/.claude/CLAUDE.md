<!-- Loaded on-demand when Claude works on Ray Train files. -->

# Ray Train

## v1 vs v2 — read first
**v2 is the default.** Toggle: `RAY_TRAIN_V2_ENABLED`; `is_v2_enabled()` in `v2/_internal/constants.py` returns True by default. Each framework `__init__.py` imports v1 first then conditionally overrides with v2. **New features → v2**. Touch v1 only for fixes affecting users who explicitly opted out. `ScalingConfig`/`RunConfig` resolve to different classes (v1 dataclasses in `ray/air/config.py`; v2 pydantic models in `v2/api/config.py`) at runtime — do not assume identity.

## Key Modules
**v1** (top-level `python/ray/train/`, non-`v2/`):
- `base_trainer.py` (`BaseTrainer`), `data_parallel_trainer.py` (`DataParallelTrainer`).
- Framework trainers: `torch/torch_trainer.py`, `tensorflow/tensorflow_trainer.py`, `xgboost/xgboost_trainer.py`, `lightgbm/lightgbm_trainer.py`, `horovod/horovod_trainer.py`. HuggingFace lives under `huggingface/transformers/`.
- `_internal/backend_executor.py`, `_internal/worker_group.py`, `_internal/session.py` (holds module-level `_session` for `train.report`), `_internal/_checkpoint.py`, `_internal/checkpoint_manager.py`.

**v2** (`python/ray/train/v2/`):
- `v2/api/data_parallel_trainer.py`, `v2/torch/torch_trainer.py`, `v2/api/config.py` (pydantic), `v2/api/train_fn_utils.py` (`report()`).
- `v2/_internal/execution/controller/controller.py` (`TrainController` — replaces `BackendExecutor`) + `controller/state.py` state machine.
- `v2/_internal/execution/worker_group/worker_group.py`, `v2/_internal/execution/context.py` (module-level `_train_context`).
- `v2/_internal/migration_utils.py` — deprecation messages for v1→v2 field removals.

## Architecture flow
**v1:** `TorchTrainer(...).fit()` → `as_trainable()` wraps in a Tune `Trainable` → `Tuner.fit()` (always runs through Tune, even for single trials). Inside the trial, `DataParallelTrainer.training_loop` builds a `BackendExecutor`, which spawns a `WorkerGroup` of Ray actors (one per `num_workers`), initializes the backend (NCCL/Gloo via `TorchConfig`), and ships the `train_loop_per_worker` to all workers. Workers call `train.report(metrics, checkpoint=...)` against the per-worker `_session`; the executor polls and forwards results to Tune.

**v2:** `DataParallelTrainer.fit()` spawns a `TrainController` Ray actor directly (no Tune wrapping unless already in a Tune trial — detected at call site). The controller owns the worker group lifecycle via a state machine (`controller/state.py`) and supports restart/resize. Workers use a module-level `_train_context` per process; `report()` handles async checkpoint upload, consistency modes, and rank-0 aggregation before surfacing to the controller.

## Gotchas
- **Train↔Tune circular-import risk.** v1 `base_trainer.py` imports `ray.tune` inside `fit()` to break the cycle. `lint/check_circular_imports.py` enforces it — run before adding cross-package imports.
- **`train.report()` is a module global**, not a method. Outside a worker it raises. v1 reads `_session`; v2 reads `_train_context`.
- **`prepare_model` / `prepare_data_loader`** wrap in DDP (or FSDP) and move to device; `prepare_data_loader` swaps in a `DistributedSampler`. v2 equivalents under `v2/torch/train_loop_utils.py`.
- **`BackendExecutor` lifecycle (v1):** `start → start_training → get_next_results → shutdown`. Skip `shutdown()` and you leak actors. v2's controller owns its own lifecycle as an actor.
- **Framework code is intentionally public** (`torch/`, `xgboost/`, …); `_internal/` is only for shared plumbing.
- **v2 requires pydantic** — `ray.train` import fails if missing and v2 is on.
