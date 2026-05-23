<!-- Loaded on-demand when Claude works on Ray Data files. -->

# Ray Data

## Key Modules
**Public API** (stable):
- `dataset.py` — `Dataset`: all lazy transforms (`map_batches`, `filter`, `sort`, …) and terminal ops (`materialize`, `iter_batches`, `write_*`).
- `read_api.py` — `ray.data.read_*` builders (each appends a `ReadLogicalOperator`).
- `grouped_data.py` — `GroupedData` from `ds.groupby()`.
- `block.py` — `Block` (Arrow or pandas), `BlockAccessor.for_block()` dispatches to `ArrowBlockAccessor`/`PandasBlockAccessor`.
- `datasource/datasource.py` — `Datasource` (read; `create_reader()`).
- `datasource/datasink.py` — `DataSink` (write; `write()`).

**Internal** (`_internal/` — unstable, do not import from outside):
- `logical/` — `LogicalPlan`, `LogicalOperator`s + rewrites (`operator_fusion.py`, `limit_pushdown.py`, `predicate_pushdown.py`, `projection_pushdown.py`).
- `planner/planner.py` — turns optimized `LogicalPlan` into a `PhysicalPlan` of `PhysicalOperator`s (per-op planners under `planner/`).
- `execution/streaming_executor.py` — drives the physical DAG.
- `execution/operators/` — `MapOperator` (base), `TaskPoolMapOperator`, `ActorPoolMapOperator`, `AllToAllOperator`, …
- `execution/backpressure_policy/` — downstream-capacity, resource-budget, concurrency-cap.
- `execution/resource_manager.py` + `interfaces/execution_options.py` — `ExecutionResources` budgeting.

## Architecture flow
`read_*().map_batches().write_*()` builds a `LogicalPlan` lazily. On a terminal op, `LogicalOptimizer` rewrites it (operator fusion, pushdown) → `Planner` emits a `PhysicalPlan` → `StreamingExecutor` runs a scheduling loop: pick operators to advance subject to backpressure, dispatch Ray tasks/actor calls, move `RefBundle`s (lists of `(ObjectRef, BlockMetadata)`) downstream. Stateless transforms → ephemeral tasks (`TaskPoolMapOperator`); stateful → autoscaling actor pool (`ActorPoolMapOperator`). Blocks live in the object store; never pass through the driver.

## Gotchas
- **Lazy by default.** Only terminal ops execute. `ds.schema()` materializes the first block — expensive on big read-many-files plans; cache or pass schema explicitly.
- **Block format.** Arrow when representable, pandas otherwise. `batch_format` controls UDF input; mismatched formats between chained maps add silent conversion overhead (fusion does NOT elide format-boundary copies). `batch_format=None` is zero-copy but UDF must handle both.
- **`_internal` is unstable** but heavily imported by third-party code anyway — keep new internal code there and avoid breaking import paths gratuitously.
- **Actor pool autoscaling** (`actor_autoscaler/`) can change pool size mid-run. Actors pin GPU memory — set `concurrency=` explicitly for GPU workloads.
- **`ExecutionResources` is a per-pipeline soft cap**, not a cluster reservation. Setting it too high can over-subscribe the cluster scheduler.
- **`concurrency=` caps actor pool size only.** Task-based maps run as many concurrent tasks as the cluster allows; gate via `num_cpus`/`num_gpus` per-task.
- **Datasource ≠ DataSink.** Separate classes (read vs write); don't conflate.
