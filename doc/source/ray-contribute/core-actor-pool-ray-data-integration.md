# Core ActorPool — Ray Data Integration

This document explains how the new C++-backed Core ActorPool integrates with Ray Data. It covers what changes for Ray Data, how to enable it, the migration path, compatibility guarantees, and testing strategy.

For the detailed internals design (pool manager, work queue, retry state machine), see `Core_Actor_Pool_0309.md`.

**Target audience:** Ray Data engineers who understand pipelines, operators, and streaming execution at a high level but may not be familiar with how actor-based execution (`ActorPoolMapOperator`, `_ActorPool`) works under the hood.

---

## 1. Background — How Ray Data Uses Actors Today

### What `map_batches` with a class does

When users pass a callable class to `map_batches()` with `ActorPoolStrategy`, Ray Data creates a pool of long-running actors — one per worker. Each actor holds the class instance (e.g., a loaded ML model). Data blocks are routed to these actors for processing:

```python
ds.map_batches(MyModel, compute=ActorPoolStrategy(size=4), num_gpus=1)
```

### How the pool works today

Ray Data's `ActorPoolMapOperator` manages the pool via an internal Python class called `_ActorPool`. This class:

- **Creates actors** by calling `ray.remote(MyClass).remote()`
- **Picks which actor gets each task** using a Python selector (`_ActorTaskSelectorImpl`) that ranks actors by locality and load
- **Tracks per-actor task counts** for backpressure (`max_tasks_in_flight_per_actor`)
- **Handles autoscaling** — scales up when utilization exceeds 1.75x (i.e., tasks in flight / (concurrency * actors) >= 1.75), scales down when utilization drops below 0.5x

### What goes wrong on failure

If an actor dies (node crash, OOM), Ray retries the task on *the same actor*. All pending tasks queue behind a single restarting actor while other healthy actors sit idle. In multi-stage pipelines, intermediate outputs are lost and must be reconstructed — but reconstruction is also actor-bound, creating cascading delays.

---

## 2. What is the Core ActorPool?

**One sentence:** A C++-backed actor pool that lives in Ray Core's runtime, replacing Ray Data's Python-side pool management with a unified, lower-level abstraction.

### Key idea

Instead of Ray Data managing actors individually and picking which one gets each task, Core provides a "pool" primitive. You register actors into a pool, then submit tasks *to the pool* (not to a specific actor). The C++ runtime picks the actor, handles retries across actors, and tracks load.

### Python API (`ray.experimental.actor_pool`)

```python
import ray
from ray.experimental.actor_pool import ActorPool, RetryPolicy

@ray.remote
class Worker:
    def process(self, data):
        return data * 2

pool = ActorPool(
    actor_cls=Worker,
    size=4,
    retry=RetryPolicy(max_attempts=3),
)

# Submit tasks — Core picks the actor
refs = [pool.submit("process", i) for i in range(10)]
results = ray.get(refs)
pool.shutdown()
```

> **Note:** Ray Data users don't use this API directly — it's wrapped by the adapter (Section 3).

Under the hood, `ActorPool.__init__` calls `core_worker.register_actor_pool()` to create a C++ pool, then `add_actor_to_pool()` for each actor. `submit()` calls `core_worker.submit_task_to_pool()`, which routes through `CCoreWorkerProcess.GetCoreWorker().SubmitTaskToActorPool()` in the Cython bridge (`python/ray/_raylet.pyx:5070-5092`). The C++ `ActorPoolManager` (`src/ray/core_worker/actor_pool_manager.h`) selects an actor, submits the task, and handles retries transparently.

---

## 3. How the Integration Works

### Feature flag

The Core ActorPool is gated behind a feature flag (defined in `python/ray/data/context.py`):

- **Environment variable:** `RAY_DATA_USE_CORE_ACTOR_POOL=1`
- **Programmatic:** `DataContext.get_current().use_core_actor_pool = True`
- **Default:** off

### What happens when enabled

`ActorPoolMapOperator` creates a `ClassBasedActorPoolAdapter` instead of `_ActorPool`. The adapter wraps `ray.experimental.ActorPool` and implements the same `AutoscalingActorPool` interface that the operator expects. From the operator's perspective, nothing changes.

The dual-path decision is in `actor_pool_map_operator.py` lines 200-207:

```python
use_core_pool = (
    data_context.use_core_actor_pool and not self._ray_remote_args_fn
)
```

### Architecture diagram

```
Legacy (default):                     Core pool (opt-in):
┌──────────────────────┐              ┌──────────────────────┐
│ ActorPoolMapOperator │              │ ActorPoolMapOperator │
│                      │              │                      │
│  ┌────────────────┐  │              │  ┌────────────────┐  │
│  │ _ActorPool     │  │              │  │ ClassBased-    │  │
│  │ (Python)       │  │              │  │ ActorPool-     │  │
│  │                │  │              │  │ Adapter        │  │
│  │ Picks actor ───┤  │              │  │                │  │
│  │ in Python      │  │              │  │  ┌──────────┐  │  │
│  └────────────────┘  │              │  │  │ C++ Pool │  │  │
│         │            │              │  │  │ Manager  │  │  │
│         ▼            │              │  │  │          │  │  │
│  actor.method.remote │              │  │  │ Picks    │  │  │
│  (direct call)       │              │  │  │ actor,   │  │  │
└──────────────────────┘              │  │  │ retries, │  │  │
                                      │  │  │ balances │  │  │
                                      │  │  └──────────┘  │  │
                                      │  └────────────────┘  │
                                      │         │            │
                                      │         ▼            │
                                      │  submit_task_to_pool │
                                      │  (C++ selects actor) │
                                      └──────────────────────┘
```

### What moves to C++ and what stays in Python

**Moved to C++:**
- Actor selection (load balancing + locality)
- Task slot tracking (backpressure source of truth via `get_occupied_task_slots()`)
- Retry routing (cross-actor — if an actor dies, retry on a different healthy actor)

**Stays in Python:**
- Autoscaling decisions (when/how much to scale) — the adapter's `scale()` method still controls pool sizing
- Operator lifecycle (start, stop, input/output queue management)
- Actor state tracking for Ray Data metrics (`_ActorState` in the adapter)

### Fallback to legacy path

If `ray_remote_args_fn` is set (dynamic per-call actor options), the legacy Python path is always used — even with the flag on. This is because the Core pool creates actors with static options baked in at construction time and has no hook for per-actor dynamic argument generation.

---

## 4. What Changes for Ray Data Users?

### Nothing in the public API

`map_batches(MyUDF, compute=ActorPoolStrategy(size=4))` works identically. No code changes needed.

### Behavioral differences when the flag is on

These matter for anyone debugging or testing. The table reflects **Phase 1 (current)** behavior:

| Aspect | Legacy | Core Pool |
|---|---|---|
| Retry on actor death | Same actor (waits for restart) | Any healthy actor in the pool |
| Lineage reconstruction | Rebuilds on original actor | Can rebuild on any pool actor |
| Actor selection | Python ranking by (locality, load) | C++ ranking with same criteria |
| Backpressure tracking | Python per-actor counters | C++ `get_occupied_task_slots()` |
| Scale-down actor choice | First idle actor found (insertion order) | First idle actor found (insertion order) |
| Stats | Basic | Adds `total_tasks_retried`, `backlog_size`, `total_in_flight` |

The adapter exposes pool stats via `ActorPool.stats()`, which queries the C++ pool manager and returns a dict with keys like `total_tasks_submitted`, `total_tasks_failed`, `total_tasks_retried`, `num_actors`, `backlog_size`, and `total_in_flight`.

---

## 5. Migration & Compatibility

### Zero breaking changes

The feature is entirely behind a flag. Existing code, tests, and configs work without modification.

### `ray.util.ActorPool` — not affected

The legacy `ray.util.ActorPool` utility class is **completely unrelated** to this work. It is not affected, not deprecated. It's a standalone helper for manual actor pool management that predates Ray Data.

### Gradual rollout path

1. **Now (Phase 1):** Opt-in via flag. Validate with nightly benchmarks and targeted test suites.
2. **Phase 2:** Default-on after performance parity confirmed. Legacy path preserved as fallback.
3. **Phase 3:** Legacy `_ActorPool` deprecated internally. `ray.util.ActorPool` remains unchanged.

---

## 6. Testing

### Core pool unit tests

**File:** `python/ray/tests/test_actor_pool_v2.py`

Tests the `ray.experimental.ActorPool` API directly: pool lifecycle, submit, scale, stats, shutdown, load balancing.

### Ray Data integration tests

**File:** `python/ray/data/tests/test_actor_pool_map_operator.py`

End-to-end tests with `ActorPoolMapOperator`. Exercises both legacy and core pool paths via the feature flag.

### Reconstruction tests

**File:** `python/ray/data/tests/test_actor_pool_reconstruction.py`

Kills worker nodes mid-pipeline, verifies no data loss. Tests multi-stage cascading reconstruction. These are the highest-signal tests for the core pool's value proposition — they validate that cross-actor retry actually prevents the cascading delays described in Section 1.

### How to run

```bash
# Core pool unit tests
bazel test //python/ray/tests:test_actor_pool_v2

# Ray Data integration (with core pool enabled)
RAY_DATA_USE_CORE_ACTOR_POOL=1 pytest python/ray/data/tests/test_actor_pool_map_operator.py -v

# Reconstruction tests (with core pool enabled)
RAY_DATA_USE_CORE_ACTOR_POOL=1 pytest python/ray/data/tests/test_actor_pool_reconstruction.py -v
```

---

## 7. Current Limitations (Phase 1)

Be aware of what doesn't work yet, to avoid surprises:

- **`ray_remote_args_fn` forces the legacy path** — dynamic per-call actor options are incompatible with the static-option pool construction model.
- **No driver recovery or pool persistence** — if the driver dies, pools are lost. Planned for Phase 2.
- **No multi-submitter support** — only the driver can submit to a pool. Planned for Phase 2.

---

## Key Source Files

| File | Role |
|---|---|
| `python/ray/experimental/actor_pool.py` | Public Python API (`ActorPool`, `RetryPolicy`) |
| `python/ray/data/_internal/execution/operators/core_actor_pool_adapter.py` | Ray Data adapter (`ClassBasedActorPoolAdapter`) |
| `python/ray/data/_internal/execution/operators/actor_pool_map_operator.py` | Operator with dual-path logic |
| `python/ray/data/context.py` | Feature flag definition |
| `src/ray/core_worker/actor_pool_manager.h` | C++ pool manager interface |
| `python/ray/_raylet.pyx` | Cython bridge methods |
