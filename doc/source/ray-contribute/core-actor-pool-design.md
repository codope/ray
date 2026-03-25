# Core Actor Pool Design Notes

**PR**: [#61622](https://github.com/ray-project/ray/pull/61622)
**Status**: Draft — design review before code review
**Design Doc**: [Core Actor Pool](https://docs.google.com/document/d/1B6IztJAYILCp_qxo3UEmKXY_5ApqPT8YW1kcA4mipxQ/edit?usp=sharing)
---

## 1. Motivation & Goals

### Problem

Ray Data's Python actor pool has three fundamental limitations:

1. **No cross-actor retry**: When an actor dies, tasks bound to it fail permanently. Ray's built-in retry (`max_retries`) is actor-bound — `TaskSpec` is bound to a specific `ActorId` at submission time, so retries always go to the SAME actor.
2. **Python-side performance overhead**: Actor selection, load balancing, and in-flight tracking all happen in Python.
3. **Dual-tracked in-flight counts**: Python `_total_num_tasks_in_flight` and C++ per-actor counts diverge on actor death, backlog changes, and pool-submitted tasks.

### Goals (Q1 2026)

- Cross-actor retry with exponential backoff
- Lineage reconstruction support for pool tasks
- Backpressure unification (C++ as single source of truth)
- Locality-aware scheduling (reuse existing `LocalityDataProviderInterface`)
- Performance parity with legacy Python pool

### Non-goals / Deferred

- Core-controlled autoscaling (`SelectActorForRemoval`)
- Grace period shutdown (drain-before-kill)
- `ray_remote_args_fn` support in core pool path

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Ray Data                                                           │
│  ActorPoolMapOperator                                               │
│    ├── _try_schedule_via_pool()     ← core pool path                │
│    └── _try_schedule_tasks_internal() ← legacy path                 │
├─────────────────────────────────────────────────────────────────────┤
│  Python Adapter                                                     │
│  CoreActorPoolAdapter (AutoscalingActorPool interface)              │
│    ├── submit_task()  → C++ pool selects actor                      │
│    ├── num_free_task_slots() → C++ GetOccupiedTaskSlots             │
│    └── num_active_actors()   → C++ GetNumActiveActors               │
├─────────────────────────────────────────────────────────────────────┤
│  Cython Bindings (_raylet.pyx)                                      │
│    ├── register_actor_pool()    ─┐                                  │
│    ├── submit_task_to_pool()     │  with nogil                      │
│    ├── get_occupied_task_slots() │  (GIL released for C++ calls)    │
│    └── add_actor_to_pool()      ─┘                                  │
├─────────────────────────────────────────────────────────────────────┤
│  C++ ActorPoolManager                                               │
│    ├── SelectActorFromPool()  (load + locality ranking)             │
│    ├── OnPoolTaskComplete()   (cross-actor retry entry point)       │
│    ├── ScheduleRetry()        (exponential backoff via boost timer) │
│    └── DrainWorkQueue()       (submit backlogged work)              │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature flag

`RAY_DATA_USE_CORE_ACTOR_POOL` (env var) → `DataContext.use_core_actor_pool` (default: off)

When off, legacy Python pool path is unchanged.

**Ref**: `python/ray/data/context.py:214`

### Key files by layer

| Layer | File | Purpose |
|-------|------|---------|
| C++ | `src/ray/core_worker/actor_pool_manager.h` | Structs, API surface |
| C++ | `src/ray/core_worker/actor_pool_manager.cc` | Full implementation |
| C++ | `src/ray/core_worker/actor_pool_work_queue.h` | Work queue interface + `UnorderedPoolWorkQueue` |
| C++ | `src/ray/core_worker/core_worker.cc:400-440` | Wiring (constructor, callbacks) |
| C++ | `src/ray/core_worker/core_worker.cc:838-874` | `InternalHeartbeat` pool task resubmission |
| C++ | `src/ray/core_worker/core_worker.cc:2847-2887` | `SubmitActorTaskForPool` |
| C++ | `src/ray/core_worker/task_submission/actor_task_submitter.cc:29-58` | `MaybeNotifyPoolTaskSubmitted` (in-flight increment at `PushActorTask`), `MaybeNotifyPoolTaskComplete` |
| C++ | `src/ray/core_worker/task_manager.cc:1064-1078` | Lineage pinning override |
| C++ | `src/ray/core_worker/task_manager.cc:541-574` | `MovePoolTaskActorDependency` (reference count transfer on redirect) |
| C++ | `src/ray/core_worker/lease_policy.h:27-39` | `LocalityDataProviderInterface` |
| Proto | `src/ray/protobuf/common.proto:625-628` | `actor_pool_id` (field 46), `actor_pool_work_item_id` (field 47) |
| Cython | `python/ray/_raylet.pyx:4825-5064` | Cython bindings (nogil calls) |
| Cython | `python/ray/includes/libcoreworker.pxd:127-157` | C++ declarations |
| Python | `python/ray/experimental/actor_pool.py` | Public Python API |
| Python | `python/ray/data/_internal/execution/operators/core_actor_pool_adapter.py` | Data adapter |
| Python | `python/ray/data/_internal/execution/operators/actor_pool_map_operator.py:200-247` | Feature flag branch |
| Python | `python/ray/data/_internal/execution/operators/actor_pool_map_operator.py:443-539` | Two submission paths |

---

## 3. Key Design Decisions & Rationale

These anticipate "why not just..." questions:

### Why `max_retries=-1` at task level?

Ray's built-in retry is actor-bound: `TaskSpec` is bound to a specific `ActorId` at submission time, so retries always go to the SAME actor. We use `max_retries=-1` (infinite) to keep the `ObjectRef` alive through actor failures. When an actor dies, `ActorTaskSubmitter` retries the task (keeping the same `ObjectRef` pending), and `InternalHeartbeat` redirects the task to a healthy pool actor. The pool does NOT submit new tasks for retries — it relies on this redirect mechanism to reuse the original `TaskID`/`ObjectRef`.

**Ref**: `core_worker.cc:2863` — `actor_handle->SetActorTaskSpec(builder, ..., /*max_retries=*/-1, ...)`

### Why a new `ActorPoolManager` vs extending `ActorTaskSubmitter`?

`ActorTaskSubmitter` is per-actor; pool needs cross-actor state (which actor to pick, load balancing, locality). Extending ATS would conflate per-actor and per-pool concerns.

### Why pool metadata on `TaskSpec` proto?

Need to identify pool tasks in the hot path (`HandlePushTaskReply`, `InternalHeartbeat`) without extra lookups. Proto fields 46-47 are `optional bytes`, safe with protobuf forward compatibility — older workers ignore unknown fields, `IsPoolTask()` returns false, tasks execute normally without pool retry semantics.

**Ref**: `common.proto:625-628`, `task_spec.h:294` — `bool IsPoolTask() const { return !GetMessage().actor_pool_id().empty(); }`

### Why single mutex vs per-pool locking?

Simplicity for Phase 1. 13 methods acquire `mu_`. Per-pool locking is a future optimization; current bottleneck is Python→C++ boundary, not mutex contention.

**Ref**: `actor_pool_manager.h:454` — `mutable absl::Mutex mu_`

### Why `dynamic_cast` for `LocalityDataProviderInterface`?

`reference_counter_` is stored as `shared_ptr<ReferenceCounterInterface>` but `ReferenceCounter` also implements `LocalityDataProviderInterface`. Cast happens ONCE at CoreWorker construction, not on hot path. Alternative: change inheritance hierarchy (larger blast radius).

**Ref**: `core_worker.cc:424` — `dynamic_cast<LocalityDataProviderInterface *>(reference_counter_.get())`

---

## 4. Interfaces & Abstractions

### `ActorPoolManager` (new)

Central orchestrator. Manages pool registration, actor selection, task submission, cross-actor retry, and work queue draining.

**Ref**: `actor_pool_manager.h:177-468`

Key methods:
- `RegisterPool()` / `UnregisterPool()`
- `AddActorToPool()` / `RemoveActorFromPool()`
- `SubmitTaskToPool()` — entry point for task submission
- `OnPoolTaskComplete()` — entry point for cross-actor retry
- `GetOccupiedTaskSlots()` / `GetNumActiveActors()` — backpressure queries
- `SelectActorForTask()` — public wrapper for reconstruction path

### `PoolWorkQueue` (new, abstract)

**Ref**: `actor_pool_work_queue.h:89-115`

Interface with `Push`, `Pop`, `HasWork`, `Size`, `Clear`. Only `UnorderedPoolWorkQueue` (FIFO deque) implemented.

### `LocalityDataProviderInterface` (reused)

**Ref**: `lease_policy.h:33-39`

Already implemented by `ReferenceCounter`. Provides `GetLocalityData(ObjectID)` → `{object_size, nodes_containing_object}`.

### `ActorTaskSubmitterInterface` (extended)

**Ref**: `actor_task_submitter.h:210-222`

Two callbacks wired once during CoreWorker construction:
- `SetPoolTaskCompletionCallback()` — `HandlePushTaskReply()` calls `MaybeNotifyPoolTaskComplete()` on every pool task terminal completion.
- `SetPoolTaskSubmittedCallback()` — `PushActorTask()` calls `MaybeNotifyPoolTaskSubmitted()` when a pool task is actually pushed to an actor's RPC queue. This is the single point for in-flight count increments, covering both original submissions and `InternalHeartbeat` redirects.

### `AutoscalingActorPool` (reused)

Ray Data's pool interface. `CoreActorPoolAdapter` implements it, delegating to C++ `ActorPool`.

**Ref**: `core_actor_pool_adapter.py:48`

### `TaskSpec` proto (extended)

**Ref**: `common.proto:625-628`

```protobuf
optional bytes actor_pool_id = 46;
optional bytes actor_pool_work_item_id = 47;
```

**Wire compatibility**: Both fields `optional bytes` — older workers ignore them; newer workers handle missing fields (empty = not a pool task).

---

## 5. Actor Pool Lifecycle & State Machine

### Lifecycle flow

```
1. Pool registration
   ActorPool.__init__() → core_worker.register_actor_pool() → C++ RegisterPool()

2. Actor creation & addition
   scale() → _create_and_add_actor() → ray.remote() → add_actor_to_pool()

3. Pending → Running transition
   get_location.remote() → pending_to_running() → add_actor_to_pool(location=NodeID)

4. Task submission (two paths)
   Core:   _try_schedule_via_pool() → adapter.submit_task() → C++ SubmitTaskToPool
   Legacy: _try_schedule_tasks_internal() → Python select → actor.submit.remote()

5. Pool shutdown
   shutdown() → ray.kill(actors) → unregister_actor_pool() → C++ UnregisterPool()
```

### Actor state machine (`ActorPoolActorState`)

**Ref**: `actor_pool_manager.h:105-117`

```
States:
  {is_alive=true, location=Nil}     → PENDING
  {is_alive=true, location=NodeID}  → RUNNING
  {is_alive=false}                  → DEAD

Transitions:
  PENDING  ──[add_actor_to_pool(location)]──→  RUNNING
  RUNNING  ──[actor death detected]──────────→  DEAD

Per-actor counters:
  num_tasks_in_flight   — incremented when PushActorTask sends the task to the
                          actor (via PoolTaskSubmittedCallback), decremented on
                          completion (via OnPoolTaskComplete), zeroed on actor
                          death (OnActorDead). This ensures redirected retries
                          are tracked on the correct actor.

Dead actors are never selected by SelectActorFromPool.
Dead actors have num_tasks_in_flight reset to 0 (stale counts cleared).
Tasks in-flight on dead actors get routed to other actors via InternalHeartbeat redirect.
```

---

## 6. Cross-Actor Retry

### Code-traced flow

```
1. Task submission
   SubmitTaskToPool()                              [actor_pool_manager.cc:182]
     → SelectActorFromPool()                       [actor_pool_manager.cc:364]
     → SubmitToActor()                             [actor_pool_manager.cc:470]
       → submit_actor_task_fn_ callback
         → CoreWorker::SubmitActorTaskForPool()    [core_worker.cc:2847]
           Sets max_retries=-1, tags pool metadata on TaskSpec
         → actor_task_submitter_->SubmitTask()
           → PushActorTask()                       [actor_task_submitter.cc:659]
             → MaybeNotifyPoolTaskSubmitted()      ← increments in-flight count
               → ActorPoolManager::OnTaskSubmitted()

2. Completion detection
   ActorTaskSubmitter::HandlePushTaskReply()        [actor_task_submitter.cc]
     → IsPoolTask() check                          [task_spec.h:294]
     → MaybeNotifyPoolTaskComplete()               [actor_task_submitter.cc:33]
       → pool_task_completion_callback_
         → ActorPoolManager::OnPoolTaskComplete()  [actor_pool_manager.cc:327]

3. Actor death → retry via InternalHeartbeat
   With max_retries=-1, ActorTaskSubmitter retries the task (will_retry=true).
   The pool is NOT notified via MaybeNotifyPoolTaskComplete (skipped when will_retry).
   Instead, OnActorDead() zeros the dead actor's in-flight count.

   CoreWorker::InternalHeartbeat()                 [core_worker.cc:863]
     → spec.IsPoolTask() check
     → SelectActorForTask() → picks healthy actor B
     → MovePoolTaskActorDependency()               [task_manager.cc]
       (updates reference counts from old to new actor creation dummy)
     → mutate TaskSpec actor_id to B
     → actor_task_submitter_->SubmitTask()
       → PushActorTask()
         → MaybeNotifyPoolTaskSubmitted()          ← increments B's in-flight
     → task completes on B
       → MaybeNotifyPoolTaskComplete()             ← decrements B's in-flight

4. Pool-level retry (for pool's own ShouldRetryTask path — currently secondary
   to the InternalHeartbeat path since max_retries=-1 handles most retries)
   OnPoolTaskComplete(status=error)
     → OnTaskFailed()                              [actor_pool_manager.cc:522]
       → ShouldRetryTask() (error classification) [actor_pool_manager.cc:687]
       → CalculateBackoff() (exponential)          [actor_pool_manager.cc:721]
       → ScheduleRetry() (boost::asio timer)       [actor_pool_manager.cc:612]
         → RetryWorkItem()                         [actor_pool_manager.cc:650]
           → SelectActorFromPool() (picks different actor)
           → SubmitToActor()
```

### Error classification

| Error Type | Retryable? | Rationale |
|-----------|-----------|-----------|
| `ACTOR_DIED` | Yes | System error — retry on different actor |
| `NODE_DIED` | Yes | System error — retry on different actor |
| `WORKER_DIED` | Yes | System error — retry on different actor |
| `ACTOR_UNAVAILABLE` | Yes | System error — retry on different actor |
| `TASK_EXECUTION_EXCEPTION` | No | User code error — retrying won't help |
| `OUT_OF_MEMORY` | No | Resource error — likely to OOM again |
| `TASK_CANCELLED` | No | Intentional cancellation |
| Unknown | No | Conservative default |

**Ref**: `actor_pool_manager.cc:687-719` — `ShouldRetryTask()`

---

## 7. Lineage Reconstruction & Memory Implications

### Code-traced flow

```
1. TaskManager lineage eligibility
   task_manager.cc:1064-1066:
     bool is_pool_task = it->second.spec_.IsPoolTask();
     bool task_retryable = (it->second.num_retries_left_ != 0 || is_pool_task) &&
                           !it->second.reconstructable_return_ids_.empty();

   Pool tasks are ALWAYS eligible for reconstruction. With max_retries=-1,
   num_retries_left_ is -1 (infinite), so the standard check already passes.
   The is_pool_task override is a defensive safeguard.

2. Lineage pinning
   task_manager.cc:1068-1071:
     if (task_retryable) {
       release_lineage = false;
       it->second.lineage_footprint_bytes_ = spec.ByteSizeLong();
       total_lineage_footprint_bytes_ += it->second.lineage_footprint_bytes_;
     }

3. Heartbeat-driven reconstruction
   CoreWorker::InternalHeartbeat()                 [core_worker.cc:838]
     → detect pool task in to_resubmit_            [core_worker.cc:854]
     → spec.IsPoolTask() check
     → actor_pool_manager_->SelectActorForTask()   [core_worker.cc:858]
       selects healthy actor
     → task_manager_->MovePoolTaskActorDependency() [task_manager.cc]
       (moves reference count from old actor's creation dummy object
        to new actor's — prevents stale dependency from blocking lineage GC)
     → redirect TaskSpec to new actor              [core_worker.cc:868-873]
       (mutates actor_id + creation_dummy_object_id)
     → resubmit task via actor_task_submitter_->SubmitTask()
       (PushActorTask fires PoolTaskSubmittedCallback → in-flight tracking
        is correctly updated for the new actor)
```

### Memory cost of lineage pinning

- Each pool task pins its `TaskSpec` (~`spec.ByteSizeLong()` bytes)
- Tracked against `total_lineage_footprint_bytes_` / `max_lineage_bytes_` limit
- When limit exceeded, oldest tasks evicted via LRU strategy (`task_manager.cc:1072-1077`)
- Lineage released when: (a) task return objects go out of scope, or (b) lineage budget forces eviction
- For long-running streaming pipelines with millions of tasks, lineage memory pressure is bounded by the existing `max_lineage_bytes_` limit — same mechanism as regular tasks

---

## 8. Backpressure Unification

### Problem

Python `_total_num_tasks_in_flight` and C++ `num_tasks_in_flight` diverge on actor death, backlog, and pool-submitted tasks.

### Solution: C++ is single source of truth

| Query | C++ Method | Implementation |
|-------|-----------|---------------|
| Occupied slots | `GetOccupiedTaskSlots(pool_id)` | `sum(in_flight) + backlog` |
| Active actors | `GetNumActiveActors(pool_id)` | `count(actors where in_flight > 0)` |

**Ref**: `actor_pool_manager.cc:284-319`

### Data flow

```
can_add_input()                                    [actor_pool_map_operator.py:355]
  → actor_pool.num_free_task_slots()               [core_actor_pool_adapter.py:211]
    → capacity - core_worker.get_occupied_task_slots(pool_id)
      → C++ GetOccupiedTaskSlots()                 [actor_pool_manager.cc:284]
        = sum(actor_state.num_tasks_in_flight) + work_queue.Size()
```

### Key fixes

- **In-flight tracking at `PushActorTask`**: `PoolTaskSubmittedCallback` fires when a pool task is actually pushed to an actor's RPC queue (not at `SubmitToActor` time). This ensures redirected retries via `InternalHeartbeat` are tracked on the correct actor. `OnActorDead` zeros the dead actor's count.
- **Per-actor C++ query**: `GetActorTasksInFlight(pool_id, actor_id)` exposed through Cython. The adapter's `_try_remove_idle_actor` queries C++ directly instead of Python-side state, preventing the autoscaler from killing actors that have redirected tasks.
- **No Python-side per-actor tracking**: The adapter removed `num_tasks_in_flight` from `_ActorState`. `on_task_submitted()` and `on_task_completed()` are no-ops for pool-submitted tasks — C++ is authoritative.
- `_release_running_actor` calls `remove_actor_from_pool` on C++ side (`core_actor_pool_adapter.py:597`)

---

## 9. Locality-Aware Scheduling

### Design

Reuses `LocalityDataProviderInterface` (already implemented by `ReferenceCounter`).

**Ref**: `lease_policy.h:33-39`

### `ComputeNodeLocalityMap(arg_ids)`

**Ref**: `actor_pool_manager.cc:450-468`

For each argument object ID, queries `GetLocalityData()` and aggregates `NodeID → total_bytes`. Hoisted outside per-actor loop (key optimization).

### `RankActor` returns `(locality_rank, load)` pair

**Ref**: `actor_pool_manager.cc:422-448`

```
Ranking (lower is better):
  Data-local node:  (-total_bytes, in_flight)  — more data = better
  Remote node:      (INT64_MAX, in_flight)
  No locality data: (0, in_flight)             — pure load balancing
```

Selection uses `std::min_element` with lexicographic comparison. Matches Python legacy `_rank_actors` semantics (locality primary, load secondary).

---

## 10. Ray Data Integration & Migration Strategy

### Feature flag

```python
# python/ray/data/context.py:214
DEFAULT_USE_CORE_ACTOR_POOL = env_bool("RAY_DATA_USE_CORE_ACTOR_POOL", False)
```

### Detection in `ActorPoolMapOperator`

**Ref**: `actor_pool_map_operator.py:200-247`

```python
use_core_pool = data_context.use_core_actor_pool and not self._ray_remote_args_fn

if use_core_pool:
    self._actor_pool = CoreActorPoolAdapter(...)  # C++ handles selection
else:
    self._actor_pool = _ActorPool(...)            # Python handles selection
```

### Two submission paths

**Ref**: `actor_pool_map_operator.py:443-539`

```
_try_schedule_tasks_internal():
  if supports_pool_submission:
    → _try_schedule_via_pool()       # C++ selects actor
  else:
    → Python task selector → actor.submit.remote()  # Python selects actor
```

### `CoreActorPoolAdapter` bridges interfaces

**Ref**: `core_actor_pool_adapter.py:48-586`

- Implements `AutoscalingActorPool` interface
- Delegates pool management to `ActorPool` (C++-backed)
- Labels: `__ray_data_logical_actor_id` for Data's actor tracking

### Automatic fallback to legacy path

When `ray_remote_args_fn` is set, core pool cannot be used because it creates actors with static options baked in at construction time — no hook for per-actor dynamic arg generation.

Both paths coexist indefinitely behind the feature flag; rollback = unset env var.

---

## 11. Failure Modes & Edge Cases

| Scenario | Behavior | Risk |
|----------|----------|------|
| **Single actor dies** | `OnActorDead` zeros in-flight count; `InternalHeartbeat` redirects task to healthy actor; `PushActorTask` callback updates tracking on new actor | Low |
| **Node crash kills N actors** | Each failed task retried independently; `SelectActorFromPool` distributes across survivors; exponential backoff prevents thundering herd | Low |
| **Force downscale during in-flight tasks** | Adapter's `_try_remove_idle_actor` queries C++ `GetActorTasksInFlight(pool_id, actor_id)` — actors with redirected tasks show correct in-flight counts and are not removed | Low (fixed) |
| **All actors dead** | `SelectActorFromPool` returns Nil → work items queued indefinitely → drained when new actor added via `AddActorToPool` | **Medium** — no timeout/error propagation; tasks wait forever |
| **Driver crash** | Pool state lost; in-flight tasks continue/fail normally on workers; no cleanup of work items or async timers | **Medium** — async timer captures `this`, potential use-after-free if destruction races with timer |
| **GCS failover** | No impact — pool metadata is entirely local to CoreWorker, no GCS storage | Low |
| **Version skew (new driver → old worker)** | Proto fields 46-47 ignored; tasks execute normally without pool retry semantics | Low |
| **Version skew (old driver → new worker)** | Fields absent = empty → `IsPoolTask()` returns false | Low |

---

## 12. Performance & Scalability Analysis

### Current design choices and their implications

| Aspect | Design | Implication |
|--------|--------|-------------|
| **Locking** | Single `absl::Mutex` for all pools (13 methods acquire it) | Adequate for Phase 1. Expected bottleneck is Python→C++ boundary, not lock contention. Future: per-pool locking if contention measured. |
| **Actor selection** | O(n_actors) per `SelectActorFromPool` — iterates all actors, then `std::min_element` | Fine for Ray Data's typical 4-64 actors per pool. At 1000+ actors, would need a priority queue. |
| **Locality computation** | `ComputeNodeLocalityMap` called per task — O(n_args) `GetLocalityData` lookups | Hoisted outside per-actor loop (key optimization). For tasks with many args, consider caching. |
| **Cython boundary** | 1 GIL release + 1 reacquire per task submission. No batching. | Acceptable for current throughput targets. |
| **Benchmarks** | None exist — only correctness tests (max 3 actors) | Need microbenchmarks before production rollout. |

---

## 13. Test Coverage Summary

| Test Suite | File | Count | Scope |
|-----------|------|-------|-------|
| C++ unit tests | `src/ray/core_worker/tests/actor_pool_manager_test.cc` | 22 | Pool lifecycle, selection, locality, backpressure, actor death/restart, in-flight tracking |
| Work queue tests | `src/ray/core_worker/tests/actor_pool_work_queue_test.cc` | 6 | FIFO, push/pop, clear, stress |
| C++ reconstruction | `src/ray/core_worker/tests/actor_pool_reconstruction_test.cc` | 2 | Actor removal redirect, cascading two-pool reconstruction |
| Task manager tests | `src/ray/core_worker/tests/task_manager_test.cc` | 4 | Pool task lineage pinning, resubmit, actor dependency move |
| Python API tests | `python/ray/tests/test_actor_pool_v2.py` | 16 | Creation, submission, scaling, stats, shutdown, cross-actor retry |
| Parity tests | `python/ray/data/tests/test_actor_pool_map_operator.py` | 3 | Legacy vs core path comparison (map, scaling, stats) |
| Data integration | `python/ray/data/tests/test_actor_pool_map_operator.py` | 39+ total | Full operator lifecycle |
| Reconstruction (Data) | `python/ray/data/tests/test_actor_pool_reconstruction.py` | 3 | E2E with Ray Data: single pool, cascading with sort, cascading without sort |

---

## 14. Known Risks, Open Questions & Future Work

### Open questions for discussion

1. **Queue timeout for "all actors dead" case?** Should we add a timeout, or rely on Ray Data's operator-level timeout?
2. **Dashboard integration** — pool-level metrics not exposed. Should `PoolStats` feed into task event buffer?
3. **`ray list actors`** shows individual actors, not pool grouping. Worth adding pool metadata to actor state?

### Known risks

| Risk | Location | Impact |
|------|----------|--------|
| Work items not cleaned up on `UnregisterPool` | `actor_pool_manager.cc:97-99` (TODO C2) | `TaskArg` objects leak until CoreWorker destruction |
| Async retry timers capture raw `this` pointer | `actor_pool_manager.cc:658-670` | Safe in practice (`io_service_.stop()` cancels timers before destruction), but relies on shutdown ordering discipline |
| No observability integration | — | No dashboard, metrics export, or task events for pool-level data |
| `ActorPool.shutdown()` always force-kills actors | `actor_pool.py:452-456` | `force=False` still calls `ray.kill(no_restart=True)`, preventing lineage reconstruction for in-flight outputs. Should drop references instead (matching legacy `_ActorPool` behavior) |

### Future work

- Core-controlled autoscaling (`SelectActorForRemoval`)
- `ray_remote_args_fn` support
- Event-driven actor state (callback from ActorManager instead of polling)
- Per-pool locking (if contention measured)
- Microbenchmarks (selection latency, throughput at scale)
- Streaming generator support in `ActorPool.submit()` (currently only returns `ObjectRef`; the Data adapter handles generators directly via Cython)
- `ActorPool` as context manager (`__enter__`/`__exit__`)
- Graceful shutdown with drain-before-kill (`grace_period_s` parameter is currently unused)

---

## Appendix: Anticipated Questions & Answers

### Cross-Actor Retry

**Q: Why not use Ray's built-in `max_retries` instead of building pool-level retry?**

Ray's task retry is actor-bound. When you submit a task to actor A with `max_retries=3`, the `TaskSpec` is stamped with actor A's `ActorId` at submission time. All 3 retries go to the *same* actor A. Cross-actor retry requires a layer that can re-route the task to actor B. We use `max_retries=-1` (infinite) to keep the `ObjectRef` alive through failures, then `InternalHeartbeat` redirects the task to a healthy pool actor. The pool's `ActorPoolManager` provides the cross-actor selection logic (`SelectActorForTask`), while `ActorTaskSubmitter`'s infinite retry mechanism keeps the task alive.

**Q: What happens if a task fails mid-stream with a streaming generator?**

`HandlePushTaskReply` fires once at the end of the generator (or on failure), not per-yielded-item. When it fires with an error, `MaybeNotifyPoolTaskComplete` notifies the pool, and the pool retries the *entire* generator on a different actor. Partial results from the failed generator are not reused — the new actor re-executes from scratch. This matches Ray Data's existing behavior where a failed map task is fully re-executed.

**Q: Can a retried task land on the same actor that just failed?**

Yes, in theory. `SelectActorFromPool` picks the actor with the best `(locality_rank, load)`. If the failed actor is still alive (e.g., a transient `ACTOR_UNAVAILABLE` error), it remains a candidate. However, `OnActorDead` marks dying actors as `is_alive=false`, so dead actors are excluded from selection. A restarted actor becomes a candidate again via `OnActorAlive`. Since other surviving actors likely have lower load, the failed actor is *unlikely* to be re-selected immediately. There is no explicit exclusion list — that's a potential Phase 2 improvement.

**Q: What if all actors die simultaneously (e.g., node crash)?**

`SelectActorFromPool` returns `ActorID::Nil()` and the work item is pushed to the `PoolWorkQueue`. It stays queued indefinitely until a new actor is added via `AddActorToPool`, which calls `DrainWorkQueue`. There is **no timeout** — tasks wait forever. This is a known medium-risk gap (Section 14). The expectation is that Ray Data's operator-level timeout or autoscaler handles this, but it's an open question for discussion.

### Lifetime & Safety

**Q: The retry timer captures `this` (ActorPoolManager). Is that safe on shutdown?**

The `boost::asio::steady_timer` in `ScheduleRetry` (`actor_pool_manager.cc:633`) captures a raw `this` pointer. This is safe in practice because:

1. CoreWorker's shutdown executor calls `io_service_.stop()` before member destruction begins (`core_worker_shutdown_executor.cc:84`)
2. `io_service_.stop()` cancels all pending async operations, causing timers to fire with an error code
3. The timer callback checks for the error code and returns early (`actor_pool_manager.cc:641-643`)
4. `actor_pool_manager_` (declared at `core_worker.h:1898`) is destroyed *before* `io_service_` (declared at `core_worker.h:1814`) due to reverse declaration order, but `io_service_` is a reference (not owned), so it's not destroyed by CoreWorker

The sequence is: `io_service_.stop()` → timers cancelled → `~ActorPoolManager()`. No use-after-free. That said, this relies on shutdown ordering discipline — there are no explicit timer cancellation calls, and if someone changes the shutdown sequence, it could break.

**Q: What happens to in-flight tasks when `UnregisterPool` is called?**

In-flight tasks on actors belonging to the unregistered pool continue executing on the worker. When they complete, `HandlePushTaskReply` fires → `MaybeNotifyPoolTaskComplete` → `OnPoolTaskComplete` checks if the pool still exists (`actor_pool_manager.cc:340-344`) and silently ignores the completion. The task's result is still returned to the caller normally via `TaskManager`. The pool just stops tracking it.

The `work_items_` entries for the unregistered pool leak (TODO C2 at `actor_pool_manager.cc:116-118`). These hold `TaskArg` objects until the CoreWorker itself is destroyed.

### Performance

**Q: `CloneArgs` serializes to proto and back on every submission. Isn't that expensive?**

Yes, and it's called on *every* `SubmitToActor` call (`actor_pool_manager.cc:500`), not just retries. The original args are moved into `work_items_` for retry tracking, so a clone is needed for the actual submission. The cost is: proto serialization + `LocalMemoryBuffer` deep copy per by-value arg + `ObjectID::FromBinary` per by-reference arg.

For the happy path (no retry needed), this is pure overhead. A future optimization could use `shared_ptr<TaskArg>` to share args between the submission and retry store, or defer cloning until an actual retry occurs. For Phase 1 with typical Ray Data workloads (by-reference args pointing to object store blocks), the cost is modest since by-reference args are just a small proto with an ObjectID + owner address.

**Q: Extracting `arg_ids` also serializes each arg to proto (`SubmitTaskToPool:198-207`). Can that be cheaper?**

Yes. `TaskArgByReference` holds the `ObjectID` directly as a member (`id_` in `task_util.h`). A virtual `GetObjectID()` method on `TaskArg` could return it without serialization. This pattern appears in both `SubmitTaskToPool` and `RetryWorkItem`, so the savings would compound. This is a good follow-up optimization.

**Q: `SelectActorFromPool` is O(n_actors). Does that matter?**

For Ray Data's typical pool sizes (4-64 actors), this is negligible. The function iterates actors twice: once to build the candidate list, and once for `std::min_element` with `RankActor`. At 1000+ actors, a priority queue (keyed by rank, updated on in-flight changes) would be better. Not needed for Phase 1.

**Q: One `absl::Mutex` for all pools — won't that be contended?**

Unlikely to be the bottleneck. The lock is held for microseconds (in-memory map lookups, counter increments). The real serialization point is the Python→C++ boundary: each `submit_task_to_pool` call acquires the GIL, releases it for C++, then reacquires. That GIL round-trip dominates. Per-pool locking is a future optimization if profiling shows mutex contention.

### Backpressure

**Q: `num_free_task_slots` uses Python-side `num_running_actors()` for capacity but C++ `GetOccupiedTaskSlots` for occupancy. Can these diverge?**

Yes. During the pending→running transition, there's a window where C++ already has the actor (added in `scale()` → `add_actor_to_pool`) but Python hasn't moved it to `_running_actors` yet (waiting for `get_location.remote()` to resolve). During this window, `num_running_actors()` under-counts, so `capacity` is lower than actual, and `num_free_task_slots` is conservatively low. This means the operator may briefly under-utilize the pool. It self-resolves once `pending_to_running` completes. Not a correctness issue — just a brief throughput dip during scale-up.

**Q: `SelectActorFromPool` falls back to over-capacity actors when all are busy (line 397-402). Doesn't that break backpressure?**

The fallback ensures callers always get valid `ObjectRef`s instead of empty results. The over-submitted tasks queue inside `ActorTaskSubmitter`'s per-actor queue and execute normally. `GetOccupiedTaskSlots` correctly reports the inflated in-flight count, so `num_free_task_slots` goes to 0, and Python stops submitting new tasks. There's no deadlock — tasks drain naturally. The backpressure signal is delayed by one task (the one that triggered the over-submission), which is acceptable.

**Q: On the legacy direct-submission path, does `GetOccupiedTaskSlots` still work?**

`CoreActorPoolAdapter` always queries C++ via `GetOccupiedTaskSlots`. For legacy-path tasks (submitted directly to actors, not through C++ pool), the C++ pool doesn't know about them — its `num_tasks_in_flight` counters won't reflect those tasks. However, the adapter is only instantiated when `supports_pool_submission = True`, meaning the legacy path uses the old `_ActorPool` class, not the adapter. So there's no mismatch in practice.

### Lineage & Memory

**Q: Pool tasks use `max_retries=-1`. How does lineage reconstruction work?**

With `max_retries=-1`, `num_retries_left_` is `-1` (infinite), so the standard `num_retries_left_ != 0` check already makes pool tasks eligible for lineage reconstruction. The `IsPoolTask()` override in the lineage check (`task_manager.cc:1064-1066`) is a defensive safeguard in case `max_retries` is ever changed to a finite value. During reconstruction, `InternalHeartbeat` detects the pool task, calls `SelectActorForTask` to pick a healthy actor, calls `MovePoolTaskActorDependency` to update reference counts, and resubmits to the new actor.

**Q: For streaming generators, what exactly gets pinned for lineage?**

Each yielded object that lands in plasma gets its own `ObjectID` added to `reconstructable_return_ids_` (`task_manager.cc:1010-1032`). It's not just the generator ID — it's the full set of plasma-stored yielded objects. The `TaskSpec` (which is what gets pinned for lineage) is shared across all of them, so the memory cost is one `TaskSpec` per generator task, not per yielded item.

**Q: Could lineage memory blow up for long-running streaming pipelines?**

Bounded by the existing `max_lineage_bytes_` limit (same mechanism as regular tasks). When `total_lineage_footprint_bytes_` exceeds the limit, oldest tasks are evicted via LRU (`task_manager.cc:1072-1077`). For streaming pipelines with millions of tasks, this budget prevents unbounded growth. The per-task cost is `spec.ByteSizeLong()` bytes — typically a few KB for Ray Data map tasks.

### Integration & Migration

**Q: What if we need to roll back from core pool to legacy?**

Unset the `RAY_DATA_USE_CORE_ACTOR_POOL` env var (or set to 0). The feature flag is checked at `ActorPoolMapOperator.__init__` time (`actor_pool_map_operator.py:204`). Both paths coexist in the codebase indefinitely. No data migration or state cleanup needed — pool state is ephemeral (lives only in CoreWorker memory for the duration of the job).

**Q: Why can't the core pool support `ray_remote_args_fn`?**

The core pool creates actors with static `actor_options` baked in at `ActorPool.__init__` time. `ray_remote_args_fn` generates *per-actor* dynamic options (e.g., rotating GPU assignments). The core pool has no hook for injecting dynamic args into actor creation. When `ray_remote_args_fn` is set, the operator automatically falls back to the legacy path (`actor_pool_map_operator.py:204-205`). Supporting this requires adding a callback-based actor factory to `ActorPool` — deferred to future work.

**Q: How do parity tests work? What guarantees behavioral equivalence?**

Three dedicated parity tests (`test_actor_pool_map_operator.py:1542-1620`) run the same workload through both the legacy and core paths and compare results: `test_actor_pool_map_operator_parity` (correctness), `test_actor_pool_scaling_parity` (scale up/down behavior), and `test_actor_pool_stats_parity` (stats reporting). They use `monkeypatch` to toggle `use_core_actor_pool` and assert output equivalence.

### Public API Design (`ray.experimental.actor_pool`)

**Q: How does this relate to the existing `ray.util.ActorPool`?**

Different API, different purpose. `ray.util.ActorPool` takes a pre-created list of actor handles, returns direct results (calls `ray.get()` internally), and provides iterator methods (`get_next`, `get_next_unordered`, `map_unordered`). It has no retries, no autoscaling, no C++ backing. The new `ray.experimental.ActorPool` takes an actor *class*, manages actor lifecycle itself, returns `ObjectRef`s (caller decides when to `ray.get()`), and delegates selection/retry/backpressure to C++. They coexist — neither replaces the other yet.

**Q: `submit()` returns `ObjectRef`, but what if C++ queues the task (no actor available)?**

If all actors are alive but at capacity, `SelectActorFromPool` falls back to over-submitting to the least-loaded alive actor (`actor_pool_manager.cc:397-402`). The caller always gets a valid `ObjectRef`. If there are *zero* alive actors, C++ queues internally in `PoolWorkQueue` and returns an empty ref list, which makes `submit()` raise `RuntimeError("Failed to submit task to pool: no alive actors with capacity")` (`actor_pool.py:346-349`). This means there's no way to "submit and wait for an actor to become available" — the caller must retry. This is a gap compared to `ray.util.ActorPool` which silently queues in `_pending_submits`.

**Q: `size=4` sets `min_size=0, max_size=-1`. Doesn't that mean the pool can scale down to 0?**

Yes. The `size` parameter only sets `initial_size` — it doesn't constrain autoscaling bounds (`actor_pool.py:137-142`). `scale(-4)` on a `size=4` pool succeeds and leaves 0 actors. Subsequent `submit()` raises. This is intentional for the Data adapter path (where `min_size`/`max_size` are set explicitly), but surprising for standalone users who expect `size=4` to mean "always 4 actors." A standalone user who wants a fixed-size pool should use `ActorPool(actor_cls=W, min_size=4, max_size=4)`.

**Q: No `get_next()` or `get_next_unordered()` — how do users process results incrementally?**

They use standard Ray primitives: `ray.wait(refs, num_returns=1)` for as-completed processing, or `ray.get(refs)` for ordered batch retrieval. The new API returns `ObjectRef`s and stays out of result iteration. This is a deliberate tradeoff — the old `ray.util.ActorPool` hides `ray.get()` behind iteration, which couples pool lifecycle to result consumption. The new API decouples them: submit tasks → hold refs → process results however you want.

**Q: `submit()` takes `method_name` as a string. Why not `pool.submit(actor.method, *args)` like the old API?**

The old API's `submit(fn, value)` takes a lambda `fn(actor, value)` where the caller picks the method. The new API needs to resolve the method signature and function descriptor for C++ submission (`actor_pool.py:322-324`). It uses `self._actor_handles[0]` to look up `_ray_method_signatures[method_name]` and `_ray_function_descriptor[method_name]`. A string name is the simplest way to do that lookup. A `pool.submit(Worker.method, *args)` style would require introspecting the unbound method to extract the descriptor — doable but more complex.

**Q: `submit()` uses `self._actor_handles[0]` to get the function descriptor. What if actors have different methods?**

All actors in a pool are instances of the same `actor_cls`, so their method signatures and descriptors are identical. Using `[0]` is safe — it's just a descriptor lookup, not a submission target (C++ picks the actual actor). The only risk is if `_actor_handles` is empty, which is guarded by the `if not self._actor_handles: raise RuntimeError` check on line 318.

**Q: `map()` is just a loop calling `submit()`. No batching, no backpressure. Won't this flood the pool?**

Yes. `map()` (`actor_pool.py:372-377`) submits all items eagerly and returns all `ObjectRef`s. If you pass 1M items, it creates 1M work items in C++. For large workloads, users should implement their own batching with `ray.wait()` between batches, or use Ray Data (which has `can_add_input()` backpressure). The `map()` method is a convenience for small-to-medium workloads, not a replacement for Ray Data's streaming execution.

**Q: Can users call actor methods directly via `pool.actors[i].method.remote()`, bypassing the pool?**

Yes. The `actors` property (`actor_pool.py:255-257`) returns actor handles, and nothing prevents direct calls. This bypasses C++ selection, retry, load balancing, and in-flight tracking. The pool's `num_tasks_in_flight` won't count these tasks, and backpressure will be wrong. This is a leaky abstraction — useful for debugging or one-off diagnostics, but misuse silently breaks pool semantics. Worth documenting.

**Q: `shutdown()` takes `grace_period_s` but it's unused. What actually happens to in-flight tasks?**

`shutdown()` calls `ray.kill(actor, no_restart=True)` on every actor immediately (`actor_pool.py:442-446`), then unregisters the pool. In-flight tasks on those actors will fail with `ACTOR_DIED`. Since the pool is unregistered, `OnPoolTaskComplete` ignores the completion (`actor_pool_manager.cc:340-344`), so no retry is attempted. The `grace_period_s` parameter is a placeholder (TODO P7). A real drain-before-kill would need to wait for `stats()["total_in_flight"] == 0` before killing.

**Q: `__del__` calls `unregister_actor_pool` but doesn't kill actors. Isn't that a resource leak?**

Yes. If a user drops all references to the pool without calling `shutdown()`, `__del__` (`actor_pool.py:455-466`) unregisters the C++ pool but leaves actors running. Those actors continue consuming cluster resources until the driver exits or they crash. This is a known gap — `__del__` is best-effort cleanup (may not even run if the interpreter is shutting down). The docstring should recommend explicit `shutdown()`, ideally via a context manager (not yet implemented).

**Q: `RetryPolicy.max_attempts` — is that total attempts or retries after the first try?**

Retries after the first try. The C++ field is `max_retry_attempts` (`actor_pool_manager.h:86`), and the check in `OnTaskFailed` is `work_item.attempt_number > config.max_retry_attempts` (`actor_pool_manager.cc:570-571`). `attempt_number` starts at 0 and increments on each failure. So `max_attempts=3` means up to 3 retries (4 total executions). The Python field is named `max_attempts` but maps to `max_retry_attempts` in C++ — this naming asymmetry could confuse users.

**Q: The `logical_id_label_key`, `logical_id_kwarg_name`, and `static_labels` parameters feel Ray Data-specific. Should they be on the public API?**

They're there because `CoreActorPoolAdapter` needs them to integrate with Ray Data's actor tracking (`core_actor_pool_adapter.py:140-142`). For a standalone user, these are noise — you'd never set them. A cleaner design would move these to the adapter or a subclass, keeping the public `ActorPool` API minimal. For Phase 1 this is acceptable since the API is `experimental`, but worth revisiting before promoting to stable.

**Q: Why isn't `ActorPool` a context manager?**

No `__enter__` / `__exit__` methods are defined. Users must remember to call `shutdown()`. Adding context manager support (`with ActorPool(...) as pool:`) would be straightforward and prevent resource leaks. This is a common Python API pattern for resource-owning objects.

### Proto & Wire Compatibility

**Q: Adding fields 46-47 to `TaskSpec` — could this break older workers?**

No. Both fields are `optional bytes` (protobuf 3 syntax). Older workers that don't know about these fields will: (a) ignore them when deserializing (standard protobuf behavior), (b) pass them through unchanged if forwarding, and (c) never set them, so `IsPoolTask()` returns false on any task they generate. Newer workers handle missing fields gracefully — empty `actor_pool_id` means "not a pool task". This is the standard Ray pattern for extending `TaskSpec`.

---
