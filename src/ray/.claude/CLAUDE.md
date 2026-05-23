<!-- Loaded on-demand when Claude works on C++ core files. -->

# Ray Core (C++ runtime + Python bridge)

## Key Modules
- `src/ray/core_worker/` — per-process worker. Task submission, object put/get, ref counting, actor mgmt. Entry: `core_worker.h/.cc`, `task_submission/{normal,actor}_task_submitter.h`, `reference_counter.h`.
- `src/ray/raylet/` — per-node daemon. Worker leasing, local scheduling, worker pool, spilling. Entry: `node_manager.h/.cc`, `scheduling/`, `worker_pool.h`.
- `src/ray/gcs/` — Global Control Store. Authoritative registry for actors, nodes, jobs, placement groups. Entry: `gcs_server.h/.cc`, `gcs_actor_manager.h`, `gcs_placement_group_scheduler.h`.
- `src/ray/object_manager/` — cross-node object transfer + Plasma. Entry: `object_manager.h/.cc`, `pull_manager.h`, `push_manager.h`, `plasma/`.
- `src/ray/rpc/` — gRPC server/client infra (`grpc_server.h`, `grpc_client.h`, `server_call.h`).
- `src/ray/common/` — IDs, function descriptors, buffers, protobuf helpers.
- `src/ray/protobuf/` — source of truth for all cross-process messages. Edit `.proto`, never the generated files.

## Python ↔ C++ bridge
- `python/ray/_raylet.pyx` is the only Cython bridge; wraps `CCoreWorkerProcess`/`CCoreWorker`. The `CoreWorker` Python class is here; `submit_task()` is what every `.remote()` call funnels through.
- `python/ray/includes/*.pxd` are the Cython declarations of C++ classes exposed to Python.

## Architecture flow
`f.remote()` → `RemoteFunction` (`_private/worker.py`) → `CoreWorker.submit_task` in `_raylet.pyx` → `CCoreWorker::SubmitTask`. `CoreWorker` runs two `boost::asio` loops (IO/RPC + task execution) on separate threads. `NormalTaskSubmitter` sends `RequestWorkerLease` to local raylet; raylet schedules a worker, then the task spec is delivered to that worker's `CoreWorker` directly (raylet is out of the data path). Results go into local Plasma; `ObjectRef`s carry an owner address and the owner's `ReferenceCounter` tracks borrowers. Cross-node fetches go through `ObjectManager` pull/push.

## Gotchas
- **Stale `_raylet.so`** after editing `.pxd` or exposed C++ headers → cryptic import errors. Use `/rebuild`; never trust an incremental build for Cython changes.
- **Protobuf** changes need a full Bazel rebuild; `*_pb2.py` and generated headers are NOT checked in. Never hand-edit them.
- **Bazel pinned to 7.5.0** via bazelisk; system gRPC/protobuf installs conflict on macOS — uninstall first.
- **Threading:** never block the IO thread. C++ → Python callbacks must `PyGILState_Ensure/Release` (see `_raylet.pyx`).
- **GCS vs raylet:** raylet holds only local scheduling state. Actor location, node liveness, PGs, jobs — all GCS. Handle `HandleRayletNotifyGCSRestart` if you cache GCS state.
- **Ownership:** every `ObjectRef` has exactly one owner. Borrowers must notify the owner before dropping or the owner leaks the object. Forgetting `AddLocalReference`/`RemoveLocalReference` in Cython causes premature eviction or leaks.
- **RPC style:** most C++ RPCs are async callback-based via `ClientCallManager`. A few sync gRPC stubs in `gcs_utils.py` can deadlock if called from the IO thread.
- **Security:** new gRPC endpoints/RPC handlers need token auth, propagated on both sides — see `.claude/rules/security.md`.
