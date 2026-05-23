<!-- Loaded on-demand when Claude works on Ray Serve files. -->

# Ray Serve

## Key Modules
**Control plane** (single actor):
- `_private/controller.py` — `ServeController`. Owns `DeploymentStateManager`, `ApplicationStateManager`, `AutoscalingStateManager`, `LongPollHost`. Checkpoints to GCS via `kv_store`.
- `_private/deployment_state.py` — reconcile loop: `target_num_replicas` → actual Ray actors. No Ray-level actor auto-restart; the manager replaces crashed replicas.
- `_private/application_state.py` — app state machine (`DEPLOYING → RUNNING / UNHEALTHY`).
- `_private/autoscaling_state.py` — aggregates pushed metrics, runs policy, emits new `target_num_replicas`.

**Data plane:**
- `_private/proxy.py` — `ProxyActor`, one per node. Runs uvicorn (HTTP) + grpcio (gRPC) in the same actor.
- `_private/proxy_router.py` — `ProxyRouter`, URL→deployment table populated lazily, long-poll-driven thereafter.
- `_private/replica.py` — `ReplicaActor`. Executes user code; asyncio semaphore gates `max_ongoing_requests`; pushes queue/util metrics to controller.
- `_private/router.py` — client-side `Router` inside each `DeploymentHandle`. Uses `PowerOfTwoChoicesRequestRouter`; queues client-side when all replicas saturated.
- `handle.py` — `DeploymentHandle`: user-facing client wrapper around `Router`. Not an actor.
- `_private/long_poll.py` — `LongPollHost` (in controller) ↔ `LongPollClient` (in each proxy/router); snapshot-ID delta protocol.

**User API** (everything outside `_private/`): `api.py` (`serve.run`), `deployment.py` (`@serve.deployment`), `handle.py`, `config.py`, `batching.py`, `multiplex.py`.

## Architecture flow
HTTP/gRPC → `ProxyActor` → `ProxyRouter` lookup → `Router` picks a replica via power-of-two-choices (probe two, pick shorter queue, respect `max_ongoing_requests`) → `ReplicaActor` runs user callable. Meanwhile the controller's tick-driven `update()` loop reconciles desired-vs-actual replicas, advances app state, and runs autoscaling against metrics *pushed* from routers/replicas. State changes broadcast via long-poll: clients block on `listen_for_change`, unblock when snapshot ID advances. `.bind()` produces an `Application` DAG resolved at `serve.run`; each DAG edge becomes a `DeploymentHandle` held by the calling replica. `@serve.multiplexed` replicas advertise loaded `model_id`s (LRU up to `max_num_models_per_replica`); router prefers replicas with the model loaded, falls back otherwise.

## Gotchas
- **Controller is a single-actor bottleneck and SPOF for control plane.** Crash → autoscaling pauses, transient metrics lost; recovery from GCS checkpoint. Requests keep flowing.
- **Long-poll is pull**; autoscaling metrics are push. Don't confuse them.
- **`serve.run` is NOT idempotent declarative.** It diffs against running config and triggers a rolling redeploy for any changed deployment. Only `user_config` changes via `reconfigure()` avoid teardown.
- **`DeploymentHandle` is per-call-site.** Two handles to the same deployment have independent `Router`s, queues, and load-balancing state — no shared backpressure.
- **Async-everywhere.** Each replica runs one asyncio loop; a sync `__call__` blocks the whole replica. Wrap blocking work in `run_in_executor` or use `async def`.
- **`@serve.batch` vs `max_ongoing_requests`:** the batched call counts as ONE ongoing request from the router's view but holds many user requests. Too-low `max_ongoing_requests` starves the batcher.
- **HTTP/gRPC parity is surface-level** — same actor, same router, but gRPC needs `grpc_servicer_functions` at startup and the replica must implement the servicer interface. Middleware/interceptors are separate code paths.
- **`Application` (named DAG from `.bind()` + `serve.run`) ≠ `Deployment` (decorated class+config) ≠ `Replica` (running actor).** Autoscaling moves replica count; config changes require a new deployment version.
- **Security:** new gRPC endpoints/RPC handlers need token auth on both sides — see `.claude/rules/security.md`.
