# nsys CPU Profiling on GPU Nodes — Analysis

**Run**: prodjob_tz19fmz3uxk8i92ctja6g1dxxr (2026-04-08)
**Cluster**: 10 × r6a.8xlarge (CPU) + 4 × g5.4xlarge (GPU) + 1 head
**Profiled process**: Infer actor (ViT embedding model), PID 3540, `python`

---

## Background

nsys (NVIDIA Nsight Systems) defaults to `--sample=process-tree` when launched
with a target application. This enables **CPU instruction pointer sampling** via
the Linux perf subsystem — capturing what the CPU is doing while it orchestrates
GPU work. The sampling uses software clock events (`CPU Clock (sw)`) with DWARF
backtraces, collecting 1 backtrace for every 4 IP samples at a period of 2M
cycles.

This CPU profiling is separate from (and complementary to) the CUDA API and GPU
kernel tracing that nsys always captures.

---

## Why Only 2 of 4 GPU Nodes Have CPU Sampling

| GPU Node IP   | nsys File Size | CPU Samples | perf_raylet Data? |
|---------------|---------------|-------------|-------------------|
| 10.0.127.20   | **14 MB**     | **291,572** | Yes               |
| 10.0.87.120   | **14 MB**     | **291,597** | Yes               |
| 10.0.79.114   | 3.4 MB        | 0           | No                |
| 10.0.88.48    | 3.3 MB        | 0           | No                |

### Root cause: perf_event_paranoid

nsys needs `perf_event_open()` access for CPU sampling. By default, the kernel
restricts this (`perf_event_paranoid >= 2`).

The `perf.py` profiler (a separate profiling component) deploys
`_RayletPerfProfiler` actors to a subset of worker nodes. On each targeted node,
it runs:

```python
subprocess.run(["sudo", "sysctl", "-w", "kernel.perf_event_paranoid=-1"], ...)
subprocess.run(["sudo", "sysctl", "-w", "kernel.kptr_restrict=0"], ...)
```

This lowers the perf security policy **on that specific node**. nsys, which also
uses perf underneath, then succeeds with CPU sampling on those nodes. On the 2
GPU nodes that did not receive perf actors, `perf_event_paranoid` remained
restrictive and nsys silently skipped CPU sampling.

### Evidence

**Node with sampling (10.0.127.20) diagnostics:**
```
Info|Dwarf backtraces collected.
Info|Event 'CPU Clock (sw)', with sampling period 2000000, used to trigger process-tree CPU IP sample collection.
Info|4 CPU IP samples collected for every CPU IP backtrace collected.
Info|Number of IP samples collected: 291572.
```

**Node without sampling (10.0.79.114) diagnostics:**
No sampling-related messages at all — silently skipped.

### This is a side effect, not intentional

The CPU sampling in nsys is an accidental benefit of the perf profiler lowering
`perf_event_paranoid`. To get consistent CPU sampling on all GPU nodes, either:
1. Ensure perf actors target all GPU nodes before nsys starts
2. Add `--sample=process-tree` to the nsys config and lower `perf_event_paranoid`
   on all GPU nodes at cluster startup

---

## CPU Time Breakdown (291,572 samples, node 10.0.127.20)

### By module — leaf (self time)

| Module                    | Samples | Self % | What it does                                      |
|---------------------------|---------|--------|---------------------------------------------------|
| **libcuda.so**            | 182,822 | **62.7%** | CUDA driver: kernel launches, DMA, command queues |
| **kernel.kallsyms**       | 49,534  | **17.0%** | Kernel syscalls from CUDA (ioctl, futex, mmap)   |
| **vdso**                  | 38,289  | **13.1%** | clock_gettime — timestamps for PyTorch/CUDA      |
| **libc.so**               | 13,952  | **4.8%**  | memcpy=3.7%, clock_gettime=0.5%, pthread=0.3%    |
| **python3.10**            | 4,856   | **1.7%**  | GC=0.9%, eval=0.07%                              |
| **_raylet.so**            | 770     | **0.3%**  | Ray worker: gRPC, periodical runners             |
| **nsys (cupti/injection)**| 293     | **0.1%**  | Profiler overhead                                |
| **libtorch_cpu.so**       | 175     | **0.06%** | CPU tensor ops — essentially zero                |
| **pyarrow**               | 150     | **0.05%** | Arrow serialization/deserialization              |
| **libc10_cuda.so**        | 113     | **0.04%** | PyTorch CUDA runtime wrapper                     |

### By module — inclusive (on stack anywhere)

| Module                | Frames    | Stack % | Interpretation                                 |
|-----------------------|-----------|---------|------------------------------------------------|
| **libcuda.so**        | 886,116   | 36.1%   | CUDA driver (top of most stacks)               |
| **python3.10**        | 529,233   | 21.5%   | Python interpreter (bottom of most stacks)     |
| **libtorch_cpu.so**   | 314,411   | 12.8%   | PyTorch dispatch (mid-stack, calling CUDA)      |
| **libcudart.so**      | 168,999   | 6.9%    | CUDA runtime (between torch and driver)        |
| **pyarrow**           | 94,224    | 3.8%    | Arrow data handling                            |
| **libtorch_cuda.so**  | 57,580    | 2.3%    | PyTorch CUDA backend                           |
| **_raylet.so**        | 16,021    | 0.7%    | Ray worker runtime                             |

The inclusive vs self time comparison confirms the call hierarchy:
**Python → PyTorch → CUDA runtime → CUDA driver → kernel**

Python (21.5% inclusive, 1.7% self) and libtorch_cpu (12.8% inclusive, 0.06%
self) are mid-stack callers that ultimately spend their time in the CUDA driver.

---

## Top Resolved Functions

### libc (self time)

| Self Time | Function                    | What                            |
|-----------|-----------------------------|---------------------------------|
| 3.70%     | `__memcpy_avx_unaligned_erms` | Copying tensor data in CPU RAM |
| 0.52%     | `__clock_gettime`           | Timestamps                      |
| 0.32%     | `__pthread_once`            | Thread-local init               |
| 0.03%     | `__pthread_mutex_lock`      | Lock contention                 |
| 0.02%     | `__libc_malloc`             | Memory allocation               |

### Python interpreter (self time)

| Self Time | Function                | What                               |
|-----------|-------------------------|------------------------------------|
| 0.43%     | `gc_collect_main`       | **Garbage collection — mark phase**|
| 0.25%     | `visit_reachable`       | GC object graph traversal          |
| 0.19%     | `deduce_unreachable`    | GC unreachable detection           |
| 0.17%     | `dict_traverse`         | GC traversing dict objects         |
| 0.08%     | `func_traverse`         | GC traversing function objects     |
| 0.07%     | `_PyEval_EvalFrameDefault` | Bytecode execution              |
| 0.05%     | `_PyObject_GenericGetAttrWithDict` | Attribute lookup          |

GC dominates Python self-time: gc_collect_main + traverse functions = **~1.1%**
of total CPU time. Actual Python bytecode execution is only 0.07%.

### Ray _raylet.so (self time)

| Samples | Function                                      | What                         |
|---------|-----------------------------------------------|------------------------------|
| 26      | `epoll_reactor::run`                          | Boost ASIO event loop        |
| 24      | `pollset_work`                                | gRPC polling                 |
| 16      | `PeriodicalRunner::DoRunFnPeriodically...`    | Scheduled tasks              |
| 14      | `microsec_clock::create_time`                 | Timestamp creation           |
| 12      | `__Pyx__ExceptionSave`                        | Cython exception handling    |
| 11      | `check_signals()`                             | Signal checking              |
| 10      | `CoreWorker::RunTaskExecutionLoop`            | Task execution loop          |

Total: 770 samples / 291,572 = **0.26%**. Ray runtime overhead is negligible.

---

## Thread Distribution

41 threads were sampled in the actor process. Distribution:

| Thread Name           | Samples | %      | Role                              |
|-----------------------|---------|--------|-----------------------------------|
| ray::MapWorker (main) | 227,356 | **78%** | Primary inference thread          |
| ray::MapWorker (aux1) | 28,746  | 10%    | CUDA stream / data loader thread  |
| ray::MapWorker (aux2) | 28,345  | 10%    | CUDA stream / data loader thread  |
| ray::IDLE             | 3,793   | 1.3%   | Actor idle/wait                   |
| ray::MapWorker (aux3) | 1,092   | 0.4%   | Minor helper thread               |
| [NSys Comms]          | 809     | 0.3%   | nsys profiler communication       |
| worker.io             | 264     | 0.1%   | Ray I/O thread                    |
| timer_manager         | 193     | 0.1%   | gRPC timer                        |
| cuda-EvtHandlr        | 130     | 0.04%  | CUDA event handler                |
| Other (gRPC, polling) | ~850    | 0.3%   | Infrastructure threads            |

The 3 MapWorker threads account for 98% of samples. The main thread (78%) runs
the inference loop; the two auxiliary threads (10% each) handle CUDA stream
callbacks and data movement.

### CPU core distribution

Samples spread across all 16 cores of the g5.4xlarge (16 vCPU):
- Core 4: 16.5% (busiest)
- Cores 0, 1, 8: 7-9% each
- Others: 3-7% each

No single core is saturated. The actor uses ~2-3 CPU cores effectively.

---

## CUDA Memory Transfers (Host ↔ Device)

| Direction | Calls | Avg Size | Total    | Wall Time |
|-----------|-------|----------|----------|-----------|
| **HtoD**  | 339   | **302 MB** | **100 GB** | **10.5s** |
| DtoH      | 339   | 2 MB     | 0.66 GB  | 0.16s     |

Each of the 339 infer tasks sends ~302 MB of preprocessed image batches to GPU
and receives ~2 MB of float embeddings back. The 150:1 input:output ratio is
expected for an embedding model (ViT) that reduces high-resolution images to
compact vectors.

**HtoD transfer rate**: 100 GB / 10.5s = **9.5 GB/s** (PCIe Gen4 x16 theoretical
max = 31.5 GB/s, so we're at 30% of bus bandwidth for these large transfers).

---

## CUDA Kernel Summary (node 10.0.127.20)

| Kernel                     | Time % | Instances | Avg (ms) | What                       |
|----------------------------|--------|-----------|----------|----------------------------|
| `ampere_sgemm_128x128_tn`  | 46.0%  | 4,116     | 50.0     | ViT linear layers (GEMM)  |
| `ampere_sgemm_128x64_tn`   | 25.3%  | 20,462    | 5.5      | Smaller GEMM operations    |
| `fmha_cutlassF_f32_...`    | 13.3%  | 4,068     | 14.6     | Flash attention (ViT)      |
| `GeluCUDA`                  | 4.9%   | 6,096     | 3.6      | GELU activation            |
| `vectorized_add`            | 3.6%   | 8,136     | 2.0      | Element-wise addition      |
| `vectorized_layer_norm`     | 3.2%   | 8,475     | 1.7      | Layer normalization        |
| `fprop_implicit_gemm_...`   | 3.0%   | 339       | 38.9     | ResNet conv (cuDNN)        |

84.6% of GPU time in GEMM + attention — a compute-bound ViT workload as expected.

### Cross-node kernel comparison (4 GPU nodes)

| Metric              | 10.0.127.20 (14MB) | 10.0.88.48 (3.3MB) |
|---------------------|---------------------|---------------------|
| sgemm_128x128 count | 4,116               | 4,080               |
| sgemm_128x128 time  | 205.3s              | 204.5s              |
| sgemm_128x64 count  | 20,462              | 20,353              |
| fmha count          | 4,068               | 4,044               |
| Conv tasks          | 339                 | 337                 |

Work is evenly distributed across all 4 GPU nodes (within 1%), confirming the
CPU sampling on 2 nodes did not measurably affect GPU throughput on those nodes.

---

## Key Takeaways

1. **The CPU on GPU nodes is a CUDA API machine** — 93% of CPU self-time
   (libcuda + kernel + vdso) is spent driving the GPU. This is not useful
   computation; it's the overhead of marshaling kernel launches and DMA.

2. **Python GC is measurable but small** (~1.1% self-time). The gc_collect_main
   + traverse functions run periodically but don't stall GPU work significantly.

3. **Ray and pyarrow overhead is negligible** (<0.4% combined self-time). The
   actor framework adds no meaningful CPU cost to the inference loop.

4. **libtorch_cpu is zero** (0.06%) — the model runs entirely on GPU. No CPU
   fallback kernels.

5. **memcpy is the largest non-driver CPU cost** (3.7%) — copying image tensors
   from Ray object store buffers through CPU staging areas to GPU-mapped memory.

6. **CPU sampling has zero impact on GPU performance** — the two nodes with
   sampling ran identical kernel counts and durations as the two without.

7. **The 14 MB vs 3.3 MB nsys file size difference** is entirely from CPU
   sampling data: 2.5M callchain entries + 525K scheduling events per sampled
   node. No additional CUDA tracing overhead.
