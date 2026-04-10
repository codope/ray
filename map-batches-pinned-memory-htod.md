# Plan: Pinned-Memory Prefetch for `map_batches` Actor Path

## Context

The nsys profile shows that on GPU inference actors (`map_batches` with `ActorPoolStrategy`),
the CPU spends 3.7% self-time in `__memcpy_avx_unaligned_erms` — the CUDA driver's internal
pageable→pinned staging copy during `.to("cuda")`. During this copy, **the GPU is completely
idle**. With 339 batches × ~47ms idle per batch, this adds ~16s of GPU waste.

The `iter_torch_batches` path already has `pin_memory` + `DefaultCollateFn` + `finalize_fn`
with `move_tensors_to_device(non_blocking=True)`. But `map_batches` actors have **none of this
infrastructure** — the UDF receives numpy dicts from pageable memory and is responsible for
its own GPU transfer.

## Detailed Root Cause Analysis

### The profiler evidence (nsys_profile.png)

The Nsight Systems timeline screenshot shows three critical rows during the selected window:

1. **CPU cores (top)**: Active — executing `__memcpy_avx_unaligned_erms` (the AVX-optimized
   bulk memory copy in libc).
2. **CUDA HW (middle)**: **Completely empty.** No kernels running, no DMA engine active.
   The GPU is idle, waiting.
3. **Memcpy HtoD (Pageable)**: Shows a host-to-device transfer labeled **"Pageable"** — this
   is the CUDA driver's DMA transfer from its internal pinned staging buffer to GPU memory.

The bottom panel shows the call stack rooted at `__memcpy_avx_unaligned_erms`, tracing up
through CUDA runtime → PyTorch → Python → Ray MapWorker.

### What "Pageable" means and why the GPU idles

When `cudaMemcpy` (or PyTorch's `.to("cuda")`) is called on **pageable** (non-pinned) host
memory, CUDA cannot DMA directly from it — the OS could swap those pages at any time. So
CUDA performs a **two-phase serial transfer**:

```
Phase 1 — CPU memcpy (pageable host buffer → CUDA driver's internal pinned staging buffer)
           This is __memcpy_avx_unaligned_erms. GPU is IDLE because DMA hasn't started.

Phase 2 — DMA (pinned staging buffer → GPU device memory)
           This is the "Memcpy HtoD (Pageable)" row. GPU compute still can't start
           because the cudaMemcpy call is synchronous — it blocks until DMA completes.

Phase 3 — GPU kernels finally run.
```

Both phases are serial. The GPU does nothing during Phase 1 (CPU copy), and Phase 2 blocks
the calling thread until complete. For each of the 339 inference batches (302 MB each), this
creates a ~47ms idle bubble (15ms CPU copy + 32ms DMA at 9.5 GB/s observed).

### Why the data is in pageable memory

The data path for `map_batches` actor inference:

1. **Plasma object store**: Arrow blocks live in **mmap'd shared memory**
   (`src/ray/object_manager/plasma/shared_memory.cc` — `mmap(NULL, length_, PROT_READ |
   PROT_WRITE, MAP_SHARED, fd, 0)`). mmap'd memory is always pageable — CUDA pinned memory
   can't be shared between processes.

2. **Arrow → NumPy**: `ArrowBlockAccessor.to_numpy()` calls `to_numpy(zero_copy_only=False)`
   (`arrow_block.py:309`). This either zero-copy views the mmap'd buffer (still pageable)
   or copies to heap for chunked/dtype-incompatible arrays (still pageable).

3. **NumPy → Torch**: `torch.as_tensor(ndarray)` in `convert_ndarray_to_torch_tensor`
   (`torch_utils.py:176`) shares memory with NumPy when possible — still pageable. The
   comment at line 169 confirms: "The numpy array is not always writeable as it can come
   from the Ray object store."

4. **UDF `.to("cuda")`**: The UDF calls `.to("cuda")` on a tensor backed by pageable memory.
   This triggers the two-phase serial transfer described above.

**No component in this chain uses pinned memory.** The `pin_memory` option that exists in
`DefaultCollateFn` is only wired up for the `iter_torch_batches` consumer-side path, not
for `map_batches` actor execution.

### The gap

`batch_blocks()` already accepts a `collate_fn` parameter (`block_batching.py:22`), and
`DefaultCollateFn(pin_memory=True)` already creates pinned torch tensors
(`torch_utils.py:183: result = result.pin_memory()`). But `BatchMapTransformFn._pre_process`
never passes a `collate_fn` to `batch_blocks()` (`map_transformer.py:340-346`).

With pinned memory, the UDF could call `.to("cuda", non_blocking=True)`:
- Phase 1 disappears (CUDA can DMA directly from pinned memory)
- Phase 2 becomes asynchronous (CPU returns immediately, DMA runs in background)
- GPU idle bubble shrinks from ~47ms to near zero per batch

## Why Option D (collate_fn passthrough), not Option B (pin_memory/device params)

**Option B** adds `pin_memory` and `device` params directly to `map_batches()`. This silently
changes what the UDF receives: user sets `batch_format="numpy"` but gets `Dict[str, torch.Tensor]`
when `pin_memory=True`. This is a hidden contract violation — the UDF's input type changes based
on an unrelated flag.

**Option D** exposes `collate_fn` on `map_batches()`, matching `iter_torch_batches`. The user
explicitly opts in by providing a collate function, and the collate function's return type IS
what the UDF receives. No surprise. The infrastructure already exists:

- `batch_blocks()` accepts `collate_fn` (block_batching.py:22) — already handles the pipeline
- `BatchMapTransformFn._pre_process` calls `batch_blocks()` but doesn't pass `collate_fn` (map_transformer.py:340-346)
- `DefaultCollateFn(pin_memory=True)` already creates pinned tensors (torch_utils.py:183)
- `iter_torch_batches` already auto-detects `batch_format` from collate_fn type (iterator.py:509-519)

The plumbing gap is ~15 lines.

---

## Recommended Approach: Option D, then C

### Phase 1: Thread `collate_fn` through `map_batches` → `batch_blocks`

**Files to modify:**

1. **`python/ray/data/dataset.py`** (`map_batches`, ~line 489)
   - Add `collate_fn: Optional[Callable] = None` parameter
   - Auto-detect `batch_format` from collate_fn type (same pattern as iterator.py:509-519):
     - `ArrowBatchCollateFn` → force `batch_format="pyarrow"`
     - `NumpyBatchCollateFn` → force `batch_format="numpy"`
     - `PandasBatchCollateFn` → force `batch_format="pandas"`
   - Pass `collate_fn` through to `MapBatches` logical operator

2. **`python/ray/data/_internal/logical/operators/map_operator.py`** (`MapBatches`, ~line 186)
   - Add `collate_fn` field

3. **`python/ray/data/_internal/planner/plan_udf_map_op.py`** (~line 306)
   - Pass `collate_fn` to `BatchMapTransformFn` constructor

4. **`python/ray/data/_internal/execution/operators/map_transformer.py`** (`BatchMapTransformFn`, ~line 310)
   - Accept `collate_fn` in constructor, store as `self._collate_fn`
   - Pass to `batch_blocks(collate_fn=self._collate_fn)` in `_pre_process` (line 340)

That's it. `batch_blocks` already calls `collate(batch_iter, collate_fn=collate_fn)` when
collate_fn is provided (block_batching.py:48-49). No new logic needed.

### Phase 2 (future): Async prefetch for true pipelining

Wrap the collate step in `make_async_gen(num_workers=1, buffer_size=1)` inside
`_pre_process` to overlap pin+collation of batch N+1 with GPU compute of batch N.
This is independent of Phase 1 and uses the existing `make_async_gen` utility
(util.py:1053).

---

## What Changes for the User

Before:
```python
ds.map_batches(InferModel, compute=ray.data.ActorPoolStrategy(), num_gpus=1)
# UDF receives Dict[str, np.ndarray], must call .to("cuda") itself (pageable HtoD, GPU idles)
```

After (pinned CPU tensors — UDF handles device transfer):
```python
from ray.data import DefaultCollateFn

ds.map_batches(
    InferModel,
    compute=ray.data.ActorPoolStrategy(),
    num_gpus=1,
    collate_fn=DefaultCollateFn(pin_memory=True),
    # batch_format auto-set to "pyarrow" because DefaultCollateFn is ArrowBatchCollateFn
)
# UDF receives Dict[str, torch.Tensor] in pinned CPU memory
# UDF does .to("cuda", non_blocking=True) → fast async DMA, no pageable staging copy
```

After (GPU tensors — custom collate handles everything):
```python
def gpu_collate(batch: Dict[str, np.ndarray]):
    import torch
    return {k: torch.as_tensor(v).pin_memory().to("cuda", non_blocking=True)
            for k, v in batch.items()}

ds.map_batches(InferModel, num_gpus=1, collate_fn=gpu_collate)
# UDF receives Dict[str, torch.Tensor] already on GPU
```

**Why this is good UX:**
- `collate_fn` is the same concept users know from `iter_torch_batches` and PyTorch DataLoader
- The user explicitly opts in — no hidden type changes
- The collate function's return type IS the UDF's input type — clear contract
- `DefaultCollateFn(pin_memory=True)` reuses proven, tested code
- Users who don't pass `collate_fn` get identical behavior to today

## Verification

1. **Unit test**: Create a small dataset, `map_batches` with
   `collate_fn=DefaultCollateFn(pin_memory=True)`. Verify UDF receives
   `Dict[str, torch.Tensor]` with `tensor.is_pinned() == True`.
2. **batch_format auto-detection test**: Verify that passing `DefaultCollateFn` (an
   `ArrowBatchCollateFn`) auto-sets `batch_format="pyarrow"`, and that explicit
   `batch_format` conflicting with collate_fn type raises a clear error.
3. **nsys validation**: Run same inference workload with/without `collate_fn`. Compare:
   - "Memcpy HtoD (Pageable)" should disappear, replaced by "Memcpy HtoD (Pinned)"
   - GPU idle gaps between kernel bursts should shrink
4. **Memory test**: Verify pinned memory is released after each batch (not leaked).
5. **Existing tests**: `python/ray/data/tests/test_map.py` should pass unchanged (no
   collate_fn = same behavior).

## Files Summary

| File | Change |
|------|--------|
| `python/ray/data/dataset.py` | Add `collate_fn` param to `map_batches`, auto-detect `batch_format` |
| `python/ray/data/_internal/logical/operators/map_operator.py` | Add `collate_fn` field to `MapBatches` |
| `python/ray/data/_internal/planner/plan_udf_map_op.py` | Pass `collate_fn` to `BatchMapTransformFn` |
| `python/ray/data/_internal/execution/operators/map_transformer.py` | Accept `collate_fn`, pass to `batch_blocks()` in `_pre_process` |
| `python/ray/data/util/torch_utils.py` | No changes — reuse existing functions |
| `python/ray/data/_internal/block_batching/block_batching.py` | No changes — `batch_blocks` already supports `collate_fn` |

---

## Slack Message (for team discussion)

> hmm.. the screenshot clearly shows that GPU is sitting completely idle during host-to-device transfer. Memcpy HtoD (Pageable) section is basically telling us all data flows through pageable memory (plasma is mmap'd which holds Arrow blocks --> numpy --> torch tensor, all pageable). This forces two-phase serial transfer: CPU memcpy to internal pinned staging buffer, then DMA to device. And all this while GPU is blocked and we can't pipeline. And, this is happening for each inference batch, so cost can compound quickly.
>
> I was digging into the code and found that Ray Data already has DefaultCollateFn(pin_memory=True) which creates pinned tensors. But, it is only wired up for the iter_torch_batches consumer-side path, not for map_batches actor execution. Any particular reason @jhsu @Balaji (Ray Team)?
>
> I think if we just expose collate_fn on map_batches (same API pattern as iter_torch_batches), thread it to batch_blocks, it would enable non-blocking transfer and unlock pipelining. User code/udf can call tensor.to(device, non_blocking=True). True pipelining will require some more work but maybe worthwhile to think about why it was not exposed and any concerns with the api surface.
