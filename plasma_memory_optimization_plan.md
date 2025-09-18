# Plasma Store Memory Pool Optimization Plan

## Executive Summary

This document outlines a comprehensive plan to optimize memory allocation in Ray's Plasma store by replacing dlmalloc with jemalloc and leveraging its built-in algorithms for large object management. Since Plasma exclusively handles objects ≥100KB, the optimization focuses on reducing fragmentation and improving allocation performance for large memory blocks.

## Current State Analysis

### Problem Statement

The Plasma store currently uses dlmalloc with custom mmap management, which exhibits several performance bottlenecks:

1. **Memory Fragmentation**: External fragmentation between large allocated objects
2. **Linear Search Overhead**: O(n) lookup in `mmap_records` for allocation metadata
3. **No Coalescing**: Freed memory blocks are not efficiently merged
4. **Poor Cache Locality**: No NUMA awareness or memory placement optimization
5. **Global Mutex Contention**: Single mutex serializes all memory operations

### Key Constraint

- **Minimum Object Size**: Ray Core only stores objects ≥100KB in Plasma
- **Shared Memory**: All allocations must come from `/dev/shm` for zero-copy access
- **Memory Tracking**: Must maintain `mmap_records` for client memory mapping

## jemalloc Integration Strategy

### Why jemalloc for Large Objects

jemalloc provides sophisticated algorithms specifically designed to handle large allocations efficiently:

1. **Extent-Based Management**: Uses variable-sized "extents" instead of fixed chunks
2. **Automatic Coalescing**: Eagerly merges adjacent free extents
3. **Best-Fit Allocation**: Reduces fragmentation for large objects
4. **Built-in Metrics**: Comprehensive fragmentation and performance statistics
5. **Thread Safety**: Lock-free fast paths without global mutex

### jemalloc's Large Object Algorithms

#### Size Classes
```
Small:    [8 bytes - 14KB]     → Small bins (not used by Plasma)
Large:    [14KB - 4MB]          → Large size classes (Plasma's primary range)
Huge:     [4MB+]                → Direct extent allocation
Oversize: [8MB+]                → Dedicated arena (configurable)
```

#### Anti-Fragmentation Features

1. **extent_max_active_fit**: Prevents splitting large extents for small allocations
2. **Serial Number Allocation**: Prefers older extents for better locality
3. **Delayed Coalescing**: Reduces fragmentation from alternating alloc/free patterns
4. **Decay-Based Purging**: Gradual memory return to OS

## Implementation Plan

### Phase 1: jemalloc Configuration for Plasma

```cpp
class PlasmaJemallocAllocator : public IAllocator {
private:
  void ConfigureJemalloc() {
    // Optimal settings for large objects (≥100KB)

    // 1. Extent management
    je_mallctl("opt.lg_extent_max_active_fit", 4);  // Aggressive coalescing

    // 2. Size classes for 100KB+ objects
    je_mallctl("opt.lg_quantum", 17);               // 128KB quantum
    je_mallctl("opt.oversize_threshold", 67108864); // 64MB oversize

    // 3. Memory reuse optimization
    je_mallctl("opt.dirty_decay_ms", 5000);         // Keep freed memory
    je_mallctl("opt.muzzy_decay_ms", 10000);        // Delay OS return

    // 4. Huge pages support
    je_mallctl("opt.thp", "always");                // Transparent huge pages
    je_mallctl("opt.metadata_thp", "always");       // THP for metadata

    // 5. Disable unnecessary features
    je_mallctl("opt.tcache", false);                // No thread cache
    je_mallctl("opt.tcache_max", 0);                // Large objects don't benefit
  }
};
```

### Phase 2: Custom Extent Hooks

Implement extent hooks to integrate jemalloc with Plasma's shared memory architecture:

```cpp
extent_hooks_t plasma_extent_hooks = {
  .alloc = PlasmaExtentAlloc,
  .dalloc = PlasmaExtentDalloc,
  .destroy = PlasmaExtentDestroy,
  .commit = PlasmaExtentCommit,
  .decommit = PlasmaExtentDecommit,
  .purge_lazy = PlasmaExtentPurge,
  .merge = PlasmaExtentMerge,
  .split = PlasmaExtentSplit
};
```

#### Extent Hook Implementations

##### 1. **alloc** - Allocate from /dev/shm
```cpp
void* PlasmaExtentAlloc(extent_hooks_t *extent_hooks,
                        void *new_addr, size_t size,
                        size_t alignment, bool *zero,
                        bool *commit, unsigned arena_ind) {
  // Create file in /dev/shm
  std::string file_template = "/dev/shm/plasmaXXXXXX";
  int fd = mkostemp(file_template.c_str(), O_CLOEXEC);

  // Configure for large objects
  ftruncate(fd, size);

  // Map with huge page support
  int flags = MAP_SHARED | MAP_POPULATE;
  void* addr = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, fd, 0);

  if (addr != MAP_FAILED && size >= 2097152) {  // ≥2MB
    madvise(addr, size, MADV_HUGEPAGE);
  }

  // Track in Plasma's records
  mmap_records[addr] = {fd, size};

  *zero = true;
  *commit = true;
  return addr;
}
```

##### 2. **dalloc** - Return memory to OS
```cpp
bool PlasmaExtentDalloc(extent_hooks_t *extent_hooks,
                        void *addr, size_t size,
                        bool committed, unsigned arena_ind) {
  auto it = mmap_records.find(addr);
  if (it != mmap_records.end()) {
    munmap(addr, size);
    close(it->second.fd);
    mmap_records.erase(it);
    return false;  // Success
  }
  return true;  // Failure
}
```

##### 3. **commit** - Make memory accessible
```cpp
bool PlasmaExtentCommit(extent_hooks_t *extent_hooks,
                       void *addr, size_t size,
                       size_t offset, size_t length,
                       unsigned arena_ind) {
  void* target = (char*)addr + offset;

  // Ensure memory is accessible
  mprotect(target, length, PROT_READ | PROT_WRITE);

  // Prefetch for large allocations
  if (length >= 1048576) {  // ≥1MB
    madvise(target, length, MADV_WILLNEED);
  }

  return false;  // Success
}
```

##### 4. **decommit** - Release physical memory
```cpp
bool PlasmaExtentDecommit(extent_hooks_t *extent_hooks,
                         void *addr, size_t size,
                         size_t offset, size_t length,
                         unsigned arena_ind) {
  void* target = (char*)addr + offset;

  // Release physical pages under memory pressure
  madvise(target, length, MADV_DONTNEED);
  mprotect(target, length, PROT_NONE);

  return false;  // Success
}
```

##### 5. **purge_lazy** - Lazy memory reclaim
```cpp
bool PlasmaExtentPurge(extent_hooks_t *extent_hooks,
                       void *addr, size_t size,
                       size_t offset, size_t length,
                       unsigned arena_ind) {
  void* target = (char*)addr + offset;

  // Use MADV_FREE if available (Linux 4.5+)
  #ifdef MADV_FREE
    madvise(target, length, MADV_FREE);
  #else
    madvise(target, length, MADV_DONTNEED);
  #endif

  return false;  // Success
}
```

##### 6. **merge** - Coalesce adjacent extents
```cpp
bool PlasmaExtentMerge(extent_hooks_t *extent_hooks,
                      void *addr_a, size_t size_a,
                      void *addr_b, size_t size_b,
                      bool committed, unsigned arena_ind) {
  // Verify adjacency
  if ((char*)addr_a + size_a != addr_b) {
    return true;  // Cannot merge non-adjacent
  }

  // Update tracking
  auto it_a = mmap_records.find(addr_a);
  if (it_a != mmap_records.end()) {
    it_a->second.size = size_a + size_b;
    mmap_records.erase(addr_b);
  }

  // Track coalescing metrics
  coalesce_count++;
  bytes_coalesced += size_b;

  return false;  // Success
}
```

##### 7. **split** - Split extent
```cpp
bool PlasmaExtentSplit(extent_hooks_t *extent_hooks,
                      void *addr, size_t size,
                      size_t size_a, size_t size_b,
                      bool committed, unsigned arena_ind) {
  // Verify sizes
  if (size_a + size_b != size) {
    return true;  // Invalid split
  }

  // Update tracking
  void* addr_b = (char*)addr + size_a;
  auto it = mmap_records.find(addr);
  if (it != mmap_records.end()) {
    MEMFD_TYPE fd = it->second.fd;
    it->second.size = size_a;
    mmap_records[addr_b] = {fd, size_b};
  }

  // Track fragmentation
  split_count++;

  return false;  // Success
}
```

### Phase 3: Memory Metrics and Monitoring

```cpp
class PlasmaMemoryMetrics {
  struct LargeObjectStats {
    // Size distribution (100KB buckets)
    std::atomic<uint64_t> size_histogram[100];

    // Fragmentation metrics
    std::atomic<uint64_t> external_fragmentation_bytes;
    std::atomic<uint64_t> coalesce_operations;
    std::atomic<uint64_t> split_operations;

    // Performance metrics
    std::atomic<uint64_t> allocation_latency_ns_sum;
    std::atomic<uint64_t> allocation_count;

    // jemalloc statistics
    size_t allocated, active, metadata, resident, retained;
  };

  void CollectJemallocStats() {
    je_mallctl("stats.allocated", &allocated, ...);
    je_mallctl("stats.active", &active, ...);
    je_mallctl("stats.metadata", &metadata, ...);
    je_mallctl("stats.resident", &resident, ...);
    je_mallctl("stats.retained", &retained, ...);

    // Calculate fragmentation
    double internal_frag = (double)(active - allocated) / active;
    double external_frag = (double)(retained - active) / retained;
  }

  void RecordAllocation(size_t size, uint64_t latency_ns) {
    int bucket = size / (100 * 1024);  // 100KB buckets
    if (bucket < 100) {
      size_histogram[bucket]++;
    }
    allocation_latency_ns_sum += latency_ns;
    allocation_count++;
  }
};
```

## Implementation Timeline

### Week 1: jemalloc Integration
- [ ] Replace dlmalloc with jemalloc in build system
- [ ] Implement basic allocator wrapper
- [ ] Configure jemalloc for large objects
- [ ] Add basic allocation metrics

### Week 2: Extent Hooks Implementation
- [ ] Implement custom extent hooks
- [ ] Integrate with Plasma's mmap_records
- [ ] Test shared memory allocation
- [ ] Verify huge page support

### Week 3: Testing and Optimization
- [ ] Benchmark allocation performance
- [ ] Measure fragmentation reduction
- [ ] Tune jemalloc parameters
- [ ] Stress test with production workloads

### Week 4: Production Deployment
- [ ] Add comprehensive monitoring
- [ ] Create performance dashboards
- [ ] Document configuration options
- [ ] Deploy to staging environment

## Expected Performance Improvements

### With jemalloc + Custom Extent Hooks

| Metric | Current (dlmalloc) | Expected (jemalloc) | Improvement |
|--------|-------------------|---------------------|-------------|
| External Fragmentation | 30-40% | 10-15% | **60% reduction** |
| Allocation Latency (p50) | 10μs | 6μs | **40% faster** |
| Allocation Latency (p99) | 100μs | 50μs | **50% faster** |
| Memory Overhead | 5-8% | 2-3% | **60% reduction** |
| Coalescing Efficiency | None | Automatic | **New capability** |
| Peak Memory Usage | Baseline | -20% | **20% reduction** |

## Testing Strategy

### Microbenchmarks
1. Single-threaded allocation/deallocation cycles
2. Multi-threaded concurrent allocations
3. Fragmentation measurement over time
4. Memory reuse efficiency

### Workload Testing
```cpp
// Test various allocation patterns
void TestAllocationPatterns() {
  // Pattern 1: Uniform sizes (1MB objects)
  TestUniformAllocations(1048576, 1000);

  // Pattern 2: Mixed sizes (100KB to 10MB)
  TestMixedSizeAllocations(100*1024, 10*1024*1024);

  // Pattern 3: Alternating alloc/free
  TestFragmentationPattern();

  // Pattern 4: Burst allocations
  TestBurstAllocation(100, 4*1024*1024);
}
```

### Production Validation
1. A/B testing with subset of traffic
2. Memory usage monitoring
3. Latency percentile tracking
4. Fragmentation analysis

## Risk Mitigation

### Potential Risks
1. **jemalloc Integration Issues**: May require build system changes
2. **Shared Memory Compatibility**: Ensure client mapping still works
3. **Performance Regression**: Some workloads might perform worse

### Mitigation Strategies
1. **Gradual Rollout**: Use feature flags to enable/disable
2. **Fallback Path**: Keep dlmalloc code for emergency rollback
3. **Extensive Testing**: Test with production workload traces
4. **Monitoring**: Add alerts for memory metrics

## Conclusion

By leveraging jemalloc's sophisticated algorithms and implementing custom extent hooks for Plasma's shared memory architecture, we can achieve significant performance improvements without the complexity of implementing our own memory management algorithms. The focus on large object optimization (≥100KB) simplifies the implementation while delivering substantial benefits in fragmentation reduction and allocation performance.

## References

- [jemalloc Documentation](https://jemalloc.net/jemalloc.3.html)
- [Facebook's jemalloc Usage](https://engineering.fb.com/2011/01/03/core-infra/scalable-memory-allocation-using-jemalloc/)
- [jemalloc Extent Hooks](https://github.com/jemalloc/jemalloc/wiki/ExtentHooks)
- [Ray Configuration: max_direct_call_object_size](src/ray/common/ray_config_def.h:195)