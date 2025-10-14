// Copyright 2025 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/object_manager/plasma/jemalloc_allocator.h"

#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <string>
#include <vector>

#include "ray/common/ray_config.h"
#include "ray/object_manager/plasma/common.h"
#include "ray/object_manager/plasma/malloc.h"
#include "ray/stats/metric_defs.h"
#include "ray/stats/tag_defs.h"
#include "ray/util/compat.h"
#include "ray/util/logging.h"

namespace plasma {

namespace {

// Alignment for large objects to utilize huge pages effectively
constexpr size_t kLargeObjectAlignment = 2 * 1024 * 1024;  // 2MB for huge pages

// Minimum object size for Plasma (100KB)
constexpr size_t kMinPlasmaObjectSize = 100 * 1024;

// Give each mmap record a unique id to disambiguate fd reuse - process-wide
static int64_t next_mmap_unique_id = INVALID_UNIQUE_FD_ID + 1;

// Process-wide pointer to the allocator instance for extent hooks
static JemallocAllocator *g_allocator_instance = nullptr;

// Process-wide flag to indicate we're in fallback allocation mode
static std::atomic<bool> in_fallback_mode{false};

// Track the single pre-allocated pool for extent hooks
static bool pool_allocated = false;
static void *pool_base = nullptr;
static size_t pool_size = 0;
static int pool_fd = -1;

// Helper functions
void *pointer_advance(void *p, ptrdiff_t n) {
  return static_cast<unsigned char *>(p) + n;
}

void *pointer_retreat(void *p, ptrdiff_t n) {
  return static_cast<unsigned char *>(p) - n;
}

// Create and mmap a buffer, handling both primary and fallback allocation
void create_and_mmap_buffer(int64_t size,
                            void **pointer,
                            int *fd,
                            const std::string &plasma_directory,
                            const std::string &fallback_directory,
                            bool hugepages_enabled,
                            bool fallback_enabled) {
  // Choose directory based on allocation mode
  std::string file_template = plasma_directory;

  // If fallback is enabled and we're in fallback mode, use fallback directory
  if (fallback_enabled && in_fallback_mode) {
    file_template = fallback_directory;
    RAY_LOG(INFO) << "Using fallback directory for allocation";
  }

  file_template += "/plasmaXXXXXX";
  RAY_LOG(INFO) << "create_and_mmap_buffer(" << size << ", " << file_template << ")";

  std::vector<char> file_name(file_template.begin(), file_template.end());
  file_name.push_back('\0');

  // O_CLOEXEC ensures fd is closed when process is forked
  *fd = mkostemp(&file_name[0], O_CLOEXEC);
  if (*fd < 0) {
    RAY_LOG(ERROR) << "Failed to create file " << &file_name[0] << ": "
                   << std::strerror(errno);
    *pointer = MAP_FAILED;
    return;
  }

  // Immediately unlink the file so we don't leave traces
  if (unlink(&file_name[0]) != 0) {
    RAY_LOG(WARNING) << "Failed to unlink file " << &file_name[0] << ": "
                     << std::strerror(errno);
  }

  if (!hugepages_enabled) {
    // Extend file to desired size (not needed for hugepage fs)
    if (ftruncate(*fd, size) != 0) {
      RAY_LOG(ERROR) << "Failed to ftruncate file: " << std::strerror(errno);
      close(*fd);
      *pointer = MAP_FAILED;
      return;
    }
  }

  // Prepare mmap flags
  int flags = MAP_SHARED;
  if (RayConfig::instance().preallocate_plasma_memory()) {
#ifdef MAP_POPULATE
    RAY_LOG(INFO) << "Preallocating all plasma memory using MAP_POPULATE";
    flags |= MAP_POPULATE;
#else
    RAY_LOG(WARNING) << "MAP_POPULATE not supported on this platform";
#endif
  }

#ifdef __linux__
  // For fallback allocation, use fallocate to ensure no SIGBUS
  if (fallback_enabled && in_fallback_mode) {
    RAY_LOG(DEBUG) << "Preallocating fallback allocation using fallocate";
    int ret = fallocate(*fd, 0, 0, size);
    if (ret != 0) {
      if (errno == EOPNOTSUPP || errno == ENOSYS) {
        RAY_LOG(DEBUG) << "fallocate not supported: " << std::strerror(errno);
      } else {
        RAY_LOG(ERROR) << "Out of disk space with fallocate error: "
                       << std::strerror(errno);
        close(*fd);
        *pointer = MAP_FAILED;
        return;
      }
    }
  }
#endif

  *pointer = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, *fd, 0);
  if (*pointer == MAP_FAILED) {
    RAY_LOG(ERROR) << "mmap failed: " << std::strerror(errno);
    if (errno == ENOMEM && hugepages_enabled) {
      RAY_LOG(ERROR) << "  (may need to increase /proc/sys/vm/nr_hugepages)";
    }
    close(*fd);
  }

#ifdef __linux__
  // Optionally exclude from core dumps
  if (*pointer != MAP_FAILED &&
      RayConfig::instance().raylet_core_dump_exclude_plasma_store()) {
    int rval = madvise(*pointer, size, MADV_DONTDUMP);
    if (rval) {
      RAY_LOG(WARNING) << "madvise(MADV_DONTDUMP) failed: " << strerror(errno);
    }
  }
#endif
}

// Extent hook functions that will be registered with jemalloc
void *PlasmaExtentAlloc(extent_hooks_t *extent_hooks,
                        void *new_addr,
                        size_t size,
                        size_t alignment,
                        bool *zero,
                        bool *commit,
                        unsigned arena_ind) {
  if (!g_allocator_instance) {
    return nullptr;
  }

  // Single-pool strategy: allocate entire pool on first call, return nullptr after
  if (!pool_allocated) {
    // First allocation: create the entire memory pool
    size_t total_size =
        g_allocator_instance->GetFootprintLimit() + plasma::kMmapRegionsGap;

    void *pointer;
    int fd;
    create_and_mmap_buffer(total_size,
                           &pointer,
                           &fd,
                           g_allocator_instance->GetPlasmaDirectory(),
                           g_allocator_instance->GetFallbackDirectory(),
                           g_allocator_instance->IsHugepageEnabled(),
                           false);  // Not fallback for main pool

    if (pointer == MAP_FAILED) {
      RAY_LOG(ERROR) << "Failed to allocate main memory pool of " << total_size
                     << " bytes";
      return nullptr;
    }

    // Store pool information
    pool_allocated = true;
    pool_base = pointer;
    pool_size = total_size;
    pool_fd = fd;

    // Update metrics
    g_allocator_instance->IncrementExtentAllocCount();

    // Track the mapping for compatibility
    {
      absl::MutexLock lock(&g_allocator_instance->GetMutex());
      JemallocMmapRecord &record = g_allocator_instance->GetMmapRecords()[pointer];
      record.fd = MEMFD_TYPE(fd, next_mmap_unique_id++);
      record.size = total_size;
    }

    // Advance pointer by gap amount (lie to jemalloc about actual location)
    void *adjusted_pointer = pointer_advance(pointer, plasma::kMmapRegionsGap);

    RAY_LOG(INFO) << "PlasmaExtentAlloc: allocated entire pool of " << total_size
                  << " bytes at " << adjusted_pointer << " (actual: " << pointer << ")";

    *zero = true;
    *commit = true;
    return adjusted_pointer;
  } else {
    // Subsequent allocations: return nullptr to force jemalloc to subdivide existing pool
    RAY_LOG(DEBUG) << "PlasmaExtentAlloc: returning nullptr for size " << size
                   << " (pool already allocated, forcing subdivision)";
    return nullptr;
  }
}

bool PlasmaExtentDalloc(extent_hooks_t *extent_hooks,
                        void *addr,
                        size_t size,
                        bool committed,
                        unsigned arena_ind) {
  if (!g_allocator_instance) {
    return true;
  }

  // Retreat pointer to actual location
  void *real_addr = pointer_retreat(addr, plasma::kMmapRegionsGap);
  size_t real_size = size + plasma::kMmapRegionsGap;

  // Check if this is the entire pool being deallocated
  if (pool_allocated && real_addr == pool_base && real_size == pool_size) {
    RAY_LOG(INFO) << "PlasmaExtentDalloc: deallocating entire pool of " << real_size
                  << " bytes at " << real_addr;

    // Update metrics
    g_allocator_instance->IncrementExtentDallocCount();

    // Actually free the pool
    int r = munmap(real_addr, real_size);
    if (r == 0) {
      close(pool_fd);
      pool_allocated = false;
      pool_base = nullptr;
      pool_size = 0;
      pool_fd = -1;

      // Remove from mmap_records
      absl::MutexLock lock(&g_allocator_instance->GetMutex());
      g_allocator_instance->GetMmapRecords().erase(real_addr);
    } else {
      RAY_LOG(ERROR) << "munmap failed: " << strerror(errno);
    }

    return r != 0;
  } else {
    // This is a subdivision of the pool - don't actually deallocate
    // Return true to indicate we're refusing to deallocate
    RAY_LOG(DEBUG) << "PlasmaExtentDalloc: refusing to deallocate subdivision at "
                   << real_addr << ", size " << real_size
                   << " (keeping memory in jemalloc's control)";
    return true;
  }
}

void PlasmaExtentDestroy(extent_hooks_t *extent_hooks,
                         void *addr,
                         size_t size,
                         bool committed,
                         unsigned arena_ind) {
  // Retreat pointer to actual location
  addr = pointer_retreat(addr, plasma::kMmapRegionsGap);
  size += plasma::kMmapRegionsGap;

  // For uncommitted extents, just unmap
  munmap(addr, size);

  if (g_allocator_instance) {
    absl::MutexLock lock(&g_allocator_instance->GetMutex());
    g_allocator_instance->GetMmapRecords().erase(addr);
  }
}

bool PlasmaExtentCommit(extent_hooks_t *extent_hooks,
                        void *addr,
                        size_t size,
                        size_t offset,
                        size_t length,
                        unsigned arena_ind) {
  // No need to adjust pointer for commit operations
  void *target = static_cast<char *>(addr) + offset;

  // Ensure memory is accessible
  if (mprotect(target, length, PROT_READ | PROT_WRITE) != 0) {
    RAY_LOG(WARNING) << "Failed to commit memory: " << strerror(errno);
    return true;
  }

  // Prefetch for large allocations
  if (length >= 1024 * 1024) {  // >= 1MB
    madvise(target, length, MADV_WILLNEED);
  }

  // Enable huge pages if configured
  if (g_allocator_instance && g_allocator_instance->IsHugepageEnabled() &&
      length >= kLargeObjectAlignment) {
    madvise(target, length, MADV_HUGEPAGE);
  }

  return false;
}

bool PlasmaExtentDecommit(extent_hooks_t *extent_hooks,
                          void *addr,
                          size_t size,
                          size_t offset,
                          size_t length,
                          unsigned arena_ind) {
  void *target = static_cast<char *>(addr) + offset;

  // Release physical pages
  if (madvise(target, length, MADV_DONTNEED) != 0) {
    RAY_LOG(WARNING) << "Failed to decommit memory: " << strerror(errno);
  }

  // Make memory inaccessible
  if (mprotect(target, length, PROT_NONE) != 0) {
    RAY_LOG(WARNING) << "Failed to protect memory: " << strerror(errno);
  }

  return false;
}

static bool PlasmaExtentPurgeImpl(void *addr, size_t size, size_t offset, size_t length) {
  void *target = static_cast<char *>(addr) + offset;

#ifdef MADV_FREE
  // Prefer MADV_FREE on Linux 4.5+ for lazy reclaim
  if (madvise(target, length, MADV_FREE) != 0) {
    // Fallback to MADV_DONTNEED
    madvise(target, length, MADV_DONTNEED);
  }
#else
  madvise(target, length, MADV_DONTNEED);
#endif

  return false;
}

bool PlasmaExtentPurgeLazy(extent_hooks_t *extent_hooks,
                           void *addr,
                           size_t size,
                           size_t offset,
                           size_t length,
                           unsigned arena_ind) {
  return PlasmaExtentPurgeImpl(addr, size, offset, length);
}

bool PlasmaExtentPurgeForced(extent_hooks_t *extent_hooks,
                             void *addr,
                             size_t size,
                             size_t offset,
                             size_t length,
                             unsigned arena_ind) {
  return PlasmaExtentPurgeImpl(addr, size, offset, length);
}

bool PlasmaExtentMerge(extent_hooks_t *extent_hooks,
                       void *addr_a,
                       size_t size_a,
                       void *addr_b,
                       size_t size_b,
                       bool committed,
                       unsigned arena_ind) {
  if (!g_allocator_instance) {
    return true;
  }

  // Verify adjacency (with gap adjustment)
  if (static_cast<char *>(addr_a) + size_a != addr_b) {
    return true;  // Cannot merge non-adjacent extents
  }

  // Retreat pointers to actual locations for tracking
  void *real_addr_a = pointer_retreat(addr_a, plasma::kMmapRegionsGap);
  void *real_addr_b = pointer_retreat(addr_b, plasma::kMmapRegionsGap);
  size_t real_size_a = size_a + plasma::kMmapRegionsGap;
  size_t real_size_b = size_b + plasma::kMmapRegionsGap;

  // Update tracking
  {
    absl::MutexLock lock(&g_allocator_instance->GetMutex());
    auto it_a = g_allocator_instance->GetMmapRecords().find(real_addr_a);
    auto it_b = g_allocator_instance->GetMmapRecords().find(real_addr_b);

    if (it_a != g_allocator_instance->GetMmapRecords().end() &&
        it_b != g_allocator_instance->GetMmapRecords().end()) {
      // Merge: extend size of first extent, remove second
      it_a->second.size = real_size_a + real_size_b - plasma::kMmapRegionsGap;
      g_allocator_instance->GetMmapRecords().erase(it_b);
    }
  }

  // Track coalescing metrics
  g_allocator_instance->IncrementCoalesceCount();

  RAY_LOG(DEBUG) << "Merged extents: " << size_a << " + " << size_b << " bytes";
  return false;
}

bool PlasmaExtentSplit(extent_hooks_t *extent_hooks,
                       void *addr,
                       size_t size,
                       size_t size_a,
                       size_t size_b,
                       bool committed,
                       unsigned arena_ind) {
  if (!g_allocator_instance) {
    return true;
  }

  // Verify sizes
  if (size_a + size_b != size) {
    return true;
  }

  // Calculate real addresses and sizes
  void *real_addr = pointer_retreat(addr, plasma::kMmapRegionsGap);
  void *real_addr_b = pointer_advance(real_addr, size_a + plasma::kMmapRegionsGap);

  // Update tracking
  {
    absl::MutexLock lock(&g_allocator_instance->GetMutex());
    auto it = g_allocator_instance->GetMmapRecords().find(real_addr);
    if (it != g_allocator_instance->GetMmapRecords().end()) {
      MEMFD_TYPE fd = it->second.fd;

      // Update first extent size
      it->second.size = size_a + plasma::kMmapRegionsGap;

      // Add second extent
      g_allocator_instance->GetMmapRecords()[real_addr_b] =
          JemallocMmapRecord{fd,
                             static_cast<size_t>(size_b + plasma::kMmapRegionsGap),
                             std::string(),
                             false};
    }
  }

  // Track split metrics
  g_allocator_instance->IncrementSplitCount();

  RAY_LOG(DEBUG) << "Split extent: " << size << " into " << size_a << " + " << size_b;
  return false;
}

// Global extent hooks structure
extent_hooks_t plasma_extent_hooks = {
    PlasmaExtentAlloc,        // alloc
    PlasmaExtentDalloc,       // dalloc
    PlasmaExtentDestroy,      // destroy
    PlasmaExtentCommit,       // commit
    PlasmaExtentDecommit,     // decommit
    PlasmaExtentPurgeLazy,    // purge_lazy
    PlasmaExtentPurgeForced,  // purge_forced
    PlasmaExtentSplit,        // split
    PlasmaExtentMerge         // merge
};

}  // namespace

JemallocAllocator::JemallocAllocator(const std::string &plasma_directory,
                                     const std::string &fallback_directory,
                                     bool hugepage_enabled,
                                     int64_t footprint_limit)
    : plasma_directory_(plasma_directory),
      fallback_directory_(fallback_directory),
      hugepage_enabled_(hugepage_enabled),
      footprint_limit_(footprint_limit) {
  RAY_CHECK(footprint_limit > 0) << "Footprint limit must be positive";

  // Set thread-local instance pointer
  g_allocator_instance = this;

  // Reset single-pool tracking
  pool_allocated = false;
  pool_base = nullptr;
  pool_size = 0;
  pool_fd = -1;

  InitializeJemalloc();

  RAY_LOG(INFO) << "JemallocAllocator initialized with " << footprint_limit
                << " bytes limit, plasma_dir=" << plasma_directory
                << ", fallback_dir=" << fallback_directory;
}

JemallocAllocator::~JemallocAllocator() {
  // Clean up any remaining allocations
  absl::MutexLock lock(&mutex_);

  // Clean up the global pool if it exists
  if (pool_allocated && pool_base && pool_size > 0) {
    munmap(pool_base, pool_size);
    close(pool_fd);
    pool_allocated = false;
    pool_base = nullptr;
    pool_size = 0;
    pool_fd = -1;
  }

  // Clean up any remaining mmap_records
  for (const auto &entry : mmap_records_) {
    void *addr = entry.first;
    const JemallocMmapRecord &record = entry.second;
    // Don't double-free the main pool
    if (addr != pool_base) {
      munmap(addr, record.size);
      close(record.fd.first);
    }
  }
  mmap_records_.clear();

  g_allocator_instance = nullptr;
}

void JemallocAllocator::InitializeJemalloc() {
  // Configure jemalloc for single pool behavior with no memory release

  // Create a dedicated arena for plasma allocations
  unsigned arena_index;
  size_t sz = sizeof(unsigned);
  if (mallctl("arenas.create", &arena_index, &sz, nullptr, 0) != 0) {
    RAY_LOG(FATAL) << "Failed to create jemalloc arena";
  }
  plasma_arena_index_ = arena_index;

  std::string arena_prefix = "arena." + std::to_string(arena_index) + ".";

  // Never release memory back to OS (set decay to -1 means never decay)
  ssize_t never_decay = -1;
  mallctl((arena_prefix + "dirty_decay_ms").c_str(),
          nullptr,
          nullptr,
          &never_decay,
          sizeof(ssize_t));
  mallctl((arena_prefix + "muzzy_decay_ms").c_str(),
          nullptr,
          nullptr,
          &never_decay,
          sizeof(ssize_t));

  // Set retain growth limit to our pool size
  // This prevents the arena from growing beyond our limit
  size_t retain_limit = footprint_limit_;
  mallctl((arena_prefix + "retain_grow_limit").c_str(),
          nullptr,
          nullptr,
          &retain_limit,
          sizeof(size_t));

  // Configure for large objects (plasma objects are typically > 100KB)

  // Set quantum for better alignment with large objects
  size_t lg_quantum = 17;  // 128KB quantum
  mallctl("opt.lg_quantum", nullptr, nullptr, &lg_quantum, sizeof(size_t));

  // Enable transparent huge pages if configured
  if (hugepage_enabled_) {
    bool thp_enabled = true;
    mallctl("opt.thp", nullptr, nullptr, &thp_enabled, sizeof(bool));
    mallctl("opt.metadata_thp", nullptr, nullptr, &thp_enabled, sizeof(bool));
  }

  // Keep thread cache enabled for better performance
  // (previous code disabled it, but tcache helps even with large objects)

  RAY_LOG(INFO) << "jemalloc arena " << arena_index << " configured with retain limit "
                << retain_limit << " and no memory release";

  // Install custom extent hooks for single-pool management
  InstallExtentHooks();
}

void JemallocAllocator::InstallExtentHooks() {
  // Install custom extent hooks for the already created arena
  std::string hooks_key =
      "arena." + std::to_string(plasma_arena_index_) + ".extent_hooks";
  extent_hooks_t *hooks_ptr = &plasma_extent_hooks;

  if (mallctl(
          hooks_key.c_str(), nullptr, nullptr, &hooks_ptr, sizeof(extent_hooks_t *)) !=
      0) {
    RAY_LOG(ERROR) << "Failed to install extent hooks";
  }

  RAY_LOG(INFO) << "Installed custom extent hooks for arena " << plasma_arena_index_;
}

std::optional<Allocation> JemallocAllocator::Allocate(size_t bytes) {
  if (bytes < kMinPlasmaObjectSize) {
    RAY_LOG(WARNING) << "Requested allocation " << bytes << " bytes is less than minimum "
                     << kMinPlasmaObjectSize;
  }

  // Round up to page boundary for better alignment
  size_t aligned_size = (bytes + 4095) & ~4095;

  // Make sure we're not in fallback mode for regular allocation
  in_fallback_mode = false;

  // Use custom arena with extent hooks
  int flags = MALLOCX_ARENA(plasma_arena_index_) | MALLOCX_ZERO;

  // Align large allocations to huge page boundaries
  if (aligned_size >= kLargeObjectAlignment) {
    flags |= MALLOCX_ALIGN(kLargeObjectAlignment);
  }

  void *ptr = mallocx(aligned_size, flags);
  if (!ptr) {
    RAY_LOG(DEBUG) << "Failed to allocate " << aligned_size << " bytes";
    return std::nullopt;
  }

  allocated_ += aligned_size;

  RAY_LOG(DEBUG) << "Allocated " << aligned_size << " bytes at " << ptr;

  return BuildAllocation(ptr, aligned_size, false);
}

std::optional<Allocation> JemallocAllocator::FallbackAllocate(size_t bytes) {
  if (fallback_directory_.empty()) {
    RAY_LOG(ERROR)
        << "Fallback allocation requested but no fallback directory configured";
    return std::nullopt;
  }

  // Round up to page boundary
  size_t aligned_size = (bytes + 4095) & ~4095;

  // Set fallback mode flag
  in_fallback_mode = true;

  // Use custom arena with extent hooks
  int flags = MALLOCX_ARENA(plasma_arena_index_) | MALLOCX_ZERO;

  void *ptr = mallocx(aligned_size, flags);

  // Reset fallback mode flag
  in_fallback_mode = false;

  if (!ptr) {
    RAY_LOG(DEBUG) << "Failed to fallback allocate " << aligned_size << " bytes";
    return std::nullopt;
  }

  // With single-pool extent hooks, fallback allocations are still from the same pool
  // Just check if we're near the pool limit
  bool is_outside = false;  // For single-pool, everything is in the main pool

  allocated_ += aligned_size;
  if (is_outside) {
    fallback_allocated_ += aligned_size;
  }

  RAY_LOG(DEBUG) << "Fallback allocated " << aligned_size << " bytes at " << ptr
                 << " (outside initial: " << is_outside << ")";

  return BuildAllocation(ptr, aligned_size, is_outside);
}

void JemallocAllocator::Free(Allocation allocation) {
  RAY_CHECK(allocation.address_ != nullptr) << "Cannot free nullptr";

  size_t size = allocation.size_;

  // Use sized deallocation for efficiency
  sdallocx(allocation.address_, size, 0);

  allocated_ -= size;
  if (allocation.fallback_allocated_) {
    fallback_allocated_ -= size;
  }

  RAY_LOG(DEBUG) << "Freed " << size << " bytes at " << allocation.address_;
}

int64_t JemallocAllocator::Allocated() const {
  // Query jemalloc for accurate stats
  size_t allocated, sz = sizeof(size_t);
  mallctl("stats.allocated", &allocated, &sz, nullptr, 0);
  return allocated;
}

JemallocAllocator::MemoryStats JemallocAllocator::GetStats() const {
  MemoryStats stats;
  size_t sz = sizeof(size_t);

  mallctl("stats.allocated", &stats.allocated_bytes, &sz, nullptr, 0);
  mallctl("stats.active", &stats.active_bytes, &sz, nullptr, 0);
  mallctl("stats.metadata", &stats.metadata_bytes, &sz, nullptr, 0);
  mallctl("stats.resident", &stats.resident_bytes, &sz, nullptr, 0);
  mallctl("stats.retained", &stats.retained_bytes, &sz, nullptr, 0);

  // Calculate fragmentation ratio
  if (stats.active_bytes > 0) {
    stats.fragmentation_ratio = 1.0 - static_cast<double>(stats.allocated_bytes) /
                                          static_cast<double>(stats.active_bytes);
  } else {
    stats.fragmentation_ratio = 0.0;
  }

  stats.coalesce_count = coalesce_count_;
  stats.split_count = split_count_;

  return stats;
}

void JemallocAllocator::RecordMetrics() const {
  auto stats = GetStats();

  // Record basic allocator statistics
  ray::stats::STATS_object_store_memory.Record(
      stats.allocated_bytes, {{ray::stats::LocationKey, "jemalloc_allocated"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.active_bytes, {{ray::stats::LocationKey, "jemalloc_active"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.resident_bytes, {{ray::stats::LocationKey, "jemalloc_resident"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.metadata_bytes, {{ray::stats::LocationKey, "jemalloc_metadata"}});

  // Record fragmentation as a percentage (0-100)
  ray::stats::STATS_object_store_memory.Record(
      static_cast<int64_t>(stats.fragmentation_ratio * 100),
      {{ray::stats::LocationKey, "jemalloc_fragmentation_pct"}});

  // Record fallback allocation stats
  ray::stats::STATS_object_store_memory.Record(
      fallback_allocated_.load(),
      {{ray::stats::LocationKey, "jemalloc_fallback_allocated"}});

  // Record extent operation counts
  ray::stats::STATS_object_store_memory.Record(
      extent_alloc_count_.load(), {{ray::stats::LocationKey, "jemalloc_extent_allocs"}});

  ray::stats::STATS_object_store_memory.Record(
      extent_dalloc_count_.load(),
      {{ray::stats::LocationKey, "jemalloc_extent_deallocs"}});

  ray::stats::STATS_object_store_memory.Record(
      coalesce_count_.load(), {{ray::stats::LocationKey, "jemalloc_coalesce_ops"}});

  ray::stats::STATS_object_store_memory.Record(
      split_count_.load(), {{ray::stats::LocationKey, "jemalloc_split_ops"}});

  ray::stats::STATS_object_store_memory.Record(
      fallback_alloc_count_.load(),
      {{ray::stats::LocationKey, "jemalloc_fallback_allocs"}});
}

void JemallocAllocator::GetDebugDump(std::stringstream &buffer) const {
  auto stats = GetStats();

  buffer << "\n=== JemallocAllocator Debug Info ===\n";
  buffer << "- Memory limit: " << footprint_limit_ << " bytes\n";
  buffer << "- Plasma directory: " << plasma_directory_ << "\n";
  buffer << "- Fallback directory: " << fallback_directory_ << "\n";
  buffer << "- Hugepages enabled: " << (hugepage_enabled_ ? "yes" : "no") << "\n";
  buffer << "\n";

  buffer << "=== Memory Statistics ===\n";
  buffer << "- Allocated: " << stats.allocated_bytes << " bytes\n";
  buffer << "- Active: " << stats.active_bytes << " bytes\n";
  buffer << "- Resident: " << stats.resident_bytes << " bytes\n";
  buffer << "- Metadata: " << stats.metadata_bytes << " bytes\n";
  buffer << "- Retained: " << stats.retained_bytes << " bytes\n";
  buffer << "- Fragmentation: " << (stats.fragmentation_ratio * 100.0) << "%\n";
  buffer << "- Fallback allocated: " << fallback_allocated_.load() << " bytes\n";
  buffer << "\n";

  buffer << "=== Operation Counts ===\n";
  buffer << "- Extent allocations: " << extent_alloc_count_.load() << "\n";
  buffer << "- Extent deallocations: " << extent_dalloc_count_.load() << "\n";
  buffer << "- Coalesce operations: " << coalesce_count_.load() << "\n";
  buffer << "- Split operations: " << split_count_.load() << "\n";
  buffer << "- Fallback allocations: " << fallback_alloc_count_.load() << "\n";

  buffer << "\n=== Memory Mappings ===\n";
  absl::MutexLock lock(&mutex_);
  buffer << "- Active mappings: " << mmap_records_.size() << "\n";

  int64_t total_mapped = 0;
  for (const auto &entry : mmap_records_) {
    const JemallocMmapRecord &record = entry.second;
    total_mapped += record.size;
  }
  buffer << "- Total mapped: " << total_mapped << " bytes\n";
}

std::optional<Allocation> JemallocAllocator::BuildAllocation(void *addr,
                                                             size_t size,
                                                             bool is_fallback_allocated) {
  if (!addr) {
    return std::nullopt;
  }

  // With single-pool extent hooks, all allocations come from the one pre-allocated pool
  // The address from jemalloc is already adjusted (has gap), need to check against
  // adjusted pool
  if (pool_allocated && pool_base && pool_fd >= 0) {
    // The pool_base is the real address, but jemalloc works with adjusted addresses
    void *adjusted_pool_base = pointer_advance(pool_base, plasma::kMmapRegionsGap);
    void *adjusted_pool_end = pointer_advance(pool_base, pool_size);

    // Check if address is within the adjusted pool range
    if (addr >= adjusted_pool_base && addr < adjusted_pool_end) {
      // Calculate offset from the real pool base
      // addr is adjusted, so we need to account for the gap
      ptrdiff_t offset =
          static_cast<char *>(addr) - static_cast<char *>(adjusted_pool_base);

      // Create a unique id for this allocation
      static std::atomic<int64_t> alloc_unique_id{1};
      MEMFD_TYPE fd_pair = MEMFD_TYPE(pool_fd, alloc_unique_id++);

      RAY_LOG(DEBUG) << "BuildAllocation: addr " << addr << " is at offset " << offset
                     << " in pool (pool_base=" << pool_base << ", pool_fd=" << pool_fd
                     << ")";

      return std::optional<Allocation>(Allocation(addr,
                                                  static_cast<int64_t>(size),
                                                  fd_pair,
                                                  offset,
                                                  0,
                                                  static_cast<int64_t>(size),
                                                  is_fallback_allocated));
    }
  }

  // Shouldn't reach here with properly configured single-pool extent hooks
  RAY_LOG(WARNING) << "Address " << addr << " not found in single pool"
                   << " (pool_allocated=" << pool_allocated << ", pool_base=" << pool_base
                   << ", pool_size=" << pool_size << ")";
  return std::nullopt;
}

}  // namespace plasma
