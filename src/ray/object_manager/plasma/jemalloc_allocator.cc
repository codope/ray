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

#include <cstring>
#include <string>
#include <vector>

#include "ray/common/ray_config.h"
#include "ray/object_manager/plasma/malloc.h"
#include "ray/stats/metric_defs.h"
#include "ray/util/logging.h"

// jemalloc functions
extern "C" {
void *je_mallocx(size_t size, int flags);
void je_sdallocx(void *ptr, size_t size, int flags);
int je_mallctl(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);

// jemalloc flags
#define MALLOCX_ZERO ((int)0x40)
#define MALLOCX_ARENA(a) ((int)(((a) + 1) << 8))
#define MALLOCX_ALIGN(a) ((int)(((a) < 63 ? (a) : 63) << 0))
}

namespace plasma {

namespace {

// Alignment for large objects to utilize huge pages effectively
constexpr size_t kLargeObjectAlignment = 2 * 1024 * 1024;  // 2MB for huge pages

// Minimum object size for Plasma (100KB)
constexpr size_t kMinPlasmaObjectSize = 100 * 1024;

// Gap between memory regions to prevent coalescing (same as dlmalloc)
constexpr int64_t kMmapRegionsGap = sizeof(size_t);

// Track whether we've allocated once (for fallback logic)
thread_local bool allocated_once = false;

// Give each mmap record a unique id to disambiguate fd reuse
thread_local int64_t next_mmap_unique_id = INVALID_UNIQUE_FD_ID + 1;

// Track the initial allocation region
thread_local void *initial_region_ptr = nullptr;
thread_local size_t initial_region_size = 0;

// Thread-local pointer to the allocator instance for extent hooks
thread_local JemallocAllocator *g_allocator_instance = nullptr;

// Thread-local flag to indicate we're in fallback allocation mode
thread_local bool in_fallback_mode = false;

// Helper functions
void *pointer_advance(void *p, ptrdiff_t n) {
  return static_cast<unsigned char *>(p) + n;
}

void *pointer_retreat(void *p, ptrdiff_t n) {
  return static_cast<unsigned char *>(p) - n;
}

// Create and mmap a buffer, handling both primary and fallback allocation
void create_and_mmap_buffer(int64_t size, void **pointer, int *fd,
                           const std::string &plasma_directory,
                           const std::string &fallback_directory,
                           bool hugepages_enabled,
                           bool fallback_enabled) {
  // Choose directory based on allocation mode
  std::string file_template = plasma_directory;

  // If we've allocated once and fallback is enabled, use fallback directory
  if (allocated_once && fallback_enabled && in_fallback_mode) {
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
  if (allocated_once && fallback_enabled && in_fallback_mode) {
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
  } else if (!allocated_once) {
    // Track the initial region for fallback detection
    initial_region_ptr = *pointer;
    initial_region_size = size;
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

// Check if pointer is outside the initial allocation
bool IsOutsideInitialAllocation(void *p) {
  if (initial_region_ptr == nullptr) {
    return false;
  }
  char *ptr = static_cast<char *>(p);
  char *initial_end = static_cast<char *>(initial_region_ptr) + initial_region_size;
  return (ptr < initial_region_ptr) || (ptr >= initial_end);
}

// Extent hook functions that will be registered with jemalloc
void *PlasmaExtentAlloc(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
                        size_t alignment, bool *zero, bool *commit, unsigned arena_ind) {
  if (!g_allocator_instance) {
    return nullptr;
  }

  // Add gap to prevent coalescing (same as dlmalloc)
  size += kMmapRegionsGap;

  // Check if this is a fallback allocation
  bool is_fallback = in_fallback_mode &&
                     g_allocator_instance->fallback_directory_ != "" &&
                     allocated_once;

  if (!is_fallback && allocated_once && g_allocator_instance->fallback_directory_ != "") {
    // Refuse to overcommit in non-fallback mode after initial allocation
    RAY_LOG(DEBUG) << "Refusing to overcommit in non-fallback mode: " << size;
    return nullptr;
  }

  void *pointer;
  int fd;
  create_and_mmap_buffer(size, &pointer, &fd,
                        g_allocator_instance->plasma_directory_,
                        g_allocator_instance->fallback_directory_,
                        g_allocator_instance->hugepage_enabled_,
                        is_fallback);

  if (pointer == MAP_FAILED) {
    return nullptr;
  }

  allocated_once = true;

  // Update metrics
  g_allocator_instance->extent_alloc_count_++;
  if (is_fallback) {
    g_allocator_instance->fallback_alloc_count_++;
  }

  // Track the mapping
  {
    absl::MutexLock lock(&g_allocator_instance->mutex_);
    MmapRecord &record = g_allocator_instance->mmap_records_[pointer];
    record.fd = MEMFD_TYPE(fd, next_mmap_unique_id++);
    record.size = size;
  }

  // Advance pointer by gap amount (lie to jemalloc about actual location)
  void *adjusted_pointer = pointer_advance(pointer, kMmapRegionsGap);

  RAY_LOG(DEBUG) << "PlasmaExtentAlloc: allocated " << size << " bytes at "
                 << adjusted_pointer << " (actual: " << pointer << ")";

  *zero = true;
  *commit = true;
  return adjusted_pointer;
}

bool PlasmaExtentDalloc(extent_hooks_t *extent_hooks, void *addr, size_t size,
                       bool committed, unsigned arena_ind) {
  if (!g_allocator_instance) {
    return true;
  }

  // Retreat pointer to actual location
  addr = pointer_retreat(addr, kMmapRegionsGap);
  size += kMmapRegionsGap;

  absl::MutexLock lock(&g_allocator_instance->mutex_);
  auto it = g_allocator_instance->mmap_records_.find(addr);

  if (it == g_allocator_instance->mmap_records_.end() || it->second.size != size) {
    // Reject requests that don't match previous allocations
    RAY_LOG(WARNING) << "PlasmaExtentDalloc: mismatch for " << addr << ", size " << size;
    return true;
  }

  RAY_LOG(INFO) << "PlasmaExtentDalloc: freeing " << size << " bytes at " << addr;

  // Update metrics
  g_allocator_instance->extent_dalloc_count_++;

  int r = munmap(addr, size);
  if (r == 0) {
    close(it->second.fd.first);
    g_allocator_instance->mmap_records_.erase(it);
  } else {
    RAY_LOG(ERROR) << "munmap failed: " << strerror(errno);
  }

  return r != 0;
}

void PlasmaExtentDestroy(extent_hooks_t *extent_hooks, void *addr, size_t size,
                        bool committed, unsigned arena_ind) {
  // Retreat pointer to actual location
  addr = pointer_retreat(addr, kMmapRegionsGap);
  size += kMmapRegionsGap;

  // For uncommitted extents, just unmap
  munmap(addr, size);

  if (g_allocator_instance) {
    absl::MutexLock lock(&g_allocator_instance->mutex_);
    g_allocator_instance->mmap_records_.erase(addr);
  }
}

bool PlasmaExtentCommit(extent_hooks_t *extent_hooks, void *addr, size_t size,
                       size_t offset, size_t length, unsigned arena_ind) {
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
  if (g_allocator_instance && g_allocator_instance->hugepage_enabled_ &&
      length >= kLargeObjectAlignment) {
    madvise(target, length, MADV_HUGEPAGE);
  }

  return false;
}

bool PlasmaExtentDecommit(extent_hooks_t *extent_hooks, void *addr, size_t size,
                          size_t offset, size_t length, unsigned arena_ind) {
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

bool PlasmaExtentPurge(extent_hooks_t *extent_hooks, void *addr, size_t size,
                      size_t offset, size_t length, unsigned arena_ind) {
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

bool PlasmaExtentMerge(extent_hooks_t *extent_hooks, void *addr_a, size_t size_a,
                      void *addr_b, size_t size_b, bool committed, unsigned arena_ind) {
  if (!g_allocator_instance) {
    return true;
  }

  // Verify adjacency (with gap adjustment)
  if (static_cast<char *>(addr_a) + size_a != addr_b) {
    return true;  // Cannot merge non-adjacent extents
  }

  // Retreat pointers to actual locations for tracking
  void *real_addr_a = pointer_retreat(addr_a, kMmapRegionsGap);
  void *real_addr_b = pointer_retreat(addr_b, kMmapRegionsGap);
  size_t real_size_a = size_a + kMmapRegionsGap;
  size_t real_size_b = size_b + kMmapRegionsGap;

  // Update tracking
  {
    absl::MutexLock lock(&g_allocator_instance->mutex_);
    auto it_a = g_allocator_instance->mmap_records_.find(real_addr_a);
    auto it_b = g_allocator_instance->mmap_records_.find(real_addr_b);

    if (it_a != g_allocator_instance->mmap_records_.end() &&
        it_b != g_allocator_instance->mmap_records_.end()) {
      // Merge: extend size of first extent, remove second
      it_a->second.size = real_size_a + real_size_b - kMmapRegionsGap;
      g_allocator_instance->mmap_records_.erase(it_b);
    }
  }

  // Track coalescing metrics
  g_allocator_instance->coalesce_count_++;

  RAY_LOG(DEBUG) << "Merged extents: " << size_a << " + " << size_b << " bytes";
  return false;
}

bool PlasmaExtentSplit(extent_hooks_t *extent_hooks, void *addr, size_t size,
                      size_t size_a, size_t size_b, bool committed, unsigned arena_ind) {
  if (!g_allocator_instance) {
    return true;
  }

  // Verify sizes
  if (size_a + size_b != size) {
    return true;
  }

  // Calculate real addresses and sizes
  void *real_addr = pointer_retreat(addr, kMmapRegionsGap);
  void *real_addr_b = pointer_advance(real_addr, size_a + kMmapRegionsGap);

  // Update tracking
  {
    absl::MutexLock lock(&g_allocator_instance->mutex_);
    auto it = g_allocator_instance->mmap_records_.find(real_addr);
    if (it != g_allocator_instance->mmap_records_.end()) {
      MEMFD_TYPE fd = it->second.fd;

      // Update first extent size
      it->second.size = size_a + kMmapRegionsGap;

      // Add second extent
      g_allocator_instance->mmap_records_[real_addr_b] = {fd,
                                                          (int64_t)(size_b + kMmapRegionsGap)};
    }
  }

  // Track split metrics
  g_allocator_instance->split_count_++;

  RAY_LOG(DEBUG) << "Split extent: " << size << " into " << size_a << " + " << size_b;
  return false;
}

// Global extent hooks structure
extent_hooks_t plasma_extent_hooks = {
    PlasmaExtentAlloc,    PlasmaExtentDalloc,  PlasmaExtentDestroy,
    PlasmaExtentCommit,   PlasmaExtentDecommit, PlasmaExtentPurge,
    PlasmaExtentMerge,    PlasmaExtentSplit};

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

  // Reset allocation tracking
  allocated_once = false;
  initial_region_ptr = nullptr;
  initial_region_size = 0;

  InitializeJemalloc();
  InstallExtentHooks();

  RAY_LOG(INFO) << "JemallocAllocator initialized with " << footprint_limit
                << " bytes limit, plasma_dir=" << plasma_directory
                << ", fallback_dir=" << fallback_directory;
}

JemallocAllocator::~JemallocAllocator() {
  // Clean up any remaining allocations
  absl::MutexLock lock(&mutex_);
  for (const auto &[addr, record] : mmap_records_) {
    munmap(addr, record.size);
    close(record.fd.first);
  }
  mmap_records_.clear();

  g_allocator_instance = nullptr;
}

void JemallocAllocator::InitializeJemalloc() {
  // Configure jemalloc for large object optimization

  // Reduce extent fragmentation
  size_t lg_extent_max_active_fit = 4;
  je_mallctl("opt.lg_extent_max_active_fit", nullptr, nullptr,
             &lg_extent_max_active_fit, sizeof(size_t));

  // Set quantum for better alignment with large objects
  size_t lg_quantum = 17;  // 128KB quantum
  je_mallctl("opt.lg_quantum", nullptr, nullptr, &lg_quantum, sizeof(size_t));

  // Configure oversize threshold
  size_t oversize_threshold = 64 * 1024 * 1024;  // 64MB
  je_mallctl("opt.oversize_threshold", nullptr, nullptr, &oversize_threshold,
             sizeof(size_t));

  // Memory decay settings for better reuse
  ssize_t dirty_decay_ms = 5000;
  ssize_t muzzy_decay_ms = 10000;
  je_mallctl("opt.dirty_decay_ms", nullptr, nullptr, &dirty_decay_ms, sizeof(ssize_t));
  je_mallctl("opt.muzzy_decay_ms", nullptr, nullptr, &muzzy_decay_ms, sizeof(ssize_t));

  // Enable transparent huge pages if configured
  if (hugepage_enabled_) {
    bool thp_enabled = true;
    je_mallctl("opt.thp", nullptr, nullptr, &thp_enabled, sizeof(bool));
    je_mallctl("opt.metadata_thp", nullptr, nullptr, &thp_enabled, sizeof(bool));
  }

  // Disable thread cache for large objects
  bool tcache_enabled = false;
  je_mallctl("opt.tcache", nullptr, nullptr, &tcache_enabled, sizeof(bool));

  RAY_LOG(INFO) << "jemalloc configured for large object optimization";
}

void JemallocAllocator::InstallExtentHooks() {
  // Create a new arena with custom extent hooks
  unsigned arena_index;
  size_t sz = sizeof(unsigned);

  if (je_mallctl("arenas.create", &arena_index, &sz, nullptr, 0) != 0) {
    RAY_LOG(FATAL) << "Failed to create jemalloc arena";
  }

  plasma_arena_index_ = arena_index;

  // Install custom extent hooks for the arena
  std::string hooks_key = "arena." + std::to_string(arena_index) + ".extent_hooks";
  extent_hooks_t *hooks_ptr = &plasma_extent_hooks;

  if (je_mallctl(hooks_key.c_str(), nullptr, nullptr, &hooks_ptr,
                 sizeof(extent_hooks_t *)) != 0) {
    RAY_LOG(ERROR) << "Failed to install extent hooks";
  }

  RAY_LOG(INFO) << "Installed custom extent hooks for arena " << arena_index;
}

std::optional<Allocation> JemallocAllocator::Allocate(size_t bytes) {
  if (bytes < kMinPlasmaObjectSize) {
    RAY_LOG(WARNING) << "Requested allocation " << bytes
                     << " bytes is less than minimum " << kMinPlasmaObjectSize;
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

  void *ptr = je_mallocx(aligned_size, flags);
  if (!ptr) {
    RAY_LOG(WARNING) << "Failed to allocate " << aligned_size << " bytes";
    return std::nullopt;
  }

  allocated_ += aligned_size;

  RAY_LOG(DEBUG) << "Allocated " << aligned_size << " bytes at " << ptr;

  return BuildAllocation(ptr, aligned_size, false);
}

std::optional<Allocation> JemallocAllocator::FallbackAllocate(size_t bytes) {
  if (fallback_directory_.empty()) {
    RAY_LOG(ERROR) << "Fallback allocation requested but no fallback directory configured";
    return std::nullopt;
  }

  // Round up to page boundary
  size_t aligned_size = (bytes + 4095) & ~4095;

  // Set fallback mode flag
  in_fallback_mode = true;

  // Use custom arena with extent hooks
  int flags = MALLOCX_ARENA(plasma_arena_index_) | MALLOCX_ZERO;

  void *ptr = je_mallocx(aligned_size, flags);

  // Reset fallback mode flag
  in_fallback_mode = false;

  if (!ptr) {
    RAY_LOG(WARNING) << "Failed to fallback allocate " << aligned_size << " bytes";
    return std::nullopt;
  }

  // Check if this allocation is actually outside the initial region
  void *real_ptr = pointer_retreat(ptr, kMmapRegionsGap);
  bool is_outside = IsOutsideInitialAllocation(real_ptr);

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
  je_sdallocx(allocation.address_, size, 0);

  allocated_ -= size;
  if (allocation.fallback_allocated_) {
    fallback_allocated_ -= size;
  }

  RAY_LOG(DEBUG) << "Freed " << size << " bytes at " << allocation.address_;
}

int64_t JemallocAllocator::Allocated() const {
  // Query jemalloc for accurate stats
  size_t allocated, sz = sizeof(size_t);
  je_mallctl("stats.allocated", &allocated, &sz, nullptr, 0);
  return allocated;
}

JemallocAllocator::MemoryStats JemallocAllocator::GetStats() const {
  MemoryStats stats;
  size_t sz = sizeof(size_t);

  je_mallctl("stats.allocated", &stats.allocated_bytes, &sz, nullptr, 0);
  je_mallctl("stats.active", &stats.active_bytes, &sz, nullptr, 0);
  je_mallctl("stats.metadata", &stats.metadata_bytes, &sz, nullptr, 0);
  je_mallctl("stats.resident", &stats.resident_bytes, &sz, nullptr, 0);
  je_mallctl("stats.retained", &stats.retained_bytes, &sz, nullptr, 0);

  // Calculate fragmentation ratio
  if (stats.active_bytes > 0) {
    stats.fragmentation_ratio =
        1.0 - (double)stats.allocated_bytes / stats.active_bytes;
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
      stats.allocated_bytes,
      {{ray::stats::LocationKey, "jemalloc_allocated"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.active_bytes,
      {{ray::stats::LocationKey, "jemalloc_active"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.resident_bytes,
      {{ray::stats::LocationKey, "jemalloc_resident"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.metadata_bytes,
      {{ray::stats::LocationKey, "jemalloc_metadata"}});

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
      extent_alloc_count_.load(),
      {{ray::stats::LocationKey, "jemalloc_extent_allocs"}});

  ray::stats::STATS_object_store_memory.Record(
      extent_dalloc_count_.load(),
      {{ray::stats::LocationKey, "jemalloc_extent_deallocs"}});

  ray::stats::STATS_object_store_memory.Record(
      coalesce_count_.load(),
      {{ray::stats::LocationKey, "jemalloc_coalesce_ops"}});

  ray::stats::STATS_object_store_memory.Record(
      split_count_.load(),
      {{ray::stats::LocationKey, "jemalloc_split_ops"}});

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
  for (const auto &[addr, record] : mmap_records_) {
    total_mapped += record.size;
  }
  buffer << "- Total mapped: " << total_mapped << " bytes\n";
}

std::optional<Allocation> JemallocAllocator::BuildAllocation(void *addr, size_t size,
                                                            bool is_fallback_allocated) {
  if (!addr) {
    return std::nullopt;
  }

  // The address from jemalloc is already adjusted, retreat to get real address
  void *real_addr = pointer_retreat(addr, kMmapRegionsGap);

  absl::MutexLock lock(&mutex_);
  auto it = mmap_records_.find(real_addr);
  if (it != mmap_records_.end()) {
    // Calculate offset from the real mmap base
    ptrdiff_t offset = kMmapRegionsGap;

    return Allocation(addr, size, it->second.fd, offset, 0,
                     it->second.size - kMmapRegionsGap, is_fallback_allocated);
  }

  RAY_LOG(WARNING) << "Address " << real_addr << " not found in mmap_records";
  return std::nullopt;
}

}  // namespace plasma