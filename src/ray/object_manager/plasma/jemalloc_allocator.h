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

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "ray/object_manager/plasma/allocator.h"
#include "ray/object_manager/plasma/common.h"
#include "ray/util/compat.h"

// Use official jemalloc API
#include <jemalloc/jemalloc.h>

namespace plasma {

// Forward declare MmapRecord to avoid conflicts with malloc.h
struct JemallocMmapRecord {
  MEMFD_TYPE fd;     ///< File descriptor pair for the memory mapping
  size_t size;       ///< Size of the memory mapping in bytes
  std::string path;  ///< File path for the memory mapping
  bool is_fallback;  ///< Whether this is fallback allocation
};

/// JemallocAllocator optimized for large object allocation (>=100KB).
/// This allocator uses jemalloc with custom extent hooks to manage
/// memory from shared memory (/dev/shm) for zero-copy object sharing.
class JemallocAllocator : public IAllocator {
 public:
  /// Constructor for JemallocAllocator.
  /// \param plasma_directory Directory for shared memory files (usually /dev/shm).
  /// \param fallback_directory Directory for fallback allocation when memory is full.
  /// \param hugepage_enabled Whether to enable transparent huge pages.
  /// \param footprint_limit Maximum memory footprint in bytes.
  JemallocAllocator(const std::string &plasma_directory,
                    const std::string &fallback_directory,
                    bool hugepage_enabled,
                    int64_t footprint_limit);

  ~JemallocAllocator();

  /// Allocate memory using jemalloc with custom extent hooks.
  /// \param bytes Number of bytes to allocate.
  /// \return Allocated memory or nullopt if allocation fails.
  std::optional<Allocation> Allocate(size_t bytes) override;

  /// Fallback allocation from disk when main memory is full.
  /// \param bytes Number of bytes to allocate.
  /// \return Allocated memory or nullopt if allocation fails.
  std::optional<Allocation> FallbackAllocate(size_t bytes) override;

  /// Free allocated memory.
  /// \param allocation The allocation to free.
  void Free(Allocation allocation) override;

  /// Get the memory footprint limit.
  /// \return Memory limit in bytes.
  int64_t GetFootprintLimit() const override { return footprint_limit_; }

  /// Get total bytes allocated.
  /// \return Number of bytes currently allocated.
  int64_t Allocated() const override;

  /// Get total bytes fallback allocated.
  /// \return Number of bytes currently fallback allocated.
  int64_t FallbackAllocated() const override { return fallback_allocated_; }

  /// Get memory statistics for monitoring.
  struct MemoryStats {
    int64_t allocated_bytes;
    int64_t active_bytes;
    int64_t metadata_bytes;
    int64_t resident_bytes;
    int64_t retained_bytes;
    double fragmentation_ratio;
    int64_t coalesce_count;
    int64_t split_count;
  };

  /// Get current memory statistics.
  MemoryStats GetStats() const;

  /// Record metrics to Ray's metrics system.
  void RecordMetrics() const;

  /// Get debug information for troubleshooting.
  void GetDebugDump(std::stringstream &buffer) const;

  // Public accessors for extent hooks (needed for friend functions)
  const std::string &GetPlasmaDirectory() const { return plasma_directory_; }
  const std::string &GetFallbackDirectory() const { return fallback_directory_; }
  bool IsHugepageEnabled() const { return hugepage_enabled_; }
  absl::Mutex &GetMutex() const { return mutex_; }
  absl::flat_hash_map<void *, JemallocMmapRecord> &GetMmapRecords()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return mmap_records_;
  }
  void IncrementExtentAllocCount() { extent_alloc_count_++; }
  void IncrementFallbackAllocCount() { fallback_alloc_count_++; }
  void IncrementExtentDallocCount() { extent_dalloc_count_++; }
  void IncrementCoalesceCount() { coalesce_count_++; }
  void IncrementSplitCount() { split_count_++; }

 private:
  /// Initialize jemalloc with optimal configuration for large objects.
  void InitializeJemalloc();

  /// Install custom extent hooks for shared memory management.
  void InstallExtentHooks();

  /// Build an Allocation object from raw memory pointer.
  std::optional<Allocation> BuildAllocation(void *addr,
                                            size_t size,
                                            bool is_fallback_allocated);

  // Configuration
  const std::string plasma_directory_;
  const std::string fallback_directory_;
  const bool hugepage_enabled_;
  const int64_t footprint_limit_;

  // Memory tracking
  std::atomic<int64_t> allocated_{0};
  std::atomic<int64_t> fallback_allocated_{0};

  // Metrics
  mutable std::atomic<int64_t> coalesce_count_{0};
  mutable std::atomic<int64_t> split_count_{0};
  mutable std::atomic<int64_t> extent_alloc_count_{0};
  mutable std::atomic<int64_t> extent_dalloc_count_{0};
  mutable std::atomic<int64_t> fallback_alloc_count_{0};

  // Jemalloc arena for plasma allocations
  unsigned plasma_arena_index_;

  // Mutex for thread safety
  mutable absl::Mutex mutex_;

  // Track memory mappings for client access
  absl::flat_hash_map<void *, JemallocMmapRecord> mmap_records_ ABSL_GUARDED_BY(mutex_);

  // Custom extent hooks for jemalloc
  std::unique_ptr<extent_hooks_t> extent_hooks_;

  // Friend functions for extent hook callbacks
  friend void *PlasmaExtentAlloc(
      extent_hooks_t *, void *, size_t, size_t, bool *, bool *, unsigned);
  friend bool PlasmaExtentDalloc(extent_hooks_t *, void *, size_t, bool, unsigned);
  friend void PlasmaExtentDestroy(extent_hooks_t *, void *, size_t, bool, unsigned);
  friend bool PlasmaExtentCommit(
      extent_hooks_t *, void *, size_t, size_t, size_t, unsigned);
  friend bool PlasmaExtentDecommit(
      extent_hooks_t *, void *, size_t, size_t, size_t, unsigned);
  friend bool PlasmaExtentPurge(
      extent_hooks_t *, void *, size_t, size_t, size_t, unsigned);
  friend bool PlasmaExtentMerge(
      extent_hooks_t *, void *, size_t, void *, size_t, bool, unsigned);
  friend bool PlasmaExtentSplit(
      extent_hooks_t *, void *, size_t, size_t, size_t, bool, unsigned);
};

}  // namespace plasma
