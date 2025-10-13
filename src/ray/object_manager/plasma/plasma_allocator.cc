// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ray/object_manager/plasma/plasma_allocator.h"

#include <string>
#include <utility>

#include "ray/common/ray_config.h"
#include "ray/object_manager/plasma/malloc.h"
#include "ray/stats/metric_defs.h"
#include "ray/stats/tag_defs.h"
#include "ray/util/logging.h"

namespace plasma {
namespace internal {
bool IsOutsideInitialAllocation(void *ptr);

void SetDLMallocConfig(const std::string &plasma_directory,
                       const std::string &fallback_directory,
                       bool hugepage_enabled,
                       bool fallback_enabled);
}  // namespace internal

extern "C" {
void *dlmemalign(size_t alignment, size_t bytes);
void dlfree(void *mem);
int dlmallopt(int param_number, int value);
}

namespace {
/* Copied from dlmalloc.c; make sure to keep in sync */
size_t MAX_SIZE_T = static_cast<size_t>(-1);
const int M_MMAP_THRESHOLD = -3;

// We align the allocated region to a 64-byte boundary. This is not
// strictly necessary, but it is an optimization that could speed up the
// computation of a hash of the data (see compute_object_hash_parallel in
// plasma_client.cc). Note that even though this pointer is 64-byte aligned,
// it is not guaranteed that the corresponding pointer in the client will be
// 64-byte aligned, but in practice it often will be.
const size_t kAllocationAlignment = 64;

// We are using a single memory-mapped file by mallocing and freeing a single
// large amount of space up front. According to the documentation,
// dlmalloc might need up to 128*sizeof(size_t) bytes for internal
// bookkeeping.
const int64_t kDlMallocReserved = 256 * sizeof(size_t);

}  // namespace

PlasmaAllocator::PlasmaAllocator(const std::string &plasma_directory,
                                 const std::string &fallback_directory,
                                 bool hugepage_enabled,
                                 int64_t footprint_limit)
    : kFootprintLimit(footprint_limit),
      kAlignment(kAllocationAlignment),
      allocated_(0),
      fallback_allocated_(0) {
  internal::SetDLMallocConfig(plasma_directory,
                              fallback_directory,
                              hugepage_enabled,
                              /*fallback_enabled=*/true);
  RAY_CHECK(kFootprintLimit > kDlMallocReserved)
      << "Footprint limit has to be greater than " << kDlMallocReserved;
  auto allocation = Allocate(kFootprintLimit - kDlMallocReserved);
  RAY_CHECK(allocation.has_value())
      << "PlasmaAllocator initialization failed."
      << " It's likely we don't have enough space in " << plasma_directory;
  // This will unmap the file, but the next one created will be as large
  // as this one (this is an implementation detail of dlmalloc).
  Free(std::move(allocation.value()));
}

std::optional<Allocation> PlasmaAllocator::Allocate(size_t bytes) {
  RAY_LOG(DEBUG) << "allocating " << bytes;
  void *mem = dlmemalign(kAlignment, bytes);
  RAY_LOG(DEBUG) << "allocated " << bytes << " at " << mem;
  if (!mem) {
    return absl::nullopt;
  }

  // Update metrics
  allocated_ += bytes;
  total_alloc_count_++;
  active_allocations_++;

  return BuildAllocation(mem, bytes, /* is_fallback_allocated */ false);
}

std::optional<Allocation> PlasmaAllocator::FallbackAllocate(size_t bytes) {
  bool is_fallback_allocated = false;

  // Forces allocation as a separate file.
  RAY_CHECK(dlmallopt(M_MMAP_THRESHOLD, 0));
  RAY_LOG(DEBUG) << "fallback allocating " << bytes;
  void *mem = dlmemalign(kAlignment, bytes);
  RAY_LOG(DEBUG) << "allocated " << bytes << " at " << mem;
  // Reset to the default value.
  RAY_CHECK(dlmallopt(M_MMAP_THRESHOLD, MAX_SIZE_T));

  if (!mem) {
    return absl::nullopt;
  }

  // Update metrics
  allocated_ += bytes;
  total_alloc_count_++;
  active_allocations_++;
  fallback_alloc_count_++;

  // The allocation was servicable using the initial region, no need to fallback.
  if (internal::IsOutsideInitialAllocation(mem)) {
    is_fallback_allocated = true;
    fallback_allocated_ += bytes;
  }
  return BuildAllocation(mem, bytes, is_fallback_allocated);
}

void PlasmaAllocator::Free(Allocation allocation) {
  RAY_CHECK(allocation.address_ != nullptr) << "Cannot free the nullptr";
  RAY_LOG(DEBUG) << "deallocating " << allocation.size_ << " at " << allocation.address_;
  dlfree(allocation.address_);

  // Update metrics
  allocated_ -= allocation.size_;
  total_free_count_++;
  active_allocations_--;

  if (internal::IsOutsideInitialAllocation(allocation.address_)) {
    fallback_allocated_ -= allocation.size_;
  }
}

int64_t PlasmaAllocator::GetFootprintLimit() const { return kFootprintLimit; }

int64_t PlasmaAllocator::Allocated() const { return allocated_; }

int64_t PlasmaAllocator::FallbackAllocated() const { return fallback_allocated_; }

std::optional<Allocation> PlasmaAllocator::BuildAllocation(void *addr,
                                                           size_t size,
                                                           bool is_fallback_allocated) {
  if (addr == nullptr) {
    return absl::nullopt;
  }
  MEMFD_TYPE fd;
  int64_t mmap_size;
  ptrdiff_t offset;

  if (internal::GetMallocMapinfo(addr, &fd, &mmap_size, &offset)) {
    return Allocation(addr,
                      static_cast<int64_t>(size),
                      std::move(fd),
                      offset,
                      0 /* device_number*/,
                      mmap_size,
                      is_fallback_allocated);
  }
  return absl::nullopt;
}

PlasmaAllocator::MemoryStats PlasmaAllocator::GetStats() const {
  MemoryStats stats;

  stats.allocated_bytes = allocated_;
  stats.footprint_limit = kFootprintLimit;
  stats.fallback_allocated_bytes = fallback_allocated_.load();
  stats.total_alloc_count = total_alloc_count_.load();
  stats.total_free_count = total_free_count_.load();
  stats.fallback_alloc_count = fallback_alloc_count_.load();
  stats.active_allocations = active_allocations_.load();

  // Estimate fragmentation as unused space in the footprint limit
  // This is a rough estimate since dlmalloc doesn't expose detailed fragmentation info
  int64_t used_space = allocated_ + kDlMallocReserved;
  if (kFootprintLimit > 0 && used_space > 0) {
    stats.fragmentation_estimate =
        1.0 - static_cast<double>(used_space) / static_cast<double>(kFootprintLimit);
    // Clamp to reasonable bounds
    if (stats.fragmentation_estimate < 0) stats.fragmentation_estimate = 0.0;
    if (stats.fragmentation_estimate > 1) stats.fragmentation_estimate = 1.0;
  } else {
    stats.fragmentation_estimate = 0.0;
  }

  return stats;
}

void PlasmaAllocator::RecordMetrics() const {
  auto stats = GetStats();

  // Record basic allocator statistics - using dlmalloc prefix for distinction
  ray::stats::STATS_object_store_memory.Record(
      stats.allocated_bytes, {{ray::stats::LocationKey, "dlmalloc_allocated"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.footprint_limit, {{ray::stats::LocationKey, "dlmalloc_footprint_limit"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.fallback_allocated_bytes,
      {{ray::stats::LocationKey, "dlmalloc_fallback_allocated"}});

  // Record fragmentation as a percentage (0-100)
  ray::stats::STATS_object_store_memory.Record(
      static_cast<int64_t>(stats.fragmentation_estimate * 100),
      {{ray::stats::LocationKey, "dlmalloc_fragmentation_pct"}});

  // Record operation counts
  ray::stats::STATS_object_store_memory.Record(
      stats.total_alloc_count, {{ray::stats::LocationKey, "dlmalloc_total_allocs"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.total_free_count, {{ray::stats::LocationKey, "dlmalloc_total_frees"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.fallback_alloc_count,
      {{ray::stats::LocationKey, "dlmalloc_fallback_allocs"}});

  ray::stats::STATS_object_store_memory.Record(
      stats.active_allocations,
      {{ray::stats::LocationKey, "dlmalloc_active_allocations"}});
}

void PlasmaAllocator::GetDebugDump(std::stringstream &buffer) const {
  auto stats = GetStats();

  buffer << "\n=== PlasmaAllocator (dlmalloc) Debug Info ===\n";
  buffer << "- Memory limit: " << stats.footprint_limit << " bytes\n";
  buffer << "- Alignment: " << kAlignment << " bytes\n";
  buffer << "- Reserved for dlmalloc: " << kDlMallocReserved << " bytes\n";
  buffer << "\n";

  buffer << "=== Memory Statistics ===\n";
  buffer << "- Allocated: " << stats.allocated_bytes << " bytes\n";
  buffer << "- Fallback allocated: " << stats.fallback_allocated_bytes << " bytes\n";
  buffer << "- Fragmentation estimate: " << (stats.fragmentation_estimate * 100.0)
         << "%\n";
  buffer << "- Available: " << (stats.footprint_limit - stats.allocated_bytes)
         << " bytes\n";
  buffer << "\n";

  buffer << "=== Operation Counts ===\n";
  buffer << "- Total allocations: " << stats.total_alloc_count << "\n";
  buffer << "- Total frees: " << stats.total_free_count << "\n";
  buffer << "- Fallback allocations: " << stats.fallback_alloc_count << "\n";
  buffer << "- Active allocations: " << stats.active_allocations << "\n";

  // Calculate allocation/free balance
  int64_t balance = stats.total_alloc_count - stats.total_free_count;
  buffer << "- Allocation balance: " << balance << "\n";
  if (balance != stats.active_allocations) {
    buffer << "  WARNING: Balance mismatch with active allocations!\n";
  }
}

}  // namespace plasma
