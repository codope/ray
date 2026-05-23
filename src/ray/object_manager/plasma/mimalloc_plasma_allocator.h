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

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "ray/object_manager/plasma/allocator.h"
#include "ray/object_manager/plasma/common.h"
#include "ray/util/compat.h"

namespace plasma {

// Spike (Phase 1.5 of the allocator evaluation plan) — minimum-viable
// IAllocator backed by mimalloc, organized around a single pre-allocated
// shared-memory file the same way dlmalloc is today.
//
// Architectural choices:
//   * Pre-allocate one giant shm file in plasma_directory and register it
//     as a mimalloc arena via `mi_manage_os_memory_ex`. All user
//     allocations are carved out of this single file. Every allocation
//     resolves to (the_one_fd, addr - base, mmap_size) cleanly — same
//     contract as dlmalloc today, no segment-header co-location concern
//     because the segment headers live inside the same shared file as the
//     user data, and that file is what every client maps.
//   * One heap (`mi_heap_t`) bound to the arena. All Allocate/Free calls
//     go through this heap, ensuring allocations stay inside our arena
//     rather than being promoted to mimalloc's default arena under
//     pressure.
//   * Decommit / purge disabled via mimalloc options — pages stay
//     committed in the shared file for the allocator's lifetime.
//
// The spike does NOT yet implement: hugepages, fallback-directory routing
// (FallbackAllocate uses the same arena for now), or Windows.
//
// NOT thread-safe — must be called only from the Plasma store thread,
// matching the IAllocator contract at allocator.h:26.
class MimallocPlasmaAllocator : public IAllocator {
 public:
  MimallocPlasmaAllocator(const std::string &plasma_directory,
                          const std::string &fallback_directory,
                          bool hugepage_enabled,
                          int64_t footprint_limit);
  ~MimallocPlasmaAllocator() override;

  std::optional<Allocation> Allocate(size_t bytes) override;
  std::optional<Allocation> FallbackAllocate(size_t bytes) override;
  void Free(Allocation allocation) override;
  int64_t GetFootprintLimit() const override;
  int64_t Allocated() const override;
  int64_t FallbackAllocated() const override;

 private:
  std::optional<Allocation> BuildAllocation(void *addr,
                                            size_t size,
                                            bool is_fallback_allocated);

  const int64_t footprint_limit_;
  const size_t alignment_;
  // The single backing file. base_addr_ / mmap_size_ define the region
  // mimalloc owns; fd_ / unique_id_ are surfaced to clients.
  void *base_addr_;
  size_t mmap_size_;
  int fd_;
  int64_t unique_id_;
  // mimalloc handles. Opaque types stored as void* to keep this header
  // mimalloc-include-free.
  void *arena_id_;
  void *heap_;
  int64_t allocated_;
  std::atomic<int64_t> fallback_allocated_;
};

}  // namespace plasma
