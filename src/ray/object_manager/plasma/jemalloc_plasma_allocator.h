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

// Spike (Phase 1.4 of the allocator evaluation plan) — minimum-viable
// IAllocator backed by jemalloc with a custom arena. Each user-visible
// allocation is backed by exactly one shared-memory file so that the
// existing Plasma client mmap-by-(fd, offset, mmap_size) contract is
// preserved.
//
// Architectural choices (validated in the spike, locked for production):
//   * One arena per process. Hooks read shared state via a process-local
//     singleton, mirroring the file-static `allocated_once` constraint
//     of the dlmalloc impl.
//   * extent_split returns false (refuse). jemalloc may otherwise split an
//     extent into two sub-extents that both live in the same backing file,
//     which destroys the unambiguous (fd, offset) mapping.
//   * extent_merge returns false (refuse). Same reasoning in reverse —
//     merging two extents from different files is meaningless under our
//     model.
//   * Each extent_hooks alloc → one mkostemp+ftruncate+mmap of exactly the
//     requested size. Each extent_hooks dalloc → munmap+close.
//
// The spike does NOT yet implement: hugepages, fallback-directory routing
// when /dev/shm is full, MAP_POPULATE preallocation, or Windows. These are
// orthogonal to the architectural feasibility question.
//
// NOT thread-safe — must be called only from the Plasma store thread,
// matching the IAllocator contract at allocator.h:26.
class JemallocPlasmaAllocator : public IAllocator {
 public:
  JemallocPlasmaAllocator(const std::string &plasma_directory,
                          const std::string &fallback_directory,
                          bool hugepage_enabled,
                          int64_t footprint_limit);
  ~JemallocPlasmaAllocator() override;

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
  unsigned arena_index_;
  int64_t allocated_;
  std::atomic<int64_t> fallback_allocated_;
};

}  // namespace plasma
