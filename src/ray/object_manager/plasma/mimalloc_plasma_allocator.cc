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

#include "ray/object_manager/plasma/mimalloc_plasma_allocator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include "mimalloc.h"
#include "ray/util/logging.h"

namespace plasma {

namespace {

// Process-monotonic counter for the unique_id portion of MEMFD_TYPE.
std::atomic<int64_t> g_next_unique_id{1};

// Open + ftruncate + mmap one large file in `directory` of `size` bytes.
// Returns true on success and writes the fd / address out-params.
bool CreateBackingFile(const std::string &directory,
                       size_t size,
                       int *out_fd,
                       void **out_addr) {
  std::string file_template = directory + "/ray_mi_plasma_XXXXXX";
  std::vector<char> tmpl(file_template.begin(), file_template.end());
  tmpl.push_back('\0');
  int fd = mkstemp(tmpl.data());
  if (fd < 0) {
    RAY_LOG(ERROR) << "mkstemp failed: " << strerror(errno);
    return false;
  }
  if (unlink(tmpl.data()) != 0) {
    RAY_LOG(WARNING) << "unlink failed: " << strerror(errno);
  }
  if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
    RAY_LOG(ERROR) << "ftruncate failed: " << strerror(errno);
    close(fd);
    return false;
  }
  void *addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    RAY_LOG(ERROR) << "mmap failed: " << strerror(errno);
    close(fd);
    return false;
  }
  *out_fd = fd;
  *out_addr = addr;
  return true;
}

}  // namespace

MimallocPlasmaAllocator::MimallocPlasmaAllocator(
    const std::string &plasma_directory,
    const std::string &fallback_directory,
    bool hugepage_enabled,
    int64_t footprint_limit)
    : footprint_limit_(footprint_limit),
      alignment_(64),
      base_addr_(nullptr),
      mmap_size_(0),
      fd_(-1),
      unique_id_(0),
      arena_id_(nullptr),
      heap_(nullptr),
      allocated_(0),
      fallback_allocated_(0) {
  RAY_CHECK(!hugepage_enabled)
      << "Hugepages not yet supported by MimallocPlasmaAllocator spike";
  RAY_CHECK(footprint_limit > 0);

  if (!CreateBackingFile(
          plasma_directory, static_cast<size_t>(footprint_limit), &fd_, &base_addr_)) {
    RAY_LOG(FATAL) << "Failed to create backing file in " << plasma_directory;
  }
  mmap_size_ = static_cast<size_t>(footprint_limit);
  unique_id_ = g_next_unique_id.fetch_add(1);

  // Disable mimalloc's purge/decommit so pages stay live in the shared
  // file for the lifetime of this allocator. Decommitting would release
  // file pages back to the OS — fine for normal mimalloc, but breaks
  // Plasma's expectation that any client can read any offset at any time.
  mi_option_set(mi_option_purge_delay, -1);
  mi_option_set(mi_option_arena_reserve, 0);

  // Register the file-backed region as a mimalloc arena. is_committed=true
  // because ftruncate-backed mmap pages are reserved by the kernel.
  mi_arena_id_t arena_id = 0;
  bool ok = mi_manage_os_memory_ex(base_addr_,
                                   mmap_size_,
                                   /*is_committed=*/true,
                                   /*is_large=*/false,
                                   /*is_zero=*/true,
                                   /*numa_node=*/-1,
                                   /*exclusive=*/true,
                                   &arena_id);
  RAY_CHECK(ok) << "mi_manage_os_memory_ex failed";
  arena_id_ = reinterpret_cast<void *>(static_cast<uintptr_t>(arena_id));

  mi_heap_t *heap = mi_heap_new_in_arena(arena_id);
  RAY_CHECK(heap != nullptr) << "mi_heap_new_in_arena returned null";
  heap_ = static_cast<void *>(heap);
}

MimallocPlasmaAllocator::~MimallocPlasmaAllocator() {
  if (heap_ != nullptr) {
    mi_heap_destroy(static_cast<mi_heap_t *>(heap_));
  }
  if (base_addr_ != nullptr) {
    munmap(base_addr_, mmap_size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

std::optional<Allocation> MimallocPlasmaAllocator::Allocate(size_t bytes) {
  auto *heap = static_cast<mi_heap_t *>(heap_);
  void *mem = mi_heap_malloc_aligned(heap, bytes, alignment_);
  if (mem == nullptr) {
    return std::nullopt;
  }
  allocated_ += static_cast<int64_t>(bytes);
  return BuildAllocation(mem, bytes, /*is_fallback_allocated=*/false);
}

std::optional<Allocation> MimallocPlasmaAllocator::FallbackAllocate(size_t bytes) {
  // Spike: fallback shares the primary arena. Production will route
  // fallback through a second arena backed by the fallback directory.
  auto *heap = static_cast<mi_heap_t *>(heap_);
  void *mem = mi_heap_malloc_aligned(heap, bytes, alignment_);
  if (mem == nullptr) {
    return std::nullopt;
  }
  allocated_ += static_cast<int64_t>(bytes);
  fallback_allocated_ += static_cast<int64_t>(bytes);
  return BuildAllocation(mem, bytes, /*is_fallback_allocated=*/true);
}

void MimallocPlasmaAllocator::Free(Allocation allocation) {
  RAY_CHECK(allocation.address_ != nullptr) << "Cannot free the nullptr";
  // mi_free is safe across heaps; it locates the owning heap from the
  // pointer metadata.
  mi_free(allocation.address_);
  allocated_ -= allocation.size_;
  if (allocation.fallback_allocated_) {
    fallback_allocated_ -= allocation.size_;
  }
}

int64_t MimallocPlasmaAllocator::GetFootprintLimit() const {
  return footprint_limit_;
}

int64_t MimallocPlasmaAllocator::Allocated() const { return allocated_; }

int64_t MimallocPlasmaAllocator::FallbackAllocated() const {
  return fallback_allocated_;
}

std::optional<Allocation> MimallocPlasmaAllocator::BuildAllocation(
    void *addr, size_t size, bool is_fallback_allocated) {
  if (addr == nullptr) {
    return std::nullopt;
  }
  auto base = reinterpret_cast<uintptr_t>(base_addr_);
  auto candidate = reinterpret_cast<uintptr_t>(addr);
  if (candidate < base || candidate >= base + mmap_size_) {
    RAY_LOG(ERROR) << "mimalloc returned pointer outside our arena: " << addr;
    return std::nullopt;
  }
  ptrdiff_t offset = static_cast<ptrdiff_t>(candidate - base);
  MEMFD_TYPE fd{fd_, unique_id_};
  return Allocation(addr,
                    static_cast<int64_t>(size),
                    std::move(fd),
                    offset,
                    0 /* device_number */,
                    static_cast<int64_t>(mmap_size_),
                    is_fallback_allocated);
}

}  // namespace plasma
