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

#include "ray/object_manager/plasma/jemalloc_plasma_allocator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "jemalloc/jemalloc.h"
#include "ray/util/logging.h"

namespace plasma {

namespace {

// Per-extent metadata. We keep enough to reconstruct the (fd, offset,
// mmap_size) triple Plasma needs and to munmap+close on free. unique_id
// is a monotonically-increasing process-local identifier carried in the
// MEMFD_TYPE pair (second element) and surfaced to clients via the wire
// protocol (see protocol.cc:301).
struct ExtentRecord {
  int fd;
  int64_t unique_id;
  void *addr;
  size_t mmap_size;
};

// Process-singleton holding all the state the C-style extent hooks need.
// Hooks have no context pointer in jemalloc 5.x, so we route through a
// static singleton. This matches the file-static `allocated_once` model of
// the dlmalloc backend and inherits the same "one allocator per process"
// constraint documented at plasma_allocator.h:33.
class JemallocBackend {
 public:
  static JemallocBackend &Get() {
    static JemallocBackend instance;
    return instance;
  }

  void Init(const std::string &plasma_directory,
            const std::string &fallback_directory,
            int64_t footprint_limit) {
    absl::MutexLock lock(&mu_);
    RAY_CHECK(!initialized_) << "JemallocBackend can only be initialized once per process";
    plasma_directory_ = plasma_directory;
    fallback_directory_ = fallback_directory;
    footprint_limit_ = footprint_limit;
    committed_bytes_ = 0;
    initialized_ = true;
  }

  // Called from the extent_hooks alloc callback. Creates one shm file
  // sized to `size` bytes and mmaps it. Records the (fd, addr, size) triple
  // so Plasma can later resolve any pointer in this extent back to its
  // backing file.
  void *AllocExtent(size_t size, size_t alignment) {
    absl::MutexLock lock(&mu_);
    if (committed_bytes_ + static_cast<int64_t>(size) > footprint_limit_) {
      return nullptr;
    }
    std::string file_template = plasma_directory_ + "/ray_je_plasma_XXXXXX";
    std::vector<char> tmpl(file_template.begin(), file_template.end());
    tmpl.push_back('\0');
    int fd = mkstemp(tmpl.data());
    if (fd < 0) {
      RAY_LOG(ERROR) << "mkstemp failed: " << strerror(errno);
      return nullptr;
    }
    // Unlink immediately; the file lives only via its open fds.
    if (unlink(tmpl.data()) != 0) {
      RAY_LOG(WARNING) << "unlink failed: " << strerror(errno);
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
      RAY_LOG(ERROR) << "ftruncate failed: " << strerror(errno);
      close(fd);
      return nullptr;
    }
    void *addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
      RAY_LOG(ERROR) << "mmap failed: " << strerror(errno);
      close(fd);
      return nullptr;
    }
    // Alignment guarantee: mmap returns page-aligned addresses, so any
    // alignment ≤ page size is satisfied. For larger alignment (64-byte
    // is typical for Plasma) the page alignment subsumes it.
    if (alignment > 0 &&
        (reinterpret_cast<uintptr_t>(addr) & (alignment - 1)) != 0) {
      RAY_LOG(ERROR) << "mmap returned mis-aligned address (requested align="
                     << alignment << ")";
      munmap(addr, size);
      close(fd);
      return nullptr;
    }
    records_[addr] = ExtentRecord{fd, next_unique_id_++, addr, size};
    committed_bytes_ += size;
    return addr;
  }

  // Called from the extent_hooks dalloc callback.
  bool FreeExtent(void *addr, size_t size) {
    absl::MutexLock lock(&mu_);
    auto it = records_.find(addr);
    if (it == records_.end()) {
      RAY_LOG(ERROR) << "FreeExtent: no record for " << addr;
      return true;  // jemalloc convention: true == failure
    }
    int fd = it->second.fd;
    size_t mmap_size = it->second.mmap_size;
    RAY_CHECK_EQ(mmap_size, size) << "extent size mismatch (split happened?)";
    records_.erase(it);
    committed_bytes_ -= size;
    munmap(addr, mmap_size);
    close(fd);
    return false;
  }

  // Lookup helper used by JemallocPlasmaAllocator::BuildAllocation to
  // resolve a pointer to its backing (fd, unique_id, offset, mmap_size).
  // Because extent_split is refused, every live allocation lives in exactly
  // one extent.
  bool LookupMapinfo(void *addr,
                     int *fd,
                     int64_t *unique_id,
                     int64_t *mmap_size,
                     ptrdiff_t *offset) {
    absl::MutexLock lock(&mu_);
    for (const auto &kv : records_) {
      auto base = reinterpret_cast<uintptr_t>(kv.first);
      auto candidate = reinterpret_cast<uintptr_t>(addr);
      if (candidate >= base && candidate < base + kv.second.mmap_size) {
        *fd = kv.second.fd;
        *unique_id = kv.second.unique_id;
        *mmap_size = static_cast<int64_t>(kv.second.mmap_size);
        *offset = static_cast<ptrdiff_t>(candidate - base);
        return true;
      }
    }
    return false;
  }

 private:
  absl::Mutex mu_;
  bool initialized_ ABSL_GUARDED_BY(mu_) = false;
  std::string plasma_directory_ ABSL_GUARDED_BY(mu_);
  std::string fallback_directory_ ABSL_GUARDED_BY(mu_);
  int64_t footprint_limit_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t committed_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t next_unique_id_ ABSL_GUARDED_BY(mu_) = 1;
  absl::flat_hash_map<void *, ExtentRecord> records_ ABSL_GUARDED_BY(mu_);
};

// ---- jemalloc extent_hooks_t callbacks ----
//
// jemalloc 5.x extent_hooks API: the hooks struct itself has no context
// pointer; all per-allocator state must be reached via a process-global.
// All callbacks documented at
// https://jemalloc.net/jemalloc.3.html#arena.i.extent_hooks

extern "C" {

static void *ExtentAlloc(extent_hooks_t * /*hooks*/,
                         void * /*new_addr*/,
                         size_t size,
                         size_t alignment,
                         bool *zero,
                         bool *commit,
                         unsigned /*arena_ind*/) {
  void *addr = JemallocBackend::Get().AllocExtent(size, alignment);
  if (addr == nullptr) {
    return nullptr;
  }
  // ftruncate-backed mmap pages are zero-initialized by the kernel and
  // already committed (resident-on-first-touch).
  if (zero != nullptr) *zero = true;
  if (commit != nullptr) *commit = true;
  return addr;
}

static bool ExtentDalloc(extent_hooks_t * /*hooks*/,
                         void *addr,
                         size_t size,
                         bool /*committed*/,
                         unsigned /*arena_ind*/) {
  return JemallocBackend::Get().FreeExtent(addr, size);
}

static void ExtentDestroy(extent_hooks_t *hooks,
                          void *addr,
                          size_t size,
                          bool committed,
                          unsigned arena_ind) {
  (void)ExtentDalloc(hooks, addr, size, committed, arena_ind);
}

static bool ExtentCommit(extent_hooks_t * /*hooks*/,
                         void * /*addr*/,
                         size_t /*size*/,
                         size_t /*offset*/,
                         size_t /*length*/,
                         unsigned /*arena_ind*/) {
  // Already committed by mmap. Returning false signals success.
  return false;
}

static bool ExtentDecommit(extent_hooks_t * /*hooks*/,
                           void * /*addr*/,
                           size_t /*size*/,
                           size_t /*offset*/,
                           size_t /*length*/,
                           unsigned /*arena_ind*/) {
  // Refuse decommit; pages stay committed for the lifetime of the extent.
  return true;
}

static bool ExtentPurgeLazy(extent_hooks_t * /*hooks*/,
                            void * /*addr*/,
                            size_t /*size*/,
                            size_t /*offset*/,
                            size_t /*length*/,
                            unsigned /*arena_ind*/) {
  return true;  // refuse
}

static bool ExtentPurgeForced(extent_hooks_t * /*hooks*/,
                              void * /*addr*/,
                              size_t /*size*/,
                              size_t /*offset*/,
                              size_t /*length*/,
                              unsigned /*arena_ind*/) {
  return true;  // refuse
}

// CRITICAL: refuse split. Without this, jemalloc may divide one extent
// (one backing file) into two sub-extents, breaking the assumption that
// each live allocation maps to exactly one (fd, mmap_size) pair.
static bool ExtentSplit(extent_hooks_t * /*hooks*/,
                        void * /*addr*/,
                        size_t /*size*/,
                        size_t /*size_a*/,
                        size_t /*size_b*/,
                        bool /*committed*/,
                        unsigned /*arena_ind*/) {
  return true;  // refuse
}

// Symmetric: refuse merge. Two extents come from different backing files
// in our design.
static bool ExtentMerge(extent_hooks_t * /*hooks*/,
                        void * /*addr_a*/,
                        size_t /*size_a*/,
                        void * /*addr_b*/,
                        size_t /*size_b*/,
                        bool /*committed*/,
                        unsigned /*arena_ind*/) {
  return true;  // refuse
}

}  // extern "C"

static extent_hooks_t g_extent_hooks = {
    ExtentAlloc,
    ExtentDalloc,
    ExtentDestroy,
    ExtentCommit,
    ExtentDecommit,
    ExtentPurgeLazy,
    ExtentPurgeForced,
    ExtentSplit,
    ExtentMerge,
};

unsigned CreateCustomArena() {
  unsigned arena_ind = 0;
  size_t sz = sizeof(arena_ind);
  // Create a new arena with our custom hooks. mallctl name format:
  // "arenas.create" returns a new arena index when written with an
  // extent_hooks_t pointer.
  extent_hooks_t *hooks = &g_extent_hooks;
  int err = ray_je_mallctl("arenas.create",
                            static_cast<void *>(&arena_ind),
                            &sz,
                            static_cast<void *>(&hooks),
                            sizeof(hooks));
  RAY_CHECK_EQ(err, 0) << "jemalloc arenas.create failed: " << err;
  return arena_ind;
}

void DestroyCustomArena(unsigned arena_ind) {
  std::string ctl = "arena." + std::to_string(arena_ind) + ".destroy";
  int err = ray_je_mallctl(ctl.c_str(), nullptr, nullptr, nullptr, 0);
  if (err != 0) {
    RAY_LOG(WARNING) << "jemalloc " << ctl << " failed: " << err;
  }
}

}  // namespace

JemallocPlasmaAllocator::JemallocPlasmaAllocator(
    const std::string &plasma_directory,
    const std::string &fallback_directory,
    bool hugepage_enabled,
    int64_t footprint_limit)
    : footprint_limit_(footprint_limit),
      alignment_(64),
      arena_index_(0),
      allocated_(0),
      fallback_allocated_(0) {
  RAY_CHECK(!hugepage_enabled)
      << "Hugepages not yet supported by JemallocPlasmaAllocator spike";
  JemallocBackend::Get().Init(plasma_directory, fallback_directory, footprint_limit);
  arena_index_ = CreateCustomArena();
}

JemallocPlasmaAllocator::~JemallocPlasmaAllocator() {
  DestroyCustomArena(arena_index_);
}

std::optional<Allocation> JemallocPlasmaAllocator::Allocate(size_t bytes) {
  int flags = MALLOCX_ARENA(arena_index_) | MALLOCX_ALIGN(alignment_);
  void *mem = ray_je_mallocx(bytes, flags);
  if (mem == nullptr) {
    return std::nullopt;
  }
  allocated_ += static_cast<int64_t>(bytes);
  return BuildAllocation(mem, bytes, /*is_fallback_allocated=*/false);
}

std::optional<Allocation> JemallocPlasmaAllocator::FallbackAllocate(size_t bytes) {
  // For the spike, fallback uses the same arena as primary — the
  // architectural question is whether we can produce (fd, offset, mmap_size)
  // triples at all. Production code will route fallback through a separate
  // arena bound to the fallback directory.
  int flags = MALLOCX_ARENA(arena_index_) | MALLOCX_ALIGN(alignment_);
  void *mem = ray_je_mallocx(bytes, flags);
  if (mem == nullptr) {
    return std::nullopt;
  }
  allocated_ += static_cast<int64_t>(bytes);
  fallback_allocated_ += static_cast<int64_t>(bytes);
  return BuildAllocation(mem, bytes, /*is_fallback_allocated=*/true);
}

void JemallocPlasmaAllocator::Free(Allocation allocation) {
  RAY_CHECK(allocation.address_ != nullptr) << "Cannot free the nullptr";
  int flags = MALLOCX_ARENA(arena_index_);
  ray_je_dallocx(allocation.address_, flags);
  allocated_ -= allocation.size_;
  if (allocation.fallback_allocated_) {
    fallback_allocated_ -= allocation.size_;
  }
}

int64_t JemallocPlasmaAllocator::GetFootprintLimit() const {
  return footprint_limit_;
}

int64_t JemallocPlasmaAllocator::Allocated() const { return allocated_; }

int64_t JemallocPlasmaAllocator::FallbackAllocated() const {
  return fallback_allocated_;
}

std::optional<Allocation> JemallocPlasmaAllocator::BuildAllocation(
    void *addr, size_t size, bool is_fallback_allocated) {
  if (addr == nullptr) {
    return std::nullopt;
  }
  int fd_raw = -1;
  int64_t unique_id = 0;
  int64_t mmap_size = 0;
  ptrdiff_t offset = 0;
  if (!JemallocBackend::Get().LookupMapinfo(
          addr, &fd_raw, &unique_id, &mmap_size, &offset)) {
    return std::nullopt;
  }
  MEMFD_TYPE fd{fd_raw, unique_id};
  return Allocation(addr,
                    static_cast<int64_t>(size),
                    std::move(fd),
                    offset,
                    0 /* device_number */,
                    mmap_size,
                    is_fallback_allocated);
}

}  // namespace plasma
