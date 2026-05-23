// Copyright 2017 The Ray Authors.
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
//
// Spike validation (Phase 1.6) — minimum-viable test that
// JemallocPlasmaAllocator satisfies the IAllocator contract and produces
// valid (fd, offset, mmap_size) triples consumable by Plasma's mmap-by-fd
// IPC model.
//
// One test per binary to side-step the process-singleton constraint that
// both PlasmaAllocator (dlmalloc) and JemallocPlasmaAllocator inherit
// from the "can only be created once per process" model.

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ray/common/id.h"
#include "ray/object_manager/plasma/jemalloc_plasma_allocator.h"

namespace plasma {
namespace {
const int64_t kMB = 1024 * 1024;

std::string CreateTestDir() {
  auto directory = std::filesystem::temp_directory_path() /
                   ray::UniqueID::FromRandom().Hex();
  std::filesystem::create_directories(directory);
  return directory.string();
}
}  // namespace

TEST(JemallocPlasmaAllocatorTest, BasicAllocateFreeAndMapinfo) {
  auto plasma_dir = CreateTestDir();
  auto fallback_dir = CreateTestDir();
  const int64_t kLimit = 8 * kMB;
  const int64_t kObjectSize = 900 * 1024;

  JemallocPlasmaAllocator allocator(
      plasma_dir, fallback_dir, /*hugepage_enabled=*/false, kLimit);

  EXPECT_EQ(kLimit, allocator.GetFootprintLimit());

  auto a1 = allocator.Allocate(kObjectSize);
  ASSERT_TRUE(a1.has_value());
  EXPECT_FALSE(a1->fallback_allocated_);
  EXPECT_EQ(kObjectSize, allocator.Allocated());
  // Every allocation must carry a valid mmap triple (the IPC contract).
  EXPECT_GE(a1->fd_.first, 0);
  EXPECT_GT(a1->mmap_size_, 0);
  EXPECT_GE(a1->offset_, 0);
  EXPECT_LE(a1->offset_ + a1->size_, a1->mmap_size_);

  auto a2 = allocator.Allocate(kObjectSize);
  ASSERT_TRUE(a2.has_value());
  EXPECT_EQ(2 * kObjectSize, allocator.Allocated());

  allocator.Free(std::move(a1.value()));
  EXPECT_EQ(kObjectSize, allocator.Allocated());

  allocator.Free(std::move(a2.value()));
  EXPECT_EQ(0, allocator.Allocated());
}

}  // namespace plasma
