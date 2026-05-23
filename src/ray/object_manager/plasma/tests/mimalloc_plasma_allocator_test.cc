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
// MimallocPlasmaAllocator satisfies the IAllocator contract and produces
// valid (fd, offset, mmap_size) triples consumable by Plasma's mmap-by-fd
// IPC model. Because all allocations are carved from one pre-mmapped
// backing file, every Allocation should carry the same fd but different
// offsets.

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ray/common/id.h"
#include "ray/object_manager/plasma/mimalloc_plasma_allocator.h"

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

TEST(MimallocPlasmaAllocatorTest, BasicAllocateFreeAndMapinfo) {
  auto plasma_dir = CreateTestDir();
  auto fallback_dir = CreateTestDir();
  const int64_t kLimit = 16 * kMB;
  const int64_t kObjectSize = 900 * 1024;

  MimallocPlasmaAllocator allocator(
      plasma_dir, fallback_dir, /*hugepage_enabled=*/false, kLimit);

  EXPECT_EQ(kLimit, allocator.GetFootprintLimit());

  auto a1 = allocator.Allocate(kObjectSize);
  ASSERT_TRUE(a1.has_value());
  EXPECT_FALSE(a1->fallback_allocated_);
  EXPECT_GE(a1->fd_.first, 0);
  EXPECT_EQ(kLimit, a1->mmap_size_) << "single backing file should equal limit";
  EXPECT_GE(a1->offset_, 0);
  EXPECT_LE(a1->offset_ + a1->size_, a1->mmap_size_);

  auto a2 = allocator.Allocate(kObjectSize);
  ASSERT_TRUE(a2.has_value());
  // Both allocations live in the same backing file.
  EXPECT_EQ(a1->fd_.first, a2->fd_.first);
  EXPECT_EQ(a1->fd_.second, a2->fd_.second);
  EXPECT_NE(a1->offset_, a2->offset_);
  EXPECT_EQ(2 * kObjectSize, allocator.Allocated());

  allocator.Free(std::move(a1.value()));
  EXPECT_EQ(kObjectSize, allocator.Allocated());

  allocator.Free(std::move(a2.value()));
  EXPECT_EQ(0, allocator.Allocated());
}

}  // namespace plasma
