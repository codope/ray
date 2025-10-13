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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ray/object_manager/plasma/allocator.h"
#include "ray/object_manager/plasma/plasma_allocator.h"

#ifdef JEMALLOC_AVAILABLE
#include "ray/object_manager/plasma/jemalloc_allocator.h"
#endif

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

namespace plasma {

// Test configuration constants
constexpr int64_t kKB = 1024;
constexpr int64_t kMB = 1024 * kKB;
constexpr int kDefaultIterations = 1000;

// Object size categories for testing
const std::vector<size_t> kObjectSizes = {
    100 * kKB,  // Small objects - 100KB
    1 * kMB,    // Medium objects - 1MB
    10 * kMB,   // Large objects - 10MB
    100 * kMB   // Very large objects - 100MB
};

// Memory pressure levels for testing
const std::vector<int64_t> kMemoryLimits = {
    256 * kMB,   // Low memory - 256MB
    1024 * kMB,  // Medium memory - 1GB
    4096 * kMB   // High memory - 4GB
};

struct BenchmarkResult {
  std::string test_name;
  std::string allocator_type;
  size_t object_size;
  int64_t memory_limit;

  // Performance metrics
  double alloc_ops_per_sec;
  double free_ops_per_sec;
  double avg_alloc_latency_us;
  double avg_free_latency_us;
  double p95_alloc_latency_us;
  double p95_free_latency_us;

  // Fragmentation metrics
  double fragmentation_ratio;
  double memory_utilization;
  int64_t wasted_bytes;
  int64_t largest_free_block;
  int64_t active_allocations;
  int64_t fallback_allocations;

  // Test-specific metrics
  int successful_allocations;
  int failed_allocations;
  double test_duration_sec;
};

// Utility class for creating temporary directories
class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("plasma_bench_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(path_);
  }

  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  std::string string() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

// Base class for fragmentation benchmarks
class FragmentationBenchmarkBase : public ::testing::Test {
 public:
  virtual ~FragmentationBenchmarkBase() = default;

 protected:
  void SetUp() override {
    plasma_dir_ = std::make_unique<TempDir>();
    fallback_dir_ = std::make_unique<TempDir>();

    // Initialize random number generator
    rng_.seed(12345);  // Fixed seed for reproducible results
  }

  void TearDown() override {
    plasma_dir_.reset();
    fallback_dir_.reset();
  }

  // Create allocator instance
  std::unique_ptr<IAllocator> CreatePlasmaAllocator(int64_t memory_limit) {
    return std::make_unique<PlasmaAllocator>(plasma_dir_->string(),
                                             fallback_dir_->string(),
                                             false,  // hugepages_enabled
                                             memory_limit);
  }

  std::unique_ptr<IAllocator> CreateJemallocAllocator(int64_t memory_limit) {
#ifdef JEMALLOC_AVAILABLE
    return std::make_unique<JemallocAllocator>(plasma_dir_->string(),
                                               fallback_dir_->string(),
                                               false,  // hugepages_enabled
                                               memory_limit);
#else
    // Return nullptr when jemalloc is not available
    return nullptr;
#endif
  }

  // Performance measurement utilities
  template <typename Func>
  double MeasureLatency(Func &&func, int iterations = 100) {
    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
      auto start = high_resolution_clock::now();
      func();
      auto end = high_resolution_clock::now();

      auto duration = duration_cast<nanoseconds>(end - start).count();
      latencies.push_back(duration / 1000.0);  // Convert to microseconds
    }

    return std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  }

  template <typename Func>
  std::vector<double> MeasureLatencies(Func &&func, int iterations) {
    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
      auto start = high_resolution_clock::now();
      func();
      auto end = high_resolution_clock::now();

      auto duration = duration_cast<nanoseconds>(end - start).count();
      latencies.push_back(duration / 1000.0);  // Convert to microseconds
    }

    return latencies;
  }

  double CalculatePercentile(std::vector<double> &values, double percentile) {
    if (values.empty()) return 0.0;

    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(values.size() * percentile / 100.0);
    index = std::min(index, values.size() - 1);
    return values[index];
  }

  // Calculate fragmentation metrics
  double CalculateFragmentationRatio(IAllocator &allocator,
                                     const std::vector<Allocation> &allocations) {
    int64_t total_allocated = allocator.Allocated();
    int64_t footprint_limit = allocator.GetFootprintLimit();

    if (footprint_limit == 0 || total_allocated == 0) return 0.0;

    // Estimate largest free block by attempting a large allocation
    int64_t remaining = footprint_limit - total_allocated;
    if (remaining <= 0) return 1.0;

    // Try to allocate largest possible block to find fragmentation.
    // Clamp to >= 100KB to avoid tiny probes that aren't supported by some allocators.
    const size_t kMinProbe = 100 * kKB;
    size_t test_size = remaining;
    while (test_size > kMinProbe) {
      auto test_alloc = allocator.Allocate(test_size);
      if (test_alloc.has_value()) {
        allocator.Free(std::move(test_alloc.value()));
        break;
      }
      test_size /= 2;
    }

    if (test_size <= kMinProbe) return 1.0;  // Highly fragmented or below probe threshold

    return 1.0 - static_cast<double>(test_size) / remaining;
  }

  // Generate benchmark report
  BenchmarkResult CreateResult(const std::string &test_name,
                               const std::string &allocator_type,
                               size_t object_size,
                               int64_t memory_limit) {
    BenchmarkResult result{};  // Value-initialize all members to zero
    result.test_name = test_name;
    result.allocator_type = allocator_type;
    result.object_size = object_size;
    result.memory_limit = memory_limit;
    return result;
  }

  void PrintResult(const BenchmarkResult &result) {
    std::cout << "=== " << result.test_name << " [" << result.allocator_type
              << "] ===" << std::endl;
    std::cout << "Object Size: " << (result.object_size / kKB) << " KB" << std::endl;
    std::cout << "Memory Limit: " << (result.memory_limit / kMB) << " MB" << std::endl;
    std::cout << "Allocation Rate: " << result.alloc_ops_per_sec << " ops/sec"
              << std::endl;
    std::cout << "Free Rate: " << result.free_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "Avg Alloc Latency: " << result.avg_alloc_latency_us << " μs"
              << std::endl;
    std::cout << "P95 Alloc Latency: " << result.p95_alloc_latency_us << " μs"
              << std::endl;
    std::cout << "Fragmentation Ratio: " << (result.fragmentation_ratio * 100) << "%"
              << std::endl;
    std::cout << "Memory Utilization: " << (result.memory_utilization * 100) << "%"
              << std::endl;
    std::cout << "Successful/Failed Allocs: " << result.successful_allocations << "/"
              << result.failed_allocations << std::endl;
    std::cout << "Fallback Allocations: " << result.fallback_allocations << std::endl;
    std::cout << "Test Duration: " << result.test_duration_sec << " sec" << std::endl;
    std::cout << std::endl;
  }

 protected:
  std::unique_ptr<TempDir> plasma_dir_;
  std::unique_ptr<TempDir> fallback_dir_;
  std::mt19937 rng_;
};

// Test case: Sequential allocation benchmark
class SequentialAllocationBenchmark : public FragmentationBenchmarkBase {
 public:
  BenchmarkResult RunBenchmark(std::unique_ptr<IAllocator> allocator,
                               const std::string &allocator_type,
                               size_t object_size,
                               int64_t memory_limit,
                               int iterations = kDefaultIterations) {
    auto result =
        CreateResult("Sequential_Allocation", allocator_type, object_size, memory_limit);

    std::vector<Allocation> allocations;
    std::vector<double> alloc_latencies;
    std::vector<double> free_latencies;

    auto start_time = high_resolution_clock::now();

    // Phase 1: Sequential allocations
    for (int i = 0; i < iterations; ++i) {
      auto alloc_start = high_resolution_clock::now();
      auto allocation = allocator->Allocate(object_size);
      auto alloc_end = high_resolution_clock::now();

      auto latency = duration_cast<nanoseconds>(alloc_end - alloc_start).count() / 1000.0;
      alloc_latencies.push_back(latency);

      if (allocation.has_value()) {
        allocations.push_back(std::move(allocation.value()));
        result.successful_allocations++;
      } else {
        result.failed_allocations++;
        break;  // Stop on first failure to measure realistic capacity
      }
    }

    // Measure fragmentation after allocations
    result.fragmentation_ratio = CalculateFragmentationRatio(*allocator, allocations);
    result.memory_utilization =
        static_cast<double>(allocator->Allocated()) / memory_limit;
    result.active_allocations = allocations.size();

    // Phase 2: Sequential frees
    for (auto &allocation : allocations) {
      auto free_start = high_resolution_clock::now();
      allocator->Free(std::move(allocation));
      auto free_end = high_resolution_clock::now();

      auto latency = duration_cast<nanoseconds>(free_end - free_start).count() / 1000.0;
      free_latencies.push_back(latency);
    }

    auto end_time = high_resolution_clock::now();
    result.test_duration_sec =
        duration_cast<microseconds>(end_time - start_time).count() / 1e6;

    // Calculate performance metrics
    if (!alloc_latencies.empty()) {
      result.avg_alloc_latency_us =
          std::accumulate(alloc_latencies.begin(), alloc_latencies.end(), 0.0) /
          alloc_latencies.size();
      result.p95_alloc_latency_us = CalculatePercentile(alloc_latencies, 95.0);
      result.alloc_ops_per_sec = 1e6 / result.avg_alloc_latency_us;
    }

    if (!free_latencies.empty()) {
      result.avg_free_latency_us =
          std::accumulate(free_latencies.begin(), free_latencies.end(), 0.0) /
          free_latencies.size();
      result.p95_free_latency_us = CalculatePercentile(free_latencies, 95.0);
      result.free_ops_per_sec = 1e6 / result.avg_free_latency_us;
    }

    result.fallback_allocations = allocator->FallbackAllocated() / object_size;

    return result;
  }
};

// Test case: Mixed allocation/deallocation benchmark
class MixedAllocationBenchmark : public FragmentationBenchmarkBase {
 public:
  BenchmarkResult RunBenchmark(std::unique_ptr<IAllocator> allocator,
                               const std::string &allocator_type,
                               size_t object_size,
                               int64_t memory_limit,
                               int iterations = kDefaultIterations) {
    auto result =
        CreateResult("Mixed_Allocation", allocator_type, object_size, memory_limit);

    std::vector<Allocation> active_allocations;
    std::vector<double> alloc_latencies;
    std::vector<double> free_latencies;

    // Use various object sizes for more realistic workload
    std::vector<size_t> sizes = {
        object_size, object_size / 2, object_size * 2, object_size / 4};
    std::uniform_int_distribution<> size_dist(0, sizes.size() - 1);
    std::uniform_real_distribution<> action_dist(0.0, 1.0);

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
      // Decide whether to allocate or free (bias towards allocation early on)
      double alloc_probability =
          active_allocations.empty()
              ? 1.0
              : std::max(0.3,
                         1.0 - static_cast<double>(active_allocations.size()) / 100.0);

      if (action_dist(rng_) < alloc_probability) {
        // Allocate
        size_t size = sizes[size_dist(rng_)];

        auto alloc_start = high_resolution_clock::now();
        auto allocation = allocator->Allocate(size);
        auto alloc_end = high_resolution_clock::now();

        auto latency =
            duration_cast<nanoseconds>(alloc_end - alloc_start).count() / 1000.0;
        alloc_latencies.push_back(latency);

        if (allocation.has_value()) {
          active_allocations.push_back(std::move(allocation.value()));
          result.successful_allocations++;
        } else {
          result.failed_allocations++;
        }
      } else if (!active_allocations.empty()) {
        // Free random allocation
        std::uniform_int_distribution<> index_dist(0, active_allocations.size() - 1);
        int index = index_dist(rng_);

        auto free_start = high_resolution_clock::now();
        allocator->Free(std::move(active_allocations[index]));
        auto free_end = high_resolution_clock::now();

        auto latency = duration_cast<nanoseconds>(free_end - free_start).count() / 1000.0;
        free_latencies.push_back(latency);

        active_allocations.erase(active_allocations.begin() + index);
      }
    }

    // Measure fragmentation during mixed workload
    result.fragmentation_ratio =
        CalculateFragmentationRatio(*allocator, active_allocations);
    result.memory_utilization =
        static_cast<double>(allocator->Allocated()) / memory_limit;
    result.active_allocations = active_allocations.size();

    // Clean up remaining allocations
    for (auto &allocation : active_allocations) {
      allocator->Free(std::move(allocation));
    }

    auto end_time = high_resolution_clock::now();
    result.test_duration_sec =
        duration_cast<microseconds>(end_time - start_time).count() / 1e6;

    // Calculate performance metrics
    if (!alloc_latencies.empty()) {
      result.avg_alloc_latency_us =
          std::accumulate(alloc_latencies.begin(), alloc_latencies.end(), 0.0) /
          alloc_latencies.size();
      result.p95_alloc_latency_us = CalculatePercentile(alloc_latencies, 95.0);
      result.alloc_ops_per_sec = 1e6 / result.avg_alloc_latency_us;
    }

    if (!free_latencies.empty()) {
      result.avg_free_latency_us =
          std::accumulate(free_latencies.begin(), free_latencies.end(), 0.0) /
          free_latencies.size();
      result.p95_free_latency_us = CalculatePercentile(free_latencies, 95.0);
      result.free_ops_per_sec = 1e6 / result.avg_free_latency_us;
    }

    result.fallback_allocations = allocator->FallbackAllocated() / object_size;

    return result;
  }
};

// Test case: Pathological fragmentation patterns
class PathologicalFragmentationBenchmark : public FragmentationBenchmarkBase {
 public:
  BenchmarkResult RunCheckerboardPattern(std::unique_ptr<IAllocator> allocator,
                                         const std::string &allocator_type,
                                         size_t object_size,
                                         int64_t memory_limit) {
    auto result =
        CreateResult("Checkerboard_Pattern", allocator_type, object_size, memory_limit);

    std::vector<Allocation> allocations;
    std::vector<double> alloc_latencies;

    auto start_time = high_resolution_clock::now();

    // Phase 1: Allocate many objects
    int max_objects =
        static_cast<int>(memory_limit / (object_size * 2));  // Leave room for gaps
    for (int i = 0; i < max_objects && allocations.size() < 100; ++i) {
      auto alloc_start = high_resolution_clock::now();
      auto allocation = allocator->Allocate(object_size);
      auto alloc_end = high_resolution_clock::now();

      auto latency = duration_cast<nanoseconds>(alloc_end - alloc_start).count() / 1000.0;
      alloc_latencies.push_back(latency);

      if (allocation.has_value()) {
        allocations.push_back(std::move(allocation.value()));
        result.successful_allocations++;
      } else {
        result.failed_allocations++;
        break;
      }
    }

    // Phase 2: Free every other allocation to create checkerboard pattern
    for (size_t i = 1; i < allocations.size(); i += 2) {
      if (i < allocations.size()) {
        allocator->Free(std::move(allocations[i]));
      }
    }

    // Remove freed allocations from vector (keep only even indices)
    std::vector<Allocation> remaining_allocations;
    for (size_t i = 0; i < allocations.size(); i += 2) {
      remaining_allocations.push_back(std::move(allocations[i]));
    }
    allocations = std::move(remaining_allocations);

    // Phase 3: Try to allocate larger objects in the gaps
    int large_alloc_attempts = 10;
    size_t large_object_size = object_size * 2;  // Double size

    for (int i = 0; i < large_alloc_attempts; ++i) {
      auto large_allocation = allocator->Allocate(large_object_size);
      if (large_allocation.has_value()) {
        allocations.push_back(std::move(large_allocation.value()));
        result.successful_allocations++;
      } else {
        result.failed_allocations++;
      }
    }

    // Measure fragmentation in checkerboard pattern
    result.fragmentation_ratio = CalculateFragmentationRatio(*allocator, allocations);
    result.memory_utilization =
        static_cast<double>(allocator->Allocated()) / memory_limit;
    result.active_allocations = allocations.size();

    // Cleanup
    for (auto &allocation : allocations) {
      allocator->Free(std::move(allocation));
    }

    auto end_time = high_resolution_clock::now();
    result.test_duration_sec =
        duration_cast<microseconds>(end_time - start_time).count() / 1e6;

    // Calculate metrics
    if (!alloc_latencies.empty()) {
      result.avg_alloc_latency_us =
          std::accumulate(alloc_latencies.begin(), alloc_latencies.end(), 0.0) /
          alloc_latencies.size();
      result.p95_alloc_latency_us = CalculatePercentile(alloc_latencies, 95.0);
      result.alloc_ops_per_sec = 1e6 / result.avg_alloc_latency_us;
    }

    result.fallback_allocations = allocator->FallbackAllocated() / object_size;

    return result;
  }

  BenchmarkResult RunBurstAllocation(std::unique_ptr<IAllocator> allocator,
                                     const std::string &allocator_type,
                                     size_t object_size,
                                     int64_t memory_limit) {
    auto result =
        CreateResult("Burst_Allocation", allocator_type, object_size, memory_limit);

    std::vector<Allocation> allocations;
    std::vector<double> alloc_latencies;

    auto start_time = high_resolution_clock::now();

    // Burst allocation - allocate as fast as possible
    int max_objects = static_cast<int>(memory_limit / object_size);
    for (int i = 0; i < max_objects && i < 1000; ++i) {  // Limit to 1000 for performance
      auto alloc_start = high_resolution_clock::now();
      auto allocation = allocator->Allocate(object_size);
      auto alloc_end = high_resolution_clock::now();

      auto latency = duration_cast<nanoseconds>(alloc_end - alloc_start).count() / 1000.0;
      alloc_latencies.push_back(latency);

      if (allocation.has_value()) {
        allocations.push_back(std::move(allocation.value()));
        result.successful_allocations++;
      } else {
        result.failed_allocations++;
        break;
      }
    }

    // Measure fragmentation after burst
    result.fragmentation_ratio = CalculateFragmentationRatio(*allocator, allocations);
    result.memory_utilization =
        static_cast<double>(allocator->Allocated()) / memory_limit;
    result.active_allocations = allocations.size();

    // Free in reverse order (worst case for some allocators)
    std::reverse(allocations.begin(), allocations.end());
    for (auto &allocation : allocations) {
      allocator->Free(std::move(allocation));
    }

    auto end_time = high_resolution_clock::now();
    result.test_duration_sec =
        duration_cast<microseconds>(end_time - start_time).count() / 1e6;

    // Calculate metrics
    if (!alloc_latencies.empty()) {
      result.avg_alloc_latency_us =
          std::accumulate(alloc_latencies.begin(), alloc_latencies.end(), 0.0) /
          alloc_latencies.size();
      result.p95_alloc_latency_us = CalculatePercentile(alloc_latencies, 95.0);
      result.alloc_ops_per_sec = 1e6 / result.avg_alloc_latency_us;
    }

    result.fallback_allocations = allocator->FallbackAllocated() / object_size;

    return result;
  }
};

// Main test cases
#if 0
// Disabled: small-object benchmark (< 100KB) is not representative for plasma/zero-copy path.
TEST_F(SequentialAllocationBenchmark, CompareAllocators_SmallObjects) {}
#endif

TEST_F(SequentialAllocationBenchmark, CompareAllocators_LargeObjects) {
  size_t object_size = 10 * kMB;
  int64_t memory_limit = 256 * kMB;  // Reduce memory limit for tmpfs

  std::cout << "\n=== Sequential Allocation Benchmark - Large Objects ===\n" << std::endl;

  // Test PlasmaAllocator (dlmalloc)
  auto plasma_allocator = CreatePlasmaAllocator(memory_limit);
  auto plasma_result = RunBenchmark(std::move(plasma_allocator),
                                    "dlmalloc",
                                    object_size,
                                    memory_limit,
                                    100);  // Fewer iterations for large objects
  PrintResult(plasma_result);

  // Test JemallocAllocator if available
  auto jemalloc_allocator = CreateJemallocAllocator(memory_limit);
  if (jemalloc_allocator) {
    auto jemalloc_result = RunBenchmark(
        std::move(jemalloc_allocator), "jemalloc", object_size, memory_limit, 100);
    PrintResult(jemalloc_result);

    // Compare results
    std::cout << "=== COMPARISON ===" << std::endl;
    std::cout << "Fragmentation Reduction: "
              << ((plasma_result.fragmentation_ratio -
                   jemalloc_result.fragmentation_ratio) *
                  100)
              << "% (negative = jemalloc better)" << std::endl;
    std::cout << "Allocation Speed Improvement: "
              << ((jemalloc_result.alloc_ops_per_sec / plasma_result.alloc_ops_per_sec -
                   1.0) *
                  100)
              << "%" << std::endl;
  } else {
    std::cout << "=== JEMALLOC NOT AVAILABLE ===" << std::endl;
    std::cout << "Only testing dlmalloc performance." << std::endl;
  }
}

TEST_F(MixedAllocationBenchmark, CompareAllocators_MixedWorkload) {
  size_t object_size = 1 * kMB;
  int64_t memory_limit = 256 * kMB;

  std::cout << "\n=== Mixed Allocation Benchmark ===\n" << std::endl;

  // Test PlasmaAllocator (dlmalloc)
  auto plasma_allocator = CreatePlasmaAllocator(memory_limit);
  auto plasma_result = RunBenchmark(
      std::move(plasma_allocator), "dlmalloc", object_size, memory_limit, 2000);
  PrintResult(plasma_result);

  // Test JemallocAllocator if available
  auto jemalloc_allocator = CreateJemallocAllocator(memory_limit);
  if (jemalloc_allocator) {
    auto jemalloc_result = RunBenchmark(
        std::move(jemalloc_allocator), "jemalloc", object_size, memory_limit, 2000);
    PrintResult(jemalloc_result);

    // Compare results
    std::cout << "=== COMPARISON ===" << std::endl;
    std::cout << "Fragmentation Reduction: "
              << ((plasma_result.fragmentation_ratio -
                   jemalloc_result.fragmentation_ratio) *
                  100)
              << "% (negative = jemalloc better)" << std::endl;
    std::cout << "Mixed Allocation Speed Improvement: "
              << ((jemalloc_result.alloc_ops_per_sec / plasma_result.alloc_ops_per_sec -
                   1.0) *
                  100)
              << "%" << std::endl;
  } else {
    std::cout << "=== JEMALLOC NOT AVAILABLE ===" << std::endl;
    std::cout << "Only testing dlmalloc performance." << std::endl;
  }
}

TEST_F(PathologicalFragmentationBenchmark, CompareAllocators_CheckerboardPattern) {
  size_t object_size = 1 * kMB;
  // Use a smaller limit to fit typical /dev/shm on CI/devboxes
  int64_t memory_limit = 256 * kMB;

  std::cout << "\n=== Checkerboard Pattern Benchmark ===\n" << std::endl;

  // Test PlasmaAllocator (dlmalloc)
  auto plasma_allocator = CreatePlasmaAllocator(memory_limit);
  auto plasma_result = RunCheckerboardPattern(
      std::move(plasma_allocator), "dlmalloc", object_size, memory_limit);
  PrintResult(plasma_result);

  // Test JemallocAllocator if available
  auto jemalloc_allocator = CreateJemallocAllocator(memory_limit);
  if (jemalloc_allocator) {
    auto jemalloc_result = RunCheckerboardPattern(
        std::move(jemalloc_allocator), "jemalloc", object_size, memory_limit);
    PrintResult(jemalloc_result);

    // Compare results (this should show the most dramatic difference)
    std::cout << "=== COMPARISON ===" << std::endl;
    std::cout << "Fragmentation Reduction: "
              << ((plasma_result.fragmentation_ratio -
                   jemalloc_result.fragmentation_ratio) *
                  100)
              << "% (negative = jemalloc better)" << std::endl;
    std::cout << "Successful Large Alloc Difference: "
              << (jemalloc_result.successful_allocations -
                  plasma_result.successful_allocations)
              << " (positive = jemalloc better)" << std::endl;
  } else {
    std::cout << "=== JEMALLOC NOT AVAILABLE ===" << std::endl;
    std::cout << "Only testing dlmalloc performance." << std::endl;
  }
}

TEST_F(PathologicalFragmentationBenchmark, CompareAllocators_BurstPattern) {
  size_t object_size = 1 * kMB;
  int64_t memory_limit = 256 * kMB;

  std::cout << "\n=== Burst Allocation Benchmark ===\n" << std::endl;

  // Test PlasmaAllocator (dlmalloc)
  auto plasma_allocator = CreatePlasmaAllocator(memory_limit);
  auto plasma_result = RunBurstAllocation(
      std::move(plasma_allocator), "dlmalloc", object_size, memory_limit);
  PrintResult(plasma_result);

  // Test JemallocAllocator if available
  auto jemalloc_allocator = CreateJemallocAllocator(memory_limit);
  if (jemalloc_allocator) {
    auto jemalloc_result = RunBurstAllocation(
        std::move(jemalloc_allocator), "jemalloc", object_size, memory_limit);
    PrintResult(jemalloc_result);

    // Compare results
    std::cout << "=== COMPARISON ===" << std::endl;
    std::cout << "Burst Allocation Speed Improvement: "
              << ((jemalloc_result.alloc_ops_per_sec / plasma_result.alloc_ops_per_sec -
                   1.0) *
                  100)
              << "%" << std::endl;
    std::cout
        << "Memory Utilization Improvement: "
        << ((jemalloc_result.memory_utilization - plasma_result.memory_utilization) * 100)
        << "% (positive = jemalloc better)" << std::endl;
  } else {
    std::cout << "=== JEMALLOC NOT AVAILABLE ===" << std::endl;
    std::cout << "Only testing dlmalloc performance." << std::endl;
  }
}

}  // namespace plasma

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "\n======================================" << std::endl;
  std::cout << "Plasma Allocator Fragmentation Benchmark" << std::endl;
  std::cout << "======================================\n" << std::endl;

  return RUN_ALL_TESTS();
}
