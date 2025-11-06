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

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

#include "ray/core_worker/shutdown_coordinator.h"

namespace ray {
namespace core {

// Mock executor for testing
class MockShutdownExecutor : public ShutdownExecutorInterface {
 public:
  void ExecuteGracefulShutdown(std::string_view exit_type,
                               std::string_view detail,
                               std::chrono::milliseconds timeout_ms) override {
    // Simulate some async work
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  void ExecuteForceShutdown(std::string_view exit_type,
                            std::string_view detail) override {
    // Quick exit
  }

  void ExecuteExit(std::string_view exit_type,
                   std::string_view detail,
                   std::chrono::milliseconds timeout_ms,
                   const std::shared_ptr<LocalMemoryBuffer>
                       &creation_task_exception_pb_bytes) override {
    // Simulate async shutdown work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void ExecuteExitIfIdle(std::string_view exit_type,
                         std::string_view detail,
                         std::chrono::milliseconds timeout_ms) override {}

  void KillChildProcessesImmediately() override {}

  bool ShouldWorkerIdleExit() const override { return false; }
};

// Test that WaitForShutdownComplete blocks until shutdown completes
TEST(ShutdownCoordinatorTest, WaitForShutdownComplete) {
  auto executor = std::make_unique<MockShutdownExecutor>();
  ShutdownCoordinator coordinator(std::move(executor), rpc::WorkerType::DRIVER);

  // Initial state should be running
  ASSERT_TRUE(coordinator.IsRunning());
  ASSERT_FALSE(coordinator.IsShutdown());

  // Start shutdown in a background thread
  std::thread shutdown_thread([&coordinator]() {
    coordinator.RequestShutdown(/*force_shutdown=*/false,
                                ShutdownReason::kGracefulExit,
                                "Test shutdown");
  });

  // Give shutdown a moment to start
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Shutdown should be in progress
  ASSERT_TRUE(coordinator.IsShuttingDown());

  // Wait for completion
  auto start = std::chrono::steady_clock::now();
  coordinator.WaitForShutdownComplete();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();

  // Should have waited for the async work to complete
  ASSERT_GE(elapsed, 40);  // Allow some slack

  // Should now be shutdown
  ASSERT_TRUE(coordinator.IsShutdown());

  shutdown_thread.join();
}

// Test that WaitForShutdownComplete returns immediately if never initiated
TEST(ShutdownCoordinatorTest, WaitForShutdownCompleteNeverInitiated) {
  auto executor = std::make_unique<MockShutdownExecutor>();
  ShutdownCoordinator coordinator(std::move(executor), rpc::WorkerType::DRIVER);

  // Should be running
  ASSERT_TRUE(coordinator.IsRunning());

  // Wait should return immediately without blocking
  auto start = std::chrono::steady_clock::now();
  coordinator.WaitForShutdownComplete();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();

  // Should return almost immediately
  ASSERT_LT(elapsed, 10);

  // Should still be running
  ASSERT_TRUE(coordinator.IsRunning());
}

// Test that WaitForShutdownComplete returns immediately if already complete
TEST(ShutdownCoordinatorTest, WaitForShutdownCompleteAlreadyComplete) {
  auto executor = std::make_unique<MockShutdownExecutor>();
  ShutdownCoordinator coordinator(std::move(executor), rpc::WorkerType::DRIVER);

  // Initiate and complete shutdown
  coordinator.RequestShutdown(/*force_shutdown=*/false,
                              ShutdownReason::kGracefulExit,
                              "Test shutdown");

  // Should be shutdown
  ASSERT_TRUE(coordinator.IsShutdown());

  // Wait should return immediately
  auto start = std::chrono::steady_clock::now();
  coordinator.WaitForShutdownComplete();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();

  // Should return almost immediately
  ASSERT_LT(elapsed, 10);
}

// Test multiple waiters
TEST(ShutdownCoordinatorTest, MultipleWaiters) {
  auto executor = std::make_unique<MockShutdownExecutor>();
  auto coordinator = std::make_shared<ShutdownCoordinator>(std::move(executor),
                                                           rpc::WorkerType::DRIVER);

  std::atomic<int> completed_waiters{0};

  // Start multiple waiter threads
  std::vector<std::thread> waiter_threads;
  for (int i = 0; i < 5; i++) {
    waiter_threads.emplace_back([coordinator, &completed_waiters]() {
      coordinator->WaitForShutdownComplete();
      completed_waiters++;
    });
  }

  // Give waiters time to start
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // No waiters should have completed yet
  ASSERT_EQ(completed_waiters.load(), 0);

  // Initiate shutdown
  coordinator->RequestShutdown(/*force_shutdown=*/false,
                               ShutdownReason::kGracefulExit,
                               "Test shutdown");

  // All waiters should complete
  for (auto &thread : waiter_threads) {
    thread.join();
  }

  ASSERT_EQ(completed_waiters.load(), 5);
  ASSERT_TRUE(coordinator->IsShutdown());
}

}  // namespace core
}  // namespace ray

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

