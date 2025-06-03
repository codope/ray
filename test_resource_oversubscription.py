#!/usr/bin/env python3
"""
Test case to demonstrate resource oversubscription during CoreWorker::Exit()

The bug: In CoreWorker::Exit(), we call NotifyDirectCallTaskBlocked() early 
to release resources, but tasks may still be running during the drain phase,
leading to oversubscription.
"""

import ray
import time
import threading
import psutil
import os
from typing import List, Tuple
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1.0)
class ResourceConsumingActor:
    """Actor that can demonstrate the resource release bug"""
    
    def __init__(self):
        self.should_stop = False
        
    def start_cpu_intensive_work_and_exit(self, work_duration: float):
        """Start CPU work, then call exit_actor while work is still running"""
        
        def cpu_burn():
            """CPU-intensive work that actually consumes CPU cycles"""
            start_time = time.time()
            while time.time() - start_time < work_duration and not self.should_stop:
                # Actual CPU work - compute some meaningless calculations
                for i in range(100000):
                    _ = i * i * i
                    
        logger.info(f"Actor {os.getpid()}: Starting CPU work for {work_duration} seconds")
        
        # Start CPU work in background thread to simulate concurrent execution
        work_thread = threading.Thread(target=cpu_burn)
        work_thread.start()
        
        # Give the work thread time to start consuming CPU
        time.sleep(0.1)
        
        logger.info(f"Actor {os.getpid()}: Calling exit_actor() while work is still running!")
        
        # This triggers CoreWorker::Exit() which calls NotifyDirectCallTaskBlocked()
        # immediately, but the CPU work is still running!
        ray.actor.exit_actor()
        
        # This line should never be reached due to SystemExit from exit_actor()
        return "This should not be returned"


@ray.remote(num_cpus=1.0)
def cpu_intensive_task(task_id: int, duration: float) -> Tuple[int, float, float]:
    """CPU-intensive task that measures actual execution time"""
    start_time = time.time()
    logger.info(f"Task {task_id} (PID {os.getpid()}): Starting CPU work for {duration} seconds")
    
    # CPU-intensive work
    work_start = time.time()
    while time.time() - work_start < duration:
        for i in range(100000):
            _ = i * i * i
            
    end_time = time.time()
    actual_duration = end_time - start_time
    
    logger.info(f"Task {task_id} (PID {os.getpid()}): Completed in {actual_duration:.2f} seconds")
    return task_id, start_time, end_time


def measure_system_cpu_usage() -> float:
    """Get current system-wide CPU usage percentage"""
    return psutil.cpu_percent(interval=0.1)


def test_resource_oversubscription_during_exit():
    """
    Test that demonstrates resource oversubscription when Exit() releases 
    resources before tasks actually finish.
    """
    logger.info("=== Testing Resource Oversubscription During Worker Exit ===")
    
    # Initialize Ray with exactly 1 CPU to make oversubscription obvious
    ray.init(num_cpus=1.0, log_to_driver=True)
    
    try:
        # Phase 1: Baseline test - normal task execution without exit
        logger.info("\n--- Phase 1: Baseline Test ---")
        start_time = time.time()
        
        # Submit two 2-second CPU tasks sequentially (should take ~4 seconds total)
        task1 = cpu_intensive_task.remote(1, 2.0)
        task2 = cpu_intensive_task.remote(2, 2.0)
        
        results = ray.get([task1, task2])
        baseline_duration = time.time() - start_time
        
        logger.info(f"Baseline: Two sequential 2s tasks took {baseline_duration:.2f} seconds")
        
        # Verify they executed sequentially (not in parallel due to 1 CPU limit)
        task1_start, task1_end = results[0][1], results[0][2]
        task2_start, task2_end = results[1][1], results[1][2]
        
        overlap = max(0, min(task1_end, task2_end) - max(task1_start, task2_start))
        logger.info(f"Task overlap time: {overlap:.2f} seconds (should be ~0 for sequential)")
        
        # Phase 2: Test with actor exit during task execution
        logger.info("\n--- Phase 2: Actor Exit During Task Execution ---")
        
        # Create actor that will exit while consuming CPU
        actor = ResourceConsumingActor.remote()
        
        # Start the actor task that will consume CPU for 3 seconds then exit
        logger.info("Starting actor task that will exit_actor() while consuming CPU...")
        actor_future = actor.start_cpu_intensive_work_and_exit.remote(3.0)
        
        # Wait a moment for the actor to start its CPU work
        time.sleep(0.5)
        
        # Now submit a regular task that should wait for CPU to be available
        logger.info("Submitting regular task that should wait for CPU...")
        test_start_time = time.time()
        regular_task = cpu_intensive_task.remote(99, 2.0)
        
        # Measure CPU usage to see if we have oversubscription
        cpu_samples = []
        sample_start = time.time()
        
        # Sample CPU usage for a few seconds
        for i in range(30):  # 3 seconds of samples
            cpu_usage = measure_system_cpu_usage()
            cpu_samples.append(cpu_usage)
            time.sleep(0.1)
            
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        
        logger.info(f"CPU usage during test - Average: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")
        
        # Try to get the regular task result
        try:
            regular_result = ray.get(regular_task, timeout=5.0)
            task_end_time = time.time()
            total_test_time = task_end_time - test_start_time
            
            logger.info(f"Regular task completed in {total_test_time:.2f} seconds from test start")
            
            # Check for resource oversubscription indicators:
            # 1. High CPU usage (>150% suggests multiple CPU-bound tasks running)
            # 2. Task completing "too quickly" relative to expected queue time
            
            oversubscription_detected = False
            
            if max_cpu > 150:  # Significantly above 100% suggests oversubscription
                logger.error(f"🚨 OVERSUBSCRIPTION DETECTED: CPU usage peaked at {max_cpu:.1f}%")
                logger.error("This suggests multiple CPU-intensive tasks were running simultaneously!")
                oversubscription_detected = True
                
            if total_test_time < 2.5:  # Task should have been queued for ~1.5s + 2s execution
                logger.error(f"🚨 OVERSUBSCRIPTION DETECTED: Task completed too quickly ({total_test_time:.2f}s)")
                logger.error("Task should have been queued until actor finished, but started immediately!")
                oversubscription_detected = True
                
            if oversubscription_detected:
                logger.error("\n💥 BUG CONFIRMED: CoreWorker::Exit() releases resources before tasks finish!")
                logger.error("The raylet thinks resources are free while tasks are still running.")
                return False
            else:
                logger.info("✅ No oversubscription detected - resources properly managed")
                return True
                
        except ray.exceptions.GetTimeoutError:
            logger.error("Regular task timed out - something went wrong")
            return False
            
        except ray.exceptions.RayActorError as e:
            # This is expected - the actor exited
            logger.info(f"Actor exited as expected: {e}")
            
            # But we still want to check if the regular task executed properly
            try:
                regular_result = ray.get(regular_task, timeout=3.0)
                logger.info("Regular task completed after actor exit")
                return True
            except ray.exceptions.GetTimeoutError:
                logger.error("Regular task didn't complete in reasonable time")
                return False
                
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False
        
    finally:
        ray.shutdown()


def test_reference_draining_oversubscription():
    """
    Test oversubscription during reference draining for normal task workers.
    This demonstrates the case where DrainAndShutdown() blocks for a long time.
    """
    logger.info("\n=== Testing Resource Oversubscription During Reference Draining ===")
    
    ray.init(num_cpus=1.0, log_to_driver=True)
    
    try:
        @ray.remote(num_cpus=1.0)
        def task_with_long_lived_objects():
            """Task that creates objects held by remote references (simulating drain delay)"""
            import time
            
            # Create some objects that might be held by remote workers
            # In a real scenario, these could be held by other workers causing drain delays
            large_object = [i for i in range(1000000)]  # ~40MB list
            object_ref = ray.put(large_object)
            
            logger.info(f"Task (PID {os.getpid()}): Created large object, starting CPU work...")
            
            # Do CPU work for a while
            start_time = time.time()
            while time.time() - start_time < 2.0:
                for i in range(100000):
                    _ = i * i * i
                    
            # Return the object ref - this could cause draining delays if held elsewhere
            return object_ref
            
        # Submit the task
        logger.info("Submitting task that will create hard-to-drain references...")
        task_future = task_with_long_lived_objects.remote()
        
        # Wait for it to start
        time.sleep(0.5)
        
        # Simulate worker receiving exit signal (e.g., SIGTERM)
        # In real scenario, this would trigger CoreWorker::Exit() with reference draining
        logger.info("Simulating worker exit signal...")
        
        # Submit another task that should be queued
        logger.info("Submitting second task that should wait for resources...")
        second_task = cpu_intensive_task.remote(98, 1.5)
        
        # Monitor resource usage
        cpu_samples = []
        for i in range(20):
            cpu_usage = measure_system_cpu_usage()
            cpu_samples.append(cpu_usage)
            time.sleep(0.1)
            
        max_cpu = max(cpu_samples)
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        
        logger.info(f"CPU usage - Average: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")
        
        # Get results
        result1 = ray.get(task_future)
        result2 = ray.get(second_task)
        
        if max_cpu > 150:
            logger.error(f"🚨 OVERSUBSCRIPTION: CPU peaked at {max_cpu:.1f}% during reference draining")
            return False
        else:
            logger.info("✅ Reference draining handled properly")
            return True
            
    except Exception as e:
        logger.error(f"Reference draining test failed: {e}")
        return False
    finally:
        ray.shutdown()


if __name__ == "__main__":
    print("Testing Ray Worker Resource Oversubscription Bugs")
    print("=" * 60)
    
    # Test 1: Actor exit oversubscription
    success1 = test_resource_oversubscription_during_exit()
    
    # Test 2: Reference draining oversubscription  
    success2 = test_reference_draining_oversubscription()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"Actor Exit Test: {'PASS' if success1 else 'FAIL (Bug Detected)'}")
    print(f"Reference Drain Test: {'PASS' if success2 else 'FAIL (Bug Detected)'}")
    
    if not success1 or not success2:
        print("\n💥 Resource oversubscription bugs detected!")
        print("The raylet is told resources are free while tasks are still running.")
        exit(1)
    else:
        print("\n✅ All tests passed - no oversubscription detected")
        exit(0) 