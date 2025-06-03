#!/usr/bin/env python3
"""
Focused test for the CoreWorker::Exit() resource release bug.

Bug: In CoreWorker::Exit(), NotifyDirectCallTaskBlocked() is called immediately
to release resources, but tasks may still be draining asynchronously, leading
to resource oversubscription.

Code path:
1. ray.actor.exit_actor() -> Exit()
2. Exit() calls NotifyDirectCallTaskBlocked() immediately
3. Exit() posts async task to drain references (can take long time)
4. Raylet thinks resources are free while draining is still happening!
"""

import ray
import time
import threading
import subprocess
import sys
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


@ray.remote(num_cpus=1.0, max_restarts=0)
class VictimActor:
    """Actor that will trigger the resource release bug"""
    
    def trigger_exit_with_running_work(self):
        """
        This method demonstrates the bug:
        1. Start CPU-intensive work in background
        2. Call ray.actor.exit_actor() 
        3. exit_actor() -> Exit() -> NotifyDirectCallTaskBlocked() (resources released!)
        4. But background work is still running and consuming CPU!
        """
        log.info(f"VictimActor PID {os.getpid()}: Starting background CPU work...")
        
        # Flag to control background work
        self.keep_working = True
        
        def background_cpu_work():
            """Simulate ongoing work that would happen during reference draining"""
            count = 0
            while self.keep_working and count < 300:  # ~3 seconds of work
                # Actual CPU work - this will show up in system monitors
                for _ in range(100000):
                    x = 2.5 ** 0.5
                count += 1
                time.sleep(0.01)
            log.info(f"Background work finished after {count} iterations")
        
        # Start background work
        work_thread = threading.Thread(target=background_cpu_work, daemon=True)
        work_thread.start()
        
        # Give it time to start consuming CPU
        time.sleep(0.2)
        
        log.info(f"VictimActor PID {os.getpid()}: Calling ray.actor.exit_actor()!")
        log.info("This will trigger CoreWorker::Exit() -> NotifyDirectCallTaskBlocked()")
        log.info("Resources will be released immediately, but background work continues!")
        
        # This is the critical call that triggers the bug
        ray.actor.exit_actor()


@ray.remote(num_cpus=1.0)
def probe_task(task_id: int):
    """
    This task should be queued if resources are properly managed.
    If it starts immediately, we have resource oversubscription!
    """
    start_time = time.time()
    log.info(f"ProbeTask {task_id} PID {os.getpid()}: Started at {start_time}")
    
    # Do some work to make the timing visible
    for _ in range(50):
        for _ in range(100000):
            x = 3.14159 ** 0.5
        time.sleep(0.02)
    
    end_time = time.time()
    duration = end_time - start_time
    log.info(f"ProbeTask {task_id} PID {os.getpid()}: Completed in {duration:.2f}s")
    
    return {"task_id": task_id, "start_time": start_time, "end_time": end_time, "pid": os.getpid()}


def run_resource_oversubscription_test():
    """Test that reproduces the resource oversubscription bug"""
    
    log.info("="*70)
    log.info("TESTING: Resource Oversubscription Bug in CoreWorker::Exit()")
    log.info("="*70)
    
    # Initialize Ray with exactly 1 CPU
    ray.init(num_cpus=1.0, log_to_driver=False)
    
    try:
        log.info("Phase 1: Baseline - measure normal task scheduling")
        
        # Baseline: Two sequential tasks (should take ~2+ seconds total)
        baseline_start = time.time()
        baseline_task1 = probe_task.remote(1)
        baseline_task2 = probe_task.remote(2)
        
        baseline_results = ray.get([baseline_task1, baseline_task2])
        baseline_duration = time.time() - baseline_start
        
        # Check if they ran sequentially (different PIDs or no time overlap)
        task1_pid = baseline_results[0]["pid"]
        task2_pid = baseline_results[1]["pid"] 
        task1_end = baseline_results[0]["end_time"]
        task2_start = baseline_results[1]["start_time"]
        
        sequential = (task1_pid != task2_pid) or (task2_start >= task1_end - 0.1)
        
        log.info(f"Baseline: Duration={baseline_duration:.2f}s, Sequential={sequential}")
        log.info(f"Task1 PID={task1_pid}, Task2 PID={task2_pid}")
        
        time.sleep(1)  # Cool down
        
        log.info("\nPhase 2: Bug reproduction - actor exit with ongoing work")
        
        # Create the victim actor
        victim = VictimActor.remote()
        
        # Start the actor method that will trigger exit_actor() with ongoing work
        log.info("Starting victim actor method (will exit with background work)...")
        victim_future = victim.trigger_exit_with_running_work.remote()
        
        # Wait for the actor to start its work and call exit_actor()
        time.sleep(0.5)
        
        # Now submit a probe task - this should be queued until CPU is actually free
        log.info("Submitting probe task (should be queued until CPU truly free)...")
        test_start_time = time.time()
        probe_future = probe_task.remote(99)
        
        # Monitor if probe task starts "too quickly"
        probe_started = False
        for i in range(10):  # Check for 1 second
            try:
                # If we can get the result quickly, the task started immediately (BAD!)
                result = ray.get(probe_future, timeout=0.1)
                probe_end_time = time.time()
                probe_duration = probe_end_time - test_start_time
                
                log.error(f"🚨 BUG DETECTED: Probe task completed in {probe_duration:.2f}s")
                log.error(f"Task PID: {result['pid']}")
                log.error("This indicates resource oversubscription!")
                log.error("The actor's background work was still running when probe started.")
                probe_started = True
                break
                
            except ray.exceptions.GetTimeoutError:
                # Good - task is still running/queued
                pass
            except ray.exceptions.RayActorError:
                # Expected - victim actor exited
                pass
            
            time.sleep(0.1)
        
        if not probe_started:
            # Task is taking longer - check if it eventually completes
            try:
                result = ray.get(probe_future, timeout=5.0)
                probe_end_time = time.time()
                probe_duration = probe_end_time - test_start_time
                
                log.info(f"✅ Probe task completed in {probe_duration:.2f}s")
                log.info(f"Task PID: {result['pid']}")
                
                if probe_duration < 1.5:
                    log.error("🚨 BUG DETECTED: Task completed too quickly!")
                    log.error("Should have been queued longer due to actor background work.")
                    return False
                else:
                    log.info("✅ Task was properly queued - no oversubscription detected")
                    return True
                    
            except ray.exceptions.GetTimeoutError:
                log.error("❌ Probe task timed out - unexpected")
                return False
        else:
            return False
        
        # Handle actor cleanup
        try:
            ray.get(victim_future, timeout=0.5)
        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
            pass  # Expected
            
    except Exception as e:
        log.error(f"Test failed with exception: {e}")
        return False
        
    finally:
        ray.shutdown()


def main():
    print("Ray CoreWorker::Exit() Resource Bug Test")
    print("This test checks if resources are released before tasks actually finish")
    
    success = run_resource_oversubscription_test()
    
    print("\n" + "="*70)
    if success:
        print("✅ TEST PASSED: No resource oversubscription detected")
        print("Resources are properly managed during worker exit")
    else:
        print("💥 TEST FAILED: Resource oversubscription bug confirmed!")
        print("")
        print("BUG EXPLANATION:")
        print("In CoreWorker::Exit(), NotifyDirectCallTaskBlocked() is called")
        print("immediately to release resources to the raylet. However, tasks")
        print("may still be running during the asynchronous draining phase,")
        print("leading to resource oversubscription.")
        print("")
        print("CODE LOCATION: src/ray/core_worker/core_worker.cc")
        print("Look for: NotifyDirectCallTaskBlocked() in Exit() method")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 