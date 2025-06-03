#!/usr/bin/env python3
"""
Comprehensive test cases for Ray worker shutdown deadlocks and race conditions.

This file contains simple, focused tests that reproduce various shutdown bugs
identified in Ray's core worker implementation.
"""

import ray
import time
import threading
import signal
import os
import sys
import logging
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


class ShutdownTestSuite:
    """Test suite for Ray worker shutdown issues"""
    
    def __init__(self):
        self.test_results = {}
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        log.info(f"\n{'='*60}")
        log.info(f"Running: {test_name}")
        log.info(f"{'='*60}")
        
        try:
            success = test_func()
            self.test_results[test_name] = success
            status = "✅ PASS" if success else "❌ FAIL"
            log.info(f"{status}: {test_name}")
            return success
        except Exception as e:
            log.error(f"💥 ERROR in {test_name}: {e}")
            self.test_results[test_name] = False
            return False
    
    def print_summary(self):
        """Print test results summary"""
        print(f"\n{'='*70}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*70}")
        
        passed = 0
        failed = 0
        
        for test_name, success in self.test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
            if success:
                passed += 1
            else:
                failed += 1
        
        print(f"\nTotal: {len(self.test_results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print(f"\n💥 {failed} shutdown bugs detected!")
            return False
        else:
            print(f"\n✅ All shutdown tests passed!")
            return True


def test_multithreading_shutdown_race():
    """
    Test: Multiple threads submitting tasks during shutdown
    Bug: Race condition when ray.shutdown() called while threads are active
    """
    ray.init(num_cpus=2)
    
    try:
        @ray.remote
        def simple_task():
            time.sleep(0.1)
            return "done"
        
        shutdown_event = Event()
        exception_caught = Event()
        
        def worker_thread(thread_id):
            """Continuously submit tasks until shutdown"""
            try:
                while not shutdown_event.is_set():
                    # Submit batch of tasks
                    futures = [simple_task.remote() for _ in range(5)]
                    results = ray.get(futures, timeout=2.0)
                    log.info(f"Thread {thread_id}: Completed batch")
                    time.sleep(0.1)
            except Exception as e:
                log.info(f"Thread {thread_id} got exception (expected): {e}")
                exception_caught.set()
        
        # Start multiple worker threads
        threads = [Thread(target=worker_thread, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        
        # Let them run for a bit
        time.sleep(2.0)
        
        # Trigger shutdown while threads are active
        log.info("Triggering shutdown while threads are submitting tasks...")
        start_time = time.time()
        
        shutdown_event.set()  # Signal threads to stop
        ray.shutdown()  # This should not hang or crash
        
        shutdown_time = time.time() - start_time
        log.info(f"Shutdown completed in {shutdown_time:.2f} seconds")
        
        # Wait for threads to finish
        for t in threads:
            t.join(timeout=3.0)
        
        # Success if shutdown completed in reasonable time and didn't crash
        return shutdown_time < 10.0  # Should not take more than 10 seconds
        
    except Exception as e:
        log.error(f"Multithreading shutdown test failed: {e}")
        return False
    finally:
        # Ensure Ray is shut down
        try:
            ray.shutdown()
        except:
            pass


def test_actor_exit_resource_oversubscription():
    """
    Test: Resource oversubscription during actor exit
    Bug: NotifyDirectCallTaskBlocked() called before tasks actually finish
    """
    ray.init(num_cpus=1)
    
    try:
        @ray.remote(num_cpus=1.0)
        class ResourceHogActor:
            def exit_with_background_work(self):
                """Exit while background work is still running"""
                
                # Start background work that consumes CPU
                def cpu_work():
                    for _ in range(200):  # ~2 seconds of work
                        for _ in range(100000):
                            x = 2.5 ** 0.5
                        time.sleep(0.01)
                
                work_thread = threading.Thread(target=cpu_work, daemon=True)
                work_thread.start()
                
                time.sleep(0.2)  # Let work start
                log.info("Actor calling exit_actor() with background work running...")
                
                # This triggers the resource release bug
                ray.actor.exit_actor()
        
        @ray.remote(num_cpus=1.0)
        def probe_task():
            """Task that should be queued if resources properly managed"""
            start_time = time.time()
            # Do some work
            for _ in range(100):
                for _ in range(50000):
                    x = 3.14 ** 0.5
                time.sleep(0.01)
            
            duration = time.time() - start_time
            return {"start_time": start_time, "duration": duration}
        
        # Create actor and trigger exit with background work
        actor = ResourceHogActor.remote()
        actor_future = actor.exit_with_background_work.remote()
        
        # Wait for actor to start exiting
        time.sleep(0.5)
        
        # Submit probe task - should be queued until CPU truly free
        test_start = time.time()
        probe_future = probe_task.remote()
        
        # Check if probe task starts too quickly (indicates oversubscription)
        quick_completion = False
        try:
            result = ray.get(probe_future, timeout=1.0)
            probe_duration = time.time() - test_start
            
            if probe_duration < 1.5:  # Started too quickly
                log.error(f"🚨 OVERSUBSCRIPTION: Probe task completed in {probe_duration:.2f}s")
                quick_completion = True
            else:
                log.info(f"✅ Probe task properly queued: {probe_duration:.2f}s")
                
        except ray.exceptions.GetTimeoutError:
            # Task taking longer - good sign
            try:
                result = ray.get(probe_future, timeout=3.0)
                log.info("✅ Task completed after proper queuing")
            except ray.exceptions.GetTimeoutError:
                log.error("Task timed out unexpectedly")
                return False
        
        # Clean up actor
        try:
            ray.get(actor_future, timeout=1.0)
        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
            pass  # Expected
        
        # Return True if no oversubscription detected
        return not quick_completion
        
    except Exception as e:
        log.error(f"Resource oversubscription test failed: {e}")
        return False
    finally:
        ray.shutdown()


def test_multiple_shutdown_calls():
    """
    Test: Multiple concurrent shutdown calls
    Bug: Race condition in atomic flag handling
    """
    ray.init()
    
    try:
        shutdown_results = []
        
        def shutdown_worker(worker_id):
            """Try to shutdown Ray"""
            try:
                start_time = time.time()
                ray.shutdown()
                duration = time.time() - start_time
                shutdown_results.append((worker_id, "success", duration))
                log.info(f"Worker {worker_id}: Shutdown succeeded in {duration:.3f}s")
            except Exception as e:
                shutdown_results.append((worker_id, "error", str(e)))
                log.info(f"Worker {worker_id}: Shutdown failed: {e}")
        
        # Launch multiple concurrent shutdown attempts
        threads = [Thread(target=shutdown_worker, args=(i,)) for i in range(5)]
        
        # Start all threads simultaneously
        for t in threads:
            t.start()
        
        # Wait for all to complete
        for t in threads:
            t.join(timeout=10.0)
        
        # Analyze results
        successes = [r for r in shutdown_results if r[1] == "success"]
        errors = [r for r in shutdown_results if r[1] == "error"]
        
        log.info(f"Shutdown attempts: {len(shutdown_results)}")
        log.info(f"Successes: {len(successes)}")
        log.info(f"Errors: {len(errors)}")
        
        # Success if at least one succeeded and no crashes
        return len(successes) >= 1 and len(shutdown_results) == 5
        
    except Exception as e:
        log.error(f"Multiple shutdown test failed: {e}")
        return False


def test_signal_handler_shutdown_race():
    """
    Test: Signal handler triggered during task execution
    Bug: Race condition between signal handling and task cleanup
    """
    if os.name == 'nt':  # Skip on Windows
        log.info("Skipping signal test on Windows")
        return True
    
    # Use subprocess to test signal handling
    test_script = '''
import ray
import signal
import time
import sys
import os

def signal_handler(signum, frame):
    print(f"Received signal {signum}, shutting down...")
    ray.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

ray.init()

@ray.remote
def long_running_task():
    time.sleep(2.0)
    return "completed"

# Submit tasks
futures = [long_running_task.remote() for _ in range(5)]

# Send signal to self after a delay
import threading
def send_signal():
    time.sleep(1.0)
    os.kill(os.getpid(), signal.SIGTERM)

threading.Thread(target=send_signal).start()

try:
    # This should be interrupted by signal
    results = ray.get(futures, timeout=5.0)
    print("ERROR: Tasks completed before signal")
    sys.exit(1)
except Exception as e:
    print(f"Expected interruption: {e}")
    sys.exit(0)
'''
    
    try:
        # Run the test script in subprocess
        process = subprocess.Popen(
            [sys.executable, '-c', test_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        
        stdout, stderr = process.communicate()
        
        # Success if process exited cleanly
        success = process.returncode == 0
        
        if success:
            log.info("✅ Signal handler shutdown worked correctly")
        else:
            log.error(f"❌ Signal handler test failed: {stderr.decode()}")
        
        return success
        
    except subprocess.TimeoutExpired:
        log.error("Signal handler test timed out")
        return False
    except Exception as e:
        log.error(f"Signal handler test failed: {e}")
        return False


def test_asyncio_actor_shutdown():
    """
    Test: Asyncio actor shutdown race conditions
    Bug: Race between asyncio event loop and worker shutdown
    """
    ray.init()
    
    try:
        @ray.remote
        class AsyncActor:
            async def async_work(self):
                """Simulate async work"""
                import asyncio
                await asyncio.sleep(1.0)
                return "async_done"
            
            def exit_during_async_work(self):
                """Exit while async work is running"""
                import asyncio
                
                async def background_work():
                    for i in range(10):
                        await asyncio.sleep(0.2)
                        print(f"Async work iteration {i}")
                
                # Start background async work
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Schedule background work
                task = loop.create_task(background_work())
                
                # Exit while async work is running
                import threading
                def delayed_exit():
                    time.sleep(0.5)
                    ray.actor.exit_actor()
                
                threading.Thread(target=delayed_exit).start()
                
                try:
                    loop.run_until_complete(task)
                except Exception as e:
                    print(f"Async work interrupted: {e}")
                finally:
                    loop.close()
        
        # Test asyncio actor shutdown
        actor = AsyncActor.remote()
        
        try:
            future = actor.exit_during_async_work.remote()
            ray.get(future, timeout=3.0)
            
            # If we get here without hanging, test passed
            log.info("✅ Asyncio actor shutdown completed without hanging")
            return True
            
        except ray.exceptions.RayActorError:
            # Expected - actor exited
            log.info("✅ Asyncio actor exited as expected")
            return True
        except ray.exceptions.GetTimeoutError:
            log.error("❌ Asyncio actor shutdown hung")
            return False
        
    except Exception as e:
        log.error(f"Asyncio actor test failed: {e}")
        return False
    finally:
        ray.shutdown()


def test_reference_draining_deadlock():
    """
    Test: Reference counter draining deadlock
    Bug: DrainAndShutdown can hang indefinitely waiting for references
    """
    ray.init()
    
    try:
        @ray.remote
        def create_circular_refs():
            """Create circular references that are hard to drain"""
            
            # Create objects that reference each other
            obj1 = {"data": list(range(10000)), "ref": None}
            obj2 = {"data": list(range(10000)), "ref": None}
            
            # Store in object store
            ref1 = ray.put(obj1)
            ref2 = ray.put(obj2)
            
            # Create circular reference
            obj1["ref"] = ref2
            obj2["ref"] = ref1
            
            # Update in object store
            ray.put(obj1)
            ray.put(obj2)
            
            return [ref1, ref2]
        
        @ray.remote
        def worker_with_refs():
            """Worker that holds references and exits"""
            refs = ray.get(create_circular_refs.remote())
            
            # Hold references for a while
            time.sleep(1.0)
            
            # Exit while holding references
            return "done_with_refs"
        
        # Start multiple workers with circular references
        futures = [worker_with_refs.remote() for _ in range(3)]
        
        # Get results to ensure tasks complete
        start_time = time.time()
        results = ray.get(futures, timeout=5.0)
        completion_time = time.time() - start_time
        
        log.info(f"Reference draining completed in {completion_time:.2f} seconds")
        
        # Success if completed in reasonable time
        return completion_time < 10.0
        
    except ray.exceptions.GetTimeoutError:
        log.error("❌ Reference draining hung - potential deadlock")
        return False
    except Exception as e:
        log.error(f"Reference draining test failed: {e}")
        return False
    finally:
        ray.shutdown()


def test_task_receiver_stop_race():
    """
    Test: Task receiver stop race condition
    Bug: Race between stopping task receiver and task execution
    """
    ray.init()
    
    try:
        @ray.remote
        def rapid_fire_task(task_id):
            """Quick task for rapid submission"""
            return f"task_{task_id}_done"
        
        # Submit many tasks rapidly
        futures = []
        for i in range(100):
            futures.append(rapid_fire_task.remote(i))
        
        # Start getting results
        results = []
        start_time = time.time()
        
        # Get first few results
        for i in range(10):
            results.append(ray.get(futures[i]))
        
        # Trigger shutdown while many tasks are still pending
        log.info("Shutting down with tasks still pending...")
        ray.shutdown()
        
        shutdown_time = time.time() - start_time
        log.info(f"Shutdown with pending tasks completed in {shutdown_time:.2f} seconds")
        
        # Success if shutdown completed quickly without hanging
        return shutdown_time < 5.0
        
    except Exception as e:
        log.error(f"Task receiver stop test failed: {e}")
        return False


def test_timeout_handling():
    """
    Test: Shutdown timeout handling
    Bug: Missing or incorrect timeout handling in shutdown sequence
    """
    ray.init()
    
    try:
        @ray.remote
        def slow_cleanup_task():
            """Task that takes long time to clean up"""
            try:
                # Simulate slow cleanup
                time.sleep(3.0)
                return "cleanup_done"
            except Exception:
                # Simulate cleanup that can't be interrupted
                time.sleep(2.0)
                raise
        
        # Submit slow task
        future = slow_cleanup_task.remote()
        
        # Try to shutdown with timeout
        start_time = time.time()
        
        # This should not hang indefinitely
        ray.shutdown()
        
        shutdown_time = time.time() - start_time
        log.info(f"Shutdown with slow cleanup completed in {shutdown_time:.2f} seconds")
        
        # Success if completed in reasonable time (with timeout enforcement)
        return shutdown_time < 10.0
        
    except Exception as e:
        log.error(f"Timeout handling test failed: {e}")
        return False


def main():
    """Run all shutdown tests"""
    print("Ray Worker Shutdown Test Suite")
    print("=" * 70)
    print("Testing various deadlocks and race conditions in Ray worker shutdown")
    
    suite = ShutdownTestSuite()
    
    # Run all tests
    tests = [
        ("Multithreading Shutdown Race", test_multithreading_shutdown_race),
        ("Actor Exit Resource Oversubscription", test_actor_exit_resource_oversubscription),
        ("Multiple Shutdown Calls", test_multiple_shutdown_calls),
        ("Signal Handler Shutdown Race", test_signal_handler_shutdown_race),
        ("Asyncio Actor Shutdown", test_asyncio_actor_shutdown),
        ("Reference Draining Deadlock", test_reference_draining_deadlock),
        ("Task Receiver Stop Race", test_task_receiver_stop_race),
        ("Timeout Handling", test_timeout_handling),
    ]
    
    for test_name, test_func in tests:
        suite.run_test(test_name, test_func)
        time.sleep(1)  # Cool down between tests
    
    # Print summary and return exit code
    success = suite.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 