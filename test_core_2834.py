"""
Reproduction script for CHECK failure in HandleTaskExecutionResult.

This demonstrates that a KeyboardInterrupt arriving during store_task_errors
in execute_task_with_cancellation_handler's except handler causes
task_execution_handler to return OK status without populating return objects.

The trigger sequence is:
  1. Task is running (e.g., time.sleep)
  2. First ray.cancel() → KeyboardInterrupt interrupts the task
  3. execute_task re-raises KeyboardInterrupt
  4. execute_task_with_cancellation_handler catches it, calls store_task_errors
  5. Second ray.cancel() → KeyboardInterrupt during store_task_errors
     (current_task_id is still set, so kill_main_task sends the interrupt)
  6. KeyboardInterrupt escapes all handlers in task_execution_handler
  7. Cython swallows exception, returns default CRayStatus (OK)
  8. HandleTaskExecutionResult finds status=OK but return_objects[0]=nullptr
  9. CHECK failure

USAGE:
  Step 1: Add a sleep to widen the error-storage window. In _raylet.pyx,
          inside store_task_errors, add "import time; time.sleep(3)"
          right before the store_task_outputs call (around line 1021).

          i.e., change:
              num_errors_stored = core_worker.store_task_outputs(
          to:
              import time; time.sleep(3)   # TEMPORARY - for repro only
              num_errors_stored = core_worker.store_task_outputs(

  Step 2: Rebuild Ray:
              cd python && pip install -e . --verbose

  Step 3: Run this script:
              python test_issue_59582.py

  Expected: Worker crashes with:
      Check failed: objects_valid 1 return objects expected, 1 returned.
      Object at idx 0 was not stored.

  Step 4: Remove the temporary sleep and rebuild.

ALTERNATIVE (non-invasive, less reliable):
  Without modifying source, you can try running the script as-is, but
  the timing window for the second cancel to hit during store_task_errors
  is very small (~milliseconds), so it may not reproduce consistently.
"""

import time

import ray


@ray.remote(num_returns=1)
def long_running_task():
    """A task that uses a busy-wait loop so KeyboardInterrupt is delivered
    promptly. time.sleep() may not be reliably interrupted by
    _thread.interrupt_main() on all platforms."""
    import time as _time

    end = _time.monotonic() + 120
    while _time.monotonic() < end:
        # Short sleeps ensure CPython signal check points fire frequently
        _time.sleep(0.05)
    return 42


def main():
    ray.init(num_cpus=1)
    try:
        ref = long_running_task.remote()

        # Wait for task to start executing on the worker
        time.sleep(2)

        print("[driver] Sending first cancel (interrupts the task)...")
        ray.cancel(ref)

        # Wait for the worker to enter store_task_errors and hit the 2s sleep.
        # The first cancel delivers KeyboardInterrupt, which propagates to
        # execute_task_with_cancellation_handler's except handler, which calls
        # store_task_errors. Our instrumented sleep(2) in store_task_errors
        # creates a 2-second window for the second cancel.
        time.sleep(1)

        print("[driver] Sending second cancel (interrupts error storage)...")
        ray.cancel(ref)

        # Wait for the worker to process
        time.sleep(5)

        try:
            result = ray.get(ref, timeout=30)
            print(f"[driver] Unexpected success: {result}")
        except ray.exceptions.TaskCancelledError:
            print("[driver] Got TaskCancelledError (expected if fix is in place)")
        except ray.exceptions.WorkerCrashedError as e:
            print(f"[driver] Worker crashed (CHECK failure triggered): {e}")
        except Exception as e:
            print(f"[driver] Got exception: {type(e).__name__}: {e}")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
