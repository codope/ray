#!/bin/bash

echo "Ray Worker Shutdown Test Suite"
echo "=============================="
echo ""
echo "This comprehensive test suite checks for various deadlocks,"
echo "race conditions, and bugs in Ray's worker shutdown logic."
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is required"
    exit 1
fi

# Check Ray
python3 -c "import ray; print(f'Ray version: {ray.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Ray is not installed. Install with: pip install ray"
    exit 1
fi

# Check psutil
python3 -c "import psutil; print(f'psutil version: {psutil.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ psutil is not installed. Install with: pip install psutil"
    exit 1
fi

echo "✅ All dependencies available"
echo ""

# Show what tests will run
echo "Tests to run:"
echo "============="
echo "1. Multithreading Shutdown Race - Multiple threads submitting during shutdown"
echo "2. Actor Exit Resource Oversubscription - Resources released too early"  
echo "3. Multiple Shutdown Calls - Concurrent shutdown attempts"
echo "4. Signal Handler Shutdown Race - SIGTERM during task execution"
echo "5. Asyncio Actor Shutdown - Asyncio event loop shutdown race"
echo "6. Reference Draining Deadlock - Circular references causing hangs"
echo "7. Task Receiver Stop Race - Task receiver shutdown race"
echo "8. Timeout Handling - Shutdown timeout enforcement"
echo ""

read -p "Press Enter to start the tests..."
echo ""

# Run the test suite
echo "Starting Ray Worker Shutdown Test Suite..."
echo "=========================================="

python3 test_worker_shutdown.py

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    echo "No shutdown bugs detected in this Ray installation."
else
    echo "💥 SOME TESTS FAILED"
    echo "Shutdown bugs detected! See test output above for details."
    echo ""
    echo "Common issues that may be detected:"
    echo "- Resource oversubscription during worker exit"
    echo "- Deadlocks in reference draining"
    echo "- Race conditions in multithreaded shutdown"
    echo "- Signal handler interference"
    echo "- Asyncio actor shutdown hangs"
    echo ""
    echo "These bugs can cause:"
    echo "- Worker processes hanging on shutdown"
    echo "- Resource leaks and oversubscription"
    echo "- Cluster instability"
    echo "- Job failures during cleanup"
fi

exit $exit_code 