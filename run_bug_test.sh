#!/bin/bash

echo "Ray Worker Resource Oversubscription Bug Test"
echo "============================================="
echo ""
echo "This test reproduces the bug where CoreWorker::Exit() releases"
echo "resources to the raylet before tasks actually finish draining."
echo ""

# Check if Ray is available
python3 -c "import ray; print(f'Ray version: {ray.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Ray is not installed. Please install Ray first:"
    echo "pip install ray"
    exit 1
fi

# Check if psutil is available (for CPU monitoring)
python3 -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ psutil is not installed. Please install it:"
    echo "pip install psutil"
    exit 1
fi

echo "Running the resource oversubscription test..."
echo ""

# Run the test
python3 test_exit_resource_bug.py

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Test completed successfully"
else
    echo "💥 Test detected the resource oversubscription bug!"
    echo ""
    echo "What this means:"
    echo "- CoreWorker::Exit() calls NotifyDirectCallTaskBlocked() immediately"
    echo "- This tells raylet that worker resources are free"
    echo "- But tasks may still be running during async reference draining"
    echo "- Result: raylet schedules more tasks than actual CPU resources"
    echo ""
    echo "To fix this bug, resources should only be released AFTER"
    echo "all tasks and references are actually drained and cleaned up."
fi

exit $exit_code 