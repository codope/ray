import ray
import subprocess
import time
import psutil
import os
import signal
import sys


@ray.remote
class Worker:
    def __init__(self):
        self.direct_child = None
        
    def spawn_processes(self):
        """Spawn direct child, which spawns grandchild."""
        # Create a script for the direct child
        child_script = f'''
import subprocess
import time
import os

print(f"Direct child {{os.getpid()}} started")

# Spawn grandchild that will leak
grandchild = subprocess.Popen([
    "{sys.executable}", "-c", 
    "import time, os; print('Grandchild', os.getpid(), 'alive'); [time.sleep(1) for _ in range(3600)]"
])

print(f"Spawned grandchild {{grandchild.pid}}")

# Keep direct child alive
time.sleep(3600)
'''
        
        # Write script to temp file
        with open('/tmp/child.py', 'w') as f:
            f.write(child_script)
        
        # Spawn direct child
        self.direct_child = subprocess.Popen([sys.executable, '/tmp/child.py'])
        
        # Give time for grandchild to spawn
        time.sleep(2)
        
        return self.direct_child.pid
    
    def get_worker_pid(self):
        return os.getpid()
    
    def trigger_oom(self):
        """Simulate OOM by allocating memory then SIGKILL."""
        # Allocate some memory
        big_list = [bytearray(100 * 1024 * 1024) for _ in range(5)]  # 500MB
        
        # Kill self with SIGKILL (simulating OOM killer)
        os.kill(os.getpid(), signal.SIGKILL)


def find_python_processes():
    """Find all Python processes that might be our test processes."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
        try:
            if 'python' in proc.info['name'] and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'Grandchild' in cmdline or 'child.py' in cmdline:
                    processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'ppid': proc.info['ppid']
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes


def test_leak(use_subreaper=False):
    """Test the process leak scenario."""
    print(f"\n{'='*50}")
    print(f"TEST {'WITH' if use_subreaper else 'WITHOUT'} SUBREAPER")
    print(f"{'='*50}")
    
    # Start Ray
    config = {}
    if use_subreaper:
        config["kill_child_processes_on_worker_exit_with_raylet_subreaper"] = True
    
    ray.init(ignore_reinit_error=True, _system_config=config)
    
    # Create worker
    worker = Worker.remote()
    worker_pid = ray.get(worker.get_worker_pid.remote())
    print(f"Ray worker PID: {worker_pid}")
    
    # Spawn processes
    direct_child_pid = ray.get(worker.spawn_processes.remote())
    print(f"Direct child PID: {direct_child_pid}")
    
    # Check what processes we have
    before_processes = find_python_processes()
    print(f"Test processes before crash: {len(before_processes)}")
    for p in before_processes:
        print(f"  PID {p['pid']}: {p['cmdline'][:60]}...")
    
    # Trigger OOM kill
    try:
        ray.get(worker.trigger_oom.remote())
    except Exception as e:
        print(f"Worker crashed as expected: {type(e).__name__}")
    
    # Wait for cleanup
    wait_time = 15 if use_subreaper else 5
    print(f"Waiting {wait_time}s for cleanup...")
    time.sleep(wait_time)
    
    # Check what's left
    after_processes = find_python_processes()
    print(f"Test processes after crash: {len(after_processes)}")
    
    leaked = 0
    for p in after_processes:
        print(f"  LEAKED PID {p['pid']}: {p['cmdline'][:60]}...")
        leaked += 1
    
    ray.shutdown()
    
    # Cleanup
    try:
        subprocess.run(['pkill', '-f', 'child.py'], check=False)
        subprocess.run(['pkill', '-f', 'Grandchild'], check=False)
        os.remove('/tmp/child.py')
    except:
        pass
    
    return leaked


def main():
    print("Simple Process Leak Test")
    print("Ray Worker -> Direct Child -> Grandchild")
    
    # Test default behavior
    leaked_default = test_leak(use_subreaper=False)
    
    # Test with subreaper  
    leaked_subreaper = test_leak(use_subreaper=True)
    
    # Results
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Leaked without subreaper: {leaked_default}")
    print(f"Leaked with subreaper: {leaked_subreaper}")
    
    if leaked_default > 0 and leaked_subreaper == 0:
        print(">>> SUCCESS: Subreaper fixed the leak!")
    elif leaked_default > 0:
        print(">>> Processes still leaked even with subreaper")
    else:
        print(">>> No leak detected - test may need adjustment")


if __name__ == "__main__":
    main()
