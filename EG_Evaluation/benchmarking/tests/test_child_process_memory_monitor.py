import subprocess
import sys
import time
import unittest

from child_process_memory_monitor import ChildProcessMemoryMonitor


class ChildProcessMemoryMonitorTests(unittest.TestCase):
    def test_cpu_only_monitor_does_not_initialize_gpu_sampling(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.05)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        monitor = ChildProcessMemoryMonitor(
            process.pid,
            physical_gpu=-1,
            interval_seconds=0.002,
        ).start()
        process.communicate(timeout=5)
        result = monitor.stop()
        self.assertIsNone(result["gpu_index"])
        self.assertIsNone(result["gpu_proc_peak_mb"])
        self.assertEqual(result["monitor_gpu_proc_samples"], 0)
        self.assertGreater(result["monitor_rss_samples"], 0)

    def test_parent_monitor_samples_child_while_child_is_busy(self):
        command = [
            sys.executable,
            "-c",
            (
                "import time; "
                "payload = bytearray(32 * 1024 * 1024); "
                "time.sleep(0.50); "
                "print(len(payload))"
            ),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        monitor = ChildProcessMemoryMonitor(
            process.pid,
            physical_gpu=0,
            interval_seconds=0.002,
        ).start()
        stdout, stderr = process.communicate(timeout=5)
        result = monitor.stop()
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(stdout.strip(), str(32 * 1024 * 1024))
        self.assertGreaterEqual(result["monitor_rss_samples"], 10)
        self.assertGreater(result["rss_mb"], 20.0)
        self.assertEqual(
            result["memory_monitor_origin"],
            "coordinator_process_child_tree",
        )


if __name__ == "__main__":
    unittest.main()
