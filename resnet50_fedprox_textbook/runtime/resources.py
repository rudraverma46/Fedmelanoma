"""GPU, RAM and Windows commit sampling, the low-commit safe stop, and keeping the PC awake."""
from __future__ import annotations

import csv
import ctypes
import os
import subprocess
import threading
import time
from typing import Dict

import psutil

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
LOW_COMMIT_SAMPLES_TO_STOP = 3


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_snapshot() -> Dict[str, float]:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    gb = 2**30
    return {"ram_free_gb": status.ullAvailPhys / gb, "commit_limit_gb": status.ullTotalPageFile / gb,
            "commit_free_gb": status.ullAvailPageFile / gb}


def gpu_snapshot() -> Dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    util, mem, temp, power = [x.strip() for x in out.split(",")]
    return {"gpu_util_pct": float(util), "gpu_mem_mb": float(mem), "gpu_temp_c": float(temp), "gpu_power_w": float(power)}


def keep_awake(enable: bool) -> None:
    """Stop Windows from sleeping during the run; released when the process ends."""
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


class ResourceMonitor(threading.Thread):
    """Appends one CSV row per interval and raises `low_commit` if free commit stays too low."""

    def __init__(self, path: str, interval: float, commit_floor_gb: float) -> None:
        super().__init__(daemon=True)
        self.path, self.interval, self.floor = path, interval, commit_floor_gb
        self.low_commit = threading.Event()
        self.halt = threading.Event()
        self.proc = psutil.Process()
        self.current_round = 0

    def run(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        new_file = not os.path.exists(self.path)
        low = 0
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = None
            while not self.halt.is_set():
                row = {"time": time.time(), "round": self.current_round, **memory_snapshot(),
                       "process_rss_gb": self.proc.memory_info().rss / 2**30}
                try:
                    row.update(gpu_snapshot())
                except Exception:
                    row.update({"gpu_util_pct": "", "gpu_mem_mb": "", "gpu_temp_c": "", "gpu_power_w": ""})
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    if new_file:
                        writer.writeheader()
                writer.writerow(row)
                fh.flush()
                low = low + 1 if row["commit_free_gb"] < self.floor else 0
                if low >= LOW_COMMIT_SAMPLES_TO_STOP:
                    self.low_commit.set()
                self.halt.wait(self.interval)
