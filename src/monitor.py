"""
GPU monitor — polls VRAM usage and GPU utilisation via nvidia-ml-py.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class GpuSnapshot:
    """Single-point GPU measurement."""
    timestamp: float
    gpu_index: int
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: int
    temperature_c: int


class GpuMonitor:
    """
    Background GPU polling thread.

    Usage:
        monitor = GpuMonitor(gpu_index=0, interval=1.0)
        monitor.start()
        ...
        snap = monitor.latest()
        monitor.stop()
    """

    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        self._gpu_index = gpu_index
        self._interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: GpuSnapshot | None = None
        self._pynvml_available = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._pynvml_available = True
        except Exception:
            self._pynvml_available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._pynvml_available:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._pynvml_available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self._sample()
                with self._lock:
                    self._latest = snap
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def _sample(self) -> GpuSnapshot:
        mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        temp = self._pynvml.nvmlDeviceGetTemperature(self._handle, 0)  # 0 = GPU die

        return GpuSnapshot(
            timestamp=time.time(),
            gpu_index=self._gpu_index,
            memory_used_mb=mem.used // (1024 * 1024),
            memory_total_mb=mem.total // (1024 * 1024),
            utilization_pct=util.gpu,
            temperature_c=temp,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def latest(self) -> GpuSnapshot | None:
        with self._lock:
            return self._latest

    @property
    def available(self) -> bool:
        return self._pynvml_available

    def status_line(self) -> str:
        """Return a one-line GPU status string for progress display."""
        snap = self.latest()
        if snap is None:
            return "GPU: N/A"
        return (
            f"GPU: {snap.utilization_pct}% util, "
            f"{snap.memory_used_mb / 1024:.1f}GB/{snap.memory_total_mb / 1024:.1f}GB, "
            f"{snap.temperature_c}°C"
        )
