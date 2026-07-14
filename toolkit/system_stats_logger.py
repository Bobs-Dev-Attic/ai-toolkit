import json
import os
import time
import threading
from typing import Callable, Optional

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a hard dep of the UI stack
    psutil = None

# NVML gives whole-GPU memory usage that matches `nvidia-smi` (i.e. what the user
# actually sees), so it is preferred over torch's process-local allocator counters.
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class SystemStatsLogger:
    """Background thread that samples system resource usage during a training run.

    Each sample is written as one JSON object per line (JSON Lines) to
    ``system_stats.jsonl`` in the job's save folder. This is append friendly (no
    full-file rewrites) and trivially parseable by the UI. Sampling runs on a
    daemon thread so it never blocks training and is torn down automatically if
    the process exits abnormally; every sample is flushed immediately so a crash
    still leaves a valid, up-to-date log.

    Tracked per sample: VRAM (used/total/%), GPU utilization, system RAM
    (used/total/%), this process' RSS, CPU load, and disk usage of the drive
    holding the job folder.
    """

    def __init__(
        self,
        log_file: str,
        device: Optional[object] = None,
        interval: float = 3.0,
        get_step: Optional[Callable[[], Optional[int]]] = None,
    ) -> None:
        self.log_file = log_file
        self.interval = max(0.5, float(interval))
        self._get_step = get_step
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = time.time()
        self._gpu_index = self._resolve_gpu_index(device)
        self._nvml_handle = None
        self._disk_path = os.path.dirname(os.path.abspath(log_file)) or "."

        # Prime the CPU counters so the first real reading is a delta, not 0/garbage.
        self._proc = None
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
                self._proc = psutil.Process()
                self._proc.cpu_percent(interval=None)
            except Exception:
                self._proc = None

    @staticmethod
    def _resolve_gpu_index(device: Optional[object]) -> Optional[int]:
        if device is None:
            return 0  # default to the first GPU
        s = str(device)
        if "cuda" not in s:
            return None  # cpu / mps -> no VRAM tracking
        if ":" in s:
            try:
                return int(s.split(":")[1])
            except Exception:
                return 0
        return 0

    def start(self) -> None:
        if self._thread is not None:
            return

        parent = os.path.dirname(os.path.abspath(self.log_file))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        # Truncate any log from a previous run of this job so the chart reflects
        # the current run.
        try:
            open(self.log_file, "w", encoding="utf-8").close()
        except Exception:
            pass

        if _NVML_OK and self._gpu_index is not None:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            except Exception:
                self._nvml_handle = None

        self._start_time = time.time()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="SystemStatsLogger", daemon=True
        )
        self._thread.start()

    def _sample(self) -> dict:
        now = time.time()
        rec: dict = {
            "t": round(now, 3),
            "elapsed": round(now - self._start_time, 3),
        }

        if self._get_step is not None:
            try:
                rec["step"] = int(self._get_step())
            except Exception:
                rec["step"] = None

        if psutil is not None:
            try:
                rec["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
            except Exception:
                pass
            try:
                vm = psutil.virtual_memory()
                rec["ram_used_mb"] = round(vm.used / (1024 * 1024), 1)
                rec["ram_total_mb"] = round(vm.total / (1024 * 1024), 1)
                rec["ram_percent"] = round(vm.percent, 1)
            except Exception:
                pass
            if self._proc is not None:
                try:
                    rec["proc_ram_mb"] = round(
                        self._proc.memory_info().rss / (1024 * 1024), 1
                    )
                except Exception:
                    pass
            try:
                du = psutil.disk_usage(self._disk_path)
                rec["disk_used_mb"] = round(du.used / (1024 * 1024), 1)
                rec["disk_total_mb"] = round(du.total / (1024 * 1024), 1)
                rec["disk_free_mb"] = round(du.free / (1024 * 1024), 1)
                rec["disk_percent"] = round(du.percent, 1)
            except Exception:
                pass

        if self._nvml_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                rec["vram_used_mb"] = round(mem.used / (1024 * 1024), 1)
                rec["vram_total_mb"] = round(mem.total / (1024 * 1024), 1)
                rec["vram_percent"] = (
                    round(100.0 * mem.used / mem.total, 1) if mem.total else None
                )
            except Exception:
                pass
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                rec["gpu_percent"] = float(util.gpu)
            except Exception:
                pass

        return rec

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rec = self._sample()
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                # never let a sampling hiccup take down training
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2.0)
            self._thread = None


def create_system_stats_logger(
    save_root: str,
    device: Optional[object] = None,
    interval: float = 3.0,
    get_step: Optional[Callable[[], Optional[int]]] = None,
) -> SystemStatsLogger:
    log_file = os.path.join(save_root, "system_stats.jsonl")
    return SystemStatsLogger(
        log_file=log_file, device=device, interval=interval, get_step=get_step
    )
