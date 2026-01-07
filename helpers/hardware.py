import os
import threading
import time
from typing import Any, Dict, Optional

import pandas as pd
import torch

_NVML_OK = False
pynvml: Optional[object] = None

try:
    import nvidia_ml_py as pynvml  # type: ignore
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        _NVML_OK = True
    except Exception:
        _NVML_OK = False


class HardwareInfo:
    """Environment metadata helpers (torch, CUDA, NVML)."""

    @staticmethod
    def get_env_info(gpu_index: int = 0) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            "python_pid": os.getpid(),
        }

        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(gpu_index)
            except Exception:
                info["gpu_name"] = None
            try:
                props = torch.cuda.get_device_properties(gpu_index)
                info["gpu_total_mem_mb"] = float(props.total_memory / (1024**2))
                info["gpu_sm_count"] = int(getattr(props, "multi_processor_count", -1))
            except Exception:
                info["gpu_total_mem_mb"] = None
                info["gpu_sm_count"] = None
        else:
            info["gpu_name"] = None
            info["gpu_total_mem_mb"] = None
            info["gpu_sm_count"] = None

        if _NVML_OK:
            try:
                drv = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(drv, (bytes, bytearray)):
                    drv = drv.decode("utf-8", errors="ignore")
                info["nv_driver_version"] = drv
            except Exception:
                info["nv_driver_version"] = None
        else:
            info["nv_driver_version"] = None

        return info


class GPUMonitor:
    """Samples GPU utilization, power, and memory using NVML when available."""

    def __init__(self, interval_s: float = 0.2, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self.samples = []
        self._handle = None

        if _NVML_OK and torch.cuda.is_available():
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._handle = None

    def start(self) -> None:
        self._stop.clear()
        self._t0 = time.time()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if hasattr(self, "_thr"):
            self._thr.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            ts = time.time() - self._t0

            util = None
            pwr_w = None
            mem_used_mb = None

            if self._handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                    pwr_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    mem_used_mb = mem.used / (1024**2)
                except Exception:
                    pass

            self.samples.append({
                "t": ts,
                "gpu_util_pct": util,
                "power_w": pwr_w,
                "gpu_mem_used_mb": mem_used_mb,
            })
            time.sleep(self.interval_s)

    def summary(self) -> Dict[str, Any]:
        df = pd.DataFrame(self.samples)
        out: Dict[str, Any] = {"n_samples": int(len(df))}
        if len(df) == 0:
            return out

        for col in ["gpu_util_pct", "power_w", "gpu_mem_used_mb"]:
            if col in df.columns and df[col].notna().any():
                s = df[col].dropna()
                out[f"{col}_mean"] = float(s.mean())
                out[f"{col}_p95"] = float(s.quantile(0.95))
                out[f"{col}_max"] = float(s.max())

        if "power_w" in df.columns and df["power_w"].notna().any():
            dfx = df.dropna(subset=["t", "power_w"]).sort_values("t")
            if len(dfx) >= 2:
                dt = dfx["t"].values[1:] - dfx["t"].values[:-1]
                pw = (dfx["power_w"].values[1:] + dfx["power_w"].values[:-1]) * 0.5
                out["energy_j_approx"] = float((pw * dt).sum())

        return out
