import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Silence noisy warnings once, on import
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*pynvml.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*DataFrame concatenation with empty or all-NA entries is deprecated.*",
)


class Utils:
    """Utility helpers for seeds, filesystem, time tags, and device selection."""

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def safe_mkdir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def now_tag() -> str:
        """Microsecond-resolution timestamp tag to avoid collisions across processes."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def get_device(gpu_index: int = 0) -> Tuple[torch.device, str]:
        """
        Returns (device, device_type_str).
        If CUDA available, selects the GPU at gpu_index.
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_index)
            dev = torch.device(f"cuda:{gpu_index}")
            return dev, "cuda"
        return torch.device("cpu"), "cpu"
