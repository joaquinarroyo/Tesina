from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


def _load_master(master_csv: str = "./runs/master_runs.csv") -> pd.DataFrame:
    p = Path(master_csv)
    if not p.exists():
        raise FileNotFoundError(f"No existe {p}. ¿Ya corriste algún experimento?")
    df = pd.read_csv(p)

    # normalizar nombres esperados (por si cambiaste columnas)
    needed = ["experiment_id", "condition", "total_time_s", "final_train_throughput_sps", "artifacts_dir"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en master_runs.csv: {missing}")
    return df


def _extract_run_tag(artifacts_dir: str) -> str:
    # artifacts_dir = ./runs/E3_cifar10_resnet18_mps_20260106_001346_123456_pid9999
    # devolvemos el nombre de carpeta
    return Path(str(artifacts_dir)).name


def _extract_datetime_key_from_dirname(dirname: str) -> str | None:
    """
    Intenta extraer un key de tiempo del dirname:
    soporta YYYYMMDD_HHMMSS o YYYYMMDD_HHMMSS_micro.
    """
    m = re.search(r"_(\d{8}_\d{6}(?:_\d{1,6})?)", dirname)
    return m.group(1) if m else None


def to_markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)