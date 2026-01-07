#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runner depurado para Colab / subprocess / MPS.

- Ejecuta 1 experimento por proceso (ideal con CUDA MPS)
- Guarda artifacts por run en carpeta única (incluye microsegundos + pid)
- Guarda logs/plots sin imprimir plots
- Guarda metadata de entorno y métricas energéticas

Uso típico:
  python3 tesis.py --list
  python3 tesis.py --exp E3 --condition sequential --out ./runs --repeat 3
  python3 tesis.py --exp E3 --condition mps --mps-group G1 --out ./runs --seed 1000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from helpers.artifacts import ArtifactManager
from helpers.hardware import HardwareInfo
from helpers.registries import Registries
from helpers.training import RunConfig, Trainer
from helpers.utils import Utils


def run_experiment(
    experiment_id: str,
    base_out_dir: str | Path = "./runs",
    condition: str = "sequential",
    seed: int = 42,
    save_plots: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    mps_group: Optional[str] = None,
    gpu_index: int = 0,
) -> Dict[str, Any]:
    if experiment_id not in Registries.RUNS:
        raise ValueError(f"Experimento '{experiment_id}' no definido. Usa --list para ver opciones.")

    cfg0 = dict(Registries.RUNS[experiment_id])
    if overrides:
        cfg0.update({k: v for k, v in overrides.items() if v is not None})

    run_cfg = RunConfig(
        dataset=cfg0["dataset"],
        model=cfg0["model"],
        batch_size=int(cfg0["batch_size"]),
        epochs=int(cfg0["epochs"]),
        lr=float(cfg0.get("lr", 1e-3)),
        num_workers=int(cfg0.get("num_workers", 2)),
        amp=bool(cfg0.get("amp", True)),
    )

    pid = os.getpid()
    tag = f"{experiment_id}_{run_cfg.dataset}_{run_cfg.model}_{condition}_{Utils.now_tag()}_pid{pid}"
    out_dir = Path(base_out_dir) / tag
    Utils.safe_mkdir(out_dir)

    print("=" * 70, flush=True)
    print(f"Ejecutando {experiment_id} | condition={condition} | seed={seed} | pid={pid}", flush=True)
    if mps_group:
        print(f"mps_group: {mps_group}", flush=True)
    print(f"Dataset: {run_cfg.dataset} | Modelo: {run_cfg.model} | bs={run_cfg.batch_size} | epochs={run_cfg.epochs}", flush=True)
    print(f"Salida: {out_dir}", flush=True)
    print("=" * 70, flush=True)

    Utils.set_seed(seed)
    env_info = HardwareInfo.get_env_info(gpu_index=gpu_index)
    result = Trainer(run_cfg, gpu_index=gpu_index).run()

    result["experiment_id"] = experiment_id
    result["condition"] = condition
    result["mps_group"] = mps_group
    result["seed"] = seed
    result["pid"] = pid
    result["experiment_notes"] = cfg0.get("notes", "")
    result["artifacts_dir"] = str(out_dir)
    result["env"] = env_info

    ArtifactManager.save_run_artifacts(result, out_dir, also_append_master_csv=True)
    if save_plots:
        ArtifactManager.plot_standard_charts(result, out_dir, save_png=True)

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Runner de entrenamientos (baseline/MPS) para tesina.")
    p.add_argument("--exp", help="ID del experimento (E1..E6).")
    p.add_argument("--list", action="store_true", help="Listar experimentos disponibles y salir.")
    p.add_argument("--out", default="./runs", help="Directorio base para artifacts.")
    p.add_argument("--condition", default="sequential", help="Etiqueta de condición (sequential/mps/ray/...).")
    p.add_argument("--mps-group", default=None, help="Etiqueta para agrupar runs concurrentes (misma para procesos en paralelo).")
    p.add_argument("--gpu-index", type=int, default=0, help="GPU index (para NVML/torch). En Colab normalmente 0.")
    p.add_argument("--seed", type=int, default=42, help="Seed base.")
    p.add_argument("--repeat", type=int, default=1, help="Cantidad de repeticiones (incrementa seed).")
    p.add_argument("--no-plots", action="store_true", help="No guardar PNGs.")

    p.add_argument("--override-batch-size", type=int, default=None)
    p.add_argument("--override-epochs", type=int, default=None)
    p.add_argument("--override-lr", type=float, default=None)
    p.add_argument("--override-num-workers", type=int, default=None)
    p.add_argument("--override-amp", type=int, default=None, help="1 para AMP, 0 para no AMP.")

    args = p.parse_args()

    if args.list:
        for k, v in Registries.RUNS.items():
            print(
                f"{k}: dataset={v['dataset']}, model={v['model']}, "
                f"bs={v['batch_size']}, epochs={v['epochs']}, notes={v.get('notes','')}"
            )
        return

    if not args.exp:
        raise SystemExit("Falta --exp (o usar --list).")

    overrides = {
        "batch_size": args.override_batch_size,
        "epochs": args.override_epochs,
        "lr": args.override_lr,
        "num_workers": args.override_num_workers,
    }
    if args.override_amp is not None:
        overrides["amp"] = bool(int(args.override_amp))

    for i in range(args.repeat):
        run_experiment(
            experiment_id=args.exp,
            base_out_dir=args.out,
            condition=args.condition,
            seed=args.seed + i,
            save_plots=(not args.no_plots),
            overrides=overrides,
            mps_group=args.mps_group,
            gpu_index=args.gpu_index,
        )


if __name__ == "__main__":
    main()