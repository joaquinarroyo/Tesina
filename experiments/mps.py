"""
Ejecutor de experimentos con NVIDIA MPS (Multi-Process Service).

Permite correr múltiples entrenamientos en paralelo compartiendo la GPU,
midiendo throughput agregado y consumo energético.

Uso típico:
  # Iniciar MPS primero (una sola vez por sesión):
  source setup_mps.sh

  # Ejecutar 2 instancias de E1 en paralelo:
  python -m experiments.mps --exp E1 --parallel 2 --out ./runs --seed 42

  # Ejecutar combinaciones de experimentos:
  python -m experiments.mps --exp E1 E3 --out ./runs --seed 42

  # Con grupo MPS identificador:
  python -m experiments.mps --exp E1 --parallel 3 --mps-group MPS_E1_x3 --out ./runs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Asegura que el repo root esté en sys.path al ejecutar como script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run_single_experiment(
    experiment_id: str,
    base_out_dir: str,
    condition: str,
    seed: int,
    save_plots: bool,
    overrides: Dict[str, Any],
    mps_group: Optional[str],
    gpu_index: int,
    worker_id: int,
) -> Dict[str, Any]:
    """
    Función que ejecuta un solo experimento.
    Se ejecuta en un proceso separado para aprovechar MPS.
    """
    # Importar aquí para evitar problemas con multiprocessing
    from helpers.runner import run_experiment

    print(f"[Worker {worker_id}] Iniciando {experiment_id} con seed={seed}", flush=True)

    result = run_experiment(
        experiment_id=experiment_id,
        base_out_dir=base_out_dir,
        condition=condition,
        seed=seed,
        save_plots=save_plots,
        overrides=overrides,
        mps_group=mps_group,
        gpu_index=gpu_index,
    )

    print(f"[Worker {worker_id}] Completado {experiment_id}", flush=True)
    return result


def check_mps_status() -> bool:
    """Verifica si el daemon MPS está corriendo."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "COMPUTE"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # MPS está activo si encontramos referencia a él
        return "MPS" in result.stdout or os.path.exists("/tmp/mps_pipe")
    except Exception:
        return False


def start_mps() -> bool:
    """Intenta iniciar MPS si no está corriendo."""
    if check_mps_status():
        print("MPS ya está activo.", flush=True)
        return True

    print("Iniciando MPS daemon...", flush=True)
    try:
        # Configurar directorios MPS
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/mps_pipe"
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/mps_log"
        os.makedirs("/tmp/mps_pipe", exist_ok=True)
        os.makedirs("/tmp/mps_log", exist_ok=True)

        # Detener cualquier instancia previa
        subprocess.run(
            "echo quit | nvidia-cuda-mps-control 2>/dev/null || true",
            shell=True,
            timeout=5,
        )
        time.sleep(1)

        # Iniciar daemon
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            timeout=5,
            check=True,
        )
        time.sleep(2)

        if check_mps_status():
            print("MPS iniciado correctamente.", flush=True)
            return True
        else:
            print("Advertencia: No se pudo verificar MPS, continuando de todas formas.", flush=True)
            return True
    except Exception as e:
        print(f"Error iniciando MPS: {e}", flush=True)
        print("Continuando sin MPS verificado (puede funcionar si ya está activo).", flush=True)
        return True


def run_mps_experiments(
    experiments: List[str],
    parallel: int,
    base_out_dir: str,
    base_seed: int,
    mps_group: Optional[str],
    gpu_index: int,
    save_plots: bool,
    overrides: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Ejecuta múltiples experimentos en paralelo usando MPS.

    Args:
        experiments: Lista de IDs de experimentos (pueden repetirse)
        parallel: Número de procesos paralelos
        base_out_dir: Directorio de salida
        base_seed: Seed base (se incrementa por cada experimento)
        mps_group: Identificador del grupo MPS
        gpu_index: Índice de GPU
        save_plots: Si guardar gráficos
        overrides: Overrides de configuración

    Returns:
        Lista de resultados de cada experimento
    """
    # Generar grupo MPS si no se especifica
    if mps_group is None:
        from helpers.utils import Utils
        mps_group = f"MPS_{Utils.now_tag()}"

    print("=" * 70, flush=True)
    print(f"Ejecutando {len(experiments)} experimentos en paralelo (max {parallel} workers)", flush=True)
    print(f"Experimentos: {experiments}", flush=True)
    print(f"MPS Group: {mps_group}", flush=True)
    print("=" * 70, flush=True)

    # Preparar trabajos
    jobs = []
    for i, exp_id in enumerate(experiments):
        jobs.append({
            "experiment_id": exp_id,
            "base_out_dir": base_out_dir,
            "condition": "mps",
            "seed": base_seed + i,
            "save_plots": save_plots,
            "overrides": overrides,
            "mps_group": mps_group,
            "gpu_index": gpu_index,
            "worker_id": i,
        })

    results = []
    start_time = time.time()

    # Ejecutar en paralelo
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(run_single_experiment, **job): job
            for job in jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✓ Completado: {job['experiment_id']} (worker {job['worker_id']})", flush=True)
            except Exception as e:
                print(f"✗ Error en {job['experiment_id']}: {e}", flush=True)
                results.append({"error": str(e), "experiment_id": job["experiment_id"]})

    total_time = time.time() - start_time

    print("=" * 70, flush=True)
    print(f"Todos los experimentos completados en {total_time:.2f}s", flush=True)
    print(f"Tiempo promedio por experimento: {total_time/len(experiments):.2f}s", flush=True)
    print(f"Resultados guardados en: {base_out_dir}", flush=True)
    print("=" * 70, flush=True)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ejecuta experimentos en paralelo usando NVIDIA MPS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # 2 instancias de E1 en paralelo
  python -m experiments.mps --exp E1 --parallel 2

  # E1 y E3 en paralelo
  python -m experiments.mps --exp E1 E3

  # 3 instancias de E3 con grupo específico
  python -m experiments.mps --exp E3 --parallel 3 --mps-group MPS_E3_x3
        """,
    )

    p.add_argument(
        "--exp",
        nargs="+",
        required=True,
        help="IDs de experimentos a ejecutar (E1..E6). Puede repetirse.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Número de instancias paralelas. Por defecto igual al número de experimentos.",
    )
    p.add_argument("--out", default="./runs", help="Directorio base para artifacts.")
    p.add_argument("--seed", type=int, default=42, help="Seed base.")
    p.add_argument("--gpu-index", type=int, default=0, help="GPU index.")
    p.add_argument("--mps-group", default=None, help="Etiqueta para agrupar runs MPS.")
    p.add_argument("--no-plots", action="store_true", help="No guardar PNGs.")
    p.add_argument("--no-mps-check", action="store_true", help="No verificar/iniciar MPS.")
    p.add_argument("--repeat", type=int, default=1, help="Repetir el conjunto de experimentos N veces.")

    # Overrides
    p.add_argument("--override-batch-size", type=int, default=None)
    p.add_argument("--override-epochs", type=int, default=None)
    p.add_argument("--override-lr", type=float, default=None)
    p.add_argument("--override-num-workers", type=int, default=None)
    p.add_argument("--override-amp", type=int, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Verificar/iniciar MPS
    if not args.no_mps_check:
        start_mps()

    # Expandir experimentos:
    # - Si --parallel N y solo 1 exp: crear N copias del mismo experimento
    # - Si --repeat N: repetir toda la lista N veces
    experiments = args.exp * args.repeat
    
    # Si se especifica --parallel y es mayor que el número de experimentos,
    # replicar para tener exactamente parallel instancias
    if args.parallel and args.parallel > len(experiments):
        # Replicar cíclicamente hasta tener parallel experimentos
        base_exps = experiments.copy()
        while len(experiments) < args.parallel:
            experiments.extend(base_exps)
        experiments = experiments[:args.parallel]

    # Determinar paralelismo
    parallel = args.parallel if args.parallel else len(experiments)

    # Construir overrides
    overrides = {
        "batch_size": args.override_batch_size,
        "epochs": args.override_epochs,
        "lr": args.override_lr,
        "num_workers": args.override_num_workers,
    }
    if args.override_amp is not None:
        overrides["amp"] = bool(int(args.override_amp))

    # Ejecutar
    run_mps_experiments(
        experiments=experiments,
        parallel=parallel,
        base_out_dir=args.out,
        base_seed=args.seed,
        mps_group=args.mps_group,
        gpu_index=args.gpu_index,
        save_plots=not args.no_plots,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
