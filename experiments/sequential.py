"""
Ejecutor secuencial de experimentos, envolviendo helpers.runner.run_experiment.

Uso típico (equivalente a la línea solicitada):
  python3 experiments/sequential.py --exp E3 --out ./runs --repeat 2
"""

import argparse
import sys
from pathlib import Path


# Asegura que el repo root esté en sys.path al ejecutar como script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from helpers.runner import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Lanza experimentos en modo sequential.")
	p.add_argument("--exp", default="E3", help="ID del experimento (E1..E6).")
	p.add_argument("--out", default="./runs", help="Directorio base para artifacts.")
	p.add_argument("--repeat", type=int, default=2, help="Cantidad de repeticiones.")
	p.add_argument("--seed", type=int, default=42, help="Seed base (se incrementa por repetición).")
	p.add_argument("--gpu-index", type=int, default=0, help="GPU index para torch/NVML.")
	p.add_argument("--no-plots", action="store_true", help="No guardar PNGs.")

	# Overrides opcionales para mantener compatibilidad con runner.py
	p.add_argument("--override-batch-size", type=int, default=None)
	p.add_argument("--override-epochs", type=int, default=None)
	p.add_argument("--override-lr", type=float, default=None)
	p.add_argument("--override-num-workers", type=int, default=None)
	p.add_argument("--override-amp", type=int, default=None, help="1 para AMP, 0 para no AMP.")
	return p.parse_args()


def main() -> None:
	args = parse_args()

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
			condition="sequential",
			seed=args.seed + i,
			save_plots=(not args.no_plots),
			overrides=overrides,
			mps_group=None,
			gpu_index=args.gpu_index,
		)


if __name__ == "__main__":
	main()
