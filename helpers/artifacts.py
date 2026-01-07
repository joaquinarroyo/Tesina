import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .utils import Utils


class ArtifactManager:
    """Handles saving artifacts and plotting standard charts for runs."""

    @staticmethod
    def save_run_artifacts(
        result: dict,
        out_dir: str | Path,
        also_append_master_csv: bool = True,
    ) -> Dict[str, str]:
        out_dir = Path(out_dir)
        Utils.safe_mkdir(out_dir)

        with open(out_dir / "run.json", "w") as f:
            json.dump(result, f, indent=2)

        epochs_df = pd.DataFrame(result.get("epochs", []))
        epochs_df.to_csv(out_dir / "epochs.csv", index=False)

        gpu_samples = result.get("gpu_samples", None)
        if isinstance(gpu_samples, list) and len(gpu_samples) > 0:
            pd.DataFrame(gpu_samples).to_csv(out_dir / "gpu_samples.csv", index=False)

        cfg = result.get("config", {})
        gpu_sum = result.get("gpu_monitor", {})
        last_epoch = result.get("epochs", [])[-1] if result.get("epochs") else {}
        env = result.get("env", {})

        energy_j = gpu_sum.get("energy_j_approx", None)
        train_samples_total = result.get("train_samples_total", None)
        energy_per_sample_j = None
        samples_per_j = None
        if energy_j is not None and train_samples_total is not None:
            try:
                energy_j = float(energy_j)
                train_samples_total = float(train_samples_total)
                if energy_j > 0:
                    energy_per_sample_j = train_samples_total / energy_j
                    samples_per_j = train_samples_total / energy_j
            except Exception:
                energy_per_sample_j = None
                samples_per_j = None

        j_per_sample = None
        if energy_j is not None and train_samples_total is not None:
            try:
                energy_j = float(energy_j)
                train_samples_total = float(train_samples_total)
                if train_samples_total > 0:
                    j_per_sample = energy_j / train_samples_total
            except Exception:
                j_per_sample = None

        summary_row = {
            "experiment_id": result.get("experiment_id", ""),
            "dataset": cfg.get("dataset", ""),
            "model": cfg.get("model", ""),
            "condition": result.get("condition", "sequential"),
            "mps_group": result.get("mps_group", None),
            "seed": result.get("seed", None),
            "pid": result.get("pid", None),
            "run_dir": out_dir.name,
            "artifacts_dir": str(out_dir),
            "batch_size": cfg.get("batch_size", None),
            "epochs": cfg.get("epochs", None),
            "lr": cfg.get("lr", None),
            "amp": cfg.get("amp", None),
            "num_workers": cfg.get("num_workers", None),
            "train_size": result.get("train_size", None),
            "val_size": result.get("val_size", None),
            "train_samples_total": result.get("train_samples_total", None),
            "total_time_s": result.get("total_time_s", None),
            "final_train_loss": last_epoch.get("train_loss", None),
            "final_train_acc": last_epoch.get("train_acc", None),
            "final_val_loss": last_epoch.get("val_loss", None),
            "final_val_acc": last_epoch.get("val_acc", None),
            "final_train_throughput_sps": last_epoch.get("train_throughput_samples_s", None),
            "final_val_throughput_sps": last_epoch.get("val_throughput_samples_s", None),
            "final_gpu_max_mem_alloc_mb": last_epoch.get("gpu_max_mem_alloc_mb", None),
            "final_gpu_max_mem_reserved_mb": last_epoch.get("gpu_max_mem_reserved_mb", None),
            "gpu_util_pct_mean": gpu_sum.get("gpu_util_pct_mean", None),
            "gpu_util_pct_p95": gpu_sum.get("gpu_util_pct_p95", None),
            "gpu_util_pct_max": gpu_sum.get("gpu_util_pct_max", None),
            "power_w_mean": gpu_sum.get("power_w_mean", None),
            "power_w_p95": gpu_sum.get("power_w_p95", None),
            "power_w_max": gpu_sum.get("power_w_max", None),
            "energy_j_approx": gpu_sum.get("energy_j_approx", None),
            "energy_j_per_sample": j_per_sample,
            "samples_per_j": samples_per_j,
            "gpu_mem_used_mb_mean": gpu_sum.get("gpu_mem_used_mb_mean", None),
            "gpu_mem_used_mb_p95": gpu_sum.get("gpu_mem_used_mb_p95", None),
            "gpu_mem_used_mb_max": gpu_sum.get("gpu_mem_used_mb_max", None),
            "gpu_name": env.get("gpu_name", None),
            "gpu_total_mem_mb": env.get("gpu_total_mem_mb", None),
            "gpu_sm_count": env.get("gpu_sm_count", None),
            "torch_version": env.get("torch_version", None),
            "cuda_version": env.get("cuda_version", None),
            "cudnn_version": env.get("cudnn_version", None),
            "nv_driver_version": env.get("nv_driver_version", None),
        }

        summary_df = pd.DataFrame([summary_row])
        summary_df.to_csv(out_dir / "summary_row.csv", index=False)

        if also_append_master_csv:
            master_path = out_dir.parent / "master_runs.csv"
            if master_path.exists():
                try:
                    master_df = pd.read_csv(master_path)
                except Exception:
                    master_df = pd.DataFrame()
            else:
                master_df = pd.DataFrame()

            if master_df.empty:
                master_df = summary_df.copy()
            else:
                master_df = pd.concat([master_df, summary_df], ignore_index=True)

            master_df.to_csv(master_path, index=False)

        return {"out_dir": str(out_dir)}

    @staticmethod
    def plot_standard_charts(result: dict, out_dir: str | Path, save_png: bool = True) -> None:
        if not save_png:
            return

        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        Utils.safe_mkdir(out_dir)

        epochs = result.get("epochs", [])
        if not epochs:
            return

        df = pd.DataFrame(epochs)
        x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

        def _finish(fig, name):
            fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        fig = plt.figure()
        if "train_loss" in df:
            plt.plot(x, df["train_loss"], label="train_loss")
        if "val_loss" in df:
            plt.plot(x, df["val_loss"], label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch"); plt.legend()
        _finish(fig, "loss_vs_epoch")

        fig = plt.figure()
        if "train_acc" in df:
            plt.plot(x, df["train_acc"], label="train_acc")
        if "val_acc" in df:
            plt.plot(x, df["val_acc"], label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch"); plt.legend()
        _finish(fig, "acc_vs_epoch")

        fig = plt.figure()
        if "train_throughput_samples_s" in df:
            plt.plot(x, df["train_throughput_samples_s"], label="train_samp/s")
        if "val_throughput_samples_s" in df:
            plt.plot(x, df["val_throughput_samples_s"], label="val_samp/s")
        plt.xlabel("Epoch"); plt.ylabel("Samples/s"); plt.title("Throughput vs Epoch"); plt.legend()
        _finish(fig, "throughput_vs_epoch")

        if "gpu_max_mem_alloc_mb" in df.columns or "gpu_max_mem_reserved_mb" in df.columns:
            fig = plt.figure()
            if "gpu_max_mem_alloc_mb" in df:
                plt.plot(x, df["gpu_max_mem_alloc_mb"], label="max_alloc_mb")
            if "gpu_max_mem_reserved_mb" in df:
                plt.plot(x, df["gpu_max_mem_reserved_mb"], label="max_reserved_mb")
            plt.xlabel("Epoch"); plt.ylabel("MB"); plt.title("GPU VRAM (peak) vs Epoch"); plt.legend()
            _finish(fig, "vram_vs_epoch")

        gpu_samples = result.get("gpu_samples", None)
        if isinstance(gpu_samples, list) and len(gpu_samples) > 0:
            gdf = pd.DataFrame(gpu_samples)
            if "t" in gdf.columns and "gpu_util_pct" in gdf.columns and gdf["gpu_util_pct"].notna().any():
                fig = plt.figure()
                plt.plot(gdf["t"], gdf["gpu_util_pct"])
                plt.xlabel("t (s)"); plt.ylabel("GPU util (%)"); plt.title("GPU utilization over time")
                _finish(fig, "gpu_util_over_time")

            if "t" in gdf.columns and "power_w" in gdf.columns and gdf["power_w"].notna().any():
                fig = plt.figure()
                plt.plot(gdf["t"], gdf["power_w"])
                plt.xlabel("t (s)"); plt.ylabel("Power (W)"); plt.title("GPU power over time")
                _finish(fig, "gpu_power_over_time")
