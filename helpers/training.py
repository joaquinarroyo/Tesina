import time
from dataclasses import dataclass, asdict
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .hardware import GPUMonitor
from .registries import Registries
from .utils import Utils


@dataclass
class RunConfig:
    dataset: str
    model: str
    batch_size: int = 128
    epochs: int = 3
    lr: float = 1e-3
    num_workers: int = 2
    amp: bool = True


class Trainer:
    """High-level training orchestrator built around RunConfig."""

    def __init__(self, cfg: RunConfig, monitor_interval_s: float = 0.2, gpu_index: int = 0):
        self.cfg = cfg
        self.monitor_interval_s = monitor_interval_s
        self.gpu_index = gpu_index

    @staticmethod
    def accuracy_top1(logits, y):
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()

    def _train_one_epoch(self, model, loader, optimizer, device, device_type, scaler=None) -> Dict[str, float]:
        model.train()
        crit = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0

        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(device_type):
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += self.accuracy_top1(logits.detach(), y) * bs
            n_samples += bs

        dt = time.time() - t0
        return {
            "train_loss": total_loss / n_samples,
            "train_acc": total_acc / n_samples,
            "train_time_s": dt,
            "train_throughput_samples_s": n_samples / dt,
            "train_samples_epoch": n_samples,
        }

    def _eval_one_epoch(self, model, loader, device) -> Dict[str, float]:
        model.eval()
        crit = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0

        t0 = time.time()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = crit(logits, y)

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_acc += self.accuracy_top1(logits, y) * bs
                n_samples += bs

        dt = time.time() - t0
        return {
            "val_loss": total_loss / n_samples,
            "val_acc": total_acc / n_samples,
            "val_time_s": dt,
            "val_throughput_samples_s": n_samples / dt,
            "val_samples_epoch": n_samples,
        }

    def run(self) -> Dict[str, Any]:
        device, device_type = Utils.get_device(self.gpu_index)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        train_ds, val_ds, n_classes, in_shape = Registries.get_dataset(self.cfg.dataset)
        model = Registries.get_model(self.cfg.model, n_classes=n_classes, in_shape=in_shape).to(device)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        optimizer = optim.AdamW(model.parameters(), lr=self.cfg.lr)
        scaler = torch.amp.GradScaler("cuda") if (self.cfg.amp and device.type == "cuda") else None

        gpu_mon = GPUMonitor(interval_s=self.monitor_interval_s, gpu_index=self.gpu_index)
        gpu_mon.start()

        per_epoch = []
        t0 = time.time()
        for epoch in range(self.cfg.epochs):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            e_train = self._train_one_epoch(model, train_loader, optimizer, device, device_type, scaler=scaler)
            e_val = self._eval_one_epoch(model, val_loader, device)

            e: Dict[str, Any] = {"epoch": epoch + 1, **e_train, **e_val}
            if device.type == "cuda":
                e["gpu_max_mem_alloc_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
                e["gpu_max_mem_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024**2)
            per_epoch.append(e)

            print(
                f"[{self.cfg.dataset} | {self.cfg.model}] epoch {epoch+1}/{self.cfg.epochs} "
                f"train_acc={e['train_acc']:.3f} val_acc={e['val_acc']:.3f} "
                f"train_thr={e['train_throughput_samples_s']:.1f} samp/s",
                flush=True,
            )

        total_time = time.time() - t0
        gpu_mon.stop()

        train_size = len(train_ds)
        val_size = len(val_ds)
        train_samples_total = int(train_size * self.cfg.epochs)

        return {
            "config": asdict(self.cfg),
            "total_time_s": total_time,
            "epochs": per_epoch,
            "gpu_monitor": gpu_mon.summary(),
            "gpu_samples": gpu_mon.samples,
            "train_size": int(train_size),
            "val_size": int(val_size),
            "train_samples_total": int(train_samples_total),
        }
