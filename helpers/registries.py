from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T


class SimpleCNN(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], n_classes: int):
        super().__init__()
        c, h, w = in_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Registries:
    RUNS: Dict[str, Dict[str, object]] = {
        "E1": {"dataset": "mnist", "model": "simplecnn", "batch_size": 256, "epochs": 3, "lr": 1e-3, "notes": "Sanity check"},
        "E2": {"dataset": "cifar10", "model": "mobilenet_v3_small", "batch_size": 256, "epochs": 5, "lr": 1e-3, "notes": "Modelo liviano"},
        "E3": {"dataset": "cifar10", "model": "resnet18", "batch_size": 128, "epochs": 5, "lr": 1e-3, "notes": "Baseline balanceado"},
        "E4": {"dataset": "cifar100", "model": "resnet18", "batch_size": 128, "epochs": 5, "lr": 1e-3, "notes": "Más clases"},
        "E5": {"dataset": "cifar10", "model": "resnet50", "batch_size": 64, "epochs": 5, "lr": 1e-3, "notes": "Compute-heavy"},
        "E6": {"dataset": "flowers102", "model": "resnet50", "batch_size": 32, "epochs": 5, "lr": 1e-3, "notes": "Memory-bound"},
    }

    @staticmethod
    def get_model(name: str, n_classes: int, in_shape: Tuple[int, int, int]) -> nn.Module:
        name = name.lower()

        if name == "simplecnn":
            return SimpleCNN(in_shape, n_classes)

        if name == "resnet18":
            m = torchvision.models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, n_classes)
            return m

        if name == "resnet50":
            m = torchvision.models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, n_classes)
            return m

        if name == "mobilenet_v3_small":
            m = torchvision.models.mobilenet_v3_small(weights=None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
            return m

        raise ValueError(f"Modelo no soportado: {name}")

    @staticmethod
    def get_dataset(name: str, root: str = "./data"):
        name = name.lower()

        if name == "mnist":
            tfm = T.Compose([T.ToTensor()])
            train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
            test = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
            return train, test, 10, (1, 28, 28)

        if name == "cifar10":
            tfm_train = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            tfm_test = T.Compose([T.ToTensor()])
            train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm_train)
            test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm_test)
            return train, test, 10, (3, 32, 32)

        if name == "cifar100":
            tfm_train = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            tfm_test = T.Compose([T.ToTensor()])
            train = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=tfm_train)
            test = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=tfm_test)
            return train, test, 100, (3, 32, 32)

        if name == "flowers102":
            tfm_train = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            tfm_test = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ])
            train = torchvision.datasets.Flowers102(root, split="train", download=True, transform=tfm_train)
            test = torchvision.datasets.Flowers102(root, split="val", download=True, transform=tfm_test)
            return train, test, 102, (3, 224, 224)

        raise ValueError(f"Dataset no soportado: {name}")


