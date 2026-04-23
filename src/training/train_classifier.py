from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABEL_TO_INDEX = {"no-yawn": 0, "yawn": 1}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
MODEL_NAMES = {"resnet18", "mobilenet_v2", "efficientnet_b0"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class MouthCropDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["processed_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = LABEL_TO_INDEX[row["label"]]
        return image, torch.tensor(label, dtype=torch.long)


def read_split(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        raise SystemExit(f"Split manifest not found: {path}")
    rows_by_split = {"train": [], "val": [], "test": []}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") in rows_by_split and row.get("label") in LABEL_TO_INDEX:
                if Path(row["processed_path"]).exists():
                    rows_by_split[row["split"]].append(row)
    if not all(rows_by_split.values()):
        counts = {split: len(rows) for split, rows in rows_by_split.items()}
        raise SystemExit(f"Split manifest must contain existing samples for train/val/test. Found: {counts}")
    return rows_by_split


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }


def build_model(model_name: str, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    raise ValueError(f"Unsupported model name: {model_name}")


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad = trainable
    if model_name == "resnet18":
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True


def class_weights(rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = np.bincount([LABEL_TO_INDEX[row["label"]] for row in rows], minlength=2)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(model, dataloader, criterion, optimizer, device: torch.device, train: bool) -> tuple[float, float]:
    model.train(train)
    losses = []
    accuracies = []
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        accuracies.append(accuracy_from_logits(logits.detach(), labels))
    return float(np.mean(losses)), float(np.mean(accuracies))


@torch.no_grad()
def collect_predictions(model, dataloader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in dataloader:
        logits = model(images.to(device))
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return y_true, y_pred


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(rows_by_split, batch_size: int, image_size: int, num_workers: int) -> dict[str, DataLoader]:
    transforms_by_split = build_transforms(image_size)
    datasets = {
        "train": MouthCropDataset(rows_by_split["train"], transform=transforms_by_split["train"]),
        "val": MouthCropDataset(rows_by_split["val"], transform=transforms_by_split["eval"]),
        "test": MouthCropDataset(rows_by_split["test"], transform=transforms_by_split["eval"]),
    }
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }


def plot_history(history: list[dict[str, float]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_acc"] * 100 for row in history], label="Train accuracy")
    plt.plot(epochs, [row["val_acc"] * 100 for row in history], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion(cm: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], [INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]])
    plt.yticks([0, 1], [INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]])
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            plt.text(x, y, str(cm[y, x]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_one(args: argparse.Namespace) -> dict[str, object]:
    rows_by_split = read_split(args.split_manifest)
    device = select_device()
    batch_size = args.batch_size

    try:
        loaders = make_loaders(rows_by_split, batch_size, args.image_size, args.num_workers)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and batch_size > 16:
            batch_size = 16
            loaders = make_loaders(rows_by_split, batch_size, args.image_size, args.num_workers)
        else:
            raise

    try:
        model = build_model(args.model, pretrained=not args.no_pretrained)
    except Exception as exc:  # noqa: BLE001 - offline environments may not have cached weights.
        print(f"Pretrained weights unavailable for {args.model}: {exc}. Falling back to random initialization.")
        model = build_model(args.model, pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(rows_by_split["train"], device))
    set_backbone_trainable(model, args.model, trainable=False)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)

    best_state = deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, args.model, trainable=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)

        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"{args.model} epoch {epoch:02d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / f"{args.model}_best.pt")

    metrics: dict[str, object] = {
        "model": args.model,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "duration_seconds": round(time.time() - start, 2),
    }
    for split in ["train", "val", "test"]:
        y_true, y_pred = collect_predictions(model, loaders[split], device)
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0
        )
        metrics[f"{split}_accuracy"] = acc
        metrics[f"{split}_precision"] = float(precision)
        metrics[f"{split}_recall"] = float(recall)
        metrics[f"{split}_f1"] = float(f1)
        if split == "test":
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            metrics["test_confusion_matrix"] = cm.tolist()
            plot_confusion(cm, args.figures_dir / f"{args.model}_confusion_matrix.png", f"{args.model} Test Confusion Matrix")

    plot_history(history, args.figures_dir / f"{args.model}_training_curve.png", f"{args.model} Training Curve")
    (args.output_dir / f"{args.model}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / f"{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one initial CNN baseline for YawDD+ mouth crops.")
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--split-manifest", type=Path, default=repo_root() / "artifacts" / "splits" / "yawdd_dash_subject_split.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "artifacts" / "results")
    parser.add_argument("--figures-dir", type=Path, default=repo_root() / "artifacts" / "figures")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_one(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
