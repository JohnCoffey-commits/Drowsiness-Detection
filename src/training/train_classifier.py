from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = PROJECT_ROOT / "artifacts" / "cache" / "matplotlib"
TORCH_CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "torch"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABEL_TO_INDEX = {"no_yawn": 0, "yawn": 1}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
LABEL_ALIASES = {"no-yawn": "no_yawn", "no_yawn": "no_yawn", "yawn": "yawn"}
MODEL_NAMES = {"resnet18", "mobilenet_v2", "efficientnet_b0"}


def repo_root() -> Path:
    return PROJECT_ROOT


def canonical_label(label: str) -> str:
    try:
        return LABEL_ALIASES[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported label {label!r}; expected one of {sorted(LABEL_TO_INDEX)}") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    skipped_bad_label = 0
    skipped_missing_file = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split")
            if split not in rows_by_split:
                continue
            try:
                label = canonical_label(row.get("label", ""))
            except ValueError:
                skipped_bad_label += 1
                continue
            processed_path = row.get("processed_path") or row.get("mouth_crop_path")
            if not processed_path or not Path(processed_path).is_file():
                skipped_missing_file += 1
                continue
            normalized_row = dict(row)
            normalized_row["label"] = label
            normalized_row["processed_path"] = processed_path
            rows_by_split[split].append(normalized_row)
    if not all(rows_by_split.values()):
        counts = {split: len(rows) for split, rows in rows_by_split.items()}
        raise SystemExit(f"Split manifest must contain existing samples for train/val/test. Found: {counts}")
    if skipped_bad_label or skipped_missing_file:
        print(
            "Skipped rows while reading split manifest: "
            f"bad_label={skipped_bad_label}, missing_file={skipped_missing_file}"
        )
    return rows_by_split


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomRotation(8),
                transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
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


def correct_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


def run_epoch(model, dataloader, criterion, optimizer, device: torch.device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
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
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct_from_logits(logits.detach(), labels)
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


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
        "train_eval": MouthCropDataset(rows_by_split["train"], transform=transforms_by_split["eval"]),
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_acc"] * 100 for row in history], label="Train")
    axes[0].plot(epochs, [row["val_acc"] * 100 for row in history], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.suptitle(title)
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
    set_seed(args.seed)
    torch.hub.set_dir(str(TORCH_CACHE_DIR))
    rows_by_split = read_split(args.split_manifest)
    device = select_device()
    if device.type == "cpu":
        cpu_threads = args.cpu_threads or (os.cpu_count() or 1)
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(min(4, cpu_threads))
    batch_size = args.batch_size

    loaders = make_loaders(rows_by_split, batch_size, args.image_size, args.num_workers)

    pretrained_used = not args.no_pretrained
    try:
        model = build_model(args.model, pretrained=not args.no_pretrained)
    except Exception as exc:  # noqa: BLE001 - offline environments may not have cached weights.
        print(f"Pretrained weights unavailable for {args.model}: {exc}. Falling back to random initialization.")
        pretrained_used = False
        model = build_model(args.model, pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(rows_by_split["train"], device))
    set_backbone_trainable(model, args.model, trainable=False)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)

    best_state = deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_train_acc = -1.0
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
            best_train_acc = train_acc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {
        "model": args.model,
        "batch_size": batch_size,
        "image_size": args.image_size,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "pretrained_requested": not args.no_pretrained,
        "pretrained_used": pretrained_used,
        "best_epoch": best_epoch,
        "duration_seconds": round(time.time() - start, 2),
        "class_to_index": LABEL_TO_INDEX,
        "train_accuracy": float(best_train_acc),
        "val_accuracy": float(best_val_acc),
    }
    y_true, y_pred = collect_predictions(model, loaders["test"], device)
    test_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["test_accuracy"] = test_acc
    metrics["test_precision"] = float(precision)
    metrics["test_recall"] = float(recall)
    metrics["test_f1"] = float(f1)
    metrics["test_confusion_matrix"] = cm.tolist()
    plot_confusion(
        cm,
        args.figures_dir / f"{args.model}_test_confusion_matrix.png",
        f"{args.model} Test Confusion Matrix",
    )

    plot_history(history, args.figures_dir / f"{args.model}_training_curve.png", f"{args.model} Training Curve")
    torch.save(
        {
            "model": args.model,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "history": history,
            "class_to_index": LABEL_TO_INDEX,
        },
        args.checkpoint_dir / f"{args.model}_best.pt",
    )
    (args.output_dir / f"{args.model}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / f"{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one initial CNN baseline for YawDD+ mouth crops.")
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--split-manifest", type=Path, default=repo_root() / "artifacts" / "splits" / "yawdd_dash_subject_split.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "artifacts" / "results")
    parser.add_argument("--figures-dir", type=Path, default=repo_root() / "artifacts" / "figures")
    parser.add_argument("--checkpoint-dir", type=Path, default=repo_root() / "checkpoints")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 uses os.cpu_count() on CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_one(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
