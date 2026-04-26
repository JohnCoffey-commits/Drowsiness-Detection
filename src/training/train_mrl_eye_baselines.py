"""Train MRL Eye open/closed CNN baselines.

This is the Stage 9 training entrypoint for the eye-state specialist module.
It consumes the Stage 8 subject-level split manifest and trains lightweight
binary classifiers that output:

    p_eye_closed, p_eye_open

Labels are fixed as:

    0 = closed
    1 = open

Example full run:

    python src/training/train_mrl_eye_baselines.py \
        --models resnet18 mobilenet_v2 efficientnet_b0 \
        --epochs 10 \
        --batch-size 64 \
        --image-size 224 \
        --manifest artifacts/mappings/mrl_eye_trainable_with_split.csv \
        --output-dir outputs/mrl_eye

Example debug run:

    python src/training/train_mrl_eye_baselines.py \
        --models resnet18 \
        --epochs 1 \
        --batch-size 16 \
        --max-samples-per-split 128 \
        --output-dir outputs/mrl_eye_debug
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = PROJECT_ROOT / "artifacts" / "cache" / "matplotlib"
TORCH_CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "torch"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        precision_recall_fscore_support,
    )
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ImportError as exc:  # pragma: no cover - depends on local training env
    raise SystemExit(
        "Missing Stage 9 training dependency. Install PyTorch, torchvision, "
        "pandas, Pillow, scikit-learn, and matplotlib in the training environment. "
        f"Original import error: {exc}"
    ) from exc


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


LABEL_TO_NAME = {0: "closed", 1: "open"}
NAME_TO_LABEL = {"closed": 0, "open": 1}
MODEL_NAMES = {"resnet18", "mobilenet_v2", "efficientnet_b0"}
SPLITS = ("train", "val", "test")
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "mappings" / "mrl_eye_trainable_with_split.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mrl_eye"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "results": output_dir / "results",
        "reports": output_dir / "reports",
        "figures": output_dir / "figures",
        "checkpoints": output_dir / "checkpoints",
        "error_analysis": output_dir / "error_analysis",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_image_path(row: pd.Series, data_root: Path | None) -> Path:
    relative = str(row.get("relative_path", "") or "")
    if data_root is not None and relative:
        candidate = data_root / relative
        if candidate.is_file():
            return candidate

        dataset_marker = "dataset/"
        if dataset_marker in relative:
            candidate = data_root / relative.split(dataset_marker, 1)[1]
            if candidate.is_file():
                return candidate

        raw_root_marker = "mrlEyes_2018_01/"
        if raw_root_marker in relative:
            candidate = data_root / relative.split(raw_root_marker, 1)[1]
            if candidate.is_file():
                return candidate

    image_path = str(row.get("image_path", "") or "")
    if image_path:
        candidate = Path(image_path)
        if candidate.is_file():
            return candidate
        if data_root is not None and relative:
            return data_root / relative
        return candidate
    if data_root is None or not relative:
        return Path("")
    return data_root / relative


def validate_manifest_columns(df: pd.DataFrame, manifest: Path) -> None:
    required = {"label", "split", "subject_id"}
    path_options = {"image_path", "relative_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Manifest is missing required columns {missing}: {manifest}")
    if not path_options.intersection(df.columns):
        raise SystemExit(
            f"Manifest must contain at least one path column from {sorted(path_options)}: {manifest}"
        )


def load_manifest(
    manifest: Path,
    data_root: Path | None,
    max_samples_per_split: int | None,
    seed: int,
) -> pd.DataFrame:
    manifest = normalize_path(manifest)
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    df = pd.read_csv(manifest, dtype={"subject_id": str, "sensor_id": str})
    validate_manifest_columns(df, manifest)

    df = df[df["split"].isin(SPLITS)].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    if "label_name" in df.columns:
        expected_names = df["label"].map(LABEL_TO_NAME)
        inconsistent = df["label_name"].astype(str).str.lower() != expected_names
        if inconsistent.any():
            bad = df.loc[inconsistent, ["filename", "label", "label_name"]].head()
            raise SystemExit(
                "Manifest label_name contradicts label mapping 0=closed, 1=open. "
                f"Examples:\n{bad.to_string(index=False)}"
            )
    else:
        df["label_name"] = df["label"].map(LABEL_TO_NAME)

    resolved_paths = [resolve_image_path(row, data_root) for _, row in df.iterrows()]
    df["resolved_image_path"] = [path.as_posix() for path in resolved_paths]
    exists = df["resolved_image_path"].map(lambda p: Path(p).is_file())
    if not exists.all():
        missing_count = int((~exists).sum())
        examples = df.loc[~exists, ["relative_path", "image_path", "resolved_image_path"]].head()
        raise SystemExit(
            f"{missing_count} manifest image path(s) do not exist after path resolution. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    if max_samples_per_split:
        sampled: list[pd.DataFrame] = []
        for split in SPLITS:
            split_df = df[df["split"] == split]
            if len(split_df) <= max_samples_per_split:
                sampled.append(split_df)
                continue
            per_class = max(1, max_samples_per_split // 2)
            parts = []
            for label in [0, 1]:
                label_df = split_df[split_df["label"] == label]
                parts.append(label_df.sample(n=min(per_class, len(label_df)), random_state=seed + label))
            merged = pd.concat(parts, ignore_index=False)
            if len(merged) < max_samples_per_split:
                remainder = split_df.drop(index=merged.index)
                need = max_samples_per_split - len(merged)
                merged = pd.concat(
                    [merged, remainder.sample(n=min(need, len(remainder)), random_state=seed + 99)]
                )
            sampled.append(merged.sample(frac=1.0, random_state=seed))
        df = pd.concat(sampled, ignore_index=True)

    counts = df.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    print("Manifest columns:", list(df.columns))
    print("Resolved manifest rows:", len(df))
    print("Split/class counts:")
    print(counts.to_string())
    return df.reset_index(drop=True)


class MRLEyeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["resolved_image_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(index, dtype=torch.long)


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.90, 1.10)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.06)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.20, contrast=0.20),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize(image_size + 16),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }


def make_loaders(
    df: pd.DataFrame,
    batch_size: int,
    image_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, DataLoader], dict[str, MRLEyeDataset]]:
    transform_map = build_transforms(image_size)
    datasets = {
        "train": MRLEyeDataset(df[df["split"] == "train"], transform_map["train"]),
        "train_eval": MRLEyeDataset(df[df["split"] == "train"], transform_map["eval"]),
        "val": MRLEyeDataset(df[df["split"] == "val"], transform_map["eval"]),
        "test": MRLEyeDataset(df[df["split"] == "test"], transform_map["eval"]),
    }
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **loader_kwargs),
        "train_eval": DataLoader(datasets["train_eval"], shuffle=False, **loader_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_kwargs),
    }
    return loaders, datasets


def try_torchvision_weights(model_name: str, pretrained: bool) -> Any:
    if not pretrained:
        return None
    if model_name == "resnet18":
        return models.ResNet18_Weights.DEFAULT
    if model_name == "mobilenet_v2":
        return models.MobileNet_V2_Weights.DEFAULT
    if model_name == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.DEFAULT
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(
    model_name: str,
    pretrained: bool,
    require_pretrained: bool = False,
) -> tuple[nn.Module, bool]:
    def instantiate(weights: Any) -> nn.Module:
        if model_name == "resnet18":
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, 2)
            return model
        if model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            return model
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            return model
        raise ValueError(f"Unsupported model: {model_name}")

    weights = try_torchvision_weights(model_name, pretrained)
    pretrained_requested = weights is not None
    try:
        return instantiate(weights), pretrained_requested
    except Exception as exc:
        if pretrained:
            if require_pretrained:
                raise SystemExit(
                    f"Required pretrained weights could not be loaded for {model_name}. "
                    f"Original error: {exc}"
                ) from exc
            print(
                f"Warning: could not load pretrained weights for {model_name}: {exc}. "
                "Falling back to random initialization."
            )
            return instantiate(None), False
        raise


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    if model_name == "resnet18":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def compute_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["label"].value_counts().reindex([0, 1], fill_value=0).to_numpy(dtype=float)
    if (counts == 0).any():
        raise SystemExit(f"Training split must contain both classes. Counts: {counts.tolist()}")
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    print(f"Class weights [closed, open]: {weights.tolist()}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def iterate(loader: DataLoader, description: str):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=description, leave=False)


def amp_context(enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = scaler is not None
    for images, labels, _indices in iterate(loader, f"epoch {epoch} train"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> dict[str, np.ndarray]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_probs: list[list[float]] = []
    all_indices: list[int] = []
    for images, labels, indices in iterate(loader, description):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_true.extend(labels.numpy().astype(int).tolist())
        all_pred.extend(preds.astype(int).tolist())
        all_probs.extend(probs.tolist())
        all_indices.extend(indices.numpy().astype(int).tolist())
    return {
        "y_true": np.asarray(all_true, dtype=int),
        "y_pred": np.asarray(all_pred, dtype=int),
        "probs": np.asarray(all_probs, dtype=float),
        "indices": np.asarray(all_indices, dtype=int),
    }


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )[2]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision_closed": float(precision[0]),
        "recall_closed": float(recall[0]),
        "f1_closed": float(f1[0]),
        "precision_open": float(precision[1]),
        "recall_open": float(recall[1]),
        "f1_open": float(f1[1]),
        "confusion_matrix": cm.astype(int).tolist(),
        "num_samples": int(len(y_true)),
        "false_open_count": int(cm[0, 1]),
        "false_closed_count": int(cm[1, 0]),
    }


def threshold_sweep(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p_closed = probs[:, 0]
    for threshold in THRESHOLDS:
        y_pred = np.where(p_closed >= threshold, 0, 1)
        metrics = metrics_from_predictions(y_true, y_pred)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "precision_closed": metrics["precision_closed"],
                "recall_closed": metrics["recall_closed"],
                "f1_closed": metrics["f1_closed"],
                "precision_open": metrics["precision_open"],
                "recall_open": metrics["recall_open"],
                "f1_open": metrics["f1_open"],
                "false_open_count": metrics["false_open_count"],
                "false_closed_count": metrics["false_closed_count"],
            }
        )
    return pd.DataFrame(rows)


def select_candidate_threshold(validation_sweep: pd.DataFrame) -> dict[str, Any]:
    default = validation_sweep[validation_sweep["threshold"] == 0.50]
    if default.empty:
        selected = validation_sweep.sort_values("macro_f1", ascending=False).iloc[0]
        return {
            "threshold": float(selected["threshold"]),
            "selection_source": "validation",
            "selection_reason": "best validation macro F1; 0.50 baseline row unavailable",
            "validation_metrics": selected.to_dict(),
        }

    baseline = default.iloc[0]
    max_macro = float(validation_sweep["macro_f1"].max())
    viable = validation_sweep[
        (validation_sweep["macro_f1"] >= max_macro - 0.02)
        & (validation_sweep["recall_closed"] >= baseline["recall_closed"])
    ].copy()
    if viable.empty:
        selected = validation_sweep.sort_values("macro_f1", ascending=False).iloc[0]
        reason = (
            "best validation macro F1; no threshold improved closed recall while "
            "remaining within 0.02 macro F1 of the best validation threshold"
        )
    else:
        selected = viable.sort_values(["recall_closed", "macro_f1"], ascending=False).iloc[0]
        reason = (
            "highest validation closed-eye recall among thresholds within 0.02 "
            "macro F1 of the best validation threshold"
        )

    return {
        "threshold": float(selected["threshold"]),
        "selection_source": "validation",
        "selection_reason": reason,
        "validation_metrics": selected.to_dict(),
    }


def metrics_at_closed_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = np.where(probs[:, 0] >= threshold, 0, 1)
    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    return metrics


def flatten_metrics(split_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for split, metrics in split_metrics.items():
        for key, value in metrics.items():
            if key == "confusion_matrix":
                continue
            flat[f"{split}_{key}"] = value
    return flat


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_training_curve(history: list[dict[str, Any]], output_path: Path, model_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_macro_f1"] for row in history], label="val macro F1")
    plt.plot(epochs, [row["val_recall_closed"] for row in history], label="val recall closed")
    plt.xlabel("Epoch")
    plt.title(f"{model_name} MRL Eye training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: list[list[int]], output_path: Path, model_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(cm)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{model_name} test confusion matrix")
    plt.xticks([0, 1], ["pred closed", "pred open"])
    plt.yticks([0, 1], ["true closed", "true open"])
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_pr_curve_closed(
    y_true: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    model_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_closed = (y_true == 0).astype(int)
    p_closed = probs[:, 0]
    precision, recall, _ = precision_recall_curve(y_closed, p_closed)
    ap = average_precision_score(y_closed, p_closed)
    plt.figure(figsize=(5.5, 4))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Closed-eye recall")
    plt.ylabel("Closed-eye precision")
    plt.title(f"{model_name} closed-class PR curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def make_contact_sheet(
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
    max_images: int = 36,
    seed: int = 42,
    shuffle: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.head(max_images).copy()
    if rows.empty:
        image = Image.new("RGB", (720, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f"{title}: no samples", fill="black", font=ImageFont.load_default())
        image.save(output_path, quality=92)
        return

    if shuffle:
        rows = rows.sample(frac=1.0, random_state=seed)
    rows = rows.head(max_images)
    thumb_size = (120, 80)
    label_height = 44
    padding = 8
    columns = 6
    cell_width = thumb_size[0] + padding * 2
    cell_height = thumb_size[1] + label_height + padding * 2
    title_height = 28
    num_rows = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * cell_width, title_height + num_rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((padding, 8), title, fill="black", font=font)

    for idx, (_, row) in enumerate(rows.iterrows()):
        col = idx % columns
        grid_row = idx // columns
        x0 = col * cell_width + padding
        y0 = title_height + grid_row * cell_height + padding
        path = Path(str(row["resolved_image_path"]))
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", thumb_size, "white")
                canvas.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
                sheet.paste(canvas, (x0, y0))
        except (OSError, UnidentifiedImageError) as exc:
            draw.rectangle((x0, y0, x0 + thumb_size[0], y0 + thumb_size[1]), outline="red")
            draw.text((x0 + 4, y0 + 4), str(exc)[:40], fill="red", font=font)
        p_closed = float(row.get("p_eye_closed", 0.0))
        text = f"{row['subject_id']} | true {row['label_name']} | pC {p_closed:.2f}"
        draw.text((x0, y0 + thumb_size[1] + 5), text[:26], fill="black", font=font)
        draw.text((x0, y0 + thumb_size[1] + 18), f"split {row['split']}", fill="black", font=font)
    sheet.save(output_path, quality=92)


def save_error_contact_sheets(
    test_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    dirs: dict[str, Path],
    model_name: str,
    seed: int,
) -> None:
    indexed = test_df.iloc[predictions["indices"]].copy().reset_index(drop=True)
    indexed["y_true"] = predictions["y_true"]
    indexed["y_pred"] = predictions["y_pred"]
    indexed["p_eye_closed"] = predictions["probs"][:, 0]
    indexed["p_eye_open"] = predictions["probs"][:, 1]
    false_open = indexed[(indexed["y_true"] == 0) & (indexed["y_pred"] == 1)].copy()
    false_closed = indexed[(indexed["y_true"] == 1) & (indexed["y_pred"] == 0)].copy()
    false_open = false_open.sort_values("p_eye_open", ascending=False)
    false_closed = false_closed.sort_values("p_eye_closed", ascending=False)
    make_contact_sheet(
        false_open,
        dirs["error_analysis"] / f"{model_name}_false_open_contact_sheet.jpg",
        f"{model_name} false open: true closed, predicted open",
        seed=seed,
        shuffle=False,
    )
    make_contact_sheet(
        false_closed,
        dirs["error_analysis"] / f"{model_name}_false_closed_contact_sheet.jpg",
        f"{model_name} false closed: true open, predicted closed",
        seed=seed,
        shuffle=False,
    )


def checkpoint_payload(
    model: nn.Module,
    model_name: str,
    epoch: int,
    best_val_macro_f1: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "epoch": epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "state_dict": model.state_dict(),
        "label_mapping": LABEL_TO_NAME,
        "image_size": args.image_size,
        "outputs": ["p_eye_closed", "p_eye_open"],
    }


def train_model(
    model_name: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    dirs: dict[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    print(f"\n=== Training {model_name} ===")
    loaders, datasets = make_loaders(df, args.batch_size, args.image_size, args.num_workers, device)
    model, pretrained_loaded = build_model(
        model_name,
        pretrained=not args.no_pretrained,
        require_pretrained=args.require_pretrained,
    )
    model = model.to(device)
    print(f"Pretrained weights loaded: {pretrained_loaded}")
    weights = compute_class_weights(df[df["split"] == "train"], device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    checkpoint_path = dirs["checkpoints"] / f"best_{model_name}_mrl_eye.pt"
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    scaler_or_none = scaler if device.type == "cuda" else None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    current_frozen: bool | None = None

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        if current_frozen is None or frozen != current_frozen:
            set_backbone_trainable(model, model_name, trainable=not frozen)
            optimizer = torch.optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=1,
                factor=0.5,
            )
            current_frozen = frozen

        started = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            scaler_or_none,
            epoch,
        )
        val_preds = predict(model, loaders["val"], device, f"epoch {epoch} val")
        val_metrics = metrics_from_predictions(val_preds["y_true"], val_preds["y_pred"])
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "phase": "head_only" if frozen else "fine_tune",
            "train_loss": train_loss,
            "train_accuracy_batch": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_recall_closed": val_metrics["recall_closed"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_recall_closed={val_metrics['recall_closed']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            torch.save(
                checkpoint_payload(model, model_name, epoch, best_val_macro_f1, args),
                checkpoint_path,
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["state_dict"])

    split_metrics: dict[str, dict[str, Any]] = {}
    split_predictions: dict[str, dict[str, np.ndarray]] = {}
    for split, loader_name in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
        preds = predict(model, loaders[loader_name], device, f"{model_name} {split} eval")
        split_predictions[split] = preds
        split_metrics[split] = metrics_from_predictions(preds["y_true"], preds["y_pred"])

    val_preds = split_predictions["val"]
    test_preds = split_predictions["test"]
    val_threshold_df = threshold_sweep(val_preds["y_true"], val_preds["probs"])
    test_threshold_df = threshold_sweep(test_preds["y_true"], test_preds["probs"])
    threshold_selection = select_candidate_threshold(val_threshold_df)
    selected_threshold = float(threshold_selection["threshold"])
    test_metrics_at_selected_threshold = metrics_at_closed_threshold(
        test_preds["y_true"],
        test_preds["probs"],
        selected_threshold,
    )
    val_threshold_path = dirs["results"] / f"{model_name}_val_threshold_sweep.csv"
    test_threshold_path = dirs["results"] / f"{model_name}_test_threshold_sweep.csv"
    val_threshold_df.to_csv(val_threshold_path, index=False)
    test_threshold_df.to_csv(test_threshold_path, index=False)

    history_path = dirs["results"] / f"{model_name}_history.json"
    metrics_path = dirs["results"] / f"{model_name}_metrics.json"
    curve_path = dirs["figures"] / f"{model_name}_training_curve.png"
    cm_path = dirs["figures"] / f"{model_name}_confusion_matrix.png"
    pr_path = dirs["figures"] / f"{model_name}_pr_curve_closed.png"

    metrics_payload = {
        "model_name": model_name,
        "pretrained_requested": not args.no_pretrained,
        "pretrained_required": args.require_pretrained,
        "pretrained_loaded": pretrained_loaded,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": checkpoint_path.as_posix(),
        "val_threshold_sweep_path": val_threshold_path.as_posix(),
        "test_threshold_sweep_path": test_threshold_path.as_posix(),
        "threshold_selection": threshold_selection,
        "test_metrics_at_validation_threshold": test_metrics_at_selected_threshold,
        "splits": split_metrics,
    }
    save_json(history_path, history)
    save_json(metrics_path, metrics_payload)
    plot_training_curve(history, curve_path, model_name)
    plot_confusion_matrix(split_metrics["test"]["confusion_matrix"], cm_path, model_name)
    plot_pr_curve_closed(test_preds["y_true"], test_preds["probs"], pr_path, model_name)
    save_error_contact_sheets(datasets["test"].frame, test_preds, dirs, model_name, args.seed)

    summary_row = {
        "model": model_name,
        "pretrained_loaded": pretrained_loaded,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "selected_threshold_from_val": selected_threshold,
        "test_selected_threshold_accuracy": test_metrics_at_selected_threshold["accuracy"],
        "test_selected_threshold_macro_f1": test_metrics_at_selected_threshold["macro_f1"],
        "test_selected_threshold_recall_closed": test_metrics_at_selected_threshold["recall_closed"],
        "test_selected_threshold_false_open_count": test_metrics_at_selected_threshold["false_open_count"],
        "test_selected_threshold_false_closed_count": test_metrics_at_selected_threshold["false_closed_count"],
        **flatten_metrics(split_metrics),
        "checkpoint_path": checkpoint_path.as_posix(),
    }
    print(
        f"{model_name} test macro_f1={split_metrics['test']['macro_f1']:.4f}, "
        f"test recall_closed={split_metrics['test']['recall_closed']:.4f}, "
        f"false_open={split_metrics['test']['false_open_count']}"
    )
    return {
        "summary_row": summary_row,
        "metrics_payload": metrics_payload,
    }


def candidate_threshold_text(threshold_df: pd.DataFrame) -> str:
    default = threshold_df[threshold_df["threshold"] == 0.50]
    if default.empty:
        return "No 0.50 baseline threshold row was available."
    baseline = default.iloc[0]
    max_macro = float(threshold_df["macro_f1"].max())
    viable = threshold_df[
        (threshold_df["macro_f1"] >= max_macro - 0.02)
        & (threshold_df["recall_closed"] >= baseline["recall_closed"])
    ].copy()
    if viable.empty:
        return (
            "No threshold improved closed-eye recall while staying within 0.02 macro F1 "
            "of the best threshold; keep threshold selection as a validation-time decision."
        )
    selected = viable.sort_values(["recall_closed", "macro_f1"], ascending=False).iloc[0]
    return (
        f"Candidate threshold: {selected['threshold']:.2f}. "
        f"It keeps macro F1 at {selected['macro_f1']:.4f} and closed recall at "
        f"{selected['recall_closed']:.4f}. This is only a candidate for later temporal fusion."
    )


def write_combined_outputs(
    model_outputs: dict[str, dict[str, Any]],
    dirs: dict[str, Path],
) -> None:
    rows = [payload["summary_row"] for payload in model_outputs.values()]
    results_csv = dirs["results"] / "mrl_eye_initial_results.csv"
    summary_json = dirs["results"] / "mrl_eye_metrics_summary.json"
    report_md = dirs["reports"] / "mrl_eye_experiment_summary.md"

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_csv, index=False)
    save_json(summary_json, {name: payload["metrics_payload"] for name, payload in model_outputs.items()})

    lines = [
        "# MRL Eye Stage 9 Experiment Summary",
        "",
        "The MRL Eye module is an eye open/closed specialist. It reports per-frame "
        "`p_eye_closed` and `p_eye_open` for later temporal fusion; it is not a full "
        "drowsiness classifier.",
        "",
        "## Results",
        "",
        "| Model | Pretrained | Val macro F1 | Test macro F1 (argmax) | Test closed recall (argmax) | Val-selected threshold | Test macro F1 (fixed threshold) | Test closed recall (fixed threshold) | False open (fixed threshold) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['pretrained_loaded']} | "
            f"{row['best_val_macro_f1']:.4f} | {row['test_macro_f1']:.4f} | "
            f"{row['test_recall_closed']:.4f} | {row['selected_threshold_from_val']:.2f} | "
            f"{row['test_selected_threshold_macro_f1']:.4f} | "
            f"{row['test_selected_threshold_recall_closed']:.4f} | "
            f"{int(row['test_selected_threshold_false_open_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Notes",
            "",
            "Threshold candidates are selected from the validation sweep only. "
            "The test sweep is saved for final reporting and audit, not for choosing "
            "the threshold.",
            "",
        ]
    )
    for model_name in model_outputs:
        sweep_path = dirs["results"] / f"{model_name}_val_threshold_sweep.csv"
        if sweep_path.is_file():
            sweep = pd.read_csv(sweep_path)
            selected = model_outputs[model_name]["metrics_payload"]["threshold_selection"]
            test_fixed = model_outputs[model_name]["metrics_payload"][
                "test_metrics_at_validation_threshold"
            ]
            lines.append(
                f"- `{model_name}`: {candidate_threshold_text(sweep)} "
                f"Selected from validation: {selected['threshold']:.2f}. "
                f"Applied unchanged to test: macro F1 {test_fixed['macro_f1']:.4f}, "
                f"closed recall {test_fixed['recall_closed']:.4f}, "
                f"false-open count {int(test_fixed['false_open_count'])}."
            )

    lines.extend(
        [
            "",
            "Closed-eye recall should be reviewed alongside macro F1. A threshold that "
            "predicts nearly everything as closed is not useful for deployment even if it "
            "reduces false-open errors in isolation.",
            "",
            "Next step: run the selected Stage 9 models in Colab, inspect error contact "
            "sheets, then pass `p_eye_closed` into Stage 10/11 temporal smoothing or "
            "PERCLOS-like fusion with the YawDD mouth/yawn module.",
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote combined results: {results_csv}")
    print(f"Wrote combined summary: {summary_json}")
    print(f"Wrote experiment report: {report_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--require-pretrained",
        action="store_true",
        help="Stop if torchvision pretrained weights cannot be loaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.models) - MODEL_NAMES)
    if unknown:
        raise SystemExit(f"Unsupported model(s): {unknown}. Supported: {sorted(MODEL_NAMES)}")
    if args.no_pretrained and args.require_pretrained:
        raise SystemExit("--no-pretrained and --require-pretrained cannot be used together.")

    set_seed(args.seed)
    output_dir = normalize_path(args.output_dir)
    dirs = ensure_output_dirs(output_dir)
    data_root = normalize_path(args.data_root) if args.data_root else None
    device = select_device()
    print(f"Device: {device}")
    print(f"Mixed precision enabled: {device.type == 'cuda'}")
    print(f"Output directory: {output_dir}")
    if data_root:
        print(f"Data root override: {data_root}")

    df = load_manifest(args.manifest, data_root, args.max_samples_per_split, args.seed)
    model_outputs: dict[str, dict[str, Any]] = {}
    for model_name in args.models:
        model_outputs[model_name] = train_model(model_name, df, args, dirs, device)
    write_combined_outputs(model_outputs, dirs)


if __name__ == "__main__":
    main()
