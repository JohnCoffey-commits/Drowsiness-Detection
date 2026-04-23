from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MODEL_ROWS = [
    ("CNN-1 (ResNet18)", "resnet18"),
    ("CNN-2 (MobileNetV2)", "mobilenet_v2"),
    ("CNN-3 (EfficientNet-B0)", "efficientnet_b0"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_command(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def write_initial_results(metrics_by_model: dict[str, dict[str, object]], results_csv: Path, summary_json: Path) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cnn_architecture",
                "model",
                "train_accuracy",
                "validation_accuracy",
                "test_accuracy",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        for display_name, model_name in MODEL_ROWS:
            metrics = metrics_by_model[model_name]
            writer.writerow(
                {
                    "cnn_architecture": display_name,
                    "model": model_name,
                    "train_accuracy": metrics["train_accuracy"],
                    "validation_accuracy": metrics["val_accuracy"],
                    "test_accuracy": metrics["test_accuracy"],
                    "precision": metrics["test_precision"],
                    "recall": metrics["test_recall"],
                    "f1": metrics["test_f1"],
                }
            )
    summary_json.write_text(json.dumps(metrics_by_model, indent=2) + "\n")


def write_unavailable_results(results_csv: Path, summary_json: Path, reason: str) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cnn_architecture",
                "model",
                "train_accuracy",
                "validation_accuracy",
                "test_accuracy",
                "precision",
                "recall",
                "f1",
                "status",
            ],
        )
        writer.writeheader()
        for display_name, model_name in MODEL_ROWS:
            writer.writerow(
                {
                    "cnn_architecture": display_name,
                    "model": model_name,
                    "train_accuracy": "N/A",
                    "validation_accuracy": "N/A",
                    "test_accuracy": "N/A",
                    "precision": "N/A",
                    "recall": "N/A",
                    "f1": "N/A",
                    "status": reason,
                }
            )
    summary_json.write_text(json.dumps({"status": "not_run", "reason": reason}, indent=2) + "\n")


def pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def generate_summary(results_csv: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_csv.exists():
        rows = []
    else:
        with results_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

    lines = [
        "# Initial Experiment Summary",
        "",
        "## A. Task Type",
        "",
        "**Image Classification**",
        "",
        "## B. Experimental settings",
        "",
        "The initial experiment uses YawDD+ Dash as the intended primary dataset for binary mouth-focused yawning classification. Samples are mapped to `yawn` for yawning frames and `no-yawn` for normal or talking frames when those labels are recoverable from the inspected folder or file names. Each image is preprocessed with MediaPipe Face Mesh to crop the mouth region from lip landmarks, with a lower-face fallback crop when landmarks are unavailable, then resized to 224 x 224 and normalized with ImageNet statistics. Splitting is performed at subject level using approximately 70% training, 15% validation, and 15% test subjects to avoid frame leakage. The three baselines are CNN-1 ResNet18, CNN-2 MobileNetV2, and CNN-3 EfficientNet-B0, trained in PyTorch with Adam, learning rate 1e-4, weighted cross entropy, batch size 32 with fallback to 16, 12 epochs, early stopping patience 3, ReduceLROnPlateau scheduling, mild rotation/brightness/contrast/scale augmentation, and a two-stage transfer-learning schedule that freezes the backbone before full fine-tuning.",
        "",
        "## C. Initial Results table",
        "",
        "| CNN Architecture        | Train Accuracy | Validation Accuracy | Test Accuracy |",
        "| ----------------------- | -------------: | ------------------: | ------------: |",
    ]

    rows_with_metrics = [row for row in rows if row.get("train_accuracy") not in {"", "N/A", None}]

    if rows_with_metrics:
        by_arch = {row["cnn_architecture"]: row for row in rows_with_metrics}
        for display_name, _ in MODEL_ROWS:
            row = by_arch[display_name]
            lines.append(
                f"| {display_name:<23} | {pct(row['train_accuracy']):>14} | {pct(row['validation_accuracy']):>19} | {pct(row['test_accuracy']):>13} |"
            )
    else:
        for display_name, _ in MODEL_ROWS:
            lines.append(f"| {display_name:<23} |            N/A |                 N/A |           N/A |")

    lines.extend(["", "## D. Short interpretation", ""])
    if rows_with_metrics:
        best = max(rows_with_metrics, key=lambda row: float(row["test_accuracy"]))
        train_gap = float(best["train_accuracy"]) - float(best["validation_accuracy"])
        lines.extend(
            [
                f"{best['cnn_architecture']} achieved the highest test accuracy in this initial run.",
                f"The train-validation gap for the best model is {train_gap * 100:.2f} percentage points, which gives the first signal about overfitting.",
                "Class imbalance is handled with weighted cross entropy and should be reviewed against the split report before drawing strong conclusions.",
                "The mouth-crop pipeline is appropriate for yawning because it focuses the classifier on the most relevant facial region.",
                "The next step should be to add the full YawDD+ Dash image frames if missing, rerun preprocessing, and then compare these results with a later leakage-safe NTHUDDD2 evaluation.",
            ]
        )
    else:
        lines.extend(
            [
                "No CNN result is available yet because the inspected local YawDD+ Dash folder does not currently contain image files.",
                "The dataset appears incomplete for the requested supervised image-classification experiment because only annotation text files were found.",
                "NTHUDDD2 has recoverable labels and subject IDs, but it was intentionally not substituted for the initial YawDD+ Dash training source.",
                "Mouth cropping cannot be evaluated until paired Dash image frames are present.",
                "The next step is to add the missing YawDD+ Dash images, rerun preprocessing, build the subject split, and then run the three CNN baselines.",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n")


def split_manifest_ready(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"split manifest not found: {path}"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False, f"split manifest has no samples: {path}"
    splits = {row.get("split") for row in rows}
    if not {"train", "val", "test"}.issubset(splits):
        return False, f"split manifest does not contain train/val/test samples: {path}"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all three initial CNN baselines.")
    parser.add_argument("--split-manifest", type=Path, default=repo_root() / "artifacts" / "splits" / "yawdd_dash_subject_split.csv")
    parser.add_argument("--results-csv", type=Path, default=repo_root() / "artifacts" / "results" / "initial_results.csv")
    parser.add_argument("--summary-json", type=Path, default=repo_root() / "artifacts" / "results" / "metrics_summary.json")
    parser.add_argument("--summary-md", type=Path, default=repo_root() / "reports" / "initial_experiment_summary.md")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    split_ready, reason = split_manifest_ready(args.split_manifest)
    if not split_ready:
        write_unavailable_results(args.results_csv, args.summary_json, reason)
        generate_summary(args.results_csv, args.summary_md)
        raise SystemExit(f"Initial baselines not run because {reason}")

    metrics_by_model: dict[str, dict[str, object]] = {}
    for _, model_name in MODEL_ROWS:
        command = [
            sys.executable,
            "-m",
            "src.training.train_classifier",
            "--model",
            model_name,
            "--split-manifest",
            str(args.split_manifest),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.no_pretrained:
            command.append("--no-pretrained")
        run_command(command, repo_root())
        metrics_path = repo_root() / "artifacts" / "results" / f"{model_name}_metrics.json"
        metrics_by_model[model_name] = json.loads(metrics_path.read_text())

    write_initial_results(metrics_by_model, args.results_csv, args.summary_json)
    generate_summary(args.results_csv, args.summary_md)
    print(f"Wrote {args.results_csv}")
    print(f"Wrote {args.summary_md}")


if __name__ == "__main__":
    main()
