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
                    "train_accuracy": f"{float(metrics['train_accuracy']):.6f}",
                    "validation_accuracy": f"{float(metrics['val_accuracy']):.6f}",
                    "test_accuracy": f"{float(metrics['test_accuracy']):.6f}",
                    "precision": f"{float(metrics['test_precision']):.6f}",
                    "recall": f"{float(metrics['test_recall']):.6f}",
                    "f1": f"{float(metrics['test_f1']):.6f}",
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


def generate_summary(results_csv: Path, output_path: Path, metrics_by_model: dict[str, dict[str, object]] | None = None) -> None:
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
        "The initial experiment uses the reconstructed YawDD+ Dash mouth-crop dataset for binary image classification with labels `no_yawn` and `yawn`. Each sample is read from the Stage 6 leakage-safe split manifest and uses the Stage 5 mouth ROI crop generated from MediaPipe Face Mesh lip landmarks, with a lower-face fallback crop only when landmarks were unavailable; failed crops are already excluded. Training uses a subject-level split of 20 train, 4 validation, and 5 test subjects, and evaluates three transfer-learning baselines: CNN-1 ResNet18, CNN-2 MobileNetV2, and CNN-3 EfficientNet-B0. All models are trained in PyTorch with Adam, learning rate 1e-4, batch size 32 with fallback to 16 if needed, weighted cross-entropy loss, ReduceLROnPlateau scheduling, early stopping patience 3, 12 maximum epochs, and a two-stage strategy that trains the classification head before full fine-tuning. Training augmentation is limited to small rotation, light affine scaling, and mild brightness/contrast jitter, while validation and test use deterministic resizing and normalization only.",
        "",
        "## C. Initial Results table",
        "",
        "| CNN Architecture | Train Accuracy | Validation Accuracy | Test Accuracy |",
        "|---|---:|---:|---:|",
    ]

    rows_with_metrics = [row for row in rows if row.get("train_accuracy") not in {"", "N/A", None}]

    if rows_with_metrics:
        by_arch = {row["cnn_architecture"]: row for row in rows_with_metrics}
        for display_name, _ in MODEL_ROWS:
            row = by_arch[display_name]
            lines.append(f"| {display_name} | {pct(row['train_accuracy'])} | {pct(row['validation_accuracy'])} | {pct(row['test_accuracy'])} |")
    else:
        for display_name, _ in MODEL_ROWS:
            lines.append(f"| {display_name} | N/A | N/A | N/A |")

    lines.extend(["", "## D. Short interpretation", ""])
    if rows_with_metrics:
        best = max(rows_with_metrics, key=lambda row: float(row["test_accuracy"]))
        train_gap = float(best["train_accuracy"]) - float(best["validation_accuracy"])
        best_metrics = metrics_by_model[best["model"]] if metrics_by_model else None
        if best_metrics:
            precision = float(best_metrics["test_precision"]) * 100
            recall = float(best_metrics["test_recall"]) * 100
            f1 = float(best_metrics["test_f1"]) * 100
        else:
            precision = recall = f1 = 0.0
        lines.extend(
            [
                f"{best['cnn_architecture']} achieved the highest test accuracy in this initial experiment.",
                f"The best model shows a train-validation gap of {train_gap * 100:.2f} percentage points, which is the main signal to watch for overfitting.",
                f"On the test split, the best model reached precision {precision:.2f}%, recall {recall:.2f}%, and F1-score {f1:.2f}%, so accuracy should be interpreted together with the minority-class retrieval metrics.",
                "The dataset remains class-imbalanced toward `no_yawn`, so weighted cross entropy helps, but the subject-level split still makes generalization harder than an image-level split would.",
                "Because validation and test subjects are unseen identities, these results are a reasonable first baseline rather than an upper bound.",
                "The next step should be targeted error analysis on the saved confusion matrices and misclassified subjects, then a controlled comparison against later drowsiness-oriented data such as NTHUDDD2.",
            ]
        )
    else:
        lines.extend(
            [
                "No CNN result is available yet because the baselines have not completed successfully.",
                "The next step is to inspect the latest training log and rerun the failed model with the same Stage 6 split manifest.",
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
    labels = {row.get("label") for row in rows}
    if not {"no_yawn", "yawn"}.issubset(labels):
        return False, f"split manifest does not contain expected labels no_yawn/yawn: {path}"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all three initial CNN baselines.")
    parser.add_argument("--split-manifest", type=Path, default=repo_root() / "artifacts" / "splits" / "yawdd_dash_subject_split.csv")
    parser.add_argument("--results-csv", type=Path, default=repo_root() / "artifacts" / "results" / "initial_results.csv")
    parser.add_argument("--summary-json", type=Path, default=repo_root() / "artifacts" / "results" / "metrics_summary.json")
    parser.add_argument("--summary-md", type=Path, default=repo_root() / "reports" / "initial_experiment_summary.md")
    parser.add_argument("--checkpoint-dir", type=Path, default=repo_root() / "checkpoints")
    parser.add_argument("--figures-dir", type=Path, default=repo_root() / "artifacts" / "figures")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
            "-u",
            "-m",
            "src.training.train_classifier",
            "--model",
            model_name,
            "--split-manifest",
            str(args.split_manifest),
            "--checkpoint-dir",
            str(args.checkpoint_dir),
            "--figures-dir",
            str(args.figures_dir),
            "--epochs",
            str(args.epochs),
            "--image-size",
            str(args.image_size),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--cpu-threads",
            str(args.cpu_threads),
            "--seed",
            str(args.seed),
        ]
        if args.no_pretrained:
            command.append("--no-pretrained")
        run_command(command, repo_root())
        metrics_path = repo_root() / "artifacts" / "results" / f"{model_name}_metrics.json"
        metrics_by_model[model_name] = json.loads(metrics_path.read_text())

    write_initial_results(metrics_by_model, args.results_csv, args.summary_json)
    generate_summary(args.results_csv, args.summary_md, metrics_by_model)
    print(f"Wrote {args.results_csv}")
    print(f"Wrote {args.summary_md}")


if __name__ == "__main__":
    main()
