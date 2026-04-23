from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXTENSIONS = {".txt", ".csv", ".json", ".xml", ".yaml", ".yml"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mkdir_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def tokenize(value: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", value.lower()) if part]


def infer_yawn_label(path: Path) -> str | None:
    tokens = tokenize(" ".join(path.parts))
    token_set = set(tokens)
    if {"yawn", "yawning"} & token_set:
        return "yawn"
    if {"normal", "talking", "talk", "no", "noyawn", "non", "notyawn"} & token_set:
        return "no-yawn"
    return None


def list_tree(root: Path, max_depth: int = 3, max_entries: int = 240) -> list[str]:
    if not root.exists():
        return [f"{root} [missing]"]

    lines: list[str] = []
    for idx, path in enumerate(sorted(root.rglob("*"))):
        if idx >= max_entries:
            lines.append(f"... truncated after {max_entries} entries")
            break
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{'  ' * (depth - 1)}- {path.name}{suffix}")
    return lines


def inspect_yawdd_dash(root: Path, report_path: Path) -> dict[str, object]:
    image_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    annotation_files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ANNOTATION_EXTENSIONS
    )
    subject_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    subfolder_counter = Counter(p.name for p in root.rglob("*") if p.is_dir()) if root.exists() else Counter()

    inferred_labels = Counter()
    unlabeled_images = 0
    for image_path in image_files:
        label = infer_yawn_label(image_path)
        if label is None:
            unlabeled_images += 1
        else:
            inferred_labels[label] += 1

    yolo_class_ids = Counter()
    txt_files = [p for p in annotation_files if p.suffix.lower() == ".txt"]
    for txt_file in txt_files:
        try:
            for line in txt_file.read_text(errors="ignore").splitlines():
                parts = line.strip().split()
                if parts and re.fullmatch(r"-?\d+", parts[0]):
                    yolo_class_ids[parts[0]] += 1
        except OSError:
            continue

    subject_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for image_path in image_files:
        try:
            subject_id = image_path.relative_to(root).parts[0]
        except IndexError:
            subject_id = "unknown"
        subject_label_counts[subject_id][infer_yawn_label(image_path) or "unknown"] += 1

    lines = [
        "# YawDD+ Dash Dataset Report",
        "",
        f"Dataset path: `{root}`",
        f"Path exists: `{root.exists()}`",
        "",
        "## Tree Summary",
        "",
        "```text",
        *list_tree(root),
        "```",
        "",
        "## Counts",
        "",
        f"- Subject folders: {len(subject_dirs)}",
        f"- Subfolders: {sum(1 for _ in root.rglob('*') if _.is_dir()) if root.exists() else 0}",
        f"- Image files: {len(image_files)}",
        f"- Annotation files: {len(annotation_files)}",
        f"- Text label files: {len(txt_files)}",
        "",
        "## Label Inference",
        "",
    ]

    if image_files:
        lines.extend(
            [
                "Image labels were inferred from folder/file-name tokens.",
                "",
                "| Label | Images |",
                "| --- | ---: |",
            ]
        )
        for label, count in sorted(inferred_labels.items()):
            lines.append(f"| {label} | {count} |")
        if unlabeled_images:
            lines.append(f"| unknown | {unlabeled_images} |")
    else:
        lines.extend(
            [
                "No image files were found under Dash. The current local Dash tree contains annotation text files only.",
                "The available `.txt` files look like YOLO bounding-box annotations, not supervised image-class labels.",
            ]
        )

    lines.extend(["", "## Annotation Class IDs", "", "| Class ID | Rows |", "| --- | ---: |"])
    if yolo_class_ids:
        for class_id, count in sorted(yolo_class_ids.items()):
            lines.append(f"| {class_id} | {count} |")
    else:
        lines.append("| none detected | 0 |")

    lines.extend(
        [
            "",
            "## Subject Folders",
            "",
            "| Subject ID | Images | Label files |",
            "| --- | ---: | ---: |",
        ]
    )
    for subject_dir in subject_dirs:
        images = sum(1 for p in subject_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        labels = sum(1 for p in subject_dir.rglob("*.txt") if p.is_file())
        lines.append(f"| {subject_dir.name} | {images} | {labels} |")

    lines.extend(
        [
            "",
            "## Assumptions",
            "",
            "- If future Dash images are organized or named with `yawning`, `normal`, and `talking`, this pipeline maps `yawning` to `yawn` and `normal`/`talking` to `no-yawn`.",
            "- If only YOLO `.txt` files are present, they are treated as localization annotations and are not sufficient for the requested binary image-classification experiment.",
            "",
            "## Status",
            "",
        ]
    )
    if image_files and inferred_labels:
        lines.append("Dash appears usable for the initial supervised experiment, subject to label coverage checks during preprocessing.")
    else:
        lines.append("Dash is not ready for supervised initial training in the current local folder because no image files were found.")

    mkdir_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n")

    return {
        "image_files": len(image_files),
        "annotation_files": len(annotation_files),
        "subject_folders": len(subject_dirs),
        "inferred_labels": dict(inferred_labels),
        "yolo_class_ids": dict(yolo_class_ids),
    }


def parse_nthu_filename(path: Path) -> dict[str, str | None]:
    stem = path.stem
    parts = stem.split("_")
    label = path.parent.name.lower() if path.parent.name.lower() in {"drowsy", "notdrowsy"} else None
    return {
        "subject_id": parts[0] if parts and re.fullmatch(r"\d+", parts[0]) else None,
        "glasses": parts[1] if len(parts) > 1 else None,
        "scenario": parts[2] if len(parts) > 2 else None,
        "frame_or_index": parts[3] if len(parts) > 3 else None,
        "label": label,
    }


def inspect_nthu(root: Path, report_path: Path) -> dict[str, object]:
    image_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    label_counts = Counter(p.parent.name for p in image_files)
    subjects = Counter()
    scenarios = Counter()
    parse_failures = 0
    rows = []

    for image_path in image_files:
        parsed = parse_nthu_filename(image_path)
        if parsed["subject_id"] is None or parsed["label"] is None:
            parse_failures += 1
        if parsed["subject_id"]:
            subjects[parsed["subject_id"]] += 1
        if parsed["scenario"]:
            scenarios[parsed["scenario"]] += 1
        rows.append(parsed)

    usable_later = bool(image_files and label_counts and subjects and parse_failures == 0)
    status = "usable later" if usable_later else "not ready for supervised initial experiment"

    lines = [
        "# NTHUDDD2 Dataset Report",
        "",
        f"Dataset path: `{root}`",
        f"Path exists: `{root.exists()}`",
        "",
        "## Tree Summary",
        "",
        "```text",
        *list_tree(root),
        "```",
        "",
        "## Counts",
        "",
        f"- Image files: {len(image_files)}",
        f"- Labels from parent folders: {', '.join(sorted(label_counts)) if label_counts else 'none detected'}",
        f"- Recoverable subject IDs: {len(subjects)}",
        f"- Filename parse failures: {parse_failures}",
        "",
        "## Label Counts",
        "",
        "| Label | Images |",
        "| --- | ---: |",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "## Subject Counts", "", "| Subject ID | Images |", "| --- | ---: |"])
    for subject_id, count in sorted(subjects.items()):
        lines.append(f"| {subject_id} | {count} |")

    lines.extend(["", "## Scenario Tokens", "", "| Scenario | Images |", "| --- | ---: |"])
    for scenario, count in sorted(scenarios.items()):
        lines.append(f"| {scenario} | {count} |")

    lines.extend(
        [
            "",
            "## Assessment",
            "",
            "NTHUDDD2 has supervised labels from the `drowsy` and `notdrowsy` folders, and subject IDs can be recovered from the leading filename token.",
            "However, it is not used for the initial experiment because the selected initial task is YawDD+ Dash mouth-focused yawning classification.",
            "",
            f"Final status: **{status}**",
        ]
    )

    mkdir_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n")

    return {
        "image_files": len(image_files),
        "label_counts": dict(label_counts),
        "subjects": len(subjects),
        "status": status,
    }


def write_summary_csv(path: Path, yawdd_stats: dict[str, object], nthu_stats: dict[str, object]) -> None:
    mkdir_parent(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "images", "subjects", "status"])
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "YawDD+ Dash",
                "images": yawdd_stats["image_files"],
                "subjects": yawdd_stats["subject_folders"],
                "status": "ready" if yawdd_stats["image_files"] else "missing images",
            }
        )
        writer.writerow(
            {
                "dataset": "NTHUDDD2 train_data",
                "images": nthu_stats["image_files"],
                "subjects": nthu_stats["subjects"],
                "status": nthu_stats["status"],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect YawDD+ Dash and NTHUDDD2 dataset structure.")
    parser.add_argument("--yawdd-root", type=Path, default=repo_root() / "dataset" / "YawDD+" / "dataset" / "Dash")
    parser.add_argument("--nthu-root", type=Path, default=repo_root() / "dataset" / "NTHUDDD2" / "train_data")
    parser.add_argument("--reports-dir", type=Path, default=repo_root() / "reports")
    args = parser.parse_args()

    yawdd_stats = inspect_yawdd_dash(args.yawdd_root, args.reports_dir / "yawdd_dash_dataset_report.md")
    nthu_stats = inspect_nthu(args.nthu_root, args.reports_dir / "nthu_dataset_report.md")
    write_summary_csv(args.reports_dir / "dataset_inspection_summary.csv", yawdd_stats, nthu_stats)

    print("Dataset inspection complete.")
    print(f"YawDD+ Dash: {yawdd_stats['image_files']} images, {yawdd_stats['subject_folders']} subject folders")
    print(f"NTHUDDD2: {nthu_stats['image_files']} images, status={nthu_stats['status']}")


if __name__ == "__main__":
    main()
