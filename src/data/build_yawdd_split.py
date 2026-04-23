from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_SUBJECT_COUNTS = {
    "train": 20,
    "val": 4,
    "test": 5,
}

LABELS = ("no_yawn", "yawn")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Stage 5 mouth-crop manifest not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def subject_attributes(subject_id: str) -> tuple[str, str]:
    if "Female" in subject_id:
        gender = "Female"
    elif "Male" in subject_id:
        gender = "Male"
    else:
        gender = "Unknown"

    if "NoGlasses" in subject_id:
        glasses = "NoGlasses"
    elif "Glasses" in subject_id:
        glasses = "Glasses"
    else:
        glasses = "Unknown"
    return gender, glasses


def filter_trainable(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    trainable = []
    for row in rows:
        if row.get("crop_method") == "failed":
            continue
        if row.get("binary_label") not in LABELS:
            continue
        if not row.get("mouth_crop_path"):
            continue
        trainable.append(row)
    return trainable


def build_subject_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for row in rows:
        subject_id = row["subject_id"]
        if subject_id not in stats:
            gender, glasses = subject_attributes(subject_id)
            stats[subject_id] = {
                "subject_id": subject_id,
                "gender": gender,
                "glasses": glasses,
                "total": 0,
                "labels": Counter(),
                "methods": Counter(),
            }
        stats[subject_id]["total"] = int(stats[subject_id]["total"]) + 1
        labels = stats[subject_id]["labels"]
        methods = stats[subject_id]["methods"]
        assert isinstance(labels, Counter)
        assert isinstance(methods, Counter)
        labels[row["binary_label"]] += 1
        methods[row["crop_method"]] += 1
    return stats


def summarize_assignment(
    assignment: dict[str, str],
    subject_stats: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    summary = {}
    for split in SPLIT_SUBJECT_COUNTS:
        summary[split] = {
            "subjects": [],
            "total": 0,
            "labels": Counter(),
            "gender": Counter(),
            "glasses": Counter(),
        }

    for subject_id, split in assignment.items():
        split_summary = summary[split]
        stats = subject_stats[subject_id]
        split_summary["subjects"].append(subject_id)
        split_summary["total"] = int(split_summary["total"]) + int(stats["total"])
        split_labels = split_summary["labels"]
        assert isinstance(split_labels, Counter)
        split_labels.update(stats["labels"])
        gender_counts = split_summary["gender"]
        glasses_counts = split_summary["glasses"]
        assert isinstance(gender_counts, Counter)
        assert isinstance(glasses_counts, Counter)
        gender_counts[str(stats["gender"])] += 1
        glasses_counts[str(stats["glasses"])] += 1

    return summary


def split_score(
    assignment: dict[str, str],
    subject_stats: dict[str, dict[str, object]],
    overall_labels: Counter[str],
    overall_gender: Counter[str],
    overall_glasses: Counter[str],
) -> float:
    summary = summarize_assignment(assignment, subject_stats)
    total_images = sum(overall_labels.values())
    total_subjects = len(subject_stats)
    overall_yawn_rate = overall_labels["yawn"] / total_images

    score = 0.0
    for split, target_subjects in SPLIT_SUBJECT_COUNTS.items():
        split_summary = summary[split]
        split_total = int(split_summary["total"])
        labels = split_summary["labels"]
        gender_counts = split_summary["gender"]
        glasses_counts = split_summary["glasses"]
        assert isinstance(labels, Counter)
        assert isinstance(gender_counts, Counter)
        assert isinstance(glasses_counts, Counter)

        if split_total == 0:
            return 1_000_000.0

        expected_image_prop = target_subjects / total_subjects
        actual_image_prop = split_total / total_images
        score += abs(actual_image_prop - expected_image_prop) * 4.0

        yawn_rate = labels["yawn"] / split_total
        score += abs(yawn_rate - overall_yawn_rate) * 6.0

        if labels["no_yawn"] == 0 or labels["yawn"] == 0:
            score += 100.0

        for key, overall_count in overall_gender.items():
            expected = target_subjects * overall_count / total_subjects
            score += abs(gender_counts[key] - expected) * 0.15
        for key, overall_count in overall_glasses.items():
            expected = target_subjects * overall_count / total_subjects
            score += abs(glasses_counts[key] - expected) * 0.15

        if split in {"val", "test"}:
            if gender_counts["Female"] == 0 or gender_counts["Male"] == 0:
                score += 10.0
            if glasses_counts["Glasses"] == 0 or glasses_counts["NoGlasses"] == 0:
                score += 10.0

    return score


def choose_subject_split(
    subject_stats: dict[str, dict[str, object]],
    seed: int,
    iterations: int,
) -> tuple[dict[str, str], float]:
    subject_ids = sorted(subject_stats)
    expected_subjects = sum(SPLIT_SUBJECT_COUNTS.values())
    if len(subject_ids) != expected_subjects:
        raise SystemExit(
            f"Expected {expected_subjects} trainable subjects, found {len(subject_ids)}."
        )

    overall_labels: Counter[str] = Counter()
    overall_gender: Counter[str] = Counter()
    overall_glasses: Counter[str] = Counter()
    for stats in subject_stats.values():
        overall_labels.update(stats["labels"])
        overall_gender[str(stats["gender"])] += 1
        overall_glasses[str(stats["glasses"])] += 1

    rng = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")

    for _ in range(iterations):
        shuffled = subject_ids[:]
        rng.shuffle(shuffled)
        assignment = {}
        start = 0
        for split, count in SPLIT_SUBJECT_COUNTS.items():
            for subject_id in shuffled[start : start + count]:
                assignment[subject_id] = split
            start += count

        score = split_score(
            assignment,
            subject_stats,
            overall_labels,
            overall_gender,
            overall_glasses,
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise SystemExit("Failed to build a subject-level split.")
    return best_assignment, best_score


def check_missing_crops(rows: list[dict[str, str]]) -> int:
    missing = 0
    for row in rows:
        if not Path(row["mouth_crop_path"]).is_file():
            missing += 1
    return missing


def write_trainable_manifest(
    path: Path,
    rows: list[dict[str, str]],
    assignment: dict[str, str],
    source_fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = source_fieldnames[:]
    if "split" not in fieldnames:
        fieldnames.append("split")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["split"] = assignment[row["subject_id"]]
            writer.writerow(out)


def write_split_manifest(
    path: Path,
    rows: list[dict[str, str]],
    assignment: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "split",
        "gender",
        "glasses",
        "frame_index",
        "original_path",
        "processed_path",
        "label",
        "class_id",
        "crop_method",
        "mouth_bbox_xyxy",
        "image_path",
        "mouth_crop_path",
        "raw_video_path",
        "annotation_txt_path",
        "yawdd_bbox_raw",
        "notes",
    ]

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            {"train": 0, "val": 1, "test": 2}[assignment[row["subject_id"]]],
            row["subject_id"],
            row["frame_index"],
        ),
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted_rows:
            gender, glasses = subject_attributes(row["subject_id"])
            writer.writerow(
                {
                    "subject_id": row["subject_id"],
                    "split": assignment[row["subject_id"]],
                    "gender": gender,
                    "glasses": glasses,
                    "frame_index": row["frame_index"],
                    "original_path": row["image_path"],
                    "processed_path": row["mouth_crop_path"],
                    "label": row["binary_label"],
                    "class_id": row["class_id"],
                    "crop_method": row["crop_method"],
                    "mouth_bbox_xyxy": row["mouth_bbox_xyxy"],
                    "image_path": row["image_path"],
                    "mouth_crop_path": row["mouth_crop_path"],
                    "raw_video_path": row["raw_video_path"],
                    "annotation_txt_path": row["annotation_txt_path"],
                    "yawdd_bbox_raw": row["yawdd_bbox_raw"],
                    "notes": row["notes"],
                }
            )


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{100 * numerator / denominator:.2f}%"


def leakage_ok(assignment: dict[str, str]) -> tuple[bool, dict[tuple[str, str], set[str]]]:
    split_to_subjects: dict[str, set[str]] = defaultdict(set)
    for subject_id, split in assignment.items():
        split_to_subjects[split].add(subject_id)
    overlaps = {}
    split_names = list(SPLIT_SUBJECT_COUNTS)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlaps[(left, right)] = split_to_subjects[left] & split_to_subjects[right]
    return all(not subjects for subjects in overlaps.values()), overlaps


def write_report(
    path: Path,
    source_manifest: Path,
    trainable_manifest: Path,
    split_manifest: Path,
    seed: int,
    iterations: int,
    raw_rows: list[dict[str, str]],
    trainable_rows: list[dict[str, str]],
    subject_stats: dict[str, dict[str, object]],
    assignment: dict[str, str],
    best_score: float,
    missing_crops: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_assignment(assignment, subject_stats)
    total_images = len(trainable_rows)
    overall_labels = Counter(row["binary_label"] for row in trainable_rows)
    leak_free, overlaps = leakage_ok(assignment)
    exact_subject_counts = all(
        len(summary[split]["subjects"]) == expected
        for split, expected in SPLIT_SUBJECT_COUNTS.items()
    )
    all_splits_have_both_classes = all(
        summary[split]["labels"]["no_yawn"] > 0 and summary[split]["labels"]["yawn"] > 0
        for split in SPLIT_SUBJECT_COUNTS
    )
    no_failed_rows = all(row["crop_method"] != "failed" for row in trainable_rows)
    ready = (
        leak_free
        and exact_subject_counts
        and all_splits_have_both_classes
        and no_failed_rows
        and missing_crops == 0
    )

    lines = [
        "# YawDD+ Dash - Stage 6 Subject-Level Split Report",
        "",
        "## Inputs and outputs",
        "",
        f"- Source mouth-crop manifest: `{source_manifest}`",
        f"- Trainable manifest: `{trainable_manifest}`",
        f"- Split manifest: `{split_manifest}`",
        "- Split unit: subject_id (no image-level randomization)",
        f"- Split search seed: {seed}",
        f"- Split search iterations: {iterations}",
        f"- Split search score: {best_score:.6f}",
        "",
        "## Filtering",
        "",
        f"- Source rows: **{len(raw_rows)}**",
        f"- Excluded rows where `crop_method == failed`: **{len(raw_rows) - len(trainable_rows)}**",
        f"- Trainable rows: **{len(trainable_rows)}**",
        f"- Missing crop files among trainable rows: **{missing_crops}**",
        "",
        "## Overall class distribution",
        "",
        "| Class | Images | Percent |",
        "|---|---:|---:|",
    ]
    for label in LABELS:
        lines.append(f"| `{label}` | {overall_labels[label]} | {pct(overall_labels[label], total_images)} |")

    lines.extend(
        [
            "",
            "## Split distribution",
            "",
            "| Split | Subjects | Images | Image % | no_yawn | no_yawn % | yawn | yawn % | Yawn rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLIT_SUBJECT_COUNTS:
        split_total = int(summary[split]["total"])
        labels = summary[split]["labels"]
        assert isinstance(labels, Counter)
        no_yawn = labels["no_yawn"]
        yawn = labels["yawn"]
        lines.append(
            f"| `{split}` | {len(summary[split]['subjects'])} | {split_total} | "
            f"{pct(split_total, total_images)} | {no_yawn} | {pct(no_yawn, split_total)} | "
            f"{yawn} | {pct(yawn, split_total)} | {pct(yawn, split_total)} |"
        )

    lines.extend(
        [
            "",
            "## Subject attribute distribution",
            "",
            "| Split | Female subjects | Male subjects | Glasses subjects | NoGlasses subjects |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in SPLIT_SUBJECT_COUNTS:
        gender_counts = summary[split]["gender"]
        glasses_counts = summary[split]["glasses"]
        assert isinstance(gender_counts, Counter)
        assert isinstance(glasses_counts, Counter)
        lines.append(
            f"| `{split}` | {gender_counts['Female']} | {gender_counts['Male']} | "
            f"{glasses_counts['Glasses']} | {glasses_counts['NoGlasses']} |"
        )

    lines.extend(
        [
            "",
            "## Leakage check",
            "",
            f"- Unique trainable subjects: **{len(subject_stats)}**",
            f"- Requested subject counts met: **{'YES' if exact_subject_counts else 'NO'}**",
            f"- No subject appears in more than one split: **{'YES' if leak_free else 'NO'}**",
            f"- No failed crop rows in trainable data: **{'YES' if no_failed_rows else 'NO'}**",
            f"- Every split contains both classes: **{'YES' if all_splits_have_both_classes else 'NO'}**",
            f"- All referenced mouth-crop files exist: **{'YES' if missing_crops == 0 else 'NO'}**",
        ]
    )
    if not leak_free:
        for pair, subjects in overlaps.items():
            lines.append(f"- Overlap `{pair[0]}` / `{pair[1]}`: {sorted(subjects)}")

    lines.extend(
        [
            "",
            "## Subject assignments",
            "",
            "| Split | subject_id | Gender | Glasses | Images | no_yawn | yawn |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for split in SPLIT_SUBJECT_COUNTS:
        for subject_id in sorted(summary[split]["subjects"]):
            stats = subject_stats[subject_id]
            labels = stats["labels"]
            assert isinstance(labels, Counter)
            lines.append(
                f"| `{split}` | `{subject_id}` | {stats['gender']} | {stats['glasses']} | "
                f"{stats['total']} | {labels['no_yawn']} | {labels['yawn']} |"
            )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "**READY** - the train/validation/test split is leakage-safe and ready "
                "for Stage 7 CNN training."
                if ready
                else "**NOT READY** - fix the failed checks above before CNN training."
            ),
        ]
    )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stage 6 YawDD+ Dash subject-level split.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=repo_root() / "artifacts" / "mappings" / "yawdd_dash_all_mouth_crops.csv",
    )
    parser.add_argument(
        "--trainable-manifest",
        type=Path,
        default=repo_root() / "artifacts" / "mappings" / "yawdd_dash_all_mouth_crops_trainable.csv",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=repo_root() / "artifacts" / "splits" / "yawdd_dash_subject_split.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root() / "reports" / "yawdd_dash_split_report.md",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--iterations",
        type=int,
        default=200_000,
        help="Random subject-assignment candidates to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = read_manifest(args.input_manifest)
    trainable_rows = filter_trainable(raw_rows)
    if not trainable_rows:
        raise SystemExit("No trainable rows found after excluding failed crops.")

    with args.input_manifest.open(newline="") as f:
        source_fieldnames = list(csv.DictReader(f).fieldnames or [])

    subject_stats = build_subject_stats(trainable_rows)
    assignment, best_score = choose_subject_split(
        subject_stats=subject_stats,
        seed=args.seed,
        iterations=args.iterations,
    )
    missing_crops = check_missing_crops(trainable_rows)

    write_trainable_manifest(
        args.trainable_manifest,
        trainable_rows,
        assignment,
        source_fieldnames,
    )
    write_split_manifest(args.split_manifest, trainable_rows, assignment)
    write_report(
        path=args.report,
        source_manifest=args.input_manifest,
        trainable_manifest=args.trainable_manifest,
        split_manifest=args.split_manifest,
        seed=args.seed,
        iterations=args.iterations,
        raw_rows=raw_rows,
        trainable_rows=trainable_rows,
        subject_stats=subject_stats,
        assignment=assignment,
        best_score=best_score,
        missing_crops=missing_crops,
    )

    print(f"Wrote trainable manifest: {args.trainable_manifest}")
    print(f"Wrote split manifest: {args.split_manifest}")
    print(f"Wrote split report: {args.report}")


if __name__ == "__main__":
    main()
