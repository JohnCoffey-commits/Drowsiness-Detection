"""Create a leakage-safe subject-level split for MRL Eye.

Input:

    artifacts/mappings/mrl_eye_trainable.csv

Outputs:

    artifacts/splits/mrl_eye_subject_split.csv
    artifacts/mappings/mrl_eye_trainable_with_split.csv
    reports/mrl_eye_split_report.md

Run from the project root:

    python src/data/split_mrl_eye_subjects.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "artifacts" / "mappings" / "mrl_eye_trainable.csv"
DEFAULT_SUBJECT_SPLIT_OUT = (
    PROJECT_ROOT / "artifacts" / "splits" / "mrl_eye_subject_split.csv"
)
DEFAULT_MANIFEST_WITH_SPLIT_OUT = (
    PROJECT_ROOT / "artifacts" / "mappings" / "mrl_eye_trainable_with_split.csv"
)
DEFAULT_REPORT_OUT = PROJECT_ROOT / "reports" / "mrl_eye_split_report.md"

SPLITS = ("train", "val", "test")
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def validate_manifest(df: pd.DataFrame, input_path: Path) -> None:
    required = {"image_path", "filename", "subject_id", "label", "label_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input manifest is missing columns {missing}: {input_path}")
    if df.empty:
        raise SystemExit(f"Input manifest has no trainable rows: {input_path}")

    labels = set(pd.to_numeric(df["label"], errors="coerce").dropna().astype(int))
    if labels != {0, 1}:
        raise SystemExit(f"Expected binary labels {{0, 1}}, found {sorted(labels)}")

    bad_names = df[~df["label_name"].isin(["closed", "open"])]
    if not bad_names.empty:
        raise SystemExit("Unexpected label_name values found in trainable manifest.")

    subject_text = df["subject_id"].astype("string").str.strip()
    missing_subjects = subject_text.isna() | subject_text.eq("") | subject_text.eq("nan")
    if missing_subjects.any():
        raise SystemExit(
            f"Cannot split by subject: {int(missing_subjects.sum())} rows have missing subject_id."
        )


def target_subject_counts(num_subjects: int) -> dict[str, int]:
    test_count = max(1, round(num_subjects * TARGET_RATIOS["test"]))
    val_count = max(1, round(num_subjects * TARGET_RATIOS["val"]))
    train_count = num_subjects - val_count - test_count
    if train_count < 1:
        raise SystemExit(f"Need at least 3 subjects for train/val/test; found {num_subjects}")
    return {"train": train_count, "val": val_count, "test": test_count}


def build_subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    labels = pd.to_numeric(df["label"], errors="raise").astype(int)
    work = df.assign(label=labels)
    stats = (
        work.groupby("subject_id", as_index=False)
        .agg(
            num_images=("filename", "size"),
            num_closed=("label", lambda s: int((s == 0).sum())),
            num_open=("label", lambda s: int((s == 1).sum())),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    stats["closed_ratio"] = stats["num_closed"] / stats["num_images"]
    stats["open_ratio"] = stats["num_open"] / stats["num_images"]
    return stats


def assignment_from_order(subject_ids: list[str], counts: dict[str, int]) -> dict[str, str]:
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    assignment: dict[str, str] = {}
    for subject_id in subject_ids[:train_end]:
        assignment[subject_id] = "train"
    for subject_id in subject_ids[train_end:val_end]:
        assignment[subject_id] = "val"
    for subject_id in subject_ids[val_end:]:
        assignment[subject_id] = "test"
    return assignment


def summarize_assignment(
    subject_stats: pd.DataFrame, assignment: dict[str, str]
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_stats = subject_stats[subject_stats["subject_id"].map(assignment) == split]
        num_images = int(split_stats["num_images"].sum())
        num_closed = int(split_stats["num_closed"].sum())
        num_open = int(split_stats["num_open"].sum())
        summaries[split] = {
            "num_subjects": int(len(split_stats)),
            "num_images": num_images,
            "num_closed": num_closed,
            "num_open": num_open,
            "closed_ratio": num_closed / num_images if num_images else 0.0,
            "open_ratio": num_open / num_images if num_images else 0.0,
        }
    return summaries


def valid_assignment(subject_stats: pd.DataFrame, assignment: dict[str, str]) -> bool:
    summaries = summarize_assignment(subject_stats, assignment)
    for split in SPLITS:
        if summaries[split]["num_subjects"] == 0:
            return False
        if summaries[split]["num_closed"] == 0 or summaries[split]["num_open"] == 0:
            return False
    return True


def score_assignment(subject_stats: pd.DataFrame, assignment: dict[str, str]) -> float:
    summaries = summarize_assignment(subject_stats, assignment)
    total_images = int(subject_stats["num_images"].sum())
    total_closed = int(subject_stats["num_closed"].sum())
    total_open = int(subject_stats["num_open"].sum())
    overall_closed_ratio = total_closed / total_images
    overall_open_ratio = total_open / total_images

    score = 0.0
    for split in SPLITS:
        summary = summaries[split]
        if summary["num_images"] == 0:
            return float("inf")
        image_ratio = summary["num_images"] / total_images
        score += abs(image_ratio - TARGET_RATIOS[split]) * 10.0
        score += abs(summary["closed_ratio"] - overall_closed_ratio) * 4.0
        score += abs(summary["open_ratio"] - overall_open_ratio) * 4.0
    return score


def choose_split(
    subject_stats: pd.DataFrame,
    seed: int,
    iterations: int,
) -> tuple[dict[str, str], dict[str, int], float]:
    subject_ids = sorted(subject_stats["subject_id"].astype(str).tolist())
    counts = target_subject_counts(len(subject_ids))
    rng = random.Random(seed)

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for _ in range(iterations):
        shuffled = subject_ids[:]
        rng.shuffle(shuffled)
        assignment = assignment_from_order(shuffled, counts)
        if not valid_assignment(subject_stats, assignment):
            continue
        score = score_assignment(subject_stats, assignment)
        if score < best_score:
            best_assignment = assignment
            best_score = score

    if best_assignment is None:
        raise SystemExit(
            "Could not create a subject-level split where every split contains both classes."
        )
    return best_assignment, counts, best_score


def verify_split(df: pd.DataFrame) -> dict[str, bool]:
    split_counts = df.groupby("subject_id")["split"].nunique()
    leakage_free = bool((split_counts == 1).all())
    no_missing_split = bool(df["split"].notna().all() and df["split"].isin(SPLITS).all())
    each_image_once = bool(len(df) == len(df.index) and no_missing_split)
    both_classes = True
    for split in SPLITS:
        labels = set(pd.to_numeric(df[df["split"] == split]["label"], errors="coerce").dropna().astype(int))
        both_classes = both_classes and labels == {0, 1}
    paths_exist = bool(df["image_path"].map(lambda p: Path(p).is_file()).all())
    return {
        "leakage_free": leakage_free,
        "no_missing_split": no_missing_split,
        "each_image_once": each_image_once,
        "each_split_has_both_classes": both_classes,
        "paths_exist": paths_exist,
    }


def write_outputs(
    df: pd.DataFrame,
    subject_stats: pd.DataFrame,
    assignment: dict[str, str],
    subject_split_out: Path,
    manifest_with_split_out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_split_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_with_split_out.parent.mkdir(parents=True, exist_ok=True)

    split_order = {split: idx for idx, split in enumerate(SPLITS)}
    subject_split = subject_stats.copy()
    subject_split["split"] = subject_split["subject_id"].map(assignment)
    subject_split = subject_split[
        [
            "subject_id",
            "split",
            "num_images",
            "num_closed",
            "num_open",
            "closed_ratio",
            "open_ratio",
        ]
    ].sort_values(["split", "subject_id"], key=lambda s: s.map(split_order).fillna(s) if s.name == "split" else s)

    with_split = df.copy()
    with_split["split"] = with_split["subject_id"].map(assignment)
    subject_split.to_csv(subject_split_out, index=False)
    with_split.to_csv(manifest_with_split_out, index=False)
    return subject_split, with_split


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    headers = [str(column) for column in df.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    with_split: pd.DataFrame,
    checks: dict[str, bool],
    seed: int,
    score: float,
    report_out: Path,
) -> None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_df = with_split[with_split["split"] == split]
        num_images = len(split_df)
        num_closed = int((split_df["label"] == 0).sum())
        num_open = int((split_df["label"] == 1).sum())
        rows.append(
            {
                "split": split,
                "subjects": split_df["subject_id"].nunique(),
                "images": num_images,
                "closed": num_closed,
                "open": num_open,
                "closed_ratio": round(num_closed / num_images, 4) if num_images else 0.0,
                "open_ratio": round(num_open / num_images, 4) if num_images else 0.0,
            }
        )
    summary = pd.DataFrame(rows)
    verdict = "READY for CNN baseline training" if all(checks.values()) else "BLOCKED"
    lines = [
        "# MRL Eye Subject Split Report",
        "",
        f"- Random seed: {seed}",
        f"- Split search score: {score:.6f}",
        f"- Final verdict: **{verdict}**",
        "",
        "## Split Summary",
        "",
        markdown_table(summary),
        "",
        "## Verification",
        "",
        f"- Leakage check result: {checks['leakage_free']}",
        f"- Missing split label check result: {checks['no_missing_split']}",
        f"- Every image receives exactly one split: {checks['each_image_once']}",
        f"- Every split contains closed and open: {checks['each_split_has_both_classes']}",
        f"- Missing file check result: {checks['paths_exist']}",
        "",
    ]
    report_out.write_text("\n".join(lines), encoding="utf-8")


def print_summary(with_split: pd.DataFrame, checks: dict[str, bool], outputs: list[Path]) -> None:
    for output in outputs:
        print(f"Wrote: {output}")
    for split in SPLITS:
        split_df = with_split[with_split["split"] == split]
        class_counts = split_df["label_name"].value_counts().sort_index().to_dict()
        print(f"{split} subjects: {split_df['subject_id'].nunique()}")
        print(f"{split} images: {len(split_df)}")
        print(f"{split} class counts: {class_counts}")
    print(f"Leakage check result: {checks['leakage_free']}")
    print(f"Missing file check result: {checks['paths_exist']}")
    print(f"Final verdict: {'READY for CNN baseline training' if all(checks.values()) else 'BLOCKED'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--subject-split-out", type=Path, default=DEFAULT_SUBJECT_SPLIT_OUT)
    parser.add_argument("--manifest-with-split-out", type=Path, default=DEFAULT_MANIFEST_WITH_SPLIT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--iterations", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input manifest not found: {args.input}")
    df = pd.read_csv(args.input, dtype={"subject_id": str, "sensor_id": str})
    validate_manifest(df, args.input)
    subject_stats = build_subject_stats(df)
    assignment, _counts, score = choose_split(subject_stats, args.seed, args.iterations)
    subject_split, with_split = write_outputs(
        df,
        subject_stats,
        assignment,
        args.subject_split_out,
        args.manifest_with_split_out,
    )
    checks = verify_split(with_split)
    write_report(with_split, checks, args.seed, score, args.report_out)
    print_summary(
        with_split,
        checks,
        [args.subject_split_out, args.manifest_with_split_out, args.report_out],
    )


if __name__ == "__main__":
    main()
