"""Create leakage-aware subject-level splits for NTHUDDD2 Kaggle JPG frames.

The split is by parsed subject ID, never by individual image. If subject IDs
are missing or invalid, the script stops instead of falling back to image-level
random splitting.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "mappings"
    / "nthuddd2_kaggle_all_images_trainable.csv"
)
DEFAULT_SUBJECT_SPLIT_OUT = (
    PROJECT_ROOT / "artifacts" / "splits" / "nthuddd2_kaggle_subject_split.csv"
)
DEFAULT_MANIFEST_WITH_SPLIT_OUT = (
    PROJECT_ROOT
    / "artifacts"
    / "mappings"
    / "nthuddd2_kaggle_all_images_trainable_with_split.csv"
)

SPLITS = ("train", "val", "test")
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def validate_manifest(df: pd.DataFrame, input_path: Path) -> None:
    required = {"subject_id", "class_id", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"Input manifest is missing required column(s): {missing}. "
            f"Input: {input_path}"
        )
    if df.empty:
        raise SystemExit(f"Input manifest has no rows: {input_path}")

    subject_text = df["subject_id"].astype("string").str.strip()
    invalid_subjects = subject_text.isna() | subject_text.eq("") | subject_text.eq("nan")
    if invalid_subjects.any():
        count = int(invalid_subjects.sum())
        raise SystemExit(
            f"Cannot create a subject-level split: {count} row(s) have missing "
            "or invalid subject_id. Refusing to fall back to image-level splitting."
        )

    class_values = set(pd.to_numeric(df["class_id"], errors="coerce").dropna().astype(int))
    if not {0, 1}.issubset(class_values):
        raise SystemExit(
            "Cannot create a binary split because class_id values do not include "
            "both 0 and 1."
        )


def build_subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.assign(class_id=pd.to_numeric(df["class_id"], errors="raise").astype(int))
        .groupby("subject_id", as_index=False)
        .agg(
            num_images=("filename", "size"),
            num_drowsy=("class_id", lambda s: int((s == 1).sum())),
            num_notdrowsy=("class_id", lambda s: int((s == 0).sum())),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    return stats


def summarize_assignment(
    subject_stats: pd.DataFrame, assignment: dict[str, str]
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_subjects = [sid for sid, value in assignment.items() if value == split]
        split_stats = subject_stats[subject_stats["subject_id"].isin(split_subjects)]
        summaries[split] = {
            "subjects": split_subjects,
            "num_images": int(split_stats["num_images"].sum()),
            "num_drowsy": int(split_stats["num_drowsy"].sum()),
            "num_notdrowsy": int(split_stats["num_notdrowsy"].sum()),
        }
    return summaries


def valid_assignment(subject_stats: pd.DataFrame, assignment: dict[str, str]) -> bool:
    summaries = summarize_assignment(subject_stats, assignment)
    for split in SPLITS:
        summary = summaries[split]
        if not summary["subjects"]:
            return False
        if summary["num_drowsy"] == 0 or summary["num_notdrowsy"] == 0:
            return False
    return True


def score_assignment(subject_stats: pd.DataFrame, assignment: dict[str, str]) -> float:
    summaries = summarize_assignment(subject_stats, assignment)
    total_images = int(subject_stats["num_images"].sum())
    total_drowsy = int(subject_stats["num_drowsy"].sum())
    overall_drowsy_rate = total_drowsy / total_images

    score = 0.0
    for split in SPLITS:
        summary = summaries[split]
        split_images = summary["num_images"]
        if split_images == 0:
            return float("inf")

        image_ratio = split_images / total_images
        drowsy_rate = summary["num_drowsy"] / split_images
        score += abs(image_ratio - TARGET_RATIOS[split]) * 10.0
        score += abs(drowsy_rate - overall_drowsy_rate) * 3.0

        if summary["num_drowsy"] == 0 or summary["num_notdrowsy"] == 0:
            score += 1_000.0
    return score


def exhaustive_assignments(subject_ids: list[str]) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    for values in itertools.product(SPLITS, repeat=len(subject_ids)):
        assignment = dict(zip(subject_ids, values))
        if set(assignment.values()) == set(SPLITS):
            assignments.append(assignment)
    return assignments


def random_assignments(
    subject_ids: list[str], seed: int, iterations: int
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    assignments: list[dict[str, str]] = []
    for _ in range(iterations):
        shuffled = subject_ids[:]
        rng.shuffle(shuffled)
        assignment = {
            shuffled[0]: "train",
            shuffled[1]: "val",
            shuffled[2]: "test",
        }
        for subject_id in shuffled[3:]:
            assignment[subject_id] = rng.choices(
                population=list(SPLITS),
                weights=[TARGET_RATIOS[s] for s in SPLITS],
                k=1,
            )[0]
        assignments.append(assignment)
    return assignments


def choose_split(
    subject_stats: pd.DataFrame, seed: int, iterations: int
) -> dict[str, str]:
    subject_ids = sorted(subject_stats["subject_id"].astype(str).tolist())
    if len(subject_ids) < 3:
        raise SystemExit(
            "Cannot create train/val/test subject-level splits with fewer than "
            f"3 parsed subjects. Found {len(subject_ids)}."
        )

    if len(subject_ids) <= 12:
        candidates = exhaustive_assignments(subject_ids)
        rng = random.Random(seed)
        rng.shuffle(candidates)
    else:
        candidates = random_assignments(subject_ids, seed, iterations)

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for assignment in candidates:
        if not valid_assignment(subject_stats, assignment):
            continue
        score = score_assignment(subject_stats, assignment)
        if score < best_score:
            best_assignment = assignment
            best_score = score

    if best_assignment is None:
        raise SystemExit(
            "Could not create a subject-level split where every split contains "
            "both classes. Refusing to fall back to image-level splitting."
        )
    return best_assignment


def write_outputs(
    df: pd.DataFrame,
    subject_stats: pd.DataFrame,
    assignment: dict[str, str],
    subject_split_out: Path,
    manifest_with_split_out: Path,
) -> None:
    subject_split_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_with_split_out.parent.mkdir(parents=True, exist_ok=True)

    split_df = subject_stats.copy()
    split_df["split"] = split_df["subject_id"].map(assignment)
    split_df = split_df[
        ["subject_id", "split", "num_images", "num_drowsy", "num_notdrowsy"]
    ].sort_values(["split", "subject_id"])
    split_df.to_csv(subject_split_out, index=False)

    with_split = df.copy()
    with_split["split"] = with_split["subject_id"].map(assignment)
    with_split.to_csv(manifest_with_split_out, index=False)


def leakage_check(df: pd.DataFrame) -> bool:
    split_counts = df.groupby("subject_id")["split"].nunique()
    return bool((split_counts == 1).all())


def print_summary(
    df: pd.DataFrame, subject_split_out: Path, manifest_with_split_out: Path
) -> None:
    print(f"wrote: {subject_split_out}")
    print(f"wrote: {manifest_with_split_out}")
    print(f"subject_leakage_free: {leakage_check(df)}")
    for split in SPLITS:
        split_df = df[df["split"] == split]
        print(f"{split}_subjects: {split_df['subject_id'].nunique()}")
        print(f"{split}_images: {len(split_df)}")
        print(
            f"{split}_class_counts: "
            f"{split_df['label'].value_counts().sort_index().to_dict()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--subject-split-out", type=Path, default=DEFAULT_SUBJECT_SPLIT_OUT)
    parser.add_argument(
        "--manifest-with-split-out",
        type=Path,
        default=DEFAULT_MANIFEST_WITH_SPLIT_OUT,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--iterations", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input manifest not found: {args.input}")

    df = pd.read_csv(args.input, dtype={"subject_id": str})
    validate_manifest(df, args.input)
    subject_stats = build_subject_stats(df)
    assignment = choose_split(subject_stats, args.seed, args.iterations)
    write_outputs(
        df,
        subject_stats,
        assignment,
        args.subject_split_out,
        args.manifest_with_split_out,
    )

    with_split = df.copy()
    with_split["split"] = with_split["subject_id"].map(assignment)
    print_summary(with_split, args.subject_split_out, args.manifest_with_split_out)


if __name__ == "__main__":
    main()
