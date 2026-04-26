"""Build a manifest for the Kaggle extracted-frame NTHUDDD2 JPG dataset.

This script is intentionally additive and does not touch any YawDD assets.
It scans:

    dataset/NTHUDDD2/train_data/drowsy
    dataset/NTHUDDD2/train_data/notdrowsy

and writes:

    artifacts/mappings/nthuddd2_kaggle_all_images.csv
    artifacts/mappings/nthuddd2_kaggle_all_images_trainable.csv

No frames are extracted, no images are resized, and original JPG files are
never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "NTHUDDD2" / "train_data"
DEFAULT_ALL_OUT = (
    PROJECT_ROOT / "artifacts" / "mappings" / "nthuddd2_kaggle_all_images.csv"
)
DEFAULT_TRAINABLE_OUT = (
    PROJECT_ROOT
    / "artifacts"
    / "mappings"
    / "nthuddd2_kaggle_all_images_trainable.csv"
)

CLASS_TO_ID = {
    "notdrowsy": 0,
    "drowsy": 1,
}
SOURCE_DATASET = "nthuddd2_kaggle"
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def parse_filename(path: Path) -> dict[str, Any]:
    parts = path.stem.split("_")
    subject_id = parts[0] if parts else ""
    condition = parts[1] if len(parts) > 1 else ""

    frame_index = None
    frame_token_index = None
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].isdigit():
            frame_index = int(parts[idx])
            frame_token_index = idx
            break

    state_candidates: list[str] = []
    for idx, token in enumerate(parts[2:], start=2):
        if idx == frame_token_index:
            continue
        if token in CLASS_TO_ID:
            continue
        if token.isdigit():
            continue
        state_candidates.append(token)

    return {
        "subject_id": subject_id,
        "condition": condition,
        "state_token": "_".join(state_candidates),
        "frame_index": frame_index,
    }


def inspect_image(path: Path) -> tuple[bool, int | None, int | None]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return True, width, height
    except (OSError, UnidentifiedImageError):
        return False, None, None


def iter_image_paths(data_root: Path) -> list[Path]:
    missing = [
        class_dir for class_dir in CLASS_TO_ID if not (data_root / class_dir).is_dir()
    ]
    if missing:
        missing_text = ", ".join(str(data_root / class_dir) for class_dir in missing)
        raise SystemExit(f"Required class folder(s) not found: {missing_text}")

    paths: list[Path] = []
    for label in CLASS_TO_ID:
        class_dir = data_root / label
        paths.extend(
            sorted(
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        )
    return paths


def build_manifest(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_path in iter_image_paths(data_root):
        label = image_path.parent.name
        parsed = parse_filename(image_path)
        image_ok, width, height = inspect_image(image_path)

        rows.append(
            {
                "image_path": image_path.resolve().as_posix(),
                "relative_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "filename": image_path.name,
                "label": label,
                "class_id": CLASS_TO_ID[label],
                "subject_id": parsed["subject_id"],
                "condition": parsed["condition"],
                "state_token": parsed["state_token"],
                "frame_index": parsed["frame_index"],
                "source_dataset": SOURCE_DATASET,
                "image_ok": image_ok,
                "width": width,
                "height": height,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, all_out: Path, trainable_out: Path) -> None:
    all_out.parent.mkdir(parents=True, exist_ok=True)
    trainable_out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(all_out, index=False)
    trainable = df[df["image_ok"]].copy()
    trainable.to_csv(trainable_out, index=False)


def print_summary(df: pd.DataFrame, all_out: Path, trainable_out: Path) -> None:
    class_counts = df["label"].value_counts().sort_index().to_dict()
    unreadable = int((~df["image_ok"]).sum())
    subjects = int(df["subject_id"].nunique(dropna=True))

    print(f"wrote: {all_out}")
    print(f"wrote: {trainable_out}")
    print(f"total_images: {len(df)}")
    print(f"class_counts: {class_counts}")
    print(f"num_subjects: {subjects}")
    print(f"unreadable_images: {unreadable}")
    print(f"trainable_images: {len(df) - unreadable}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--all-out", type=Path, default=DEFAULT_ALL_OUT)
    parser.add_argument("--trainable-out", type=Path, default=DEFAULT_TRAINABLE_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_manifest(args.data_root)
    write_outputs(df, args.all_out, args.trainable_out)
    print_summary(df, args.all_out, args.trainable_out)


if __name__ == "__main__":
    main()
