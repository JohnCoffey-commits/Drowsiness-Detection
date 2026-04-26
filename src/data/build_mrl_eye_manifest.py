"""Build clean MRL Eye manifests from the raw image folders.

Outputs:

    artifacts/mappings/mrl_eye_all_images.csv
    artifacts/mappings/mrl_eye_trainable.csv

Run from the project root:

    python src/data/build_mrl_eye_manifest.py
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "mrlEyes_2018_01"
DEFAULT_ALL_OUT = PROJECT_ROOT / "artifacts" / "mappings" / "mrl_eye_all_images.csv"
DEFAULT_TRAINABLE_OUT = (
    PROJECT_ROOT / "artifacts" / "mappings" / "mrl_eye_trainable.csv"
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FILENAME_RE = re.compile(
    r"^(?P<subject_id>s\d{3,4})_"
    r"(?P<image_id>\d+)_"
    r"(?P<gender>[01])_"
    r"(?P<glasses>[01])_"
    r"(?P<eye_state>[01])_"
    r"(?P<reflections>[012])_"
    r"(?P<lighting>[01])_"
    r"(?P<sensor_id>\d{2})$"
)
SUBJECT_RE = re.compile(r"^s\d{3,4}$")

GENDER_MAP = {"0": "male", "1": "female"}
GLASSES_MAP = {"0": "no", "1": "yes"}
LABEL_NAME_MAP = {0: "closed", 1: "open"}
REFLECTIONS_MAP = {"0": "none", "1": "low", "2": "high"}
LIGHTING_MAP = {"0": "bad", "1": "good"}
SENSOR_IDS = {"01", "02", "03"}


def annotation_eye_state_mapping(annotation_path: Path) -> dict[str, str]:
    if not annotation_path.is_file():
        return {}
    lines = annotation_path.read_text(encoding="utf-8", errors="replace").splitlines()
    mapping: dict[str, str] = {}
    in_eye_state = False
    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("eye state"):
            in_eye_state = True
            continue
        if in_eye_state and not line:
            break
        if in_eye_state:
            match = re.match(r"^([01])\s*-\s*(.+)$", lower)
            if match:
                value = match.group(2).strip()
                if value == "close":
                    value = "closed"
                mapping[match.group(1)] = value
    return mapping


def validate_annotation_label_mapping(data_root: Path) -> None:
    mapping = annotation_eye_state_mapping(data_root / "annotation.txt")
    expected = {"0": "closed", "1": "open"}
    normalized = {
        key: ("closed" if value.split()[0] == "close" else value.split()[0])
        for key, value in mapping.items()
    }
    if normalized != expected:
        raise SystemExit(
            "MRL annotation label mapping is missing or inconsistent with "
            f"0=closed, 1=open. Parsed mapping: {mapping}"
        )


def image_paths(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        raise SystemExit(f"Dataset root not found: {data_root}")
    return sorted(
        p
        for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_filename(path: Path) -> dict[str, Any]:
    parent_subject = path.parent.name if SUBJECT_RE.match(path.parent.name) else ""
    match = FILENAME_RE.match(path.stem)
    if not match:
        return {
            "subject_id": parent_subject or pd.NA,
            "image_id": pd.NA,
            "gender": pd.NA,
            "glasses": pd.NA,
            "eye_state": pd.NA,
            "label": pd.NA,
            "label_name": pd.NA,
            "reflections": pd.NA,
            "lighting": pd.NA,
            "sensor_id": pd.NA,
            "parse_ok": False,
            "parse_error": "filename does not match expected MRL format",
        }

    parts = match.groupdict()
    errors: list[str] = []
    if parent_subject and parent_subject != parts["subject_id"]:
        errors.append(
            f"subject folder {parent_subject} conflicts with filename {parts['subject_id']}"
        )
    if parts["sensor_id"] not in SENSOR_IDS:
        errors.append(f"unknown sensor_id {parts['sensor_id']}")

    label = int(parts["eye_state"])
    return {
        "subject_id": parts["subject_id"],
        "image_id": parts["image_id"],
        "gender": GENDER_MAP[parts["gender"]],
        "glasses": GLASSES_MAP[parts["glasses"]],
        "eye_state": label,
        "label": label,
        "label_name": LABEL_NAME_MAP[label],
        "reflections": REFLECTIONS_MAP[parts["reflections"]],
        "lighting": LIGHTING_MAP[parts["lighting"]],
        "sensor_id": parts["sensor_id"],
        "parse_ok": not errors,
        "parse_error": "; ".join(errors),
    }


def inspect_image(path: Path) -> tuple[bool, int | None, int | None, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return True, width, height, ""
    except (OSError, UnidentifiedImageError) as exc:
        return False, None, None, str(exc)


def build_manifest(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in image_paths(data_root):
        parsed = parse_filename(path)
        exists = path.exists()
        read_ok, width, height, read_error = inspect_image(path) if exists else (False, None, None, "missing file")
        parse_ok = bool(parsed["parse_ok"])
        label = parsed["label"]
        valid_label = label in {0, 1}
        subject_id = parsed["subject_id"]
        has_subject = not pd.isna(subject_id) and str(subject_id).strip() != ""
        is_valid = bool(parse_ok and read_ok and exists and valid_label and has_subject)
        error = "; ".join(
            text
            for text in [parsed.pop("parse_error", ""), read_error]
            if text
        )
        rows.append(
            {
                "image_path": path.resolve().as_posix(),
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "filename": path.name,
                **parsed,
                "width": width,
                "height": height,
                "extension": path.suffix.lower(),
                "is_valid": is_valid,
                "read_ok": read_ok,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def trainable_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    trainable = df[
        (df["parse_ok"] == True)
        & (df["read_ok"] == True)
        & (df["is_valid"] == True)
        & (df["label"].isin([0, 1]))
        & (df["subject_id"].notna())
        & (df["subject_id"].astype(str).str.strip() != "")
        & (df["image_path"].map(lambda p: Path(p).is_file()))
    ].copy()
    return trainable


def write_outputs(df: pd.DataFrame, all_out: Path, trainable_out: Path) -> pd.DataFrame:
    all_out.parent.mkdir(parents=True, exist_ok=True)
    trainable_out.parent.mkdir(parents=True, exist_ok=True)
    trainable = trainable_rows(df)
    df.to_csv(all_out, index=False)
    trainable.to_csv(trainable_out, index=False)
    return trainable


def print_summary(df: pd.DataFrame, trainable: pd.DataFrame, all_out: Path, trainable_out: Path) -> None:
    print(f"Total images found: {len(df)}")
    print(f"All manifest rows: {len(df)}")
    print(f"Trainable rows: {len(trainable)}")
    print(f"Invalid rows: {len(df) - len(trainable)}")
    print(f"Class distribution: {dict(Counter(trainable['label_name'])) if not trainable.empty else {}}")
    print(f"Subject count: {trainable['subject_id'].nunique() if not trainable.empty else 0}")
    print(f"Output paths: {all_out}, {trainable_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--all-out", type=Path, default=DEFAULT_ALL_OUT)
    parser.add_argument("--trainable-out", type=Path, default=DEFAULT_TRAINABLE_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_annotation_label_mapping(args.data_root)
    df = build_manifest(args.data_root)
    trainable = write_outputs(df, args.all_out, args.trainable_out)
    print_summary(df, trainable, args.all_out, args.trainable_out)


if __name__ == "__main__":
    main()
