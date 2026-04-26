"""Inspect the raw MRL Eye dataset and write a preparation report.

This script is intentionally read-only for the raw dataset. It confirms the
MRL filename label mapping, scans images recursively, and writes:

    reports/mrl_eye_dataset_report.md

Run from the project root:

    python src/data/inspect_mrl_eye.py
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "mrlEyes_2018_01"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "reports" / "mrl_eye_dataset_report.md"

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
EYE_STATE_MAP = {"0": "closed", "1": "open"}
REFLECTIONS_MAP = {"0": "none", "1": "low", "2": "high"}
LIGHTING_MAP = {"0": "bad", "1": "good"}
SENSOR_MAP = {
    "01": "RealSense SR300 640x480",
    "02": "IDS Imaging 1280x1024",
    "03": "Aptina Imaging 752x480",
}


def image_paths(data_root: Path) -> list[Path]:
    return sorted(
        p
        for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


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


def validate_annotation_label_mapping(annotation_path: Path) -> tuple[bool, str]:
    mapping = annotation_eye_state_mapping(annotation_path)
    if not mapping:
        return False, "annotation.txt eye state mapping was not found"

    expected = {"0": "closed", "1": "open"}
    normalized = {key: value.split()[0] for key, value in mapping.items()}
    normalized = {
        key: ("closed" if value == "close" else value)
        for key, value in normalized.items()
    }
    if normalized != expected:
        return (
            False,
            f"annotation.txt contradicts requested label mapping: {mapping}",
        )
    return True, "annotation.txt confirms 0=closed and 1=open"


def parse_filename(path: Path, data_root: Path) -> dict[str, Any]:
    parent_subject = path.parent.name if SUBJECT_RE.match(path.parent.name) else ""
    match = FILENAME_RE.match(path.stem)
    if not match:
        return {
            "parse_ok": False,
            "subject_id": parent_subject,
            "error": "filename does not match expected MRL format",
        }

    parsed = match.groupdict()
    errors: list[str] = []
    if parent_subject and parent_subject != parsed["subject_id"]:
        errors.append(
            f"subject folder {parent_subject} conflicts with filename {parsed['subject_id']}"
        )

    sensor_id = parsed["sensor_id"]
    if sensor_id not in SENSOR_MAP:
        errors.append(f"unknown sensor_id {sensor_id}")

    return {
        "parse_ok": not errors,
        "subject_id": parsed["subject_id"],
        "image_id": parsed["image_id"],
        "gender": GENDER_MAP.get(parsed["gender"], parsed["gender"]),
        "glasses": GLASSES_MAP.get(parsed["glasses"], parsed["glasses"]),
        "eye_state": int(parsed["eye_state"]),
        "label": int(parsed["eye_state"]),
        "label_name": EYE_STATE_MAP[parsed["eye_state"]],
        "reflections": REFLECTIONS_MAP.get(parsed["reflections"], parsed["reflections"]),
        "lighting": LIGHTING_MAP.get(parsed["lighting"], parsed["lighting"]),
        "sensor_id": sensor_id,
        "sensor": SENSOR_MAP.get(sensor_id, "unknown"),
        "error": "; ".join(errors),
    }


def inspect_image(path: Path) -> tuple[bool, int | None, int | None, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return True, width, height, ""
    except (OSError, UnidentifiedImageError) as exc:
        return False, None, None, str(exc)


def scan_dataset(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in image_paths(data_root):
        parsed = parse_filename(path, data_root)
        read_ok, width, height, read_error = inspect_image(path)
        error_parts = [text for text in [parsed.get("error", ""), read_error] if text]
        rows.append(
            {
                "image_path": path.resolve().as_posix(),
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "read_ok": read_ok,
                "width": width,
                "height": height,
                "error": "; ".join(error_parts),
                **parsed,
            }
        )
    return pd.DataFrame(rows)


def count_dict(series: pd.Series) -> dict[str, int]:
    clean = series.dropna().astype(str)
    clean = clean[clean.ne("")]
    return dict(Counter(clean))


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
    df: pd.DataFrame,
    data_root: Path,
    report_out: Path,
    annotation_ok: bool,
    annotation_message: str,
) -> None:
    report_out.parent.mkdir(parents=True, exist_ok=True)

    total_images = len(df)
    parsed_df = df[df.get("parse_ok", False) == True] if not df.empty else df
    subjects = sorted(parsed_df["subject_id"].dropna().astype(str).unique()) if not parsed_df.empty else []
    unreadable = int((df["read_ok"] == False).sum()) if not df.empty else 0
    unparseable = int((df["parse_ok"] == False).sum()) if not df.empty else 0
    class_counts = (
        parsed_df["label_name"].value_counts().rename_axis("label_name").reset_index(name="count")
        if not parsed_df.empty
        else pd.DataFrame(columns=["label_name", "count"])
    )

    per_subject = (
        parsed_df.groupby("subject_id", as_index=False)
        .agg(
            num_images=("filename", "size"),
            num_closed=("label_name", lambda s: int((s == "closed").sum())),
            num_open=("label_name", lambda s: int((s == "open").sum())),
        )
        .sort_values("subject_id")
        if not parsed_df.empty
        else pd.DataFrame(columns=["subject_id", "num_images", "num_closed", "num_open"])
    )

    blocks: list[str] = []
    warnings: list[str] = []
    if not data_root.is_dir():
        blocks.append(f"Dataset root not found: {data_root}")
    if not annotation_ok:
        blocks.append(annotation_message)
    if total_images == 0:
        blocks.append("No image files were found.")
    if total_images and parsed_df.empty:
        blocks.append("No image filenames matched the expected MRL format.")
    if total_images and set(parsed_df.get("label_name", [])) != {"closed", "open"}:
        blocks.append("Parsed labels do not contain both closed and open classes.")
    if unparseable:
        warnings.append(f"{unparseable} filename(s) were unparseable or invalid.")
    if unreadable:
        warnings.append(f"{unreadable} image(s) could not be opened by Pillow.")

    verdict = "READY for manifest generation" if not blocks else "BLOCKED"

    lines = [
        "# MRL Eye Dataset Inspection Report",
        "",
        f"- Dataset root: `{data_root}`",
        f"- Annotation check: {annotation_message}",
        f"- Final verdict: **{verdict}**",
        "",
        "## Summary",
        "",
        f"- Total images: {total_images}",
        f"- Total subjects: {len(subjects)}",
        f"- Unreadable images: {unreadable}",
        f"- Unparseable filenames: {unparseable}",
        f"- Image extensions: {count_dict(df['extension']) if not df.empty else {}}",
        "",
        "## Class Distribution",
        "",
        markdown_table(class_counts),
        "",
        "## Metadata Distributions",
        "",
        f"- Gender: {count_dict(parsed_df['gender']) if not parsed_df.empty else {}}",
        f"- Glasses: {count_dict(parsed_df['glasses']) if not parsed_df.empty else {}}",
        f"- Lighting: {count_dict(parsed_df['lighting']) if not parsed_df.empty else {}}",
        f"- Reflections: {count_dict(parsed_df['reflections']) if not parsed_df.empty else {}}",
        f"- Sensor ID: {count_dict(parsed_df['sensor_id']) if not parsed_df.empty else {}}",
        "",
        "## Per-Subject Counts",
        "",
        markdown_table(per_subject),
        "",
        "## Warnings",
        "",
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "_None._",
        "",
        "## Blockers",
        "",
        "\n".join(f"- {block}" for block in blocks) if blocks else "_None._",
        "",
    ]
    report_out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = args.data_root / "annotation.txt"
    annotation_ok, annotation_message = validate_annotation_label_mapping(annotation_path)
    if annotation_path.is_file() and not annotation_ok and "contradicts" in annotation_message:
        raise SystemExit(annotation_message)

    df = scan_dataset(args.data_root) if args.data_root.is_dir() else pd.DataFrame()
    write_report(df, args.data_root, args.report_out, annotation_ok, annotation_message)

    print(f"Total images: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique() if not df.empty and 'subject_id' in df else 0}")
    print(f"Unreadable images: {int((df['read_ok'] == False).sum()) if not df.empty else 0}")
    print(f"Unparseable filenames: {int((df['parse_ok'] == False).sum()) if not df.empty else 0}")
    print(f"Wrote report: {args.report_out}")


if __name__ == "__main__":
    main()
