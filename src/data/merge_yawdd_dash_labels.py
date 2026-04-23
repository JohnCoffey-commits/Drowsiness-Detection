"""Stage 4B — merge per-subject label CSVs into a single manifest.

Reads every `labels_csv/<subject>.csv` produced by
`extract_yawdd_dash_labeled_frames.py` and concatenates them into
`artifacts/mappings/yawdd_dash_all_labeled_frames.csv`.

The merged CSV is the single source of truth used by Stage 5 (mouth-ROI
generation) and by the verification script.

Run:
    .venv/bin/python src/data/merge_yawdd_dash_labels.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_DIR = (
    PROJECT_ROOT / "dataset" / "YawDD_plus_reconstructed" / "Dash" / "labels_csv"
)
DEFAULT_MERGED_OUT = (
    PROJECT_ROOT / "artifacts" / "mappings" / "yawdd_dash_all_labeled_frames.csv"
)

EXPECTED_COLUMNS = [
    "subject_id",
    "frame_index",
    "image_path",
    "raw_video_path",
    "annotation_txt_path",
    "class_id",
    "binary_label",
    "kept_object_id",
    "had_duplicate_box",
    "yawdd_bbox_raw",
    "extraction_status",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_MERGED_OUT)
    args = parser.parse_args()

    if not args.labels_dir.is_dir():
        raise SystemExit(f"labels dir not found: {args.labels_dir}")

    subject_csvs: List[Path] = sorted(
        p for p in args.labels_dir.iterdir()
        if p.is_file() and p.suffix == ".csv"
    )
    if not subject_csvs:
        raise SystemExit(f"no per-subject CSVs under {args.labels_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with args.out.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        for csv_path in subject_csvs:
            with csv_path.open("r", encoding="utf-8") as in_fh:
                reader = csv.DictReader(in_fh)
                if reader.fieldnames != EXPECTED_COLUMNS:
                    raise SystemExit(
                        f"{csv_path} has unexpected columns: {reader.fieldnames}"
                    )
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1

    print(f"[merge] merged {len(subject_csvs)} per-subject CSVs "
          f"-> {total_rows} rows at {args.out}")


if __name__ == "__main__":
    main()
