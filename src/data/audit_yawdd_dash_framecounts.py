"""Quick cross-dataset frame-count audit.

For every (YawDD+ subject, raw video) pair in the Stage-3 mapping, this script
reads only the raw video's reported frame count (`CAP_PROP_FRAME_COUNT`) and
compares it with the maximum YawDD+ frame index for the same subject. No
frames are actually decoded.

The output is a small CSV at
`artifacts/visual_checks/framecount_audit.csv` and a summary printed to stdout.
It is used by the Stage-4A report to back up the claim that YawDD+ frame
indices are native 0-based indices into the raw `.avi`.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = PROJECT_ROOT / "artifacts" / "mappings" / "yawdd_dash_mapping.csv"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "visual_checks" / "framecount_audit.csv"

FRAME_NAME_RE = re.compile(r"^(\d{8})_\d+\.txt$")


def max_annotation_frame(ann_dir: Path) -> Tuple[int, int]:
    """Return (count_of_txt_files, max_frame_index_seen) for a labels/ dir."""
    n = 0
    mx = -1
    for p in ann_dir.iterdir():
        if not (p.is_file() and p.suffix == ".txt"):
            continue
        m = FRAME_NAME_RE.match(p.name)
        if not m:
            continue
        n += 1
        idx = int(m.group(1))
        if idx > mx:
            mx = idx
    return n, mx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []
    with args.mapping.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit_cols = [
        "subject_id", "annotation_folder_basename", "raw_video_basename",
        "yawdd_plus_file_count", "yawdd_plus_max_frame_index",
        "raw_reported_frame_count", "raw_reported_fps",
        "matches_native_indexing", "mismatch_delta",
    ]
    n_match = 0
    n_total = 0

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_cols)
        w.writeheader()
        for r in rows:
            ann_dir = Path(r["annotation_folder"]) / "labels"
            if not ann_dir.is_dir():
                continue
            fc, mx_idx = max_annotation_frame(ann_dir)
            raw_path = Path(r["raw_source_path"])
            cap = cv2.VideoCapture(str(raw_path))
            reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else -1.0
            cap.release()

            # "native indexing" claim: raw video has `reported` frames indexed 0..reported-1,
            # so max YawDD+ index should be <= reported - 1. A tight match is
            # mx_idx == reported - 1 (i.e. the last raw frame is still annotated).
            matches = reported > 0 and mx_idx == reported - 1
            delta = (reported - 1) - mx_idx if reported > 0 else None

            n_total += 1
            if matches:
                n_match += 1

            w.writerow({
                "subject_id": r["subject_id"],
                "annotation_folder_basename": Path(r["annotation_folder"]).name,
                "raw_video_basename": raw_path.name,
                "yawdd_plus_file_count": fc,
                "yawdd_plus_max_frame_index": mx_idx,
                "raw_reported_frame_count": reported,
                "raw_reported_fps": f"{fps:.4f}",
                "matches_native_indexing": "yes" if matches else "no",
                "mismatch_delta": delta,
            })

    print(f"[audit] {n_match}/{n_total} subjects have max(YawDD+ index) == raw_frame_count - 1")
    print(f"[audit] csv: {args.out}")


if __name__ == "__main__":
    main()
