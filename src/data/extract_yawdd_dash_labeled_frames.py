"""Stage 4B — labeled-frame extractor for the YawDD+ ↔ raw Dash pipeline.

Responsibilities:

1. For every subject in the Stage-3 mapping CSV, read every YawDD+ `.txt` in
   `labels/`.
2. Group annotation files by frame index. Prefer the `_0` file; if a `_1`
   sibling exists, record `had_duplicate_box = True` but **ignore** the
   `_1` file for labelling (verified in Stage 4A as a spurious duplicate).
3. Open the matching raw `.avi` once and iterate it sequentially, saving
   only frames whose index is in the target set. Sequential iteration is
   orders of magnitude faster and more reliable on AVI than random seek
   (`CAP_PROP_POS_FRAMES`) which is known to mis-behave on some builds.
4. Write the decoded frame to
     `dataset/YawDD_plus_reconstructed/Dash/full_frames/<subject>/<frame_index>.jpg`
5. Append a row to `dataset/YawDD_plus_reconstructed/Dash/labels_csv/<subject>.csv`
   with full provenance (subject, frame, class id, bbox, etc.).

The script is idempotent: rerunning with the same flags will skip frames
whose output JPEG already exists unless `--force` is given.

Run:
    .venv/bin/python src/data/extract_yawdd_dash_labeled_frames.py
Flags:
    --subjects S1,S2    only process these YawDD+ folder names
    --limit N           stop after N frames per subject (for quick trials)
    --force             overwrite existing JPEGs
    --jpeg-quality Q    override default 90
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = PROJECT_ROOT / "artifacts" / "mappings" / "yawdd_dash_mapping.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "dataset" / "YawDD_plus_reconstructed" / "Dash"

FRAME_NAME_RE = re.compile(r"^(\d{8})_(\d+)\.txt$")

BINARY_LABEL = {0: "no_yawn", 1: "yawn"}

CSV_COLUMNS = [
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


# ----------------------------------------------------------------------------
# annotation loading
# ----------------------------------------------------------------------------

@dataclass
class FrameTarget:
    frame_index: int
    kept_obj_id: int              # the object index whose class we keep (always 0 in practice)
    annotation_txt_path: Path     # the .txt used for labelling
    class_id: int
    bbox_raw: str                 # the 5-token YOLO line, verbatim
    had_duplicate_box: bool       # True if a `_1` sibling exists


def parse_yolo_line(path: Path) -> Tuple[int, str]:
    """Return (class_id, first_line_stripped). Raises on malformed files."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"empty annotation file: {path}")
    first = text.splitlines()[0].strip()
    tokens = first.split()
    if len(tokens) != 5:
        raise ValueError(f"expected 5 tokens, got {len(tokens)}: {first!r} at {path}")
    return int(tokens[0]), first


def collect_targets(labels_dir: Path) -> Tuple[List[FrameTarget], int]:
    """Scan a subject's labels/ dir; return one FrameTarget per frame index.

    Returns (targets, num_duplicate_1_files_ignored).
    """
    by_idx: Dict[int, Dict[int, Path]] = {}
    for p in labels_dir.iterdir():
        if not (p.is_file() and p.suffix == ".txt"):
            continue
        m = FRAME_NAME_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        obj = int(m.group(2))
        by_idx.setdefault(idx, {})[obj] = p

    targets: List[FrameTarget] = []
    duplicates_ignored = 0
    for idx in sorted(by_idx):
        by_obj = by_idx[idx]
        had_dup = len(by_obj) > 1
        if had_dup:
            # Per Stage-4A verification: `_1` files are spurious duplicates.
            duplicates_ignored += sum(1 for k in by_obj if k != 0)
        # We deterministically prefer _0; if _0 is missing (shouldn't happen in
        # this corpus) we fall back to the smallest obj id we do have, and
        # record a note in the per-row CSV later.
        if 0 in by_obj:
            chosen_obj = 0
        else:
            chosen_obj = min(by_obj)
        ann_path = by_obj[chosen_obj]
        try:
            cls, raw_line = parse_yolo_line(ann_path)
        except Exception as e:
            # Defer the error: we still want a row in the CSV flagging the
            # problem. Use sentinel class_id = -1.
            targets.append(FrameTarget(
                frame_index=idx,
                kept_obj_id=chosen_obj,
                annotation_txt_path=ann_path,
                class_id=-1,
                bbox_raw=f"parse_error: {e}",
                had_duplicate_box=had_dup,
            ))
            continue
        targets.append(FrameTarget(
            frame_index=idx,
            kept_obj_id=chosen_obj,
            annotation_txt_path=ann_path,
            class_id=cls,
            bbox_raw=raw_line,
            had_duplicate_box=had_dup,
        ))
    return targets, duplicates_ignored


# ----------------------------------------------------------------------------
# video extraction
# ----------------------------------------------------------------------------

def extract_subject(
    subject_name: str,
    raw_video: Path,
    targets: List[FrameTarget],
    out_root: Path,
    force: bool,
    jpeg_quality: int,
    limit: Optional[int],
) -> List[Dict[str, str]]:
    """Decode every target frame from `raw_video` and write + return CSV rows."""
    images_dir = out_root / "full_frames" / subject_name
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    if limit is not None:
        targets = targets[:limit]

    target_set: Dict[int, FrameTarget] = {t.frame_index: t for t in targets}
    remaining_indices = set(target_set.keys())

    cap = cv2.VideoCapture(str(raw_video))
    if not cap.isOpened():
        # Everything fails — emit rows for completeness.
        for t in targets:
            rows.append(_failed_row(subject_name, t, raw_video, "",
                                    "video_open_failed"))
        return rows

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    max_target = max(target_set) if target_set else -1
    if total_frames >= 0 and max_target >= total_frames:
        # The Stage-4A audit proved this shouldn't happen, but guard anyway.
        print(f"  [warn] {subject_name}: max target {max_target} >= "
              f"reported frame count {total_frames}")

    # Sequential iteration is reliable on AVI. We grab every frame and only
    # decode+save when the current index is a target.
    current_idx = 0
    last_target_idx = max_target
    while remaining_indices:
        # If we've already walked past every target, stop.
        if current_idx > last_target_idx:
            break
        if current_idx in target_set:
            # Use read() (grab + retrieve) on target frames.
            ok, frame = cap.read()
            t = target_set[current_idx]
            if not ok or frame is None:
                rows.append(_failed_row(
                    subject_name, t, raw_video, "", "decode_failed"))
                remaining_indices.discard(current_idx)
                current_idx += 1
                continue

            out_path = images_dir / f"{current_idx:08d}.jpg"
            wrote = False
            if out_path.exists() and not force:
                wrote = True  # treat as already-extracted success
                status = "skipped_existing"
                notes = ""
            else:
                ok_write = cv2.imwrite(
                    str(out_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
                )
                if ok_write:
                    wrote = True
                    status = "extracted"
                    notes = ""
                else:
                    status = "imwrite_failed"
                    notes = f"cv2.imwrite returned False for {out_path}"

            rows.append(_ok_row(
                subject=subject_name,
                t=t,
                raw_video=raw_video,
                out_path=out_path if wrote else None,
                status=status,
                notes=notes,
            ))
            remaining_indices.discard(current_idx)
        else:
            # Skip without decoding: grab() advances the position much faster.
            if not cap.grab():
                break
        current_idx += 1

    cap.release()

    # Anything left in remaining_indices was beyond EOF or unreachable.
    for idx in sorted(remaining_indices):
        t = target_set[idx]
        rows.append(_failed_row(
            subject_name, t, raw_video, "",
            "unreachable_index_past_eof",
        ))

    return rows


def _ok_row(subject: str, t: FrameTarget, raw_video: Path,
            out_path: Optional[Path], status: str, notes: str) -> Dict[str, str]:
    binary = BINARY_LABEL.get(t.class_id, "")
    return {
        "subject_id": subject,
        "frame_index": f"{t.frame_index:08d}",
        "image_path": out_path.as_posix() if out_path else "",
        "raw_video_path": raw_video.as_posix(),
        "annotation_txt_path": t.annotation_txt_path.as_posix(),
        "class_id": str(t.class_id),
        "binary_label": binary,
        "kept_object_id": str(t.kept_obj_id),
        "had_duplicate_box": "true" if t.had_duplicate_box else "false",
        "yawdd_bbox_raw": t.bbox_raw,
        "extraction_status": status,
        "notes": notes,
    }


def _failed_row(subject: str, t: FrameTarget, raw_video: Path,
                out_path: str, status: str) -> Dict[str, str]:
    return {
        "subject_id": subject,
        "frame_index": f"{t.frame_index:08d}",
        "image_path": out_path,
        "raw_video_path": raw_video.as_posix(),
        "annotation_txt_path": t.annotation_txt_path.as_posix(),
        "class_id": str(t.class_id),
        "binary_label": BINARY_LABEL.get(t.class_id, ""),
        "kept_object_id": str(t.kept_obj_id),
        "had_duplicate_box": "true" if t.had_duplicate_box else "false",
        "yawdd_bbox_raw": t.bbox_raw,
        "extraction_status": status,
        "notes": "",
    }


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def read_mapping(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_subject_csv(rows: List[Dict[str, str]], out_root: Path, subject: str) -> Path:
    csv_dir = out_root / "labels_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"{subject}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--subjects", type=str, default="",
                        help="comma-separated YawDD+ folder names to limit to")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N frames per subject (test only)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing JPEGs")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    mapping = read_mapping(args.mapping)
    wanted = {s.strip() for s in args.subjects.split(",") if s.strip()}
    if wanted:
        mapping = [r for r in mapping
                   if Path(r["annotation_folder"]).name in wanted]

    if not mapping:
        raise SystemExit("no subjects to process (check --subjects / mapping CSV)")

    (args.out_root / "full_frames").mkdir(parents=True, exist_ok=True)
    (args.out_root / "labels_csv").mkdir(parents=True, exist_ok=True)

    total_targets = 0
    total_extracted = 0
    total_skipped = 0
    total_failed = 0
    total_duplicates_ignored = 0
    t0 = time.time()

    for row in mapping:
        subject = Path(row["annotation_folder"]).name
        raw_video = Path(row["raw_source_path"])
        ann_dir = Path(row["annotation_folder"]) / "labels"
        if not raw_video.is_file():
            print(f"[skip] {subject}: raw video missing at {raw_video}")
            continue
        if not ann_dir.is_dir():
            print(f"[skip] {subject}: annotation dir missing at {ann_dir}")
            continue

        targets, duplicates_ignored = collect_targets(ann_dir)
        total_duplicates_ignored += duplicates_ignored
        total_targets += len(targets)

        print(f"[extract] {subject}: {len(targets)} frames "
              f"({duplicates_ignored} `_1` duplicates ignored)")

        rows = extract_subject(
            subject_name=subject,
            raw_video=raw_video,
            targets=targets,
            out_root=args.out_root,
            force=args.force,
            jpeg_quality=args.jpeg_quality,
            limit=args.limit,
        )

        status_counts = {}
        for r in rows:
            status_counts[r["extraction_status"]] = status_counts.get(r["extraction_status"], 0) + 1
        total_extracted += status_counts.get("extracted", 0)
        total_skipped += status_counts.get("skipped_existing", 0)
        total_failed += sum(v for k, v in status_counts.items()
                            if k not in ("extracted", "skipped_existing"))

        csv_path = write_subject_csv(rows, args.out_root, subject)
        print(f"            wrote {csv_path.as_posix()} ({status_counts})")

    dt = time.time() - t0
    print("")
    print(f"[extract] done in {dt:.1f}s")
    print(f"[extract] target frames:            {total_targets}")
    print(f"[extract] newly extracted JPEGs:    {total_extracted}")
    print(f"[extract] skipped (already on disk):{total_skipped}")
    print(f"[extract] failed rows:              {total_failed}")
    print(f"[extract] `_1` duplicates ignored:  {total_duplicates_ignored}")


if __name__ == "__main__":
    main()
