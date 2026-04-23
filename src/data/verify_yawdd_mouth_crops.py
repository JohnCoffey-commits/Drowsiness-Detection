"""
Stage 5 - Verify the mouth-crop manifest produced by
`src/preprocessing/generate_yawdd_mouth_crops.py`.

This script:
  1. Loads `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`.
  2. Aggregates per-method counts, per-subject counts, and class distribution.
  3. Confirms that every non-failed crop exists on disk and is readable.
  4. Samples a small number of crops (balanced by class & method) and saves
     visual QC side-by-side images to
     `artifacts/visual_checks/mouth_crops/`.
  5. Writes `reports/yawdd_dash_mouth_crop_report.md`.

Usage:
    .venv/bin/python src/data/verify_yawdd_mouth_crops.py
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "mappings" / "yawdd_dash_all_mouth_crops.csv"
)
DEFAULT_QC_DIR = (
    PROJECT_ROOT / "artifacts" / "visual_checks" / "mouth_crops"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "yawdd_dash_mouth_crop_report.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--qc-per-bucket",
        type=int,
        default=4,
        help="How many QC examples to render per (method, class) bucket.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_bbox(text: str) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    try:
        parts = [int(p) for p in text.split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    return tuple(parts)  # type: ignore[return-value]


def load_rows(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def render_qc(
    row: dict, out_path: Path, target_h: int = 360
) -> Optional[Path]:
    """Render a side-by-side image:  [full frame with bbox overlay] | [crop]."""
    full_path = Path(row["image_path"])
    crop_path = Path(row["mouth_crop_path"]) if row["mouth_crop_path"] else None
    if not full_path.is_file() or crop_path is None or not crop_path.is_file():
        return None

    full = cv2.imread(str(full_path))
    crop = cv2.imread(str(crop_path))
    if full is None or crop is None:
        return None

    bbox = parse_bbox(row["mouth_bbox_xyxy"])
    full_annot = full.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if row["crop_method"] == "face_mesh" else (0, 165, 255)
        cv2.rectangle(full_annot, (x1, y1), (x2, y2), color, 2)
        label = "{m} | {lbl}".format(m=row["crop_method"], lbl=row["binary_label"])
        cv2.putText(
            full_annot,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Resize both panels to a common height.
    def resize_h(img: np.ndarray, h: int) -> np.ndarray:
        ratio = h / img.shape[0]
        new_w = max(1, int(round(img.shape[1] * ratio)))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    left = resize_h(full_annot, target_h)
    right = resize_h(crop, target_h)
    gap = np.full((target_h, 16, 3), 0, dtype=np.uint8)
    panel = np.concatenate([left, gap, right], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    return out_path


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.manifest.is_file():
        print(f"[error] manifest not found: {args.manifest}")
        return 2

    rows = load_rows(args.manifest)
    total = len(rows)

    method_counts: Counter = Counter()
    class_by_method: Dict[str, Counter] = defaultdict(Counter)
    per_subject: Dict[str, Counter] = defaultdict(Counter)
    missing_files: List[dict] = []
    unreadable_files: List[dict] = []
    bucketed: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    for row in rows:
        method = row["crop_method"]
        method_counts[method] += 1
        class_by_method[method][row["binary_label"]] += 1
        per_subject[row["subject_id"]][method] += 1

        if method in {"face_mesh", "fallback_lower_face", "resumed"}:
            crop_path = Path(row["mouth_crop_path"]) if row["mouth_crop_path"] else None
            if crop_path is None or not crop_path.is_file():
                missing_files.append(row)
            bucketed[(method, row["binary_label"])].append(row)

    # Randomly verify readability on a small sample so we don't read 60k files.
    readable_sample_size = min(500, max(0, total - len(missing_files)))
    readable_candidates = [
        r
        for r in rows
        if r["crop_method"] in {"face_mesh", "fallback_lower_face", "resumed"}
        and r["mouth_crop_path"]
        and Path(r["mouth_crop_path"]).is_file()
    ]
    rng.shuffle(readable_candidates)
    for row in readable_candidates[:readable_sample_size]:
        img = cv2.imread(row["mouth_crop_path"])
        if img is None or img.size == 0:
            unreadable_files.append(row)

    # ---------- QC images ----------
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    qc_written: List[Path] = []
    for (method, label), items in sorted(bucketed.items()):
        rng.shuffle(items)
        for j, item in enumerate(items[: args.qc_per_bucket]):
            out = (
                args.qc_dir
                / f"{method}__{label}__{item['subject_id']}__{item['frame_index']}.jpg"
            )
            rendered = render_qc(item, out)
            if rendered is not None:
                qc_written.append(rendered)

    # ---------- Report ----------
    total_saved = (
        method_counts.get("face_mesh", 0)
        + method_counts.get("fallback_lower_face", 0)
        + method_counts.get("resumed", 0)
    )
    success_rate = (total_saved / total * 100.0) if total else 0.0

    class_totals = Counter()
    for m, c in class_by_method.items():
        if m in {"face_mesh", "fallback_lower_face", "resumed"}:
            class_totals.update(c)

    # A handful of random successful crop paths for manual spot-check.
    success_rows = [
        r
        for r in rows
        if r["crop_method"] in {"face_mesh", "fallback_lower_face"}
        and r["mouth_crop_path"]
    ]
    rng.shuffle(success_rows)
    sample_rows = success_rows[:8]

    lines: List[str] = []
    lines.append("# YawDD+ Dash - Stage 5 Mouth-Crop Report")
    lines.append("")
    lines.append(f"- Merged manifest: `{args.manifest.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Crop root: `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/`")
    lines.append(f"- QC samples: `{args.qc_dir.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    lines.append("## Processing statistics")
    lines.append("")
    lines.append(f"- Total frames processed: **{total}**")
    lines.append(
        f"- MediaPipe Face Mesh crops (`face_mesh`): "
        f"**{method_counts.get('face_mesh', 0)}**"
    )
    lines.append(
        f"- Fallback lower-face crops (`fallback_lower_face`): "
        f"**{method_counts.get('fallback_lower_face', 0)}**"
    )
    lines.append(
        f"- Resumed from a prior run (`resumed`): "
        f"**{method_counts.get('resumed', 0)}**"
    )
    lines.append(
        f"- Failed (no crop saved, `failed`): "
        f"**{method_counts.get('failed', 0)}**"
    )
    lines.append(f"- Success rate: **{success_rate:.2f}%**")
    lines.append("")

    lines.append("## Class distribution across saved crops")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("|---|---|")
    for cls in sorted(class_totals):
        lines.append(f"| `{cls}` | {class_totals[cls]} |")
    lines.append("")

    lines.append("## Per-method class breakdown")
    lines.append("")
    lines.append("| Method | no_yawn | yawn |")
    lines.append("|---|---|---|")
    for m in ("face_mesh", "fallback_lower_face", "resumed", "failed"):
        c = class_by_method.get(m, Counter())
        lines.append(f"| `{m}` | {c.get('no_yawn', 0)} | {c.get('yawn', 0)} |")
    lines.append("")

    lines.append("## Per-subject method counts")
    lines.append("")
    lines.append("| subject_id | face_mesh | fallback | resumed | failed |")
    lines.append("|---|---|---|---|---|")
    for subj in sorted(per_subject):
        c = per_subject[subj]
        lines.append(
            f"| `{subj}` | {c.get('face_mesh', 0)} | "
            f"{c.get('fallback_lower_face', 0)} | "
            f"{c.get('resumed', 0)} | "
            f"{c.get('failed', 0)} |"
        )
    lines.append("")

    lines.append("## File-integrity checks")
    lines.append("")
    lines.append(f"- Crop rows whose image file is MISSING on disk: **{len(missing_files)}**")
    lines.append(
        "- Readability sample size: "
        f"{min(readable_sample_size, len(readable_candidates))} "
        f"(of {len(readable_candidates)} saved crops)"
    )
    lines.append(f"- Unreadable JPEGs in the sample: **{len(unreadable_files)}**")
    if missing_files:
        lines.append("")
        lines.append("Example missing-file rows:")
        for r in missing_files[:5]:
            lines.append(
                f"  - `{r['subject_id']}` / `{r['frame_index']}` -> "
                f"`{r['mouth_crop_path']}` ({r['crop_method']})"
            )
    lines.append("")

    lines.append("## Random example crops (manual inspection)")
    lines.append("")
    for r in sample_rows:
        rel = Path(r["mouth_crop_path"])
        try:
            rel = rel.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
        lines.append(
            f"- `{r['subject_id']}` / `{r['frame_index']}` / `{r['binary_label']}` "
            f"/ `{r['crop_method']}` -> `{rel}`"
        )
    lines.append("")

    lines.append("## Visual QC samples")
    lines.append("")
    lines.append(
        f"A side-by-side (full-frame with bbox | crop) image was rendered for up "
        f"to {args.qc_per_bucket} examples per (method, class) bucket."
    )
    lines.append(f"Total QC panels written: **{len(qc_written)}**.")
    lines.append(f"Location: `{args.qc_dir.relative_to(PROJECT_ROOT)}/`.")
    lines.append("")
    for p in qc_written[:8]:
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        lines.append(f"- `{rel}`")
    lines.append("")

    lines.append("## Readiness for subject-level splitting")
    lines.append("")
    lines.append(
        "The Stage 5 output is ready for subject-level train/val/test splitting "
        "when ALL of the following hold:"
    )
    lines.append("")
    lines.append("1. Success rate >= 95%.")
    lines.append("2. No subject has 0 usable crops (`face_mesh + fallback > 0`).")
    lines.append("3. Both classes (`no_yawn` and `yawn`) are non-empty.")
    lines.append("4. No missing/unreadable files in the file-integrity checks.")
    lines.append("")

    subjects_with_no_crops = [
        s
        for s, c in per_subject.items()
        if (c.get("face_mesh", 0) + c.get("fallback_lower_face", 0) + c.get("resumed", 0)) == 0
    ]
    both_classes_present = class_totals.get("no_yawn", 0) > 0 and class_totals.get("yawn", 0) > 0

    ready = (
        success_rate >= 95.0
        and not subjects_with_no_crops
        and both_classes_present
        and not missing_files
        and not unreadable_files
    )

    lines.append("### Verdict")
    lines.append("")
    if ready:
        lines.append(
            "**READY** - the mouth-crop dataset passes all readiness checks and "
            "can be used as input to subject-level splitting (Stage 6)."
        )
    else:
        lines.append("**NOT READY** - the following checks did not pass:")
        if success_rate < 95.0:
            lines.append(f"- Success rate is {success_rate:.2f}% (< 95%).")
        if subjects_with_no_crops:
            lines.append(
                "- Subjects with zero usable crops: "
                + ", ".join(f"`{s}`" for s in subjects_with_no_crops)
            )
        if not both_classes_present:
            lines.append("- One of the binary classes is empty after cropping.")
        if missing_files:
            lines.append(f"- {len(missing_files)} manifest rows point to missing crop files.")
        if unreadable_files:
            lines.append(f"- {len(unreadable_files)} sampled crops were unreadable.")
    lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")

    print(f"[done] report: {args.report}")
    print(
        f"[done] total={total}  face_mesh={method_counts.get('face_mesh', 0)}  "
        f"fallback={method_counts.get('fallback_lower_face', 0)}  "
        f"failed={method_counts.get('failed', 0)}  "
        f"success_rate={success_rate:.2f}%"
    )
    return 0 if not missing_files and not unreadable_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
