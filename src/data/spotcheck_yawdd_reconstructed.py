"""Spot-check tool for the reconstructed YawDD+ Dash (Stage 4B) dataset.

For a small set of subjects, randomly samples:
  * N frames with class_id = 0
  * N frames with class_id = 1
from each subject's `labels_csv/<subject>.csv`, copies the corresponding
JPEGs into `artifacts/visual_checks/spotcheck/<subject>/`, builds a
2-row contact sheet per subject, and writes a merged CSV + markdown summary
so you can quickly verify that labels match the images.

No model training. Only filesystem + Pillow.

Example:
    .venv/bin/python src/data/spotcheck_yawdd_reconstructed.py

With custom subjects and seed:
    .venv/bin/python src/data/spotcheck_yawdd_reconstructed.py \\
        --subjects 1-FemaleNoGlasses,8-MaleNoGlasses,13-MaleNoGlasses --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise SystemExit(
        "Pillow is required: pip install pillow  "
        f"(or: pip install -r requirements.txt). Original error: {e}"
    ) from e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_DIR = (
    PROJECT_ROOT / "dataset" / "YawDD_plus_reconstructed" / "Dash" / "labels_csv"
)
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "visual_checks" / "spotcheck"
DEFAULT_SUBJECTS = "1-FemaleNoGlasses,8-MaleNoGlasses,13-MaleNoGlasses"
SAMPLE_PER_CLASS = 5
CELL_W, CELL_H = 220, 165
PADDING = 4
TITLE_BAR = 32


@dataclass
class Row:
    subject_id: str
    frame_index: str
    image_path: str
    class_id: int
    binary_label: str


def load_subject_rows(labels_csv: Path) -> List[Row]:
    rows: List[Row] = []
    with labels_csv.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("extraction_status", "") not in ("extracted", "skipped_existing"):
                continue
            try:
                cid = int(r["class_id"].strip())
            except (KeyError, ValueError):
                continue
            rows.append(
                Row(
                    subject_id=r["subject_id"].strip(),
                    frame_index=r["frame_index"].strip(),
                    image_path=r["image_path"].strip(),
                    class_id=cid,
                    binary_label=r.get("binary_label", "").strip(),
                )
            )
    return rows


def sample_class(rows: List[Row], class_id: int, k: int, rng: random.Random) -> List[Row]:
    pool = [r for r in rows if r.class_id == class_id]
    n = min(k, len(pool))
    if n == 0:
        return []
    return rng.sample(pool, n)


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    for name in (
        "Helvetica.ttc",
        "Arial Unicode.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact_sheet(
    c0: List[Row],
    c1: List[Row],
    out_path: Path,
    subject: str,
    n_per_class: int,
) -> None:
    """2 rows: first row = class 0, second row = class 1."""
    font = _get_font(13)
    font_small = _get_font(11)

    n0, n1 = min(n_per_class, len(c0)), min(n_per_class, len(c1))
    cols = max(n0, n1, 1)
    W = PADDING * 2 + cols * (CELL_W + PADDING)
    H = TITLE_BAR + PADDING * 3 + 2 * (CELL_H + 24 + 22) + PADDING
    sheet = Image.new("RGB", (W, H), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    draw.text((PADDING, 6), f"{subject}  —  random spot-check  (row1: class 0, row2: class 1)", fill=(255, 255, 255), font=font)

    def place_row(
        y_base: int, label: str, color: Tuple[int, int, int], samples: List[Row]
    ) -> None:
        draw.text((PADDING, y_base), label, fill=color, font=font_small)
        y0 = y_base + 20
        for j, r in enumerate(samples):
            x0 = PADDING + j * (CELL_W + PADDING)
            p = Path(r.image_path)
            if not p.is_file():
                cell = Image.new("RGB", (CELL_W, CELL_H), (64, 0, 0))
                d2 = ImageDraw.Draw(cell)
                d2.text((8, 8), "MISSING", fill=(255, 200, 200), font=font_small)
            else:
                im = Image.open(p).convert("RGB")
                im.thumbnail((CELL_W, CELL_H), Image.Resampling.LANCZOS)
                cell = Image.new("RGB", (CELL_W, CELL_H), (16, 16, 16))
                ox = (CELL_W - im.size[0]) // 2
                oy = (CELL_H - im.size[1]) // 2
                cell.paste(im, (ox, oy))
            sheet.paste(cell, (x0, y0))
            cap = f"f={r.frame_index}  {r.binary_label}"
            draw.text((x0, y0 + CELL_H + 2), cap[:40], fill=(200, 200, 200), font=font_small)

    y1 = TITLE_BAR + 8
    place_row(
        y1, "class_id = 0  (no_yawn)", (100, 220, 100), c0[:n_per_class]
    )
    # Leave room for thumbnails + per-cell captions under row 1.
    y2 = y1 + 24 + CELL_H + 22 + PADDING * 2
    place_row(
        y2, "class_id = 1  (yawn)", (255, 120, 120), c1[:n_per_class]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def copy_samples(subject: str, c0: List[Row], c1: List[Row], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for r in c0:
        src = Path(r.image_path)
        if src.is_file():
            dst = dest_dir / f"c0_{r.frame_index}.jpg"
            shutil.copy2(src, dst)
    for r in c1:
        src = Path(r.image_path)
        if src.is_file():
            dst = dest_dir / f"c1_{r.frame_index}.jpg"
            shutil.copy2(src, dst)


def run_spotcheck(
    subjects: List[str],
    labels_dir: Path,
    out_root: Path,
    seed: int,
    n_per_class: int,
) -> List[Row]:
    rng = random.Random(seed)
    all_sampled: List[Row] = []
    for subject in subjects:
        csv_path = labels_dir / f"{subject}.csv"
        if not csv_path.is_file():
            print(f"[warn] missing {csv_path}, skipping")
            continue
        rows = load_subject_rows(csv_path)
        c0 = sample_class(rows, 0, n_per_class, rng)
        c1 = sample_class(rows, 1, n_per_class, rng)
        if len(c0) < n_per_class:
            print(
                f"[warn] {subject}: only {len(c0)} class-0 rows available (wanted {n_per_class})"
            )
        if len(c1) < n_per_class:
            print(
                f"[warn] {subject}: only {len(c1)} class-1 rows available (wanted {n_per_class})"
            )
        sub_dir = out_root / subject
        copy_samples(subject, c0, c1, sub_dir)
        build_contact_sheet(
            c0, c1, sub_dir / f"contact_sheet_{subject}.jpg", subject, n_per_class
        )
        all_sampled.extend(c0)
        all_sampled.extend(c1)
        print(
            f"[spotcheck] {subject}: {len(c0)} class-0, {len(c1)} class-1 -> {sub_dir}"
        )
    return all_sampled


def write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["subject_id", "frame_index", "image_path", "class_id", "binary_label"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "subject_id": r.subject_id,
                    "frame_index": r.frame_index,
                    "image_path": r.image_path,
                    "class_id": r.class_id,
                    "binary_label": r.binary_label,
                }
            )


def _rel_to_project(p: str, project_root: Path) -> str:
    try:
        return Path(p).resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return p


def write_markdown(
    path: Path, rows: List[Row], seed: int, subjects: List[str], project_root: Path
) -> None:
    lines: List[str] = [
        "# YawDD+ Dash reconstructed — spot-check sample",
        "",
        f"Random seed: **{seed}**  •  Subjects: **{', '.join(subjects)}**",
        "",
        "Open each `contact_sheet_<subject>.jpg` in this folder for a 2×5 "
        "visual grid, or browse the per-subject copies: `c0_*.jpg` = class 0, "
        "`c1_*.jpg` = class 1.",
        "",
        "| subject_id | frame_index | class_id | binary_label | image_path (rel.) |",
        "|--------------|-------------|----------|--------------|-------------------|",
    ]
    for r in rows:
        rel = _rel_to_project(r.image_path, project_root)
        lines.append(
            f"| {r.subject_id} | {r.frame_index} | {r.class_id} | {r.binary_label} | `{rel}` |"
        )
    lines.append("")
    lines.append("### Full absolute `image_path` (one per line)")
    lines.append("")
    for r in rows:
        lines.append(f"- `{r.image_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subjects",
        type=str,
        default=DEFAULT_SUBJECTS,
        help="Comma-separated subject folder names (default: 3 diverse subjects)",
    )
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible samples)")
    p.add_argument(
        "-n",
        "--per-class",
        type=int,
        default=SAMPLE_PER_CLASS,
        help="Number of random frames per class per subject (default: 5)",
    )
    args = p.parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    rows = run_spotcheck(
        subjects=subjects,
        labels_dir=args.labels_dir,
        out_root=args.out,
        seed=args.seed,
        n_per_class=args.per_class,
    )
    if not rows:
        raise SystemExit("no samples written (check --subjects and labels CSVs)")

    write_csv(args.out / "spotcheck_samples.csv", rows)
    write_markdown(
        args.out / "spotcheck_report.md", rows, args.seed, subjects, PROJECT_ROOT
    )
    print(f"[spotcheck] merged CSV:  {args.out / 'spotcheck_samples.csv'}")
    print(f"[spotcheck] summary MD:  {args.out / 'spotcheck_report.md'}")


if __name__ == "__main__":
    main()
