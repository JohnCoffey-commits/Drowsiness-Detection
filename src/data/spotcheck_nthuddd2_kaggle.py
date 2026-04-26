"""Build visual spot-check contact sheets for NTHUDDD2 Kaggle JPG frames."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "mappings"
    / "nthuddd2_kaggle_all_images_trainable_with_split.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "visual_checks"
DEFAULT_SAMPLE_PER_CLASS = 24
SEED = 42

CELL_W = 190
IMAGE_H = 140
CAPTION_H = 58
GAP = 8
MARGIN = 12
COLS = 6


def get_font(size: int) -> ImageFont.ImageFont:
    for name in ("Helvetica.ttc", "Arial Unicode.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (width, height), (22, 24, 28))
            x = (width - image.width) // 2
            y = (height - image.height) // 2
            cell.paste(image, (x, y))
            return cell
    except (OSError, UnidentifiedImageError):
        cell = Image.new("RGB", (width, height), (90, 20, 20))
        draw = ImageDraw.Draw(cell)
        draw.text((8, 8), "unreadable", fill=(255, 230, 230), font=get_font(12))
        return cell


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_height: int,
) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def build_contact_sheet(rows: pd.DataFrame, label: str, out_path: Path) -> None:
    rows = rows.reset_index(drop=True)
    cols = min(COLS, max(1, len(rows)))
    rows_count = max(1, (len(rows) + cols - 1) // cols)
    width = MARGIN * 2 + cols * CELL_W + (cols - 1) * GAP
    height = MARGIN * 2 + 34 + rows_count * (IMAGE_H + CAPTION_H) + (rows_count - 1) * GAP

    sheet = Image.new("RGB", (width, height), (245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(17)
    caption_font = get_font(10)
    draw.text(
        (MARGIN, MARGIN),
        f"NTHUDDD2 Kaggle spot-check: {label}",
        fill=(20, 24, 30),
        font=title_font,
    )

    y0 = MARGIN + 34
    for idx, row in rows.iterrows():
        grid_row = idx // cols
        grid_col = idx % cols
        x = MARGIN + grid_col * (CELL_W + GAP)
        y = y0 + grid_row * (IMAGE_H + CAPTION_H + GAP)

        image = fit_image(Path(row["image_path"]), CELL_W, IMAGE_H)
        sheet.paste(image, (x, y))

        caption_lines = [
            truncate_middle(str(row["filename"]), 32),
            f"label={row['label']}  subject={row['subject_id']}",
            f"split={row['split']}",
        ]
        draw_wrapped_text(
            draw,
            (x, y + IMAGE_H + 4),
            caption_lines,
            caption_font,
            (45, 50, 58),
            15,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def sample_label(
    df: pd.DataFrame, label: str, sample_count: int, rng: random.Random
) -> pd.DataFrame:
    pool = df[df["label"] == label].copy()
    if pool.empty:
        raise SystemExit(f"No rows found for label={label!r}")
    n = min(sample_count, len(pool))
    indices = rng.sample(pool.index.tolist(), n)
    return pool.loc[indices].sort_values(["split", "subject_id", "filename"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-per-class", type=int, default=DEFAULT_SAMPLE_PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input manifest not found: {args.input}")

    df = pd.read_csv(args.input, dtype={"subject_id": str})
    required = {"image_path", "filename", "label", "subject_id", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input manifest is missing required column(s): {missing}")

    rng = random.Random(args.seed)
    outputs = {
        "drowsy": args.out_dir / "nthuddd2_kaggle_drowsy_contact_sheet.jpg",
        "notdrowsy": args.out_dir / "nthuddd2_kaggle_notdrowsy_contact_sheet.jpg",
    }
    for label, out_path in outputs.items():
        sampled = sample_label(df, label, args.sample_per_class, rng)
        build_contact_sheet(sampled, label, out_path)
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
