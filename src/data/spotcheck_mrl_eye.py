"""Generate MRL Eye visual contact sheets for label sanity checks.

Input:

    artifacts/mappings/mrl_eye_trainable_with_split.csv

Outputs:

    artifacts/visual_checks/mrl_eye_closed_contact_sheet.jpg
    artifacts/visual_checks/mrl_eye_open_contact_sheet.jpg
    artifacts/visual_checks/mrl_eye_by_split_contact_sheet.jpg

Run from the project root:

    python src/data/spotcheck_mrl_eye.py
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "artifacts" / "mappings" / "mrl_eye_trainable_with_split.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "visual_checks"

SEED = 42
SPLITS = ("train", "val", "test")


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sample(n=min(n, len(df)), random_state=seed)


def sample_by_split(df: pd.DataFrame, per_split: int, seed: int) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for idx, split in enumerate(SPLITS):
        split_df = df[df["split"] == split]
        samples.append(sample_rows(split_df, per_split, seed + idx))
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:2]


def load_thumbnail(path: Path, thumb_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", thumb_size, "white")
        x = (thumb_size[0] - image.width) // 2
        y = (thumb_size[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def make_contact_sheet(
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
    columns: int = 6,
    thumb_size: tuple[int, int] = (120, 80),
    label_height: int = 34,
) -> None:
    if rows.empty:
        raise SystemExit(f"No rows available for contact sheet: {title}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    title_height = 28
    padding = 8
    cell_width = thumb_size[0] + padding * 2
    cell_height = thumb_size[1] + label_height + padding * 2
    num_rows = math.ceil(len(rows) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_width, title_height + num_rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((padding, 8), title, fill="black", font=font)

    for idx, row in rows.reset_index(drop=True).iterrows():
        col = idx % columns
        grid_row = idx // columns
        x0 = col * cell_width + padding
        y0 = title_height + grid_row * cell_height + padding
        path = Path(str(row["image_path"]))
        try:
            thumb = load_thumbnail(path, thumb_size)
            sheet.paste(thumb, (x0, y0))
        except (OSError, UnidentifiedImageError) as exc:
            draw.rectangle((x0, y0, x0 + thumb_size[0], y0 + thumb_size[1]), outline="red")
            draw.text((x0 + 4, y0 + 4), f"unreadable: {exc}", fill="red", font=font)

        label = f"{row['subject_id']} | {row['label_name']} | {row['split']}"
        for line_idx, line in enumerate(fit_text(draw, label, thumb_size[0], font)):
            draw.text(
                (x0, y0 + thumb_size[1] + 5 + line_idx * 12),
                line,
                fill="black",
                font=font,
            )

    sheet.save(output_path, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--per-split", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input manifest with split not found: {args.input}")
    df = pd.read_csv(args.input, dtype={"subject_id": str, "sensor_id": str})
    required = {"image_path", "subject_id", "label_name", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")

    rng = random.Random(args.seed)
    output_paths = {
        "closed": args.out_dir / "mrl_eye_closed_contact_sheet.jpg",
        "open": args.out_dir / "mrl_eye_open_contact_sheet.jpg",
        "split": args.out_dir / "mrl_eye_by_split_contact_sheet.jpg",
    }

    closed = sample_rows(df[df["label_name"] == "closed"], args.per_class, args.seed)
    open_rows = sample_rows(df[df["label_name"] == "open"], args.per_class, args.seed + 1)
    by_split = sample_by_split(df, args.per_split, args.seed + 10)

    make_contact_sheet(closed, output_paths["closed"], "MRL Eye closed samples")
    make_contact_sheet(open_rows, output_paths["open"], "MRL Eye open samples")
    make_contact_sheet(by_split, output_paths["split"], "MRL Eye samples by split")

    for output_path in output_paths.values():
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
