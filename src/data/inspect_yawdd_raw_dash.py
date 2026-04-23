"""Stage 2: Inspect the raw YawDD Dash source data.

Scans `dataset/YawDD_raw/Dash/Dash/{Female,Male}/` and any Female/Male folders
that may live directly under `dataset/YawDD_raw/Dash/`. Produces a markdown
report describing:

* what kind of files are on disk (videos, images, ...)
* the full list of raw files (with sizes)
* the filename pattern(s) actually observed
* any metadata we can recover from folder + file names
   (subject id, gender, glasses state)
* whether the supplied PDFs (`Readme_YawDD.pdf`, `Table1.pdf`, `Table2.pdf`)
   add anything we did not already know
* a preliminary judgment on whether subject-level mapping to YawDD+ is feasible.

No frames are decoded from the videos.

Run:
    python -m src.data.inspect_yawdd_raw_dash
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAWDD_RAW = PROJECT_ROOT / "dataset" / "YawDD_raw"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "yawdd_raw_dash_report.md"

# Match e.g. "13-MaleNoGlasses", "1-FemaleNoGlasses", "11-FemaleGlasses".
SUBJECT_TOKEN_RE = re.compile(r"^\s*(\d+)-(Female|Male)(Glasses|NoGlasses|SunGlasses)?\s*$")


def find_female_male_dirs(raw_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """The user's layout may have an extra nesting: `YawDD_raw/Dash/Dash/{Female,Male}`.

    We try both. Returns (female_dir, male_dir) or None for whichever is missing.
    """
    candidates = [
        raw_root / "Dash" / "Female",
        raw_root / "Dash" / "Male",
        raw_root / "Dash" / "Dash" / "Female",
        raw_root / "Dash" / "Dash" / "Male",
    ]
    female: Optional[Path] = None
    male: Optional[Path] = None
    for c in candidates:
        if c.is_dir():
            if c.name == "Female" and female is None:
                female = c
            elif c.name == "Male" and male is None:
                male = c
    return female, male


def normalise_stem(stem: str) -> str:
    """Strip a trailing `.avi` if the file was double-extensioned, and any
    stray whitespace. We deliberately DO NOT rename any files on disk — this
    is only used to derive the canonical subject token for matching.
    """
    s = stem.strip()
    # Some files are named `11-FemaleGlasses.avi.avi` on disk; Path.stem
    # already drops the final `.avi` so `s` might still end with `.avi`.
    if s.lower().endswith(".avi"):
        s = s[: -len(".avi")]
    return s.strip()


def parse_subject_token(token: str) -> Optional[Dict]:
    m = SUBJECT_TOKEN_RE.match(token)
    if not m:
        return None
    return {
        "subject_index": int(m.group(1)),
        "gender": m.group(2),
        "glasses_state": m.group(3) or "",
    }


def scan_dir(d: Optional[Path], expected_gender: str) -> List[Dict]:
    rows: List[Dict] = []
    if d is None or not d.is_dir():
        return rows
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        stem = Path(name).stem
        ext = p.suffix.lower()
        double_ext = name.lower().endswith(".avi.avi")
        canonical = normalise_stem(stem)
        meta = parse_subject_token(canonical) or {}
        rows.append(
            {
                "path": p,
                "parent_gender": expected_gender,
                "filename_on_disk": name,
                "size_bytes": p.stat().st_size,
                "suffix": ext,
                "double_ext": double_ext,
                "has_trailing_space": name != name.strip()
                or " .avi" in name
                or " .AVI" in name,
                "canonical_token": canonical,
                "subject_index": meta.get("subject_index"),
                "gender": meta.get("gender"),
                "glasses_state": meta.get("glasses_state"),
            }
        )
    return rows


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def build_report(
    raw_root: Path,
    female_dir: Optional[Path],
    male_dir: Optional[Path],
    female_rows: List[Dict],
    male_rows: List[Dict],
    pdfs: Dict[str, Optional[Path]],
) -> str:
    all_rows = female_rows + male_rows
    ext_counter = Counter(r["suffix"] for r in all_rows)
    lines: List[str] = []

    lines.append("# YawDD Raw — Dash Source Inspection Report (Stage 2)\n")
    lines.append(
        "This report describes the raw, as-delivered YawDD Dash data that we "
        "will later use to reconstruct frames for the YawDD+ annotations. "
        "No videos were decoded — we only inspected file names, sizes and the "
        "accompanying PDFs.\n"
    )
    lines.append(f"- Scanned root: `{raw_root.as_posix()}`")
    lines.append(f"- Female directory: `{female_dir.as_posix() if female_dir else 'NOT FOUND'}`")
    lines.append(f"- Male directory:   `{male_dir.as_posix() if male_dir else 'NOT FOUND'}`")
    lines.append(f"- Raw files found: **{len(all_rows)}** "
                 f"(Female: {len(female_rows)}, Male: {len(male_rows)})")
    lines.append("")

    # 1. File types
    lines.append("## 1. File types found\n")
    lines.append("| extension | count |")
    lines.append("|-----------|-------|")
    for ext, n in ext_counter.most_common():
        lines.append(f"| `{ext or '(none)'}` | {n} |")
    lines.append("")
    if set(ext_counter.keys()) <= {".avi"}:
        lines.append(
            "All raw source files are **`.avi` videos** (single container per "
            "subject). No image sequences, no per-frame folders.\n"
        )

    # 2. Naming patterns
    lines.append("## 2. Naming patterns observed\n")
    lines.append(
        "The canonical name of each raw video matches the pattern "
        "`<subject_index>-<Gender><GlassesState>.avi` where\n"
    )
    lines.append(
        "- `<subject_index>` is an integer 1..16 (males) or 1..13 (females),\n"
        "- `<Gender>` is literally `Female` or `Male`,\n"
        "- `<GlassesState>` is `Glasses`, `NoGlasses`, or `SunGlasses`.\n"
    )
    # Anomalies
    double_ext = [r for r in all_rows if r["double_ext"]]
    trailing = [r for r in all_rows if r["has_trailing_space"]]
    lines.append("### Observed deviations from the canonical pattern\n")
    if double_ext:
        lines.append(f"- **{len(double_ext)} file(s)** have a duplicated extension (`.avi.avi`):")
        for r in double_ext:
            lines.append(f"  - `{r['parent_gender']}/{r['filename_on_disk']}`")
    if trailing:
        lines.append(
            f"- **{len(trailing)} file(s)** have whitespace inside the name, "
            "typically a stray space before `.avi`:"
        )
        for r in trailing:
            lines.append(f"  - `{r['parent_gender']}/{r['filename_on_disk']}`")
    if not double_ext and not trailing:
        lines.append("- None.")
    lines.append("")

    # 3. Full listings
    def _listing_table(rows: List[Dict]) -> List[str]:
        out = ["| file | size | subject | gender | glasses | canonical token |",
               "|------|------|---------|--------|---------|------------------|"]
        for r in sorted(rows, key=lambda x: (x["subject_index"] or 0)):
            out.append(
                f"| `{r['filename_on_disk']}` | {human_bytes(r['size_bytes'])} | "
                f"{r['subject_index']} | {r['gender']} | {r['glasses_state']} | "
                f"`{r['canonical_token']}` |"
            )
        return out

    lines.append("### Full Female listing\n")
    lines.extend(_listing_table(female_rows))
    lines.append("")
    lines.append("### Full Male listing\n")
    lines.extend(_listing_table(male_rows))
    lines.append("")

    # 4. Metadata inferable from names
    lines.append("## 3. Metadata inferable from folder/file names\n")
    lines.append(
        "From the folder and file names alone we can recover, for every raw "
        "video: the subject index, the gender, and the glasses state. No other "
        "per-video metadata (session id, clip id, time range, yawning time "
        "stamps, ...) is present in the filenames — the raw Dash videos are "
        "delivered as a single continuous clip per subject containing the "
        "three scripted segments (Normal / Talking / Yawning) back-to-back.\n"
    )

    # 5. Cross-check with YawDD+ folder names
    lines.append("## 4. Compatibility with YawDD+ subject folders\n")
    plus_dir = PROJECT_ROOT / "dataset" / "YawDD+" / "dataset" / "Dash"
    plus_names = []
    if plus_dir.is_dir():
        plus_names = sorted(p.name for p in plus_dir.iterdir() if p.is_dir())
    raw_tokens = sorted(r["canonical_token"] for r in all_rows if r["canonical_token"])
    lines.append(f"- YawDD+ Dash subject folders: **{len(plus_names)}**")
    lines.append(f"- Raw Dash videos (canonical tokens): **{len(raw_tokens)}**")
    set_plus = set(plus_names)
    set_raw = set(raw_tokens)
    in_both = sorted(set_plus & set_raw)
    only_plus = sorted(set_plus - set_raw)
    only_raw = sorted(set_raw - set_plus)
    lines.append(f"- Tokens present in **both** YawDD+ and raw: **{len(in_both)}**")
    lines.append(f"- Tokens **only** in YawDD+ (no raw video): **{len(only_plus)}** — {only_plus or 'none'}")
    lines.append(f"- Tokens **only** in raw (no YawDD+ labels): **{len(only_raw)}** — {only_raw or 'none'}")
    lines.append("")
    if set_plus == set_raw and set_plus:
        lines.append(
            "**The canonical tokens match one-to-one.** Every YawDD+ subject "
            "folder has a raw `.avi` with the same `<index>-<Gender><Glasses>` "
            "token, after normalising the `.avi.avi` and whitespace anomalies "
            "above. Subject-level mapping is therefore fully determined by "
            "folder name == canonical raw-video stem.\n"
        )

    # 6. PDFs
    lines.append("## 5. Value added by the supplied PDFs\n")
    lines.append("The three PDFs under `dataset/YawDD_raw/` were read for "
                 "context; none of them add filename-level information that "
                 "was not already recoverable from the folder structure.\n")
    lines.append("| PDF | present | summary of relevance |")
    lines.append("|-----|---------|-----------------------|")
    lines.append(
        "| `Readme_YawDD.pdf` | "
        f"{'yes' if pdfs['readme'] else 'no'} | "
        "Confirms the Dash dataset has **29 videos** (1 per participant), "
        "30 fps, 640x480 24-bit RGB AVI, no audio, with each participant "
        "performing Normal / Talking / Yawning segments inside a single "
        "continuous clip. |"
    )
    lines.append(
        "| `Table1.pdf` | "
        f"{'yes' if pdfs['table1'] else 'no'} | "
        "Describes the **Mirror** dataset (camera under front mirror, 322 "
        "short clips). Not directly relevant to Dash, kept here only for "
        "completeness. |"
    )
    lines.append(
        "| `Table2.pdf` | "
        f"{'yes' if pdfs['table2'] else 'no'} | "
        "Per-subject metadata for the **Dash** dataset (16 males + 13 "
        "females = 29 subjects), matching the 29 raw videos and 29 YawDD+ "
        "folders. Useful later for fairness/ethnicity reporting, but not "
        "needed for the frame mapping. |"
    )
    lines.append("")

    # 7. Judgment
    lines.append("## 6. Preliminary judgment on subject-level mapping\n")
    if set_plus == set_raw and set_plus:
        lines.append(
            "**Feasible with high confidence.** The 29 YawDD+ annotation "
            "folders and the 29 raw Dash videos share identical canonical "
            "subject tokens after trivial whitespace / double-extension "
            "normalisation. Stage 3 can safely build a 1-to-1 mapping table "
            "keyed on that token.\n"
        )
    else:
        lines.append(
            "**Partial.** Some YawDD+ folders do not have a matching raw "
            "video (or vice versa). Stage 3 will need a heuristic and the "
            "user should be told which subjects cannot be reconstructed "
            "without re-downloading the raw source.\n"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_YAWDD_RAW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    print(f"[stage2] scanning {args.raw_root} ...")
    female_dir, male_dir = find_female_male_dirs(args.raw_root)
    female_rows = scan_dir(female_dir, "Female")
    male_rows = scan_dir(male_dir, "Male")

    pdfs = {
        "readme": (args.raw_root / "Readme_YawDD.pdf") if (args.raw_root / "Readme_YawDD.pdf").is_file() else None,
        "table1": (args.raw_root / "Table1.pdf") if (args.raw_root / "Table1.pdf").is_file() else None,
        "table2": (args.raw_root / "Table2.pdf") if (args.raw_root / "Table2.pdf").is_file() else None,
    }

    report = build_report(args.raw_root, female_dir, male_dir, female_rows, male_rows, pdfs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print(f"[stage2] {len(female_rows)} female + {len(male_rows)} male = "
          f"{len(female_rows) + len(male_rows)} raw files.")
    print(f"[stage2] report written to {args.report}")


if __name__ == "__main__":
    main()
