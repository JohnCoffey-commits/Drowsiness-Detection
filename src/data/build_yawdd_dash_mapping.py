"""Stage 3: Build a mapping table between YawDD+ Dash annotations and raw YawDD Dash videos.

Produces:

* `artifacts/mappings/yawdd_dash_mapping.csv`
    one row per YawDD+ subject folder, with columns
        subject_id, annotation_folder, annotation_txt_path,
        raw_source_path, mapping_confidence, mapping_notes

* `reports/yawdd_dash_mapping_report.md`
    human-readable description of the matching logic, per-subject results,
    and recommendations before Stage 4 (frame extraction).

Run:
    python -m src.data.build_yawdd_dash_mapping
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAWDD_PLUS_DASH = PROJECT_ROOT / "dataset" / "YawDD+" / "dataset" / "Dash"
DEFAULT_YAWDD_RAW = PROJECT_ROOT / "dataset" / "YawDD_raw"
DEFAULT_CSV = PROJECT_ROOT / "artifacts" / "mappings" / "yawdd_dash_mapping.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "yawdd_dash_mapping_report.md"

SUBJECT_TOKEN_RE = re.compile(r"^(\d+)-(Female|Male)(Glasses|NoGlasses|SunGlasses)?$")


def canonical_token_from_filename(name: str) -> str:
    """Remove a trailing `.avi` (once or twice) and strip whitespace."""
    s = name.strip()
    for _ in range(2):
        if s.lower().endswith(".avi"):
            s = s[: -len(".avi")]
    return s.strip()


def index_raw_videos(raw_root: Path) -> Dict[str, Path]:
    """Return {canonical_token: Path} for every raw .avi found under raw_root.

    Handles the double-nested `Dash/Dash/{Female,Male}` layout and the shallower
    `Dash/{Female,Male}` layout.
    """
    candidates: List[Path] = []
    for sub in (
        raw_root / "Dash" / "Female",
        raw_root / "Dash" / "Male",
        raw_root / "Dash" / "Dash" / "Female",
        raw_root / "Dash" / "Dash" / "Male",
    ):
        if sub.is_dir():
            for p in sub.iterdir():
                if p.is_file() and p.name.lower().endswith(".avi"):
                    candidates.append(p)

    index: Dict[str, Path] = {}
    for p in candidates:
        token = canonical_token_from_filename(p.name)
        # Prefer the first occurrence; warn on collision by keeping the larger file.
        if token in index:
            if p.stat().st_size > index[token].stat().st_size:
                index[token] = p
        else:
            index[token] = p
    return index


def first_label_txt(labels_dir: Path) -> Optional[Path]:
    if not labels_dir.is_dir():
        return None
    txts = sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    return txts[0] if txts else None


def match_subject(folder_name: str, raw_index: Dict[str, Path]) -> Tuple[Optional[Path], str, str]:
    """Return (raw_path, confidence, notes) for a single YawDD+ subject folder."""
    # Try exact canonical-token match first.
    if folder_name in raw_index:
        return raw_index[folder_name], "high", "exact canonical-token match"

    # Fallback: try case-insensitive match.
    lower = folder_name.lower()
    for tok, p in raw_index.items():
        if tok.lower() == lower:
            return p, "medium", f"case-insensitive match to '{tok}'"

    # Fallback: match by (subject_index, gender) only — ignore glasses state.
    m = SUBJECT_TOKEN_RE.match(folder_name)
    if m:
        idx = int(m.group(1))
        gender = m.group(2)
        candidates = [
            (tok, p) for tok, p in raw_index.items()
            if (SUBJECT_TOKEN_RE.match(tok) or SUBJECT_TOKEN_RE.match(tok))
            and int(SUBJECT_TOKEN_RE.match(tok).group(1)) == idx
            and SUBJECT_TOKEN_RE.match(tok).group(2) == gender
        ]
        if len(candidates) == 1:
            tok, p = candidates[0]
            return p, "medium", (
                f"glasses-state differs: YawDD+ says "
                f"'{m.group(3) or ''}' but raw token is '{tok}'"
            )
        if len(candidates) > 1:
            return None, "low", (
                f"ambiguous: multiple raw videos share (subject={idx}, "
                f"gender={gender}): {[t for t, _ in candidates]}"
            )

    return None, "none", "no raw video matches this subject"


def build_rows(plus_root: Path, raw_root: Path) -> List[Dict]:
    if not plus_root.is_dir():
        raise FileNotFoundError(f"YawDD+ Dash directory not found: {plus_root}")
    raw_index = index_raw_videos(raw_root)

    rows: List[Dict] = []
    for subject_dir in sorted(p for p in plus_root.iterdir() if p.is_dir()):
        labels_dir = subject_dir / "labels"
        sample_txt = first_label_txt(labels_dir)
        raw_path, confidence, notes = match_subject(subject_dir.name, raw_index)

        m = SUBJECT_TOKEN_RE.match(subject_dir.name)
        subject_id = f"{m.group(1)}-{m.group(2)}" if m else subject_dir.name

        rows.append(
            {
                "subject_id": subject_id,
                "annotation_folder": subject_dir.as_posix(),
                "annotation_txt_path": sample_txt.as_posix() if sample_txt else "",
                "raw_source_path": raw_path.as_posix() if raw_path else "",
                "mapping_confidence": confidence,
                "mapping_notes": notes,
            }
        )
    return rows


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "annotation_folder",
        "annotation_txt_path",
        "raw_source_path",
        "mapping_confidence",
        "mapping_notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_report(rows: List[Dict], plus_root: Path, raw_root: Path, out_csv: Path) -> str:
    high = [r for r in rows if r["mapping_confidence"] == "high"]
    medium = [r for r in rows if r["mapping_confidence"] == "medium"]
    low = [r for r in rows if r["mapping_confidence"] == "low"]
    none = [r for r in rows if r["mapping_confidence"] == "none"]

    lines: List[str] = []
    lines.append("# YawDD Dash — Annotation ↔ Raw Video Mapping Report (Stage 3)\n")
    lines.append(
        "This report is produced by `src/data/build_yawdd_dash_mapping.py`. "
        "It pairs every subject folder under `YawDD+/dataset/Dash/` with the "
        "matching raw `.avi` under `YawDD_raw/Dash/`. The CSV output is\n"
    )
    lines.append(f"`{out_csv.as_posix()}`\n")
    lines.append(f"- YawDD+ root: `{plus_root.as_posix()}`")
    lines.append(f"- Raw root:    `{raw_root.as_posix()}`")
    lines.append(f"- Subjects processed: **{len(rows)}**")
    lines.append(
        f"- Confidence: high={len(high)}, medium={len(medium)}, "
        f"low={len(low)}, none={len(none)}\n"
    )

    # Matching logic
    lines.append("## 1. Matching logic\n")
    lines.append(
        "The algorithm normalises each raw `.avi` filename by repeatedly "
        "stripping a trailing `.avi` (to collapse names like "
        "`11-FemaleGlasses.avi.avi`) and removing stray whitespace. The "
        "resulting string — for example `13-MaleNoGlasses` — is the "
        "**canonical token**.\n"
    )
    lines.append(
        "Each YawDD+ subject folder is already in canonical form, so the "
        "primary key is an exact string match between the folder name and "
        "the canonical token of a raw `.avi`.\n"
    )
    lines.append("Confidence levels used in the CSV:\n")
    lines.append(
        "- **high** — exact canonical-token match (no ambiguity, no "
        "heuristic). Safe to use for frame extraction as-is.\n"
        "- **medium** — match obtained by a case-insensitive comparison, or "
        "by dropping the `Glasses` / `NoGlasses` / `SunGlasses` suffix "
        "because the raw file and the YawDD+ folder disagree on it. The "
        "subject is the same person but the attribute label must be "
        "reviewed by hand.\n"
        "- **low** — multiple candidates remain even after the heuristics "
        "above. The script refuses to pick one.\n"
        "- **none** — no raw video could be associated with this YawDD+ "
        "folder. Frame extraction is blocked for this subject.\n"
    )

    # Heuristics used
    lines.append("## 2. Concrete rules applied in code\n")
    lines.append(
        "1. `canonical_token = name.strip().removesuffix('.avi').removesuffix('.avi').strip()`\n"
        "2. Exact token match → confidence **high**.\n"
        "3. Case-insensitive token match → confidence **medium** (note the raw token).\n"
        "4. Match on `(subject_index, gender)` ignoring `GlassesState` →\n"
        "   - exactly one candidate → **medium** with a note describing the disagreement.\n"
        "   - more than one candidate → **low** with the list of candidates.\n"
        "5. No candidate → **none**.\n"
    )

    # Per-subject table
    lines.append("## 3. Per-subject mapping\n")
    lines.append(
        "| subject_id | YawDD+ folder | raw video (basename) | confidence | notes |"
    )
    lines.append("|------------|---------------|-----------------------|------------|-------|")
    for r in rows:
        raw_bn = Path(r["raw_source_path"]).name if r["raw_source_path"] else "—"
        folder_bn = Path(r["annotation_folder"]).name
        lines.append(
            f"| {r['subject_id']} | `{folder_bn}` | `{raw_bn}` | "
            f"{r['mapping_confidence']} | {r['mapping_notes']} |"
        )
    lines.append("")

    if medium:
        lines.append("### Medium-confidence subjects (review)\n")
        for r in medium:
            lines.append(f"- **{r['subject_id']}** — {r['mapping_notes']}")
        lines.append("")
    if low:
        lines.append("### Low-confidence / ambiguous subjects\n")
        for r in low:
            lines.append(f"- **{r['subject_id']}** — {r['mapping_notes']}")
        lines.append("")
    if none:
        lines.append("### Unmatched subjects\n")
        for r in none:
            lines.append(f"- **{r['subject_id']}** — {r['mapping_notes']}")
        lines.append("")

    # Recommended next step
    lines.append("## 4. Recommended Next Step Before Frame Extraction\n")
    blocked = len(low) + len(none)
    if blocked == 0 and len(high) == len(rows):
        lines.append(
            "All 29 subjects mapped with **high** confidence. It is safe to "
            "proceed to Stage 4 (frame extraction from the raw videos) using "
            "the `raw_source_path` column of the CSV as the input list.\n"
        )
        lines.append(
            "Before running a bulk decode, do a **tiny sanity check** on one "
            "subject first:\n"
        )
        lines.append(
            "1. Decode frame `00000000` from the raw video for, e.g., "
            "`1-FemaleNoGlasses.avi`.\n"
            "2. Load the matching annotation `1-FemaleNoGlasses/labels/"
            "00000000_0.txt` and draw its bounding box on the decoded image.\n"
            "3. Confirm visually that the box frames the driver's face.\n"
            "4. Repeat for a frame whose label is class `1` (e.g. the "
            "`00001661_0.txt` file under `1-FemaleNoGlasses`) and verify the "
            "driver is visibly yawning on that frame.\n"
        )
        lines.append(
            "Only after those two checks pass should you run the bulk "
            "extractor. That will simultaneously confirm the frame-index "
            "convention and the class-id semantics that Stage 1 flagged as "
            "uncertain.\n"
        )
    elif len(high) + len(medium) == len(rows):
        lines.append(
            "All subjects have a candidate, but some were matched with "
            "medium confidence (attribute disagreement). Before Stage 4:\n"
            "- Manually review the medium-confidence rows listed above.\n"
            "- For each, open the raw video and the YawDD+ folder and "
            "confirm the same person/session.\n"
            "- Keep, drop, or re-label subjects as appropriate, then "
            "proceed to frame extraction on the cleaned list.\n"
        )
    else:
        lines.append(
            "Frame extraction is **not yet safe** for the whole set. "
            "Resolve the unmatched/low-confidence subjects first — either "
            "re-download the missing raw videos, or decide to drop those "
            "subjects — and re-run this script before proceeding to "
            "Stage 4.\n"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plus-root", type=Path, default=DEFAULT_YAWDD_PLUS_DASH)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_YAWDD_RAW)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    print(f"[stage3] building mapping for {args.plus_root} <-> {args.raw_root} ...")
    rows = build_rows(args.plus_root, args.raw_root)
    write_csv(rows, args.csv_out)
    report = build_report(rows, args.plus_root, args.raw_root, args.csv_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    by_conf = {c: sum(1 for r in rows if r["mapping_confidence"] == c)
               for c in ("high", "medium", "low", "none")}
    print(f"[stage3] rows written to {args.csv_out}")
    print(f"[stage3] report written to {args.report}")
    print(f"[stage3] confidence breakdown: {by_conf}")


if __name__ == "__main__":
    main()
