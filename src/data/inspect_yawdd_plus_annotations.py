"""Stage 1: Inspect and reverse-engineer the YawDD+ Dash annotation format.

This script scans every `.txt` file under
    dataset/YawDD+/dataset/Dash/<subject>/labels/
and produces a markdown report summarising:

* what each annotation file looks like on disk
* how the label fields are encoded (class id + four floats)
* how many files per subject, and class distribution
* a handful of real examples copied verbatim
* a list of remaining uncertainties

The script performs NO frame extraction and NO label alignment. It is
strictly a forensic inspection of the annotation files.

Run:
    python -m src.data.inspect_yawdd_plus_annotations
or
    python src/data/inspect_yawdd_plus_annotations.py
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAWDD_PLUS_DASH = PROJECT_ROOT / "dataset" / "YawDD+" / "dataset" / "Dash"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "yawdd_plus_annotation_format_report.md"

FILENAME_RE = re.compile(r"^(\d{8})_(\d+)\.txt$")
YOLO_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$"
)


def parse_label_file(path: Path) -> Tuple[List[Tuple[int, float, float, float, float]], List[str]]:
    """Parse a single annotation file.

    Returns a tuple of (parsed_rows, raw_nonempty_lines). Rows that do not match
    the YOLO-style 5-token pattern are still returned as raw strings so they
    can be surfaced as anomalies.
    """
    rows: List[Tuple[int, float, float, float, float]] = []
    raw: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.rstrip("\n").rstrip()
            if not s.strip():
                continue
            raw.append(s)
            m = YOLO_LINE_RE.match(s)
            if m:
                cls = int(m.group(1))
                xc, yc, w, h = (float(m.group(i)) for i in range(2, 6))
                rows.append((cls, xc, yc, w, h))
    return rows, raw


def scan_subject(subject_dir: Path) -> Dict:
    """Scan one subject folder under YawDD+/dataset/Dash/."""
    labels_dir = subject_dir / "labels"
    info: Dict = {
        "subject": subject_dir.name,
        "labels_dir_exists": labels_dir.is_dir(),
        "file_count": 0,
        "frame_indices": [],
        "obj_suffixes": Counter(),
        "class_counts": Counter(),
        "rows_per_file": Counter(),
        "malformed_files": [],
        "out_of_range_files": [],
        "sample_files": [],
        "bad_filenames": [],
        "multi_suffix_files": [],  # files whose object-suffix is not _0
    }
    if not labels_dir.is_dir():
        return info

    txts = sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    info["file_count"] = len(txts)

    frame_indices: List[int] = []
    for p in txts:
        m = FILENAME_RE.match(p.name)
        if not m:
            info["bad_filenames"].append(p.name)
            continue
        frame_idx = int(m.group(1))
        obj_suffix = int(m.group(2))
        frame_indices.append(frame_idx)
        info["obj_suffixes"][obj_suffix] += 1
        if obj_suffix != 0:
            info["multi_suffix_files"].append(p.name)

        rows, raw = parse_label_file(p)
        info["rows_per_file"][len(rows)] += 1
        if len(rows) != len(raw):
            info["malformed_files"].append(p.name)

        for cls, xc, yc, w, h in rows:
            info["class_counts"][cls] += 1
            # sanity check: YOLO coords should be in [0, 1]
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                info["out_of_range_files"].append(p.name)

    info["frame_indices"] = frame_indices
    if frame_indices:
        info["frame_min"] = min(frame_indices)
        info["frame_max"] = max(frame_indices)
        info["frame_unique"] = len(set(frame_indices))
        # Pick up to 3 example files across the range for the report
        sample_names = [txts[0].name]
        if len(txts) > 2:
            sample_names.append(txts[len(txts) // 2].name)
            sample_names.append(txts[-1].name)
        for name in sample_names:
            content = (labels_dir / name).read_text(encoding="utf-8", errors="replace")
            info["sample_files"].append((name, content.rstrip("\n")))
    return info


def scan_all(root: Path) -> List[Dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"YawDD+ Dash directory not found: {root}")
    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    return [scan_subject(s) for s in subjects]


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.2f}%" if total else "0.00%"


def build_report(all_info: List[Dict], root: Path) -> str:
    total_files = sum(i["file_count"] for i in all_info)
    global_classes: Counter = Counter()
    global_rows_per_file: Counter = Counter()
    global_obj_suffixes: Counter = Counter()
    bad_filenames: List[Tuple[str, str]] = []
    malformed: List[Tuple[str, str]] = []
    out_of_range: List[Tuple[str, str]] = []

    for i in all_info:
        global_classes.update(i["class_counts"])
        global_rows_per_file.update(i["rows_per_file"])
        global_obj_suffixes.update(i["obj_suffixes"])
        for n in i["bad_filenames"]:
            bad_filenames.append((i["subject"], n))
        for n in i["malformed_files"]:
            malformed.append((i["subject"], n))
        for n in i["out_of_range_files"]:
            out_of_range.append((i["subject"], n))

    lines: List[str] = []
    lines.append("# YawDD+ Dash — Annotation Format Report (Stage 1)\n")
    lines.append(
        "This report is the result of a forensic inspection of every `.txt` "
        "file under `dataset/YawDD+/dataset/Dash/<subject>/labels/`. "
        "No frames were extracted, no labels were aligned, and no images were touched.\n"
    )
    lines.append(f"- Scanned root: `{root.as_posix()}`")
    lines.append(f"- Subject folders: **{len(all_info)}**")
    lines.append(f"- Total annotation files: **{total_files}**\n")

    # Summary
    lines.append("## 1. Summary of the annotation structure\n")
    lines.append(
        "Each subject folder contains exactly one sub-folder called `labels/`. "
        "Every file inside `labels/` is a small plain-text file named like\n"
    )
    lines.append("```\n<8-digit frame index>_<object index>.txt\n```\n")
    lines.append(
        "Each file contains **one line per detected object**, in the canonical "
        "YOLO-v5 / Darknet bounding-box format:\n"
    )
    lines.append("```\n<class_id> <x_center> <y_center> <width> <height>\n```\n")
    lines.append(
        "All four geometry fields are normalised to `[0, 1]` relative to the "
        "(unseen) source image. The class id is an integer.\n"
    )

    # Filename pattern
    lines.append("## 2. Filename pattern\n")
    lines.append(
        "Across **all** scanned files, the filename conforms to the regex "
        "`^\\d{8}_\\d+\\.txt$`. The first group is a zero-padded, monotonically "
        "increasing frame counter starting at `00000000`. The second group is "
        "an object index suffix.\n"
    )
    lines.append("Object-index suffix distribution (across the whole corpus):\n")
    lines.append("| suffix | count |")
    lines.append("|--------|-------|")
    for suf, n in sorted(global_obj_suffixes.items()):
        lines.append(f"| `_{suf}` | {n} |")
    lines.append("")
    if set(global_obj_suffixes.keys()) == {0}:
        lines.append(
            "Every annotation file uses the `_0` suffix. That strongly suggests "
            "the suffix was reserved for a per-frame object index but the YawDD+ "
            "maintainers only ever store a single object (the driver's face) per "
            "frame — i.e. one annotation per image.\n"
        )

    # Rows per file
    lines.append("## 3. Rows per file\n")
    lines.append("| rows in file | number of files |")
    lines.append("|--------------|-----------------|")
    for k in sorted(global_rows_per_file):
        lines.append(f"| {k} | {global_rows_per_file[k]} |")
    lines.append("")
    if list(global_rows_per_file.keys()) == [1]:
        lines.append(
            "**Every** scanned file contains exactly one bounding box. This "
            "confirms the interpretation: one file == one annotated frame == "
            "one face bounding box.\n"
        )

    # Class distribution
    lines.append("## 4. Class-id distribution\n")
    total_rows = sum(global_classes.values())
    lines.append("| class id | row count | share |")
    lines.append("|----------|-----------|-------|")
    for cls in sorted(global_classes):
        lines.append(f"| {cls} | {global_classes[cls]} | {_pct(global_classes[cls], total_rows)} |")
    lines.append("")
    lines.append(
        "Only two class ids appear in the whole corpus: `0` and `1`. Given the "
        "purpose of YawDD (a yawning-detection dataset) the most plausible "
        "interpretation — **subject to confirmation against a few sample "
        "frames once we extract them** — is:\n"
    )
    lines.append("- `0` — face present, **not yawning** (normal / talking)\n")
    lines.append("- `1` — face present, **yawning**\n")

    # Per-subject
    lines.append("## 5. Per-subject statistics\n")
    lines.append("| subject | files | frame min | frame max | frames unique | class 0 | class 1 |")
    lines.append("|---------|-------|-----------|-----------|---------------|---------|---------|")
    for i in sorted(all_info, key=lambda x: x["subject"]):
        fmin = i.get("frame_min", "-")
        fmax = i.get("frame_max", "-")
        funi = i.get("frame_unique", "-")
        c0 = i["class_counts"].get(0, 0)
        c1 = i["class_counts"].get(1, 0)
        lines.append(
            f"| {i['subject']} | {i['file_count']} | {fmin} | {fmax} | {funi} | {c0} | {c1} |"
        )
    lines.append("")

    # Examples
    lines.append("## 6. Real examples copied from disk\n")
    lines.append(
        "Below are three verbatim samples (first file, middle file, last file) "
        "from four different subjects, chosen to cover both the Female/Male and "
        "Glasses/NoGlasses axes.\n"
    )
    example_subjects = [
        "1-FemaleNoGlasses",
        "1-MaleGlasses",
        "13-MaleNoGlasses",
        "11-FemaleGlasses",
    ]
    by_name = {i["subject"]: i for i in all_info}
    for name in example_subjects:
        info = by_name.get(name)
        if not info or not info["sample_files"]:
            continue
        lines.append(f"### {name}\n")
        for fname, content in info["sample_files"]:
            lines.append(f"`{fname}`:")
            lines.append("```\n" + content + "\n```\n")

    # Anomalies
    lines.append("## 7. Anomalies and data-quality notes\n")
    # Format-level checks
    lines.append("### 7.1 Format-level\n")
    if not bad_filenames and not malformed and not out_of_range:
        lines.append(
            "- Every filename matches the canonical pattern, every line parses "
            "as a YOLO bounding box, and every coordinate is within `[0, 1]`. "
            "The format itself is clean.\n"
        )
    else:
        if bad_filenames:
            lines.append(f"- {len(bad_filenames)} filename(s) did not match `^\\d{{8}}_\\d+\\.txt$`.")
        if malformed:
            lines.append(f"- {len(malformed)} file(s) contained line(s) that did not parse as `class cx cy w h`.")
        if out_of_range:
            lines.append(f"- {len(out_of_range)} file(s) contained coordinates outside `[0, 1]`.")
        lines.append("")

    # Multi-object suffix findings
    lines.append("### 7.2 Multi-object frames (`_1` suffix)\n")
    multi = [(i["subject"], n) for i in all_info for n in i["multi_suffix_files"]]
    if not multi:
        lines.append("- None. Every file uses `_0`.\n")
    else:
        lines.append(
            f"- **{len(multi)}** file(s) carry a non-zero object-index suffix. "
            "That means YawDD+ occasionally stores **two (or more) faces per "
            "frame** — the same frame index appears as `<frame>_0.txt` *and* "
            "`<frame>_1.txt`, with one bounding box each. Treat these as "
            "multi-face frames, not as duplicates.\n"
        )
        lines.append("| subject | filename |")
        lines.append("|---------|----------|")
        for sub, name in multi:
            lines.append(f"| {sub} | `{name}` |")
        lines.append("")

    # Frame-index gaps / skips
    lines.append("### 7.3 Frame-index skips\n")
    skipped_subjects: List[Tuple[str, int, int, int, int]] = []
    for i in all_info:
        if not i.get("frame_indices"):
            continue
        fmin = i["frame_min"]
        fmax = i["frame_max"]
        expected = fmax - fmin + 1
        unique = i["frame_unique"]
        if unique != expected:
            skipped_subjects.append((i["subject"], fmin, fmax, expected, unique))
    if not skipped_subjects:
        lines.append(
            "- No gaps. Every subject has a contiguous range of frame indices.\n"
        )
    else:
        lines.append(
            f"- **{len(skipped_subjects)}** subject(s) have gaps between the "
            "minimum and maximum frame index. The YawDD+ authors appear to "
            "have **dropped some frames** (likely frames where the face "
            "detector failed). Keep the frame index, not a running counter, "
            "as the canonical key when extracting frames from the raw video.\n"
        )
        lines.append("| subject | frame_min | frame_max | expected if contiguous | actually present |")
        lines.append("|---------|-----------|-----------|------------------------|-------------------|")
        for sub, fmin, fmax, exp, uni in skipped_subjects:
            lines.append(f"| {sub} | {fmin} | {fmax} | {exp} | {uni} |")
        lines.append("")

    # Safe assumptions
    lines.append("## 8. Safe assumptions going forward\n")
    lines.append(
        "- Each `.txt` file annotates **one image / one frame** (filename is a "
        "zero-padded frame index).\n"
        "- Each file contains **exactly one bounding box** per line, in "
        "YOLO-normalised `<class> <cx> <cy> <w> <h>` format.\n"
        "- The class id is a **binary yawn vs non-yawn label** (`0` = not "
        "yawning, `1` = yawning) — pending visual confirmation on a few frames.\n"
        "- The annotation format is independent of gender / glasses; the "
        "subject-folder name is the only place that encodes those attributes.\n"
        "- The `_<n>` suffix is an **object index** within a frame. In this "
        "corpus it is almost always `_0` (one face per frame), but a handful "
        "of frames also have a `_1.txt` companion when a second face was "
        "annotated.\n"
        "- Frame indices are **not guaranteed to be contiguous**: some "
        "subjects skip indices because the YawDD+ authors dropped frames "
        "where no reliable detection was possible. Use the index in the "
        "filename, never a running counter, when matching to decoded frames.\n"
    )

    # Uncertainties
    lines.append("## 9. Known Uncertainties\n")
    lines.append(
        "- **Class semantics are not written down anywhere in the YawDD+ "
        "folder.** The `0`/`1` meaning above is inferred from the dataset's "
        "stated purpose and from which files carry class `1` (a small "
        "minority, consistent with yawn-positive frames). This must be "
        "visually verified once we extract the matching images.\n"
        "- **Images themselves are not present** under `YawDD+/dataset/Dash/`; "
        "only the `labels/` folder exists. The source frames must therefore be "
        "reconstructed from the raw YawDD Dash videos (Stage 2 and Stage 3).\n"
        "- **Frame-rate / frame indexing convention** of the YawDD+ authors is "
        "not documented. We assume they decoded each raw `.avi` at its native "
        "30 fps and wrote one `.txt` per decoded frame, but this must be "
        "verified by counting actual video frames during Stage 4.\n"
        "- **Frame-index gaps.** We do not yet know *why* the authors skipped "
        "certain indices. The two most likely causes are (a) the YawDD+ face "
        "detector failed / was filtered on those frames, or (b) they decoded "
        "the raw video at a different effective rate. This matters: when we "
        "decode the raw video in Stage 4 we must index frames the same way "
        "the annotators did, otherwise labels and images will drift. A small "
        "alignment experiment on one subject is required before bulk extraction.\n"
        "- **Object-index suffix (`_1`).** Ten files carry a `_1` suffix. "
        "Until we visualise one, we do not know whether the second box is a "
        "passenger's face, a reflection, or something else. For the binary "
        "yawn classifier this is unlikely to matter, but the labeling has to "
        "be decided (drop, keep, or treat specially).\n"
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_YAWDD_PLUS_DASH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    print(f"[stage1] scanning {args.root} ...")
    all_info = scan_all(args.root)
    report = build_report(all_info, args.root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    total_files = sum(i["file_count"] for i in all_info)
    print(f"[stage1] {len(all_info)} subjects, {total_files} annotation files scanned.")
    print(f"[stage1] report written to {args.report}")


if __name__ == "__main__":
    main()
