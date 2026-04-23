"""Stage 4A: small-scale visual sanity check for the YawDD+ ↔ raw Dash pipeline.

For a small hand-picked set of subjects this script:

1. reads the YawDD+ <-> raw .avi mapping produced by Stage 3,
2. seeks to specific frame indices in the raw `.avi` (using the YawDD+ filename
   as ground truth for the index),
3. loads the matching YOLO annotation, converts normalised coords to pixels,
4. draws the bounding box + class label onto the decoded frame,
5. writes the annotated image to `artifacts/visual_checks/<subject>/...`.

It does **not** perform bulk extraction. It emits a small JSON log that the
companion report can cite.

Run:
    .venv/bin/python src/data/validate_yawdd_dash_frames.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = PROJECT_ROOT / "artifacts" / "mappings" / "yawdd_dash_mapping.csv"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "visual_checks"
DEFAULT_LOG = PROJECT_ROOT / "artifacts" / "visual_checks" / "validation_log.json"

CLASS_NAMES = {0: "class0 (non-yawn)", 1: "class1 (yawn)"}
CLASS_COLORS_BGR = {0: (0, 200, 0), 1: (0, 0, 230)}

# Subjects + targeted frames for the check. Kept deliberately small. The frames
# were chosen from the Stage-1 scan:
#   - frame 0                          -> baseline alignment check
#   - a middle class-0 frame           -> non-yawn interior
#   - a known class-1 frame            -> yawn
#   - (optional) a _1 multi-object fr. -> multi-box diagnostic
DEFAULT_TARGETS: List[Dict] = [
    {
        "subject": "1-FemaleNoGlasses",
        "frames": [
            {"idx": 0,    "obj": 0, "purpose": "first frame / alignment check"},
            {"idx": 1370, "obj": 0, "purpose": "middle class-0 (non-yawn)"},
            {"idx": 1661, "obj": 0, "purpose": "class-1 (yawn)"},
        ],
    },
    {
        "subject": "13-MaleNoGlasses",
        "frames": [
            {"idx": 0,    "obj": 0, "purpose": "first frame / alignment check"},
            {"idx": 500,  "obj": 0, "purpose": "middle class-0 (non-yawn)"},
            {"idx": 1001, "obj": 0, "purpose": "class-1 (yawn)"},
        ],
    },
    {
        "subject": "8-MaleNoGlasses",
        "frames": [
            {"idx": 0,    "obj": 0, "purpose": "first frame / alignment check"},
            {"idx": 800,  "obj": 0, "purpose": "middle class-0 (non-yawn)"},
            {"idx": 1287, "obj": 0, "purpose": "class-0 paired with _1 sibling"},
            {"idx": 1287, "obj": 1, "purpose": "multi-object _1 file"},
            {"idx": 1756, "obj": 0, "purpose": "class-1 (yawn)"},
        ],
    },
]


@dataclass
class FrameResult:
    subject: str
    frame_index: int
    obj_index: int
    purpose: str
    annotation_path: str
    raw_video_path: str
    decoded_ok: bool
    decoded_shape: Optional[Tuple[int, int, int]] = None
    reported_frame_count: Optional[int] = None
    reported_fps: Optional[float] = None
    yolo_line: Optional[str] = None
    class_id: Optional[int] = None
    pixel_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    output_image: Optional[str] = None
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def read_mapping(csv_path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            folder_name = Path(row["annotation_folder"]).name
            out[folder_name] = row
    return out


def load_yolo_line(path: Path) -> Tuple[int, float, float, float, float, str]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"empty annotation file: {path}")
    first = text.splitlines()[0]
    tokens = first.split()
    if len(tokens) != 5:
        raise ValueError(f"expected 5 tokens, got {len(tokens)}: {first!r}")
    cls = int(tokens[0])
    cx, cy, w, h = (float(t) for t in tokens[1:])
    return cls, cx, cy, w, h, first


def yolo_to_pixel(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    bx, by = cx * img_w, cy * img_h
    bw, bh = w * img_w, h * img_h
    x1 = int(round(bx - bw / 2))
    y1 = int(round(by - bh / 2))
    x2 = int(round(bx + bw / 2))
    y2 = int(round(by + bh / 2))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2


def seek_and_read(cap: cv2.VideoCapture, frame_idx: int) -> Tuple[bool, Optional[np.ndarray]]:
    """Seek to the given frame index and read it.

    OpenCV's CAP_PROP_POS_FRAMES + read() is occasionally unreliable on AVI
    containers: we therefore use it as the primary strategy and, on failure,
    fall back to a linear grab loop from the current position.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if ok and frame is not None:
        return True, frame
    # Fallback: rewind and grab frame by frame (slow but robust).
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    for _ in range(frame_idx):
        if not cap.grab():
            return False, None
    ok, frame = cap.read()
    return ok, frame if ok else None


def annotate_frame(frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int],
                   class_id: int, title: str) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = bbox_xyxy
    color = CLASS_COLORS_BGR.get(class_id, (255, 255, 255))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    label = CLASS_NAMES.get(class_id, f"class{class_id}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
    cv2.rectangle(out, (x1, max(0, y1 - th - 8)),
                  (x1 + tw + 6, max(0, y1)), color, -1)
    cv2.putText(out, label, (x1 + 3, max(th + 2, y1 - 4)),
                font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Thin top banner with the overall title (subject + frame + purpose).
    banner_h = 28
    banner = np.zeros((banner_h, out.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title[:140], (8, 20), font, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, out])


def process_subject(subject: str, frames: List[Dict],
                    mapping_row: Dict[str, str], out_root: Path) -> List[FrameResult]:
    raw_video = Path(mapping_row["raw_source_path"])
    ann_dir = Path(mapping_row["annotation_folder"]) / "labels"
    results: List[FrameResult] = []

    cap = cv2.VideoCapture(str(raw_video))
    reported_count = None
    reported_fps = None
    if cap.isOpened():
        reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))

    subject_out = out_root / subject
    subject_out.mkdir(parents=True, exist_ok=True)

    if not cap.isOpened():
        for tf in frames:
            results.append(FrameResult(
                subject=subject,
                frame_index=tf["idx"],
                obj_index=tf["obj"],
                purpose=tf["purpose"],
                annotation_path=str(ann_dir / f"{tf['idx']:08d}_{tf['obj']}.txt"),
                raw_video_path=str(raw_video),
                decoded_ok=False,
                reported_frame_count=reported_count,
                reported_fps=reported_fps,
                error=f"failed to open video: {raw_video}",
            ))
        return results

    try:
        for tf in frames:
            idx = tf["idx"]
            obj = tf["obj"]
            ann_path = ann_dir / f"{idx:08d}_{obj}.txt"
            res = FrameResult(
                subject=subject,
                frame_index=idx,
                obj_index=obj,
                purpose=tf["purpose"],
                annotation_path=str(ann_path),
                raw_video_path=str(raw_video),
                decoded_ok=False,
                reported_frame_count=reported_count,
                reported_fps=reported_fps,
            )

            if not ann_path.is_file():
                res.error = f"annotation missing: {ann_path}"
                results.append(res)
                continue

            try:
                cls, cx, cy, w, h, first_line = load_yolo_line(ann_path)
            except Exception as e:
                res.error = f"annotation parse error: {e}"
                results.append(res)
                continue
            res.class_id = cls
            res.yolo_line = first_line

            ok, frame = seek_and_read(cap, idx)
            if not ok or frame is None:
                res.error = f"failed to decode frame {idx} from {raw_video.name}"
                results.append(res)
                continue
            res.decoded_ok = True
            res.decoded_shape = tuple(frame.shape)  # (H, W, C)

            img_h, img_w = frame.shape[:2]
            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
            res.pixel_bbox_xyxy = (x1, y1, x2, y2)

            title = (
                f"{subject}  frame={idx:08d}  obj={obj}  "
                f"{CLASS_NAMES.get(cls, cls)}  purpose={tf['purpose']}"
            )
            annotated = annotate_frame(frame, (x1, y1, x2, y2), cls, title)
            out_name = f"frame_{idx:08d}_obj{obj}_class{cls}.jpg"
            out_path = subject_out / out_name
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            res.output_image = str(out_path)

            # Sanity notes collected for the report.
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                res.notes.append("YOLO coordinates outside [0, 1]")
            if x1 >= x2 or y1 >= y2:
                res.notes.append("degenerate pixel bbox")

            results.append(res)
    finally:
        cap.release()

    return results


def save_log(all_results: List[FrameResult], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = [asdict(r) for r in all_results]
    log_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    mapping = read_mapping(args.mapping)
    all_results: List[FrameResult] = []
    for target in DEFAULT_TARGETS:
        subject = target["subject"]
        if subject not in mapping:
            print(f"[warn] subject {subject!r} not in mapping CSV, skipping")
            continue
        print(f"[stage4a] processing {subject} ({len(target['frames'])} frames)")
        res = process_subject(subject, target["frames"], mapping[subject], args.out_root)
        for r in res:
            status = "ok" if r.decoded_ok else f"FAIL: {r.error}"
            print(f"  frame {r.frame_index:08d} obj={r.obj_index} class={r.class_id} -> {status}")
        all_results.extend(res)

    save_log(all_results, args.log)
    print(f"[stage4a] validation log: {args.log}")
    print(f"[stage4a] images under:    {args.out_root}")


if __name__ == "__main__":
    main()
