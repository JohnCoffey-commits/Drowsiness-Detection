"""
Stage 5 - Generate mouth ROI crops for the reconstructed YawDD+ Dash labeled frames.

Primary method:
    MediaPipe Face Mesh (via the MediaPipe Tasks `FaceLandmarker`) is used to
    detect 478 facial landmarks per frame.  The mouth ROI is computed from a
    fixed outer-lip + inner-lip landmark set, then expanded with a margin of
    max(10 px, 10% of the raw mouth bbox width/height) in each axis and clipped
    to the image boundaries.

Fallback A (`fallback_lower_face`):
    If Face Mesh does not return landmarks, an OpenCV Haar cascade frontal face
    detector is run.  When a face is found the lower 40% of the face box is
    used as the crop.

Fallback B (`failed`):
    If neither Face Mesh nor the Haar face detector find anything, no crop is
    saved and the sample is logged as `failed`.

The original YawDD+ YOLO bounding box (`yawdd_bbox_raw`) is preserved only as
traceability metadata and is NEVER used as the cropping source.

Outputs:
    - dataset/YawDD_plus_reconstructed/Dash/mouth_crops/<subject_id>/<frame_index>.jpg
    - artifacts/mappings/yawdd_dash_all_mouth_crops.csv  (merged manifest)

Usage (from the repo root):
    .venv/bin/python src/preprocessing/generate_yawdd_mouth_crops.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


# Outer + inner lip landmark indices. These are the canonical MediaPipe Face
# Mesh lip indices; the MediaPipe Tasks FaceLandmarker uses the same topology
# (478 landmarks, first 468 shared with classic Face Mesh).
MOUTH_LANDMARK_IDS: Tuple[int, ...] = (
    # outer lip
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    # inner lip
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "mappings" / "yawdd_dash_all_labeled_frames.csv"
)
DEFAULT_CROP_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "YawDD_plus_reconstructed"
    / "Dash"
    / "mouth_crops"
)
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "mappings" / "yawdd_dash_all_mouth_crops.csv"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "face_landmarker.task"
)


OUTPUT_FIELDS = [
    "subject_id",
    "frame_index",
    "image_path",
    "mouth_crop_path",
    "class_id",
    "binary_label",
    "crop_method",
    "mouth_bbox_xyxy",
    "raw_video_path",
    "annotation_txt_path",
    "yawdd_bbox_raw",
    "notes",
]


def clamp_bbox(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def mouth_bbox_from_landmarks(
    landmarks, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    xs, ys = [], []
    for idx in MOUTH_LANDMARK_IDS:
        lm = landmarks[idx]
        xs.append(lm.x * img_w)
        ys.append(lm.y * img_h)
    raw_x1, raw_x2 = min(xs), max(xs)
    raw_y1, raw_y2 = min(ys), max(ys)

    bbox_w = raw_x2 - raw_x1
    bbox_h = raw_y2 - raw_y1

    x_margin = max(10.0, 0.10 * bbox_w)
    y_margin = max(10.0, 0.10 * bbox_h)

    x1 = int(round(raw_x1 - x_margin))
    y1 = int(round(raw_y1 - y_margin))
    x2 = int(round(raw_x2 + x_margin))
    y2 = int(round(raw_y2 + y_margin))
    return clamp_bbox(x1, y1, x2, y2, img_w, img_h)


def lower_face_bbox_from_face_box(
    face_xyxy: Tuple[int, int, int, int], img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    fx1, fy1, fx2, fy2 = face_xyxy
    face_h = fy2 - fy1
    # Take the lower ~40% of the face box as the mouth/lower-face ROI.
    y1 = int(round(fy1 + 0.60 * face_h))
    return clamp_bbox(fx1, y1, fx2, fy2, img_w, img_h)


def detect_face_haar(
    cascade: cv2.CascadeClassifier, gray: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(x + w), int(y + h)


def build_landmarker(model_path: Path) -> FaceLandmarker:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return FaceLandmarker.create_from_options(options)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when saving mouth crops.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject_id whitelist (default: all subjects).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip frames that already have a crop file on disk.  Existing rows "
            "are re-emitted to the manifest so the manifest stays complete."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="How often to print progress.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_manifest.is_file():
        print(f"[error] input manifest not found: {args.input_manifest}", file=sys.stderr)
        return 2
    if not args.model_path.is_file():
        print(
            f"[error] MediaPipe model not found: {args.model_path}\n"
            "Download face_landmarker.task into artifacts/models/ first.",
            file=sys.stderr,
        )
        return 2

    subjects_filter = None
    if args.subjects:
        subjects_filter = {s.strip() for s in args.subjects.split(",") if s.strip()}

    args.crop_root.mkdir(parents=True, exist_ok=True)
    ensure_parent(args.output_manifest)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"[error] failed to load Haar cascade: {cascade_path}", file=sys.stderr)
        return 2

    print(f"[info] loading MediaPipe FaceLandmarker from {args.model_path}")
    landmarker = build_landmarker(args.model_path)

    stats = {
        "total": 0,
        "face_mesh": 0,
        "fallback_lower_face": 0,
        "failed": 0,
        "skipped_missing_image": 0,
        "resumed_existing": 0,
    }

    start = time.time()

    with args.input_manifest.open("r", newline="") as fin, args.output_manifest.open(
        "w", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for i, row in enumerate(reader):
            if args.limit is not None and i >= args.limit:
                break

            subject_id = row["subject_id"]
            if subjects_filter is not None and subject_id not in subjects_filter:
                continue

            frame_index = row["frame_index"]
            image_path = Path(row["image_path"])

            stats["total"] += 1

            crop_path = args.crop_root / subject_id / f"{frame_index}.jpg"
            ensure_parent(crop_path)

            out_row = {
                "subject_id": subject_id,
                "frame_index": frame_index,
                "image_path": str(image_path),
                "mouth_crop_path": "",
                "class_id": row["class_id"],
                "binary_label": row["binary_label"],
                "crop_method": "failed",
                "mouth_bbox_xyxy": "",
                "raw_video_path": row.get("raw_video_path", ""),
                "annotation_txt_path": row.get("annotation_txt_path", ""),
                "yawdd_bbox_raw": row.get("yawdd_bbox_raw", ""),
                "notes": "",
            }

            # Resume: if crop already exists on disk, trust the previous run
            # and emit a face_mesh row for it (we cannot recover the original
            # bbox metadata without re-running, so leave mouth_bbox_xyxy empty
            # and mark the note).  Only used when --resume is set.
            if args.resume and crop_path.is_file():
                out_row["mouth_crop_path"] = str(crop_path)
                out_row["crop_method"] = "resumed"
                out_row["notes"] = "resumed_from_existing_file"
                stats["resumed_existing"] += 1
                writer.writerow(out_row)
                if stats["total"] % args.progress_every == 0:
                    _log_progress(stats, start)
                continue

            if not image_path.is_file():
                out_row["notes"] = "image_file_missing"
                stats["skipped_missing_image"] += 1
                stats["failed"] += 1
                writer.writerow(out_row)
                continue

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                out_row["notes"] = "image_unreadable"
                stats["skipped_missing_image"] += 1
                stats["failed"] += 1
                writer.writerow(out_row)
                continue

            h, w = image_bgr.shape[:2]

            # ---------- Primary: MediaPipe Face Mesh ----------
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            try:
                result = landmarker.detect(mp_image)
            except Exception as exc:  # noqa: BLE001
                result = None
                out_row["notes"] = f"face_mesh_exception:{exc}"

            bbox: Optional[Tuple[int, int, int, int]] = None
            method = "failed"

            if result is not None and result.face_landmarks:
                try:
                    bbox = mouth_bbox_from_landmarks(result.face_landmarks[0], w, h)
                    method = "face_mesh"
                except Exception as exc:  # noqa: BLE001
                    out_row["notes"] = f"mouth_bbox_exception:{exc}"
                    bbox = None

            # ---------- Fallback A: Haar face -> lower 40% ----------
            if bbox is None:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                face_box = detect_face_haar(cascade, gray)
                if face_box is not None:
                    bbox = lower_face_bbox_from_face_box(face_box, w, h)
                    method = "fallback_lower_face"
                    if not out_row["notes"]:
                        out_row["notes"] = "face_mesh_failed_haar_used"

            # ---------- Fallback B: no crop ----------
            if bbox is None:
                method = "failed"
                if not out_row["notes"]:
                    out_row["notes"] = "face_mesh_and_haar_failed"
                stats["failed"] += 1
                out_row["crop_method"] = method
                writer.writerow(out_row)
                if stats["total"] % args.progress_every == 0:
                    _log_progress(stats, start)
                continue

            x1, y1, x2, y2 = bbox
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                out_row["notes"] = "empty_crop_after_clip"
                stats["failed"] += 1
                out_row["crop_method"] = "failed"
                writer.writerow(out_row)
                continue

            ok = cv2.imwrite(
                str(crop_path),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            if not ok:
                out_row["notes"] = "jpeg_write_failed"
                stats["failed"] += 1
                out_row["crop_method"] = "failed"
                writer.writerow(out_row)
                continue

            out_row["mouth_crop_path"] = str(crop_path)
            out_row["crop_method"] = method
            out_row["mouth_bbox_xyxy"] = f"{x1},{y1},{x2},{y2}"
            stats[method] += 1
            writer.writerow(out_row)

            if stats["total"] % args.progress_every == 0:
                _log_progress(stats, start)

    landmarker.close()

    elapsed = time.time() - start
    print(
        "\n[done] processed {total} rows in {elapsed:.1f}s "
        "(face_mesh={face_mesh}, fallback_lower_face={fallback_lower_face}, "
        "failed={failed}, resumed={resumed})".format(
            total=stats["total"],
            elapsed=elapsed,
            face_mesh=stats["face_mesh"],
            fallback_lower_face=stats["fallback_lower_face"],
            failed=stats["failed"],
            resumed=stats["resumed_existing"],
        )
    )
    print(f"[done] merged manifest written to {args.output_manifest}")
    print(f"[done] crop images written under {args.crop_root}")
    return 0


def _log_progress(stats: dict, start_ts: float) -> None:
    dt = max(time.time() - start_ts, 1e-6)
    rate = stats["total"] / dt
    print(
        "[progress] total={total:6d}  face_mesh={face_mesh:6d}  "
        "fallback={fallback_lower_face:5d}  failed={failed:4d}  "
        "resumed={resumed:5d}  rate={rate:5.1f} fps".format(
            total=stats["total"],
            face_mesh=stats["face_mesh"],
            fallback_lower_face=stats["fallback_lower_face"],
            failed=stats["failed"],
            resumed=stats["resumed_existing"],
            rate=rate,
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
