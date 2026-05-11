"""Stage 10 runtime eye ROI consistency test.

This script does not train models and does not report final drowsiness
accuracy. It checks whether eye crops taken from full-frame runtime images or
videos can be passed through the selected MRL Eye MobileNetV2 specialist and
logged as stable per-eye p_eye_closed values.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "mrl_eye" / "checkpoints" / "best_mobilenet_v2_mrl_eye.pt"
DEFAULT_FACE_LANDMARKER = PROJECT_ROOT / "artifacts" / "models" / "face_landmarker.task"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage10_eye_roi_consistency"

MODEL_NAME = "mobilenet_v2"
LABEL_MAPPING = {0: "closed", 1: "open"}
DECISION_RULE = "argmax / p_eye_closed >= 0.50 default; runtime threshold uses --closed-threshold"
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# MediaPipe landmark names are anatomical. In a mirrored camera preview, the
# visually left and right eyes may appear reversed. For Stage 10 the priority is
# deterministic crop geometry, side logging, and repeatable p_eye_closed traces.
LEFT_EYE_LANDMARK_IDS = (
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
)
RIGHT_EYE_LANDMARK_IDS = (
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
)
EYE_LANDMARKS = {
    "left": LEFT_EYE_LANDMARK_IDS,
    "right": RIGHT_EYE_LANDMARK_IDS,
}

CSV_FIELDS = [
    "source_id",
    "source_path",
    "source_type",
    "frame_index",
    "timestamp_sec",
    "face_detected",
    "num_faces",
    "eye_side",
    "eye_bbox_x1",
    "eye_bbox_y1",
    "eye_bbox_x2",
    "eye_bbox_y2",
    "eye_crop_path",
    "debug_frame_path",
    "crop_width",
    "crop_height",
    "crop_aspect_ratio",
    "landmark_ids",
    "crop_method",
    "model_name",
    "checkpoint_path",
    "device",
    "p_eye_closed",
    "p_eye_open",
    "pred_label",
    "closed_threshold",
    "decision_rule",
    "status",
    "error",
]


class Stage10Error(RuntimeError):
    """Raised for clear preflight/runtime failures."""


@dataclass(frozen=True)
class RuntimeDeps:
    cv2: Any
    np: Any
    mp: Any
    BaseOptions: Any
    FaceLandmarker: Any
    FaceLandmarkerOptions: Any
    RunningMode: Any
    torch: Any
    nn: Any
    models: Any
    transforms: Any
    Image: Any
    ImageDraw: Any
    ImageFont: Any
    ImageOps: Any


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    crops: Path
    debug_frames: Path
    contact_sheets: Path
    predictions_csv: Path
    failures_csv: Path
    summary_json: Path
    report_md: Path


@dataclass
class CropSample:
    eye_side: str
    p_eye_closed: float
    pred_label: str
    source_id: str
    image: Any


def load_runtime_deps() -> RuntimeDeps:
    missing: list[str] = []

    try:
        import cv2
    except ImportError:
        cv2 = None
        missing.append("opencv-python")

    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    except ImportError:
        mp = None
        BaseOptions = None
        FaceLandmarker = None
        FaceLandmarkerOptions = None
        RunningMode = None
        missing.append("mediapipe")

    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
    except ImportError:
        torch = None
        nn = None
        models = None
        transforms = None
        missing.append("torch/torchvision")

    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError:
        Image = None
        ImageDraw = None
        ImageFont = None
        ImageOps = None
        missing.append("pillow")

    if missing:
        unique = sorted(set(missing))
        raise Stage10Error(
            "Missing runtime dependencies: "
            + ", ".join(unique)
            + ". Suggested install command: pip install mediapipe opencv-python pillow numpy torch torchvision"
        )

    return RuntimeDeps(
        cv2=cv2,
        np=np,
        mp=mp,
        BaseOptions=BaseOptions,
        FaceLandmarker=FaceLandmarker,
        FaceLandmarkerOptions=FaceLandmarkerOptions,
        RunningMode=RunningMode,
        torch=torch,
        nn=nn,
        models=models,
        transforms=transforms,
        Image=Image,
        ImageDraw=ImageDraw,
        ImageFont=ImageFont,
        ImageOps=ImageOps,
    )


def normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_output_paths(output_dir: Path) -> OutputPaths:
    root = normalize_path(output_dir)
    paths = OutputPaths(
        root=root,
        crops=root / "crops",
        debug_frames=root / "debug_frames",
        contact_sheets=root / "contact_sheets",
        predictions_csv=root / "runtime_eye_roi_predictions.csv",
        failures_csv=root / "failures.csv",
        summary_json=root / "summary.json",
        report_md=root / "STAGE10_RUNTIME_EYE_ROI_REPORT.md",
    )
    for path in [paths.root, paths.crops, paths.debug_frames, paths.contact_sheets]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def base_row(
    *,
    source_id: str,
    source_path: Path,
    source_type: str,
    frame_index: int | None,
    timestamp_sec: float | None,
    checkpoint_path: Path,
    device: str,
    closed_threshold: float,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_path": source_path.as_posix(),
        "source_type": source_type,
        "frame_index": "" if frame_index is None else frame_index,
        "timestamp_sec": "" if timestamp_sec is None else f"{timestamp_sec:.6f}",
        "face_detected": "",
        "num_faces": "",
        "eye_side": "",
        "eye_bbox_x1": "",
        "eye_bbox_y1": "",
        "eye_bbox_x2": "",
        "eye_bbox_y2": "",
        "eye_crop_path": "",
        "debug_frame_path": "",
        "crop_width": "",
        "crop_height": "",
        "crop_aspect_ratio": "",
        "landmark_ids": "",
        "crop_method": "",
        "model_name": MODEL_NAME,
        "checkpoint_path": checkpoint_path.as_posix(),
        "device": device,
        "p_eye_closed": "",
        "p_eye_open": "",
        "pred_label": "",
        "closed_threshold": f"{closed_threshold:.4f}",
        "decision_rule": DECISION_RULE,
        "status": "",
        "error": "",
    }


def failure_row(status: str, error: str, **kwargs: Any) -> dict[str, Any]:
    row = base_row(**kwargs)
    row["status"] = status
    row["error"] = error
    return row


def select_device(deps: RuntimeDeps, requested: str) -> Any:
    torch = deps.torch
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise Stage10Error("Requested CUDA device, but torch.cuda.is_available() is false.")
        if requested == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise Stage10Error("Requested MPS device, but torch.backends.mps.is_available() is false.")
        if requested not in {"cuda", "mps", "cpu"}:
            raise Stage10Error(f"Unsupported --device value: {requested}")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_eval_transform(deps: RuntimeDeps, image_size: int) -> Any:
    return deps.transforms.Compose(
        [
            deps.transforms.Resize(image_size + 16),
            deps.transforms.CenterCrop(image_size),
            deps.transforms.ToTensor(),
            deps.transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def build_mobilenet_v2(deps: RuntimeDeps) -> Any:
    model = deps.models.mobilenet_v2(weights=None)
    model.classifier[-1] = deps.nn.Linear(model.classifier[-1].in_features, 2)
    return model


def serializable_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ["model_name", "epoch", "best_val_macro_f1", "label_mapping", "image_size", "outputs"]:
        if key in payload:
            value = payload[key]
            if key == "label_mapping" and isinstance(value, dict):
                metadata[key] = {str(k): v for k, v in value.items()}
            else:
                metadata[key] = value
    return metadata


def validate_checkpoint_metadata(payload: dict[str, Any], image_size: int) -> list[str]:
    warnings: list[str] = []
    model_name = payload.get("model_name")
    if model_name is not None and model_name != MODEL_NAME:
        warnings.append(f"checkpoint model_name is {model_name!r}; expected {MODEL_NAME!r}")

    label_mapping = payload.get("label_mapping")
    if label_mapping is not None:
        normalized = {int(k): v for k, v in label_mapping.items()}
        if normalized != LABEL_MAPPING:
            warnings.append(f"checkpoint label_mapping is {normalized!r}; expected {LABEL_MAPPING!r}")

    checkpoint_image_size = payload.get("image_size")
    if checkpoint_image_size is not None and int(checkpoint_image_size) != int(image_size):
        warnings.append(f"checkpoint image_size is {checkpoint_image_size}; runtime image_size is {image_size}")
    return warnings


def load_model_and_metadata(
    deps: RuntimeDeps,
    checkpoint_path: Path,
    device: Any,
    image_size: int,
) -> tuple[Any, dict[str, Any], list[str]]:
    checkpoint_path = normalize_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise Stage10Error(f"Missing checkpoint: {checkpoint_path}")

    model = build_mobilenet_v2(deps)
    try:
        payload = deps.torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = deps.torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise Stage10Error("Checkpoint payload is not a dict.")
    if "state_dict" not in payload:
        raise Stage10Error('Checkpoint payload is missing required key "state_dict".')

    metadata_warnings = validate_checkpoint_metadata(payload, image_size)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, serializable_metadata(payload), metadata_warnings


def validate_assets(checkpoint_path: Path, face_landmarker_path: Path) -> None:
    checkpoint_path = normalize_path(checkpoint_path)
    face_landmarker_path = normalize_path(face_landmarker_path)
    if not checkpoint_path.is_file():
        raise Stage10Error(f"Missing checkpoint: {checkpoint_path}")
    if not face_landmarker_path.is_file():
        raise Stage10Error(f"Missing FaceLandmarker asset: {face_landmarker_path}")


def build_landmarker(deps: RuntimeDeps, model_path: Path) -> Any:
    options = deps.FaceLandmarkerOptions(
        base_options=deps.BaseOptions(model_asset_path=str(model_path)),
        running_mode=deps.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return deps.FaceLandmarker.create_from_options(options)


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def eye_bbox_from_landmarks(
    landmarks: Any,
    landmark_ids: tuple[int, ...],
    width: int,
    height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    usable_ids = [idx for idx in landmark_ids if idx < len(landmarks)]
    if not usable_ids:
        raise ValueError("no requested eye landmarks are available")

    xs = [landmarks[idx].x * width for idx in usable_ids]
    ys = [landmarks[idx].y * height for idx in usable_ids]
    raw_x1, raw_x2 = min(xs), max(xs)
    raw_y1, raw_y2 = min(ys), max(ys)
    bbox_w = raw_x2 - raw_x1
    bbox_h = raw_y2 - raw_y1
    if bbox_w <= 0 or bbox_h <= 0:
        raise ValueError("landmark bbox has non-positive size")

    x_margin = max(4.0, margin * bbox_w)
    y_margin = max(4.0, margin * bbox_h)
    return clamp_bbox(
        int(round(raw_x1 - x_margin)),
        int(round(raw_y1 - y_margin)),
        int(round(raw_x2 + x_margin)),
        int(round(raw_y2 + y_margin)),
        width,
        height,
    )


def resolve_input_images(input_images: str | None) -> list[Path]:
    if not input_images:
        return []

    raw = Path(input_images)
    if raw.exists() and raw.is_file():
        return [normalize_path(raw)]
    if raw.exists() and raw.is_dir():
        return sorted(
            path
            for path in normalize_path(raw).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    matches = sorted(Path(match) for match in glob.glob(input_images))
    matches = [normalize_path(path) for path in matches if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return matches


def pil_from_bgr(deps: RuntimeDeps, image_bgr: Any) -> Any:
    image_rgb = deps.cv2.cvtColor(image_bgr, deps.cv2.COLOR_BGR2RGB)
    return deps.Image.fromarray(image_rgb).convert("RGB")


def run_eye_inference(
    deps: RuntimeDeps,
    model: Any,
    transform: Any,
    device: Any,
    crop_bgr: Any,
    closed_threshold: float,
) -> tuple[float, float, str]:
    pil_image = pil_from_bgr(deps, crop_bgr)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with deps.torch.no_grad():
        logits = model(tensor)
        probs = deps.torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    p_eye_closed = float(probs[0])
    p_eye_open = float(probs[1])
    pred_label = "closed" if p_eye_closed >= closed_threshold else "open"
    return p_eye_closed, p_eye_open, pred_label


def draw_debug_frame(
    deps: RuntimeDeps,
    frame_bgr: Any,
    overlay_items: list[tuple[str, tuple[int, int, int, int], float | None, str]],
    output_path: Path,
) -> str:
    debug = frame_bgr.copy()
    colors = {"left": (0, 255, 0), "right": (255, 160, 0)}
    for eye_side, bbox, p_closed, status in overlay_items:
        x1, y1, x2, y2 = bbox
        color = colors.get(eye_side, (255, 255, 255))
        deps.cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        if p_closed is None:
            label = f"{eye_side} {status}"
        else:
            label = f"{eye_side} pC={p_closed:.2f}"
        deps.cv2.putText(
            debug,
            label,
            (x1, max(14, y1 - 6)),
            deps.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            deps.cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = deps.cv2.imwrite(str(output_path), debug)
    return output_path.as_posix() if ok else ""


def process_frame(
    *,
    deps: RuntimeDeps,
    landmarker: Any,
    model: Any,
    transform: Any,
    device: Any,
    frame_bgr: Any,
    source_id: str,
    source_path: Path,
    source_type: str,
    frame_index: int | None,
    timestamp_sec: float | None,
    paths: OutputPaths,
    args: argparse.Namespace,
    prediction_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    crop_samples: list[CropSample],
) -> None:
    checkpoint_path = normalize_path(args.checkpoint)
    base_kwargs = {
        "source_id": source_id,
        "source_path": source_path,
        "source_type": source_type,
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "closed_threshold": args.closed_threshold,
    }

    height, width = frame_bgr.shape[:2]
    image_rgb = deps.cv2.cvtColor(frame_bgr, deps.cv2.COLOR_BGR2RGB)
    mp_image = deps.mp.Image(image_format=deps.mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if result is None or not result.face_landmarks:
        row = failure_row("no_face", "MediaPipe FaceLandmarker returned no face landmarks.", **base_kwargs)
        row["face_detected"] = False
        row["num_faces"] = 0
        failure_rows.append(row)
        return

    landmarks = result.face_landmarks[0]
    num_faces = len(result.face_landmarks)
    overlay_items: list[tuple[str, tuple[int, int, int, int], float | None, str]] = []
    debug_path = ""
    if args.save_debug_frames:
        debug_path = (paths.debug_frames / f"{source_id}.jpg").as_posix()

    pending_debug_rows: list[dict[str, Any]] = []
    for eye_side, landmark_ids in EYE_LANDMARKS.items():
        row = base_row(**base_kwargs)
        row["face_detected"] = True
        row["num_faces"] = num_faces
        row["eye_side"] = eye_side
        row["landmark_ids"] = ",".join(str(idx) for idx in landmark_ids)
        row["crop_method"] = "mediapipe_face_landmarker_eye_bbox"
        row["debug_frame_path"] = debug_path

        try:
            x1, y1, x2, y2 = eye_bbox_from_landmarks(landmarks, landmark_ids, width, height, args.eye_margin)
            row.update(
                {
                    "eye_bbox_x1": x1,
                    "eye_bbox_y1": y1,
                    "eye_bbox_x2": x2,
                    "eye_bbox_y2": y2,
                }
            )
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError("empty crop after bbox clamp")

            crop_height, crop_width = crop.shape[:2]
            row.update(
                {
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                    "crop_aspect_ratio": f"{crop_width / crop_height:.6f}",
                }
            )
            crop_path = ""
            if args.save_crops:
                crop_out = paths.crops / f"{source_id}_{eye_side}.jpg"
                if deps.cv2.imwrite(str(crop_out), crop):
                    crop_path = crop_out.as_posix()

            p_eye_closed, p_eye_open, pred_label = run_eye_inference(
                deps,
                model,
                transform,
                device,
                crop,
                args.closed_threshold,
            )

            row.update(
                {
                    "eye_crop_path": crop_path,
                    "p_eye_closed": f"{p_eye_closed:.8f}",
                    "p_eye_open": f"{p_eye_open:.8f}",
                    "pred_label": pred_label,
                    "status": "ok",
                }
            )
            prediction_rows.append(row)
            crop_samples.append(
                CropSample(
                    eye_side=eye_side,
                    p_eye_closed=p_eye_closed,
                    pred_label=pred_label,
                    source_id=source_id,
                    image=pil_from_bgr(deps, crop),
                )
            )
            overlay_items.append((eye_side, (x1, y1, x2, y2), p_eye_closed, "ok"))
            pending_debug_rows.append(row)
        except ValueError as exc:
            row["status"] = "invalid_crop"
            row["error"] = str(exc)
            failure_rows.append(row)
        except Exception as exc:  # noqa: BLE001 - preserve runtime inference failures.
            row["status"] = "inference_failed"
            row["error"] = repr(exc)
            failure_rows.append(row)

    if args.save_debug_frames and overlay_items:
        written_path = draw_debug_frame(deps, frame_bgr, overlay_items, Path(debug_path),)
        for row in pending_debug_rows:
            row["debug_frame_path"] = written_path


def process_images(
    *,
    deps: RuntimeDeps,
    landmarker: Any,
    model: Any,
    transform: Any,
    device: Any,
    image_paths: list[Path],
    paths: OutputPaths,
    args: argparse.Namespace,
    prediction_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    crop_samples: list[CropSample],
) -> None:
    for index, image_path in enumerate(image_paths, start=1):
        source_id = f"image_{index:06d}"
        image_bgr = deps.cv2.imread(str(image_path))
        if image_bgr is None:
            row = failure_row(
                "image_decode_failed",
                "OpenCV could not decode image.",
                source_id=source_id,
                source_path=image_path,
                source_type="image",
                frame_index=None,
                timestamp_sec=None,
                checkpoint_path=normalize_path(args.checkpoint),
                device=str(device),
                closed_threshold=args.closed_threshold,
            )
            failure_rows.append(row)
            continue

        process_frame(
            deps=deps,
            landmarker=landmarker,
            model=model,
            transform=transform,
            device=device,
            frame_bgr=image_bgr,
            source_id=source_id,
            source_path=image_path,
            source_type="image",
            frame_index=None,
            timestamp_sec=None,
            paths=paths,
            args=args,
            prediction_rows=prediction_rows,
            failure_rows=failure_rows,
            crop_samples=crop_samples,
        )


def process_video(
    *,
    deps: RuntimeDeps,
    landmarker: Any,
    model: Any,
    transform: Any,
    device: Any,
    video_path: Path,
    paths: OutputPaths,
    args: argparse.Namespace,
    prediction_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    crop_samples: list[CropSample],
) -> int:
    video_path = normalize_path(video_path)
    if not video_path.is_file():
        raise Stage10Error(f"Input video not found: {video_path}")
    cap = deps.cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Stage10Error(f"Could not open input video: {video_path}")

    fps = float(cap.get(deps.cv2.CAP_PROP_FPS) or 0.0)
    frame_index = 0
    sampled = 0
    try:
        while sampled < args.max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % args.sample_every_n_frames == 0:
                sampled += 1
                timestamp_sec = frame_index / fps if fps > 0 else None
                source_id = f"video_{sampled:06d}_frame_{frame_index:08d}"
                process_frame(
                    deps=deps,
                    landmarker=landmarker,
                    model=model,
                    transform=transform,
                    device=device,
                    frame_bgr=frame_bgr,
                    source_id=source_id,
                    source_path=video_path,
                    source_type="video_frame",
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    paths=paths,
                    args=args,
                    prediction_rows=prediction_rows,
                    failure_rows=failure_rows,
                    crop_samples=crop_samples,
                )
            frame_index += 1
    finally:
        cap.release()
    return sampled


def contact_sheet(samples: list[CropSample], output_path: Path, title: str, deps: RuntimeDeps, max_images: int) -> None:
    if not samples:
        return
    selected = samples[:max_images]
    thumb_size = (112, 72)
    padding = 8
    label_height = 34
    title_height = 28
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    sheet = deps.Image.new(
        "RGB",
        (columns * (thumb_size[0] + 2 * padding), title_height + rows * (thumb_size[1] + label_height + 2 * padding)),
        "white",
    )
    draw = deps.ImageDraw.Draw(sheet)
    font = deps.ImageFont.load_default()
    draw.text((padding, 8), title, fill="black", font=font)

    for idx, sample in enumerate(selected):
        col = idx % columns
        row = idx // columns
        x0 = col * (thumb_size[0] + 2 * padding) + padding
        y0 = title_height + row * (thumb_size[1] + label_height + 2 * padding) + padding
        image = sample.image.copy()
        image.thumbnail(thumb_size, deps.Image.Resampling.LANCZOS)
        canvas = deps.Image.new("RGB", thumb_size, "white")
        canvas.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
        sheet.paste(canvas, (x0, y0))
        draw.text((x0, y0 + thumb_size[1] + 4), f"{sample.eye_side} pC={sample.p_eye_closed:.2f}", fill="black", font=font)
        draw.text((x0, y0 + thumb_size[1] + 17), sample.pred_label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_contact_sheets(samples: list[CropSample], paths: OutputPaths, deps: RuntimeDeps, max_images: int) -> list[str]:
    created: list[str] = []
    plans = [
        ("left_eye_samples.jpg", "Left eye runtime samples", [s for s in samples if s.eye_side == "left"]),
        ("right_eye_samples.jpg", "Right eye runtime samples", [s for s in samples if s.eye_side == "right"]),
        ("high_p_eye_closed.jpg", "Highest p_eye_closed runtime samples", sorted(samples, key=lambda s: s.p_eye_closed, reverse=True)),
        ("low_p_eye_closed.jpg", "Lowest p_eye_closed runtime samples", sorted(samples, key=lambda s: s.p_eye_closed)),
        ("mixed_runtime_eye_samples.jpg", "Mixed runtime eye samples", samples),
    ]
    for filename, title, group in plans:
        if not group:
            continue
        output_path = paths.contact_sheets / filename
        contact_sheet(group, output_path, title, deps, max_images)
        created.append(output_path.as_posix())
    return created


def p_closed_stats(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["p_eye_closed"]) for row in prediction_rows if row.get("status") == "ok"]
    if not values:
        return {
            "p_eye_closed_overall": None,
            "mean_p_eye_closed_by_eye_side": {},
            "num_predicted_closed": 0,
            "num_predicted_open": 0,
        }

    by_side: dict[str, list[float]] = {}
    for row in prediction_rows:
        if row.get("status") != "ok":
            continue
        by_side.setdefault(str(row["eye_side"]), []).append(float(row["p_eye_closed"]))

    return {
        "p_eye_closed_overall": {
            "mean": mean(values),
            "std": pstdev(values),
            "min": min(values),
            "max": max(values),
        },
        "mean_p_eye_closed_by_eye_side": {side: mean(side_values) for side, side_values in by_side.items()},
        "num_predicted_closed": sum(1 for row in prediction_rows if row.get("pred_label") == "closed"),
        "num_predicted_open": sum(1 for row in prediction_rows if row.get("pred_label") == "open"),
    }


def pstdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def build_summary(
    *,
    args: argparse.Namespace,
    mode: str,
    image_count: int,
    sampled_frame_count: int,
    prediction_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    checkpoint_metadata: dict[str, Any],
    metadata_warnings: list[str],
    device: Any,
    contact_sheets: list[str],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in failure_rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    stats = p_closed_stats(prediction_rows)
    attempted = 0 if mode == "preflight_only" else image_count + sampled_frame_count
    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input_image_count": image_count,
        "video_path": normalize_path(args.input_video).as_posix() if args.input_video else None,
        "sampled_frame_count": sampled_frame_count,
        "attempted_frame_image_count": attempted,
        "successful_eye_crop_count": len(prediction_rows),
        "failure_count": len(failure_rows),
        "no_face_count": status_counts.get("no_face", 0),
        "invalid_crop_count": status_counts.get("invalid_crop", 0),
        "inference_failed_count": status_counts.get("inference_failed", 0),
        "model_name": MODEL_NAME,
        "checkpoint_path": normalize_path(args.checkpoint).as_posix(),
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_metadata_warnings": metadata_warnings,
        "device": str(device),
        "image_size": args.image_size,
        "closed_threshold": args.closed_threshold,
        "decision_rule": DECISION_RULE,
        "contact_sheets": contact_sheets,
        **stats,
        "warning": (
            "This is runtime eye ROI consistency testing only. It is not final "
            "system-level drowsiness accuracy and it is not a fatigue score."
        ),
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(paths: OutputPaths, summary: dict[str, Any], prediction_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 10 Runtime Eye ROI Consistency Report",
        "",
        "This report documents a runtime eye ROI consistency check for the MRL Eye MobileNetV2 specialist.",
        "",
        "It is not a final drowsiness classifier, not a fatigue score, and not final system-level drowsiness accuracy.",
        "",
        "## Run Summary",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Device: `{summary['device']}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        f"- Image size: `{summary['image_size']}`",
        f"- Closed threshold: `{summary['closed_threshold']}`",
        f"- Input images: {summary['input_image_count']}",
        f"- Sampled video frames: {summary['sampled_frame_count']}",
        f"- Attempted frames/images: {summary['attempted_frame_image_count']}",
        f"- Successful eye crops: {summary['successful_eye_crop_count']}",
        f"- Failure rows: {summary['failure_count']}",
        f"- No-face rows: {summary['no_face_count']}",
        f"- Invalid-crop rows: {summary['invalid_crop_count']}",
        f"- Inference-failed rows: {summary['inference_failed_count']}",
        "",
        "## Artifacts",
        "",
        f"- Predictions CSV: `{paths.predictions_csv.as_posix()}`",
        f"- Failures CSV: `{paths.failures_csv.as_posix()}`",
        f"- Summary JSON: `{paths.summary_json.as_posix()}`",
        f"- Contact sheets directory: `{paths.contact_sheets.as_posix()}`",
        f"- Crops directory: `{paths.crops.as_posix()}`",
        f"- Debug frames directory: `{paths.debug_frames.as_posix()}`",
        "",
    ]
    if summary.get("preflight_error"):
        lines.extend(
            [
                "## Preflight Error",
                "",
                summary["preflight_error"],
                "",
            ]
        )
    if summary.get("checkpoint_metadata_warnings"):
        lines.extend(["## Checkpoint Metadata Warnings", ""])
        for warning in summary["checkpoint_metadata_warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    if prediction_rows:
        stats = summary.get("p_eye_closed_overall") or {}
        lines.extend(
            [
                "## Runtime Probability Snapshot",
                "",
                f"- Mean p_eye_closed: {stats.get('mean')}",
                f"- Min p_eye_closed: {stats.get('min')}",
                f"- Max p_eye_closed: {stats.get('max')}",
                f"- Predicted closed eyes: {summary['num_predicted_closed']}",
                f"- Predicted open eyes: {summary['num_predicted_open']}",
                "",
            ]
        )

    if failure_rows:
        lines.extend(
            [
                "## Failure Handling",
                "",
                "Failures are logged in `failures.csv`. Frames/images are not silently dropped.",
                "",
            ]
        )

    paths.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failed_preflight_artifacts(args: argparse.Namespace, error: str) -> Path | None:
    try:
        paths = ensure_output_paths(args.output_dir)
        write_csv(paths.predictions_csv, [])
        write_csv(paths.failures_csv, [])
        summary = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "preflight_only",
            "preflight_status": "failed",
            "preflight_error": error,
            "input_image_count": 0,
            "video_path": normalize_path(args.input_video).as_posix() if args.input_video else None,
            "sampled_frame_count": 0,
            "attempted_frame_image_count": 0,
            "successful_eye_crop_count": 0,
            "failure_count": 1,
            "no_face_count": 0,
            "invalid_crop_count": 0,
            "inference_failed_count": 0,
            "model_name": MODEL_NAME,
            "checkpoint_path": normalize_path(args.checkpoint).as_posix(),
            "checkpoint_metadata": {},
            "checkpoint_metadata_warnings": [],
            "device": "unavailable",
            "image_size": args.image_size,
            "closed_threshold": args.closed_threshold,
            "decision_rule": DECISION_RULE,
            "contact_sheets": [],
            "p_eye_closed_overall": None,
            "mean_p_eye_closed_by_eye_side": {},
            "num_predicted_closed": 0,
            "num_predicted_open": 0,
            "warning": (
                "This is runtime eye ROI consistency testing only. It is not final "
                "system-level drowsiness accuracy and it is not a fatigue score."
            ),
        }
        write_summary(paths.summary_json, summary)
        write_report(paths, summary, [], [])
        return paths.summary_json
    except Exception:  # noqa: BLE001 - keep original preflight error visible.
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-images", type=str, default=None)
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--sample-every-n-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--face-landmarker", type=Path, default=DEFAULT_FACE_LANDMARKER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--closed-threshold", type=float, default=0.50)
    parser.add_argument("--eye-margin", type=float, default=0.35)
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--save-debug-frames", action="store_true")
    parser.add_argument("--contact-sheet-max", type=int, default=64)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.sample_every_n_frames <= 0:
        raise Stage10Error("--sample-every-n-frames must be positive.")
    if args.max_frames <= 0:
        raise Stage10Error("--max-frames must be positive.")
    if args.image_size <= 0:
        raise Stage10Error("--image-size must be positive.")
    if args.eye_margin < 0:
        raise Stage10Error("--eye-margin must be non-negative.")
    if not args.preflight and not args.input_images and not args.input_video:
        raise Stage10Error("At least one of --input-images or --input-video is required unless --preflight is set.")

    deps = load_runtime_deps()
    checkpoint_path = normalize_path(args.checkpoint)
    face_landmarker_path = normalize_path(args.face_landmarker)
    validate_assets(checkpoint_path, face_landmarker_path)
    paths = ensure_output_paths(args.output_dir)
    device = select_device(deps, args.device)
    transform = build_eval_transform(deps, args.image_size)
    model, checkpoint_metadata, metadata_warnings = load_model_and_metadata(
        deps,
        checkpoint_path,
        device,
        args.image_size,
    )

    prediction_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    crop_samples: list[CropSample] = []
    image_paths: list[Path] = []
    sampled_frames = 0

    if args.input_images:
        image_paths = resolve_input_images(args.input_images)
        if not image_paths:
            raise Stage10Error(f"--input-images matched no supported image files: {args.input_images}")

    if not args.preflight:
        landmarker = build_landmarker(deps, face_landmarker_path)
        try:
            if image_paths:
                process_images(
                    deps=deps,
                    landmarker=landmarker,
                    model=model,
                    transform=transform,
                    device=device,
                    image_paths=image_paths,
                    paths=paths,
                    args=args,
                    prediction_rows=prediction_rows,
                    failure_rows=failure_rows,
                    crop_samples=crop_samples,
                )
            if args.input_video:
                sampled_frames = process_video(
                    deps=deps,
                    landmarker=landmarker,
                    model=model,
                    transform=transform,
                    device=device,
                    video_path=args.input_video,
                    paths=paths,
                    args=args,
                    prediction_rows=prediction_rows,
                    failure_rows=failure_rows,
                    crop_samples=crop_samples,
                )
        finally:
            landmarker.close()

    contact_sheets = write_contact_sheets(crop_samples, paths, deps, args.contact_sheet_max)
    write_csv(paths.predictions_csv, prediction_rows)
    write_csv(paths.failures_csv, failure_rows)
    summary = build_summary(
        args=args,
        mode="preflight_only" if args.preflight else "runtime_processing",
        image_count=len(image_paths),
        sampled_frame_count=sampled_frames,
        prediction_rows=prediction_rows,
        failure_rows=failure_rows,
        checkpoint_metadata=checkpoint_metadata,
        metadata_warnings=metadata_warnings,
        device=device,
        contact_sheets=contact_sheets,
    )
    write_summary(paths.summary_json, summary)
    write_report(paths, summary, prediction_rows, failure_rows)
    print(f"[done] predictions: {paths.predictions_csv}")
    print(f"[done] failures: {paths.failures_csv}")
    print(f"[done] summary: {paths.summary_json}")
    print(f"[done] report: {paths.report_md}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Stage10Error as exc:
        summary_path = write_failed_preflight_artifacts(args, str(exc)) if args.preflight else None
        if summary_path is not None:
            print(f"[error] {exc}; failure summary written to {summary_path}", file=sys.stderr)
            return 2
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
