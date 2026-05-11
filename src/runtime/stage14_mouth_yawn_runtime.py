#!/usr/bin/env python3
"""Stage 14 runtime mouth/yawn ROI consistency validation.

This script loads the recovered Stage 7 YawDD/YawDD+ Dash mouth/yawn
ResNet18 specialist and produces timestamped p_yawn timelines from full-face
videos. It does not train models and does not claim final drowsiness accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "resnet18_best.pt"
DEFAULT_FACE_LANDMARKER = PROJECT_ROOT / "artifacts" / "models" / "face_landmarker.task"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage14_mouth_yawn_runtime"

MODEL_NAME = "resnet18"
LABEL_MAPPING = {0: "no_yawn", 1: "yawn"}
P_YAWN_CLASS_INDEX = 1
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
WARNING = "This is runtime mouth/yawn inference, not final system-level drowsiness accuracy."

# Same MediaPipe Face Mesh mouth/lip topology as Stage 5 mouth ROI generation.
MOUTH_LANDMARK_IDS = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
)

PREDICTION_FIELDS = [
    "video_path",
    "video_slug",
    "frame_index",
    "timestamp_sec",
    "mouth_bbox_x1",
    "mouth_bbox_y1",
    "mouth_bbox_x2",
    "mouth_bbox_y2",
    "p_yawn",
    "p_no_yawn",
    "predicted_label",
    "yawn_event",
    "recent_yawn_event",
    "mouth_signal_status",
    "checkpoint_path",
    "model_name",
    "label_mapping",
    "mouth_crop_path",
    "debug_frame_path",
]

FAILURE_FIELDS = [
    "video_path",
    "video_slug",
    "frame_index",
    "timestamp_sec",
    "failure_type",
    "failure_reason",
]


class Stage14Error(RuntimeError):
    """Raised for explicit Stage 14 preflight/runtime failures."""


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


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    crops: Path
    debug_frames: Path
    contact_sheets: Path
    figures: Path
    predictions_csv: Path
    failures_csv: Path
    summary_json: Path
    report_md: Path


@dataclass
class CropSample:
    image: Any
    p_yawn: float
    predicted_label: str
    frame_index: int
    timestamp_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 14 runtime mouth/yawn inference.")
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--face-landmarker", type=Path, default=DEFAULT_FACE_LANDMARKER)
    parser.add_argument("--sample-every-n-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--yawn-threshold", type=float, default=0.50)
    parser.add_argument("--recent-yawn-window-sec", type=float, default=8.0)
    parser.add_argument("--mouth-margin", type=float, default=0.10)
    parser.add_argument("--min-mouth-margin-px", type=int, default=10)
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--save-debug-frames", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


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
        from PIL import Image, ImageDraw
    except ImportError:
        Image = None
        ImageDraw = None
        missing.append("pillow")
    if missing:
        raise Stage14Error(
            "Missing dependencies: "
            + ", ".join(sorted(set(missing)))
            + ". Suggested install: pip install mediapipe opencv-python pillow numpy matplotlib torch torchvision"
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
    )


def ensure_output_paths(output_dir: Path) -> OutputPaths:
    root = normalize_path(output_dir)
    paths = OutputPaths(
        root=root,
        crops=root / "crops",
        debug_frames=root / "debug_frames",
        contact_sheets=root / "contact_sheets",
        figures=root / "figures",
        predictions_csv=root / "runtime_mouth_yawn_predictions.csv",
        failures_csv=root / "failures.csv",
        summary_json=root / "summary.json",
        report_md=root / "STAGE14_RUNTIME_MOUTH_YAWN_REPORT.md",
    )
    for path in [paths.root, paths.crops, paths.debug_frames, paths.contact_sheets, paths.figures]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_transform(deps: RuntimeDeps):
    return deps.transforms.Compose(
        [
            deps.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            deps.transforms.ToTensor(),
            deps.transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def build_model(deps: RuntimeDeps):
    model = deps.models.resnet18(weights=None)
    model.fc = deps.nn.Linear(model.fc.in_features, 2)
    return model


def load_checkpoint_model(deps: RuntimeDeps, checkpoint_path: Path):
    checkpoint_path = normalize_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise Stage14Error(f"Checkpoint not found: {checkpoint_path}")
    payload = deps.torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "model_state_dict"}
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    elif isinstance(payload, dict):
        state_dict = payload
        metadata = {}
    else:
        raise Stage14Error(f"Unsupported checkpoint payload type: {type(payload)}")

    clean_state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    fc_weight = clean_state_dict.get("fc.weight")
    fc_bias = clean_state_dict.get("fc.bias")
    if fc_weight is None or tuple(fc_weight.shape) != (2, 512):
        raise Stage14Error(f"Invalid fc.weight shape: {None if fc_weight is None else tuple(fc_weight.shape)}")
    if fc_bias is None or tuple(fc_bias.shape) != (2,):
        raise Stage14Error(f"Invalid fc.bias shape: {None if fc_bias is None else tuple(fc_bias.shape)}")

    class_to_index = metadata.get("class_to_index")
    if class_to_index and class_to_index != {"no_yawn": 0, "yawn": 1}:
        raise Stage14Error(f"Unexpected class_to_index metadata: {class_to_index}")
    image_size = metadata.get("image_size")
    if image_size and int(image_size) != IMAGE_SIZE:
        raise Stage14Error(f"Unexpected image_size metadata: {image_size}")

    model = build_model(deps)
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    return model, metadata


def build_landmarker(deps: RuntimeDeps, face_landmarker_path: Path):
    face_landmarker_path = normalize_path(face_landmarker_path)
    if not face_landmarker_path.is_file():
        raise Stage14Error(f"FaceLandmarker asset not found: {face_landmarker_path}")
    options = deps.FaceLandmarkerOptions(
        base_options=deps.BaseOptions(model_asset_path=str(face_landmarker_path)),
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


def mouth_bbox_from_landmarks(
    landmarks: Any,
    img_w: int,
    img_h: int,
    mouth_margin: float,
    min_margin_px: int,
) -> tuple[int, int, int, int]:
    xs: list[float] = []
    ys: list[float] = []
    for idx in MOUTH_LANDMARK_IDS:
        lm = landmarks[idx]
        xs.append(lm.x * img_w)
        ys.append(lm.y * img_h)
    raw_x1, raw_x2 = min(xs), max(xs)
    raw_y1, raw_y2 = min(ys), max(ys)
    bbox_w = raw_x2 - raw_x1
    bbox_h = raw_y2 - raw_y1
    x_margin = max(float(min_margin_px), mouth_margin * bbox_w)
    y_margin = max(float(min_margin_px), mouth_margin * bbox_h)
    x1 = int(round(raw_x1 - x_margin))
    y1 = int(round(raw_y1 - y_margin))
    x2 = int(round(raw_x2 + x_margin))
    y2 = int(round(raw_y2 + y_margin))
    return clamp_bbox(x1, y1, x2, y2, img_w, img_h)


def detect_mouth_bbox(
    deps: RuntimeDeps,
    landmarker: Any,
    frame_bgr: Any,
    mouth_margin: float,
    min_margin_px: int,
) -> tuple[tuple[int, int, int, int] | None, int, str | None]:
    frame_rgb = deps.cv2.cvtColor(frame_bgr, deps.cv2.COLOR_BGR2RGB)
    mp_image = deps.mp.Image(image_format=deps.mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    faces = result.face_landmarks or []
    if not faces:
        return None, 0, "no face detected"
    height, width = frame_bgr.shape[:2]
    bbox = mouth_bbox_from_landmarks(faces[0], width, height, mouth_margin, min_margin_px)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None, len(faces), "invalid mouth crop bbox"
    return bbox, len(faces), None


def infer_crop(deps: RuntimeDeps, model: Any, transform: Any, crop_bgr: Any) -> tuple[float, float, str]:
    crop_rgb = deps.cv2.cvtColor(crop_bgr, deps.cv2.COLOR_BGR2RGB)
    image = deps.Image.fromarray(crop_rgb).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with deps.torch.no_grad():
        logits = model(tensor)
        probs = deps.torch.softmax(logits, dim=1)[0].cpu()
    p_no_yawn = float(probs[0].item())
    p_yawn = float(probs[P_YAWN_CLASS_INDEX].item())
    predicted_label = "yawn" if p_yawn >= p_no_yawn else "no_yawn"
    return p_yawn, p_no_yawn, predicted_label


def add_recent_yawn_flags(rows: list[dict[str, Any]], recent_window_sec: float) -> None:
    last_event_time: float | None = None
    for row in rows:
        timestamp = float(row["timestamp_sec"])
        yawn_event = str(row["yawn_event"]).lower() in {"true", "1"}
        if yawn_event:
            last_event_time = timestamp
        row["recent_yawn_event"] = bool(
            last_event_time is not None and (timestamp - last_event_time) <= recent_window_sec
        )


def draw_debug_frame(deps: RuntimeDeps, frame_bgr: Any, bbox: tuple[int, int, int, int], label: str) -> Any:
    debug = frame_bgr.copy()
    x1, y1, x2, y2 = bbox
    deps.cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
    deps.cv2.putText(
        debug,
        label,
        (x1, max(20, y1 - 8)),
        deps.cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        deps.cv2.LINE_AA,
    )
    return debug


def make_contact_sheet(
    deps: RuntimeDeps,
    samples: list[CropSample],
    output_path: Path,
    title: str,
    max_samples: int = 32,
    thumb_size: tuple[int, int] = (160, 120),
) -> bool:
    if not samples:
        return False
    selected = samples[:max_samples]
    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)
    sheet_w = cols * thumb_size[0]
    sheet_h = rows * (thumb_size[1] + 32) + 28
    sheet = deps.Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = deps.ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, sample in enumerate(selected):
        col = idx % cols
        row = idx // cols
        x = col * thumb_size[0]
        y = 28 + row * (thumb_size[1] + 32)
        thumb = sample.image.copy().resize(thumb_size)
        sheet.paste(thumb, (x, y))
        label = f"f{sample.frame_index} t={sample.timestamp_sec:.1f} p={sample.p_yawn:.2f}"
        draw.text((x + 4, y + thumb_size[1] + 4), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return True


def write_figures(output_paths: OutputPaths, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_paths: list[str] = []
    if not rows:
        return figure_paths
    times = [float(row["timestamp_sec"]) for row in rows]
    p_yawns = [float(row["p_yawn"]) for row in rows]
    events = [1 if str(row["yawn_event"]).lower() in {"true", "1"} else 0 for row in rows]

    p_path = output_paths.figures / "p_yawn_over_time.png"
    plt.figure(figsize=(10, 4))
    plt.plot(times, p_yawns, marker="o", markersize=2, linewidth=1)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="threshold 0.50")
    plt.xlabel("Time (sec)")
    plt.ylabel("p_yawn")
    plt.title("Stage 14 p_yawn Over Time")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p_path, dpi=160)
    plt.close()
    figure_paths.append(str(p_path))

    e_path = output_paths.figures / "yawn_event_timeline.png"
    plt.figure(figsize=(10, 3))
    plt.step(times, events, where="post")
    plt.xlabel("Time (sec)")
    plt.ylabel("yawn_event")
    plt.yticks([0, 1])
    plt.title("Stage 14 Yawn Event Timeline")
    plt.tight_layout()
    plt.savefig(e_path, dpi=160)
    plt.close()
    figure_paths.append(str(e_path))
    return figure_paths


def write_report(
    output_paths: OutputPaths,
    summary: dict[str, Any],
    contact_sheet_notes: list[str],
) -> None:
    lines = [
        "# Stage 14 Runtime Mouth/Yawn Report",
        "",
        "## Purpose",
        "",
        "Runtime mouth/yawn inference and mouth ROI consistency validation. This is not final drowsiness accuracy and not mouth-eye fusion.",
        "",
        "## Model",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        "- Architecture: torchvision ResNet18 with a two-class `fc` head.",
        "- Label mapping: `0 = no_yawn`, `1 = yawn`.",
        "- `p_yawn = softmax(logits)[1]`.",
        "",
        "## Runtime Summary",
        "",
        f"- Video: `{summary['video_path']}`",
        f"- Sampled frames: {summary['sampled_frame_count']}",
        f"- Successful mouth crops: {summary['successful_mouth_crop_count']}",
        f"- Failures: {summary['failure_count']}",
        f"- Yawn events: {summary['yawn_event_count']}",
        f"- Recent-yawn rows: {summary['recent_yawn_event_count']}",
        f"- Mean p_yawn: {summary.get('mean_p_yawn')}",
        f"- Min p_yawn: {summary.get('min_p_yawn')}",
        f"- Max p_yawn: {summary.get('max_p_yawn')}",
        "",
        "## Contact Sheets",
        "",
        *[f"- {note}" for note in contact_sheet_notes],
        "",
        "## Visual Inspection Requirement",
        "",
        "A human must inspect contact sheets and debug frames. Mouth crops must show the mouth region, and high `p_yawn` crops should visually correspond to yawning or mouth-open/yawn-like frames.",
        "",
        "## Limitations",
        "",
        "- Small controlled validation set.",
        "- No final drowsiness labels.",
        "- Not mouth-eye fusion yet.",
        "- Runtime mouth ROI may fail under occlusion, head pose, or poor lighting.",
        "- The mouth/yawn specialist was trained on YawDD/YawDD+ Dash mouth crops, not necessarily these runtime videos.",
        f"- {WARNING}",
    ]
    output_paths.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight(args: argparse.Namespace) -> None:
    deps = load_runtime_deps()
    checkpoint = normalize_path(args.checkpoint)
    face_landmarker = normalize_path(args.face_landmarker)
    if not face_landmarker.is_file():
        raise Stage14Error(f"FaceLandmarker asset missing: {face_landmarker}")
    if not checkpoint.is_file():
        raise Stage14Error(f"Checkpoint missing: {checkpoint}")
    transform = build_transform(deps)
    model, metadata = load_checkpoint_model(deps, checkpoint)
    dummy_image = deps.Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "black")
    dummy = transform(dummy_image).unsqueeze(0)
    with deps.torch.no_grad():
        logits = model(dummy)
        probs = deps.torch.softmax(logits, dim=1)
    if tuple(logits.shape) != (1, 2) or tuple(probs.shape) != (1, 2):
        raise Stage14Error(f"Unexpected dummy output shape logits={tuple(logits.shape)} probs={tuple(probs.shape)}")
    print("STAGE14_PREFLIGHT_PASSED")
    print(f"checkpoint={checkpoint}")
    print(f"face_landmarker={face_landmarker}")
    print(f"model_name={MODEL_NAME}")
    print(f"label_mapping={LABEL_MAPPING}")
    print(f"p_yawn_class_index={P_YAWN_CLASS_INDEX}")
    print(f"checkpoint_metadata_keys={sorted(metadata.keys())}")


def process_video(args: argparse.Namespace) -> dict[str, Any]:
    deps = load_runtime_deps()
    checkpoint = normalize_path(args.checkpoint)
    face_landmarker_path = normalize_path(args.face_landmarker)
    input_video = normalize_path(args.input_video)
    if not input_video.is_file():
        raise Stage14Error(f"Input video not found: {input_video}")
    output_paths = ensure_output_paths(args.output_dir)

    transform = build_transform(deps)
    model, metadata = load_checkpoint_model(deps, checkpoint)
    landmarker = build_landmarker(deps, face_landmarker_path)

    cap = deps.cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise Stage14Error(f"Could not open video: {input_video}")
    fps = float(cap.get(deps.cv2.CAP_PROP_FPS) or 0.0)
    video_slug = input_video.stem

    prediction_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    samples: list[CropSample] = []
    frame_index = -1
    sampled_count = 0

    try:
        while sampled_count < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % args.sample_every_n_frames != 0:
                continue
            timestamp_sec = frame_index / fps if fps > 0 else float(sampled_count)
            sampled_count += 1
            try:
                bbox, num_faces, reason = detect_mouth_bbox(
                    deps,
                    landmarker,
                    frame,
                    args.mouth_margin,
                    args.min_mouth_margin_px,
                )
                if bbox is None:
                    failure_type = "no_face" if num_faces == 0 else "invalid_mouth_crop"
                    failure_rows.append(
                        {
                            "video_path": str(input_video),
                            "video_slug": video_slug,
                            "frame_index": frame_index,
                            "timestamp_sec": f"{timestamp_sec:.6f}",
                            "failure_type": failure_type,
                            "failure_reason": reason or failure_type,
                        }
                    )
                    continue
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    failure_rows.append(
                        {
                            "video_path": str(input_video),
                            "video_slug": video_slug,
                            "frame_index": frame_index,
                            "timestamp_sec": f"{timestamp_sec:.6f}",
                            "failure_type": "invalid_mouth_crop",
                            "failure_reason": "empty crop after bbox clamp",
                        }
                    )
                    continue
                p_yawn, p_no_yawn, predicted_label = infer_crop(deps, model, transform, crop)
                yawn_event = p_yawn >= args.yawn_threshold

                crop_path = ""
                if args.save_crops:
                    crop_file = output_paths.crops / f"{video_slug}_frame_{frame_index:06d}_mouth.jpg"
                    deps.cv2.imwrite(str(crop_file), crop)
                    crop_path = str(crop_file)

                debug_path = ""
                if args.save_debug_frames:
                    label = f"p_yawn={p_yawn:.3f} {predicted_label}"
                    debug = draw_debug_frame(deps, frame, bbox, label)
                    debug_file = output_paths.debug_frames / f"{video_slug}_frame_{frame_index:06d}_debug.jpg"
                    deps.cv2.imwrite(str(debug_file), debug)
                    debug_path = str(debug_file)

                crop_rgb = deps.cv2.cvtColor(crop, deps.cv2.COLOR_BGR2RGB)
                samples.append(
                    CropSample(
                        image=deps.Image.fromarray(crop_rgb).convert("RGB"),
                        p_yawn=p_yawn,
                        predicted_label=predicted_label,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                    )
                )
                prediction_rows.append(
                    {
                        "video_path": str(input_video),
                        "video_slug": video_slug,
                        "frame_index": frame_index,
                        "timestamp_sec": f"{timestamp_sec:.6f}",
                        "mouth_bbox_x1": x1,
                        "mouth_bbox_y1": y1,
                        "mouth_bbox_x2": x2,
                        "mouth_bbox_y2": y2,
                        "p_yawn": f"{p_yawn:.8f}",
                        "p_no_yawn": f"{p_no_yawn:.8f}",
                        "predicted_label": predicted_label,
                        "yawn_event": bool(yawn_event),
                        "recent_yawn_event": False,
                        "mouth_signal_status": "ok",
                        "checkpoint_path": str(checkpoint),
                        "model_name": MODEL_NAME,
                        "label_mapping": json.dumps(LABEL_MAPPING, sort_keys=True),
                        "mouth_crop_path": crop_path,
                        "debug_frame_path": debug_path,
                    }
                )
            except Exception as exc:
                failure_rows.append(
                    {
                        "video_path": str(input_video),
                        "video_slug": video_slug,
                        "frame_index": frame_index,
                        "timestamp_sec": f"{timestamp_sec:.6f}",
                        "failure_type": "inference_failed",
                        "failure_reason": repr(exc),
                    }
                )
    finally:
        cap.release()
        if hasattr(landmarker, "close"):
            landmarker.close()

    add_recent_yawn_flags(prediction_rows, args.recent_yawn_window_sec)
    write_csv(output_paths.predictions_csv, prediction_rows, PREDICTION_FIELDS)
    write_csv(output_paths.failures_csv, failure_rows, FAILURE_FIELDS)

    sorted_by_high = sorted(samples, key=lambda s: s.p_yawn, reverse=True)
    sorted_by_low = sorted(samples, key=lambda s: s.p_yawn)
    event_samples = [s for s in sorted_by_high if s.p_yawn >= args.yawn_threshold]
    contact_sheet_notes: list[str] = []
    for name, sample_set, title in [
        ("mouth_samples.jpg", samples, "Mouth Samples"),
        ("high_p_yawn.jpg", sorted_by_high, "High p_yawn Samples"),
        ("low_p_yawn.jpg", sorted_by_low, "Low p_yawn Samples"),
        ("yawn_event_samples.jpg", event_samples, "Yawn Event Samples"),
    ]:
        ok = make_contact_sheet(deps, sample_set, output_paths.contact_sheets / name, title)
        if ok:
            contact_sheet_notes.append(f"`contact_sheets/{name}` generated")
        else:
            contact_sheet_notes.append(f"`contact_sheets/{name}` not generated because no samples were available")

    figure_paths = write_figures(output_paths, prediction_rows)
    p_yawns = [float(row["p_yawn"]) for row in prediction_rows]
    status_counts = {"ok": len(prediction_rows)}
    no_face_count = sum(1 for row in failure_rows if row["failure_type"] == "no_face")
    invalid_count = sum(1 for row in failure_rows if row["failure_type"] == "invalid_mouth_crop")
    inference_failed_count = sum(1 for row in failure_rows if row["failure_type"] == "inference_failed")

    summary = {
        "stage": 14,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "video_path": str(input_video),
        "video_slug": video_slug,
        "checkpoint_path": str(checkpoint),
        "model_name": MODEL_NAME,
        "label_mapping": LABEL_MAPPING,
        "p_yawn_class_index": P_YAWN_CLASS_INDEX,
        "checkpoint_metadata": metadata,
        "sample_every_n_frames": args.sample_every_n_frames,
        "max_frames": args.max_frames,
        "sampled_frame_count": sampled_count,
        "successful_mouth_crop_count": len(prediction_rows),
        "failure_count": len(failure_rows),
        "no_face_count": no_face_count,
        "invalid_mouth_crop_count": invalid_count,
        "inference_failed_count": inference_failed_count,
        "mean_p_yawn": mean(p_yawns) if p_yawns else None,
        "min_p_yawn": min(p_yawns) if p_yawns else None,
        "max_p_yawn": max(p_yawns) if p_yawns else None,
        "yawn_event_count": sum(1 for row in prediction_rows if str(row["yawn_event"]).lower() in {"true", "1"}),
        "recent_yawn_event_count": sum(1 for row in prediction_rows if str(row["recent_yawn_event"]).lower() in {"true", "1"}),
        "mouth_signal_status_counts": status_counts,
        "output_dir": str(output_paths.root),
        "prediction_csv": str(output_paths.predictions_csv),
        "failure_csv": str(output_paths.failures_csv),
        "figure_paths": figure_paths,
        "limitations": [
            "Not final system-level drowsiness accuracy.",
            "Not mouth-eye fusion yet.",
            "Small controlled validation set.",
            "Runtime mouth ROI may fail under occlusion, head pose, or low light.",
        ],
        "warning": WARNING,
    }
    write_json(output_paths.summary_json, summary)
    write_report(output_paths, summary, contact_sheet_notes)
    return summary


def main() -> int:
    args = parse_args()
    if args.sample_every_n_frames <= 0:
        raise Stage14Error("--sample-every-n-frames must be positive")
    if args.max_frames <= 0:
        raise Stage14Error("--max-frames must be positive")
    if args.preflight:
        run_preflight(args)
        return 0
    if args.input_video is None:
        raise Stage14Error("--input-video is required unless --preflight is set")
    summary = process_video(args)
    print(f"STAGE14_RUNTIME_COMPLETED video={summary['video_slug']}")
    print(f"summary={summary['output_dir']}/summary.json")
    print(f"successful_mouth_crop_count={summary['successful_mouth_crop_count']}")
    print(f"failure_count={summary['failure_count']}")
    print(f"yawn_event_count={summary['yawn_event_count']}")
    print(WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
