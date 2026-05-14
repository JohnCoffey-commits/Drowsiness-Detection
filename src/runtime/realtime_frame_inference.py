"""Realtime single-frame webcam evidence inference.

This module is intentionally frame-level only. It reuses the specialist eye and
mouth/yawn models to produce per-frame evidence for the Live Monitor prototype;
it does not compute temporal warning state, alarms, or final drowsiness truth.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EYE_CHECKPOINT = PROJECT_ROOT / "outputs" / "mrl_eye" / "checkpoints" / "best_mobilenet_v2_mrl_eye.pt"
MOUTH_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "resnet18_best.pt"
FACE_LANDMARKER = PROJECT_ROOT / "artifacts" / "models" / "face_landmarker.task"
SAFE_INTERPRETATION = (
    "Frame-level evidence for realtime warning-candidate state. This is not final "
    "system-level drowsiness accuracy."
)
PERMANENT_WARNING = (
    "This output is a realtime rule-based warning-candidate analysis, "
    "not final system-level drowsiness accuracy."
)


class RealtimeFrameInferenceError(RuntimeError):
    """Raised for controlled realtime inference failures."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bbox_xywh(bbox: tuple[int, int, int, int] | None) -> list[int] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def select_realtime_device(torch: Any) -> Any:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def inference_mode(torch: Any):
    mode = getattr(torch, "inference_mode", None)
    if mode is not None:
        with mode():
            yield
    else:
        with torch.no_grad():
            yield


def classify_eye_evidence(mean_p_eye_closed: float | None) -> str:
    if mean_p_eye_closed is None:
        return "unavailable"
    if mean_p_eye_closed >= 0.85:
        return "strong_eye_closure_candidate"
    if mean_p_eye_closed >= 0.70:
        return "moderate_eye_closure_candidate"
    if mean_p_eye_closed >= 0.50:
        return "weak_reduced_eye_openness_candidate"
    return "normal_open"


class RealtimeFrameInferenceService:
    """Lazy-loaded single-frame inference service for webcam sampling."""

    def __init__(self) -> None:
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._deps: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._landmarker: Any | None = None
        self._eye_model: Any | None = None
        self._eye_transform: Any | None = None
        self._mouth_model: Any | None = None
        self._mouth_transform: Any | None = None

    @property
    def models_loaded(self) -> bool:
        return self._loaded

    def device_name(self) -> str:
        if self._device is not None:
            return str(self._device)
        try:
            import torch

            return str(select_realtime_device(torch))
        except Exception:
            return "unavailable"

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "realtime_frame_inference",
            "eye_checkpoint_found": EYE_CHECKPOINT.is_file(),
            "mouth_checkpoint_found": MOUTH_CHECKPOINT.is_file(),
            "models_loaded": self._loaded,
            "device": self.device_name(),
            "note": (
                "Realtime frame inference health only. Models are lazy-loaded on the first "
                "frame request; no expensive inference is run by this endpoint."
            ),
        }

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return
            try:
                from src.runtime import stage10_eye_roi_consistency as eye_runtime
                from src.runtime import stage14_mouth_yawn_runtime as mouth_runtime
            except Exception as exc:  # pragma: no cover - environment dependent.
                self._load_error = f"Could not import runtime modules: {exc}"
                raise RealtimeFrameInferenceError(self._load_error) from exc

            try:
                deps = eye_runtime.load_runtime_deps()
                device = select_realtime_device(deps.torch)
                eye_transform = eye_runtime.build_eval_transform(deps, image_size=224)
                eye_model, _, _ = eye_runtime.load_model_and_metadata(
                    deps,
                    EYE_CHECKPOINT,
                    device,
                    image_size=224,
                )
                mouth_transform = mouth_runtime.build_transform(deps)
                mouth_model, _ = mouth_runtime.load_checkpoint_model(deps, MOUTH_CHECKPOINT)
                mouth_model.to(device)
                mouth_model.eval()
                landmarker = eye_runtime.build_landmarker(deps, FACE_LANDMARKER)
            except Exception as exc:
                self._load_error = str(exc)
                raise RealtimeFrameInferenceError(self._load_error) from exc

            self._deps = deps
            self._torch = deps.torch
            self._device = device
            self._eye_model = eye_model
            self._eye_transform = eye_transform
            self._mouth_model = mouth_model
            self._mouth_transform = mouth_transform
            self._landmarker = landmarker
            self._loaded = True
            self._load_error = None

    def decode_jpeg(self, frame_bytes: bytes) -> Any:
        self.ensure_loaded()
        deps = self._deps
        if deps is None:
            raise RealtimeFrameInferenceError("Runtime dependencies are unavailable.")

        array = deps.np.frombuffer(frame_bytes, dtype=deps.np.uint8)
        frame_bgr = deps.cv2.imdecode(array, deps.cv2.IMREAD_COLOR)
        if frame_bgr is None or frame_bgr.size == 0:
            raise RealtimeFrameInferenceError("Could not decode JPEG frame.")
        return frame_bgr

    def detect_face_landmarks(self, frame_bgr: Any) -> tuple[Any | None, int]:
        deps = self._deps
        landmarker = self._landmarker
        if deps is None or landmarker is None:
            raise RealtimeFrameInferenceError("Face landmarker is unavailable.")

        frame_rgb = deps.cv2.cvtColor(frame_bgr, deps.cv2.COLOR_BGR2RGB)
        mp_image = deps.mp.Image(image_format=deps.mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)
        faces = result.face_landmarks or [] if result is not None else []
        if not faces:
            return None, 0
        return faces[0], len(faces)

    def infer_eye_crop(self, crop_bgr: Any) -> float:
        deps = self._deps
        torch = self._torch
        model = self._eye_model
        transform = self._eye_transform
        device = self._device
        if deps is None or torch is None or model is None or transform is None or device is None:
            raise RealtimeFrameInferenceError("Eye model is unavailable.")

        crop_rgb = deps.cv2.cvtColor(crop_bgr, deps.cv2.COLOR_BGR2RGB)
        image = deps.Image.fromarray(crop_rgb).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with inference_mode(torch):
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        return float(probs[0].item())

    def infer_mouth_crop(self, crop_bgr: Any) -> float:
        deps = self._deps
        torch = self._torch
        model = self._mouth_model
        transform = self._mouth_transform
        device = self._device
        if deps is None or torch is None or model is None or transform is None or device is None:
            raise RealtimeFrameInferenceError("Mouth/yawn model is unavailable.")

        crop_rgb = deps.cv2.cvtColor(crop_bgr, deps.cv2.COLOR_BGR2RGB)
        image = deps.Image.fromarray(crop_rgb).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with inference_mode(torch):
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        return float(probs[1].item())

    def analyze_frame(
        self,
        *,
        session_id: str,
        frame_bytes: bytes,
        client_timestamp_ms: float | None,
        frame_width: int | None,
        frame_height: int | None,
        sampling_fps: float | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        server_received_at = now_iso()
        frame_id = f"frame_{uuid.uuid4().hex[:12]}"

        try:
            frame_bgr = self.decode_jpeg(frame_bytes)
        except RealtimeFrameInferenceError as exc:
            return self._failure_response(
                session_id=session_id,
                frame_id=frame_id,
                server_received_at=server_received_at,
                client_timestamp_ms=client_timestamp_ms,
                width=frame_width,
                height=frame_height,
                sampling_fps=sampling_fps,
                status="decode_failure",
                reason=str(exc),
                started=started,
            )

        height, width = frame_bgr.shape[:2]
        input_width = frame_width or int(width)
        input_height = frame_height or int(height)

        with self._inference_lock:
            return self._analyze_decoded_frame(
                session_id=session_id,
                frame_id=frame_id,
                server_received_at=server_received_at,
                client_timestamp_ms=client_timestamp_ms,
                input_width=input_width,
                input_height=input_height,
                sampling_fps=sampling_fps,
                frame_bgr=frame_bgr,
                started=started,
            )

    def _analyze_decoded_frame(
        self,
        *,
        session_id: str,
        frame_id: str,
        server_received_at: str,
        client_timestamp_ms: float | None,
        input_width: int,
        input_height: int,
        sampling_fps: float | None,
        frame_bgr: Any,
        started: float,
    ) -> dict[str, Any]:
        from src.runtime import stage10_eye_roi_consistency as eye_runtime
        from src.runtime import stage14_mouth_yawn_runtime as mouth_runtime

        try:
            landmarks, num_faces = self.detect_face_landmarks(frame_bgr)
        except Exception as exc:
            return self._failure_response(
                session_id=session_id,
                frame_id=frame_id,
                server_received_at=server_received_at,
                client_timestamp_ms=client_timestamp_ms,
                width=input_width,
                height=input_height,
                sampling_fps=sampling_fps,
                status="model_inference_failure",
                reason=f"Face landmark inference failed: {exc}",
                started=started,
            )

        if landmarks is None:
            return {
                **self._base_response(
                    ok=True,
                    session_id=session_id,
                    frame_id=frame_id,
                    server_received_at=server_received_at,
                    client_timestamp_ms=client_timestamp_ms,
                    width=input_width,
                    height=input_height,
                    sampling_fps=sampling_fps,
                    started=started,
                ),
                "face": {"detected": False, "tracking_status": "no_face", "num_faces": 0},
                "eye": self._empty_eye("no face detected"),
                "mouth": self._empty_mouth("no face detected"),
                "signal_quality": {
                    "status": "no_face",
                    "reason": "MediaPipe FaceLandmarker returned no face landmarks for this frame.",
                },
                "safe_interpretation": SAFE_INTERPRETATION,
                "warning": PERMANENT_WARNING,
            }

        height, width = frame_bgr.shape[:2]
        eye_result = self._infer_eyes(eye_runtime, landmarks, frame_bgr, int(width), int(height))
        mouth_result = self._infer_mouth(
            mouth_runtime,
            landmarks,
            frame_bgr,
            int(width),
            int(height),
        )

        signal_status, signal_reason = self._signal_quality(eye_result, mouth_result)
        return {
            **self._base_response(
                ok=True,
                session_id=session_id,
                frame_id=frame_id,
                server_received_at=server_received_at,
                client_timestamp_ms=client_timestamp_ms,
                width=input_width,
                height=input_height,
                sampling_fps=sampling_fps,
                started=started,
            ),
            "face": {
                "detected": True,
                "tracking_status": "ok",
                "num_faces": int(num_faces),
            },
            "eye": eye_result,
            "mouth": mouth_result,
            "signal_quality": {
                "status": signal_status,
                "reason": signal_reason,
            },
            "safe_interpretation": SAFE_INTERPRETATION,
            "warning": PERMANENT_WARNING,
        }

    def _infer_eyes(
        self,
        eye_runtime: Any,
        landmarks: Any,
        frame_bgr: Any,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        values: dict[str, float | None] = {"left": None, "right": None}
        boxes: dict[str, list[int] | None] = {"left": None, "right": None}
        errors: dict[str, str] = {}

        for side, landmark_ids in eye_runtime.EYE_LANDMARKS.items():
            try:
                bbox = eye_runtime.eye_bbox_from_landmarks(
                    landmarks,
                    landmark_ids,
                    width,
                    height,
                    margin=0.60,
                )
                x1, y1, x2, y2 = bbox
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    raise ValueError("empty eye crop after bbox clamp")
                values[side] = self.infer_eye_crop(crop)
                boxes[side] = bbox_xywh(bbox)
            except Exception as exc:
                errors[side] = str(exc)

        valid_values = [value for value in values.values() if value is not None]
        mean_p_eye_closed = (
            float(sum(valid_values) / len(valid_values)) if valid_values else None
        )
        available = bool(valid_values)
        reason = None
        if not available:
            reason = "; ".join(f"{side}: {error}" for side, error in errors.items()) or "eye ROI unavailable"

        return {
            "available": available,
            "left_p_eye_closed": values["left"],
            "right_p_eye_closed": values["right"],
            "mean_p_eye_closed": mean_p_eye_closed,
            "evidence_strength": classify_eye_evidence(mean_p_eye_closed),
            "left_roi_box": boxes["left"],
            "right_roi_box": boxes["right"],
            "status": "ok" if available else "eye_roi_unavailable",
            "reason": reason,
        }

    def _infer_mouth(
        self,
        mouth_runtime: Any,
        landmarks: Any,
        frame_bgr: Any,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        try:
            bbox = mouth_runtime.mouth_bbox_from_landmarks(
                landmarks,
                width,
                height,
                mouth_margin=0.10,
                min_margin_px=10,
            )
            x1, y1, x2, y2 = bbox
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError("empty mouth crop after bbox clamp")
            p_yawn = self.infer_mouth_crop(crop)
            return {
                "available": True,
                "p_yawn": p_yawn,
                "mouth_roi_box": bbox_xywh(bbox),
                "status": "ok",
                "reason": None,
            }
        except Exception as exc:
            return {
                "available": False,
                "p_yawn": None,
                "mouth_roi_box": None,
                "status": "mouth_roi_unavailable",
                "reason": str(exc),
            }

    def _signal_quality(self, eye_result: dict[str, Any], mouth_result: dict[str, Any]) -> tuple[str, str]:
        if eye_result.get("available") and mouth_result.get("available"):
            return "ok", "Face, eye ROI, and mouth ROI available for this frame."
        if eye_result.get("available") or mouth_result.get("available"):
            return "partial", "Face detected, but one specialist ROI was unavailable for this frame."
        return "roi_unavailable", "Face detected, but eye and mouth ROI evidence was unavailable."

    def _base_response(
        self,
        *,
        ok: bool,
        session_id: str,
        frame_id: str,
        server_received_at: str,
        client_timestamp_ms: float | None,
        width: int | None,
        height: int | None,
        sampling_fps: float | None,
        started: float,
    ) -> dict[str, Any]:
        return {
            "ok": ok,
            "session_id": session_id,
            "frame_id": frame_id,
            "server_received_at": server_received_at,
            "client_timestamp_ms": client_timestamp_ms,
            "input": {
                "width": width,
                "height": height,
                "sampling_fps": sampling_fps,
            },
            "device": self.device_name(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _failure_response(
        self,
        *,
        session_id: str,
        frame_id: str,
        server_received_at: str,
        client_timestamp_ms: float | None,
        width: int | None,
        height: int | None,
        sampling_fps: float | None,
        status: str,
        reason: str,
        started: float,
    ) -> dict[str, Any]:
        return {
            **self._base_response(
                ok=False,
                session_id=session_id,
                frame_id=frame_id,
                server_received_at=server_received_at,
                client_timestamp_ms=client_timestamp_ms,
                width=width,
                height=height,
                sampling_fps=sampling_fps,
                started=started,
            ),
            "face": {"detected": False, "tracking_status": status, "num_faces": 0},
            "eye": self._empty_eye(reason),
            "mouth": self._empty_mouth(reason),
            "signal_quality": {"status": status, "reason": reason},
            "safe_interpretation": SAFE_INTERPRETATION,
            "warning": PERMANENT_WARNING,
        }

    def _empty_eye(self, reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "left_p_eye_closed": None,
            "right_p_eye_closed": None,
            "mean_p_eye_closed": None,
            "evidence_strength": "unavailable",
            "left_roi_box": None,
            "right_roi_box": None,
            "status": "eye_roi_unavailable",
            "reason": reason,
        }

    def _empty_mouth(self, reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "p_yawn": None,
            "mouth_roi_box": None,
            "status": "mouth_roi_unavailable",
            "reason": reason,
        }


_SERVICE = RealtimeFrameInferenceService()


def get_realtime_service() -> RealtimeFrameInferenceService:
    return _SERVICE
