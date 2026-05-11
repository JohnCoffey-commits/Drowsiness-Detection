#!/usr/bin/env python3
"""Stage 17 single-video upload analysis pipeline.

Runs the existing eye branch, mouth branch, and rule-based fusion for one
uploaded video. Outputs are warning-candidate analysis artifacts, not final
system-level drowsiness accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.keyframe_extractor import extract_keyframes  # noqa: E402
from src.runtime.stage13_mouth_eye_fusion_design import (  # noqa: E402
    align_real_mouth_timeline,
    build_fusion_timeline,
)


WARNING = (
    "This output is a rule-based drowsiness warning-candidate analysis, "
    "not final system-level drowsiness accuracy."
)
EYE_RULE_NAME = "quality_gated_perclos_mean_ge_0.60_consec"
FUSION_RULE_NAME = "F5_tiered_quality_aware_fusion"
MOUTH_SOURCE = "stage14_runtime_mouth_yawn_model"
SUSTAINED_EYE_GATE_MIN_DURATION_SEC = 1.0
SUSTAINED_EYE_GATE_MIN_SAMPLED_FRAMES = 5
EYE_EVIDENCE_CALIBRATION_VERSION = "Stage 17.5 provisional rule-based calibration"
EYE_EVIDENCE_WEAK_MIN = 0.50
EYE_EVIDENCE_MODERATE_MIN = 0.70
EYE_EVIDENCE_STRONG_MIN = 0.85
EYE_STRENGTH_GATE_MIN_MEAN_P_EYE_CLOSED = 0.70
EYE_STRENGTH_GATE_MIN_MAX_P_EYE_CLOSED = 0.85
EYE_STRENGTH_GATE_MIN_STRONG_FRAMES = 1
EYE_STRENGTH_GATE_MIN_MODERATE_OR_STRONG_FRAMES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--sample-every-n-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--yawn-threshold", type=float, default=0.50)
    parser.add_argument("--recent-yawn-window-sec", type=float, default=8.0)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--save-keyframes", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) or math.isinf(parsed) else parsed


def classify_eye_evidence(
    p_eye_closed: float | None,
    *,
    eye_warning_candidate: bool = False,
    signal_unreliable: bool = False,
) -> dict[str, object]:
    """Classify eye evidence strength without changing the eye model output.

    Stage 17.5 is a rule-based interpretation calibration layer. It keeps the
    existing p_eye_closed formula and base eye-warning rule intact, but adds
    safer wording and a strength signal for high-confidence escalation.
    """

    probability = safe_float(p_eye_closed)
    if signal_unreliable:
        return {
            "eye_evidence_strength": "signal_unreliable",
            "eye_evidence_strength_rank": -1,
            "eye_evidence_label": "Signal unreliable",
            "eye_evidence_interpretation": (
                "Face/eye ROI quality may be unreliable; this is not eye-warning evidence."
            ),
            "eye_evidence_calibration_version": EYE_EVIDENCE_CALIBRATION_VERSION,
        }

    if probability is None:
        return {
            "eye_evidence_strength": "none",
            "eye_evidence_strength_rank": 0,
            "eye_evidence_label": "No calibrated eye-warning evidence",
            "eye_evidence_interpretation": (
                "No valid p_eye_closed value was available for this sampled row."
            ),
            "eye_evidence_calibration_version": EYE_EVIDENCE_CALIBRATION_VERSION,
        }

    if probability >= EYE_EVIDENCE_STRONG_MIN:
        return {
            "eye_evidence_strength": "strong",
            "eye_evidence_strength_rank": 3,
            "eye_evidence_label": "Strong eye-closure candidate",
            "eye_evidence_interpretation": (
                "p_eye_closed is high enough for strong eye-closure candidate evidence; "
                "manual review is still recommended."
            ),
            "eye_evidence_calibration_version": EYE_EVIDENCE_CALIBRATION_VERSION,
        }

    if probability >= EYE_EVIDENCE_MODERATE_MIN:
        return {
            "eye_evidence_strength": "moderate",
            "eye_evidence_strength_rank": 2,
            "eye_evidence_label": "Moderate eye-closure candidate",
            "eye_evidence_interpretation": (
                "p_eye_closed is in a moderate range; it may reflect partial closure, "
                "reduced eye openness, blink-like activity, or ROI-sensitive appearance."
            ),
            "eye_evidence_calibration_version": EYE_EVIDENCE_CALIBRATION_VERSION,
        }

    if probability >= EYE_EVIDENCE_WEAK_MIN or eye_warning_candidate:
        return {
            "eye_evidence_strength": "weak",
            "eye_evidence_strength_rank": 1,
            "eye_evidence_label": "Weak eye-warning evidence",
            "eye_evidence_interpretation": (
                "Eye-warning evidence is weak; it may reflect reduced eye openness, "
                "fatigue-like appearance, blink-like activity, or ROI-sensitive cases."
            ),
            "eye_evidence_calibration_version": EYE_EVIDENCE_CALIBRATION_VERSION,
        }

    return {
        "eye_evidence_strength": "none",
        "eye_evidence_strength_rank": 0,
        "eye_evidence_label": "No calibrated eye-warning evidence",
        "eye_evidence_interpretation": (
            "p_eye_closed is below the provisional weak evidence range for this row."
        ),
        "eye_evidence_calibration_version": EYE_EVIDENCE_CALIBRATION_VERSION,
    }


def add_eye_evidence_calibration(df: pd.DataFrame) -> pd.DataFrame:
    calibrated = df.copy()
    rows = [
        classify_eye_evidence(
            safe_float(row.get("p_eye_closed")),
            eye_warning_candidate=to_bool(row.get("eye_warning_candidate")),
            signal_unreliable=to_bool(row.get("signal_unreliable")),
        )
        for _, row in calibrated.iterrows()
    ]
    calibration = pd.DataFrame(rows, index=calibrated.index)
    for column in calibration.columns:
        calibrated[column] = calibration[column]
    calibrated["weak_eye_warning_evidence"] = (
        calibrated["eye_evidence_strength"].astype(str) == "weak"
    )
    calibrated["moderate_eye_closure_candidate"] = (
        calibrated["eye_evidence_strength"].astype(str) == "moderate"
    )
    calibrated["strong_eye_closure_candidate"] = (
        calibrated["eye_evidence_strength"].astype(str) == "strong"
    )
    calibrated["moderate_or_strong_eye_evidence"] = calibrated[
        "eye_evidence_strength"
    ].isin(["moderate", "strong"])
    return calibrated


def eye_strength_gate_reason(
    *,
    mean_p_eye_closed: float,
    max_p_eye_closed: float,
    strong_frames: int,
    moderate_or_strong_frames: int,
) -> str:
    if max_p_eye_closed >= EYE_STRENGTH_GATE_MIN_MAX_P_EYE_CLOSED:
        return "Stage 17.5 strength gate passed: strong eye-closure candidate frame present"
    if strong_frames >= EYE_STRENGTH_GATE_MIN_STRONG_FRAMES:
        return "Stage 17.5 strength gate passed: strong eye evidence count threshold met"
    if mean_p_eye_closed >= EYE_STRENGTH_GATE_MIN_MEAN_P_EYE_CLOSED:
        return "Stage 17.5 strength gate passed: interval mean p_eye_closed is moderate"
    if moderate_or_strong_frames >= EYE_STRENGTH_GATE_MIN_MODERATE_OR_STRONG_FRAMES:
        return "Stage 17.5 strength gate passed: multiple moderate-or-strong eye evidence frames"
    return (
        "Stage 17.5 strength gate not passed: eye evidence remained weak or reduced-eye-openness "
        "candidate evidence"
    )


def resolve_output_dir(session_id: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    return PROJECT_ROOT / "outputs" / "system_video_upload_runs" / session_id


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(result.stdout)
        log.write(f"\n[exit_code] {result.returncode}\n")
        log.write(f"[duration_sec] {time.time() - started:.3f}\n\n")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}. "
            f"See {log_path}"
        )


def sustained_alert(raw_condition: pd.Series, min_consecutive: int) -> pd.Series:
    raw = raw_condition.fillna(False).astype(bool).tolist()
    alert = [False] * len(raw)
    start: int | None = None
    for idx, value in enumerate(raw + [False]):
        if value and start is None:
            start = idx
        if not value and start is not None:
            end = idx
            if end - start >= min_consecutive:
                for mark_idx in range(start, end):
                    alert[mark_idx] = True
            start = None
    return pd.Series(alert, index=raw_condition.index)


def build_eye_alert_timeline(
    session_id: str,
    stage10_dir: Path,
    stage11_dir: Path,
    output_dir: Path,
    recent_quality_window: int = 5,
    max_recent_no_face_ratio: float = 0.20,
    threshold: float = 0.60,
    min_consecutive: int = 2,
) -> pd.DataFrame:
    temporal_path = stage11_dir / "stage11_eye_temporal_summary.csv"
    if not temporal_path.exists():
        raise FileNotFoundError(f"Missing Stage 11 temporal summary: {temporal_path}")

    timeline = pd.read_csv(temporal_path)
    timeline["frame_index"] = pd.to_numeric(timeline["frame_index"], errors="raise").astype(int)
    timeline["timestamp_sec"] = pd.to_numeric(timeline["timestamp_sec"], errors="raise")
    timeline["has_prediction"] = True
    timeline["failure_status"] = ""
    timeline["no_face_binary"] = 0
    timeline["tracking_failure_binary"] = 0

    failure_rows: list[dict[str, Any]] = []
    failures_path = stage10_dir / "failures.csv"
    if failures_path.exists():
        failures = pd.read_csv(failures_path)
        for _, row in failures.iterrows():
            frame_index = row.get("frame_index")
            if pd.isna(frame_index):
                continue
            status = str(row.get("status", "failure"))
            failure_rows.append(
                {
                    "frame_index": int(frame_index),
                    "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
                    "has_prediction": False,
                    "failure_status": status,
                    "no_face_binary": 1 if status == "no_face" else 0,
                    "tracking_failure_binary": 1,
                }
            )

    if failure_rows:
        timeline = pd.concat([timeline, pd.DataFrame(failure_rows)], ignore_index=True, sort=False)

    timeline = timeline.sort_values(["frame_index", "has_prediction"], ascending=[True, False])
    timeline = timeline.drop_duplicates(subset=["frame_index"], keep="first")
    timeline = timeline.sort_values("frame_index").reset_index(drop=True)
    timeline["video_slug"] = session_id

    for col in [
        "left_closed_binary",
        "right_closed_binary",
        "both_eyes_closed_binary",
        "either_eye_closed_binary",
        "mean_closed_binary",
        "no_face_binary",
        "tracking_failure_binary",
    ]:
        if col not in timeline.columns:
            timeline[col] = 0
        timeline[col] = timeline[col].fillna(0).astype(int)

    for col in [
        "mean_p_eye_closed",
        "rolling_mean_p_eye_closed",
        "rolling_max_p_eye_closed",
        "rolling_perclos_mean_binary",
        "rolling_perclos_either_eye",
        "rolling_perclos_both_eyes",
    ]:
        if col not in timeline.columns:
            timeline[col] = pd.NA
        timeline[col] = pd.to_numeric(timeline[col], errors="coerce")

    timeline["recent_no_face_ratio"] = (
        timeline["no_face_binary"].rolling(recent_quality_window, min_periods=1).mean()
    )
    timeline["signal_unreliable"] = (
        (timeline["tracking_failure_binary"] == 1)
        | (timeline["recent_no_face_ratio"] > max_recent_no_face_ratio)
    )
    raw = (timeline["rolling_perclos_mean_binary"] >= threshold) & (
        ~timeline["signal_unreliable"].fillna(False).astype(bool)
    )
    alert = sustained_alert(raw, min_consecutive)
    timeline["recommended_rule_name"] = EYE_RULE_NAME
    timeline["recommended_raw_condition"] = raw.fillna(False).astype(int)
    timeline["recommended_alert"] = alert.fillna(False).astype(int)

    columns = [
        "video_slug",
        "frame_index",
        "timestamp_sec",
        "has_prediction",
        "failure_status",
        "no_face_binary",
        "recent_no_face_ratio",
        "signal_unreliable",
        "mean_p_eye_closed",
        "rolling_mean_p_eye_closed",
        "rolling_perclos_mean_binary",
        "rolling_perclos_both_eyes",
        "mean_closed_binary",
        "both_eyes_closed_binary",
        "either_eye_closed_binary",
        "recommended_rule_name",
        "recommended_raw_condition",
        "recommended_alert",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / f"stage12_video_alert_timeline_{session_id}.csv"
    timeline[columns].to_csv(timeline_path, index=False)
    write_json(
        output_dir / "eye_alert_summary.json",
        {
            "stage": "17_eye_alert_adapter",
            "session_id": session_id,
            "rule_name": EYE_RULE_NAME,
            "threshold": threshold,
            "min_consecutive_windows": min_consecutive,
            "max_recent_no_face_ratio": max_recent_no_face_ratio,
            "recent_quality_window": recent_quality_window,
            "total_frames": int(len(timeline)),
            "eye_warning_candidate_frames": int(timeline["recommended_alert"].sum()),
            "signal_unreliable_frames": int(timeline["signal_unreliable"].sum()),
            "warning": WARNING,
        },
    )
    return timeline[columns].copy()


def prepare_eye_for_fusion(eye_df: pd.DataFrame, session_id: str) -> pd.DataFrame:
    df = eye_df.copy()
    df["video_slug"] = session_id
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64")
    df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
    df["signal_unreliable"] = df["signal_unreliable"].map(to_bool)
    df["eye_warning_candidate"] = df["recommended_alert"].map(to_bool)
    df["p_eye_closed"] = pd.to_numeric(df["mean_p_eye_closed"], errors="coerce").fillna(0.0)
    df["eye_state"] = "normal"
    df.loc[df["eye_warning_candidate"], "eye_state"] = "eye_warning_candidate"
    df.loc[df["signal_unreliable"], "eye_state"] = "signal_unreliable"
    return df.sort_values("timestamp_sec").reset_index(drop=True)


def load_stage14_mouth_timeline(session_id: str, mouth_dir: Path) -> pd.DataFrame:
    pred_path = mouth_dir / "runtime_mouth_yawn_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing Stage 14 mouth predictions: {pred_path}")
    pred = pd.read_csv(pred_path)
    rows: list[dict[str, Any]] = []
    for _, row in pred.iterrows():
        rows.append(
            {
                "video_slug": session_id,
                "timestamp_sec": float(row["timestamp_sec"]),
                "frame_index": int(row["frame_index"]),
                "p_yawn": float(row["p_yawn"]),
                "yawn_event": to_bool(row["yawn_event"]),
                "recent_yawn_event": to_bool(row["recent_yawn_event"]),
                "mouth_signal_status": str(row.get("mouth_signal_status", "ok")),
                "mouth_source": MOUTH_SOURCE,
                "notes": "model-generated Stage 14 p_yawn timeline for uploaded video",
            }
        )

    failures_path = mouth_dir / "failures.csv"
    if failures_path.exists():
        failures = pd.read_csv(failures_path)
        for _, row in failures.iterrows():
            if pd.isna(row.get("frame_index")):
                continue
            rows.append(
                {
                    "video_slug": session_id,
                    "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
                    "frame_index": int(row["frame_index"]),
                    "p_yawn": 0.0,
                    "yawn_event": False,
                    "recent_yawn_event": False,
                    "mouth_signal_status": str(row.get("failure_type", "mouth_signal_unavailable")),
                    "mouth_source": MOUTH_SOURCE,
                    "notes": (
                        "Stage 14 mouth failure; no p_yawn generated; not treated as yawn"
                    ),
                }
            )
    mouth = pd.DataFrame(rows).sort_values(["timestamp_sec", "frame_index"]).reset_index(drop=True)
    return mouth


def augment_fusion_timeline(timeline: pd.DataFrame, mouth_df: pd.DataFrame) -> pd.DataFrame:
    extras = mouth_df[
        [
            "video_slug",
            "timestamp_sec",
            "frame_index",
            "yawn_event",
            "mouth_signal_status",
            "notes",
        ]
    ].copy()
    merged = timeline.merge(
        extras,
        on=["video_slug", "timestamp_sec", "frame_index"],
        how="left",
        suffixes=("", "_mouth_extra"),
    )
    merged["yawn_event"] = merged["yawn_event"].map(to_bool)
    merged["mouth_signal_status"] = merged["mouth_signal_status"].fillna("missing")
    return merged


def apply_sustained_eye_gate(fusion: pd.DataFrame) -> pd.DataFrame:
    """Suppress high-confidence escalation for brief or weak eye-warning intervals.

    Stage 17.1 keeps the existing eye-warning and mouth-warning logic intact, but
    requires sustained eye-warning evidence before upgrading eye+recent-yawn
    overlap to high-confidence warning candidate. Stage 17.5 adds a conservative
    strength-aware interpretation gate without changing the base eye-warning rule.
    """

    df = add_eye_evidence_calibration(fusion)
    df["eye_warning_interval_id"] = 0
    df["eye_warning_interval_duration_sec"] = 0.0
    df["eye_warning_interval_sampled_frames"] = 0
    df["sustained_eye_warning"] = False
    df["eye_strength_interval_mean_p_eye_closed"] = 0.0
    df["eye_strength_interval_max_p_eye_closed"] = 0.0
    df["eye_strength_interval_strong_frame_count"] = 0
    df["eye_strength_interval_moderate_or_strong_frame_count"] = 0
    df["eye_strength_gate_passed"] = False
    df["eye_strength_gate_reason"] = ""
    df["high_confidence_suppressed_by_brief_eye_warning"] = False
    df["high_confidence_suppressed_by_weak_eye_evidence"] = False

    eye_warning = df["eye_warning_candidate"].map(to_bool).tolist()
    interval_id = 0
    start: int | None = None
    for idx, value in enumerate(eye_warning + [False]):
        if value and start is None:
            start = idx
        if not value and start is not None:
            end = idx - 1
            interval_id += 1
            segment = df.iloc[start : end + 1]
            sampled_frames = int(len(segment))
            duration_sec = float(segment["timestamp_sec"].iloc[-1] - segment["timestamp_sec"].iloc[0])
            p_eye = pd.to_numeric(segment["p_eye_closed"], errors="coerce")
            mean_p_eye_closed = float(p_eye.mean()) if len(p_eye.dropna()) else 0.0
            max_p_eye_closed = float(p_eye.max()) if len(p_eye.dropna()) else 0.0
            strong_frames = int(segment["strong_eye_closure_candidate"].map(to_bool).sum())
            moderate_or_strong_frames = int(
                segment["moderate_or_strong_eye_evidence"].map(to_bool).sum()
            )
            sustained = (
                duration_sec >= SUSTAINED_EYE_GATE_MIN_DURATION_SEC
                or sampled_frames >= SUSTAINED_EYE_GATE_MIN_SAMPLED_FRAMES
            )
            strength_gate_passed = (
                mean_p_eye_closed >= EYE_STRENGTH_GATE_MIN_MEAN_P_EYE_CLOSED
                or max_p_eye_closed >= EYE_STRENGTH_GATE_MIN_MAX_P_EYE_CLOSED
                or strong_frames >= EYE_STRENGTH_GATE_MIN_STRONG_FRAMES
                or moderate_or_strong_frames
                >= EYE_STRENGTH_GATE_MIN_MODERATE_OR_STRONG_FRAMES
            )
            strength_reason = eye_strength_gate_reason(
                mean_p_eye_closed=mean_p_eye_closed,
                max_p_eye_closed=max_p_eye_closed,
                strong_frames=strong_frames,
                moderate_or_strong_frames=moderate_or_strong_frames,
            )
            interval_index = df.index[start : end + 1]
            df.loc[interval_index, "eye_warning_interval_id"] = interval_id
            df.loc[interval_index, "eye_warning_interval_duration_sec"] = duration_sec
            df.loc[interval_index, "eye_warning_interval_sampled_frames"] = sampled_frames
            df.loc[interval_index, "sustained_eye_warning"] = sustained
            df.loc[interval_index, "eye_strength_interval_mean_p_eye_closed"] = (
                mean_p_eye_closed
            )
            df.loc[interval_index, "eye_strength_interval_max_p_eye_closed"] = max_p_eye_closed
            df.loc[interval_index, "eye_strength_interval_strong_frame_count"] = strong_frames
            df.loc[interval_index, "eye_strength_interval_moderate_or_strong_frame_count"] = (
                moderate_or_strong_frames
            )
            df.loc[interval_index, "eye_strength_gate_passed"] = strength_gate_passed
            df.loc[interval_index, "eye_strength_gate_reason"] = strength_reason
            start = None

    initial_high_confidence = df["fusion_state"] == "high_confidence_drowsiness_candidate"
    suppress_brief = (
        initial_high_confidence
        & df["recent_yawn_event"].map(to_bool)
        & df["eye_warning_candidate"].map(to_bool)
        & (~df["sustained_eye_warning"].map(to_bool))
    )
    suppress_weak = (
        initial_high_confidence
        & df["recent_yawn_event"].map(to_bool)
        & df["eye_warning_candidate"].map(to_bool)
        & df["sustained_eye_warning"].map(to_bool)
        & (~df["eye_strength_gate_passed"].map(to_bool))
    )
    suppress = suppress_brief | suppress_weak
    df.loc[suppress_brief, "high_confidence_suppressed_by_brief_eye_warning"] = True
    df.loc[suppress_weak, "high_confidence_suppressed_by_weak_eye_evidence"] = True
    df.loc[suppress_brief, "fusion_reason"] = (
        "recent yawn event; high-confidence suppressed because eye-warning interval was brief"
    )
    df.loc[suppress_weak, "fusion_reason"] = (
        "recent yawn event; high-confidence suppressed by Stage 17.5 because calibrated "
        "eye evidence remained weak"
    )
    df.loc[suppress, "fusion_state"] = "mouth_warning_candidate"
    df.loc[suppress, "high_confidence_drowsiness_candidate"] = False
    df.loc[suppress, "mouth_warning_candidate"] = True
    df.loc[suppress, "mouth_state"] = "mouth_warning_candidate"
    return df


def strongest_eye_evidence(segment: pd.DataFrame) -> dict[str, Any]:
    if segment.empty or "eye_evidence_strength_rank" not in segment.columns:
        return {}
    rank = pd.to_numeric(segment["eye_evidence_strength_rank"], errors="coerce").fillna(0)
    idx = rank.idxmax()
    row = segment.loc[idx]
    gate_reasons = (
        segment["eye_strength_gate_reason"].replace("", pd.NA).dropna()
        if "eye_strength_gate_reason" in segment.columns
        else pd.Series([], dtype=object)
    )
    mean_strength = (
        pd.to_numeric(segment["eye_strength_interval_mean_p_eye_closed"], errors="coerce").max()
        if "eye_strength_interval_mean_p_eye_closed" in segment.columns
        else 0.0
    )
    max_strength = (
        pd.to_numeric(segment["eye_strength_interval_max_p_eye_closed"], errors="coerce").max()
        if "eye_strength_interval_max_p_eye_closed" in segment.columns
        else 0.0
    )
    strong_count = (
        pd.to_numeric(segment["eye_strength_interval_strong_frame_count"], errors="coerce").max()
        if "eye_strength_interval_strong_frame_count" in segment.columns
        else 0
    )
    moderate_count = (
        pd.to_numeric(
            segment["eye_strength_interval_moderate_or_strong_frame_count"],
            errors="coerce",
        ).max()
        if "eye_strength_interval_moderate_or_strong_frame_count" in segment.columns
        else 0
    )
    return {
        "eye_evidence_strength": str(row.get("eye_evidence_strength", "none")),
        "eye_evidence_label": str(row.get("eye_evidence_label", "")),
        "eye_evidence_interpretation": str(row.get("eye_evidence_interpretation", "")),
        "weak_eye_warning_evidence_frames": int(
            (segment["eye_evidence_strength"].astype(str) == "weak").sum()
        ),
        "moderate_eye_closure_candidate_frames": int(
            (segment["eye_evidence_strength"].astype(str) == "moderate").sum()
        ),
        "strong_eye_closure_candidate_frames": int(
            (segment["eye_evidence_strength"].astype(str) == "strong").sum()
        ),
        "eye_strength_gate_passed": bool(segment.get("eye_strength_gate_passed", False).map(to_bool).any())
        if "eye_strength_gate_passed" in segment.columns
        else False,
        "eye_strength_gate_reason": str(gate_reasons.iloc[0]) if len(gate_reasons) else "",
        "eye_strength_interval_mean_p_eye_closed": float(mean_strength)
        if not pd.isna(mean_strength)
        else 0.0,
        "eye_strength_interval_max_p_eye_closed": float(max_strength)
        if not pd.isna(max_strength)
        else 0.0,
        "eye_strength_interval_strong_frame_count": int(strong_count)
        if not pd.isna(strong_count)
        else 0,
        "eye_strength_interval_moderate_or_strong_frame_count": int(moderate_count)
        if not pd.isna(moderate_count)
        else 0,
        "high_confidence_suppressed_by_brief_eye_warning": bool(
            segment.get(
                "high_confidence_suppressed_by_brief_eye_warning",
                pd.Series(False, index=segment.index),
            )
            .map(to_bool)
            .any()
        ),
        "high_confidence_suppressed_by_weak_eye_evidence": bool(
            segment.get(
                "high_confidence_suppressed_by_weak_eye_evidence",
                pd.Series(False, index=segment.index),
            )
            .map(to_bool)
            .any()
        ),
    }


def run_lengths(mask: pd.Series) -> dict[str, Any]:
    longest = 0
    count = 0
    current = 0
    in_run = False
    for value in mask.map(to_bool):
        if value:
            current += 1
            if not in_run:
                count += 1
                in_run = True
            longest = max(longest, current)
        else:
            current = 0
            in_run = False
    return {"count": count, "longest": longest}


def intervals_for_state(df: pd.DataFrame, state: str) -> list[dict[str, Any]]:
    mask = (df["fusion_state"].astype(str) == state).tolist()
    intervals: list[dict[str, Any]] = []
    start: int | None = None
    for idx, value in enumerate(mask + [False]):
        if value and start is None:
            start = idx
        if not value and start is not None:
            end = idx - 1
            segment = df.iloc[start : end + 1]
            interval = {
                "start_frame_index": int(segment["frame_index"].iloc[0]),
                "end_frame_index": int(segment["frame_index"].iloc[-1]),
                "start_timestamp_sec": float(segment["timestamp_sec"].iloc[0]),
                "end_timestamp_sec": float(segment["timestamp_sec"].iloc[-1]),
                "duration_sampled_frames": int(len(segment)),
                "max_p_eye_closed": float(pd.to_numeric(segment["p_eye_closed"], errors="coerce").max()),
                "max_p_yawn": float(pd.to_numeric(segment["p_yawn"], errors="coerce").max()),
            }
            interval.update(strongest_eye_evidence(segment))
            intervals.append(interval)
            start = None
    return intervals


def plot_fusion_timeline(df: pd.DataFrame, path: Path, session_id: str) -> None:
    state_map = {
        "normal": 0,
        "eye_warning_candidate": 1,
        "mouth_warning_candidate": 2,
        "high_confidence_drowsiness_candidate": 3,
        "signal_unreliable": -1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["timestamp_sec"], df["p_eye_closed"], label="p_eye_closed", color="#2f5597")
    ax1.plot(df["timestamp_sec"], df["p_yawn"], label="p_yawn", color="#c55a11")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Probability")
    ax2 = ax1.twinx()
    ax2.step(
        df["timestamp_sec"],
        df["fusion_state"].map(state_map),
        label="fusion_state",
        color="#548235",
        where="post",
        alpha=0.75,
    )
    ax2.set_yticks([-1, 0, 1, 2, 3])
    ax2.set_yticklabels(
        ["signal_unreliable", "normal", "eye_warning", "mouth_warning", "high_conf"]
    )
    ax2.set_ylabel("Fusion state")
    ax1.set_title(f"Stage 17 Video Upload Fusion Timeline: {session_id}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_series(df: pd.DataFrame, y_col: str, path: Path, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp_sec"], df[y_col], color="#2f5597", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_summary(
    session_id: str,
    input_video: Path,
    created_at: str,
    duration_sec: float,
    fusion_df: pd.DataFrame,
    keyframes: list[dict[str, Any]],
    output_dir: Path,
    runtime_sec: float,
) -> dict[str, Any]:
    state_counts = fusion_df["fusion_state"].value_counts().to_dict()
    warning_mask = fusion_df["fusion_state"].isin(
        [
            "eye_warning_candidate",
            "mouth_warning_candidate",
            "high_confidence_drowsiness_candidate",
        ]
    )
    warning_ts = fusion_df.loc[warning_mask, "timestamp_sec"]
    high_intervals = intervals_for_state(fusion_df, "high_confidence_drowsiness_candidate")
    eye_intervals = intervals_for_state(fusion_df, "eye_warning_candidate")
    mouth_intervals = intervals_for_state(fusion_df, "mouth_warning_candidate")
    unreliable_intervals = intervals_for_state(fusion_df, "signal_unreliable")
    eye_strength = fusion_df.get(
        "eye_evidence_strength",
        pd.Series("none", index=fusion_df.index),
    ).astype(str)
    weak_eye_warning_frames = int(
        ((eye_strength == "weak") & fusion_df["eye_warning_candidate"].map(to_bool)).sum()
    )
    moderate_eye_frames = int((eye_strength == "moderate").sum())
    strong_eye_frames = int((eye_strength == "strong").sum())
    strength_suppressed_frames = int(
        fusion_df.get(
            "high_confidence_suppressed_by_weak_eye_evidence",
            pd.Series(False, index=fusion_df.index),
        )
        .map(to_bool)
        .sum()
    )
    return {
        "session_id": session_id,
        "input_video_path": str(input_video),
        "created_at": created_at,
        "pipeline_status": "completed",
        "runtime_sec": runtime_sec,
        "total_frames_sampled": int(len(fusion_df)),
        "duration_sec": duration_sec,
        "normal_frames": int(state_counts.get("normal", 0)),
        "eye_warning_candidate_frames": int(state_counts.get("eye_warning_candidate", 0)),
        "mouth_warning_candidate_frames": int(state_counts.get("mouth_warning_candidate", 0)),
        "high_confidence_drowsiness_candidate_frames": int(
            state_counts.get("high_confidence_drowsiness_candidate", 0)
        ),
        "signal_unreliable_frames": int(state_counts.get("signal_unreliable", 0)),
        "first_warning_timestamp_sec": float(warning_ts.min()) if len(warning_ts) else None,
        "last_warning_timestamp_sec": float(warning_ts.max()) if len(warning_ts) else None,
        "high_confidence_intervals": high_intervals,
        "eye_warning_intervals": eye_intervals,
        "mouth_warning_intervals": mouth_intervals,
        "signal_unreliable_intervals": unreliable_intervals,
        "yawn_event_count": int(fusion_df["yawn_event"].map(to_bool).sum()),
        "recent_yawn_event_count": int(fusion_df["recent_yawn_event"].map(to_bool).sum()),
        "suppressed_high_confidence_brief_eye_warning_frames": int(
            fusion_df.get(
                "high_confidence_suppressed_by_brief_eye_warning",
                pd.Series(False, index=fusion_df.index),
            )
            .map(to_bool)
            .sum()
        ),
        "suppressed_high_confidence_weak_eye_evidence_frames": strength_suppressed_frames,
        "sustained_eye_gate_min_duration_sec": SUSTAINED_EYE_GATE_MIN_DURATION_SEC,
        "sustained_eye_gate_min_sampled_frames": SUSTAINED_EYE_GATE_MIN_SAMPLED_FRAMES,
        "stage17_5_eye_evidence_calibration": {
            "version": EYE_EVIDENCE_CALIBRATION_VERSION,
            "weak_min_p_eye_closed": EYE_EVIDENCE_WEAK_MIN,
            "moderate_min_p_eye_closed": EYE_EVIDENCE_MODERATE_MIN,
            "strong_min_p_eye_closed": EYE_EVIDENCE_STRONG_MIN,
            "strength_gate_min_mean_p_eye_closed": EYE_STRENGTH_GATE_MIN_MEAN_P_EYE_CLOSED,
            "strength_gate_min_max_p_eye_closed": EYE_STRENGTH_GATE_MIN_MAX_P_EYE_CLOSED,
            "strength_gate_min_strong_frames": EYE_STRENGTH_GATE_MIN_STRONG_FRAMES,
            "strength_gate_min_moderate_or_strong_frames": (
                EYE_STRENGTH_GATE_MIN_MODERATE_OR_STRONG_FRAMES
            ),
            "note": (
                "Provisional rule-based calibration thresholds for interpretation and "
                "high-confidence gating; not final system-level drowsiness accuracy."
            ),
        },
        "weak_eye_warning_evidence_frames": weak_eye_warning_frames,
        "moderate_eye_closure_candidate_frames": moderate_eye_frames,
        "strong_eye_closure_candidate_frames": strong_eye_frames,
        "eye_evidence_strength_counts": {
            "none": int((eye_strength == "none").sum()),
            "weak": int((eye_strength == "weak").sum()),
            "moderate": moderate_eye_frames,
            "strong": strong_eye_frames,
            "signal_unreliable": int((eye_strength == "signal_unreliable").sum()),
        },
        "mean_p_eye_closed": float(pd.to_numeric(fusion_df["p_eye_closed"], errors="coerce").mean()),
        "max_p_eye_closed": float(pd.to_numeric(fusion_df["p_eye_closed"], errors="coerce").max()),
        "mean_p_yawn": float(pd.to_numeric(fusion_df["p_yawn"], errors="coerce").mean()),
        "max_p_yawn": float(pd.to_numeric(fusion_df["p_yawn"], errors="coerce").max()),
        "keyframes": keyframes,
        "output_dir": str(output_dir),
        "limitations": [
            "Video-upload inference/demo MVP only.",
            "Rule-based fusion, not a trained fusion classifier.",
            "Not final system-level drowsiness accuracy.",
            "Not deployment readiness.",
        ],
        "warning": WARNING,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    text = f"""# System Video Upload Analysis Report

## Purpose

This report summarizes one uploaded-video rule-based warning-candidate analysis session.

This is not final system-level drowsiness accuracy, not deployment readiness, and not a trained fusion classifier.

## Session

- Session ID: `{summary["session_id"]}`
- Input video: `{summary["input_video_path"]}`
- Pipeline status: `{summary["pipeline_status"]}`
- Sampled frames: {summary["total_frames_sampled"]}
- Runtime seconds: {summary["runtime_sec"]:.3f}

## Warning Candidate Counts

| State | Frames |
| --- | ---: |
| `normal` | {summary["normal_frames"]} |
| `eye_warning_candidate` | {summary["eye_warning_candidate_frames"]} |
| `mouth_warning_candidate` | {summary["mouth_warning_candidate_frames"]} |
| `high_confidence_drowsiness_candidate` | {summary["high_confidence_drowsiness_candidate_frames"]} |
| `signal_unreliable` | {summary["signal_unreliable_frames"]} |

## Stage 17.1 Sustained-Eye Gate

High-confidence warning candidates now require sustained eye-warning evidence in addition to recent-yawn evidence.

Gate rule:

- `recent_yawn_event == true`
- `eye_warning_candidate == true`
- `sustained_eye_warning == true`

`sustained_eye_warning` means the current eye-warning interval is at least {summary["sustained_eye_gate_min_duration_sec"]:.1f} second or at least {summary["sustained_eye_gate_min_sampled_frames"]} sampled frames.

Brief normal-blink-like eye events that overlap with recent-yawn evidence are suppressed from high-confidence escalation and remain mouth-warning candidates when recent-yawn evidence is active.

- Suppressed high-confidence frames due to brief eye warning: {summary["suppressed_high_confidence_brief_eye_warning_frames"]}

## Stage 17.5 Eye Evidence Calibration

Stage 17.5 adds provisional rule-based calibration for eye-warning evidence strength. It does not change the eye model, the `p_eye_closed = softmax(logits)[0]` formula, or the base Stage 17 eye-warning rule.

Evidence ranges:

- Weak eye-warning evidence: `p_eye_closed >= {summary["stage17_5_eye_evidence_calibration"]["weak_min_p_eye_closed"]:.2f}`
- Moderate eye-closure candidate: `p_eye_closed >= {summary["stage17_5_eye_evidence_calibration"]["moderate_min_p_eye_closed"]:.2f}`
- Strong eye-closure candidate: `p_eye_closed >= {summary["stage17_5_eye_evidence_calibration"]["strong_min_p_eye_closed"]:.2f}`

High-confidence warning candidates must still pass Stage 17.1 sustained-eye gating and now must also pass the Stage 17.5 strength gate. The strength gate passes when the eye-warning interval has interval mean `p_eye_closed >= {summary["stage17_5_eye_evidence_calibration"]["strength_gate_min_mean_p_eye_closed"]:.2f}`, interval max `p_eye_closed >= {summary["stage17_5_eye_evidence_calibration"]["strength_gate_min_max_p_eye_closed"]:.2f}`, at least {summary["stage17_5_eye_evidence_calibration"]["strength_gate_min_strong_frames"]} strong frame, or at least {summary["stage17_5_eye_evidence_calibration"]["strength_gate_min_moderate_or_strong_frames"]} moderate-or-strong frames.

- Weak eye-warning evidence frames: {summary["weak_eye_warning_evidence_frames"]}
- Moderate eye-closure candidate frames: {summary["moderate_eye_closure_candidate_frames"]}
- Strong eye-closure candidate frames: {summary["strong_eye_closure_candidate_frames"]}
- Suppressed high-confidence frames due to weak calibrated eye evidence: {summary["suppressed_high_confidence_weak_eye_evidence_frames"]}

## Mouth/Yawn Signal

- Yawn-event count: {summary["yawn_event_count"]}
- Recent-yawn-event count: {summary["recent_yawn_event_count"]}
- Mean/max `p_yawn`: {summary["mean_p_yawn"]:.6f} / {summary["max_p_yawn"]:.6f}

## Eye Signal

- Mean/max `p_eye_closed`: {summary["mean_p_eye_closed"]:.6f} / {summary["max_p_eye_closed"]:.6f}

## Keyframes

- Keyframes saved: {len(summary["keyframes"])}
- Keyframe metadata: `keyframes/keyframes_metadata.csv`

## Warning

{WARNING}
"""
    path.write_text(text, encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    session_id = args.session_id or f"upload_{uuid.uuid4().hex[:12]}"
    input_video = args.input_video if args.input_video.is_absolute() else PROJECT_ROOT / args.input_video
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if args.sample_every_n_frames <= 0 or args.max_frames <= 0:
        raise ValueError("--sample-every-n-frames and --max-frames must be positive")

    output_dir = resolve_output_dir(session_id, args.output_dir)
    if output_dir.exists() and not args.force and (output_dir / "summary.json").exists():
        raise FileExistsError(f"Output directory already has summary.json; use --force: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    figures_dir = output_dir / "figures"
    created_at = now_iso()

    eye_stage10_dir = output_dir / "eye_stage10"
    eye_stage11_dir = output_dir / "eye_stage11"
    eye_stage12_dir = output_dir / "eye_stage12"
    mouth_stage14_dir = output_dir / "mouth_stage14"

    base_eye_cmd = [
        sys.executable,
        "src/runtime/stage10_eye_roi_consistency.py",
        "--input-video",
        str(input_video),
        "--sample-every-n-frames",
        str(args.sample_every_n_frames),
        "--max-frames",
        str(args.max_frames),
        "--output-dir",
        str(eye_stage10_dir),
    ]
    if args.save_debug:
        base_eye_cmd.extend(["--save-crops", "--save-debug-frames"])
    run_command(base_eye_cmd, logs_dir / "stage10_eye.log")

    run_command(
        [
            sys.executable,
            "src/runtime/stage11_eye_temporal_analysis.py",
            "--input-csv",
            str(eye_stage10_dir / "runtime_eye_roi_predictions.csv"),
            "--output-dir",
            str(eye_stage11_dir),
            "--closed-threshold",
            "0.50",
            "--rolling-window",
            "5",
        ],
        logs_dir / "stage11_eye_temporal.log",
    )

    eye_alert = build_eye_alert_timeline(
        session_id=session_id,
        stage10_dir=eye_stage10_dir,
        stage11_dir=eye_stage11_dir,
        output_dir=eye_stage12_dir,
    )

    mouth_cmd = [
        sys.executable,
        "src/runtime/stage14_mouth_yawn_runtime.py",
        "--input-video",
        str(input_video),
        "--sample-every-n-frames",
        str(args.sample_every_n_frames),
        "--max-frames",
        str(args.max_frames),
        "--yawn-threshold",
        str(args.yawn_threshold),
        "--recent-yawn-window-sec",
        str(args.recent_yawn_window_sec),
        "--output-dir",
        str(mouth_stage14_dir),
    ]
    if args.save_debug:
        mouth_cmd.extend(["--save-crops", "--save-debug-frames"])
    run_command(mouth_cmd, logs_dir / "stage14_mouth.log")

    eye_for_fusion = prepare_eye_for_fusion(eye_alert, session_id)
    mouth_timeline = load_stage14_mouth_timeline(session_id, mouth_stage14_dir)
    mouth_timeline.to_csv(output_dir / "mouth_timeline_stage13_schema.csv", index=False)
    aligned_mouth = align_real_mouth_timeline(eye_for_fusion, mouth_timeline, session_id)
    fusion = build_fusion_timeline(eye_for_fusion, aligned_mouth, session_id, FUSION_RULE_NAME)
    fusion = augment_fusion_timeline(fusion, aligned_mouth)
    fusion["yawn_event"] = fusion["yawn_event"].map(to_bool)
    fusion = apply_sustained_eye_gate(fusion)

    fusion_path = output_dir / "fusion_timeline.csv"
    timeline_path = output_dir / "timeline.csv"
    fusion.to_csv(fusion_path, index=False)
    fusion.to_csv(timeline_path, index=False)

    plot_fusion_timeline(fusion, figures_dir / "fusion_timeline.png", session_id)
    plot_series(
        fusion,
        "p_eye_closed",
        figures_dir / "p_eye_closed_over_time.png",
        f"Stage 17 p_eye_closed Over Time: {session_id}",
        "p_eye_closed",
    )
    plot_series(
        fusion,
        "p_yawn",
        figures_dir / "p_yawn_over_time.png",
        f"Stage 17 p_yawn Over Time: {session_id}",
        "p_yawn",
    )

    keyframes: list[dict[str, Any]] = []
    if args.save_keyframes:
        keyframes, keyframe_summary = extract_keyframes(
            video_path=input_video,
            fusion_timeline=fusion,
            output_dir=output_dir / "keyframes",
            session_id=session_id,
            max_keyframes=20,
        )
        write_json(output_dir / "keyframes" / "keyframes_summary.json", keyframe_summary)
    else:
        (output_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_dir / "keyframes" / "keyframes_metadata.csv", index=False)
        write_json(output_dir / "keyframes" / "keyframes_metadata.json", [])

    duration_sec = float(fusion["timestamp_sec"].max()) if len(fusion) else 0.0
    summary = build_summary(
        session_id=session_id,
        input_video=input_video,
        created_at=created_at,
        duration_sec=duration_sec,
        fusion_df=fusion,
        keyframes=keyframes,
        output_dir=output_dir,
        runtime_sec=time.time() - started,
    )
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "fusion_summary.json", summary)
    write_report(output_dir / "SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md", summary)

    with (output_dir / "pipeline_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artifact", "path"])
        for artifact in [
            "summary.json",
            "timeline.csv",
            "fusion_timeline.csv",
            "fusion_summary.json",
            "figures/fusion_timeline.png",
            "figures/p_eye_closed_over_time.png",
            "figures/p_yawn_over_time.png",
            "keyframes/keyframes_metadata.csv",
            "SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md",
        ]:
            writer.writerow([artifact, str(output_dir / artifact)])

    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run_pipeline(args)
    except Exception as exc:
        session_id = args.session_id or "unknown_session"
        output_dir = resolve_output_dir(session_id, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "session_id": session_id,
            "pipeline_status": "failed",
            "error": str(exc),
            "warning": WARNING,
        }
        write_json(output_dir / "summary.json", failure)
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    print(f"STAGE17_VIDEO_UPLOAD_PIPELINE_COMPLETED session_id={summary['session_id']}")
    print(f"summary={summary['output_dir']}/summary.json")
    print(f"timeline={summary['output_dir']}/timeline.csv")
    print(f"keyframes={len(summary['keyframes'])}")
    print(WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
