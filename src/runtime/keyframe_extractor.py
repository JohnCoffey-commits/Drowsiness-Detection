#!/usr/bin/env python3
"""Keyframe extraction for warning-candidate timeline segments.

This module saves a small set of original-video screenshots for rule-based
warning-candidate intervals. It does not classify drowsiness by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


WARNING_STATES = [
    "high_confidence_drowsiness_candidate",
    "eye_warning_candidate",
    "mouth_warning_candidate",
    "signal_unreliable",
]

STATE_TO_DIR = {
    "high_confidence_drowsiness_candidate": "high_confidence",
    "eye_warning_candidate": "eye_warning",
    "mouth_warning_candidate": "mouth_warning",
    "signal_unreliable": "signal_unreliable",
}


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for keyframe extraction") from exc
    return cv2


def segment_state(df: pd.DataFrame, state: str) -> list[pd.DataFrame]:
    mask = df["fusion_state"].astype(str) == state
    segments: list[pd.DataFrame] = []
    start: int | None = None
    values = mask.tolist()
    for idx, value in enumerate(values + [False]):
        if value and start is None:
            start = idx
        if not value and start is not None:
            segment = df.iloc[start:idx].copy()
            if not segment.empty:
                segments.append(segment)
            start = None
    return segments


def score_column_for_state(state: str) -> str:
    if state == "high_confidence_drowsiness_candidate":
        return "_combined_score"
    if state == "eye_warning_candidate":
        return "p_eye_closed"
    if state == "mouth_warning_candidate":
        return "p_yawn"
    return "_combined_score"


def select_rows_for_segment(segment: pd.DataFrame, state: str) -> list[pd.Series]:
    rows: list[pd.Series] = []
    positions = [0, len(segment) // 2]
    if len(segment) >= 4:
        positions.append(len(segment) - 1)

    score_col = score_column_for_state(state)
    if score_col in segment.columns and not segment[score_col].dropna().empty:
        positions.append(int(segment[score_col].astype(float).idxmax()))

    selected_indices: list[int] = []
    for pos in positions:
        if pos in segment.index:
            idx = int(pos)
        else:
            idx = int(segment.index[min(max(int(pos), 0), len(segment) - 1)])
        if idx not in selected_indices:
            selected_indices.append(idx)
    return [segment.loc[idx] for idx in selected_indices]


def overlay_text(cv2, frame, lines: list[str]) -> None:
    x, y = 18, 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    for line in lines:
        (width, height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(
            frame,
            (x - 6, y - height - 7),
            (x + width + 8, y + baseline + 7),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += height + 16


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def save_frame(cv2, cap, video_path: Path, row: pd.Series, output_path: Path) -> bool:
    frame_index = safe_float(row.get("frame_index"))
    if frame_index is None:
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    lines = [
        f"t={safe_float(row.get('timestamp_sec')):.2f}s frame={int(frame_index)}",
        str(row.get("fusion_state", "")),
        f"p_eye_closed={safe_float(row.get('p_eye_closed')) or 0.0:.3f} p_yawn={safe_float(row.get('p_yawn')) or 0.0:.3f}",
    ]
    eye_label = str(row.get("eye_evidence_label", ""))
    if eye_label:
        lines.append(eye_label[:72])
    reason = str(row.get("fusion_reason", ""))
    if reason:
        lines.append(reason[:72])
    overlay_text(cv2, frame, lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


def extract_keyframes(
    video_path: Path,
    fusion_timeline: pd.DataFrame,
    output_dir: Path,
    session_id: str,
    max_keyframes: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cv2 = load_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in STATE_TO_DIR.values():
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    df = fusion_timeline.copy()
    if df.empty:
        metadata: list[dict[str, Any]] = []
        return metadata, {"keyframe_count": 0, "warning": "No fusion rows available."}

    df["p_eye_closed"] = pd.to_numeric(df.get("p_eye_closed", 0.0), errors="coerce").fillna(0.0)
    df["p_yawn"] = pd.to_numeric(df.get("p_yawn", 0.0), errors="coerce").fillna(0.0)
    df["_combined_score"] = df["p_eye_closed"] + df["p_yawn"]

    selected: list[tuple[str, int, pd.Series, bool]] = []
    high_segments = segment_state(df, "high_confidence_drowsiness_candidate")

    if high_segments:
        for segment_id, segment in enumerate(high_segments, start=1):
            for row in select_rows_for_segment(segment, "high_confidence_drowsiness_candidate"):
                selected.append(("high_confidence", segment_id, row, True))
    else:
        for state in ("eye_warning_candidate", "mouth_warning_candidate"):
            for segment_id, segment in enumerate(segment_state(df, state), start=1):
                for row in select_rows_for_segment(segment, state):
                    selected.append((STATE_TO_DIR[state], segment_id, row, False))

    for segment_id, segment in enumerate(segment_state(df, "signal_unreliable"), start=1):
        for row in select_rows_for_segment(segment, "signal_unreliable")[:2]:
            selected.append(("signal_unreliable", segment_id, row, False))

    seen_frames: set[tuple[str, int]] = set()
    deduped: list[tuple[str, int, pd.Series, bool]] = []
    for warning_type, segment_id, row, is_primary in selected:
        frame_index = safe_float(row.get("frame_index"))
        if frame_index is None:
            continue
        key = (warning_type, int(frame_index))
        if key in seen_frames:
            continue
        seen_frames.add(key)
        deduped.append((warning_type, segment_id, row, is_primary))
        if len(deduped) >= max_keyframes:
            break

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for keyframe extraction: {video_path}")

    metadata: list[dict[str, Any]] = []
    try:
        for warning_type, segment_id, row, is_primary in deduped:
            frame_index = int(safe_float(row.get("frame_index")) or 0)
            timestamp = safe_float(row.get("timestamp_sec")) or 0.0
            filename = (
                f"{session_id}_{warning_type}_seg{segment_id:02d}_"
                f"frame{frame_index:06d}.jpg"
            )
            keyframe_path = output_dir / warning_type / filename
            if not save_frame(cv2, cap, video_path, row, keyframe_path):
                continue
            metadata.append(
                {
                    "keyframe_path": str(keyframe_path),
                    "video_path": str(video_path),
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp,
                    "fusion_state": str(row.get("fusion_state", "")),
                    "p_eye_closed": safe_float(row.get("p_eye_closed")),
                    "p_yawn": safe_float(row.get("p_yawn")),
                    "recent_yawn_event": to_bool(row.get("recent_yawn_event")),
                    "sustained_eye_warning": to_bool(row.get("sustained_eye_warning")),
                    "eye_evidence_strength": str(row.get("eye_evidence_strength", "")),
                    "eye_evidence_label": str(row.get("eye_evidence_label", "")),
                    "eye_evidence_interpretation": str(
                        row.get("eye_evidence_interpretation", "")
                    ),
                    "eye_strength_gate_passed": to_bool(row.get("eye_strength_gate_passed")),
                    "eye_strength_gate_reason": str(row.get("eye_strength_gate_reason", "")),
                    "high_confidence_suppressed_by_brief_eye_warning": to_bool(
                        row.get("high_confidence_suppressed_by_brief_eye_warning")
                    ),
                    "high_confidence_suppressed_by_weak_eye_evidence": to_bool(
                        row.get("high_confidence_suppressed_by_weak_eye_evidence")
                    ),
                    "warning_type": warning_type,
                    "reason": str(row.get("fusion_reason", "")),
                    "segment_id": segment_id,
                    "is_primary": bool(is_primary),
                }
            )
    finally:
        cap.release()

    metadata_csv = output_dir / "keyframes_metadata.csv"
    metadata_json = output_dir / "keyframes_metadata.json"
    pd.DataFrame(metadata).to_csv(metadata_csv, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata, {
        "keyframe_count": len(metadata),
        "metadata_csv": str(metadata_csv),
        "metadata_json": str(metadata_json),
        "max_keyframes": max_keyframes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract keyframes from a fusion timeline.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--max-keyframes", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeline = pd.read_csv(args.timeline)
    metadata, summary = extract_keyframes(
        video_path=args.video,
        fusion_timeline=timeline,
        output_dir=args.output_dir,
        session_id=args.session_id,
        max_keyframes=args.max_keyframes,
    )
    print(f"keyframes={len(metadata)}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
