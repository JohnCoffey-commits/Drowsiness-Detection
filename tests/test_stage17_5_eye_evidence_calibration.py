from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.system_video_upload_pipeline import (
    apply_sustained_eye_gate,
    classify_eye_evidence,
)


def _base_rows(probabilities: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_slug": ["test"] * len(probabilities),
            "frame_index": list(range(len(probabilities))),
            "timestamp_sec": [index * 0.2 for index in range(len(probabilities))],
            "fusion_state": ["high_confidence_drowsiness_candidate"] * len(probabilities),
            "fusion_reason": ["eye warning + recent yawn event"] * len(probabilities),
            "signal_unreliable": [False] * len(probabilities),
            "eye_warning_candidate": [True] * len(probabilities),
            "recent_yawn_event": [True] * len(probabilities),
            "high_confidence_drowsiness_candidate": [True] * len(probabilities),
            "mouth_warning_candidate": [True] * len(probabilities),
            "mouth_state": ["mouth_warning_candidate"] * len(probabilities),
            "p_eye_closed": probabilities,
            "p_yawn": [0.9] * len(probabilities),
        }
    )


def test_classify_eye_evidence_thresholds() -> None:
    assert classify_eye_evidence(0.49)["eye_evidence_strength"] == "none"
    assert classify_eye_evidence(0.55)["eye_evidence_strength"] == "weak"
    assert classify_eye_evidence(0.70)["eye_evidence_strength"] == "moderate"
    assert classify_eye_evidence(0.85)["eye_evidence_strength"] == "strong"
    assert (
        classify_eye_evidence(0.95, signal_unreliable=True)["eye_evidence_strength"]
        == "signal_unreliable"
    )


def test_stage17_5_suppresses_sustained_but_weak_eye_evidence() -> None:
    result = apply_sustained_eye_gate(_base_rows([0.55, 0.58, 0.60, 0.61, 0.57]))

    assert result["sustained_eye_warning"].all()
    assert not result["eye_strength_gate_passed"].any()
    assert result["high_confidence_suppressed_by_weak_eye_evidence"].all()
    assert (result["fusion_state"] == "mouth_warning_candidate").all()


def test_stage17_5_keeps_high_confidence_when_strength_gate_passes() -> None:
    result = apply_sustained_eye_gate(_base_rows([0.56, 0.72, 0.89, 0.73, 0.58]))

    assert result["sustained_eye_warning"].all()
    assert result["eye_strength_gate_passed"].all()
    assert not result["high_confidence_suppressed_by_weak_eye_evidence"].any()
    assert (result["fusion_state"] == "high_confidence_drowsiness_candidate").all()


if __name__ == "__main__":
    test_classify_eye_evidence_thresholds()
    test_stage17_5_suppresses_sustained_but_weak_eye_evidence()
    test_stage17_5_keeps_high_confidence_when_strength_gate_passes()
    print("Stage 17.5 eye evidence calibration tests passed.")
