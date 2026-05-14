"""Session-local realtime temporal warning-candidate state.

This module converts single-frame webcam evidence into a rolling rule-based
warning-candidate state. It does not train models, produce alarms, write
history, or claim final drowsiness truth.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any


REALTIME_RULE_WARNING = (
    "This output is a realtime rule-based warning-candidate analysis, "
    "not final system-level drowsiness accuracy."
)

EYE_CLOSED_THRESHOLD = 0.50
EYE_WARNING_ENTER_ROLLING_MEAN = 0.60
EYE_WARNING_ENTER_CONSECUTIVE_FRAMES = 2
EYE_WARNING_EXIT_ROLLING_MEAN = 0.40
EYE_WARNING_EXIT_CONSECUTIVE_FRAMES = 2
SUSTAINED_EYE_WARNING_MIN_SECONDS = 1.0
SUSTAINED_EYE_WARNING_MIN_FRAMES = 5
RECENT_EYE_WARNING_REMINDER_SECONDS = 4.0
RECENT_EYE_WARNING_REMINDER_REQUIRES_SUSTAINED = True
RECENT_EYE_WARNING_REMINDER_REQUIRES_MODERATE_OR_STRONG = True
YAWN_ON_THRESHOLD = 0.50
YAWN_OFF_THRESHOLD = 0.35
YAWN_OFF_CONSECUTIVE_FRAMES = 2
MOUTH_ACTIVE_MAX_HOLD_SECONDS = 1.5
RECENT_YAWN_CONTEXT_SECONDS = 4.0
RECENT_YAWN_REMINDER_SECONDS = 8.0
ROLLING_WINDOW_FRAMES = 5
SIGNAL_FAILURE_RATIO_THRESHOLD = 0.20

MODERATE_OR_STRONG_EYE_STRENGTHS = {
    "moderate_eye_closure_candidate",
    "strong_eye_closure_candidate",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


class RealtimeTemporalState:
    """Rolling state for one in-memory realtime session."""

    def __init__(self, max_frames: int = 240) -> None:
        self.frames: deque[dict[str, Any]] = deque(maxlen=max_frames)
        self.last_yawn_monotonic: float | None = None
        self.last_mouth_active_monotonic: float | None = None
        self.mouth_active = False
        self.yawn_off_consecutive_frames = 0
        self.eye_warning_active = False
        self.eye_warning_enter_consecutive_frames = 0
        self.eye_warning_exit_consecutive_frames = 0
        self.current_eye_warning_interval_start: float | None = None
        self.current_eye_warning_frames = 0
        self.current_eye_warning_peak_p_eye_closed: float | None = None
        self.current_eye_warning_peak_strength = "unavailable"
        self.current_eye_warning_has_moderate_or_strong = False
        self.last_sustained_eye_warning_end_monotonic: float | None = None
        self.last_sustained_eye_warning_duration_seconds: float | None = None
        self.last_sustained_eye_warning_peak_strength: str | None = None
        self.last_sustained_eye_warning_peak_p_eye_closed: float | None = None
        self.stopped_at: str | None = None

    def freeze(self) -> str:
        self.stopped_at = now_iso()
        self.frames.clear()
        self.last_yawn_monotonic = None
        self.last_mouth_active_monotonic = None
        self.mouth_active = False
        self.yawn_off_consecutive_frames = 0
        self.eye_warning_active = False
        self.eye_warning_enter_consecutive_frames = 0
        self.eye_warning_exit_consecutive_frames = 0
        self.current_eye_warning_interval_start = None
        self.current_eye_warning_frames = 0
        self.current_eye_warning_peak_p_eye_closed = None
        self.current_eye_warning_peak_strength = "unavailable"
        self.current_eye_warning_has_moderate_or_strong = False
        self.last_sustained_eye_warning_end_monotonic = None
        self.last_sustained_eye_warning_duration_seconds = None
        self.last_sustained_eye_warning_peak_strength = None
        self.last_sustained_eye_warning_peak_p_eye_closed = None
        return self.stopped_at

    def update_from_frame(self, frame_result: dict[str, Any], now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        frame = self._extract_frame(frame_result, now)
        self.frames.append(frame)

        self._update_mouth_activity(frame, now)

        last_yawn_age = (
            None if self.last_yawn_monotonic is None else now - self.last_yawn_monotonic
        )
        recent_yawn_event = (
            last_yawn_age is not None and last_yawn_age <= RECENT_YAWN_CONTEXT_SECONDS
        )
        recent_yawn_reminder = (
            last_yawn_age is not None and last_yawn_age <= RECENT_YAWN_REMINDER_SECONDS
        )

        recent_frames = list(self.frames)[-ROLLING_WINDOW_FRAMES:]
        signal_failures = sum(1 for item in recent_frames if item["signal_failure"])
        recent_signal_failure_ratio = (
            signal_failures / len(recent_frames) if recent_frames else 0.0
        )
        signal_unreliable = recent_signal_failure_ratio > SIGNAL_FAILURE_RATIO_THRESHOLD

        valid_eye_frames = [
            item for item in recent_frames if item["eye_closed_binary"] is not None
        ]
        rolling_eye_closed_mean = (
            sum(1 for item in valid_eye_frames if item["eye_closed_binary"])
            / len(valid_eye_frames)
            if valid_eye_frames
            else None
        )

        eye_state = self._update_eye_warning_state(
            frame=frame,
            rolling_eye_closed_mean=rolling_eye_closed_mean,
            signal_unreliable=signal_unreliable,
            now=now,
        )
        current_strength = str(frame["eye_evidence_strength"])
        eye_warning_active = bool(eye_state["eye_warning_active"])
        eye_warning_candidate = eye_warning_active
        current_eye_warning_duration_seconds = float(
            eye_state["current_eye_warning_duration_seconds"]
        )
        sustained_eye_warning = bool(eye_state["sustained_eye_warning"])
        moderate_or_strong_eye_evidence = bool(
            eye_state["moderate_or_strong_eye_evidence"]
        )
        last_sustained_eye_warning_age = (
            None
            if self.last_sustained_eye_warning_end_monotonic is None
            else now - self.last_sustained_eye_warning_end_monotonic
        )
        recent_eye_warning_reminder = (
            last_sustained_eye_warning_age is not None
            and last_sustained_eye_warning_age <= RECENT_EYE_WARNING_REMINDER_SECONDS
        )

        fusion_state, suppressed, reason = self._fuse(
            signal_unreliable=signal_unreliable,
            mouth_active=self.mouth_active,
            recent_yawn_event=recent_yawn_event,
            recent_yawn_reminder=recent_yawn_reminder,
            recent_eye_warning_reminder=recent_eye_warning_reminder,
            eye_warning_candidate=eye_warning_candidate,
            sustained_eye_warning=sustained_eye_warning,
            moderate_or_strong_eye_evidence=moderate_or_strong_eye_evidence,
        )

        mouth_warning_candidate = self.mouth_active
        return {
            "fusion_state": fusion_state,
            "eye_warning_candidate": eye_warning_candidate,
            "mouth_warning_candidate": mouth_warning_candidate,
            "current_eye_evidence": current_strength,
            "eye_warning_active": eye_warning_active,
            "eye_warning_exit_consecutive_frames": self.eye_warning_exit_consecutive_frames,
            "current_eye_warning_interval_frames": self.current_eye_warning_frames,
            "current_eye_warning_peak_p_eye_closed": round(
                self.current_eye_warning_peak_p_eye_closed,
                3,
            )
            if self.current_eye_warning_peak_p_eye_closed is not None
            else None,
            "current_eye_warning_peak_strength": self.current_eye_warning_peak_strength,
            "recent_eye_warning_reminder": recent_eye_warning_reminder,
            "recent_eye_warning_reminder_seconds": RECENT_EYE_WARNING_REMINDER_SECONDS,
            "last_sustained_eye_warning_age_seconds": round(
                last_sustained_eye_warning_age,
                3,
            )
            if last_sustained_eye_warning_age is not None
            else None,
            "last_sustained_eye_warning_duration_seconds": round(
                self.last_sustained_eye_warning_duration_seconds,
                3,
            )
            if self.last_sustained_eye_warning_duration_seconds is not None
            else None,
            "last_sustained_eye_warning_peak_strength": (
                self.last_sustained_eye_warning_peak_strength
            ),
            "last_sustained_eye_warning_peak_p_eye_closed": round(
                self.last_sustained_eye_warning_peak_p_eye_closed,
                3,
            )
            if self.last_sustained_eye_warning_peak_p_eye_closed is not None
            else None,
            "mouth_active": self.mouth_active,
            "recent_yawn_event": recent_yawn_event,
            "recent_yawn_context_seconds": RECENT_YAWN_CONTEXT_SECONDS,
            "recent_yawn_reminder": recent_yawn_reminder,
            "recent_yawn_reminder_seconds": RECENT_YAWN_REMINDER_SECONDS,
            "last_yawn_age_seconds": round(last_yawn_age, 3)
            if last_yawn_age is not None
            else None,
            "yawn_off_consecutive_frames": self.yawn_off_consecutive_frames,
            "mouth_active_max_hold_seconds": MOUTH_ACTIVE_MAX_HOLD_SECONDS,
            "rolling_eye_closed_mean": round(rolling_eye_closed_mean, 3)
            if rolling_eye_closed_mean is not None
            else None,
            "eye_warning_consecutive_frames": self.eye_warning_enter_consecutive_frames,
            "current_eye_warning_duration_seconds": round(
                current_eye_warning_duration_seconds,
                3,
            ),
            "sustained_eye_warning": sustained_eye_warning,
            "eye_evidence_strength": current_strength,
            "moderate_or_strong_eye_evidence": moderate_or_strong_eye_evidence,
            "signal_unreliable": signal_unreliable,
            "recent_signal_failure_ratio": round(recent_signal_failure_ratio, 3),
            "high_confidence_suppressed_by_brief_or_weak_eye_warning": suppressed,
            "safe_reason_text": reason,
            "yawn_event": frame["yawn_event"],
            "eye_closed_binary": frame["eye_closed_binary"],
            "recent_window_frame_count": len(recent_frames),
            "valid_eye_frame_count": len(valid_eye_frames),
            "warning": REALTIME_RULE_WARNING,
        }

    def _update_mouth_activity(self, frame: dict[str, Any], now: float) -> None:
        p_yawn = frame["p_yawn"]
        mouth_available = bool(frame["mouth_available"])

        if mouth_available and p_yawn is not None:
            if p_yawn >= YAWN_ON_THRESHOLD:
                self.mouth_active = True
                self.yawn_off_consecutive_frames = 0
                self.last_yawn_monotonic = now
                self.last_mouth_active_monotonic = now
                return

            if p_yawn < YAWN_OFF_THRESHOLD:
                if self.mouth_active:
                    self.yawn_off_consecutive_frames += 1
                    if self.yawn_off_consecutive_frames >= YAWN_OFF_CONSECUTIVE_FRAMES:
                        self.mouth_active = False
                return

            self.yawn_off_consecutive_frames = 0
            return

        if (
            self.mouth_active
            and self.last_mouth_active_monotonic is not None
            and now - self.last_mouth_active_monotonic > MOUTH_ACTIVE_MAX_HOLD_SECONDS
        ):
            self.mouth_active = False
            self.yawn_off_consecutive_frames = 0

    def _update_eye_warning_state(
        self,
        *,
        frame: dict[str, Any],
        rolling_eye_closed_mean: float | None,
        signal_unreliable: bool,
        now: float,
    ) -> dict[str, Any]:
        enter_condition = (
            rolling_eye_closed_mean is not None
            and rolling_eye_closed_mean >= EYE_WARNING_ENTER_ROLLING_MEAN
            and not signal_unreliable
        )

        if signal_unreliable:
            self.eye_warning_enter_consecutive_frames = 0
            self.eye_warning_exit_consecutive_frames = 0
            if self.eye_warning_active:
                self._close_eye_warning_interval(now)
            return self._current_eye_state(now)

        if enter_condition:
            self.eye_warning_enter_consecutive_frames += 1
        else:
            self.eye_warning_enter_consecutive_frames = 0

        if self.eye_warning_active:
            if (
                rolling_eye_closed_mean is not None
                and rolling_eye_closed_mean < EYE_WARNING_EXIT_ROLLING_MEAN
            ):
                self.eye_warning_exit_consecutive_frames += 1
            else:
                self.eye_warning_exit_consecutive_frames = 0

            self._record_eye_warning_frame(frame)

            if self.eye_warning_exit_consecutive_frames >= EYE_WARNING_EXIT_CONSECUTIVE_FRAMES:
                self._close_eye_warning_interval(now)
        elif (
            self.eye_warning_enter_consecutive_frames
            >= EYE_WARNING_ENTER_CONSECUTIVE_FRAMES
        ):
            self._start_eye_warning_interval(now)
            self._record_eye_warning_frame(frame)

        return self._current_eye_state(now)

    def _start_eye_warning_interval(self, now: float) -> None:
        self.eye_warning_active = True
        self.eye_warning_exit_consecutive_frames = 0
        self.current_eye_warning_interval_start = now
        self.current_eye_warning_frames = 0
        self.current_eye_warning_peak_p_eye_closed = None
        self.current_eye_warning_peak_strength = "unavailable"
        self.current_eye_warning_has_moderate_or_strong = False

    def _record_eye_warning_frame(self, frame: dict[str, Any]) -> None:
        self.current_eye_warning_frames += 1
        mean_p_eye_closed = frame["mean_p_eye_closed"]
        strength = str(frame["eye_evidence_strength"])

        if strength in MODERATE_OR_STRONG_EYE_STRENGTHS:
            self.current_eye_warning_has_moderate_or_strong = True

        if mean_p_eye_closed is None:
            return

        if (
            self.current_eye_warning_peak_p_eye_closed is None
            or mean_p_eye_closed > self.current_eye_warning_peak_p_eye_closed
        ):
            self.current_eye_warning_peak_p_eye_closed = mean_p_eye_closed
            self.current_eye_warning_peak_strength = strength

    def _current_eye_state(self, now: float) -> dict[str, Any]:
        current_duration = (
            0.0
            if not self.eye_warning_active
            or self.current_eye_warning_interval_start is None
            else now - self.current_eye_warning_interval_start
        )
        sustained = self.eye_warning_active and (
            current_duration >= SUSTAINED_EYE_WARNING_MIN_SECONDS
            or self.current_eye_warning_frames >= SUSTAINED_EYE_WARNING_MIN_FRAMES
        )
        moderate_or_strong = (
            self.eye_warning_active and self.current_eye_warning_has_moderate_or_strong
        )

        return {
            "eye_warning_active": self.eye_warning_active,
            "current_eye_warning_duration_seconds": current_duration,
            "sustained_eye_warning": sustained,
            "moderate_or_strong_eye_evidence": moderate_or_strong,
        }

    def _close_eye_warning_interval(self, now: float) -> None:
        duration = (
            0.0
            if self.current_eye_warning_interval_start is None
            else now - self.current_eye_warning_interval_start
        )
        sustained = (
            duration >= SUSTAINED_EYE_WARNING_MIN_SECONDS
            or self.current_eye_warning_frames >= SUSTAINED_EYE_WARNING_MIN_FRAMES
        )
        moderate_or_strong = self.current_eye_warning_has_moderate_or_strong

        if (
            (sustained or not RECENT_EYE_WARNING_REMINDER_REQUIRES_SUSTAINED)
            and (
                moderate_or_strong
                or not RECENT_EYE_WARNING_REMINDER_REQUIRES_MODERATE_OR_STRONG
            )
        ):
            self.last_sustained_eye_warning_end_monotonic = now
            self.last_sustained_eye_warning_duration_seconds = duration
            self.last_sustained_eye_warning_peak_strength = (
                self.current_eye_warning_peak_strength
            )
            self.last_sustained_eye_warning_peak_p_eye_closed = (
                self.current_eye_warning_peak_p_eye_closed
            )

        self.eye_warning_active = False
        self.eye_warning_exit_consecutive_frames = 0
        self.current_eye_warning_interval_start = None
        self.current_eye_warning_frames = 0
        self.current_eye_warning_peak_p_eye_closed = None
        self.current_eye_warning_peak_strength = "unavailable"
        self.current_eye_warning_has_moderate_or_strong = False

    def _extract_frame(self, frame_result: dict[str, Any], now: float) -> dict[str, Any]:
        eye = frame_result.get("eye") if isinstance(frame_result.get("eye"), dict) else {}
        mouth = frame_result.get("mouth") if isinstance(frame_result.get("mouth"), dict) else {}
        face = frame_result.get("face") if isinstance(frame_result.get("face"), dict) else {}
        signal = (
            frame_result.get("signal_quality")
            if isinstance(frame_result.get("signal_quality"), dict)
            else {}
        )

        mean_p_eye_closed = safe_float(eye.get("mean_p_eye_closed"))
        p_yawn = safe_float(mouth.get("p_yawn"))
        eye_available = bool(eye.get("available"))
        mouth_available = bool(mouth.get("available"))
        face_detected = bool(face.get("detected"))
        signal_status = str(signal.get("status") or "unknown")
        tracking_status = str(face.get("tracking_status") or "unknown")
        no_face_or_tracking_failure = not face_detected or tracking_status != "ok"
        required_roi_unavailable = not eye_available or not mouth_available
        signal_failure = (
            no_face_or_tracking_failure
            or required_roi_unavailable
            or signal_status != "ok"
        )
        eye_closed_binary = (
            mean_p_eye_closed >= EYE_CLOSED_THRESHOLD
            if mean_p_eye_closed is not None and eye_available
            else None
        )
        yawn_event = (
            p_yawn >= YAWN_ON_THRESHOLD
            if p_yawn is not None and mouth_available
            else False
        )

        return {
            "frame_id": frame_result.get("frame_id"),
            "server_timestamp": frame_result.get("server_received_at"),
            "client_timestamp_ms": frame_result.get("client_timestamp_ms"),
            "received_monotonic": now,
            "mean_p_eye_closed": mean_p_eye_closed,
            "p_yawn": p_yawn,
            "face_detected": face_detected,
            "eye_available": eye_available,
            "mouth_available": mouth_available,
            "signal_quality_status": signal_status,
            "eye_evidence_strength": classify_eye_evidence(mean_p_eye_closed)
            if eye_available
            else "unavailable",
            "yawn_event": yawn_event,
            "eye_closed_binary": eye_closed_binary,
            "no_face_or_tracking_failure": no_face_or_tracking_failure,
            "required_roi_unavailable": required_roi_unavailable,
            "signal_failure": signal_failure,
        }

    def _fuse(
        self,
        *,
        signal_unreliable: bool,
        mouth_active: bool,
        recent_yawn_event: bool,
        recent_yawn_reminder: bool,
        recent_eye_warning_reminder: bool,
        eye_warning_candidate: bool,
        sustained_eye_warning: bool,
        moderate_or_strong_eye_evidence: bool,
    ) -> tuple[str, bool, str]:
        high_confidence_ready = (
            recent_yawn_event
            and eye_warning_candidate
            and sustained_eye_warning
            and moderate_or_strong_eye_evidence
        )
        suppressed = recent_yawn_event and eye_warning_candidate and not high_confidence_ready

        if signal_unreliable and not mouth_active:
            if recent_yawn_event:
                reason = (
                    "Signal is unreliable because recent face/ROI evidence is unstable. "
                    "Current mouth activity is no longer active, but recent yawn context is retained briefly for fusion."
                )
            elif recent_yawn_reminder:
                reason = (
                    "Signal is unreliable because recent face/ROI evidence is unstable. "
                    "A recent yawn is shown as a reminder only and does not keep the main warning state active."
                )
            else:
                reason = "Signal is unreliable because recent face/ROI evidence is unstable."
            return (
                "signal_unreliable",
                suppressed,
                reason,
            )

        if high_confidence_ready:
            return (
                "high_confidence_drowsiness_candidate",
                False,
                "High-confidence warning candidate is a rule-based review cue, not final drowsiness truth.",
            )

        if mouth_active:
            return (
                "mouth_warning_candidate",
                suppressed,
                "Mouth-warning candidate is based on current mouth/yawn activity.",
            )

        if eye_warning_candidate:
            reason = "Eye-warning candidate is based on temporal eye evidence, not a system-level conclusion."
            if suppressed:
                reason = (
                    "Eye-warning candidate is active. Current mouth activity is no longer active, "
                    "but recent yawn context is retained briefly for fusion; high-confidence escalation is suppressed."
                )
            return (
                "eye_warning_candidate",
                suppressed,
                reason,
            )

        if recent_eye_warning_reminder:
            return (
                "normal",
                False,
                "Recent sustained eye-warning candidate is shown as a reminder only.",
            )

        if recent_yawn_event:
            return (
                "normal",
                False,
                "Current mouth activity is no longer active, but recent yawn context is retained briefly for fusion.",
            )

        if recent_yawn_reminder:
            return (
                "normal",
                False,
                "A recent yawn is shown as a reminder only and does not keep the main warning state active.",
            )

        return (
            "normal",
            False,
            "No realtime warning-candidate state is active in the current temporal window.",
        )
