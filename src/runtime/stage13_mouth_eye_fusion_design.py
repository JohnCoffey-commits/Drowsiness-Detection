#!/usr/bin/env python3
"""Stage 13 rule-based mouth-eye fusion design prototype.

This script consumes Stage 12 eye alert timelines and either a real synchronized
mouth/yawn timeline or clearly marked synthetic design-demo mouth timelines.
It does not train models, modify checkpoints, or claim final drowsiness accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_STAGE12_OUTPUT_DIR = Path("outputs/stage12_eye_alert_rule_analysis")
DEFAULT_OUTPUT_DIR = Path("outputs/stage13_mouth_eye_fusion_design")
DEFAULT_REPORT_PATH = Path("reports/stage13_mouth_eye_fusion_design_report.md")
DEFAULT_VIDEOS = (
    "A_normal_open_baseline,"
    "B_realistic_drowsy_simulation,"
    "C_mild_head_motion,"
    "D_controlled_long_open_closed"
)

FUSION_RULES = {
    "F1_eye_only_baseline": "Eye-only baseline",
    "F2_mouth_only_baseline": "Mouth-only baseline",
    "F3_or_fusion": "OR fusion",
    "F4_and_near_window_fusion": "AND/near-window fusion",
    "F5_tiered_quality_aware_fusion": "Recommended tiered quality-aware fusion",
}

FUSION_STATES = [
    "normal",
    "eye_warning_candidate",
    "mouth_warning_candidate",
    "high_confidence_drowsiness_candidate",
    "signal_unreliable",
]

REQUIRED_MOUTH_COLUMNS = {
    "video_slug",
    "timestamp_sec",
    "p_yawn",
    "yawn_event",
    "recent_yawn_event",
    "mouth_signal_status",
    "mouth_source",
    "notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype rule-based Stage 13 mouth-eye fusion design."
    )
    parser.add_argument("--stage12-output-dir", type=Path, default=DEFAULT_STAGE12_OUTPUT_DIR)
    parser.add_argument("--mouth-timeline", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recent-yawn-window-sec", type=float, default=8.0)
    parser.add_argument("--yawn-threshold", type=float, default=0.50)
    parser.add_argument(
        "--demo-mode",
        choices=["no_yawn", "synthetic_yawn", "both"],
        default="both",
    )
    parser.add_argument("--videos", default=DEFAULT_VIDEOS)
    return parser.parse_args()


def parse_videos(value: str) -> list[str]:
    videos = [item.strip() for item in value.split(",") if item.strip()]
    if not videos:
        raise ValueError("At least one video slug is required.")
    return videos


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "timelines": output_dir / "timelines",
        "synthetic_mouth_timelines": output_dir / "synthetic_mouth_timelines",
        "figures": output_dir / "figures",
        "reports": Path("reports"),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    payload = json_safe(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


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


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_stage12_timeline(stage12_output_dir: Path, slug: str) -> pd.DataFrame:
    path = stage12_output_dir / f"stage12_video_alert_timeline_{slug}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 12 alert timeline: {path}")

    df = pd.read_csv(path)
    required = {
        "video_slug",
        "frame_index",
        "timestamp_sec",
        "signal_unreliable",
        "mean_p_eye_closed",
        "rolling_perclos_mean_binary",
        "recommended_alert",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["video_slug"] = slug
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64")
    df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
    df["signal_unreliable"] = df["signal_unreliable"].map(to_bool)
    df["eye_warning_candidate"] = df["recommended_alert"].map(to_bool)
    df["p_eye_closed"] = pd.to_numeric(df["mean_p_eye_closed"], errors="coerce")
    df["rolling_perclos_mean_binary"] = pd.to_numeric(
        df["rolling_perclos_mean_binary"], errors="coerce"
    )
    df["eye_state"] = "normal"
    df.loc[df["eye_warning_candidate"], "eye_state"] = "eye_warning_candidate"
    df.loc[df["signal_unreliable"], "eye_state"] = "signal_unreliable"
    return df.sort_values("timestamp_sec").reset_index(drop=True)


def add_recent_yawn_event(df: pd.DataFrame, recent_window_sec: float) -> pd.DataFrame:
    df = df.sort_values("timestamp_sec").reset_index(drop=True).copy()
    last_event_time: float | None = None
    recent_flags: list[bool] = []
    for _, row in df.iterrows():
        timestamp = float(row["timestamp_sec"])
        if to_bool(row.get("yawn_event", False)):
            last_event_time = timestamp
        recent_flags.append(
            last_event_time is not None and (timestamp - last_event_time) <= recent_window_sec
        )
    df["recent_yawn_event"] = recent_flags
    return df


def synthetic_event_timestamps(eye_df: pd.DataFrame, slug: str, demo_mode: str) -> list[float]:
    if demo_mode == "no_yawn":
        return []
    if slug != "B_realistic_drowsy_simulation":
        return []

    alerts = eye_df[eye_df["eye_warning_candidate"]].copy()
    if alerts.empty:
        fallback = eye_df["timestamp_sec"].dropna()
        return [float(fallback.iloc[len(fallback) // 2])] if len(fallback) else []

    starts: list[float] = []
    previous_index: int | None = None
    for _, row in alerts.iterrows():
        frame_index = int(row["frame_index"]) if not pd.isna(row["frame_index"]) else None
        if previous_index is None or frame_index is None or frame_index - previous_index > 5:
            starts.append(float(row["timestamp_sec"]))
        previous_index = frame_index
        if len(starts) >= 2:
            break
    return starts


def make_synthetic_mouth_timeline(
    eye_df: pd.DataFrame,
    slug: str,
    demo_mode: str,
    recent_yawn_window_sec: float,
    yawn_threshold: float,
) -> pd.DataFrame:
    event_times = synthetic_event_timestamps(eye_df, slug, demo_mode)
    rows: list[dict[str, Any]] = []
    for _, row in eye_df.iterrows():
        timestamp = float(row["timestamp_sec"])
        is_event = any(abs(timestamp - event_time) < 1e-6 for event_time in event_times)
        rows.append(
            {
                "video_slug": slug,
                "timestamp_sec": timestamp,
                "frame_index": int(row["frame_index"]) if not pd.isna(row["frame_index"]) else None,
                "p_yawn": 0.92 if is_event else 0.05,
                "yawn_event": is_event,
                "recent_yawn_event": False,
                "mouth_signal_status": "ok",
                "mouth_source": "synthetic_design_demo",
                "notes": (
                    "synthetic yawn event for Stage 13 design demo"
                    if is_event
                    else "synthetic no-yawn baseline for Stage 13 design demo"
                ),
            }
        )
    mouth_df = pd.DataFrame(rows)
    mouth_df["yawn_event"] = (
        pd.to_numeric(mouth_df["p_yawn"], errors="coerce").fillna(0.0) >= yawn_threshold
    )
    return add_recent_yawn_event(mouth_df, recent_yawn_window_sec)


def validate_mouth_timeline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing mouth timeline CSV: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_MOUTH_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required mouth timeline columns: {sorted(missing)}")
    df = df.copy()
    df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="raise")
    if "frame_index" in df.columns:
        df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64")
    df["p_yawn"] = pd.to_numeric(df["p_yawn"], errors="raise")
    df["yawn_event"] = df["yawn_event"].map(to_bool)
    df["recent_yawn_event"] = df["recent_yawn_event"].map(to_bool)
    return df.sort_values(["video_slug", "timestamp_sec"]).reset_index(drop=True)


def align_real_mouth_timeline(eye_df: pd.DataFrame, mouth_all: pd.DataFrame, slug: str) -> pd.DataFrame:
    mouth_df = mouth_all[mouth_all["video_slug"] == slug].copy()
    if mouth_df.empty:
        raise ValueError(f"No real mouth timeline rows found for video_slug={slug}")

    eye_sorted = eye_df.sort_values("timestamp_sec").copy()
    mouth_sorted = mouth_df.sort_values("timestamp_sec").copy()
    aligned = pd.merge_asof(
        eye_sorted,
        mouth_sorted,
        on="timestamp_sec",
        direction="nearest",
        suffixes=("", "_mouth"),
    )

    for col in REQUIRED_MOUTH_COLUMNS:
        if col not in aligned.columns:
            aligned[col] = "missing" if col.endswith("status") else pd.NA
    if "frame_index_mouth" in aligned.columns:
        aligned["mouth_frame_index"] = aligned["frame_index_mouth"]
    return aligned[
        [
            "video_slug",
            "timestamp_sec",
            "frame_index",
            "p_yawn",
            "yawn_event",
            "recent_yawn_event",
            "mouth_signal_status",
            "mouth_source",
            "notes",
        ]
    ].copy()


def apply_fusion_rule(row: pd.Series, rule_name: str) -> tuple[str, str]:
    eye_unreliable = to_bool(row["signal_unreliable"])
    eye_warning = to_bool(row["eye_warning_candidate"])
    recent_yawn = to_bool(row["recent_yawn_event"])
    mouth_status = str(row.get("mouth_signal_status", "ok"))
    mouth_unreliable = mouth_status not in {"ok", "synthetic_ok"}

    if rule_name == "F1_eye_only_baseline":
        if eye_unreliable:
            return "signal_unreliable", "eye signal unreliable"
        if eye_warning:
            return "eye_warning_candidate", "eye-only Stage 12 warning candidate"
        return "normal", "no eye warning candidate"

    if rule_name == "F2_mouth_only_baseline":
        if mouth_unreliable:
            return "signal_unreliable", "mouth signal unreliable"
        if recent_yawn:
            return "mouth_warning_candidate", "recent yawn event"
        return "normal", "no recent yawn event"

    if rule_name == "F3_or_fusion":
        if eye_warning and recent_yawn:
            return (
                "high_confidence_drowsiness_candidate",
                "eye warning candidate and recent yawn event",
            )
        if eye_warning:
            return "eye_warning_candidate", "eye warning candidate"
        if recent_yawn:
            return "mouth_warning_candidate", "recent yawn event"
        if eye_unreliable or mouth_unreliable:
            return "signal_unreliable", "one or more signals unreliable"
        return "normal", "no warning candidate"

    if rule_name == "F4_and_near_window_fusion":
        if eye_warning and recent_yawn:
            return (
                "high_confidence_drowsiness_candidate",
                "eye warning candidate and recent yawn event",
            )
        if eye_unreliable and not recent_yawn:
            return "signal_unreliable", "eye signal unreliable without mouth support"
        return "normal", "AND condition not met"

    if rule_name == "F5_tiered_quality_aware_fusion":
        if eye_unreliable and not recent_yawn:
            return "signal_unreliable", "eye signal unreliable and no recent yawn"
        if eye_unreliable and recent_yawn:
            return (
                "mouth_warning_candidate",
                "recent yawn event while eye signal is unreliable",
            )
        if eye_warning and recent_yawn:
            return (
                "high_confidence_drowsiness_candidate",
                "eye warning candidate and recent yawn event",
            )
        if eye_warning:
            return "eye_warning_candidate", "eye warning candidate"
        if recent_yawn:
            return "mouth_warning_candidate", "recent yawn event"
        return "normal", "no warning candidate"

    raise ValueError(f"Unknown fusion rule: {rule_name}")


def consecutive_run_metrics(states: pd.Series, warning_states: set[str]) -> dict[str, int]:
    longest = 0
    count = 0
    current = 0
    in_run = False
    for state in states:
        is_warning = state in warning_states
        if is_warning:
            current += 1
            if not in_run:
                count += 1
                in_run = True
            longest = max(longest, current)
        else:
            current = 0
            in_run = False
    return {"alert_count": count, "longest_run": longest}


def build_fusion_timeline(
    eye_df: pd.DataFrame,
    mouth_df: pd.DataFrame,
    slug: str,
    rule_name: str,
) -> pd.DataFrame:
    merged = eye_df.merge(
        mouth_df,
        on=["video_slug", "timestamp_sec", "frame_index"],
        how="left",
        suffixes=("", "_mouth"),
    )
    merged["p_yawn"] = pd.to_numeric(merged["p_yawn"], errors="coerce").fillna(0.0)
    merged["yawn_event"] = merged["yawn_event"].map(to_bool)
    merged["recent_yawn_event"] = merged["recent_yawn_event"].map(to_bool)
    merged["mouth_signal_status"] = merged["mouth_signal_status"].fillna("missing")
    merged["mouth_source"] = merged["mouth_source"].fillna("missing")
    merged["notes"] = merged["notes"].fillna("")

    states: list[str] = []
    reasons: list[str] = []
    for _, row in merged.iterrows():
        state, reason = apply_fusion_rule(row, rule_name)
        states.append(state)
        reasons.append(reason)

    merged["fusion_rule"] = rule_name
    merged["fusion_state"] = states
    merged["fusion_reason"] = reasons
    merged["mouth_warning_candidate"] = merged["recent_yawn_event"].map(to_bool)
    merged["mouth_state"] = "normal"
    merged.loc[merged["mouth_warning_candidate"], "mouth_state"] = "mouth_warning_candidate"
    merged.loc[
        ~merged["mouth_signal_status"].isin(["ok", "synthetic_ok"]),
        "mouth_state",
    ] = "signal_unreliable"
    merged["signal_quality"] = "ok"
    merged.loc[merged["signal_unreliable"], "signal_quality"] = "eye_unreliable"
    merged.loc[
        ~merged["mouth_signal_status"].isin(["ok", "synthetic_ok"]),
        "signal_quality",
    ] = "mouth_unreliable"
    merged["high_confidence_drowsiness_candidate"] = (
        merged["fusion_state"] == "high_confidence_drowsiness_candidate"
    )

    return merged[
        [
            "video_slug",
            "timestamp_sec",
            "frame_index",
            "eye_state",
            "mouth_state",
            "fusion_state",
            "fusion_reason",
            "signal_quality",
            "p_eye_closed",
            "p_yawn",
            "recent_yawn_event",
            "eye_warning_candidate",
            "mouth_warning_candidate",
            "high_confidence_drowsiness_candidate",
            "mouth_source",
            "fusion_rule",
        ]
    ].copy()


def summarize_rule_timeline(
    timeline: pd.DataFrame,
    slug: str,
    rule_name: str,
    uses_real_mouth_timeline: bool,
    uses_synthetic_mouth_timeline: bool,
) -> dict[str, Any]:
    state_counts = {state: int((timeline["fusion_state"] == state).sum()) for state in FUSION_STATES}
    warning_states = {
        "eye_warning_candidate",
        "mouth_warning_candidate",
        "high_confidence_drowsiness_candidate",
    }
    any_warning = timeline["fusion_state"].isin(warning_states)
    high_conf = timeline["fusion_state"] == "high_confidence_drowsiness_candidate"
    warning_metrics = consecutive_run_metrics(timeline["fusion_state"], warning_states)
    high_conf_metrics = consecutive_run_metrics(
        timeline["fusion_state"], {"high_confidence_drowsiness_candidate"}
    )
    warning_timestamps = timeline.loc[any_warning, "timestamp_sec"]

    notes = []
    if uses_synthetic_mouth_timeline:
        notes.append("synthetic mouth timeline; design demo only")
    if rule_name == "F5_tiered_quality_aware_fusion":
        notes.append("recommended tiered quality-aware rule")

    return {
        "video_slug": slug,
        "rule_name": rule_name,
        "rule_label": FUSION_RULES[rule_name],
        "fusion_state_counts": json.dumps(state_counts, sort_keys=True),
        "normal_frames": state_counts["normal"],
        "eye_warning_candidate_frames": state_counts["eye_warning_candidate"],
        "mouth_warning_candidate_frames": state_counts["mouth_warning_candidate"],
        "high_confidence_drowsiness_candidate_frames": state_counts[
            "high_confidence_drowsiness_candidate"
        ],
        "signal_unreliable_frames": state_counts["signal_unreliable"],
        "alert_count": warning_metrics["alert_count"],
        "longest_high_confidence_run": high_conf_metrics["longest_run"],
        "longest_any_warning_run": warning_metrics["longest_run"],
        "first_warning_timestamp_sec": (
            float(warning_timestamps.min()) if len(warning_timestamps) else None
        ),
        "last_warning_timestamp_sec": (
            float(warning_timestamps.max()) if len(warning_timestamps) else None
        ),
        "uses_real_mouth_timeline": uses_real_mouth_timeline,
        "uses_synthetic_mouth_timeline": uses_synthetic_mouth_timeline,
        "notes": "; ".join(notes),
    }


def write_schema(path: Path) -> None:
    text = """# Stage 13 Fusion Schema

## Required Mouth Timeline Columns

| Column | Required | Description |
| --- | --- | --- |
| `video_slug` | Yes | Scenario/video identifier matching the eye timeline slug. |
| `timestamp_sec` | Yes | Timestamp in seconds for timeline alignment. |
| `frame_index` | Optional | Frame index for fallback alignment. |
| `p_yawn` | Yes | Mouth/yawn probability or score. |
| `yawn_event` | Yes | Boolean or 0/1 yawn event marker. |
| `recent_yawn_event` | Yes | Boolean or 0/1 recent-yawn marker. |
| `mouth_signal_status` | Yes | Signal status such as `ok`, `missing`, or `unreliable`. |
| `mouth_source` | Yes | Source such as `runtime_yawn_model` or `synthetic_design_demo`. |
| `notes` | Yes | Provenance or caveat notes. |

## Fusion Timeline Columns

| Column | Description |
| --- | --- |
| `video_slug` | Scenario/video identifier. |
| `timestamp_sec` | Timeline timestamp in seconds. |
| `frame_index` | Frame index when available. |
| `eye_state` | Eye state from Stage 12. |
| `mouth_state` | Mouth/yawn state. |
| `fusion_state` | Rule-based fused state. |
| `fusion_reason` | Human-readable rule reason. |
| `signal_quality` | Signal-quality summary. |
| `p_eye_closed` | Eye closed probability proxy. |
| `p_yawn` | Mouth/yawn probability or score. |
| `recent_yawn_event` | Boolean recent-yawn marker. |
| `eye_warning_candidate` | Boolean eye warning candidate. |
| `mouth_warning_candidate` | Boolean mouth warning candidate. |
| `high_confidence_drowsiness_candidate` | Boolean high-confidence candidate. |

This schema supports design/prototype fusion. It is not final drowsiness accuracy.
"""
    path.write_text(text, encoding="utf-8")


def plot_state_counts(comparison: pd.DataFrame, output_path: Path) -> None:
    recommended = comparison[comparison["rule_name"] == "F5_tiered_quality_aware_fusion"].copy()
    if recommended.empty:
        return
    pivot = recommended.set_index("video_slug")[
        [
            "normal_frames",
            "eye_warning_candidate_frames",
            "mouth_warning_candidate_frames",
            "high_confidence_drowsiness_candidate_frames",
            "signal_unreliable_frames",
        ]
    ]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title("Stage 13 F5 Fusion State Counts by Video")
    ax.set_xlabel("Video")
    ax.set_ylabel("Sampled frames")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_video_timeline(timeline: pd.DataFrame, output_path: Path, slug: str) -> None:
    state_map = {
        "normal": 0,
        "eye_warning_candidate": 1,
        "mouth_warning_candidate": 2,
        "high_confidence_drowsiness_candidate": 3,
        "signal_unreliable": -1,
    }
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(
        timeline["timestamp_sec"],
        timeline["p_eye_closed"],
        label="p_eye_closed",
        color="#2f5597",
        linewidth=1.5,
    )
    ax1.scatter(
        timeline.loc[timeline["recent_yawn_event"], "timestamp_sec"],
        timeline.loc[timeline["recent_yawn_event"], "p_yawn"],
        label="recent_yawn_event",
        color="#c55a11",
        s=20,
    )
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Probability / score")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.step(
        timeline["timestamp_sec"],
        timeline["fusion_state"].map(state_map),
        label="fusion_state",
        color="#548235",
        where="post",
        alpha=0.75,
    )
    ax2.set_yticks([-1, 0, 1, 2, 3])
    ax2.set_yticklabels(
        [
            "signal_unreliable",
            "normal",
            "eye_warning",
            "mouth_warning",
            "high_conf",
        ],
        fontsize=8,
    )
    ax2.set_ylabel("Fusion state")
    ax1.set_title(f"Stage 13 F5 Fusion Timeline: {slug}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in table.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                text = ""
            elif isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    summary: dict[str, Any],
    uses_synthetic_mouth_timeline: bool,
    output_dir: Path,
) -> None:
    recommended = comparison[comparison["rule_name"] == "F5_tiered_quality_aware_fusion"]
    comparison_cols = [
        "video_slug",
        "rule_name",
        "normal_frames",
        "eye_warning_candidate_frames",
        "mouth_warning_candidate_frames",
        "high_confidence_drowsiness_candidate_frames",
        "signal_unreliable_frames",
        "alert_count",
        "longest_any_warning_run",
    ]
    recommended_cols = [
        "video_slug",
        "normal_frames",
        "eye_warning_candidate_frames",
        "mouth_warning_candidate_frames",
        "high_confidence_drowsiness_candidate_frames",
        "signal_unreliable_frames",
        "alert_count",
        "longest_any_warning_run",
    ]

    synthetic_note = (
        "No real synchronized `p_yawn` timelines were found, so this run used synthetic "
        "mouth timelines for design demonstration only. Stage 13 is fusion design plus "
        "offline prototype, not validated runtime fusion."
        if uses_synthetic_mouth_timeline
        else "A real mouth timeline CSV was provided for this run."
    )

    text = f"""# Stage 13 Mouth-Eye Fusion Design Report

## 1. Purpose

Stage 13 designs rule-based mouth-eye fusion for the modular drowsiness monitoring system. It is not a trained fusion classifier, not final system-level drowsiness accuracy, and not deployment readiness.

## 2. Mouth/Yawn Runtime Audit Summary

The Stage 13 audit found no existing Python runtime mouth/yawn inference pipeline, no verified local runtime-ready mouth/yawn checkpoint path, no `p_yawn` timelines for A/B/C/D, and no synchronized mouth-eye timelines for A/B/C/D.

{synthetic_note}

## 3. Inputs

- Stage 12 eye alert timelines: `outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_<slug>.csv`
- Mouth timeline source: {"synthetic design demo" if uses_synthetic_mouth_timeline else "real provided CSV"}
- Stage 13 output directory: `{output_dir}`

## 4. Fusion States

- `normal`: no warning candidate.
- `eye_warning_candidate`: Stage 12 eye-only temporal warning is active.
- `mouth_warning_candidate`: recent mouth/yawn signal is active.
- `high_confidence_drowsiness_candidate`: eye warning and recent yawn co-occur.
- `signal_unreliable`: signal quality is insufficient; this must not be counted as drowsiness.

## 5. Fusion Rules Compared

- F1 eye-only baseline: uses the Stage 12 eye warning and preserves eye signal unreliability.
- F2 mouth-only baseline: uses recent yawn events only.
- F3 OR fusion: emits a warning when either eye or mouth warning is present.
- F4 AND/near-window fusion: emits high-confidence only when eye warning and recent yawn co-occur.
- F5 recommended tiered quality-aware fusion: preserves unreliable eye intervals, supports eye-only and mouth-only warnings, and upgrades only on eye-mouth co-occurrence.

## 6. Rule Comparison Table

{markdown_table(comparison, comparison_cols)}

## 7. Recommended Rule

Recommended rule: `F5_tiered_quality_aware_fusion`.

This rule is selected because it:

- Preserves `signal_unreliable` instead of treating tracking failure as drowsiness.
- Supports an eye-only warning candidate when the Stage 12 eye rule is active.
- Supports a mouth-only warning candidate when a recent yawn exists and the eye signal is not usable.
- Upgrades to `high_confidence_drowsiness_candidate` only when eye warning and recent yawn co-occur.

Recommended-rule summary:

{markdown_table(recommended, recommended_cols)}

## 8. Scenario-Level Interpretation

- `A_normal_open_baseline`: should stay mostly normal in demo mode.
- `B_realistic_drowsy_simulation`: may become high-confidence when synthetic yawn events overlap eye-warning periods.
- `C_mild_head_motion`: is a mixed fatigue-like eye closure, mild head motion, and partial occlusion scenario; signal-unreliable intervals should remain quality markers rather than drowsiness labels.
- `D_controlled_long_open_closed`: should remain eye-warning driven unless a yawn event is present.

## 9. Limitations

- If synthetic mouth timelines are used, Stage 13 is not validated synchronized fusion.
- No real `p_yawn` runtime timelines for A/B/C/D were found in this audit.
- No ground-truth drowsiness timeline is used.
- No trained fusion classifier is used.
- The validation set is small.
- This is not final system-level drowsiness accuracy.

## 10. Next Steps

1. Implement real runtime mouth/yawn inference on the same videos.
2. Generate synchronized `p_yawn` timelines.
3. Rerun Stage 13 with real mouth timelines.
4. Optionally collect synchronized labeled data for future fusion classifier extension.

## Machine-Readable Summary

- Stage: {summary["stage"]}
- Status: {summary["status"]}
- Recommended rule: `{summary["recommended_rule_name"]}`
- Uses synthetic mouth timeline: {summary["uses_synthetic_mouth_timeline"]}
- Warning: {summary["warning"]}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    videos = parse_videos(args.videos)
    dirs = ensure_output_dirs(args.output_dir)

    uses_real_mouth_timeline = args.mouth_timeline is not None
    uses_synthetic_mouth_timeline = not uses_real_mouth_timeline
    real_mouth = validate_mouth_timeline(args.mouth_timeline) if args.mouth_timeline else None

    write_schema(args.output_dir / "fusion_schema.md")

    comparison_rows: list[dict[str, Any]] = []
    recommended_timelines: dict[str, pd.DataFrame] = {}
    synthetic_paths: list[str] = []

    for slug in videos:
        eye_df = load_stage12_timeline(args.stage12_output_dir, slug)
        if real_mouth is not None:
            mouth_df = align_real_mouth_timeline(eye_df, real_mouth, slug)
        else:
            mouth_df = make_synthetic_mouth_timeline(
                eye_df=eye_df,
                slug=slug,
                demo_mode=args.demo_mode,
                recent_yawn_window_sec=args.recent_yawn_window_sec,
                yawn_threshold=args.yawn_threshold,
            )
            synthetic_path = (
                dirs["synthetic_mouth_timelines"] / f"synthetic_mouth_timeline_{slug}.csv"
            )
            mouth_df.to_csv(synthetic_path, index=False)
            synthetic_paths.append(str(synthetic_path))

        for rule_name in FUSION_RULES:
            timeline = build_fusion_timeline(eye_df, mouth_df, slug, rule_name)
            comparison_rows.append(
                summarize_rule_timeline(
                    timeline=timeline,
                    slug=slug,
                    rule_name=rule_name,
                    uses_real_mouth_timeline=uses_real_mouth_timeline,
                    uses_synthetic_mouth_timeline=uses_synthetic_mouth_timeline,
                )
            )
            if rule_name == "F5_tiered_quality_aware_fusion":
                recommended_timelines[slug] = timeline
                timeline_path = dirs["timelines"] / f"fusion_timeline_{slug}.csv"
                timeline.to_csv(timeline_path, index=False)

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = args.output_dir / "stage13_fusion_rule_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    figure_paths = [
        args.output_dir / "figures" / "fusion_state_counts_by_video.png",
    ]
    plot_state_counts(comparison, figure_paths[0])
    for slug, timeline in recommended_timelines.items():
        figure_path = args.output_dir / "figures" / f"fusion_timeline_{slug}.png"
        plot_video_timeline(timeline, figure_path, slug)
        figure_paths.append(figure_path)

    recommended_rows = comparison[comparison["rule_name"] == "F5_tiered_quality_aware_fusion"]
    summary = {
        "stage": 13,
        "status": (
            "DESIGN_PROTOTYPE_SYNTHETIC_MOUTH_TIMELINE"
            if uses_synthetic_mouth_timeline
            else "DESIGN_WITH_REAL_MOUTH_TIMELINE"
        ),
        "recommended_rule_name": "F5_tiered_quality_aware_fusion",
        "recommended_rule_parameters": {
            "eye_rule": "Stage 12 quality_gated_perclos_mean_ge_0.60_consec",
            "recent_yawn_window_sec": args.recent_yawn_window_sec,
            "yawn_threshold": args.yawn_threshold,
            "tiered_quality_aware": True,
        },
        "validation_videos": videos,
        "uses_real_mouth_timeline": uses_real_mouth_timeline,
        "uses_synthetic_mouth_timeline": uses_synthetic_mouth_timeline,
        "synthetic_mouth_timeline_paths": synthetic_paths,
        "rule_comparison_csv": str(comparison_path),
        "recommended_timeline_paths": {
            slug: str(dirs["timelines"] / f"fusion_timeline_{slug}.csv") for slug in videos
        },
        "figure_paths": [str(path) for path in figure_paths],
        "recommended_rule_metrics": recommended_rows.to_dict(orient="records"),
        "limitations": [
            "Stage 13 is design/prototype only when synthetic mouth timelines are used.",
            "No real synchronized p_yawn timelines for A/B/C/D were found in the audit.",
            "No final system-level drowsiness accuracy is claimed.",
            "No trained fusion classifier is used.",
        ],
        "next_missing_piece_before_true_synchronized_fusion_validation": (
            "Implement real runtime mouth/yawn inference and generate synchronized p_yawn timelines "
            "for the same videos."
        ),
        "next_stage_recommendation": (
            "Generate real mouth/yawn timelines, then rerun Stage 13 before any deployment or final "
            "accuracy claim."
        ),
        "warning": "This is not final system-level drowsiness accuracy.",
    }
    write_json(args.output_dir / "stage13_fusion_summary.json", summary)

    write_report(
        args.output_dir / "STAGE13_MOUTH_EYE_FUSION_REPORT.md",
        comparison,
        summary,
        uses_synthetic_mouth_timeline,
        args.output_dir,
    )
    write_report(
        DEFAULT_REPORT_PATH,
        comparison,
        summary,
        uses_synthetic_mouth_timeline,
        args.output_dir,
    )

    print(f"Wrote Stage 13 fusion comparison: {comparison_path}")
    print(f"Wrote Stage 13 summary: {args.output_dir / 'stage13_fusion_summary.json'}")
    if uses_synthetic_mouth_timeline:
        print("Used synthetic mouth timelines for design demonstration only.")
    else:
        print(f"Used real mouth timeline: {args.mouth_timeline}")
    print("Warning: this is not final system-level drowsiness accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
