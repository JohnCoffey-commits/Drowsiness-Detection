#!/usr/bin/env python3
"""Stage 15 real synchronized mouth-eye fusion validation.

This wrapper combines real Stage 14 model-generated mouth/yawn timelines with
real Stage 12 eye alert timelines, then reuses the Stage 13 rule-based fusion
logic. It does not train models, modify checkpoints, or use synthetic/manual
mouth timelines for fusion decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.stage13_mouth_eye_fusion_design import (  # noqa: E402
    FUSION_RULES,
    FUSION_STATES,
    align_real_mouth_timeline,
    build_fusion_timeline,
    load_stage12_timeline,
    markdown_table,
    summarize_rule_timeline,
    validate_mouth_timeline,
)


DEFAULT_STAGE12_OUTPUT_DIR = Path("outputs/stage12_eye_alert_rule_analysis")
DEFAULT_STAGE14_ROOT_PREFIX = "outputs/stage14_mouth_yawn_runtime_"
DEFAULT_OUTPUT_DIR = Path("outputs/stage15_real_mouth_eye_fusion")
DEFAULT_AUDIT_DIR = Path("artifacts/audits/stage15_real_mouth_eye_fusion_2026-05-09")
DEFAULT_REPORT_PATH = Path("reports/stage15_real_mouth_eye_fusion_validation_report.md")
DEFAULT_DOC_LOG_PATH = Path("docs/STAGE15_REAL_MOUTH_EYE_FUSION_LOG.md")
DEFAULT_VIDEOS = (
    "A_normal_open_baseline,"
    "B_realistic_drowsy_simulation,"
    "C_mild_head_motion,"
    "D_controlled_long_open_closed"
)
RECOMMENDED_RULE = "F5_tiered_quality_aware_fusion"
MOUTH_SOURCE = "stage14_runtime_mouth_yawn_model"


SCENARIO_EXPECTATIONS = {
    "A_normal_open_baseline": (
        "Expected mostly normal; no high-confidence candidate, no mouth warning, "
        "no yawn events, and short eye spikes suppressed."
    ),
    "B_realistic_drowsy_simulation": (
        "Expected model-generated yawn/mouth warning around the real 14.3s-16.8s "
        "yawn interval, with high-confidence candidates when recent yawn overlaps "
        "eye warning."
    ),
    "C_mild_head_motion": (
        "Expected signal-unreliable intervals from tracking/visibility issues, no "
        "mouth/yawn false positives, and no treating no-face as drowsiness."
    ),
    "D_controlled_long_open_closed": (
        "Expected eye-warning candidates during long eye closure, no mouth warning, "
        "and no high-confidence candidate without yawn/recent-yawn evidence."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 15 real synchronized rule-based mouth-eye fusion validation."
    )
    parser.add_argument("--stage12-output-dir", type=Path, default=DEFAULT_STAGE12_OUTPUT_DIR)
    parser.add_argument("--stage14-root-prefix", default=DEFAULT_STAGE14_ROOT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--recent-yawn-window-sec", type=float, default=8.0)
    parser.add_argument("--yawn-threshold", type=float, default=0.50)
    parser.add_argument("--videos", default=DEFAULT_VIDEOS)
    return parser.parse_args()


def parse_videos(value: str) -> list[str]:
    videos = [item.strip() for item in value.split(",") if item.strip()]
    if not videos:
        raise ValueError("At least one video slug is required.")
    return videos


def ensure_dirs(output_dir: Path, audit_dir: Path) -> dict[str, Path]:
    dirs = {
        "output": output_dir,
        "timelines": output_dir / "timelines",
        "figures": output_dir / "figures",
        "audit": audit_dir,
        "reports": Path("reports"),
        "docs": Path("docs"),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def stage14_dir(prefix: str, slug: str) -> Path:
    return Path(f"{prefix}{slug}")


def prediction_path(prefix: str, slug: str) -> Path:
    return stage14_dir(prefix, slug) / "runtime_mouth_yawn_predictions.csv"


def failure_path(prefix: str, slug: str) -> Path:
    return stage14_dir(prefix, slug) / "failures.csv"


def stage12_path(stage12_output_dir: Path, slug: str) -> Path:
    return stage12_output_dir / f"stage12_video_alert_timeline_{slug}.csv"


def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


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


def first_last_timestamp(df: pd.DataFrame, mask: pd.Series) -> tuple[float | None, float | None]:
    timestamps = df.loc[mask.map(to_bool), "timestamp_sec"]
    if timestamps.empty:
        return None, None
    return float(timestamps.min()), float(timestamps.max())


def plot_stage15_state_counts(comparison: pd.DataFrame, output_path: Path) -> None:
    recommended = comparison[comparison["rule_name"] == RECOMMENDED_RULE].copy()
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
    ax.set_title("Stage 15 F5 Real Mouth-Eye Fusion State Counts by Video")
    ax.set_xlabel("Video")
    ax.set_ylabel("Sampled frames")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_stage15_video_timeline(timeline: pd.DataFrame, output_path: Path, slug: str) -> None:
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
    ax1.set_title(f"Stage 15 F5 Real Mouth-Eye Fusion Timeline: {slug}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_stage14_predictions(path: Path, slug: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "video_slug",
        "frame_index",
        "timestamp_sec",
        "p_yawn",
        "yawn_event",
        "recent_yawn_event",
        "mouth_signal_status",
        "checkpoint_path",
        "model_name",
        "label_mapping",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing Stage 14 prediction columns: {sorted(missing)}")
    if set(df["video_slug"].dropna().unique()) != {slug}:
        raise ValueError(f"{path} does not contain only video_slug={slug}")
    out = pd.DataFrame(
        {
            "video_slug": slug,
            "timestamp_sec": pd.to_numeric(df["timestamp_sec"], errors="raise"),
            "frame_index": pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64"),
            "p_yawn": pd.to_numeric(df["p_yawn"], errors="raise"),
            "yawn_event": df["yawn_event"].map(to_bool),
            "recent_yawn_event": df["recent_yawn_event"].map(to_bool),
            "mouth_signal_status": df["mouth_signal_status"].fillna("missing").astype(str),
            "mouth_source": MOUTH_SOURCE,
            "notes": "model-generated Stage 14 p_yawn timeline",
        }
    )
    out["checkpoint_path"] = df["checkpoint_path"].astype(str)
    out["model_name"] = df["model_name"].astype(str)
    out["label_mapping"] = df["label_mapping"].astype(str)
    return out


def load_stage14_failures(path: Path, slug: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    required = {"video_slug", "frame_index", "timestamp_sec", "failure_type", "failure_reason"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing Stage 14 failure columns: {sorted(missing)}")
    df = df[df["video_slug"] == slug].copy()
    if df.empty:
        return pd.DataFrame()
    failure_type = df["failure_type"].fillna("mouth_signal_unavailable").astype(str)
    failure_reason = df["failure_reason"].fillna("").astype(str)
    return pd.DataFrame(
        {
            "video_slug": slug,
            "timestamp_sec": pd.to_numeric(df["timestamp_sec"], errors="raise"),
            "frame_index": pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64"),
            "p_yawn": 0.0,
            "yawn_event": False,
            "recent_yawn_event": False,
            "mouth_signal_status": failure_type,
            "mouth_source": MOUTH_SOURCE,
            "notes": [
                f"Stage 14 {ft} failure; no p_yawn generated; not treated as yawn"
                + (f"; reason: {fr}" if fr else "")
                for ft, fr in zip(failure_type, failure_reason, strict=False)
            ],
            "checkpoint_path": "",
            "model_name": "",
            "label_mapping": "",
        }
    )


def build_combined_mouth_timeline(
    stage14_root_prefix: str,
    videos: list[str],
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    provenance: dict[str, dict[str, Any]] = {}
    for slug in videos:
        pred_path = prediction_path(stage14_root_prefix, slug)
        fail_path = failure_path(stage14_root_prefix, slug)
        pred = load_stage14_predictions(pred_path, slug)
        fail = load_stage14_failures(fail_path, slug)
        combined = pd.concat([pred, fail], ignore_index=True, sort=False)
        combined = combined.sort_values(["timestamp_sec", "frame_index"]).reset_index(drop=True)
        rows.append(combined)

        yawn_window = pred[
            (pd.to_numeric(pred["timestamp_sec"], errors="coerce") >= 14.3)
            & (pd.to_numeric(pred["timestamp_sec"], errors="coerce") <= 16.8)
        ]
        provenance[slug] = {
            "stage14_prediction_path": str(pred_path),
            "stage14_failure_path": str(fail_path),
            "prediction_rows": int(len(pred)),
            "failure_rows_added_to_mouth_timeline": int(len(fail)),
            "combined_rows": int(len(combined)),
            "yawn_event_count": int(pred["yawn_event"].sum()),
            "mouth_signal_status_counts": combined["mouth_signal_status"].value_counts().to_dict(),
            "model_names": sorted(set(pred["model_name"].dropna().astype(str))),
            "checkpoint_paths": sorted(set(pred["checkpoint_path"].dropna().astype(str))),
            "label_mappings": sorted(set(pred["label_mapping"].dropna().astype(str))),
            "b_yawn_interval_rows": int(len(yawn_window)) if slug == "B_realistic_drowsy_simulation" else None,
            "b_yawn_interval_event_rows": (
                int(yawn_window["yawn_event"].sum()) if slug == "B_realistic_drowsy_simulation" else None
            ),
            "b_yawn_interval_mean_p_yawn": (
                float(yawn_window["p_yawn"].mean())
                if slug == "B_realistic_drowsy_simulation" and len(yawn_window)
                else None
            ),
            "b_yawn_interval_min_p_yawn": (
                float(yawn_window["p_yawn"].min())
                if slug == "B_realistic_drowsy_simulation" and len(yawn_window)
                else None
            ),
            "b_yawn_interval_max_p_yawn": (
                float(yawn_window["p_yawn"].max())
                if slug == "B_realistic_drowsy_simulation" and len(yawn_window)
                else None
            ),
        }

    mouth = pd.concat(rows, ignore_index=True, sort=False)
    stage13_columns = [
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
    mouth[stage13_columns].to_csv(output_path, index=False)
    return mouth, provenance


def nearest_alignment_metrics(eye_df: pd.DataFrame, mouth_df: pd.DataFrame) -> dict[str, Any]:
    eye = eye_df[["timestamp_sec"]].copy()
    mouth = mouth_df[["timestamp_sec"]].rename(columns={"timestamp_sec": "mouth_timestamp_sec"})
    eye = eye.sort_values("timestamp_sec")
    mouth = mouth.sort_values("mouth_timestamp_sec")
    aligned = pd.merge_asof(
        eye,
        mouth,
        left_on="timestamp_sec",
        right_on="mouth_timestamp_sec",
        direction="nearest",
    )
    deltas = (aligned["timestamp_sec"] - aligned["mouth_timestamp_sec"]).abs()
    return {
        "eye_rows": int(len(eye_df)),
        "mouth_rows_available": int(len(mouth_df)),
        "exact_timestamp_matches": int((deltas.fillna(float("inf")) <= 1e-6).sum()),
        "max_nearest_timestamp_delta_sec": float(deltas.max()) if len(deltas) else None,
        "mean_nearest_timestamp_delta_sec": float(deltas.mean()) if len(deltas) else None,
        "aligned_by_timestamp": bool(len(deltas) and deltas.max() <= 0.001),
    }


def audit_inputs(
    stage12_output_dir: Path,
    stage14_root_prefix: str,
    videos: list[str],
    combined_mouth: pd.DataFrame,
    provenance: dict[str, dict[str, Any]],
    audit_path: Path,
    blocked_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blocking: list[str] = []
    for slug in videos:
        eye_path = stage12_path(stage12_output_dir, slug)
        pred_path = prediction_path(stage14_root_prefix, slug)
        fail_path = failure_path(stage14_root_prefix, slug)
        if not nonempty(eye_path):
            blocking.append(f"Missing or empty Stage 12 eye timeline: {eye_path}")
        if not nonempty(pred_path):
            blocking.append(f"Missing or empty Stage 14 mouth timeline: {pred_path}")
        eye_df = pd.read_csv(eye_path) if nonempty(eye_path) else pd.DataFrame()
        mouth_df = combined_mouth[combined_mouth["video_slug"] == slug].copy()
        align = nearest_alignment_metrics(eye_df, mouth_df) if not eye_df.empty else {}
        if align and not align["aligned_by_timestamp"]:
            blocking.append(
                f"Timestamp alignment exceeds tolerance for {slug}: "
                f"max delta {align['max_nearest_timestamp_delta_sec']}"
            )
        rows.append(
            {
                "video_slug": slug,
                "stage12_eye_path": str(eye_path),
                "stage12_exists_nonempty": nonempty(eye_path),
                "stage14_mouth_path": str(pred_path),
                "stage14_exists_nonempty": nonempty(pred_path),
                "stage14_failures_path": str(fail_path),
                "stage14_failure_rows_added": provenance[slug]["failure_rows_added_to_mouth_timeline"],
                "eye_rows": align.get("eye_rows"),
                "combined_mouth_rows": align.get("mouth_rows_available"),
                "exact_timestamp_matches": align.get("exact_timestamp_matches"),
                "max_nearest_timestamp_delta_sec": align.get("max_nearest_timestamp_delta_sec"),
                "aligned_by_timestamp": align.get("aligned_by_timestamp"),
                "stage14_yawn_event_count": provenance[slug]["yawn_event_count"],
                "mouth_source": MOUTH_SOURCE,
            }
        )

    b = provenance.get("B_realistic_drowsy_simulation", {})
    a_yawns = provenance.get("A_normal_open_baseline", {}).get("yawn_event_count")
    c_yawns = provenance.get("C_mild_head_motion", {}).get("yawn_event_count")
    d_yawns = provenance.get("D_controlled_long_open_closed", {}).get("yawn_event_count")

    if a_yawns != 0:
        blocking.append(f"A_normal_open_baseline Stage 14 yawn_event_count expected 0, got {a_yawns}")
    if c_yawns != 0:
        blocking.append(f"C_mild_head_motion Stage 14 yawn_event_count expected 0, got {c_yawns}")
    if d_yawns != 0:
        blocking.append(f"D_controlled_long_open_closed Stage 14 yawn_event_count expected 0, got {d_yawns}")
    if b.get("b_yawn_interval_rows") != 12 or b.get("b_yawn_interval_event_rows") != 12:
        blocking.append(
            "B_realistic_drowsy_simulation did not contain the expected 12/12 "
            "Stage 14 yawn events in 14.3s-16.8s."
        )

    audit = {
        "all_required_inputs_available": not blocking,
        "mouth_timeline_source": MOUTH_SOURCE,
        "uses_real_stage14_mouth_timeline": True,
        "uses_synthetic_mouth_timeline": False,
        "uses_manual_mouth_annotation": False,
        "input_rows": rows,
        "stage14_provenance": provenance,
        "b_yawn_interval_check": {
            "interval_sec": [14.3, 16.8],
            "rows": b.get("b_yawn_interval_rows"),
            "yawn_event_rows": b.get("b_yawn_interval_event_rows"),
            "mean_p_yawn": b.get("b_yawn_interval_mean_p_yawn"),
            "min_p_yawn": b.get("b_yawn_interval_min_p_yawn"),
            "max_p_yawn": b.get("b_yawn_interval_max_p_yawn"),
        },
        "blocking_issues": blocking,
    }

    lines = [
        "# Stage 15 Input Audit",
        "",
        "## Result",
        "",
        f"- All required inputs available: {not blocking}",
        f"- Mouth timeline source: `{MOUTH_SOURCE}`",
        "- Synthetic mouth timelines used: false",
        "- Manual mouth annotation used: false",
        "- Alignment method: timestamp alignment; exact matches are present after adding Stage 14 failure rows to avoid silently dropping no-face mouth frames.",
        "",
        "## Input Table",
        "",
        markdown_table(pd.DataFrame(rows), list(rows[0].keys()) if rows else []),
        "",
        "## B Yawn Interval Check",
        "",
        f"- Interval: 14.3s-16.8s",
        f"- Rows in interval: {b.get('b_yawn_interval_rows')}",
        f"- Yawn-event rows in interval: {b.get('b_yawn_interval_event_rows')}",
        f"- Mean/min/max `p_yawn`: {b.get('b_yawn_interval_mean_p_yawn')}, {b.get('b_yawn_interval_min_p_yawn')}, {b.get('b_yawn_interval_max_p_yawn')}",
        "",
        "## Notes",
        "",
        "- Stage 14 timelines are model-generated by the recovered ResNet18 mouth/yawn specialist, not synthetic timelines and not manual annotation timelines.",
        "- Stage 14 C no-face failure rows were included in the combined mouth timeline as `mouth_signal_status=no_face`, `p_yawn=0`, and `yawn_event=false`; these rows are signal-quality evidence, not yawn evidence.",
        "- This audit does not claim final system-level drowsiness accuracy.",
    ]
    if blocking:
        lines.extend(["", "## Blocking Issues", ""])
        lines.extend([f"- {issue}" for issue in blocking])
        blocked_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def augment_timeline(
    timeline: pd.DataFrame,
    mouth_df: pd.DataFrame,
) -> pd.DataFrame:
    mouth_extra = mouth_df[
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
        mouth_extra,
        on=["video_slug", "timestamp_sec", "frame_index"],
        how="left",
        suffixes=("", "_mouth_extra"),
    )
    merged["yawn_event"] = merged["yawn_event"].map(to_bool)
    merged["mouth_signal_status"] = merged["mouth_signal_status"].fillna("missing")
    merged["uses_real_stage14_mouth_timeline"] = True
    merged["uses_synthetic_mouth_timeline"] = False
    merged["uses_manual_mouth_annotation"] = False
    return merged


def scenario_match(slug: str, row: dict[str, Any]) -> tuple[bool, str]:
    if slug == "A_normal_open_baseline":
        ok = (
            row["high_confidence_drowsiness_candidate_frames"] == 0
            and row["mouth_warning_candidate_frames"] == 0
            and row["yawn_event_count"] == 0
        )
        return ok, "Baseline has no high-confidence or mouth-warning frames and no yawn events."
    if slug == "B_realistic_drowsy_simulation":
        ok = row["yawn_event_count"] > 0 and row["high_confidence_drowsiness_candidate_frames"] > 0
        return ok, (
            "Model-generated yawn events are present, and recent-yawn overlap with eye warning "
            "creates high-confidence candidate frames."
        )
    if slug == "C_mild_head_motion":
        ok = (
            row["yawn_event_count"] == 0
            and row["high_confidence_drowsiness_candidate_frames"] == 0
            and row["signal_unreliable_frames"] > 0
        )
        return ok, "No yawn false positives; signal-unreliable frames remain quality markers."
    if slug == "D_controlled_long_open_closed":
        ok = (
            row["eye_warning_candidate_frames"] > 0
            and row["mouth_warning_candidate_frames"] == 0
            and row["high_confidence_drowsiness_candidate_frames"] == 0
            and row["yawn_event_count"] == 0
        )
        return ok, "Long eye closures remain eye-warning candidates without mouth/yawn escalation."
    return False, "No scenario expectation configured."


def summarize_stage15_timeline(timeline: pd.DataFrame, slug: str) -> dict[str, Any]:
    state_counts = {state: int((timeline["fusion_state"] == state).sum()) for state in FUSION_STATES}
    eye_mask = timeline["fusion_state"] == "eye_warning_candidate"
    mouth_mask = timeline["fusion_state"] == "mouth_warning_candidate"
    high_mask = timeline["fusion_state"] == "high_confidence_drowsiness_candidate"
    first_high, last_high = first_last_timestamp(timeline, high_mask)

    row = {
        "video_slug": slug,
        "total_rows": int(len(timeline)),
        "normal_frames": state_counts["normal"],
        "eye_warning_candidate_frames": state_counts["eye_warning_candidate"],
        "mouth_warning_candidate_frames": state_counts["mouth_warning_candidate"],
        "high_confidence_drowsiness_candidate_frames": state_counts[
            "high_confidence_drowsiness_candidate"
        ],
        "signal_unreliable_frames": state_counts["signal_unreliable"],
        "eye_warning_count": run_lengths(eye_mask)["count"],
        "mouth_warning_count": run_lengths(mouth_mask)["count"],
        "high_confidence_count": run_lengths(high_mask)["count"],
        "longest_eye_warning_run": run_lengths(eye_mask)["longest"],
        "longest_mouth_warning_run": run_lengths(mouth_mask)["longest"],
        "longest_high_confidence_run": run_lengths(high_mask)["longest"],
        "first_high_confidence_timestamp_sec": first_high,
        "last_high_confidence_timestamp_sec": last_high,
        "mean_p_eye_closed": float(pd.to_numeric(timeline["p_eye_closed"], errors="coerce").mean()),
        "max_p_eye_closed": float(pd.to_numeric(timeline["p_eye_closed"], errors="coerce").max()),
        "mean_p_yawn": float(pd.to_numeric(timeline["p_yawn"], errors="coerce").mean()),
        "max_p_yawn": float(pd.to_numeric(timeline["p_yawn"], errors="coerce").max()),
        "yawn_event_count": int(timeline["yawn_event"].map(to_bool).sum()),
        "recent_yawn_event_count": int(timeline["recent_yawn_event"].map(to_bool).sum()),
        "signal_unreliable_ratio": (
            state_counts["signal_unreliable"] / len(timeline) if len(timeline) else 0.0
        ),
        "uses_real_stage14_mouth_timeline": True,
        "uses_synthetic_mouth_timeline": False,
        "uses_manual_mouth_annotation": False,
    }
    ok, notes = scenario_match(slug, row)
    row["scenario_expectation_match"] = ok
    row["notes"] = notes
    return row


def b_specific_analysis(timeline: pd.DataFrame) -> dict[str, Any]:
    interval = timeline[
        (pd.to_numeric(timeline["timestamp_sec"], errors="coerce") >= 14.3)
        & (pd.to_numeric(timeline["timestamp_sec"], errors="coerce") <= 16.8)
    ].copy()
    high = timeline[timeline["fusion_state"] == "high_confidence_drowsiness_candidate"]
    mouth = timeline[timeline["fusion_state"] == "mouth_warning_candidate"]
    return {
        "manual_observed_yawn_interval_sec": [14.3, 16.8],
        "yawn_interval_rows": int(len(interval)),
        "yawn_interval_yawn_event_rows": int(interval["yawn_event"].map(to_bool).sum()),
        "yawn_interval_mean_p_yawn": (
            float(pd.to_numeric(interval["p_yawn"], errors="coerce").mean())
            if len(interval)
            else None
        ),
        "yawn_interval_min_p_yawn": (
            float(pd.to_numeric(interval["p_yawn"], errors="coerce").min()) if len(interval) else None
        ),
        "yawn_interval_max_p_yawn": (
            float(pd.to_numeric(interval["p_yawn"], errors="coerce").max()) if len(interval) else None
        ),
        "mouth_warning_candidate_rows": int(len(mouth)),
        "mouth_warning_first_timestamp_sec": (
            float(mouth["timestamp_sec"].min()) if len(mouth) else None
        ),
        "mouth_warning_last_timestamp_sec": (
            float(mouth["timestamp_sec"].max()) if len(mouth) else None
        ),
        "high_confidence_rows": int(len(high)),
        "high_confidence_first_timestamp_sec": (
            float(high["timestamp_sec"].min()) if len(high) else None
        ),
        "high_confidence_last_timestamp_sec": (
            float(high["timestamp_sec"].max()) if len(high) else None
        ),
        "aligns_with_real_yawn_behavior": bool(
            len(interval)
            and int(interval["yawn_event"].map(to_bool).sum()) > 0
            and len(high) > 0
        ),
    }


def write_report(
    output_report_path: Path,
    canonical_report_path: Path,
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    stage15_metrics: pd.DataFrame,
    input_audit_path: Path,
    output_dir: Path,
) -> None:
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
    metric_cols = [
        "video_slug",
        "total_rows",
        "normal_frames",
        "eye_warning_candidate_frames",
        "mouth_warning_candidate_frames",
        "high_confidence_drowsiness_candidate_frames",
        "signal_unreliable_frames",
        "yawn_event_count",
        "recent_yawn_event_count",
        "scenario_expectation_match",
    ]
    b = summary["b_specific_analysis"]
    text = f"""# Stage 15 Real Mouth-Eye Fusion Validation Report

## 1. Purpose

Stage 15 validates real synchronized rule-based mouth-eye fusion using Stage 12 real eye timelines and Stage 14 model-generated `p_yawn` timelines. This run is not synthetic, not manual mouth annotation, not a trained fusion classifier, and not final system-level drowsiness accuracy.

## 2. Inputs

- Stage 12 eye timelines: `outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_<slug>.csv`
- Stage 14 mouth/yawn timelines: `outputs/stage14_mouth_yawn_runtime_<slug>/runtime_mouth_yawn_predictions.csv`
- Combined Stage 14 real mouth timeline: `{output_dir / 'combined_stage14_real_mouth_timeline.csv'}`
- Mouth timeline source: `stage14_runtime_mouth_yawn_model`
- Input audit: `{input_audit_path}`

## 3. Input Audit Result

- All required inputs available: {summary["input_audit"]["all_required_inputs_available"]}
- Real Stage 14 mouth timelines used: true
- Synthetic mouth timelines used: false
- Manual mouth annotation used: false
- Stage 14 C no-face rows were represented as mouth signal-quality rows and were not treated as yawn.

## 4. Fusion Rule

The validated rule is `F5_tiered_quality_aware_fusion`:

- If eye signal is unreliable and no recent yawn exists, output `signal_unreliable`.
- If eye signal is unreliable and a recent yawn exists, output `mouth_warning_candidate`.
- If eye warning candidate and recent yawn event co-occur, output `high_confidence_drowsiness_candidate`.
- If only eye warning is active, output `eye_warning_candidate`.
- If only recent yawn is active, output `mouth_warning_candidate`.
- Otherwise output `normal`.

`signal_unreliable`, `eye_warning_candidate`, `mouth_warning_candidate`, and `high_confidence_drowsiness_candidate` remain rule-based candidate states, not final driver drowsiness accuracy.

## 5. Per-Video Results

{markdown_table(stage15_metrics, metric_cols)}

## 6. Rule Comparison

{markdown_table(comparison, comparison_cols)}

## 7. B-Specific Real Yawn Validation

The user manually observed yawning in `B_realistic_drowsy_simulation` around 14.3s-16.8s. Stage 15 did not use that manual annotation for fusion decisions; it used Stage 14 model-generated `p_yawn` only.

- Rows in 14.3s-16.8s: {b["yawn_interval_rows"]}
- Yawn-event rows in 14.3s-16.8s: {b["yawn_interval_yawn_event_rows"]}
- Mean/min/max `p_yawn` in 14.3s-16.8s: {b["yawn_interval_mean_p_yawn"]}, {b["yawn_interval_min_p_yawn"]}, {b["yawn_interval_max_p_yawn"]}
- Mouth-warning candidate interval: {b["mouth_warning_first_timestamp_sec"]}s to {b["mouth_warning_last_timestamp_sec"]}s
- High-confidence candidate interval: {b["high_confidence_first_timestamp_sec"]}s to {b["high_confidence_last_timestamp_sec"]}s

High-confidence candidates can occur after the visible yawn interval because Stage 14 `recent_yawn_event` remains active for the recent-yawn window and can later overlap with eye warning candidates.

## 8. Scenario-Level Interpretation

- `A_normal_open_baseline`: mostly normal, with no high-confidence or mouth-warning frames.
- `B_realistic_drowsy_simulation`: Stage 14 generated high `p_yawn` during the observed yawn interval, and F5 fusion produced mouth/high-confidence candidates when recent yawn overlapped eye state.
- `C_mild_head_motion`: no mouth/yawn false positives were used; signal-quality intervals remain quality markers rather than confirmed drowsiness.
- `D_controlled_long_open_closed`: eye closure produced eye-warning candidates without mouth/yawn escalation.

## 9. Visual Acceptance Note

Stage 14 mouth contact sheets and debug frames were visually accepted as sufficient for Stage 15. High `p_yawn` crops in B corresponded to yawning/open-mouth frames. Some lower-probability or transition frames existed, but they did not materially affect yawn-event detection in the manually observed B interval.

## 10. Limitations

- Small A/B/C/D validation set.
- One or few subjects.
- No final drowsiness ground-truth timeline.
- No trained fusion classifier.
- No real-world deployment validation.
- This is not final system-level drowsiness accuracy.

## 11. Next Step

If Stage 15 behavior is accepted, the project can move to final integration summary and demo planning. A learned fusion classifier should only be considered after collecting synchronized annotated mouth-eye data.
"""
    output_report_path.write_text(text, encoding="utf-8")
    canonical_report_path.write_text(text, encoding="utf-8")


def write_doc_log(path: Path, summary: dict[str, Any], output_dir: Path, audit_path: Path) -> None:
    b = summary["b_specific_analysis"]
    lines = [
        "# Stage 15 Real Mouth-Eye Fusion Log",
        "",
        "## Purpose",
        "",
        "Stage 15 performs real synchronized rule-based mouth-eye fusion validation using Stage 12 eye timelines and Stage 14 model-generated `p_yawn` timelines.",
        "",
        "This is not synthetic mouth fusion, not manual mouth annotation fusion, and not final system-level drowsiness accuracy.",
        "",
        "## Inputs",
        "",
        "- Stage 12 eye timelines from `outputs/stage12_eye_alert_rule_analysis/`.",
        "- Stage 14 mouth/yawn timelines from `outputs/stage14_mouth_yawn_runtime_<slug>/runtime_mouth_yawn_predictions.csv`.",
        f"- Combined real mouth timeline: `{output_dir / 'combined_stage14_real_mouth_timeline.csv'}`.",
        f"- Input audit: `{audit_path}`.",
        "",
        "## Run Result",
        "",
        f"- Status: {summary['status']}",
        "- Real Stage 14 mouth timelines used: true",
        "- Synthetic mouth timelines used: false",
        "- Manual mouth annotation used: false",
        f"- Rule validated: `{summary['recommended_fusion_rule']}`",
        "",
        "## B Yawn Interval",
        "",
        "- Manual observed yawn interval: 14.3s-16.8s.",
        "- Fusion decisions used Stage 14 model output, not manual labels.",
        f"- Stage 14 yawn-event rows in interval: {b['yawn_interval_yawn_event_rows']}/{b['yawn_interval_rows']}.",
        f"- Mean/min/max `p_yawn` in interval: {b['yawn_interval_mean_p_yawn']}, {b['yawn_interval_min_p_yawn']}, {b['yawn_interval_max_p_yawn']}.",
        f"- High-confidence candidate interval: {b['high_confidence_first_timestamp_sec']}s to {b['high_confidence_last_timestamp_sec']}s.",
        "",
        "## Outputs",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Report: `{DEFAULT_REPORT_PATH}`",
        "",
        "## Warning",
        "",
        "Stage 15 validates rule-based fusion behavior on a small controlled-realistic set. It is not final system-level drowsiness accuracy and not deployment readiness.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    videos = parse_videos(args.videos)
    dirs = ensure_dirs(args.output_dir, args.audit_dir)

    combined_path = args.output_dir / "combined_stage14_real_mouth_timeline.csv"
    combined_mouth_raw, provenance = build_combined_mouth_timeline(
        args.stage14_root_prefix,
        videos,
        combined_path,
    )
    combined_mouth = validate_mouth_timeline(combined_path)

    audit_path = args.audit_dir / "stage15_input_audit.md"
    blocked_path = args.audit_dir / "STAGE15_BLOCKED_MISSING_INPUTS.md"
    input_audit = audit_inputs(
        stage12_output_dir=args.stage12_output_dir,
        stage14_root_prefix=args.stage14_root_prefix,
        videos=videos,
        combined_mouth=combined_mouth_raw,
        provenance=provenance,
        audit_path=audit_path,
        blocked_path=blocked_path,
    )
    if input_audit["blocking_issues"]:
        print(f"Stage 15 blocked. See {blocked_path}")
        return 2

    comparison_rows: list[dict[str, Any]] = []
    recommended_timelines: dict[str, pd.DataFrame] = {}
    stage15_metric_rows: list[dict[str, Any]] = []

    for slug in videos:
        eye_df = load_stage12_timeline(args.stage12_output_dir, slug)
        mouth_df = align_real_mouth_timeline(eye_df, combined_mouth, slug)

        for rule_name in FUSION_RULES:
            timeline = build_fusion_timeline(eye_df, mouth_df, slug, rule_name)
            timeline = augment_timeline(timeline, mouth_df)
            comparison_row = summarize_rule_timeline(
                timeline=timeline,
                slug=slug,
                rule_name=rule_name,
                uses_real_mouth_timeline=True,
                uses_synthetic_mouth_timeline=False,
            )
            comparison_row["uses_real_stage14_mouth_timeline"] = True
            comparison_row["uses_manual_mouth_annotation"] = False
            comparison_rows.append(comparison_row)

            if rule_name == RECOMMENDED_RULE:
                recommended_timelines[slug] = timeline
                timeline_path = dirs["timelines"] / f"fusion_timeline_{slug}.csv"
                timeline.to_csv(timeline_path, index=False)
                stage15_metric_rows.append(summarize_stage15_timeline(timeline, slug))

    comparison = pd.DataFrame(comparison_rows)
    stage15_metrics = pd.DataFrame(stage15_metric_rows)

    comparison_path = args.output_dir / "stage15_real_fusion_rule_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    figure_paths = [args.output_dir / "figures" / "fusion_state_counts_by_video.png"]
    plot_stage15_state_counts(comparison, figure_paths[0])
    for slug, timeline in recommended_timelines.items():
        figure_path = args.output_dir / "figures" / f"fusion_timeline_{slug}.png"
        plot_stage15_video_timeline(timeline, figure_path, slug)
        figure_paths.append(figure_path)

    b_timeline = recommended_timelines["B_realistic_drowsy_simulation"]
    b_analysis = b_specific_analysis(b_timeline)

    summary = {
        "stage": 15,
        "status": "REAL_SYNCHRONIZED_RULE_BASED_FUSION_VALIDATION_COMPLETED",
        "recommended_fusion_rule": RECOMMENDED_RULE,
        "eye_rule": "Stage 12 quality_gated_perclos_mean_ge_0.60_consec",
        "mouth_timeline_source": MOUTH_SOURCE,
        "uses_real_stage14_mouth_timeline": True,
        "uses_synthetic_mouth_timeline": False,
        "uses_manual_mouth_annotation": False,
        "validation_videos": videos,
        "input_audit_path": str(audit_path),
        "combined_mouth_timeline_path": str(combined_path),
        "rule_comparison_csv": str(comparison_path),
        "timeline_paths": {
            slug: str(dirs["timelines"] / f"fusion_timeline_{slug}.csv") for slug in videos
        },
        "figure_paths": [str(path) for path in figure_paths],
        "per_video_metrics": stage15_metrics.to_dict(orient="records"),
        "scenario_expectations_met": {
            row["video_slug"]: bool(row["scenario_expectation_match"])
            for row in stage15_metric_rows
        },
        "b_specific_analysis": b_analysis,
        "input_audit": input_audit,
        "limitations": [
            "Small controlled-realistic validation set.",
            "No final drowsiness ground-truth timeline.",
            "No trained fusion classifier.",
            "No real-world deployment validation.",
            "Not final system-level drowsiness accuracy.",
        ],
        "next_stage_recommendation": (
            "If accepted, prepare final project integration summary and demo planning; "
            "learned fusion should wait for synchronized annotated data."
        ),
        "warning": "This is not final system-level drowsiness accuracy.",
    }
    summary_path = args.output_dir / "stage15_real_fusion_summary.json"
    write_json(summary_path, summary)

    write_report(
        output_report_path=args.output_dir / "STAGE15_REAL_MOUTH_EYE_FUSION_REPORT.md",
        canonical_report_path=DEFAULT_REPORT_PATH,
        summary=summary,
        comparison=comparison,
        stage15_metrics=stage15_metrics,
        input_audit_path=audit_path,
        output_dir=args.output_dir,
    )
    write_doc_log(DEFAULT_DOC_LOG_PATH, summary, args.output_dir, audit_path)

    print(f"Wrote combined real mouth timeline: {combined_path}")
    print(f"Wrote Stage 15 comparison: {comparison_path}")
    print(f"Wrote Stage 15 summary: {summary_path}")
    print(f"Wrote Stage 15 report: {DEFAULT_REPORT_PATH}")
    print("Used real Stage 14 mouth/yawn timelines: true")
    print("Used synthetic mouth timelines: false")
    print("Used manual mouth annotation: false")
    print("Warning: this is not final system-level drowsiness accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
