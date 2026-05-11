#!/usr/bin/env python3
"""Stage 12 eye-only alert rule comparison.

This script evaluates temporal eye-only alert rules from Stage 10/11 outputs.
It does not train models, does not modify checkpoints, and does not claim final
system-level drowsiness accuracy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MULTI_VIDEO_SUMMARY = Path("outputs/stage11_multi_video_validation_summary.csv")
DEFAULT_STAGE10_PREFIX = "outputs/stage10_eye_roi_consistency_"
DEFAULT_STAGE11_PREFIX = "outputs/stage11_eye_temporal_analysis_"
DEFAULT_OUTPUT_DIR = Path("outputs/stage12_eye_alert_rule_analysis")
DEFAULT_REPORT_PATH = Path("reports/stage12_eye_alert_rule_analysis_report.md")

SCENARIO_NAMES = {
    "A_normal_open_baseline": "Normal-open baseline",
    "B_realistic_drowsy_simulation": "Realistic drowsy simulation",
    "C_mild_head_motion": "Mild head motion",
    "D_controlled_long_open_closed": "Controlled long open/closed reference",
}


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    threshold: float | None = None
    min_duration: int | None = None
    uses_quality_gate: bool = False


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare eye-only Stage 12 temporal alert rules."
    )
    parser.add_argument("--multi-video-summary", type=Path, default=DEFAULT_MULTI_VIDEO_SUMMARY)
    parser.add_argument("--stage10-root-prefix", default=DEFAULT_STAGE10_PREFIX)
    parser.add_argument("--stage11-root-prefix", default=DEFAULT_STAGE11_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--closed-threshold", type=float, default=0.50)
    parser.add_argument("--rolling-perclos-thresholds", default="0.4,0.5,0.6,0.7")
    parser.add_argument("--rolling-mean-thresholds", default="0.5,0.6,0.7")
    parser.add_argument("--min-consecutive-windows", type=int, default=2)
    parser.add_argument("--max-recent-no-face-ratio", type=float, default=0.20)
    parser.add_argument("--recent-quality-window", type=int, default=5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_rules(
    rolling_mean_thresholds: list[float],
    rolling_perclos_thresholds: list[float],
) -> list[Rule]:
    rules: list[Rule] = []
    for threshold in rolling_mean_thresholds:
        rules.append(
            Rule(
                name=f"rolling_mean_prob_ge_{threshold:.2f}_consec",
                family="rolling_mean_probability",
                threshold=threshold,
            )
        )
    for threshold in rolling_perclos_thresholds:
        rules.append(
            Rule(
                name=f"rolling_perclos_mean_ge_{threshold:.2f}_consec",
                family="rolling_perclos_mean_binary",
                threshold=threshold,
            )
        )
    for threshold in rolling_perclos_thresholds:
        rules.append(
            Rule(
                name=f"rolling_perclos_both_ge_{threshold:.2f}_consec",
                family="rolling_perclos_both_eyes",
                threshold=threshold,
            )
        )
    for duration in (3, 5, 8):
        rules.append(
            Rule(
                name=f"candidate_event_duration_ge_{duration}",
                family="candidate_event_duration",
                min_duration=duration,
            )
        )
    for threshold in (0.5, 0.6, 0.7):
        rules.append(
            Rule(
                name=f"quality_gated_perclos_mean_ge_{threshold:.2f}_consec",
                family="quality_gated_rolling_perclos_mean",
                threshold=threshold,
                uses_quality_gate=True,
            )
        )
    return rules


def infer_output_dir(prefix: str, slug: str, summary_value: str | None) -> Path:
    if summary_value and isinstance(summary_value, str) and summary_value.strip():
        return Path(summary_value)
    return Path(f"{prefix}{slug}")


def read_timeline(
    slug: str,
    stage10_dir: Path,
    stage11_dir: Path,
    recent_quality_window: int,
    max_recent_no_face_ratio: float,
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

    failures_path = stage10_dir / "failures.csv"
    failure_rows: list[dict[str, Any]] = []
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
        failure_df = pd.DataFrame(failure_rows)
        timeline = pd.concat([timeline, failure_df], ignore_index=True, sort=False)

    timeline = timeline.sort_values(["frame_index", "has_prediction"], ascending=[True, False])
    timeline = timeline.drop_duplicates(subset=["frame_index"], keep="first")
    timeline = timeline.sort_values("frame_index").reset_index(drop=True)
    timeline["video_slug"] = slug

    numeric_fill_zero = [
        "left_closed_binary",
        "right_closed_binary",
        "both_eyes_closed_binary",
        "either_eye_closed_binary",
        "mean_closed_binary",
        "no_face_binary",
        "tracking_failure_binary",
    ]
    for col in numeric_fill_zero:
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
    return timeline


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


def alert_run_metrics(timeline: pd.DataFrame, alert: pd.Series) -> dict[str, Any]:
    values = alert.fillna(False).astype(bool).tolist()
    alert_count = 0
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
        else:
            if current:
                alert_count += 1
                longest = max(longest, current)
                current = 0
    if current:
        alert_count += 1
        longest = max(longest, current)

    alert_frames = timeline.loc[alert.fillna(False).astype(bool), "frame_index"]
    return {
        "alert_count": int(alert_count),
        "total_alert_frames": int(alert.fillna(False).sum()),
        "longest_alert_run": int(longest),
        "first_alert_frame_index": int(alert_frames.iloc[0]) if len(alert_frames) else None,
        "last_alert_frame_index": int(alert_frames.iloc[-1]) if len(alert_frames) else None,
    }


def event_alert(timeline: pd.DataFrame, events: pd.DataFrame, min_duration: int) -> pd.Series:
    alert = pd.Series(False, index=timeline.index)
    if events.empty:
        return alert
    for _, event in events.iterrows():
        if int(event["duration_sampled_frames"]) < min_duration:
            continue
        mask = (
            (timeline["frame_index"] >= int(event["start_frame_index"]))
            & (timeline["frame_index"] <= int(event["end_frame_index"]))
            & timeline["has_prediction"].fillna(False).astype(bool)
        )
        alert.loc[mask] = True
    return alert


def evaluate_rule(
    rule: Rule,
    slug: str,
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    min_consecutive_windows: int,
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    if rule.family == "rolling_mean_probability":
        raw = timeline["rolling_mean_p_eye_closed"] >= float(rule.threshold)
        alert = sustained_alert(raw, min_consecutive_windows)
    elif rule.family == "rolling_perclos_mean_binary":
        raw = timeline["rolling_perclos_mean_binary"] >= float(rule.threshold)
        alert = sustained_alert(raw, min_consecutive_windows)
    elif rule.family == "rolling_perclos_both_eyes":
        raw = timeline["rolling_perclos_both_eyes"] >= float(rule.threshold)
        alert = sustained_alert(raw, min_consecutive_windows)
    elif rule.family == "candidate_event_duration":
        raw = event_alert(timeline, events, int(rule.min_duration))
        alert = raw
    elif rule.family == "quality_gated_rolling_perclos_mean":
        raw = (timeline["rolling_perclos_mean_binary"] >= float(rule.threshold)) & (
            ~timeline["signal_unreliable"].fillna(False).astype(bool)
        )
        alert = sustained_alert(raw, min_consecutive_windows)
    else:
        raise ValueError(f"Unknown rule family: {rule.family}")

    alert = alert.fillna(False).astype(bool)
    raw = raw.fillna(False).astype(bool)
    run_metrics = alert_run_metrics(timeline, alert)
    signal_unreliable_frames = int(timeline["signal_unreliable"].fillna(False).sum())
    no_face_frames = int(timeline["no_face_binary"].fillna(0).sum())
    alert_on_no_face_frames = int(
        (alert & (timeline["no_face_binary"].fillna(0).astype(int) == 1)).sum()
    )
    alert_on_unreliable_frames = int(
        (alert & timeline["signal_unreliable"].fillna(False).astype(bool)).sum()
    )

    is_a = slug == "A_normal_open_baseline"
    is_b = slug == "B_realistic_drowsy_simulation"
    is_c = slug == "C_mild_head_motion"
    is_d = slug == "D_controlled_long_open_closed"

    false_warning_on_a = bool(is_a and run_metrics["total_alert_frames"] > 0)
    detected_b = bool(is_b and run_metrics["alert_count"] > 0)
    handled_c = bool(
        is_c
        and rule.uses_quality_gate
        and no_face_frames > 0
        and signal_unreliable_frames >= no_face_frames
        and alert_on_no_face_frames == 0
    )
    detected_d = bool(is_d and run_metrics["alert_count"] > 0)

    if is_a:
        scenario_match = not false_warning_on_a
        notes = (
            "A baseline suppressed."
            if scenario_match
            else "A baseline produced alert frames; review threshold/window."
        )
    elif is_b:
        scenario_match = detected_b
        notes = (
            "B drowsy simulation produced warning candidates."
            if scenario_match
            else "B drowsy simulation was not detected."
        )
    elif is_c:
        scenario_match = handled_c
        notes = (
            "C no-face rows represented as signal_unreliable and not alerted directly."
            if scenario_match
            else "C quality issue not fully handled by this rule family."
        )
    elif is_d:
        scenario_match = detected_d
        notes = (
            "D long closure produced warning candidates."
            if scenario_match
            else "D long closure was not detected."
        )
    else:
        scenario_match = False
        notes = "Unknown scenario."

    row = {
        "rule_name": rule.name,
        "rule_family": rule.family,
        "rule_threshold": rule.threshold,
        "rule_min_duration": rule.min_duration,
        "uses_quality_gate": rule.uses_quality_gate,
        "video_slug": slug,
        **run_metrics,
        "signal_unreliable_frames": signal_unreliable_frames,
        "no_face_frames": no_face_frames,
        "alert_on_no_face_frames": alert_on_no_face_frames,
        "alert_on_unreliable_frames": alert_on_unreliable_frames,
        "false_warning_on_A_baseline": false_warning_on_a,
        "detected_B_drowsy_simulation": detected_b,
        "handled_C_quality_issue": handled_c,
        "detected_D_long_closure": detected_d,
        "scenario_expectation_match": bool(scenario_match),
        "notes": notes,
    }
    return row, raw, alert


def select_recommended_rule(comparison: pd.DataFrame) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for rule_name, group in comparison.groupby("rule_name"):
        by_slug = {row["video_slug"]: row for _, row in group.iterrows()}
        if not all(slug in by_slug for slug in SCENARIO_NAMES):
            continue
        a_suppressed = not bool(by_slug["A_normal_open_baseline"]["false_warning_on_A_baseline"])
        b_detected = bool(by_slug["B_realistic_drowsy_simulation"]["detected_B_drowsy_simulation"])
        c_handled = bool(by_slug["C_mild_head_motion"]["handled_C_quality_issue"])
        d_detected = bool(by_slug["D_controlled_long_open_closed"]["detected_D_long_closure"])
        all_criteria = a_suppressed and b_detected and c_handled and d_detected
        family = str(group.iloc[0]["rule_family"])
        threshold = group.iloc[0]["rule_threshold"]
        threshold_distance = abs(float(threshold) - 0.6) if pd.notna(threshold) else 99.0
        score = (
            int(a_suppressed) * 10
            + int(b_detected) * 10
            + int(c_handled) * 12
            + int(d_detected) * 10
            + (5 if family == "quality_gated_rolling_perclos_mean" else 0)
            - threshold_distance
        )
        candidates.append(
            {
                "rule_name": rule_name,
                "rule_family": family,
                "rule_threshold": None if pd.isna(threshold) else float(threshold),
                "rule_min_duration": None
                if pd.isna(group.iloc[0]["rule_min_duration"])
                else int(group.iloc[0]["rule_min_duration"]),
                "uses_quality_gate": bool(group.iloc[0]["uses_quality_gate"]),
                "A_baseline_suppressed": a_suppressed,
                "B_drowsy_detected": b_detected,
                "C_quality_issue_handled": c_handled,
                "D_long_closure_detected": d_detected,
                "all_criteria": all_criteria,
                "score": float(score),
            }
        )
    candidates_df = pd.DataFrame(candidates)
    passing = candidates_df[candidates_df["all_criteria"]].copy()
    if not passing.empty:
        passing["threshold_distance"] = passing["rule_threshold"].fillna(99.0).sub(0.6).abs()
        passing["quality_rank"] = passing["rule_family"].eq(
            "quality_gated_rolling_perclos_mean"
        ).astype(int)
        chosen = passing.sort_values(
            ["quality_rank", "threshold_distance", "score"],
            ascending=[False, True, False],
        ).iloc[0]
    else:
        chosen = candidates_df.sort_values("score", ascending=False).iloc[0]
    result = chosen.to_dict()
    for key in ("rule_threshold", "rule_min_duration"):
        if key in result and pd.isna(result[key]):
            result[key] = None
    if result.get("rule_min_duration") is not None:
        result["rule_min_duration"] = int(result["rule_min_duration"])
    if result.get("rule_threshold") is not None:
        result["rule_threshold"] = float(result["rule_threshold"])
    result["uses_quality_gate"] = bool(result.get("uses_quality_gate"))
    result["A_baseline_suppressed"] = bool(result.get("A_baseline_suppressed"))
    result["B_drowsy_detected"] = bool(result.get("B_drowsy_detected"))
    result["C_quality_issue_handled"] = bool(result.get("C_quality_issue_handled"))
    result["D_long_closure_detected"] = bool(result.get("D_long_closure_detected"))
    return result


def write_timeline(
    output_dir: Path,
    slug: str,
    timeline: pd.DataFrame,
    raw: pd.Series,
    alert: pd.Series,
    rule_name: str,
) -> Path:
    out = timeline.copy()
    out["recommended_rule_name"] = rule_name
    out["recommended_raw_condition"] = raw.fillna(False).astype(int)
    out["recommended_alert"] = alert.fillna(False).astype(int)
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
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    path = output_dir / f"stage12_video_alert_timeline_{slug}.csv"
    out[columns].to_csv(path, index=False)
    return path


def plot_comparison(comparison: pd.DataFrame, output_dir: Path) -> Path:
    pivot = comparison.pivot_table(
        index="rule_name", columns="video_slug", values="total_alert_frames", aggfunc="sum"
    ).fillna(0)
    path = output_dir / "figures" / "alert_rule_comparison_by_video.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, max(6, len(pivot) * 0.28)))
    image = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(image, label="Total alert frames")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Stage 12 alert rule comparison by video")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_timeline(timeline_csv: Path, output_dir: Path, slug: str) -> Path:
    df = pd.read_csv(timeline_csv)
    path = output_dir / "figures" / f"alert_timeline_{slug}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    valid = df["has_prediction"].astype(str).isin(["True", "true", "1"])
    plt.plot(
        df.loc[valid, "frame_index"],
        df.loc[valid, "mean_p_eye_closed"],
        linewidth=1,
        alpha=0.5,
        label="mean p_eye_closed",
    )
    plt.plot(
        df.loc[valid, "frame_index"],
        df.loc[valid, "rolling_perclos_mean_binary"],
        linewidth=2,
        label="rolling PERCLOS-like mean binary",
    )
    alert = df["recommended_alert"].fillna(0).astype(int) == 1
    unreliable = df["signal_unreliable"].astype(str).isin(["True", "true", "1"])
    if alert.any():
        plt.scatter(
            df.loc[alert, "frame_index"],
            [1.05] * int(alert.sum()),
            marker="s",
            s=18,
            label="recommended alert",
        )
    if unreliable.any():
        plt.scatter(
            df.loc[unreliable, "frame_index"],
            [1.15] * int(unreliable.sum()),
            marker="x",
            s=28,
            label="signal unreliable",
        )
    plt.ylim(-0.05, 1.22)
    plt.xlabel("Frame index")
    plt.ylabel("Eye-only signal")
    plt.title(f"Stage 12 recommended alert timeline: {slug}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_reports(
    output_dir: Path,
    comparison: pd.DataFrame,
    recommended: dict[str, Any],
    summary_df: pd.DataFrame,
    timeline_paths: list[Path],
    figure_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    compact_cols = [
        "rule_name",
        "video_slug",
        "alert_count",
        "total_alert_frames",
        "longest_alert_run",
        "signal_unreliable_frames",
        "scenario_expectation_match",
    ]
    recommended_rows = comparison[comparison["rule_name"] == recommended["rule_name"]]
    rec_cols = [
        "video_slug",
        "alert_count",
        "total_alert_frames",
        "longest_alert_run",
        "signal_unreliable_frames",
        "false_warning_on_A_baseline",
        "detected_B_drowsy_simulation",
        "handled_C_quality_issue",
        "detected_D_long_closure",
        "scenario_expectation_match",
    ]

    report = f"""# Stage 12 Eye-Only Alert Rule Analysis Report

## 1. Purpose

Stage 12 designs and compares eye-only temporal alert rules using the completed Stage 10/11 multi-video validation outputs.

It is not mouth/yawn fusion. It is not final system-level drowsiness accuracy. It is not deployment readiness.

## 2. Literature-Inspired Rationale

PERCLOS is commonly described in driver monitoring literature as the percentage of time, over a time window, that the eyes are substantially closed, often more than 80% closed.

This project does not directly measure eyelid aperture percentage. It uses the trained MRL Eye specialist probability `p_eye_closed = softmax(logits)[0]` as a proxy. Therefore Stage 12 uses a **PERCLOS-like** or **PERCLOS-inspired** metric, not standard PERCLOS.

Temporal persistence and rolling windows are needed because single-frame probability spikes can create false warnings. Signal-quality gating is needed because no-face or tracking failures are not drowsiness; they are unreliable signal intervals.

## 3. Inputs

- Multi-video summary: `{args.multi_video_summary}`
- Stage 10 prefix: `{args.stage10_root_prefix}`
- Stage 11 prefix: `{args.stage11_root_prefix}`
- Videos: `A_normal_open_baseline`, `B_realistic_drowsy_simulation`, `C_mild_head_motion`, `D_controlled_long_open_closed`

## 4. Rules Compared

- Rule 1: Rolling mean probability, `rolling_mean_p_eye_closed >= threshold` for at least `{args.min_consecutive_windows}` sampled frames.
- Rule 2: Rolling PERCLOS-like mean-binary ratio, `rolling_perclos_mean_binary >= threshold` for at least `{args.min_consecutive_windows}` sampled frames.
- Rule 3: Rolling PERCLOS-like both-eyes ratio, `rolling_perclos_both_eyes >= threshold` for at least `{args.min_consecutive_windows}` sampled frames.
- Rule 4: Candidate closure event duration, event duration >= 3, 5, or 8 sampled frames.
- Rule 5: Quality-gated rolling PERCLOS-like mean-binary ratio. If recent no-face ratio over `{args.recent_quality_window}` sampled frames is greater than `{args.max_recent_no_face_ratio:.2f}`, the frame/window is marked `signal_unreliable`; otherwise the rolling PERCLOS-like threshold is applied.

## 5. Rule Comparison Table

{md_table(comparison[compact_cols], compact_cols)}

## 6. Recommended Rule

Recommended rule:

```text
{recommended["rule_name"]}
```

Parameters:

```json
{json.dumps({k: recommended[k] for k in ["rule_family", "rule_threshold", "rule_min_duration", "uses_quality_gate"]}, indent=2)}
```

Recommended rule behavior:

{md_table(recommended_rows[rec_cols], rec_cols)}

Selection rationale:

- A's short false event was suppressed: `{recommended["A_baseline_suppressed"]}`.
- B produced expected warning candidates: `{recommended["B_drowsy_detected"]}`.
- C's no-face rows were handled as signal quality issues: `{recommended["C_quality_issue_handled"]}`.
- D produced expected long warning candidates: `{recommended["D_long_closure_detected"]}`.
- The rule is simple to explain: quality-gated rolling PERCLOS-like mean-binary ratio with persistence.

## 7. Limitations

- Small validation set.
- One/few subjects.
- No ground-truth temporal annotation.
- No mouth/yawn fusion yet.
- No live webcam validation.
- Not final drowsiness accuracy.
- PERCLOS-like proxy is based on CNN probability, not true eyelid aperture percentage.

## 8. Next Step

If the recommended eye-only rule behaves correctly after human review of the generated timelines and figures, proceed to Stage 13 mouth-eye fusion design. Otherwise adjust thresholds/windowing and rerun Stage 12.

## Artifact Paths

- Rule comparison CSV: `{output_dir / "stage12_rule_comparison.csv"}`
- Summary JSON: `{output_dir / "stage12_eye_alert_summary.json"}`
- Output report: `{output_dir / "STAGE12_EYE_ALERT_RULE_REPORT.md"}`
- Timeline CSVs: {", ".join(str(path) for path in timeline_paths)}
- Figures: {", ".join(str(path) for path in figure_paths)}

This report is eye-only alert rule design. It is not final system-level drowsiness accuracy.
"""

    (output_dir / "STAGE12_EYE_ALERT_RULE_REPORT.md").write_text(report, encoding="utf-8")
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    multi = pd.read_csv(args.multi_video_summary)
    rolling_perclos_thresholds = parse_float_list(args.rolling_perclos_thresholds)
    rolling_mean_thresholds = parse_float_list(args.rolling_mean_thresholds)
    rules = make_rules(rolling_mean_thresholds, rolling_perclos_thresholds)

    timelines: dict[str, pd.DataFrame] = {}
    events_by_slug: dict[str, pd.DataFrame] = {}
    comparison_rows: list[dict[str, Any]] = []
    rule_alerts: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}

    for _, video in multi.iterrows():
        slug = str(video["video_slug"])
        stage10_dir = infer_output_dir(
            args.stage10_root_prefix, slug, video.get("stage10_output_dir")
        )
        stage11_dir = infer_output_dir(
            args.stage11_root_prefix, slug, video.get("stage11_output_dir")
        )
        timeline = read_timeline(
            slug=slug,
            stage10_dir=stage10_dir,
            stage11_dir=stage11_dir,
            recent_quality_window=args.recent_quality_window,
            max_recent_no_face_ratio=args.max_recent_no_face_ratio,
        )
        events_path = stage11_dir / "stage11_eye_events.csv"
        events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
        timelines[slug] = timeline
        events_by_slug[slug] = events

        for rule in rules:
            row, raw, alert = evaluate_rule(
                rule=rule,
                slug=slug,
                timeline=timeline,
                events=events,
                min_consecutive_windows=args.min_consecutive_windows,
            )
            comparison_rows.append(row)
            rule_alerts[(slug, rule.name)] = (raw, alert)

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "stage12_rule_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    recommended = select_recommended_rule(comparison)

    timeline_paths: list[Path] = []
    for slug, timeline in timelines.items():
        raw, alert = rule_alerts[(slug, recommended["rule_name"])]
        timeline_paths.append(
            write_timeline(output_dir, slug, timeline, raw, alert, recommended["rule_name"])
        )

    figure_paths = [plot_comparison(comparison, output_dir)]
    for path in timeline_paths:
        slug = path.stem.replace("stage12_video_alert_timeline_", "")
        figure_paths.append(plot_timeline(path, output_dir, slug))

    validation_videos = list(multi["video_slug"].astype(str))
    summary = {
        "stage": 12,
        "status": "COMPLETED_EYE_ONLY_ALERT_RULE_ANALYSIS",
        "recommended_rule_name": recommended["rule_name"],
        "recommended_rule_parameters": {
            "rule_family": recommended["rule_family"],
            "rule_threshold": recommended["rule_threshold"],
            "rule_min_duration": recommended["rule_min_duration"],
            "uses_quality_gate": recommended["uses_quality_gate"],
            "min_consecutive_windows": args.min_consecutive_windows,
            "max_recent_no_face_ratio": args.max_recent_no_face_ratio,
            "recent_quality_window": args.recent_quality_window,
        },
        "validation_videos": validation_videos,
        "A_baseline_suppressed": recommended["A_baseline_suppressed"],
        "B_drowsy_detected": recommended["B_drowsy_detected"],
        "C_quality_issue_handled": recommended["C_quality_issue_handled"],
        "D_long_closure_detected": recommended["D_long_closure_detected"],
        "limitations": [
            "small validation set",
            "one/few subjects",
            "no ground-truth temporal annotation",
            "no mouth/yawn fusion yet",
            "no live webcam validation",
            "not final drowsiness accuracy",
            "PERCLOS-like proxy uses CNN probability, not true eyelid aperture percentage",
        ],
        "next_stage_recommendation": (
            "Proceed to Stage 13 mouth-eye fusion design if human review accepts "
            "the recommended eye-only alert timelines; otherwise adjust thresholds/windowing "
            "and rerun Stage 12."
        ),
        "warning": "This is not final system-level drowsiness accuracy.",
    }
    summary_path = output_dir / "stage12_eye_alert_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")

    write_reports(
        output_dir=output_dir,
        comparison=comparison,
        recommended=recommended,
        summary_df=multi,
        timeline_paths=timeline_paths,
        figure_paths=figure_paths,
        args=args,
    )

    print(f"[done] comparison: {comparison_path}")
    for path in timeline_paths:
        print(f"[done] timeline: {path}")
    print(f"[done] summary: {summary_path}")
    print(f"[done] output report: {output_dir / 'STAGE12_EYE_ALERT_RULE_REPORT.md'}")
    print(f"[done] repo report: {DEFAULT_REPORT_PATH}")
    for path in figure_paths:
        print(f"[done] figure: {path}")
    print(f"[done] recommended_rule: {recommended['rule_name']}")


if __name__ == "__main__":
    main()
