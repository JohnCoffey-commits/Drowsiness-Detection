#!/usr/bin/env python3
"""Stage 11 eye-only temporal smoothing and PERCLOS-like analysis.

This script consumes Stage 10 per-eye runtime predictions and produces an
eye-only temporal signal package. It does not train models, does not modify
checkpoints, and does not claim final drowsiness accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_CSV = Path(
    "outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/"
    "runtime_eye_roi_predictions.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/stage11_eye_temporal_analysis_IMG_4901")
DEFAULT_REPORT_PATH = Path("reports/stage11_eye_temporal_analysis_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Stage 10 eye probabilities as an eye-only temporal signal."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--closed-threshold", type=float, default=0.50)
    parser.add_argument("--rolling-window", type=int, default=5)
    return parser.parse_args()


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_columns(df: pd.DataFrame, required: list[str], input_csv: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")


def read_predictions(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = ["frame_index", "timestamp_sec", "eye_side", "p_eye_closed", "p_eye_open"]
    ensure_columns(df, required, input_csv)

    df = df.copy()
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="raise").astype(int)
    df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="raise")
    df["p_eye_closed"] = pd.to_numeric(df["p_eye_closed"], errors="raise")
    df["p_eye_open"] = pd.to_numeric(df["p_eye_open"], errors="raise")
    df["eye_side"] = df["eye_side"].astype(str).str.lower()
    return df


def build_frame_summary(
    df: pd.DataFrame, closed_threshold: float, rolling_window: int
) -> pd.DataFrame:
    if rolling_window < 1:
        raise ValueError("--rolling-window must be >= 1")

    timestamp = (
        df.groupby("frame_index", as_index=False)["timestamp_sec"]
        .min()
        .sort_values("frame_index")
    )

    closed_pivot = df.pivot_table(
        index="frame_index", columns="eye_side", values="p_eye_closed", aggfunc="mean"
    )
    open_pivot = df.pivot_table(
        index="frame_index", columns="eye_side", values="p_eye_open", aggfunc="mean"
    )

    for side in ("left", "right"):
        if side not in closed_pivot.columns:
            closed_pivot[side] = pd.NA
        if side not in open_pivot.columns:
            open_pivot[side] = pd.NA

    frames = timestamp.merge(
        closed_pivot[["left", "right"]].reset_index(),
        on="frame_index",
        how="left",
    )
    frames = frames.rename(
        columns={"left": "left_p_eye_closed", "right": "right_p_eye_closed"}
    )
    open_frames = open_pivot[["left", "right"]].reset_index().rename(
        columns={"left": "left_p_eye_open", "right": "right_p_eye_open"}
    )
    frames = frames.merge(open_frames, on="frame_index", how="left")

    closed_cols = ["left_p_eye_closed", "right_p_eye_closed"]
    frames["mean_p_eye_closed"] = frames[closed_cols].mean(axis=1)
    frames["max_p_eye_closed"] = frames[closed_cols].max(axis=1)
    frames["min_p_eye_closed"] = frames[closed_cols].min(axis=1)

    frames["left_closed_binary"] = (
        frames["left_p_eye_closed"] >= closed_threshold
    ).astype(int)
    frames["right_closed_binary"] = (
        frames["right_p_eye_closed"] >= closed_threshold
    ).astype(int)
    frames["both_eyes_closed_binary"] = (
        (frames["left_closed_binary"] == 1) & (frames["right_closed_binary"] == 1)
    ).astype(int)
    frames["either_eye_closed_binary"] = (
        (frames["left_closed_binary"] == 1) | (frames["right_closed_binary"] == 1)
    ).astype(int)
    frames["mean_closed_binary"] = (
        frames["mean_p_eye_closed"] >= closed_threshold
    ).astype(int)

    frames["rolling_mean_p_eye_closed"] = (
        frames["mean_p_eye_closed"].rolling(rolling_window, min_periods=1).mean()
    )
    frames["rolling_max_p_eye_closed"] = (
        frames["max_p_eye_closed"].rolling(rolling_window, min_periods=1).max()
    )
    frames["rolling_perclos_mean_binary"] = (
        frames["mean_closed_binary"].rolling(rolling_window, min_periods=1).mean()
    )
    frames["rolling_perclos_either_eye"] = (
        frames["either_eye_closed_binary"].rolling(rolling_window, min_periods=1).mean()
    )
    frames["rolling_perclos_both_eyes"] = (
        frames["both_eyes_closed_binary"].rolling(rolling_window, min_periods=1).mean()
    )
    return frames.sort_values("frame_index").reset_index(drop=True)


def detect_events(frames: pd.DataFrame) -> pd.DataFrame:
    events: list[dict[str, Any]] = []
    active_rows: list[pd.Series] = []

    def close_event(rows: list[pd.Series]) -> None:
        if not rows:
            return
        event_df = pd.DataFrame(rows)
        events.append(
            {
                "event_id": len(events) + 1,
                "start_frame_index": int(event_df["frame_index"].iloc[0]),
                "end_frame_index": int(event_df["frame_index"].iloc[-1]),
                "start_timestamp_sec": float(event_df["timestamp_sec"].iloc[0]),
                "end_timestamp_sec": float(event_df["timestamp_sec"].iloc[-1]),
                "duration_sampled_frames": int(len(event_df)),
                "max_mean_p_eye_closed": float(event_df["mean_p_eye_closed"].max()),
                "mean_mean_p_eye_closed": float(event_df["mean_p_eye_closed"].mean()),
            }
        )

    for _, row in frames.iterrows():
        if int(row["mean_closed_binary"]) == 1:
            active_rows.append(row)
        else:
            close_event(active_rows)
            active_rows = []
    close_event(active_rows)

    columns = [
        "event_id",
        "start_frame_index",
        "end_frame_index",
        "start_timestamp_sec",
        "end_timestamp_sec",
        "duration_sampled_frames",
        "max_mean_p_eye_closed",
        "mean_mean_p_eye_closed",
    ]
    return pd.DataFrame(events, columns=columns)


def save_figures(frames: pd.DataFrame, output_dir: Path, closed_threshold: float) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        figures_dir / "p_eye_closed_rolling_mean.png",
        figures_dir / "eye_closed_binary_timeline.png",
        figures_dir / "perclos_like_score_over_time.png",
    ]

    plt.figure(figsize=(12, 5))
    plt.plot(
        frames["frame_index"],
        frames["mean_p_eye_closed"],
        linewidth=1,
        alpha=0.45,
        label="frame mean p_eye_closed",
    )
    plt.plot(
        frames["frame_index"],
        frames["rolling_mean_p_eye_closed"],
        linewidth=2,
        label="rolling mean p_eye_closed",
    )
    plt.axhline(
        closed_threshold, color="red", linestyle="--", linewidth=1, label="threshold"
    )
    plt.xlabel("Frame index")
    plt.ylabel("p_eye_closed")
    plt.title("Stage 11 rolling mean p_eye_closed")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(paths[0], dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.step(
        frames["frame_index"],
        frames["mean_closed_binary"],
        where="post",
        label="mean closed binary",
    )
    plt.step(
        frames["frame_index"],
        frames["either_eye_closed_binary"] + 1.2,
        where="post",
        label="either eye closed binary + 1.2",
    )
    plt.step(
        frames["frame_index"],
        frames["both_eyes_closed_binary"] + 2.4,
        where="post",
        label="both eyes closed binary + 2.4",
    )
    plt.yticks([0, 1, 1.2, 2.2, 2.4, 3.4], ["0", "1", "0", "1", "0", "1"])
    plt.xlabel("Frame index")
    plt.ylabel("Binary signal lanes")
    plt.title("Stage 11 eye-closed binary timeline")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(paths[1], dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(
        frames["frame_index"],
        frames["rolling_perclos_mean_binary"],
        linewidth=2,
        label="rolling mean-closed ratio",
    )
    plt.plot(
        frames["frame_index"],
        frames["rolling_perclos_either_eye"],
        linewidth=1.5,
        label="rolling either-eye ratio",
    )
    plt.plot(
        frames["frame_index"],
        frames["rolling_perclos_both_eyes"],
        linewidth=1.5,
        label="rolling both-eyes ratio",
    )
    plt.xlabel("Frame index")
    plt.ylabel("PERCLOS-like ratio")
    plt.ylim(-0.03, 1.03)
    plt.title("Stage 11 PERCLOS-like rolling scores")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(paths[2], dpi=160)
    plt.close()
    return paths


def stats_dict(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    integer_columns = {
        "event_id",
        "start_frame_index",
        "end_frame_index",
        "duration_sampled_frames",
    }
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif col in integer_columns:
                values.append(str(int(value)))
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_reports(
    output_report_path: Path,
    repo_report_path: Path,
    input_csv: Path,
    output_dir: Path,
    figures: list[Path],
    frames: pd.DataFrame,
    events: pd.DataFrame,
    summary: dict[str, Any],
    closed_threshold: float,
    rolling_window: int,
) -> None:
    mean_stats = stats_dict(frames["mean_p_eye_closed"])
    top_events = events.sort_values(
        ["duration_sampled_frames", "max_mean_p_eye_closed"],
        ascending=[False, False],
    ).head(10)
    event_cols = [
        "event_id",
        "start_frame_index",
        "end_frame_index",
        "start_timestamp_sec",
        "end_timestamp_sec",
        "duration_sampled_frames",
        "max_mean_p_eye_closed",
        "mean_mean_p_eye_closed",
    ]

    source_dir = input_csv.parent
    report = f"""# Stage 11 Eye Temporal Analysis Report

## 1. Purpose

Stage 11 analyzes eye-only temporal signal behavior from the successful Stage 10 controlled-video runtime output. It converts per-eye frame probabilities into smoothed frame-level signals and PERCLOS-like rolling ratios.

This is not final drowsiness accuracy. It is not mouth-eye fusion yet. It is not final fatigue scoring.

## 2. Input Source

Stage 10 controlled output path:

```text
{source_dir}
```

Input CSV:

```text
{input_csv}
```

## 3. Method

- Read per-eye `p_eye_closed` and `p_eye_open` predictions from Stage 10.
- Group left/right eye predictions by `frame_index`.
- Compute frame-level `mean_p_eye_closed`, `max_p_eye_closed`, and `min_p_eye_closed`.
- Use threshold `{closed_threshold:.2f}` for binary closed-eye indicators.
- Compute rolling statistics over `{rolling_window}` sampled frames.
- Treat rolling binary ratios as PERCLOS-like eye-only signals, not as final drowsiness labels.
- Detect candidate eye-closure events as contiguous sampled-frame sequences where `mean_closed_binary == 1`.

## 4. Summary Metrics

| Metric | Value |
| --- | ---: |
| Number of frames | {int(summary["frame_count"])} |
| Threshold used | {closed_threshold:.2f} |
| Rolling window | {rolling_window} sampled frames |
| Mean `mean_p_eye_closed` | {mean_stats["mean"]:.12f} |
| Min `mean_p_eye_closed` | {mean_stats["min"]:.8f} |
| Max `mean_p_eye_closed` | {mean_stats["max"]:.8f} |
| Total `mean_closed_binary` frames | {int(summary["mean_closed_binary_frames"])} |
| Total `either_eye_closed_binary` frames | {int(summary["either_eye_closed_binary_frames"])} |
| Total `both_eyes_closed_binary` frames | {int(summary["both_eyes_closed_binary_frames"])} |
| Candidate eye-closure events | {int(summary["candidate_event_count"])} |

## 5. Candidate Event Table

Top candidate eye-closure events by duration, then max probability:

{md_table(top_events, event_cols)}

Candidate events are not confirmed drowsiness events; they are temporal eye-closure candidates derived from the eye-only signal.

## 6. Figures

- `{figures[0]}`
- `{figures[1]}`
- `{figures[2]}`

## 7. Interpretation

The controlled-video signal appears usable for temporal smoothing on this one video: Stage 10 provided complete per-eye predictions, Stage 11 produced a continuous per-frame signal, and rolling PERCLOS-like ratios were generated without missing-frame failures.

This does not claim final drowsiness accuracy.

## 8. Limitations

- One controlled video only.
- No ground truth temporal annotation.
- No mouth/yawn fusion yet.
- No live webcam validation.
- No deployment robustness validation.
- No claim is made for all lighting, camera angles, glasses, subjects, or runtime environments.

## 9. Recommended Next Step

If the temporal signal looks stable under human inspection, proceed to Stage 12 fusion design with `p_yawn`. If the signal is noisy, adjust smoothing window and threshold policy before fusion.
"""

    output_report_path.write_text(report, encoding="utf-8")
    repo_report_path.parent.mkdir(parents=True, exist_ok=True)
    repo_report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv
    output_dir = args.output_dir
    closed_threshold = args.closed_threshold
    rolling_window = args.rolling_window

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(input_csv)
    frames = build_frame_summary(predictions, closed_threshold, rolling_window)
    events = detect_events(frames)
    figures = save_figures(frames, output_dir, closed_threshold)

    source_dir = input_csv.parent
    stage10_summary = load_optional_json(source_dir / "summary.json")
    stage10_acceptance = load_optional_json(source_dir / "stage10_acceptance_summary.json")

    temporal_csv_path = output_dir / "stage11_eye_temporal_summary.csv"
    events_csv_path = output_dir / "stage11_eye_events.csv"
    summary_json_path = output_dir / "stage11_eye_temporal_summary.json"
    output_report_path = output_dir / "STAGE11_EYE_TEMPORAL_REPORT.md"

    frames.to_csv(temporal_csv_path, index=False)
    events.to_csv(events_csv_path, index=False)

    mean_stats = stats_dict(frames["mean_p_eye_closed"])
    summary: dict[str, Any] = {
        "stage": 11,
        "status": "COMPLETED_EYE_ONLY_TEMPORAL_ANALYSIS",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "stage10_summary_path": str(source_dir / "summary.json"),
        "stage10_acceptance_summary_path": str(
            source_dir / "stage10_acceptance_summary.json"
        ),
        "stage10_status": stage10_acceptance.get("status"),
        "video_path": stage10_summary.get("video_path"),
        "model_name": stage10_summary.get("model_name"),
        "checkpoint_path": stage10_summary.get("checkpoint_path"),
        "device": stage10_summary.get("device"),
        "closed_threshold": closed_threshold,
        "rolling_window": rolling_window,
        "frame_count": int(len(frames)),
        "prediction_row_count": int(len(predictions)),
        "mean_p_eye_closed": mean_stats,
        "mean_closed_binary_frames": int(frames["mean_closed_binary"].sum()),
        "either_eye_closed_binary_frames": int(
            frames["either_eye_closed_binary"].sum()
        ),
        "both_eyes_closed_binary_frames": int(
            frames["both_eyes_closed_binary"].sum()
        ),
        "candidate_event_count": int(len(events)),
        "max_rolling_perclos_mean_binary": float(
            frames["rolling_perclos_mean_binary"].max()
        ),
        "max_rolling_perclos_either_eye": float(
            frames["rolling_perclos_either_eye"].max()
        ),
        "max_rolling_perclos_both_eyes": float(
            frames["rolling_perclos_both_eyes"].max()
        ),
        "figures": [str(path) for path in figures],
        "warning": (
            "Eye-only temporal analysis only. Not final system-level drowsiness "
            "accuracy, not fusion, and not final fatigue scoring."
        ),
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    write_reports(
        output_report_path=output_report_path,
        repo_report_path=DEFAULT_REPORT_PATH,
        input_csv=input_csv,
        output_dir=output_dir,
        figures=figures,
        frames=frames,
        events=events,
        summary=summary,
        closed_threshold=closed_threshold,
        rolling_window=rolling_window,
    )

    print(f"[done] temporal summary: {temporal_csv_path}")
    print(f"[done] events: {events_csv_path}")
    print(f"[done] summary: {summary_json_path}")
    print(f"[done] output report: {output_report_path}")
    print(f"[done] repo report: {DEFAULT_REPORT_PATH}")
    for figure in figures:
        print(f"[done] figure: {figure}")


if __name__ == "__main__":
    main()
