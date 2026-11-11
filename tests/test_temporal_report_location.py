"""Regression for the real non-root container failure in Stage 11 reporting."""

import csv
import subprocess
import sys
from pathlib import Path


def test_temporal_cli_keeps_reports_in_selected_session(tmp_path):
    source = tmp_path / "predictions.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "timestamp_sec", "eye_side", "p_eye_closed", "p_eye_open"])
        for i in range(6):
            for side in ("left", "right"):
                writer.writerow([i, i * 0.2, side, 0.8, 0.2])
    output = tmp_path / "session"
    # A file at the old global directory makes accidental fallback fail even as root.
    (tmp_path / "reports").write_text("Do not overwrite global report storage")
    script = Path(__file__).resolve().parents[1] / "src/runtime/stage11_eye_temporal_analysis.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-csv",
            str(source),
            "--output-dir",
            str(output),
            "--repo-report",
            str(output / "report.md"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "report.md").is_file()
    assert (output / "stage11_eye_temporal_summary.csv").is_file()
    assert (tmp_path / "reports").read_text() == "Do not overwrite global report storage"
