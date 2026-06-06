# Stage 17 Video Upload Result Schema

Stage 17 outputs rule-based drowsiness warning-candidate analysis for one uploaded video. It does not output final drowsy/not-drowsy truth.

## `summary.json`

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `session_id` | string | Unique upload/run session identifier. |
| `input_video_path` | string | Local input video path used by the pipeline. |
| `created_at` | string | UTC timestamp when the pipeline started. |
| `pipeline_status` | string | `completed` or `failed`. |
| `total_frames_sampled` | integer | Number of sampled timeline rows after alignment. |
| `duration_sec` | number | Last sampled timestamp in seconds. |
| `normal_frames` | integer | Frames where fusion state is `normal`. |
| `eye_warning_candidate_frames` | integer | Frames where fusion state is `eye_warning_candidate`. |
| `mouth_warning_candidate_frames` | integer | Frames where fusion state is `mouth_warning_candidate`. |
| `high_confidence_drowsiness_candidate_frames` | integer | Frames where eye-warning evidence and recent-yawn evidence overlapped. |
| `signal_unreliable_frames` | integer | Frames where signal quality was unreliable. |
| `first_warning_timestamp_sec` | number or null | First timestamp of any warning-candidate state. |
| `last_warning_timestamp_sec` | number or null | Last timestamp of any warning-candidate state. |
| `high_confidence_intervals` | array | Contiguous high-confidence warning-candidate intervals. |
| `eye_warning_intervals` | array | Contiguous eye-warning candidate intervals. |
| `mouth_warning_intervals` | array | Contiguous mouth-warning candidate intervals. |
| `signal_unreliable_intervals` | array | Contiguous signal-unreliable intervals. |
| `yawn_event_count` | integer | Number of sampled rows where `p_yawn >= threshold`. |
| `recent_yawn_event_count` | integer | Number of sampled rows with active recent-yawn memory. |
| `suppressed_high_confidence_brief_eye_warning_frames` | integer | Frames where eye+recent-yawn overlap was suppressed from high-confidence because the eye-warning interval was too brief. |
| `suppressed_high_confidence_weak_eye_evidence_frames` | integer | Frames where eye+recent-yawn overlap passed the sustained-eye gate but was suppressed from high-confidence by Stage 17.5 because calibrated eye evidence remained weak. |
| `sustained_eye_gate_min_duration_sec` | number | Minimum eye-warning interval duration for high-confidence escalation. |
| `sustained_eye_gate_min_sampled_frames` | integer | Minimum sampled-frame count for high-confidence escalation. |
| `stage17_5_eye_evidence_calibration` | object | Provisional Stage 17.5 interpretation thresholds and strength-gate settings. |
| `weak_eye_warning_evidence_frames` | integer | Eye-warning candidate frames calibrated as weak evidence. |
| `moderate_eye_closure_candidate_frames` | integer | Sampled rows calibrated as moderate eye-closure candidate evidence. |
| `strong_eye_closure_candidate_frames` | integer | Sampled rows calibrated as strong eye-closure candidate evidence. |
| `eye_evidence_strength_counts` | object | Counts of `none`, `weak`, `moderate`, `strong`, and `signal_unreliable` calibrated eye evidence rows. |
| `mean_p_eye_closed` | number | Mean eye-closed probability over aligned sampled rows. |
| `max_p_eye_closed` | number | Maximum eye-closed probability over aligned sampled rows. |
| `mean_p_yawn` | number | Mean yawn probability over aligned sampled rows. |
| `max_p_yawn` | number | Maximum yawn probability over aligned sampled rows. |
| `keyframes` | array | Keyframe metadata rows. |
| `limitations` | array | Run-level limitations and claim boundaries. |
| `warning` | string | Required claim-boundary warning. |

Required warning string:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

## Interval Object

Interval arrays use:

| Field | Type |
| --- | --- |
| `start_frame_index` | integer |
| `end_frame_index` | integer |
| `start_timestamp_sec` | number |
| `end_timestamp_sec` | number |
| `duration_sampled_frames` | integer |
| `max_p_eye_closed` | number |
| `max_p_yawn` | number |
| `eye_evidence_strength` | string |
| `eye_evidence_label` | string |
| `eye_evidence_interpretation` | string |
| `eye_strength_gate_passed` | boolean |
| `eye_strength_gate_reason` | string |
| `eye_strength_interval_mean_p_eye_closed` | number |
| `eye_strength_interval_max_p_eye_closed` | number |
| `eye_strength_interval_strong_frame_count` | integer |
| `eye_strength_interval_moderate_or_strong_frame_count` | integer |
| `high_confidence_suppressed_by_weak_eye_evidence` | boolean |

## `timeline.csv` and `fusion_timeline.csv`

Both files contain the aligned Stage 17 fusion timeline. Important columns include:

| Column | Description |
| --- | --- |
| `video_slug` | Session identifier. |
| `timestamp_sec` | Timeline timestamp. |
| `frame_index` | Original video frame index. |
| `eye_state` | Eye-side state. |
| `mouth_state` | Mouth-side state. |
| `fusion_state` | Final rule-based fusion state. |
| `fusion_reason` | Human-readable rule reason. |
| `signal_quality` | Signal-quality summary. |
| `p_eye_closed` | Eye closed probability. |
| `p_yawn` | Yawn probability. |
| `recent_yawn_event` | Recent-yawn memory flag. |
| `yawn_event` | Current-frame yawn event flag. |
| `eye_warning_candidate` | Eye warning flag. |
| `mouth_warning_candidate` | Mouth warning flag. |
| `high_confidence_drowsiness_candidate` | High-confidence warning-candidate flag. |
| `eye_warning_interval_id` | Contiguous eye-warning interval id, 0 outside eye-warning intervals. |
| `eye_warning_interval_duration_sec` | Duration of the current eye-warning interval in seconds. |
| `eye_warning_interval_sampled_frames` | Number of sampled frames in the current eye-warning interval. |
| `sustained_eye_warning` | True when the current eye-warning interval is long enough for high-confidence escalation. |
| `high_confidence_suppressed_by_brief_eye_warning` | True when high-confidence escalation was suppressed by the Stage 17.1 sustained-eye gate. |
| `eye_evidence_strength` | Stage 17.5 calibrated value: `none`, `weak`, `moderate`, `strong`, or `signal_unreliable`. |
| `eye_evidence_label` | Safe user-facing label such as `Weak eye-warning evidence`, `Moderate eye-closure candidate`, or `Strong eye-closure candidate`. |
| `eye_evidence_interpretation` | Safe interpretation text for the calibrated eye evidence. |
| `eye_strength_gate_passed` | True when the current eye-warning interval passes Stage 17.5 strength-aware high-confidence gating. |
| `eye_strength_gate_reason` | Rule-based explanation for the Stage 17.5 strength gate result. |
| `high_confidence_suppressed_by_weak_eye_evidence` | True when high-confidence escalation was suppressed because calibrated eye evidence remained weak. |

## Stage 17.1 Sustained-Eye Gate

High-confidence warning candidates require:

1. `recent_yawn_event == true`
2. `eye_warning_candidate == true`
3. `sustained_eye_warning == true`

`sustained_eye_warning` is true when either:

- current eye-warning interval duration is at least `1.0` second, or
- current eye-warning interval has at least `5` sampled frames.

If recent-yawn and eye-warning evidence overlap but the eye-warning interval is too brief, the frame remains a `mouth_warning_candidate` rather than being upgraded to `high_confidence_drowsiness_candidate`.

## Stage 17.5 Eye Evidence Calibration

Stage 17.5 adds a provisional rule-based interpretation layer for eye evidence strength. It does not change the MRL Eye model, checkpoint, preprocessing, `p_eye_closed = softmax(logits)[0]`, or the base eye-warning rule.

Calibration thresholds:

| Calibrated field | Rule |
|---|---|
| `weak` | `p_eye_closed >= 0.50`, or a temporal eye-warning candidate row with lower current `p_eye_closed` |
| `moderate` | `p_eye_closed >= 0.70` |
| `strong` | `p_eye_closed >= 0.85` |
| `signal_unreliable` | Eye signal quality is unreliable |

Stage 17.5 strength-aware high-confidence gate:

High-confidence warning candidates require the existing Stage 17.1 sustained-eye gate and at least one of:

- eye-warning interval mean `p_eye_closed >= 0.70`
- eye-warning interval max `p_eye_closed >= 0.85`
- at least `1` strong eye-closure candidate frame
- at least `2` moderate-or-strong eye evidence frames

If recent-yawn and sustained eye-warning evidence overlap but Stage 17.5 eye evidence remains weak, the frame remains a `mouth_warning_candidate`. This keeps reduced-eye-openness and fatigue-like weak evidence visible without overstating it as high-confidence.

## Keyframe Metadata

`keyframes/keyframes_metadata.csv` and `keyframes/keyframes_metadata.json` contain:

| Column | Description |
| --- | --- |
| `keyframe_path` | Local screenshot path. |
| `video_path` | Input video path. |
| `session_id` | Session identifier. |
| `frame_index` | Original frame index. |
| `timestamp_sec` | Timestamp in seconds. |
| `fusion_state` | Fusion state for the screenshot. |
| `p_eye_closed` | Eye closed probability. |
| `p_yawn` | Yawn probability. |
| `recent_yawn_event` | Recent-yawn memory flag. |
| `sustained_eye_warning` | Stage 17.1 sustained-eye flag. |
| `eye_evidence_strength` | Stage 17.5 calibrated eye evidence strength. |
| `eye_evidence_label` | Safe label for the calibrated eye evidence. |
| `eye_evidence_interpretation` | Safe interpretation text for the calibrated eye evidence. |
| `eye_strength_gate_passed` | Stage 17.5 strength gate flag. |
| `eye_strength_gate_reason` | Stage 17.5 gate explanation. |
| `high_confidence_suppressed_by_weak_eye_evidence` | True when Stage 17.5 suppressed high-confidence escalation. |
| `warning_type` | Keyframe bucket. |
| `reason` | Fusion reason. |
| `segment_id` | Contiguous segment identifier. |
| `is_primary` | True for high-confidence keyframes. |

## API Response

`POST /api/analyze-video` returns:

| Field | Description |
| --- | --- |
| `session_id` | Run id. |
| `status` | Pipeline status. |
| `summary` | Parsed `summary.json`. |
| `warning_counts` | Frame counts by fusion state. |
| `timeline_url` | CSV endpoint. |
| `fusion_figure_url` | Fusion timeline figure endpoint. |
| `keyframes` | Keyframe metadata plus static URLs. |
| `report_url` | Markdown report endpoint. |
| `warning` | Claim-boundary warning. |
