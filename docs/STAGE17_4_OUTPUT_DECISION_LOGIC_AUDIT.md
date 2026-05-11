# Stage 17.4 Output Decision Logic Audit

## 1. Scope and claim boundary

This audit covers the current Stage 17.4 uploaded-video output decision logic only. The audited path is the local FastAPI + SystemUI video-upload MVP that runs existing eye and mouth/yawn specialists, applies rule-based temporal fusion, writes run artifacts, and displays the result on `/video-upload`.

This audit does not implement Stage 17.5. It does not retrain models, modify checkpoints, implement webcam behavior, change runtime decision logic, change UI behavior, or overwrite existing outputs.

Permanent claim boundary:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

Safe terms for this document:

- rule-based drowsiness warning-candidate analysis
- eye-warning candidate
- mouth-warning candidate
- high-confidence warning candidate
- signal unreliable

Out-of-scope claims:

- final system-level drowsiness accuracy
- final drowsiness detected
- deployment-ready behavior
- real-world validation
- webcam system behavior

Primary source files checked:

- `src/runtime/system_video_upload_pipeline.py`
- `src/runtime/keyframe_extractor.py`
- `src/runtime/stage10_eye_roi_consistency.py`
- `src/runtime/stage11_eye_temporal_analysis.py`
- `src/runtime/stage12_eye_alert_rule_analysis.py`
- `src/runtime/stage13_mouth_eye_fusion_design.py`
- `src/runtime/stage14_mouth_yawn_runtime.py`
- `src/runtime/stage15_real_mouth_eye_fusion_validation.py`
- `src/backend/app.py`
- `SystemUI/src/app/video-upload/page.tsx`
- `SystemUI/src/components/video-upload/*`
- `SystemUI/src/lib/videoUploadTypes.ts`
- `SystemUI/src/lib/videoUploadUtils.ts`

Supporting docs/reports checked:

- `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md`
- `docs/STAGE17_2_RESULT_INTERPRETATION_SCHEMA_ADDENDUM.md`
- `docs/STAGE17_2_MANUAL_REVIEW_INTERPRETATION_NOTES.md`
- `docs/STAGE17_3_VIDEO_UPLOAD_UI_PAGE_REPORT.md`
- `docs/STAGE17_4_VIDEO_UPLOAD_UI_ACCEPTANCE_CHECKLIST.md`
- `docs/STAGE17_4_DEMO_SCRIPT.md`
- `reports/stage17_video_upload_detection_mvp_report.md`
- `reports/stage17_2_manual_review_interpretation_report.md`
- `reports/stage17_4_video_upload_mvp_stabilization_report.md`
- `reports/stage15_real_mouth_eye_fusion_validation_report.md`

Runtime outputs inspected for schema confirmation:

- `outputs/system_video_upload_runs/upload_680971e85f3e/`
- `outputs/system_video_upload_runs/upload_028476634500/`
- `outputs/system_video_upload_runs/upload_0c505cc8208c/`

## 2. End-to-end decision flow

Current Stage 17.4 flow:

```text
uploaded video
-> FastAPI upload validation and session directory
-> sampled frames
-> eye ROI extraction
-> MRL Eye MobileNetV2 eye model
-> per-eye p_eye_closed rows
-> frame-level p_eye_closed aggregation
-> rolling PERCLOS-like eye temporal rule
-> eye_warning_candidate and signal_unreliable flags
-> mouth ROI extraction
-> YawDD/YawDD+ ResNet18 mouth/yawn model
-> p_yawn timeline
-> yawn_event and recent_yawn_event logic
-> nearest timestamp alignment
-> F5 tiered quality-aware rule-based fusion
-> Stage 17.1 sustained-eye gate
-> final fusion_state rows
-> intervals
-> summary metrics
-> figures
-> keyframes
-> backend JSON response and safe file URLs
-> SystemUI display
```

The backend starts the pipeline with:

```text
src/runtime/system_video_upload_pipeline.py
  --sample-every-n-frames 5
  --max-frames 300
  --save-debug
  --save-keyframes
  --force
```

Source: `src/backend/app.py`, `run_pipeline()`.

## 3. Eye model output logic

Source: `src/runtime/stage10_eye_roi_consistency.py`.

| Item | Current value / logic |
|---|---|
| Selected eye checkpoint path | `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` |
| Model architecture | `torchvision.models.mobilenet_v2(weights=None)` with classifier output replaced by `Linear(..., 2)` |
| Label mapping | `0 = closed`, `1 = open` |
| Preprocessing | `Resize(image_size + 16)`, `CenterCrop(image_size)`, `ToTensor()`, ImageNet normalization |
| Runtime image size | default `224` |
| Exact probability formula | `probs = softmax(logits, dim=1)` |
| `p_eye_closed` | `float(probs[0])` |
| `p_eye_open` | `float(probs[1])` |
| Default closed threshold | `0.50` |
| Threshold location | `stage10_eye_roi_consistency.py` CLI argument `--closed-threshold`, default `0.50`; Stage 17 calls Stage 11 with `--closed-threshold 0.50` |
| Per-eye prediction label | `"closed"` if `p_eye_closed >= closed_threshold`, else `"open"` |
| Eye ROI source | MediaPipe FaceLandmarker eye landmarks for left and right eye crops |

Important confirmation:

- MRL Eye mapping is `0 = closed`, `1 = open`.
- `p_eye_closed` uses softmax index `0`.

Stage 10 writes per-eye rows to `runtime_eye_roi_predictions.csv`. Important fields:

- `frame_index`
- `timestamp_sec`
- `eye_side`
- `p_eye_closed`
- `p_eye_open`
- `pred_label`
- `closed_threshold`
- `status`
- `error`
- crop/debug metadata if enabled

Stage 11 aggregates both eyes by `frame_index` in `src/runtime/stage11_eye_temporal_analysis.py`, `build_frame_summary()`:

| Aggregated field | Exact calculation |
|---|---|
| `left_p_eye_closed` | mean `p_eye_closed` for left eye rows at the frame |
| `right_p_eye_closed` | mean `p_eye_closed` for right eye rows at the frame |
| `mean_p_eye_closed` | mean of left and right `p_eye_closed` |
| `max_p_eye_closed` | max of left and right `p_eye_closed` |
| `min_p_eye_closed` | min of left and right `p_eye_closed` |
| `left_closed_binary` | `left_p_eye_closed >= 0.50` |
| `right_closed_binary` | `right_p_eye_closed >= 0.50` |
| `both_eyes_closed_binary` | left and right closed binaries both true |
| `either_eye_closed_binary` | left or right closed binary true |
| `mean_closed_binary` | `mean_p_eye_closed >= 0.50` |

Stage 17 uses `mean_p_eye_closed` as the aligned fusion column `p_eye_closed`.

## 4. Eye temporal rule logic

Stage 11 builds rolling eye signal fields. Stage 17 then adapts the Stage 12 recommended rule inside `src/runtime/system_video_upload_pipeline.py`, `build_eye_alert_timeline()`.

Core parameters:

| Parameter | Value | Source |
|---|---:|---|
| Stage 11 closed threshold | `0.50` | `system_video_upload_pipeline.py` calls Stage 11 with `--closed-threshold 0.50` |
| Stage 11 rolling window | `5` sampled frames | `system_video_upload_pipeline.py` calls Stage 11 with `--rolling-window 5` |
| Stage 17 eye rule name | `quality_gated_perclos_mean_ge_0.60_consec` | `EYE_RULE_NAME` |
| Eye-warning threshold | `rolling_perclos_mean_binary >= 0.60` | `build_eye_alert_timeline()` |
| Consecutive-frame requirement | at least `2` consecutive sampled rows | `min_consecutive=2` |
| Recent signal-quality window | `5` sampled rows | `recent_quality_window=5` |
| Max recent no-face ratio | `0.20` | `max_recent_no_face_ratio=0.20` |

Rule table:

| Field / rule name | Exact condition | Threshold | Source file/function | Output effect |
|---|---|---:|---|---|
| `mean_closed_binary` | `mean_p_eye_closed >= closed_threshold` | `0.50` | `stage11_eye_temporal_analysis.py`, `build_frame_summary()` | Input to rolling PERCLOS-like mean-binary |
| `rolling_perclos_mean_binary` | rolling mean of `mean_closed_binary` over `5` sampled frames, `min_periods=1` | window `5` | `stage11_eye_temporal_analysis.py`, `build_frame_summary()` | Eye temporal ratio proxy |
| `no_face_binary` | `1` for Stage 10 failure rows where `status == "no_face"` | n/a | `system_video_upload_pipeline.py`, `build_eye_alert_timeline()` | Input to recent no-face quality gate |
| `tracking_failure_binary` | `1` for any Stage 10 failure row appended to the timeline | n/a | `system_video_upload_pipeline.py`, `build_eye_alert_timeline()` | Direct signal-unreliable condition |
| `recent_no_face_ratio` | rolling mean of `no_face_binary` over `5` sampled rows | window `5` | `system_video_upload_pipeline.py`, `build_eye_alert_timeline()` | Input to signal-unreliable condition |
| `signal_unreliable` | `tracking_failure_binary == 1 OR recent_no_face_ratio > 0.20` | `0.20` | `system_video_upload_pipeline.py`, `build_eye_alert_timeline()` | Marks eye signal quality as unreliable |
| `recommended_raw_condition` | `rolling_perclos_mean_binary >= 0.60 AND NOT signal_unreliable` | `0.60` | `system_video_upload_pipeline.py`, `build_eye_alert_timeline()` | Raw eye-warning condition |
| `recommended_alert` | raw condition persists for at least `2` consecutive sampled rows; all rows in qualifying run are marked | `2` rows | `system_video_upload_pipeline.py`, `sustained_alert()` | Becomes `eye_warning_candidate` |
| `eye_warning_candidate` | boolean conversion of `recommended_alert` | n/a | `prepare_eye_for_fusion()` | Input to F5 fusion |

No-face handling:

- Stage 10 no-face rows are not treated as eye-warning evidence.
- Stage 17 sets `signal_unreliable` for any Stage 10 failure row and for recent no-face ratio above `0.20`.
- The raw eye-warning condition explicitly excludes `signal_unreliable`.
- No-face is therefore signal quality evidence, not drowsiness evidence.

## 5. Mouth/yawn model output logic

Source: `src/runtime/stage14_mouth_yawn_runtime.py`.

| Item | Current value / logic |
|---|---|
| Selected mouth checkpoint path | `checkpoints/resnet18_best.pt` |
| Model architecture | `torchvision.models.resnet18(weights=None)` with `fc = Linear(..., 2)` |
| Label mapping | `0 = no_yawn`, `1 = yawn` |
| Preprocessing | RGB conversion, `Resize((224, 224))`, `ToTensor()`, ImageNet normalization |
| Exact probability formula | `probs = softmax(logits, dim=1)` |
| `p_no_yawn` | `float(probs[0])` |
| `p_yawn` | `float(probs[1])`, using `P_YAWN_CLASS_INDEX = 1` |
| Predicted label | `"yawn"` if `p_yawn >= p_no_yawn`, else `"no_yawn"` |
| Yawn event threshold | `p_yawn >= 0.50` by default |
| Threshold location | `stage14_mouth_yawn_runtime.py` CLI `--yawn-threshold`, default `0.50`; backend uses this default |
| Mouth ROI source | MediaPipe FaceLandmarker mouth/lip landmarks |

Important confirmation:

- YawDD/YawDD+ mapping is `0 = no_yawn`, `1 = yawn`.
- `p_yawn` uses softmax index `1`.

Stage 14 writes successful mouth rows to `runtime_mouth_yawn_predictions.csv`. Important fields:

- `frame_index`
- `timestamp_sec`
- `mouth_bbox_*`
- `p_yawn`
- `p_no_yawn`
- `predicted_label`
- `yawn_event`
- `recent_yawn_event`
- `mouth_signal_status`
- `checkpoint_path`
- `model_name`
- `label_mapping`

Mouth failures are written to `mouth_stage14/failures.csv` with:

- `failure_type`: `no_face`, `invalid_mouth_crop`, or `inference_failed`
- `failure_reason`

Stage 17 converts mouth failures into non-yawn mouth timeline rows in `load_stage14_mouth_timeline()`:

- `p_yawn = 0.0`
- `yawn_event = False`
- `recent_yawn_event = False`
- `mouth_signal_status = failure_type`
- note: "not treated as yawn"

## 6. Recent-yawn logic

Source: `src/runtime/stage14_mouth_yawn_runtime.py`, `add_recent_yawn_flags()`.

Logic:

1. Stage 14 processes successful mouth prediction rows in timestamp order.
2. If `yawn_event` is true, `last_event_time` is set to that row timestamp.
3. `recent_yawn_event` is true when:

```text
last_event_time is not None
AND timestamp_sec - last_event_time <= recent_yawn_window_sec
```

Current backend defaults:

| Parameter | Value |
|---|---:|
| `yawn_threshold` | `0.50` |
| `recent_yawn_window_sec` | `8.0` seconds |

Source: `system_video_upload_pipeline.py`, CLI defaults and backend call path.

Recent-yawn persistence:

- `recent_yawn_event` is true on the yawn-event row itself.
- It remains true for rows up to `8.0` seconds after the most recent yawn event.
- It can persist after the visible yawn interval and later interact with eye-warning evidence.

Fusion impact:

- `recent_yawn_event` alone can produce `mouth_warning_candidate`.
- `recent_yawn_event` plus `eye_warning_candidate` initially produces `high_confidence_drowsiness_candidate` under F5.
- Stage 17.1 then requires `sustained_eye_warning` before allowing the high-confidence warning candidate to remain.

## 7. Fusion state decision logic

The current fusion rule is `F5_tiered_quality_aware_fusion`.

Primary source: `src/runtime/stage13_mouth_eye_fusion_design.py`, `apply_fusion_rule()`.

Stage 17 calls:

- `align_real_mouth_timeline()`
- `build_fusion_timeline()`
- `apply_sustained_eye_gate()`

Decision table before the Stage 17.1 sustained-eye gate:

| Priority | Condition | Output `fusion_state` | Reason / explanation | Source file/function |
|---:|---|---|---|---|
| 1 | `eye_unreliable == true` and `recent_yawn_event == false` | `signal_unreliable` | eye signal unreliable and no recent yawn | `stage13_mouth_eye_fusion_design.py`, `apply_fusion_rule()` |
| 2 | `eye_unreliable == true` and `recent_yawn_event == true` | `mouth_warning_candidate` | recent yawn event while eye signal is unreliable | same |
| 3 | `eye_warning_candidate == true` and `recent_yawn_event == true` | `high_confidence_drowsiness_candidate` | eye warning candidate and recent yawn event | same |
| 4 | `eye_warning_candidate == true` | `eye_warning_candidate` | eye warning candidate | same |
| 5 | `recent_yawn_event == true` | `mouth_warning_candidate` | recent yawn event | same |
| 6 | otherwise | `normal` | no warning candidate | same |

Stage 17.1 modifies the row after F5:

| Condition after F5 | Final output | Effect |
|---|---|---|
| `fusion_state == high_confidence_drowsiness_candidate` AND `recent_yawn_event == true` AND `eye_warning_candidate == true` AND `sustained_eye_warning == false` | `mouth_warning_candidate` | Set `high_confidence_suppressed_by_brief_eye_warning = True`, set `high_confidence_drowsiness_candidate = False`, set `mouth_warning_candidate = True`, update `fusion_reason` |

Important exact behavior:

- Mouth signal failures alone do not directly create `fusion_state == signal_unreliable` under F5.
- Mouth failures are represented as `mouth_state == signal_unreliable` and `signal_quality == mouth_unreliable`, but the F5 `fusion_state` branches are controlled by eye unreliability, eye warning, and recent-yawn state.
- This is exact current behavior, not a recommendation.

## 8. Stage 17.1 sustained-eye gate

Source: `src/runtime/system_video_upload_pipeline.py`, `apply_sustained_eye_gate()`.

Constants:

```text
SUSTAINED_EYE_GATE_MIN_DURATION_SEC = 1.0
SUSTAINED_EYE_GATE_MIN_SAMPLED_FRAMES = 5
```

Eye-warning intervals are computed from contiguous rows where:

```text
eye_warning_candidate == true
```

For each contiguous eye-warning interval:

```text
sampled_frames = len(segment)
duration_sec = last_timestamp_sec - first_timestamp_sec
sustained_eye_warning =
  duration_sec >= 1.0
  OR sampled_frames >= 5
```

The condition is OR, not AND.

Field table:

| Field | Meaning | Type | Exact calculation | Source file/function |
|---|---|---|---|---|
| `eye_warning_interval_id` | Contiguous eye-warning interval id | integer | Incremented for each contiguous `eye_warning_candidate` run; `0` outside eye-warning rows | `apply_sustained_eye_gate()` |
| `eye_warning_interval_duration_sec` | Duration of current eye-warning interval | number | last interval timestamp minus first interval timestamp | `apply_sustained_eye_gate()` |
| `eye_warning_interval_sampled_frames` | Sampled rows in current eye-warning interval | integer | `len(segment)` | `apply_sustained_eye_gate()` |
| `sustained_eye_warning` | Whether current eye-warning interval passes Stage 17.1 gate | boolean | `duration_sec >= 1.0 OR sampled_frames >= 5` | `apply_sustained_eye_gate()` |
| `high_confidence_suppressed_by_brief_eye_warning` | Whether a high-confidence escalation was suppressed | boolean | true for initial high-confidence rows with recent-yawn + eye-warning but no sustained-eye evidence | `apply_sustained_eye_gate()` |
| `suppressed_high_confidence_brief_eye_warning_frames` | Summary count of suppressed rows | integer | sum of `high_confidence_suppressed_by_brief_eye_warning` over final fusion timeline | `build_summary()` |
| `sustained_eye_gate_min_duration_sec` | Summary copy of gate duration threshold | number | constant `1.0` | `build_summary()` |
| `sustained_eye_gate_min_sampled_frames` | Summary copy of sampled-frame threshold | integer | constant `5` | `build_summary()` |

Suppression behavior:

- The original `eye_warning_candidate` flag remains true.
- The final `fusion_state` becomes `mouth_warning_candidate`.
- `fusion_reason` becomes:

```text
recent yawn event; high-confidence suppressed because eye-warning interval was brief
```

- `high_confidence_drowsiness_candidate` is set to false.
- `mouth_warning_candidate` is set to true.
- `mouth_state` is set to `mouth_warning_candidate`.

## 9. Signal-unreliable logic

Signal-unreliable source conditions:

| Source condition | Exact current behavior |
|---|---|
| Stage 10 `no_face` failure | appended as an eye failure row with `no_face_binary = 1`, `tracking_failure_binary = 1` |
| Stage 10 `invalid_crop` or `inference_failed` failure | appended as an eye failure row with `tracking_failure_binary = 1` |
| Recent no-face quality window | `recent_no_face_ratio = rolling_mean(no_face_binary, 5 sampled rows)` |
| Eye signal unreliable | `tracking_failure_binary == 1 OR recent_no_face_ratio > 0.20` |
| Eye warning raw condition | excludes signal-unreliable rows |
| F5 without recent yawn | eye unreliable + no recent yawn -> `signal_unreliable` |
| F5 with recent yawn | eye unreliable + recent yawn -> `mouth_warning_candidate` |

No-face is never treated as drowsiness evidence in the current Stage 17.4 fusion path. It is a signal-quality condition.

UI display:

- Summary card: `Signal unreliable`.
- Interval table group: `Signal unreliable`.
- Friendly label: `Signal unreliable`.
- Evidence text: `Face/ROI signal quality may be unreliable`.
- Keyframe group: `Signal unreliable`.

Keyframe extraction:

- `signal_unreliable` intervals are segmented separately.
- Up to two rows per signal-unreliable segment are selected.
- Signal-unreliable keyframes are still subject to global `max_keyframes`.

Important current caveat:

- Mouth ROI failures are not yawn evidence.
- Under F5, mouth failures alone do not make final `fusion_state` become `signal_unreliable`; they can appear in `mouth_state` and `signal_quality`.

## 10. Output files and schemas

Output folder pattern:

```text
outputs/system_video_upload_runs/<session_id>/
```

Actual Stage 17.4 upload-run files include:

| File | Produced by | Main fields | How decision fields are computed | Used by backend/UI? |
|---|---|---|---|---|
| `summary.json` | `system_video_upload_pipeline.py`, `build_summary()` | counts, intervals, means/maxes, keyframes, warning | counts and intervals from final post-gate `fusion_state` timeline | backend returns as `summary`; UI uses heavily |
| `fusion_summary.json` | same as `summary.json` | same payload | same as summary | exposed through safe file URL if requested; not primary UI link |
| `timeline.csv` | `system_video_upload_pipeline.py` | full final fusion timeline | final post-gate timeline | backend `/api/runs/{session_id}/timeline`; UI technical link |
| `fusion_timeline.csv` | `system_video_upload_pipeline.py` | same final fusion timeline | final post-gate timeline | UI technical link |
| `figures/fusion_timeline.png` | `plot_fusion_timeline()` | image | plots `p_eye_closed`, `p_yawn`, and `fusion_state` | backend `fusion_figure_url`; UI figure |
| `figures/p_eye_closed_over_time.png` | `plot_series()` | image | plots final timeline `p_eye_closed` | UI figure |
| `figures/p_yawn_over_time.png` | `plot_series()` | image | plots final timeline `p_yawn` | UI figure |
| `keyframes/high_confidence/*.jpg` | `keyframe_extractor.py` | screenshots | selected from high-confidence intervals if any exist | backend serves URL; UI gallery |
| `keyframes/eye_warning/*.jpg` | `keyframe_extractor.py` | screenshots | selected only when no high-confidence intervals exist | backend serves URL if generated; UI gallery |
| `keyframes/mouth_warning/*.jpg` | `keyframe_extractor.py` | screenshots | selected only when no high-confidence intervals exist | backend serves URL if generated; UI gallery |
| `keyframes/signal_unreliable/*.jpg` | `keyframe_extractor.py` | screenshots | selected from signal-unreliable intervals | backend serves URL if generated; UI gallery |
| `keyframes/keyframes_metadata.csv` | `keyframe_extractor.py` | keyframe rows | copied from selected final timeline rows | backend-derived keyframe URL list; UI gallery/technical link |
| `keyframes/keyframes_metadata.json` | `keyframe_extractor.py` | same as metadata CSV | same | UI technical link |
| `keyframes/keyframes_summary.json` | `system_video_upload_pipeline.py` | keyframe count, metadata paths, max keyframes | output from `extract_keyframes()` | not directly shown by current UI |
| `SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md` | `write_report()` | human-readable run report | from summary | backend `report_url`; UI technical link |
| `pipeline_manifest.csv` | `system_video_upload_pipeline.py` | artifact path table | static manifest list | not directly shown by current UI |
| `mouth_timeline_stage13_schema.csv` | `system_video_upload_pipeline.py` | Stage 14 mouth timeline converted to Stage 13 schema | from successful and failed mouth rows | intermediate |
| `eye_stage10/runtime_eye_roi_predictions.csv` | Stage 10 | per-eye probabilities | model output | intermediate |
| `eye_stage10/failures.csv` | Stage 10 | eye ROI failures | MediaPipe/model errors | intermediate, used by eye alert adapter |
| `eye_stage11/stage11_eye_temporal_summary.csv` | Stage 11 | frame-level/rolling eye fields | aggregated per-eye rows | intermediate, used by eye alert adapter |
| `eye_stage12/stage12_video_alert_timeline_<session_id>.csv` | Stage 17 eye adapter | recommended eye-alert fields | quality-gated rolling rule | intermediate |
| `eye_stage12/eye_alert_summary.json` | Stage 17 eye adapter | eye warning and signal counts | same eye adapter | intermediate |
| `mouth_stage14/runtime_mouth_yawn_predictions.csv` | Stage 14 | mouth/yawn probabilities | model output | intermediate |
| `mouth_stage14/failures.csv` | Stage 14 | mouth ROI failures | MediaPipe/model errors | intermediate |
| `logs/*.log` | backend/pipeline subprocess runner | command stdout/stderr | runtime evidence only | not UI |

Current `keyframes_metadata` fields:

- `keyframe_path`
- `video_path`
- `session_id`
- `frame_index`
- `timestamp_sec`
- `fusion_state`
- `p_eye_closed`
- `p_yawn`
- `recent_yawn_event`
- `warning_type`
- `reason`
- `segment_id`
- `is_primary`

Note: current keyframe metadata does not write `sustained_eye_warning`, even though the UI type can display it if a future backend returns it.

## 11. Summary metrics logic

Source: `src/runtime/system_video_upload_pipeline.py`, `build_summary()`.

| Summary field | Exact source column(s) | Calculation | Source file/function |
|---|---|---|---|
| `total_frames_sampled` | final `fusion_df` rows | `len(fusion_df)` | `build_summary()` |
| `duration_sec` | `timestamp_sec` | max sampled timestamp; not container duration | `run_pipeline()` / `build_summary()` |
| `normal_frames` | `fusion_state` | count final rows equal `normal` | `build_summary()` |
| `eye_warning_candidate_frames` | `fusion_state` | count final rows equal `eye_warning_candidate` | `build_summary()` |
| `mouth_warning_candidate_frames` | `fusion_state` | count final rows equal `mouth_warning_candidate` | `build_summary()` |
| `high_confidence_drowsiness_candidate_frames` | `fusion_state` | count final rows equal `high_confidence_drowsiness_candidate` after Stage 17.1 gate | `build_summary()` |
| `signal_unreliable_frames` | `fusion_state` | count final rows equal `signal_unreliable` | `build_summary()` |
| `first_warning_timestamp_sec` | `fusion_state`, `timestamp_sec` | min timestamp where final state is eye, mouth, or high-confidence warning candidate | `build_summary()` |
| `last_warning_timestamp_sec` | `fusion_state`, `timestamp_sec` | max timestamp where final state is eye, mouth, or high-confidence warning candidate | `build_summary()` |
| `yawn_event_count` | `yawn_event` | boolean sum | `build_summary()` |
| `recent_yawn_event_count` | `recent_yawn_event` | boolean sum | `build_summary()` |
| `mean_p_eye_closed` | `p_eye_closed` | numeric mean over final aligned rows | `build_summary()` |
| `max_p_eye_closed` | `p_eye_closed` | numeric max over final aligned rows | `build_summary()` |
| `mean_p_yawn` | `p_yawn` | numeric mean over final aligned rows | `build_summary()` |
| `max_p_yawn` | `p_yawn` | numeric max over final aligned rows | `build_summary()` |
| `keyframes` | `extract_keyframes()` return | list of metadata rows | `run_pipeline()` |
| keyframe count | `summary["keyframes"]` | `len(summary["keyframes"])` in UI/report context | `write_report()` / UI |
| `suppressed_high_confidence_brief_eye_warning_frames` | `high_confidence_suppressed_by_brief_eye_warning` | boolean sum | `build_summary()` |

Important nuance:

- `first_warning_timestamp_sec` and `last_warning_timestamp_sec` exclude `signal_unreliable`; signal-unreliable intervals are tracked separately.
- `duration_sec` is the last sampled timestamp, not necessarily exact video container duration.

## 12. Interval generation logic

Source: `src/runtime/system_video_upload_pipeline.py`, `intervals_for_state()`.

Frame-level final states are grouped into intervals by scanning the final post-gate `fusion_state` sequence.

Exact behavior:

- Only intervals for requested warning states are generated.
- `normal` intervals are excluded.
- Adjacent rows with the same target `fusion_state` are merged when contiguous in the dataframe order.
- The merge does not check timestamp gaps; it relies on sorted timeline row order.
- `start_frame_index` and `end_frame_index` use first/last row frame indices.
- `start_timestamp_sec` and `end_timestamp_sec` use first/last row timestamps.
- `duration_sampled_frames = len(segment)`.
- `max_p_eye_closed` and `max_p_yawn` are numeric max values within the segment.
- Backend interval objects currently do not include `fusion_reason`, `sustained_eye_warning`, or suppression flags.

Generated summary arrays:

- `high_confidence_intervals`
- `eye_warning_intervals`
- `mouth_warning_intervals`
- `signal_unreliable_intervals`

UI interval table differences:

- UI merges the four arrays and sorts by `start_timestamp_sec`.
- UI adds a local `kind` and `id`.
- UI adds friendly labels and evidence text from `INTERVAL_CONFIG`.
- UI computes display duration as `end_timestamp_sec - start_timestamp_sec`.
- UI marks `brief` if duration `< 1` second or sampled frames `<= 2`.
- UI can mark `Suppressed escalation` only if interval optional fields/text include suppression/downgrade wording. Current backend interval objects do not carry per-interval suppression flags, so the main current suppression display is the summary-level suppressed-frame banner/count.

## 13. Keyframe extraction logic

Source: `src/runtime/keyframe_extractor.py`.

States eligible for keyframes:

- `high_confidence_drowsiness_candidate`
- `eye_warning_candidate`
- `mouth_warning_candidate`
- `signal_unreliable`

Directory mapping:

| `fusion_state` | keyframe directory / UI group |
|---|---|
| `high_confidence_drowsiness_candidate` | `keyframes/high_confidence/` |
| `eye_warning_candidate` | `keyframes/eye_warning/` |
| `mouth_warning_candidate` | `keyframes/mouth_warning/` |
| `signal_unreliable` | `keyframes/signal_unreliable/` |

Selection strategy:

1. Segment the final fusion timeline by state.
2. If any high-confidence segments exist:
   - select rows only from high-confidence segments as primary keyframes.
   - eye-warning and mouth-warning segments are not selected in that case.
3. If no high-confidence segments exist:
   - select rows from eye-warning and mouth-warning segments.
4. Always add signal-unreliable segment examples separately, up to two rows per signal-unreliable segment.
5. For each selected segment, candidate row positions are:
   - first row
   - midpoint row
   - last row if segment length is at least 4
   - max-score row
6. Score columns:
   - high-confidence: `p_eye_closed + p_yawn`
   - eye-warning: `p_eye_closed`
   - mouth-warning: `p_yawn`
   - signal-unreliable: `p_eye_closed + p_yawn`
7. Deduplicate by `(warning_type, frame_index)`.
8. Stop at `max_keyframes`, currently `20` in Stage 17.

Metadata written:

- `keyframe_path`
- `video_path`
- `session_id`
- `frame_index`
- `timestamp_sec`
- `fusion_state`
- `p_eye_closed`
- `p_yawn`
- `recent_yawn_event`
- `warning_type`
- `reason`
- `segment_id`
- `is_primary`

UI grouping:

- UI groups keyframes by `warning_type` or `fusion_state` text via `keyframeKind()`.
- UI displays image, timestamp, frame index, friendly fusion state, `p_eye_closed`, `p_yawn`, `recent_yawn_event`, optional `sustained_eye_warning`, reason, and optional manual review fields.

## 14. Backend API response logic

Source: `src/backend/app.py`.

`POST /api/analyze-video` request:

- `multipart/form-data`
- required file field: `file`
- allowed extensions: `.mp4`, `.mov`, `.avi`, `.m4v`
- max upload size: `750 MB`

Session and storage:

- Session id: `upload_` + first 12 hex chars of UUID4.
- Uploaded file path:

```text
outputs/system_video_upload_runs/<session_id>/input/<sanitized_filename>
```

Backend response fields from `build_response()`:

| Field | Source |
|---|---|
| `session_id` | generated backend session id |
| `status` | `summary.pipeline_status`, default `completed` |
| `summary` | parsed `summary.json` |
| `warning_counts` | selected frame counts copied from summary |
| `timeline_url` | `/api/runs/{session_id}/timeline` |
| `fusion_figure_url` | `/api/runs/{session_id}/files/figures/fusion_timeline.png` |
| `keyframes` | summary keyframes with safe API `url` added |
| `report_url` | `/api/runs/{session_id}/files/SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md` |
| `warning` | permanent warning text |
| `runtime_duration_sec` | backend wall-clock pipeline duration |
| `audit_log` | backend pipeline log path |

Backend file-serving routes:

- `GET /api/runs/{session_id}/summary`
- `GET /api/runs/{session_id}/timeline`
- `GET /api/runs/{session_id}/keyframes`
- `GET /api/runs/{session_id}/files/{relative_path:path}`

Path safety:

- `session_dir(session_id)` accepts only `[A-Za-z0-9_.-]+`.
- `safe_session_file()` resolves the session root and candidate file path.
- A file is served only if the resolved candidate is inside the resolved session directory and exists as a file.
- `keyframe_urls()` adds a URL only when the stored keyframe path resolves inside the session directory.
- The backend does not expose arbitrary local paths through the response URL fields.

Backend transformation:

- The backend does not recompute fusion decisions.
- It runs the pipeline, loads `summary.json`, copies selected fields into `warning_counts`, and adds API URLs.

## 15. SystemUI display and interpretation logic

SystemUI source:

- `SystemUI/src/app/video-upload/page.tsx`
- `SystemUI/src/components/video-upload/*`
- `SystemUI/src/lib/videoUploadTypes.ts`
- `SystemUI/src/lib/videoUploadUtils.ts`

The UI does not recompute model probabilities, eye-warning candidates, mouth-warning candidates, high-confidence warning candidates, signal-unreliable states, or Stage 17.1 gate decisions. It displays backend output with local formatting and safe URL handling.

UI component table:

| UI component | Input fields | Displayed labels | Any derived logic | Source file |
|---|---|---|---|---|
| `VideoUploadAnalysis` upload card | local file object, backend URL | file name, size, type, backend URL | validates URL starts with `http://` or `https://`; sends `FormData(file)` | `VideoUploadAnalysis.tsx` |
| `PipelineIndicator` | local loading state | seven static pipeline steps | simulated active step every 1.3s; no real progress | `VideoUploadAnalysis.tsx` |
| `ResultHeader` | `result.summary`, `warning_counts` | analysis completed/failed, session id, pipeline status, duration, sampled frames, Stage 17.1/17.2 labels | high-confidence presence from summary with warning-count fallback; signal-unreliable overview | `VideoUploadAnalysis.tsx` |
| `InterpretationNotice` | permanent constant | permanent warning; rule-based fusion guidance | no backend logic | `InterpretationNotice.tsx` |
| `AnalysisSummaryCards` | `summary`, `warning_counts` | duration, sampled frames, normal, eye-warning, mouth-warning, high-confidence, signal unreliable, yawn events, recent-yawn, suppressed brief-eye escalation | `frameCount()` uses summary first, warning-count fallback | `AnalysisSummaryCards.tsx` |
| `IntervalReviewTable` | summary interval arrays | friendly state labels, time range, duration, sampled frames, evidence, max probabilities, priority, evidence link | merges arrays, sorts by start time, computes brief badge, optional suppressed badge, friendly evidence text | `IntervalReviewTable.tsx`, `videoUploadUtils.ts` |
| `InterpretationCard` | static content | Stage 17.1 / Stage 17.2 interpretation guidance | no backend logic | `VideoUploadAnalysis.tsx` |
| `FiguresSection` | `session_id`, figure URL fields | fusion timeline, eye signal over time, mouth/yawn signal over time | constructs safe file URLs; hides failed images | `VideoUploadAnalysis.tsx` |
| `KeyframeEvidenceGallery` | `keyframes` | grouped keyframe cards | groups by `warning_type` / `fusion_state`; friendly labels; safe image URL | `KeyframeEvidenceGallery.tsx`, `videoUploadUtils.ts` |
| `TechnicalEvidencePanel` | `session_id`, `report_url`, `timeline_url` | report, summary, timeline, fusion timeline, keyframe metadata links | constructs safe backend URLs only | `TechnicalEvidencePanel.tsx` |
| copy summary | `VideoUploadResponse` | safe text summary | builds safe warning-candidate wording; does not change backend data | `buildCopyableSummary()` |

Friendly state mapping:

| Raw state text contains | UI label |
|---|---|
| `high_confidence` | High-confidence warning candidate |
| `eye_warning` | Eye-warning candidate |
| `mouth_warning` | Mouth-warning candidate |
| `signal_unreliable` | Signal unreliable |
| `normal` | Normal |

Safe URL logic:

- `safeBackendUrl()` rejects `file:`, local absolute paths such as `/Users/`, `/private/`, `/var/`, `/tmp/`, and Windows drive paths.
- `safeSessionFileUrl()` requires a safe session id and rejects `..` or `~` in relative paths.
- Current UI technical links use backend API paths, not absolute local filesystem paths.

Permanent warning locations:

- Upload panel side warning.
- `InterpretationNotice`.
- Copyable summary.
- Backend response and summary also carry the same warning text.

## 16. Current known limitation relevant to Stage 17.5

Current limitation:

`eye_warning_candidate` is too coarse for interpretation. It can mix:

- true eye closure
- reduced eye openness
- smiling/squinting
- blink-like activity
- fatigue-like small eye opening
- uncertain ROI / angle / lighting effects

The current Stage 17.4 system treats `p_eye_closed` as a model probability input to rule-based fusion. It does not calibrate that probability into human interpretation tiers.

Observed runtime-output examples from existing `B_upload_test.mp4` sessions `upload_028476634500` and `upload_0c505cc8208c`:

- `p_eye_closed` around `0.55` to `0.61` appears in several `eye_warning_candidate` rows because the rolling PERCLOS-like rule and persistence condition are active. These values should not automatically be presented as strong full-closure evidence.
- Some rows with similar `p_eye_closed` values can remain `normal` when the rolling temporal condition is not active. This confirms that Stage 17.4 output is temporal-rule based, not single-probability based.
- `p_eye_closed` around `0.894` appears as stronger eye-closure evidence in the same output set. This is still model-derived evidence, not final truth.

Stage 17.5 should add interpretation calibration and wording refinement. It should not retrain models or change runtime decision logic unless a separate implementation task explicitly authorizes that.

Recommended Stage 17.5 direction:

- Keep the Stage 17.1 sustained-eye gate unchanged.
- Add calibrated interpretation bands for `p_eye_closed` and/or interval evidence.
- Distinguish weak/moderate/strong eye-warning evidence in reporting.
- Preserve warning-candidate wording.

## 17. Exact questions / uncertainties found during audit

| Uncertainty | What is unclear | Files checked | Recommended follow-up |
|---|---|---|---|
| Keyframe `sustained_eye_warning` metadata | UI can display `sustained_eye_warning`, but current `keyframe_extractor.py` does not write this field into metadata | `keyframe_extractor.py`, `KeyframeEvidenceGallery.tsx`, `videoUploadTypes.ts` | Stage 17.5 docs or a later UI/data task should decide whether to add this metadata field |
| Interval-level suppression metadata | Summary has `suppressed_high_confidence_brief_eye_warning_frames`, but current interval objects do not carry suppression flags or reasons | `system_video_upload_pipeline.py`, `IntervalReviewTable.tsx` | Add interval-level suppression fields only in a future implementation task if needed |
| Mouth failure effect on final fusion state | F5 marks mouth failures as `mouth_state` / `signal_quality`, but does not set final `fusion_state` to `signal_unreliable` for mouth-only failure | `stage13_mouth_eye_fusion_design.py`, `system_video_upload_pipeline.py` | Confirm whether Stage 18/17.5 interpretation should explain mouth ROI failure separately |
| Timestamp alignment tolerance | `align_real_mouth_timeline()` uses nearest timestamp alignment without explicit tolerance | `stage13_mouth_eye_fusion_design.py` | Consider documenting or adding a tolerance in a future implementation task if sampling paths diverge |
| Interval duration semantics | Interval duration is `last_timestamp - first_timestamp`, so one-row intervals have `0.0` seconds even though they represent one sampled frame | `system_video_upload_pipeline.py`, `videoUploadUtils.ts` | Preserve current behavior for audit; consider display clarification later |
| `duration_sec` semantics | `summary.duration_sec` is max sampled timestamp, not exact video container duration | `system_video_upload_pipeline.py` | Update future docs/UI wording if exact container duration is required |
| Eye probability calibration | The system has no calibrated interpretation tiers for `p_eye_closed` | runtime outputs, Stage 17.2 docs | Stage 17.5 should address interpretation calibration without retraining |

## 18. Machine-readable checklist for ChatGPT

### Checklist to tell ChatGPT

- Scope: Stage 17.4 is local uploaded-video rule-based drowsiness warning-candidate analysis only; not webcam, not deployment-ready, not final system-level drowsiness accuracy.
- Permanent warning: "This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy."
- Eye model: MobileNetV2 checkpoint `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`; MRL Eye mapping `0 = closed`, `1 = open`; `p_eye_closed = softmax(logits)[0]`; `p_eye_open = softmax(logits)[1]`.
- Eye preprocessing: resize to `image_size + 16`, center crop to `224`, tensor conversion, ImageNet normalization.
- Eye binary threshold: `p_eye_closed >= 0.50`; frame-level `mean_p_eye_closed` is mean of left/right eye probabilities; `mean_closed_binary = mean_p_eye_closed >= 0.50`.
- Eye rolling rule: `rolling_perclos_mean_binary` is rolling mean of `mean_closed_binary` over `5` sampled rows.
- Stage 17 eye-warning rule: `recommended_raw_condition = rolling_perclos_mean_binary >= 0.60 AND NOT signal_unreliable`; `recommended_alert` requires at least `2` consecutive sampled rows.
- Signal-unreliable rule: Stage 10 failure row or `recent_no_face_ratio > 0.20` over `5` sampled rows makes `signal_unreliable`; no-face is quality evidence, not drowsiness evidence.
- Mouth model: ResNet18 checkpoint `checkpoints/resnet18_best.pt`; YawDD/YawDD+ mapping `0 = no_yawn`, `1 = yawn`; `p_yawn = softmax(logits)[1]`.
- Mouth preprocessing: RGB mouth ROI, resize to `224 x 224`, tensor conversion, ImageNet normalization.
- Yawn event: `yawn_event = p_yawn >= 0.50`.
- Recent-yawn: after any yawn event, `recent_yawn_event` remains true through `timestamp_sec - last_yawn_event_time <= 8.0`.
- F5 fusion before Stage 17.1 gate: eye unreliable + no recent yawn -> `signal_unreliable`; eye unreliable + recent yawn -> `mouth_warning_candidate`; eye warning + recent yawn -> `high_confidence_drowsiness_candidate`; eye warning only -> `eye_warning_candidate`; recent yawn only -> `mouth_warning_candidate`; otherwise -> `normal`.
- Stage 17.1 sustained-eye gate: high-confidence remains only if `sustained_eye_warning == true`; sustained means current eye-warning interval duration `>= 1.0s` OR sampled frames `>= 5`.
- Stage 17.1 suppression: if initial high-confidence has recent-yawn + eye-warning but not sustained-eye, final state becomes `mouth_warning_candidate`, `high_confidence_suppressed_by_brief_eye_warning = true`, and summary increments `suppressed_high_confidence_brief_eye_warning_frames`.
- Summary counts are computed from final post-gate `fusion_state`; `duration_sec` is max sampled timestamp; means/maxes are over aligned fusion rows.
- Intervals are contiguous same-state runs for high-confidence, eye-warning, mouth-warning, and signal-unreliable states; normal intervals are excluded.
- Keyframes: if high-confidence intervals exist, keyframes are selected from high-confidence intervals; otherwise from eye-warning and mouth-warning intervals; signal-unreliable examples are added separately; max keyframes is `20`.
- Output files: `summary.json`, `timeline.csv`, `fusion_timeline.csv`, `fusion_summary.json`, figures (`fusion_timeline.png`, `p_eye_closed_over_time.png`, `p_yawn_over_time.png`), keyframe folders, `keyframes_metadata.csv/json`, `keyframes_summary.json`, `SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md`, `pipeline_manifest.csv`, plus intermediate `eye_stage10`, `eye_stage11`, `eye_stage12`, and `mouth_stage14` artifacts.
- Backend: `POST /api/analyze-video` accepts multipart `file`; validates `.mp4/.mov/.avi/.m4v`; max size `750 MB`; session id `upload_<uuid12>`; returns summary, warning counts, timeline URL, fusion figure URL, keyframe URLs, report URL, warning.
- Backend path safety: session id regex plus resolved-path containment under `outputs/system_video_upload_runs/<session_id>`; keyframe URLs only generated for paths inside the session directory.
- UI: SystemUI does not recompute decisions; it formats backend fields, merges/sorts intervals, creates safe URLs, maps raw states to friendly labels, displays permanent warning text, and supports copyable safe-worded summary.
- Known Stage 17.5 gap: `eye_warning_candidate` is too coarse; it can mix true closure, reduced eye openness, squint/smile, blink-like activity, fatigue-like small eye opening, and ROI/angle/lighting effects. Stage 17.5 should add interpretation calibration, not retrain.

Chinese summary to paste into ChatGPT:

当前 Stage 17.4 是本地上传视频的 rule-based drowsiness warning-candidate analysis，不是 webcam，也不是最终系统级疲劳准确率。眼模型使用 `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`，MRL Eye 标签是 `0=closed, 1=open`，`p_eye_closed=softmax(logits)[0]`；嘴部模型使用 `checkpoints/resnet18_best.pt`，YawDD/YawDD+ 标签是 `0=no_yawn, 1=yawn`，`p_yawn=softmax(logits)[1]`。眼部规则是 `rolling_perclos_mean_binary >= 0.60` 且连续至少 2 个 sampled rows，并排除 `signal_unreliable`；`signal_unreliable` 来自 Stage 10 failure 或 5 帧窗口内 `recent_no_face_ratio > 0.20`。`yawn_event=p_yawn>=0.50`，`recent_yawn_event` 在 yawn 后 8 秒内保持 true。F5 fusion 规则是：eye unreliable + no recent yawn -> signal unreliable；eye unreliable + recent yawn -> mouth-warning candidate；eye warning + recent yawn -> high-confidence warning candidate；eye warning only -> eye-warning candidate；recent yawn only -> mouth-warning candidate；否则 normal。Stage 17.1 sustained-eye gate 要求 eye-warning interval 持续至少 1.0 秒或至少 5 个 sampled frames，否则 high-confidence 会被降为 mouth-warning candidate 并记录 suppressed brief-eye escalation。Stage 17.5 的重点应是 eye evidence calibration / interpretation refinement，因为当前 `eye_warning_candidate` 可能混合 true closure、reduced eye openness、smiling/squinting、blink-like activity、fatigue-like small eye opening 和 ROI/angle/lighting effects；不要 retrain，不要改 fusion logic。
