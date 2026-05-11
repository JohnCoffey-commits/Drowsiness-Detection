# Stage 13 Mouth-Eye Fusion Design

## 1. Purpose

Stage 13 defines a rule-based mouth-eye fusion design for the modular drowsiness monitoring system.

This stage is:

- A design and prototype stage.
- Not final system-level drowsiness accuracy.
- Not a trained fusion classifier.
- Not deployment readiness.
- Not synchronized runtime fusion validation unless real synchronized mouth/yawn timelines are provided.

The current system has separate specialist modules:

- Mouth/yawn specialist conceptually producing `p_yawn`.
- Eye open/closed specialist producing `p_eye_closed`.

Stage 13 describes how those signals should be combined once synchronized mouth and eye timelines exist.

## 2. Inputs

### Eye-side inputs

Stage 13 consumes Stage 12 eye alert timelines with these key fields:

| Field | Meaning |
| --- | --- |
| `eye_alert_state` | Interpretable eye-side state derived from the Stage 12 rule. |
| `eye_closure_warning_candidate` | Boolean warning candidate from the recommended Stage 12 eye-only rule. |
| `signal_unreliable` | Boolean signal-quality marker, especially for no-face or tracking failures. |
| `mean_p_eye_closed` | Frame-level mean of left/right `p_eye_closed`. |
| `rolling_perclos_mean_binary` | PERCLOS-like rolling ratio based on binary closed-eye predictions. |

### Mouth-side inputs

True fusion requires a synchronized mouth/yawn timeline with:

| Field | Meaning |
| --- | --- |
| `p_yawn` | Mouth/yawn specialist probability or score for yawn. |
| `yawn_event` | Boolean marker for a frame/timepoint classified as a yawn event. |
| `recent_yawn_event` | Boolean marker indicating a yawn event occurred recently. |
| `mouth_signal_status` | Mouth signal status such as `ok`, `missing`, or `unreliable`. |

## 3. Required Mouth Timeline Schema

Any real mouth timeline used for Stage 13 should provide:

| Column | Required | Description |
| --- | --- | --- |
| `video_slug` | Yes | Scenario/video identifier matching the eye timeline slug. |
| `timestamp_sec` | Yes | Timestamp in seconds for timeline alignment. |
| `frame_index` | Optional | Frame index for fallback alignment when timestamps are unavailable. |
| `p_yawn` | Yes | Mouth/yawn probability or score. |
| `yawn_event` | Yes | Boolean or 0/1 event marker. |
| `recent_yawn_event` | Yes | Boolean or 0/1 recent-event marker. |
| `mouth_signal_status` | Yes | Signal status such as `ok`, `missing`, or `unreliable`. |
| `mouth_source` | Yes | Source description, for example `runtime_yawn_model` or `synthetic_design_demo`. |
| `notes` | Yes | Free-text notes for provenance or caveats. |

## 4. Fusion Output Schema

Stage 13 fusion timelines should include:

| Column | Description |
| --- | --- |
| `video_slug` | Scenario/video identifier. |
| `timestamp_sec` | Timeline timestamp in seconds. |
| `frame_index` | Frame index from the eye timeline when available. |
| `eye_state` | Eye-side state: `normal`, `eye_warning_candidate`, or `signal_unreliable`. |
| `mouth_state` | Mouth-side state: `normal`, `mouth_warning_candidate`, or `signal_unreliable`. |
| `fusion_state` | Final fused state. |
| `fusion_reason` | Human-readable reason for the fused state. |
| `signal_quality` | Overall signal-quality summary. |
| `p_eye_closed` | Eye closed probability proxy from Stage 12. |
| `p_yawn` | Mouth/yawn probability or score. |
| `recent_yawn_event` | Boolean recent-yawn marker. |
| `eye_warning_candidate` | Boolean eye warning candidate. |
| `mouth_warning_candidate` | Boolean mouth warning candidate. |
| `high_confidence_drowsiness_candidate` | Boolean high-confidence candidate marker. |

## 5. Fusion States

| State | Meaning |
| --- | --- |
| `normal` | No current eye or mouth warning candidate. |
| `eye_warning_candidate` | Eye-only temporal closure signal is active. |
| `mouth_warning_candidate` | Recent mouth/yawn signal is active. |
| `high_confidence_drowsiness_candidate` | Eye warning and recent yawn co-occur. |
| `signal_unreliable` | Eye/mouth evidence is insufficient or unreliable; this must not be counted as drowsiness. |

## 6. Recommended Tiered Rule

The recommended Stage 13 rule is a tiered quality-aware rule:

1. If the eye signal is unreliable and no recent yawn signal exists:
   - `fusion_state = signal_unreliable`
2. Else if the eye signal is unreliable and a recent yawn exists:
   - `fusion_state = mouth_warning_candidate`
   - Reason should state that the eye signal is unreliable.
3. Else if an eye warning candidate and recent yawn event co-occur:
   - `fusion_state = high_confidence_drowsiness_candidate`
4. Else if an eye warning candidate exists:
   - `fusion_state = eye_warning_candidate`
5. Else if a recent yawn event exists:
   - `fusion_state = mouth_warning_candidate`
6. Else:
   - `fusion_state = normal`

This rule keeps tracking failure separate from drowsiness evidence and only upgrades to high-confidence when independent mouth and eye signals co-occur.

## 7. Why Rule-Based Fusion Is Used Now

Rule-based fusion is used because the repository does not yet contain synchronized runtime `p_yawn` timelines for the A/B/C/D videos. A trained fusion classifier would be premature without:

- Synchronized mouth and eye runtime predictions.
- Temporal labels for drowsiness states.
- Broader validation videos across subjects, lighting, glasses, camera pose, and occlusion.

The rule-based design is interpretable, easy to audit, and consistent with the evidence currently available.

## 8. Future Extension

After collecting synchronized mouth-eye videos with temporal labels, the project may add:

- Runtime mouth/yawn inference on full videos.
- Real synchronized mouth-eye fusion validation.
- A learned fusion classifier or calibrated temporal model.
- Live webcam validation.

Until those artifacts exist, Stage 13 remains fusion design plus offline prototype only.
