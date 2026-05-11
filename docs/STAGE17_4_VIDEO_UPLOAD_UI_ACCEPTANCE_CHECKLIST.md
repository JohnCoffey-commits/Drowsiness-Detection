# Stage 17.4 Video Upload UI Acceptance Checklist

This checklist is for manual acceptance of the Stage 17.3/17.4 video-upload warning-candidate MVP and the Stage 17.5 eye evidence calibration refinement.

Acceptance scope:

- Local video-upload UI only.
- Rule-based warning-candidate analysis only.
- No webcam validation.
- No deployment-readiness claim.
- No final system-level drowsiness accuracy claim.
- No model logic, checkpoint, probability-index formula, or Stage 17.1 sustained-eye gate removal during UI validation.

## Manual Acceptance Checklist

Use the repository root:

```bash
cd /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection
make stage17-ui
```

| Check | Expected result | Pass/Fail | Notes |
|---|---|---|---|
| One-command launcher works | `make stage17-ui` starts both local services |  |  |
| Backend starts | Backend is reachable at `http://127.0.0.1:8000` |  |  |
| Frontend starts | Frontend is reachable at `http://127.0.0.1:3000/video-upload` |  |  |
| Dashboard still opens | `http://127.0.0.1:3000/` opens without regression |  |  |
| Sidebar placement | `Video Upload Analysis` appears directly under `Dashboard` |  |  |
| `/video-upload` opens | Upload analysis workstation loads |  |  |
| Test upload succeeds | `C_upload_test.mp4` upload completes successfully |  |  |
| Summary cards render | Duration, sampled frames, normal frames, warning-candidate counts, signal-unreliable count, yawn events, and recent-yawn frames/events are visible |  |  |
| High-confidence frames show Stage 17.1 result | High-confidence warning candidate frame count is visible and interpreted as Stage 17.1 rule-based fusion output |  |  |
| Suppressed brief-eye escalation appears | Suppressed brief-eye escalation metric or notice is visible when returned |  |  |
| Stage 17.5 eye calibration appears | Weak eye-warning, moderate eye-closure candidate, strong eye-closure candidate, and weak-eye suppression fields appear when returned |  |  |
| Strength-aware high-confidence gating appears | High-confidence warning candidates are described as requiring recent mouth/yawn evidence, sustained eye-warning evidence, and calibrated eye-strength evidence |  |  |
| Interval table renders | `Warning-candidate intervals` table is visible |  |  |
| Fusion timeline figure renders | `Fusion timeline` image is visible |  |  |
| `p_eye_closed` figure renders | `Eye signal over time: p_eye_closed` image is visible |  |  |
| `p_yawn` figure renders | `Mouth/yawn signal over time: p_yawn` image is visible |  |  |
| Keyframe gallery renders | Keyframe evidence gallery appears with metadata, not images alone |  |  |
| Technical links render | Report, summary, timeline, fusion timeline, and keyframe metadata links appear when available |  |  |
| Permanent warning text is visible | “This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.” |  |  |
| Unsafe wording is absent | The UI does not claim: `driver is drowsy`, `final drowsiness detected`, `final accuracy`, `deployment-ready`, or `certified alert` |  |  |
| Model/fusion boundary preserved | No model code, checkpoints, `p_eye_closed`, `p_yawn`, or Stage 17.1 sustained-eye gate behavior was removed during validation |  |  |

## Test Video

Recommended manual acceptance video:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/upload_test/C_upload_test.mp4
```

Expected validation markers for this video, based on the Stage 17.3 local UI verification:

- High-confidence warning candidate frames: `9`
- Suppressed brief-eye escalation frames: `8`
- Keyframes: `4`
- Figures: `3`
- Interval table: present

## Manual Notes

| Field | Value |
|---|---|
| Tested date |  |
| Tester |  |
| Test video |  |
| `session_id` |  |
| Observed high-confidence frames |  |
| Observed suppressed brief-eye frames |  |
| Observed keyframes |  |
| Observed figures |  |
| Pass/fail |  |
| Notes |  |

## Acceptance Boundary

Passing this checklist means the local Stage 17.3/17.4 video-upload warning-candidate MVP is ready for demonstration and manual review. It does not mean the system has final system-level drowsiness accuracy, deployment readiness, real-time webcam support, or a trained fusion classifier.
