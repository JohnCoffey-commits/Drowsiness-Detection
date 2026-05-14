# Project Final Status at Stage 16

Last updated: 2026-05-13

Status note: this file is a historical Stage 16 final-integration snapshot. It is no longer the latest project status. The current repository has since added Stage 17 uploaded-video backend/UI work, Stage 17.5 eye-evidence interpretation and `/video-upload` UI cleanup, Stage 18 frontend-only `/history-48h` history page, and Stage 19 `/` Live Monitor local realtime warning-candidate prototype.

For the current live status, read:

- `docs/PROJECT_CURRENT_STATUS.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/STAGE17_5_VIDEO_UPLOAD_UI_SECOND_PASS_CLEANUP.md`
- `docs/STAGE18_HISTORY_48H_UI_PAGE_PLAN.md`

## Current Stage

Stage 16 is complete and superseded.

The project has reached final integration summary, evidence packaging, and demo planning for a controlled-validation prototype.

After Stage 16, the project continued into:

| Later stage | Current result |
| --- | --- |
| Stage 17 | Backend-connected uploaded-video rule-based warning-candidate analysis MVP. |
| Stage 17.5 | Conservative eye-evidence calibration and `/video-upload` evidence review UI polish. |
| Stage 18 | Frontend-only `48h History` page with demo/local warning-candidate history stored in browser `localStorage`. |
| Stage 19 | Local `Live Monitor` webcam preview, 2 FPS sampling, realtime frame evidence, and rule-based temporal warning-candidate state. |

## Final Validated Pipeline

The validated pipeline is modular:

1. Full-face video input.
2. Eye ROI extraction with MediaPipe FaceLandmarker.
3. MRL Eye MobileNetV2 inference to produce `p_eye_closed`.
4. Eye-only temporal analysis and Stage 12 alert rule.
5. Mouth ROI extraction with MediaPipe FaceLandmarker.
6. YawDD/YawDD+ Dash ResNet18 inference to produce `p_yawn`.
7. Stage 15 real synchronized rule-based mouth-eye fusion.

## Completed Components

| Component | Status |
| --- | --- |
| YawDD/YawDD+ Dash mouth/yawn specialist | Completed; ResNet18 selected. |
| MRL Eye open/closed specialist | Completed; MobileNetV2 selected. |
| Runtime eye ROI consistency | Completed on controlled validation videos. |
| Eye-only temporal analysis | Completed. |
| Eye-only alert rule | Completed; `quality_gated_perclos_mean_ge_0.60_consec`. |
| Runtime mouth/yawn inference | Completed on controlled validation videos. |
| Real synchronized mouth-eye fusion | Completed using Stage 12 eye timelines and Stage 14 model-generated mouth timelines. |
| Final evidence package and demo plan | Completed. |

## Key Results

Stage 15 F5 fusion results:

| Video | Key result |
| --- | --- |
| `A_normal_open_baseline` | Mostly normal; no mouth warning or high-confidence candidate. |
| `B_realistic_drowsy_simulation` | Stage 14 detected high `p_yawn` during 14.3s-16.8s; Stage 15 produced high-confidence candidates from 16.882456s to 17.924583s when recent-yawn evidence overlapped eye warning. |
| `C_mild_head_motion` | Signal-unreliable intervals were preserved; no mouth/yawn false positives were used. |
| `D_controlled_long_open_closed` | Eye-warning behavior remained eye-only without mouth escalation. |

## Known Limitations

- Not final system-level drowsiness accuracy.
- Not deployment-ready.
- Not a trained fusion classifier.
- Small controlled-realistic validation set.
- One or few subjects.
- No final ground-truth drowsiness timeline.
- No real-world road validation.

## Next Recommended Action

Use the Stage 16 evidence package as historical integration evidence, but present the current project as a Stage 17.5 / Stage 18 / Stage 19 local warning-candidate frontend/backend MVP:

- Backend-connected `/video-upload` remains the real uploaded-video analysis route.
- `/history-48h` is frontend-only demo/local warning-candidate history.
- `/` is now Live Monitor with local webcam preview, frame sampling, realtime frame evidence, and realtime rule-based warning-candidate state.
- The sidebar currently exposes Live Monitor, Video Upload Analysis, 48h History, and Insights.
- The safe boundary remains: rule-based warning-candidate analysis, not final system-level drowsiness accuracy.

Future technical work should collect more synchronized mouth-eye videos with temporal labels, evaluate across more subjects and conditions, and only then consider live deployment or learned fusion.

Next work should keep the warning-candidate boundary: history work needs a deliberate persistence design, and Live Monitor work still needs alert debounce, alarm policy, and history ingestion design before any user-facing alert behavior.
