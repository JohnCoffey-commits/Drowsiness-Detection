# Report Evidence and Claims Boundary Guide

## 1. Purpose of This Document

This guide helps you describe VisionGuard correctly in reports, viva presentations, GitHub READMEs, resumes, portfolios, or interviews without overstating the evidence.

The goal is to map “what evidence exists” to “what can be claimed.”

## 2. Main Claim Boundary

Recommended main wording:

> VisionGuard is a modular driver drowsiness monitoring system that combines specialist visual evidence and rule-based temporal fusion to produce warning-candidate states.

Do not describe it as:

- a fully validated drowsiness diagnosis system;
- a certified driver safety system;
- an end-to-end drowsy/not-drowsy classifier;
- a system that detects drowsiness with final accuracy.

Unless future labelled video-level full-system evaluation exists, do not claim final full-system drowsiness accuracy.

## 3. Evidence Types in This Project

| Evidence type | What it supports | What it does not support | Where to find it |
|---|---|---|---|
| dataset/preprocessing evidence | Data source, label mapping, manifest, split, crop traceability | Runtime accuracy | `artifacts/mappings/`, preprocessing docs |
| specialist model metrics | Eye open/closed or no-yawn/yawn classifier performance | Full-system drowsiness accuracy | `outputs/mrl_eye/results/`, `report_assets/mouth_yawn_evaluation_refresh/` |
| confusion matrices | Class-level correct/incorrect patterns | Temporal fusion correctness | `outputs/mrl_eye/figures/`, `report_assets/.../figures/` |
| ROC/AUC / PR curves | Probability-ranking behavior | Correctness of every runtime alert | `report_assets/mouth_yawn_evaluation_refresh/figures/`, MRL figures if available |
| training curves | Training/validation behavior | Real-driving generalization | `outputs/mrl_eye/figures/` |
| runtime probability traces | `p_eye_closed`, `p_yawn` over time | Ground-truth drowsiness labels | `outputs/system_video_upload_runs/` |
| upload warning intervals | Rule-based warning-candidate timeline | Manually labelled fatigue segments | upload summaries/timelines |
| keyframes | Visual evidence from intervals | Proof the model is correct | upload keyframe artifacts |
| evidence figures | Runtime evidence visualization | Accuracy/ROC/PR figures | upload figure artifacts |
| History/Insights charts | Live Monitor product summaries | Model evaluation reports | `SystemUI/src/components/history-48h/`, `SystemUI/src/components/insights/` |
| archive records | Compact runtime summaries | Raw frame/video storage or ground truth | `src/backend/local_archive.py` |
| frontend screenshots | Product UI demonstration | Algorithm performance | `report_assets/all_figures/` |
| deployment screenshots | Remote demo reachability | Production readiness | deployment docs |

## 4. Specialist Metrics vs Full-System Accuracy

Eye model metrics evaluate MRL Eye open/closed classification. They describe the model’s ability to classify eye ROI images as closed/open.

Mouth/yawn model metrics evaluate YawDD/YAWDD+ mouth-crop no-yawn/yawn classification. They describe the model’s ability to classify mouth ROI images as no-yawn/yawn.

Neither is full drowsiness accuracy, because the full system also includes:

- face/landmark detection;
- ROI crop quality;
- runtime webcam domain shift;
- signal quality;
- temporal fusion;
- debounce/cooldown;
- UI alert policy;
- History/Insights product summaries.

Full-system accuracy would require:

1. clear video-level or segment-level ground-truth drowsiness labels;
2. a defined evaluation protocol;
3. inference settings matching the runtime pipeline;
4. definitions for false positives, false negatives, latency, and signal failure;
5. an independent test set.

Source: `docs/tech_learning/MODEL_EVALUATION_AND_SELECTION_GUIDE_ZH.md`, `docs/tech_learning/RUNTIME_INFERENCE_AND_TEMPORAL_FUSION_GUIDE_ZH.md`

## 5. Runtime Evidence and Warning-Candidate Intervals

Correct interpretation:

- `p_eye_closed` is eye-closure evidence;
- `p_yawn` is yawn evidence;
- signal quality affects evidence reliability;
- rule-based temporal fusion produces warning-candidate states;
- a warning interval means runtime rules stayed active over a timeline segment.

Incorrect interpretation:

- `p_eye_closed` is fatigue probability;
- `p_yawn` is fatigue diagnosis;
- a warning interval is a ground-truth drowsiness segment;
- a high-confidence candidate proves the driver is tired.

Source: `src/runtime/realtime_temporal_state.py`, `src/runtime/system_video_upload_pipeline.py`

## 6. Video Upload Evidence

Video Upload can demonstrate how the runtime pipeline behaves on a selected video:

- summary;
- timeline;
- alert intervals;
- evidence figures;
- keyframes;
- HTML/Markdown report artifacts;
- technical details.

Good wording:

> The uploaded-video analysis demonstrates how the runtime evidence pipeline produces warning-candidate intervals and supporting figures for selected videos.

Bad wording:

> The uploaded video proves the model detects drowsiness accurately.

Source: `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx`, `src/runtime/system_video_upload_pipeline.py`

## 7. Live Monitor, History, and Insights Evidence

Live Monitor demonstrates realtime product behavior. History summarizes Live Monitor records. Insights summarizes recent Live Monitor alert patterns.

These are not model evaluation reports:

- History counts are runtime records;
- Insights bullets are product analytics;
- Recent Drives are session/drive summaries;
- signal interruptions are camera/ROI reliability issues;
- charts should not be treated as precision/recall/F1.

Source: `SystemUI/src/components/history-48h/History48hPage.tsx`, `SystemUI/src/components/insights/InsightsPage.tsx`

## 8. Correct Wording Examples

- “The system produces warning-candidate intervals based on sustained eye-closure and yawn evidence.”
- “The eye specialist estimates `p_eye_closed`, which is later used by the temporal fusion layer.”
- “The mouth/yawn specialist estimates `p_yawn` from mouth ROI crops.”
- “The upload analysis demonstrates how the runtime pipeline behaves on selected videos.”
- “History and Insights summarize recent Live Monitor warning-candidate records.”
- “Specialist model metrics evaluate image-level classification tasks, not final system-level drowsiness detection.”

## 9. Incorrect Wording Examples

| Incorrect wording | Problem |
|---|---|
| “The system detects drowsiness with 99% accuracy.” | Inflates specialist metrics into full-system accuracy |
| “The model proves the driver is tired.” | Model output is not medical or safety truth |
| “Every closed-eye frame means drowsiness.” | Normal blinking also closes the eye |
| “History shows model accuracy over 48 hours.” | History is runtime records, not evaluation |
| “Video Upload results are ground-truth labels.” | Upload intervals are rule-based outputs |
| “Vercel deployment means the backend is in the cloud.” | The current backend is reached through a tunnel to a local Mac |

## 10. How to Present This in a Report

Recommended report structure:

| Report section | What to write |
|---|---|
| Methodology | Modular system, specialist models, MediaPipe ROI, temporal fusion |
| Data preprocessing | Manifest, label mapping, subject-level split, crop generation |
| Model evaluation | Eye/mouth specialist metrics, confusion matrix, PR/ROC if available |
| System implementation | FastAPI backend, Next.js frontend, runtime pipeline |
| Runtime demonstration | Upload evidence figures, keyframes, Live Monitor behavior |
| Limitations | No final video-level drowsiness accuracy, domain shift, signal quality |

## 11. How to Present This in a Resume or PhD Portfolio

Truthful concise framing:

- built a modular driver-monitoring prototype;
- trained/evaluated specialist CNNs for eye-state and mouth/yawn evidence;
- used MediaPipe landmarks for ROI extraction;
- integrated rule-based temporal fusion for warning-candidate states;
- built FastAPI backend and Next.js frontend;
- implemented local archive, History, Insights, and upload evidence reports;
- deployed frontend to Vercel with remote testing through Cloudflare tunnel.

Avoid making it sound like a commercial safety-certified product or medical diagnosis tool.

## 12. Red-Line Claims

Do not claim these unless future evidence is added:

- certified driver safety system;
- medical fatigue diagnosis;
- production-ready autonomous safety product;
- final full-system drowsiness accuracy;
- real-world safety validation;
- generalized performance across all drivers/cameras/lighting;
- clinical fatigue measurement;
- cloud-native production backend.

## 13. Beginner Checklist

- Am I writing a specialist metric or a full-system claim?
- Did I explain the evidence nature of `p_eye_closed` / `p_yawn`?
- Did I separate warning-candidate output from ground truth?
- Did I include source paths for figures/tables?
- Did I explain runtime demonstration limitations?
- Did I avoid treating History/Insights as accuracy?

## 14. Common Mistakes

- Using the strongest single metric as system accuracy.
- Showing only strong results and hiding limitations.
- Mixing demo evidence with evaluation evidence.
- Saying “detects fatigue” without explaining warning-candidate monitoring.
- Omitting dataset and runtime boundaries.
- Treating frontend screenshots as model performance evidence.
- Treating upload keyframes as ground-truth labels.
