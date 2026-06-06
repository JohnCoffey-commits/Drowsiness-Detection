# Stage 17.4 Demo Script

This script presents the Stage 17.4 video-upload warning-candidate MVP with the Stage 17.5 eye evidence calibration refinement using safe interpretation wording.

## 1. Start Services

From the repository root:

```bash
cd /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection
make stage17-ui
```

Open:

```text
http://127.0.0.1:3000/video-upload
```

Say:

> This starts the local FastAPI backend and the local Next.js frontend for the Stage 17.4 video-upload warning-candidate MVP.

## 2. Explain Project Boundary

Say:

> This is a video-upload MVP. It analyzes an uploaded video using the existing eye and mouth/yawn model outputs plus rule-based fusion and Stage 17.5 rule-based eye evidence calibration. It is not a webcam page, it is not deployment-ready, and it does not report final system-level drowsiness accuracy.

Point to the permanent warning text:

> This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.

## 3. Upload Test Video

Use:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/upload_test/C_upload_test.mp4
```

Say:

> I will upload the C test video and run the local video-upload warning-candidate analysis.

Click `Analyze Video`.

## 4. Explain Loading Pipeline

While the loading state is visible, say:

> The UI shows the processing pipeline. The backend saves the upload, extracts eye and mouth ROI evidence, runs the eye-warning model, runs the mouth/yawn model, applies Stage 17.1 and Stage 17.5 rule-based fusion refinements, extracts warning-candidate keyframes, and prepares the analysis report.

Mention:

> This is processing an uploaded video. It is not live webcam monitoring.

## 5. Explain Result Sections

After analysis completes, walk through the page top to bottom.

### Summary Cards

Say:

> The summary cards show duration, sampled frames, normal frames, eye-warning candidate frames, mouth-warning candidate frames, high-confidence warning candidate frames, signal-unreliable frames, yawn events, recent-yawn frames/events, and suppressed brief-eye escalation when present.

Say:

> Stage 17.5 adds weak, moderate, and strong eye evidence categories. Weak eye-warning evidence remains visible for manual review, but it does not by itself support high-confidence escalation.

### Warning-candidate Intervals

Say:

> The interval table is the main professional evidence review section. It groups high-confidence warning candidates, eye-warning candidates, mouth-warning candidates, and signal-unreliable intervals with timing, sampled frames, probability maxima, evidence notes, and review priority.

### Figures

Say:

> The fusion timeline shows the rule-based fusion state over time. The `p_eye_closed` figure shows the eye model signal over time, and the `p_yawn` figure shows the mouth/yawn model signal over time.

### Keyframes

Say:

> The keyframe gallery displays saved warning-candidate evidence with metadata: timestamp, frame index, fusion state, `p_eye_closed`, `p_yawn`, recent-yawn status, sustained eye-warning status when available, and reason.

### Technical Evidence

Say:

> The technical evidence section links to backend-generated files such as the report, summary JSON, timeline CSV files, and keyframe metadata. The UI uses backend API paths rather than exposing local absolute paths.

## 6. Explain Stage 17.1 Sustained-eye Gate

Say:

> Stage 17.1 adds a sustained-eye gate for high-confidence escalation. A high-confidence warning candidate requires recent mouth/yawn evidence plus sustained eye-warning evidence.

Say:

> If brief blink-like activity overlaps recent-yawn evidence, the rule-based fusion suppresses high-confidence escalation and keeps the result more conservative.

## 7. Explain Stage 17.5 Eye Evidence Calibration

Say:

> Stage 17.5 does not retrain the eye model or change the `p_eye_closed` formula. It adds provisional rule-based calibration so `p_eye_closed` around 0.55 to 0.61 is shown as weak or reduced-eye-openness evidence, while higher values around 0.85 or above can be shown as strong eye-closure candidate evidence.

Say:

> High-confidence warning candidates now require recent mouth/yawn evidence, sustained eye-warning evidence, and calibrated eye-strength evidence. This reduces overstatement of weak eye-warning intervals while preserving them for manual review.

## 8. Explain Stage 17.2 Interpretation Layer

Say:

> Stage 17.2 adds interpretation guidance. Eye-warning evidence is not automatically described as sustained full eye closure. It may reflect reduced eye openness, blink-like activity, brief closure, fatigue-like appearance, or ROI-sensitive cases.

Say:

> The UI therefore uses warning-candidate wording and supports manual review rather than claiming final output truth.

## 9. Safe Final Conclusion

Say:

> The system produces rule-based drowsiness warning candidates for uploaded videos. It supports a user-friendly summary and professional evidence review, but it does not output final drowsiness truth.

## Terms to Use

- Warning candidate
- High-confidence warning candidate
- Eye-warning candidate
- Mouth-warning candidate
- Signal unreliable
- Reduced eye openness
- Weak eye-warning evidence
- Moderate eye-closure candidate
- Strong eye-closure candidate
- Rule-based fusion

## Terms to Avoid During the Demo

- `driver is drowsy`
- `final detected`
- `final accuracy`
- `deployment-ready`
