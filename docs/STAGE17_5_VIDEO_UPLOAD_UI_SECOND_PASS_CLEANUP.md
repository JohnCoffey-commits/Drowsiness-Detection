# Stage 17.5 Video Upload UI Second-Pass Cleanup

Date: 2026-05-11

## Task Scope

This pass cleaned up the `/video-upload` evidence review UI for the local Stage 17.x uploaded-video rule-based warning-candidate MVP.

No model checkpoint, retraining path, Python inference formula, backend fusion rule, `p_eye_closed` formula, or `p_yawn` formula was changed.

## Problems Found

- The warning-candidate interval table placed long prose directly in table cells, which made rows tall and hard to scan.
- Stage 17.5 summary metric scope was under-explained. Runtime/schema review showed weak eye evidence is counted within eye-warning candidate rows, while moderate and strong eye-evidence counts are across sampled timeline rows.
- Interval rows mixed backend fusion state with descriptive eye evidence without enough explanation.
- Keyframe cards used `Eye strength gate`, which made interval-level gate output look like a single-frame result.
- Missing optional interval fields created repeated `Not provided` text.
- Result Overview repeated details already covered by Summary Metrics.
- Evidence figures rendered as several large stacked cards.
- A mascot/cartoon overlay was not found in SystemUI code or public assets.

## Files Changed

- `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx`
- `SystemUI/src/components/video-upload/AnalysisSummaryCards.tsx`
- `SystemUI/src/components/video-upload/IntervalReviewTable.tsx`
- `SystemUI/src/components/video-upload/EvidenceFigures.tsx`
- `SystemUI/src/components/video-upload/KeyframeEvidenceGallery.tsx`
- `SystemUI/src/lib/videoUploadUtils.ts`
- `docs/STAGE17_5_VIDEO_UPLOAD_UI_SECOND_PASS_CLEANUP.md`

## Interval Table Cleanup

The interval table now keeps the main row compact:

- State
- Start
- End
- Duration
- Frames
- Max `p_eye_closed`
- Max `p_yawn`
- Peak eye evidence
- Sustained
- Review
- Details

Long explanations moved into a per-interval expandable details panel. The details panel includes fusion state explanation, fusion state reason, peak eye evidence explanation, sustained-eye gate information, interval eye-strength gate information, suppression notes, and a reminder that the UI does not recompute backend fusion state.

## Stage 17.5 Metric Scope Wording

The Stage 17.5 metric section now explains that calibration fields describe eye-evidence strength and may not equal final fusion-state counts.

Current runtime/schema scope:

- Weak eye evidence frames: within eye-warning candidate rows.
- Moderate eye evidence frames: across sampled timeline rows.
- Strong eye-closure candidate frames: across sampled timeline rows.
- Suppressed weak-eye escalation frames: backend-provided suppressed high-confidence candidate count.

## Fusion State vs Descriptive Evidence

The interval table now labels the evidence column as `Peak eye evidence`.

The page explains that this is descriptive eye-probability evidence within the interval and does not recompute or override the backend fusion state. A mouth-warning interval can therefore show moderate peak eye evidence without the UI changing it to high-confidence.

## Interval-Level Eye-Strength Gate Wording

Keyframe cards now use `Interval eye-strength gate` instead of `Eye strength gate`.

When gate data is shown, the card explains that the gate is evaluated over the eye-warning interval and that the current keyframe may still be weak evidence.

## Missing Optional Fields

Compact interval cells now render missing optional booleans and text as `—`.

Keyframe cards still hide missing optional rows where that keeps the card cleaner. Missing optional booleans are not displayed as false.

## Evidence Figures

Evidence figures now render as tabs. The Fusion timeline is visible by default, while `p_eye_closed` and `p_yawn` figures remain accessible through tab buttons and open links.

If a figure image fails to load, the UI shows a safe fallback message instead of crashing.

## Mascot / Overlay Search Result

Searched SystemUI source and public assets for:

- mascot
- avatar
- chibi
- assistant
- character
- sticker
- gif
- fixed
- bottom
- right
- floating
- overlay
- widget

No mascot/cartoon overlay was found in SystemUI application code. Public image assets are limited to `eye.png`, `yawn.png`, and default SVG assets. The observed overlay is likely from the browser, plugin, or development environment rather than this frontend code.

## Safe Wording Boundary

The permanent warning remains visible:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

The UI keeps warning-candidate wording and does not claim a final user state, production readiness, certification status, or system-level accuracy.

## Validation Commands

From `SystemUI/`:

```bash
npm run lint
npm run build
npm run dev
```

Open:

```text
http://127.0.0.1:3000/video-upload
```

Optional backend-connected validation from repository root:

```bash
make stage17-ui
```

Then upload:

```text
upload_test/C_upload_test.mp4
upload_test/B_upload_test.mp4
```

## Validation Results

- `npm run lint`: passed.
- `npm run build`: passed.
- `npm run dev`: an existing Next.js dev server was already running on `http://localhost:3000`; browser validation used that server.
- `/video-upload`: rendered without errors after refresh.
- Backend-connected upload: `upload_test/C_upload_test.mp4` completed through the UI and returned result session `upload_7d1c96312d92`.
- Interval details: expandable details opened and showed fusion state, peak eye evidence, interval eye-strength gate wording, and the backend-fusion-state reminder.
- Evidence figure tabs: Fusion timeline was the default visible tab; `p_eye_closed` and `p_yawn` tabs were both selectable.
- Dashboard `/`: rendered with page title `VisionGuard Dashboard`.
- Mascot/overlay: no SystemUI mascot/cartoon code was found; the only observed floating control was the Next.js development tools button.
- Optional `upload_test/B_upload_test.mp4`: not run in this pass.

## Known Limitations

- Stage 17.5 field scope follows the current backend summary schema.
- Backend progress is not streamed, so processing steps remain an approximate UI guide.
- Browser/plugin overlays are outside SystemUI code and were not removed.
- This is frontend evidence review cleanup only, not backend calibration or model improvement.
