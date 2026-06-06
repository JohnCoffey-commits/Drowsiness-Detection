# Stage 17.5 Video Upload UI Fallback Polish

Date: 2026-05-11

## Problem Found

The `/video-upload` keyframe cards repeated long missing-field fallback copy for Stage 17.5 eye evidence and manual-review metadata. This made normal Stage 17.4-compatible backend output look unfinished and distracted from useful fields such as timestamp, frame index, `p_eye_closed`, `p_yawn`, `recent_yawn_event`, warning type, segment, and fusion state.

The UI also needed clearer wording for `recent_yawn_event` because that flag can remain active after the visible yawn event. A low current-frame `p_yawn` can still appear with `recent_yawn_event = true` when the yawn occurred inside the recent temporal window.

## Files Changed

- `SystemUI/src/components/video-upload/KeyframeEvidenceGallery.tsx`
- `SystemUI/src/components/video-upload/IntervalReviewTable.tsx`
- `SystemUI/src/lib/videoUploadTypes.ts`
- `SystemUI/src/lib/videoUploadUtils.ts`
- `docs/stages/stage17/STAGE17_5_VIDEO_UPLOAD_UI_FALLBACK_POLISH.md`

## What Was Fixed

- Added `formatOptionalBoolean()` so missing optional booleans are not collapsed into false.
- Added keyframe eye-evidence helpers for detecting whether Stage 17.5 eye evidence strength fields are actually present.
- Added `recentYawnExplanation()` for temporal-window yawn interpretation.
- Added support for backend-provided `eye_evidence_label` and `eye_evidence_interpretation`.
- Replaced the vague keyframe badge with `Primary evidence keyframe` and `Supporting keyframe`.
- Grouped keyframe metadata into Frame, Model evidence, Temporal evidence, and Review sections.
- Moved the Stage 17.5 missing-field explanation to one gallery-level notice when no keyframes include those fields.
- Kept per-keyframe missing eye-evidence copy short only for mixed responses where some keyframes have Stage 17.5 fields and some do not.

## What Was Not Changed

- No model checkpoints were changed.
- No model retraining was run.
- No Python model inference formulas were changed.
- No backend fusion logic was changed.
- No `p_eye_closed` or `p_yawn` formula was changed.
- No webcam workflow was added.
- Dashboard `/` was not rewritten.
- Raw local filesystem paths are still not rendered in the UI.

## Why Repeated Missing-Field Text Was Removed

Stage 17.4-compatible runs may legitimately omit Stage 17.5 eye evidence strength fields. Repeating a long fallback inside every keyframe card made the result look broken even when the backend run was valid. The UI now explains that case once at the Keyframe Evidence Gallery level and keeps individual keyframes focused on available uploaded-video warning-candidate metadata.

## Optional Boolean Handling

Optional booleans are now formatted with explicit missing-value handling:

- `true` renders as `Yes` unless a component supplies a more specific label.
- `false` renders as `No` when the field is actually present and rendered.
- `undefined` and `null` render as `Not provided` or the row is hidden for visual clarity.

This prevents missing metadata such as `sustained_eye_warning`, `manual_review_recommended`, candidate flags, eye-strength gate flags, and suppression flags from being silently displayed as false.

## Recent-Yawn Temporal Window

`recent_yawn_event` means a yawn event occurred within the recent temporal window. It does not necessarily mean the exact keyframe has high current-frame `p_yawn`.

When a keyframe has `recent_yawn_event = true` and low `p_yawn`, the UI shows:

```text
Recent yawn: Yes - within recent temporal window
```

and explains that the current frame may have a low `p_yawn` because the evidence can persist after the visible yawn event.

## Safe Wording Boundary

The permanent warning remains:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

The UI continues to use safe terms such as:

- rule-based warning-candidate analysis
- uploaded-video analysis
- warning-candidate review
- eye-warning candidate
- mouth-warning candidate
- high-confidence warning candidate
- signal unreliable
- possible eye-closure candidate
- reduced eye openness
- manual review recommended
- evidence not provided by this backend run

The UI does not make final driver-state claims or present the page as a production deployment or live monitoring workflow.

## Validation Commands

From the frontend directory:

```bash
cd /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/SystemUI
npm run lint
npm run build
npm run dev
```

Open:

```text
http://127.0.0.1:3000/video-upload
```

Optional backend-connected validation from the repository root:

```bash
cd /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection
make stage17-ui
```

Then upload:

```text
upload_test/C_upload_test.mp4
```

Optional second upload:

```text
upload_test/B_upload_test.mp4
```

## Validation Result

- `npm run lint`: passed.
- `npm run build`: passed. The production build included `/` and `/video-upload`.
- `npm run dev`: a Next dev server for this workspace was already running on port `3000`, so the new dev command did not start a duplicate server. Browser validation used the active server at `http://127.0.0.1:3000`.
- Browser check for `http://127.0.0.1:3000/video-upload`: rendered without page errors; permanent warning was visible; old repeated keyframe fallback text was absent; `Secondary` was absent; raw `/Users/` paths were absent from the page.
- Browser check for `http://127.0.0.1:3000/`: rendered without page errors.
- Backend upload endpoint check with `upload_test/C_upload_test.mp4`: returned HTTP `200` with completed session `upload_f6a77d544214` and `9` keyframes.
- `make stage17-ui`: not run in this pass because the frontend port was already occupied by the active workspace dev server and the backend was already reachable.

## Known Limitations

- The UI only displays Stage 17.5 eye evidence fields when the backend response includes them.
- If an older backend serialized a missing optional boolean as explicit `false`, the frontend cannot recover the original missing-vs-false distinction from that value alone.
- Backend processing progress is not streamed; processing steps remain an approximate UI guide.
- This task is frontend fallback and interpretation polish only, not Stage 17.5 backend calibration.
- Browser automation in this environment did not expose a safe file-picker file-selection API for the React page. The backend upload endpoint was validated directly with the requested test video.
