# Stage 17.5 Video Upload UI Evidence Review Page

Date: 2026-05-11

## Task Scope

This frontend task built `/video-upload` as a professional uploaded-video evidence review workstation for the Stage 17 local warning-candidate MVP.

The work was limited to the Next.js SystemUI frontend and launcher route target. No model checkpoints, training code, Python inference formulas, or backend fusion logic were changed.

## What Changed

- Added a dedicated App Router route at `SystemUI/src/app/video-upload/page.tsx`.
- Added a client-side evidence review workstation component.
- Added typed backend response models for Stage 17.4 fields and optional Stage 17.5 eye-evidence fields.
- Added safe URL and display helpers for backend API links, probabilities, durations, intervals, keyframes, and copied summaries.
- Added uploaded-video preview using a native non-autoplay `<video controls>` element.
- Added backend URL validation and a compact backend status check.
- Added result overview, grouped summary metrics, warning-candidate interval review, evidence figures, keyframe gallery, interpretation notes, and technical evidence links.
- Restored the local launcher frontend URL to `http://127.0.0.1:3000/video-upload`.

## What Did Not Change

- No Python backend code was edited.
- No model checkpoints were changed.
- No model retraining was run.
- No eye or mouth probability formula was changed.
- No fusion-state computation was moved into the UI.
- Dashboard `/` was not rewritten.
- Webcam or live-monitoring behavior was not implemented.

## Page Sections

1. Page Header
2. Upload & Backend Control
3. Uploaded Video Preview
4. Processing Status
5. Result Overview
6. Summary Metrics
7. Warning-Candidate Intervals
8. Evidence Figures
9. Keyframe Evidence Gallery
10. Interpretation / Manual Review Notes
11. Technical Evidence Panel

## Backend API Used

Default backend URL:

```text
http://127.0.0.1:8000
```

Upload endpoint:

```text
POST {backendUrl}/api/analyze-video
```

Safe evidence links are constructed from:

- `GET /api/runs/{session_id}/summary`
- `GET /api/runs/{session_id}/timeline`
- `GET /api/runs/{session_id}/keyframes`
- `GET /api/runs/{session_id}/files/{relative_path}`

The UI does not display `audit_log`, `keyframe_path`, `video_path`, or other local absolute filesystem paths in the browser.

## Safe Wording Boundary

The permanent warning appears on the page:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

The UI uses warning-candidate wording such as:

- uploaded-video analysis
- warning-candidate review
- eye-warning candidate
- mouth-warning candidate
- high-confidence warning candidate
- signal unreliable
- possible eye-closure candidate
- reduced eye openness
- manual review recommended
- technical evidence

The UI avoids final driver-state claims and does not present backend outputs as final system-level accuracy.

## Stage 17.5 Optional Field Support

The TypeScript types and display helpers support optional Stage 17.5 fields for summaries, intervals, and keyframes.

When present, the UI renders weak, moderate, strong, uncertain, and normal-open eye-evidence labels. When absent, it shows safe fallback text:

```text
Eye evidence strength fields are not present in this backend response.
```

The UI does not synthesize missing Stage 17.5 fields.

## Local Commands

From `SystemUI/`:

```bash
npm run lint
npm run build
npm run dev
```

From repository root:

```bash
make stage17-ui
```

Open:

```text
http://127.0.0.1:3000/video-upload
```

## Validation Result

Frontend checks:

- `npm run lint`: passed
- `npx tsc --noEmit --incremental false`: passed
- `npm run build`: passed
- Next.js build route output included `/video-upload`

Browser checks with Playwright CLI:

- `/video-upload` rendered at `http://127.0.0.1:3000/video-upload`
- selected `upload_test/C_upload_test.mp4`
- native video preview loaded and read browser duration
- `POST http://127.0.0.1:8000/api/analyze-video` returned `200 OK`
- result overview rendered
- summary metrics rendered
- warning-candidate intervals rendered
- evidence figures rendered
- keyframe gallery rendered
- technical evidence links used backend-safe API URLs
- page text did not contain local absolute path markers
- page text did not contain forbidden final driver-state wording

Backend-connected C upload session:

```text
session_id: upload_513046570cf4
high_confidence_drowsiness_candidate_frames: 0
suppressed_high_confidence_brief_eye_warning_frames: 8
keyframes: 9
figures: 3
weak_eye_warning_evidence_frames: 29
moderate_eye_closure_candidate_frames: 8
strong_eye_closure_candidate_frames: 0
suppressed_high_confidence_weak_eye_evidence_frames: 9
```

Note: the current backend response for this run differed from the older Stage 17.4 expectation of 9 high-confidence warning candidate frames and 4 keyframes. The UI displayed the backend response as returned and did not alter fusion results.

Screenshot captured:

```text
.playwright-cli/page-2026-05-11T06-25-45-291Z.png
```

## Known Limitations

- Backend progress is not streamed, so processing steps are an approximate UI guide only.
- Stage 17.5 values are displayed only if the backend response includes them.
- Browser upload validation checks extension and size; backend remains the authority for accepted upload content.
- The technical evidence panel links to expected backend evidence paths and handles missing assets through normal browser/backend behavior.
