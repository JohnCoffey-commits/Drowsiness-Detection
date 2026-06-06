# Stage 18 48h History UI Page Plan

Date: 2026-05-11

## Task Scope

Build `/history-48h` as a frontend-only VisionGuard history page for recent driver-state warning-candidate review.

This task does not add webcam capture, backend database storage, model checkpoint changes, retraining, Python inference changes, or Stage 17 fusion-rule changes.

## Route

```text
http://127.0.0.1:3000/history-48h
```

## Current Frontend-Only Data Source

The page uses local demo history data in browser storage. On first load it seeds a realistic recent 48-hour history dataset. Later page loads read the browser-stored data, filter out records older than 48 hours, and keep user review-state changes local.

## localStorage Key

```text
visionguard.history48h.v1
```

## Page Sections

- Header and boundary notice
- Filters and controls
- 48h summary cards
- Candidate severity trend chart
- Warning-candidate event distribution chart
- State breakdown chart
- High-risk warning candidates
- Event timeline
- Recent sessions
- Manual review queue
- Interpretation note

## State/Event Model

The page defines strongly typed event/session models in `SystemUI/src/lib/history48hTypes.ts`.

Core event states:

- `normal`
- `eye_warning_candidate`
- `mouth_warning_candidate`
- `high_confidence_drowsiness_candidate`
- `signal_unreliable`

Core metadata:

- timestamp and duration
- source: demo/local, video upload, or future webcam placeholder
- max `p_eye_closed`
- max `p_yawn`
- candidate severity score
- eye evidence strength
- safe reason text
- local review status

## Safe Wording Boundary

The permanent warning boundary remains:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

The UI uses warning-candidate wording and avoids confirmed-state, certified-alert, active-camera, and production-readiness claims.

## Future Webcam Integration Note

The page includes a future webcam-ready source type so later frontend/backend work can write compatible history records. This implementation does not request camera permissions, start a camera stream, or create backend persistence.

## Files Changed

- `SystemUI/src/app/history-48h/page.tsx`
- `SystemUI/next.config.ts`
- `SystemUI/src/components/dashboard/Sidebar.tsx`
- `SystemUI/src/components/history-48h/History48hPage.tsx`
- `SystemUI/src/components/history-48h/HistoryHeader.tsx`
- `SystemUI/src/components/history-48h/HistoryFilters.tsx`
- `SystemUI/src/components/history-48h/HistorySummaryCards.tsx`
- `SystemUI/src/components/history-48h/CandidateSeverityTrend.tsx`
- `SystemUI/src/components/history-48h/EventDistributionChart.tsx`
- `SystemUI/src/components/history-48h/StateBreakdownChart.tsx`
- `SystemUI/src/components/history-48h/HighRiskCandidates.tsx`
- `SystemUI/src/components/history-48h/EventTimelineTable.tsx`
- `SystemUI/src/components/history-48h/RecentSessionsSummary.tsx`
- `SystemUI/src/components/history-48h/ManualReviewQueue.tsx`
- `SystemUI/src/components/history-48h/HistoryInterpretationNote.tsx`
- `SystemUI/src/components/history-48h/useChartSize.ts`
- `SystemUI/src/lib/history48hTypes.ts`
- `SystemUI/src/lib/history48hMockData.ts`
- `SystemUI/src/lib/history48hStorage.ts`
- `SystemUI/src/lib/history48hUtils.ts`
- `docs/stages/stage18/STAGE18_HISTORY_48H_UI_PAGE_PLAN.md`

## Validation Commands

From `SystemUI/`:

```bash
npm run lint
npm run build
npm run dev
```

Open:

```text
http://127.0.0.1:3000/history-48h
http://127.0.0.1:3000
http://127.0.0.1:3000/video-upload
```

## Validation Results

- `npm run lint`: passed.
- `npm run build`: passed.
- `npm run dev`: started on `http://localhost:3000` and validated through `http://127.0.0.1:3000/history-48h`.
- `/history-48h`: rendered with seeded demo/local data, summary cards, charts, event timeline, recent sessions, manual review queue, and interpretation note.
- Sidebar: only Dashboard, Video Upload Analysis, 48h History, and Insights are shown; 48h History has active page state.
- Filters: time window, event type, review, and source filters were exercised.
- Local controls: copy summary, clear history, refresh persistence, reset demo data, details expansion, session filtering, and review-state update were exercised.
- `/`: rendered with page title `VisionGuard Dashboard`.
- `/video-upload`: rendered with page title `Video Upload Analysis | VisionGuard`.

## Known Limitations

- History data is demo/local browser data only.
- Clearing history keeps an empty local store until the user resets demo data.
- Manual review actions are localStorage updates only.
- Future webcam and video-upload history ingestion are modelled as compatible sources, not implemented capture or backend storage.
