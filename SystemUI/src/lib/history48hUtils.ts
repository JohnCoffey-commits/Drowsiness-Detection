import type {
  DriverHistoryEvent,
  DriverHistorySession,
  EventTypeFilter,
  HistoryFilters,
  HistorySeverity,
  HistorySource,
  HistoryState,
  ReviewStatus,
  SourceFilter,
  StateBreakdownItem,
  TimeBucketSummary,
  TimeWindowHours,
  TrendPoint,
} from "@/lib/history48hTypes";

export const HISTORY_48H_BOUNDARY_NOTICE =
  "This page shows frontend-local warning-candidate history for review. It is not final system-level drowsiness accuracy.";

export const TIME_WINDOW_OPTIONS: Array<{
  label: string;
  value: TimeWindowHours;
}> = [
  { label: "Last 6 hours", value: 6 },
  { label: "Last 12 hours", value: 12 },
  { label: "Last 24 hours", value: 24 },
  { label: "Last 48 hours", value: 48 },
  { label: "Last 7 days", value: 168 },
  { label: "Last 30 days", value: 720 },
  { label: "All archive", value: 876000 },
];

export const EVENT_TYPE_OPTIONS: Array<{
  label: string;
  value: EventTypeFilter;
}> = [
  { label: "All", value: "all" },
  { label: "Eye warning candidate", value: "eye_warning_candidate" },
  { label: "Yawn warning candidate", value: "mouth_warning_candidate" },
  {
    label: "Critical eye warning candidate",
    value: "high_confidence_drowsiness_candidate",
  },
  { label: "Signal quality issue", value: "signal_unreliable" },
];

export const REVIEW_OPTIONS: Array<{
  label: string;
  value: ReviewStatus | "all";
}> = [
  { label: "All", value: "all" },
  { label: "Pending review", value: "pending" },
  { label: "Reviewed", value: "reviewed" },
  { label: "Not required", value: "not_required" },
];

export const SOURCE_OPTIONS: Array<{
  label: string;
  value: SourceFilter;
}> = [
  { label: "All", value: "all" },
  { label: "Live Monitor", value: "live_monitor" },
  { label: "Video Upload", value: "video_upload" },
  { label: "Demo", value: "demo" },
];

export const STATE_META: Record<
  HistoryState,
  {
    label: string;
    shortLabel: string;
    color: string;
    bgClass: string;
    textClass: string;
    dotClass: string;
  }
> = {
  normal: {
    label: "Normal",
    shortLabel: "Normal",
    color: "#10b981",
    bgClass: "bg-emerald-50 border-emerald-100",
    textClass: "text-emerald-700",
    dotClass: "bg-emerald-500",
  },
  eye_warning_candidate: {
    label: "Eye warning candidate",
    shortLabel: "Eye warning",
    color: "#f97316",
    bgClass: "bg-orange-50 border-orange-100",
    textClass: "text-orange-700",
    dotClass: "bg-orange-500",
  },
  mouth_warning_candidate: {
    label: "Yawn warning candidate",
    shortLabel: "Yawn warning",
    color: "#ec4899",
    bgClass: "bg-pink-50 border-pink-100",
    textClass: "text-pink-700",
    dotClass: "bg-pink-500",
  },
  high_confidence_drowsiness_candidate: {
    label: "Critical eye warning candidate",
    shortLabel: "Critical eye",
    color: "#ef4444",
    bgClass: "bg-red-50 border-red-100",
    textClass: "text-red-700",
    dotClass: "bg-red-500",
  },
  signal_unreliable: {
    label: "Signal quality issue",
    shortLabel: "Signal quality",
    color: "#64748b",
    bgClass: "bg-slate-100 border-slate-200",
    textClass: "text-slate-700",
    dotClass: "bg-slate-500",
  },
};

export const SEVERITY_META: Record<
  HistorySeverity,
  { label: string; className: string; rank: number }
> = {
  low: {
    label: "Low",
    className: "border-emerald-100 bg-emerald-50 text-emerald-700",
    rank: 1,
  },
  medium: {
    label: "Medium",
    className: "border-amber-100 bg-amber-50 text-amber-700",
    rank: 2,
  },
  high: {
    label: "High",
    className: "border-red-100 bg-red-50 text-red-700",
    rank: 3,
  },
  unreliable: {
    label: "Unreliable",
    className: "border-slate-200 bg-slate-100 text-slate-700",
    rank: 2,
  },
};

export const SOURCE_LABELS: Record<HistorySource, string> = {
  mock: "Demo",
  live_monitor: "Live Monitor",
  video_upload: "Video Upload",
  manual: "Manual",
  webcam_future: "Demo",
};

export const REVIEW_LABELS: Record<ReviewStatus, string> = {
  pending: "Pending review",
  reviewed: "Reviewed",
  not_required: "Not required",
};

export interface HistorySummary {
  sessionCount: number;
  monitoredTimeMin: number;
  totalEvents: number;
  normalCount: number;
  eyeWarningCount: number;
  mouthWarningCount: number;
  highConfidenceCount: number;
  signalUnreliableCount: number;
  reviewPendingCount: number;
  warningCandidateCount: number;
  highPriorityCount: number;
  normalRatio: number;
  peakCandidateSeverity: number;
  lastEventTime: string | null;
}

export interface SessionSummaryRow extends DriverHistorySession {
  warningCandidateCount: number;
  reviewPendingCount: number;
}

function dateValue(value: string): number {
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : 0;
}

function getWindowStart(now: Date, hours: TimeWindowHours): Date {
  return new Date(now.getTime() - hours * 60 * 60_000);
}

function isEventInWindow(
  event: DriverHistoryEvent,
  now: Date,
  hours: TimeWindowHours
): boolean {
  const timestamp = dateValue(event.timestamp);
  return timestamp >= getWindowStart(now, hours).getTime() && timestamp <= now.getTime();
}

export function filterHistoryEvents(
  events: DriverHistoryEvent[],
  filters: HistoryFilters,
  now = new Date()
): DriverHistoryEvent[] {
  return events
    .filter((event) => isEventInWindow(event, now, filters.timeWindowHours))
    .filter((event) => event.state !== "normal")
    .filter((event) =>
      filters.eventType === "all" ? true : event.state === filters.eventType
    )
    .filter((event) =>
      filters.review === "all" ? true : event.reviewStatus === filters.review
    )
    .filter((event) =>
      filters.source === "all"
        ? true
        : filters.source === "demo"
          ? event.source === "mock" || event.source === "webcam_future"
          : event.source === filters.source
    )
    .filter((event) =>
      filters.sessionId ? event.sessionId === filters.sessionId : true
    )
    .sort((a, b) => dateValue(b.timestamp) - dateValue(a.timestamp));
}

export function summarizeHistory(
  events: DriverHistoryEvent[],
  sessions: DriverHistorySession[]
): HistorySummary {
  const sessionIds = new Set(events.map((event) => event.sessionId));
  const monitoredTimeMin = sessions
    .filter((session) => sessionIds.has(session.id))
    .reduce((sum, session) => sum + session.durationMin, 0);
  const totalEvents = events.length;
  const normalCount = countByState(events, "normal");
  const warningCandidateCount = events.filter(
    (event) => event.state !== "normal"
  ).length;
  const highPriorityCount = events.filter(
    (event) =>
      event.state === "high_confidence_drowsiness_candidate" ||
      event.severity === "high"
  ).length;
  const peakCandidateSeverity = events.reduce(
    (max, event) => Math.max(max, event.candidateSeverityScore ?? 0),
    0
  );

  return {
    sessionCount: sessionIds.size,
    monitoredTimeMin,
    totalEvents,
    normalCount,
    eyeWarningCount: countByState(events, "eye_warning_candidate"),
    mouthWarningCount: countByState(events, "mouth_warning_candidate"),
    highConfidenceCount: countByState(
      events,
      "high_confidence_drowsiness_candidate"
    ),
    signalUnreliableCount: countByState(events, "signal_unreliable"),
    reviewPendingCount: events.filter((event) => event.reviewStatus === "pending")
      .length,
    warningCandidateCount,
    highPriorityCount,
    normalRatio: totalEvents === 0 ? 0 : normalCount / totalEvents,
    peakCandidateSeverity,
    lastEventTime: events[0]?.timestamp ?? null,
  };
}

export function countByState(
  events: DriverHistoryEvent[],
  state: HistoryState
): number {
  return events.filter((event) => event.state === state).length;
}

export function getBucketMinutes(hours: TimeWindowHours): number {
  if (hours <= 6) return 30;
  if (hours <= 12) return 60;
  return 120;
}

function bucketLabel(date: Date, hours: TimeWindowHours): string {
  if (hours === 48) {
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function buildBuckets(now: Date, hours: TimeWindowHours): Date[] {
  const bucketMinutes = getBucketMinutes(hours);
  const bucketCount = Math.ceil((hours * 60) / bucketMinutes);
  const start = getWindowStart(now, hours);

  return Array.from({ length: bucketCount }, (_, index) =>
    new Date(start.getTime() + index * bucketMinutes * 60_000)
  );
}

export function aggregateSeverityTrend(
  events: DriverHistoryEvent[],
  now: Date,
  hours: TimeWindowHours
): TrendPoint[] {
  const bucketMinutes = getBucketMinutes(hours);
  const bucketMs = bucketMinutes * 60_000;

  return buildBuckets(now, hours).map((bucketStart) => {
    const bucketEnd = new Date(bucketStart.getTime() + bucketMs);
    const bucketEvents = events.filter((event) => {
      const timestamp = dateValue(event.timestamp);
      return timestamp >= bucketStart.getTime() && timestamp < bucketEnd.getTime();
    });
    const scoredEvents = bucketEvents.filter(
      (event) => event.candidateSeverityScore != null
    );
    const score =
      scoredEvents.length === 0
        ? null
        : Math.max(
            ...scoredEvents.map((event) => event.candidateSeverityScore ?? 0)
          );

    return {
      timestamp: bucketStart.toISOString(),
      label: bucketLabel(bucketStart, hours),
      score,
      unreliableCount: bucketEvents.filter(
        (event) => event.state === "signal_unreliable"
      ).length,
    };
  });
}

export function aggregateEventDistribution(
  events: DriverHistoryEvent[],
  now: Date,
  hours: TimeWindowHours
): TimeBucketSummary[] {
  const bucketMinutes = getBucketMinutes(hours);
  const bucketMs = bucketMinutes * 60_000;

  return buildBuckets(now, hours).map((bucketStart) => {
    const bucketEnd = new Date(bucketStart.getTime() + bucketMs);
    const bucketEvents = events.filter((event) => {
      const timestamp = dateValue(event.timestamp);
      return timestamp >= bucketStart.getTime() && timestamp < bucketEnd.getTime();
    });

    return {
      timestamp: bucketStart.toISOString(),
      label: bucketLabel(bucketStart, hours),
      eyeWarning: countByState(bucketEvents, "eye_warning_candidate"),
      mouthWarning: countByState(bucketEvents, "mouth_warning_candidate"),
      highConfidence: countByState(
        bucketEvents,
        "high_confidence_drowsiness_candidate"
      ),
      signalUnreliable: countByState(bucketEvents, "signal_unreliable"),
    };
  });
}

export function buildStateBreakdown(
  events: DriverHistoryEvent[]
): StateBreakdownItem[] {
  const total = Math.max(events.length, 1);
  const states: HistoryState[] = [
    "normal",
    "eye_warning_candidate",
    "mouth_warning_candidate",
    "high_confidence_drowsiness_candidate",
    "signal_unreliable",
  ];

  return states.map((state) => {
    const count = countByState(events, state);
    return {
      state,
      label: STATE_META[state].label,
      count,
      percentage: count / total,
      color: STATE_META[state].color,
    };
  });
}

export function getHighRiskCandidates(
  events: DriverHistoryEvent[],
  limit = 6
): DriverHistoryEvent[] {
  const isMediumOrHighSeverity = (severity: HistorySeverity) =>
    severity === "medium" || severity === "high";

  return events
    .filter(
      (event) =>
        event.state === "high_confidence_drowsiness_candidate" ||
        event.severity === "high" ||
        (event.reviewStatus === "pending" &&
          isMediumOrHighSeverity(event.severity))
    )
    .sort((a, b) => {
      const severityDelta =
        SEVERITY_META[b.severity].rank - SEVERITY_META[a.severity].rank;
      if (severityDelta !== 0) return severityDelta;
      return (
        (b.candidateSeverityScore ?? 0) - (a.candidateSeverityScore ?? 0) ||
        dateValue(b.timestamp) - dateValue(a.timestamp)
      );
    })
    .slice(0, limit);
}

export function getManualReviewQueue(
  events: DriverHistoryEvent[],
  limit = 8
): DriverHistoryEvent[] {
  const sessionEvents = new Map<string, DriverHistoryEvent[]>();
  for (const event of events) {
    const current = sessionEvents.get(event.sessionId) ?? [];
    current.push(event);
    sessionEvents.set(event.sessionId, current);
  }

  function priority(event: DriverHistoryEvent): number {
    const sameSessionEvents = sessionEvents.get(event.sessionId) ?? [];
    const yawnCount = countByState(sameSessionEvents, "mouth_warning_candidate");
    const signalCount = countByState(sameSessionEvents, "signal_unreliable");

    if (
      event.state === "high_confidence_drowsiness_candidate" ||
      event.severity === "high"
    ) {
      return 500;
    }

    if (
      event.state === "eye_warning_candidate" &&
      (event.eyeEvidenceStrength === "strong" ||
        event.eyeEvidenceStrength === "moderate")
    ) {
      return 400;
    }

    if (event.state === "mouth_warning_candidate" && yawnCount > 1) {
      return 300;
    }

    if (event.state === "signal_unreliable" && signalCount > 1) {
      return 250;
    }

    if (event.state === "signal_unreliable") {
      return 200;
    }

    return 100;
  }

  return events
    .filter(
      (event) =>
        event.reviewStatus === "pending" ||
        event.state === "high_confidence_drowsiness_candidate" ||
        event.state === "signal_unreliable" ||
        event.eyeEvidenceStrength === "weak" ||
        event.eyeEvidenceStrength === "moderate"
    )
    .sort((a, b) => {
      const priorityDelta = priority(b) - priority(a);
      if (priorityDelta !== 0) return priorityDelta;
      const pendingDelta =
        Number(b.reviewStatus === "pending") - Number(a.reviewStatus === "pending");
      if (pendingDelta !== 0) return pendingDelta;
      const severityDelta =
        SEVERITY_META[b.severity].rank - SEVERITY_META[a.severity].rank;
      if (severityDelta !== 0) return severityDelta;
      return dateValue(b.timestamp) - dateValue(a.timestamp);
    })
    .slice(0, limit);
}

export function buildSessionRows(
  sessions: DriverHistorySession[],
  events: DriverHistoryEvent[]
): SessionSummaryRow[] {
  const eventSessionIds = new Set(events.map((event) => event.sessionId));

  return sessions
    .filter((session) => eventSessionIds.has(session.id))
    .map((session) => {
      const sessionEvents = events.filter((event) => event.sessionId === session.id);
      return {
        ...session,
        normalCount: countByState(sessionEvents, "normal"),
        eyeWarningCount: countByState(sessionEvents, "eye_warning_candidate"),
        mouthWarningCount: countByState(sessionEvents, "mouth_warning_candidate"),
        highConfidenceCount: countByState(
          sessionEvents,
          "high_confidence_drowsiness_candidate"
        ),
        signalUnreliableCount: countByState(sessionEvents, "signal_unreliable"),
        reviewPendingCount: sessionEvents.filter(
          (event) => event.reviewStatus === "pending"
        ).length,
        warningCandidateCount: sessionEvents.filter(
          (event) => event.state !== "normal"
        ).length,
      };
    })
    .sort((a, b) => dateValue(b.startedAt) - dateValue(a.startedAt));
}

export function updateHistoryEventReviewStatus(
  events: DriverHistoryEvent[],
  eventId: string,
  reviewStatus: ReviewStatus
): DriverHistoryEvent[] {
  return events.map((event) =>
    event.id === eventId ? { ...event, reviewStatus } : event
  );
}

export function formatDateTime(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatTimeRange(start: string, end: string): string {
  const endDate = new Date(end);
  return `${formatDateTime(start)} - ${endDate.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
}

export function formatMinutes(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const hours = Math.floor(minutes / 60);
  const remainder = Math.round(minutes % 60);
  return remainder > 0 ? `${hours}h ${remainder}m` : `${hours}h`;
}

export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatProbability(value: number | undefined): string {
  return value == null ? "-" : value.toFixed(3);
}

export function formatCandidateScore(value: number | undefined): string {
  return value == null ? "-" : String(Math.round(value));
}

export function evidenceLabel(event: DriverHistoryEvent): string {
  if (event.state === "signal_unreliable") return "Signal quality";
  if (event.state === "high_confidence_drowsiness_candidate") {
    return "High-priority evidence";
  }
  if (event.state === "eye_warning_candidate") return "Eye";
  if (event.state === "mouth_warning_candidate") return "Yawn";
  return "Baseline";
}

export function buildHistorySummaryText(
  summary: HistorySummary,
  filters: HistoryFilters
): string {
  const windowLabel =
    TIME_WINDOW_OPTIONS.find((option) => option.value === filters.timeWindowHours)
      ?.label ?? `Last ${filters.timeWindowHours} hours`;

  return [
    "VisionGuard 48h warning-candidate history summary",
    `Selected window: ${windowLabel}`,
    `Sessions with local records: ${summary.sessionCount}`,
    `Warning-candidate events: ${summary.warningCandidateCount}`,
    `High-priority candidates: ${summary.highPriorityCount}`,
    `Eye warning candidates: ${summary.eyeWarningCount}`,
    `Yawn warning candidates: ${summary.mouthWarningCount}`,
    `Critical eye warning candidates: ${summary.highConfidenceCount}`,
    `Signal quality issues: ${summary.signalUnreliableCount}`,
    `Manual review pending: ${summary.reviewPendingCount}`,
    HISTORY_48H_BOUNDARY_NOTICE,
  ].join("\n");
}
