import type {
  DriverHistoryEvent,
  DriverHistorySession,
  HistorySource,
  TimeWindowHours,
} from "@/lib/history48hTypes";
import { SOURCE_LABELS, getBucketMinutes } from "@/lib/history48hUtils";
import type {
  InsightCompositionItem,
  InsightEventKind,
  InsightKindMeta,
  InsightRecommendation,
  InsightSessionComparisonRow,
  InsightSignalQualitySummary,
  InsightSummary,
  InsightTimeOfDayItem,
  InsightTrendPoint,
} from "@/lib/insightsTypes";

export const INSIGHTS_BOUNDARY_NOTICE =
  "This page summarizes Live Monitor warning-candidate history for review. It is not final system-level drowsiness accuracy.";

export const INSIGHT_KIND_META: Record<InsightEventKind, InsightKindMeta> = {
  eye_warning_candidate: {
    kind: "eye_warning_candidate",
    label: "Eye warning candidate",
    shortLabel: "Eye warning",
    color: "#f97316",
  },
  yawn_warning_candidate: {
    kind: "yawn_warning_candidate",
    label: "Yawn warning candidate",
    shortLabel: "Yawn warning",
    color: "#ec4899",
  },
  critical_eye_warning_candidate: {
    kind: "critical_eye_warning_candidate",
    label: "Critical eye warning candidate",
    shortLabel: "Critical eye",
    color: "#ef4444",
  },
  signal_quality_issue: {
    kind: "signal_quality_issue",
    label: "Signal quality issue",
    shortLabel: "Signal quality",
    color: "#64748b",
  },
};

const INSIGHT_KIND_ORDER: InsightEventKind[] = [
  "eye_warning_candidate",
  "yawn_warning_candidate",
  "critical_eye_warning_candidate",
  "signal_quality_issue",
];

function dateValue(value: string): number {
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : 0;
}

function percent(part: number, total: number): number {
  return total === 0 ? 0 : part / total;
}

function formatPercentValue(value: number): string {
  return `${Math.round(value * 100)}%`;
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

function sourceSortLabel(source: HistorySource): string {
  return SOURCE_LABELS[source] ?? source;
}

export function normalizeHistoryEventKind(
  record: DriverHistoryEvent
): InsightEventKind | null {
  if (record.state === "eye_warning_candidate") return "eye_warning_candidate";
  if (record.state === "mouth_warning_candidate") return "yawn_warning_candidate";
  if (record.state === "high_confidence_drowsiness_candidate") {
    return "critical_eye_warning_candidate";
  }
  if (record.state === "signal_unreliable") return "signal_quality_issue";
  return null;
}

export function getInsightRecords(
  records: DriverHistoryEvent[],
  now = new Date(),
  hours: TimeWindowHours = 48
): DriverHistoryEvent[] {
  const seen = new Set<string>();
  return records
    .filter((record) => normalizeHistoryEventKind(record) !== null)
    .filter((record) => isEventInWindow(record, now, hours))
    .filter((record) => {
      const key = record.ingestionKey || record.id;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .sort((a, b) => dateValue(b.timestamp) - dateValue(a.timestamp));
}

function countKind(
  records: DriverHistoryEvent[],
  kind: InsightEventKind
): number {
  return records.filter((record) => normalizeHistoryEventKind(record) === kind).length;
}

function getDominantKind(records: DriverHistoryEvent[]): {
  kind: InsightEventKind | null;
  count: number;
  share: number;
} {
  const total = records.length;
  const ranked = INSIGHT_KIND_ORDER.map((kind) => ({
    kind,
    count: countKind(records, kind),
  })).sort((a, b) => b.count - a.count);
  const top = ranked[0];

  return {
    kind: top && top.count > 0 ? top.kind : null,
    count: top?.count ?? 0,
    share: percent(top?.count ?? 0, total),
  };
}

export function getInsightSummary(records: DriverHistoryEvent[]): InsightSummary {
  const totalWarningCandidates = records.length;
  const dominant = getDominantKind(records);
  const highPriorityCount = records.filter(
    (record) =>
      normalizeHistoryEventKind(record) === "critical_eye_warning_candidate" ||
      record.severity === "high"
  ).length;
  const signalQualityCount = countKind(records, "signal_quality_issue");
  const reviewedCount = records.filter(
    (record) => record.reviewStatus === "reviewed"
  ).length;
  const reviewableCount = records.filter(
    (record) => record.reviewStatus !== "not_required"
  ).length;
  const signalQualityShare = percent(signalQualityCount, totalWarningCandidates);

  let signalQualityBurdenLabel = "Low signal-quality burden";
  if (signalQualityShare >= 0.35) {
    signalQualityBurdenLabel = "High signal-quality burden";
  } else if (signalQualityShare >= 0.15) {
    signalQualityBurdenLabel = "Moderate signal-quality burden";
  } else if (signalQualityCount === 0) {
    signalQualityBurdenLabel = "No signal-quality burden";
  }

  return {
    totalWarningCandidates,
    dominantPatternLabel: dominant.kind
      ? `${INSIGHT_KIND_META[dominant.kind].shortLabel} dominant`
      : "No dominant pattern",
    dominantPatternShare: dominant.share,
    highPriorityShare: percent(highPriorityCount, totalWarningCandidates),
    highPriorityCount,
    signalQualityBurdenLabel,
    signalQualityShare,
    signalQualityCount,
    reviewCompletionShare: percent(reviewedCount, Math.max(reviewableCount, 1)),
    reviewedCount,
    reviewableCount,
  };
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

export function getWarningCandidateTrend(
  records: DriverHistoryEvent[],
  now = new Date(),
  hours: TimeWindowHours = 48
): InsightTrendPoint[] {
  const bucketMinutes = getBucketMinutes(hours);
  const bucketMs = bucketMinutes * 60_000;

  return buildBuckets(now, hours).map((bucketStart) => {
    const bucketEnd = new Date(bucketStart.getTime() + bucketMs);
    const bucketRecords = records.filter((record) => {
      const timestamp = dateValue(record.timestamp);
      return timestamp >= bucketStart.getTime() && timestamp < bucketEnd.getTime();
    });

    return {
      timestamp: bucketStart.toISOString(),
      label: bucketLabel(bucketStart, hours),
      eyeWarning: countKind(bucketRecords, "eye_warning_candidate"),
      yawnWarning: countKind(bucketRecords, "yawn_warning_candidate"),
      criticalEyeWarning: countKind(
        bucketRecords,
        "critical_eye_warning_candidate"
      ),
      signalQualityIssue: countKind(bucketRecords, "signal_quality_issue"),
    };
  });
}

export function getEventTypeDistribution(
  records: DriverHistoryEvent[]
): InsightCompositionItem[] {
  const total = records.length;
  return INSIGHT_KIND_ORDER.map((kind) => ({
    kind,
    label: INSIGHT_KIND_META[kind].label,
    count: countKind(records, kind),
    share: percent(countKind(records, kind), total),
    color: INSIGHT_KIND_META[kind].color,
  }));
}

export function getTimeOfDayPattern(
  records: DriverHistoryEvent[]
): InsightTimeOfDayItem[] {
  const groups: InsightTimeOfDayItem[] = [
    { id: "night", label: "Night", timeRange: "00:00-06:00", count: 0, share: 0 },
    {
      id: "morning",
      label: "Morning",
      timeRange: "06:00-12:00",
      count: 0,
      share: 0,
    },
    {
      id: "afternoon",
      label: "Afternoon",
      timeRange: "12:00-18:00",
      count: 0,
      share: 0,
    },
    {
      id: "evening",
      label: "Evening",
      timeRange: "18:00-24:00",
      count: 0,
      share: 0,
    },
  ];

  for (const record of records) {
    const date = new Date(record.timestamp);
    if (!Number.isFinite(date.getTime())) continue;
    const hour = date.getHours();
    const index = hour < 6 ? 0 : hour < 12 ? 1 : hour < 18 ? 2 : 3;
    groups[index] = {
      ...groups[index],
      count: groups[index].count + 1,
    };
  }

  return groups.map((group) => ({
    ...group,
    share: percent(group.count, records.length),
  }));
}

export function describeTimeOfDayPattern(
  items: InsightTimeOfDayItem[]
): string {
  const top = [...items].sort((a, b) => b.count - a.count)[0];
  if (!top || top.count === 0) {
    return "No local warning-candidate time-of-day pattern is available in the selected window.";
  }

  return `More warning-candidate events were recorded during ${top.label.toLowerCase()} sessions (${top.timeRange}).`;
}

export function getSessionComparison(
  records: DriverHistoryEvent[],
  sessions: DriverHistorySession[]
): InsightSessionComparisonRow[] {
  const sessionMap = new Map(sessions.map((session) => [session.id, session]));
  const grouped = new Map<string, DriverHistoryEvent[]>();

  for (const record of records) {
    const current = grouped.get(record.sessionId) ?? [];
    current.push(record);
    grouped.set(record.sessionId, current);
  }

  return Array.from(grouped.entries())
    .map(([sessionId, sessionRecords]) => {
      const session = sessionMap.get(sessionId);
      const dominant = getDominantKind(sessionRecords);
      const startedAt =
        session?.startedAt ??
        sessionRecords.reduce(
          (min, record) =>
            dateValue(record.timestamp) < dateValue(min) ? record.timestamp : min,
          sessionRecords[0]?.timestamp ?? new Date(0).toISOString()
        );

      return {
        sessionId,
        source: session?.source ?? sessionRecords[0]?.source ?? "mock",
        startedAt,
        eventCount: sessionRecords.length,
        highPriorityCount: sessionRecords.filter(
          (record) =>
            normalizeHistoryEventKind(record) ===
              "critical_eye_warning_candidate" || record.severity === "high"
        ).length,
        signalQualityIssueCount: countKind(
          sessionRecords,
          "signal_quality_issue"
        ),
        pendingReviewCount: sessionRecords.filter(
          (record) => record.reviewStatus === "pending"
        ).length,
        dominantPattern: dominant.kind
          ? INSIGHT_KIND_META[dominant.kind].shortLabel
          : "No dominant pattern",
      };
    })
    .filter(
      (row) =>
        row.sessionId.trim().length > 0 &&
        Number.isFinite(new Date(row.startedAt).getTime())
    )
    .sort((a, b) => dateValue(b.startedAt) - dateValue(a.startedAt));
}

export function getSignalQualityInsights(
  records: DriverHistoryEvent[]
): InsightSignalQualitySummary {
  const signalRecords = records.filter(
    (record) => normalizeHistoryEventKind(record) === "signal_quality_issue"
  );
  const countsBySession = new Map<string, number>();

  for (const record of signalRecords) {
    countsBySession.set(
      record.sessionId,
      (countsBySession.get(record.sessionId) ?? 0) + 1
    );
  }

  const [mostLimitedSessionId, mostLimitedSessionCount] =
    Array.from(countsBySession.entries()).sort((a, b) => b[1] - a[1])[0] ?? [
      null,
      0,
    ];

  return {
    count: signalRecords.length,
    share: percent(signalRecords.length, records.length),
    affectedSessionCount: countsBySession.size,
    mostLimitedSessionId,
    mostLimitedSessionCount,
  };
}

export function getReviewRecommendations(
  records: DriverHistoryEvent[]
): InsightRecommendation[] {
  const recommendations: InsightRecommendation[] = [];
  const criticalCount = countKind(records, "critical_eye_warning_candidate");
  const signalQuality = getSignalQualityInsights(records);
  const sessionRows = getSessionComparison(records, []);
  const repeatedEyeSession = sessionRows.find((row) => {
    const sessionRecords = records.filter(
      (record) => record.sessionId === row.sessionId
    );
    return countKind(sessionRecords, "eye_warning_candidate") >= 2;
  });
  const repeatedYawnSession = sessionRows.find((row) => {
    const sessionRecords = records.filter(
      (record) => record.sessionId === row.sessionId
    );
    return countKind(sessionRecords, "yawn_warning_candidate") >= 2;
  });

  if (criticalCount > 0) {
    recommendations.push({
      id: "review-critical-eye",
      priority: "high",
      title: "Review critical eye warning candidates first.",
      body: `${criticalCount} high-priority warning-candidate record${criticalCount === 1 ? "" : "s"} should be reviewed before lower-priority items.`,
    });
  }

  if (signalQuality.count > 0) {
    recommendations.push({
      id: "review-signal-quality",
      priority: signalQuality.share >= 0.2 ? "high" : "medium",
      title: "Check signal-quality-heavy sessions before interpreting patterns.",
      body: `Signal quality issues account for ${formatPercentValue(signalQuality.share)} of local warning-candidate records.`,
    });
  }

  if (repeatedEyeSession) {
    recommendations.push({
      id: "review-repeated-eye",
      priority: "medium",
      title: "Review sessions with repeated eye warning candidates.",
      body: `Session ${repeatedEyeSession.sessionId} contains repeated eye warning-candidate records.`,
    });
  }

  if (repeatedYawnSession) {
    recommendations.push({
      id: "compare-yawn-eye",
      priority: "low",
      title: "Compare yawn-warning clusters with eye-warning clusters.",
      body: `Session ${repeatedYawnSession.sessionId} includes repeated yawn warning-candidate records for local review context.`,
    });
  }

  if (recommendations.length === 0) {
    recommendations.push({
      id: "continue-review",
      priority: "low",
      title: "Continue reviewing new local warning-candidate records.",
      body: "No high-priority review pattern stands out in the selected local history window.",
    });
  }

  return recommendations.slice(0, 4);
}

export function formatInsightPercent(value: number): string {
  return formatPercentValue(value);
}

export function formatInsightSource(source: HistorySource): string {
  return sourceSortLabel(source);
}
