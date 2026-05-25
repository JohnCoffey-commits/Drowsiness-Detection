import type {
  DriverHistoryEvent,
  DriverHistorySession,
  HistorySource,
  TimeWindowHours,
} from "@/lib/history48hTypes";
import {
  SOURCE_LABELS,
  formatMinutes,
  formatTimeRange,
  getBucketMinutes,
} from "@/lib/history48hUtils";
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
  "Insights are derived from lightweight Live Monitor alert summaries. They show patterns in recent visual cues such as eye closure, yawning, and camera signal quality.";

export const INSIGHT_KIND_META: Record<InsightEventKind, InsightKindMeta> = {
  eye_warning_candidate: {
    kind: "eye_warning_candidate",
    label: "Eye-closure alert",
    shortLabel: "Eye closure",
    color: "#f97316",
  },
  yawn_warning_candidate: {
    kind: "yawn_warning_candidate",
    label: "Yawn alert",
    shortLabel: "Yawn",
    color: "#ec4899",
  },
  critical_eye_warning_candidate: {
    kind: "critical_eye_warning_candidate",
    label: "High-risk eye alert",
    shortLabel: "High-risk eye",
    color: "#ef4444",
  },
  signal_quality_issue: {
    kind: "signal_quality_issue",
    label: "Camera signal interruption",
    shortLabel: "Camera signal",
    color: "#64748b",
  },
};

const INSIGHT_KIND_ORDER: InsightEventKind[] = [
  "critical_eye_warning_candidate",
  "eye_warning_candidate",
  "yawn_warning_candidate",
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
  })).sort(
    (a, b) =>
      b.count - a.count ||
      INSIGHT_KIND_ORDER.indexOf(a.kind) - INSIGHT_KIND_ORDER.indexOf(b.kind)
  );
  const top = ranked[0];

  return {
    kind: top && top.count > 0 ? top.kind : null,
    count: top?.count ?? 0,
    share: percent(top?.count ?? 0, total),
  };
}

export function getInsightSummary(records: DriverHistoryEvent[]): InsightSummary {
  const totalAlerts = records.length;
  const dominant = getDominantKind(records);
  const highPriorityCount = records.filter(
    (record) =>
      normalizeHistoryEventKind(record) === "critical_eye_warning_candidate" ||
      record.severity === "high"
  ).length;
  const signalInterruptionCount = countKind(records, "signal_quality_issue");

  return {
    totalAlerts,
    dominantAlertLabel: dominant.kind
      ? INSIGHT_KIND_META[dominant.kind].label
      : "No dominant alert",
    dominantAlertCount: dominant.count,
    dominantAlertShare: dominant.share,
    highPriorityShare: percent(highPriorityCount, totalAlerts),
    highPriorityCount,
    signalInterruptionShare: percent(signalInterruptionCount, totalAlerts),
    signalInterruptionCount,
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
    return "No alert timing pattern is available in the selected window.";
  }

  return `Most alerts were recorded during ${top.label.toLowerCase()} drives.`;
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
      const endedAt =
        session?.endedAt ??
        sessionRecords.reduce(
          (max, record) =>
            dateValue(record.timestamp) > dateValue(max) ? record.timestamp : max,
          sessionRecords[0]?.timestamp ?? startedAt
        );
      const durationMin = session?.durationMin ?? Math.max(
        1,
        Math.round((dateValue(endedAt) - dateValue(startedAt)) / 60_000)
      );
      const criticalEyeCount = countKind(
        sessionRecords,
        "critical_eye_warning_candidate"
      );
      const eyeClosureCount = countKind(sessionRecords, "eye_warning_candidate");
      const yawnCount = countKind(sessionRecords, "yawn_warning_candidate");
      const signalInterruptionCount = countKind(
        sessionRecords,
        "signal_quality_issue"
      );

      return {
        sessionId,
        source: session?.source ?? sessionRecords[0]?.source ?? "mock",
        startedAt,
        endedAt,
        driveLabel: formatTimeRange(startedAt, endedAt),
        durationLabel: formatMinutes(durationMin),
        eventCount: sessionRecords.length,
        highPriorityCount: sessionRecords.filter(
          (record) =>
            normalizeHistoryEventKind(record) ===
              "critical_eye_warning_candidate" || record.severity === "high"
        ).length,
        signalInterruptionCount,
        criticalEyeCount,
        eyeClosureCount,
        yawnCount,
        dominantPattern: dominant.kind
          ? INSIGHT_KIND_META[dominant.kind].label
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
  records: DriverHistoryEvent[],
  sessionRows: InsightSessionComparisonRow[] = []
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
  const mostLimitedDriveLabel =
    sessionRows.find((row) => row.sessionId === mostLimitedSessionId)
      ?.driveLabel ?? null;

  return {
    count: signalRecords.length,
    share: percent(signalRecords.length, records.length),
    affectedSessionCount: countsBySession.size,
    mostLimitedSessionId,
    mostLimitedDriveLabel,
    mostLimitedSessionCount,
  };
}

export function getKeyInsights({
  records,
  summary,
  timeOfDay,
  sessionRows,
  signalQuality,
}: {
  records: DriverHistoryEvent[];
  summary: InsightSummary;
  timeOfDay: InsightTimeOfDayItem[];
  sessionRows: InsightSessionComparisonRow[];
  signalQuality: InsightSignalQualitySummary;
}): string[] {
  if (records.length === 0) {
    return ["No insights yet. Start a Live Monitor drive to generate alert patterns."];
  }

  const insights: string[] = [];
  const topTime = [...timeOfDay].sort((a, b) => b.count - a.count)[0];

  if (summary.dominantAlertCount > 0) {
    insights.push(
      `Most alerts were ${pluralizeAlertLabel(summary.dominantAlertLabel).toLowerCase()}.`
    );
  }
  if (topTime && topTime.count > 0) {
    insights.push(
      `Alerts were concentrated during ${topTime.label.toLowerCase()} drives.`
    );
  }
  if (signalQuality.affectedSessionCount > 0) {
    insights.push(
      `Camera signal interruptions affected ${formatCount(signalQuality.affectedSessionCount, "drive")}.`
    );
  }
  insights.push(
    `Based on ${formatCount(records.length, "alert")} across ${formatCount(sessionRows.length, "recent drive")}.`
  );

  if (records.length < 10 || sessionRows.length < 3) {
    insights.push(
      "Insights are based on a small number of recent drives, so treat patterns as early signals."
    );
  } else {
    insights.push(
      `Patterns become more reliable as more Live Monitor drives are recorded.`
    );
  }

  return insights;
}

export function getAttentionAreas(
  records: DriverHistoryEvent[],
  sessionRows: InsightSessionComparisonRow[] = []
): InsightRecommendation[] {
  const areas: InsightRecommendation[] = [];
  const criticalCount = countKind(records, "critical_eye_warning_candidate");
  const eyeCount = countKind(records, "eye_warning_candidate");
  const yawnCount = countKind(records, "yawn_warning_candidate");
  const signalQuality = getSignalQualityInsights(records, sessionRows);

  if (criticalCount > 0) {
    areas.push({
      id: "high-risk-eye-alerts",
      priority: "high",
      title: "High-risk eye alerts were frequent.",
      body: `${formatCount(criticalCount, "alert")} showed stronger eye-closure-related cues in the selected window.`,
    });
  }

  if (signalQuality.count > 0) {
    areas.push({
      id: "camera-signal-interruptions",
      priority: signalQuality.share >= 0.2 ? "high" : "medium",
      title: `Camera signal interruptions appeared in ${formatCount(signalQuality.affectedSessionCount, "drive")}.`,
      body: "Check camera angle, lighting, and face visibility if this repeats.",
    });
  }

  if (yawnCount > 0 && eyeCount + criticalCount > 0) {
    areas.push({
      id: "yawn-with-eye-alerts",
      priority: "medium",
      title: "Yawn alerts appeared alongside eye-related alerts.",
      body: "Open History to see when these alerts occurred during each drive.",
    });
  }

  if (areas.length === 0) {
    areas.push({
      id: "limited-alert-patterns",
      priority: "low",
      title: "No single alert pattern stands out yet.",
      body: "Keep recording Live Monitor drives to make recent patterns easier to compare.",
    });
  }

  return areas.slice(0, 4);
}

export function formatInsightPercent(value: number): string {
  return formatPercentValue(value);
}

export function formatInsightSource(source: HistorySource): string {
  return sourceSortLabel(source);
}

export function formatCount(count: number, singular: string): string {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

export function pluralizeAlertLabel(label: string): string {
  if (label === "No dominant alert") return label;
  return label.endsWith("alert") ? `${label}s` : label;
}
