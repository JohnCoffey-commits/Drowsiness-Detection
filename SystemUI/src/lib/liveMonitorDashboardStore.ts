import type {
  LiveMonitorDashboardCounts,
  LiveMonitorDashboardEvent,
  LiveMonitorDashboardEventDraft,
  LiveMonitorDashboardEventKind,
  LiveMonitorDashboardStore,
  LiveMonitorRiskPoint,
} from "@/lib/liveMonitorDashboardTypes";
import type { LiveAlertEvent, LiveAlertKind } from "@/lib/liveMonitorAlertUtils";
import type {
  LiveMonitorRiskSeverity,
  LiveMonitorRiskState,
} from "@/lib/liveMonitorRiskUtils";

export const LIVE_MONITOR_DASHBOARD_STORAGE_KEY =
  "visionguard.liveMonitorDashboard.v1";

const LAST_HOUR_MS = 60 * 60 * 1000;
const RISK_POINT_BUCKET_MS = 5_000;
const MAX_RISK_POINTS = Math.ceil(LAST_HOUR_MS / RISK_POINT_BUCKET_MS) + 24;

const DISPLAY_SEVERITY_RANGES: Record<
  LiveMonitorRiskSeverity,
  { min: number; max: number; baseline: number }
> = {
  idle: { min: 0, max: 0, baseline: 0 },
  low: { min: 8, max: 30, baseline: 20 },
  medium: { min: 35, max: 65, baseline: 52 },
  high: { min: 72, max: 82, baseline: 76 },
  critical: { min: 86, max: 96, baseline: 91 },
  signal_quality: { min: 25, max: 30, baseline: 28 },
};

const LIVE_MONITOR_RISK_SEVERITIES = new Set<LiveMonitorRiskSeverity>([
  "idle",
  "low",
  "medium",
  "high",
  "critical",
  "signal_quality",
]);

export const EMPTY_LIVE_MONITOR_DASHBOARD_STORE: LiveMonitorDashboardStore = {
  events: [],
  riskPoints: [],
  updatedAt: "",
};

const EVENT_LABELS: Record<LiveMonitorDashboardEventKind, string> = {
  normal: "Normal",
  eye_warning: "Eye Warning",
  yawn_warning: "Yawn Warning",
  critical_eye_warning: "Critical Eye Warning",
  signal_quality: "Signal Check",
};

const EVENT_SEVERITY_SCORES: Record<LiveMonitorDashboardEventKind, number> = {
  normal: 20,
  eye_warning: 74,
  yawn_warning: 55,
  critical_eye_warning: 92,
  signal_quality: 30,
};

function hasBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function parseStore(value: string | null): LiveMonitorDashboardStore | null {
  if (!value) return null;

  try {
    const parsed = JSON.parse(value) as Partial<LiveMonitorDashboardStore>;
    if (!Array.isArray(parsed.events) || !Array.isArray(parsed.riskPoints)) {
      return null;
    }

    return {
      events: parsed.events as LiveMonitorDashboardEvent[],
      riskPoints: compactLiveMonitorRiskPoints(
        parsed.riskPoints as LiveMonitorRiskPoint[],
        new Date()
      ),
      updatedAt:
        typeof parsed.updatedAt === "string"
          ? parsed.updatedAt
          : new Date().toISOString(),
    };
  } catch {
    return null;
  }
}

function dateValue(value: string): number {
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : 0;
}

function riskPointTimestampMs(point: LiveMonitorRiskPoint): number {
  const directTimestampMs = (point as { timestampMs?: unknown }).timestampMs;
  if (typeof directTimestampMs === "number" && Number.isFinite(directTimestampMs)) {
    return directTimestampMs;
  }

  return dateValue(point.timestamp);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function sessionSeed(sessionId: string): number {
  return Array.from(sessionId).reduce(
    (seed, character) => (seed * 31 + character.charCodeAt(0)) % 9973,
    17
  );
}

function createDisplaySeverityWaveOffset(
  timestampMs: number,
  sessionId: string,
  amplitude: number
): number {
  const seed = sessionSeed(sessionId);
  const slowWave = Math.sin(timestampMs / 14_000 + seed * 0.013);
  const secondaryWave = Math.sin(timestampMs / 31_000 + seed * 0.029) * 0.35;
  return (slowWave + secondaryWave) * amplitude;
}

function deriveDisplaySeverityScore(
  severity: LiveMonitorRiskSeverity,
  sessionId: string,
  timestampMs: number
): number {
  const range = DISPLAY_SEVERITY_RANGES[severity];
  if (severity === "idle") {
    return 0;
  }

  const amplitude = (range.max - range.min) / 2;
  const rawScore =
    range.baseline +
    createDisplaySeverityWaveOffset(timestampMs, sessionId, amplitude * 0.55);

  return Math.round(clamp(rawScore, range.min, range.max));
}

function isLiveMonitorRiskSeverity(
  severity: unknown
): severity is LiveMonitorRiskSeverity {
  return (
    typeof severity === "string" &&
    LIVE_MONITOR_RISK_SEVERITIES.has(severity as LiveMonitorRiskSeverity)
  );
}

function normalizeRiskPoint(
  point: LiveMonitorRiskPoint
): LiveMonitorRiskPoint | null {
  const timestampMs = riskPointTimestampMs(point);
  const score = Number(point.score);
  const severity = point.severity;

  if (
    !Number.isFinite(timestampMs) ||
    timestampMs <= 0 ||
    !Number.isFinite(score) ||
    score < 0 ||
    score > 100 ||
    !isLiveMonitorRiskSeverity(severity)
  ) {
    return null;
  }

  const timestamp = new Date(timestampMs).toISOString();
  const safeSessionId =
    typeof point.sessionId === "string" && point.sessionId.length > 0
      ? point.sessionId
      : "unknown-live-monitor-session";

  return {
    id:
      typeof point.id === "string" && point.id.length > 0
        ? point.id
        : `live-risk-${safeSessionId}-${timestampMs}-${severity}-${score}`,
    userId:
      typeof point.userId === "string" && point.userId.length > 0
        ? point.userId
        : undefined,
    timestamp,
    sessionId: safeSessionId,
    score,
    displaySeverityScore: deriveDisplaySeverityScore(
      severity,
      safeSessionId,
      timestampMs
    ),
    severity,
  };
}

function riskPointBucketKey(point: LiveMonitorRiskPoint): number {
  return (
    Math.floor(riskPointTimestampMs(point) / RISK_POINT_BUCKET_MS) *
    RISK_POINT_BUCKET_MS
  );
}

function shouldReplaceBucketPoint(
  current: LiveMonitorRiskPoint,
  next: LiveMonitorRiskPoint
): boolean {
  const currentTimestamp = riskPointTimestampMs(current);
  const nextTimestamp = riskPointTimestampMs(next);

  if (nextTimestamp !== currentTimestamp) {
    return nextTimestamp > currentTimestamp;
  }

  return next.displaySeverityScore >= current.displaySeverityScore;
}

function areRiskPointsEqual(
  current: LiveMonitorRiskPoint[],
  next: LiveMonitorRiskPoint[]
): boolean {
  if (current.length !== next.length) {
    return false;
  }

  return current.every((point, index) => {
    const candidate = next[index];
    return (
      point.id === candidate.id &&
      point.userId === candidate.userId &&
      point.timestamp === candidate.timestamp &&
      point.sessionId === candidate.sessionId &&
      point.score === candidate.score &&
      point.displaySeverityScore === candidate.displaySeverityScore &&
      point.severity === candidate.severity
    );
  });
}

function compactLiveMonitorRiskPoints(
  points: LiveMonitorRiskPoint[],
  now = new Date()
): LiveMonitorRiskPoint[] {
  const end = now.getTime();
  const start = end - LAST_HOUR_MS;
  const buckets = new Map<number, LiveMonitorRiskPoint>();

  for (const point of points) {
    const normalizedPoint = normalizeRiskPoint(point);
    if (!normalizedPoint) {
      continue;
    }

    const timestamp = riskPointTimestampMs(normalizedPoint);
    if (timestamp < start || timestamp > end) {
      continue;
    }

    const bucketKey = riskPointBucketKey(normalizedPoint);
    const current = buckets.get(bucketKey);

    if (!current || shouldReplaceBucketPoint(current, normalizedPoint)) {
      buckets.set(bucketKey, normalizedPoint);
    }
  }

  return Array.from(buckets.values())
    .sort((a, b) => riskPointTimestampMs(a) - riskPointTimestampMs(b))
    .slice(-MAX_RISK_POINTS);
}

export function createLiveMonitorDriveSessionId(now = new Date()): string {
  // TODO: replace this browser-run drive session with authenticated user-session logic.
  return `live-drive-${now.getTime()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createEmptyLiveMonitorDashboardStore(
  now = new Date()
): LiveMonitorDashboardStore {
  return {
    ...EMPTY_LIVE_MONITOR_DASHBOARD_STORE,
    updatedAt: now.toISOString(),
  };
}

export function loadLiveMonitorDashboardStore(): LiveMonitorDashboardStore {
  if (!hasBrowserStorage()) {
    return createEmptyLiveMonitorDashboardStore();
  }

  return (
    parseStore(window.localStorage.getItem(LIVE_MONITOR_DASHBOARD_STORAGE_KEY)) ??
    createEmptyLiveMonitorDashboardStore()
  );
}

export function saveLiveMonitorDashboardStore(
  store: LiveMonitorDashboardStore
): void {
  if (!hasBrowserStorage()) return;
  window.localStorage.setItem(
    LIVE_MONITOR_DASHBOARD_STORAGE_KEY,
    JSON.stringify(store)
  );
}

export function appendLiveMonitorDashboardEvent(
  store: LiveMonitorDashboardStore,
  draft: LiveMonitorDashboardEventDraft,
  sessionId: string,
  userId?: string
): LiveMonitorDashboardStore {
  if (store.events.some((event) => event.id === draft.id)) {
    return store;
  }

  const event: LiveMonitorDashboardEvent = {
    ...draft,
    userId,
    sessionId,
    source: "live_monitor_prototype",
  };

  return {
    ...store,
    events: [event, ...store.events].sort(
      (a, b) => dateValue(b.timestamp) - dateValue(a.timestamp)
    ),
    updatedAt: new Date().toISOString(),
  };
}

export function appendLiveMonitorRiskPoint(
  store: LiveMonitorDashboardStore,
  point: LiveMonitorRiskPoint
): LiveMonitorDashboardStore {
  const normalizedPoint = normalizeRiskPoint(point);
  if (!normalizedPoint) {
    return store;
  }

  const nextRiskPoints = compactLiveMonitorRiskPoints(
    [...store.riskPoints, normalizedPoint],
    new Date(Math.max(Date.now(), riskPointTimestampMs(normalizedPoint)))
  );

  if (areRiskPointsEqual(store.riskPoints, nextRiskPoints)) {
    return store;
  }

  return {
    ...store,
    riskPoints: nextRiskPoints,
    updatedAt: new Date().toISOString(),
  };
}

export function createLiveMonitorRiskPoint(
  riskState: LiveMonitorRiskState,
  sessionId: string,
  now = new Date(),
  userId?: string
): LiveMonitorRiskPoint {
  const timestamp = now.toISOString();
  const timestampMs = now.getTime();
  const score = Number.isFinite(riskState.score)
    ? Math.max(0, Math.min(100, riskState.score))
    : 0;

  return {
    id: `live-risk-${sessionId}-${timestampMs}-${riskState.severity}-${score}`,
    userId,
    timestamp,
    sessionId,
    score,
    displaySeverityScore: deriveDisplaySeverityScore(
      riskState.severity,
      sessionId,
      timestampMs
    ),
    severity: riskState.severity,
  };
}

function isVisibleToUser(
  recordUserId: string | undefined,
  userId?: string,
  includeLegacyRecords = false
): boolean {
  if (!userId) {
    return includeLegacyRecords && !recordUserId;
  }

  if (recordUserId) {
    return recordUserId === userId;
  }

  return includeLegacyRecords;
}

export function filterLiveMonitorEventsForUser(
  events: LiveMonitorDashboardEvent[],
  userId?: string,
  includeLegacyRecords = false
): LiveMonitorDashboardEvent[] {
  return events.filter((event) =>
    isVisibleToUser(event.userId, userId, includeLegacyRecords)
  );
}

export function filterLiveMonitorRiskPointsForUser(
  points: LiveMonitorRiskPoint[],
  userId?: string,
  includeLegacyRecords = false
): LiveMonitorRiskPoint[] {
  return points.filter((point) =>
    isVisibleToUser(point.userId, userId, includeLegacyRecords)
  );
}

export function createNormalDashboardEventDraft(
  reason: "monitoring_started" | "returned_to_monitoring",
  now = new Date()
): LiveMonitorDashboardEventDraft {
  return {
    id: `live-event-normal-${reason}-${now.getTime()}`,
    timestamp: now.toISOString(),
    kind: "normal",
    label: EVENT_LABELS.normal,
    severityScore: EVENT_SEVERITY_SCORES.normal,
  };
}

export function dashboardEventDraftFromLiveAlertEvent(
  event: LiveAlertEvent,
  overrideKind?: LiveMonitorDashboardEventKind
): LiveMonitorDashboardEventDraft {
  const kind = overrideKind ?? dashboardKindFromAlertKind(event.kind);

  return {
    id: `dashboard-${kind}-${event.id}`,
    timestamp: new Date(event.timestamp).toISOString(),
    kind,
    label: EVENT_LABELS[kind],
    severityScore: EVENT_SEVERITY_SCORES[kind],
  };
}

export function dashboardKindFromAlertKind(
  kind: LiveAlertKind
): LiveMonitorDashboardEventKind {
  if (kind === "mouth_warning") return "yawn_warning";
  if (kind === "high_confidence") return "critical_eye_warning";
  if (kind === "signal_quality") return "signal_quality";
  return "eye_warning";
}

export function summarizeCurrentDriveEvents(
  events: LiveMonitorDashboardEvent[],
  sessionId: string | null,
  userId?: string,
  includeLegacyRecords = false
): LiveMonitorDashboardCounts {
  if (!sessionId) {
    return {
      eyeWarnings: 0,
      yawnWarnings: 0,
    };
  }

  return filterLiveMonitorEventsForUser(events, userId, includeLegacyRecords)
    .filter((event) => event.sessionId === sessionId)
    .reduce<LiveMonitorDashboardCounts>(
      (summary, event) => {
        if (
          event.kind === "eye_warning" ||
          event.kind === "critical_eye_warning"
        ) {
          summary.eyeWarnings += 1;
        }

        if (event.kind === "yawn_warning") {
          summary.yawnWarnings += 1;
        }

        return summary;
      },
      {
        eyeWarnings: 0,
        yawnWarnings: 0,
      }
    );
}

export function isSameLocalCalendarDate(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  );
}

export function getTodayLiveMonitorEvents(
  events: LiveMonitorDashboardEvent[],
  now = new Date(),
  userId?: string,
  includeLegacyRecords = false
): LiveMonitorDashboardEvent[] {
  return filterLiveMonitorEventsForUser(events, userId, includeLegacyRecords)
    .filter((event) => {
      const timestamp = new Date(event.timestamp);
      return (
        Number.isFinite(timestamp.getTime()) &&
        isSameLocalCalendarDate(timestamp, now)
      );
    })
    .sort((a, b) => dateValue(b.timestamp) - dateValue(a.timestamp));
}

export function getLastHourLiveMonitorRiskPoints(
  points: LiveMonitorRiskPoint[],
  now = new Date()
): LiveMonitorRiskPoint[] {
  return compactLiveMonitorRiskPoints(points, now);
}

export function getCurrentSessionLiveMonitorRiskPoints(
  points: LiveMonitorRiskPoint[],
  sessionId: string,
  now = new Date(),
  userId?: string,
  includeLegacyRecords = false
): LiveMonitorRiskPoint[] {
  return compactLiveMonitorRiskPoints(
    filterLiveMonitorRiskPointsForUser(
      points,
      userId,
      includeLegacyRecords
    ).filter((point) => point.sessionId === sessionId),
    now
  );
}
