import {
  createDemoHistory48hStore,
  createUserDemoHistory48hStore,
} from "@/lib/history48hMockData";
import type {
  DriverHistoryEvent,
  DriverHistorySession,
  EyeEvidenceStrength,
  History48hStore,
  HistorySeverity,
  HistorySource,
  HistoryState,
  ReviewStatus,
} from "@/lib/history48hTypes";

export const HISTORY_48H_STORAGE_KEY = "visionguard.history48h.v1";

const WINDOW_MS = 48 * 60 * 60 * 1_000;
const LEGACY_USER_KEY = "__legacy__";

function createEmptyHistory48hStore(now = new Date()): History48hStore {
  return {
    events: [],
    sessions: [],
    updatedAt: now.toISOString(),
  };
}

const HISTORY_STATES = new Set<HistoryState>([
  "normal",
  "eye_warning_candidate",
  "mouth_warning_candidate",
  "high_confidence_drowsiness_candidate",
  "signal_unreliable",
]);

const HISTORY_SEVERITIES = new Set<HistorySeverity>([
  "low",
  "medium",
  "high",
  "unreliable",
]);

const HISTORY_SOURCES = new Set<HistorySource>([
  "mock",
  "live_monitor",
  "video_upload",
  "manual",
  "webcam_future",
]);

const REVIEW_STATUSES = new Set<ReviewStatus>([
  "pending",
  "reviewed",
  "not_required",
]);

const EYE_EVIDENCE_STRENGTHS = new Set<EyeEvidenceStrength>([
  "none",
  "weak",
  "moderate",
  "strong",
  "unknown",
]);

function hasBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function textValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function positiveNumber(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : fallback;
}

function optionalNumber(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function isoDateValue(value: unknown): string | null {
  const text = textValue(value);
  const timestamp = new Date(text).getTime();
  return Number.isFinite(timestamp) ? new Date(timestamp).toISOString() : null;
}

function sourceValue(value: unknown): HistorySource {
  return typeof value === "string" &&
    HISTORY_SOURCES.has(value as HistorySource)
    ? (value as HistorySource)
    : "mock";
}

function stateValue(value: unknown): HistoryState | null {
  return typeof value === "string" && HISTORY_STATES.has(value as HistoryState)
    ? (value as HistoryState)
    : null;
}

function severityValue(value: unknown, state: HistoryState): HistorySeverity {
  if (
    typeof value === "string" &&
    HISTORY_SEVERITIES.has(value as HistorySeverity)
  ) {
    return value as HistorySeverity;
  }

  if (state === "normal") return "low";
  if (state === "signal_unreliable") return "unreliable";
  if (state === "high_confidence_drowsiness_candidate") return "high";
  return "medium";
}

function reviewStatusValue(value: unknown, state: HistoryState): ReviewStatus {
  if (typeof value === "string" && REVIEW_STATUSES.has(value as ReviewStatus)) {
    return value as ReviewStatus;
  }

  return state === "normal" ? "not_required" : "pending";
}

function eyeEvidenceStrengthValue(
  value: unknown
): EyeEvidenceStrength | undefined {
  return typeof value === "string" &&
    EYE_EVIDENCE_STRENGTHS.has(value as EyeEvidenceStrength)
    ? (value as EyeEvidenceStrength)
    : undefined;
}

function relatedRouteValue(
  value: unknown
): DriverHistoryEvent["relatedRoute"] | undefined {
  return value === "/" || value === "/video-upload" || value === "/history-48h"
    ? value
    : undefined;
}

function normalizeHistoryEvent(value: unknown): DriverHistoryEvent | null {
  if (!value || typeof value !== "object") return null;

  const record = value as Partial<DriverHistoryEvent>;
  const id = textValue(record.id);
  const sessionId = textValue(record.sessionId);
  const timestamp = isoDateValue(record.timestamp);
  const state = stateValue(record.state);

  if (!id || !sessionId || !timestamp || !state) {
    return null;
  }

  const event: DriverHistoryEvent = {
    id,
    sessionId,
    timestamp,
    durationSec: positiveNumber(record.durationSec, 0),
    state,
    severity: severityValue(record.severity, state),
    source: sourceValue(record.source),
    reason:
      textValue(record.reason) ||
      "Legacy local warning-candidate history record.",
    reviewStatus: reviewStatusValue(record.reviewStatus, state),
  };

  const userId = textValue(record.userId);
  const sourceEventId = textValue(record.sourceEventId);
  const ingestionKey = textValue(record.ingestionKey);
  const endTimestamp = isoDateValue(record.endTimestamp);
  const title = textValue(record.title);
  const summary = textValue(record.summary);
  const relatedRoute = relatedRouteValue(record.relatedRoute);
  const eyeEvidenceStrength = eyeEvidenceStrengthValue(record.eyeEvidenceStrength);
  const pEyeClosedMax = optionalNumber(record.pEyeClosedMax);
  const pYawnMax = optionalNumber(record.pYawnMax);
  const candidateSeverityScore = optionalNumber(record.candidateSeverityScore);

  if (userId) event.userId = userId;
  if (sourceEventId) event.sourceEventId = sourceEventId;
  if (ingestionKey) event.ingestionKey = ingestionKey;
  if (endTimestamp) event.endTimestamp = endTimestamp;
  if (title) event.title = title;
  if (summary) event.summary = summary;
  if (relatedRoute) event.relatedRoute = relatedRoute;
  if (eyeEvidenceStrength) event.eyeEvidenceStrength = eyeEvidenceStrength;
  if (pEyeClosedMax != null) event.pEyeClosedMax = pEyeClosedMax;
  if (pYawnMax != null) event.pYawnMax = pYawnMax;
  if (candidateSeverityScore != null) {
    event.candidateSeverityScore = candidateSeverityScore;
  }

  return event;
}

function normalizeHistorySession(value: unknown): DriverHistorySession | null {
  if (!value || typeof value !== "object") return null;

  const record = value as Partial<DriverHistorySession>;
  const id = textValue(record.id);
  const startedAt = isoDateValue(record.startedAt);
  const endedAt = isoDateValue(record.endedAt);

  if (!id || !startedAt || !endedAt) {
    return null;
  }

  const session: DriverHistorySession = {
    id,
    source: sourceValue(record.source),
    startedAt,
    endedAt,
    durationMin: positiveNumber(record.durationMin, 0),
    status:
      record.status === "completed" ||
      record.status === "partial" ||
      record.status === "demo"
        ? record.status
        : "partial",
    normalCount: positiveNumber(record.normalCount, 0),
    eyeWarningCount: positiveNumber(record.eyeWarningCount, 0),
    mouthWarningCount: positiveNumber(record.mouthWarningCount, 0),
    highConfidenceCount: positiveNumber(record.highConfidenceCount, 0),
    signalUnreliableCount: positiveNumber(record.signalUnreliableCount, 0),
    reviewPendingCount: positiveNumber(record.reviewPendingCount, 0),
  };

  const userId = textValue(record.userId);
  if (userId) session.userId = userId;

  return session;
}

function eventTimestamp(event: DriverHistoryEvent): number {
  const timestamp = new Date(event.timestamp).getTime();
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function sessionTimestamp(session: DriverHistorySession): number {
  const timestamp = new Date(session.startedAt).getTime();
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function userKey(userId?: string): string {
  return userId || LEGACY_USER_KEY;
}

function getEventDedupeKey(event: DriverHistoryEvent): string {
  return event.ingestionKey || `${userKey(event.userId)}:${event.id}`;
}

function getSessionDedupeKey(session: DriverHistorySession): string {
  return `${userKey(session.userId)}:${session.id}`;
}

function hasSameSessionOwner(
  session: DriverHistorySession,
  event: DriverHistoryEvent
): boolean {
  return session.id === event.sessionId && userKey(session.userId) === userKey(event.userId);
}

function isEventWithin48h(event: DriverHistoryEvent, now: Date): boolean {
  const timestamp = eventTimestamp(event);
  return Number.isFinite(timestamp) && now.getTime() - timestamp <= WINDOW_MS;
}

function isSessionRelevant(
  session: DriverHistorySession,
  events: DriverHistoryEvent[],
  now: Date
): boolean {
  const sessionHasFreshEvent = events.some((event) =>
    hasSameSessionOwner(session, event)
  );
  if (sessionHasFreshEvent) return true;

  const endedAt = new Date(session.endedAt).getTime();
  return Number.isFinite(endedAt) && now.getTime() - endedAt <= WINDOW_MS;
}

export function dedupeHistory48hStore(store: History48hStore): History48hStore {
  const eventMap = new Map<string, DriverHistoryEvent>();
  for (const event of [...store.events].sort(
    (a, b) => eventTimestamp(b) - eventTimestamp(a)
  )) {
    const key = getEventDedupeKey(event);
    if (!eventMap.has(key)) {
      eventMap.set(key, event);
    }
  }

  const sessionMap = new Map<string, DriverHistorySession>();
  for (const session of [...store.sessions].sort(
    (a, b) => sessionTimestamp(b) - sessionTimestamp(a)
  )) {
    const key = getSessionDedupeKey(session);
    if (!sessionMap.has(key)) {
      sessionMap.set(key, session);
    }
  }

  return {
    events: Array.from(eventMap.values()).sort(
      (a, b) => eventTimestamp(b) - eventTimestamp(a)
    ),
    sessions: Array.from(sessionMap.values()).sort(
      (a, b) => sessionTimestamp(b) - sessionTimestamp(a)
    ),
    updatedAt: store.updatedAt,
  };
}

export function pruneHistory48hStore(
  store: History48hStore,
  now = new Date()
): History48hStore {
  const events = store.events.filter((event) => isEventWithin48h(event, now));
  const sessions = store.sessions.filter((session) =>
    isSessionRelevant(session, events, now)
  );

  return dedupeHistory48hStore({
    events,
    sessions,
    updatedAt: store.updatedAt,
  });
}

function parseHistoryStore(value: string | null): History48hStore | null {
  if (!value) return null;

  try {
    const parsed = JSON.parse(value) as Partial<History48hStore>;
    if (!Array.isArray(parsed.events) || !Array.isArray(parsed.sessions)) {
      return null;
    }

    return dedupeHistory48hStore({
      events: parsed.events
        .map((event) => normalizeHistoryEvent(event))
        .filter(
          (event): event is DriverHistoryEvent => Boolean(event)
        ),
      sessions: parsed.sessions
        .map((session) => normalizeHistorySession(session))
        .filter(
          (session): session is DriverHistorySession => Boolean(session)
        ),
      updatedAt:
        typeof parsed.updatedAt === "string"
          ? parsed.updatedAt
          : new Date().toISOString(),
    });
  } catch {
    return null;
  }
}

export function saveHistory48hStore(store: History48hStore): void {
  if (!hasBrowserStorage()) return;
  window.localStorage.setItem(
    HISTORY_48H_STORAGE_KEY,
    JSON.stringify(dedupeHistory48hStore(store))
  );
}

function loadRawHistory48hStore(
  now = new Date(),
  seedUserId?: string,
  seedWhenMissing = true
): History48hStore {
  if (!hasBrowserStorage()) {
    const store = seedWhenMissing
      ? createUserDemoHistory48hStore(now, seedUserId)
      : createEmptyHistory48hStore(now);
    return pruneHistory48hStore(store, now);
  }

  const existing = parseHistoryStore(
    window.localStorage.getItem(HISTORY_48H_STORAGE_KEY)
  );

  if (!existing) {
    if (!seedWhenMissing) {
      return createEmptyHistory48hStore(now);
    }

    const seeded = seedUserId
      ? createUserDemoHistory48hStore(now, seedUserId)
      : createDemoHistory48hStore(now);
    saveHistory48hStore(seeded);
    return seeded;
  }

  const pruned = pruneHistory48hStore(existing, now);
  if (
    pruned.events.length !== existing.events.length ||
    pruned.sessions.length !== existing.sessions.length
  ) {
    saveHistory48hStore(pruned);
  }

  return pruned;
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

export function filterHistory48hStoreForUser(
  store: History48hStore,
  userId?: string,
  includeLegacyRecords = false
): History48hStore {
  const events = store.events.filter((event) =>
    isVisibleToUser(event.userId, userId, includeLegacyRecords)
  );
  const sessions = store.sessions.filter((session) =>
    isVisibleToUser(session.userId, userId, includeLegacyRecords)
  );

  return {
    events,
    sessions,
    updatedAt: store.updatedAt,
  };
}

export function filterHistory48hStoreBySource(
  store: History48hStore,
  source: HistorySource
): History48hStore {
  const events = store.events.filter((event) => event.source === source);

  return {
    events,
    sessions: store.sessions.filter(
      (session) => session.source === source
    ),
    updatedAt: store.updatedAt,
  };
}

export function loadHistory48hStore(
  now = new Date(),
  userId?: string,
  includeLegacyRecords = false
): History48hStore {
  return filterHistory48hStoreForUser(
    loadRawHistory48hStore(now, userId),
    userId,
    includeLegacyRecords
  );
}

function shouldReplaceUserRecord(
  recordUserId: string | undefined,
  userId?: string,
  includeLegacyRecords = false
): boolean {
  return isVisibleToUser(recordUserId, userId, includeLegacyRecords);
}

export function saveHistory48hUserStore(
  store: History48hStore,
  userId?: string,
  includeLegacyRecords = false,
  now = new Date()
): History48hStore {
  const nextUserStore = pruneHistory48hStore(
    {
      ...store,
      updatedAt: now.toISOString(),
    },
    now
  );

  if (!userId || !hasBrowserStorage()) {
    saveHistory48hStore(nextUserStore);
    return nextUserStore;
  }

  const rawStore = loadRawHistory48hStore(now, undefined, false);
  const mergedStore = pruneHistory48hStore(
    {
      events: [
        ...rawStore.events.filter(
          (event) =>
            !shouldReplaceUserRecord(event.userId, userId, includeLegacyRecords)
        ),
        ...nextUserStore.events,
      ],
      sessions: [
        ...rawStore.sessions.filter(
          (session) =>
            !shouldReplaceUserRecord(
              session.userId,
              userId,
              includeLegacyRecords
            )
        ),
        ...nextUserStore.sessions,
      ],
      updatedAt: now.toISOString(),
    },
    now
  );

  saveHistory48hStore(mergedStore);
  return filterHistory48hStoreForUser(
    mergedStore,
    userId,
    includeLegacyRecords
  );
}

function sameEventSession(
  event: DriverHistoryEvent,
  sessionId: string,
  userId?: string
): boolean {
  return event.sessionId === sessionId && userKey(event.userId) === userKey(userId);
}

function countSessionState(
  events: DriverHistoryEvent[],
  state: HistoryState
): number {
  return events.filter((event) => event.state === state).length;
}

function dateMs(value: string): number {
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function earliestIso(...values: Array<string | undefined>): string {
  const timestamps = values
    .map((value) => (value ? dateMs(value) : 0))
    .filter((timestamp) => timestamp > 0);
  return timestamps.length > 0
    ? new Date(Math.min(...timestamps)).toISOString()
    : new Date().toISOString();
}

function latestIso(...values: Array<string | undefined>): string {
  const timestamps = values
    .map((value) => (value ? dateMs(value) : 0))
    .filter((timestamp) => timestamp > 0);
  return timestamps.length > 0
    ? new Date(Math.max(...timestamps)).toISOString()
    : new Date().toISOString();
}

function sessionDurationMin(startedAt: string, endedAt: string): number {
  const startedAtMs = dateMs(startedAt);
  const endedAtMs = dateMs(endedAt);
  if (!startedAtMs || !endedAtMs) return 0;
  return Math.max(0, Math.round(((endedAtMs - startedAtMs) / 60_000) * 10) / 10);
}

function sessionStatus(
  existingStatus: DriverHistorySession["status"] | undefined,
  nextStatus: DriverHistorySession["status"]
): DriverHistorySession["status"] {
  if (nextStatus === "completed" || existingStatus === "completed") {
    return "completed";
  }
  if (nextStatus === "demo" || existingStatus === "demo") {
    return "demo";
  }
  return "partial";
}

function buildSessionFromEvents(
  baseSession: DriverHistorySession,
  events: DriverHistoryEvent[]
): DriverHistorySession {
  return {
    ...baseSession,
    durationMin: Math.max(
      baseSession.durationMin,
      sessionDurationMin(baseSession.startedAt, baseSession.endedAt)
    ),
    normalCount: countSessionState(events, "normal"),
    eyeWarningCount: countSessionState(events, "eye_warning_candidate"),
    mouthWarningCount: countSessionState(events, "mouth_warning_candidate"),
    highConfidenceCount: countSessionState(
      events,
      "high_confidence_drowsiness_candidate"
    ),
    signalUnreliableCount: countSessionState(events, "signal_unreliable"),
    reviewPendingCount: events.filter(
      (candidate) => candidate.reviewStatus === "pending"
    ).length,
  };
}

function createOrUpdateSessionForEvent(
  store: History48hStore,
  event: DriverHistoryEvent
): History48hStore {
  const sessionEvents = store.events
    .filter((candidate) =>
      sameEventSession(candidate, event.sessionId, event.userId)
    )
    .sort((a, b) => eventTimestamp(a) - eventTimestamp(b));
  const firstEvent = sessionEvents[0] ?? event;
  const lastEvent = sessionEvents[sessionEvents.length - 1] ?? event;
  const existingSession = store.sessions.find(
    (candidate) =>
      candidate.id === event.sessionId &&
      userKey(candidate.userId) === userKey(event.userId)
  );
  const eventEndedAt =
    lastEvent.endTimestamp ??
    new Date(eventTimestamp(lastEvent) + lastEvent.durationSec * 1_000).toISOString();
  const startedAt = existingSession
    ? earliestIso(existingSession.startedAt, firstEvent.timestamp)
    : firstEvent.timestamp;
  const endedAt = existingSession
    ? latestIso(existingSession.endedAt, eventEndedAt)
    : eventEndedAt;

  const baseSession: DriverHistorySession = {
    id: event.sessionId,
    userId: event.userId,
    source: existingSession?.source ?? event.source,
    startedAt,
    endedAt,
    durationMin: Math.max(
      existingSession?.durationMin ?? 0,
      sessionDurationMin(startedAt, endedAt)
    ),
    status: sessionStatus(
      existingSession?.status,
      event.source === "live_monitor" ? "partial" : "completed"
    ),
    normalCount: 0,
    eyeWarningCount: 0,
    mouthWarningCount: 0,
    highConfidenceCount: 0,
    signalUnreliableCount: 0,
    reviewPendingCount: 0,
  };
  const session = buildSessionFromEvents(baseSession, sessionEvents);

  return {
    ...store,
    sessions: [
      session,
      ...store.sessions.filter(
        (candidate) =>
          !(
            candidate.id === session.id &&
            userKey(candidate.userId) === userKey(session.userId)
          )
      ),
    ],
  };
}

export function appendHistory48hRecord(
  record: DriverHistoryEvent,
  now = new Date()
): History48hStore {
  const normalizedRecord = normalizeHistoryEvent(record);
  if (!normalizedRecord) {
    return loadRawHistory48hStore(now, undefined, false);
  }

  const rawStore = loadRawHistory48hStore(now, undefined, false);
  const dedupeKey = getEventDedupeKey(normalizedRecord);
  if (rawStore.events.some((event) => getEventDedupeKey(event) === dedupeKey)) {
    return rawStore;
  }

  const withEvent = pruneHistory48hStore(
    {
      ...rawStore,
      events: [normalizedRecord, ...rawStore.events],
      updatedAt: now.toISOString(),
    },
    now
  );
  const withSession = pruneHistory48hStore(
    createOrUpdateSessionForEvent(withEvent, normalizedRecord),
    now
  );

  saveHistory48hStore(withSession);
  return withSession;
}

export function upsertHistory48hSession(
  session: DriverHistorySession,
  now = new Date()
): History48hStore {
  const normalizedSession = normalizeHistorySession(session);
  if (!normalizedSession) {
    return loadRawHistory48hStore(now, undefined, false);
  }

  const rawStore = loadRawHistory48hStore(now, undefined, false);
  const existingSession = rawStore.sessions.find(
    (candidate) =>
      candidate.id === normalizedSession.id &&
      userKey(candidate.userId) === userKey(normalizedSession.userId)
  );
  const startedAt = existingSession
    ? earliestIso(existingSession.startedAt, normalizedSession.startedAt)
    : normalizedSession.startedAt;
  const endedAt = existingSession
    ? latestIso(existingSession.endedAt, normalizedSession.endedAt)
    : normalizedSession.endedAt;
  const sessionEvents = rawStore.events.filter((event) =>
    sameEventSession(event, normalizedSession.id, normalizedSession.userId)
  );
  const baseSession: DriverHistorySession = {
    ...normalizedSession,
    startedAt,
    endedAt,
    durationMin: Math.max(
      existingSession?.durationMin ?? 0,
      normalizedSession.durationMin,
      sessionDurationMin(startedAt, endedAt)
    ),
    status: sessionStatus(existingSession?.status, normalizedSession.status),
  };
  const nextSession = buildSessionFromEvents(baseSession, sessionEvents);
  const nextStore = pruneHistory48hStore(
    {
      ...rawStore,
      sessions: [
        nextSession,
        ...rawStore.sessions.filter(
          (candidate) =>
            !(
              candidate.id === nextSession.id &&
              userKey(candidate.userId) === userKey(nextSession.userId)
            )
        ),
      ],
      updatedAt: now.toISOString(),
    },
    now
  );

  saveHistory48hStore(nextStore);
  return nextStore;
}

export function resetHistory48hDemoData(
  now = new Date(),
  userId?: string,
  includeLegacyRecords = true
): History48hStore {
  if (!userId) {
    const store = createDemoHistory48hStore(now);
    saveHistory48hStore(store);
    return store;
  }

  const demoStore = createUserDemoHistory48hStore(now, userId);
  return saveHistory48hUserStore(demoStore, userId, includeLegacyRecords, now);
}

export function clearHistory48hStore(
  now = new Date(),
  userId?: string,
  includeLegacyRecords = false
): History48hStore {
  if (!userId || !hasBrowserStorage()) {
    const store = createEmptyHistory48hStore(now);
    saveHistory48hStore(store);
    return store;
  }

  const rawStore = loadRawHistory48hStore(now, undefined, false);
  const nextStore = pruneHistory48hStore(
    {
      events: rawStore.events.filter(
        (event) =>
          !shouldReplaceUserRecord(event.userId, userId, includeLegacyRecords)
      ),
      sessions: rawStore.sessions.filter(
        (session) =>
          !shouldReplaceUserRecord(session.userId, userId, includeLegacyRecords)
      ),
      updatedAt: now.toISOString(),
    },
    now
  );

  saveHistory48hStore(nextStore);
  return filterHistory48hStoreForUser(nextStore, userId, includeLegacyRecords);
}
