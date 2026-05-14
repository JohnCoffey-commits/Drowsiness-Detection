import { createDemoHistory48hStore } from "@/lib/history48hMockData";
import type {
  DriverHistoryEvent,
  DriverHistorySession,
  History48hStore,
} from "@/lib/history48hTypes";

export const HISTORY_48H_STORAGE_KEY = "visionguard.history48h.v1";

const WINDOW_MS = 48 * 60 * 60 * 1_000;

function hasBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function isEventWithin48h(event: DriverHistoryEvent, now: Date): boolean {
  const timestamp = new Date(event.timestamp).getTime();
  return Number.isFinite(timestamp) && now.getTime() - timestamp <= WINDOW_MS;
}

function isSessionRelevant(
  session: DriverHistorySession,
  events: DriverHistoryEvent[],
  now: Date
): boolean {
  const sessionHasFreshEvent = events.some((event) => event.sessionId === session.id);
  if (sessionHasFreshEvent) return true;

  const endedAt = new Date(session.endedAt).getTime();
  return Number.isFinite(endedAt) && now.getTime() - endedAt <= WINDOW_MS;
}

export function pruneHistory48hStore(
  store: History48hStore,
  now = new Date()
): History48hStore {
  const events = store.events.filter((event) => isEventWithin48h(event, now));
  const sessions = store.sessions.filter((session) =>
    isSessionRelevant(session, events, now)
  );

  return {
    events,
    sessions,
    updatedAt: store.updatedAt,
  };
}

function parseHistoryStore(value: string | null): History48hStore | null {
  if (!value) return null;

  try {
    const parsed = JSON.parse(value) as Partial<History48hStore>;
    if (!Array.isArray(parsed.events) || !Array.isArray(parsed.sessions)) {
      return null;
    }

    return {
      events: parsed.events as DriverHistoryEvent[],
      sessions: parsed.sessions as DriverHistorySession[],
      updatedAt:
        typeof parsed.updatedAt === "string"
          ? parsed.updatedAt
          : new Date().toISOString(),
    };
  } catch {
    return null;
  }
}

export function saveHistory48hStore(store: History48hStore): void {
  if (!hasBrowserStorage()) return;
  window.localStorage.setItem(HISTORY_48H_STORAGE_KEY, JSON.stringify(store));
}

export function loadHistory48hStore(now = new Date()): History48hStore {
  if (!hasBrowserStorage()) {
    return pruneHistory48hStore(createDemoHistory48hStore(now), now);
  }

  const existing = parseHistoryStore(
    window.localStorage.getItem(HISTORY_48H_STORAGE_KEY)
  );

  if (!existing) {
    const seeded = createDemoHistory48hStore(now);
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

export function resetHistory48hDemoData(now = new Date()): History48hStore {
  const store = createDemoHistory48hStore(now);
  saveHistory48hStore(store);
  return store;
}

export function clearHistory48hStore(now = new Date()): History48hStore {
  const store: History48hStore = {
    events: [],
    sessions: [],
    updatedAt: now.toISOString(),
  };
  saveHistory48hStore(store);
  return store;
}
