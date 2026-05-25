"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { EventTimelineTable } from "@/components/history-48h/EventTimelineTable";
import { HistoryFilters } from "@/components/history-48h/HistoryFilters";
import { HistoryHeader } from "@/components/history-48h/HistoryHeader";
import { HistoryInterpretationNote } from "@/components/history-48h/HistoryInterpretationNote";
import { HistorySummaryCards } from "@/components/history-48h/HistorySummaryCards";
import { RecentSessionsSummary } from "@/components/history-48h/RecentSessionsSummary";
import {
  filterHistory48hStoreBySource,
  loadHistory48hStore,
} from "@/lib/history48hStorage";
import { useVisionGuardAuth } from "@/lib/authStore";
import {
  archiveRecordsToHistoryStore,
  getArchiveHealth,
  getArchiveRecords,
} from "@/lib/backendArchiveApi";
import type { BackendArchiveRange } from "@/lib/backendArchiveTypes";
import type {
  DriverHistoryEvent,
  DriverHistorySession,
  History48hStore,
  HistoryFilters as HistoryFilterState,
  TimeWindowHours,
} from "@/lib/history48hTypes";
import {
  EVENT_TYPE_OPTIONS,
  HISTORY_EVENT_PAGE_SIZE,
  TIME_WINDOW_OPTIONS,
  buildHistorySummaryText,
  buildSessionRows,
  clampPage,
  filterHistoryEvents,
  formatTimeRange,
  paginateItems,
  summarizeHistory,
} from "@/lib/history48hUtils";
import {
  buildHistoryCsv,
  buildHistorySummaryHtml,
  buildRawHistoryJson,
  downloadTextFile,
  historyCsvFilename,
  historyRawJsonFilename,
  historySummaryFilename,
  type HistoryExportPayload,
} from "@/lib/history48hExportUtils";

const DEFAULT_FILTERS: HistoryFilterState = {
  timeWindowHours: 48,
  eventType: "all",
  review: "all",
  source: "live_monitor",
};

const EMPTY_STORE: History48hStore = {
  events: [],
  sessions: [],
  updatedAt: "",
};

function archiveRangeFromTimeWindow(
  timeWindowHours: TimeWindowHours
): BackendArchiveRange {
  if (timeWindowHours <= 48) return "48h";
  if (timeWindowHours <= 168) return "7d";
  if (timeWindowHours <= 720) return "30d";
  return "all";
}

function liveMonitorOnly(store: History48hStore): History48hStore {
  return filterHistory48hStoreBySource(store, "live_monitor");
}

function timestampMs(value: string | undefined): number {
  const timestamp = value ? new Date(value).getTime() : 0;
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function eventKey(event: DriverHistoryEvent): string {
  return event.ingestionKey || event.id;
}

function sessionKey(session: DriverHistorySession): string {
  return `${session.userId || "__legacy__"}:${session.id}`;
}

function mergeHistoryEvents(
  preferredEvents: DriverHistoryEvent[],
  fallbackEvents: DriverHistoryEvent[]
): DriverHistoryEvent[] {
  const eventMap = new Map<string, DriverHistoryEvent>();
  for (const event of [...preferredEvents, ...fallbackEvents]) {
    const key = eventKey(event);
    if (!eventMap.has(key)) {
      eventMap.set(key, event);
    }
  }
  return Array.from(eventMap.values()).sort(
    (a, b) => timestampMs(b.timestamp) - timestampMs(a.timestamp)
  );
}

function mergeSessionStatus(
  a: DriverHistorySession["status"],
  b: DriverHistorySession["status"]
): DriverHistorySession["status"] {
  if (a === "completed" || b === "completed") return "completed";
  if (a === "demo" || b === "demo") return "demo";
  return "partial";
}

function mergeSession(
  preferredSession: DriverHistorySession,
  fallbackSession: DriverHistorySession
): DriverHistorySession {
  const startMs = Math.min(
    timestampMs(preferredSession.startedAt),
    timestampMs(fallbackSession.startedAt)
  );
  const endMs = Math.max(
    timestampMs(preferredSession.endedAt),
    timestampMs(fallbackSession.endedAt)
  );
  const startedAt =
    startMs > 0 ? new Date(startMs).toISOString() : preferredSession.startedAt;
  const endedAt =
    endMs > 0 ? new Date(endMs).toISOString() : preferredSession.endedAt;
  const durationMin =
    startMs > 0 && endMs > 0
      ? Math.max(
          preferredSession.durationMin,
          fallbackSession.durationMin,
          Math.round(((endMs - startMs) / 60_000) * 10) / 10
        )
      : Math.max(preferredSession.durationMin, fallbackSession.durationMin);

  return {
    ...fallbackSession,
    ...preferredSession,
    startedAt,
    endedAt,
    durationMin,
    status: mergeSessionStatus(preferredSession.status, fallbackSession.status),
    normalCount: Math.max(
      preferredSession.normalCount,
      fallbackSession.normalCount
    ),
    eyeWarningCount: Math.max(
      preferredSession.eyeWarningCount,
      fallbackSession.eyeWarningCount
    ),
    mouthWarningCount: Math.max(
      preferredSession.mouthWarningCount,
      fallbackSession.mouthWarningCount
    ),
    highConfidenceCount: Math.max(
      preferredSession.highConfidenceCount,
      fallbackSession.highConfidenceCount
    ),
    signalUnreliableCount: Math.max(
      preferredSession.signalUnreliableCount,
      fallbackSession.signalUnreliableCount
    ),
    reviewPendingCount: Math.max(
      preferredSession.reviewPendingCount,
      fallbackSession.reviewPendingCount
    ),
  };
}

function mergeHistorySessions(
  preferredSessions: DriverHistorySession[],
  fallbackSessions: DriverHistorySession[]
): DriverHistorySession[] {
  const sessionMap = new Map<string, DriverHistorySession>();
  for (const session of fallbackSessions) {
    sessionMap.set(sessionKey(session), session);
  }
  for (const session of preferredSessions) {
    const key = sessionKey(session);
    const existing = sessionMap.get(key);
    sessionMap.set(key, existing ? mergeSession(session, existing) : session);
  }
  return Array.from(sessionMap.values()).sort(
    (a, b) => timestampMs(b.startedAt) - timestampMs(a.startedAt)
  );
}

function mergeHistoryStores(
  localStore: History48hStore,
  archiveStore: History48hStore
): History48hStore {
  return {
    events: mergeHistoryEvents(archiveStore.events, localStore.events),
    sessions: mergeHistorySessions(localStore.sessions, archiveStore.sessions),
    updatedAt: archiveStore.updatedAt || localStore.updatedAt,
  };
}

function isSessionInWindow(
  session: DriverHistorySession,
  now: Date,
  timeWindowHours: TimeWindowHours
): boolean {
  const nowMs = now.getTime();
  const windowStartMs = nowMs - timeWindowHours * 60 * 60_000;
  const sessionStartMs = timestampMs(session.startedAt);
  const sessionEndMs = timestampMs(session.endedAt) || sessionStartMs;
  return sessionEndMs >= windowStartMs && sessionStartMs <= nowMs;
}

function scopeFreeFilters(filters: HistoryFilterState): HistoryFilterState {
  return {
    ...filters,
    review: "all",
    source: "live_monitor",
    sessionId: undefined,
  };
}

export function History48hPage() {
  const searchParams = useSearchParams();
  const [store, setStore] = useState<History48hStore>(EMPTY_STORE);
  const [archiveStore, setArchiveStore] = useState<History48hStore | null>(null);
  const [archiveAvailable, setArchiveAvailable] = useState(false);
  const [filters, setFilters] = useState<HistoryFilterState>(DEFAULT_FILTERS);
  const [eventListPage, setEventListPage] = useState(1);
  const [referenceNow, setReferenceNow] = useState<Date>(new Date());
  const [copyStatus, setCopyStatus] = useState("");
  const { currentUser, isLegacyRecordVisible } = useVisionGuardAuth();
  const currentUserId = currentUser?.id;
  const includeLegacyRecords = currentUser
    ? isLegacyRecordVisible(undefined)
    : false;
  const searchParamsKey = searchParams.toString();

  const resetListPages = useCallback(() => {
    setEventListPage(1);
  }, []);

  const handleFilterChange = useCallback(
    (nextFilters: HistoryFilterState) => {
      setFilters({
        ...nextFilters,
        review: "all",
        source: "live_monitor",
      });
      resetListPages();
    },
    [resetListPages]
  );

  useEffect(() => {
    const query = searchParamsKey;
    if (!query) return;

    const id = window.setTimeout(() => {
      const params = new URLSearchParams(query);
      const nextFilters: HistoryFilterState = { ...DEFAULT_FILTERS };
      const sessionId = params.get("sessionId")?.trim();
      const eventType = params.get("eventType");
      const timeWindowHours = Number(params.get("timeWindowHours"));

      if (sessionId) nextFilters.sessionId = sessionId;
      if (EVENT_TYPE_OPTIONS.some((option) => option.value === eventType)) {
        nextFilters.eventType = eventType as HistoryFilterState["eventType"];
      }
      if (
        TIME_WINDOW_OPTIONS.some((option) => option.value === timeWindowHours)
      ) {
        nextFilters.timeWindowHours = timeWindowHours as TimeWindowHours;
      }

      setFilters(nextFilters);
      resetListPages();
    }, 0);

    return () => window.clearTimeout(id);
  }, [resetListPages, searchParamsKey]);

  useEffect(() => {
    const id = window.setTimeout(() => {
      const now = new Date();
      setReferenceNow(now);
      setStore(
        liveMonitorOnly(
          loadHistory48hStore(now, currentUserId, includeLegacyRecords)
        )
      );
    }, 0);

    return () => window.clearTimeout(id);
  }, [currentUserId, includeLegacyRecords]);

  const refreshArchive = useCallback(async () => {
    try {
      const health = await getArchiveHealth();
      if (!health.ok || !health.enabled) {
        setArchiveAvailable(false);
        setArchiveStore(null);
        return;
      }

      const archiveRange = archiveRangeFromTimeWindow(filters.timeWindowHours);
      const response = await getArchiveRecords({
        range: archiveRange,
        source: "live_monitor",
        limit: 500,
      });
      const nextStore = liveMonitorOnly(
        archiveRecordsToHistoryStore(response.records)
      );
      setArchiveStore(nextStore);
      setArchiveAvailable(response.enabled && nextStore.events.length > 0);
    } catch {
      setArchiveAvailable(false);
      setArchiveStore(null);
    }
  }, [filters.timeWindowHours]);

  useEffect(() => {
    const id = window.setTimeout(() => {
      void refreshArchive();
    }, 0);

    return () => window.clearTimeout(id);
  }, [refreshArchive]);

  const activeStore = useMemo(
    () =>
      archiveAvailable && archiveStore
        ? mergeHistoryStores(store, archiveStore)
        : store,
    [archiveAvailable, archiveStore, store]
  );
  const baseFilters = useMemo(() => scopeFreeFilters(filters), [filters]);

  const driveScopeEvents = useMemo(
    () => filterHistoryEvents(activeStore.events, baseFilters, referenceNow),
    [activeStore.events, baseFilters, referenceNow]
  );

  const driveScopeSessions = useMemo(
    () =>
      activeStore.sessions.filter(
        (session) =>
          session.source === "live_monitor" &&
          isSessionInWindow(session, referenceNow, filters.timeWindowHours)
      ),
    [activeStore.sessions, filters.timeWindowHours, referenceNow]
  );

  const visibleEvents = useMemo(
    () =>
      filters.sessionId
        ? driveScopeEvents.filter((event) => event.sessionId === filters.sessionId)
        : driveScopeEvents,
    [driveScopeEvents, filters.sessionId]
  );

  const sessionRows = useMemo(
    () => buildSessionRows(driveScopeSessions, driveScopeEvents),
    [driveScopeEvents, driveScopeSessions]
  );

  const scopedSessionRows = useMemo(
    () =>
      filters.sessionId
        ? sessionRows.filter((session) => session.id === filters.sessionId)
        : sessionRows,
    [filters.sessionId, sessionRows]
  );

  const selectedSession = useMemo(
    () => sessionRows.find((session) => session.id === filters.sessionId),
    [filters.sessionId, sessionRows]
  );

  const driveLabels = useMemo(
    () =>
      Object.fromEntries(
        sessionRows.map((session) => [
          session.id,
          formatTimeRange(session.startedAt, session.endedAt),
        ])
      ),
    [sessionRows]
  );

  const summary = useMemo(
    () => summarizeHistory(visibleEvents, scopedSessionRows),
    [scopedSessionRows, visibleEvents]
  );

  useEffect(() => {
    const id = window.setTimeout(() => {
      setEventListPage((page) =>
        clampPage(page, visibleEvents.length, HISTORY_EVENT_PAGE_SIZE)
      );
    }, 0);

    return () => window.clearTimeout(id);
  }, [visibleEvents.length]);

  const paginatedVisibleEvents = useMemo(
    () => paginateItems(visibleEvents, eventListPage, HISTORY_EVENT_PAGE_SIZE),
    [eventListPage, visibleEvents]
  );

  const handleSelectSession = useCallback(
    (sessionId?: string) => {
      setFilters((current) => ({ ...current, sessionId }));
      resetListPages();
    },
    [resetListPages]
  );

  async function handleCopySummary() {
    if (visibleEvents.length === 0) {
      setCopyStatus("No alert summary to copy");
      return;
    }

    const text = buildHistorySummaryText(summary, filters);
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus("Summary copied");
    } catch {
      setCopyStatus("Copy unavailable in this browser");
    }
  }

  function createExportPayload(): HistoryExportPayload {
    return {
      exportedAt: new Date().toISOString(),
      filters,
      summary,
      events: visibleEvents,
      sessions: scopedSessionRows,
    };
  }

  function handleDownloadSummary() {
    if (visibleEvents.length === 0) {
      setCopyStatus("No alert summary to download");
      return;
    }

    try {
      const payload = createExportPayload();
      downloadTextFile(
        historySummaryFilename(filters),
        buildHistorySummaryHtml(payload),
        "text/html;charset=utf-8"
      );
      setCopyStatus("Summary downloaded");
    } catch {
      setCopyStatus("Summary download failed");
    }
  }

  function handleDownloadCsv() {
    if (visibleEvents.length === 0) {
      setCopyStatus("No alert table to download");
      return;
    }

    try {
      downloadTextFile(
        historyCsvFilename(),
        buildHistoryCsv(createExportPayload()),
        "text/csv;charset=utf-8"
      );
      setCopyStatus("CSV downloaded");
    } catch {
      setCopyStatus("CSV download failed");
    }
  }

  function handleExportRawData() {
    if (visibleEvents.length === 0) {
      setCopyStatus("No raw history data to export");
      return;
    }

    try {
      downloadTextFile(
        historyRawJsonFilename(),
        buildRawHistoryJson(createExportPayload()),
        "application/json;charset=utf-8"
      );
      setCopyStatus("Raw data exported");
    } catch {
      setCopyStatus("Raw data export failed");
    }
  }

  const scopeText = selectedSession
    ? `Showing alerts from ${formatTimeRange(
        selectedSession.startedAt,
        selectedSession.endedAt
      )}.`
    : "Showing all alerts from the selected time window.";

  const emptyMessage = filters.sessionId
    ? "No matching alerts for this drive. Try another drive or show all drives."
    : filters.eventType === "all"
      ? "No alerts in this time window. Try a wider time window."
      : "No matching alerts. Try a different alert type or time window.";

  return (
    <main className="flex-1 overflow-y-auto bg-[#f4f7f9] px-4 py-5 lg:px-6">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-5 pb-10">
        <HistoryHeader />
        <HistoryFilters
          filters={filters}
          copyStatus={copyStatus}
          eventCount={visibleEvents.length}
          onChange={handleFilterChange}
          onDownloadSummary={handleDownloadSummary}
          onCopySummary={handleCopySummary}
          onDownloadCsv={handleDownloadCsv}
          onExportRawData={handleExportRawData}
        />
        <HistorySummaryCards summary={summary} />
        <RecentSessionsSummary
          sessions={sessionRows}
          selectedSessionId={filters.sessionId}
          totalAlertCount={driveScopeEvents.length}
          onSelectSession={handleSelectSession}
        />
        <EventTimelineTable
          events={paginatedVisibleEvents}
          totalCount={visibleEvents.length}
          scopeText={scopeText}
          selectedSessionId={filters.sessionId}
          driveLabels={driveLabels}
          page={eventListPage}
          pageSize={HISTORY_EVENT_PAGE_SIZE}
          onPageChange={setEventListPage}
          onShowAllDrives={() => handleSelectSession(undefined)}
          emptyMessage={emptyMessage}
        />
        <HistoryInterpretationNote />
      </div>
    </main>
  );
}
