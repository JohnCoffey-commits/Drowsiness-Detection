"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { EventTimelineTable } from "@/components/history-48h/EventTimelineTable";
import { HistoryFilters } from "@/components/history-48h/HistoryFilters";
import { HistoryHeader } from "@/components/history-48h/HistoryHeader";
import { HistoryInterpretationNote } from "@/components/history-48h/HistoryInterpretationNote";
import { HistorySummaryCards } from "@/components/history-48h/HistorySummaryCards";
import { ManualReviewQueue } from "@/components/history-48h/ManualReviewQueue";
import { RecentSessionsSummary } from "@/components/history-48h/RecentSessionsSummary";
import {
  clearHistory48hStore,
  loadHistory48hStore,
  resetHistory48hDemoData,
  saveHistory48hUserStore,
} from "@/lib/history48hStorage";
import { useVisionGuardAuth } from "@/lib/authStore";
import {
  archiveRecordsToHistoryStore,
  exportArchiveRecords,
  getArchiveHealth,
  getArchiveRecords,
  updateArchiveRecordReview,
} from "@/lib/backendArchiveApi";
import type { BackendArchiveRange } from "@/lib/backendArchiveTypes";
import type {
  History48hStore,
  HistoryFilters as HistoryFilterState,
  ReviewStatus,
  TimeWindowHours,
} from "@/lib/history48hTypes";
import {
  buildHistorySummaryText,
  buildSessionRows,
  filterHistoryEvents,
  getManualReviewQueue,
  summarizeHistory,
  updateHistoryEventReviewStatus,
} from "@/lib/history48hUtils";

const DEFAULT_FILTERS: HistoryFilterState = {
  timeWindowHours: 48,
  eventType: "all",
  review: "all",
  source: "all",
};

const EMPTY_STORE: History48hStore = {
  events: [],
  sessions: [],
  updatedAt: "",
};

const ARCHIVE_RANGE_TO_WINDOW: Record<BackendArchiveRange, TimeWindowHours> = {
  "48h": 48,
  "7d": 168,
  "30d": 720,
  all: 876000,
};

export function History48hPage() {
  const [store, setStore] = useState<History48hStore>(EMPTY_STORE);
  const [archiveStore, setArchiveStore] = useState<History48hStore | null>(null);
  const [archiveRange, setArchiveRange] = useState<BackendArchiveRange>("48h");
  const [archiveStatus, setArchiveStatus] = useState("Backend archive not checked yet.");
  const [archiveAvailable, setArchiveAvailable] = useState(false);
  const [filters, setFilters] = useState<HistoryFilterState>(DEFAULT_FILTERS);
  const [referenceNow, setReferenceNow] = useState<Date>(new Date());
  const [copyStatus, setCopyStatus] = useState("");
  const { currentUser, isLegacyRecordVisible } = useVisionGuardAuth();
  const includeLegacyRecords = currentUser
    ? isLegacyRecordVisible(undefined)
    : false;

  useEffect(() => {
    const id = window.setTimeout(() => {
      const now = new Date();
      setReferenceNow(now);
      setStore(
        loadHistory48hStore(now, currentUser?.id, includeLegacyRecords)
      );
    }, 0);

    return () => window.clearTimeout(id);
  }, [currentUser?.id, includeLegacyRecords]);

  const refreshArchive = useCallback(async () => {
    setArchiveStatus("Checking backend archive.");
    try {
      const health = await getArchiveHealth();
      if (!health.ok || !health.enabled) {
        setArchiveAvailable(false);
        setArchiveStore(null);
        setArchiveStatus("Backend archive unavailable; showing local_only history.");
        return;
      }
      const response = await getArchiveRecords({
        range: archiveRange,
        limit: 500,
      });
      const nextStore = archiveRecordsToHistoryStore(response.records);
      setArchiveStore(nextStore);
      setArchiveAvailable(response.enabled && response.records.length > 0);
      setArchiveStatus(
        response.records.length > 0
          ? `backend_archive loaded ${response.records.length} records from ${archiveRange}.`
          : "Backend archive is reachable but has no records in this range; showing local_only history.",
      );
    } catch {
      setArchiveAvailable(false);
      setArchiveStore(null);
      setArchiveStatus(
        "Backend archive unavailable. Check FastAPI, Cloudflare Tunnel, NEXT_PUBLIC_API_BASE_URL, and CORS.",
      );
    }
  }, [archiveRange]);

  useEffect(() => {
    const id = window.setTimeout(() => {
      void refreshArchive();
    }, 0);

    return () => window.clearTimeout(id);
  }, [refreshArchive]);

  const activeStore = archiveAvailable && archiveStore ? archiveStore : store;
  const activeDataSource = archiveAvailable && archiveStore ? "backend_archive" : "local_only";

  const filteredEvents = useMemo(
    () => filterHistoryEvents(activeStore.events, filters, referenceNow),
    [activeStore.events, filters, referenceNow]
  );

  const summary = useMemo(
    () => summarizeHistory(filteredEvents, activeStore.sessions),
    [activeStore.sessions, filteredEvents]
  );

  const sessionRows = useMemo(
    () => buildSessionRows(activeStore.sessions, filteredEvents),
    [activeStore.sessions, filteredEvents]
  );

  const reviewQueue = useMemo(
    () => getManualReviewQueue(filteredEvents),
    [filteredEvents]
  );

  function persistStore(nextStore: History48hStore) {
    setStore(
      saveHistory48hUserStore(
        nextStore,
        currentUser?.id,
        includeLegacyRecords
      )
    );
  }

  function handleSetReviewStatus(eventId: string, status: ReviewStatus) {
    if (activeDataSource === "backend_archive" && archiveStore) {
      const nextStore = {
        ...archiveStore,
        events: updateHistoryEventReviewStatus(archiveStore.events, eventId, status),
        updatedAt: new Date().toISOString(),
      };
      setArchiveStore(nextStore);
      void updateArchiveRecordReview(eventId, {
        reviewed: status === "reviewed",
      }).then((result) => {
        if (!result.ok) {
          setCopyStatus("Archive review update failed");
        }
      });
      return;
    }

    const nextStore = {
      ...store,
      events: updateHistoryEventReviewStatus(store.events, eventId, status),
      updatedAt: new Date().toISOString(),
    };
    persistStore(nextStore);
  }

  function handleResetDemoData() {
    const now = new Date();
    setReferenceNow(now);
    setFilters(DEFAULT_FILTERS);
    setStore(resetHistory48hDemoData(now, currentUser?.id, includeLegacyRecords));
    setCopyStatus("Demo data reset for this local user");
  }

  function handleClearHistory() {
    const now = new Date();
    setReferenceNow(now);
    setFilters(DEFAULT_FILTERS);
    setStore(clearHistory48hStore(now, currentUser?.id, includeLegacyRecords));
    setCopyStatus("History cleared for this local user");
  }

  async function handleCopySummary() {
    const text = buildHistorySummaryText(summary, filters);
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus("Summary copied");
    } catch {
      setCopyStatus("Copy unavailable in this browser");
    }
  }

  function handleViewSession(sessionId: string) {
    setFilters((current) => ({ ...current, sessionId }));
  }

  function handleClearSessionFilter() {
    setFilters((current) => ({ ...current, sessionId: undefined }));
  }

  function handleArchiveRangeChange(nextRange: BackendArchiveRange) {
    setArchiveRange(nextRange);
    setFilters((current) => ({
      ...current,
      timeWindowHours: ARCHIVE_RANGE_TO_WINDOW[nextRange],
    }));
  }

  async function handleExportArchive() {
    try {
      const payload = await exportArchiveRecords();
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const date = new Date().toISOString().slice(0, 10);
      link.href = url;
      link.download = `visionguard-archive-export-${date}.json`;
      link.click();
      URL.revokeObjectURL(url);
      setCopyStatus(`Exported ${payload.record_count} backend_archive records`);
    } catch {
      setCopyStatus("Archive export failed");
    }
  }

  return (
    <main className="flex-1 overflow-y-auto bg-[#f4f7f9] px-4 py-5 lg:px-6">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-5 pb-10">
        <HistoryHeader />
        <HistoryFilters
          filters={filters}
          selectedSessionId={filters.sessionId}
          copyStatus={copyStatus}
          archiveRange={archiveRange}
          archiveStatus={archiveStatus}
          activeDataSource={activeDataSource}
          onChange={setFilters}
          onArchiveRangeChange={handleArchiveRangeChange}
          onRefreshArchive={refreshArchive}
          onExportArchive={handleExportArchive}
          onResetDemoData={handleResetDemoData}
          onClearHistory={handleClearHistory}
          onCopySummary={handleCopySummary}
          onClearSessionFilter={handleClearSessionFilter}
        />
        <HistorySummaryCards summary={summary} />
        <ManualReviewQueue
          events={reviewQueue}
          onSetReviewStatus={handleSetReviewStatus}
        />
        <EventTimelineTable
          events={filteredEvents}
          emptyMessage={
            activeStore.events.length === 0
              ? "No warning-candidate history records are available for the selected source."
              : "No events match the current filters."
          }
        />
        <RecentSessionsSummary
          sessions={sessionRows}
          onViewEvents={handleViewSession}
        />
        <HistoryInterpretationNote />
      </div>
    </main>
  );
}
