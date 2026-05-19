"use client";

import { useEffect, useMemo, useState } from "react";
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
import type {
  History48hStore,
  HistoryFilters as HistoryFilterState,
  ReviewStatus,
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

export function History48hPage() {
  const [store, setStore] = useState<History48hStore>(EMPTY_STORE);
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

  const filteredEvents = useMemo(
    () => filterHistoryEvents(store.events, filters, referenceNow),
    [filters, referenceNow, store.events]
  );

  const summary = useMemo(
    () => summarizeHistory(filteredEvents, store.sessions),
    [filteredEvents, store.sessions]
  );

  const sessionRows = useMemo(
    () => buildSessionRows(store.sessions, filteredEvents),
    [filteredEvents, store.sessions]
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

  return (
    <main className="flex-1 overflow-y-auto bg-[#f4f7f9] px-4 py-5 lg:px-6">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-5 pb-10">
        <HistoryHeader />
        <HistoryFilters
          filters={filters}
          selectedSessionId={filters.sessionId}
          copyStatus={copyStatus}
          onChange={setFilters}
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
            store.events.length === 0
              ? "No local warning-candidate history for this user in the last 48 hours."
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
