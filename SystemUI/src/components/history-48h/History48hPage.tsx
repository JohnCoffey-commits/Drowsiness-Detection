"use client";

import { useEffect, useMemo, useState } from "react";
import { CandidateSeverityTrend } from "@/components/history-48h/CandidateSeverityTrend";
import { EventDistributionChart } from "@/components/history-48h/EventDistributionChart";
import { EventTimelineTable } from "@/components/history-48h/EventTimelineTable";
import { HighRiskCandidates } from "@/components/history-48h/HighRiskCandidates";
import { HistoryFilters } from "@/components/history-48h/HistoryFilters";
import { HistoryHeader } from "@/components/history-48h/HistoryHeader";
import { HistoryInterpretationNote } from "@/components/history-48h/HistoryInterpretationNote";
import { HistorySummaryCards } from "@/components/history-48h/HistorySummaryCards";
import { ManualReviewQueue } from "@/components/history-48h/ManualReviewQueue";
import { RecentSessionsSummary } from "@/components/history-48h/RecentSessionsSummary";
import { StateBreakdownChart } from "@/components/history-48h/StateBreakdownChart";
import {
  clearHistory48hStore,
  loadHistory48hStore,
  resetHistory48hDemoData,
  saveHistory48hStore,
} from "@/lib/history48hStorage";
import type {
  History48hStore,
  HistoryFilters as HistoryFilterState,
  ReviewStatus,
} from "@/lib/history48hTypes";
import {
  aggregateEventDistribution,
  aggregateSeverityTrend,
  buildHistorySummaryText,
  buildSessionRows,
  buildStateBreakdown,
  filterHistoryEvents,
  getHighRiskCandidates,
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

  useEffect(() => {
    const id = window.setTimeout(() => {
      const now = new Date();
      setReferenceNow(now);
      setStore(loadHistory48hStore(now));
    }, 0);

    return () => window.clearTimeout(id);
  }, []);

  const filteredEvents = useMemo(
    () => filterHistoryEvents(store.events, filters, referenceNow),
    [filters, referenceNow, store.events]
  );

  const summary = useMemo(
    () => summarizeHistory(filteredEvents, store.sessions),
    [filteredEvents, store.sessions]
  );

  const trendData = useMemo(
    () =>
      aggregateSeverityTrend(
        filteredEvents,
        referenceNow,
        filters.timeWindowHours
      ),
    [filteredEvents, filters.timeWindowHours, referenceNow]
  );

  const distributionData = useMemo(
    () =>
      aggregateEventDistribution(
        filteredEvents,
        referenceNow,
        filters.timeWindowHours
      ),
    [filteredEvents, filters.timeWindowHours, referenceNow]
  );

  const stateBreakdown = useMemo(
    () => buildStateBreakdown(filteredEvents),
    [filteredEvents]
  );

  const highRiskCandidates = useMemo(
    () => getHighRiskCandidates(filteredEvents),
    [filteredEvents]
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
    setStore(nextStore);
    saveHistory48hStore(nextStore);
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
    setStore(resetHistory48hDemoData(now));
    setCopyStatus("Demo data reset");
  }

  function handleClearHistory() {
    const now = new Date();
    setReferenceNow(now);
    setFilters(DEFAULT_FILTERS);
    setStore(clearHistory48hStore(now));
    setCopyStatus("History cleared");
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
        <CandidateSeverityTrend data={trendData} />
        <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
          <EventDistributionChart data={distributionData} />
          <StateBreakdownChart data={stateBreakdown} />
        </div>
        <HighRiskCandidates events={highRiskCandidates} />
        <EventTimelineTable events={filteredEvents} />
        <RecentSessionsSummary
          sessions={sessionRows}
          onViewEvents={handleViewSession}
        />
        <ManualReviewQueue
          events={reviewQueue}
          onSetReviewStatus={handleSetReviewStatus}
        />
        <HistoryInterpretationNote />
      </div>
    </main>
  );
}
