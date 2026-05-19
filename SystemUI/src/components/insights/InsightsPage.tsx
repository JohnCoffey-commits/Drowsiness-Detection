"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { EventCompositionChart } from "@/components/insights/EventCompositionChart";
import { InsightSummaryCards } from "@/components/insights/InsightSummaryCards";
import { InsightsEmptyState } from "@/components/insights/InsightsEmptyState";
import { InsightsHeader } from "@/components/insights/InsightsHeader";
import { ReviewRecommendations } from "@/components/insights/ReviewRecommendations";
import { SessionComparisonTable } from "@/components/insights/SessionComparisonTable";
import { SignalQualityInsights } from "@/components/insights/SignalQualityInsights";
import { TimeOfDayPattern } from "@/components/insights/TimeOfDayPattern";
import { WarningCandidateTrend } from "@/components/insights/WarningCandidateTrend";
import { useVisionGuardAuth } from "@/lib/authStore";
import {
  archiveRecordsToHistoryStore,
  getArchiveHealth,
  getArchiveRecords,
} from "@/lib/backendArchiveApi";
import { loadHistory48hStore } from "@/lib/history48hStorage";
import type { History48hStore } from "@/lib/history48hTypes";
import {
  describeTimeOfDayPattern,
  getEventTypeDistribution,
  getInsightRecords,
  getInsightSummary,
  getReviewRecommendations,
  getSessionComparison,
  getSignalQualityInsights,
  getTimeOfDayPattern,
  getWarningCandidateTrend,
} from "@/lib/insightsUtils";

const EMPTY_STORE: History48hStore = {
  events: [],
  sessions: [],
  updatedAt: "",
};

export function InsightsPage() {
  const router = useRouter();
  const { currentUser, isLegacyRecordVisible } = useVisionGuardAuth();
  const [store, setStore] = useState<History48hStore>(EMPTY_STORE);
  const [archiveStore, setArchiveStore] = useState<History48hStore | null>(null);
  const [archiveStatus, setArchiveStatus] = useState("Backend archive not checked yet.");
  const [archiveAvailable, setArchiveAvailable] = useState(false);
  const [referenceNow, setReferenceNow] = useState<Date>(new Date());
  const includeLegacyRecords = currentUser
    ? isLegacyRecordVisible(undefined)
    : false;

  useEffect(() => {
    const id = window.setTimeout(() => {
      const now = new Date();
      setReferenceNow(now);
      setStore(loadHistory48hStore(now, currentUser?.id, includeLegacyRecords));
    }, 0);

    return () => window.clearTimeout(id);
  }, [currentUser?.id, includeLegacyRecords]);

  useEffect(() => {
    const id = window.setTimeout(async () => {
      try {
        const health = await getArchiveHealth();
        if (!health.ok || !health.enabled) {
          setArchiveAvailable(false);
          setArchiveStore(null);
          setArchiveStatus("local_only");
          return;
        }
        const response = await getArchiveRecords({ range: "48h", limit: 500 });
        setArchiveStore(archiveRecordsToHistoryStore(response.records));
        setArchiveAvailable(response.records.length > 0);
        setArchiveStatus(response.records.length > 0 ? "backend_archive" : "local_only");
      } catch {
        setArchiveAvailable(false);
        setArchiveStore(null);
        setArchiveStatus("local_only");
      }
    }, 0);

    return () => window.clearTimeout(id);
  }, []);

  const activeStore = archiveAvailable && archiveStore ? archiveStore : store;

  const records = useMemo(
    () => getInsightRecords(activeStore.events, referenceNow, 48),
    [activeStore.events, referenceNow]
  );
  const summary = useMemo(() => getInsightSummary(records), [records]);
  const trend = useMemo(
    () => getWarningCandidateTrend(records, referenceNow, 48),
    [records, referenceNow]
  );
  const composition = useMemo(
    () => getEventTypeDistribution(records),
    [records]
  );
  const timeOfDay = useMemo(() => getTimeOfDayPattern(records), [records]);
  const timeOfDayDescription = useMemo(
    () => describeTimeOfDayPattern(timeOfDay),
    [timeOfDay]
  );
  const sessionRows = useMemo(
    () => getSessionComparison(records, activeStore.sessions),
    [activeStore.sessions, records]
  );
  const signalQuality = useMemo(
    () => getSignalQualityInsights(records),
    [records]
  );
  const recommendations = useMemo(
    () => getReviewRecommendations(records),
    [records]
  );

  function handleViewInHistory() {
    // TODO: add query-based session filtering when /history-48h supports URL filters.
    router.push("/history-48h");
  }

  return (
    <main className="flex-1 overflow-y-auto bg-[#f4f7f9] px-4 py-5 transition-colors duration-300 dark:bg-slate-950 lg:px-6">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-5 pb-10">
        <InsightsHeader
          displayName={currentUser?.displayName}
          recordCount={records.length}
          dataSource={archiveStatus}
        />

        {records.length === 0 ? (
          <InsightsEmptyState />
        ) : (
          <>
            <InsightSummaryCards summary={summary} />
            <WarningCandidateTrend data={trend} />
            <div className="grid gap-5 xl:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
              <EventCompositionChart data={composition} />
              <TimeOfDayPattern
                data={timeOfDay}
                description={timeOfDayDescription}
              />
            </div>
            <div className="grid gap-5 xl:grid-cols-[minmax(0,1.08fr)_minmax(0,0.92fr)]">
              <SessionComparisonTable
                rows={sessionRows}
                onViewInHistory={handleViewInHistory}
              />
              <div className="flex flex-col gap-5">
                <SignalQualityInsights summary={signalQuality} />
                <ReviewRecommendations recommendations={recommendations} />
              </div>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
