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
import {
  filterHistory48hStoreBySource,
  loadHistory48hStore,
} from "@/lib/history48hStorage";
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

function liveMonitorOnly(store: History48hStore): History48hStore {
  return filterHistory48hStoreBySource(store, "live_monitor");
}

type ArchiveConnectionState =
  | "unchecked"
  | "checking"
  | "connected"
  | "disconnected";

function formatArchiveCheckedAt(value: string | null): string {
  if (!value) return "Not checked";
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function InsightsPage() {
  const router = useRouter();
  const { currentUser, isLegacyRecordVisible } = useVisionGuardAuth();
  const [store, setStore] = useState<History48hStore>(EMPTY_STORE);
  const [archiveStore, setArchiveStore] = useState<History48hStore | null>(null);
  const [archiveDataSource, setArchiveDataSource] = useState("local_only");
  const [archiveConnection, setArchiveConnection] =
    useState<ArchiveConnectionState>("unchecked");
  const [archiveCheckedAt, setArchiveCheckedAt] = useState<string | null>(null);
  const [archiveAvailable, setArchiveAvailable] = useState(false);
  const [referenceNow, setReferenceNow] = useState<Date>(new Date());
  const includeLegacyRecords = currentUser
    ? isLegacyRecordVisible(undefined)
    : false;

  useEffect(() => {
    const id = window.setTimeout(() => {
      const now = new Date();
      setReferenceNow(now);
      setStore(
        liveMonitorOnly(
          loadHistory48hStore(now, currentUser?.id, includeLegacyRecords)
        )
      );
    }, 0);

    return () => window.clearTimeout(id);
  }, [currentUser?.id, includeLegacyRecords]);

  useEffect(() => {
    const id = window.setTimeout(async () => {
      setArchiveConnection("checking");
      try {
        const health = await getArchiveHealth();
        const checkedAt = new Date().toISOString();
        setArchiveCheckedAt(checkedAt);
        if (!health.ok) {
          setArchiveAvailable(false);
          setArchiveStore(null);
          setArchiveDataSource("local_only");
          setArchiveConnection("disconnected");
          return;
        }
        setArchiveConnection("connected");
        if (!health.enabled) {
          setArchiveAvailable(false);
          setArchiveStore(null);
          setArchiveDataSource("local_only");
          return;
        }
        const response = await getArchiveRecords({
          range: "48h",
          source: "live_monitor",
          limit: 500,
        });
        const nextStore = liveMonitorOnly(
          archiveRecordsToHistoryStore(response.records)
        );
        setArchiveStore(nextStore);
        setArchiveAvailable(nextStore.events.length > 0);
        setArchiveDataSource(
          nextStore.events.length > 0 ? "backend_archive" : "local_only"
        );
      } catch {
        setArchiveAvailable(false);
        setArchiveStore(null);
        setArchiveDataSource("local_only");
        setArchiveConnection("disconnected");
        setArchiveCheckedAt(new Date().toISOString());
      }
    }, 0);

    return () => window.clearTimeout(id);
  }, [currentUser?.id]);

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

  function handleViewInHistory(sessionId: string) {
    const params = new URLSearchParams({
      sessionId,
      timeWindowHours: "48",
    });
    router.push(`/history-48h?${params.toString()}`);
  }

  return (
    <main className="flex-1 overflow-y-auto bg-[#f4f7f9] px-4 py-5 transition-colors duration-300 dark:bg-slate-950 lg:px-6">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-5 pb-10">
        <InsightsHeader
          displayName={currentUser?.displayName}
          recordCount={records.length}
          dataSource={archiveDataSource}
          archiveConnection={archiveConnection}
          archiveCheckedAt={formatArchiveCheckedAt(archiveCheckedAt)}
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
