"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { AboutInsightsNote } from "@/components/insights/AboutInsightsNote";
import { AlertsByDriveChart } from "@/components/insights/AlertsByDriveChart";
import { EventCompositionChart } from "@/components/insights/EventCompositionChart";
import { InsightSummaryCards } from "@/components/insights/InsightSummaryCards";
import { InsightsEmptyState } from "@/components/insights/InsightsEmptyState";
import { InsightsHeader } from "@/components/insights/InsightsHeader";
import { KeyInsights } from "@/components/insights/KeyInsights";
import { AttentionAreas } from "@/components/insights/AttentionAreas";
import { SessionComparisonTable } from "@/components/insights/SessionComparisonTable";
import { SignalQualityInsights } from "@/components/insights/SignalQualityInsights";
import { TimeOfDayPattern } from "@/components/insights/TimeOfDayPattern";
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
  getAttentionAreas,
  getEventTypeDistribution,
  getKeyInsights,
  getInsightRecords,
  getInsightSummary,
  getSessionComparison,
  getSignalQualityInsights,
  getTimeOfDayPattern,
} from "@/lib/insightsUtils";
import {
  buildInsightsReportHtml,
  downloadTextFile,
  insightsReportFilename,
  type InsightsReportPayload,
} from "@/lib/insightsExportUtils";

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

export function InsightsPage() {
  const router = useRouter();
  const { currentUser, isLegacyRecordVisible } = useVisionGuardAuth();
  const [store, setStore] = useState<History48hStore>(EMPTY_STORE);
  const [archiveStore, setArchiveStore] = useState<History48hStore | null>(null);
  const [archiveDataSource, setArchiveDataSource] = useState("local_only");
  const [archiveConnection, setArchiveConnection] =
    useState<ArchiveConnectionState>("unchecked");
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
    () => getSignalQualityInsights(records, sessionRows),
    [records, sessionRows]
  );
  const keyInsights = useMemo(
    () =>
      getKeyInsights({
        records,
        summary,
        timeOfDay,
        sessionRows,
        signalQuality,
      }),
    [records, sessionRows, signalQuality, summary, timeOfDay]
  );
  const attentionAreas = useMemo(
    () => getAttentionAreas(records, sessionRows),
    [records, sessionRows]
  );

  const dataSourceLabel =
    archiveDataSource === "backend_archive"
      ? "Data synced"
      : archiveConnection === "checking"
        ? "Checking sync"
        : "Local history";
  const userLabel = currentUser?.displayName ?? "Local user";
  const dataBasisLabel = `${records.length} ${
    records.length === 1 ? "alert" : "alerts"
  } across ${sessionRows.length} ${
    sessionRows.length === 1 ? "recent drive" : "recent drives"
  }`;

  function handleViewInHistory(sessionId?: string) {
    if (!sessionId) {
      router.push("/history-48h");
      return;
    }

    const params = new URLSearchParams({
      sessionId,
      timeWindowHours: "48",
    });
    router.push(`/history-48h?${params.toString()}`);
  }

  function handleDownloadReport() {
    if (records.length === 0) return;

    const payload: InsightsReportPayload = {
      exportedAt: new Date().toISOString(),
      timeWindowLabel: "Last 48 hours",
      userLabel,
      dataSourceLabel,
      dataBasisLabel,
      keyInsights,
      summary,
      driveRows: sessionRows,
      composition,
      timeOfDay,
      signalQuality,
      attentionAreas,
    };
    downloadTextFile(
      insightsReportFilename(),
      buildInsightsReportHtml(payload),
      "text/html;charset=utf-8"
    );
  }

  return (
    <main className="flex-1 overflow-y-auto bg-[#f4f7f9] px-4 py-5 transition-colors duration-300 dark:bg-slate-950 lg:px-6">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-5 pb-10">
        <InsightsHeader
          displayName={userLabel}
          recordCount={records.length}
          dataSourceLabel={dataSourceLabel}
          onDownloadReport={handleDownloadReport}
        />

        {records.length === 0 ? (
          <InsightsEmptyState />
        ) : (
          <>
            <KeyInsights insights={keyInsights} />
            <InsightSummaryCards
              summary={summary}
              driveCount={sessionRows.length}
            />
            <AlertsByDriveChart rows={sessionRows} />
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
                <AttentionAreas areas={attentionAreas} />
              </div>
            </div>
            <AboutInsightsNote />
          </>
        )}
      </div>
    </main>
  );
}
