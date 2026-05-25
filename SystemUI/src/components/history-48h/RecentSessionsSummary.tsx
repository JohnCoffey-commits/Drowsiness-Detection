"use client";

import { AlertTriangle, Clock3, Gauge, Route } from "lucide-react";
import { useMemo, useState } from "react";
import type { SessionSummaryRow } from "@/lib/history48hUtils";
import {
  SEVERITY_META,
  formatMinutes,
  formatTimeRange,
} from "@/lib/history48hUtils";

interface RecentSessionsSummaryProps {
  sessions: SessionSummaryRow[];
  selectedSessionId?: string;
  totalAlertCount: number;
  onSelectSession: (sessionId?: string) => void;
}

const DEFAULT_VISIBLE_DRIVE_COUNT = 3;

export function RecentSessionsSummary({
  sessions,
  selectedSessionId,
  totalAlertCount,
  onSelectSession,
}: RecentSessionsSummaryProps) {
  const [showAllDriveCards, setShowAllDriveCards] = useState(false);
  const hasOverflowDrives = sessions.length > DEFAULT_VISIBLE_DRIVE_COUNT;
  const selectedSessionPinned =
    !showAllDriveCards &&
    hasOverflowDrives &&
    Boolean(
      selectedSessionId &&
        !sessions
          .slice(0, DEFAULT_VISIBLE_DRIVE_COUNT)
          .some((session) => session.id === selectedSessionId)
    );
  const visibleSessions = useMemo(() => {
    if (showAllDriveCards || !hasOverflowDrives) return sessions;

    const recentSessions = sessions.slice(0, DEFAULT_VISIBLE_DRIVE_COUNT);
    const selectedSession = sessions.find(
      (session) => session.id === selectedSessionId
    );
    if (
      selectedSession &&
      !recentSessions.some((session) => session.id === selectedSession.id)
    ) {
      return [...recentSessions, selectedSession];
    }

    return recentSessions;
  }, [hasOverflowDrives, selectedSessionId, sessions, showAllDriveCards]);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-base font-bold text-slate-900">Recent Drives</h2>
          <p className="mt-1 text-sm text-slate-500">
            Select a drive to narrow the alert timeline.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {hasOverflowDrives && (
            <button
              type="button"
              onClick={() => setShowAllDriveCards((current) => !current)}
              className="inline-flex h-9 items-center justify-center rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-100"
            >
              {showAllDriveCards
                ? "Show recent drives"
                : `Show all ${sessions.length} drives`}
            </button>
          )}
          {selectedSessionId && (
            <button
              type="button"
              onClick={() => onSelectSession(undefined)}
              className="inline-flex h-9 items-center justify-center rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-100"
            >
              Show all drives
            </button>
          )}
        </div>
      </div>

      <div
        className={`grid gap-2 md:grid-cols-2 xl:grid-cols-4 ${
          showAllDriveCards ? "max-h-[420px] overflow-y-auto pr-1" : ""
        }`}
      >
        <button
          type="button"
          onClick={() => onSelectSession(undefined)}
          aria-pressed={!selectedSessionId}
          className={`rounded-xl border p-3 text-left transition focus:outline-none focus:ring-4 focus:ring-blue-100 ${
            !selectedSessionId
              ? "border-blue-200 bg-blue-50 shadow-sm"
              : "border-slate-200 bg-slate-50/60 hover:bg-slate-50"
          }`}
        >
          <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
            <Route className="h-4 w-4 text-blue-700" />
            All drives
          </div>
          <p className="mt-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Selected time window
          </p>
          <p className="mt-1 text-2xl font-bold text-slate-900">
            {totalAlertCount}
          </p>
          <p className="text-xs font-medium text-slate-500">alerts</p>
        </button>

        {visibleSessions.map((session) => {
          const selected = selectedSessionId === session.id;
          return (
            <button
              type="button"
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              aria-pressed={selected}
              className={`rounded-xl border p-3 text-left transition focus:outline-none focus:ring-4 focus:ring-blue-100 ${
                selected
                  ? "border-blue-200 bg-blue-50 shadow-sm"
                  : "border-slate-200 bg-slate-50/60 hover:bg-slate-50"
              }`}
            >
              <p className="line-clamp-2 text-sm font-bold leading-5 text-slate-900">
                {formatTimeRange(session.startedAt, session.endedAt)}
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-600">
                <DriveMetric
                  icon={Clock3}
                  label="Duration"
                  value={formatMinutes(session.durationMin)}
                />
                <DriveMetric
                  icon={AlertTriangle}
                  label="Alerts"
                  value={session.warningCandidateCount}
                />
                <DriveMetric
                  icon={Gauge}
                  label="Highest"
                  value={
                    session.warningCandidateCount > 0
                      ? SEVERITY_META[session.highestSeverity].label
                      : "None"
                  }
                />
                <DriveMetric
                  icon={Route}
                  label="Signal"
                  value={session.signalUnreliableCount}
                />
              </div>
            </button>
          );
        })}
      </div>

      {!showAllDriveCards && hasOverflowDrives && (
        <p className="mt-3 text-xs font-semibold text-slate-500">
          Showing the latest{" "}
          {Math.min(DEFAULT_VISIBLE_DRIVE_COUNT, sessions.length)} of{" "}
          {sessions.length} drives.
          {selectedSessionPinned ? " The selected drive stays visible." : ""}
        </p>
      )}

      {sessions.length === 0 && (
        <div className="mt-3 rounded-xl border border-dashed border-slate-200 bg-slate-50 p-5 text-center text-sm font-medium text-slate-500">
          No drives match the current filters.
        </div>
      )}
    </section>
  );
}

function DriveMetric({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Clock3;
  label: string;
  value: number | string;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-slate-200 bg-white px-2.5 py-2">
      <div className="flex items-start gap-1.5 text-[10px] leading-tight text-slate-400">
        <Icon className="h-3.5 w-3.5 shrink-0" />
        <span className="min-w-0 font-semibold uppercase tracking-wide">
          {label}
        </span>
      </div>
      <p className="mt-1 truncate font-bold text-slate-800">{value}</p>
    </div>
  );
}
