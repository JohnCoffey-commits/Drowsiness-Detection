"use client";

import { ArrowRight, Clock3, Database } from "lucide-react";
import type { SessionSummaryRow } from "@/lib/history48hUtils";
import {
  SOURCE_LABELS,
  formatMinutes,
  formatTimeRange,
} from "@/lib/history48hUtils";

interface RecentSessionsSummaryProps {
  sessions: SessionSummaryRow[];
  onViewEvents: (sessionId: string) => void;
}

export function RecentSessionsSummary({
  sessions,
  onViewEvents,
}: RecentSessionsSummaryProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-base font-bold text-slate-900">Recent sessions</h2>
        <p className="mt-1 text-sm text-slate-500">
          Session-level counts derived from filtered local history events.
        </p>
      </div>

      {sessions.length === 0 ? (
        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm font-medium text-slate-500">
          No recent sessions match the current filters.
        </div>
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {sessions.map((session) => (
            <article
              key={session.id}
              className="rounded-xl border border-slate-200 bg-slate-50/60 p-4"
            >
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="min-w-0">
                  <p className="truncate text-sm font-bold text-slate-900">
                    {session.id}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className="inline-flex items-center gap-1 rounded-full border border-blue-100 bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
                      <Database className="h-3.5 w-3.5" />
                      {SOURCE_LABELS[session.source]}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-600">
                      <Clock3 className="h-3.5 w-3.5" />
                      {formatMinutes(session.durationMin)}
                    </span>
                    <span className="inline-flex rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold capitalize text-slate-600">
                      {session.status}
                    </span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => onViewEvents(session.id)}
                  className="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-lg bg-blue-600 px-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-100"
                >
                  View events
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>

              <p className="mt-3 text-sm text-slate-600">
                {formatTimeRange(session.startedAt, session.endedAt)}
              </p>

              <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                <Metric label="Warning candidates" value={session.warningCandidateCount} />
                <Metric label="Review pending" value={session.reviewPendingCount} />
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p className="mt-1 text-xl font-bold text-slate-900">{value}</p>
    </div>
  );
}
