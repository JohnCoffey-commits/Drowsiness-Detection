"use client";

import { CheckCircle2, RotateCcw } from "lucide-react";
import type { DriverHistoryEvent, ReviewStatus } from "@/lib/history48hTypes";
import {
  REVIEW_LABELS,
  SEVERITY_META,
  SOURCE_LABELS,
  STATE_META,
  evidenceLabel,
  formatDateTime,
  formatDuration,
} from "@/lib/history48hUtils";

interface ManualReviewQueueProps {
  events: DriverHistoryEvent[];
  onSetReviewStatus: (eventId: string, status: ReviewStatus) => void;
}

export function ManualReviewQueue({
  events,
  onSetReviewStatus,
}: ManualReviewQueueProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-base font-bold text-slate-900">Review Queue</h2>
        <p className="mt-1 text-sm text-slate-500">
          Prioritized local warning-candidate records for manual review.
        </p>
      </div>

      {events.length === 0 ? (
        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm font-medium text-slate-500">
          No manual review items in the selected window.
        </div>
      ) : (
        <div className="grid gap-3">
          {events.map((event) => {
            const stateMeta = STATE_META[event.state];
            const severityMeta = SEVERITY_META[event.severity];
            return (
              <article
                key={event.id}
                className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50/60 p-4 lg:grid-cols-[1.1fr_1fr_auto]"
              >
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${stateMeta.bgClass} ${stateMeta.textClass}`}
                    >
                      {stateMeta.label}
                    </span>
                    <span
                      className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${severityMeta.className}`}
                    >
                      {severityMeta.label}
                    </span>
                    <span className="inline-flex rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-600">
                      {REVIEW_LABELS[event.reviewStatus]}
                    </span>
                    <span className="inline-flex rounded-full border border-blue-100 bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
                      {SOURCE_LABELS[event.source]}
                    </span>
                    <span className="inline-flex rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-500">
                      {event.archiveSource ?? "local_only"}
                    </span>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-sm font-semibold text-slate-800">
                    <span>{formatDateTime(event.timestamp)}</span>
                    <span className="text-slate-500">
                      {formatDuration(event.durationSec)}
                    </span>
                  </div>
                </div>

                <div className="text-sm leading-6 text-slate-600">
                  <p className="font-semibold text-slate-800">
                    {evidenceLabel(event)}
                  </p>
                  <p>{event.reason}</p>
                </div>

                <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                  <button
                    type="button"
                    onClick={() => onSetReviewStatus(event.id, "reviewed")}
                    className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-emerald-100 bg-emerald-50 px-3 text-sm font-semibold text-emerald-700 transition hover:bg-emerald-100 focus:outline-none focus:ring-4 focus:ring-emerald-100"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    Mark reviewed
                  </button>
                  <button
                    type="button"
                    onClick={() => onSetReviewStatus(event.id, "pending")}
                    className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 focus:outline-none focus:ring-4 focus:ring-slate-100"
                  >
                    <RotateCcw className="h-4 w-4" />
                    Reset review state
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
