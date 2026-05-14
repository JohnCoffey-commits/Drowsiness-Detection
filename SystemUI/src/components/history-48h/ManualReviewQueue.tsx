"use client";

import { CheckCircle2, RotateCcw } from "lucide-react";
import type { DriverHistoryEvent, ReviewStatus } from "@/lib/history48hTypes";
import {
  REVIEW_LABELS,
  SEVERITY_META,
  STATE_META,
  formatDateTime,
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
        <h2 className="text-base font-bold text-slate-900">Manual review queue</h2>
        <p className="mt-1 text-sm text-slate-500">
          Local review actions update browser storage only.
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
                className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50/60 p-4 lg:grid-cols-[1.2fr_1fr_auto]"
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
                  </div>
                  <p className="mt-2 text-sm font-semibold text-slate-800">
                    {formatDateTime(event.timestamp)}
                  </p>
                </div>

                <p className="text-sm leading-6 text-slate-600">{event.reason}</p>

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
