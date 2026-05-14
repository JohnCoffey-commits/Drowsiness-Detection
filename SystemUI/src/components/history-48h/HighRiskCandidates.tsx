import { AlertTriangle, ShieldAlert } from "lucide-react";
import type { DriverHistoryEvent } from "@/lib/history48hTypes";
import {
  REVIEW_LABELS,
  SEVERITY_META,
  STATE_META,
  evidenceLabel,
  formatCandidateScore,
  formatDateTime,
  formatDuration,
  formatProbability,
} from "@/lib/history48hUtils";

interface HighRiskCandidatesProps {
  events: DriverHistoryEvent[];
}

export function HighRiskCandidates({ events }: HighRiskCandidatesProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-slate-900">
            High-risk warning candidates
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Highest-priority candidate intervals for manual review.
          </p>
        </div>
        <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-red-100 bg-red-50 text-red-700">
          <ShieldAlert className="h-5 w-5" />
        </span>
      </div>

      {events.length === 0 ? (
        <div className="flex items-center gap-3 rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm font-medium text-slate-500">
          <AlertTriangle className="h-5 w-5 text-slate-400" />
          No high-confidence warning candidates in the selected window.
        </div>
      ) : (
        <div className="grid gap-3 xl:grid-cols-2">
          {events.map((event) => {
            const stateMeta = STATE_META[event.state];
            const severityMeta = SEVERITY_META[event.severity];
            return (
              <article
                key={event.id}
                className="rounded-xl border border-slate-200 bg-slate-50/70 p-4"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${stateMeta.bgClass} ${stateMeta.textClass}`}
                  >
                    {stateMeta.label}
                  </span>
                  <span
                    className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${severityMeta.className}`}
                  >
                    {severityMeta.label}
                  </span>
                  <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-600">
                    {REVIEW_LABELS[event.reviewStatus]}
                  </span>
                </div>
                <div className="mt-3 grid gap-2 text-sm text-slate-600 sm:grid-cols-2">
                  <div>
                    <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Time
                    </span>
                    {formatDateTime(event.timestamp)}
                  </div>
                  <div>
                    <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Duration
                    </span>
                    {formatDuration(event.durationSec)}
                  </div>
                  <div>
                    <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                      p_eye_closed max
                    </span>
                    {formatProbability(event.pEyeClosedMax)}
                  </div>
                  <div>
                    <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                      p_yawn max
                    </span>
                    {formatProbability(event.pYawnMax)}
                  </div>
                  <div>
                    <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Candidate severity score
                    </span>
                    {formatCandidateScore(event.candidateSeverityScore)}
                  </div>
                  <div>
                    <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Evidence
                    </span>
                    {evidenceLabel(event)}
                  </div>
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-700">
                  {event.reason}
                </p>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
