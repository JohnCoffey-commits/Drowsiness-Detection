"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import type { DriverHistoryEvent } from "@/lib/history48hTypes";
import {
  REVIEW_LABELS,
  SEVERITY_META,
  SOURCE_LABELS,
  STATE_META,
  evidenceLabel,
  formatCandidateScore,
  formatDateTime,
  formatDuration,
  formatProbability,
} from "@/lib/history48hUtils";

interface EventTimelineTableProps {
  events: DriverHistoryEvent[];
  emptyMessage?: string;
}

function DetailPanel({ event }: { event: DriverHistoryEvent }) {
  const archiveSource = event.archiveSource ?? "local_only";
  return (
    <div className="rounded-xl border border-blue-100 bg-blue-50/60 p-4 text-sm text-slate-700">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[repeat(7,minmax(0,1fr))]">
        <div>
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Session id
          </span>
          <span className="font-semibold text-slate-800">{event.sessionId}</span>
        </div>
        <div>
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Source event id
          </span>
          <span className="font-semibold text-slate-800">
            {event.sourceEventId ?? "-"}
          </span>
        </div>
        <div>
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            p_eye_closed max
          </span>
          {formatProbability(event.pEyeClosedMax)}
        </div>
        <div>
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            p_yawn max
          </span>
          {formatProbability(event.pYawnMax)}
        </div>
        <div>
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Eye evidence strength
          </span>
          {event.eyeEvidenceStrength ?? "unknown"}
        </div>
        <div>
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Candidate severity score
          </span>
          {formatCandidateScore(event.candidateSeverityScore)}
        </div>
        <div>
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Archive source
          </span>
          {archiveSource}
        </div>
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-[1fr_1fr]">
        <p className="leading-6">
          <span className="font-semibold text-slate-900">Reason: </span>
          {event.reason}
        </p>
        <p className="leading-6 text-slate-600">
          Safe interpretation: this is a frontend-local warning-candidate history
          item, not final system-level drowsiness accuracy.
        </p>
      </div>
    </div>
  );
}

function SourceLabel({ event }: { event: DriverHistoryEvent }) {
  return (
    <div className="leading-5">
      <div>{SOURCE_LABELS[event.source]}</div>
      <div className="text-[11px] font-semibold text-slate-400">
        {event.archiveSource ?? "local_only"}
      </div>
    </div>
  );
}

function StatePill({ event }: { event: DriverHistoryEvent }) {
  const stateMeta = STATE_META[event.state];
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${stateMeta.bgClass} ${stateMeta.textClass}`}
    >
      {stateMeta.label}
    </span>
  );
}

function SeverityPill({ event }: { event: DriverHistoryEvent }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${SEVERITY_META[event.severity].className}`}
    >
      {SEVERITY_META[event.severity].label}
    </span>
  );
}

export function EventTimelineTable({
  events,
  emptyMessage = "No events match the current filters.",
}: EventTimelineTableProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-base font-bold text-slate-900">Event timeline</h2>
        <p className="mt-1 text-sm text-slate-500">
          Compact warning-candidate history rows. Open details for model evidence
          and safe interpretation notes.
        </p>
      </div>

      {events.length === 0 ? (
        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm font-medium text-slate-500">
          {emptyMessage}
        </div>
      ) : (
        <>
          <div className="hidden xl:block">
            <table className="w-full table-fixed border-separate border-spacing-0 text-sm">
              <thead>
                <tr className="text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  <th className="w-[15%] rounded-l-lg bg-slate-50 px-3 py-3">
                    Time
                  </th>
                  <th className="w-[9%] bg-slate-50 px-3 py-3">Duration</th>
                  <th className="w-[18%] bg-slate-50 px-3 py-3">State</th>
                  <th className="w-[10%] bg-slate-50 px-3 py-3">Severity</th>
                  <th className="w-[12%] bg-slate-50 px-3 py-3">Evidence</th>
                  <th className="w-[11%] bg-slate-50 px-3 py-3">Source</th>
                  <th className="w-[12%] bg-slate-50 px-3 py-3">Review</th>
                  <th className="w-[13%] rounded-r-lg bg-slate-50 px-3 py-3 text-right">
                    Details
                  </th>
                </tr>
              </thead>
              <tbody>
                {events.map((event) => {
                  const isExpanded = expandedId === event.id;
                  return (
                    <tr key={event.id} className="align-top">
                      <td colSpan={8} className="pt-2">
                        <div className="rounded-xl border border-slate-200 bg-white">
                          <div className="grid grid-cols-[15%_9%_18%_10%_12%_11%_12%_13%] items-center gap-0 px-3 py-3">
                            <div className="pr-3 font-semibold text-slate-800">
                              {formatDateTime(event.timestamp)}
                            </div>
                            <div className="pr-3 text-slate-600">
                              {formatDuration(event.durationSec)}
                            </div>
                            <div className="pr-3">
                              <StatePill event={event} />
                            </div>
                            <div className="pr-3">
                              <SeverityPill event={event} />
                            </div>
                            <div className="pr-3 font-medium text-slate-700">
                              {evidenceLabel(event)}
                            </div>
                            <div className="pr-3 text-slate-600">
                              <SourceLabel event={event} />
                            </div>
                            <div className="pr-3 text-slate-600">
                              {REVIEW_LABELS[event.reviewStatus]}
                            </div>
                            <div className="text-right">
                              <button
                                type="button"
                                onClick={() =>
                                  setExpandedId(isExpanded ? null : event.id)
                                }
                                aria-expanded={isExpanded}
                                className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 focus:outline-none focus:ring-4 focus:ring-blue-100"
                              >
                                {isExpanded ? (
                                  <ChevronUp className="h-3.5 w-3.5" />
                                ) : (
                                  <ChevronDown className="h-3.5 w-3.5" />
                                )}
                                {isExpanded ? "Hide" : "Details"}
                              </button>
                            </div>
                          </div>
                          {isExpanded && (
                            <div className="border-t border-slate-100 p-3">
                              <DetailPanel event={event} />
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="grid gap-3 xl:hidden">
            {events.map((event) => {
              const isExpanded = expandedId === event.id;
              return (
                <article
                  key={event.id}
                  className="rounded-xl border border-slate-200 bg-white p-4"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <StatePill event={event} />
                    <SeverityPill event={event} />
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
                        Evidence
                      </span>
                      {evidenceLabel(event)}
                    </div>
                    <div>
                      <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                        Source
                      </span>
                      <SourceLabel event={event} />
                    </div>
                    <div>
                      <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                        Review
                      </span>
                      {REVIEW_LABELS[event.reviewStatus]}
                    </div>
                    <div>
                      <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
                        Candidate severity score
                      </span>
                      {formatCandidateScore(event.candidateSeverityScore)}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setExpandedId(isExpanded ? null : event.id)}
                    aria-expanded={isExpanded}
                    className="mt-3 inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 focus:outline-none focus:ring-4 focus:ring-blue-100"
                  >
                    {isExpanded ? (
                      <ChevronUp className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5" />
                    )}
                    {isExpanded ? "Hide" : "Details"}
                  </button>
                  {isExpanded && <div className="mt-3"><DetailPanel event={event} /></div>}
                </article>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
