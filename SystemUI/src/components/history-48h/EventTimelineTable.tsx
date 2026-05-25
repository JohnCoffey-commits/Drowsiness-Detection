"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { useMemo, useState } from "react";
import { PaginationControls } from "@/components/history-48h/PaginationControls";
import type { DriverHistoryEvent } from "@/lib/history48hTypes";
import {
  SEVERITY_META,
  STATE_META,
  evidenceLabel,
  formatCandidateScore,
  formatDateTime,
  formatDuration,
  formatProbability,
} from "@/lib/history48hUtils";

interface EventTimelineTableProps {
  events: DriverHistoryEvent[];
  totalCount: number;
  scopeText: string;
  selectedSessionId?: string;
  driveLabels: Record<string, string>;
  page: number;
  pageSize: number;
  emptyMessage?: string;
  onPageChange: (page: number) => void;
  onShowAllDrives: () => void;
}

function LongDetailText({ text }: { text: string }) {
  const [page, setPage] = useState(1);
  const pageSize = 420;
  const chunks = useMemo(() => {
    if (text.length <= pageSize) return [text];
    const result: string[] = [];
    for (let index = 0; index < text.length; index += pageSize) {
      result.push(text.slice(index, index + pageSize));
    }
    return result;
  }, [text]);
  const pageCount = chunks.length;
  const safePage = Math.min(page, pageCount);

  return (
    <div>
      <p className="leading-6">{chunks[safePage - 1]}</p>
      {pageCount > 1 ? (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-600">
          <span>
            Detail page {safePage} of {pageCount}
          </span>
          <button
            type="button"
            onClick={() => setPage((current) => Math.max(1, current - 1))}
            disabled={safePage === 1}
            className="rounded-lg border border-slate-200 bg-white px-2 py-1 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Previous
          </button>
          <button
            type="button"
            onClick={() => setPage((current) => Math.min(pageCount, current + 1))}
            disabled={safePage === pageCount}
            className="rounded-lg border border-slate-200 bg-white px-2 py-1 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Next
          </button>
        </div>
      ) : null}
    </div>
  );
}

function DetailPanel({
  event,
  driveLabel,
}: {
  event: DriverHistoryEvent;
  driveLabel: string;
}) {
  return (
    <div className="rounded-xl border border-blue-100 bg-blue-50/60 p-4 text-sm text-slate-700">
      <div className="grid gap-4 lg:grid-cols-3">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            What happened
          </h3>
          <div className="mt-1 text-slate-800">
            <LongDetailText text={event.summary || event.reason} />
          </div>
        </div>
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Evidence
          </h3>
          <p className="mt-1 font-semibold text-slate-800">
            {evidenceLabel(event)}
          </p>
          <p className="mt-1 text-slate-600">
            Severity: {SEVERITY_META[event.severity].label}
          </p>
        </div>
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Drive
          </h3>
          <p className="mt-1 font-semibold text-slate-800">{driveLabel}</p>
          <p className="mt-1 text-slate-600">
            Duration: {formatDuration(event.durationSec)}
          </p>
        </div>
      </div>

      <details className="mt-4 rounded-lg border border-blue-100 bg-white/80 px-3 py-2">
        <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500">
          Technical details
        </summary>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <TechnicalItem label="Session ID" value={event.sessionId} />
          <TechnicalItem
            label="Record reference"
            value={event.sourceEventId ?? "-"}
          />
          <TechnicalItem
            label="Eye closed score"
            value={formatProbability(event.pEyeClosedMax)}
          />
          <TechnicalItem
            label="Yawn score"
            value={formatProbability(event.pYawnMax)}
          />
          <TechnicalItem
            label="Eye evidence"
            value={event.eyeEvidenceStrength ?? "unknown"}
          />
          <TechnicalItem
            label="Priority score"
            value={formatCandidateScore(event.candidateSeverityScore)}
          />
        </div>
      </details>
    </div>
  );
}

function TechnicalItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </span>
      <span className="break-words font-semibold text-slate-800">{value}</span>
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

function signalLabel(event: DriverHistoryEvent): string {
  if (event.state === "signal_unreliable") return "Interrupted";
  if (
    event.eyeEvidenceStrength === "weak" ||
    event.eyeEvidenceStrength === "unknown"
  ) {
    return "Limited";
  }
  return "Available";
}

export function EventTimelineTable({
  events,
  totalCount,
  scopeText,
  selectedSessionId,
  driveLabels,
  page,
  pageSize,
  emptyMessage = "No matching alerts.",
  onPageChange,
  onShowAllDrives,
}: EventTimelineTableProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4 flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <h2 className="text-base font-bold text-slate-900">
            Alert Timeline
          </h2>
          <p className="mt-1 text-sm text-slate-500">{scopeText}</p>
        </div>
        {selectedSessionId && (
          <button
            type="button"
            onClick={onShowAllDrives}
            className="inline-flex h-9 items-center justify-center rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-100"
          >
            Show all drives
          </button>
        )}
      </div>

      {totalCount === 0 ? (
        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm font-medium text-slate-500">
          {emptyMessage}
        </div>
      ) : (
        <>
          <div className="hidden xl:block">
            <table className="w-full table-fixed border-separate border-spacing-0 text-sm">
              <thead>
                <tr className="text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  <th className="w-[14%] rounded-l-lg bg-slate-50 px-3 py-3">
                    Time
                  </th>
                  <th className="w-[18%] bg-slate-50 px-3 py-3">Alert</th>
                  <th className="w-[10%] bg-slate-50 px-3 py-3">Severity</th>
                  <th className="w-[10%] bg-slate-50 px-3 py-3">Duration</th>
                  <th className="w-[17%] bg-slate-50 px-3 py-3">Evidence</th>
                  <th className="w-[13%] bg-slate-50 px-3 py-3">Signal</th>
                  <th className="w-[18%] rounded-r-lg bg-slate-50 px-3 py-3 text-right">
                    Details
                  </th>
                </tr>
              </thead>
              <tbody>
                {events.map((event) => {
                  const isExpanded = expandedId === event.id;
                  return (
                    <tr key={event.id} className="align-top">
                      <td colSpan={7} className="pt-2">
                        <div className="rounded-xl border border-slate-200 bg-white">
                          <div className="grid grid-cols-[14%_18%_10%_10%_17%_13%_18%] items-center gap-0 px-3 py-3">
                            <div className="pr-3 font-semibold text-slate-800">
                              {formatDateTime(event.timestamp)}
                            </div>
                            <div className="pr-3">
                              <StatePill event={event} />
                            </div>
                            <div className="pr-3">
                              <SeverityPill event={event} />
                            </div>
                            <div className="pr-3 text-slate-600">
                              {formatDuration(event.durationSec)}
                            </div>
                            <div className="pr-3 font-medium text-slate-700">
                              {evidenceLabel(event)}
                            </div>
                            <div className="pr-3 text-slate-600">
                              {signalLabel(event)}
                            </div>
                            <div className="flex justify-end">
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
                              <DetailPanel
                                event={event}
                                driveLabel={
                                  driveLabels[event.sessionId] ?? "Live Monitor drive"
                                }
                              />
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
                    <TimelineField label="Time" value={formatDateTime(event.timestamp)} />
                    <TimelineField
                      label="Duration"
                      value={formatDuration(event.durationSec)}
                    />
                    <TimelineField label="Evidence" value={evidenceLabel(event)} />
                    <TimelineField label="Signal" value={signalLabel(event)} />
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
                  {isExpanded && (
                    <div className="mt-3">
                      <DetailPanel
                        event={event}
                        driveLabel={
                          driveLabels[event.sessionId] ?? "Live Monitor drive"
                        }
                      />
                    </div>
                  )}
                </article>
              );
            })}
          </div>
          <PaginationControls
            label="alerts"
            page={page}
            pageSize={pageSize}
            totalItems={totalCount}
            onPageChange={onPageChange}
          />
        </>
      )}
    </section>
  );
}

function TimelineField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </span>
      {value}
    </div>
  );
}
