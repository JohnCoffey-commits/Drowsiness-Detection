import {
  AlertCircle,
  Clock3,
  FileWarning,
  History,
  ShieldAlert,
} from "lucide-react";
import type { HistorySummary } from "@/lib/history48hUtils";
import { formatDateTime } from "@/lib/history48hUtils";

interface HistorySummaryCardsProps {
  summary: HistorySummary;
}

export function HistorySummaryCards({ summary }: HistorySummaryCardsProps) {
  const cards = [
    {
      label: "Warning-candidate events",
      value: String(summary.warningCandidateCount),
      note: "Non-normal local history records",
      icon: History,
      accent: "text-blue-600 bg-blue-50 border-blue-100",
    },
    {
      label: "High-priority candidates",
      value: String(summary.highPriorityCount),
      note: "Critical eye or high severity records",
      icon: ShieldAlert,
      accent: "text-red-700 bg-red-50 border-red-100",
    },
    {
      label: "Signal quality issues",
      value: String(summary.signalUnreliableCount),
      note: "Camera, face, ROI, or signal uncertainty",
      icon: AlertCircle,
      accent: "text-slate-700 bg-slate-100 border-slate-200",
    },
    {
      label: "Manual review pending",
      value: String(summary.reviewPendingCount),
      note: "Local review state",
      icon: FileWarning,
      accent: "text-amber-700 bg-amber-50 border-amber-100",
    },
    {
      label: "Last event time",
      value: summary.lastEventTime ? formatDateTime(summary.lastEventTime) : "-",
      note: "Most recent filtered event",
      icon: Clock3,
      accent: "text-blue-700 bg-blue-50 border-blue-100",
    },
  ];

  return (
    <section aria-label="48h summary cards">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {cards.map(({ label, value, note, icon: Icon, accent }) => (
          <article
            key={label}
            className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {label}
                </p>
                <p className="mt-2 truncate text-2xl font-bold tracking-tight text-slate-900">
                  {value}
                </p>
              </div>
              <span
                className={`inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border ${accent}`}
              >
                <Icon className="h-5 w-5" />
              </span>
            </div>
            <p className="mt-3 text-xs font-medium text-slate-500">{note}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
