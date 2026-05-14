import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Clock3,
  Eye,
  FileWarning,
  Gauge,
  MessageCircleWarning,
  ShieldAlert,
  Users,
} from "lucide-react";
import type { HistorySummary } from "@/lib/history48hUtils";
import {
  formatDateTime,
  formatMinutes,
  formatPercent,
} from "@/lib/history48hUtils";

interface HistorySummaryCardsProps {
  summary: HistorySummary;
}

export function HistorySummaryCards({ summary }: HistorySummaryCardsProps) {
  const cards = [
    {
      label: "Monitored sessions",
      value: String(summary.sessionCount),
      note: "Sessions in selected window",
      icon: Users,
      accent: "text-blue-600 bg-blue-50 border-blue-100",
    },
    {
      label: "Monitored time",
      value: formatMinutes(summary.monitoredTimeMin),
      note: "Demo/local session duration",
      icon: Clock3,
      accent: "text-slate-700 bg-slate-50 border-slate-200",
    },
    {
      label: "Normal state ratio",
      value: formatPercent(summary.normalRatio),
      note: `${summary.normalCount} normal periods`,
      icon: CheckCircle2,
      accent: "text-emerald-700 bg-emerald-50 border-emerald-100",
    },
    {
      label: "Eye-warning candidates",
      value: String(summary.eyeWarningCount),
      note: "Temporal eye evidence",
      icon: Eye,
      accent: "text-orange-700 bg-orange-50 border-orange-100",
    },
    {
      label: "Mouth-warning candidates",
      value: String(summary.mouthWarningCount),
      note: "Recent mouth/yawn evidence",
      icon: MessageCircleWarning,
      accent: "text-pink-700 bg-pink-50 border-pink-100",
    },
    {
      label: "High-confidence warning candidates",
      value: String(summary.highConfidenceCount),
      note: "Candidate intervals for review",
      icon: ShieldAlert,
      accent: "text-red-700 bg-red-50 border-red-100",
    },
    {
      label: "Signal-unreliable periods",
      value: String(summary.signalUnreliableCount),
      note: "Quality issue candidates",
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
      label: "Peak candidate severity",
      value: String(summary.peakCandidateSeverity),
      note: "UI-level display score",
      icon: Gauge,
      accent: "text-red-700 bg-red-50 border-red-100",
    },
    {
      label: "Last event time",
      value: summary.lastEventTime ? formatDateTime(summary.lastEventTime) : "-",
      note: "Most recent filtered event",
      icon: BarChart3,
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
