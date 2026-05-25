import { AlertCircle, Clock3, History, ShieldAlert } from "lucide-react";
import type { HistorySummary } from "@/lib/history48hUtils";
import { formatDateTime } from "@/lib/history48hUtils";

interface HistorySummaryCardsProps {
  summary: HistorySummary;
}

export function HistorySummaryCards({ summary }: HistorySummaryCardsProps) {
  const cards = [
    {
      label: "Total Alerts",
      value: String(summary.warningCandidateCount),
      note: "Alerts in the selected scope",
      icon: History,
      accent: "text-blue-600 bg-blue-50 border-blue-100",
    },
    {
      label: "High-Risk Alerts",
      value: String(summary.highPriorityCount),
      note: "Stronger fatigue-related cues",
      icon: ShieldAlert,
      accent: "text-red-700 bg-red-50 border-red-100",
    },
    {
      label: "Signal Interruptions",
      value: String(summary.signalUnreliableCount),
      note: "Camera, face, or signal visibility gaps",
      icon: AlertCircle,
      accent: "text-slate-700 bg-slate-100 border-slate-200",
    },
    {
      label: "Latest Alert",
      value: summary.lastEventTime ? formatDateTime(summary.lastEventTime) : "-",
      note: "Most recent alert shown",
      icon: Clock3,
      accent: "text-blue-700 bg-blue-50 border-blue-100",
    },
  ];

  return (
    <section aria-label="History summary cards">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
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
                <p
                  className={`mt-2 break-words font-bold tracking-tight text-slate-900 ${
                    label === "Latest Alert"
                      ? "text-lg leading-6"
                      : "text-2xl"
                  }`}
                >
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
