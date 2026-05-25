import {
  AlertTriangle,
  CarFront,
  Radar,
  ShieldAlert,
} from "lucide-react";
import type { InsightSummary } from "@/lib/insightsTypes";
import { formatInsightPercent } from "@/lib/insightsUtils";

interface InsightSummaryCardsProps {
  summary: InsightSummary;
  driveCount: number;
}

export function InsightSummaryCards({
  summary,
  driveCount,
}: InsightSummaryCardsProps) {
  const cards = [
    {
      title: "Dominant Alert",
      value: summary.dominantAlertLabel,
      helper: `${summary.dominantAlertCount} of ${summary.totalAlerts} alerts`,
      icon: Radar,
      accent: "text-blue-600 dark:text-cyan-300",
      bg: "bg-blue-50 dark:bg-cyan-400/10",
    },
    {
      title: "High-Risk Share",
      value: formatInsightPercent(summary.highPriorityShare),
      helper: "Alerts with stronger fatigue-related cues",
      icon: ShieldAlert,
      accent: "text-red-600 dark:text-red-300",
      bg: "bg-red-50 dark:bg-red-400/10",
    },
    {
      title: "Signal Interruptions",
      value: formatInsightPercent(summary.signalInterruptionShare),
      helper: `${summary.signalInterruptionCount} alerts involved camera or tracking uncertainty`,
      icon: AlertTriangle,
      accent: "text-amber-600 dark:text-amber-300",
      bg: "bg-amber-50 dark:bg-amber-400/10",
    },
    {
      title: "Drives Analyzed",
      value: String(driveCount),
      helper: "Recent drives in the selected window",
      icon: CarFront,
      accent: "text-emerald-600 dark:text-emerald-300",
      bg: "bg-emerald-50 dark:bg-emerald-400/10",
    },
  ];

  return (
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {cards.map(({ accent, bg, helper, icon: Icon, title, value }) => (
        <article
          key={title}
          className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900"
        >
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-black uppercase tracking-[0.12em] text-slate-400 dark:text-slate-500">
                {title}
              </p>
              <p className="mt-2 text-xl font-black leading-tight text-slate-950 dark:text-white">
                {value}
              </p>
            </div>
            <span className={`rounded-xl p-2.5 ${bg} ${accent}`}>
              <Icon className="h-5 w-5" />
            </span>
          </div>
          <p className="mt-3 text-sm font-medium leading-5 text-slate-500 dark:text-slate-400">
            {helper}
          </p>
        </article>
      ))}
    </section>
  );
}
