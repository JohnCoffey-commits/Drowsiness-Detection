import { RadioTower } from "lucide-react";
import type { InsightSignalQualitySummary } from "@/lib/insightsTypes";
import { formatInsightPercent } from "@/lib/insightsUtils";

interface SignalQualityInsightsProps {
  summary: InsightSignalQualitySummary;
}

export function SignalQualityInsights({ summary }: SignalQualityInsightsProps) {
  const mostLimitedSession =
    summary.mostLimitedSessionId && summary.mostLimitedSessionCount > 0
      ? `${summary.mostLimitedSessionId} (${summary.mostLimitedSessionCount})`
      : "None";

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-5">
      <div className="mb-4 flex items-start gap-3">
        <span className="rounded-xl bg-amber-50 p-2.5 text-amber-600 dark:bg-amber-400/10 dark:text-amber-300">
          <RadioTower className="h-5 w-5" />
        </span>
        <div>
          <h2 className="text-base font-black text-slate-950 dark:text-white">
            Signal quality insights
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
            Camera, face visibility, ROI, or signal uncertainty burden.
          </p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-950">
          <p className="text-xs font-black uppercase tracking-[0.12em] text-slate-400">
            Issue count
          </p>
          <p className="mt-2 text-2xl font-black text-slate-950 dark:text-white">
            {summary.count}
          </p>
        </div>
        <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-950">
          <p className="text-xs font-black uppercase tracking-[0.12em] text-slate-400">
            Share
          </p>
          <p className="mt-2 text-2xl font-black text-slate-950 dark:text-white">
            {formatInsightPercent(summary.share)}
          </p>
        </div>
        <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-950">
          <p className="text-xs font-black uppercase tracking-[0.12em] text-slate-400">
            Affected sessions
          </p>
          <p className="mt-2 text-2xl font-black text-slate-950 dark:text-white">
            {summary.affectedSessionCount}
          </p>
        </div>
        <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-950">
          <p className="text-xs font-black uppercase tracking-[0.12em] text-slate-400">
            Most limited session
          </p>
          <p className="mt-2 break-all text-sm font-black text-slate-950 dark:text-white">
            {mostLimitedSession}
          </p>
        </div>
      </div>

      <p className="mt-4 rounded-xl bg-amber-50 px-4 py-3 text-sm font-semibold leading-6 text-amber-900 dark:bg-amber-400/10 dark:text-amber-100">
        Signal quality issues indicate camera, face visibility, ROI, or signal
        uncertainty. Review signal-quality-heavy sessions before interpreting
        warning-candidate patterns.
      </p>
    </section>
  );
}
