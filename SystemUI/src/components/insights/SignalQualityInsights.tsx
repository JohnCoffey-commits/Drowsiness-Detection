import { RadioTower } from "lucide-react";
import type { InsightSignalQualitySummary } from "@/lib/insightsTypes";
import { formatInsightPercent } from "@/lib/insightsUtils";

interface SignalQualityInsightsProps {
  summary: InsightSignalQualitySummary;
}

export function SignalQualityInsights({ summary }: SignalQualityInsightsProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-4">
      <div className="mb-3 flex items-start gap-3">
        <span className="rounded-lg bg-amber-50 p-2 text-amber-600 dark:bg-amber-400/10 dark:text-amber-300">
          <RadioTower className="h-4 w-4" />
        </span>
        <div>
          <h2 className="text-base font-bold text-slate-950 dark:text-white">
            Camera Signal
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
            Camera, face visibility, and tracking reliability.
          </p>
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-3">
        <SignalMetric label="Interruptions" value={`${summary.count}`} />
        <SignalMetric
          label="Affected drives"
          value={`${summary.affectedSessionCount}`}
        />
        <SignalMetric
          label="Share"
          value={formatInsightPercent(summary.share)}
        />
      </div>

      <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs font-semibold leading-5 text-amber-900 dark:bg-amber-400/10 dark:text-amber-100">
        Signal interruptions can affect alert interpretation. Check lighting,
        camera angle, and face visibility if this repeats.
      </p>
    </section>
  );
}

function SignalMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-950">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p className="mt-1 text-base font-bold text-slate-950 dark:text-white">
        {value}
      </p>
    </div>
  );
}
