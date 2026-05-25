import { Sparkles } from "lucide-react";

interface KeyInsightsProps {
  insights: string[];
  stats: {
    dominant: string;
    highRiskShare: string;
    signalCount: number;
  };
}

export function KeyInsights({ insights, stats }: KeyInsightsProps) {
  const limitedDataNote = insights.find((insight) =>
    insight.toLowerCase().includes("small number")
  );
  const primarySummary = insights
    .filter((insight) => insight !== limitedDataNote)
    .slice(0, 3)
    .join(" ");

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <span className="rounded-lg bg-blue-50 p-2 text-blue-600 dark:bg-cyan-400/10 dark:text-cyan-300">
            <Sparkles className="h-4 w-4" />
          </span>
          <div>
            <h2 className="text-base font-bold text-slate-950 dark:text-white">
              Key Insight Summary
            </h2>
            <p className="mt-1.5 max-w-5xl text-sm font-medium leading-5 text-slate-600 dark:text-slate-300">
              {primarySummary}
            </p>
            {limitedDataNote ? (
              <p className="mt-1.5 text-xs font-semibold text-slate-500 dark:text-slate-400">
                {limitedDataNote}
              </p>
            ) : null}
          </div>
        </div>

        <div className="grid shrink-0 gap-2 sm:grid-cols-3 lg:min-w-[360px]">
          <InsightStat label="Dominant" value={stats.dominant} />
          <InsightStat label="High-risk" value={stats.highRiskShare} />
          <InsightStat label="Signal" value={`${stats.signalCount} alerts`} />
        </div>
      </div>
    </section>
  );
}

function InsightStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 dark:border-slate-800 dark:bg-slate-950">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
        {label}
      </p>
      <p className="mt-1 truncate text-sm font-bold text-slate-900 dark:text-white">
        {value}
      </p>
    </div>
  );
}
