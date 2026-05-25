import type { InsightTimeOfDayItem } from "@/lib/insightsTypes";
import { formatInsightPercent } from "@/lib/insightsUtils";

interface TimeOfDayPatternProps {
  data: InsightTimeOfDayItem[];
  description: string;
}

export function TimeOfDayPattern({
  data,
  description,
}: TimeOfDayPatternProps) {
  const maxCount = Math.max(1, ...data.map((item) => item.count));

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-4">
      <div className="mb-3">
        <h2 className="text-base font-bold text-slate-950 dark:text-white">
          Time of Day
        </h2>
        <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
          When alerts occurred during the selected window.
        </p>
      </div>

      <div className="space-y-2">
        {data.map((item) => (
          <article
            key={item.id}
            className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 dark:border-slate-800 dark:bg-slate-950 sm:grid-cols-[128px_minmax(0,1fr)_92px] sm:items-center"
          >
            <div>
              <p className="text-sm font-bold text-slate-900 dark:text-white">
                {item.label}
              </p>
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                {item.timeRange}
              </p>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-white dark:bg-slate-800">
              <div
                className="h-full rounded-full bg-blue-600 dark:bg-cyan-400"
                style={{
                  width:
                    item.count === 0
                      ? "0%"
                      : `${Math.max(4, Math.round((item.count / maxCount) * 100))}%`,
                }}
              />
            </div>
            <p className="text-left text-xs font-semibold text-slate-700 dark:text-slate-200 sm:text-right">
              {item.count} {item.count === 1 ? "alert" : "alerts"}
              <span className="ml-1 font-medium text-slate-400">
                {formatInsightPercent(item.share)}
              </span>
            </p>
          </article>
        ))}
      </div>

      <p className="mt-3 rounded-lg bg-blue-50 px-3 py-2 text-sm font-semibold leading-5 text-blue-900 dark:bg-cyan-400/10 dark:text-cyan-100">
        {description}
      </p>
    </section>
  );
}
