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
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-5">
      <div className="mb-5">
        <h2 className="text-base font-black text-slate-950 dark:text-white">
          Time-of-day pattern
        </h2>
        <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
          Conservative grouping by local record timestamp.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {data.map((item) => (
          <article
            key={item.id}
            className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-950"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-black text-slate-900 dark:text-white">
                  {item.label}
                </p>
                <p className="text-xs font-bold text-slate-500 dark:text-slate-400">
                  {item.timeRange}
                </p>
              </div>
              <span className="text-sm font-black text-slate-950 dark:text-white">
                {item.count}
              </span>
            </div>
            <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-white dark:bg-slate-800">
              <div
                className="h-full rounded-full bg-blue-600 dark:bg-cyan-400"
                style={{
                  width: `${Math.max(4, Math.round((item.count / maxCount) * 100))}%`,
                }}
              />
            </div>
            <p className="mt-2 text-xs font-bold text-slate-500 dark:text-slate-400">
              {formatInsightPercent(item.share)} of local records
            </p>
          </article>
        ))}
      </div>

      <p className="mt-4 rounded-xl bg-blue-50 px-4 py-3 text-sm font-semibold leading-6 text-blue-900 dark:bg-cyan-400/10 dark:text-cyan-100">
        {description}
      </p>
    </section>
  );
}
