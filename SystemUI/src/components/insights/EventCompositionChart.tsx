import type { InsightCompositionItem } from "@/lib/insightsTypes";
import { formatInsightPercent } from "@/lib/insightsUtils";

interface EventCompositionChartProps {
  data: InsightCompositionItem[];
}

export function EventCompositionChart({ data }: EventCompositionChartProps) {
  const hasEvents = data.some((item) => item.count > 0);
  const visibleData = data.filter((item) => item.count > 0);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-5">
      <div className="mb-5">
        <h2 className="text-base font-black text-slate-950 dark:text-white">
          Alert Composition
        </h2>
        <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
          Breakdown of alert types in the selected history window.
        </p>
      </div>

      {hasEvents ? (
        <div className="space-y-4">
          {visibleData.map((item) => (
            <div key={item.kind}>
              <div className="mb-2 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm font-bold text-slate-700 dark:text-slate-200">
                    {item.label}
                  </span>
                </div>
                <span className="text-sm font-black text-slate-950 dark:text-white">
                  {item.count} · {formatInsightPercent(item.share)}
                </span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                <div
                  className="h-full rounded-full"
                  style={{
                    backgroundColor: item.color,
                    width: `${Math.max(2, Math.round(item.share * 100))}%`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex min-h-[220px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-center text-sm font-bold text-slate-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">
          No alert composition is available yet.
        </div>
      )}
    </section>
  );
}
