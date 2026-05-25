import type { InsightSessionComparisonRow } from "@/lib/insightsTypes";

interface AlertsByDriveChartProps {
  rows: InsightSessionComparisonRow[];
}

const SEGMENTS = [
  {
    key: "criticalEyeCount",
    label: "High-risk eye",
    color: "#ef4444",
  },
  {
    key: "eyeClosureCount",
    label: "Eye closure",
    color: "#f97316",
  },
  {
    key: "yawnCount",
    label: "Yawn",
    color: "#ec4899",
  },
  {
    key: "signalInterruptionCount",
    label: "Signal",
    color: "#64748b",
  },
] as const;

export function AlertsByDriveChart({ rows }: AlertsByDriveChartProps) {
  const hasAlerts = rows.some((row) => row.eventCount > 0);
  const maxAlerts = Math.max(1, ...rows.map((row) => row.eventCount));

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-5">
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-base font-bold text-slate-950 dark:text-white">
            Alerts by Drive
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
            Alert counts grouped by recent Live Monitor drives.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {SEGMENTS.map((segment) => (
            <span
              key={segment.key}
              className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-2.5 py-1 text-xs font-bold text-slate-600 dark:bg-slate-800 dark:text-slate-300"
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: segment.color }}
              />
              {segment.label}
            </span>
          ))}
        </div>
      </div>

      {hasAlerts ? (
        <div className="space-y-3">
          {rows.map((row) => (
            <article
              key={row.sessionId}
              className="rounded-xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950"
            >
              <div className="mb-2 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                    {row.driveLabel}
                  </h3>
                  <p className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                    {row.durationLabel} · {row.dominantPattern}
                  </p>
                </div>
                <p className="text-sm font-bold text-slate-950 dark:text-white">
                  {row.eventCount} {row.eventCount === 1 ? "alert" : "alerts"}
                </p>
              </div>

              <div className="h-4 overflow-hidden rounded-full bg-white shadow-inner dark:bg-slate-800">
                <div
                  className="flex h-full overflow-hidden rounded-full"
                  style={{
                    width: `${Math.max(6, (row.eventCount / maxAlerts) * 100)}%`,
                  }}
                >
                  {SEGMENTS.map((segment) => {
                    const value = row[segment.key];
                    if (value <= 0) return null;
                    return (
                      <div
                        key={segment.key}
                        title={`${segment.label}: ${value}`}
                        style={{
                          backgroundColor: segment.color,
                          width: `${Math.max(4, (value / row.eventCount) * 100)}%`,
                        }}
                      />
                    );
                  })}
                </div>
              </div>
            </article>
          ))}
          {rows.length === 1 ? (
            <p className="rounded-xl bg-blue-50 px-4 py-3 text-sm font-semibold text-blue-900 dark:bg-cyan-400/10 dark:text-cyan-100">
              Alerts are currently concentrated in one recorded drive.
            </p>
          ) : null}
        </div>
      ) : (
        <div className="flex min-h-[180px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-center text-sm font-medium text-slate-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">
          No alerts in this time window.
        </div>
      )}
    </section>
  );
}
