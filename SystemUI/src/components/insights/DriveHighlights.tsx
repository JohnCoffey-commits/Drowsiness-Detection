import Link from "next/link";
import { ExternalLink } from "lucide-react";
import type { InsightSessionComparisonRow } from "@/lib/insightsTypes";

interface DriveHighlightsProps {
  rows: InsightSessionComparisonRow[];
  visibleCount?: number;
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

function sortDriveHighlights(
  rows: InsightSessionComparisonRow[]
): InsightSessionComparisonRow[] {
  return [...rows].sort(
    (a, b) =>
      b.highPriorityCount - a.highPriorityCount ||
      b.eventCount - a.eventCount ||
      b.signalInterruptionCount - a.signalInterruptionCount ||
      new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime()
  );
}

export function DriveHighlights({
  rows,
  visibleCount = 3,
}: DriveHighlightsProps) {
  const highlightedRows = sortDriveHighlights(rows).slice(0, visibleCount);
  const hasRows = highlightedRows.length > 0;
  const maxAlerts = Math.max(1, ...highlightedRows.map((row) => row.eventCount));

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-4">
      <div className="mb-3 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-base font-bold text-slate-950 dark:text-white">
            Drive Highlights
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
            Recent drives with the most alert activity.
          </p>
        </div>
        <Link
          href="/history-48h"
          className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 transition-colors hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-cyan-400/40 dark:hover:bg-cyan-400/10 dark:hover:text-cyan-100"
        >
          Open full History
          <ExternalLink className="h-4 w-4" />
        </Link>
      </div>

      {hasRows ? (
        <div className="space-y-2">
          {highlightedRows.map((row, index) => (
            <article
              key={row.sessionId}
              className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:items-center"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-white text-xs font-semibold text-slate-500 ring-1 ring-slate-200 dark:bg-slate-900 dark:text-slate-300 dark:ring-slate-800">
                    {index + 1}
                  </span>
                  <h3 className="truncate text-sm font-bold text-slate-950 dark:text-white">
                    {row.driveLabel}
                  </h3>
                </div>
                <p className="mt-1 text-xs font-semibold text-slate-500 dark:text-slate-400">
                  {row.durationLabel} · Main pattern: {row.dominantPattern}
                </p>
              </div>

              <div className="min-w-0">
                <p className="mb-2 text-xs font-semibold text-slate-600 dark:text-slate-300">
                  {row.eventCount} {row.eventCount === 1 ? "alert" : "alerts"} ·{" "}
                  {row.highPriorityCount} high-risk ·{" "}
                  {row.signalInterruptionCount} signal{" "}
                  {row.signalInterruptionCount === 1
                    ? "interruption"
                    : "interruptions"}
                </p>
                <div className="h-3 overflow-hidden rounded-full bg-white shadow-inner dark:bg-slate-800">
                  <div
                    className="flex h-full overflow-hidden rounded-full"
                    style={{
                      width: `${Math.max(8, (row.eventCount / maxAlerts) * 100)}%`,
                    }}
                  >
                    {SEGMENTS.map((segment) => {
                      const value = row[segment.key];
                      if (value <= 0 || row.eventCount <= 0) return null;
                      return (
                        <div
                          key={segment.key}
                          title={`${segment.label}: ${value}`}
                          style={{
                            backgroundColor: segment.color,
                            width: `${Math.max(5, (value / row.eventCount) * 100)}%`,
                          }}
                        />
                      );
                    })}
                  </div>
                </div>
              </div>
            </article>
          ))}
          <p className="pt-1 text-xs font-semibold text-slate-500 dark:text-slate-400">
            Showing top {highlightedRows.length} of {rows.length} drives.
          </p>
        </div>
      ) : (
        <div className="flex min-h-[120px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-center text-sm font-medium text-slate-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">
          No drive highlights are available yet.
        </div>
      )}
    </section>
  );
}
