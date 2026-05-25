import { ArrowRight, Clock3 } from "lucide-react";
import type { InsightSessionComparisonRow } from "@/lib/insightsTypes";

interface SessionComparisonTableProps {
  rows: InsightSessionComparisonRow[];
  onViewInHistory: (sessionId?: string) => void;
}

export function SessionComparisonTable({
  rows,
  onViewInHistory,
}: SessionComparisonTableProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-5">
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-base font-black text-slate-950 dark:text-white">
            Drive Comparison
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
            Compare recent drives by alert count, high-risk alerts, and signal
            interruptions.
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-3 py-1 text-xs font-black text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          <Clock3 className="h-3.5 w-3.5" />
          Last 48 hours
        </span>
      </div>

      {rows.length > 0 ? (
        <div className="overflow-hidden">
          <table className="w-full table-fixed border-separate border-spacing-0 text-left text-sm">
            <thead>
              <tr className="text-xs font-black uppercase tracking-[0.12em] text-slate-400 dark:text-slate-500">
                <th className="w-[28%] border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Drive
                </th>
                <th className="w-[12%] border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Duration
                </th>
                <th className="w-[10%] border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Alerts
                </th>
                <th className="w-[12%] border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  High-risk
                </th>
                <th className="w-[16%] border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Signal interruptions
                </th>
                <th className="w-[16%] border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Main pattern
                </th>
                <th className="w-[6%] border-b border-slate-200 py-3 dark:border-slate-800">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.sessionId}>
                  <td className="border-b border-slate-100 py-3 pr-4 align-top dark:border-slate-800/70">
                    <p className="break-words font-black text-slate-900 dark:text-white">
                      {row.driveLabel}
                    </p>
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-semibold text-slate-600 dark:border-slate-800/70 dark:text-slate-300">
                    {row.durationLabel}
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-black text-slate-900 dark:border-slate-800/70 dark:text-white">
                    {row.eventCount}
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-black text-red-600 dark:border-slate-800/70 dark:text-red-300">
                    {row.highPriorityCount}
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-black text-amber-600 dark:border-slate-800/70 dark:text-amber-300">
                    {row.signalInterruptionCount}
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-semibold text-slate-600 dark:border-slate-800/70 dark:text-slate-300">
                    {row.dominantPattern}
                  </td>
                  <td className="border-b border-slate-100 py-3 align-top dark:border-slate-800/70">
                    <button
                      type="button"
                      onClick={() => onViewInHistory(row.sessionId)}
                      className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-700 transition-colors hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-cyan-400/40 dark:hover:bg-cyan-400/10 dark:hover:text-cyan-100"
                      aria-label={`Open ${row.driveLabel} in History`}
                    >
                      <ArrowRight className="h-3.5 w-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="flex min-h-[220px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-center text-sm font-bold text-slate-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">
          No drive comparison is available yet.
        </div>
      )}
      <div className="mt-4">
        <button
          type="button"
          onClick={() => onViewInHistory()}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-black text-slate-700 transition-colors hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-cyan-400/40 dark:hover:bg-cyan-400/10 dark:hover:text-cyan-100"
        >
          Open in History
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </section>
  );
}
