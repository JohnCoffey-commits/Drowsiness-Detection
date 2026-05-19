import { ArrowRight, Clock3 } from "lucide-react";
import type { InsightSessionComparisonRow } from "@/lib/insightsTypes";
import { formatInsightSource } from "@/lib/insightsUtils";

interface SessionComparisonTableProps {
  rows: InsightSessionComparisonRow[];
  onViewInHistory: (sessionId: string) => void;
}

function formatSessionTime(value: string): string {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "Unknown";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
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
            Session comparison
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
            Read-only comparison of sessions represented in local history.
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-3 py-1 text-xs font-black text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          <Clock3 className="h-3.5 w-3.5" />
          Last 48 hours
        </span>
      </div>

      {rows.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="min-w-[880px] w-full border-separate border-spacing-0 text-left text-sm">
            <thead>
              <tr className="text-xs font-black uppercase tracking-[0.12em] text-slate-400 dark:text-slate-500">
                <th className="border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Session
                </th>
                <th className="border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Source
                </th>
                <th className="border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Event count
                </th>
                <th className="border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  High priority
                </th>
                <th className="border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Signal quality
                </th>
                <th className="border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Pending review
                </th>
                <th className="border-b border-slate-200 py-3 pr-4 dark:border-slate-800">
                  Dominant pattern
                </th>
                <th className="border-b border-slate-200 py-3 dark:border-slate-800">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.sessionId}>
                  <td className="border-b border-slate-100 py-3 pr-4 align-top dark:border-slate-800/70">
                    <p className="font-black text-slate-900 dark:text-white">
                      {row.sessionId}
                    </p>
                    <p className="mt-1 text-xs font-semibold text-slate-500 dark:text-slate-400">
                      {formatSessionTime(row.startedAt)}
                    </p>
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 align-top dark:border-slate-800/70">
                    <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-black text-blue-700 dark:bg-cyan-400/10 dark:text-cyan-200">
                      {formatInsightSource(row.source)}
                    </span>
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-black text-slate-900 dark:border-slate-800/70 dark:text-white">
                    {row.eventCount}
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-black text-red-600 dark:border-slate-800/70 dark:text-red-300">
                    {row.highPriorityCount}
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-black text-amber-600 dark:border-slate-800/70 dark:text-amber-300">
                    {row.signalQualityIssueCount}
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-black text-slate-900 dark:border-slate-800/70 dark:text-white">
                    {row.pendingReviewCount}
                  </td>
                  <td className="border-b border-slate-100 py-3 pr-4 font-semibold text-slate-600 dark:border-slate-800/70 dark:text-slate-300">
                    {row.dominantPattern}
                  </td>
                  <td className="border-b border-slate-100 py-3 align-top dark:border-slate-800/70">
                    <button
                      type="button"
                      onClick={() => onViewInHistory(row.sessionId)}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-black text-slate-700 transition-colors hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-cyan-400/40 dark:hover:bg-cyan-400/10 dark:hover:text-cyan-100"
                    >
                      View in History
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
          No session comparison is available for this user.
        </div>
      )}
    </section>
  );
}
