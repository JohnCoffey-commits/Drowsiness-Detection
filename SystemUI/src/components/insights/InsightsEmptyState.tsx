import Link from "next/link";
import { History, LayoutDashboard } from "lucide-react";

export function InsightsEmptyState() {
  return (
    <section className="rounded-2xl border border-dashed border-slate-300 bg-white px-5 py-12 text-center shadow-sm transition-colors duration-300 dark:border-slate-700 dark:bg-slate-900">
      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50 text-blue-600 dark:bg-cyan-400/10 dark:text-cyan-300">
        <History className="h-6 w-6" />
      </div>
      <h2 className="mt-4 text-xl font-bold text-slate-950 dark:text-white">
        No insights yet
      </h2>
      <p className="mx-auto mt-2 max-w-xl text-sm font-medium leading-6 text-slate-500 dark:text-slate-400">
        Start a Live Monitor drive to generate alert patterns. Insights become
        more useful as more recent drives are recorded.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-3">
        <Link
          href="/"
          className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 dark:bg-cyan-500 dark:text-slate-950 dark:hover:bg-cyan-400"
        >
          <LayoutDashboard className="h-4 w-4" />
          Open Live Monitor
        </Link>
        <Link
          href="/history-48h"
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-sm transition-colors hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-cyan-400/40 dark:hover:bg-cyan-400/10 dark:hover:text-cyan-100"
        >
          <History className="h-4 w-4" />
          Open History
        </Link>
      </div>
    </section>
  );
}
