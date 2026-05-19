import { BarChart3, Database, ShieldCheck, UserRound } from "lucide-react";
import { INSIGHTS_BOUNDARY_NOTICE } from "@/lib/insightsUtils";

interface InsightsHeaderProps {
  displayName?: string;
  recordCount: number;
}

function HeaderBadge({
  icon: Icon,
  label,
}: {
  icon: typeof Database;
  label: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-bold text-slate-600 shadow-sm dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
      <Icon className="h-3.5 w-3.5" />
      {label}
    </span>
  );
}

export function InsightsHeader({ displayName, recordCount }: InsightsHeaderProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white shadow-lg shadow-blue-900/20 dark:bg-cyan-500">
              <BarChart3 className="h-5 w-5" />
            </span>
            <div>
              <h1 className="text-2xl font-black tracking-tight text-slate-950 dark:text-white sm:text-3xl">
                Insights
              </h1>
              <p className="mt-1 text-sm font-semibold text-slate-500 dark:text-slate-400">
                User-scoped local warning-candidate analytics
              </p>
            </div>
          </div>
          <p className="mt-4 max-w-3xl text-sm font-medium leading-6 text-slate-600 dark:text-slate-300">
            {INSIGHTS_BOUNDARY_NOTICE}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          <HeaderBadge icon={Database} label="Local history" />
          <HeaderBadge icon={ShieldCheck} label="Last 48 hours" />
          <HeaderBadge icon={UserRound} label="Current user" />
          <span className="inline-flex items-center rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-600 dark:bg-slate-800 dark:text-slate-300">
            {displayName ?? "Local user"} · {recordCount} records
          </span>
        </div>
      </div>
    </section>
  );
}
