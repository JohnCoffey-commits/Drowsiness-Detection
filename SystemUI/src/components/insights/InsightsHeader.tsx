import {
  BarChart3,
  CarFront,
  Download,
  ShieldCheck,
  UserRound,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface InsightsHeaderProps {
  displayName?: string;
  recordCount: number;
  driveCount: number;
  onDownloadReport: () => void;
}

function HeaderBadge({
  icon: Icon,
  label,
}: {
  icon: LucideIcon;
  label: string;
}) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600 shadow-sm dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
      <Icon className="h-3.5 w-3.5" />
      <span className="truncate">{label}</span>
    </span>
  );
}

export function InsightsHeader({
  displayName,
  recordCount,
  driveCount,
  onDownloadReport,
}: InsightsHeaderProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600 text-white shadow-lg shadow-blue-900/20 dark:bg-cyan-500">
              <BarChart3 className="h-4 w-4" />
            </span>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-950 dark:text-white">
                Insights
              </h1>
              <p className="mt-1 text-sm font-semibold text-slate-500 dark:text-slate-400">
                Patterns from recent Live Monitor alerts.
              </p>
            </div>
          </div>
          <p className="mt-3 max-w-3xl text-sm font-medium leading-6 text-slate-600 dark:text-slate-300">
            These insights summarize fatigue-related visual cues, alert timing,
            drive-level patterns, and camera signal interruptions.
          </p>
        </div>

        <div className="flex flex-col gap-3 lg:items-end">
          <button
            type="button"
            onClick={onDownloadReport}
            disabled={recordCount === 0}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-emerald-100 bg-emerald-50 px-3 text-sm font-semibold text-emerald-700 shadow-sm transition hover:bg-emerald-100 focus:outline-none focus:ring-4 focus:ring-emerald-100 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400 disabled:shadow-none dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-200"
          >
            <Download className="h-4 w-4" />
            Download insights report
          </button>
          <div className="grid w-full grid-cols-2 gap-2 sm:w-auto sm:min-w-[320px]">
            <HeaderBadge icon={ShieldCheck} label="Last 48 hours" />
            <HeaderBadge icon={UserRound} label={displayName ?? "Local user"} />
            <HeaderBadge
              icon={BarChart3}
              label={`${recordCount} ${recordCount === 1 ? "alert" : "alerts"}`}
            />
            <HeaderBadge
              icon={CarFront}
              label={`${driveCount} ${driveCount === 1 ? "drive" : "drives"}`}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
