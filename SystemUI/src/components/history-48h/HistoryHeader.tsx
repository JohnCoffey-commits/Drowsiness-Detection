import { Database, History, ShieldCheck } from "lucide-react";

const badges = [
  "Last 48 hours",
  "Local warning-candidate history",
  "Current user data",
];

export function HistoryHeader() {
  return (
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 bg-gradient-to-r from-slate-900 via-slate-800 to-blue-900 px-5 py-6 text-white sm:px-6">
        <div className="flex flex-wrap gap-2">
          {badges.map((badge) => (
            <span
              key={badge}
              className="inline-flex items-center rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold text-blue-50"
            >
              {badge}
            </span>
          ))}
        </div>
        <div className="mt-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-blue-100">
              <History className="h-4 w-4" />
              Warning-candidate history
            </div>
            <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">
              48h History
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-blue-50 sm:text-base">
              Recent driver-state warning-candidate history
            </p>
          </div>
          <div className="grid gap-2 text-sm text-blue-50 sm:grid-cols-2 lg:w-[420px]">
            <div className="rounded-xl border border-white/10 bg-white/10 p-3">
              <Database className="mb-2 h-4 w-4 text-blue-100" />
              Local history records
            </div>
            <div className="rounded-xl border border-white/10 bg-white/10 p-3">
              <ShieldCheck className="mb-2 h-4 w-4 text-blue-100" />
              Candidate-state review
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
