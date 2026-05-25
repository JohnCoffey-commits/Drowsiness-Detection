import { History } from "lucide-react";

export function HistoryHeader() {
  return (
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 bg-gradient-to-r from-slate-900 via-slate-800 to-blue-900 px-5 py-5 text-white sm:px-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-blue-100">
              <History className="h-4 w-4" />
              Live Monitor history
            </div>
            <h1 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">
              48h History
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-blue-50 sm:text-base">
              View recent driving alerts, fatigue-related visual cues, and
              camera signal interruptions from Live Monitor.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
