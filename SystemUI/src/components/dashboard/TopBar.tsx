"use client";

import { Bell, Clock } from "lucide-react";
import { useEffect, useState } from "react";
import { dashboardData, formatHMS } from "@/lib/mockData";

export function TopBar() {
  const { driver, status, notifications } = dashboardData;
  const [elapsed, setElapsed] = useState(driver.sessionStartedSecondsAgo);

  useEffect(() => {
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b border-slate-200/60 bg-[#f8fafc]/80 px-4 py-3 backdrop-blur-md lg:px-6">
      <h2 className="min-w-0 truncate text-lg font-bold tracking-tight text-slate-800 xl:text-xl">
        <span>Live Monitor</span>
      </h2>

      <div className="flex shrink-0 items-center gap-2 lg:gap-3">
        {status.isLive ? (
          <span className="flex items-center gap-1.5 rounded-full border border-red-100 bg-red-50 px-2.5 py-1.5 text-xs font-semibold text-red-600 shadow-sm sm:text-sm sm:gap-2 sm:px-3">
            <span className="h-2 w-2 animate-pulse rounded-full bg-red-600 sm:h-2.5 sm:w-2.5" />
            LIVE
          </span>
        ) : (
          <span className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm font-semibold text-slate-500">
            OFFLINE
          </span>
        )}

        <div className="flex items-center gap-1.5 rounded-full border border-slate-200/70 bg-white px-2.5 py-1.5 text-sm shadow-sm sm:gap-2 sm:px-3">
          <Clock className="h-3.5 w-3.5 text-slate-400 sm:h-4 sm:w-4" strokeWidth={2.2} />
          <span className="hidden text-[11px] font-medium uppercase tracking-wider text-slate-400 lg:inline">
            Drive
          </span>
          <span className="text-xs font-semibold tabular-nums text-slate-700 sm:text-sm">
            {formatHMS(elapsed)}
          </span>
        </div>

        <div className="hidden sm:flex items-center gap-1.5 rounded-full border border-slate-200/70 bg-white px-2.5 py-1.5 text-sm shadow-sm sm:gap-2 sm:px-3">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5 text-orange-400 sm:h-4 sm:w-4">
            <circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path><path d="M2 12h2"></path><path d="M20 12h2"></path><path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path>
          </svg>
          <span className="text-xs font-semibold text-slate-700 sm:text-sm">
            Day
          </span>
        </div>

        <button
          type="button"
          aria-label={`Notifications${notifications ? ` (${notifications} unread)` : ""}`}
          className="relative flex h-9 w-9 items-center justify-center rounded-full border border-slate-200/70 bg-white text-slate-500 shadow-sm outline-none transition-colors hover:bg-slate-50 hover:text-slate-700 focus-visible:ring-2 focus-visible:ring-blue-400/60 sm:h-10 sm:w-10"
        >
          <Bell className="h-4 w-4" strokeWidth={2} />
          {notifications > 0 && (
            <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-rose-500 px-1 text-[10px] font-bold leading-none text-white ring-2 ring-[#f8fafc]">
              {notifications}
            </span>
          )}
        </button>

        <div className="flex items-center gap-2 rounded-full border border-slate-200/70 bg-white py-1 pl-1 pr-1 shadow-sm xl:pr-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 text-xs font-bold text-white shadow-inner">
            {driver.initials}
          </div>
          <div className="hidden flex-col leading-tight xl:flex">
            <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
              Driver
            </span>
            <span className="text-sm font-semibold text-slate-700">
              {driver.name}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
