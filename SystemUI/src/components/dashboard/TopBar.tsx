"use client";

import { Clock } from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { dashboardData, formatHMS } from "@/lib/mockData";
import { ThemeToggle } from "@/components/dashboard/ThemeToggle";
import { NotificationCenter } from "@/components/dashboard/NotificationCenter";
import { UserProfileMenu } from "@/components/dashboard/UserProfileMenu";

const ROUTE_TITLES: Record<string, string> = {
  "/": "Live Monitor",
  "/video-upload": "Video Upload Analysis",
  "/history-48h": "48h History",
  "/insights": "Insights",
};

export function TopBar() {
  const { driver, status } = dashboardData;
  const pathname = usePathname();
  const [elapsed, setElapsed] = useState(driver.sessionStartedSecondsAgo);
  const title = ROUTE_TITLES[pathname ?? ""] ?? "VisionGuard";

  useEffect(() => {
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="sticky top-0 z-40 flex items-center justify-between gap-3 border-b border-slate-200/60 bg-[#f8fafc]/80 px-4 py-3 backdrop-blur-md transition-colors duration-300 lg:px-6 dark:border-slate-800/80 dark:bg-slate-950/80">
      <h2 className="min-w-0 truncate text-lg font-bold tracking-tight text-slate-800 xl:text-xl">
        <span>{title}</span>
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

        <div className="flex items-center gap-1.5 rounded-full border border-slate-200/70 bg-white px-2.5 py-1.5 text-sm shadow-sm transition-colors duration-200 sm:gap-2 sm:px-3 dark:border-slate-700 dark:bg-slate-900">
          <Clock className="h-3.5 w-3.5 text-slate-400 sm:h-4 sm:w-4" strokeWidth={2.2} />
          <span className="hidden text-[11px] font-medium uppercase tracking-wider text-slate-400 lg:inline">
            Current drive
          </span>
          <span className="text-xs font-semibold tabular-nums text-slate-700 sm:text-sm">
            {formatHMS(elapsed)}
          </span>
        </div>

        <ThemeToggle />
        <NotificationCenter />
        <UserProfileMenu />
      </div>
    </header>
  );
}
