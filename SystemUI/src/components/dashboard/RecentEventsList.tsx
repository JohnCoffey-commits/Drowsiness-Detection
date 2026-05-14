"use client";

import { Card } from "@/components/ui/card";
import { ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { dashboardData, eventStyle, formatClock } from "@/lib/mockData";

export function RecentEventsList() {
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setNow(new Date());
  }, []);

  return (
    <Card className="col-span-3 flex h-full min-h-0 flex-col rounded-[2rem] border border-slate-200/70 bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-md lg:col-span-1">
      <div className="mb-3 flex shrink-0 items-center justify-between">
        <h3 className="text-lg font-bold tracking-tight text-slate-800">
          Recent Events
        </h3>
        <span className="text-[11px] font-medium text-slate-400">
          {dashboardData.events.length} today
        </span>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto pr-1">
        {dashboardData.events.map((event, i) => {
          const style = eventStyle[event.type];
          const time = now
            ? formatClock(new Date(now.getTime() - event.secondsAgo * 1000))
            : "--:--:--";
          return (
            <div
              key={i}
              className="group flex items-center gap-4 text-sm"
            >
              <span className="w-16 font-mono text-xs font-medium text-slate-400 transition-colors group-hover:text-slate-600">
                {time}
              </span>
              <div className="flex items-center gap-2.5">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${style.dot} shadow-sm`}
                />
                <span className={`font-medium ${style.label}`}>
                  {event.type}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <button
        type="button"
        className="mt-3 flex shrink-0 items-center justify-center gap-1 rounded-xl border border-slate-100 bg-slate-50/60 py-2 text-xs font-semibold text-slate-500 transition-colors outline-none hover:bg-slate-100 hover:text-slate-700 focus-visible:ring-2 focus-visible:ring-blue-400/60"
      >
        View all events
        <ChevronRight className="h-3.5 w-3.5" strokeWidth={2.4} />
      </button>
    </Card>
  );
}
