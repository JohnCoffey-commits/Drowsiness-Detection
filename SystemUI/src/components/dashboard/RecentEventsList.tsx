"use client";

import { Card } from "@/components/ui/card";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import type {
  LiveMonitorDashboardEvent,
  LiveMonitorDashboardEventKind,
} from "@/lib/liveMonitorDashboardTypes";

interface RecentEventsListProps {
  events: LiveMonitorDashboardEvent[];
  className?: string;
  expanded?: boolean;
  actionTabIndex?: 0 | -1;
  onCollapse?: () => void;
  onExpand?: () => void;
}

const EVENT_STYLE: Record<
  LiveMonitorDashboardEventKind,
  { dot: string; label: string }
> = {
  normal: { dot: "bg-emerald-500", label: "text-slate-700" },
  eye_warning: { dot: "bg-orange-500", label: "text-slate-700" },
  yawn_warning: { dot: "bg-rose-500", label: "text-slate-700" },
  critical_eye_warning: { dot: "bg-red-600", label: "text-red-700" },
  signal_quality: { dot: "bg-slate-500", label: "text-slate-700" },
};

function formatEventTime(timestamp: string): string {
  const date = new Date(timestamp);
  if (!Number.isFinite(date.getTime())) {
    return "--:--:--";
  }

  return [date.getHours(), date.getMinutes(), date.getSeconds()]
    .map((value) => String(value).padStart(2, "0"))
    .join(":");
}

function RecentEventsBody({
  events,
  expanded = false,
}: {
  events: LiveMonitorDashboardEvent[];
  expanded?: boolean;
}) {
  if (events.length === 0) {
    return (
      <div
        className={`flex items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 px-4 text-center text-sm font-medium leading-6 text-slate-500 ${
          expanded ? "min-h-[320px]" : "h-full min-h-[120px]"
        }`}
      >
        No warning-candidate events today.
      </div>
    );
  }

  return (
    <>
      {events.map((event) => {
        const style = EVENT_STYLE[event.kind];
        return (
          <div key={event.id} className="group flex items-center gap-4 text-sm">
            <span className="w-16 font-mono text-xs font-medium text-slate-400 transition-colors group-hover:text-slate-600">
              {formatEventTime(event.timestamp)}
            </span>
            <div className="flex min-w-0 items-center gap-2.5">
              <span
                className={`h-2.5 w-2.5 shrink-0 rounded-full ${style.dot} shadow-sm`}
              />
              <span className={`truncate font-medium ${style.label}`}>
                {event.label}
              </span>
            </div>
          </div>
        );
      })}
    </>
  );
}

export function RecentEventsList({
  events,
  className = "",
  expanded = false,
  actionTabIndex = 0,
  onCollapse,
  onExpand,
}: RecentEventsListProps) {
  const [uncontrolledExpanded, setUncontrolledExpanded] = useState(false);
  const isExpanded = expanded || uncontrolledExpanded;
  const expand = () => {
    onExpand?.();
    if (!onExpand) {
      setUncontrolledExpanded(true);
    }
  };
  const collapse = () => {
    onCollapse?.();
    if (!onCollapse) {
      setUncontrolledExpanded(false);
    }
  };

  return (
    <Card
      className={`col-span-3 flex h-full min-h-0 flex-col rounded-[2rem] border border-slate-200/70 bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-md lg:col-span-1 ${className}`}
    >
      <div className="mb-3 flex shrink-0 items-center justify-between">
        <div>
          <h3 className="text-lg font-bold tracking-tight text-slate-800">
            Recent Events
          </h3>
          {isExpanded && (
            <p className="mt-1 text-xs font-medium text-slate-400">
              Session-local warning-candidate events
            </p>
          )}
        </div>
        <span className="text-[11px] font-medium text-slate-400">Today</span>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        <RecentEventsBody events={events} expanded={isExpanded} />
      </div>

      <button
        type="button"
        tabIndex={actionTabIndex}
        onClick={isExpanded ? collapse : expand}
        aria-expanded={isExpanded}
        className="mt-3 flex shrink-0 items-center justify-center gap-1 rounded-xl border border-slate-100 bg-slate-50/60 py-2 text-xs font-semibold text-slate-500 transition-colors outline-none hover:bg-slate-100 hover:text-slate-700 focus-visible:ring-2 focus-visible:ring-blue-400/60"
      >
        {isExpanded ? "Collapse events" : "Expand events"}
        {isExpanded ? (
          <ChevronDown className="h-3.5 w-3.5" strokeWidth={2.4} />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" strokeWidth={2.4} />
        )}
      </button>
    </Card>
  );
}
