"use client";

import {
  Bell,
  Car,
  CheckCheck,
  Server,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import {
  getNotificationProductCategory,
  type ProductNotificationCategory,
  useVisionGuardNotifications,
} from "@/lib/notificationStore";
import type {
  VisionGuardNotification,
  VisionGuardNotificationSeverity,
} from "@/lib/notificationTypes";

type NotificationFilter = "all" | ProductNotificationCategory;

const FILTERS: Array<{ id: NotificationFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "driving", label: "Driving" },
  { id: "uploads", label: "Uploads" },
  { id: "system", label: "System" },
];

const severityStyle: Record<
  VisionGuardNotificationSeverity,
  { icon: string; dot: string }
> = {
  info: {
    icon: "bg-blue-50 text-blue-600 dark:bg-cyan-400/10 dark:text-cyan-300",
    dot: "bg-blue-500",
  },
  success: {
    icon: "bg-emerald-50 text-emerald-600 dark:bg-emerald-400/10 dark:text-emerald-300",
    dot: "bg-emerald-500",
  },
  warning: {
    icon: "bg-amber-50 text-amber-600 dark:bg-amber-400/10 dark:text-amber-300",
    dot: "bg-amber-500",
  },
  critical: {
    icon: "bg-rose-50 text-rose-600 dark:bg-rose-400/10 dark:text-rose-300",
    dot: "bg-rose-500",
  },
};

function formatNotificationTime(createdAt: string): string {
  const created = new Date(createdAt).getTime();
  if (!Number.isFinite(created)) {
    return "Now";
  }

  const elapsedMs = Date.now() - created;
  if (elapsedMs < 60_000) return "Now";
  if (elapsedMs < 60 * 60_000) return `${Math.floor(elapsedMs / 60_000)}m`;
  if (elapsedMs < 24 * 60 * 60_000) {
    return `${Math.floor(elapsedMs / (60 * 60_000))}h`;
  }
  return `${Math.floor(elapsedMs / (24 * 60 * 60_000))}d`;
}

function NotificationIcon({
  notification,
}: {
  notification: VisionGuardNotification;
}) {
  const className = cn(
    "flex h-9 w-9 shrink-0 items-center justify-center rounded-full",
    severityStyle[notification.severity].icon
  );

  const category = getNotificationProductCategory(notification);

  if (category === "driving") {
    return (
      <span className={className}>
        <Car className="h-4 w-4" strokeWidth={2.3} />
      </span>
    );
  }

  if (category === "uploads") {
    return (
      <span className={className}>
        <UploadCloud className="h-4 w-4" strokeWidth={2.3} />
      </span>
    );
  }

  return (
    <span className={className}>
      <Server className="h-4 w-4" strokeWidth={2.3} />
    </span>
  );
}

export function NotificationCenter() {
  const router = useRouter();
  const {
    notifications,
    unreadCount,
    markAllRead,
    markNotificationRead,
    clearRead,
  } = useVisionGuardNotifications();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState<NotificationFilter>("all");
  const containerRef = useRef<HTMLDivElement | null>(null);

  const visibleNotifications = useMemo(
    () =>
      notifications.filter(
        (notification) =>
          filter === "all" ||
          getNotificationProductCategory(notification) === filter
      ),
    [filter, notifications]
  );
  const totalUnreadCount = useMemo(
    () => notifications.filter((notification) => !notification.readAt).length,
    [notifications]
  );

  useEffect(() => {
    if (!open) return;

    function handlePointerDown(event: PointerEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  function handleNotificationClick(notification: VisionGuardNotification) {
    markNotificationRead(notification.id);
    if (notification.relatedRoute) {
      router.push(notification.relatedRoute);
      setOpen(false);
    }
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        aria-label={`Notifications${
          unreadCount ? ` (${unreadCount} attention unread)` : ""
        }`}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
        className="relative flex h-9 w-9 items-center justify-center rounded-full border border-slate-200/70 bg-white text-slate-500 shadow-sm outline-none transition-colors hover:bg-slate-50 hover:text-slate-700 focus-visible:ring-2 focus-visible:ring-blue-400/60 sm:h-10 sm:w-10 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-slate-50"
      >
        <Bell className="h-4 w-4" strokeWidth={2} />
        {unreadCount > 0 && (
          <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-rose-500 px-1 text-[10px] font-bold leading-none text-white ring-2 ring-[#f8fafc] dark:ring-slate-950">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      <div
        role="region"
        aria-label="Notifications"
        className={`absolute right-0 top-12 z-[1000] w-[min(92vw,420px)] origin-top-right rounded-2xl border border-slate-200 bg-white p-3 text-slate-900 shadow-2xl shadow-slate-950/15 outline-none transition duration-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-50 dark:shadow-black/35 ${
          open
            ? "pointer-events-auto translate-y-0 scale-100 opacity-100"
            : "pointer-events-none -translate-y-1 scale-95 opacity-0"
        }`}
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-black tracking-tight">Notifications</h2>
            <p className="mt-0.5 text-xs font-medium text-slate-500 dark:text-slate-400">
              {totalUnreadCount} unread
            </p>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={markAllRead}
              className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-bold text-blue-600 outline-none transition hover:bg-blue-50 focus-visible:ring-2 focus-visible:ring-blue-400 dark:text-cyan-300 dark:hover:bg-cyan-400/10"
            >
              <CheckCheck className="h-3.5 w-3.5" />
              Mark all read
            </button>
            <button
              type="button"
              onClick={clearRead}
              aria-label="Clear read notifications"
              className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-slate-400 outline-none transition hover:bg-slate-100 hover:text-slate-600 focus-visible:ring-2 focus-visible:ring-blue-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {FILTERS.map((candidate) => (
            <button
              key={candidate.id}
              type="button"
              onClick={() => setFilter(candidate.id)}
              className={cn(
                "rounded-full border px-2.5 py-1 text-xs font-bold outline-none transition focus-visible:ring-2 focus-visible:ring-blue-400",
                filter === candidate.id
                  ? "border-blue-200 bg-blue-50 text-blue-700 dark:border-cyan-400/30 dark:bg-cyan-400/10 dark:text-cyan-200"
                  : "border-slate-200 bg-slate-50 text-slate-500 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
              )}
            >
              {candidate.label}
            </button>
          ))}
        </div>

        <div className="mt-3 max-h-[360px] space-y-2 overflow-y-auto pr-1">
          {visibleNotifications.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm font-medium text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400">
              No notifications in this view.
            </div>
          ) : (
            visibleNotifications.map((notification) => (
              <button
                key={notification.id}
                type="button"
                onClick={() => handleNotificationClick(notification)}
                className="flex w-full items-start gap-3 rounded-2xl border border-slate-100 bg-white p-3 text-left outline-none transition hover:border-blue-100 hover:bg-blue-50/50 focus-visible:ring-2 focus-visible:ring-blue-400 dark:border-slate-800 dark:bg-slate-950 dark:hover:border-cyan-400/25 dark:hover:bg-cyan-400/10"
              >
                <NotificationIcon notification={notification} />
                <span className="min-w-0 flex-1">
                  <span className="flex items-start justify-between gap-2">
                    <span className="truncate text-sm font-black text-slate-900 dark:text-slate-50">
                      {notification.title}
                    </span>
                    <span className="shrink-0 text-[11px] font-bold text-slate-400">
                      {formatNotificationTime(notification.createdAt)}
                    </span>
                  </span>
                  <span className="mt-1 line-clamp-2 text-xs font-medium leading-5 text-slate-500 dark:text-slate-400">
                    {notification.message}
                  </span>
                </span>
                {!notification.readAt && (
                  <span
                    className={cn(
                      "mt-1.5 h-2 w-2 shrink-0 rounded-full",
                      severityStyle[notification.severity].dot
                    )}
                  />
                )}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
