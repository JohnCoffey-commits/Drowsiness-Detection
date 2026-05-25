"use client";

import { LogOut, Settings } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { formatVisionGuardRole } from "@/lib/authTypes";
import { useVisionGuardAuth } from "@/lib/authStore";

function initialsFromName(name: string): string {
  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");

  return initials || "VG";
}

export function UserProfileMenu() {
  const { currentUser, logout } = useVisionGuardAuth();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const initials = useMemo(
    () => initialsFromName(currentUser?.displayName ?? "VisionGuard Driver"),
    [currentUser?.displayName]
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

  if (!currentUser) {
    return null;
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex items-center gap-2 rounded-full border border-slate-200/70 bg-white py-1 pl-1 pr-1 shadow-sm outline-none transition-colors hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400/60 xl:pr-3 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
      >
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 text-xs font-bold text-white shadow-inner">
          {initials}
        </span>
        <span className="hidden flex-col leading-tight xl:flex">
          <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
            {formatVisionGuardRole(currentUser.role)}
          </span>
          <span className="max-w-[150px] truncate text-sm font-semibold text-slate-700 dark:text-slate-100">
            {currentUser.displayName}
          </span>
        </span>
      </button>

      <div
        role="menu"
        className={`absolute right-0 top-12 z-50 w-[min(88vw,320px)] origin-top-right rounded-2xl border border-slate-200 bg-white p-3 text-slate-900 shadow-2xl shadow-slate-950/15 outline-none transition duration-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-50 dark:shadow-black/35 ${
          open
            ? "pointer-events-auto translate-y-0 scale-100 opacity-100"
            : "pointer-events-none -translate-y-1 scale-95 opacity-0"
        }`}
      >
        <div className="flex items-start gap-3 rounded-xl bg-slate-50 p-3 dark:bg-slate-800">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 text-sm font-black text-white">
            {initials}
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-black">
              {currentUser.displayName}
            </p>
            <p className="truncate text-xs font-medium text-slate-500 dark:text-slate-400">
              {currentUser.email}
            </p>
            <p className="mt-1 text-xs font-semibold text-blue-600 dark:text-cyan-300">
              {formatVisionGuardRole(currentUser.role)} · Local MVP account
            </p>
          </div>
        </div>

        <button
          type="button"
          role="menuitem"
          className="mt-2 flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm font-bold text-slate-700 outline-none transition hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-blue-400 dark:text-slate-100 dark:hover:bg-slate-800"
        >
          <Settings className="h-4 w-4" />
          Settings
        </button>

        <button
          type="button"
          role="menuitem"
          onClick={() => {
            setOpen(false);
            logout();
          }}
          className="mt-1 flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm font-bold text-rose-600 outline-none transition hover:bg-rose-50 focus-visible:ring-2 focus-visible:ring-rose-300 dark:text-rose-300 dark:hover:bg-rose-500/10"
        >
          <LogOut className="h-4 w-4" />
          Logout
        </button>
      </div>
    </div>
  );
}
