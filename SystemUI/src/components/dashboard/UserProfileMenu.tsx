"use client";

import { LogOut, Settings, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { formatVisionGuardRole } from "@/lib/authTypes";
import { useVisionGuardAuth } from "@/lib/authStore";
import { useVisionGuardSettings } from "@/lib/settingsStore";

function initialsFromName(name: string): string {
  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");

  return initials || "VG";
}

function SettingsModal({ onClose }: { onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const { settings, setMinimalLiveMonitorMode } = useVisionGuardSettings();
  const minimalMode = settings.liveMonitor.minimalMode;

  useEffect(() => {
    closeButtonRef.current?.focus();

    function handlePointerDown(event: PointerEvent) {
      if (
        panelRef.current &&
        !panelRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  if (typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <div
      ref={panelRef}
      role="dialog"
      aria-labelledby="visionguard-settings-title"
      className="fixed right-4 top-[72px] z-[1000] w-[min(92vw,420px)] origin-top-right rounded-2xl border border-slate-200 bg-white p-3 text-slate-900 shadow-2xl shadow-slate-950/15 outline-none transition duration-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-50 dark:shadow-black/35 lg:right-6"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2
            id="visionguard-settings-title"
            className="text-sm font-black tracking-tight text-slate-900 dark:text-slate-50"
          >
            Settings
          </h2>
          <p className="mt-0.5 text-xs font-medium text-slate-500 dark:text-slate-400">
            Live Monitor display
          </p>
        </div>

        <button
          ref={closeButtonRef}
          type="button"
          aria-label="Close settings"
          onClick={onClose}
          className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-slate-400 outline-none transition hover:bg-slate-100 hover:text-slate-600 focus-visible:ring-2 focus-visible:ring-blue-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="mt-3 rounded-2xl border border-slate-100 bg-white p-3 text-left outline-none transition dark:border-slate-800 dark:bg-slate-950">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-sm font-black text-slate-900 dark:text-slate-50">
              Minimal Live Monitor Mode
            </h3>
            <p className="mt-1 line-clamp-4 text-xs font-medium leading-5 text-slate-500 dark:text-slate-400">
              Hide the camera preview, recent events, charts, and extra dashboard
              panels while keeping realtime monitoring, sound alerts, and critical
              warning popups active.
            </p>
          </div>

          <button
            type="button"
            role="switch"
            aria-checked={minimalMode}
            aria-label="Minimal Live Monitor Mode"
            onClick={() => setMinimalLiveMonitorMode(!minimalMode)}
            className={`relative mt-0.5 h-6 w-11 shrink-0 rounded-full outline-none ring-offset-2 transition focus-visible:ring-2 focus-visible:ring-blue-400 ${
              minimalMode ? "bg-blue-600" : "bg-slate-300 dark:bg-slate-600"
            }`}
          >
            <span
              className={`absolute left-1 top-1 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
                minimalMode ? "translate-x-5" : "translate-x-0"
              }`}
            />
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}

export function UserProfileMenu() {
  const { currentUser, logout } = useVisionGuardAuth();
  const [open, setOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
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
          onClick={() => {
            setOpen(false);
            setSettingsOpen(true);
          }}
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

      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}
