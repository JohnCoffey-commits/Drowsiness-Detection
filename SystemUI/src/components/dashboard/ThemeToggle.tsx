"use client";

import { Moon, Sun } from "lucide-react";
import { useVisionGuardTheme } from "@/lib/themeStore";

export function ThemeToggle() {
  const { theme, toggleTheme } = useVisionGuardTheme();
  const isNight = theme === "night";
  const Icon = isNight ? Moon : Sun;

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={`Switch to ${isNight ? "Day" : "Night"} theme`}
      className="hidden items-center gap-1.5 rounded-full border border-slate-200/70 bg-white px-2.5 py-1.5 text-sm shadow-sm outline-none transition-colors duration-200 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400/60 sm:flex sm:gap-2 sm:px-3 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
    >
      <Icon
        className={`h-3.5 w-3.5 sm:h-4 sm:w-4 ${
          isNight ? "text-cyan-300" : "text-orange-400"
        }`}
        strokeWidth={2.2}
      />
      <span className="text-xs font-semibold text-slate-700 sm:text-sm dark:text-slate-100">
        {isNight ? "Night" : "Day"}
      </span>
    </button>
  );
}
