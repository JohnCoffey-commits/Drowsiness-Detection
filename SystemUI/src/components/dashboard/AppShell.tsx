"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import { Sidebar } from "./Sidebar";
import { LiveMonitorPage } from "@/components/dashboard/LiveMonitorPage";
import { TopBar } from "@/components/dashboard/TopBar";
import { LoginScreen } from "@/components/auth/LoginScreen";
import {
  VisionGuardAuthProvider,
  useVisionGuardAuth,
} from "@/lib/authStore";
import { VisionGuardThemeProvider, useVisionGuardTheme } from "@/lib/themeStore";
import { VisionGuardNotificationsProvider } from "@/lib/notificationStore";
import { cn } from "@/lib/utils";

function AppFrame({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();
  const isLiveMonitorRoute = pathname === "/";
  const { currentUser, isReady } = useVisionGuardAuth();
  const { theme } = useVisionGuardTheme();

  if (!isReady) {
    return (
      <div
        data-theme={theme}
        className={cn(
          "vg-themed flex min-h-dvh items-center justify-center bg-[#f4f7f9] text-slate-900 transition-colors duration-300",
          theme === "day" && "theme-day",
          theme === "night" && "theme-night dark bg-slate-950 text-slate-50"
        )}
      >
        <div className="rounded-2xl border border-slate-200 bg-white px-5 py-4 text-sm font-bold shadow-sm dark:border-slate-700 dark:bg-slate-900">
          Loading VisionGuard...
        </div>
      </div>
    );
  }

  if (!currentUser) {
    return (
      <div
        data-theme={theme}
        className={cn(
          "vg-themed min-h-dvh",
          theme === "day" && "theme-day",
          theme === "night" && "theme-night dark"
        )}
      >
        <LoginScreen />
      </div>
    );
  }

  return (
    <div
      data-theme={theme}
      className={cn(
        "vg-themed flex h-dvh w-full overflow-hidden bg-[#f4f7f9] text-slate-900 transition-colors duration-300",
        theme === "day" && "theme-day",
        theme === "night" && "theme-night dark bg-slate-950 text-slate-50"
      )}
    >
      <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
      <div className="flex-1 min-w-0 flex flex-col h-dvh overflow-hidden transition-all duration-300">
        <TopBar />
        <div
          className={
            isLiveMonitorRoute ? "flex h-full min-h-0 flex-col" : "hidden"
          }
        >
          <LiveMonitorPage />
        </div>
        <div
          className={
            isLiveMonitorRoute ? "hidden" : "flex h-full min-h-0 flex-col"
          }
        >
          {children}
        </div>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <VisionGuardAuthProvider>
      <VisionGuardThemeProvider>
        <VisionGuardNotificationsProvider>
          <AppFrame>{children}</AppFrame>
        </VisionGuardNotificationsProvider>
      </VisionGuardThemeProvider>
    </VisionGuardAuthProvider>
  );
}
