"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Eye,
  History,
  LayoutDashboard,
  LineChart,
  PanelLeftClose,
  PanelLeftOpen,
  UploadCloud,
} from "lucide-react";

const NAV_ITEMS = [
  { id: "/", icon: LayoutDashboard, label: "Live Monitor" },
  { id: "/video-upload", icon: UploadCloud, label: "Video Upload Analysis" },
  { id: "/history-48h", icon: History, label: "48h History" },
  { id: "/insights", icon: LineChart, label: "Insights" },
] as const;

interface SidebarProps {
  collapsed: boolean;
  setCollapsed: (v: boolean) => void;
}

export function Sidebar({ collapsed, setCollapsed }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside
      className={`relative z-20 flex h-full flex-col justify-between border-r border-slate-200 bg-white text-slate-900 shadow-xl shadow-slate-950/5 transition-all duration-300 ease-in-out dark:border-slate-800 dark:bg-slate-950 dark:text-white dark:shadow-black/25 ${
        collapsed ? "w-[112px]" : "w-[288px]"
      }`}
    >
      <div className="flex flex-col overflow-hidden">
        <div
          className={`flex ${
            collapsed
              ? "items-center justify-center gap-2 px-4 pb-5 pt-8"
              : "items-center justify-between p-6 pt-8"
          }`}
        >
          <div className="flex items-center gap-3">
            <div
              className={`flex shrink-0 items-center justify-center bg-blue-600/90 shadow-lg shadow-blue-900/20 ring-1 ring-blue-400/30 dark:shadow-blue-950/40 ${
                collapsed ? "h-11 w-11 rounded-2xl" : "h-10 w-10 rounded-xl"
              }`}
            >
              <Eye className="h-5 w-5 text-white" strokeWidth={2.4} />
            </div>
            {!collapsed && (
              <div className="leading-tight overflow-hidden whitespace-nowrap">
                <h1 className="text-xl font-bold tracking-tight">VisionGuard</h1>
                <p className="mt-0.5 text-[11px] font-medium text-slate-500 dark:text-slate-400">
                  Driver Drowsiness System
                </p>
              </div>
            )}
          </div>
          <button
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className={`flex shrink-0 items-center justify-center border border-slate-200 bg-white text-slate-500 shadow-sm transition-all duration-200 hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 dark:border-slate-700/70 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-cyan-300/40 dark:hover:bg-white/10 dark:hover:text-white ${
              collapsed
                ? "h-10 w-10 rounded-2xl"
                : "h-8 w-8 rounded-lg"
            }`}
          >
            {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={20} />}
          </button>
        </div>

        <nav className="mt-2 flex flex-col gap-1 px-3 overflow-y-auto" aria-label="Primary">
          {NAV_ITEMS.map(({ id, icon: Icon, label }) => {
            const isActive = pathname === id || (id !== "/" && pathname?.startsWith(id));
            return (
              <Link
                key={id}
                href={id}
                title={collapsed ? label : undefined}
                aria-current={isActive ? "page" : undefined}
                className={`group flex items-center ${collapsed ? "justify-center" : "gap-3"} rounded-xl p-3 text-left text-sm font-medium transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 focus-visible:ring-offset-2 focus-visible:ring-offset-white dark:focus-visible:ring-offset-slate-950 ${
                  isActive
                    ? "bg-blue-600 !text-white shadow-md shadow-blue-900/20 dark:bg-blue-500/90 dark:!text-white dark:ring-1 dark:ring-blue-300/30"
                    : "text-slate-600 hover:bg-blue-50 hover:text-slate-950 dark:text-slate-300 dark:hover:bg-white/10 dark:hover:text-white"
                }`}
              >
                <Icon
                  size={20}
                  strokeWidth={isActive ? 2.4 : 2}
                  className={`shrink-0 transition-colors ${
                    isActive
                      ? "!text-white"
                      : "text-slate-600 group-hover:text-slate-950 dark:text-slate-300 dark:group-hover:text-white"
                  }`}
                />
                {!collapsed && (
                  <span
                    className={`truncate transition-colors ${
                      isActive ? "!text-white" : ""
                    }`}
                  >
                    {label}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      <div className={`mx-3 mb-6 shrink-0 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 shadow-inner transition-all duration-300 dark:border-slate-800 dark:bg-slate-900 ${collapsed ? "p-3 flex justify-center" : "p-4"}`}>
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 shrink-0 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
          {!collapsed && <span className="text-sm font-medium whitespace-nowrap">System Status</span>}
        </div>
        {!collapsed && <p className="mt-1.5 text-xs text-slate-500 whitespace-nowrap dark:text-slate-400">All systems operational</p>}
      </div>
    </aside>
  );
}
