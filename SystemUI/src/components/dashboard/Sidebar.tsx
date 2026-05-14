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
      className={`relative z-20 flex h-full flex-col justify-between bg-gradient-to-b from-[#0B1A30] to-[#0A1121] text-white shadow-2xl transition-all duration-300 ease-in-out ${
        collapsed ? "w-[84px]" : "w-[288px]"
      }`}
    >
      <div className="flex flex-col overflow-hidden">
        <div className={`flex items-center ${collapsed ? "justify-center p-6 pt-8" : "justify-between p-6 pt-8"} `}>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-600/90 shadow-lg shadow-blue-900/40 ring-1 ring-blue-400/30">
              <Eye className="h-5 w-5 text-white" strokeWidth={2.4} />
            </div>
            {!collapsed && (
              <div className="leading-tight overflow-hidden whitespace-nowrap">
                <h1 className="text-xl font-bold tracking-tight">VisionGuard</h1>
                <p className="mt-0.5 text-[11px] font-medium text-slate-400">
                  Driver Drowsiness System
                </p>
              </div>
            )}
          </div>
          <button
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className={`text-slate-400 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 rounded-md ${collapsed ? "absolute right-[-14px] top-9 bg-[#0B1A30] p-1 border border-slate-700/50 rounded-full" : ""}`}
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
                className={`group flex items-center ${collapsed ? "justify-center" : "gap-3"} rounded-xl p-3 text-left text-sm font-medium transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0B1A30] ${
                  isActive
                    ? "bg-blue-600 text-white shadow-md shadow-blue-900/40"
                    : "text-slate-300 hover:bg-white/10 hover:text-white"
                }`}
              >
                <Icon size={20} strokeWidth={isActive ? 2.4 : 2} className="shrink-0" />
                {!collapsed && <span className="truncate">{label}</span>}
              </Link>
            );
          })}
        </nav>
      </div>

      <div className={`mx-3 mb-6 shrink-0 rounded-xl border border-slate-700/50 bg-[#11233F] shadow-inner transition-all duration-300 overflow-hidden ${collapsed ? "p-3 flex justify-center" : "p-4"}`}>
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 shrink-0 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
          {!collapsed && <span className="text-sm font-medium whitespace-nowrap">System Status</span>}
        </div>
        {!collapsed && <p className="mt-1.5 text-xs text-slate-400 whitespace-nowrap">All systems operational</p>}
      </div>
    </aside>
  );
}
