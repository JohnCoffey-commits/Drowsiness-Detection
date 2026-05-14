"use client";

import { useState } from "react";
import { Sidebar } from "./Sidebar";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-[#f4f7f9] text-slate-900">
      <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
      <div className="flex-1 min-w-0 flex flex-col h-dvh overflow-hidden transition-all duration-300">
        {children}
      </div>
    </div>
  );
}
