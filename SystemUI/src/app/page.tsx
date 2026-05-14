"use client";

import { useCallback, useState } from "react";
import { TopBar } from "@/components/dashboard/TopBar";
import { LiveVideoCard } from "@/components/dashboard/LiveVideoCard";
import { StatusMetricCard } from "@/components/dashboard/StatusMetricCard";
import { DrowsinessRiskCard } from "@/components/dashboard/DrowsinessRiskCard";
import { DrowsinessLevelChart } from "@/components/dashboard/DrowsinessLevelChart";
import { RecentEventsList } from "@/components/dashboard/RecentEventsList";
import { dashboardData } from "@/lib/mockData";
import {
  IDLE_LIVE_MONITOR_RISK_STATE,
  type LiveMonitorRiskState,
} from "@/lib/liveMonitorRiskUtils";

const LIVE_MONITOR_WARNING =
  "This output is a realtime rule-based warning-candidate analysis, not final system-level drowsiness accuracy.";

export default function Dashboard() {
  const { status } = dashboardData;
  const [riskState, setRiskState] = useState<LiveMonitorRiskState>(
    IDLE_LIVE_MONITOR_RISK_STATE
  );
  const handleRiskStateChange = useCallback((nextRiskState: LiveMonitorRiskState) => {
    setRiskState(nextRiskState);
  }, []);

  return (
    <>
      <TopBar />
      <main className="flex-1 overflow-hidden px-4 py-4 lg:px-6 lg:py-4 flex flex-col min-h-0">
        <div className="mx-auto w-full max-w-[1600px] flex-1 flex flex-col min-h-0 gap-4 xl:gap-5">
          <div className="grid grid-cols-1 gap-4 xl:gap-5 lg:grid-cols-[1.35fr_1fr] flex-[1.35] min-h-0">
            <div className="h-full min-h-0 overflow-hidden">
              <LiveVideoCard onRiskStateChange={handleRiskStateChange} />
            </div>

            <div className="flex flex-col gap-4 xl:gap-5 min-h-0">
              <div className="grid grid-cols-1 gap-4 xl:gap-5 sm:grid-cols-2 shrink-0">
                <StatusMetricCard type="closed" events={status.eyesClosed} />
                <StatusMetricCard type="yawn" events={status.yawn} />
              </div>
              <div className="flex-1 min-h-0 overflow-hidden">
                <DrowsinessRiskCard riskState={riskState} />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 xl:gap-5 lg:grid-cols-[1.35fr_1fr] flex-[0.85] min-h-0">
            <div className="h-full min-h-0 overflow-hidden">
              <DrowsinessLevelChart />
            </div>
            <div className="h-full min-h-0 overflow-hidden">
              <RecentEventsList />
            </div>
          </div>
          <p className="shrink-0 px-1 text-[11px] leading-4 text-slate-500">
            {LIVE_MONITOR_WARNING}
          </p>
        </div>
      </main>
    </>
  );
}
