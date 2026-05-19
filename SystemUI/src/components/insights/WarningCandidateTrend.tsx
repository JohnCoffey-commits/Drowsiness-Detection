"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useChartSize } from "@/components/history-48h/useChartSize";
import type { InsightTrendPoint } from "@/lib/insightsTypes";

interface WarningCandidateTrendProps {
  data: InsightTrendPoint[];
}

export function WarningCandidateTrend({ data }: WarningCandidateTrendProps) {
  const { containerRef, height, isReady, width } = useChartSize<HTMLDivElement>();
  const hasEvents = data.some(
    (point) =>
      point.eyeWarning +
        point.yawnWarning +
        point.criticalEyeWarning +
        point.signalQualityIssue >
      0
  );

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-5">
      <div className="mb-4">
        <h2 className="text-base font-black text-slate-950 dark:text-white">
          Warning-candidate trend
        </h2>
        <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
          Stacked local record counts from the current user&apos;s 48h history.
        </p>
      </div>
      <div ref={containerRef} className="h-[320px] min-w-0">
        {hasEvents && isReady ? (
          <BarChart
            data={data}
            height={height}
            margin={{ left: 0, right: 10, top: 10, bottom: 0 }}
            width={width}
          >
            <CartesianGrid
              stroke="#e2e8f0"
              strokeDasharray="3 3"
              vertical={false}
            />
            <XAxis
              dataKey="label"
              minTickGap={28}
              tick={{ fill: "#64748b", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fill: "#64748b", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={28}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 12,
                borderColor: "#dbeafe",
                boxShadow: "0 12px 30px rgba(15, 23, 42, 0.12)",
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12, fontWeight: 700 }} />
            <Bar
              dataKey="eyeWarning"
              stackId="candidate"
              name="Eye warning candidate"
              fill="#f97316"
              radius={[5, 5, 0, 0]}
              isAnimationActive={false}
            />
            <Bar
              dataKey="yawnWarning"
              stackId="candidate"
              name="Yawn warning candidate"
              fill="#ec4899"
              radius={[5, 5, 0, 0]}
              isAnimationActive={false}
            />
            <Bar
              dataKey="criticalEyeWarning"
              stackId="candidate"
              name="Critical eye warning candidate"
              fill="#ef4444"
              radius={[5, 5, 0, 0]}
              isAnimationActive={false}
            />
            <Bar
              dataKey="signalQualityIssue"
              stackId="candidate"
              name="Signal quality issue"
              fill="#64748b"
              radius={[5, 5, 0, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        ) : (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-center text-sm font-bold text-slate-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">
            {hasEvents
              ? "Preparing trend chart."
              : "No warning-candidate records in the selected window."}
          </div>
        )}
      </div>
    </section>
  );
}
