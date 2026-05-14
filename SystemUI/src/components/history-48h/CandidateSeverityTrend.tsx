"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useChartSize } from "@/components/history-48h/useChartSize";
import type { TrendPoint } from "@/lib/history48hTypes";

interface CandidateSeverityTrendProps {
  data: TrendPoint[];
}

export function CandidateSeverityTrend({ data }: CandidateSeverityTrendProps) {
  const { containerRef, height, isReady, width } = useChartSize<HTMLDivElement>();
  const hasScore = data.some((point) => point.score != null);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-base font-bold text-slate-900">
          Candidate severity trend
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          UI-level severity trend based on warning-candidate events. Not final
          drowsiness accuracy.
        </p>
      </div>
      <div ref={containerRef} className="h-[300px] min-w-0">
        {hasScore && isReady ? (
          <AreaChart
            data={data}
            height={height}
            margin={{ left: 0, right: 10, top: 10, bottom: 0 }}
            width={width}
          >
              <defs>
                <linearGradient id="severityTrend" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="5%" stopColor="#2563eb" stopOpacity={0.28} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="label"
                minTickGap={28}
                tick={{ fill: "#64748b", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: "#64748b", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={32}
              />
              <Tooltip
                contentStyle={{
                  borderRadius: 12,
                  borderColor: "#dbeafe",
                  boxShadow: "0 12px 30px rgba(15, 23, 42, 0.12)",
                }}
                formatter={(value) => [`${value}`, "Candidate severity score"]}
                labelFormatter={(label) => `Time block: ${label}`}
              />
              <Area
                type="monotone"
                dataKey="score"
                name="Candidate severity score"
                stroke="#2563eb"
                strokeWidth={2.5}
                fill="url(#severityTrend)"
                connectNulls
                isAnimationActive={false}
              />
          </AreaChart>
        ) : (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-sm font-medium text-slate-500">
            {hasScore
              ? "Preparing severity trend chart."
              : "No severity-score events in the selected window."}
          </div>
        )}
      </div>
    </section>
  );
}
