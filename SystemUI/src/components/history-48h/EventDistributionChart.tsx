"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useChartSize } from "@/components/history-48h/useChartSize";
import type { TimeBucketSummary } from "@/lib/history48hTypes";

interface EventDistributionChartProps {
  data: TimeBucketSummary[];
}

export function EventDistributionChart({ data }: EventDistributionChartProps) {
  const { containerRef, height, isReady, width } = useChartSize<HTMLDivElement>();
  const hasEvents = data.some(
    (point) =>
      point.eyeWarning +
        point.mouthWarning +
        point.highConfidence +
        point.signalUnreliable >
      0
  );

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-base font-bold text-slate-900">
          Warning-candidate events by time block
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Stacked counts for candidate states in the selected window.
        </p>
      </div>
      <div ref={containerRef} className="h-[300px] min-w-0">
        {hasEvents && isReady ? (
          <BarChart
            data={data}
            height={height}
            margin={{ left: 0, right: 10, top: 10, bottom: 0 }}
            width={width}
          >
              <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
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
              <Bar
                dataKey="eyeWarning"
                stackId="candidate"
                name="Eye-warning candidate"
                fill="#f97316"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
              <Bar
                dataKey="mouthWarning"
                stackId="candidate"
                name="Mouth-warning candidate"
                fill="#ec4899"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
              <Bar
                dataKey="highConfidence"
                stackId="candidate"
                name="High-confidence warning candidate"
                fill="#ef4444"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
              <Bar
                dataKey="signalUnreliable"
                stackId="candidate"
                name="Signal unreliable"
                fill="#64748b"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
          </BarChart>
        ) : (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-sm font-medium text-slate-500">
            {hasEvents
              ? "Preparing event distribution chart."
              : "No warning-candidate events in the selected window."}
          </div>
        )}
      </div>
    </section>
  );
}
