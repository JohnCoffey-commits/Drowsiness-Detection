"use client";

import { Cell, Pie, PieChart, Tooltip } from "recharts";
import { useChartSize } from "@/components/history-48h/useChartSize";
import type { StateBreakdownItem } from "@/lib/history48hTypes";
import { formatPercent } from "@/lib/history48hUtils";

interface StateBreakdownChartProps {
  data: StateBreakdownItem[];
}

export function StateBreakdownChart({ data }: StateBreakdownChartProps) {
  const { containerRef, height, isReady, width } = useChartSize<HTMLDivElement>();
  const nonEmptyData = data.filter((item) => item.count > 0);
  const total = data.reduce((sum, item) => sum + item.count, 0);
  const radiusBase = Math.max(48, Math.min(width, height) / 2 - 14);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-base font-bold text-slate-900">State breakdown</h2>
        <p className="mt-1 text-sm text-slate-500">
          Counts and percentages for candidate states in the selected window.
        </p>
      </div>
      <div className="grid gap-4 lg:grid-cols-[220px_1fr] lg:items-center">
        <div ref={containerRef} className="h-[220px] min-w-0">
          {total > 0 && isReady ? (
            <PieChart width={width} height={height}>
                <Pie
                  data={nonEmptyData}
                  dataKey="count"
                  nameKey="label"
                  innerRadius={Math.max(30, radiusBase - 30)}
                  outerRadius={radiusBase}
                  paddingAngle={2}
                  isAnimationActive={false}
                >
                  {nonEmptyData.map((item) => (
                    <Cell key={item.state} fill={item.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    borderColor: "#dbeafe",
                    boxShadow: "0 12px 30px rgba(15, 23, 42, 0.12)",
                  }}
                />
            </PieChart>
          ) : (
            <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-sm font-medium text-slate-500">
              {total > 0 ? "Preparing state breakdown chart." : "No events"}
            </div>
          )}
        </div>
        <div className="space-y-3">
          {data.map((item) => (
            <div key={item.state}>
              <div className="mb-1 flex items-center justify-between gap-3 text-sm">
                <span className="flex min-w-0 items-center gap-2 font-semibold text-slate-700">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="truncate">{item.label}</span>
                </span>
                <span className="shrink-0 text-xs font-semibold text-slate-500">
                  {item.count} / {formatPercent(item.percentage)}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.round(item.percentage * 100)}%`,
                    backgroundColor: item.color,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
