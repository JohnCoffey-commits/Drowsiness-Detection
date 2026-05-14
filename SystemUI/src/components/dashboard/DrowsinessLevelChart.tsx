"use client";

import { Card } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { useEffect, useState } from "react";
import { formatHM } from "@/lib/mockData";

interface DataPoint {
  time: string;
  value: number;
}

const SAMPLES = 60;

const generateData = (now: Date): DataPoint[] => {
  return Array.from({ length: SAMPLES }).map((_, i) => {
    let val =
      25 + Math.sin(i / 6) * 10 + Math.cos(i / 3) * 5 + Math.random() * 8;
    if (i > 42 && i < 50) val += 35 + Math.sin((i - 42) / 2) * 15;
    if (i > 49) val = 40 + Math.random() * 10;

    const minutesAgo = SAMPLES - 1 - i;
    const t = new Date(now.getTime() - minutesAgo * 60 * 1000);

    return {
      time: i === SAMPLES - 1 ? "Now" : formatHM(t),
      value: Math.max(0, Math.min(100, val)),
    };
  });
};

export function DrowsinessLevelChart() {
  const [data, setData] = useState<DataPoint[]>([]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(generateData(new Date()));
  }, []);

  if (data.length === 0) {
    return (
      <Card className="col-span-3 min-h-[260px] rounded-[2rem] border border-slate-200/70 bg-white shadow-sm lg:col-span-2" />
    );
  }

  return (
    <Card className="col-span-3 flex h-full min-h-0 flex-col rounded-[2rem] border border-slate-200/70 bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-md lg:col-span-2">
      <div className="mb-4 flex shrink-0 items-center justify-between">
        <h3 className="text-lg font-bold tracking-tight text-slate-800">
          Drowsiness Level{" "}
          <span className="ml-2 text-sm font-normal text-slate-400">
            (Last 1 Hour)
          </span>
        </h3>
        <div className="flex items-center gap-1.5 text-xs text-slate-500">
          <span className="inline-block h-2 w-2 rounded-full bg-orange-500" />
          <span className="font-medium">Drowsiness score</span>
        </div>
      </div>

      <div className="flex-1 min-h-0 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 10, left: -25, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e2e8f0" />
            <XAxis
              dataKey="time"
              axisLine={{ stroke: "#e2e8f0" }}
              tickLine={false}
              tick={{ fontSize: 12, fill: "#64748b", dy: 10 }}
              minTickGap={40}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 12, fill: "#64748b" }}
              domain={[0, 100]}
              ticks={[0, 50, 100]}
            />
            <Tooltip
              contentStyle={{
                borderRadius: "12px",
                border: "1px solid #e2e8f0",
                boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
              }}
              itemStyle={{ color: "#f97316", fontWeight: "bold" }}
              labelStyle={{ color: "#64748b", marginBottom: "4px" }}
              cursor={{
                stroke: "#cbd5e1",
                strokeWidth: 1,
                strokeDasharray: "4 4",
              }}
            />
            <defs>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" />
                <stop offset="50%" stopColor="#f97316" />
                <stop offset="100%" stopColor="#10b981" />
              </linearGradient>
            </defs>
            <Line
              type="monotone"
              dataKey="value"
              stroke="url(#lineGradient)"
              strokeWidth={3}
              dot={false}
              activeDot={{
                r: 6,
                fill: "#f97316",
                strokeWidth: 2,
                stroke: "#fff",
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
