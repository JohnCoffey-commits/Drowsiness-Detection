"use client";

import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import {
  IDLE_LIVE_MONITOR_RISK_STATE,
  type LiveMonitorRiskSeverity,
  type LiveMonitorRiskState,
} from "@/lib/liveMonitorRiskUtils";
import { useVisionGuardTheme } from "@/lib/themeStore";
import { cn } from "@/lib/utils";

const riskStyle: Record<
  LiveMonitorRiskSeverity,
  { text: string; needle: string }
> = {
  idle: {
    text: "text-slate-500",
    needle: "#64748b",
  },
  low: {
    text: "text-emerald-500",
    needle: "#10b981",
  },
  medium: {
    text: "text-orange-500",
    needle: "#f97316",
  },
  high: {
    text: "text-red-500",
    needle: "#ef4444",
  },
  critical: {
    text: "text-rose-600",
    needle: "#dc2626",
  },
  signal_quality: {
    text: "text-sky-600",
    needle: "#0284c7",
  },
};

function easeOutCubic(value: number): number {
  return 1 - (1 - value) ** 3;
}

function useAnimatedNumber(target: number, durationMs = 700): number {
  const [value, setValue] = useState(target);
  const valueRef = useRef(target);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    const startValue = valueRef.current;
    const delta = target - startValue;
    const startedAt = performance.now();

    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
    }

    if (delta === 0) {
      setValue(target);
      valueRef.current = target;
      return undefined;
    }

    const tick = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / durationMs);
      const nextValue = startValue + delta * easeOutCubic(progress);

      valueRef.current = nextValue;
      setValue(nextValue);

      if (progress < 1) {
        frameRef.current = requestAnimationFrame(tick);
      } else {
        frameRef.current = null;
      }
    };

    frameRef.current = requestAnimationFrame(tick);

    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [durationMs, target]);

  return value;
}

interface DrowsinessRiskCardProps {
  riskState?: LiveMonitorRiskState;
  variant?: "default" | "prominent";
}

export function DrowsinessRiskCard({
  riskState = IDLE_LIVE_MONITOR_RISK_STATE,
  variant = "default",
}: DrowsinessRiskCardProps) {
  const { theme } = useVisionGuardTheme();
  const animatedScore = useAnimatedNumber(riskState.score);
  const displayScore = Math.round(animatedScore);
  const style = riskStyle[riskState.severity];
  const rotation = (Math.max(0, Math.min(100, animatedScore)) / 100) * 180 - 90;
  const gaugeTrack = theme === "night" ? "#1f2937" : "#f1f5f9";
  const hubColor = theme === "night" ? "#e2e8f0" : "#0f172a";
  const hubDotColor = theme === "night" ? "#020617" : "#ffffff";
  const isProminent = variant === "prominent";

  return (
    <Card
      className={cn(
        "flex h-full min-h-0 flex-col rounded-[2rem] border border-slate-200/70 bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-md lg:p-6 dark:border-slate-800 dark:bg-slate-900 dark:hover:shadow-slate-950/30",
        isProminent && "p-6 lg:p-8"
      )}
    >
      <div className="mb-4 flex shrink-0 items-center justify-between lg:mb-6">
        <h3
          className={cn(
            "font-bold tracking-tight text-slate-800",
            isProminent ? "text-xl sm:text-2xl" : "text-lg"
          )}
        >
          Drowsiness Risk
        </h3>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Realtime
        </span>
      </div>

      <div
        className={cn(
          "flex min-h-0 flex-1 flex-col items-center justify-center gap-4 lg:flex-row lg:gap-8",
          isProminent && "gap-6 lg:flex-col lg:gap-7"
        )}
      >
        <div
          className={cn(
            "relative w-full shrink-0",
            isProminent ? "max-w-[320px]" : "max-w-[200px]"
          )}
        >
          <svg viewBox="0 0 200 120" className="w-full overflow-visible drop-shadow-sm">
            <path
              d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none"
              stroke={gaugeTrack}
              strokeWidth="24"
              strokeLinecap="round"
            />
            <path
              d="M 20 100 A 80 80 0 0 1 70 26"
              fill="none"
              stroke="#10b981"
              strokeWidth="24"
              strokeLinecap="round"
            />
            <path
              d="M 70 26 A 80 80 0 0 1 140 30"
              fill="none"
              stroke="#f97316"
              strokeWidth="24"
            />
            <path
              d="M 140 30 A 80 80 0 0 1 180 100"
              fill="none"
              stroke="#ef4444"
              strokeWidth="24"
              strokeLinecap="round"
            />

            <g
              style={{
                transform: `rotate(${rotation}deg)`,
                transformOrigin: "100px 100px",
                transition: "transform 180ms ease-out",
              }}
            >
              <path d="M 96 100 L 100 18 L 104 100 Z" fill={style.needle} />
              <circle cx="100" cy="100" r="12" fill={hubColor} />
              <circle cx="100" cy="100" r="4" fill={hubDotColor} />
            </g>
          </svg>
        </div>

        <div
          className={cn(
            "flex flex-col items-center text-center",
            isProminent ? "lg:items-center lg:text-center" : "lg:items-start lg:text-left"
          )}
        >
          <div
            className={cn(
              "font-bold leading-none tracking-tight",
              isProminent ? "text-5xl sm:text-6xl" : "text-[2.5rem]",
              style.text
            )}
          >
            {riskState.label}
          </div>
          <div className="mt-2 flex items-baseline gap-1.5">
            <span
              className={cn(
                "font-bold tabular-nums text-slate-700",
                isProminent ? "text-4xl" : "text-2xl"
              )}
            >
              {displayScore}
            </span>
            <span className="text-sm font-medium text-slate-400">/ 100</span>
          </div>
          <div
            className={cn(
              "mt-3 font-medium text-slate-500",
              isProminent ? "text-base" : "text-sm"
            )}
          >
            {riskState.helper}
          </div>
        </div>
      </div>
    </Card>
  );
}
