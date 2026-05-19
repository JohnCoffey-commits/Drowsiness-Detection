"use client";

import { Card } from "@/components/ui/card";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import { useEffect, useMemo, useRef, useState } from "react";
import type { LiveMonitorRiskPoint } from "@/lib/liveMonitorDashboardTypes";
import type { LiveMonitorRiskSeverity } from "@/lib/liveMonitorRiskUtils";
import { useVisionGuardTheme } from "@/lib/themeStore";

type SeverityBand = "low" | "medium" | "high";
type SeverityLine = "idle" | SeverityBand;

interface BaseWaveformPoint {
  timestamp: string;
  timestampMs: number;
  candidateSeverityScore: number;
  displaySeverityScore: number;
  severity: LiveMonitorRiskSeverity;
  severityLabel: string;
}

interface SegmentedWaveformPoint extends BaseWaveformPoint {
  idleScore: number | null;
  lowScore: number | null;
  mediumScore: number | null;
  highScore: number | null;
  areaScore: number | null;
  isIdleOnly: boolean;
}

interface DrowsinessLevelChartProps {
  points: LiveMonitorRiskPoint[];
  now: Date;
  sessionStartedAt: Date;
  isMonitoringActive: boolean;
}

interface ChartSize {
  width: number;
  height: number;
}

const LAST_HOUR_MS = 60 * 60 * 1000;
const CHART_BUCKET_MS = 5_000;
const IDLE_DISPLAY_SCORE = 20;
const LOW_THRESHOLD = 30;
const HIGH_THRESHOLD = 70;

const SEVERITY_COLORS: Record<SeverityLine, string> = {
  idle: "#94a3b8",
  low: "#10b981",
  medium: "#f97316",
  high: "#ef4444",
};

const LIVE_MONITOR_RISK_SEVERITIES = new Set<LiveMonitorRiskSeverity>([
  "idle",
  "low",
  "medium",
  "high",
  "critical",
  "signal_quality",
]);

function formatHM(date: Date): string {
  return [date.getHours(), date.getMinutes()]
    .map((value) => String(value).padStart(2, "0"))
    .join(":");
}

function formatHMS(date: Date): string {
  return [date.getHours(), date.getMinutes(), date.getSeconds()]
    .map((value) => String(value).padStart(2, "0"))
    .join(":");
}

function getPointTimestampMs(point: LiveMonitorRiskPoint): number {
  const directTimestampMs = (point as { timestampMs?: unknown }).timestampMs;
  if (typeof directTimestampMs === "number" && Number.isFinite(directTimestampMs)) {
    return directTimestampMs;
  }

  const parsed = new Date(point.timestamp).getTime();
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function getDisplaySeverityScore(point: LiveMonitorRiskPoint): number {
  const displaySeverityScore = Number(point.displaySeverityScore);
  if (Number.isFinite(displaySeverityScore)) {
    return displaySeverityScore;
  }

  return Number(point.score);
}

function isLiveMonitorRiskSeverity(
  severity: unknown
): severity is LiveMonitorRiskSeverity {
  return (
    typeof severity === "string" &&
    LIVE_MONITOR_RISK_SEVERITIES.has(severity as LiveMonitorRiskSeverity)
  );
}

function getBucketKey(timestampMs: number): number {
  return Math.floor(timestampMs / CHART_BUCKET_MS) * CHART_BUCKET_MS;
}

function bandForDisplayScore(score: number): SeverityBand {
  if (score <= LOW_THRESHOLD) return "low";
  if (score < HIGH_THRESHOLD) return "medium";
  return "high";
}

function labelForPoint(
  severity: LiveMonitorRiskSeverity,
  displaySeverityScore: number
): string {
  if (severity === "signal_quality") {
    return "Signal quality";
  }

  const band = bandForDisplayScore(displaySeverityScore);
  if (band === "low") return "Low";
  if (band === "medium") return "Medium";
  return "High";
}

function emptySegmentedPoint(
  point: BaseWaveformPoint,
  isIdleOnly = false
): SegmentedWaveformPoint {
  return {
    ...point,
    idleScore: isIdleOnly ? IDLE_DISPLAY_SCORE : null,
    lowScore: null,
    mediumScore: null,
    highScore: null,
    areaScore: isIdleOnly ? null : point.displaySeverityScore,
    isIdleOnly,
  };
}

function setScoreForBand(
  point: SegmentedWaveformPoint,
  band: SeverityBand,
  score: number
): SegmentedWaveformPoint {
  if (band === "low") {
    return { ...point, lowScore: score };
  }
  if (band === "medium") {
    return { ...point, mediumScore: score };
  }
  return { ...point, highScore: score };
}

function toSegmentedPoint(point: BaseWaveformPoint): SegmentedWaveformPoint {
  return setScoreForBand(
    emptySegmentedPoint(point),
    bandForDisplayScore(point.displaySeverityScore),
    point.displaySeverityScore
  );
}

function createIdleBasePoint(timestampMs: number): BaseWaveformPoint {
  return {
    timestamp: new Date(timestampMs).toISOString(),
    timestampMs,
    candidateSeverityScore: 0,
    displaySeverityScore: IDLE_DISPLAY_SCORE,
    severity: "idle",
    severityLabel: "Idle",
  };
}

function toIdleSegmentedPoint(timestampMs: number): SegmentedWaveformPoint {
  return emptySegmentedPoint(createIdleBasePoint(timestampMs), true);
}

function bridgePoint(
  previous: BaseWaveformPoint,
  next: BaseWaveformPoint,
  threshold: number,
  previousBand: SeverityBand,
  nextBand: SeverityBand
): SegmentedWaveformPoint {
  const scoreDelta = next.displaySeverityScore - previous.displaySeverityScore;
  const progress =
    scoreDelta === 0
      ? 0
      : (threshold - previous.displaySeverityScore) / scoreDelta;
  const timestampMs =
    previous.timestampMs + (next.timestampMs - previous.timestampMs) * progress;
  const basePoint: BaseWaveformPoint = {
    timestamp: new Date(timestampMs).toISOString(),
    timestampMs,
    candidateSeverityScore: next.candidateSeverityScore,
    displaySeverityScore: threshold,
    severity: next.severity,
    severityLabel: labelForPoint(next.severity, threshold),
  };
  return setScoreForBand(
    setScoreForBand(emptySegmentedPoint(basePoint), previousBand, threshold),
    nextBand,
    threshold
  );
}

function nextBandAfterThreshold(
  threshold: number,
  isRising: boolean
): SeverityBand {
  if (threshold === LOW_THRESHOLD) {
    return isRising ? "medium" : "low";
  }

  return isRising ? "high" : "medium";
}

function crossingThresholds(
  previousScore: number,
  nextScore: number
): number[] {
  if (previousScore === nextScore) {
    return [];
  }

  const thresholds =
    previousScore < nextScore
      ? [LOW_THRESHOLD, HIGH_THRESHOLD]
      : [HIGH_THRESHOLD, LOW_THRESHOLD];

  return thresholds.filter((threshold) =>
    previousScore < nextScore
      ? previousScore < threshold && nextScore >= threshold
      : previousScore > threshold && nextScore <= threshold
  );
}

function shouldReplacePoint(
  current: BaseWaveformPoint,
  next: BaseWaveformPoint
): boolean {
  return next.timestampMs >= current.timestampMs;
}

function toWaveformBaseData(
  points: LiveMonitorRiskPoint[],
  startMs: number,
  nowMs: number
): BaseWaveformPoint[] {
  const buckets = new Map<number, BaseWaveformPoint>();

  for (const point of points) {
    const timestampMs = getPointTimestampMs(point);
    const candidateSeverityScore = Number(point.score);
    const displaySeverityScore = getDisplaySeverityScore(point);

    if (
      !Number.isFinite(timestampMs) ||
      timestampMs < startMs ||
      timestampMs > nowMs ||
      !Number.isFinite(candidateSeverityScore) ||
      candidateSeverityScore < 0 ||
      candidateSeverityScore > 100 ||
      !Number.isFinite(displaySeverityScore) ||
      displaySeverityScore < 0 ||
      displaySeverityScore > 100 ||
      !isLiveMonitorRiskSeverity(point.severity)
    ) {
      continue;
    }

    const chartPoint: BaseWaveformPoint = {
      timestamp: new Date(timestampMs).toISOString(),
      timestampMs,
      candidateSeverityScore,
      displaySeverityScore,
      severity: point.severity,
      severityLabel: labelForPoint(point.severity, displaySeverityScore),
    };

    const bucketKey = getBucketKey(timestampMs);
    const current = buckets.get(bucketKey);

    if (!current || shouldReplacePoint(current, chartPoint)) {
      buckets.set(bucketKey, chartPoint);
    }
  }

  return Array.from(buckets.values()).sort(
    (a, b) => a.timestampMs - b.timestampMs
  );
}

function toSegmentedWaveformData(
  points: BaseWaveformPoint[]
): SegmentedWaveformPoint[] {
  if (points.length === 0) {
    return [];
  }

  const segmentedPoints: SegmentedWaveformPoint[] = [
    toSegmentedPoint(points[0]),
  ];

  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1];
    const next = points[index];
    const isRising = previous.displaySeverityScore < next.displaySeverityScore;
    let activeBand = bandForDisplayScore(previous.displaySeverityScore);

    for (const threshold of crossingThresholds(
      previous.displaySeverityScore,
      next.displaySeverityScore
    )) {
      const nextBand = nextBandAfterThreshold(threshold, isRising);
      segmentedPoints.push(
        bridgePoint(previous, next, threshold, activeBand, nextBand)
      );
      activeBand = nextBand;
    }

    segmentedPoints.push(toSegmentedPoint(next));
  }

  return segmentedPoints;
}

function extendActiveTrendToNow(
  points: BaseWaveformPoint[],
  nowMs: number,
  isMonitoringActive: boolean
): BaseWaveformPoint[] {
  if (!isMonitoringActive || points.length === 0) {
    return points;
  }

  const latestPoint = points[points.length - 1];
  if (nowMs - latestPoint.timestampMs < 1_000) {
    return points;
  }

  return [
    ...points,
    {
      ...latestPoint,
      timestamp: new Date(nowMs).toISOString(),
      timestampMs: nowMs,
    },
  ];
}

function mergeSegmentedPoint(
  current: SegmentedWaveformPoint,
  next: SegmentedWaveformPoint
): SegmentedWaveformPoint {
  return {
    ...current,
    ...next,
    idleScore: next.idleScore ?? current.idleScore,
    lowScore: next.lowScore ?? current.lowScore,
    mediumScore: next.mediumScore ?? current.mediumScore,
    highScore: next.highScore ?? current.highScore,
    areaScore: next.areaScore ?? current.areaScore,
    isIdleOnly: current.isIdleOnly && next.isIdleOnly,
  };
}

function withIdleBaseline(
  points: SegmentedWaveformPoint[],
  domainStartMs: number,
  nowMs: number,
  isMonitoringActive: boolean
): SegmentedWaveformPoint[] {
  const byTimestamp = new Map<string, SegmentedWaveformPoint>();
  const addPoint = (point: SegmentedWaveformPoint) => {
    const key = String(point.timestampMs);
    const current = byTimestamp.get(key);
    byTimestamp.set(key, current ? mergeSegmentedPoint(current, point) : point);
  };

  if (points.length === 0) {
    addPoint(toIdleSegmentedPoint(domainStartMs));
    addPoint(toIdleSegmentedPoint(nowMs));
    return Array.from(byTimestamp.values()).sort(
      (a, b) => a.timestampMs - b.timestampMs
    );
  }

  const firstPoint = points[0];
  const latestPoint = points[points.length - 1];

  addPoint(toIdleSegmentedPoint(domainStartMs));
  if (firstPoint.timestampMs > domainStartMs) {
    addPoint(toIdleSegmentedPoint(firstPoint.timestampMs));
  }

  for (const point of points) {
    addPoint(point);
  }

  if (!isMonitoringActive) {
    addPoint(toIdleSegmentedPoint(latestPoint.timestampMs));
    addPoint(toIdleSegmentedPoint(nowMs));
  }

  return Array.from(byTimestamp.values()).sort(
    (a, b) => a.timestampMs - b.timestampMs
  );
}

function buildTimeTicks(domainStartMs: number, nowMs: number): number[] {
  const durationMs = Math.max(1, nowMs - domainStartMs);
  if (durationMs < 30_000) {
    return [nowMs];
  }

  const tickCount =
    durationMs <= 2 * 60 * 1000
      ? 2
      : durationMs <= 10 * 60 * 1000
        ? 3
        : durationMs <= 25 * 60 * 1000
        ? 4
        : durationMs <= 45 * 60 * 1000
          ? 6
          : 7;
  const interval = durationMs / (tickCount - 1);

  const ticks = Array.from({ length: tickCount }, (_, index) => {
    if (index === tickCount - 1) {
      return nowMs;
    }
    return domainStartMs + index * interval;
  });

  return Array.from(new Set(ticks.map((tick) => Math.round(tick)))).sort(
    (a, b) => a - b
  );
}

function formatWindowLabel(windowDurationMs: number): string {
  if (windowDurationMs >= LAST_HOUR_MS - 1_000) {
    return "Last 1 Hour";
  }

  if (windowDurationMs < 60_000) {
    return "Last <1 Min";
  }

  const minutes = Math.max(1, Math.floor(windowDurationMs / 60_000));
  return `Last ${minutes} Min`;
}

function WaveformTooltip({
  active,
  payload,
  label,
}: TooltipContentProps) {
  const point = payload?.[0]?.payload as SegmentedWaveformPoint | undefined;
  const timestamp = typeof label === "number" ? label : Number(label);

  if (!active || !point || !Number.isFinite(timestamp)) {
    return null;
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white/95 px-3 py-2 text-xs shadow-lg shadow-slate-950/10">
      <div className="font-medium text-slate-500">
        {formatHMS(new Date(timestamp))}
      </div>
      <div className="mt-1 font-semibold text-slate-800">
        Display severity score: {Math.round(point.displaySeverityScore)}
      </div>
      <div className="mt-0.5 text-slate-500">
        Severity: {point.severityLabel}
      </div>
    </div>
  );
}

function IdleChartPlaceholder() {
  return (
    <div className="relative h-full min-h-[180px] overflow-hidden rounded-2xl border border-slate-100 bg-slate-50/40">
      <div className="absolute inset-x-4 top-[72%] border-t-2 border-slate-300/80" />
    </div>
  );
}

export function DrowsinessLevelChart({
  points,
  now,
  sessionStartedAt,
  isMonitoringActive,
}: DrowsinessLevelChartProps) {
  const { theme } = useVisionGuardTheme();
  const isNight = theme === "night";
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const [chartSize, setChartSize] = useState<ChartSize | null>(null);
  const parsedNowMs = now.getTime();
  const nowMs = Number.isFinite(parsedNowMs) ? parsedNowMs : 0;
  const parsedSessionStartedMs = sessionStartedAt.getTime();
  const sessionStartedMs =
    Number.isFinite(parsedSessionStartedMs) && parsedSessionStartedMs <= nowMs
      ? parsedSessionStartedMs
      : nowMs;
  const domainStart = Math.max(sessionStartedMs, nowMs - LAST_HOUR_MS);
  const chartEndMs = nowMs > domainStart ? nowMs : domainStart + 1;
  const windowDurationMs = chartEndMs - domainStart;
  const windowLabel = formatWindowLabel(windowDurationMs);
  const timeTicks = useMemo(
    () => buildTimeTicks(domainStart, chartEndMs),
    [domainStart, chartEndMs]
  );
  const data = useMemo(() => {
    const baseData = toWaveformBaseData(points, domainStart, chartEndMs);
    const activeBaseData = extendActiveTrendToNow(
      baseData,
      chartEndMs,
      isMonitoringActive
    );
    return withIdleBaseline(
      toSegmentedWaveformData(activeBaseData),
      domainStart,
      chartEndMs,
      isMonitoringActive
    );
  }, [domainStart, points, chartEndMs, isMonitoringActive]);
  const gridColor = isNight ? "#334155" : "#e2e8f0";
  const tickColor = isNight ? "#94a3b8" : "#64748b";
  const cursorColor = isNight ? "#475569" : "#cbd5e1";

  useEffect(() => {
    const element = chartContainerRef.current;
    if (!element) {
      return;
    }

    let animationFrame = 0;
    const updateChartSize = () => {
      const rect = element.getBoundingClientRect();
      const width = Math.floor(rect.width);
      const height = Math.max(180, Math.floor(rect.height));

      if (width <= 0) {
        return;
      }

      setChartSize((current) =>
        current?.width === width && current.height === height
          ? current
          : { width, height }
      );
    };

    const scheduleChartSizeUpdate = () => {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = window.requestAnimationFrame(updateChartSize);
    };

    scheduleChartSizeUpdate();
    const resizeObserver = new ResizeObserver(scheduleChartSizeUpdate);
    resizeObserver.observe(element);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <Card className="col-span-3 flex h-full min-h-0 flex-col rounded-[2rem] border border-slate-200/70 bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-md lg:col-span-2 dark:border-slate-800 dark:bg-slate-900 dark:hover:shadow-slate-950/30">
      <div className="mb-4 flex shrink-0 flex-wrap items-start justify-between gap-3">
        <h3 className="text-lg font-bold tracking-tight text-slate-800">
          Drowsiness Level{" "}
          <span className="ml-2 text-sm font-normal text-slate-400">
            ({windowLabel})
          </span>
        </h3>

        <div className="flex items-center gap-4 text-sm font-semibold">
          <span className="inline-flex items-center gap-1.5 text-slate-500">
            <span className="h-2.5 w-2.5 rounded-full bg-slate-400 shadow-sm shadow-slate-400/20" />
            Idle
          </span>
          <span className="inline-flex items-center gap-1.5 text-emerald-600">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/30" />
            Low
          </span>
          <span className="inline-flex items-center gap-1.5 text-orange-600">
            <span className="h-2.5 w-2.5 rounded-full bg-orange-500 shadow-sm shadow-orange-500/30" />
            Medium
          </span>
          <span className="inline-flex items-center gap-1.5 text-red-600">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500 shadow-sm shadow-red-500/30" />
            High
          </span>
        </div>
      </div>

      <div ref={chartContainerRef} className="min-h-0 w-full flex-1">
        {!chartSize ? (
          <IdleChartPlaceholder />
        ) : (
          <ComposedChart
            width={chartSize.width}
            height={chartSize.height}
            data={data}
            margin={{ top: 8, right: 14, left: -4, bottom: 0 }}
          >
              <defs>
                <linearGradient
                  id="stateDerivedSeverityFill"
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.18} />
                  <stop offset="30%" stopColor="#f97316" stopOpacity={0.13} />
                  <stop offset="70%" stopColor="#10b981" stopOpacity={0.09} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
                </linearGradient>
              </defs>

              <CartesianGrid
                strokeDasharray="4 4"
                vertical={false}
                stroke={gridColor}
              />
              <ReferenceLine
                y={LOW_THRESHOLD}
                stroke={SEVERITY_COLORS.low}
                strokeDasharray="5 7"
                strokeOpacity={0.45}
              />
              <ReferenceLine
                y={HIGH_THRESHOLD}
                stroke={SEVERITY_COLORS.medium}
                strokeDasharray="5 7"
                strokeOpacity={0.45}
              />
              <ReferenceLine
                y={100}
                stroke={SEVERITY_COLORS.high}
                strokeDasharray="5 7"
                strokeOpacity={0.35}
              />
              <XAxis
                dataKey="timestampMs"
                type="number"
                scale="time"
                domain={[domainStart, chartEndMs]}
                ticks={timeTicks}
                tickFormatter={(value) => {
                  const timestamp =
                    typeof value === "number" ? value : Number(value);
                  return Math.abs(chartEndMs - timestamp) < 60_000
                    ? "Now"
                    : formatHM(new Date(timestamp));
                }}
                axisLine={{ stroke: gridColor }}
                tickLine={false}
                tick={{ fontSize: 12, fill: tickColor, dy: 10 }}
                minTickGap={30}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 12, fill: tickColor }}
                domain={[0, 100]}
                ticks={[0, 30, 70, 100]}
                width={36}
                allowDecimals={false}
              />
              <Tooltip
                content={(tooltipProps) => (
                  <WaveformTooltip {...tooltipProps} />
                )}
                cursor={{
                  stroke: cursorColor,
                  strokeWidth: 1,
                  strokeDasharray: "4 4",
                }}
              />
              <Area
                type="monotone"
                dataKey="areaScore"
                name="Display severity score"
                stroke="none"
                fill="url(#stateDerivedSeverityFill)"
                isAnimationActive={false}
              />
              <Line
                type="linear"
                dataKey="idleScore"
                name="Idle"
                stroke={SEVERITY_COLORS.idle}
                strokeWidth={2}
                strokeOpacity={0.72}
                dot={false}
                activeDot={false}
                connectNulls={false}
                isAnimationActive
                animationDuration={450}
                animationEasing="ease-out"
              />
              <Line
                type="monotone"
                dataKey="lowScore"
                name="Low"
                stroke={SEVERITY_COLORS.low}
                strokeWidth={3}
                dot={false}
                activeDot={false}
                connectNulls={false}
                isAnimationActive
                animationDuration={450}
                animationEasing="ease-out"
              />
              <Line
                type="monotone"
                dataKey="mediumScore"
                name="Medium"
                stroke={SEVERITY_COLORS.medium}
                strokeWidth={3}
                dot={false}
                activeDot={false}
                connectNulls={false}
                isAnimationActive
                animationDuration={450}
                animationEasing="ease-out"
              />
              <Line
                type="monotone"
                dataKey="highScore"
                name="High"
                stroke={SEVERITY_COLORS.high}
                strokeWidth={3}
                dot={false}
                activeDot={false}
                connectNulls={false}
                isAnimationActive
                animationDuration={450}
                animationEasing="ease-out"
              />
          </ComposedChart>
        )}
      </div>
    </Card>
  );
}
