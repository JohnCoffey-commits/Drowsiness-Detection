import type { DriverHistoryEvent, HistorySource } from "@/lib/history48hTypes";

export type InsightEventKind =
  | "eye_warning_candidate"
  | "yawn_warning_candidate"
  | "critical_eye_warning_candidate"
  | "signal_quality_issue";

export interface InsightKindMeta {
  kind: InsightEventKind;
  label: string;
  shortLabel: string;
  color: string;
}

export interface InsightSummary {
  totalAlerts: number;
  dominantAlertLabel: string;
  dominantAlertCount: number;
  dominantAlertShare: number;
  highPriorityShare: number;
  highPriorityCount: number;
  signalInterruptionShare: number;
  signalInterruptionCount: number;
}

export interface InsightTrendPoint {
  timestamp: string;
  label: string;
  eyeWarning: number;
  yawnWarning: number;
  criticalEyeWarning: number;
  signalQualityIssue: number;
}

export interface InsightCompositionItem {
  kind: InsightEventKind;
  label: string;
  count: number;
  share: number;
  color: string;
}

export interface InsightTimeOfDayItem {
  id: "night" | "morning" | "afternoon" | "evening";
  label: string;
  timeRange: string;
  count: number;
  share: number;
}

export interface InsightSessionComparisonRow {
  sessionId: string;
  source: HistorySource;
  startedAt: string;
  endedAt: string;
  driveLabel: string;
  durationLabel: string;
  eventCount: number;
  highPriorityCount: number;
  signalInterruptionCount: number;
  criticalEyeCount: number;
  eyeClosureCount: number;
  yawnCount: number;
  dominantPattern: string;
}

export interface InsightSignalQualitySummary {
  count: number;
  share: number;
  affectedSessionCount: number;
  mostLimitedSessionId: string | null;
  mostLimitedDriveLabel: string | null;
  mostLimitedSessionCount: number;
}

export interface InsightRecommendation {
  id: string;
  title: string;
  body: string;
  priority: "high" | "medium" | "low";
}

export type InsightHistoryRecord = DriverHistoryEvent;
