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
  totalWarningCandidates: number;
  dominantPatternLabel: string;
  dominantPatternShare: number;
  highPriorityShare: number;
  highPriorityCount: number;
  signalQualityBurdenLabel: string;
  signalQualityShare: number;
  signalQualityCount: number;
  reviewCompletionShare: number;
  reviewedCount: number;
  reviewableCount: number;
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
  eventCount: number;
  highPriorityCount: number;
  signalQualityIssueCount: number;
  pendingReviewCount: number;
  dominantPattern: string;
}

export interface InsightSignalQualitySummary {
  count: number;
  share: number;
  affectedSessionCount: number;
  mostLimitedSessionId: string | null;
  mostLimitedSessionCount: number;
}

export interface InsightRecommendation {
  id: string;
  title: string;
  body: string;
  priority: "high" | "medium" | "low";
}

export type InsightHistoryRecord = DriverHistoryEvent;
