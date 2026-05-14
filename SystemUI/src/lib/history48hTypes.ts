export type HistoryState =
  | "normal"
  | "eye_warning_candidate"
  | "mouth_warning_candidate"
  | "high_confidence_drowsiness_candidate"
  | "signal_unreliable";

export type HistorySeverity = "low" | "medium" | "high" | "unreliable";

export type HistorySource = "mock" | "video_upload" | "webcam_future";

export type ReviewStatus = "pending" | "reviewed" | "not_required";

export type EyeEvidenceStrength =
  | "none"
  | "weak"
  | "moderate"
  | "strong"
  | "unknown";

export interface DriverHistoryEvent {
  id: string;
  sessionId: string;
  timestamp: string;
  endTimestamp?: string;
  durationSec: number;
  state: HistoryState;
  severity: HistorySeverity;
  source: HistorySource;
  pEyeClosedMax?: number;
  pYawnMax?: number;
  candidateSeverityScore?: number;
  eyeEvidenceStrength?: EyeEvidenceStrength;
  reason: string;
  reviewStatus: ReviewStatus;
}

export interface DriverHistorySession {
  id: string;
  source: HistorySource;
  startedAt: string;
  endedAt: string;
  durationMin: number;
  status: "completed" | "partial" | "demo";
  normalCount: number;
  eyeWarningCount: number;
  mouthWarningCount: number;
  highConfidenceCount: number;
  signalUnreliableCount: number;
  reviewPendingCount: number;
}

export interface History48hStore {
  events: DriverHistoryEvent[];
  sessions: DriverHistorySession[];
  updatedAt: string;
}

export interface TrendPoint {
  timestamp: string;
  label: string;
  score: number | null;
  unreliableCount: number;
}

export interface TimeBucketSummary {
  timestamp: string;
  label: string;
  eyeWarning: number;
  mouthWarning: number;
  highConfidence: number;
  signalUnreliable: number;
}

export interface StateBreakdownItem {
  state: HistoryState;
  label: string;
  count: number;
  percentage: number;
  color: string;
}

export type TimeWindowHours = 6 | 12 | 24 | 48;

export type EventTypeFilter =
  | "all"
  | "eye_warning_candidate"
  | "mouth_warning_candidate"
  | "high_confidence_drowsiness_candidate"
  | "signal_unreliable";

export type ReviewFilter = "all" | "pending" | "reviewed" | "not_required";

export type SourceFilter = "all" | HistorySource;

export interface HistoryFilters {
  timeWindowHours: TimeWindowHours;
  eventType: EventTypeFilter;
  review: ReviewFilter;
  source: SourceFilter;
  sessionId?: string;
}
