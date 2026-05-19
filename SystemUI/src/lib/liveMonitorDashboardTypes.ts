import type { LiveMonitorRiskSeverity } from "@/lib/liveMonitorRiskUtils";

export type LiveMonitorDashboardEventKind =
  | "normal"
  | "eye_warning"
  | "yawn_warning"
  | "critical_eye_warning"
  | "signal_quality";

export interface LiveMonitorDashboardEvent {
  id: string;
  userId?: string;
  timestamp: string;
  sessionId: string;
  source: "live_monitor_prototype";
  kind: LiveMonitorDashboardEventKind;
  label: string;
  severityScore: number;
}

export interface LiveMonitorDashboardEventDraft {
  id: string;
  timestamp: string;
  kind: LiveMonitorDashboardEventKind;
  label: string;
  severityScore: number;
}

export interface LiveMonitorRiskPoint {
  id: string;
  userId?: string;
  timestamp: string;
  sessionId: string;
  score: number;
  displaySeverityScore: number;
  severity: LiveMonitorRiskSeverity;
}

export interface LiveMonitorDashboardStore {
  events: LiveMonitorDashboardEvent[];
  riskPoints: LiveMonitorRiskPoint[];
  updatedAt: string;
}

export interface LiveMonitorDashboardCounts {
  eyeWarnings: number;
  yawnWarnings: number;
}
