import { appendHistory48hRecord } from "@/lib/history48hStorage";
import type { DriverHistoryEvent } from "@/lib/history48hTypes";
import type { LiveMonitorDashboardEvent } from "@/lib/liveMonitorDashboardTypes";

const LIVE_MONITOR_HISTORY_EVENT_DURATION_SEC = 15;

function addSeconds(timestamp: string, seconds: number): string | undefined {
  const timestampMs = new Date(timestamp).getTime();
  if (!Number.isFinite(timestampMs)) {
    return undefined;
  }

  return new Date(timestampMs + seconds * 1_000).toISOString();
}

function sanitizeId(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]/g, "-");
}

export function mapLiveMonitorEventToHistoryRecord(
  event: LiveMonitorDashboardEvent,
  userId: string
): DriverHistoryEvent | null {
  if (event.kind === "normal") {
    return null;
  }

  const ingestionKey = `live_monitor:${userId}:${event.id}`;
  const common = {
    id: `history-${sanitizeId(ingestionKey)}`,
    userId,
    sessionId: event.sessionId,
    sourceEventId: event.id,
    ingestionKey,
    timestamp: event.timestamp,
    endTimestamp: addSeconds(
      event.timestamp,
      LIVE_MONITOR_HISTORY_EVENT_DURATION_SEC
    ),
    durationSec: LIVE_MONITOR_HISTORY_EVENT_DURATION_SEC,
    source: "live_monitor" as const,
    relatedRoute: "/" as const,
    candidateSeverityScore: event.severityScore,
    reviewStatus: "pending" as const,
  };

  if (event.kind === "critical_eye_warning") {
    return {
      ...common,
      state: "high_confidence_drowsiness_candidate",
      severity: "high",
      title: "Critical eye warning candidate",
      summary:
        "Live Monitor emitted a stable critical eye warning-candidate event.",
      reason:
        "Stable Live Monitor visual alert created a critical eye warning-candidate history item.",
      eyeEvidenceStrength: "strong",
    };
  }

  if (event.kind === "eye_warning") {
    return {
      ...common,
      state: "eye_warning_candidate",
      severity: "high",
      title: "Eye warning candidate",
      summary: "Live Monitor emitted a stable eye warning-candidate event.",
      reason:
        "Stable Live Monitor visual alert created an eye warning-candidate history item.",
      eyeEvidenceStrength: "moderate",
    };
  }

  if (event.kind === "yawn_warning") {
    return {
      ...common,
      state: "mouth_warning_candidate",
      severity: "medium",
      title: "Yawn warning candidate",
      summary: "Live Monitor emitted a stable yawn warning-candidate event.",
      reason:
        "Stable Live Monitor visual alert created a yawn warning-candidate history item.",
      eyeEvidenceStrength: "none",
    };
  }

  return {
    ...common,
    state: "signal_unreliable",
    severity: "unreliable",
    title: "Camera signal quality issue",
    summary: "Live Monitor emitted a stable signal quality issue event.",
    reason:
      "Stable Live Monitor visual alert created a camera signal quality issue history item.",
    eyeEvidenceStrength: "unknown",
  };
}

export function appendLiveMonitorDashboardEventToHistory(
  event: LiveMonitorDashboardEvent,
  userId: string
): DriverHistoryEvent | null {
  const historyRecord = mapLiveMonitorEventToHistoryRecord(event, userId);
  if (!historyRecord) {
    return null;
  }

  appendHistory48hRecord(historyRecord, new Date(historyRecord.timestamp));
  return historyRecord;
}
