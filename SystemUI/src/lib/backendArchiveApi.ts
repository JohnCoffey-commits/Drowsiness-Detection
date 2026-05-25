import { buildApiUrl } from "@/lib/apiConfig";
import type {
  BackendArchiveExport,
  BackendArchiveHealth,
  BackendArchiveRange,
  BackendArchiveRecord,
  BackendArchiveRecordsResponse,
  BackendArchiveRecordType,
  BackendArchiveSaveResult,
  BackendArchiveSource,
  LiveArchiveEventPayload,
  LiveArchiveSessionPayload,
  VideoArchiveRunPayload,
} from "@/lib/backendArchiveTypes";
import type {
  DriverHistoryEvent,
  DriverHistorySession,
  EyeEvidenceStrength,
  History48hStore,
  HistorySeverity,
  HistorySource,
  HistoryState,
} from "@/lib/history48hTypes";
import type { VideoUploadResponse, VideoUploadSummary } from "@/lib/videoUploadTypes";

interface ArchiveRecordsQuery {
  range?: BackendArchiveRange;
  source?: BackendArchiveSource;
  recordType?: BackendArchiveRecordType;
  limit?: number;
  offset?: number;
}

interface VideoArchiveFileMetadata {
  filename?: string;
  fileSizeBytes?: number;
  mimeType?: string;
  browserDurationSec?: number;
  figureCount?: number;
}

function jsonHeaders(): HeadersInit {
  return { "Content-Type": "application/json" };
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown; error?: unknown };
    const detail = payload.detail ?? payload.error;
    return typeof detail === "string" ? detail : JSON.stringify(detail ?? payload);
  } catch {
    return `HTTP ${response.status}`;
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(buildApiUrl(path), {
    cache: "no-store",
    ...init,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as T;
}

export async function getArchiveHealth(): Promise<BackendArchiveHealth> {
  return fetchJson<BackendArchiveHealth>("/api/archive/health");
}

export async function getArchiveRecords({
  range = "48h",
  source,
  recordType,
  limit = 200,
  offset = 0,
}: ArchiveRecordsQuery = {}): Promise<BackendArchiveRecordsResponse> {
  const params = new URLSearchParams({
    range,
    limit: String(limit),
    offset: String(offset),
  });
  if (source) params.set("source", source);
  if (recordType) params.set("record_type", recordType);
  return fetchJson<BackendArchiveRecordsResponse>(
    `/api/archive/records?${params.toString()}`,
  );
}

export async function saveLiveArchiveEvent(
  record: LiveArchiveEventPayload,
): Promise<BackendArchiveSaveResult> {
  try {
    return await fetchJson<BackendArchiveSaveResult>("/api/archive/live-event", {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify(record),
    });
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Archive save failed.",
    };
  }
}

export async function saveLiveArchiveSession(
  record: LiveArchiveSessionPayload,
): Promise<BackendArchiveSaveResult> {
  try {
    return await fetchJson<BackendArchiveSaveResult>("/api/archive/live-session", {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify(record),
    });
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Archive session save failed.",
    };
  }
}

export async function saveVideoArchiveRun(
  record: VideoArchiveRunPayload,
): Promise<BackendArchiveSaveResult> {
  try {
    return await fetchJson<BackendArchiveSaveResult>("/api/archive/video-run", {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify(record),
    });
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Archive save failed.",
    };
  }
}

export async function updateArchiveRecordReview(
  recordId: string,
  payload: { reviewed?: boolean; review_note?: string },
): Promise<BackendArchiveSaveResult> {
  try {
    return await fetchJson<BackendArchiveSaveResult>(
      `/api/archive/records/${encodeURIComponent(recordId)}/review`,
      {
        method: "PATCH",
        headers: jsonHeaders(),
        body: JSON.stringify(payload),
      },
    );
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Archive review update failed.",
    };
  }
}

export async function exportArchiveRecords(): Promise<BackendArchiveExport> {
  return fetchJson<BackendArchiveExport>("/api/archive/export");
}

function eventTypeFromHistoryState(state: HistoryState): string {
  switch (state) {
    case "high_confidence_drowsiness_candidate":
      return "critical_eye_warning";
    case "eye_warning_candidate":
      return "eye_warning";
    case "mouth_warning_candidate":
      return "yawn_warning";
    case "signal_unreliable":
      return "signal_quality";
    case "normal":
      return "normal";
  }
}

export function buildLiveArchiveEventPayload(
  event: DriverHistoryEvent,
  clientId: string,
  accountId?: string,
): LiveArchiveEventPayload {
  return {
    id: `live-${event.id}`,
    client_id: clientId,
    account_id: accountId,
    session_id: event.sessionId,
    event_type: eventTypeFromHistoryState(event.state),
    severity: event.severity,
    title: event.title,
    summary: event.summary || event.reason,
    started_at: event.timestamp,
    ended_at: event.endTimestamp,
    created_at: event.timestamp,
    reviewed: event.reviewStatus === "reviewed",
    evidence: {
      p_eye_closed_max: event.pEyeClosedMax,
      p_yawn_max: event.pYawnMax,
      candidate_severity_score: event.candidateSeverityScore,
      eye_evidence_strength: event.eyeEvidenceStrength,
    },
    metadata: {
      history_state: event.state,
      source_event_id: event.sourceEventId,
      ingestion_key: event.ingestionKey,
      related_route: event.relatedRoute,
      review_status: event.reviewStatus,
    },
  };
}

export function buildLiveArchiveSessionPayload(
  session: DriverHistorySession,
  clientId: string,
  accountId?: string,
): LiveArchiveSessionPayload {
  return {
    id: `live-session-${session.id}`,
    client_id: clientId,
    account_id: accountId,
    session_id: session.id,
    event_type: "drive_session",
    severity: "low",
    title: "Live Monitor drive",
    summary: `Live Monitor drive recorded from ${session.startedAt} to ${session.endedAt}.`,
    started_at: session.startedAt,
    ended_at: session.endedAt,
    created_at: session.startedAt,
    reviewed: true,
    evidence: {},
    metadata: {
      history_record_type: "drive_session",
      status: session.status,
      duration_min: session.durationMin,
      normal_count: session.normalCount,
      eye_warning_count: session.eyeWarningCount,
      mouth_warning_count: session.mouthWarningCount,
      high_confidence_count: session.highConfidenceCount,
      signal_unreliable_count: session.signalUnreliableCount,
      review_pending_count: session.reviewPendingCount,
    },
  };
}

function numeric(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function positiveCount(value: unknown): number {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function countIntervals(summary: VideoUploadSummary): number {
  return [
    summary.high_confidence_intervals,
    summary.eye_warning_intervals,
    summary.mouth_warning_intervals,
    summary.signal_unreliable_intervals,
  ].reduce((sum, intervals) => sum + (intervals?.length ?? 0), 0);
}

function inferVideoSeverity(summary: VideoUploadSummary): HistorySeverity {
  if ((summary.high_confidence_drowsiness_candidate_frames ?? 0) > 0) return "high";
  if (
    (summary.eye_warning_candidate_frames ?? 0) > 0 ||
    (summary.mouth_warning_candidate_frames ?? 0) > 0
  ) {
    return "medium";
  }
  if ((summary.signal_unreliable_frames ?? 0) > 0) return "unreliable";
  return "low";
}

function videoSummaryText(summary: VideoUploadSummary): string {
  const high = summary.high_confidence_drowsiness_candidate_frames ?? 0;
  const eye = summary.eye_warning_candidate_frames ?? 0;
  const mouth = summary.mouth_warning_candidate_frames ?? 0;
  const signal = summary.signal_unreliable_frames ?? 0;
  return `Uploaded-video analysis completed: ${eye} eye warning-candidate frames, ${mouth} yawn warning-candidate frames, ${high} critical eye warning-candidate frames, ${signal} signal quality issue frames.`;
}

export function buildVideoArchiveRunPayload(
  response: VideoUploadResponse,
  clientId: string,
  accountId?: string,
  fileMetadata: VideoArchiveFileMetadata = {},
): VideoArchiveRunPayload {
  const summary = response.summary || {};
  const createdAt = summary.created_at || new Date().toISOString();
  const sessionId = response.session_id || summary.session_id;
  const keyframeCount = (response.keyframes || summary.keyframes || []).length;
  return {
    id: `video-${sessionId || createdAt}`,
    client_id: clientId,
    account_id: accountId,
    session_id: sessionId,
    event_type: "upload_analysis",
    severity: inferVideoSeverity(summary),
    title: "Uploaded-video analysis summary",
    summary: videoSummaryText(summary),
    started_at: createdAt,
    ended_at: createdAt,
    created_at: createdAt,
    reviewed: false,
    evidence: {
      max_p_eye_closed: summary.max_p_eye_closed,
      mean_p_eye_closed: summary.mean_p_eye_closed,
      max_p_yawn: summary.max_p_yawn,
      mean_p_yawn: summary.mean_p_yawn,
      eye_warning_candidate_frames: summary.eye_warning_candidate_frames,
      mouth_warning_candidate_frames: summary.mouth_warning_candidate_frames,
      high_confidence_drowsiness_candidate_frames:
        summary.high_confidence_drowsiness_candidate_frames,
      signal_unreliable_frames: summary.signal_unreliable_frames,
    },
    metadata: {
      filename: fileMetadata.filename,
      file_size_bytes: fileMetadata.fileSizeBytes,
      mime_type: fileMetadata.mimeType,
      browser_duration_sec: fileMetadata.browserDurationSec,
      sampled_frames: summary.total_frames_sampled,
      analyzed_duration_sec: summary.duration_sec,
      warning_counts: response.warning_counts,
      interval_count: countIntervals(summary),
      keyframe_count: keyframeCount,
      figure_count: fileMetadata.figureCount,
      pipeline_status: response.status || summary.pipeline_status,
      runtime_duration_sec: response.runtime_duration_sec,
    },
  };
}

function severityFromRecord(record: BackendArchiveRecord): HistorySeverity {
  if (
    record.severity === "low" ||
    record.severity === "medium" ||
    record.severity === "high" ||
    record.severity === "unreliable"
  ) {
    return record.severity;
  }
  return "medium";
}

function stateFromRecord(record: BackendArchiveRecord): HistoryState {
  const metadataState = record.metadata?.history_state;
  if (
    metadataState === "eye_warning_candidate" ||
    metadataState === "mouth_warning_candidate" ||
    metadataState === "high_confidence_drowsiness_candidate" ||
    metadataState === "signal_unreliable" ||
    metadataState === "normal"
  ) {
    return metadataState;
  }
  if (record.event_type === "critical_eye_warning") {
    return "high_confidence_drowsiness_candidate";
  }
  if (record.event_type === "eye_warning") return "eye_warning_candidate";
  if (record.event_type === "yawn_warning") return "mouth_warning_candidate";
  if (record.event_type === "signal_quality") return "signal_unreliable";
  if (record.event_type === "upload_analysis") {
    if (
      numeric(record.evidence?.high_confidence_drowsiness_candidate_frames) &&
      numeric(record.evidence?.high_confidence_drowsiness_candidate_frames)! > 0
    ) {
      return "high_confidence_drowsiness_candidate";
    }
    if (
      numeric(record.evidence?.eye_warning_candidate_frames) &&
      numeric(record.evidence?.eye_warning_candidate_frames)! > 0
    ) {
      return "eye_warning_candidate";
    }
    if (
      numeric(record.evidence?.mouth_warning_candidate_frames) &&
      numeric(record.evidence?.mouth_warning_candidate_frames)! > 0
    ) {
      return "mouth_warning_candidate";
    }
    if (
      numeric(record.evidence?.signal_unreliable_frames) &&
      numeric(record.evidence?.signal_unreliable_frames)! > 0
    ) {
      return "signal_unreliable";
    }
  }
  return "normal";
}

function sourceFromRecord(record: BackendArchiveRecord): HistorySource {
  if (record.source === "video_upload") return "video_upload";
  if (record.source === "live_monitor") return "live_monitor";
  if (record.source === "manual") return "manual";
  return "mock";
}

function eyeEvidenceFromRecord(record: BackendArchiveRecord): EyeEvidenceStrength {
  const value = record.evidence?.eye_evidence_strength;
  if (
    value === "none" ||
    value === "weak" ||
    value === "moderate" ||
    value === "strong" ||
    value === "unknown"
  ) {
    return value;
  }
  if (stateFromRecord(record) === "eye_warning_candidate") return "moderate";
  if (stateFromRecord(record) === "high_confidence_drowsiness_candidate") return "strong";
  if (stateFromRecord(record) === "mouth_warning_candidate") return "none";
  return "unknown";
}

export function archiveRecordToHistoryEvent(
  record: BackendArchiveRecord,
): DriverHistoryEvent {
  const timestamp = record.started_at || record.created_at;
  const durationSec =
    typeof record.metadata?.analyzed_duration_sec === "number"
      ? record.metadata.analyzed_duration_sec
      : record.ended_at
        ? Math.max(
            0,
            (new Date(record.ended_at).getTime() - new Date(timestamp).getTime()) /
              1_000,
          )
        : record.record_type === "video_run"
          ? 0
          : 15;
  const source = sourceFromRecord(record);
  return {
    id: record.id,
    userId: record.account_id || undefined,
    sessionId: record.session_id || record.id,
    sourceEventId: String(record.metadata?.source_event_id || record.id),
    ingestionKey: String(record.metadata?.ingestion_key || `archive:${record.id}`),
    timestamp,
    endTimestamp: record.ended_at || undefined,
    durationSec,
    state: stateFromRecord(record),
    severity: severityFromRecord(record),
    source,
    archiveSource: "backend_archive",
    title: record.title || undefined,
    summary: record.summary || undefined,
    relatedRoute: source === "video_upload" ? "/video-upload" : "/",
    pEyeClosedMax: numeric(record.evidence?.p_eye_closed_max ?? record.evidence?.max_p_eye_closed),
    pYawnMax: numeric(record.evidence?.p_yawn_max ?? record.evidence?.max_p_yawn),
    candidateSeverityScore: numeric(record.evidence?.candidate_severity_score),
    eyeEvidenceStrength: eyeEvidenceFromRecord(record),
    reason: record.summary || record.title || "Backend archive summary record.",
    reviewStatus: record.reviewed ? "reviewed" : "pending",
  };
}

function timestampMs(value?: string | null): number {
  const timestamp = value ? new Date(value).getTime() : 0;
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function sessionDurationMin(startedAt: string, endedAt: string): number {
  const startMs = timestampMs(startedAt);
  const endMs = timestampMs(endedAt);
  if (!startMs || !endMs || endMs < startMs) return 0;
  return Math.round(((endMs - startMs) / 60_000) * 10) / 10;
}

function sessionStatusFromRecord(
  record: BackendArchiveRecord,
): DriverHistorySession["status"] {
  const status = record.metadata?.status;
  if (status === "completed" || status === "partial" || status === "demo") {
    return status;
  }
  return "completed";
}

function archiveRecordToHistorySession(
  record: BackendArchiveRecord,
): DriverHistorySession | null {
  const sessionId = record.session_id || record.id.replace(/^live-session-/, "");
  const startedAt = record.started_at || record.created_at;
  const endedAt = record.ended_at || startedAt;
  if (!sessionId || !startedAt || !endedAt) return null;

  const durationMin =
    numeric(record.metadata?.duration_min) ?? sessionDurationMin(startedAt, endedAt);

  return {
    id: sessionId,
    userId: record.account_id || undefined,
    source: "live_monitor",
    startedAt,
    endedAt,
    durationMin,
    status: sessionStatusFromRecord(record),
    normalCount: positiveCount(record.metadata?.normal_count),
    eyeWarningCount: positiveCount(record.metadata?.eye_warning_count),
    mouthWarningCount: positiveCount(record.metadata?.mouth_warning_count),
    highConfidenceCount: positiveCount(record.metadata?.high_confidence_count),
    signalUnreliableCount: positiveCount(record.metadata?.signal_unreliable_count),
    reviewPendingCount: positiveCount(record.metadata?.review_pending_count),
  };
}

function mergeSessionStatus(
  a: DriverHistorySession["status"],
  b: DriverHistorySession["status"],
): DriverHistorySession["status"] {
  if (a === "completed" || b === "completed") return "completed";
  if (a === "demo" || b === "demo") return "demo";
  return "partial";
}

function mergeArchiveSession(
  preferredSession: DriverHistorySession,
  fallbackSession: DriverHistorySession,
): DriverHistorySession {
  const startMs = Math.min(
    timestampMs(preferredSession.startedAt),
    timestampMs(fallbackSession.startedAt),
  );
  const endMs = Math.max(
    timestampMs(preferredSession.endedAt),
    timestampMs(fallbackSession.endedAt),
  );
  const startedAt =
    startMs > 0 ? new Date(startMs).toISOString() : preferredSession.startedAt;
  const endedAt =
    endMs > 0 ? new Date(endMs).toISOString() : preferredSession.endedAt;

  return {
    ...fallbackSession,
    ...preferredSession,
    startedAt,
    endedAt,
    durationMin: Math.max(
      preferredSession.durationMin,
      fallbackSession.durationMin,
      sessionDurationMin(startedAt, endedAt),
    ),
    status: mergeSessionStatus(preferredSession.status, fallbackSession.status),
    normalCount: Math.max(preferredSession.normalCount, fallbackSession.normalCount),
    eyeWarningCount: Math.max(
      preferredSession.eyeWarningCount,
      fallbackSession.eyeWarningCount,
    ),
    mouthWarningCount: Math.max(
      preferredSession.mouthWarningCount,
      fallbackSession.mouthWarningCount,
    ),
    highConfidenceCount: Math.max(
      preferredSession.highConfidenceCount,
      fallbackSession.highConfidenceCount,
    ),
    signalUnreliableCount: Math.max(
      preferredSession.signalUnreliableCount,
      fallbackSession.signalUnreliableCount,
    ),
    reviewPendingCount: Math.max(
      preferredSession.reviewPendingCount,
      fallbackSession.reviewPendingCount,
    ),
  };
}

function mergeArchiveSessions(
  preferredSessions: DriverHistorySession[],
  fallbackSessions: DriverHistorySession[],
): DriverHistorySession[] {
  const sessionMap = new Map<string, DriverHistorySession>();
  const keyForSession = (session: DriverHistorySession) =>
    `${session.userId || "__legacy__"}:${session.id}`;

  for (const session of fallbackSessions) {
    sessionMap.set(keyForSession(session), session);
  }
  for (const session of preferredSessions) {
    const key = keyForSession(session);
    const existing = sessionMap.get(key);
    sessionMap.set(key, existing ? mergeArchiveSession(session, existing) : session);
  }

  return Array.from(sessionMap.values()).sort(
    (a, b) => timestampMs(b.startedAt) - timestampMs(a.startedAt),
  );
}

export function archiveRecordsToHistoryStore(
  records: BackendArchiveRecord[],
): History48hStore {
  const events = records
    .filter((record) => record.record_type === "live_event")
    .map(archiveRecordToHistoryEvent);
  const sessionMap = new Map<string, DriverHistoryEvent[]>();
  for (const event of events) {
    sessionMap.set(event.sessionId, [...(sessionMap.get(event.sessionId) ?? []), event]);
  }
  const eventSessions: DriverHistorySession[] = Array.from(sessionMap.entries()).map(
    ([id, sessionEvents]) => {
      const sorted = [...sessionEvents].sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
      );
      const startedAt = sorted[0]?.timestamp || new Date().toISOString();
      const endedAt = sorted[sorted.length - 1]?.endTimestamp || sorted[sorted.length - 1]?.timestamp || startedAt;
      const durationMin = Math.max(
        0,
        (new Date(endedAt).getTime() - new Date(startedAt).getTime()) / 60_000,
      );
      const count = (state: HistoryState) =>
        sessionEvents.filter((event) => event.state === state).length;
      return {
        id,
        userId: sorted[0]?.userId,
        source: sorted[0]?.source || "live_monitor",
        startedAt,
        endedAt,
        durationMin,
        status: "completed",
        normalCount: count("normal"),
        eyeWarningCount: count("eye_warning_candidate"),
        mouthWarningCount: count("mouth_warning_candidate"),
        highConfidenceCount: count("high_confidence_drowsiness_candidate"),
        signalUnreliableCount: count("signal_unreliable"),
        reviewPendingCount: sessionEvents.filter(
          (event) => event.reviewStatus === "pending",
        ).length,
      };
    },
  );
  const summarySessions = records
    .filter((record) => record.record_type === "session_summary")
    .map(archiveRecordToHistorySession)
    .filter((session): session is DriverHistorySession => Boolean(session));

  return {
    events,
    sessions: mergeArchiveSessions(summarySessions, eventSessions),
    updatedAt: new Date().toISOString(),
  };
}
