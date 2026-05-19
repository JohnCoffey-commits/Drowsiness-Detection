import type { HistorySeverity } from "@/lib/history48hTypes";

export type BackendArchiveRange = "48h" | "7d" | "30d" | "all";

export type BackendArchiveRecordType =
  | "live_event"
  | "video_run"
  | "session_summary"
  | "manual_note";

export type BackendArchiveSource =
  | "live_monitor"
  | "video_upload"
  | "manual"
  | "demo_seed";

export interface BackendArchiveHealth {
  ok: boolean;
  enabled: boolean;
  db_path: string;
  db_exists: boolean;
  db_writable: boolean;
  record_count: number;
  latest_record_timestamp?: string | null;
  archive_version?: string;
  error?: string;
}

export interface BackendArchiveRecord {
  id: string;
  record_type: BackendArchiveRecordType;
  source: BackendArchiveSource;
  client_id?: string | null;
  account_id?: string | null;
  session_id?: string | null;
  event_type?: string | null;
  severity?: HistorySeverity | string | null;
  title?: string | null;
  summary?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  created_at: string;
  updated_at?: string | null;
  reviewed: boolean;
  review_note?: string | null;
  evidence?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface BackendArchiveRecordsResponse {
  ok: boolean;
  enabled: boolean;
  range: BackendArchiveRange;
  source?: BackendArchiveSource | string | null;
  record_type?: BackendArchiveRecordType | string | null;
  limit: number;
  offset: number;
  total: number;
  records: BackendArchiveRecord[];
}

export interface BackendArchiveExport {
  ok: boolean;
  archive_version: string;
  exported_at: string;
  db_path?: string;
  record_count: number;
  records: BackendArchiveRecord[];
}

export interface BackendArchiveSaveResult {
  ok: boolean;
  record?: BackendArchiveRecord;
  error?: string;
}

export interface LiveArchiveEventPayload {
  id: string;
  client_id: string;
  account_id?: string;
  session_id?: string;
  event_type: string;
  severity: HistorySeverity;
  title?: string;
  summary?: string;
  started_at: string;
  ended_at?: string;
  created_at: string;
  reviewed?: boolean;
  evidence?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface VideoArchiveRunPayload {
  id: string;
  client_id: string;
  account_id?: string;
  session_id?: string;
  event_type: "upload_analysis";
  severity: HistorySeverity;
  title: string;
  summary: string;
  started_at?: string;
  ended_at?: string;
  created_at: string;
  reviewed?: boolean;
  evidence?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}
