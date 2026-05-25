export const PERMANENT_WARNING =
  "This video analysis shows rule-based fatigue-related alert candidates based on visual evidence. It is intended for awareness and evidence review, not medical diagnosis, final system-level drowsiness accuracy, or a guarantee of driving safety.";

export type FusionState =
  | "normal"
  | "eye_warning_candidate"
  | "mouth_warning_candidate"
  | "high_confidence_drowsiness_candidate"
  | "signal_unreliable";

export type WarningIntervalSource =
  | "high_confidence_intervals"
  | "eye_warning_intervals"
  | "mouth_warning_intervals"
  | "signal_unreliable_intervals";

export interface WarningInterval {
  start_frame_index?: number;
  end_frame_index?: number;
  start_timestamp_sec?: number;
  end_timestamp_sec?: number;
  duration_sec?: number;
  duration_sampled_frames?: number;
  sampled_frames?: number;
  max_p_eye_closed?: number;
  mean_p_eye_closed?: number;
  min_p_eye_closed?: number;
  max_p_yawn?: number;
  mean_p_yawn?: number;
  weak_eye_evidence_frames?: number;
  moderate_eye_evidence_frames?: number;
  strong_eye_evidence_frames?: number;
  moderate_or_strong_eye_evidence_frames?: number;
  dominant_eye_evidence_strength?: string;
  dominant_eye_evidence_level?: string;
  eye_evidence_strength?: string;
  eye_evidence_label?: string;
  eye_evidence_interpretation?: string;
  manual_review_recommended?: boolean;
  eye_warning_strength_reason?: string;
  eye_strength_gate_passed?: boolean;
  eye_strength_gate_reason?: string;
  sustained_eye_warning?: boolean;
  eye_warning_interval_duration_sec?: number;
  eye_warning_interval_sampled_frames?: number;
  high_confidence_suppressed_by_brief_eye_warning?: boolean;
  high_confidence_suppressed_by_weak_eye_evidence?: boolean;
  reason?: string;
}

export interface MergedWarningInterval extends WarningInterval {
  id: string;
  state: Exclude<FusionState, "normal">;
  source: WarningIntervalSource;
}

export interface VideoUploadKeyframe {
  url?: string;
  session_id?: string;
  frame_index?: number;
  timestamp_sec?: number;
  fusion_state?: FusionState | string;
  p_eye_closed?: number;
  p_yawn?: number;
  recent_yawn_event?: boolean;
  warning_type?: string;
  reason?: string;
  segment_id?: number | string;
  is_primary?: boolean;
  sustained_eye_warning?: boolean;
  eye_evidence_level?: string;
  eye_evidence_strength?: string;
  eye_evidence_label?: string;
  eye_evidence_interpretation?: string;
  is_strong_eye_closure_candidate?: boolean;
  is_reduced_eye_openness_candidate?: boolean;
  is_blink_like_candidate?: boolean;
  manual_review_recommended?: boolean;
  eye_warning_strength_reason?: string;
  eye_strength_gate_passed?: boolean;
  eye_strength_gate_reason?: string;
  high_confidence_suppressed_by_brief_eye_warning?: boolean;
  high_confidence_suppressed_by_weak_eye_evidence?: boolean;
}

export interface VideoUploadSummary {
  created_at?: string;
  session_id?: string;
  pipeline_status?: string;
  warning?: string;
  total_frames_sampled?: number;
  duration_sec?: number;
  runtime_sec?: number;
  normal_frames?: number;
  eye_warning_candidate_frames?: number;
  mouth_warning_candidate_frames?: number;
  high_confidence_drowsiness_candidate_frames?: number;
  signal_unreliable_frames?: number;
  yawn_event_count?: number;
  recent_yawn_event_count?: number;
  mean_p_yawn?: number;
  max_p_yawn?: number;
  mean_p_eye_closed?: number;
  max_p_eye_closed?: number;
  first_warning_timestamp_sec?: number | null;
  last_warning_timestamp_sec?: number | null;
  suppressed_high_confidence_brief_eye_warning_frames?: number;
  suppressed_high_confidence_weak_eye_evidence_frames?: number;
  suppressed_high_confidence_weak_eye_warning_frames?: number;
  sustained_eye_gate_min_duration_sec?: number;
  sustained_eye_gate_min_sampled_frames?: number;
  weak_eye_warning_candidate_frames?: number;
  weak_eye_warning_evidence_frames?: number;
  moderate_eye_closure_candidate_frames?: number;
  strong_eye_closure_candidate_frames?: number;
  reduced_eye_openness_candidate_frames?: number;
  manual_review_recommended_eye_frames?: number;
  high_confidence_suppressed_by_weak_eye_evidence_frames?: number;
  stage17_5_eye_evidence_calibration_enabled?: boolean;
  high_confidence_intervals?: WarningInterval[];
  eye_warning_intervals?: WarningInterval[];
  mouth_warning_intervals?: WarningInterval[];
  signal_unreliable_intervals?: WarningInterval[];
  keyframes?: VideoUploadKeyframe[];
}

export interface WarningCounts {
  normal_frames?: number;
  eye_warning_candidate_frames?: number;
  mouth_warning_candidate_frames?: number;
  high_confidence_drowsiness_candidate_frames?: number;
  signal_unreliable_frames?: number;
  weak_eye_warning_evidence_frames?: number;
  moderate_eye_closure_candidate_frames?: number;
  strong_eye_closure_candidate_frames?: number;
  suppressed_high_confidence_weak_eye_evidence_frames?: number;
}

export interface VideoUploadResponse {
  session_id: string;
  status?: string;
  summary: VideoUploadSummary;
  warning_counts?: WarningCounts;
  timeline_url?: string;
  fusion_figure_url?: string;
  keyframes?: VideoUploadKeyframe[];
  report_url?: string;
  warning?: string;
  runtime_duration_sec?: number;
  audit_log?: string;
}

export type BackendStatus =
  | "unchecked"
  | "checking"
  | "connected"
  | "disconnected";

export type AnalysisStatus =
  | "idle"
  | "file-selected"
  | "uploading"
  | "analyzing"
  | "completed"
  | "failed"
  | "backend-unavailable";
