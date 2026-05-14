import {
  type FusionState,
  type MergedWarningInterval,
  PERMANENT_WARNING,
  type VideoUploadKeyframe,
  type VideoUploadResponse,
  type VideoUploadSummary,
  type WarningInterval,
  type WarningIntervalSource,
} from "@/lib/videoUploadTypes";

export const DEFAULT_BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export const MAX_UPLOAD_BYTES = 750 * 1024 * 1024;

const LOCAL_PATH_PATTERN = /\/Users\/[^\s"',)]+/g;

const INTERVAL_SOURCES: Array<{
  key: WarningIntervalSource;
  state: Exclude<FusionState, "normal">;
}> = [
  {
    key: "high_confidence_intervals",
    state: "high_confidence_drowsiness_candidate",
  },
  { key: "eye_warning_intervals", state: "eye_warning_candidate" },
  { key: "mouth_warning_intervals", state: "mouth_warning_candidate" },
  { key: "signal_unreliable_intervals", state: "signal_unreliable" },
];

export function sanitizeBrowserText(value: unknown): string {
  if (value == null) return "";
  const text =
    typeof value === "string" ? value : JSON.stringify(value, null, 2) || "";
  return text
    .replace(LOCAL_PATH_PATTERN, "[local server path hidden]")
    .replace(/\s+/g, " ")
    .trim();
}

export function validateBackendUrl(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return "Backend URL is required.";
  if (trimmed.startsWith("file://")) return "file:// URLs are not allowed.";
  if (!/^https?:\/\//i.test(trimmed)) {
    return "Backend URL must start with http:// or https://.";
  }
  try {
    const url = new URL(trimmed);
    if (!["http:", "https:"].includes(url.protocol)) {
      return "Backend URL must use http:// or https://.";
    }
    return null;
  } catch {
    return "Backend URL is not a valid URL.";
  }
}

export function normalizeBackendUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

export function buildApiUrl(backendUrl: string, path: string): string {
  const base = normalizeBackendUrl(backendUrl);
  const safePath = path.startsWith("/") ? path : `/${path}`;
  return `${base}${safePath}`;
}

export function safeResponsePath(path?: string): string | null {
  if (!path || path.startsWith("file://") || path.includes("/Users/")) {
    return null;
  }
  return path.startsWith("/") ? path : `/${path}`;
}

export function safeBackendLink(
  backendUrl: string,
  path?: string,
): string | null {
  const safePath = safeResponsePath(path);
  return safePath ? buildApiUrl(backendUrl, safePath) : null;
}

export function sessionFilePath(sessionId: string, relativePath: string): string {
  return `/api/runs/${encodeURIComponent(sessionId)}/files/${relativePath}`;
}

export function formatBytes(bytes?: number): string {
  if (bytes == null || Number.isNaN(bytes)) return "Not available";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), 3);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatNumber(value?: number): string {
  if (value == null || Number.isNaN(value)) return "Not available";
  return new Intl.NumberFormat("en-AU").format(value);
}

export function formatSeconds(value?: number): string {
  if (value == null || Number.isNaN(value)) return "Not available";
  return `${value.toFixed(value >= 10 ? 1 : 2)}s`;
}

export function formatProbability(value?: number): string {
  if (value == null || Number.isNaN(value)) return "Not available";
  return value.toFixed(3);
}

export function formatPercent(value?: number): string {
  if (value == null || Number.isNaN(value)) return "Not available";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatOptionalBoolean(
  value: boolean | null | undefined,
  options?: {
    trueLabel?: string;
    falseLabel?: string;
    missingLabel?: string;
  },
): string {
  if (value == null) return options?.missingLabel ?? "Not provided";
  return value ? (options?.trueLabel ?? "Yes") : (options?.falseLabel ?? "No");
}

export function booleanLabel(value?: boolean | null): string {
  return formatOptionalBoolean(value);
}

export function formatOptionalBooleanCompact(
  value: boolean | null | undefined,
): string {
  return formatOptionalBoolean(value, { missingLabel: "—" });
}

export function formatOptionalTextCompact(
  value: string | number | null | undefined,
): string {
  if (value == null) return "—";
  const text = String(value).trim();
  return text || "—";
}

export function fusionStateLabel(state?: string): string {
  switch (state) {
    case "high_confidence_drowsiness_candidate":
      return "High-confidence warning candidate";
    case "eye_warning_candidate":
      return "Eye-warning candidate";
    case "mouth_warning_candidate":
      return "Mouth-warning candidate";
    case "signal_unreliable":
      return "Signal unreliable";
    case "normal":
      return "Normal";
    default:
      return "Warning candidate";
  }
}

export function fusionStateDescription(state?: string): string {
  switch (state) {
    case "high_confidence_drowsiness_candidate":
      return "Recent mouth/yawn evidence overlapped with sustained eye-warning evidence. This is still a warning candidate, not final drowsiness truth.";
    case "eye_warning_candidate":
      return "Eye temporal rule produced an eye-warning candidate. This may reflect reduced eye openness, blink-like activity, possible closure, or ROI-sensitive evidence.";
    case "mouth_warning_candidate":
      return "Recent mouth/yawn evidence is active. This alone is not final drowsiness truth.";
    case "signal_unreliable":
      return "Face/eye ROI signal quality may be unreliable. This is not treated as drowsiness evidence.";
    default:
      return "Backend output prepared for manual warning-candidate review.";
  }
}

export function stateTone(state?: string): string {
  switch (state) {
    case "high_confidence_drowsiness_candidate":
      return "border-red-200 bg-red-50 text-red-700";
    case "eye_warning_candidate":
      return "border-orange-200 bg-orange-50 text-orange-700";
    case "mouth_warning_candidate":
      return "border-pink-200 bg-pink-50 text-pink-700";
    case "signal_unreliable":
      return "border-slate-300 bg-slate-100 text-slate-700";
    case "normal":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    default:
      return "border-blue-200 bg-blue-50 text-blue-700";
  }
}

export function mergeWarningIntervals(
  summary?: VideoUploadSummary,
): MergedWarningInterval[] {
  if (!summary) return [];
  return INTERVAL_SOURCES.flatMap(({ key, state }) => {
    const rows = summary[key] || [];
    return rows.map((interval: WarningInterval, index) => ({
      ...interval,
      id: `${key}-${index}-${interval.start_frame_index ?? "na"}`,
      state,
      source: key,
    }));
  }).sort((a, b) => {
    const startA = a.start_timestamp_sec ?? Number.POSITIVE_INFINITY;
    const startB = b.start_timestamp_sec ?? Number.POSITIVE_INFINITY;
    return startA - startB;
  });
}

export function intervalDuration(interval: WarningInterval): number | undefined {
  if (interval.duration_sec != null) return interval.duration_sec;
  if (
    interval.start_timestamp_sec != null &&
    interval.end_timestamp_sec != null
  ) {
    return Math.max(0, interval.end_timestamp_sec - interval.start_timestamp_sec);
  }
  return undefined;
}

export function intervalFrameCount(interval: WarningInterval): number | undefined {
  return interval.duration_sampled_frames ?? interval.sampled_frames;
}

export function hasStage175SummaryFields(summary?: VideoUploadSummary): boolean {
  if (!summary) return false;
  return [
    summary.weak_eye_warning_candidate_frames,
    summary.weak_eye_warning_evidence_frames,
    summary.moderate_eye_closure_candidate_frames,
    summary.strong_eye_closure_candidate_frames,
    summary.reduced_eye_openness_candidate_frames,
    summary.manual_review_recommended_eye_frames,
    summary.high_confidence_suppressed_by_weak_eye_evidence_frames,
    summary.stage17_5_eye_evidence_calibration_enabled,
  ].some((value) => value != null);
}

export function hasStage175IntervalFields(interval?: WarningInterval): boolean {
  if (!interval) return false;
  return [
    interval.dominant_eye_evidence_strength,
    interval.dominant_eye_evidence_level,
    interval.eye_evidence_strength,
    interval.eye_evidence_label,
    interval.eye_evidence_interpretation,
    interval.eye_warning_strength_reason,
    interval.eye_strength_gate_reason,
    interval.weak_eye_evidence_frames,
    interval.moderate_eye_evidence_frames,
    interval.strong_eye_evidence_frames,
    interval.moderate_or_strong_eye_evidence_frames,
  ].some((value) => hasProvidedValue(value));
}

function hasProvidedString(value: string | null | undefined): boolean {
  return typeof value === "string" && value.trim().length > 0;
}

function hasProvidedValue(value: unknown): boolean {
  if (typeof value === "string") return value.trim().length > 0;
  return value != null;
}

export function hasEyeEvidenceStrengthFields(
  keyframe?: VideoUploadKeyframe,
): boolean {
  if (!keyframe) return false;
  return [
    keyframe.eye_evidence_level,
    keyframe.eye_evidence_strength,
    keyframe.eye_evidence_label,
    keyframe.eye_evidence_interpretation,
    keyframe.eye_warning_strength_reason,
    keyframe.eye_strength_gate_reason,
  ].some(hasProvidedString);
}

export function hasAnyEyeEvidenceStrengthFields(
  keyframes: VideoUploadKeyframe[],
): boolean {
  return keyframes.some(hasEyeEvidenceStrengthFields);
}

export function hasStage175KeyframeFields(
  keyframe?: VideoUploadKeyframe,
): boolean {
  return hasEyeEvidenceStrengthFields(keyframe);
}

export function recentYawnExplanation(
  keyframe: VideoUploadKeyframe,
): string | null {
  if (keyframe.recent_yawn_event !== true) return null;
  if (
    keyframe.p_yawn != null &&
    !Number.isNaN(keyframe.p_yawn) &&
    keyframe.p_yawn >= 0.5
  ) {
    return null;
  }
  return "Recent yawn means a yawn event occurred within the recent temporal window; the current frame may have a low p_yawn.";
}

export function eyeEvidenceBadgeText(strength?: string): string {
  const normalized = strength?.toLowerCase();
  if (!normalized) return "Not provided";
  if (normalized === "none") return "None";
  if (normalized.includes("weak")) return "Weak";
  if (normalized.includes("moderate")) return "Moderate";
  if (normalized.includes("strong")) return "Strong";
  if (normalized.includes("uncertain") || normalized.includes("unreliable")) {
    return "Signal unreliable";
  }
  if (normalized.includes("normal") || normalized.includes("open")) {
    return "Normal-open evidence";
  }
  return strength || "Not provided";
}

export function compactEyeEvidenceLabel(
  strength?: string,
  level?: string,
  providedLabel?: string,
): string {
  const normalizedStrength = strength?.toLowerCase();
  const label = providedLabel?.toLowerCase() || level?.toLowerCase() || "";
  if (normalizedStrength === "none" || label.includes("no calibrated")) {
    return "None";
  }
  if (normalizedStrength?.includes("weak") || label.includes("weak")) {
    return "Weak";
  }
  if (normalizedStrength?.includes("moderate") || label.includes("moderate")) {
    return "Moderate";
  }
  if (normalizedStrength?.includes("strong") || label.includes("strong")) {
    return "Strong";
  }
  if (
    normalizedStrength?.includes("signal_unreliable") ||
    normalizedStrength?.includes("unreliable") ||
    label.includes("unreliable")
  ) {
    return "Signal unreliable";
  }
  const expanded = eyeEvidenceLabel(level, strength, providedLabel);
  return expanded === "Not provided" ? "—" : expanded;
}

export function eyeEvidenceLabel(
  level?: string,
  strength?: string,
  providedLabel?: string,
): string {
  const trimmedLabel = providedLabel?.trim();
  if (trimmedLabel) return trimmedLabel;
  const normalizedStrength = strength?.toLowerCase();
  const normalizedLevel = level?.toLowerCase();
  if (normalizedStrength === "none") {
    return "No calibrated eye-warning evidence";
  }
  if (
    normalizedStrength?.includes("weak") ||
    normalizedLevel?.includes("reduced")
  ) {
    return "Weak - reduced eye openness candidate";
  }
  if (
    normalizedStrength?.includes("moderate") ||
    normalizedLevel?.includes("possible")
  ) {
    return "Moderate - possible eye-closure candidate";
  }
  if (
    normalizedStrength?.includes("strong") ||
    normalizedLevel?.includes("strong")
  ) {
    return "Strong - strong eye-closure candidate";
  }
  if (
    normalizedStrength?.includes("signal_unreliable") ||
    normalizedStrength?.includes("uncertain") ||
    normalizedStrength?.includes("unreliable") ||
    normalizedLevel?.includes("roi")
  ) {
    return "Signal unreliable";
  }
  if (
    normalizedStrength?.includes("normal") ||
    normalizedLevel?.includes("open")
  ) {
    return "Normal-open evidence";
  }
  return "Not provided";
}

export function eyeEvidenceDescription(
  level?: string,
  strength?: string,
  interpretation?: string,
): string {
  const trimmedInterpretation = interpretation?.trim();
  if (trimmedInterpretation) return trimmedInterpretation;
  const label = eyeEvidenceLabel(level, strength);
  switch (label) {
    case "No calibrated eye-warning evidence":
      return "No calibrated eye-warning evidence was provided for this sampled row.";
    case "Weak - reduced eye openness candidate":
      return "Elevated eye-closed probability may reflect reduced eye openness, but not strong closure evidence.";
    case "Moderate - possible eye-closure candidate":
      return "Eye evidence is stronger than weak reduced-openness evidence and should be reviewed manually.";
    case "Strong - strong eye-closure candidate":
      return "Eye evidence is marked as a strong eye-closure candidate for manual review.";
    case "Signal unreliable":
      return "Eye evidence may be affected by ROI or signal quality and should not be treated as proof.";
    case "Normal-open evidence":
      return "The available eye evidence is consistent with normal-open evidence.";
    default:
      return "Evidence not provided by this backend run.";
  }
}

export function manualReviewLabel(
  value?: boolean | null,
): string {
  return formatOptionalBoolean(value);
}

export function stage175MetricScopeNote(): string {
  return "Stage 17.5 calibration fields describe eye-evidence strength, not final fusion-state counts. Weak eye evidence is counted within eye-warning candidate rows in the current backend summary, while moderate and strong eye-evidence counts are across sampled timeline rows. These counts may not equal eye-warning candidate frames because fusion states and eye-evidence calibration are separate backend outputs.";
}

export function keyframeGroupLabel(keyframe: VideoUploadKeyframe): string {
  const stateLabel = fusionStateLabel(keyframe.fusion_state);
  if (stateLabel !== "Warning candidate") return stateLabel;
  switch (keyframe.warning_type) {
    case "high_confidence":
      return "High-confidence warning candidate";
    case "eye_warning":
      return "Eye-warning candidate";
    case "mouth_warning":
      return "Mouth-warning candidate";
    case "signal_unreliable":
      return "Signal unreliable";
    default:
      return "Warning candidate";
  }
}

export function figureDefinitions(
  backendUrl: string,
  response?: VideoUploadResponse,
) {
  const sessionId = response?.session_id;
  if (!sessionId) return [];
  return [
    {
      id: "fusion",
      title: "Fusion timeline",
      description:
        "Rule-based fusion states over sampled time for warning-candidate review.",
      url:
        safeBackendLink(backendUrl, response?.fusion_figure_url) ||
        buildApiUrl(backendUrl, sessionFilePath(sessionId, "figures/fusion_timeline.png")),
    },
    {
      id: "eye",
      title: "p_eye_closed over time",
      description:
        "Eye specialist model probability over sampled frames. This is technical evidence, not a final finding.",
      url: buildApiUrl(
        backendUrl,
        sessionFilePath(sessionId, "figures/p_eye_closed_over_time.png"),
      ),
    },
    {
      id: "yawn",
      title: "p_yawn over time",
      description:
        "Mouth/yawn specialist model probability over sampled frames for uploaded-video analysis.",
      url: buildApiUrl(
        backendUrl,
        sessionFilePath(sessionId, "figures/p_yawn_over_time.png"),
      ),
    },
  ];
}

export function safeKeyframeUrl(
  backendUrl: string,
  keyframe: VideoUploadKeyframe,
): string | null {
  return safeBackendLink(backendUrl, keyframe.url);
}

export function buildCopySummary(response: VideoUploadResponse): string {
  const summary = response.summary || {};
  const lines = [
    "Uploaded-video warning-candidate review summary",
    `Session ID: ${response.session_id}`,
    `Pipeline status: ${response.status || summary.pipeline_status || "Not available"}`,
    `Runtime duration: ${formatSeconds(response.runtime_duration_sec ?? summary.runtime_sec)}`,
    `Sampled frames: ${formatNumber(summary.total_frames_sampled)}`,
    `Analyzed duration: ${formatSeconds(summary.duration_sec)} (sampled timeline timestamps)`,
    "Fusion state counts:",
    `- Normal frames: ${formatNumber(summary.normal_frames)}`,
    `- Eye-warning candidate frames: ${formatNumber(summary.eye_warning_candidate_frames)}`,
    `- Mouth-warning candidate frames: ${formatNumber(summary.mouth_warning_candidate_frames)}`,
    `- High-confidence warning candidate frames: ${formatNumber(summary.high_confidence_drowsiness_candidate_frames)}`,
    `- Signal-unreliable frames: ${formatNumber(summary.signal_unreliable_frames)}`,
    `Yawn event count: ${formatNumber(summary.yawn_event_count)}`,
    `Recent-yawn count: ${formatNumber(summary.recent_yawn_event_count)}`,
    `Suppressed brief-eye escalation frames: ${formatNumber(summary.suppressed_high_confidence_brief_eye_warning_frames)}`,
    `Keyframe count: ${formatNumber((response.keyframes || summary.keyframes || []).length)}`,
  ];

  if (hasStage175SummaryFields(summary)) {
    lines.push(
      "Stage 17.5 optional eye evidence fields:",
      `- Weak eye evidence frames: ${formatNumber(summary.weak_eye_warning_candidate_frames ?? summary.weak_eye_warning_evidence_frames)}`,
      `- Moderate eye evidence frames: ${formatNumber(summary.moderate_eye_closure_candidate_frames)}`,
      `- Strong eye evidence frames: ${formatNumber(summary.strong_eye_closure_candidate_frames)}`,
      `- Manual review recommended eye frames: ${formatNumber(summary.manual_review_recommended_eye_frames)}`,
    );
  }

  lines.push(
    PERMANENT_WARNING,
    "The uploaded video produced rule-based warning-candidate states for manual review. This is not final system-level drowsiness accuracy.",
  );

  return lines.join("\n");
}

export function resultMessage(summary?: VideoUploadSummary): string {
  if (!summary) return "No result has been returned yet.";
  const high = summary.high_confidence_drowsiness_candidate_frames ?? 0;
  const unreliable = summary.signal_unreliable_frames ?? 0;
  const warning =
    (summary.eye_warning_candidate_frames ?? 0) +
    (summary.mouth_warning_candidate_frames ?? 0) +
    high +
    unreliable;
  if (high > 0) return "High-confidence warning candidate intervals are present.";
  if (unreliable > 0) return "Signal-unreliable intervals are present.";
  if (warning > 0) {
    return "Warning-candidate states were produced for manual review.";
  }
  return "No high-confidence warning candidate was returned.";
}
