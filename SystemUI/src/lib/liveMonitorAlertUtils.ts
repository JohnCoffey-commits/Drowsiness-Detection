export type RealtimeFusionState =
  | "normal"
  | "eye_warning_candidate"
  | "mouth_warning_candidate"
  | "high_confidence_drowsiness_candidate"
  | "signal_unreliable";

export type LiveAlertKind =
  | "eye_warning"
  | "mouth_warning"
  | "high_confidence"
  | "signal_quality";

export type LiveAlertSeverity = "medium" | "high" | "quality";

export interface LiveAlertMeta {
  kind: LiveAlertKind;
  severity: LiveAlertSeverity;
  title: string;
  message: string;
}

export interface LiveAlertEvent {
  id: string;
  timestamp: number;
  kind: LiveAlertKind;
  fusionState: RealtimeFusionState;
  severity: LiveAlertSeverity;
  message: string;
  reason: string;
  source: "live-monitor-session";
  meanPEyeClosed?: number | null;
  pYawn?: number | null;
  eyeEvidence?: string;
  signalQuality?: string;
}

export interface LiveActiveAlert {
  kind: LiveAlertKind;
  fusionState: RealtimeFusionState;
  severity: LiveAlertSeverity;
  title: string;
  message: string;
  reason: string;
  stableStartedAt: number;
  createdAt: number;
  stableForSeconds: number;
  cooldownRemainingSeconds: number;
}

export interface LiveAlertControllerState {
  pendingAlertKind: LiveAlertKind | null;
  pendingAlertStartedAt: number | null;
  pendingFusionState: RealtimeFusionState | null;
  activeAlert: LiveActiveAlert | null;
  normalStartedAt: number | null;
  lastStateSeenAt: number | null;
  lastAlertCreatedAtByKind: Partial<Record<LiveAlertKind, number>>;
}

export interface EvaluateLiveAlertInput {
  fusionState?: string;
  reason?: string;
  now: number;
  meanPEyeClosed?: number | null;
  pYawn?: number | null;
  eyeEvidence?: string;
  signalQuality?: string;
}

export interface LiveAlertEvaluationResult {
  state: LiveAlertControllerState;
  event?: LiveAlertEvent;
}

export const LIVE_ALERT_DEBOUNCE_SECONDS = 1.0;
export const LIVE_ALERT_NORMAL_CLEAR_SECONDS = 2.0;

const LIVE_ALERT_DEBOUNCE_MS = LIVE_ALERT_DEBOUNCE_SECONDS * 1000;
const LIVE_ALERT_NORMAL_CLEAR_MS = LIVE_ALERT_NORMAL_CLEAR_SECONDS * 1000;

const LIVE_ALERT_COOLDOWN_MS: Record<LiveAlertKind, number> = {
  eye_warning: 8000,
  mouth_warning: 8000,
  high_confidence: 10000,
  signal_quality: 5000,
};

const LIVE_ALERT_META: Record<LiveAlertKind, LiveAlertMeta> = {
  eye_warning: {
    kind: "eye_warning",
    severity: "high",
    title: "Eye Warning",
    message:
      "Reduced eye openness or eye-closure candidate evidence was observed. Please keep your eyes open and stay alert.",
  },
  mouth_warning: {
    kind: "mouth_warning",
    severity: "medium",
    title: "Yawn Warning",
    message:
      "Yawn-like mouth activity was observed. Please stay alert and review your condition.",
  },
  high_confidence: {
    kind: "high_confidence",
    severity: "high",
    title: "Critical Eye Warning",
    message:
      "Sustained or repeated eye warning candidate evidence was observed. Please stop and rest when safe.",
  },
  signal_quality: {
    kind: "signal_quality",
    severity: "quality",
    title: "Face Not Visible",
    message:
      "Please center your face in the camera frame and keep your eyes and mouth visible.",
  },
};

export function createInitialLiveAlertControllerState(): LiveAlertControllerState {
  return {
    pendingAlertKind: null,
    pendingAlertStartedAt: null,
    pendingFusionState: null,
    activeAlert: null,
    normalStartedAt: null,
    lastStateSeenAt: null,
    lastAlertCreatedAtByKind: {},
  };
}

export function getLiveAlertMeta(kind: LiveAlertKind): LiveAlertMeta {
  return LIVE_ALERT_META[kind];
}

export function getLiveAlertCooldownSeconds(kind: LiveAlertKind): number {
  return LIVE_ALERT_COOLDOWN_MS[kind] / 1000;
}

export function getLiveAlertKindForFusionState(
  fusionState: RealtimeFusionState
): LiveAlertKind | null {
  if (fusionState === "eye_warning_candidate") {
    return "eye_warning";
  }
  if (fusionState === "mouth_warning_candidate") {
    return "mouth_warning";
  }
  if (fusionState === "high_confidence_drowsiness_candidate") {
    return "high_confidence";
  }
  if (fusionState === "signal_unreliable") {
    return "signal_quality";
  }
  return null;
}

export function normalizeRealtimeFusionState(
  fusionState: string | undefined
): RealtimeFusionState {
  if (
    fusionState === "eye_warning_candidate" ||
    fusionState === "mouth_warning_candidate" ||
    fusionState === "high_confidence_drowsiness_candidate" ||
    fusionState === "signal_unreliable"
  ) {
    return fusionState;
  }
  return "normal";
}

export function formatLiveAlertKind(kind: LiveAlertKind): string {
  return kind.replaceAll("_", " ");
}

export function formatLiveAlertSeverity(severity: LiveAlertSeverity): string {
  if (severity === "quality") {
    return "quality";
  }
  return severity;
}

export function getCooldownRemainingSeconds(
  kind: LiveAlertKind,
  now: number,
  lastAlertCreatedAtByKind: Partial<Record<LiveAlertKind, number>>
): number {
  const lastCreatedAt = lastAlertCreatedAtByKind[kind];

  if (!lastCreatedAt) {
    return 0;
  }

  return Math.max(0, (LIVE_ALERT_COOLDOWN_MS[kind] - (now - lastCreatedAt)) / 1000);
}

function cloneAlertState(
  state: LiveAlertControllerState
): LiveAlertControllerState {
  return {
    ...state,
    activeAlert: state.activeAlert ? { ...state.activeAlert } : null,
    lastAlertCreatedAtByKind: { ...state.lastAlertCreatedAtByKind },
  };
}

function refreshActiveAlertRuntime(
  activeAlert: LiveActiveAlert,
  now: number,
  lastAlertCreatedAtByKind: Partial<Record<LiveAlertKind, number>>,
  reason?: string
): LiveActiveAlert {
  return {
    ...activeAlert,
    reason: reason ?? activeAlert.reason,
    stableForSeconds: Math.max(0, (now - activeAlert.stableStartedAt) / 1000),
    cooldownRemainingSeconds: getCooldownRemainingSeconds(
      activeAlert.kind,
      now,
      lastAlertCreatedAtByKind
    ),
  };
}

export function evaluateLiveAlertState(
  state: LiveAlertControllerState,
  input: EvaluateLiveAlertInput
): LiveAlertEvaluationResult {
  const now = input.now;
  const fusionState = normalizeRealtimeFusionState(input.fusionState);
  const alertKind = getLiveAlertKindForFusionState(fusionState);
  const next = cloneAlertState(state);

  next.lastStateSeenAt = now;

  if (!alertKind) {
    next.pendingAlertKind = null;
    next.pendingAlertStartedAt = null;
    next.pendingFusionState = null;

    if (!next.activeAlert) {
      next.normalStartedAt = null;
      return { state: next };
    }

    if (!next.normalStartedAt) {
      next.normalStartedAt = now;
    }

    if (now - next.normalStartedAt >= LIVE_ALERT_NORMAL_CLEAR_MS) {
      next.activeAlert = null;
      next.normalStartedAt = null;
      return { state: next };
    }

    next.activeAlert = refreshActiveAlertRuntime(
      next.activeAlert,
      now,
      next.lastAlertCreatedAtByKind,
      input.reason
    );
    return { state: next };
  }

  next.normalStartedAt = null;

  if (next.pendingAlertKind !== alertKind) {
    next.pendingAlertKind = alertKind;
    next.pendingAlertStartedAt = now;
    next.pendingFusionState = fusionState;
  }

  const pendingStartedAt = next.pendingAlertStartedAt ?? now;
  const stableForMs = now - pendingStartedAt;

  if (stableForMs < LIVE_ALERT_DEBOUNCE_MS) {
    if (next.activeAlert) {
      next.activeAlert = refreshActiveAlertRuntime(
        next.activeAlert,
        now,
        next.lastAlertCreatedAtByKind
      );
    }
    return { state: next };
  }

  const meta = getLiveAlertMeta(alertKind);
  const reason = input.reason ?? meta.message;
  const isNewActiveAlert = !next.activeAlert || next.activeAlert.kind !== alertKind;
  const cooldownRemainingSeconds = getCooldownRemainingSeconds(
    alertKind,
    now,
    next.lastAlertCreatedAtByKind
  );
  let event: LiveAlertEvent | undefined;

  if (isNewActiveAlert && cooldownRemainingSeconds <= 0) {
    event = {
      id: `live-alert-${alertKind}-${now}`,
      timestamp: now,
      kind: alertKind,
      fusionState,
      severity: meta.severity,
      message: meta.title,
      reason,
      source: "live-monitor-session",
      meanPEyeClosed: input.meanPEyeClosed,
      pYawn: input.pYawn,
      eyeEvidence: input.eyeEvidence,
      signalQuality: input.signalQuality,
    };
    next.lastAlertCreatedAtByKind[alertKind] = now;
  }

  next.activeAlert = {
    kind: alertKind,
    fusionState,
    severity: meta.severity,
    title: meta.title,
    message: meta.message,
    reason,
    stableStartedAt: pendingStartedAt,
    createdAt: isNewActiveAlert ? now : next.activeAlert?.createdAt ?? now,
    stableForSeconds: Math.max(0, stableForMs / 1000),
    cooldownRemainingSeconds: getCooldownRemainingSeconds(
      alertKind,
      now,
      next.lastAlertCreatedAtByKind
    ),
  };

  return { state: next, event };
}
