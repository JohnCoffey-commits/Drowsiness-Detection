import type { LiveAlertKind } from "@/lib/liveMonitorAlertUtils";

export type LiveMonitorRiskSeverity =
  | "idle"
  | "low"
  | "medium"
  | "high"
  | "critical"
  | "signal_quality";

export interface LiveMonitorRiskState {
  score: number;
  label: "Idle" | "Low" | "Medium" | "High" | "Critical" | "Signal Check";
  helper: string;
  severity: LiveMonitorRiskSeverity;
}

interface LiveMonitorRiskTemporalState {
  fusion_state?: string;
  eye_warning_candidate?: boolean;
  mouth_warning_candidate?: boolean;
  eye_warning_active?: boolean;
  mouth_active?: boolean;
  sustained_eye_warning?: boolean;
  signal_unreliable?: boolean;
}

export interface LiveMonitorRiskFrameEvidence {
  face?: {
    detected?: boolean;
    tracking_status?: string;
  };
  eye?: {
    available?: boolean;
  };
  mouth?: {
    available?: boolean;
  };
  signal_quality?: {
    status?: string;
  };
  temporal?: LiveMonitorRiskTemporalState;
}

export interface BuildLiveMonitorRiskStateInput {
  cameraActive: boolean;
  samplingActive: boolean;
  activeAlertKind?: LiveAlertKind | null;
  criticalEyeWarningActive?: boolean;
  frameEvidence?: LiveMonitorRiskFrameEvidence | null;
}

export const IDLE_LIVE_MONITOR_RISK_STATE: LiveMonitorRiskState = {
  score: 0,
  label: "Idle",
  helper: "Start camera to monitor",
  severity: "idle",
};

function isFaceSignalUnreliable(
  evidence: LiveMonitorRiskFrameEvidence | null | undefined
): boolean {
  if (!evidence) {
    return false;
  }

  const faceKnown =
    typeof evidence.face?.detected === "boolean" ||
    typeof evidence.face?.tracking_status === "string";
  const eyeKnown = typeof evidence.eye?.available === "boolean";
  const mouthKnown = typeof evidence.mouth?.available === "boolean";
  const signalKnown = typeof evidence.signal_quality?.status === "string";

  const faceOk =
    evidence.face?.detected === true &&
    (!evidence.face.tracking_status || evidence.face.tracking_status === "ok");
  const eyeOk = evidence.eye?.available === true;
  const mouthOk = evidence.mouth?.available === true;
  const signalOk = evidence.signal_quality?.status === "ok";

  return (
    (faceKnown && !faceOk) ||
    (eyeKnown && !eyeOk) ||
    (mouthKnown && !mouthOk) ||
    (signalKnown && !signalOk) ||
    evidence.temporal?.signal_unreliable === true
  );
}

export function buildLiveMonitorRiskState(
  input: BuildLiveMonitorRiskStateInput
): LiveMonitorRiskState {
  if (!input.cameraActive || !input.samplingActive) {
    return IDLE_LIVE_MONITOR_RISK_STATE;
  }

  const activeAlertKind = input.activeAlertKind ?? null;
  const temporal = input.frameEvidence?.temporal;
  const fusionState = temporal?.fusion_state;

  if (
    input.criticalEyeWarningActive ||
    activeAlertKind === "high_confidence" ||
    fusionState === "high_confidence_drowsiness_candidate" ||
    temporal?.sustained_eye_warning === true
  ) {
    return {
      score: 92,
      label: "Critical",
      helper: "Stop and rest when safe",
      severity: "critical",
    };
  }

  if (
    activeAlertKind === "eye_warning" ||
    fusionState === "eye_warning_candidate" ||
    temporal?.eye_warning_candidate === true ||
    temporal?.eye_warning_active === true
  ) {
    return {
      score: 74,
      label: "High",
      helper: "Eye warning candidate",
      severity: "high",
    };
  }

  if (
    activeAlertKind === "mouth_warning" ||
    fusionState === "mouth_warning_candidate" ||
    temporal?.mouth_warning_candidate === true ||
    temporal?.mouth_active === true
  ) {
    return {
      score: 55,
      label: "Medium",
      helper: "Yawn warning candidate",
      severity: "medium",
    };
  }

  if (
    activeAlertKind === "signal_quality" ||
    fusionState === "signal_unreliable" ||
    isFaceSignalUnreliable(input.frameEvidence)
  ) {
    return {
      score: 30,
      label: "Signal Check",
      helper: "Center face in frame",
      severity: "signal_quality",
    };
  }

  return {
    score: 20,
    label: "Low",
    helper: "Monitoring",
    severity: "low",
  };
}

export function getLiveMonitorRiskStateKey(state: LiveMonitorRiskState): string {
  return `${state.severity}:${state.score}:${state.label}:${state.helper}`;
}
