"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Coffee,
  Eye,
  ShieldAlert,
  Smile,
  VideoOff,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import {
  createInitialLiveAlertControllerState,
  evaluateLiveAlertState,
  type LiveAlertControllerState,
  type LiveAlertEvent,
  type LiveAlertKind,
} from "@/lib/liveMonitorAlertUtils";
import {
  closeAudioContextSafely,
  createAudioContextSafely,
  playLiveMonitorAlertSound,
} from "@/lib/liveMonitorSoundUtils";
import {
  buildLiveMonitorRiskState,
  getLiveMonitorRiskStateKey,
  type LiveMonitorRiskState,
} from "@/lib/liveMonitorRiskUtils";
import {
  dashboardEventDraftFromLiveAlertEvent,
} from "@/lib/liveMonitorDashboardStore";
import type { LiveMonitorDashboardEventDraft } from "@/lib/liveMonitorDashboardTypes";
import { buildApiUrl, getApiBaseUrl } from "@/lib/apiConfig";

const REALTIME_API_BASE_URL = getApiBaseUrl();

const CRITICAL_EYE_REPEAT_WINDOW_MS = 60_000;
const CRITICAL_SOUND_REPEAT_MS = 2_200;

type CameraStatus =
  | "Idle"
  | "Requesting camera"
  | "Camera active"
  | "Permission denied"
  | "No camera found"
  | "Camera error"
  | "Stopped";

type RealtimeBackendStatus =
  | "Not connected"
  | "Checking health"
  | "Health check failed"
  | "Starting session"
  | "Session ready"
  | "Stopping session"
  | "Session stopped"
  | "Sending frame"
  | "Frame evidence"
  | "Backend error";

interface FrameSize {
  width: number;
  height: number;
}

interface RealtimeTemporalState {
  fusion_state?: string;
  eye_warning_candidate?: boolean;
  mouth_warning_candidate?: boolean;
  current_eye_evidence?: string;
  eye_warning_active?: boolean;
  mouth_active?: boolean;
  recent_yawn_event?: boolean;
  recent_eye_warning_reminder?: boolean;
  last_sustained_eye_warning_age_seconds?: number | null;
  rolling_eye_closed_mean?: number | null;
  current_eye_warning_duration_seconds?: number;
  sustained_eye_warning?: boolean;
  eye_evidence_strength?: string;
  moderate_or_strong_eye_evidence?: boolean;
  signal_unreliable?: boolean;
  recent_signal_failure_ratio?: number;
  safe_reason_text?: string;
  yawn_event?: boolean;
  eye_closed_binary?: boolean | null;
  recent_window_frame_count?: number;
  valid_eye_frame_count?: number;
  warning?: string;
}

interface RealtimeFrameEvidence {
  ok: boolean;
  session_id: string;
  frame_id: string;
  device?: string;
  latency_ms?: number;
  face?: {
    detected?: boolean;
    tracking_status?: string;
    num_faces?: number;
  };
  eye?: {
    available?: boolean;
    left_p_eye_closed?: number | null;
    right_p_eye_closed?: number | null;
    mean_p_eye_closed?: number | null;
    evidence_strength?: string;
    left_roi_box?: number[] | null;
    right_roi_box?: number[] | null;
    status?: string;
    reason?: string | null;
  };
  mouth?: {
    available?: boolean;
    p_yawn?: number | null;
    mouth_roi_box?: number[] | null;
    status?: string;
    reason?: string | null;
  };
  signal_quality?: {
    status?: string;
    reason?: string;
  };
  temporal?: RealtimeTemporalState;
  safe_interpretation?: string;
  warning?: string;
}

interface RealtimeSessionStartResponse {
  ok?: boolean;
  session_id?: string;
  started_at?: string;
  note?: string;
  warning?: string;
}

interface RealtimeSessionStopResponse {
  ok?: boolean;
  session_id?: string;
  stopped_at?: string;
  note?: string;
  warning?: string;
}

interface ProductAlertCopy {
  title: string;
  body: string;
  label: string;
}

interface CriticalEyeAlert {
  id: string;
  createdAt: number;
  reason: string;
  acknowledged: boolean;
}

interface FaceVisibilityItem {
  label: string;
  ok: boolean;
}

interface FaceVisibilityIssue {
  items: FaceVisibilityItem[];
  reason: string;
}

interface LiveVideoCardProps {
  onRiskStateChange?: (riskState: LiveMonitorRiskState) => void;
  onDashboardEvent?: (event: LiveMonitorDashboardEventDraft) => void;
}

const PRODUCT_ALERT_COPY: Record<LiveAlertKind, ProductAlertCopy> = {
  mouth_warning: {
    title: "Yawn Warning",
    body: "Yawn-like mouth activity was observed. Please stay alert and review your condition.",
    label: "Warning-candidate alert",
  },
  eye_warning: {
    title: "Eye Warning",
    body: "Reduced eye openness or eye-closure candidate evidence was observed. Please keep your eyes open and stay alert.",
    label: "Serious warning-candidate alert",
  },
  high_confidence: {
    title: "Critical Eye Warning",
    body: "Sustained or repeated eye warning candidate evidence was observed. Please stop and rest when safe.",
    label: "High-priority warning-candidate alert",
  },
  signal_quality: {
    title: "Face Not Visible",
    body: "Please center your face in the camera frame and keep your eyes and mouth visible.",
    label: "Signal-quality warning",
  },
};

const cameraStatusStyle: Record<CameraStatus, { dot: string; chip: string }> = {
  Idle: {
    dot: "bg-slate-400",
    chip: "border-white/60 bg-white/85 text-slate-800",
  },
  "Requesting camera": {
    dot: "animate-pulse bg-blue-500",
    chip: "border-blue-100 bg-white/90 text-blue-900",
  },
  "Camera active": {
    dot: "animate-pulse bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.8)]",
    chip: "border-emerald-100 bg-white/90 text-emerald-900",
  },
  "Permission denied": {
    dot: "bg-rose-500",
    chip: "border-rose-100 bg-white/90 text-rose-900",
  },
  "No camera found": {
    dot: "bg-amber-500",
    chip: "border-amber-100 bg-white/90 text-amber-900",
  },
  "Camera error": {
    dot: "bg-orange-500",
    chip: "border-orange-100 bg-white/90 text-orange-900",
  },
  Stopped: {
    dot: "bg-slate-400",
    chip: "border-white/60 bg-white/85 text-slate-800",
  },
};

function getCameraError(error: unknown): {
  status: CameraStatus;
  message: string;
} {
  if (!(error instanceof DOMException)) {
    return {
      status: "Camera error",
      message: "Camera could not be started. Please check the browser and device settings.",
    };
  }

  if (
    error.name === "NotAllowedError" ||
    error.name === "PermissionDeniedError" ||
    error.name === "SecurityError"
  ) {
    return {
      status: "Permission denied",
      message: "Camera permission was denied. Allow camera access in the browser to preview the feed.",
    };
  }

  if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
    return {
      status: "No camera found",
      message: "No camera device was found. Connect a camera and try again.",
    };
  }

  if (error.name === "NotReadableError" || error.name === "TrackStartError") {
    return {
      status: "Camera error",
      message: "The camera is busy or cannot be read by the browser.",
    };
  }

  if (error.name === "OverconstrainedError") {
    return {
      status: "Camera error",
      message: "The requested camera settings are not available on this device.",
    };
  }

  return {
    status: "Camera error",
    message: "Camera could not be started. Please check the browser and device settings.",
  };
}

function getSampleTargetSize(videoWidth: number, videoHeight: number): FrameSize {
  const maxWidth = 640;
  const maxHeight = 360;
  const scale = Math.min(maxWidth / videoWidth, maxHeight / videoHeight, 1);

  return {
    width: Math.max(1, Math.round(videoWidth * scale)),
    height: Math.max(1, Math.round(videoHeight * scale)),
  };
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return "Unexpected realtime backend error.";
}

function getResponseError(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (detail && typeof detail === "object" && "error" in detail) {
      const error = (detail as { error?: unknown }).error;
      if (typeof error === "string") {
        return error;
      }
    }
  }
  return fallback;
}

function realtimeReachabilityMessage(apiBaseUrl: string): string {
  return `Realtime service is not reachable from this frontend. Backend unavailable. Check that the local FastAPI server and Cloudflare Tunnel are running. Verify NEXT_PUBLIC_API_BASE_URL (${apiBaseUrl}) and CORS allowed origins.`;
}

function realtimeActionErrorMessage(
  action: string,
  apiBaseUrl: string,
  error?: unknown
): string {
  const details = getErrorMessage(error);
  const lowerDetails = details.toLowerCase();
  if (details.includes("NEXT_PUBLIC_API_BASE_URL")) {
    return details;
  }
  if (
    details === "Failed to fetch" ||
    details === "Unexpected realtime backend error." ||
    lowerDetails.includes("network") ||
    lowerDetails.includes("load failed")
  ) {
    return `${action}. ${realtimeReachabilityMessage(apiBaseUrl)}`;
  }
  return `${action}. ${details} Verify NEXT_PUBLIC_API_BASE_URL (${apiBaseUrl}) and CORS allowed origins.`;
}

function formatCameraStatusLabel(status: CameraStatus): string {
  if (status === "Camera active") {
    return "Camera Active";
  }
  if (status === "Requesting camera") {
    return "Starting Camera";
  }
  if (status === "Stopped" || status === "Idle") {
    return "Camera Off";
  }
  return status;
}

function getFaceVisibilityIssue(
  evidence: RealtimeFrameEvidence | null
): FaceVisibilityIssue | null {
  if (!evidence) {
    return null;
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

  const hasIssue =
    (faceKnown && !faceOk) ||
    (eyeKnown && !eyeOk) ||
    (mouthKnown && !mouthOk) ||
    (signalKnown && !signalOk) ||
    evidence.temporal?.signal_unreliable === true;

  if (!hasIssue) {
    return null;
  }

  const items: FaceVisibilityItem[] = [];
  if (faceKnown) {
    items.push({ label: "Face signal", ok: faceOk });
  }
  if (eyeKnown) {
    items.push({ label: "Eye region", ok: eyeOk });
  }
  if (mouthKnown) {
    items.push({ label: "Mouth region", ok: mouthOk });
  }
  if (signalKnown) {
    items.push({ label: "Lighting / signal quality", ok: signalOk });
  }

  return {
    items,
    reason:
      evidence.signal_quality?.reason ??
      "The current frame does not provide a reliable face, eye, and mouth signal.",
  };
}

function getProductAlertCardStyle(kind: LiveAlertKind): string {
  if (kind === "mouth_warning") {
    return "border-amber-200 bg-amber-50/95 text-amber-950 shadow-amber-950/10";
  }
  if (kind === "eye_warning") {
    return "border-rose-200 bg-rose-50/95 text-rose-950 shadow-rose-950/15";
  }
  return "border-slate-200 bg-white/95 text-slate-900 shadow-slate-950/10";
}

function getProductAlertIconStyle(kind: LiveAlertKind): string {
  if (kind === "mouth_warning") {
    return "bg-amber-500 text-white";
  }
  if (kind === "eye_warning") {
    return "bg-rose-600 text-white";
  }
  return "bg-slate-800 text-white";
}

function getCriticalReason(
  event: LiveAlertEvent,
  events: LiveAlertEvent[],
  temporal: RealtimeTemporalState | undefined
): string | null {
  if (event.kind === "high_confidence") {
    return "High-priority warning-candidate evidence was produced by the existing realtime rule layer.";
  }

  if (event.kind !== "eye_warning") {
    return null;
  }

  if (temporal?.sustained_eye_warning === true) {
    return "Sustained eye warning candidate evidence is active in the current realtime window.";
  }

  const recentEyeWarningCount = events.filter(
    (candidate) =>
      candidate.kind === "eye_warning" &&
      event.timestamp - candidate.timestamp <= CRITICAL_EYE_REPEAT_WINDOW_MS
  ).length;

  if (recentEyeWarningCount >= 2) {
    return "Repeated eye warning candidate events occurred within the recent monitoring window.";
  }

  return null;
}

function ProductWarningOverlay({ kind }: { kind: LiveAlertKind }) {
  const copy = PRODUCT_ALERT_COPY[kind];
  const Icon = kind === "mouth_warning" ? Smile : Eye;

  return (
    <div className="pointer-events-none absolute left-1/2 top-1/2 z-20 w-[min(92%,430px)] -translate-x-1/2 -translate-y-1/2">
      <div
        className={`rounded-2xl border p-4 shadow-2xl backdrop-blur-md ${getProductAlertCardStyle(
          kind
        )}`}
      >
        <div className="flex gap-3">
          <div
            className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full ${getProductAlertIconStyle(
              kind
            )}`}
          >
            <Icon className="h-5 w-5" strokeWidth={2.3} />
          </div>
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-current/65">
              {copy.label}
            </p>
            <h3 className="mt-1 text-lg font-semibold">{copy.title}</h3>
            <p className="mt-1 text-sm leading-5 text-current/80">{copy.body}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function CriticalEyeModal({
  alert,
  onAcknowledge,
}: {
  alert: CriticalEyeAlert;
  onAcknowledge: () => void;
}) {
  const titleId = `${alert.id}-title`;

  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/62 p-5 backdrop-blur-[3px]">
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="w-full max-w-[720px] overflow-hidden rounded-[2rem] border border-rose-200 bg-white text-slate-950 shadow-2xl shadow-slate-950/35"
      >
        <div className="grid gap-6 px-6 py-7 sm:grid-cols-[180px_1fr] sm:px-8 sm:py-9">
          <div className="flex items-center justify-center">
            <div className="relative flex h-36 w-36 items-center justify-center rounded-full bg-rose-100">
              <div className="absolute inset-4 rounded-full bg-rose-200/70" />
              <div className="relative flex h-24 w-24 items-center justify-center rounded-full bg-rose-600 text-white shadow-2xl shadow-rose-600/30">
                <ShieldAlert className="h-12 w-12" strokeWidth={2.2} />
              </div>
            </div>
          </div>

          <div className="flex min-w-0 flex-col justify-center text-center sm:text-left">
            <div className="mx-auto inline-flex w-fit items-center gap-2 rounded-full bg-rose-50 px-4 py-2 text-xs font-bold uppercase tracking-[0.18em] text-rose-700 sm:mx-0">
              <AlertTriangle className="h-4 w-4" strokeWidth={2.4} />
              High Priority
            </div>

            <h3
              id={titleId}
              className="mt-5 text-3xl font-black tracking-tight text-slate-950 sm:text-4xl"
            >
              <span className="text-rose-600">Critical</span> Eye Warning
            </h3>

            <p className="mt-3 text-lg font-semibold text-slate-400">
              Sustained eye warning candidate
            </p>

            <div className="mx-auto mt-6 flex w-full max-w-sm items-center justify-center gap-4 rounded-2xl bg-slate-100/80 px-5 py-4 text-left text-base font-semibold leading-6 text-slate-600 sm:mx-0">
              <Coffee className="h-7 w-7 shrink-0 text-slate-500" strokeWidth={2.2} />
              <span>Please stop and rest when safe.</span>
            </div>
          </div>
        </div>

        <div className="bg-rose-600 px-6 py-5 sm:px-8">
          <button
            type="button"
            onClick={onAcknowledge}
            className="mx-auto flex h-14 w-full max-w-sm items-center justify-center rounded-full bg-white/18 px-6 text-lg font-bold text-white shadow-lg shadow-rose-950/20 outline-none ring-1 ring-white/20 transition hover:bg-white/24 focus-visible:ring-2 focus-visible:ring-white"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}

function FaceVisibilityOverlay({ issue }: { issue: FaceVisibilityIssue }) {
  return (
    <div className="pointer-events-none absolute left-1/2 top-1/2 z-10 w-[min(92%,430px)] -translate-x-1/2 -translate-y-1/2">
      <div className="rounded-2xl border border-slate-200 bg-white/92 p-4 text-slate-900 shadow-2xl shadow-slate-950/10 backdrop-blur-md">
        <div className="flex gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-900 text-white">
            <AlertTriangle className="h-5 w-5" strokeWidth={2.3} />
          </div>
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Signal-quality warning
            </p>
            <h3 className="mt-1 text-lg font-semibold">Face Not Visible</h3>
            <p className="mt-1 text-sm leading-5 text-slate-700">
              Please center your face in the camera frame and keep your eyes and mouth
              visible.
            </p>
          </div>
        </div>

        {issue.items.length > 0 && (
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {issue.items.map((item) => (
              <div
                key={item.label}
                className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/80 px-3 py-2 text-xs font-medium"
              >
                <span>{item.label}</span>
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 ${
                    item.ok
                      ? "bg-emerald-50 text-emerald-700"
                      : "bg-amber-50 text-amber-700"
                  }`}
                >
                  <CheckCircle2 className="h-3.5 w-3.5" strokeWidth={2.2} />
                  {item.ok ? "Available" : "Check"}
                </span>
              </div>
            ))}
          </div>
        )}

        <p className="mt-3 text-xs leading-5 text-slate-500">
          Avoid strong backlight and keep the camera unobstructed.
        </p>
      </div>
    </div>
  );
}

function CameraOffPlaceholder() {
  return (
    <div className="absolute inset-0 overflow-hidden bg-[linear-gradient(135deg,#020617_0%,#0f172a_48%,#132239_100%)]">
      <div
        aria-hidden="true"
        className="absolute inset-0 opacity-[0.14]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(226,232,240,0.7) 1px, transparent 1px), linear-gradient(90deg, rgba(226,232,240,0.7) 1px, transparent 1px)",
          backgroundSize: "42px 42px",
        }}
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(59,130,246,0.22),transparent_45%),linear-gradient(to_bottom,rgba(15,23,42,0.08),rgba(2,6,23,0.72))]"
      />

      <div className="absolute inset-0 flex items-center justify-center px-6 pb-16 text-center sm:pb-12">
        <div className="flex max-w-md flex-col items-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/15 bg-white/10 text-white shadow-2xl shadow-slate-950/30 backdrop-blur-md">
            <VideoOff className="h-8 w-8" strokeWidth={2.2} />
          </div>
          <div className="mt-5 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.16em] text-slate-200 backdrop-blur-md">
            <span className="h-2 w-2 rounded-full bg-slate-300" />
            Camera Off
          </div>
          <h3 className="mt-4 text-2xl font-black tracking-tight text-white sm:text-3xl">
            Camera is off
          </h3>
          <p className="mt-2 text-sm font-medium leading-6 text-slate-300">
            Click Start Camera to begin live monitoring.
          </p>
          <p className="mt-4 rounded-full border border-white/10 bg-white/[0.08] px-3 py-1.5 text-xs font-semibold text-slate-300 backdrop-blur-md">
            No webcam frames are stored.
          </p>
        </div>
      </div>
    </div>
  );
}

export function LiveVideoCard({
  onRiskStateChange,
  onDashboardEvent,
}: LiveVideoCardProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const realtimeSessionIdRef = useRef<string | null>(null);
  const samplingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const samplingStartedAtRef = useRef<number | null>(null);
  const samplingRunIdRef = useRef(0);
  const sampledFrameCountRef = useRef(0);
  const skippedFrameCountRef = useRef(0);
  const isSampleInProgressRef = useRef(false);
  const isMountedRef = useRef(true);
  const lastRiskStateKeyRef = useRef("");
  const alertEventsRef = useRef<LiveAlertEvent[]>([]);
  const alertControllerRef = useRef<LiveAlertControllerState>(
    createInitialLiveAlertControllerState()
  );
  const audioContextRef = useRef<AudioContext | null>(null);
  const soundAlertsEnabledRef = useRef(false);

  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("Idle");
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [, setCameraLabel] = useState<string | null>(null);
  const [isStoppingCamera, setIsStoppingCamera] = useState(false);
  const [isSampling, setIsSampling] = useState(false);
  const [samplingFps] = useState(2);
  const [samplingError, setSamplingError] = useState<string | null>(null);
  const [isStartingRealtime, setIsStartingRealtime] = useState(false);
  const [, setRealtimeSessionId] = useState<string | null>(null);
  const [, setBackendStatus] = useState<RealtimeBackendStatus>("Not connected");
  const [backendError, setBackendError] = useState<string | null>(null);
  const [lastFrameEvidence, setLastFrameEvidence] = useState<RealtimeFrameEvidence | null>(null);
  const [alertController, setAlertController] = useState<LiveAlertControllerState>(() =>
    createInitialLiveAlertControllerState()
  );
  const [criticalEyeAlert, setCriticalEyeAlert] = useState<CriticalEyeAlert | null>(null);
  const [, setSoundError] = useState<string | null>(null);
  const [autoStartSamplingRequested, setAutoStartSamplingRequested] = useState(false);

  const resetVisualAlertLayer = useCallback(
    (clearEvents = false, updateState = true) => {
      const nextState = createInitialLiveAlertControllerState();
      alertControllerRef.current = nextState;

      if (clearEvents) {
        alertEventsRef.current = [];
      }

      if (updateState && isMountedRef.current) {
        setAlertController(nextState);
        setCriticalEyeAlert(null);
      }
    },
    []
  );

  const clearActiveVisualAlert = useCallback((updateState = true) => {
    const nextState: LiveAlertControllerState = {
      ...alertControllerRef.current,
      pendingAlertKind: null,
      pendingAlertStartedAt: null,
      pendingFusionState: null,
      activeAlert: null,
      normalStartedAt: null,
    };

    alertControllerRef.current = nextState;

    if (updateState && isMountedRef.current) {
      setAlertController(nextState);
    }
  }, []);

  const getReadyAudioContext = useCallback(async (): Promise<AudioContext | null> => {
    const result = await createAudioContextSafely(audioContextRef.current);

    if (!isMountedRef.current) {
      return null;
    }

    if (!result.ok || !result.context) {
      setSoundError(result.error ?? "Audio output could not be initialized.");
      return null;
    }

    audioContextRef.current = result.context;
    setSoundError(null);
    return result.context;
  }, []);

  const enableSoundAlertsFromGesture = useCallback(async () => {
    soundAlertsEnabledRef.current = true;
    setSoundError(null);

    const context = await getReadyAudioContext();

    if (!isMountedRef.current) {
      return;
    }

    if (!context) {
      soundAlertsEnabledRef.current = false;
    }
  }, [getReadyAudioContext]);

  const disableSoundAlerts = useCallback((updateState = true) => {
    soundAlertsEnabledRef.current = false;
    const context = audioContextRef.current;
    audioContextRef.current = null;
    void closeAudioContextSafely(context);

    if (updateState && isMountedRef.current) {
      setSoundError(null);
    }
  }, []);

  const playSoundForAlertKind = useCallback(
    async (kind: LiveAlertKind, samplingRunId: number) => {
      if (
        !soundAlertsEnabledRef.current ||
        !isMountedRef.current ||
        samplingRunId !== samplingRunIdRef.current ||
        samplingStartedAtRef.current == null
      ) {
        return;
      }

      const context = await getReadyAudioContext();

      if (
        !context ||
        !soundAlertsEnabledRef.current ||
        !isMountedRef.current ||
        samplingRunId !== samplingRunIdRef.current ||
        samplingStartedAtRef.current == null
      ) {
        return;
      }

      const result = await playLiveMonitorAlertSound(context, kind);

      if (!isMountedRef.current || !soundAlertsEnabledRef.current) {
        return;
      }

      if (!result.ok) {
        setSoundError(result.error ?? "Sound alert could not be played.");
        return;
      }

      setSoundError(null);
    },
    [getReadyAudioContext]
  );

  const evaluateVisualAlert = useCallback(
    (payload: RealtimeFrameEvidence, samplingRunId: number) => {
      const temporalState = payload.temporal;
      const evaluation = evaluateLiveAlertState(alertControllerRef.current, {
        fusionState: temporalState?.fusion_state,
        reason:
          temporalState?.safe_reason_text ??
          payload.signal_quality?.reason ??
          payload.safe_interpretation,
        now: Date.now(),
        meanPEyeClosed: payload.eye?.mean_p_eye_closed,
        pYawn: payload.mouth?.p_yawn,
        eyeEvidence:
          temporalState?.current_eye_evidence ??
          temporalState?.eye_evidence_strength ??
          payload.eye?.evidence_strength,
        signalQuality: payload.signal_quality?.status,
      });

      alertControllerRef.current = evaluation.state;
      setAlertController(evaluation.state);

      if (evaluation.event) {
        const alertEvent = evaluation.event;
        const nextEvents = [alertEvent, ...alertEventsRef.current].slice(0, 20);
        const criticalReason = getCriticalReason(alertEvent, nextEvents, temporalState);

        alertEventsRef.current = nextEvents;

        if (criticalReason) {
          onDashboardEvent?.(
            dashboardEventDraftFromLiveAlertEvent(alertEvent, "critical_eye_warning")
          );
          setCriticalEyeAlert({
            id: `${alertEvent.id}-critical`,
            createdAt: alertEvent.timestamp,
            reason: criticalReason,
            acknowledged: false,
          });
          void playSoundForAlertKind("high_confidence", samplingRunId);
          return;
        }

        onDashboardEvent?.(dashboardEventDraftFromLiveAlertEvent(alertEvent));
        void playSoundForAlertKind(alertEvent.kind, samplingRunId);
      }
    },
    [onDashboardEvent, playSoundForAlertKind]
  );

  const clearSamplingInterval = useCallback(() => {
    if (samplingIntervalRef.current) {
      clearInterval(samplingIntervalRef.current);
      samplingIntervalRef.current = null;
    }

    isSampleInProgressRef.current = false;
  }, []);

  const stopSampling = useCallback(
    (nextError: string | null = null, updateState = true) => {
      clearSamplingInterval();
      samplingStartedAtRef.current = null;
      samplingRunIdRef.current += 1;
      clearActiveVisualAlert(updateState);

      if (updateState && isMountedRef.current) {
        setIsSampling(false);
        setSamplingError(nextError);
      }
    },
    [clearActiveVisualAlert, clearSamplingInterval]
  );

  const checkRealtimeHealth = useCallback(async () => {
    setBackendStatus("Checking health");
    setBackendError(null);

    try {
      const response = await fetch(
        buildApiUrl("/api/realtime/health"),
        {
          method: "GET",
          cache: "no-store",
        }
      );
      const payload = (await response.json().catch(() => null)) as
        | { ok?: boolean; detail?: unknown }
        | null;

      if (!response.ok || payload?.ok === false) {
        throw new Error(
          getResponseError(payload, `Realtime health check failed with HTTP ${response.status}.`)
        );
      }
    } catch (error) {
      const message = realtimeActionErrorMessage(
        "Realtime health check failed",
        REALTIME_API_BASE_URL,
        error
      );
      setBackendStatus("Health check failed");
      setBackendError(message);
      throw new Error(message);
    }
  }, []);

  const stopRealtimeSession = useCallback(async (updateState = true) => {
    const sessionId = realtimeSessionIdRef.current;

    if (!sessionId) {
      if (updateState && isMountedRef.current) {
        setRealtimeSessionId(null);
        setBackendStatus("Not connected");
      }
      return;
    }

    realtimeSessionIdRef.current = null;

    if (updateState && isMountedRef.current) {
      setRealtimeSessionId(null);
      setBackendStatus("Stopping session");
    }

    const formData = new FormData();
    formData.append("session_id", sessionId);

    try {
      const response = await fetch(
        buildApiUrl("/api/realtime/session/stop"),
        {
          method: "POST",
          body: formData,
          keepalive: true,
        }
      );
      const payload = (await response.json().catch(() => null)) as
        | RealtimeSessionStopResponse
        | { detail?: unknown }
        | null;

      if (!response.ok || !payload || !("ok" in payload) || !payload.ok) {
        throw new Error(getResponseError(payload, "Could not stop realtime backend session."));
      }

      if (updateState && isMountedRef.current) {
        setBackendStatus("Session stopped");
        setBackendError(null);
      }
    } catch (error) {
      if (updateState && isMountedRef.current) {
        setBackendStatus("Backend error");
        setBackendError(
          realtimeActionErrorMessage(
            "Realtime session stop failed",
            REALTIME_API_BASE_URL,
            error
          )
        );
      }
    }
  }, []);

  const stopRealtimeEvidence = useCallback(
    (nextError: string | null = null, updateState = true) => {
      stopSampling(nextError, updateState);
      void stopRealtimeSession(updateState);
    },
    [stopRealtimeSession, stopSampling]
  );

  const stopCameraTracks = useCallback(
    (nextStatus: CameraStatus = "Stopped", updateState = true) => {
      stopRealtimeEvidence(null, updateState);
      resetVisualAlertLayer(true, updateState);
      disableSoundAlerts(updateState);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      }

      if (updateState && isMountedRef.current) {
        setMediaStream(null);
        setCameraLabel(null);
        setCameraStatus(nextStatus);
        setRealtimeSessionId(null);
        setBackendStatus("Not connected");
        setAutoStartSamplingRequested(false);
      }
    },
    [disableSoundAlerts, resetVisualAlertLayer, stopRealtimeEvidence]
  );

  const startRealtimeSession = useCallback(async (): Promise<string> => {
    await checkRealtimeHealth();

    setBackendStatus("Starting session");
    setBackendError(null);

    const response = await fetch(
      buildApiUrl("/api/realtime/session/start"),
      { method: "POST" }
    );
    const payload = (await response.json().catch(() => null)) as RealtimeSessionStartResponse | null;

    if (!response.ok || !payload?.ok || !payload.session_id) {
      throw new Error(getResponseError(payload, "Could not start realtime backend session."));
    }

    realtimeSessionIdRef.current = payload.session_id;
    setRealtimeSessionId(payload.session_id);
    setBackendStatus("Session ready");
    return payload.session_id;
  }, [checkRealtimeHealth]);

  const sendFrameToBackend = useCallback(
    async (
      blob: Blob,
      targetSize: FrameSize,
      capturedAt: number,
      samplingRunId: number
    ) => {
      const sessionId = realtimeSessionIdRef.current;

      if (!sessionId) {
        setBackendStatus("Backend error");
        setBackendError("Realtime session is unavailable. Stop and restart the camera.");
        return;
      }

      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("client_timestamp_ms", String(capturedAt));
      formData.append("frame_width", String(targetSize.width));
      formData.append("frame_height", String(targetSize.height));
      formData.append("sampling_fps", String(samplingFps));
      formData.append("frame", blob, `webcam-frame-${capturedAt}.jpg`);

      try {
        setBackendStatus("Sending frame");
        const response = await fetch(buildApiUrl("/api/realtime/frame"), {
          method: "POST",
          body: formData,
        });
        const payload = (await response.json().catch(() => null)) as
          | RealtimeFrameEvidence
          | { detail?: unknown }
          | null;

        if (!isMountedRef.current || samplingRunId !== samplingRunIdRef.current) {
          return;
        }

        if (!response.ok || !payload || !("ok" in payload) || !payload.ok) {
          setBackendStatus("Backend error");
          setBackendError(getResponseError(payload, "Realtime frame inference failed."));
          return;
        }

        setLastFrameEvidence(payload);
        evaluateVisualAlert(payload, samplingRunId);
        setBackendStatus("Frame evidence");
        setBackendError(null);
      } catch (error) {
        if (!isMountedRef.current || samplingRunId !== samplingRunIdRef.current) {
          return;
        }
        setBackendStatus("Backend error");
        setBackendError(
          realtimeActionErrorMessage(
            "Realtime frame submission failed",
            REALTIME_API_BASE_URL,
            error
          )
        );
      }
    },
    [evaluateVisualAlert, samplingFps]
  );

  const sampleVideoFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const stream = streamRef.current;
    const hasLiveVideoTrack =
      stream?.getVideoTracks().some((track) => track.readyState === "live") ?? false;

    if (!video || !canvas || !stream || !hasLiveVideoTrack) {
      stopRealtimeEvidence("Video stream unavailable. Sampling stopped.");
      return;
    }

    if (!video.videoWidth || !video.videoHeight) {
      return;
    }

    if (isSampleInProgressRef.current) {
      const nextSkipped = skippedFrameCountRef.current + 1;
      skippedFrameCountRef.current = nextSkipped;
      return;
    }

    isSampleInProgressRef.current = true;

    const samplingRunId = samplingRunIdRef.current;
    const targetSize = getSampleTargetSize(video.videoWidth, video.videoHeight);
    const context = canvas.getContext("2d");

    if (!context) {
      isSampleInProgressRef.current = false;
      stopRealtimeEvidence("Canvas is unavailable. Sampling stopped.");
      return;
    }

    canvas.width = targetSize.width;
    canvas.height = targetSize.height;
    context.drawImage(video, 0, 0, targetSize.width, targetSize.height);

    canvas.toBlob(
      (blob) => {
        void (async () => {
          try {
            if (!isMountedRef.current || samplingRunId !== samplingRunIdRef.current) {
              return;
            }

            if (!blob) {
              stopRealtimeEvidence("Frame encoding failed. Sampling stopped.");
              return;
            }

            const now = Date.now();
            if (!samplingStartedAtRef.current) {
              return;
            }

            const nextCount = sampledFrameCountRef.current + 1;

            sampledFrameCountRef.current = nextCount;
            setSamplingError(null);

            await sendFrameToBackend(blob, targetSize, now, samplingRunId);
          } finally {
            isSampleInProgressRef.current = false;
          }
        })();
      },
      "image/jpeg",
      0.85
    );
  }, [sendFrameToBackend, stopRealtimeEvidence]);

  const startSampling = useCallback(async () => {
    if (!streamRef.current || isSampling || samplingIntervalRef.current || isStartingRealtime) {
      return;
    }

    setIsStartingRealtime(true);
    setSamplingError(null);
    setBackendError(null);
    setLastFrameEvidence(null);
    resetVisualAlertLayer(true);

    try {
      await startRealtimeSession();

      if (!isMountedRef.current || !streamRef.current) {
        void stopRealtimeSession(false);
        return;
      }

      sampledFrameCountRef.current = 0;
      skippedFrameCountRef.current = 0;
      samplingRunIdRef.current += 1;
      setIsSampling(true);
      samplingStartedAtRef.current = Date.now();

      samplingIntervalRef.current = setInterval(sampleVideoFrame, Math.round(1000 / samplingFps));
      sampleVideoFrame();
    } catch (error) {
      const message = realtimeActionErrorMessage(
        "Camera is available, but backend evidence analysis is unavailable",
        REALTIME_API_BASE_URL,
        error
      );
      setSamplingError(message);
      setBackendError(message);
      setBackendStatus("Backend error");
      realtimeSessionIdRef.current = null;
      setRealtimeSessionId(null);
    } finally {
      if (isMountedRef.current) {
        setIsStartingRealtime(false);
      }
    }
  }, [
    isSampling,
    isStartingRealtime,
    resetVisualAlertLayer,
    sampleVideoFrame,
    samplingFps,
    startRealtimeSession,
    stopRealtimeSession,
  ]);

  useEffect(() => {
    const video = videoRef.current;

    if (!video) {
      return;
    }

    if (!mediaStream) {
      video.srcObject = null;
      return;
    }

    video.srcObject = mediaStream;
    void video.play().catch(() => undefined);
  }, [mediaStream]);

  useEffect(() => {
    if (!autoStartSamplingRequested || !streamRef.current || isSampling || isStartingRealtime) {
      return;
    }

    setAutoStartSamplingRequested(false);
    void startSampling();
  }, [autoStartSamplingRequested, isSampling, isStartingRealtime, startSampling]);

  useEffect(() => {
    isMountedRef.current = true;

    return () => {
      isMountedRef.current = false;
      soundAlertsEnabledRef.current = false;
      stopCameraTracks("Stopped", false);
      void closeAudioContextSafely(audioContextRef.current);
      audioContextRef.current = null;
    };
  }, [stopCameraTracks]);

  useEffect(() => {
    if (!criticalEyeAlert || criticalEyeAlert.acknowledged || !isSampling) {
      return;
    }

    const samplingRunId = samplingRunIdRef.current;
    const interval = setInterval(() => {
      void playSoundForAlertKind("high_confidence", samplingRunId);
    }, CRITICAL_SOUND_REPEAT_MS);

    return () => clearInterval(interval);
  }, [criticalEyeAlert, isSampling, playSoundForAlertKind]);

  const startCamera = useCallback(async () => {
    if (streamRef.current || cameraStatus === "Requesting camera" || isStartingRealtime) {
      return;
    }

    setCameraError(null);
    setCameraLabel(null);
    setSamplingError(null);
    setBackendError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus("Camera error");
      setCameraError("This browser does not support webcam capture.");
      return;
    }

    setCameraStatus("Requesting camera");

    try {
      await enableSoundAlertsFromGesture();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      const [videoTrack] = stream.getVideoTracks();

      if (!isMountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      setMediaStream(stream);
      setCameraLabel(videoTrack?.label ?? "Webcam");
      setCameraStatus("Camera active");
      setAutoStartSamplingRequested(true);
    } catch (error) {
      if (!isMountedRef.current) {
        return;
      }

      stopCameraTracks("Camera error");
      const cameraFailure = getCameraError(error);
      setCameraStatus(cameraFailure.status);
      setCameraError(cameraFailure.message);
    }
  }, [cameraStatus, enableSoundAlertsFromGesture, isStartingRealtime, stopCameraTracks]);

  const handleStopCamera = useCallback(() => {
    if (isStoppingCamera) {
      return;
    }

    setIsStoppingCamera(true);
    setCameraError(null);
    stopCameraTracks("Stopped");

    window.setTimeout(() => {
      if (isMountedRef.current) {
        setIsStoppingCamera(false);
      }
    }, 250);
  }, [isStoppingCamera, stopCameraTracks]);

  const handleCameraToggle = useCallback(() => {
    if (cameraStatus === "Camera active") {
      handleStopCamera();
      return;
    }

    void startCamera();
  }, [cameraStatus, handleStopCamera, startCamera]);

  const acknowledgeCriticalAlert = useCallback(() => {
    setCriticalEyeAlert((alert) =>
      alert
        ? {
            ...alert,
            acknowledged: true,
          }
        : alert
    );
  }, []);

  const isRequesting = cameraStatus === "Requesting camera";
  const isActive = cameraStatus === "Camera active" && mediaStream !== null;
  const isStartingCamera = isRequesting || isStartingRealtime;
  const isCameraTransitioning = isStartingCamera || isStoppingCamera;
  const currentStatusStyle = cameraStatusStyle[cameraStatus];
  const cameraButtonLabel = isStoppingCamera
    ? "Stopping..."
    : isStartingCamera
      ? "Starting..."
      : isActive
        ? "Stop Camera"
        : "Start Camera";
  const CameraButtonIcon = isActive ? VideoOff : Camera;
  const activeVisualAlert = alertController.activeAlert;
  const activeProductAlert =
    activeVisualAlert?.kind === "mouth_warning" || activeVisualAlert?.kind === "eye_warning"
      ? activeVisualAlert
      : null;
  const visibleCriticalAlert =
    criticalEyeAlert && !criticalEyeAlert.acknowledged ? criticalEyeAlert : null;
  const faceVisibilityIssue =
    isActive && isSampling && !visibleCriticalAlert && !activeProductAlert
      ? getFaceVisibilityIssue(lastFrameEvidence)
      : null;
  const sourceLabel = isActive ? "Webcam" : "Preview";
  const riskState = useMemo(
    () =>
      buildLiveMonitorRiskState({
        cameraActive: isActive,
        samplingActive: isSampling,
        activeAlertKind: activeVisualAlert?.kind ?? null,
        criticalEyeWarningActive:
          criticalEyeAlert !== null && !criticalEyeAlert.acknowledged,
        frameEvidence: lastFrameEvidence,
      }),
    [
      activeVisualAlert?.kind,
      criticalEyeAlert,
      isActive,
      isSampling,
      lastFrameEvidence,
    ]
  );

  useEffect(() => {
    if (!onRiskStateChange) {
      return;
    }

    const nextRiskStateKey = getLiveMonitorRiskStateKey(riskState);

    if (lastRiskStateKeyRef.current === nextRiskStateKey) {
      return;
    }

    lastRiskStateKeyRef.current = nextRiskStateKey;
    onRiskStateChange(riskState);
  }, [onRiskStateChange, riskState]);

  return (
    <Card className="group relative col-span-2 row-span-2 flex h-full min-h-[400px] flex-col overflow-hidden rounded-[2rem] border border-slate-200/70 bg-white p-2 shadow-sm transition-all duration-300 hover:shadow-md xl:min-h-0">
      <div className="relative min-h-[340px] w-full flex-1 overflow-hidden rounded-[1.5rem] bg-slate-950 xl:min-h-0">
        <canvas ref={canvasRef} className="hidden" aria-hidden="true" />

        {!isActive && <CameraOffPlaceholder />}

        <video
          ref={videoRef}
          aria-label="Live webcam preview"
          className={`absolute inset-0 h-full w-full transform-gpu object-cover transition-[opacity,transform] duration-300 ${
            isActive ? "opacity-100" : "opacity-0"
          } -scale-x-100`}
          autoPlay
          playsInline
          muted
        />

        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-slate-950/25 via-transparent to-slate-950/35" />

        <div
          className={`absolute left-5 top-5 z-20 flex max-w-[calc(100%-10rem)] items-center gap-2.5 rounded-full border px-3.5 py-1.5 text-xs font-semibold shadow-lg backdrop-blur-md ${currentStatusStyle.chip}`}
        >
          <div className={`h-2 w-2 shrink-0 rounded-full ${currentStatusStyle.dot}`} />
          <span className="truncate">{formatCameraStatusLabel(cameraStatus)}</span>
        </div>

        <div className="absolute right-5 top-5 z-20 rounded-full border border-white/60 bg-white/85 px-3.5 py-1.5 text-xs font-semibold text-slate-800 shadow-lg backdrop-blur-md">
          {sourceLabel}
        </div>

        {activeProductAlert && !visibleCriticalAlert && (
          <ProductWarningOverlay kind={activeProductAlert.kind} />
        )}

        {faceVisibilityIssue && <FaceVisibilityOverlay issue={faceVisibilityIssue} />}

        {visibleCriticalAlert && (
          <CriticalEyeModal alert={visibleCriticalAlert} onAcknowledge={acknowledgeCriticalAlert} />
        )}

        {(cameraError || samplingError || backendError) && !isRequesting && (
          <div className="absolute bottom-20 left-5 z-20 max-w-[calc(100%-2.5rem)] rounded-xl border border-rose-100 bg-white/92 px-3.5 py-2 text-xs font-medium text-rose-800 shadow-lg backdrop-blur-md">
            {cameraError ?? samplingError ?? backendError}
          </div>
        )}

        {isRequesting && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/35 text-sm font-semibold text-white backdrop-blur-[2px]">
            Waiting for camera permission...
          </div>
        )}

        <button
          type="button"
          onClick={handleCameraToggle}
          disabled={isCameraTransitioning}
          className={`absolute bottom-5 right-5 z-40 inline-flex h-11 items-center gap-2 rounded-full border px-5 text-sm font-semibold shadow-xl outline-none transition focus-visible:ring-2 disabled:cursor-not-allowed disabled:opacity-60 ${
            isActive
              ? "border-rose-200/30 bg-rose-600 text-white shadow-rose-950/25 hover:bg-rose-700 focus-visible:ring-rose-200"
              : "border-white/20 bg-gradient-to-r from-blue-600 to-emerald-500 text-white shadow-blue-950/25 hover:from-blue-700 hover:to-emerald-600 focus-visible:ring-blue-200"
          }`}
        >
          <CameraButtonIcon className="h-4.5 w-4.5" strokeWidth={2.3} />
          {cameraButtonLabel}
        </button>
      </div>
    </Card>
  );
}
