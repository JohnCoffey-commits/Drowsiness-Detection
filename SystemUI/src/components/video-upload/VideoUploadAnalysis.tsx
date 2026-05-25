"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FileVideo,
  RefreshCw,
  RotateCcw,
  Server,
  UploadCloud,
  XCircle,
} from "lucide-react";
import { AnalysisSummaryCards } from "@/components/video-upload/AnalysisSummaryCards";
import { EvidenceFigures } from "@/components/video-upload/EvidenceFigures";
import { IntervalReviewTable } from "@/components/video-upload/IntervalReviewTable";
import { KeyframeEvidenceGallery } from "@/components/video-upload/KeyframeEvidenceGallery";
import { TechnicalEvidencePanel } from "@/components/video-upload/TechnicalEvidencePanel";
import {
  type AnalysisStatus,
  type BackendStatus,
  type VideoUploadResponse,
} from "@/lib/videoUploadTypes";
import {
  buildApiUrl,
  buildVideoUploadReportHtml,
  DEFAULT_BACKEND_URL,
  downloadTextFile,
  figureDefinitions,
  formatBytes,
  formatNumber,
  formatSeconds,
  mergeWarningIntervals,
  normalizeBackendUrl,
  resultMessage,
  sanitizeBrowserText,
  validateBackendUrl,
  videoUploadReportFilename,
} from "@/lib/videoUploadUtils";
import { getArchiveClientId } from "@/lib/archiveClientId";
import {
  buildVideoArchiveRunPayload,
  saveVideoArchiveRun,
} from "@/lib/backendArchiveApi";
import { useVisionGuardAuth } from "@/lib/authStore";
import { useVisionGuardNotifications } from "@/lib/notificationStore";

const ACCEPTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".m4v"];
const BACKEND_URL_STORAGE_KEY = "visionguard.videoUpload.backendUrl";
const BACKEND_HEALTH_PATH = "/";

const processingSteps = [
  "Upload validation",
  "Frame sampling",
  "Eye ROI extraction",
  "Eye model inference",
  "Eye temporal rule",
  "Mouth/yawn inference",
  "Rule-based fusion",
  "Keyframe and report generation",
];

function statusLabel(status: AnalysisStatus): string {
  switch (status) {
    case "idle":
      return "Idle";
    case "file-selected":
      return "File selected";
    case "uploading":
      return "Uploading";
    case "analyzing":
      return "Analyzing";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    case "backend-unavailable":
      return "Backend unavailable";
  }
}

function backendStatusLabel(status: BackendStatus): string {
  switch (status) {
    case "connected":
      return "Backend connected";
    case "disconnected":
      return "Backend unavailable";
    case "checking":
      return "Checking backend";
    case "unchecked":
      return "Backend not checked";
  }
}

function backendStatusTone(status: BackendStatus): string {
  switch (status) {
    case "connected":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "disconnected":
      return "border-red-200 bg-red-50 text-red-700";
    case "checking":
      return "border-blue-200 bg-blue-50 text-blue-700";
    case "unchecked":
      return "border-slate-200 bg-slate-50 text-slate-600";
  }
}

function formatBackendCheckedAt(value: string | null): string {
  if (!value) return "Not checked";
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function processingProgress(status: AnalysisStatus): {
  activeIndex: number;
  completeThrough: number;
  progress: number;
} {
  switch (status) {
    case "file-selected":
      return { activeIndex: 0, completeThrough: -1, progress: 8 };
    case "uploading":
      return { activeIndex: 0, completeThrough: -1, progress: 18 };
    case "analyzing":
      return { activeIndex: 5, completeThrough: 4, progress: 68 };
    case "completed":
      return {
        activeIndex: processingSteps.length - 1,
        completeThrough: processingSteps.length - 1,
        progress: 100,
      };
    case "failed":
    case "backend-unavailable":
      return { activeIndex: -1, completeThrough: -1, progress: 0 };
    case "idle":
      return { activeIndex: -1, completeThrough: -1, progress: 0 };
  }
}

function fileValidationError(file: File): string | null {
  const extension = `.${file.name.split(".").pop()?.toLowerCase() || ""}`;
  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    return "Unsupported video format. Use MP4, MOV, AVI, or M4V.";
  }
  if (file.size > 750 * 1024 * 1024) {
    return "The selected video is larger than the 750 MB upload limit.";
  }
  return null;
}

function usefulBackendError(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const record = payload as Record<string, unknown>;
  const detail = record.detail;
  if (detail && typeof detail === "object") {
    const detailRecord = detail as Record<string, unknown>;
    return sanitizeBrowserText(
      detailRecord.error || detailRecord.detail || detailRecord.message || detail,
    );
  }
  return sanitizeBrowserText(record.error || detail || record.message || fallback);
}

function backendReachabilityMessage(backendUrl: string): string {
  return `Backend unavailable. Check that the local FastAPI server and Cloudflare Tunnel are running. Verify NEXT_PUBLIC_API_BASE_URL uses the current backend URL (${normalizeBackendUrl(
    backendUrl,
  )}) and CORS allowed origins include this frontend.`;
}

function SectionCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}
    >
      {children}
    </section>
  );
}

function UploadedVideoPreview({
  file,
  previewUrl,
  onDuration,
}: {
  file: File | null;
  previewUrl: string | null;
  onDuration: (value?: number) => void;
}) {
  return (
    <SectionCard>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-950">
            Video preview
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Preview the selected file before sending it to backend analysis.
          </p>
        </div>
        <FileVideo className="h-5 w-5 shrink-0 text-blue-600" />
      </div>

      {file && previewUrl ? (
        <div className="mt-4">
          <video
            controls
            preload="metadata"
            src={previewUrl}
            className="aspect-video w-full rounded-xl border border-slate-200 bg-slate-950"
            onLoadedMetadata={(event) =>
              onDuration(event.currentTarget.duration || undefined)
            }
          />
        </div>
      ) : (
        <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-sm text-slate-500">
          Select a video file to preview it here.
        </div>
      )}
    </SectionCard>
  );
}

function ProcessingStatus({
  status,
  error,
  response,
}: {
  status: AnalysisStatus;
  error: string | null;
  response?: VideoUploadResponse | null;
}) {
  const [showCompletedSteps, setShowCompletedSteps] = useState(false);
  const progressInfo = processingProgress(status);
  const summary = response?.summary;
  const keyframeCount =
    response ? (response.keyframes || response.summary.keyframes || []).length : 0;
  const showStepList =
    (status !== "idle" && status !== "completed") || showCompletedSteps;

  return (
    <SectionCard>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-950">Processing Status</h2>
          <p className="mt-1 text-sm text-slate-600">
            Progress from upload through backend evidence generation.
          </p>
        </div>
        <span
          className={`inline-flex w-fit rounded-full border px-3 py-1 text-xs font-semibold ${
            status === "completed"
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : status === "failed" || status === "backend-unavailable"
                ? "border-red-200 bg-red-50 text-red-700"
                : status === "idle"
                  ? "border-slate-200 bg-slate-50 text-slate-600"
                  : "border-blue-200 bg-blue-50 text-blue-700"
          }`}
        >
          {statusLabel(status)}
        </span>
      </div>

      {status === "completed" ? (
        <div className="mt-4 flex flex-col gap-3 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2 font-semibold">
            <CheckCircle2 className="h-4 w-4" />
            <span>
              Analysis completed - {formatNumber(summary?.total_frames_sampled)} sampled frames -{" "}
              {formatNumber(keyframeCount)} keyframes
            </span>
          </div>
          <button
            type="button"
            onClick={() => setShowCompletedSteps((value) => !value)}
            className="w-fit rounded-lg border border-emerald-200 bg-white px-3 py-2 text-xs font-semibold text-emerald-800 outline-none transition hover:bg-emerald-100 focus-visible:ring-2 focus-visible:ring-emerald-300"
          >
            {showCompletedSteps ? "Hide pipeline steps" : "View pipeline steps"}
          </button>
        </div>
      ) : (
        <div className="mt-4">
          <div className="mb-2 flex items-center justify-between text-xs font-semibold text-slate-500">
            <span>Progress</span>
            <span>{progressInfo.progress}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-slate-100">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                status === "failed" || status === "backend-unavailable"
                  ? "bg-red-500"
                  : "bg-blue-600"
              }`}
              style={{ width: `${progressInfo.progress}%` }}
            />
          </div>
        </div>
      )}

      {showStepList ? (
        <div className="mb-2 flex items-center justify-between text-xs font-semibold text-slate-500">
          <span>Pipeline steps</span>
        </div>
      ) : null}

      {showStepList ? (
        <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
          {processingSteps.map((step, index) => {
            const complete = index <= progressInfo.completeThrough;
            const active = index === progressInfo.activeIndex;
            return (
              <div
                key={step}
                className={`rounded-xl border p-3 text-sm ${
                  complete
                    ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                    : active
                      ? "border-blue-200 bg-blue-50 text-blue-800"
                      : "border-slate-200 bg-slate-50 text-slate-500"
                }`}
              >
                <div className="flex items-center gap-2">
                  {complete ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : active ? (
                    <RefreshCw className="h-4 w-4" />
                  ) : (
                    <span className="h-4 w-4 rounded-full border border-current" />
                  )}
                  <span className="font-semibold">{step}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : null}

      {error ? (
        <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <div className="flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <div className="font-bold">Failure reason</div>
              <div className="mt-1">{error}</div>
            </div>
          </div>
        </div>
      ) : null}
    </SectionCard>
  );
}

function ResultOverview({
  response,
  figuresCount,
}: {
  response: VideoUploadResponse;
  figuresCount: number;
}) {
  const summary = response.summary || {};
  const keyframeCount = (response.keyframes || summary.keyframes || []).length;
  const alertIntervals = mergeWarningIntervals(summary).length;
  const statusItems = [
    ["Video duration", formatSeconds(summary.duration_sec)],
    ["Sampled frames", formatNumber(summary.total_frames_sampled)],
    ["Alert intervals", formatNumber(alertIntervals)],
    ["Keyframes", formatNumber(keyframeCount)],
    ["Evidence figures", formatNumber(figuresCount)],
    ["Processing time", formatSeconds(response.runtime_duration_sec ?? summary.runtime_sec)],
  ];
  const highRiskCount =
    summary.high_confidence_intervals?.length ??
    (summary.high_confidence_drowsiness_candidate_frames ? 1 : 0);
  const overviewText =
    highRiskCount > 0
      ? "The uploaded video contains alert intervals with stronger eye-closure-related evidence."
      : alertIntervals > 0
        ? "The uploaded video contains fatigue-related visual alert intervals."
        : "No alert intervals were returned for this uploaded video.";

  return (
    <SectionCard className="border-blue-200 bg-blue-50/30 p-4">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-950">Analysis Summary</h2>
          <p className="mt-1 text-sm font-semibold text-blue-700">
            {resultMessage(summary)}
          </p>
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-600">
            {overviewText}
          </p>
        </div>
        <div className="rounded-lg border border-blue-200 bg-white px-3 py-2 text-xs text-slate-600">
          Duration is based on sampled timeline timestamps.
        </div>
      </div>

      <dl className="mt-4 flex flex-wrap gap-2">
        {statusItems.map(([label, value]) => (
          <div
            key={label}
            className="rounded-lg border border-slate-200 bg-white px-3 py-2"
          >
            <dt className="text-[10px] font-semibold uppercase text-slate-400">
              {label}
            </dt>
            <dd className="mt-0.5 max-w-[220px] truncate text-xs font-bold text-slate-900">
              {value}
            </dd>
          </div>
        ))}
      </dl>

      <p className="mt-3 text-xs leading-relaxed text-slate-600">
        Technical run identifiers and raw backend artifacts are available in
        Technical Details.
      </p>
    </SectionCard>
  );
}

function ResultSectionNav() {
  const links = [
    ["Overview", "overview"],
    ["Alert Intervals", "alert-intervals"],
    ["Evidence Figures", "evidence-figures"],
    ["Keyframes", "keyframes"],
    ["Technical Details", "technical-details"],
  ];

  return (
    <nav
      aria-label="Video analysis result sections"
      className="sticky top-0 z-10 flex flex-wrap gap-2 rounded-2xl border border-slate-200 bg-white/95 p-3 text-sm shadow-sm backdrop-blur"
    >
      {links.map(([label, id]) => (
        <a
          key={id}
          href={`#${id}`}
          className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 outline-none transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus-visible:ring-2 focus-visible:ring-blue-400"
        >
          {label}
        </a>
      ))}
    </nav>
  );
}

export function VideoUploadAnalysis() {
  const { currentUser } = useVisionGuardAuth();
  const { addNotification } = useVisionGuardNotifications();
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [backendUrlLoaded, setBackendUrlLoaded] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("unchecked");
  const [backendCheckedAt, setBackendCheckedAt] = useState<string | null>(null);
  const [allowBackendOverride, setAllowBackendOverride] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number | undefined>();
  const [status, setStatus] = useState<AnalysisStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<VideoUploadResponse | null>(null);
  const [reportState, setReportState] = useState<"idle" | "downloaded" | "failed">("idle");
  const [archiveState, setArchiveState] = useState<
    "idle" | "saving" | "saved" | "failed"
  >("idle");
  const [archiveMessage, setArchiveMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const analyzingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (analyzingTimerRef.current) clearTimeout(analyzingTimerRef.current);
    };
  }, [previewUrl]);

  useEffect(() => {
    const id = window.setTimeout(() => {
      try {
        const storedBackendUrl = window.localStorage.getItem(BACKEND_URL_STORAGE_KEY);
        if (storedBackendUrl) {
          setBackendUrl(storedBackendUrl);
        }
      } catch {
        // Local storage can be unavailable in private or restricted browser contexts.
      } finally {
        setBackendUrlLoaded(true);
      }
    }, 0);

    return () => window.clearTimeout(id);
  }, []);

  const backendUrlError = validateBackendUrl(backendUrl);
  const normalizedBackendUrl = normalizeBackendUrl(backendUrl);
  const canAnalyze =
    Boolean(file) &&
    !backendUrlError &&
    (backendStatus === "connected" || allowBackendOverride) &&
    status !== "uploading" &&
    status !== "analyzing";

  const setSelectedFile = useCallback(
    (nextFile: File | null) => {
      setError(null);
      setResponse(null);
      setVideoDuration(undefined);
      setReportState("idle");
      setArchiveState("idle");
      setArchiveMessage("");
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (!nextFile) {
        setFile(null);
        setPreviewUrl(null);
        setStatus("idle");
        return;
      }
      const validation = fileValidationError(nextFile);
      if (validation) {
        setFile(null);
        setPreviewUrl(null);
        setStatus("failed");
        setError(validation);
        return;
      }
      setFile(nextFile);
      setPreviewUrl(URL.createObjectURL(nextFile));
      setStatus("file-selected");
    },
    [previewUrl],
  );

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(event.target.files?.[0] || null);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setSelectedFile(event.dataTransfer.files?.[0] || null);
  };

  const checkBackend = useCallback(async (url = backendUrl) => {
    const validation = validateBackendUrl(url);
    if (validation) {
      setBackendStatus("disconnected");
      setBackendCheckedAt(new Date().toISOString());
      setError(validation);
      return;
    }
    setBackendStatus("checking");
    setError(null);
    try {
      const result = await fetch(buildApiUrl(url, BACKEND_HEALTH_PATH), {
        method: "GET",
        cache: "no-store",
        redirect: "follow",
      });
      setBackendStatus(result.ok ? "connected" : "disconnected");
      setBackendCheckedAt(new Date().toISOString());
      if (!result.ok) {
        setError(
          `Backend check failed with HTTP ${result.status}. Verify the backend URL and CORS allowed origins.`,
        );
      }
    } catch {
      setBackendStatus("disconnected");
      setBackendCheckedAt(new Date().toISOString());
      setError(backendReachabilityMessage(url));
    }
  }, [backendUrl]);

  useEffect(() => {
    if (!backendUrlLoaded) return;
    const id = window.setTimeout(() => {
      void checkBackend(backendUrl);
    }, 500);

    return () => window.clearTimeout(id);
  }, [backendUrl, backendUrlLoaded, checkBackend]);

  function handleBackendUrlChange(value: string) {
    setBackendUrl(value);
    setBackendStatus("unchecked");
    setBackendCheckedAt(null);
    setAllowBackendOverride(false);
    try {
      window.localStorage.setItem(BACKEND_URL_STORAGE_KEY, value);
    } catch {
      // Local storage is optional; the typed URL still applies for this session.
    }
  }

  const saveVideoResultToArchive = useCallback(
    async (payload: VideoUploadResponse) => {
      setArchiveState("saving");
      setArchiveMessage("Saving compact video analysis summary.");
      const archivePayload = buildVideoArchiveRunPayload(
        payload,
        getArchiveClientId(),
        currentUser?.id,
        {
          filename: file?.name,
          fileSizeBytes: file?.size,
          mimeType: file?.type,
          browserDurationSec: videoDuration,
          figureCount: figureDefinitions(normalizedBackendUrl, payload).length,
        },
      );
      const result = await saveVideoArchiveRun(archivePayload);
      if (result.ok) {
        setArchiveState("saved");
        setArchiveMessage("Saved compact video analysis summary.");
      } else {
        setArchiveState("failed");
        setArchiveMessage(
          result.error ||
            "Archive save failed. The upload analysis result is still available.",
        );
      }
    },
    [currentUser?.id, file, normalizedBackendUrl, videoDuration],
  );

  const analyzeVideo = async () => {
    if (!file) return;
    const validation = validateBackendUrl(backendUrl);
    if (validation) {
      setError(validation);
      setStatus("backend-unavailable");
      return;
    }
    if (backendStatus !== "connected" && !allowBackendOverride) {
      setError(
        "Backend is not connected. Run the backend check or enable the explicit override before analyzing."
      );
      setStatus("backend-unavailable");
      return;
    }

    setError(null);
    setResponse(null);
    setReportState("idle");
    setArchiveState("idle");
    setArchiveMessage("");
    setStatus("uploading");
    if (analyzingTimerRef.current) clearTimeout(analyzingTimerRef.current);
    analyzingTimerRef.current = setTimeout(() => setStatus("analyzing"), 900);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const result = await fetch(buildApiUrl(backendUrl, "/api/analyze-video"), {
        method: "POST",
        body: formData,
      });
      if (analyzingTimerRef.current) clearTimeout(analyzingTimerRef.current);

      if (!result.ok) {
        let payload: unknown = null;
        try {
          payload = await result.json();
        } catch {
          payload = null;
        }
        const message = usefulBackendError(
          payload,
          `Backend returned HTTP ${result.status}.`,
        );
        setStatus(result.status >= 500 ? "failed" : "backend-unavailable");
        setBackendStatus("connected");
        setBackendCheckedAt(new Date().toISOString());
        setError(message || `Backend returned HTTP ${result.status}.`);
        return;
      }

      const payload = (await result.json()) as VideoUploadResponse;
      setResponse(payload);
      setStatus("completed");
      setBackendStatus("connected");
      setBackendCheckedAt(new Date().toISOString());
      void saveVideoResultToArchive(payload);
      if (currentUser) {
        addNotification({
          id: `video-upload-${currentUser.id}-${payload.session_id || Date.now()}`,
          userId: currentUser.id,
          category: "review",
          severity: "success",
          title: "Video upload analysis completed",
          message:
            "Uploaded-video analysis finished and evidence is ready to inspect.",
          source: "video_upload",
          relatedRoute: "/video-upload",
        });
      }
    } catch {
      if (analyzingTimerRef.current) clearTimeout(analyzingTimerRef.current);
      setStatus("backend-unavailable");
      setBackendStatus("disconnected");
      setBackendCheckedAt(new Date().toISOString());
      setError(backendReachabilityMessage(backendUrl));
    }
  };

  const resetAll = () => {
    setSelectedFile(null);
    setResponse(null);
    setError(null);
    setReportState("idle");
    setArchiveState("idle");
    setArchiveMessage("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const clearResult = () => {
    setResponse(null);
    setReportState("idle");
    setArchiveState("idle");
    setArchiveMessage("");
    setStatus(file ? "file-selected" : "idle");
  };

  const downloadReport = () => {
    if (!response) return;
    try {
      downloadTextFile(
        videoUploadReportFilename(),
        buildVideoUploadReportHtml(response),
        "text/html;charset=utf-8",
      );
      setReportState("downloaded");
      window.setTimeout(() => setReportState("idle"), 1800);
    } catch {
      setReportState("failed");
    }
  };

  const intervals = useMemo(
    () => mergeWarningIntervals(response?.summary),
    [response],
  );
  const keyframes = response ? response.keyframes || response.summary.keyframes || [] : [];
  const figures = useMemo(
    () => figureDefinitions(normalizedBackendUrl, response || undefined),
    [normalizedBackendUrl, response],
  );

  return (
    <main className="h-full overflow-y-auto bg-[#f4f7f9] px-4 py-5 lg:px-6 xl:px-8">
      <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-6 pb-10">
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-slate-950">
                Video Upload Analysis
              </h1>
              <p className="mt-2 text-base font-medium text-slate-600">
                Upload a driving video to detect fatigue-related visual alerts
                and review supporting evidence.
              </p>
            </div>

            <div className="min-w-0 rounded-2xl border border-slate-200 bg-slate-50 p-4 xl:w-[360px]">
              <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                <Server className="h-4 w-4 text-blue-600" />
                Analysis backend
              </div>
              <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center">
                <span
                  className={`inline-flex w-fit rounded-full border px-2.5 py-1 text-xs font-semibold ${backendStatusTone(
                    backendStatus,
                  )}`}
                >
                  {backendStatusLabel(backendStatus)}
                </span>
              </div>
              <div className="mt-2 text-xs font-semibold text-slate-500">
                Last checked: {formatBackendCheckedAt(backendCheckedAt)}
              </div>
            </div>
          </div>
        </section>

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.05fr_0.95fr]">
          <SectionCard>
            <h2 className="text-lg font-bold text-slate-950">
              Upload video
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              Choose a driving video for backend-connected fatigue alert
              analysis.
            </p>

            <div
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
              className="mt-4 flex min-h-[190px] flex-col items-center justify-center rounded-2xl border-2 border-dashed border-blue-200 bg-blue-50/50 p-6 text-center transition hover:border-blue-300 hover:bg-blue-50"
            >
              <UploadCloud className="h-10 w-10 text-blue-600" />
              <div className="mt-3 text-base font-bold text-slate-950">
                Drop video here or select a file
              </div>
              <div className="mt-1 text-sm text-slate-600">
                Supported formats: MP4, MOV, AVI, M4V. Max upload size: 750 MB.
              </div>
              <label
                htmlFor="video-upload-file"
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                className="mt-4 cursor-pointer rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white outline-none transition hover:bg-blue-700 focus-visible:ring-2 focus-visible:ring-blue-400"
              >
                Click to select file
              </label>
              <input
                id="video-upload-file"
                ref={fileInputRef}
                type="file"
                accept=".mp4,.mov,.avi,.m4v,video/mp4,video/quicktime,video/x-msvideo"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>

            {file ? (
              <div className="mt-4 grid grid-cols-1 gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm sm:grid-cols-2 xl:grid-cols-4">
                <div>
                  <div className="text-xs font-semibold uppercase text-slate-400">
                    Filename
                  </div>
                  <div className="mt-1 break-words font-semibold text-slate-900">
                    {file.name}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase text-slate-400">
                    Size
                  </div>
                  <div className="mt-1 font-semibold text-slate-900">
                    {formatBytes(file.size)}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase text-slate-400">
                    Duration
                  </div>
                  <div className="mt-1 font-semibold text-slate-900">
                    {formatSeconds(videoDuration)}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase text-slate-400">
                    MIME type
                  </div>
                  <div className="mt-1 break-words font-semibold text-slate-900">
                    {file.type || "Not provided"}
                  </div>
                </div>
              </div>
            ) : null}

            <details className="mt-4 rounded-2xl border border-slate-200 bg-white">
              <summary className="cursor-pointer list-none px-4 py-3 text-sm font-bold text-slate-900 outline-none transition hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400">
                Advanced backend settings
              </summary>
              <div className="border-t border-slate-100 p-4">
                <label className="block">
                  <span className="text-sm font-semibold text-slate-900">
                    Backend URL
                  </span>
                  <input
                    value={backendUrl}
                    onChange={(event) =>
                      handleBackendUrlChange(event.target.value)
                    }
                    className={`mt-2 w-full rounded-xl border bg-white px-3 py-2.5 font-mono text-sm text-slate-900 outline-none transition focus-visible:ring-2 focus-visible:ring-blue-400 ${
                      backendUrlError ? "border-red-300" : "border-slate-200"
                    }`}
                    placeholder={DEFAULT_BACKEND_URL}
                  />
                </label>
                {backendUrlError ? (
                  <p className="mt-2 text-sm text-red-600">{backendUrlError}</p>
                ) : null}
                <label className="mt-3 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                  <input
                    type="checkbox"
                    checked={allowBackendOverride}
                    onChange={(event) =>
                      setAllowBackendOverride(event.target.checked)
                    }
                    className="mt-1 h-4 w-4 rounded border-amber-300"
                  />
                  <span>
                    Allow analysis without a connected backend check. Use only if
                    the health check is blocked but the analysis API is known to
                    be reachable.
                  </span>
                </label>
                <p className="mt-3 break-all text-xs text-slate-500">
                  Current backend URL: {normalizedBackendUrl}
                </p>
              </div>
            </details>

            <div className="mt-5 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={analyzeVideo}
                disabled={!canAnalyze}
                className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-bold text-white outline-none transition hover:bg-blue-700 focus-visible:ring-2 focus-visible:ring-blue-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <UploadCloud className="h-4 w-4" />
                Analyze Video
              </button>
              <button
                type="button"
                onClick={resetAll}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 outline-none transition hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
              {response ? (
                <button
                  type="button"
                  onClick={clearResult}
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 outline-none transition hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400"
                >
                  <XCircle className="h-4 w-4" />
                  Clear result
                </button>
              ) : null}
            </div>
            {!canAnalyze && file ? (
              <p className="mt-3 text-sm font-semibold text-slate-500">
                Analyze Video requires a valid file and connected backend, unless
                the explicit override is enabled.
              </p>
            ) : null}
          </SectionCard>

          <UploadedVideoPreview
            file={file}
            previewUrl={previewUrl}
            onDuration={setVideoDuration}
          />
        </div>

        <ProcessingStatus status={status} error={error} response={response} />

        {response ? (
          <>
            <ResultSectionNav />

            <div id="overview" className="scroll-mt-20">
              <ResultOverview response={response} figuresCount={figures.length} />
            </div>

            <div
              className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border p-4 text-sm shadow-sm ${
                archiveState === "failed"
                  ? "border-amber-200 bg-amber-50 text-amber-800"
                  : archiveState === "saved"
                    ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                    : "border-slate-200 bg-white text-slate-700"
              }`}
            >
              <div className="flex items-start gap-2">
                {archiveState === "saved" ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
                ) : archiveState === "failed" ? (
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                ) : (
                  <RefreshCw
                    className={`mt-0.5 h-4 w-4 shrink-0 ${
                      archiveState === "saving" ? "animate-spin" : ""
                    }`}
                  />
                )}
                <span>
                  {archiveMessage ||
                    "Compact video analysis summary can be saved after analysis."}
                </span>
              </div>
              {archiveState === "failed" ? (
                <button
                  type="button"
                  onClick={() => saveVideoResultToArchive(response)}
                  className="inline-flex items-center gap-2 rounded-lg border border-amber-200 bg-white px-3 py-2 text-xs font-semibold text-amber-800 outline-none transition hover:bg-amber-100 focus-visible:ring-2 focus-visible:ring-amber-300"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Retry save
                </button>
              ) : null}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div>
                <h2 className="text-lg font-bold text-slate-950">
                  Analysis Report
                </h2>
                <p className="mt-1 text-sm text-slate-600">
                  Download a user-readable HTML report for this video analysis.
                </p>
              </div>
              <button
                type="button"
                onClick={downloadReport}
                className="inline-flex items-center gap-2 rounded-xl border border-blue-200 bg-blue-50 px-4 py-2.5 text-sm font-bold text-blue-700 outline-none transition hover:bg-blue-100 focus-visible:ring-2 focus-visible:ring-blue-400"
              >
                <Download className="h-4 w-4" />
                {reportState === "downloaded"
                  ? "Downloaded"
                  : reportState === "failed"
                    ? "Download failed"
                    : "Download report"}
              </button>
            </div>

            <AnalysisSummaryCards response={response} />
            <div id="alert-intervals" className="scroll-mt-20">
              <IntervalReviewTable intervals={intervals} />
            </div>
            <div id="evidence-figures" className="scroll-mt-20">
              <EvidenceFigures figures={figures} />
            </div>
            <div id="keyframes" className="scroll-mt-20">
              <KeyframeEvidenceGallery
                backendUrl={normalizedBackendUrl}
                keyframes={keyframes}
              />
            </div>
            <div id="technical-details" className="scroll-mt-20">
              <TechnicalEvidencePanel
                backendUrl={normalizedBackendUrl}
                response={response}
              />
            </div>
          </>
        ) : (
          <section className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm leading-relaxed text-slate-600">
            <h2 className="text-lg font-bold text-slate-950">
              Analysis workspace
            </h2>
            <p className="mt-2">
              Select an uploaded-video file and run analysis to populate result
              summary, alert intervals, backend-generated evidence figures,
              keyframes, and advanced evidence links.
            </p>
          </section>
        )}
      </div>
    </main>
  );
}
