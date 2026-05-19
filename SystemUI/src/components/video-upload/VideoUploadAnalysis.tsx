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
  Clipboard,
  ClipboardCheck,
  FileVideo,
  RefreshCw,
  RotateCcw,
  Server,
  UploadCloud,
  XCircle,
} from "lucide-react";
import { AnalysisSummaryCards } from "@/components/video-upload/AnalysisSummaryCards";
import { EvidenceFigures } from "@/components/video-upload/EvidenceFigures";
import { InterpretationNotice } from "@/components/video-upload/InterpretationNotice";
import { IntervalReviewTable } from "@/components/video-upload/IntervalReviewTable";
import { KeyframeEvidenceGallery } from "@/components/video-upload/KeyframeEvidenceGallery";
import { TechnicalEvidencePanel } from "@/components/video-upload/TechnicalEvidencePanel";
import {
  type AnalysisStatus,
  type BackendStatus,
  PERMANENT_WARNING,
  type VideoUploadResponse,
} from "@/lib/videoUploadTypes";
import {
  buildApiUrl,
  buildCopySummary,
  DEFAULT_BACKEND_URL,
  figureDefinitions,
  formatBytes,
  formatNumber,
  formatSeconds,
  mergeWarningIntervals,
  normalizeBackendUrl,
  resultMessage,
  sanitizeBrowserText,
  validateBackendUrl,
} from "@/lib/videoUploadUtils";
import { useVisionGuardAuth } from "@/lib/authStore";
import { useVisionGuardNotifications } from "@/lib/notificationStore";

const ACCEPTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".m4v"];

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
      return "connected";
    case "failed":
      return "failed";
    case "checking":
      return "checking";
    case "unchecked":
      return "unchecked";
  }
}

function backendStatusTone(status: BackendStatus): string {
  switch (status) {
    case "connected":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "failed":
      return "border-red-200 bg-red-50 text-red-700";
    case "checking":
      return "border-blue-200 bg-blue-50 text-blue-700";
    case "unchecked":
      return "border-slate-200 bg-slate-50 text-slate-600";
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
  duration,
  onDuration,
}: {
  file: File | null;
  previewUrl: string | null;
  duration?: number;
  onDuration: (value?: number) => void;
}) {
  return (
    <SectionCard>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-950">
            Uploaded video preview
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Native browser preview for the selected upload. It does not auto-play
            and is not a live feed.
          </p>
        </div>
        <FileVideo className="h-5 w-5 shrink-0 text-blue-600" />
      </div>

      {file && previewUrl ? (
        <div className="mt-4 space-y-4">
          <video
            controls
            preload="metadata"
            src={previewUrl}
            className="aspect-video w-full rounded-xl border border-slate-200 bg-slate-950"
            onLoadedMetadata={(event) =>
              onDuration(event.currentTarget.duration || undefined)
            }
          />
          <dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
            <div className="rounded-xl bg-slate-50 p-3">
              <dt className="text-xs font-semibold uppercase text-slate-400">
                File name
              </dt>
              <dd className="mt-1 break-words font-semibold text-slate-900">
                {file.name}
              </dd>
            </div>
            <div className="rounded-xl bg-slate-50 p-3">
              <dt className="text-xs font-semibold uppercase text-slate-400">
                File size
              </dt>
              <dd className="mt-1 font-semibold text-slate-900">
                {formatBytes(file.size)}
              </dd>
            </div>
            <div className="rounded-xl bg-slate-50 p-3">
              <dt className="text-xs font-semibold uppercase text-slate-400">
                MIME type
              </dt>
              <dd className="mt-1 break-words font-semibold text-slate-900">
                {file.type || "Not provided by browser"}
              </dd>
            </div>
            <div className="rounded-xl bg-slate-50 p-3">
              <dt className="text-xs font-semibold uppercase text-slate-400">
                Browser duration
              </dt>
              <dd className="mt-1 font-semibold text-slate-900">
                {formatSeconds(duration)}
              </dd>
            </div>
          </dl>
        </div>
      ) : (
        <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-sm text-slate-500">
          Select an uploaded-video file to preview it here.
        </div>
      )}
    </SectionCard>
  );
}

function ProcessingStatus({
  status,
  error,
}: {
  status: AnalysisStatus;
  error: string | null;
}) {
  const activeIndex =
    status === "uploading" ? 0 : status === "analyzing" ? 5 : status === "completed" ? 7 : -1;

  return (
    <SectionCard>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-950">Processing Status</h2>
          <p className="mt-1 text-sm text-slate-600">
            The backend is processing the uploaded video. Step indicators are an
            approximate UI guide.
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

      <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
        {processingSteps.map((step, index) => {
          const complete = status === "completed" || (activeIndex > index && activeIndex !== -1);
          const active = activeIndex === index || (status === "analyzing" && index >= 1 && index <= 7);
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

      {error ? (
        <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <div className="flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
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
  const statusItems = [
    ["Session", response.session_id],
    ["Status", response.status || summary.pipeline_status || "Not available"],
    ["Runtime", formatSeconds(response.runtime_duration_sec ?? summary.runtime_sec)],
    ["Sampled frames", formatNumber(summary.total_frames_sampled)],
    ["Analyzed", formatSeconds(summary.duration_sec)],
    ["Keyframes", formatNumber(keyframeCount)],
    ["Figures", formatNumber(figuresCount)],
  ];

  return (
    <SectionCard className="border-blue-200 bg-blue-50/30 p-4">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-950">Result Overview</h2>
          <p className="mt-1 text-sm font-semibold text-blue-700">
            {resultMessage(summary)}
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
        {PERMANENT_WARNING}
      </p>
    </SectionCard>
  );
}

export function VideoUploadAnalysis() {
  const { currentUser } = useVisionGuardAuth();
  const { addNotification } = useVisionGuardNotifications();
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("unchecked");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number | undefined>();
  const [status, setStatus] = useState<AnalysisStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<VideoUploadResponse | null>(null);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const analyzingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (analyzingTimerRef.current) clearTimeout(analyzingTimerRef.current);
    };
  }, [previewUrl]);

  const backendUrlError = validateBackendUrl(backendUrl);
  const normalizedBackendUrl = normalizeBackendUrl(backendUrl);
  const canAnalyze = Boolean(file) && !backendUrlError && status !== "uploading" && status !== "analyzing";

  const setSelectedFile = useCallback(
    (nextFile: File | null) => {
      setError(null);
      setResponse(null);
      setVideoDuration(undefined);
      setCopyState("idle");
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

  const checkBackend = useCallback(async () => {
    const validation = validateBackendUrl(backendUrl);
    if (validation) {
      setBackendStatus("failed");
      setError(validation);
      return;
    }
    setBackendStatus("checking");
    setError(null);
    try {
      const result = await fetch(buildApiUrl(backendUrl, "/static/upload_test.html"), {
        method: "GET",
        cache: "no-store",
      });
      setBackendStatus(result.ok ? "connected" : "failed");
      if (!result.ok) {
        setError(`Backend check failed with HTTP ${result.status}.`);
      }
    } catch {
      setBackendStatus("failed");
      setError("Backend is not reachable at the configured URL.");
    }
  }, [backendUrl]);

  const analyzeVideo = async () => {
    if (!file) return;
    const validation = validateBackendUrl(backendUrl);
    if (validation) {
      setError(validation);
      setStatus("backend-unavailable");
      return;
    }

    setError(null);
    setResponse(null);
    setCopyState("idle");
    setStatus("uploading");
    setBackendStatus("unchecked");
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
        setBackendStatus("failed");
        setError(message || `Backend returned HTTP ${result.status}.`);
        return;
      }

      const payload = (await result.json()) as VideoUploadResponse;
      setResponse(payload);
      setStatus("completed");
      setBackendStatus("connected");
      if (currentUser) {
        addNotification({
          id: `video-upload-${currentUser.id}-${payload.session_id || Date.now()}`,
          userId: currentUser.id,
          category: "review",
          severity: "success",
          title: "Video upload analysis completed",
          message:
            "Uploaded-video warning-candidate analysis finished and is ready for review.",
          source: "video_upload",
          relatedRoute: "/video-upload",
        });
      }
    } catch {
      if (analyzingTimerRef.current) clearTimeout(analyzingTimerRef.current);
      setStatus("backend-unavailable");
      setBackendStatus("failed");
      setError("Backend request failed. Check the backend URL and retry.");
    }
  };

  const resetAll = () => {
    setSelectedFile(null);
    setResponse(null);
    setError(null);
    setCopyState("idle");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const clearResult = () => {
    setResponse(null);
    setCopyState("idle");
    setStatus(file ? "file-selected" : "idle");
  };

  const copySummary = async () => {
    if (!response) return;
    try {
      await navigator.clipboard.writeText(buildCopySummary(response));
      setCopyState("copied");
      window.setTimeout(() => setCopyState("idle"), 1800);
    } catch {
      setCopyState("failed");
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
              <div className="flex flex-wrap gap-2">
                {[
                  "Uploaded Video",
                  "Rule-based Fusion",
                  "Warning Candidate",
                  "Not Webcam",
                  "Technical Evidence",
                  "Eye Evidence Aware",
                ].map((badge) => (
                  <span
                    key={badge}
                    className="rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700"
                  >
                    {badge}
                  </span>
                ))}
              </div>
              <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-950">
                Video Upload Analysis
              </h1>
              <p className="mt-2 text-base font-medium text-slate-600">
                Uploaded-video rule-based warning-candidate review
              </p>
              <p className="mt-3 max-w-4xl text-sm leading-relaxed text-slate-600">
                {PERMANENT_WARNING}
              </p>
            </div>

            <div className="min-w-0 rounded-2xl border border-slate-200 bg-slate-50 p-4 xl:w-[430px]">
              <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                <Server className="h-4 w-4 text-blue-600" />
                Backend status
              </div>
              <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center">
                <span
                  className={`inline-flex w-fit rounded-full border px-2.5 py-1 text-xs font-semibold ${backendStatusTone(
                    backendStatus,
                  )}`}
                >
                  {backendStatusLabel(backendStatus)}
                </span>
                <span className="break-all font-mono text-xs text-slate-600">
                  {normalizedBackendUrl}
                </span>
              </div>
              <button
                type="button"
                onClick={checkBackend}
                disabled={backendStatus === "checking"}
                className="mt-3 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 outline-none transition hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <RefreshCw
                  className={`h-3.5 w-3.5 ${backendStatus === "checking" ? "animate-spin" : ""}`}
                />
                Check backend
              </button>
            </div>
          </div>
        </section>

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.05fr_0.95fr]">
          <SectionCard>
            <h2 className="text-lg font-bold text-slate-950">
              Upload & Backend Control
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              Upload a local video file for backend-connected evidence review.
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
              <div className="mt-4 grid grid-cols-1 gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm sm:grid-cols-3">
                <div>
                  <div className="text-xs font-semibold uppercase text-slate-400">
                    File
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
                    MIME type
                  </div>
                  <div className="mt-1 break-words font-semibold text-slate-900">
                    {file.type || "Not provided"}
                  </div>
                </div>
              </div>
            ) : null}

            <label className="mt-4 block">
              <span className="text-sm font-semibold text-slate-900">
                Backend URL
              </span>
              <input
                value={backendUrl}
                onChange={(event) => {
                  setBackendUrl(event.target.value);
                  setBackendStatus("unchecked");
                }}
                className={`mt-2 w-full rounded-xl border bg-white px-3 py-2.5 font-mono text-sm text-slate-900 outline-none transition focus-visible:ring-2 focus-visible:ring-blue-400 ${
                  backendUrlError ? "border-red-300" : "border-slate-200"
                }`}
                placeholder="http://127.0.0.1:8000"
              />
            </label>
            {backendUrlError ? (
              <p className="mt-2 text-sm text-red-600">{backendUrlError}</p>
            ) : null}

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
          </SectionCard>

          <UploadedVideoPreview
            file={file}
            previewUrl={previewUrl}
            duration={videoDuration}
            onDuration={setVideoDuration}
          />
        </div>

        <ProcessingStatus status={status} error={error} />

        {response ? (
          <>
            <ResultOverview response={response} figuresCount={figures.length} />

            <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div>
                <h2 className="text-lg font-bold text-slate-950">
                  Copyable Review Summary
                </h2>
                <p className="mt-1 text-sm text-slate-600">
                  Copies safe warning-candidate wording and omits local server
                  paths.
                </p>
              </div>
              <button
                type="button"
                onClick={copySummary}
                className="inline-flex items-center gap-2 rounded-xl border border-blue-200 bg-blue-50 px-4 py-2.5 text-sm font-bold text-blue-700 outline-none transition hover:bg-blue-100 focus-visible:ring-2 focus-visible:ring-blue-400"
              >
                {copyState === "copied" ? (
                  <ClipboardCheck className="h-4 w-4" />
                ) : (
                  <Clipboard className="h-4 w-4" />
                )}
                {copyState === "copied"
                  ? "Copied"
                  : copyState === "failed"
                    ? "Copy failed"
                    : "Copy summary"}
              </button>
            </div>

            <AnalysisSummaryCards response={response} />
            <IntervalReviewTable intervals={intervals} />
            <EvidenceFigures figures={figures} />
            <KeyframeEvidenceGallery
              backendUrl={normalizedBackendUrl}
              keyframes={keyframes}
            />
            <InterpretationNotice />
            <TechnicalEvidencePanel
              backendUrl={normalizedBackendUrl}
              response={response}
            />
          </>
        ) : (
          <section className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm leading-relaxed text-slate-600">
            <h2 className="text-lg font-bold text-slate-950">
              Evidence Review Workspace
            </h2>
            <p className="mt-2">
              Select an uploaded-video file and run analysis to populate result
              overview, summary metrics, warning-candidate intervals, evidence
              figures, keyframes, interpretation notes, and technical evidence
              links.
            </p>
          </section>
        )}
      </div>
    </main>
  );
}
