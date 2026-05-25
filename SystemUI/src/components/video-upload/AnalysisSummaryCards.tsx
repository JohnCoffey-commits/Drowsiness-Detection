import { ChevronRight } from "lucide-react";
import type { VideoUploadResponse } from "@/lib/videoUploadTypes";
import {
  formatNumber,
  formatProbability,
  formatSeconds,
} from "@/lib/videoUploadUtils";

interface MetricItem {
  label: string;
  value: string;
  note?: string;
  tone?: "neutral" | "blue" | "orange" | "red" | "pink" | "green" | "slate";
}

interface AnalysisSummaryCardsProps {
  response: VideoUploadResponse;
}

const toneClasses: Record<NonNullable<MetricItem["tone"]>, string> = {
  neutral: "border-slate-200 bg-white",
  blue: "border-blue-200 bg-blue-50/70",
  orange: "border-orange-200 bg-orange-50/70",
  red: "border-red-200 bg-red-50/70",
  pink: "border-pink-200 bg-pink-50/70",
  green: "border-emerald-200 bg-emerald-50/70",
  slate: "border-slate-300 bg-slate-100/70",
};

function MetricCard({ item }: { item: MetricItem }) {
  return (
    <div
      className={`rounded-xl border px-3 py-2.5 ${toneClasses[item.tone || "neutral"]}`}
    >
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        {item.label}
      </div>
      <div className="mt-1 break-words text-base font-bold leading-tight text-slate-900">
        {item.value}
      </div>
      {item.note ? (
        <div className="mt-1 text-xs leading-snug text-slate-500">
          {item.note}
        </div>
      ) : null}
    </div>
  );
}

function TechnicalDetails({
  title,
  description,
  items,
  note,
}: {
  title: string;
  description: string;
  items: MetricItem[];
  note?: string;
}) {
  return (
    <details className="group rounded-2xl border border-slate-200 bg-white shadow-sm">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-4 outline-none focus-visible:ring-2 focus-visible:ring-blue-400">
        <span>
          <span className="block text-base font-bold text-slate-950">
            {title}
          </span>
          <span className="mt-1 block text-sm text-slate-600">
            {description}
          </span>
        </span>
        <ChevronRight className="h-5 w-5 shrink-0 text-slate-500 transition-transform group-open:rotate-90" />
      </summary>
      <div className="border-t border-slate-100 p-4">
        {note ? (
          <p className="mb-3 rounded-lg border border-blue-100 bg-blue-50 px-3 py-2 text-xs leading-relaxed text-blue-800">
            {note}
          </p>
        ) : null}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {items.map((item) => (
            <MetricCard key={item.label} item={item} />
          ))}
        </div>
      </div>
    </details>
  );
}

export function AnalysisSummaryCards({ response }: AnalysisSummaryCardsProps) {
  const summary = response.summary || {};
  const alertItems: MetricItem[] = [
    {
      label: "Normal frames",
      value: formatNumber(summary.normal_frames),
      tone: "green",
    },
    {
      label: "High-risk eye alert frames",
      value: formatNumber(summary.high_confidence_drowsiness_candidate_frames),
      tone: "red",
    },
    {
      label: "Yawn alert frames",
      value: formatNumber(summary.mouth_warning_candidate_frames),
      tone: "pink",
    },
    {
      label: "Eye-closure alert frames",
      value: formatNumber(summary.eye_warning_candidate_frames),
      tone: "orange",
    },
    {
      label: "Camera signal interruption frames",
      value: formatNumber(summary.signal_unreliable_frames),
      tone: "slate",
    },
  ];
  const totalKnownFrames = alertItems.reduce((sum, item) => {
    const numeric = Number(item.value.replace(/,/g, ""));
    return Number.isFinite(numeric) ? sum + numeric : sum;
  }, 0);
  const modelItems: MetricItem[] = [
    {
      label: "Yawn event count",
      value: formatNumber(summary.yawn_event_count),
      tone: "pink",
    },
    {
      label: "Recent-yawn count",
      value: formatNumber(summary.recent_yawn_event_count),
    },
    {
      label: "Mean p_yawn",
      value: formatProbability(summary.mean_p_yawn),
    },
    {
      label: "Max p_yawn",
      value: formatProbability(summary.max_p_yawn),
      note: "Specialist model output",
    },
    {
      label: "Mean p_eye_closed",
      value: formatProbability(summary.mean_p_eye_closed),
    },
    {
      label: "Max p_eye_closed",
      value: formatProbability(summary.max_p_eye_closed),
      note: "Specialist model output",
    },
  ];
  const safeguardItems: MetricItem[] = [
    {
      label: "Suppressed brief-eye escalation frames",
      value: formatNumber(
        summary.suppressed_high_confidence_brief_eye_warning_frames,
      ),
      tone: "orange",
    },
    {
      label: "Gate min duration",
      value: formatSeconds(summary.sustained_eye_gate_min_duration_sec),
    },
    {
      label: "Gate min sampled frames",
      value: formatNumber(summary.sustained_eye_gate_min_sampled_frames),
    },
  ];
  const calibrationItems: MetricItem[] = [
    {
      label: "Weak eye evidence frames",
      value: formatNumber(
        summary.weak_eye_warning_candidate_frames ??
          summary.weak_eye_warning_evidence_frames,
      ),
      note: "Within eye-alert rows",
      tone: "orange",
    },
    {
      label: "Moderate eye evidence frames",
      value: formatNumber(summary.moderate_eye_closure_candidate_frames),
      note: "Across sampled timeline",
      tone: "orange",
    },
    {
      label: "Strong eye evidence frames",
      value: formatNumber(summary.strong_eye_closure_candidate_frames),
      note: "Across sampled timeline",
      tone: "red",
    },
  ];

  return (
    <section className="space-y-4" aria-labelledby="alert-summary-title">
      <div>
        <h2 id="alert-summary-title" className="text-xl font-bold text-slate-950">
          Alert Summary
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          High-level frame distribution from the uploaded-video analysis.
        </p>
      </div>

      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
          {alertItems.map((item) => (
            <MetricCard key={item.label} item={item} />
          ))}
        </div>
        <div className="mt-4 overflow-hidden rounded-full bg-slate-100">
          <div className="flex h-3">
            {alertItems.map((item) => {
              const numeric = Number(item.value.replace(/,/g, ""));
              if (!Number.isFinite(numeric) || numeric <= 0 || totalKnownFrames <= 0) {
                return null;
              }
              const width = Math.max(2, (numeric / totalKnownFrames) * 100);
              const color =
                item.tone === "red"
                  ? "bg-red-500"
                  : item.tone === "orange"
                    ? "bg-orange-500"
                    : item.tone === "pink"
                      ? "bg-pink-500"
                      : item.tone === "slate"
                        ? "bg-slate-500"
                        : "bg-emerald-500";
              return (
                <span
                  key={item.label}
                  className={color}
                  title={`${item.label}: ${item.value}`}
                  style={{ width: `${width}%` }}
                />
              );
            })}
          </div>
        </div>
      </section>

      <TechnicalDetails
        title="Technical model evidence"
        description="Specialist model outputs and temporal evidence values."
        items={modelItems}
      />

      <TechnicalDetails
        title="Technical safeguards"
        description="Backend gates and suppression fields used to reduce brief or weak escalation."
        items={safeguardItems}
      />

      <TechnicalDetails
        title="Eye evidence calibration"
        description="Optional backend-provided eye-evidence strength fields."
        items={calibrationItems}
      />
    </section>
  );
}
