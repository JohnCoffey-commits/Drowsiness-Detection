import {
  PERMANENT_WARNING,
  type VideoUploadResponse,
} from "@/lib/videoUploadTypes";
import {
  formatNumber,
  formatProbability,
  formatSeconds,
  hasStage175SummaryFields,
  stage175MetricScopeNote,
} from "@/lib/videoUploadUtils";

interface MetricItem {
  label: string;
  value: string;
  note?: string;
  tone?: "neutral" | "blue" | "orange" | "red" | "pink" | "green" | "slate";
}

interface MetricGroup {
  title: string;
  description: string;
  note?: string;
  items: MetricItem[];
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
      className={`rounded-xl border p-3 shadow-sm ${toneClasses[item.tone || "neutral"]}`}
    >
      <div className="text-[11px] font-semibold uppercase text-slate-500">
        {item.label}
      </div>
      <div className="mt-1 break-words text-lg font-bold leading-tight text-slate-900">
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

function MetricGroupCard({ group }: { group: MetricGroup }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3">
        <h3 className="text-sm font-bold text-slate-900">{group.title}</h3>
        <p className="mt-1 text-xs leading-relaxed text-slate-500">
          {group.description}
        </p>
        {group.note ? (
          <p className="mt-2 rounded-lg border border-blue-100 bg-blue-50 px-3 py-2 text-xs leading-relaxed text-blue-800">
            {group.note}
          </p>
        ) : null}
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {group.items.map((item) => (
          <MetricCard key={item.label} item={item} />
        ))}
      </div>
    </section>
  );
}

export function AnalysisSummaryCards({ response }: AnalysisSummaryCardsProps) {
  const summary = response.summary || {};
  const hasStage175 = hasStage175SummaryFields(summary);

  const groups: MetricGroup[] = [
    {
      title: "Fusion State Counts",
      description: "Frame counts returned by the rule-based fusion backend.",
      items: [
        {
          label: "Normal frames",
          value: formatNumber(summary.normal_frames),
          tone: "green",
        },
        {
          label: "Eye-warning candidate frames",
          value: formatNumber(summary.eye_warning_candidate_frames),
          tone: "orange",
        },
        {
          label: "Mouth-warning candidate frames",
          value: formatNumber(summary.mouth_warning_candidate_frames),
          tone: "pink",
        },
        {
          label: "High-confidence warning candidate frames",
          value: formatNumber(summary.high_confidence_drowsiness_candidate_frames),
          tone: "red",
        },
        {
          label: "Signal-unreliable frames",
          value: formatNumber(summary.signal_unreliable_frames),
          tone: "slate",
        },
      ],
    },
    {
      title: "Mouth / Yawn Metrics",
      description: "Specialist mouth/yawn model evidence for uploaded-video review.",
      items: [
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
      ],
    },
    {
      title: "Eye Metrics",
      description: "Specialist eye model evidence and eye-warning candidate counts.",
      items: [
        {
          label: "Mean p_eye_closed",
          value: formatProbability(summary.mean_p_eye_closed),
        },
        {
          label: "Max p_eye_closed",
          value: formatProbability(summary.max_p_eye_closed),
          note: "Specialist model output",
        },
        {
          label: "Eye-warning candidate frames",
          value: formatNumber(summary.eye_warning_candidate_frames),
          tone: "orange",
        },
        {
          label: "Signal-unreliable frames",
          value: formatNumber(summary.signal_unreliable_frames),
          tone: "slate",
        },
      ],
    },
    {
      title: "Stage 17.1 Sustained-Eye Gate",
      description:
        "Rule-based gate that suppresses brief blink-like escalation before high-confidence warning candidates are retained.",
      items: [
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
      ],
    },
  ];

  const stage175Group: MetricGroup = {
    title: "Stage 17.5 Eye Evidence Calibration",
    description:
      "Backend-provided calibration fields for sampled eye evidence. These are descriptive technical evidence fields, not recomputed fusion states.",
    note: stage175MetricScopeNote(),
    items: [
      {
        label: "Weak eye evidence frames",
        value: formatNumber(
          summary.weak_eye_warning_candidate_frames ??
            summary.weak_eye_warning_evidence_frames,
        ),
        note: "Within eye-warning candidate rows",
        tone: "orange",
      },
      {
        label: "Moderate eye evidence frames",
        value: formatNumber(summary.moderate_eye_closure_candidate_frames),
        note: "Across sampled timeline",
        tone: "orange",
      },
      {
        label: "Strong eye-closure candidate frames",
        value: formatNumber(summary.strong_eye_closure_candidate_frames),
        note: "Across sampled timeline",
        tone: "red",
      },
      {
        label: "Reduced eye openness candidate frames",
        value: formatNumber(summary.reduced_eye_openness_candidate_frames),
        note: "Backend-provided if available",
      },
      {
        label: "Manual review recommended eye frames",
        value: formatNumber(summary.manual_review_recommended_eye_frames),
        note: "Backend-provided if available",
      },
      {
        label: "Suppressed weak-eye escalation frames",
        value: formatNumber(
          summary.high_confidence_suppressed_by_weak_eye_evidence_frames ??
            summary.suppressed_high_confidence_weak_eye_evidence_frames,
        ),
        note: "Suppressed from high-confidence by calibrated weak eye evidence",
        tone: "slate",
      },
    ],
  };

  return (
    <section className="space-y-4" aria-labelledby="summary-metrics-title">
      <div>
        <h2 id="summary-metrics-title" className="text-xl font-bold text-slate-950">
          Summary Metrics
        </h2>
        <p className="mt-1 text-sm text-slate-600">{PERMANENT_WARNING}</p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {groups.map((group) => (
          <MetricGroupCard key={group.title} group={group} />
        ))}

        {hasStage175 ? (
          <MetricGroupCard group={stage175Group} />
        ) : (
          <section className="rounded-2xl border border-dashed border-slate-300 bg-white p-4 text-sm text-slate-600">
            <h3 className="font-bold text-slate-900">
              Stage 17.5 Eye Evidence Calibration
            </h3>
            <p className="mt-1">
              Eye evidence strength fields are not present in this backend
              response.
            </p>
          </section>
        )}
      </div>
    </section>
  );
}
