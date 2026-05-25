"use client";

import { Fragment, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import type { MergedWarningInterval } from "@/lib/videoUploadTypes";
import {
  compactEyeEvidenceLabel,
  eyeEvidenceDescription,
  eyeEvidenceLabel,
  formatOptionalBooleanCompact,
  formatOptionalTextCompact,
  formatProbability,
  formatSeconds,
  fusionStateDescription,
  fusionStateLabel,
  hasStage175IntervalFields,
  intervalDuration,
  intervalFrameCount,
  primaryEvidenceSummary,
  stateTone,
} from "@/lib/videoUploadUtils";

interface IntervalReviewTableProps {
  intervals: MergedWarningInterval[];
}

function intervalEyeEvidence(interval: MergedWarningInterval) {
  return {
    strength:
      interval.dominant_eye_evidence_strength || interval.eye_evidence_strength,
    level: interval.dominant_eye_evidence_level,
    label: interval.eye_evidence_label,
    interpretation: interval.eye_evidence_interpretation,
  };
}

function fusionReason(interval: MergedWarningInterval): string {
  const reason = interval.reason?.trim();
  if (reason) return reason.replace(/recent yawn event/gi, "recent mouth/yawn evidence");
  if (interval.state === "mouth_warning_candidate") {
    return "recent mouth/yawn evidence";
  }
  return fusionStateDescription(interval.state);
}

function DetailItem({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="text-[11px] font-semibold uppercase text-slate-400">
        {label}
      </div>
      <div className="mt-1 text-xs leading-relaxed text-slate-700">
        {children}
      </div>
    </div>
  );
}

function IntervalDetails({ interval }: { interval: MergedWarningInterval }) {
  const stage175 = hasStage175IntervalFields(interval);
  const evidence = intervalEyeEvidence(interval);
  const expandedEvidence = stage175
    ? eyeEvidenceLabel(evidence.level, evidence.strength, evidence.label)
    : "—";
  const evidenceDescription = stage175
    ? eyeEvidenceDescription(
        evidence.level,
        evidence.strength,
        evidence.interpretation,
      )
    : "—";
  const gateReason = formatOptionalTextCompact(interval.eye_strength_gate_reason);
  const strengthReason = formatOptionalTextCompact(
    interval.eye_warning_strength_reason,
  );
  const hasSuppression =
    interval.high_confidence_suppressed_by_brief_eye_warning ||
    interval.high_confidence_suppressed_by_weak_eye_evidence;

  return (
    <div className="space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        <DetailItem label="Fusion state explanation">
          {fusionStateDescription(interval.state)}
        </DetailItem>
        <DetailItem label="Fusion state reason">
          {fusionReason(interval)}
        </DetailItem>
        <DetailItem label="Peak eye evidence">
          <span className="font-semibold text-slate-900">{expandedEvidence}</span>
          <p className="mt-1">
            This describes eye-probability evidence within the interval. It does
            not recompute or override the backend fusion state.
          </p>
        </DetailItem>
        <DetailItem label="Model probabilities">
          <p>Max p_eye_closed: {formatProbability(interval.max_p_eye_closed)}</p>
          <p>Max p_yawn: {formatProbability(interval.max_p_yawn)}</p>
          <p>Mean p_eye_closed: {formatProbability(interval.mean_p_eye_closed)}</p>
          <p>Mean p_yawn: {formatProbability(interval.mean_p_yawn)}</p>
        </DetailItem>
        <DetailItem label="Eye evidence explanation">
          {evidenceDescription}
        </DetailItem>
        <DetailItem label="Sustained-eye gate">
          {formatOptionalBooleanCompact(interval.sustained_eye_warning)}
          {interval.eye_warning_interval_duration_sec != null ||
          interval.eye_warning_interval_sampled_frames != null ? (
            <p className="mt-1">
              Interval duration{" "}
              {formatSeconds(interval.eye_warning_interval_duration_sec)};
              sampled frames{" "}
              {formatOptionalTextCompact(interval.eye_warning_interval_sampled_frames)}.
            </p>
          ) : null}
        </DetailItem>
        <DetailItem label="Interval eye-strength gate">
          {formatOptionalBooleanCompact(interval.eye_strength_gate_passed)}
          <p className="mt-1">
            Evaluated over the eye-warning interval, not a single table row or
            keyframe.
          </p>
        </DetailItem>
        {gateReason !== "—" ? (
          <DetailItem label="Eye evidence calibration reason">
            {gateReason}
          </DetailItem>
        ) : null}
        {strengthReason !== "—" ? (
          <DetailItem label="Eye evidence strength reason">
            {strengthReason}
          </DetailItem>
        ) : null}
        {hasSuppression ? (
          <DetailItem label="Suppression reason">
            {interval.high_confidence_suppressed_by_brief_eye_warning ? (
              <p>High-confidence escalation suppressed: brief eye-warning interval.</p>
            ) : null}
            {interval.high_confidence_suppressed_by_weak_eye_evidence ? (
              <p>
                High-confidence escalation suppressed: calibrated eye evidence
                remained weak or reduced-eye-openness evidence only.
              </p>
            ) : null}
          </DetailItem>
        ) : null}
      </div>
      <p className="text-xs leading-relaxed text-slate-500">
        The UI formats backend output only; it does not recompute or override
        the backend fusion state.
      </p>
    </div>
  );
}

function DetailsButton({
  expanded,
  onClick,
}: {
  expanded: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={expanded}
      className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 outline-none transition hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400"
    >
      <ChevronDown
        className={`h-3.5 w-3.5 transition-transform ${
          expanded ? "rotate-180" : ""
        }`}
      />
      {expanded ? "Hide" : "Details"}
    </button>
  );
}

function MobileIntervalCard({
  interval,
  expanded,
  onToggle,
}: {
  interval: MergedWarningInterval;
  expanded: boolean;
  onToggle: () => void;
}) {
  const stage175 = hasStage175IntervalFields(interval);
  const evidence = intervalEyeEvidence(interval);
  const primaryEvidence = primaryEvidenceSummary(interval);

  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span
          className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${stateTone(
            interval.state,
          )}`}
        >
          {fusionStateLabel(interval.state)}
        </span>
        <DetailsButton expanded={expanded} onClick={onToggle} />
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-3 text-sm">
        <div>
          <dt className="text-xs font-semibold uppercase text-slate-400">
            Time
          </dt>
          <dd className="font-semibold text-slate-900">
            {formatSeconds(interval.start_timestamp_sec)} -{" "}
            {formatSeconds(interval.end_timestamp_sec)}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-semibold uppercase text-slate-400">
            Frames
          </dt>
          <dd className="font-semibold text-slate-900">
            {intervalFrameCount(interval) ?? "—"}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-semibold uppercase text-slate-400">
            Evidence
          </dt>
          <dd className="font-semibold text-slate-900">
            {primaryEvidence}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-semibold uppercase text-slate-400">
            Strength
          </dt>
          <dd className="font-semibold text-slate-900">
            {stage175
              ? compactEyeEvidenceLabel(
                  evidence.strength,
                  evidence.level,
                  evidence.label,
                )
              : "—"}
          </dd>
        </div>
      </dl>
      {expanded ? (
        <div className="mt-3">
          <IntervalDetails interval={interval} />
        </div>
      ) : null}
    </article>
  );
}

export function IntervalReviewTable({ intervals }: IntervalReviewTableProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const toggle = (id: string) =>
    setExpandedId((current) => (current === id ? null : id));

  return (
    <section className="space-y-4" aria-labelledby="intervals-title">
      <div>
        <h2 id="intervals-title" className="text-xl font-bold text-slate-950">
          Alert Intervals
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Normal intervals are omitted. Backend fusion states remain the source
          of truth; details include the technical probability and gate fields.
        </p>
      </div>

      {intervals.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-5 text-sm text-slate-600">
          No alert intervals were returned.
        </div>
      ) : (
        <>
          <div className="space-y-3 xl:hidden">
            {intervals.map((interval) => (
              <MobileIntervalCard
                key={interval.id}
                interval={interval}
                expanded={expandedId === interval.id}
                onToggle={() => toggle(interval.id)}
              />
            ))}
          </div>

          <div className="hidden rounded-2xl border border-slate-200 bg-white shadow-sm xl:block">
            <table className="w-full table-fixed border-collapse text-left text-xs">
              <thead className="bg-slate-50 text-xs font-semibold uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="w-[17%] px-2 py-3">Alert</th>
                  <th className="w-[8%] px-2 py-3">Start</th>
                  <th className="w-[8%] px-2 py-3">End</th>
                  <th className="w-[9%] px-2 py-3">Duration</th>
                  <th className="w-[8%] px-2 py-3">Frames</th>
                  <th className="w-[21%] px-2 py-3">Evidence</th>
                  <th
                    className="w-[17%] px-2 py-3"
                    title="Descriptive eye-probability evidence within the interval; does not override backend fusion state."
                  >
                    Strength
                  </th>
                  <th className="w-[12%] px-2 py-3">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {intervals.map((interval) => {
                  const stage175 = hasStage175IntervalFields(interval);
                  const evidence = intervalEyeEvidence(interval);
                  const expanded = expandedId === interval.id;
                  const primaryEvidence = primaryEvidenceSummary(interval);
                  return (
                    <Fragment key={interval.id}>
                      <tr className="align-middle">
                        <td className="px-2 py-2">
                          <span
                            className={`inline-flex rounded-full border px-2 py-1 font-semibold leading-tight ${stateTone(
                              interval.state,
                            )}`}
                          >
                            {fusionStateLabel(interval.state)}
                          </span>
                        </td>
                        <td className="whitespace-nowrap px-2 py-2 font-mono text-slate-700">
                          {formatSeconds(interval.start_timestamp_sec)}
                        </td>
                        <td className="whitespace-nowrap px-2 py-2 font-mono text-slate-700">
                          {formatSeconds(interval.end_timestamp_sec)}
                        </td>
                        <td className="whitespace-nowrap px-2 py-2 font-mono text-slate-700">
                          {formatSeconds(intervalDuration(interval))}
                        </td>
                        <td className="whitespace-nowrap px-2 py-2 font-mono text-slate-700">
                          {intervalFrameCount(interval) ?? "—"}
                        </td>
                        <td className="px-2 py-2 font-semibold text-slate-700">
                          {primaryEvidence}
                        </td>
                        <td className="px-2 py-2 font-semibold text-slate-700">
                          {stage175
                            ? compactEyeEvidenceLabel(
                                evidence.strength,
                                evidence.level,
                                evidence.label,
                              )
                            : "—"}
                        </td>
                        <td className="px-2 py-2">
                          <DetailsButton
                            expanded={expanded}
                            onClick={() => toggle(interval.id)}
                          />
                        </td>
                      </tr>
                      {expanded ? (
                        <tr>
                          <td colSpan={8} className="px-2 pb-3">
                            <IntervalDetails interval={interval} />
                          </td>
                        </tr>
                      ) : null}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
