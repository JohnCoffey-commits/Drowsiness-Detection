"use client";

import { useMemo, useState, type ReactNode } from "react";
import { ImageOff, Info, ShieldAlert, X } from "lucide-react";
import type { VideoUploadKeyframe } from "@/lib/videoUploadTypes";
import {
  eyeEvidenceDescription,
  eyeEvidenceLabel,
  formatOptionalBoolean,
  formatProbability,
  formatSeconds,
  fusionStateLabel,
  hasAnyEyeEvidenceStrengthFields,
  hasEyeEvidenceStrengthFields,
  keyframeGroupLabel,
  recentYawnExplanation,
  safeKeyframeUrl,
  stateTone,
} from "@/lib/videoUploadUtils";

interface KeyframeEvidenceGalleryProps {
  backendUrl: string;
  keyframes: VideoUploadKeyframe[];
}

function keyframeId(keyframe: VideoUploadKeyframe, index: number): string {
  return [
    keyframe.warning_type || keyframe.fusion_state || "keyframe",
    keyframe.frame_index ?? index,
    keyframe.timestamp_sec ?? index,
  ].join("-");
}

function KeyframeImage({
  src,
  alt,
  className = "",
}: {
  src: string | null;
  alt: string;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);

  if (!src || failed) {
    return (
      <div
        className={`flex aspect-video items-center justify-center rounded-xl border border-slate-200 bg-slate-100 text-sm text-slate-500 ${className}`}
      >
        <div className="flex flex-col items-center gap-2">
          <ImageOff className="h-6 w-6 text-slate-400" />
          Keyframe image unavailable
        </div>
      </div>
    );
  }

  return (
    // Dynamic backend evidence images are intentionally served directly.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={alt}
      className={`aspect-video w-full rounded-xl border border-slate-200 bg-slate-100 object-contain ${className}`}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

function MetadataRow({
  label,
  value,
  muted = false,
}: {
  label: string;
  value: ReactNode;
  muted?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-slate-100 py-1.5 last:border-0">
      <span className="text-xs font-semibold uppercase text-slate-400">
        {label}
      </span>
      <span
        className={`text-right text-xs font-semibold ${
          muted ? "text-slate-500" : "text-slate-800"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

function MetadataSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <h4 className="text-[11px] font-bold uppercase tracking-wide text-slate-500">
        {title}
      </h4>
      <div className="mt-1">{children}</div>
    </div>
  );
}

function OptionalBooleanRow({
  label,
  value,
}: {
  label: string;
  value: boolean | null | undefined;
}) {
  if (value == null) return null;
  return <MetadataRow label={label} value={formatOptionalBoolean(value)} />;
}

function cleanText(value?: string | null): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

function RecentYawnValue({ keyframe }: { keyframe: VideoUploadKeyframe }) {
  const explanation = recentYawnExplanation(keyframe);
  if (keyframe.recent_yawn_event === true && explanation) {
    return (
      <span className="inline-flex items-center justify-end gap-1">
        Yes - within recent temporal window
        <Info aria-label={explanation} className="h-3.5 w-3.5 text-blue-500">
          <title>{explanation}</title>
        </Info>
      </span>
    );
  }
  return <>{formatOptionalBoolean(keyframe.recent_yawn_event)}</>;
}

function KeyframeThumbnail({
  backendUrl,
  keyframe,
  selected,
  onSelect,
}: {
  backendUrl: string;
  keyframe: VideoUploadKeyframe;
  selected: boolean;
  onSelect: () => void;
}) {
  const safeUrl = safeKeyframeUrl(backendUrl, keyframe);
  const alertLabel = keyframeGroupLabel(keyframe);
  const timestamp = formatSeconds(keyframe.timestamp_sec);

  return (
    <button
      type="button"
      onClick={onSelect}
      className={`rounded-2xl border bg-white p-3 text-left shadow-sm outline-none transition hover:border-blue-200 hover:bg-blue-50/40 focus-visible:ring-2 focus-visible:ring-blue-400 ${
        selected ? "border-blue-300 ring-2 ring-blue-100" : "border-slate-200"
      }`}
      aria-pressed={selected}
    >
      <KeyframeImage
        src={safeUrl}
        alt={`${alertLabel} keyframe at ${timestamp}`}
        className="max-h-44"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-700">
          {timestamp}
        </span>
        <span
          className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${stateTone(
            keyframe.fusion_state,
          )}`}
        >
          {alertLabel}
        </span>
        <span
          className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${
            keyframe.is_primary
              ? "border-blue-200 bg-blue-50 text-blue-700"
              : "border-slate-200 bg-slate-50 text-slate-600"
          }`}
        >
          {keyframe.is_primary ? "Primary" : "Supporting"}
        </span>
      </div>
    </button>
  );
}

function KeyframeDetails({
  backendUrl,
  keyframe,
  galleryHasEyeEvidenceFields,
  onClose,
}: {
  backendUrl: string;
  keyframe: VideoUploadKeyframe;
  galleryHasEyeEvidenceFields: boolean;
  onClose: () => void;
}) {
  const safeUrl = safeKeyframeUrl(backendUrl, keyframe);
  const state = keyframe.fusion_state;
  const hasCalibration = hasEyeEvidenceStrengthFields(keyframe);
  const evidenceLabel = hasCalibration
    ? eyeEvidenceLabel(
        keyframe.eye_evidence_level,
        keyframe.eye_evidence_strength,
        keyframe.eye_evidence_label,
      )
    : null;
  const evidenceDescription = hasCalibration
    ? eyeEvidenceDescription(
        keyframe.eye_evidence_level,
        keyframe.eye_evidence_strength,
        keyframe.eye_evidence_interpretation,
      )
    : null;
  const recentExplanation = recentYawnExplanation(keyframe);
  const reason = cleanText(keyframe.reason);
  const eyeStrengthReason = cleanText(keyframe.eye_warning_strength_reason);
  const gateReason = cleanText(keyframe.eye_strength_gate_reason);

  return (
    <article className="rounded-2xl border border-blue-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h3 className="text-lg font-bold text-slate-950">
            Keyframe details
          </h3>
          <p className="mt-1 text-sm text-slate-600">
            Backend-generated evidence image and available model evidence for
            the selected keyframe.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="inline-flex w-fit items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 outline-none transition hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400"
        >
          <X className="h-3.5 w-3.5" />
          Close
        </button>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
        <KeyframeImage
          src={safeUrl}
          alt={`${fusionStateLabel(state)} keyframe at ${formatSeconds(
            keyframe.timestamp_sec,
          )}`}
        />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-1">
          <MetadataSection title="Frame">
            <MetadataRow
              label="Timestamp"
              value={formatSeconds(keyframe.timestamp_sec)}
            />
            <MetadataRow
              label="Frame index"
              value={keyframe.frame_index ?? "Not available"}
            />
            <MetadataRow
              label="Segment"
              value={keyframe.segment_id ?? "Not available"}
            />
            <MetadataRow label="Alert" value={keyframeGroupLabel(keyframe)} />
          </MetadataSection>

          <MetadataSection title="Model evidence">
            <MetadataRow
              label="p_eye_closed"
              value={formatProbability(keyframe.p_eye_closed)}
            />
            <MetadataRow
              label="p_yawn"
              value={formatProbability(keyframe.p_yawn)}
            />
            {reason ? <MetadataRow label="Reason" value={reason} /> : null}
          </MetadataSection>

          <MetadataSection title="Temporal evidence">
            <MetadataRow
              label="Recent yawn"
              value={<RecentYawnValue keyframe={keyframe} />}
            />
            {recentExplanation ? (
              <p className="mt-2 rounded-lg border border-blue-100 bg-blue-50 px-2.5 py-2 text-xs text-blue-800">
                {recentExplanation}
              </p>
            ) : null}
            <OptionalBooleanRow
              label="Sustained eye alert"
              value={keyframe.sustained_eye_warning}
            />
          </MetadataSection>

          <MetadataSection title="Eye evidence calibration">
            {evidenceLabel ? (
              <>
                <MetadataRow label="Eye evidence" value={evidenceLabel} />
                {evidenceDescription ? (
                  <p className="mt-2 text-xs text-slate-600">
                    {evidenceDescription}
                  </p>
                ) : null}
              </>
            ) : galleryHasEyeEvidenceFields ? (
              <p className="text-xs text-slate-500">
                Eye evidence strength unavailable for this keyframe.
              </p>
            ) : (
              <p className="text-xs text-slate-500">
                Eye evidence calibration fields were not returned for this run.
              </p>
            )}
            <OptionalBooleanRow
              label="Careful interpretation flag"
              value={keyframe.manual_review_recommended}
            />
            <OptionalBooleanRow
              label="Strong eye-closure cue"
              value={keyframe.is_strong_eye_closure_candidate}
            />
            <OptionalBooleanRow
              label="Reduced eye openness"
              value={keyframe.is_reduced_eye_openness_candidate}
            />
            <OptionalBooleanRow
              label="Blink-like cue"
              value={keyframe.is_blink_like_candidate}
            />
            <OptionalBooleanRow
              label="Interval gate passed"
              value={keyframe.eye_strength_gate_passed}
            />
            {keyframe.high_confidence_suppressed_by_brief_eye_warning ? (
              <div className="mt-2 font-semibold text-orange-700">
                High-confidence escalation suppressed because the eye interval
                was brief.
              </div>
            ) : null}
            {keyframe.high_confidence_suppressed_by_weak_eye_evidence ? (
              <div className="mt-2 font-semibold text-orange-700">
                High-confidence escalation suppressed because calibrated eye
                evidence remained weak.
              </div>
            ) : null}
            {eyeStrengthReason ? (
              <div className="mt-2 text-xs text-slate-700">
                Eye evidence reason: {eyeStrengthReason}
              </div>
            ) : null}
            {gateReason ? (
              <div className="mt-2 text-xs text-slate-700">
                Interval gate reason: {gateReason}
              </div>
            ) : null}
          </MetadataSection>
        </div>
      </div>
    </article>
  );
}

export function KeyframeEvidenceGallery({
  backendUrl,
  keyframes,
}: KeyframeEvidenceGalleryProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const hasAnyEyeEvidenceFields = useMemo(
    () => hasAnyEyeEvidenceStrengthFields(keyframes),
    [keyframes],
  );
  const keyedKeyframes = useMemo(
    () =>
      keyframes.map((keyframe, index) => ({
        id: keyframeId(keyframe, index),
        keyframe,
      })),
    [keyframes],
  );
  const selectedKeyframe =
    keyedKeyframes.find((item) => item.id === selectedId)?.keyframe || null;

  return (
    <section className="space-y-4" aria-labelledby="keyframes-title">
      <div>
        <h2 id="keyframes-title" className="text-xl font-bold text-slate-950">
          Keyframe Evidence Gallery
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Backend-selected keyframes are shown as compact thumbnails. Select a
          keyframe to inspect probabilities and calibration details.
        </p>
        {keyframes.length > 0 && !hasAnyEyeEvidenceFields ? (
          <div className="mt-3 rounded-xl border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-600">
            This backend run does not include eye evidence calibration fields.
            Keyframes are shown with the available alert metadata.
          </div>
        ) : null}
      </div>

      {keyframes.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-5 text-sm text-slate-600">
          <div className="flex items-start gap-3">
            <ShieldAlert className="mt-0.5 h-5 w-5 text-slate-400" />
            No keyframes were returned for this run.
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {keyedKeyframes.map(({ id, keyframe }) => (
              <KeyframeThumbnail
                key={id}
                backendUrl={backendUrl}
                keyframe={keyframe}
                selected={selectedId === id}
                onSelect={() =>
                  setSelectedId((current) => (current === id ? null : id))
                }
              />
            ))}
          </div>

          {selectedKeyframe ? (
            <KeyframeDetails
              backendUrl={backendUrl}
              keyframe={selectedKeyframe}
              galleryHasEyeEvidenceFields={hasAnyEyeEvidenceFields}
              onClose={() => setSelectedId(null)}
            />
          ) : null}
        </div>
      )}
    </section>
  );
}
