"use client";

import { useMemo, useState, type ReactNode } from "react";
import { ImageOff, Info, ShieldAlert } from "lucide-react";
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

function KeyframeImage({
  src,
  alt,
}: {
  src: string | null;
  alt: string;
}) {
  const [failed, setFailed] = useState(false);

  if (!src || failed) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-xl border border-slate-200 bg-slate-100 text-sm text-slate-500">
        <div className="flex flex-col items-center gap-2">
          <ImageOff className="h-7 w-7 text-slate-400" />
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
      className="aspect-video w-full rounded-xl border border-slate-200 bg-slate-100 object-contain"
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
    <div className="border-t border-slate-100 pt-3 first:border-t-0 first:pt-0">
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

function warningTypeLabel(value?: string | null): string {
  const trimmed = cleanText(value);
  return trimmed ? trimmed.replace(/_/g, " ") : "Not available";
}

function RecentYawnValue({ keyframe }: { keyframe: VideoUploadKeyframe }) {
  const explanation = recentYawnExplanation(keyframe);
  if (keyframe.recent_yawn_event === true && explanation) {
    return (
      <span className="inline-flex items-center justify-end gap-1">
        Yes - within recent temporal window
        <Info
          aria-label={explanation}
          className="h-3.5 w-3.5 text-blue-500"
        >
          <title>{explanation}</title>
        </Info>
      </span>
    );
  }
  return <>{formatOptionalBoolean(keyframe.recent_yawn_event)}</>;
}

function KeyframeCard({
  backendUrl,
  keyframe,
  galleryHasEyeEvidenceFields,
}: {
  backendUrl: string;
  keyframe: VideoUploadKeyframe;
  galleryHasEyeEvidenceFields: boolean;
}) {
  const stage175 = hasEyeEvidenceStrengthFields(keyframe);
  const safeUrl = safeKeyframeUrl(backendUrl, keyframe);
  const state = keyframe.fusion_state;
  const evidenceLabel = stage175
    ? eyeEvidenceLabel(
        keyframe.eye_evidence_level,
        keyframe.eye_evidence_strength,
        keyframe.eye_evidence_label,
      )
    : null;
  const evidenceDescription = stage175
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
    <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <KeyframeImage
        src={safeUrl}
        alt={`${fusionStateLabel(state)} keyframe at ${formatSeconds(
          keyframe.timestamp_sec,
        )}`}
      />

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span
          className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${stateTone(
            state,
          )}`}
        >
          {fusionStateLabel(state)}
        </span>
        {keyframe.is_primary ? (
          <span className="rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
            Primary evidence keyframe
          </span>
        ) : (
          <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-600">
            Supporting keyframe
          </span>
        )}
      </div>

      <div className="mt-3 space-y-3 text-xs leading-relaxed">
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
        </MetadataSection>

        <MetadataSection title="Temporal evidence">
          <MetadataRow label="Recent yawn" value={<RecentYawnValue keyframe={keyframe} />} />
          {recentExplanation ? (
            <p className="mt-2 rounded-lg border border-blue-100 bg-blue-50 px-2.5 py-2 text-xs text-blue-800">
              {recentExplanation}
            </p>
          ) : null}
          <OptionalBooleanRow
            label="Sustained eye-warning"
            value={keyframe.sustained_eye_warning}
          />
          <MetadataRow
            label="Warning type"
            value={warningTypeLabel(keyframe.warning_type)}
          />
        </MetadataSection>

        <MetadataSection title="Review">
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
              Eye evidence strength unavailable
            </p>
          ) : null}
          <OptionalBooleanRow
            label="Manual review"
            value={keyframe.manual_review_recommended}
          />
          {stage175 ? (
            <>
              <OptionalBooleanRow
                label="Strong eye-closure candidate"
                value={keyframe.is_strong_eye_closure_candidate}
              />
              <OptionalBooleanRow
                label="Reduced eye openness"
                value={keyframe.is_reduced_eye_openness_candidate}
              />
              <OptionalBooleanRow
                label="Blink-like candidate"
                value={keyframe.is_blink_like_candidate}
              />
            </>
          ) : null}
          {keyframe.eye_strength_gate_passed != null || gateReason ? (
            <>
              <OptionalBooleanRow
                label="Interval eye-strength gate"
                value={keyframe.eye_strength_gate_passed}
              />
              <p className="mt-2 text-xs text-slate-600">
                Evaluated over the eye-warning interval; this keyframe may
                still be weak evidence.
              </p>
            </>
          ) : null}
          {keyframe.high_confidence_suppressed_by_brief_eye_warning ? (
            <div className="mt-2 font-semibold text-orange-700">
              High-confidence escalation suppressed: brief eye-warning interval
            </div>
          ) : null}
          {stage175 && keyframe.high_confidence_suppressed_by_weak_eye_evidence ? (
            <div className="mt-2 font-semibold text-orange-700">
              High-confidence escalation suppressed: weak/reduced-eye-openness
              evidence only
            </div>
          ) : null}
          {reason ? <div className="mt-2 text-slate-700">Reason: {reason}</div> : null}
          {eyeStrengthReason ? (
            <div className="mt-2 text-slate-700">
              Eye evidence reason: {eyeStrengthReason}
            </div>
          ) : null}
          {gateReason ? (
            <div className="mt-2 text-slate-700">
              Interval eye-strength gate reason: {gateReason}
            </div>
          ) : null}
        </MetadataSection>
      </div>
    </article>
  );
}

export function KeyframeEvidenceGallery({
  backendUrl,
  keyframes,
}: KeyframeEvidenceGalleryProps) {
  const hasAnyEyeEvidenceFields = useMemo(
    () => hasAnyEyeEvidenceStrengthFields(keyframes),
    [keyframes],
  );
  const groups = useMemo(() => {
    const grouped = new Map<string, VideoUploadKeyframe[]>();
    keyframes.forEach((keyframe) => {
      const label = keyframeGroupLabel(keyframe);
      grouped.set(label, [...(grouped.get(label) || []), keyframe]);
    });
    return Array.from(grouped.entries());
  }, [keyframes]);

  return (
    <section className="space-y-4" aria-labelledby="keyframes-title">
      <div>
        <h2 id="keyframes-title" className="text-xl font-bold text-slate-950">
          Keyframe Evidence Gallery
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Keyframes are grouped by backend warning type/state and shown with
          model probabilities, gate metadata, and manual-review wording.
        </p>
        {keyframes.length > 0 && !hasAnyEyeEvidenceFields ? (
          <div className="mt-3 rounded-xl border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-600">
            This backend run does not include Stage 17.5 eye evidence strength
            fields. Keyframes are shown with available Stage 17.4
            warning-candidate metadata.
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
        <div className="space-y-5">
          {groups.map(([label, items]) => (
            <div key={label} className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-base font-bold text-slate-900">{label}</h3>
                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">
                  {items.length} keyframes
                </span>
              </div>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 2xl:grid-cols-3">
                {items.map((keyframe) => (
                  <KeyframeCard
                    key={`${keyframe.warning_type}-${keyframe.frame_index}-${keyframe.timestamp_sec}`}
                    backendUrl={backendUrl}
                    keyframe={keyframe}
                    galleryHasEyeEvidenceFields={hasAnyEyeEvidenceFields}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
