"use client";

import { useState } from "react";
import { ChevronDown, ExternalLink, FileText } from "lucide-react";
import type { VideoUploadResponse } from "@/lib/videoUploadTypes";
import { buildApiUrl, safeBackendLink, sessionFilePath } from "@/lib/videoUploadUtils";

interface TechnicalEvidencePanelProps {
  backendUrl: string;
  response: VideoUploadResponse;
}

interface EvidenceLink {
  label: string;
  href: string;
  description: string;
}

export function TechnicalEvidencePanel({
  backendUrl,
  response,
}: TechnicalEvidencePanelProps) {
  const [open, setOpen] = useState(false);
  const sessionId = response.session_id;
  const links: EvidenceLink[] = [
    {
      label: "Summary JSON",
      href: buildApiUrl(backendUrl, `/api/runs/${sessionId}/summary`),
      description: "Parsed backend summary for this uploaded-video run.",
    },
    {
      label: "Timeline CSV",
      href: safeBackendLink(backendUrl, response.timeline_url) ||
        buildApiUrl(backendUrl, `/api/runs/${sessionId}/timeline`),
      description: "Frame-level sampled analysis timeline.",
    },
    {
      label: "Fusion timeline CSV",
      href: buildApiUrl(backendUrl, sessionFilePath(sessionId, "fusion_timeline.csv")),
      description: "Rule-based fusion timeline evidence.",
    },
    {
      label: "Report Markdown",
      href: safeBackendLink(backendUrl, response.report_url) ||
        buildApiUrl(
          backendUrl,
          sessionFilePath(sessionId, "SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md"),
        ),
      description: "Backend-generated evidence report.",
    },
    {
      label: "Keyframe metadata CSV",
      href: buildApiUrl(
        backendUrl,
        sessionFilePath(sessionId, "keyframes/keyframes_metadata.csv"),
      ),
      description: "Keyframe table with warning-candidate metadata.",
    },
    {
      label: "Keyframe metadata JSON",
      href: buildApiUrl(
        backendUrl,
        sessionFilePath(sessionId, "keyframes/keyframes_metadata.json"),
      ),
      description: "Keyframe metadata in JSON format.",
    },
    {
      label: "Fusion timeline figure",
      href: safeBackendLink(backendUrl, response.fusion_figure_url) ||
        buildApiUrl(backendUrl, sessionFilePath(sessionId, "figures/fusion_timeline.png")),
      description: "Readable fusion timeline image.",
    },
    {
      label: "p_eye_closed figure",
      href: buildApiUrl(
        backendUrl,
        sessionFilePath(sessionId, "figures/p_eye_closed_over_time.png"),
      ),
      description: "Eye specialist probability figure.",
    },
    {
      label: "p_yawn figure",
      href: buildApiUrl(
        backendUrl,
        sessionFilePath(sessionId, "figures/p_yawn_over_time.png"),
      ),
      description: "Mouth/yawn specialist probability figure.",
    },
  ];

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-4 p-5 text-left outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
        aria-expanded={open}
      >
        <span className="flex items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-100 text-slate-600">
            <FileText className="h-4 w-4" />
          </span>
          <span>
            <span className="block text-lg font-bold text-slate-950">
              Technical Evidence Panel
            </span>
            <span className="mt-1 block text-sm text-slate-600">
              Safe backend links only. Local server-side log paths are not shown
              in the browser.
            </span>
          </span>
        </span>
        <ChevronDown
          className={`h-5 w-5 shrink-0 text-slate-500 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open ? (
        <div className="border-t border-slate-100 p-5">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {links.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target="_blank"
                rel="noreferrer"
                className="rounded-xl border border-slate-200 bg-slate-50 p-3 outline-none transition hover:border-blue-200 hover:bg-blue-50 focus-visible:ring-2 focus-visible:ring-blue-400"
              >
                <span className="flex items-center justify-between gap-3">
                  <span className="text-sm font-bold text-slate-900">
                    {link.label}
                  </span>
                  <ExternalLink className="h-3.5 w-3.5 text-slate-500" />
                </span>
                <span className="mt-1 block text-xs leading-relaxed text-slate-600">
                  {link.description}
                </span>
              </a>
            ))}
          </div>
          {response.audit_log ? (
            <p className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
              Backend pipeline logs are local server-side evidence and are not
              browser-accessible links.
            </p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
