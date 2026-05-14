"use client";

import { useState } from "react";
import { ChevronDown, Info } from "lucide-react";
import { PERMANENT_WARNING } from "@/lib/videoUploadTypes";

const details = [
  "This page is for uploaded-video analysis, not webcam or real-time monitoring.",
  "The output is a rule-based warning-candidate analysis, not final system-level accuracy.",
  "p_eye_closed and p_yawn are specialist model outputs that support technical review.",
  "eye_warning_candidate is not verified sustained eye closure.",
  "high-confidence warning candidate is not final drowsiness truth.",
  "signal_unreliable is a quality condition, not drowsiness evidence.",
  "Stage 17.1 suppresses brief blink-like escalation before high-confidence warning candidates are retained.",
  "Stage 17.5 optional fields may distinguish weak, moderate, and strong eye evidence when available.",
];

export function InterpretationNotice() {
  const [open, setOpen] = useState(false);

  return (
    <section
      className="rounded-2xl border border-blue-200 bg-blue-50/70 p-5 shadow-sm"
      aria-labelledby="interpretation-title"
    >
      <div className="flex items-start gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white">
          <Info className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <h2 id="interpretation-title" className="text-lg font-bold text-slate-950">
            Interpretation / Manual Review Notes
          </h2>
          <p className="mt-1 text-sm leading-relaxed text-slate-700">
            {PERMANENT_WARNING}
          </p>
          <p className="mt-2 text-sm leading-relaxed text-slate-600">
            Treat each state as a manual-review cue. The UI formats backend
            evidence and safe links; it does not recompute fusion decisions.
          </p>
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-blue-200 bg-white px-3 py-2 text-xs font-semibold text-blue-700 outline-none transition hover:bg-blue-50 focus-visible:ring-2 focus-visible:ring-blue-400"
            aria-expanded={open}
          >
            <ChevronDown
              className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
            />
            {open ? "Hide details" : "Show details"}
          </button>

          {open ? (
            <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-2">
              {details.map((item) => (
                <div
                  key={item}
                  className="rounded-xl border border-blue-100 bg-white p-3 text-sm leading-relaxed text-slate-700"
                >
                  {item}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
