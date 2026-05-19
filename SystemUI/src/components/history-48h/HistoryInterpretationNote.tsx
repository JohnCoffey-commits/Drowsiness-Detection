import { Info, Signal, ShieldAlert, SlidersHorizontal } from "lucide-react";
import { HISTORY_48H_BOUNDARY_NOTICE } from "@/lib/history48hUtils";

export function HistoryInterpretationNote() {
  return (
    <section className="rounded-2xl border border-blue-100 bg-blue-50 p-5 text-blue-950 shadow-sm">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="max-w-3xl">
          <div className="flex items-center gap-2">
            <Info className="h-5 w-5 text-blue-700" />
            <h2 className="text-base font-bold">Interpretation note</h2>
          </div>
          <p className="mt-2 text-sm leading-6">
            This page summarizes recent warning-candidate history. Events are
            compact candidate states produced by stable Live Monitor visual
            alerts, uploaded-video summaries, backend_archive records, or local
            fallback history. {HISTORY_48H_BOUNDARY_NOTICE}
          </p>
        </div>
        <div className="grid gap-3 text-sm sm:grid-cols-3 lg:w-[560px]">
          <div className="rounded-xl border border-blue-100 bg-white/70 p-3">
            <ShieldAlert className="mb-2 h-4 w-4 text-red-600" />
            High-priority warning candidate is still a review cue, not final
            system output.
          </div>
          <div className="rounded-xl border border-blue-100 bg-white/70 p-3">
            <Signal className="mb-2 h-4 w-4 text-slate-600" />
            Signal quality issue means camera, face visibility, ROI, or signal
            uncertainty.
          </div>
          <div className="rounded-xl border border-blue-100 bg-white/70 p-3">
            <SlidersHorizontal className="mb-2 h-4 w-4 text-blue-700" />
            Candidate severity score is a UI-level display value only.
          </div>
        </div>
      </div>
      <p className="mt-4 text-sm leading-6 text-blue-900">
        Local history and backend archive records are lightweight summaries only.
        They do not save raw frames, webcam images, uploaded videos, blobs, or
        base64 payloads.
      </p>
    </section>
  );
}
