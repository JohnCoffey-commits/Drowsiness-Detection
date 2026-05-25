import { Info } from "lucide-react";
import { HISTORY_48H_BOUNDARY_NOTICE } from "@/lib/history48hUtils";

export function HistoryInterpretationNote() {
  return (
    <section className="rounded-2xl border border-blue-100 bg-blue-50/70 p-4 text-blue-950 shadow-sm">
      <div className="flex items-start gap-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-700" />
        <div className="text-sm leading-6">
          <p>
            {HISTORY_48H_BOUNDARY_NOTICE} Alerts are intended for awareness and
            are not a medical diagnosis or a guarantee of driving safety.
          </p>
          <p className="mt-2 text-blue-900">
            History records are lightweight summaries only. Raw webcam frames,
            uploaded videos, blobs, and base64 payloads are not stored.
          </p>
        </div>
      </div>
    </section>
  );
}
