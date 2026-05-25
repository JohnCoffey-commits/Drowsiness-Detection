import { Info } from "lucide-react";
import { INSIGHTS_BOUNDARY_NOTICE } from "@/lib/insightsUtils";

export function AboutInsightsNote() {
  return (
    <section className="rounded-2xl border border-blue-100 bg-blue-50/70 p-4 text-blue-950 shadow-sm dark:border-cyan-400/20 dark:bg-cyan-400/10 dark:text-cyan-100">
      <div className="flex items-start gap-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-700 dark:text-cyan-200" />
        <div className="text-sm leading-6">
          <h2 className="font-black">About these insights</h2>
          <p className="mt-1">
            {INSIGHTS_BOUNDARY_NOTICE} They are intended for awareness only and
            are not a medical diagnosis, a measure of final system-level
            accuracy, or a guarantee of driving safety.
          </p>
          <p className="mt-2">
            Raw webcam frames, uploaded videos, blobs, and base64 payloads are
            not stored or used in this view.
          </p>
        </div>
      </div>
    </section>
  );
}
