import { ListChecks } from "lucide-react";
import type { InsightRecommendation } from "@/lib/insightsTypes";

interface AttentionAreasProps {
  areas: InsightRecommendation[];
}

const PRIORITY_STYLES: Record<InsightRecommendation["priority"], string> = {
  high: "bg-red-50 text-red-700 dark:bg-red-400/10 dark:text-red-200",
  medium:
    "bg-amber-50 text-amber-700 dark:bg-amber-400/10 dark:text-amber-200",
  low: "bg-blue-50 text-blue-700 dark:bg-cyan-400/10 dark:text-cyan-200",
};

export function AttentionAreas({ areas }: AttentionAreasProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-4">
      <div className="mb-3 flex items-start gap-3">
        <span className="rounded-lg bg-blue-50 p-2 text-blue-600 dark:bg-cyan-400/10 dark:text-cyan-300">
          <ListChecks className="h-4 w-4" />
        </span>
        <div>
          <h2 className="text-base font-bold text-slate-950 dark:text-white">
            Attention Areas
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
            What recent alert patterns suggest you may want to check.
          </p>
        </div>
      </div>

      <div className="space-y-2">
        {areas.map((recommendation) => (
          <article
            key={recommendation.id}
            className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-800 dark:bg-slate-950"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={`rounded-full px-2.5 py-0.5 text-xs font-semibold capitalize ${PRIORITY_STYLES[recommendation.priority]}`}
              >
                {recommendation.priority}
              </span>
              <h3 className="text-sm font-bold leading-5 text-slate-900 dark:text-white">
                {recommendation.title}
              </h3>
            </div>
            <p className="mt-1 text-sm font-medium leading-5 text-slate-600 dark:text-slate-300">
              {recommendation.body}
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}
