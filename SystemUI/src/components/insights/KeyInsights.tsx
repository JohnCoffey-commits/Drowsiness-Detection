import { Sparkles } from "lucide-react";

interface KeyInsightsProps {
  insights: string[];
}

export function KeyInsights({ insights }: KeyInsightsProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors duration-300 dark:border-slate-800 dark:bg-slate-900 sm:p-5">
      <div className="mb-4 flex items-start gap-3">
        <span className="rounded-xl bg-blue-50 p-2.5 text-blue-600 dark:bg-cyan-400/10 dark:text-cyan-300">
          <Sparkles className="h-5 w-5" />
        </span>
        <div>
          <h2 className="text-base font-black text-slate-950 dark:text-white">
            Key Insights
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-500 dark:text-slate-400">
            Plain-language patterns from the selected history window.
          </p>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {insights.map((insight, index) => (
          <article
            key={`${insight}-${index}`}
            className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold leading-6 text-slate-700 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300"
          >
            {insight}
          </article>
        ))}
      </div>
    </section>
  );
}
