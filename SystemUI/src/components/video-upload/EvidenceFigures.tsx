"use client";

import { useState } from "react";
import { ExternalLink, ImageOff } from "lucide-react";

interface EvidenceFigure {
  id: string;
  title: string;
  description: string;
  url: string;
}

interface EvidenceFiguresProps {
  figures: EvidenceFigure[];
}

function FigureImage({ figure }: { figure: EvidenceFigure }) {
  const [failed, setFailed] = useState(false);

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
      {failed ? (
        <div className="flex min-h-[260px] flex-col items-center justify-center gap-2 p-6 text-center text-sm text-slate-500">
          <ImageOff className="h-8 w-8 text-slate-400" />
          Figure could not be loaded from the backend evidence route.
        </div>
      ) : (
        // Dynamic backend evidence images are intentionally served directly.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={figure.url}
          alt={`${figure.title} evidence figure`}
          className="max-h-[520px] w-full object-contain"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}

export function EvidenceFigures({ figures }: EvidenceFiguresProps) {
  const [activeId, setActiveId] = useState("fusion");
  const activeFigure =
    figures.find((figure) => figure.id === activeId) || figures[0] || null;

  return (
    <section className="space-y-4" aria-labelledby="figures-title">
      <div>
        <h2 id="figures-title" className="text-xl font-bold text-slate-950">
          Evidence Figures
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Fusion timeline is shown by default. Specialist probability figures
          remain available through tabs without expanding every figure at once.
        </p>
      </div>

      {figures.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-5 text-sm text-slate-600">
          No figure URLs are available yet.
        </div>
      ) : (
        <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div
            className="flex flex-wrap gap-2"
            role="tablist"
            aria-label="Evidence figure tabs"
          >
            {figures.map((figure) => (
              <button
                key={figure.id}
                type="button"
                role="tab"
                aria-selected={activeFigure?.id === figure.id}
                onClick={() => setActiveId(figure.id)}
                className={`rounded-lg border px-3 py-2 text-xs font-semibold outline-none transition focus-visible:ring-2 focus-visible:ring-blue-400 ${
                  activeFigure?.id === figure.id
                    ? "border-blue-200 bg-blue-50 text-blue-700"
                    : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                {figure.title}
              </button>
            ))}
          </div>

          {activeFigure ? (
            <div className="mt-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-950">
                    {activeFigure.title}
                  </h3>
                  <p className="mt-1 text-sm leading-relaxed text-slate-600">
                    {activeFigure.description}
                  </p>
                </div>
                <a
                  href={activeFigure.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-700 outline-none transition hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-blue-400"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Open
                </a>
              </div>
              <div className="mt-4">
                <FigureImage key={activeFigure.id} figure={activeFigure} />
              </div>
            </div>
          ) : null}
        </article>
      )}
    </section>
  );
}
