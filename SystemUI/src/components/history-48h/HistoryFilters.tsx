"use client";

import {
  Braces,
  CheckCircle2,
  Clipboard,
  Download,
  Table2,
} from "lucide-react";
import type {
  EventTypeFilter,
  HistoryFilters,
  TimeWindowHours,
} from "@/lib/history48hTypes";
import {
  EVENT_TYPE_OPTIONS,
  TIME_WINDOW_OPTIONS,
} from "@/lib/history48hUtils";

interface HistoryFiltersProps {
  filters: HistoryFilters;
  copyStatus: string;
  eventCount: number;
  onChange: (filters: HistoryFilters) => void;
  onDownloadSummary: () => void;
  onCopySummary: () => void;
  onDownloadCsv: () => void;
  onExportRawData: () => void;
}

function SelectField({
  label,
  value,
  children,
  onChange,
}: {
  label: string;
  value: string | number;
  children: React.ReactNode;
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex min-w-0 flex-col gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold normal-case tracking-normal text-slate-800 shadow-sm outline-none transition focus:border-blue-300 focus:ring-4 focus:ring-blue-100"
      >
        {children}
      </select>
    </label>
  );
}

export function HistoryFilters({
  filters,
  copyStatus,
  eventCount,
  onChange,
  onDownloadSummary,
  onCopySummary,
  onDownloadCsv,
  onExportRawData,
}: HistoryFiltersProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="grid gap-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <SelectField
            label="Time window"
            value={filters.timeWindowHours}
            onChange={(value) =>
              onChange({
                ...filters,
                timeWindowHours: Number(value) as TimeWindowHours,
              })
            }
          >
            {TIME_WINDOW_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>

          <SelectField
            label="Alert type"
            value={filters.eventType}
            onChange={(value) =>
              onChange({ ...filters, eventType: value as EventTypeFilter })
            }
          >
            {EVENT_TYPE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onDownloadSummary}
            disabled={eventCount === 0}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-emerald-100 bg-emerald-50 px-3 text-sm font-semibold text-emerald-700 shadow-sm transition hover:bg-emerald-100 focus:outline-none focus:ring-4 focus:ring-emerald-100 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400 disabled:shadow-none"
          >
            <Download className="h-4 w-4" />
            Download summary
          </button>
          <button
            type="button"
            onClick={onCopySummary}
            disabled={eventCount === 0}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400 disabled:shadow-none"
          >
            <Clipboard className="h-4 w-4" />
            Copy Summary
          </button>
          <button
            type="button"
            onClick={onDownloadCsv}
            disabled={eventCount === 0}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400 disabled:shadow-none"
          >
            <Table2 className="h-4 w-4" />
            Download table (.csv)
          </button>
          <button
            type="button"
            onClick={onExportRawData}
            disabled={eventCount === 0}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400 disabled:shadow-none"
          >
            <Braces className="h-4 w-4" />
            Export raw data (.json)
          </button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 text-sm text-slate-600">
        <span className="text-xs font-semibold text-slate-500">
          Only Live Monitor records are included.
        </span>
        {copyStatus && (
          <span className="inline-flex items-center gap-2 text-xs font-semibold text-emerald-700">
            <CheckCircle2 className="h-3.5 w-3.5" />
            {copyStatus}
          </span>
        )}
      </div>
    </section>
  );
}
