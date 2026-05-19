"use client";

import {
  Database,
  Download,
  Clipboard,
  Filter,
  RefreshCcw,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import type { BackendArchiveRange } from "@/lib/backendArchiveTypes";
import type {
  EventTypeFilter,
  HistoryFilters,
  ReviewFilter,
  SourceFilter,
  TimeWindowHours,
} from "@/lib/history48hTypes";
import {
  EVENT_TYPE_OPTIONS,
  REVIEW_OPTIONS,
  SOURCE_OPTIONS,
  TIME_WINDOW_OPTIONS,
} from "@/lib/history48hUtils";

interface HistoryFiltersProps {
  filters: HistoryFilters;
  selectedSessionId?: string;
  copyStatus: string;
  archiveRange: BackendArchiveRange;
  archiveStatus: string;
  activeDataSource: "backend_archive" | "local_only";
  onChange: (filters: HistoryFilters) => void;
  onArchiveRangeChange: (range: BackendArchiveRange) => void;
  onRefreshArchive: () => void;
  onExportArchive: () => void;
  onResetDemoData: () => void;
  onClearHistory: () => void;
  onCopySummary: () => void;
  onClearSessionFilter: () => void;
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
  selectedSessionId,
  copyStatus,
  archiveRange,
  archiveStatus,
  activeDataSource,
  onChange,
  onArchiveRangeChange,
  onRefreshArchive,
  onExportArchive,
  onResetDemoData,
  onClearHistory,
  onCopySummary,
  onClearSessionFilter,
}: HistoryFiltersProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="grid gap-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-5">
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
            label="Event type"
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

          <SelectField
            label="Review"
            value={filters.review}
            onChange={(value) =>
              onChange({ ...filters, review: value as ReviewFilter })
            }
          >
            {REVIEW_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>

          <SelectField
            label="Source"
            value={filters.source}
            onChange={(value) =>
              onChange({ ...filters, source: value as SourceFilter })
            }
          >
            {SOURCE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>

          <SelectField
            label="Archive range"
            value={archiveRange}
            onChange={(value) => onArchiveRangeChange(value as BackendArchiveRange)}
          >
            <option value="48h">48h</option>
            <option value="7d">7d</option>
            <option value="30d">30d</option>
            <option value="all">All</option>
          </SelectField>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onRefreshArchive}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-100"
          >
            <Database className="h-4 w-4" />
            Refresh archive
          </button>
          <button
            type="button"
            onClick={onExportArchive}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-emerald-100 bg-emerald-50 px-3 text-sm font-semibold text-emerald-700 shadow-sm transition hover:bg-emerald-100 focus:outline-none focus:ring-4 focus:ring-emerald-100"
          >
            <Download className="h-4 w-4" />
            Export archive
          </button>
          <button
            type="button"
            onClick={onResetDemoData}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-blue-100 bg-blue-50 px-3 text-sm font-semibold text-blue-700 shadow-sm transition hover:bg-blue-100 focus:outline-none focus:ring-4 focus:ring-blue-100"
          >
            <RefreshCcw className="h-4 w-4" />
            Reset demo data
          </button>
          <button
            type="button"
            onClick={onClearHistory}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-100"
          >
            <Trash2 className="h-4 w-4" />
            Clear current user history
          </button>
          <button
            type="button"
            onClick={onCopySummary}
            className="inline-flex h-10 items-center gap-2 rounded-lg bg-blue-600 px-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-100"
          >
            <Clipboard className="h-4 w-4" />
            Copy history summary
          </button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 text-sm text-slate-600">
        <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 font-semibold text-slate-700">
          <Filter className="h-3.5 w-3.5" />
          Active source: {activeDataSource}
        </span>
        <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 font-semibold text-slate-700">
          <Database className="h-3.5 w-3.5" />
          {archiveStatus}
        </span>
        {selectedSessionId && (
          <span className="inline-flex items-center gap-2 rounded-full border border-blue-100 bg-blue-50 px-3 py-1.5 font-semibold text-blue-700">
            Session filter: {selectedSessionId}
            <button
              type="button"
              onClick={onClearSessionFilter}
              className="rounded-full p-0.5 text-blue-600 hover:bg-blue-100 focus:outline-none focus:ring-2 focus:ring-blue-300"
              aria-label="Clear session filter"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </span>
        )}
        {copyStatus && (
          <span className="inline-flex items-center gap-2 text-xs font-semibold text-emerald-700">
            <RotateCcw className="h-3.5 w-3.5" />
            {copyStatus}
          </span>
        )}
      </div>
    </section>
  );
}
