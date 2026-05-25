"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { getPageCount } from "@/lib/history48hUtils";

interface PaginationControlsProps {
  page: number;
  pageSize: number;
  totalItems: number;
  label: string;
  onPageChange: (page: number) => void;
}

export function PaginationControls({
  page,
  pageSize,
  totalItems,
  label,
  onPageChange,
}: PaginationControlsProps) {
  const pageCount = getPageCount(totalItems, pageSize);
  const start = totalItems === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = Math.min(totalItems, page * pageSize);
  const isFirst = page <= 1;
  const isLast = page >= pageCount;

  if (totalItems <= pageSize) {
    return (
      <div className="mt-4 border-t border-slate-100 pt-4 text-sm font-semibold text-slate-500">
        {totalItems} {label} shown
      </div>
    );
  }

  return (
    <div className="mt-4 flex flex-col gap-3 border-t border-slate-100 pt-4 text-sm text-slate-600 sm:flex-row sm:items-center sm:justify-between">
      <div className="font-semibold">
        Showing {start}-{end} of {totalItems} {label}
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onPageChange(page - 1)}
          disabled={isFirst}
          className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-700 transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <ChevronLeft className="h-4 w-4" />
          Previous
        </button>
        <button
          type="button"
          onClick={() => onPageChange(page + 1)}
          disabled={isLast}
          className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-700 transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Next
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
