import type {
  DriverHistoryEvent,
  HistoryFilters,
} from "@/lib/history48hTypes";
import type {
  HistorySummary,
  SessionSummaryRow,
} from "@/lib/history48hUtils";
import {
  EVENT_TYPE_OPTIONS,
  HISTORY_48H_BOUNDARY_NOTICE,
  SEVERITY_META,
  STATE_META,
  TIME_WINDOW_OPTIONS,
  evidenceLabel,
  formatDateTime,
  formatDuration,
  formatMinutes,
  formatTimeRange,
} from "@/lib/history48hUtils";

export interface HistoryExportPayload {
  exportedAt: string;
  filters: HistoryFilters;
  summary: HistorySummary;
  events: DriverHistoryEvent[];
  sessions: SessionSummaryRow[];
}

export function downloadTextFile(
  filename: string,
  content: string,
  type: string
): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function buildHistorySummaryHtml(payload: HistoryExportPayload): string {
  const { exportedAt, filters, summary, events, sessions } = payload;
  const sessionMap = sessionMapById(sessions);
  const selectedWindow = timeWindowLabel(filters);
  const alertType = alertTypeLabel(filters);
  const latestAlert = summary.lastEventTime
    ? formatDateTime(summary.lastEventTime)
    : "-";

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VisionGuard History Summary</title>
  <style>
    :root {
      color: #0f172a;
      background: #f8fafc;
      font-family: Arial, Helvetica, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f8fafc; color: #0f172a; }
    .report { max-width: 1120px; margin: 0 auto; padding: 32px 24px 48px; }
    .header {
      border: 1px solid #dbe3ef;
      border-radius: 14px;
      background: #ffffff;
      padding: 24px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    h1 { margin: 0; font-size: 28px; line-height: 1.2; }
    h2 { margin: 28px 0 12px; font-size: 18px; }
    p { margin: 0; line-height: 1.6; }
    .meta { margin-top: 14px; display: grid; gap: 6px; color: #475569; font-size: 14px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }
    .card { border: 1px solid #dbe3ef; border-radius: 12px; background: #ffffff; padding: 16px; }
    .label { color: #64748b; font-size: 12px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
    .value { margin-top: 8px; font-size: 26px; font-weight: 800; }
    table { width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #dbe3ef; border-radius: 12px; overflow: hidden; }
    th, td { border-bottom: 1px solid #e2e8f0; padding: 10px 12px; text-align: left; vertical-align: top; font-size: 13px; }
    th { background: #f1f5f9; color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
    tr:last-child td { border-bottom: 0; }
    .badge { display: inline-block; border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 700; border: 1px solid #cbd5e1; }
    .severity-low { color: #047857; background: #ecfdf5; border-color: #a7f3d0; }
    .severity-medium { color: #92400e; background: #fffbeb; border-color: #fde68a; }
    .severity-high { color: #b91c1c; background: #fef2f2; border-color: #fecaca; }
    .severity-unreliable { color: #475569; background: #f1f5f9; border-color: #cbd5e1; }
    .note { margin-top: 24px; border: 1px solid #bfdbfe; border-radius: 12px; background: #eff6ff; padding: 16px; color: #1e3a8a; font-size: 14px; }
    @media (max-width: 760px) {
      .report { padding: 20px 12px 32px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      table { display: block; overflow-x: auto; }
    }
    @media print {
      body { background: #ffffff; }
      .report { max-width: none; padding: 0; }
      .header, .card, table, .note { box-shadow: none; }
      .cards { break-inside: avoid; }
      tr, .card { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <main class="report">
    <section class="header">
      <h1>VisionGuard History Summary</h1>
      <div class="meta">
        <p><strong>Generated:</strong> ${escapeHtml(formatGeneratedAt(exportedAt))}</p>
        <p><strong>Time window:</strong> ${escapeHtml(selectedWindow)}</p>
        <p><strong>Alert type:</strong> ${escapeHtml(alertType)}</p>
        <p><strong>Source:</strong> Live Monitor</p>
      </div>
    </section>

    <section class="cards" aria-label="Overview metrics">
      ${metricCard("Total Alerts", summary.warningCandidateCount)}
      ${metricCard("High-Risk Alerts", summary.highPriorityCount)}
      ${metricCard("Signal Interruptions", summary.signalUnreliableCount)}
      ${metricCard("Latest Alert", latestAlert)}
    </section>

    <section>
      <h2>Recent Drives</h2>
      ${drivesTable(sessions)}
    </section>

    <section>
      <h2>Alert Timeline</h2>
      ${alertsTable(events, sessionMap)}
    </section>

    <section class="note">
      <p>${escapeHtml(HISTORY_48H_BOUNDARY_NOTICE)} Alerts are intended for awareness and are not a medical diagnosis or a guarantee of driving safety.</p>
      <p style="margin-top: 8px;">History records are lightweight summaries only. Raw webcam frames, uploaded videos, blobs, and base64 payloads are not stored or exported.</p>
    </section>
  </main>
</body>
</html>`;
}

export function buildHistoryCsv(payload: HistoryExportPayload): string {
  const sessionMap = sessionMapById(payload.sessions);
  const rows = [
    [
      "timestamp",
      "drive_start",
      "drive_end",
      "alert_type",
      "severity",
      "duration_seconds",
      "evidence_summary",
      "signal_status",
    ],
    ...payload.events.map((event) => {
      const session = sessionMap.get(event.sessionId);
      return [
        event.timestamp,
        session?.startedAt ?? "",
        session?.endedAt ?? "",
        STATE_META[event.state].label,
        SEVERITY_META[event.severity].label,
        String(Math.round(event.durationSec)),
        evidenceLabel(event),
        signalStatusLabel(event),
      ];
    }),
  ];

  return rows.map((row) => row.map(formatCsvValue).join(",")).join("\n");
}

export function buildRawHistoryJson(payload: HistoryExportPayload): string {
  return JSON.stringify(
    {
      ok: true,
      exported_at: payload.exportedAt,
      source: "live_monitor",
      filters: payload.filters,
      alert_count: payload.events.length,
      drive_count: payload.sessions.length,
      events: payload.events,
      sessions: payload.sessions,
    },
    null,
    2
  );
}

export function historyExportDate(): string {
  return new Date().toISOString().slice(0, 10);
}

export function historySummaryFilename(): string {
  const date = historyExportDate();
  return `visionguard-history-summary-${date}.html`;
}

export function historyCsvFilename(): string {
  return `visionguard-history-alerts-${historyExportDate()}.csv`;
}

export function historyRawJsonFilename(): string {
  return `visionguard-history-raw-data-${historyExportDate()}.json`;
}

export function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function formatCsvValue(value: unknown): string {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function metricCard(label: string, value: number | string): string {
  return `<article class="card"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div></article>`;
}

function drivesTable(sessions: SessionSummaryRow[]): string {
  if (sessions.length === 0) {
    return `<p>No drives match the selected scope.</p>`;
  }

  return `<table>
    <thead>
      <tr>
        <th>Drive time range</th>
        <th>Duration</th>
        <th>Alert count</th>
        <th>Highest severity</th>
        <th>Signal interruptions</th>
      </tr>
    </thead>
    <tbody>
      ${sessions
        .map(
          (session) => `<tr>
            <td>${escapeHtml(formatTimeRange(session.startedAt, session.endedAt))}</td>
            <td>${escapeHtml(formatMinutes(session.durationMin))}</td>
            <td>${escapeHtml(session.warningCandidateCount)}</td>
            <td>${severityBadge(session.highestSeverity)}</td>
            <td>${escapeHtml(session.signalUnreliableCount)}</td>
          </tr>`
        )
        .join("")}
    </tbody>
  </table>`;
}

function alertsTable(
  events: DriverHistoryEvent[],
  sessionMap: Map<string, SessionSummaryRow>
): string {
  if (events.length === 0) {
    return `<p>No alerts match the selected scope.</p>`;
  }

  return `<table>
    <thead>
      <tr>
        <th>Time</th>
        <th>Alert</th>
        <th>Severity</th>
        <th>Duration</th>
        <th>Evidence</th>
        <th>Signal</th>
      </tr>
    </thead>
    <tbody>
      ${events
        .map((event) => {
          const session = sessionMap.get(event.sessionId);
          const driveSuffix = session
            ? `<br><span style="color:#64748b;font-size:12px;">${escapeHtml(formatTimeRange(session.startedAt, session.endedAt))}</span>`
            : "";
          return `<tr>
            <td>${escapeHtml(formatDateTime(event.timestamp))}${driveSuffix}</td>
            <td>${escapeHtml(STATE_META[event.state].label)}</td>
            <td>${severityBadge(event.severity)}</td>
            <td>${escapeHtml(formatDuration(event.durationSec))}</td>
            <td>${escapeHtml(evidenceLabel(event))}</td>
            <td>${escapeHtml(signalStatusLabel(event))}</td>
          </tr>`;
        })
        .join("")}
    </tbody>
  </table>`;
}

function severityBadge(severity: DriverHistoryEvent["severity"]): string {
  return `<span class="badge severity-${escapeHtml(severity)}">${escapeHtml(SEVERITY_META[severity].label)}</span>`;
}

function signalStatusLabel(event: DriverHistoryEvent): string {
  if (event.state === "signal_unreliable") return "Interrupted";
  if (
    event.eyeEvidenceStrength === "weak" ||
    event.eyeEvidenceStrength === "unknown"
  ) {
    return "Limited";
  }
  return "Available";
}

function sessionMapById(sessions: SessionSummaryRow[]): Map<string, SessionSummaryRow> {
  return new Map(sessions.map((session) => [session.id, session]));
}

function timeWindowLabel(filters: HistoryFilters): string {
  return (
    TIME_WINDOW_OPTIONS.find((option) => option.value === filters.timeWindowHours)
      ?.label ?? `Last ${filters.timeWindowHours} hours`
  );
}

function alertTypeLabel(filters: HistoryFilters): string {
  return (
    EVENT_TYPE_OPTIONS.find((option) => option.value === filters.eventType)
      ?.label ?? "All"
  );
}

function formatGeneratedAt(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}
