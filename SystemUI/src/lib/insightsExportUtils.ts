import type {
  InsightCompositionItem,
  InsightRecommendation,
  InsightSessionComparisonRow,
  InsightSignalQualitySummary,
  InsightSummary,
  InsightTimeOfDayItem,
} from "@/lib/insightsTypes";
import { formatInsightPercent } from "@/lib/insightsUtils";

export interface InsightsReportPayload {
  exportedAt: string;
  timeWindowLabel: string;
  userLabel: string;
  dataSourceLabel: string;
  dataBasisLabel: string;
  keyInsights: string[];
  summary: InsightSummary;
  driveRows: InsightSessionComparisonRow[];
  composition: InsightCompositionItem[];
  timeOfDay: InsightTimeOfDayItem[];
  signalQuality: InsightSignalQualitySummary;
  attentionAreas: InsightRecommendation[];
}

const ALERT_SEGMENTS = [
  {
    key: "criticalEyeCount",
    label: "High-risk eye",
    color: "#ef4444",
  },
  {
    key: "eyeClosureCount",
    label: "Eye closure",
    color: "#f97316",
  },
  {
    key: "yawnCount",
    label: "Yawn",
    color: "#ec4899",
  },
  {
    key: "signalInterruptionCount",
    label: "Signal",
    color: "#64748b",
  },
] as const;

export function downloadTextFile(
  filename: string,
  content: string,
  mimeType: string
): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function insightsReportFilename(): string {
  return `visionguard-insights-report-${reportDate()}.html`;
}

export function buildInsightsReportHtml(payload: InsightsReportPayload): string {
  const {
    exportedAt,
    timeWindowLabel,
    userLabel,
    dataSourceLabel,
    dataBasisLabel,
    keyInsights,
    summary,
    driveRows,
    composition,
    timeOfDay,
    signalQuality,
    attentionAreas,
  } = payload;

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VisionGuard Insights Report</title>
  <style>
    :root { color: #0f172a; background: #f8fafc; font-family: Arial, Helvetica, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f8fafc; color: #0f172a; }
    .report { max-width: 1120px; margin: 0 auto; padding: 32px 24px 48px; }
    .header, .card, .section, .note { border: 1px solid #dbe3ef; background: #ffffff; border-radius: 14px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06); }
    .header { padding: 24px; }
    .section { padding: 20px; margin-top: 18px; }
    h1 { margin: 0; font-size: 28px; line-height: 1.2; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    h3 { margin: 0 0 6px; font-size: 14px; }
    p { margin: 0; line-height: 1.6; }
    .meta { margin-top: 14px; display: grid; gap: 6px; color: #475569; font-size: 14px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }
    .card { padding: 16px; }
    .label { color: #64748b; font-size: 12px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
    .value { margin-top: 8px; font-size: 24px; font-weight: 800; line-height: 1.15; }
    ul { margin: 0; padding-left: 20px; }
    li { margin: 7px 0; line-height: 1.55; }
    .chart { width: 100%; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #dbe3ef; border-radius: 12px; overflow: hidden; }
    th, td { border-bottom: 1px solid #e2e8f0; padding: 10px 12px; text-align: left; vertical-align: top; font-size: 13px; }
    th { background: #f1f5f9; color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
    tr:last-child td { border-bottom: 0; }
    .priority { display: inline-block; border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 700; text-transform: capitalize; }
    .priority-high { color: #b91c1c; background: #fef2f2; }
    .priority-medium { color: #92400e; background: #fffbeb; }
    .priority-low { color: #1d4ed8; background: #eff6ff; }
    .note { margin-top: 18px; padding: 16px; color: #1e3a8a; background: #eff6ff; border-color: #bfdbfe; font-size: 14px; }
    @media (max-width: 760px) {
      .report { padding: 20px 12px 32px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      table { display: block; overflow-x: auto; }
    }
    @media print {
      body { background: #ffffff; }
      .report { max-width: none; padding: 0; }
      .header, .card, .section, .note { box-shadow: none; }
      .cards, tr, .section { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <main class="report">
    <section class="header">
      <h1>VisionGuard Insights Report</h1>
      <div class="meta">
        <p><strong>Generated:</strong> ${escapeHtml(formatReportDate(exportedAt))}</p>
        <p><strong>Selected time window:</strong> ${escapeHtml(timeWindowLabel)}</p>
        <p><strong>User:</strong> ${escapeHtml(userLabel)}</p>
        <p><strong>Data basis:</strong> ${escapeHtml(dataBasisLabel)}</p>
        <p><strong>Data source:</strong> ${escapeHtml(dataSourceLabel)}</p>
      </div>
    </section>

    <section class="section">
      <h2>Key Insights</h2>
      <ul>${keyInsights.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </section>

    <section class="cards" aria-label="Metric summary">
      ${metricCard("Dominant Alert", summary.dominantAlertLabel, `${summary.dominantAlertCount} of ${summary.totalAlerts} alerts`)}
      ${metricCard("High-Risk Share", formatInsightPercent(summary.highPriorityShare), "Alerts with stronger fatigue-related cues")}
      ${metricCard("Signal Interruptions", formatInsightPercent(summary.signalInterruptionShare), `${summary.signalInterruptionCount} camera or tracking alerts`)}
      ${metricCard("Drives Analyzed", String(driveRows.length), "Recent drives in the selected window")}
    </section>

    <section class="section">
      <h2>Alerts by Drive</h2>
      <div class="chart">${buildAlertsByDriveSvg(driveRows)}</div>
    </section>

    <section class="section">
      <h2>Alert Composition</h2>
      <div class="chart">${buildAlertCompositionSvg(composition)}</div>
    </section>

    <section class="section">
      <h2>Time of Day</h2>
      <div class="chart">${buildTimeOfDaySvg(timeOfDay)}</div>
    </section>

    <section class="section">
      <h2>Camera Signal Summary</h2>
      <p>${escapeHtml(signalQuality.count)} interruptions, ${escapeHtml(formatInsightPercent(signalQuality.share))} of alerts, ${escapeHtml(signalQuality.affectedSessionCount)} affected drives.</p>
      <p style="margin-top: 8px;">Most affected drive: ${escapeHtml(signalQuality.mostLimitedDriveLabel ?? "None")}</p>
    </section>

    <section class="section">
      <h2>Drive Comparison</h2>
      ${driveComparisonTable(driveRows)}
    </section>

    <section class="section">
      <h2>Attention Areas</h2>
      ${attentionAreaList(attentionAreas)}
    </section>

    <section class="note">
      <p>Insights are intended for awareness only and are not a medical diagnosis, a measure of final system-level accuracy, or a guarantee of driving safety.</p>
      <p style="margin-top: 8px;">Raw webcam frames, uploaded videos, blobs, and base64 payloads are not stored or exported.</p>
    </section>
  </main>
</body>
</html>`;
}

export function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function reportDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function formatReportDate(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

function metricCard(label: string, value: string, helper: string): string {
  return `<article class="card"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div><p style="margin-top:8px;color:#64748b;font-size:13px;">${escapeHtml(helper)}</p></article>`;
}

function buildAlertsByDriveSvg(rows: InsightSessionComparisonRow[]): string {
  if (rows.length === 0) return `<p>No drive alerts are available.</p>`;
  const width = 920;
  const rowHeight = 58;
  const height = 36 + rows.length * rowHeight;
  const barX = 300;
  const barWidth = 440;
  const maxTotal = Math.max(1, ...rows.map((row) => row.eventCount));

  const body = rows
    .map((row, index) => {
      const y = 30 + index * rowHeight;
      let x = barX;
      const segments = ALERT_SEGMENTS.map((segment) => {
        const value = row[segment.key];
        if (value <= 0) return "";
        const widthValue = Math.max(5, (value / maxTotal) * barWidth);
        const segmentSvg = `<rect x="${x}" y="${y}" width="${widthValue}" height="18" rx="5" fill="${segment.color}" />`;
        x += widthValue;
        return segmentSvg;
      }).join("");
      return `<g>
        <text x="20" y="${y + 13}" font-size="13" font-weight="700" fill="#0f172a">${escapeHtml(row.driveLabel)}</text>
        <text x="20" y="${y + 31}" font-size="12" fill="#64748b">${escapeHtml(row.durationLabel)} - ${escapeHtml(row.dominantPattern)}</text>
        <rect x="${barX}" y="${y}" width="${barWidth}" height="18" rx="5" fill="#e2e8f0" />
        ${segments}
        <text x="${barX + barWidth + 18}" y="${y + 13}" font-size="13" font-weight="700" fill="#0f172a">${row.eventCount} alerts</text>
      </g>`;
    })
    .join("");

  return `<svg role="img" aria-label="Alerts by Drive" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">${legend(20, 14)}${body}</svg>`;
}

function buildAlertCompositionSvg(items: InsightCompositionItem[]): string {
  const visible = items.filter((item) => item.count > 0);
  if (visible.length === 0) return `<p>No alert composition is available.</p>`;
  const width = 920;
  const rowHeight = 46;
  const height = 18 + visible.length * rowHeight;
  const barX = 260;
  const barWidth = 430;
  const maxCount = Math.max(1, ...visible.map((item) => item.count));
  const body = visible
    .map((item, index) => {
      const y = 18 + index * rowHeight;
      const widthValue = Math.max(6, (item.count / maxCount) * barWidth);
      return `<g>
        <text x="20" y="${y + 14}" font-size="13" font-weight="700" fill="#0f172a">${escapeHtml(item.label)}</text>
        <rect x="${barX}" y="${y}" width="${barWidth}" height="16" rx="5" fill="#e2e8f0" />
        <rect x="${barX}" y="${y}" width="${widthValue}" height="16" rx="5" fill="${escapeHtml(item.color)}" />
        <text x="${barX + barWidth + 18}" y="${y + 13}" font-size="13" font-weight="700" fill="#0f172a">${item.count} - ${escapeHtml(formatInsightPercent(item.share))}</text>
      </g>`;
    })
    .join("");
  return `<svg role="img" aria-label="Alert Composition" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
}

function buildTimeOfDaySvg(items: InsightTimeOfDayItem[]): string {
  const width = 760;
  const height = 260;
  const chartX = 60;
  const chartY = 30;
  const chartHeight = 150;
  const barWidth = 92;
  const gap = 70;
  const maxCount = Math.max(1, ...items.map((item) => item.count));
  const body = items
    .map((item, index) => {
      const x = chartX + index * (barWidth + gap);
      const barHeight = (item.count / maxCount) * chartHeight;
      const y = chartY + chartHeight - barHeight;
      return `<g>
        <rect x="${x}" y="${chartY}" width="${barWidth}" height="${chartHeight}" rx="8" fill="#f1f5f9" />
        <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="8" fill="#2563eb" />
        <text x="${x + barWidth / 2}" y="${chartY + chartHeight + 24}" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">${escapeHtml(item.label)}</text>
        <text x="${x + barWidth / 2}" y="${chartY + chartHeight + 42}" text-anchor="middle" font-size="12" fill="#64748b">${item.count} alerts</text>
        <text x="${x + barWidth / 2}" y="${chartY + chartHeight + 58}" text-anchor="middle" font-size="12" fill="#64748b">${escapeHtml(formatInsightPercent(item.share))}</text>
      </g>`;
    })
    .join("");
  return `<svg role="img" aria-label="Time of Day" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
}

function legend(x: number, y: number): string {
  return ALERT_SEGMENTS.map((segment, index) => {
    const itemX = x + index * 142;
    return `<g><rect x="${itemX}" y="${y - 9}" width="10" height="10" rx="3" fill="${segment.color}" /><text x="${itemX + 16}" y="${y}" font-size="12" font-weight="700" fill="#475569">${escapeHtml(segment.label)}</text></g>`;
  }).join("");
}

function driveComparisonTable(rows: InsightSessionComparisonRow[]): string {
  if (rows.length === 0) return `<p>No drive comparison is available.</p>`;
  return `<table>
    <thead>
      <tr>
        <th>Drive</th>
        <th>Duration</th>
        <th>Alerts</th>
        <th>High-risk</th>
        <th>Signal interruptions</th>
        <th>Main pattern</th>
      </tr>
    </thead>
    <tbody>
      ${rows
        .map(
          (row) => `<tr>
            <td>${escapeHtml(row.driveLabel)}</td>
            <td>${escapeHtml(row.durationLabel)}</td>
            <td>${escapeHtml(row.eventCount)}</td>
            <td>${escapeHtml(row.highPriorityCount)}</td>
            <td>${escapeHtml(row.signalInterruptionCount)}</td>
            <td>${escapeHtml(row.dominantPattern)}</td>
          </tr>`
        )
        .join("")}
    </tbody>
  </table>`;
}

function attentionAreaList(areas: InsightRecommendation[]): string {
  return areas
    .map(
      (area) => `<article style="margin-top:10px;padding:12px;border:1px solid #dbe3ef;border-radius:12px;background:#f8fafc;">
        <span class="priority priority-${escapeHtml(area.priority)}">${escapeHtml(area.priority)}</span>
        <h3 style="margin-top:8px;">${escapeHtml(area.title)}</h3>
        <p>${escapeHtml(area.body)}</p>
      </article>`
    )
    .join("");
}
