export type RiskLevel = "Low" | "Medium" | "High";

export type EventType = "Normal" | "Eyes Closed" | "Yawning";

export interface DriverEvent {
  /** Seconds before "now" the event occurred. */
  secondsAgo: number;
  type: EventType;
}

export interface DashboardData {
  driver: {
    name: string;
    initials: string;
    /** Seconds since current driving session started. */
    sessionStartedSecondsAgo: number;
  };
  status: {
    eyesClosed: number;
    yawn: number;
    fps: number;
    isLive: boolean;
  };
  risk: {
    level: RiskLevel;
    score: number;
  };
  notifications: number;
  events: DriverEvent[];
}

export const dashboardData: DashboardData = {
  driver: {
    name: "Local Driver",
    initials: "LD",
    sessionStartedSecondsAgo: 12 * 60 + 45,
  },
  status: {
    eyesClosed: 5,
    yawn: 3,
    fps: 24.6,
    isLive: true,
  },
  risk: {
    level: "Medium",
    score: 62,
  },
  notifications: 0,
  events: [
    { secondsAgo: 95, type: "Normal" },
    { secondsAgo: 168, type: "Eyes Closed" },
    { secondsAgo: 213, type: "Yawning" },
    { secondsAgo: 299, type: "Eyes Closed" },
    { secondsAgo: 375, type: "Normal" },
    { secondsAgo: 614, type: "Normal" },
    { secondsAgo: 1085, type: "Yawning" },
  ],
};

export const eventStyle: Record<
  EventType,
  { dot: string; label: string }
> = {
  Normal: { dot: "bg-emerald-500", label: "text-slate-700" },
  "Eyes Closed": { dot: "bg-orange-500", label: "text-slate-700" },
  Yawning: { dot: "bg-rose-500", label: "text-slate-700" },
};

export const riskStyle: Record<
  RiskLevel,
  { text: string; subtitle: string; gauge: string }
> = {
  Low: {
    text: "text-emerald-500",
    subtitle: "All Clear",
    gauge: "#10b981",
  },
  Medium: {
    text: "text-orange-500",
    subtitle: "Stay Alert",
    gauge: "#f97316",
  },
  High: {
    text: "text-red-500",
    subtitle: "Take a Break",
    gauge: "#ef4444",
  },
};

export function formatHMS(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return [h, m, sec].map((n) => String(n).padStart(2, "0")).join(":");
}

export function formatClock(date: Date): string {
  return [date.getHours(), date.getMinutes(), date.getSeconds()]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
}

export function formatHM(date: Date): string {
  return [date.getHours(), date.getMinutes()]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
}
