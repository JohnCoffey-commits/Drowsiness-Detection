import type {
  DriverHistoryEvent,
  DriverHistorySession,
  History48hStore,
  HistorySeverity,
  HistorySource,
  HistoryState,
  ReviewStatus,
} from "@/lib/history48hTypes";

const REASONS = {
  normal: "Normal sampled period in demo history.",
  eye_warning_candidate: "Eye warning candidate from temporal eye evidence.",
  mouth_warning_candidate: "Recent mouth/yawn evidence active.",
  high_confidence_drowsiness_candidate:
    "Recent mouth/yawn evidence overlapped with eye warning candidate.",
  signal_unreliable: "Face/ROI signal quality may be unreliable.",
} satisfies Record<HistoryState, string>;

const SESSION_PLANS: Array<{
  id: string;
  hoursAgo: number;
  durationMin: number;
  source: HistorySource;
  pattern: HistoryState[];
}> = [
  {
    id: "demo-session-48h-01",
    hoursAgo: 45.5,
    durationMin: 56,
    source: "mock",
    pattern: [
      "normal",
      "normal",
      "eye_warning_candidate",
      "normal",
      "mouth_warning_candidate",
      "normal",
      "signal_unreliable",
      "normal",
    ],
  },
  {
    id: "demo-session-48h-02",
    hoursAgo: 39,
    durationMin: 72,
    source: "mock",
    pattern: [
      "normal",
      "eye_warning_candidate",
      "eye_warning_candidate",
      "mouth_warning_candidate",
      "high_confidence_drowsiness_candidate",
      "normal",
      "normal",
      "signal_unreliable",
      "mouth_warning_candidate",
    ],
  },
  {
    id: "demo-session-48h-03",
    hoursAgo: 31.5,
    durationMin: 48,
    source: "mock",
    pattern: [
      "normal",
      "normal",
      "mouth_warning_candidate",
      "normal",
      "eye_warning_candidate",
      "normal",
      "mouth_warning_candidate",
    ],
  },
  {
    id: "demo-session-48h-04",
    hoursAgo: 24,
    durationMin: 95,
    source: "mock",
    pattern: [
      "normal",
      "eye_warning_candidate",
      "eye_warning_candidate",
      "mouth_warning_candidate",
      "high_confidence_drowsiness_candidate",
      "mouth_warning_candidate",
      "signal_unreliable",
      "normal",
      "eye_warning_candidate",
      "normal",
    ],
  },
  {
    id: "demo-session-48h-05",
    hoursAgo: 17.75,
    durationMin: 64,
    source: "mock",
    pattern: [
      "normal",
      "normal",
      "eye_warning_candidate",
      "mouth_warning_candidate",
      "normal",
      "signal_unreliable",
      "normal",
      "mouth_warning_candidate",
      "eye_warning_candidate",
    ],
  },
  {
    id: "demo-session-48h-06",
    hoursAgo: 9.25,
    durationMin: 82,
    source: "mock",
    pattern: [
      "normal",
      "eye_warning_candidate",
      "mouth_warning_candidate",
      "eye_warning_candidate",
      "high_confidence_drowsiness_candidate",
      "mouth_warning_candidate",
      "signal_unreliable",
      "eye_warning_candidate",
      "normal",
      "normal",
    ],
  },
  {
    id: "demo-session-48h-07",
    hoursAgo: 2.8,
    durationMin: 58,
    source: "mock",
    pattern: [
      "normal",
      "normal",
      "mouth_warning_candidate",
      "eye_warning_candidate",
      "normal",
      "signal_unreliable",
      "mouth_warning_candidate",
      "normal",
    ],
  },
];

function scoreForState(state: HistoryState): number | undefined {
  if (state === "normal") return 10;
  if (state === "eye_warning_candidate") return 45;
  if (state === "mouth_warning_candidate") return 55;
  if (state === "high_confidence_drowsiness_candidate") return 80;
  return undefined;
}

function severityForState(state: HistoryState): HistorySeverity {
  if (state === "normal") return "low";
  if (state === "high_confidence_drowsiness_candidate") return "high";
  if (state === "signal_unreliable") return "unreliable";
  return "medium";
}

function reviewStatusForState(
  state: HistoryState,
  eventIndex: number
): ReviewStatus {
  if (state === "normal") return "not_required";
  if (state === "high_confidence_drowsiness_candidate") return "pending";
  if (state === "signal_unreliable") return eventIndex % 2 === 0 ? "pending" : "reviewed";
  return eventIndex % 4 === 0 ? "pending" : eventIndex % 5 === 0 ? "reviewed" : "not_required";
}

function pEyeForState(state: HistoryState, eventIndex: number): number | undefined {
  if (state === "normal") return 0.12 + (eventIndex % 4) * 0.04;
  if (state === "eye_warning_candidate") return 0.55 + (eventIndex % 5) * 0.07;
  if (state === "mouth_warning_candidate") return 0.24 + (eventIndex % 6) * 0.06;
  if (state === "high_confidence_drowsiness_candidate") return 0.68 + (eventIndex % 4) * 0.06;
  return undefined;
}

function pYawnForState(state: HistoryState, eventIndex: number): number | undefined {
  if (state === "normal") return 0.03 + (eventIndex % 3) * 0.02;
  if (state === "eye_warning_candidate") return 0.04 + (eventIndex % 3) * 0.03;
  if (state === "mouth_warning_candidate") return 0.5 + (eventIndex % 5) * 0.08;
  if (state === "high_confidence_drowsiness_candidate") return 0.62 + (eventIndex % 4) * 0.07;
  return undefined;
}

function eyeStrengthForState(
  state: HistoryState,
  eventIndex: number
): DriverHistoryEvent["eyeEvidenceStrength"] {
  if (state === "normal") return "none";
  if (state === "signal_unreliable") return "unknown";
  if (state === "high_confidence_drowsiness_candidate") return eventIndex % 2 === 0 ? "strong" : "moderate";
  if (state === "eye_warning_candidate") return eventIndex % 3 === 0 ? "moderate" : "weak";
  return eventIndex % 4 === 0 ? "moderate" : "none";
}

function roundProbability(value: number | undefined): number | undefined {
  return value == null ? undefined : Number(value.toFixed(3));
}

function addMinutes(date: Date, minutes: number): Date {
  return new Date(date.getTime() + minutes * 60_000);
}

function addSeconds(date: Date, seconds: number): Date {
  return new Date(date.getTime() + seconds * 1_000);
}

function countState(
  events: DriverHistoryEvent[],
  state: HistoryState
): number {
  return events.filter((event) => event.state === state).length;
}

export function createDemoHistory48hStore(now = new Date()): History48hStore {
  const events: DriverHistoryEvent[] = [];
  const sessions: DriverHistorySession[] = [];
  return createDemoHistory48hStoreForUser(now, undefined, events, sessions);
}

export function createUserDemoHistory48hStore(
  now = new Date(),
  userId?: string
): History48hStore {
  return createDemoHistory48hStoreForUser(now, userId, [], []);
}

function createDemoHistory48hStoreForUser(
  now: Date,
  userId: string | undefined,
  events: DriverHistoryEvent[],
  sessions: DriverHistorySession[]
): History48hStore {
  const userIdSuffix = userId ? `-${userId.replace(/[^a-zA-Z0-9_-]/g, "-")}` : "";

  SESSION_PLANS.forEach((plan, sessionIndex) => {
    const sessionId = `${plan.id}${userIdSuffix}`;
    const startedAt = new Date(now.getTime() - plan.hoursAgo * 60 * 60_000);
    const endedAt = addMinutes(startedAt, plan.durationMin);
    const eventSpacingMin = plan.durationMin / (plan.pattern.length + 1);
    const sessionEvents: DriverHistoryEvent[] = [];

    plan.pattern.forEach((state, eventIndex) => {
      const timestamp = addMinutes(
        startedAt,
        eventSpacingMin * (eventIndex + 1)
      );
      const durationSec =
        state === "normal"
          ? 70 + ((eventIndex + sessionIndex) % 4) * 25
          : state === "signal_unreliable"
            ? 18 + (eventIndex % 3) * 11
            : 22 + ((eventIndex + sessionIndex) % 5) * 14;
      const event: DriverHistoryEvent = {
        id: `${sessionId}-event-${String(eventIndex + 1).padStart(2, "0")}`,
        userId,
        sessionId,
        timestamp: timestamp.toISOString(),
        endTimestamp: addSeconds(timestamp, durationSec).toISOString(),
        durationSec,
        state,
        severity: severityForState(state),
        source: plan.source,
        pEyeClosedMax: roundProbability(pEyeForState(state, eventIndex + sessionIndex)),
        pYawnMax: roundProbability(pYawnForState(state, eventIndex + sessionIndex)),
        candidateSeverityScore: scoreForState(state),
        eyeEvidenceStrength: eyeStrengthForState(state, eventIndex + sessionIndex),
        reason:
          state === "eye_warning_candidate" && eventIndex % 3 === 0
            ? "Manual review recommended for candidate interval."
            : REASONS[state],
        reviewStatus: reviewStatusForState(state, eventIndex + sessionIndex),
      };

      sessionEvents.push(event);
      events.push(event);
    });

    sessions.push({
      id: sessionId,
      userId,
      source: plan.source,
      startedAt: startedAt.toISOString(),
      endedAt: endedAt.toISOString(),
      durationMin: plan.durationMin,
      status: "demo",
      normalCount: countState(sessionEvents, "normal"),
      eyeWarningCount: countState(sessionEvents, "eye_warning_candidate"),
      mouthWarningCount: countState(sessionEvents, "mouth_warning_candidate"),
      highConfidenceCount: countState(
        sessionEvents,
        "high_confidence_drowsiness_candidate"
      ),
      signalUnreliableCount: countState(sessionEvents, "signal_unreliable"),
      reviewPendingCount: sessionEvents.filter(
        (event) => event.reviewStatus === "pending"
      ).length,
    });
  });

  return {
    events: events.sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    ),
    sessions: sessions.sort(
      (a, b) =>
        new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime()
    ),
    updatedAt: now.toISOString(),
  };
}
