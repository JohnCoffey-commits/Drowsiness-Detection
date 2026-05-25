"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LiveVideoCard } from "@/components/dashboard/LiveVideoCard";
import { StatusMetricCard } from "@/components/dashboard/StatusMetricCard";
import { DrowsinessRiskCard } from "@/components/dashboard/DrowsinessRiskCard";
import { DrowsinessLevelChart } from "@/components/dashboard/DrowsinessLevelChart";
import { RecentEventsList } from "@/components/dashboard/RecentEventsList";
import { upsertHistory48hSession } from "@/lib/history48hStorage";
import type { DriverHistorySession } from "@/lib/history48hTypes";
import {
  IDLE_LIVE_MONITOR_RISK_STATE,
  getLiveMonitorRiskStateKey,
  type LiveMonitorRiskState,
} from "@/lib/liveMonitorRiskUtils";
import {
  appendLiveMonitorDashboardEvent,
  appendLiveMonitorRiskPoint,
  createEmptyLiveMonitorDashboardStore,
  createLiveMonitorDriveSessionId,
  createLiveMonitorRiskPoint,
  createNormalDashboardEventDraft,
  getCurrentSessionLiveMonitorRiskPoints,
  getTodayLiveMonitorEvents,
  loadLiveMonitorDashboardStore,
  saveLiveMonitorDashboardStore,
  summarizeCurrentDriveEvents,
} from "@/lib/liveMonitorDashboardStore";
import { useVisionGuardAuth } from "@/lib/authStore";
import { useVisionGuardNotifications } from "@/lib/notificationStore";
import { appendLiveMonitorDashboardEventToHistory } from "@/lib/liveMonitorHistoryIngestion";
import { getArchiveClientId } from "@/lib/archiveClientId";
import {
  buildLiveArchiveEventPayload,
  buildLiveArchiveSessionPayload,
  saveLiveArchiveEvent,
  saveLiveArchiveSession,
} from "@/lib/backendArchiveApi";
import type {
  LiveMonitorDashboardEventDraft,
  LiveMonitorDashboardStore,
} from "@/lib/liveMonitorDashboardTypes";
import { useVisionGuardSettings } from "@/lib/settingsStore";

const LIVE_MONITOR_WARNING =
  "This output is a realtime rule-based warning-candidate analysis, not final system-level drowsiness accuracy.";

interface CurrentDriveSession {
  id: string;
  startedAt: Date;
}

function createCurrentDriveSession(startedAt = new Date()): CurrentDriveSession {
  return {
    id: createLiveMonitorDriveSessionId(startedAt),
    startedAt,
  };
}

function driveDurationMin(startedAt: Date, endedAt: Date): number {
  const durationMs = Math.max(0, endedAt.getTime() - startedAt.getTime());
  return Math.round((durationMs / 60_000) * 10) / 10;
}

function createDriveHistorySession(
  session: CurrentDriveSession,
  endedAt: Date,
  userId: string | undefined,
  status: DriverHistorySession["status"]
): DriverHistorySession {
  return {
    id: session.id,
    userId,
    source: "live_monitor",
    startedAt: session.startedAt.toISOString(),
    endedAt: endedAt.toISOString(),
    durationMin: driveDurationMin(session.startedAt, endedAt),
    status,
    normalCount: 0,
    eyeWarningCount: 0,
    mouthWarningCount: 0,
    highConfidenceCount: 0,
    signalUnreliableCount: 0,
    reviewPendingCount: 0,
  };
}

export function LiveMonitorPage() {
  const [riskState, setRiskState] = useState<LiveMonitorRiskState>(
    IDLE_LIVE_MONITOR_RISK_STATE
  );
  const [dashboardStore, setDashboardStore] = useState<LiveMonitorDashboardStore>(() =>
    createEmptyLiveMonitorDashboardStore()
  );
  const [currentDriveSession, setCurrentDriveSession] = useState(() =>
    createCurrentDriveSession()
  );
  const currentDriveSessionRef = useRef(currentDriveSession);
  const [referenceNow, setReferenceNow] = useState<Date>(() => new Date());
  const [recentEventsExpanded, setRecentEventsExpanded] = useState(false);
  const [archiveStatus, setArchiveStatus] = useState("");
  const { authState, currentUser } = useVisionGuardAuth();
  const { upsertDrivingDigestNotification } = useVisionGuardNotifications();
  const { settings } = useVisionGuardSettings();
  const lastRiskPointKeyRef = useRef("");
  const previousRiskSeverityRef = useRef(riskState.severity);
  const lastNormalEventAtRef = useRef(0);
  const minimalMode = settings.liveMonitor.minimalMode;
  const legacyRecordsVisible =
    Boolean(currentUser?.id) && authState.users[0]?.id === currentUser?.id;
  const currentUserId = currentUser?.id;

  const recordDriveSession = useCallback(
    (
      session: CurrentDriveSession,
      endedAt: Date,
      status: DriverHistorySession["status"]
    ) => {
      const historySession = createDriveHistorySession(
        session,
        endedAt,
        currentUserId,
        status
      );
      upsertHistory48hSession(historySession, endedAt);

      if (!currentUserId) return;

      const archivePayload = buildLiveArchiveSessionPayload(
        historySession,
        getArchiveClientId(),
        currentUserId
      );
      void saveLiveArchiveSession(archivePayload).then((result) => {
        setArchiveStatus(
          result.ok
            ? "Saved Live Monitor drive summary to local backend archive."
            : "Archive save failed for the latest Live Monitor drive."
        );
      });
    },
    [currentUserId]
  );

  const handleMonitoringSessionStart = useCallback(
    (startedAt: Date) => {
      const nextSession = createCurrentDriveSession(startedAt);
      currentDriveSessionRef.current = nextSession;
      setCurrentDriveSession(nextSession);
      recordDriveSession(nextSession, startedAt, "partial");
    },
    [recordDriveSession]
  );

  const handleMonitoringSessionEnd = useCallback(
    (endedAt: Date) => {
      const session = currentDriveSessionRef.current;
      const safeEndedAt =
        endedAt.getTime() >= session.startedAt.getTime()
          ? endedAt
          : session.startedAt;
      recordDriveSession(session, safeEndedAt, "completed");
    },
    [recordDriveSession]
  );

  const handleRiskStateChange = useCallback((nextRiskState: LiveMonitorRiskState) => {
    setRiskState(nextRiskState);
  }, []);

  const updateDashboardStore = useCallback(
    (updater: (store: LiveMonitorDashboardStore) => LiveMonitorDashboardStore) => {
      setDashboardStore((current) => {
        const nextStore = updater(current);
        if (nextStore !== current) {
          saveLiveMonitorDashboardStore(nextStore);
        }
        return nextStore;
      });
    },
    []
  );

  const recordDashboardEvent = useCallback(
    (event: LiveMonitorDashboardEventDraft) => {
      const activeSession = currentDriveSessionRef.current;
      const eventTimestamp = new Date(event.timestamp);
      if (Number.isFinite(eventTimestamp.getTime())) {
        setReferenceNow(eventTimestamp);
      }

      updateDashboardStore((store) =>
        appendLiveMonitorDashboardEvent(
          store,
          event,
          activeSession.id,
          currentUser?.id
        )
      );

      if (currentUser) {
        const persistedEvent = {
          ...event,
          sessionId: activeSession.id,
          source: "live_monitor_prototype" as const,
          userId: currentUser.id,
        };
        upsertDrivingDigestNotification(persistedEvent);
        const historyRecord = appendLiveMonitorDashboardEventToHistory(
          persistedEvent,
          currentUser.id
        );
        if (historyRecord) {
          const archivePayload = buildLiveArchiveEventPayload(
            historyRecord,
            getArchiveClientId(),
            currentUser.id
          );
          void saveLiveArchiveEvent(archivePayload).then((result) => {
            setArchiveStatus(
              result.ok
                ? "Saved stable Live Monitor event to local backend archive."
                : "Archive save failed for the latest stable Live Monitor event."
            );
          });
        }
      }
    },
    [currentUser, updateDashboardStore, upsertDrivingDigestNotification]
  );

  const recordRiskPoint = useCallback(
    (state: LiveMonitorRiskState, now = new Date()) => {
      const activeSession = currentDriveSessionRef.current;
      setReferenceNow(now);
      updateDashboardStore((store) =>
        appendLiveMonitorRiskPoint(
          store,
          createLiveMonitorRiskPoint(
            state,
            activeSession.id,
            now,
            currentUser?.id
          )
        )
      );
    },
    [currentUser?.id, updateDashboardStore]
  );

  useEffect(() => {
    const loadStoredDashboard = window.setTimeout(() => {
      setDashboardStore(loadLiveMonitorDashboardStore());
    }, 0);

    const interval = window.setInterval(() => {
      setReferenceNow(new Date());
    }, 5_000);

    return () => {
      window.clearTimeout(loadStoredDashboard);
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const nextRiskPointKey = getLiveMonitorRiskStateKey(riskState);
    const isIdle = riskState.severity === "idle";

    if (nextRiskPointKey === lastRiskPointKeyRef.current) {
      return;
    }

    lastRiskPointKeyRef.current = nextRiskPointKey;

    if (isIdle) {
      return;
    }

    const recordRiskPointTimeout = window.setTimeout(() => {
      recordRiskPoint(riskState);
    }, 0);

    return () => window.clearTimeout(recordRiskPointTimeout);
  }, [recordRiskPoint, riskState]);

  useEffect(() => {
    if (riskState.severity === "idle") {
      return;
    }

    const interval = window.setInterval(() => {
      recordRiskPoint(riskState);
    }, 10_000);

    return () => window.clearInterval(interval);
  }, [recordRiskPoint, riskState]);

  useEffect(() => {
    const previousSeverity = previousRiskSeverityRef.current;

    if (riskState.severity === "low" && previousSeverity !== "low") {
      const now = new Date();
      const nowMs = now.getTime();

      if (nowMs - lastNormalEventAtRef.current >= 15_000) {
        recordDashboardEvent(
          createNormalDashboardEventDraft(
            previousSeverity === "idle"
              ? "monitoring_started"
              : "returned_to_monitoring",
            now
          )
        );
        lastNormalEventAtRef.current = nowMs;
      }
    }

    previousRiskSeverityRef.current = riskState.severity;
  }, [recordDashboardEvent, riskState.severity]);

  useEffect(() => {
    if (!recentEventsExpanded) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setRecentEventsExpanded(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [recentEventsExpanded]);

  const currentDriveCounts = useMemo(
    () =>
      summarizeCurrentDriveEvents(
        dashboardStore.events,
        currentDriveSession.id,
        currentUser?.id,
        legacyRecordsVisible
      ),
    [
      currentDriveSession.id,
      currentUser?.id,
      dashboardStore.events,
      legacyRecordsVisible,
    ]
  );

  const todayEvents = useMemo(
    () =>
      getTodayLiveMonitorEvents(
        dashboardStore.events,
        referenceNow,
        currentUser?.id,
        legacyRecordsVisible
      ),
    [currentUser?.id, dashboardStore.events, legacyRecordsVisible, referenceNow]
  );

  const currentSessionRiskPoints = useMemo(
    () =>
      getCurrentSessionLiveMonitorRiskPoints(
        dashboardStore.riskPoints,
        currentDriveSession.id,
        referenceNow,
        currentUser?.id,
        legacyRecordsVisible
      ),
    [
      currentDriveSession.id,
      currentUser?.id,
      dashboardStore.riskPoints,
      legacyRecordsVisible,
      referenceNow,
    ]
  );

  return (
      <main className="flex min-h-0 flex-1 flex-col overflow-x-hidden overflow-y-auto px-4 py-4 lg:px-6 lg:py-4 xl:overflow-hidden">
        <div
          className={`mx-auto flex w-full flex-col gap-4 xl:min-h-0 xl:flex-1 xl:gap-5 ${
            minimalMode ? "max-w-[920px]" : "max-w-[1600px]"
          }`}
        >
          <div
            className={
              minimalMode
                ? "flex min-h-0 flex-1 flex-col gap-4 xl:gap-5"
                : "grid grid-cols-1 gap-4 xl:min-h-0 xl:flex-1 xl:grid-cols-[minmax(0,1.45fr)_minmax(420px,1fr)] xl:gap-5 2xl:grid-cols-[minmax(0,1.5fr)_minmax(460px,1fr)]"
            }
          >
            <div
              className={
                minimalMode
                  ? "order-2 flex min-h-0 flex-col gap-4 xl:gap-5"
                  : "flex min-h-0 flex-col gap-4 xl:gap-5"
              }
            >
              <div
                className={
                  minimalMode
                    ? "min-h-[112px] overflow-visible"
                    : "min-h-[360px] overflow-hidden xl:min-h-0 xl:flex-[1.35]"
                }
              >
                <LiveVideoCard
                  minimalMode={minimalMode}
                  onRiskStateChange={handleRiskStateChange}
                  onDashboardEvent={recordDashboardEvent}
                  onMonitoringSessionStart={handleMonitoringSessionStart}
                  onMonitoringSessionEnd={handleMonitoringSessionEnd}
                />
              </div>

              {!minimalMode && (
                <div className="min-h-[260px] overflow-hidden xl:min-h-0 xl:flex-[0.85]">
                  <DrowsinessLevelChart
                    points={currentSessionRiskPoints}
                    now={referenceNow}
                    sessionStartedAt={currentDriveSession.startedAt}
                    isMonitoringActive={riskState.severity !== "idle"}
                  />
                </div>
              )}
            </div>

            <div
              className={
                minimalMode
                  ? "order-1 min-h-[440px] min-w-0 overflow-visible sm:min-h-[520px] xl:min-h-0 xl:flex-1"
                  : "relative flex min-h-[640px] min-w-0 flex-col gap-4 overflow-visible xl:min-h-0 xl:min-w-[420px] xl:gap-5 2xl:min-w-[460px]"
              }
            >
              {minimalMode ? (
                <DrowsinessRiskCard riskState={riskState} variant="prominent" />
              ) : (
                <>
                  <div
                    aria-hidden={recentEventsExpanded}
                    className={`flex min-h-0 flex-[1.35] flex-col gap-4 transition-all duration-[360ms] ease-out xl:gap-5 ${
                      recentEventsExpanded
                        ? "pointer-events-none scale-[0.96] opacity-0"
                        : "scale-100 opacity-100"
                    }`}
                  >
                    <div className="grid shrink-0 grid-cols-1 gap-4 xl:gap-5 sm:grid-cols-2">
                      <StatusMetricCard
                        type="closed"
                        events={currentDriveCounts.eyeWarnings}
                      />
                      <StatusMetricCard
                        type="yawn"
                        events={currentDriveCounts.yawnWarnings}
                      />
                    </div>
                    <div className="min-h-0 flex-1 overflow-hidden">
                      <DrowsinessRiskCard riskState={riskState} />
                    </div>
                  </div>

                  <div
                    aria-hidden={recentEventsExpanded}
                    className={`min-h-0 flex-[0.85] overflow-hidden transition-all duration-[360ms] ease-out ${
                      recentEventsExpanded
                        ? "pointer-events-none scale-[0.98] opacity-0"
                        : "scale-100 opacity-100"
                    }`}
                  >
                    <RecentEventsList
                      events={todayEvents}
                      actionTabIndex={recentEventsExpanded ? -1 : 0}
                      onExpand={() => setRecentEventsExpanded(true)}
                    />
                  </div>

                  <div
                    aria-hidden={!recentEventsExpanded}
                    className={`absolute inset-x-0 bottom-0 z-30 h-full origin-bottom transform-gpu overflow-hidden transition-all duration-[420ms] ease-out ${
                      recentEventsExpanded
                        ? "pointer-events-auto opacity-100"
                        : "pointer-events-none opacity-0"
                    }`}
                    style={{
                      clipPath: recentEventsExpanded
                        ? "inset(0 0 0 0 round 2rem)"
                        : "inset(62% 0 0 0 round 2rem)",
                      transform: recentEventsExpanded
                        ? "translateY(0) scale(1)"
                        : "translateY(10px) scale(0.985)",
                    }}
                  >
                    <RecentEventsList
                      events={todayEvents}
                      expanded
                      actionTabIndex={recentEventsExpanded ? 0 : -1}
                      onCollapse={() => setRecentEventsExpanded(false)}
                      className="border-slate-200/80 shadow-2xl shadow-slate-950/20 hover:shadow-2xl"
                    />
                  </div>
                </>
              )}
            </div>
          </div>
          <p className="shrink-0 px-1 text-[11px] leading-4 text-slate-500">
            {LIVE_MONITOR_WARNING}
          </p>
          {archiveStatus ? (
            <p className="shrink-0 px-1 text-[11px] leading-4 text-slate-500">
              Archive: {archiveStatus}
            </p>
          ) : null}
        </div>
      </main>
  );
}
