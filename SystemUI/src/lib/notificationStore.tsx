"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useVisionGuardAuth } from "@/lib/authStore";
import type {
  VisionGuardNotification,
  VisionGuardNotificationCategory,
  VisionGuardNotificationRoute,
  VisionGuardNotificationSeverity,
  VisionGuardNotificationSource,
} from "@/lib/notificationTypes";
import type { LiveMonitorDashboardEvent } from "@/lib/liveMonitorDashboardTypes";

export const VISION_GUARD_NOTIFICATIONS_STORAGE_KEY =
  "visionguard.notifications.v1";

export type VisionGuardNotificationDraft = Omit<
  VisionGuardNotification,
  "createdAt" | "readAt"
> & {
  createdAt?: string;
};

interface VisionGuardNotificationsContextValue {
  notifications: VisionGuardNotification[];
  unreadCount: number;
  addNotification: (notification: VisionGuardNotificationDraft) => void;
  markNotificationRead: (notificationId: string) => void;
  markAllRead: () => void;
  clearRead: () => void;
}

const VisionGuardNotificationsContext =
  createContext<VisionGuardNotificationsContextValue | null>(null);

function hasBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function normalizeText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function isCategory(value: unknown): value is VisionGuardNotificationCategory {
  return (
    value === "warning_candidate" || value === "system" || value === "review"
  );
}

function isSeverity(value: unknown): value is VisionGuardNotificationSeverity {
  return (
    value === "info" ||
    value === "success" ||
    value === "warning" ||
    value === "critical"
  );
}

function isSource(value: unknown): value is VisionGuardNotificationSource {
  return (
    value === "live_monitor" ||
    value === "video_upload" ||
    value === "history_48h" ||
    value === "system"
  );
}

function isRoute(value: unknown): value is VisionGuardNotificationRoute {
  return value === "/" || value === "/video-upload" || value === "/history-48h";
}

function normalizeDate(value: unknown): string {
  const text = normalizeText(value);
  const parsed = new Date(text).getTime();
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : new Date().toISOString();
}

function normalizeNotification(value: unknown): VisionGuardNotification | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Partial<VisionGuardNotification>;
  const id = normalizeText(record.id);
  const userId = normalizeText(record.userId);
  const title = normalizeText(record.title);
  const message = normalizeText(record.message);

  if (
    !id ||
    !userId ||
    !title ||
    !message ||
    !isCategory(record.category) ||
    !isSeverity(record.severity) ||
    !isSource(record.source)
  ) {
    return null;
  }

  const notification: VisionGuardNotification = {
    id,
    userId,
    category: record.category,
    severity: record.severity,
    title,
    message,
    createdAt: normalizeDate(record.createdAt),
    source: record.source,
  };

  if (record.readAt) {
    notification.readAt = normalizeDate(record.readAt);
  }

  if (isRoute(record.relatedRoute)) {
    notification.relatedRoute = record.relatedRoute;
  }

  if (notification.id.endsWith("-review-history")) {
    notification.title = "Open driving alert history";
    notification.message =
      "The 48h History page contains Live Monitor alert history for product workflow checks.";
  }

  return notification;
}

function loadNotifications(): VisionGuardNotification[] {
  if (!hasBrowserStorage()) return [];

  try {
    const parsed = JSON.parse(
      window.localStorage.getItem(VISION_GUARD_NOTIFICATIONS_STORAGE_KEY) ?? "[]"
    );
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .map((notification) => normalizeNotification(notification))
      .filter(
        (notification): notification is VisionGuardNotification =>
          Boolean(notification)
      );
  } catch {
    return [];
  }
}

function saveNotifications(notifications: VisionGuardNotification[]): void {
  if (!hasBrowserStorage()) return;
  window.localStorage.setItem(
    VISION_GUARD_NOTIFICATIONS_STORAGE_KEY,
    JSON.stringify(notifications)
  );
}

function sortNotifications(
  notifications: VisionGuardNotification[]
): VisionGuardNotification[] {
  return [...notifications].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );
}

function createSeedNotifications(userId: string, now = new Date()) {
  const reviewDate = new Date(now.getTime() - 5 * 60_000).toISOString();
  return [
    {
      id: `seed-${userId}-system-local-account`,
      userId,
      category: "system",
      severity: "success",
      title: "Local MVP account active",
      message:
        "This browser is using local account state for the VisionGuard app shell.",
      createdAt: now.toISOString(),
      source: "system",
    },
    {
      id: `seed-${userId}-review-history`,
      userId,
      category: "review",
      severity: "info",
      title: "Open driving alert history",
      message:
        "The 48h History page contains Live Monitor alert history for product workflow checks.",
      createdAt: reviewDate,
      source: "history_48h",
      relatedRoute: "/history-48h",
    },
  ] satisfies VisionGuardNotification[];
}

function ensureSeedNotifications(
  notifications: VisionGuardNotification[],
  userId: string
): VisionGuardNotification[] {
  if (notifications.some((notification) => notification.userId === userId)) {
    return notifications;
  }

  return sortNotifications([...createSeedNotifications(userId), ...notifications]);
}

function upsertNotification(
  notifications: VisionGuardNotification[],
  draft: VisionGuardNotificationDraft
): VisionGuardNotification[] {
  const notification: VisionGuardNotification = {
    ...draft,
    createdAt: draft.createdAt ?? new Date().toISOString(),
  };

  if (notifications.some((candidate) => candidate.id === notification.id)) {
    return notifications;
  }

  return sortNotifications([notification, ...notifications]).slice(0, 80);
}

export function notificationFromLiveMonitorDashboardEvent(
  event: LiveMonitorDashboardEvent,
  userId: string
): VisionGuardNotificationDraft | null {
  if (event.kind === "normal") {
    return null;
  }

  const common = {
    id: `live-monitor-${userId}-${event.id}`,
    userId,
    category: "warning_candidate" as const,
    createdAt: event.timestamp,
    source: "live_monitor" as const,
    relatedRoute: "/" as const,
  };

  if (event.kind === "critical_eye_warning") {
    return {
      ...common,
      severity: "critical",
      title: "Critical eye warning candidate",
      message:
        "A stable high-priority eye warning-candidate alert was emitted in Live Monitor.",
    };
  }

  if (event.kind === "eye_warning") {
    return {
      ...common,
      severity: "warning",
      title: "Eye warning candidate",
      message:
        "A stable eye warning-candidate alert was emitted in Live Monitor.",
    };
  }

  if (event.kind === "yawn_warning") {
    return {
      ...common,
      severity: "warning",
      title: "Yawn warning candidate",
      message:
        "A stable yawn warning-candidate alert was emitted in Live Monitor.",
    };
  }

  return {
    ...common,
    category: "system",
    severity: "info",
    title: "Camera signal quality issue",
    message:
      "Live Monitor reported a face, eye, mouth, or camera signal quality issue.",
  };
}

export function VisionGuardNotificationsProvider({
  children,
}: {
  children: ReactNode;
}) {
  const { currentUser } = useVisionGuardAuth();
  const [allNotifications, setAllNotifications] = useState<
    VisionGuardNotification[]
  >([]);

  useEffect(() => {
    const id = window.setTimeout(() => {
      const loadedNotifications = loadNotifications();
      const nextNotifications = currentUser
        ? ensureSeedNotifications(loadedNotifications, currentUser.id)
        : loadedNotifications;

      setAllNotifications(nextNotifications);
      if (currentUser) {
        saveNotifications(nextNotifications);
      }
    }, 0);

    return () => window.clearTimeout(id);
  }, [currentUser]);

  const persist = useCallback((nextNotifications: VisionGuardNotification[]) => {
    setAllNotifications(nextNotifications);
    saveNotifications(nextNotifications);
  }, []);

  const notifications = useMemo(
    () =>
      currentUser
        ? sortNotifications(
            allNotifications.filter(
              (notification) => notification.userId === currentUser.id
            )
          )
        : [],
    [allNotifications, currentUser]
  );

  const unreadCount = useMemo(
    () => notifications.filter((notification) => !notification.readAt).length,
    [notifications]
  );

  const addNotification = useCallback(
    (notification: VisionGuardNotificationDraft) => {
      persist(upsertNotification(allNotifications, notification));
    },
    [allNotifications, persist]
  );

  const markNotificationRead = useCallback(
    (notificationId: string) => {
      if (!currentUser) return;
      const readAt = new Date().toISOString();
      persist(
        allNotifications.map((notification) =>
          notification.userId === currentUser.id &&
          notification.id === notificationId &&
          !notification.readAt
            ? { ...notification, readAt }
            : notification
        )
      );
    },
    [allNotifications, currentUser, persist]
  );

  const markAllRead = useCallback(() => {
    if (!currentUser) return;
    const readAt = new Date().toISOString();
    persist(
      allNotifications.map((notification) =>
        notification.userId === currentUser.id && !notification.readAt
          ? { ...notification, readAt }
          : notification
      )
    );
  }, [allNotifications, currentUser, persist]);

  const clearRead = useCallback(() => {
    if (!currentUser) return;
    persist(
      allNotifications.filter(
        (notification) =>
          notification.userId !== currentUser.id || !notification.readAt
      )
    );
  }, [allNotifications, currentUser, persist]);

  const value = useMemo<VisionGuardNotificationsContextValue>(
    () => ({
      notifications,
      unreadCount,
      addNotification,
      markNotificationRead,
      markAllRead,
      clearRead,
    }),
    [
      addNotification,
      clearRead,
      markAllRead,
      markNotificationRead,
      notifications,
      unreadCount,
    ]
  );

  return (
    <VisionGuardNotificationsContext.Provider value={value}>
      {children}
    </VisionGuardNotificationsContext.Provider>
  );
}

export function useVisionGuardNotifications(): VisionGuardNotificationsContextValue {
  const context = useContext(VisionGuardNotificationsContext);
  if (!context) {
    throw new Error(
      "useVisionGuardNotifications must be used within VisionGuardNotificationsProvider"
    );
  }
  return context;
}
