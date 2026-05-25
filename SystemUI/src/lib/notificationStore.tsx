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
  readAt?: string;
};

interface VisionGuardNotificationsContextValue {
  notifications: VisionGuardNotification[];
  unreadCount: number;
  addNotification: (notification: VisionGuardNotificationDraft) => void;
  upsertNotificationByDedupeKey: (
    notification: VisionGuardNotificationDraft
  ) => void;
  upsertDrivingDigestNotification: (
    event: LiveMonitorDashboardEvent
  ) => void;
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
    value === "driving" ||
    value === "uploads" ||
    value === "warning_candidate" ||
    value === "system" ||
    value === "review"
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

function normalizeMetadata(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }

  return value as Record<string, unknown>;
}

function localDateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function metadataStringArray(
  metadata: Record<string, unknown> | undefined,
  key: string
): string[] {
  const value = metadata?.[key];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function metadataNumber(
  metadata: Record<string, unknown> | undefined,
  key: string
): number {
  const value = metadata?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function metadataBoolean(
  metadata: Record<string, unknown> | undefined,
  key: string
): boolean {
  return metadata?.[key] === true;
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

  const dedupeKey = normalizeText(record.dedupeKey);
  if (dedupeKey) {
    notification.dedupeKey = dedupeKey;
  }

  const metadata = normalizeMetadata(record.metadata);
  if (metadata) {
    notification.metadata = metadata;
  }

  if (isRoute(record.relatedRoute)) {
    notification.relatedRoute = record.relatedRoute;
  }

  if (notification.id.endsWith("-review-history")) {
    notification.title = "Open History";
    notification.message =
      "The History page contains Live Monitor alert history for product workflow checks.";
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
      category: "system",
      severity: "info",
      title: "Open History",
      message:
        "The History page contains Live Monitor alert history for product workflow checks.",
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

function notificationFromDraft(
  draft: VisionGuardNotificationDraft
): VisionGuardNotification {
  const notification: VisionGuardNotification = {
    ...draft,
    createdAt: draft.createdAt ?? new Date().toISOString(),
  };

  if (draft.readAt) {
    notification.readAt = normalizeDate(draft.readAt);
  }

  return notification;
}

function appendNotification(
  notifications: VisionGuardNotification[],
  draft: VisionGuardNotificationDraft
): VisionGuardNotification[] {
  const notification = notificationFromDraft(draft);

  if (
    notifications.some(
      (candidate) =>
        candidate.userId === notification.userId &&
        candidate.id === notification.id
    )
  ) {
    return notifications;
  }

  return sortNotifications([notification, ...notifications]).slice(0, 80);
}

function upsertNotification(
  notifications: VisionGuardNotification[],
  draft: VisionGuardNotificationDraft
): VisionGuardNotification[] {
  const notification = notificationFromDraft(draft);
  const matchIndex = notifications.findIndex(
    (candidate) =>
      candidate.userId === notification.userId &&
      ((notification.dedupeKey &&
        candidate.dedupeKey === notification.dedupeKey) ||
        candidate.id === notification.id)
  );

  if (matchIndex === -1) {
    return sortNotifications([notification, ...notifications]).slice(0, 80);
  }

  const existing = notifications[matchIndex];
  const next = [...notifications];
  next[matchIndex] = {
    ...existing,
    ...notification,
    id: existing.id,
    readAt: existing.readAt ?? notification.readAt,
  };

  return sortNotifications(next).slice(0, 80);
}

function drivingDigestPattern(metadata: Record<string, unknown>): string {
  const highRiskEyeAlerts = metadataNumber(metadata, "highRiskEyeAlerts");
  const yawnAlerts = metadataNumber(metadata, "yawnAlerts");
  const signalInterruptions = metadataNumber(metadata, "signalInterruptions");

  if (
    highRiskEyeAlerts > 0 &&
    highRiskEyeAlerts >= yawnAlerts &&
    highRiskEyeAlerts >= signalInterruptions
  ) {
    return "High-risk eye alerts were the dominant pattern.";
  }

  if (yawnAlerts > 0 && yawnAlerts >= signalInterruptions) {
    return "Yawn alerts were the dominant pattern.";
  }

  if (signalInterruptions > 0) {
    return "Signal interruptions were the dominant pattern.";
  }

  return "No dominant alert pattern was detected.";
}

function drivingDigestMessage(metadata: Record<string, unknown>): string {
  const totalAlerts = metadataNumber(metadata, "totalAlerts");
  const driveCount = metadataStringArray(metadata, "driveSessionIds").length;
  return `Today's Live Monitor history recorded ${pluralize(
    totalAlerts,
    "alert"
  )} across ${pluralize(driveCount, "drive")}. ${drivingDigestPattern(metadata)}`;
}

function upsertDrivingDigest(
  notifications: VisionGuardNotification[],
  event: LiveMonitorDashboardEvent,
  userId: string
): VisionGuardNotification[] {
  if (event.kind === "normal") {
    return notifications;
  }

  const eventDate = new Date(event.timestamp);
  if (!Number.isFinite(eventDate.getTime())) {
    return notifications;
  }

  const dateKey = localDateKey(eventDate);
  const dedupeKey = `driving-digest:${userId}:${dateKey}`;
  const existing = notifications.find(
    (notification) =>
      notification.userId === userId && notification.dedupeKey === dedupeKey
  );
  const existingMetadata = existing?.metadata;
  const eventIds = new Set(metadataStringArray(existingMetadata, "eventIds"));

  if (eventIds.has(event.id)) {
    return notifications;
  }

  eventIds.add(event.id);
  const driveSessionIds = new Set(
    metadataStringArray(existingMetadata, "driveSessionIds")
  );
  if (event.sessionId) {
    driveSessionIds.add(event.sessionId);
  }

  const nextMetadata: Record<string, unknown> = {
    notificationKind: "driving_digest",
    eventIds: Array.from(eventIds),
    driveSessionIds: Array.from(driveSessionIds),
    dateKey,
    totalAlerts: metadataNumber(existingMetadata, "totalAlerts") + 1,
    highRiskEyeAlerts:
      metadataNumber(existingMetadata, "highRiskEyeAlerts") +
      (event.kind === "eye_warning" || event.kind === "critical_eye_warning"
        ? 1
        : 0),
    yawnAlerts:
      metadataNumber(existingMetadata, "yawnAlerts") +
      (event.kind === "yawn_warning" ? 1 : 0),
    signalInterruptions:
      metadataNumber(existingMetadata, "signalInterruptions") +
      (event.kind === "signal_quality" ? 1 : 0),
  };

  nextMetadata.requiresAttention =
    metadataNumber(nextMetadata, "highRiskEyeAlerts") > 0 ||
    metadataNumber(nextMetadata, "signalInterruptions") > 0;

  return upsertNotification(notifications, {
    id: existing?.id ?? `driving-digest-${userId}-${dateKey}`,
    userId,
    category: "driving",
    severity: metadataBoolean(nextMetadata, "requiresAttention")
      ? "warning"
      : "info",
    title: "Driving alerts summary",
    message: drivingDigestMessage(nextMetadata),
    createdAt: event.timestamp,
    source: "live_monitor",
    relatedRoute: "/",
    dedupeKey,
    metadata: nextMetadata,
  });
}

export type ProductNotificationCategory = "driving" | "uploads" | "system";

export function getNotificationProductCategory(
  notification: VisionGuardNotification
): ProductNotificationCategory {
  if (
    notification.category === "driving" ||
    notification.category === "warning_candidate" ||
    notification.source === "live_monitor"
  ) {
    return "driving";
  }

  if (
    notification.category === "uploads" ||
    notification.source === "video_upload" ||
    (notification.category === "review" &&
      notification.relatedRoute === "/video-upload")
  ) {
    return "uploads";
  }

  return "system";
}

function requiresNotificationAttention(
  notification: VisionGuardNotification
): boolean {
  if (notification.readAt) {
    return false;
  }

  if (metadataBoolean(notification.metadata, "requiresAttention")) {
    return true;
  }

  const productCategory = getNotificationProductCategory(notification);

  if (productCategory === "uploads") {
    return notification.severity === "warning" || notification.severity === "critical";
  }

  if (productCategory === "system") {
    return notification.severity === "warning" || notification.severity === "critical";
  }

  return notification.metadata?.notificationKind === "driving_digest"
    ? metadataBoolean(notification.metadata, "requiresAttention")
    : false;
}

export function notificationFromLiveMonitorDashboardEvent(
  event: LiveMonitorDashboardEvent,
  userId: string
): VisionGuardNotificationDraft | null {
  void event;
  void userId;
  return null;
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

  const updatePersistedNotifications = useCallback(
    (
      updater: (
        currentNotifications: VisionGuardNotification[]
      ) => VisionGuardNotification[]
    ) => {
      setAllNotifications((currentNotifications) => {
        const nextNotifications = updater(currentNotifications);
        saveNotifications(nextNotifications);
        return nextNotifications;
      });
    },
    []
  );

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
    () => notifications.filter(requiresNotificationAttention).length,
    [notifications]
  );

  const addNotification = useCallback(
    (notification: VisionGuardNotificationDraft) => {
      updatePersistedNotifications((currentNotifications) =>
        notification.dedupeKey
          ? upsertNotification(currentNotifications, notification)
          : appendNotification(currentNotifications, notification)
      );
    },
    [updatePersistedNotifications]
  );

  const upsertNotificationByDedupeKey = useCallback(
    (notification: VisionGuardNotificationDraft) => {
      updatePersistedNotifications((currentNotifications) =>
        upsertNotification(currentNotifications, notification)
      );
    },
    [updatePersistedNotifications]
  );

  const upsertDrivingDigestNotification = useCallback(
    (event: LiveMonitorDashboardEvent) => {
      if (!currentUser) return;
      updatePersistedNotifications((currentNotifications) =>
        upsertDrivingDigest(currentNotifications, event, currentUser.id)
      );
    },
    [currentUser, updatePersistedNotifications]
  );

  const markNotificationRead = useCallback(
    (notificationId: string) => {
      if (!currentUser) return;
      const readAt = new Date().toISOString();
      updatePersistedNotifications((currentNotifications) =>
        currentNotifications.map((notification) =>
          notification.userId === currentUser.id &&
          notification.id === notificationId &&
          !notification.readAt
            ? { ...notification, readAt }
            : notification
        )
      );
    },
    [currentUser, updatePersistedNotifications]
  );

  const markAllRead = useCallback(() => {
    if (!currentUser) return;
    const readAt = new Date().toISOString();
    updatePersistedNotifications((currentNotifications) =>
      currentNotifications.map((notification) =>
        notification.userId === currentUser.id && !notification.readAt
          ? { ...notification, readAt }
          : notification
      )
    );
  }, [currentUser, updatePersistedNotifications]);

  const clearRead = useCallback(() => {
    if (!currentUser) return;
    updatePersistedNotifications((currentNotifications) =>
      currentNotifications.filter(
        (notification) =>
          notification.userId !== currentUser.id || !notification.readAt
      )
    );
  }, [currentUser, updatePersistedNotifications]);

  const value = useMemo<VisionGuardNotificationsContextValue>(
    () => ({
      notifications,
      unreadCount,
      addNotification,
      upsertNotificationByDedupeKey,
      upsertDrivingDigestNotification,
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
      upsertDrivingDigestNotification,
      upsertNotificationByDedupeKey,
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
