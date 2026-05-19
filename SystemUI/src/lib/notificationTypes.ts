export type VisionGuardNotificationCategory =
  | "warning_candidate"
  | "system"
  | "review";

export type VisionGuardNotificationSeverity =
  | "info"
  | "success"
  | "warning"
  | "critical";

export type VisionGuardNotificationSource =
  | "live_monitor"
  | "video_upload"
  | "history_48h"
  | "system";

export type VisionGuardNotificationRoute =
  | "/"
  | "/video-upload"
  | "/history-48h";

export interface VisionGuardNotification {
  id: string;
  userId: string;
  category: VisionGuardNotificationCategory;
  severity: VisionGuardNotificationSeverity;
  title: string;
  message: string;
  createdAt: string;
  readAt?: string;
  source: VisionGuardNotificationSource;
  relatedRoute?: VisionGuardNotificationRoute;
}
