export type VisionGuardUserRole = "driver" | "reviewer" | "admin";

export interface VisionGuardUser {
  id: string;
  username: string;
  displayName: string;
  email: string;
  role: VisionGuardUserRole;
  createdAt: string;
}

export interface VisionGuardAuthState {
  currentUser: VisionGuardUser | null;
  users: VisionGuardUser[];
  activeSession?: {
    userId: string;
    startedAt: string;
  };
}

export interface LoginVisionGuardUserInput {
  username: string;
  password: string;
}

export const VISION_GUARD_USER_ROLES: VisionGuardUserRole[] = [
  "driver",
  "reviewer",
  "admin",
];

export function formatVisionGuardRole(role: VisionGuardUserRole): string {
  if (role === "admin") return "Admin";
  if (role === "reviewer") return "Reviewer";
  return "Driver";
}
