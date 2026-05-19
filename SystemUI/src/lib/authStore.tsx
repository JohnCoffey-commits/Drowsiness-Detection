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
import type {
  LoginVisionGuardUserInput,
  VisionGuardAuthState,
  VisionGuardUser,
  VisionGuardUserRole,
} from "@/lib/authTypes";

export const VISION_GUARD_AUTH_STORAGE_KEY = "visionguard.auth.v1";
export const VISION_GUARD_LOCAL_ACCOUNT_USERNAME = "John_Coffey";

const VISION_GUARD_LOCAL_ACCOUNT_PASSWORD = "25591280";
const VISION_GUARD_LOCAL_ACCOUNT_USER: VisionGuardUser = {
  id: "local-user-john-coffey",
  username: VISION_GUARD_LOCAL_ACCOUNT_USERNAME,
  displayName: "John_Coffey",
  email: "john.coffey@visionguard.local",
  role: "driver",
  createdAt: "2026-05-17T00:00:00.000Z",
};

const EMPTY_AUTH_STATE: VisionGuardAuthState = {
  currentUser: null,
  users: [VISION_GUARD_LOCAL_ACCOUNT_USER],
};

interface VisionGuardAuthContextValue {
  authState: VisionGuardAuthState;
  currentUser: VisionGuardUser | null;
  users: VisionGuardUser[];
  isReady: boolean;
  loginWithPassword: (input: LoginVisionGuardUserInput) => boolean;
  logout: () => void;
  isLegacyRecordVisible: (recordUserId?: string) => boolean;
}

const VisionGuardAuthContext =
  createContext<VisionGuardAuthContextValue | null>(null);

function hasBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function normalizeText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeEmail(value: unknown): string {
  return normalizeText(value).toLowerCase();
}

function normalizeUsername(value: unknown): string {
  return normalizeText(value);
}

function normalizeRole(value: unknown): VisionGuardUserRole {
  if (value === "reviewer" || value === "admin" || value === "driver") {
    return value;
  }
  return "driver";
}

function normalizeUser(value: unknown): VisionGuardUser | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Partial<VisionGuardUser>;
  const id = normalizeText(record.id);
  const username = normalizeUsername(record.username);
  const email = normalizeEmail(record.email);
  const displayName = normalizeText(record.displayName);
  const createdAt = normalizeText(record.createdAt);

  if (!id || !username || !email || !displayName) {
    return null;
  }

  return {
    id,
    username,
    displayName,
    email,
    role: normalizeRole(record.role),
    createdAt: createdAt || new Date().toISOString(),
  };
}

function isConfiguredLocalUser(user: VisionGuardUser | null): boolean {
  return Boolean(
    user &&
      user.id === VISION_GUARD_LOCAL_ACCOUNT_USER.id &&
      user.username === VISION_GUARD_LOCAL_ACCOUNT_USERNAME
  );
}

function normalizeAuthState(value: unknown): VisionGuardAuthState {
  if (!value || typeof value !== "object") {
    return EMPTY_AUTH_STATE;
  }

  const record = value as Partial<VisionGuardAuthState>;
  const storedUsers = Array.isArray(record.users)
    ? record.users
        .map((user) => normalizeUser(user))
        .filter((user): user is VisionGuardUser => isConfiguredLocalUser(user))
    : [];
  const users = storedUsers.length > 0 ? [storedUsers[0]] : [VISION_GUARD_LOCAL_ACCOUNT_USER];

  const activeSessionUserId =
    record.activeSession && typeof record.activeSession === "object"
      ? normalizeText(record.activeSession.userId)
      : "";
  const storedCurrentUser = normalizeUser(record.currentUser);
  const hasAllowedActiveSession =
    activeSessionUserId === VISION_GUARD_LOCAL_ACCOUNT_USER.id;
  const hasAllowedCurrentUser = isConfiguredLocalUser(storedCurrentUser);
  const currentUser =
    hasAllowedActiveSession || hasAllowedCurrentUser ? users[0] : null;

  return {
    currentUser,
    users,
    activeSession: currentUser
      ? {
          userId: currentUser.id,
          startedAt:
            record.activeSession &&
            typeof record.activeSession === "object" &&
            typeof record.activeSession.startedAt === "string"
              ? record.activeSession.startedAt
              : new Date().toISOString(),
        }
      : undefined,
  };
}

function loadAuthState(): VisionGuardAuthState {
  if (!hasBrowserStorage()) {
    return EMPTY_AUTH_STATE;
  }

  try {
    return normalizeAuthState(
      JSON.parse(window.localStorage.getItem(VISION_GUARD_AUTH_STORAGE_KEY) ?? "null")
    );
  } catch {
    return EMPTY_AUTH_STATE;
  }
}

function saveAuthState(state: VisionGuardAuthState): void {
  if (!hasBrowserStorage()) return;
  window.localStorage.setItem(
    VISION_GUARD_AUTH_STORAGE_KEY,
    JSON.stringify(state)
  );
}

export function getLegacyLiveMonitorOwnerUserId(
  state: VisionGuardAuthState
): string | null {
  return state.users[0]?.id ?? null;
}

export function VisionGuardAuthProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [authState, setAuthState] =
    useState<VisionGuardAuthState>(EMPTY_AUTH_STATE);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const id = window.setTimeout(() => {
      setAuthState(loadAuthState());
      setIsReady(true);
    }, 0);

    return () => window.clearTimeout(id);
  }, []);

  const loginWithPassword = useCallback((input: LoginVisionGuardUserInput) => {
    const username = normalizeUsername(input.username);
    const password = normalizeText(input.password);
    const isAllowed =
      username === VISION_GUARD_LOCAL_ACCOUNT_USERNAME &&
      password === VISION_GUARD_LOCAL_ACCOUNT_PASSWORD;

    if (!isAllowed) {
      return false;
    }

    setAuthState(() => {
      const nextState: VisionGuardAuthState = {
        currentUser: VISION_GUARD_LOCAL_ACCOUNT_USER,
        users: [VISION_GUARD_LOCAL_ACCOUNT_USER],
        activeSession: {
          userId: VISION_GUARD_LOCAL_ACCOUNT_USER.id,
          startedAt: new Date().toISOString(),
        },
      };
      saveAuthState(nextState);
      return nextState;
    });

    return true;
  }, []);

  const logout = useCallback(() => {
    setAuthState((currentState) => {
      const nextState: VisionGuardAuthState = {
        ...currentState,
        currentUser: null,
        users: [VISION_GUARD_LOCAL_ACCOUNT_USER],
        activeSession: undefined,
      };
      saveAuthState(nextState);
      return nextState;
    });
  }, []);

  const isLegacyRecordVisible = useCallback(
    (recordUserId?: string) => {
      if (recordUserId) {
        return recordUserId === authState.currentUser?.id;
      }

      const legacyOwnerUserId = getLegacyLiveMonitorOwnerUserId(authState);
      return Boolean(
        authState.currentUser && legacyOwnerUserId === authState.currentUser.id
      );
    },
    [authState]
  );

  const value = useMemo<VisionGuardAuthContextValue>(
    () => ({
      authState,
      currentUser: authState.currentUser,
      users: authState.users,
      isReady,
      loginWithPassword,
      logout,
      isLegacyRecordVisible,
    }),
    [authState, isReady, isLegacyRecordVisible, loginWithPassword, logout]
  );

  return (
    <VisionGuardAuthContext.Provider value={value}>
      {children}
    </VisionGuardAuthContext.Provider>
  );
}

export function useVisionGuardAuth(): VisionGuardAuthContextValue {
  const context = useContext(VisionGuardAuthContext);
  if (!context) {
    throw new Error("useVisionGuardAuth must be used within VisionGuardAuthProvider");
  }
  return context;
}
