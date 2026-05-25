"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useSyncExternalStore,
  type ReactNode,
} from "react";

export const VISION_GUARD_SETTINGS_STORAGE_KEY = "visionguard.settings.v1";

export interface VisionGuardSettings {
  liveMonitor: {
    minimalMode: boolean;
  };
}

interface VisionGuardSettingsContextValue {
  settings: VisionGuardSettings;
  setMinimalLiveMonitorMode: (minimalMode: boolean) => void;
}

const DEFAULT_SETTINGS: VisionGuardSettings = {
  liveMonitor: {
    minimalMode: false,
  },
};

const VisionGuardSettingsContext =
  createContext<VisionGuardSettingsContextValue | null>(null);
const settingsSubscribers = new Set<() => void>();
let currentSettings = DEFAULT_SETTINGS;
let browserSettingsLoaded = false;

function hasBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function normalizeSettings(value: unknown): VisionGuardSettings {
  if (!value || typeof value !== "object") {
    return DEFAULT_SETTINGS;
  }

  const settings = value as Partial<VisionGuardSettings>;

  return {
    liveMonitor: {
      minimalMode: settings.liveMonitor?.minimalMode === true,
    },
  };
}

function loadSettings(): VisionGuardSettings {
  if (!hasBrowserStorage()) {
    return DEFAULT_SETTINGS;
  }

  try {
    return normalizeSettings(
      JSON.parse(
        window.localStorage.getItem(VISION_GUARD_SETTINGS_STORAGE_KEY) ?? "null"
      )
    );
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(settings: VisionGuardSettings): void {
  if (!hasBrowserStorage()) {
    return;
  }

  window.localStorage.setItem(
    VISION_GUARD_SETTINGS_STORAGE_KEY,
    JSON.stringify(settings)
  );
}

function emitSettingsChange(): void {
  settingsSubscribers.forEach((listener) => listener());
}

function getSettingsSnapshot(): VisionGuardSettings {
  if (hasBrowserStorage() && !browserSettingsLoaded) {
    currentSettings = loadSettings();
    browserSettingsLoaded = true;
  }

  return currentSettings;
}

function getServerSettingsSnapshot(): VisionGuardSettings {
  return DEFAULT_SETTINGS;
}

function subscribeToSettings(listener: () => void): () => void {
  settingsSubscribers.add(listener);

  function handleStorage(event: StorageEvent) {
    if (event.key !== VISION_GUARD_SETTINGS_STORAGE_KEY) {
      return;
    }

    currentSettings = loadSettings();
    browserSettingsLoaded = true;
    emitSettingsChange();
  }

  if (typeof window !== "undefined") {
    window.addEventListener("storage", handleStorage);
  }

  return () => {
    settingsSubscribers.delete(listener);
    if (typeof window !== "undefined") {
      window.removeEventListener("storage", handleStorage);
    }
  };
}

export function VisionGuardSettingsProvider({
  children,
}: {
  children: ReactNode;
}) {
  const settings = useSyncExternalStore(
    subscribeToSettings,
    getSettingsSnapshot,
    getServerSettingsSnapshot
  );

  const setMinimalLiveMonitorMode = useCallback((minimalMode: boolean) => {
    const settings = getSettingsSnapshot();
    const nextSettings = normalizeSettings({
      ...settings,
      liveMonitor: {
        ...settings.liveMonitor,
        minimalMode,
      },
    });

    currentSettings = nextSettings;
    browserSettingsLoaded = true;
    saveSettings(nextSettings);
    emitSettingsChange();
  }, []);

  const value = useMemo<VisionGuardSettingsContextValue>(
    () => ({
      settings,
      setMinimalLiveMonitorMode,
    }),
    [setMinimalLiveMonitorMode, settings]
  );

  return (
    <VisionGuardSettingsContext.Provider value={value}>
      {children}
    </VisionGuardSettingsContext.Provider>
  );
}

export function useVisionGuardSettings(): VisionGuardSettingsContextValue {
  const context = useContext(VisionGuardSettingsContext);
  if (!context) {
    throw new Error(
      "useVisionGuardSettings must be used within VisionGuardSettingsProvider"
    );
  }
  return context;
}
