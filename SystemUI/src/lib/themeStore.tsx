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

export type VisionGuardTheme = "day" | "night";

export const VISION_GUARD_THEME_STORAGE_KEY = "visionguard.theme.v1";
const VISION_GUARD_THEME_RUNTIME_STYLE_ID = "visionguard-theme-runtime-style";

type RuntimeThemeStyle = Partial<
  Pick<CSSStyleDeclaration, "backgroundColor" | "borderColor" | "color">
>;

interface RuntimeThemeRule {
  selector: string;
  style: RuntimeThemeStyle;
}

const NIGHT_RUNTIME_RULES: RuntimeThemeRule[] = [
  {
    selector: ".bg-\\[\\#f4f7f9\\]",
    style: { backgroundColor: "#0f172a" },
  },
  { selector: ".bg-white", style: { backgroundColor: "#111827" } },
  {
    selector:
      ".bg-white\\/95, .bg-white\\/92, .bg-white\\/90, .bg-white\\/85, .bg-white\\/80, .bg-white\\/70",
    style: { backgroundColor: "rgb(15 23 42 / 0.92)" },
  },
  {
    selector:
      ".bg-slate-50, .bg-slate-50\\/70, .bg-slate-50\\/60, .bg-slate-50\\/40, .bg-slate-100, .bg-slate-100\\/80",
    style: { backgroundColor: "rgb(30 41 59 / 0.72)" },
  },
  {
    selector: ".bg-blue-50, .bg-blue-50\\/80, .bg-blue-50\\/50, .bg-blue-50\\/30",
    style: { backgroundColor: "rgb(37 99 235 / 0.14)" },
  },
  {
    selector: ".bg-emerald-50",
    style: { backgroundColor: "rgb(16 185 129 / 0.12)" },
  },
  {
    selector: ".bg-amber-50, .bg-orange-50",
    style: { backgroundColor: "rgb(245 158 11 / 0.13)" },
  },
  {
    selector: ".bg-red-50, .bg-rose-50, .bg-rose-100",
    style: { backgroundColor: "rgb(244 63 94 / 0.13)" },
  },
  {
    selector:
      ".border-slate-100, .border-slate-200, .border-slate-200\\/70, .border-slate-200\\/60, .border-slate-200\\/80, .border-slate-300",
    style: { borderColor: "rgb(51 65 85 / 0.88)" },
  },
  {
    selector: ".border-blue-100, .border-blue-200",
    style: { borderColor: "rgb(56 189 248 / 0.28)" },
  },
  {
    selector: ".border-emerald-200",
    style: { borderColor: "rgb(52 211 153 / 0.28)" },
  },
  {
    selector: ".border-amber-200, .border-orange-100",
    style: { borderColor: "rgb(251 191 36 / 0.28)" },
  },
  {
    selector: ".border-red-200, .border-rose-100, .border-rose-200",
    style: { borderColor: "rgb(251 113 133 / 0.3)" },
  },
  {
    selector: ".text-slate-950, .text-slate-900, .text-slate-800",
    style: { color: "#f8fafc" },
  },
  { selector: ".text-slate-700", style: { color: "#dbeafe" } },
  {
    selector: ".text-slate-600, .text-slate-500, .text-slate-400",
    style: { color: "#94a3b8" },
  },
  {
    selector: ".text-blue-700, .text-blue-600",
    style: { color: "#67e8f9" },
  },
  {
    selector: ".text-emerald-700, .text-emerald-600",
    style: { color: "#6ee7b7" },
  },
  {
    selector:
      ".text-amber-700, .text-amber-600, .text-orange-600, .text-orange-500",
    style: { color: "#fbbf24" },
  },
  {
    selector:
      ".text-red-700, .text-red-600, .text-red-500, .text-rose-700, .text-rose-600, .text-rose-500",
    style: { color: "#fb7185" },
  },
];

const DAY_RUNTIME_RULES: RuntimeThemeRule[] = [
  {
    selector: ".bg-\\[\\#f4f7f9\\]",
    style: { backgroundColor: "#f4f7f9" },
  },
  { selector: ".bg-white", style: { backgroundColor: "#ffffff" } },
  {
    selector:
      ".bg-white\\/95, .bg-white\\/92, .bg-white\\/90, .bg-white\\/85, .bg-white\\/80, .bg-white\\/70",
    style: { backgroundColor: "rgb(255 255 255 / 0.92)" },
  },
  {
    selector: ".bg-slate-50, .bg-slate-50\\/70, .bg-slate-50\\/60, .bg-slate-50\\/40",
    style: { backgroundColor: "rgb(248 250 252 / 0.9)" },
  },
  {
    selector: ".bg-slate-100, .bg-slate-100\\/80",
    style: { backgroundColor: "rgb(241 245 249 / 0.9)" },
  },
  {
    selector:
      ".border-slate-100, .border-slate-200, .border-slate-200\\/70, .border-slate-200\\/60, .border-slate-200\\/80, .border-slate-300",
    style: { borderColor: "rgb(226 232 240 / 0.9)" },
  },
  {
    selector: ".text-slate-950",
    style: { color: "#020617" },
  },
  {
    selector: ".text-slate-900",
    style: { color: "#0f172a" },
  },
  {
    selector: ".text-slate-800",
    style: { color: "#1e293b" },
  },
  {
    selector: ".text-slate-700",
    style: { color: "#334155" },
  },
  {
    selector: ".text-slate-600",
    style: { color: "#475569" },
  },
  {
    selector: ".text-slate-500",
    style: { color: "#64748b" },
  },
  {
    selector: ".text-slate-400",
    style: { color: "#94a3b8" },
  },
];

interface VisionGuardThemeContextValue {
  theme: VisionGuardTheme;
  setTheme: (theme: VisionGuardTheme) => void;
  toggleTheme: () => void;
}

const VisionGuardThemeContext =
  createContext<VisionGuardThemeContextValue | null>(null);

function hasBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function normalizeTheme(value: unknown): VisionGuardTheme {
  return value === "night" ? "night" : "day";
}

function loadTheme(): VisionGuardTheme {
  if (!hasBrowserStorage()) return "day";
  return normalizeTheme(
    window.localStorage.getItem(VISION_GUARD_THEME_STORAGE_KEY)
  );
}

function saveTheme(theme: VisionGuardTheme): void {
  if (!hasBrowserStorage()) return;
  window.localStorage.setItem(VISION_GUARD_THEME_STORAGE_KEY, theme);
}

function applyInlineStyle(
  element: Element,
  style: RuntimeThemeStyle | null
): void {
  if (!(element instanceof HTMLElement)) return;

  if (!style) {
    element.style.removeProperty("background-color");
    element.style.removeProperty("border-color");
    element.style.removeProperty("color");
    return;
  }

  if (style.backgroundColor) {
    element.style.backgroundColor = style.backgroundColor;
  }
  if (style.borderColor) {
    element.style.borderColor = style.borderColor;
  }
  if (style.color) {
    element.style.color = style.color;
  }
}

function applyRuntimeElementTheme(theme: VisionGuardTheme): void {
  if (typeof document === "undefined") return;
  const root = document.querySelector(".vg-themed");
  if (!root) return;

  const themedElements = new Set<Element>();
  for (const rule of NIGHT_RUNTIME_RULES) {
    root.querySelectorAll(rule.selector).forEach((element) => {
      themedElements.add(element);
    });
  }

  themedElements.forEach((element) => applyInlineStyle(element, null));

  const activeRules = theme === "night" ? NIGHT_RUNTIME_RULES : DAY_RUNTIME_RULES;
  for (const rule of activeRules) {
    root.querySelectorAll(rule.selector).forEach((element) => {
      applyInlineStyle(element, rule.style);
    });
  }
}

function observeRuntimeElementTheme(theme: VisionGuardTheme): () => void {
  if (typeof document === "undefined") return () => undefined;
  const root = document.querySelector(".vg-themed");
  if (!root) return () => undefined;

  let animationFrame = window.requestAnimationFrame(() =>
    applyRuntimeElementTheme(theme)
  );
  const observer = new MutationObserver(() => {
    window.cancelAnimationFrame(animationFrame);
    animationFrame = window.requestAnimationFrame(() =>
      applyRuntimeElementTheme(theme)
    );
  });

  observer.observe(root, {
    attributes: true,
    attributeFilter: ["class"],
    childList: true,
    subtree: true,
  });

  return () => {
    window.cancelAnimationFrame(animationFrame);
    observer.disconnect();
  };
}

function ensureRuntimeThemeStyles(): void {
  if (typeof document === "undefined") return;
  const existingStyle = document.getElementById(
    VISION_GUARD_THEME_RUNTIME_STYLE_ID
  );
  if (existingStyle) {
    document.head.appendChild(existingStyle);
    return;
  }

  const style = document.createElement("style");
  style.id = VISION_GUARD_THEME_RUNTIME_STYLE_ID;
  style.textContent = `
    .theme-night .bg-\\[\\#f4f7f9\\] { background-color: #0f172a !important; }
    .theme-night .bg-white { background-color: #111827 !important; }
    .theme-night .bg-white\\/95,
    .theme-night .bg-white\\/92,
    .theme-night .bg-white\\/90,
    .theme-night .bg-white\\/85,
    .theme-night .bg-white\\/80,
    .theme-night .bg-white\\/70 { background-color: rgb(15 23 42 / 0.92) !important; }
    .theme-night .bg-slate-50,
    .theme-night .bg-slate-50\\/70,
    .theme-night .bg-slate-50\\/60,
    .theme-night .bg-slate-50\\/40,
    .theme-night .bg-slate-100,
    .theme-night .bg-slate-100\\/80 { background-color: rgb(30 41 59 / 0.72) !important; }
    .theme-night .bg-blue-50,
    .theme-night .bg-blue-50\\/80,
    .theme-night .bg-blue-50\\/50,
    .theme-night .bg-blue-50\\/30 { background-color: rgb(37 99 235 / 0.14) !important; }
    .theme-night .bg-emerald-50 { background-color: rgb(16 185 129 / 0.12) !important; }
    .theme-night .bg-amber-50,
    .theme-night .bg-orange-50 { background-color: rgb(245 158 11 / 0.13) !important; }
    .theme-night .bg-red-50,
    .theme-night .bg-rose-50,
    .theme-night .bg-rose-100 { background-color: rgb(244 63 94 / 0.13) !important; }
    .theme-night .border-slate-100,
    .theme-night .border-slate-200,
    .theme-night .border-slate-200\\/70,
    .theme-night .border-slate-200\\/60,
    .theme-night .border-slate-200\\/80,
    .theme-night .border-slate-300 { border-color: rgb(51 65 85 / 0.88) !important; }
    .theme-night .text-slate-950,
    .theme-night .text-slate-900,
    .theme-night .text-slate-800 { color: #f8fafc !important; }
    .theme-night .text-slate-700 { color: #dbeafe !important; }
    .theme-night .text-slate-600,
    .theme-night .text-slate-500,
    .theme-night .text-slate-400 { color: #94a3b8 !important; }
    .theme-day .bg-white { background-color: #ffffff !important; }
    .theme-day .bg-slate-50,
    .theme-day .bg-slate-50\\/70,
    .theme-day .bg-slate-50\\/60,
    .theme-day .bg-slate-50\\/40 { background-color: rgb(248 250 252 / 0.9) !important; }
    .theme-day .bg-slate-100,
    .theme-day .bg-slate-100\\/80 { background-color: rgb(241 245 249 / 0.9) !important; }
  `;
  document.head.appendChild(style);
}

export function VisionGuardThemeProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [theme, setThemeState] = useState<VisionGuardTheme>("day");
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const id = window.setTimeout(() => {
      ensureRuntimeThemeStyles();
      setThemeState(loadTheme());
      setIsReady(true);
    }, 0);

    return () => window.clearTimeout(id);
  }, []);

  useEffect(() => {
    if (!isReady) return;
    ensureRuntimeThemeStyles();
    const cleanupRuntimeTheme = observeRuntimeElementTheme(theme);
    saveTheme(theme);
    document.documentElement.dataset.theme = theme;
    document.documentElement.classList.toggle("dark", theme === "night");

    return cleanupRuntimeTheme;
  }, [isReady, theme]);

  const setTheme = useCallback((nextTheme: VisionGuardTheme) => {
    setThemeState(normalizeTheme(nextTheme));
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState((current) => (current === "day" ? "night" : "day"));
  }, []);

  const value = useMemo<VisionGuardThemeContextValue>(
    () => ({
      theme,
      setTheme,
      toggleTheme,
    }),
    [setTheme, theme, toggleTheme]
  );

  return (
    <VisionGuardThemeContext.Provider value={value}>
      {children}
    </VisionGuardThemeContext.Provider>
  );
}

export function useVisionGuardTheme(): VisionGuardThemeContextValue {
  const context = useContext(VisionGuardThemeContext);
  if (!context) {
    throw new Error(
      "useVisionGuardTheme must be used within VisionGuardThemeProvider"
    );
  }
  return context;
}
