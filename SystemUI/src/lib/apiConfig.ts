export const LOCAL_API_BASE_URL = "http://127.0.0.1:8000";

export function normalizeApiBaseUrl(value?: string | null): string {
  const trimmed = value?.trim() || LOCAL_API_BASE_URL;
  return trimmed.replace(/\/+$/, "");
}

export function getApiBaseUrl(): string {
  return normalizeApiBaseUrl(
    process.env.NEXT_PUBLIC_API_BASE_URL || LOCAL_API_BASE_URL,
  );
}

export function validateApiBaseUrl(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return "Backend URL is required.";
  if (trimmed.startsWith("file://")) return "file:// URLs are not allowed.";
  if (!/^https?:\/\//i.test(trimmed)) {
    return "Backend URL must start with http:// or https://.";
  }
  try {
    const url = new URL(trimmed);
    if (!["http:", "https:"].includes(url.protocol)) {
      return "Backend URL must use http:// or https://.";
    }
    return null;
  } catch {
    return "Backend URL is not a valid URL.";
  }
}

export function buildApiUrl(path: string, apiBaseUrl = getApiBaseUrl()): string {
  const base = normalizeApiBaseUrl(apiBaseUrl);
  const safePath = path.startsWith("/") ? path : `/${path}`;
  return `${base}${safePath}`;
}

export function buildApiUrlWithBase(apiBaseUrl: string, path: string): string {
  return buildApiUrl(path, apiBaseUrl);
}
