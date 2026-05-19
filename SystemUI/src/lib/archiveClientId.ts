const ARCHIVE_CLIENT_ID_KEY = "visionguard.archiveClientId.v1";

function createArchiveClientId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `client-${crypto.randomUUID()}`;
  }
  return `client-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 10)}`;
}

export function getArchiveClientId(): string {
  if (typeof window === "undefined") {
    return "server-render";
  }

  const existing = window.localStorage.getItem(ARCHIVE_CLIENT_ID_KEY);
  if (existing) {
    return existing;
  }

  const nextId = createArchiveClientId();
  window.localStorage.setItem(ARCHIVE_CLIENT_ID_KEY, nextId);
  return nextId;
}
