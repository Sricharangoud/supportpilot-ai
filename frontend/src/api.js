const API_BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  chat: (message) =>
    request("/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    }),
  ingest: () => request("/ingest", { method: "POST" }),
  tickets: () => request("/admin/tickets"),
  conversations: () => request("/admin/conversations"),
  metrics: () => request("/admin/metrics"),
};

