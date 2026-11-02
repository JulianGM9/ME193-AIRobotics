// Thin wrappers around the backend API (server/app.py).

async function request(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}

export const api = {
  sessions: () => request("/api/sessions"),
  session: (id) => request(`/api/sessions/${id}`),
  saveLabels: (id, strides, segmentOk) =>
    request(`/api/sessions/${id}/labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strides, segment_ok: segmentOk }),
    }),
};
