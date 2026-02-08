// Service worker to reduce "stale ESM module" issues when running via simple static servers.
// It forces fresh fetches (no-store) for project modules under /src, /test, and /web.

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const p = url.pathname;
  if (!(p.startsWith("/src/") || p.startsWith("/test/") || p.startsWith("/web/"))) return;

  event.respondWith(fetch(req, { cache: "no-store" }));
});

