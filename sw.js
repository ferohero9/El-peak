/*
 * El-Peak service worker - retirement / cleanup version.
 *
 * The frontend no longer registers a service worker. Older deployments
 * registered a cache-first worker that kept serving stale or corrupted
 * cached copies of the app shell (symptom: page shows only the aurora
 * background). Browsers re-fetch this file for every registered worker,
 * so this version exists solely to tear that state down: delete every
 * cache, unregister this worker, and pass requests straight to the
 * network while it is still controlling a page.
 */
self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map((key) => caches.delete(key)));
      await self.registration.unregister();
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const path = new URL(event.request.url).pathname;
  if (path.startsWith("/api/")) return;
  event.respondWith(fetch(event.request));
});
