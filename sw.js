const CACHE_NAME = "el-peak-v1";
const STATIC_ASSETS = [
  "/",
  "/manifest.json"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls.
  if (url.pathname.startsWith("/api/")) {
    return;
  }

  // Never cache non-GET requests.
  if (event.request.method !== "GET") {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) {
        return cached;
      }

      return fetch(event.request)
        .then((response) => {
          const copy = response.clone();

          if (response.ok && response.type === "basic") {
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, copy);
            });
          }

          return response;
        })
        .catch(async () => {
          if (event.request.mode === "navigate") {
            const fallback = await caches.match("/");
            if (fallback) return fallback;
          }

          return new Response("Offline", {
            status: 503,
            headers: { "Content-Type": "text/plain" }
          });
        });
    })
  );
});
