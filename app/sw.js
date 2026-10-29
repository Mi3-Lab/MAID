// App shell: cache-first (updated in background). /img/*: cache-first, stored on first fetch. /api/*, /export/*: network only.
const SHELL = 'wv-shell-v5', IMG = 'wv-img-v2';
const ASSETS = ['/', '/index.html', '/app.js', '/style.css', '/manifest.webmanifest', '/icon-192.png', '/icon-512.png', '/adjudicate.html', '/adjudicate.js', '/removed.html', '/removed.js'];
self.addEventListener('install', e => { e.waitUntil(caches.open(SHELL).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => ![SHELL, IMG].includes(k)).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  if (u.pathname.startsWith('/api/') || u.pathname.startsWith('/export/')) return;      // network only
  if (u.pathname.startsWith('/img/')) {
    e.respondWith(caches.open(IMG).then(async c => {
      const hit = await c.match(e.request); if (hit) return hit;
      const r = await fetch(e.request); if (r.ok && !r.redirected) c.put(e.request, r.clone()); return r;
    })); return;
  }
  e.respondWith(caches.open(SHELL).then(async c => {
    const key = e.request.mode === 'navigate' && u.pathname === '/' ? '/index.html' : e.request;
    const hit = await c.match(key, { ignoreSearch: true });
    const net = fetch(e.request).then(r => { if (r.ok && !r.redirected && !u.pathname.startsWith('/unlock')) c.put(key, r.clone()); return r; }).catch(() => null);
    return hit || (await net) || (e.request.mode === 'navigate' ? c.match('/index.html') : Response.error());
  }));
});
