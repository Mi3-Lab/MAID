// App shell: network-first (so schema/UI deploys are not stuck behind an old cache). /img/*: cache-first. /api/*, /export/*: network only.
const SHELL = 'wv-shell-v9', IMG = 'wv-img-v2';
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
  // network-first for HTML/JS/CSS (do not ignore query strings: a cache-busted /app.js?v= must not hit a stale /app.js)
  e.respondWith((async () => {
    const c = await caches.open(SHELL);
    const key = e.request.mode === 'navigate' && u.pathname === '/' ? '/index.html' : e.request;
    try {
      const r = await fetch(e.request);
      if (r.ok && !r.redirected && !u.pathname.startsWith('/unlock')) c.put(key, r.clone());
      return r;
    } catch (err) {
      const hit = await c.match(key);
      if (hit) return hit;
      if (e.request.mode === 'navigate') return (await c.match('/index.html')) || Response.error();
      return Response.error();
    }
  })());
});
