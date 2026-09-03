/* Raqam service worker — offline-first app shell.
 * Shell (HTML/CSS/JS/model/icons) is cache-first so the app opens with no signal.
 * /api/* is network-first; scanning + queue + review all work offline in the page. */
const CACHE = 'raqam-v10';
const SHELL = [
  '/', '/static/app.css', '/static/app.js', '/static/recognize.js',
  '/static/numerals_cnn.json', '/static/icon-192.png', '/static/icon-512.png',
  '/manifest.webmanifest', '/api/sample-form',
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE)
    .then(c => Promise.allSettled(SHELL.map(u => c.add(u))))  // tolerate a missing asset
    .then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const { request } = e;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== location.origin) return;

  if (url.pathname.startsWith('/api/') && url.pathname !== '/api/sample-form') {
    e.respondWith(fetch(request).catch(() => new Response('{"offline":true}',
      { headers: { 'content-type': 'application/json' }, status: 503 })));
    return;
  }
  e.respondWith(
    caches.match(request, { ignoreSearch: true }).then(hit => hit || fetch(request).then(res => {
      const copy = res.clone();
      caches.open(CACHE).then(c => c.put(request, copy)).catch(() => {});
      return res;
    }).catch(() => caches.match('/')))
  );
});
