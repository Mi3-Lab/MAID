'use strict';
// ---------- IndexedDB (stores: kv, images, events) ----------
const DBP = new Promise((res, rej) => {
  const r = indexedDB.open('wheelview', 2);
  r.onupgradeneeded = ev0 => {
    const d = r.result;
    if (ev0.oldVersion < 1) {
      d.createObjectStore('kv');
      d.createObjectStore('images', { keyPath: 'image_id' });
      const ev = d.createObjectStore('events', { keyPath: 'uuid' });
      ev.createIndex('synced', 'synced'); ev.createIndex('image_id', 'image_id');
    }
    if (ev0.oldVersion < 2) d.createObjectStore('auto', { keyPath: 'image_id' });   // machine suggestions, cached with review batches
  };
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
});
async function tx(store, mode, fn) {
  const d = await DBP;
  return new Promise((res, rej) => {
    const t = d.transaction(store, mode); const s = t.objectStore(store); let out;
    Promise.resolve(fn(s)).then(v => out = v);
    t.oncomplete = () => res(out); t.onerror = () => rej(t.error);
  });
}
const req = r => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
const kvGet = k => tx('kv', 'readonly', s => req(s.get(k)));
const kvSet = (k, v) => tx('kv', 'readwrite', s => { s.put(v, k); });
const allOf = store => tx(store, 'readonly', s => req(s.getAll()));

// ---------- state ----------
const S = { schema: null, annotator: null, device: null, images: [], idx: 0, mine: {}, row: 0, exclMode: false, bboxMode: false, syncing: false,
  mode: 'label', order: 'lowconf', auto: {}, armed: null, faces: true };   // faces: show the blur detector's face boxes (review mode)   // mode: 'label' | 'review' (Review auto labels)
const RISK = { child: 'likely child', clinical_setting: 'likely clinical setting', not_a_photo: 'likely not a photo' };   // need a 2nd 'a' press
const $ = id => document.getElementById(id);
const _fetch = window.fetch.bind(window);
window.fetch = async (...a) => { const r = await _fetch(...a); if (r.status === 401) { location.href = '/unlock.html'; throw new Error('locked'); } return r; };
const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c == 'x' ? r : (r & 3 | 8)).toString(16); }));
const slug = s => s.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '').slice(0, 40);

// Renamed option values (schema.value_aliases, e.g. device_type 'lookalike' -> 'ambiguous'): old events keep their
// recorded value; every read maps it to the canonical option so old and new labels merge.
const canon = (f, v) => (typeof v === 'string' && S.schema?.value_aliases?.[f]?.[v]) || v;
function foldMine(events) {   // latest value per (image, field) for this annotator; ts then uuid
  const m = {};
  for (const e of events) {
    if (e.annotator !== S.annotator || e.kind !== 'label') continue;
    const im = m[e.image_id] ||= {}; const k = [e.ts, e.uuid];
    const cur = im[e.field];
    if (!cur || k[0] > cur.k[0] || (k[0] === cur.k[0] && k[1] > cur.k[1])) im[e.field] = { v: canon(e.field, e.value), k, n: (cur?.n || 0) + 1 }; else cur.n++;
  }
  return m;
}
async function refreshMine() { S.mine = foldMine(await allOf('events')); }
const val = (img, f) => S.mine[img]?.[f]?.v;

function prov(image_id, field) {   // review mode: every human edit carries "model-suggested" provenance
  const a = S.mode === 'review' && S.auto[image_id]; if (!a) return {};
  let sv = null, cf = null;
  if (a.suggestions?.[field]) { sv = canon(field, a.suggestions[field].value); cf = a.suggestions[field].confidence; }
  else if (field === 'face_blur_missed') { sv = (a.flags || []).includes('visible_face'); cf = a.screening?.visible_face ?? null; }
  return { source: 'auto_review', model_tag: a.model_tag, suggested: sv, confidence: cf };
}
async function emit(field, value, image_id) {
  image_id ||= cur().image_id;
  const e = { uuid: uuid(), image_id, annotator: S.annotator, field, value, ts: new Date().toISOString(), device_id: S.device, kind: 'label', ...prov(image_id, field), synced: 0 };
  await tx('events', 'readwrite', s => { s.put(e); });
  const im = S.mine[image_id] ||= {}; im[field] = { v: value, k: [e.ts, e.uuid], n: (im[field]?.n || 0) + 1 };
  render(); updQueue(); scheduleSync(800);
}

// ---------- sync: push queued events, pull own events from other devices ----------
let syncTimer = null;
function scheduleSync(ms) { clearTimeout(syncTimer); syncTimer = setTimeout(sync, ms); }
async function updQueue() {
  const n = await tx('events', 'readonly', s => req(s.index('synced').count(0)));
  $('queue').textContent = n + ' queued'; return n;
}
function setNet() { const on = navigator.onLine; $('net').textContent = on ? 'online' : 'offline'; $('net').className = 'pill ' + (on ? 'on' : 'off'); }
async function sync() {
  if (S.syncing || !S.annotator) return; S.syncing = true;
  try {
    const q = await tx('events', 'readonly', s => req(s.index('synced').getAll(0)));
    for (let i = 0; i < q.length; i += 500) {
      const chunk = q.slice(i, i + 500).map(({ synced, ...e }) => e);
      const r = await fetch('/api/events', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ events: chunk }) });
      if (!r.ok) throw new Error('push ' + r.status);
      const j = await r.json(); const ok = new Set([...j.accepted, ...j.duplicate]);
      const bad = new Map(j.rejected.map(x => [x.uuid, x.reason]));
      await tx('events', 'readwrite', s => { for (const e of q.slice(i, i + 500)) { if (ok.has(e.uuid)) s.put({ ...e, synced: 1 }); else if (bad.has(e.uuid)) s.put({ ...e, synced: -1, error: bad.get(e.uuid) }); } });
    }
    let since = (await kvGet('last_seq_' + S.annotator)) || 0, newImgs = new Set();
    for (;;) {
      const r = await fetch(`/api/events?annotator=${encodeURIComponent(S.annotator)}&since=${since}`);
      if (!r.ok) throw new Error('pull ' + r.status);
      const j = await r.json(); if (!j.events.length) break;
      await tx('events', 'readwrite', s => { for (const { seq, ...e } of j.events) s.put({ ...e, synced: 1 }); });
      j.events.forEach(e => newImgs.add(e.image_id)); since = j.max_seq; await kvSet('last_seq_' + S.annotator, since);
    }
    const known = new Set(S.images.map(i => i.image_id));
    for (const id of newImgs) if (!known.has(id)) {      // labelled on another device -> make it correctable here
      const r = await fetch('/api/images/' + id); if (r.ok) { const m = await r.json(); await addImages([m]); }
    }
    await refreshMine(); render();
    $('net').title = 'last sync ' + new Date().toLocaleTimeString();
  } catch (err) { console.warn('sync failed (will retry)', err); }
  finally { S.syncing = false; updQueue(); }
}
window.addEventListener('online', () => { setNet(); sync(); });
window.addEventListener('offline', setNet);
setInterval(() => { if (navigator.onLine) sync(); }, 15000);   // periodic retry

// ---------- batch ----------
async function addImages(list, review = false) {
  const have = new Map((await allOf('images')).map(h => [h.image_id, h]));
  let ord = Math.max(-1, ...[...have.values()].map(h => h.order ?? -1)) + 1, rord = Math.max(-1, ...[...have.values()].map(h => h.rorder ?? -1)) + 1;
  await tx('images', 'readwrite', s => {
    for (const { auto, ...m } of list) {
      const h = have.get(m.image_id);
      // existing records get fresh metadata (e.g. a face-blurred copy now exists) but keep their queue positions
      if (!h) s.put({ ...m, annotator: S.annotator, ...(review ? { rorder: rord++ } : { order: ord++ }) });
      else s.put({ ...h, ...m, ...(review ? { rorder: h.rorder ?? rord++ } : { order: h.order ?? ord++ }) });
    }
  });
  if (review) await tx('auto', 'readwrite', s => { for (const m of list) if (m.auto) s.put(m.auto); });
  await loadImages();
  if ('caches' in window) { const c = await caches.open('wv-img-v2'); for (const m of list) { try { if (!(await c.match(m.img))) await c.add(m.img); } catch (e) { } } }
}
async function loadImages() {
  const k = S.mode === 'review' ? 'rorder' : 'order';
  S.images = (await allOf('images')).filter(i => i.annotator === S.annotator && i[k] !== undefined).sort((a, b) => a[k] - b[k]);
  if (S.mode === 'review') { S.auto = {}; for (const a of await allOf('auto')) S.auto[a.image_id] = a; }
}
async function getBatch() {
  if (S.mode === 'review') return getReviewBatch();
  $('btnBatch').disabled = true;
  try {
    const r = await fetch(`/api/assign?annotator=${encodeURIComponent(S.annotator)}&n=30`);
    if (r.status === 503) { alert((await r.json()).detail); $('btnBatch').disabled = false; return; }
    if (!r.ok) throw new Error(r.status); const j = await r.json(); await addImages(j.images);
    if (S.idx >= S.images.length) S.idx = 0; render();
  } catch (e) { alert('Could not fetch a batch (offline?). You can keep labelling what is already cached.'); }
  $('btnBatch').disabled = false;
}

async function getReviewBatch() {
  $('btnBatch').disabled = true;
  try {
    const r = await fetch(`/api/review/assign?annotator=${encodeURIComponent(S.annotator)}&n=30&order=${S.order}`);
    if (r.status === 404) { alert('No machine suggestions on this server yet. (If the server was just updated, it needs a restart.)'); $('btnBatch').disabled = false; return; }
    if (r.status === 503) { alert((await r.json()).detail); $('btnBatch').disabled = false; return; }
    if (!r.ok) throw new Error(r.status); const j = await r.json(); await addImages(j.images, true);
    firstOpen(); render();
  } catch (e) { alert('Could not fetch a review batch (offline?). You can keep reviewing what is already cached.'); }
  $('btnBatch').disabled = false;
}
const isDone = id => val(id, '_status') === 'done';
function firstOpen() { const i = S.images.findIndex(m => !isDone(m.image_id)); S.idx = i < 0 ? Math.min(S.idx, Math.max(0, S.images.length - 1)) : i; }
function hashKey(s) { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; }
async function reorder() {   // re-sort the cached, not-yet-done review queue by the chosen order (done ones stay first)
  const conf = m => S.auto[m.image_id]?.suggestions?.device_type?.confidence ?? 1;
  const key = { lowconf: m => [conf(m), hashKey(m.image_id)], wheelchair: m => [m.proposed_class === 'wheelchair_unspecified' ? 0 : 1, conf(m)], random: m => [hashKey(m.image_id + S.device), 0] }[S.order];
  const done = S.images.filter(m => isDone(m.image_id)), open = S.images.filter(m => !isDone(m.image_id));
  open.sort((a, b) => { const x = key(a), y = key(b); return x[0] - y[0] || x[1] - y[1]; });
  const list = [...done, ...open];
  await tx('images', 'readwrite', s => { list.forEach((m, i) => s.put({ ...m, rorder: i })); });
  await loadImages(); firstOpen(); render();
}
async function setMode(m) {
  S.mode = m; S.images = []; S.idx = 0; S.exclMode = false; S.armed = null; await kvSet('mode', m);
  document.body.classList.toggle('review', m === 'review');
  $('btnReview').classList.toggle('primary', m === 'review'); $('btnReview').textContent = m === 'review' ? 'Review auto labels: ON' : 'Review auto labels';
  $('rorder').hidden = m !== 'review';
  await loadImages();
  if (m === 'review') { firstOpen(); if (!S.images.length && navigator.onLine) await getReviewBatch(); }
  render();
}

// ---------- rendering ----------
const cur = () => S.images[S.idx];
function autoSug(img, f) { const s = S.mode === 'review' ? S.auto[img.image_id]?.suggestions?.[f] : null; return s ? { ...s, value: canon(f, s.value) } : s; }
function suggestion(img, f) {
  if (S.mode === 'review') return autoSug(img, f)?.value ?? null;
  return S.schema.prefill[img.proposed_class]?.[f] ?? null;
}
const pct = c => Math.round((c || 0) * 100) + '%';
// suggest "face blur missed" only when the classifier sees a face that the blur pass did not cover (or no blurred copy exists)
function missSug(im) { const A = S.auto[im.image_id]; return !!A && (A.flags || []).includes('visible_face') && !(im.blurred && (im.faces || []).length); }
function rows() { return [...S.schema.fields.map(f => f.name)]; }
function render() {
  if (!S.images.length) { $('pos').textContent = 'No images yet — press "Get batch".'; $('fields').innerHTML = ''; $('img').removeAttribute('src'); return; }
  const im = cur(); const id = im.image_id;
  if ($('img').dataset.id !== id) { $('img').dataset.id = id; $('img').src = im.img; }
  const done = val(id, '_status') === 'done', A = S.mode === 'review' ? S.auto[id] : null;
  $('pos').innerHTML = `${S.mode === 'review' ? '<span class="tag ai">REVIEW</span> ' : ''}${S.idx + 1} / ${S.images.length} · ${id} ${done ? '<span class="tag done">done</span>' : ''} ${val(id, 'exclude') ? '<span class="tag excl">excluded</span>' : ''}`;
  const risks = A ? (A.flags || []).filter(k => RISK[k]) : [];
  $('aibanner').hidden = !A;
  if (A) $('aibanner').innerHTML = `<b>Machine suggestions</b> (${A.model_tag}) — orange = model's guess with its confidence; <kbd>a</kbd>/<kbd>Enter</kbd> confirms all, number keys fix a field.` +
    (risks.length ? `<div class="risk">⚠ screening: ${risks.map(k => `${RISK[k]} (${pct(A.screening[k])})`).join(', ')} — check it; exclude with <kbd>e</kbd> if true.${S.armed === id ? ' <b>Press a again to confirm anyway.</b>' : ''}</div>` : '') +
    (missSug(im) ? `<div class="risk">model sees a face (${pct(A.screening.visible_face)}) but the blur found none → "face blur missed" is pre-checked; untick (<kbd>f</kbd>) if no face is visible</div>` : '') +
    (im.blurred ? `<div class="muted">image shown face-blurred (${(im.faces || []).length} face box${(im.faces || []).length === 1 ? '' : 'es'}, <kbd>o</kbd> toggles overlay). Any face still recognisable? press <kbd>f</kbd>.</div>` : '');
  $('meta').innerHTML = `weak label: <b>${im.proposed_class || '—'}</b> · ${im.source} · ${im.license}<br><a href="${im.source_page_url}" target="_blank" rel="noopener">source page</a> · ${(im.title || '').slice(0, 80)}${['1','yes','True','true'].includes(im.needs_visual_screen) ? '<br>⚠ screen: ' + im.screen_reasons : ''}`;
  $('fields').innerHTML = S.schema.fields.map((f, ri) => {
    const v = val(id, f.name), sug = v === undefined ? suggestion(im, f.name) : null, as = autoSug(im, f.name);
    const cls = S.mode === 'review' ? 'ai' : 'sug';
    return `<div class="row ${ri === S.row && !S.exclMode ? 'active' : ''}" data-r="${ri}"><span class="lbl">${f.label}</span>` +
      f.options.map((o, oi) => `<span class="opt ${v === o ? 'sel' : ''} ${sug === o ? cls : ''}" data-f="${f.name}" data-v="${o}"><kbd>${oi + 1}</kbd> ${o}${sug === o && as ? ` <small class="conf">${pct(as.confidence)}</small>` : ''}</span>`).join('') +
      (as ? `<span class="aiv ${v === undefined ? '' : v === as.value ? 'ok' : 'fixed'}" title="model suggestion">AI: ${as.value} ${pct(as.confidence)}${v !== undefined ? (v === as.value ? ' ✓' : ' ✎') : ''}</span>` : '') + '</div>';
  }).join('');
  const ex = val(id, 'exclude') || '';
  $('excl').innerHTML = S.schema.exclude_reasons.map((r, i) => `<span class="opt ex ${ex === r ? 'sel' : ''}" data-f="exclude" data-v="${r}">${S.exclMode ? `<kbd>${i + 1}</kbd> ` : ''}${r}</span>`).join('') +
    (ex ? ` <span class="opt" data-f="exclude" data-v="">un-exclude</span>` : '');
  $('excl').parentElement.classList.toggle('active', S.exclMode);
  const fv = val(id, 'face_blur_missed'), fs = A && fv === undefined && missSug(im);
  $('face').checked = fv === undefined ? !!fs : !!fv; $('face').parentElement.classList.toggle('ai', !!fs);
  const bb = val(id, 'bbox') || []; $('bboxinfo').textContent = bb.length + ' box(es)' + (S.bboxMode ? ' — drag on image' : '');
  $('btnBbox').classList.toggle('primary', S.bboxMode);
  const h = S.mine[id] || {}; const edits = Object.values(h).reduce((a, x) => a + x.n, 0);
  $('hist').textContent = edits ? `${edits} edit event(s) by you on this image; any field can be changed — the latest edit wins for you only.` : '';
  drawBoxes();
}
function drawBoxes() {
  const img = $('img'), cv = $('cv');
  cv.width = img.clientWidth; cv.height = img.clientHeight; cv.style.left = img.offsetLeft + 'px'; cv.style.top = img.offsetTop + 'px';
  cv.style.pointerEvents = S.bboxMode ? 'auto' : 'none';
  const g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, cv.height); g.lineWidth = 2; g.strokeStyle = '#ffcc00';
  for (const [x, y, w, h] of (S.images.length ? val(cur().image_id, 'bbox') || [] : [])) g.strokeRect(x * cv.width, y * cv.height, w * cv.width, h * cv.height);
  const A = S.mode === 'review' && S.images.length ? S.auto[cur().image_id] : null;
  const fim = S.images.length ? cur() : null;
  if (S.mode === 'review' && S.faces && fim?.faces?.length) {   // blur detector's face boxes: thin magenta, score label
    g.save(); g.setLineDash([2, 3]); g.lineWidth = 1.5; g.strokeStyle = '#e0119d'; g.fillStyle = '#e0119d'; g.font = '10px system-ui';
    for (const [x, y, w, h, s] of fim.faces) { g.strokeRect(x * cv.width, y * cv.height, w * cv.width, h * cv.height); g.fillText(`face ${pct(s)}`, x * cv.width, y * cv.height - 2); }
    g.restore();
  }
  if (A?.boxes?.length && !(val(cur().image_id, 'bbox') || []).length) {   // model boxes: dashed orange, saved only with Shift+B
    g.save(); g.setLineDash([6, 4]); g.strokeStyle = '#ff8c00'; g.fillStyle = '#ff8c00'; g.font = '12px system-ui';
    for (const b of A.boxes) { const [x, y, w, h] = b.box; g.strokeRect(x * cv.width, y * cv.height, w * cv.width, h * cv.height); g.fillText(`${b.label} ${pct(b.score)}`, x * cv.width + 3, y * cv.height + 13); }
    g.restore();
  }
  if (drag) { g.strokeStyle = '#00e5ff'; g.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0); }
}
let drag = null;
$('cv').addEventListener('pointerdown', e => { if (!S.bboxMode) return; drag = { x0: e.offsetX, y0: e.offsetY, x1: e.offsetX, y1: e.offsetY }; });
$('cv').addEventListener('pointermove', e => { if (drag) { drag.x1 = e.offsetX; drag.y1 = e.offsetY; drawBoxes(); } });
$('cv').addEventListener('pointerup', () => {
  if (!drag) return; const cv = $('cv'); const d = drag; drag = null;
  const x = Math.min(d.x0, d.x1) / cv.width, y = Math.min(d.y0, d.y1) / cv.height, w = Math.abs(d.x1 - d.x0) / cv.width, h = Math.abs(d.y1 - d.y0) / cv.height;
  if (w > 0.01 && h > 0.01) emit('bbox', [...(val(cur().image_id, 'bbox') || []), [x, y, w, h].map(v => Math.round(v * 1e4) / 1e4)]); else drawBoxes();
});
$('img').addEventListener('load', drawBoxes); window.addEventListener('resize', drawBoxes);
$('img').addEventListener('error', () => { $('imgmsg').textContent = 'image not cached / unavailable offline'; });
$('img').addEventListener('load', () => { $('imgmsg').textContent = ''; });

// ---------- actions ----------
function go(d) { if (!S.images.length) return; S.idx = Math.max(0, Math.min(S.images.length - 1, S.idx + d)); S.exclMode = false; render(); }
async function confirmAll() {   // review mode: accept every remaining suggestion as MY label (with provenance), then done
  const im = cur(); if (!im) return; const id = im.image_id, A = S.auto[id];
  if (!A) return done();
  const risks = (A.flags || []).filter(k => RISK[k]);
  if (risks.length && !val(id, 'exclude') && S.armed !== id) { S.armed = id; render(); return; }   // screening flag: 2nd press needed
  S.armed = null;
  if (!val(id, 'exclude')) {
    for (const f of rows()) { const s = canon(f, A.suggestions?.[f]?.value); if (val(id, f) === undefined && s != null) await emit(f, s); }
    if (val(id, 'face_blur_missed') === undefined && missSug(im)) await emit('face_blur_missed', true);
  }
  if (val(id, 'exclude') === undefined) await emit('exclude', '');
  await emit('_status', 'done');
  const nx = S.images.findIndex((m, i) => i > S.idx && !isDone(m.image_id));
  if (nx >= 0) { S.idx = nx; S.exclMode = false; render(); } else if (S.idx < S.images.length - 1) go(1); else render();
}
async function acceptBoxes() {
  const A = S.auto[cur()?.image_id]; if (!A?.boxes?.length) return;
  emit('bbox', A.boxes.map(b => b.box));
}
async function done() {
  if (S.mode === 'review' && S.auto[cur()?.image_id]) return confirmAll();
  const im = cur(); if (!im) return; const id = im.image_id;
  if (!val(id, 'exclude')) for (const f of rows()) { const s = suggestion(im, f); if (val(id, f) === undefined && s != null) await emit(f, s); }
  if (val(id, 'exclude') === undefined) await emit('exclude', '');
  await emit('_status', 'done');
  if (S.idx < S.images.length - 1) go(1); else render();
}
$('fields').addEventListener('click', e => { const o = e.target.closest('.opt'); if (o) { S.row = +o.closest('.row').dataset.r; emit(o.dataset.f, val(cur().image_id, o.dataset.f) === o.dataset.v ? null : o.dataset.v); } });
$('excl').addEventListener('click', e => { const o = e.target.closest('.opt'); if (o) { S.exclMode = false; emit('exclude', o.dataset.v); } });
$('face').addEventListener('change', e => emit('face_blur_missed', e.target.checked));
$('btnBbox').onclick = () => { S.bboxMode = !S.bboxMode; render(); };
$('btnBboxClear').onclick = () => emit('bbox', []);
$('btnPrev').onclick = () => go(-1); $('btnNext').onclick = () => go(1); $('btnDone').onclick = done;
$('btnSync').onclick = sync; $('btnBatch').onclick = getBatch; $('btnHelp').onclick = () => $('help').showModal();
$('btnReview').onclick = () => setMode(S.mode === 'review' ? 'label' : 'review');
$('rorder').onchange = async e => { S.order = e.target.value; e.target.blur(); await kvSet('order', S.order); reorder(); };
$('btnLogout').onclick = async () => { await kvSet('annotator', null); location.reload(); };
$('btnList').onclick = () => {
  const l = $('list'); const show = l.hidden; l.hidden = !show; $('main').hidden = show;
  if (!show) return render();
  l.innerHTML = '<h3>My images (click to open / correct)</h3><table><tr><th>#</th><th>id</th><th>status</th>' + S.schema.fields.map(f => `<th>${f.name}</th>`).join('') + '<th>exclude</th></tr>' +
    S.images.map((im, i) => `<tr data-i="${i}" style="cursor:pointer"><td>${i + 1}</td><td>${im.image_id}</td><td>${val(im.image_id, '_status') || ''}</td>` +
      S.schema.fields.map(f => `<td>${val(im.image_id, f.name) ?? ''}</td>`).join('') + `<td>${val(im.image_id, 'exclude') || ''}</td></tr>`).join('') + '</table>';
};
$('list').addEventListener('click', e => { const tr = e.target.closest('tr[data-i]'); if (tr) { S.idx = +tr.dataset.i; $('list').hidden = true; $('main').hidden = false; render(); } });

document.addEventListener('keydown', e => {
  if (!S.annotator || $('main').hidden || e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' || e.metaKey || e.ctrlKey) return;
  const k = e.key; const id = cur()?.image_id; let handled = true;
  if (S.exclMode && /^[0-9]$/.test(k)) { S.exclMode = false; const r = k === '0' ? '' : S.schema.exclude_reasons[+k - 1]; if (r !== undefined) emit('exclude', r); else render(); }
  else if (k === 'ArrowUp') { S.row = Math.max(0, S.row - 1); S.exclMode = false; render(); }
  else if (k === 'ArrowDown') { S.row = Math.min(S.schema.fields.length - 1, S.row + 1); S.exclMode = false; render(); }
  else if (/^[0-9]$/.test(k) && id) { const f = S.schema.fields[S.row]; if (k === '0') emit(f.name, null); else if (f.options[+k - 1]) { emit(f.name, f.options[+k - 1]); if (S.row < S.schema.fields.length - 1) S.row++; } }
  else if (k === 'ArrowLeft' || k === 'p') go(-1);
  else if (k === 'ArrowRight' || k === 'n') go(1);
  else if (k === 'Enter') done();
  else if (k === 'a' && S.mode === 'review') confirmAll();
  else if (k === 'B' && S.mode === 'review') acceptBoxes();
  else if (k === 'r') setMode(S.mode === 'review' ? 'label' : 'review');
  else if (k === 'o' && S.mode === 'review') { S.faces = !S.faces; render(); }
  else if (k === 'e') { S.exclMode = !S.exclMode; render(); }
  else if (k === 'f' && id) emit('face_blur_missed', !$('face').checked); // relative to what is shown (review may pre-tick it)
  else if (k === 'b' && !e.shiftKey) { S.bboxMode = !S.bboxMode; render(); }
  else if ((k === 'Backspace' || k === 'Delete') && id) emit('bbox', []);
  else if (k === '?') $('help').showModal();
  else if (k === 'Escape') { S.exclMode = false; S.bboxMode = false; render(); }
  else handled = false;
  if (handled) e.preventDefault();
});

// ---------- boot ----------
async function start(name) {
  S.annotator = name; await kvSet('annotator', name);
  document.activeElement?.blur(); $('who').textContent = '👤 ' + name; $('login').hidden = true; $('main').hidden = false;
  S.order = (await kvGet('order')) || 'lowconf'; $('rorder').value = S.order;
  const m = (await kvGet('mode')) || 'label';
  await refreshMine();
  if (m === 'review') await setMode('review'); else { await loadImages(); render(); }
  updQueue();
  if (navigator.onLine) { await sync(); if (!S.images.length) await getBatch(); }
}
(async () => {
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(e => console.warn('sw', e));
  setNet();
  S.device = await kvGet('device_id'); if (!S.device) { S.device = uuid(); await kvSet('device_id', S.device); }
  try { const r = await fetch('/api/schema'); S.schema = await r.json(); await kvSet('schema', S.schema); }
  catch (e) { S.schema = await kvGet('schema'); }
  if (!S.schema) { document.body.innerHTML = '<p style="padding:20px">First load needs a connection.</p>'; return; }
  const name = await kvGet('annotator');
  if (name) start(name); else { $('login').hidden = false; $('name').focus(); }
  const go = () => { const n = slug($('name').value); if (n) start(n); };
  $('btnLogin').onclick = go; $('name').addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
})();
