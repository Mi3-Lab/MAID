'use strict';
const $ = id => document.getElementById(id);
const slug = s => s.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '').slice(0, 40);
const S = { list: [], idx: 0, busy: false };
$('who').value = localStorage.getItem('wv_reviewer') || localStorage.getItem('wv_adjudicator') || '';
$('who').onchange = () => { $('who').value = slug($('who').value); localStorage.setItem('wv_reviewer', $('who').value); };
const qs = new URLSearchParams(location.search);
if (qs.has('rule')) $('rule').value = qs.get('rule');
if (qs.has('status')) $('status').value = qs.get('status');
async function load() {
  const p = new URLSearchParams(); if ($('rule').value) p.set('rule', $('rule').value); if ($('status').value) p.set('status', $('status').value);
  const r = await fetch('/api/removed?' + p); if (r.status === 401) { location.href = '/unlock.html'; return; }
  const j = await r.json(); S.list = j.images; S.idx = 0;
  const c = j.counts; $('counts').textContent = `${j.total} shown | ` + Object.entries(c.rule).map(([k, v]) => `${k} ${v}`).join(', ') +
    ' | ' + Object.entries(c.status).map(([k, v]) => `${k} ${v}`).join(', ');
  show();
}
function show() {
  const it = S.list[S.idx];
  if (!it) { $('iid').textContent = 'Nothing left in this filter'; $('pic').hidden = true; $('nopic').hidden = true; ['rules', 'reason', 'meta', 'dec', 'pos', 'st'].forEach(k => $(k).textContent = ''); return; }
  $('iid').textContent = it.image_id; $('pos').textContent = `${S.idx + 1} / ${S.list.length}`;
  $('st').textContent = it.status; $('st').className = 'tag st-' + it.status;
  $('rules').textContent = it.rules.join(', ') + '  (' + it.lists.join(', ') + ')';
  $('reason').textContent = it.reason || '-';
  $('meta').textContent = `${it.proposed_class} / ${it.source}` + (it.title ? ` - ${it.title}` : '');
  $('dec').textContent = it.decided_by ? `last decision: ${it.status} by ${it.decided_by} at ${new Date(it.decided_at).toLocaleString()}` : '';
  $('pic').hidden = !it.has_blurred; $('nopic').hidden = it.has_blurred; if (it.has_blurred) $('pic').src = it.img;
  const nx = S.list[S.idx + 1]; if (nx && nx.has_blurred) new Image().src = nx.img;
}
async function decide(action) {
  const it = S.list[S.idx]; if (!it || S.busy) return;
  const who = slug($('who').value);
  if (!who) { $('msg').textContent = 'Enter your name (Reviewer) first.'; $('who').focus(); return; }
  S.busy = true;
  try {
    const r = await fetch('/api/removed/decision', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ annotator: who, image_id: it.image_id, action }) });
    const j = await r.json();
    if (!r.ok) { $('msg').textContent = 'Error: ' + (j.detail || r.status); return; }
    it.status = j.status; it.decided_by = who; it.decided_at = new Date().toISOString();
    $('msg').textContent = `${it.image_id}: ${action === 'restore' ? 'restored (back in the queue)' : 'kept removed'}`;
    next();
  } catch (e) { $('msg').textContent = 'Network error: ' + e; } finally { S.busy = false; }
}
function next() { if (S.idx < S.list.length) S.idx++; show(); }
function prev() { if (S.idx > 0) S.idx--; show(); }
$('bRestore').onclick = () => decide('restore'); $('bConfirm').onclick = () => decide('confirm_removed');
$('bNext').onclick = next; $('bPrev').onclick = prev; $('rule').onchange = load; $('status').onchange = load;
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'r') decide('restore'); else if (k === 'x') decide('confirm_removed');
  else if (k === 'arrowright' || k === ' ' || k === 'j') { e.preventDefault(); next(); }
  else if (k === 'arrowleft' || k === 'k') prev(); else return;
});
load();