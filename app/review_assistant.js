'use strict';
const $ = id => document.getElementById(id);
const esc = x => String(x ?? '').replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
const slug = s => s.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '').slice(0, 40);
let schema, items = [], pos = 0;
$('rev').value = localStorage.getItem('wv_reviewer') || localStorage.getItem('wv_annotator') || '';
$('rev').onchange = () => { $('rev').value = slug($('rev').value); localStorage.setItem('wv_reviewer', $('rev').value); };
const opts = f => (schema.fields.find(x => x.name === f) || { options: [] }).options;
const show = v => Array.isArray(v) ? v.join('+') : v;
async function load(keep) {
  schema ||= await (await fetch('/api/schema')).json();
  const id = keep && items[pos] && items[pos].image_id;
  const q = await (await fetch('/api/assistant_review/queue?status=' + $('status').value)).json();
  items = q.items; const p = q.progress;
  $('progress').textContent = `Progress: ${p.reviewed} of ${p.images} images reviewed (${p.partial} partial, ${p.pending} pending); ${p.fields_reviewed} of ${p.fields} fields`;
  const k = id ? items.findIndex(it => it.image_id === id) : -1; pos = k >= 0 ? k : Math.min(pos, Math.max(0, items.length - 1));
  render();
}
function render() {
  const it = items[pos];
  $('pos').textContent = items.length ? `${pos + 1} / ${items.length}` : '';
  if (!it) { $('item').innerHTML = '<p>Nothing to review here.</p>'; return; }
  $('item').innerHTML = `<h3>${esc(it.image_id)} <span class="tag">${esc(it.tier)}</span>${it.audit ? '<span class="tag">audit sample</span>' : ''}<span class="tag">${esc(it.review_status)}</span>
    <span class="muted">rank ${it.rank}</span></h3><div class="wrap"><img src="${esc(it.img)}"><div><table class="r">
    <tr><th>field</th><th>assistant (confidence)</th><th>other model</th><th>human labels</th><th>your review</th></tr>
    ${it.fields.map(x => `<tr class="${x.disagree_human || x.disagree_model ? 'dis' : ''} ${x.outcome ? 'done' : ''}" data-f="${esc(x.field)}">
      <td>${esc(x.field)}</td><td><b>${esc(show(x.value))}</b> (${esc(x.confidence)})</td>
      <td>${x.other == null ? '' : esc(x.other)}</td>
      <td>${Object.entries(x.human).map(([a, v]) => `${esc(a)}=<b>${esc(v)}</b>`).join(' ') || '<span class="muted">none</span>'}</td>
      <td>${x.outcome ? `${esc(x.outcome.action)}${x.outcome.value != null ? ' = ' + esc(show(x.outcome.value)) : ''} <span class="muted">(${esc(x.outcome.result)}, ${esc(x.outcome.reviewer)})</span> ` : ''}
        <button data-a="accept">Accept</button>
        <select data-edit><option value="">edit...</option>${opts(x.field).map(o => `<option>${esc(o)}</option>`).join('')}</select>
        <button data-a="reject">Reject</button></td></tr>`).join('')}
    </table>${it.source_page_url ? `<p><a href="${esc(it.source_page_url)}" target="_blank" rel="noopener">source</a></p>` : ''}</div></div>`;
}
async function decide(decisions) {
  const reviewer = slug($('rev').value);
  if (!reviewer) { alert('Enter your reviewer name first'); $('rev').focus(); return; }
  const it = items[pos];
  const r = await fetch('/api/assistant_review/decide', { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ reviewer, image_id: it.image_id, decisions }) });
  const j = await r.json();
  if (!r.ok) { alert(j.detail || 'error'); return; }
  const errs = j.results.filter(x => x.error); if (errs.length) alert(errs.map(x => `${x.field}: ${x.error}`).join('\n'));
  const routed = j.results.filter(x => x.result === 'conflict_routed'); if (routed.length) alert(`${routed.length} field(s) differ from existing human labels: sent to the adjudication queue`);
  const wasLast = it.fields.every(x => x.outcome || decisions.some(d => d.field === x.field));
  await load(true);
  if (wasLast && items[pos] && items[pos].image_id === it.image_id && pos < items.length - 1) { pos++; render(); }
}
$('item').addEventListener('click', e => {
  const b = e.target.closest('button[data-a]'); if (!b) return;
  decide([{ field: b.closest('tr').dataset.f, action: b.dataset.a }]);
});
$('item').addEventListener('change', e => {
  const s = e.target.closest('select[data-edit]'); if (!s || !s.value) return;
  decide([{ field: s.closest('tr').dataset.f, action: 'edit', value: s.value }]);
});
const openFields = () => (items[pos] ? items[pos].fields.filter(x => !x.outcome) : []);
$('acceptAll').onclick = () => { const f = openFields(); if (f.length) decide(f.map(x => ({ field: x.field, action: 'accept' }))); };
$('rejectAll').onclick = () => { const f = openFields(); if (f.length) decide(f.map(x => ({ field: x.field, action: 'reject' }))); };
$('prev').onclick = () => { if (pos > 0) { pos--; render(); } };
$('next').onclick = () => { if (pos < items.length - 1) { pos++; render(); } };
$('status').onchange = () => { pos = 0; load(); };
load();
