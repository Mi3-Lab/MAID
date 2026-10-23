'use strict';
let schema;
const slug = s => s.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '').slice(0, 40);
const adjEl = document.getElementById('adj');
adjEl.value = localStorage.getItem('wv_adjudicator') || '';
adjEl.onchange = () => localStorage.setItem('wv_adjudicator', slug(adjEl.value));
function opts(f) { return f === 'exclude' ? ['none', ...schema.exclude_reasons] : schema.fields.find(x => x.name === f).options; }
async function load() {
  schema ||= await (await fetch('/api/schema')).json();
  const st = await (await fetch('/api/stats')).json();
  const p = st.progress;
  document.getElementById('stats').innerHTML = `<b>Progress:</b> ${p.images_touched} images touched of ${p.candidates}; ` +
    Object.entries(p.status_counts).map(([k, v]) => `${k}: ${v}`).join(' · ') + ` · ${p.events} events<br>` +
    '<table class="k"><tr><th>field</th><th>pooled κ</th><th>n</th><th>pairwise κ (n≥5)</th></tr>' +
    Object.entries(st.kappa).map(([f, k]) => `<tr><td>${f}</td><td>${k.pooled_kappa ?? '—'}</td><td>${k.n_items}</td><td>${Object.entries(k.pairwise).map(([p, x]) => `${p}: ${x.kappa} (${x.n})`).join('; ')}</td></tr>`).join('') + '</table>';
  const { conflicts } = await (await fetch('/api/conflicts')).json();
  document.getElementById('list').innerHTML = conflicts.length ? conflicts.map(c => `<div class="card" data-id="${c.image_id}"><img src="${c.img}" loading="lazy"><div>
    <b>${c.image_id}</b> <span class="tag">${c.status}</span> <span class="muted">weak: ${c.proposed_class} · annotators: ${c.done.join(', ')}</span>
    ${Object.entries(c.conflicts).map(([f, vals]) => `<div class="cf"><b>${f}</b>: ` +
      Object.entries(vals).map(([a, v]) => `<span class="who">${a}=<b>${v}</b></span>`).join(' ') + '<br>' +
      opts(f).map(o => `<span class="opt" data-f="${f}" data-v="${o}">${o}</span>`).join('') + '</div>').join('')}
    <div class="muted"><a href="${c.source_page_url}" target="_blank" rel="noopener">source</a></div></div></div>`).join('') : '<p>No open disagreements 🎉</p>';
}
document.getElementById('list').addEventListener('click', async e => {
  const o = e.target.closest('.opt'); if (!o) return;
  const who = slug(adjEl.value); if (!who) { alert('Enter your adjudicator name first'); adjEl.focus(); return; }
  const id = o.closest('.card').dataset.id; const f = o.dataset.f; const v = f === 'exclude' && o.dataset.v === 'none' ? '' : o.dataset.v;
  const ev = { uuid: crypto.randomUUID(), image_id: id, annotator: who, field: f, value: v, ts: new Date().toISOString(), device_id: 'adjudicator-web', kind: 'adjudication' };
  const r = await fetch('/api/events', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ events: [ev] }) });
  const j = await r.json(); if (j.rejected.length) alert('rejected: ' + j.rejected[0].reason);
  load();
});
document.getElementById('reload').onclick = load;
load();
