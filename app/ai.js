/* WheelView AI review page. Read-only: never POSTs labels. */
(function () {
  const $ = (id) => document.getElementById(id);
  const state = { offset: 0, limit: 40, filters: { verdict: '', field: '', value: '', source: '', action: '', basis: '', q: '' }, facets: {}, total: 0 };

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function badge(kind, text) {
    return `<span class="ai-badge ai-badge-${esc(kind)}">${esc(text)}</span>`;
  }
  function fmtConf(v, c) {
    if (v == null || v === '') return '<span class="muted">—</span>';
    const p = (c == null || c === '') ? '' : ` ${(Number(c) * 100).toFixed(0)}%`;
    return `${esc(v)}<small>${p}</small>`;
  }
  function voterChips(row) {
    let voters = row.voters;
    if (typeof voters === 'string') {
      try { voters = JSON.parse(voters); } catch (e) { voters = []; }
    }
    if (!Array.isArray(voters) || !voters.length) return '<span class="muted">no votes</span>';
    return voters.map(v => {
      const typ = v.type === 'ai' ? 'ai' : 'human';
      return `<span class="ai-vote ai-vote-${typ}" title="${esc(typ)}">${esc(v.id)}: ${esc(v.value)}</span>`;
    }).join(' ');
  }
  function etaText(job) {
    if (!job || job.status !== 'running' || job.eta_s == null) return '';
    const s = Number(job.eta_s);
    if (!isFinite(s) || s < 0) return '';
    if (s < 90) return `ETA ~${Math.round(s)}s`;
    return `ETA ~${Math.round(s / 60)} min`;
  }

  async function getJSON(url) {
    const r = await fetch(url, { cache: 'no-store', credentials: 'same-origin' });
    if (r.status === 401) { location.href = '/unlock.html'; throw new Error('auth'); }
    if (!r.ok) throw new Error(url + ' ' + r.status);
    return r.json();
  }

  function renderStatus(st) {
    const job = st.job || {};
    const pill = $('jobpill');
    const running = job.status === 'running';
    const done = job.status === 'done';
    const err = job.status === 'error';
    pill.textContent = job.job ? `${job.job}: ${job.status || 'unknown'}` : 'no job';
    pill.className = 'pill ' + (running ? 'on' : err ? 'off' : '');
    const pct = job.n_total ? Math.min(100, Math.round(100 * (job.n_done || 0) / job.n_total)) : (done ? 100 : 0);
    const bits = [];
    if (job.job) bits.push(`<div><b>${esc(job.job)}</b> · ${esc(job.stage || '')} · ${esc(job.status || '')}</div>`);
    else bits.push('<div class="muted">No AI job has written a progress file yet.</div>');
    if (job.n_total) bits.push(`<div>Progress <b>${esc(job.n_done)}</b> / ${esc(job.n_total)} (${pct}%)</div>`);
    if (job.started) bits.push(`<div>Started ${esc(job.started)}${job.last_update ? ` · updated ${esc(job.last_update)}` : ''} ${esc(etaText(job))}</div>`);
    if (job.message) bits.push(`<div>${esc(job.message)}</div>`);
    $('statusbody').innerHTML = bits.join('');
    const barwrap = $('barwrap');
    if (job.job) { barwrap.hidden = false; $('bar').style.width = pct + '%'; $('bar').className = err ? 'err' : (done ? 'ok' : ''); }
    else barwrap.hidden = true;
    const log = (st.log_tail || []).join('\n');
    const pre = $('logtail');
    if (log) { pre.hidden = false; pre.textContent = log; pre.scrollTop = pre.scrollHeight; }
    else pre.hidden = true;
  }

  function metricCard(title, body) {
    return `<div class="ai-metric"><div class="ai-metric-h">${esc(title)}</div><div>${body}</div></div>`;
  }
  function pct(x) { return x == null ? '—' : (100 * x).toFixed(1) + '%'; }
  function ci(lo, hi) {
    if (lo == null || hi == null) return '';
    return ` (95% CI ${pct(lo)}–${pct(hi)})`;
  }

  function renderMetrics(sm) {
    const parts = [];
    const c3 = sm.consensus_3model || {};
    if (c3.device_type || c3.occupied) {
      parts.push('<h3>3-model consensus vs human (held-out / OOF)</h3><div class="ai-metrics">');
      for (const field of ['device_type', 'occupied']) {
        const m = c3[field] || {};
        if (!m || m.emit == null && m.precision == null) continue;
        const acc = m.per_model_accuracy || {};
        parts.push(metricCard(field,
          `consensus precision <b>${pct(m.precision)}</b>${ci(m.precision_ci95_lo, m.precision_ci95_hi)}` +
          ` · coverage ${pct(m.coverage)} · threshold ${esc(m.threshold)}` +
          `<div class="muted">per-model acc: SigLIP ${pct(acc.siglip)}, CLIP ${pct(acc.clip)}, DINOv2-OOF ${pct(acc.dinov2_oof)}</div>`));
      }
      parts.push('</div>');
    }
    const v = sm.verification;
    if (v) {
      parts.push('<h3>Verification of prior AI / assistant labels</h3>');
      parts.push(`<p class="muted">${esc((v.majority_rule && v.majority_rule.description) || 'Majority rule: ai+human or human+human proceeds; ties go to review.')}</p>`);
      parts.push('<div class="ai-metrics">');
      const vo = v.verdicts_overall || {};
      parts.push(metricCard('Verdicts', Object.keys(vo).sort().map(k => `${esc(k)}: <b>${vo[k]}</b>`).join('<br>')));
      const acc = (v.accuracy_vs_human_folded || {}).overall || {};
      parts.push(metricCard('Accuracy vs human (where they overlap)',
        acc.n_overlap ? `<b>${pct(acc.accuracy)}</b> (${acc.n_match} / ${acc.n_overlap})` : 'no overlap yet'));
      const mo = (v.majority_rule || {}).outcomes || {};
      parts.push(metricCard('Majority outcomes',
        `passes: <b>${(v.majority_rule || {}).passes || 0}</b> · ties: <b>${(v.majority_rule || {}).ties || 0}</b>` +
        '<div class="muted">' + Object.keys(mo).map(k => `${esc(k)}: ${mo[k]}`).join(' · ') + '</div>'));
      const sim = v.split_simulation || {};
      const maj = sim.if_majority_rule_auto_kept || {};
      const orig = sim.original || {};
      if (orig.n_total) {
        parts.push(metricCard('Training split (counts only, file unchanged)',
          `original ${orig.n_total} (${orig.n_human} human + ${orig.n_auto} auto)` +
          `<div>if only confirmed auto kept: <b>${(sim.if_only_confirmed_auto_kept || {}).n_total || '—'}</b></div>` +
          `<div>if majority-rule auto kept: <b>${maj.n_total || '—'}</b> (${maj.n_human} human + ${maj.n_auto_kept} auto; ${maj.n_auto_dropped_or_review} auto not kept)</div>`));
      }
      parts.push('</div>');
      if ((v.verdicts_by_source) ) {
        parts.push('<div class="muted">By source: ' + Object.keys(v.verdicts_by_source).map(s => {
          const c = v.verdicts_by_source[s];
          return `<b>${esc(s)}</b> ` + Object.keys(c).map(k => `${k} ${c[k]}`).join(', ');
        }).join(' · ') + '</div>');
      }
      if (v.problem_areas && v.problem_areas.length) {
        parts.push('<h3>Problem areas</h3><ul>' + v.problem_areas.map(p =>
          `<li>${esc(p.area)} — n=${esc(p.n)}${p.share != null ? ` (${pct(p.share)})` : ''}</li>`).join('') + '</ul>');
      }
    } else {
      parts.push('<p class="muted">Verification has not finished yet. The table will fill in when <code>derived/ai_label_verification.csv</code> is written.</p>');
    }
    $('metrics').innerHTML = parts.join('') || '<p class="muted">No summaries on disk yet.</p>';
  }

  function select(name, key, facets, extra) {
    const opts = facets[key] || {};
    const cur = state.filters[name] || '';
    const keys = Object.keys(opts).sort();
    return `<label>${esc(extra || name)} <select data-f="${esc(name)}">` +
      `<option value="">all</option>` +
      keys.map(k => `<option value="${esc(k)}" ${k === cur ? 'selected' : ''}>${esc(k)} (${opts[k]})</option>`).join('') +
      `</select></label>`;
  }

  function renderFilters(facets) {
    state.facets = facets || {};
    $('filters').innerHTML =
      select('source', 'source', state.facets, 'source') +
      select('field', 'field', state.facets, 'field') +
      select('value', 'value', state.facets, 'value') +
      select('verdict', 'verdict', state.facets, 'verdict') +
      select('basis', 'majority_basis', state.facets, 'basis') +
      select('action', 'recommended_action', state.facets, 'action') +
      `<label>search <input data-q type="search" placeholder="image id or text" value="${esc(state.filters.q)}"></label>` +
      `<button type="button" id="btnReset">Reset</button>`;
    $('filters').querySelectorAll('select').forEach(el => {
      el.addEventListener('change', () => { state.filters[el.dataset.f] = el.value; state.offset = 0; loadLabels(); });
    });
    const inp = $('filters').querySelector('input[data-q]');
    let t;
    inp.addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => { state.filters.q = inp.value.trim(); state.offset = 0; loadLabels(); }, 250); });
    $('btnReset').onclick = () => {
      state.filters = { verdict: '', field: '', value: '', source: '', action: '', basis: '', q: '' };
      state.offset = 0; loadLabels();
    };
  }

  function annotatorLink(id) {
    return `/?image=${encodeURIComponent(id)}`;
  }

  function renderTable(data) {
    state.total = data.total || 0;
    if (data.facets) renderFilters(data.facets);
    $('tabinfo').textContent = data.available
      ? `Showing ${data.rows.length} of ${data.total} matching rows (${data.n_all} total).`
      : 'No verification CSV yet.';
    const thead = $('tbl').tHead;
    thead.innerHTML = `<tr>
      <th></th><th>image</th><th>field</th><th>AI value</th>
      <th>SigLIP</th><th>CLIP</th><th>DINOv2</th>
      <th>voters</th><th>majority</th><th>human (folded)</th>
      <th>verdict</th><th>action</th><th>source</th><th></th>
    </tr>`;
    const tb = $('tbl').tBodies[0];
    tb.innerHTML = (data.rows || []).map(r => {
      const oof = r.dino_oof ? ' <small>OOF</small>' : '';
      const maj = r.majority_outcome === 'majority_pass'
        ? badge('pass', (r.majority_basis || 'majority') + ' → ' + (r.majority_value || ''))
        : (r.majority_outcome ? badge(r.majority_outcome, r.majority_outcome) : '');
      return `<tr class="ai-row-${esc(r.verdict || '')}">
        <td class="ai-thumb"><a href="${annotatorLink(r.image_id)}" title="Open in annotator"><img src="/img/${esc(r.image_id)}" alt="" loading="lazy" width="64" height="64"></a></td>
        <td><a href="${annotatorLink(r.image_id)}">${esc(r.image_id)}</a>${r.in_training_split ? ' <small>split</small>' : ''}</td>
        <td>${esc(r.field)}</td>
        <td><b>${esc(r.value)}</b></td>
        <td>${fmtConf(r.model_a_value, r.model_a_confidence)}</td>
        <td>${fmtConf(r.model_b_value, r.model_b_confidence)}</td>
        <td>${fmtConf(r.model_c_value, r.model_c_confidence)}${oof}</td>
        <td>${voterChips(r)}</td>
        <td>${maj}</td>
        <td>${r.human_value_folded ? esc(r.human_value_folded) : '<span class="muted">—</span>'}</td>
        <td>${badge(r.verdict, r.verdict)}${r.reason ? `<div class="muted">${esc(r.reason)}</div>` : ''}</td>
        <td>${esc(r.recommended_action || '')}</td>
        <td>${esc(r.source)}<div class="muted">${esc(r.source_file || '')}</div></td>
        <td><a class="ai-open" href="${annotatorLink(r.image_id)}">Open</a></td>
      </tr>`;
    }).join('') || '<tr><td colspan="14" class="muted">No rows match these filters.</td></tr>';
    const pages = Math.max(1, Math.ceil(state.total / state.limit));
    const page = Math.floor(state.offset / state.limit) + 1;
    $('pager').innerHTML = `<button ${state.offset <= 0 ? 'disabled' : ''} data-d="-1">Prev</button>` +
      ` <span>page ${page} / ${pages}</span> ` +
      `<button ${state.offset + state.limit >= state.total ? 'disabled' : ''} data-d="1">Next</button>`;
    $('pager').querySelectorAll('button').forEach(b => {
      b.onclick = () => {
        const d = Number(b.dataset.d);
        state.offset = Math.max(0, state.offset + d * state.limit);
        loadLabels();
      };
    });
  }

  async function loadLabels() {
    const p = new URLSearchParams();
    p.set('offset', String(state.offset));
    p.set('limit', String(state.limit));
    const map = { verdict: 'verdict', field: 'field', value: 'value', source: 'source', action: 'action', basis: 'basis', q: 'q' };
    for (const k in map) if (state.filters[k]) p.set(map[k], state.filters[k]);
    try {
      const data = await getJSON('/api/ai/labels?' + p.toString());
      renderTable(data);
    } catch (e) {
      if (String(e.message) !== 'auth') $('tabinfo').textContent = 'Could not load labels: ' + e.message;
    }
  }

  async function tick() {
    try {
      const st = await getJSON('/api/ai/status?n=40');
      renderStatus(st);
    } catch (e) {
      if (String(e.message) !== 'auth') $('statusbody').textContent = 'Status unavailable: ' + e.message;
    }
  }

  async function loadSummary() {
    try {
      renderMetrics(await getJSON('/api/ai/summary'));
    } catch (e) {
      if (String(e.message) !== 'auth') $('metrics').textContent = 'Summary unavailable: ' + e.message;
    }
  }

  tick(); loadSummary(); loadLabels();
  setInterval(tick, 4000);
  setInterval(loadSummary, 15000);
})();
