/* Page logic: loads the run results and calls the Python API. */
(function () {
  const $ = (id) => document.getElementById(id);
  const ROADS = ['north', 'east', 'south', 'west'];
  const fmt = (t) => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /* theme */
  const root = document.documentElement;
  const themeBtn = $('theme-btn');
  const theme = () => root.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  const paint = () => (themeBtn.textContent = theme() === 'dark' ? 'Light' : 'Dark');
  try { const t = localStorage.getItem('ats-theme'); if (t) root.setAttribute('data-theme', t); } catch (e) {}
  paint();
  themeBtn.addEventListener('click', () => {
    const next = theme() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('ats-theme', next); } catch (e) {}
    paint();
  });

  /* GitHub link, set by <body data-repo="https://github.com/..."> */
  const repo = document.body.dataset.repo;
  if (repo) { const a = $('repo-link'); a.href = repo; a.hidden = false; }

  /* results of the recorded run */
  Promise.all([fetch('media/results.json').then((r) => r.json()), fetch('media/ground_truth.json').then((r) => r.json())])
    .then(([res, truth]) => {
      const tb = $('cycles-table').querySelector('tbody');
      tb.innerHTML = res.cycles.map((c) => `<tr><td>${c.number}</td><td>${fmt(c.started)}</td><td>${c.greens.map((g) => Math.round(g)).join(' · ')}</td><td>${c.counts ? c.counts.join(' · ') : '–'}</td></tr>`).join('');
      $('events').innerHTML = res.events.map((e) => `<li><span>${fmt(e.t)}</span><span class="${e.kind === 'emergency' ? 'em' : ''}">${esc(e.text)}</span></li>`).join('');
      const ab = $('acc-table').querySelector('tbody');
      let ta = 0, tc = 0;
      ab.innerHTML = res.roads.map((r) => {
        const a = truth.counts[r], c = res.vehicles_counted[r];
        ta += a; tc += c;
        return `<tr><td>${r[0].toUpperCase() + r.slice(1)}</td><td>${a}</td><td>${c}</td><td>${Math.round((c / a) * 100)}%</td></tr>`;
      }).join('') + `<tr><td>All</td><td>${ta}</td><td>${tc}</td><td>${Math.round((tc / ta) * 100)}%</td></tr>`;
    })
    .catch(() => {});

  /* API status */
  const status = $('api-status');
  let apiUp = false;
  fetch('/api/health').then((r) => (r.ok ? r.json() : Promise.reject())).then((h) => {
    apiUp = true;
    status.textContent = `API online · OpenCV ${h.opencv}`;
    status.className = 'status ok';
  }).catch(() => {
    status.textContent = 'API not running here. Deploy to Vercel or run python dev.py';
    status.className = 'status off';
  });

  async function post(url, body) {
    if (!apiUp) throw new Error('The API is not running on this page. Deploy to Vercel or run python dev.py, then try again.');
    const r = await fetch(url, body instanceof FormData ? { method: 'POST', body } : { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `The server answered ${r.status}.`);
    return data;
  }
  const busy = (el, text) => { el.hidden = false; el.innerHTML = `<p class="hint">${text}</p>`; };
  const fail = (el, e) => { el.hidden = false; el.innerHTML = `<p class="err">${esc(e.message)}</p>`; };

  /* 1. count */
  const cLine = $('count-line'), cArea = $('count-area');
  cLine.addEventListener('input', () => ($('count-line-out').textContent = `${cLine.value}%`));
  cArea.addEventListener('input', () => ($('count-area-out').textContent = `${cArea.value} px`));
  async function runCount(blob, name) {
    const out = $('count-out');
    busy(out, 'Counting with OpenCV on the server…');
    const fd = new FormData();
    fd.append('video', blob, name);
    fd.append('line', String(cLine.value / 100));
    fd.append('min_area', cArea.value);
    fd.append('direction', $('count-dir').value);
    try {
      const r = await post('/api/count', fd);
      out.innerHTML = `
        <div><span class="big">${r.count}</span> vehicles crossed the line <span class="hint">(${r.down} moving down, ${r.up} moving up, ${r.seconds} s at ${r.fps} fps)</span></div>
        ${lineChart(r.timeline.map((p) => [p.t, p.count]), 'vehicles counted so far')}
        <div class="shots">${r.keyframes.map((k) => `<img src="${k}" alt="Frame with detected vehicles boxed">`).join('')}</div>`;
    } catch (e) { fail(out, e); }
  }
  $('count-file').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) runCount(f, f.name); e.target.value = ''; });
  $('count-sample').addEventListener('click', async () => {
    const b = await (await fetch('media/sample-north-20s.mp4')).blob();
    runCount(b, 'sample-north-20s.mp4');
  });

  /* 2. plan */
  $('plan-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const out = $('plan-out');
    const counts = ['pn', 'pe', 'ps', 'pw'].map((id) => Math.max(0, Number($(id).value) || 0));
    try {
      const r = await post('/api/plan', { counts });
      const max = Math.max(...r.greens, 1);
      out.hidden = false;
      out.innerHTML = `<div class="bars">${r.greens.map((g, i) => `<div><span>${ROADS[i][0].toUpperCase() + ROADS[i].slice(1)}</span><i style="width:${(g / max) * 100}%"></i><span>${g} s</span></div>`).join('')}</div><p class="hint">${esc(r.rule)}</p>`;
    } catch (err) { fail(out, err); }
  });

  /* 3. siren */
  const audio = $('siren-audio');
  async function runSiren(blob, name) {
    const out = $('siren-out');
    if (audio.src) URL.revokeObjectURL(audio.src);
    audio.src = URL.createObjectURL(blob);
    audio.hidden = false;
    busy(out, 'Running the FFT on the server…');
    const fd = new FormData();
    fd.append('audio', blob, name);
    try {
      const r = await post('/api/siren', fd);
      const spans = r.intervals.length ? r.intervals.map(([a, b]) => `${a.toFixed(1)} s to ${b.toFixed(1)} s`).join(', ') : 'none';
      out.innerHTML = `<div><span class="big">${r.intervals.length ? 'Siren heard' : 'No siren'}</span> <span class="hint">${spans}</span></div>
        ${ratioChart(r)}`;
    } catch (e) { fail(out, e); }
  }
  $('siren-file').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) runSiren(f, f.name); e.target.value = ''; });
  $('siren-sample').addEventListener('click', async () => {
    const b = await (await fetch('media/sample-siren-35s.wav')).blob();
    runSiren(b, 'sample-siren-35s.wav');
  });

  /* small SVG charts */
  function lineChart(points, label) {
    if (points.length < 2) return '';
    const W = 320, H = 90, px = 28, py = 10;
    const maxT = points[points.length - 1][0] || 1, maxY = Math.max(1, ...points.map((p) => p[1]));
    const X = (t) => px + (t / maxT) * (W - px - 6), Y = (v) => H - py - (v / maxY) * (H - 2 * py);
    const d = points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}">
      <line x1="${px}" x2="${W - 6}" y1="${H - py}" y2="${H - py}" stroke="var(--line)"/>
      <path d="${d}" fill="none" stroke="var(--paint)" stroke-width="2"/>
      <text x="${px - 4}" y="${Y(maxY) + 4}" text-anchor="end" font-size="10" fill="var(--muted)">${maxY}</text>
      <text x="${px - 4}" y="${H - py + 4}" text-anchor="end" font-size="10" fill="var(--muted)">0</text>
      <text x="${W - 6}" y="${H - py + 0}" dy="-4" text-anchor="end" font-size="10" fill="var(--muted)">${maxT.toFixed(0)} s</text></svg>`;
  }
  function ratioChart(r) {
    const pts = r.timeline;
    if (pts.length < 2) return '';
    const W = 320, H = 100, px = 24, py = 10, cap = Math.max(r.threshold * 2, ...pts.map((p) => Math.min(p.ratio, 40)));
    const X = (t) => px + (t / r.seconds) * (W - px - 6), Y = (v) => H - py - (Math.min(v, cap) / cap) * (H - 2 * py);
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.t).toFixed(1)},${Y(p.ratio).toFixed(1)}`).join('');
    const shade = r.intervals.map(([a, b]) => `<rect x="${X(a)}" y="${py}" width="${Math.max(1, X(b) - X(a))}" height="${H - 2 * py}" fill="var(--red)" opacity="0.15"/>`).join('');
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Siren band ratio over time">${shade}
      <line x1="${px}" x2="${W - 6}" y1="${Y(r.threshold)}" y2="${Y(r.threshold)}" stroke="var(--muted)" stroke-dasharray="3 3"/>
      <text x="${px - 4}" y="${Y(r.threshold) + 3}" text-anchor="end" font-size="10" fill="var(--muted)">${r.threshold}×</text>
      <path d="${d}" fill="none" stroke="var(--paint)" stroke-width="1.5"/>
      <text x="${W - 6}" y="${H - 2}" text-anchor="end" font-size="10" fill="var(--muted)">${r.seconds.toFixed(0)} s</text></svg>
      <p class="hint">Yellow: siren-band power ÷ the rest of the spectrum. Dashed: the 4× threshold. Red: where a siren was detected.</p>`;
  }
})();
