import { loadModel, modelReady, classify, segment } from '/static/recognize.js';

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const UR = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
const showDigit = d => (d === '?' ? '?' : ($('#numeral').value === 'urdu' ? UR[+d] : d));

/* ---------------- form / field catalogue ---------------- */
const FORMS = {
  'marksheet':              ['roll_no', 'marks_obtained', 'marks_total'],
  'polio-tally':            ['children_vaccinated', 'children_missed', 'houses_visited'],
  'epi-tally':              ['doses_given', 'children_present', 'zero_dose'],
  'flood-registration':     ['household_id', 'family_size', 'tent_no'],
  'meter-reading':          ['reading_kwh', 'meter_no', 'previous_reading'],
  'union-council-register': ['entry_no', 'cnic_last6', 'age_years'],
};
function fillForms() {
  const f = $('#formName');
  f.innerHTML = Object.keys(FORMS).map(k => `<option value="${k}">${k}</option>`).join('');
  fillFields();
  f.onchange = fillFields;
}
function fillFields() {
  $('#fieldName').innerHTML = FORMS[$('#formName').value]
    .map(k => `<option value="${k}">${k}</option>`).join('');
}
fillForms();

/* ---------------- IndexedDB queue ---------------- */
const DB = new Promise((res, rej) => {
  const r = indexedDB.open('raqam', 1);
  r.onupgradeneeded = () => r.result.createObjectStore('records', { keyPath: 'localId', autoIncrement: true });
  r.onsuccess = () => res(r.result);
  r.onerror = () => rej(r.error);
});
async function tx(mode, fn) {
  const db = await DB;
  return new Promise((res, rej) => {
    const t = db.transaction('records', mode);
    const store = t.objectStore('records');
    const out = fn(store);
    t.oncomplete = () => res(out.result ?? out);
    t.onerror = () => rej(t.error);
  });
}
const putRecord = r => tx('readwrite', s => s.put(r));
const allRecords = () => tx('readonly', s => s.getAll());
const getRecord = id => tx('readonly', s => s.get(id));

/* ---------------- connectivity ---------------- */
function setConn() {
  const on = navigator.onLine;
  $('#conn').classList.toggle('off', !on);
  $('#connText').textContent = on ? 'Online' : 'Offline — saved locally';
}
addEventListener('online', () => { setConn(); trySync(); });
addEventListener('offline', setConn);
setConn();

/* ---------------- tabs ---------------- */
function selectTab(name) {
  $$('nav.tabs button').forEach(x => x.setAttribute('aria-selected', x.dataset.tab === name));
  $$('.panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
  if (name === 'review') renderReview();
  if (name === 'records') renderRecords();
  if (name === 'dashboard') renderDashboard();
  scrollTo(0, 0);
}
$$('nav.tabs button').forEach(b => b.onclick = () => selectTab(b.dataset.tab));
const jump = (id, tab) => { const el = $(id); if (el) el.onclick = e => { e.preventDefault(); selectTab(tab); }; };
jump('#aboutJump', 'about'); jump('#programJump', 'program');
{ const p = $('#policyLink'); if (p) { p.href = '/policy'; p.target = '_blank'; } }

/* ---------------- scan ---------------- */
async function sha256(buf) {
  const h = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, '0')).join('');
}

async function imageToCanvas(blob) {
  const bmp = await createImageBitmap(blob);
  const scale = Math.min(1, 1100 / bmp.width);
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * scale);
  c.height = Math.round(bmp.height * scale);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  return c;
}

function threshold() { return parseFloat($('#thresh').value) || 0.95; }

async function digitizeOffline(canvas) {
  if (!modelReady()) await loadModel('/static/numerals_cnn.json');
  const ctx = canvas.getContext('2d');
  const { cells, boxes, gray, w, h } = segment(ctx.getImageData(0, 0, canvas.width, canvas.height));
  const THRESH = threshold();
  const results = cells.map((px, i) => {
    const { digit, conf } = classify(px);
    return { digit, confidence: +conf.toFixed(4), flagged: conf < THRESH, bbox: [boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h] };
  });
  // annotated review image
  const rc = document.createElement('canvas'); rc.width = w; rc.height = h;
  const rx = rc.getContext('2d');
  const g = rx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) { g.data[i*4]=g.data[i*4+1]=g.data[i*4+2]=gray[i]; g.data[i*4+3]=255; }
  rx.putImageData(g, 0, 0);
  rx.lineWidth = 2; rx.font = '16px sans-serif';
  results.forEach(r => {
    rx.strokeStyle = rx.fillStyle = r.flagged ? '#b3261e' : '#12703a';
    rx.strokeRect(...r.bbox);
    rx.fillText(String(r.digit), r.bbox[0], r.bbox[1] - 4);
  });
  const value = results.map(r => r.flagged ? '?' : r.digit).join('');
  return { cells: results, value, needs_review: results.some(r => r.flagged) || !results.length,
           review_image: rc.toDataURL('image/png'), engine: 'on-device CNN' };
}

async function digitizeOnline(blob) {
  const fd = new FormData(); fd.append('file', blob, 'scan.png');
  const r = await fetch(`/api/scan?form=${encodeURIComponent($('#formName').value)}`
    + `&field=${encodeURIComponent($('#fieldName').value)}&threshold=${threshold()}`,
    { method: 'POST', body: fd });
  if (!r.ok) throw new Error('server scan failed');
  const j = await r.json();
  j.engine = 'server · ' + (j.engine || 'OpenCV');
  return j;
}

async function runScan(blob) {
  $('#scanOut').innerHTML = '<p class="help">Processing…</p>';
  const canvas = await imageToCanvas(blob);
  let rec;
  try {
    rec = navigator.onLine ? await digitizeOnline(blob) : await digitizeOffline(canvas);
  } catch (e) {
    rec = await digitizeOffline(canvas);            // graceful fallback
  }
  const imgSha = await sha256(await blob.arrayBuffer());
  const stored = {
    ts: Date.now(), form: $('#formName').value, field: $('#fieldName').value,
    value: rec.value, needsReview: rec.needs_review, reviewed: false,
    cells: rec.cells, imgSha, synced: false, engine: rec.engine, threshold: threshold(),
  };
  const key = await putRecord(stored);
  stored.localId = key;
  renderScanResult(stored, rec.review_image);
  $('#engineTag').textContent = 'engine: ' + rec.engine;
}

function renderScanResult(rec, reviewImg) {
  const digits = rec.cells.length
    ? rec.cells.map(c => `<span class="d ${c.flagged ? 'flag' : 'ok'}">${showDigit(c.flagged ? '?' : String(c.digit))}</span>`).join('')
    : '<span class="help">No digit boxes detected — try a straighter, closer photo.</span>';
  const conf = rec.cells.map(c => (c.confidence * 100 | 0) + '%').join('  ');
  $('#scanOut').innerHTML = `
    <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid var(--line)">
      <p><strong>#${rec.localId}</strong> · ${rec.form} / ${rec.field} ·
        ${rec.needsReview ? '<span class="pill flag">needs review</span>' : '<span class="pill ok">auto-accepted</span>'}
        <span class="help">· threshold ${Math.round((rec.threshold || 0.95) * 100)}%</span></p>
      <p class="digits">${digits}</p>
      <p class="confrow">${conf}</p>
      ${reviewImg ? `<img class="review" src="${reviewImg}" alt="annotated scan">` : ''}
      ${rec.needsReview ? '<p class="help">Saved to the review queue.</p>' : ''}
    </div>`;
}

$('#scanFile').onchange = e => e.target.files[0] && runScan(e.target.files[0]);
$('#sampleBtn').onclick = async () => {
  try {
    const b = await fetch('/api/sample-form').then(r => r.blob());
    runScan(b);
  } catch { $('#scanOut').innerHTML = '<p class="help">Sample needs a connection the first time.</p>'; }
};
$('#numeral').onchange = () => { renderRecords(); renderReview(); };

/* ---------------- review ---------------- */
async function renderReview() {
  const rows = (await allRecords()).filter(r => r.needsReview && !r.reviewed);
  if (!rows.length) { $('#reviewOut').innerHTML = '<p class="help">Nothing pending. <span class="ur">کچھ باقی نہیں۔</span></p>'; return; }
  $('#reviewOut').innerHTML = '<table><thead><tr><th>#</th><th>Form / field</th><th>Model read</th><th>Correct value</th><th></th></tr></thead><tbody>' +
    rows.map(r => {
      const d = r.cells.map(c => `<span class="d ${c.flagged ? 'flag' : 'ok'}">${showDigit(c.flagged ? '?' : String(c.digit))}</span>`).join('');
      return `<tr><td>${r.localId}</td><td>${r.form}<br><span class="help">${r.field}</span></td>
        <td><span class="digits" style="font-size:1.15rem">${d}</span></td>
        <td><input type="text" id="fix${r.localId}" value="${r.value.replace(/\?/g, '')}" inputmode="numeric"></td>
        <td><button class="btn" onclick="window.__save(${r.localId})">Save</button></td></tr>`;
    }).join('') + '</tbody></table>';
}
window.__save = async id => {
  const rec = await getRecord(id);
  rec.value = $('#fix' + id).value;
  rec.reviewed = true; rec.needsReview = false; rec.synced = false;
  await putRecord(rec);
  renderReview();
};

/* ---------------- records ---------------- */
async function renderRecords() {
  const rows = (await allRecords()).sort((a, b) => b.ts - a.ts);
  if (!rows.length) { $('#recordsOut').innerHTML = '<p class="help">No records yet.</p>'; return; }
  $('#recordsOut').innerHTML = '<table><thead><tr><th>#</th><th>When</th><th>Form / field</th><th>Value</th><th>Status</th><th>Sync</th></tr></thead><tbody>' +
    rows.map(r => `<tr>
      <td>${r.localId}</td>
      <td class="help">${new Date(r.ts).toLocaleString()}</td>
      <td>${r.form} / <span class="help">${r.field}</span></td>
      <td class="digits" style="font-size:1.15rem">${[...r.value].map(showDigit).join('')}</td>
      <td>${r.reviewed ? '<span class="pill ok">reviewed</span>' : r.needsReview ? '<span class="pill flag">pending</span>' : '<span class="pill ok">auto</span>'}</td>
      <td>${r.synced ? '✓' : '—'}</td></tr>`).join('') + '</tbody></table>';
}

function toCSV(rows) {
  const head = ['localId', 'ts', 'form', 'field', 'value', 'needsReview', 'reviewed', 'threshold', 'imgSha', 'engine'];
  return [head.join(','), ...rows.map(r => head.map(k => JSON.stringify(r[k] ?? '')).join(','))].join('\n');
}
$('#expCsv').onclick = async () => {
  const blob = new Blob([toCSV(await allRecords())], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'raqam_records.csv'; a.click();
};
$('#expXlsx').onclick = () => location.href = '/api/export.csv';  // server-side, synced data

/* ---------------- sync ---------------- */
async function trySync(manual) {
  const rows = (await allRecords()).filter(r => !r.synced);
  if (!rows.length) { if (manual) $('#syncMsg').textContent = 'Nothing to sync.'; return; }
  if (!navigator.onLine) { $('#syncMsg').textContent = 'Offline — will sync when connected.'; return; }
  try {
    const r = await fetch('/api/sync', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ records: rows }),
    });
    if (!r.ok) throw new Error();
    for (const rec of rows) { rec.synced = true; await putRecord(rec); }
    $('#syncMsg').textContent = `Synced ${rows.length}.`;
    renderRecords();
  } catch { $('#syncMsg').textContent = 'Sync failed — server unreachable.'; }
}
$('#syncBtn').onclick = () => trySync(true);

$('#wipeBtn').onclick = async () => {
  const rows = await allRecords();
  const unsynced = rows.filter(r => !r.synced).length;
  const msg = unsynced
    ? `Delete all ${rows.length} local records? ${unsynced} are NOT yet synced and will be lost.`
    : `Delete all ${rows.length} local records from this device?`;
  if (!confirm(msg)) return;
  await tx('readwrite', s => s.clear());
  $('#syncMsg').textContent = 'Local queue cleared.';
  renderRecords(); renderReview(); renderDashboard();
};

/* ---------------- dashboard ---------------- */
async function renderDashboard() {
  const rows = await allRecords();
  const total = rows.length;
  const auto = rows.filter(r => !r.needsReview).length;
  const pending = rows.filter(r => r.needsReview && !r.reviewed).length;
  const synced = rows.filter(r => r.synced).length;
  const kpi = (n, l) => `<div class="kpi"><div class="n">${n}</div><div class="l">${l}</div></div>`;
  $('#kpis').innerHTML =
    kpi(total, 'Fields digitized') +
    kpi(total ? Math.round(100 * auto / total) + '%' : '—', 'Auto-accepted (no human touch)') +
    kpi(pending, 'Awaiting review') +
    kpi(synced, 'Synced to server');
  const byForm = {};
  rows.forEach(r => { (byForm[r.form] ??= { n: 0, auto: 0 }).n++; if (!r.needsReview) byForm[r.form].auto++; });
  $('#byForm').innerHTML = Object.keys(byForm).length
    ? '<table><thead><tr><th>Form</th><th>Fields</th><th>Auto-accepted</th></tr></thead><tbody>' +
      Object.entries(byForm).map(([f, v]) => `<tr><td>${f}</td><td>${v.n}</td><td>${Math.round(100 * v.auto / v.n)}%</td></tr>`).join('') +
      '</tbody></table>'
    : '<p class="help">No data yet.</p>';
  try {
    const ev = await fetch('/api/evaluation').then(r => r.json());
    if (ev && ev.n) $('#evalOut').innerHTML =
      `<table><tbody>
        <tr><th>Scans evaluated</th><td>${ev.n}</td></tr>
        <tr><th>Digit error rate (pre-review)</th><td>${(ev.digit_error * 100).toFixed(2)}%</td></tr>
        <tr><th>Digit error rate reaching data (post-review)</th><td>${(ev.residual_error * 100).toFixed(2)}%</td></tr>
        <tr><th>Auto-accept rate</th><td>${(ev.auto_rate * 100).toFixed(1)}%</td></tr>
        <tr><th>Processing / field vs. ~25s manual entry</th><td>${(ev.sec_per_field * 1000).toFixed(0)} ms</td></tr>
      </tbody></table>`;
  } catch {}
}

/* ---------------- engine tab (learning demo) ---------------- */
const cx = $('#chart').getContext('2d');
let hist = [];
function drawChart() {
  const W = 1000, H = 130; cx.clearRect(0, 0, W, H);
  cx.fillStyle = '#fff'; cx.fillRect(0, 0, W, H);
  if (hist.length < 2) return;
  const maxL = Math.max(...hist.map(h => h.loss)) || 1;
  const line = (sel, color, scale) => {
    const pts = hist.map((h, i) => [i / (hist.length - 1) * W, H - sel(h) * scale * H * 0.92 - 4]).filter(p => !isNaN(p[1]));
    cx.strokeStyle = color; cx.lineWidth = 2; cx.beginPath();
    pts.forEach((p, i) => i ? cx.lineTo(...p) : cx.moveTo(...p)); cx.stroke();
  };
  line(h => h.loss / maxL, '#d97a2b', 1);
  line(h => h.val_acc || NaN, '#12703a', 1);
}
$('#trainBtn').onclick = () => {
  if (!navigator.onLine) { $('#trainStat').textContent = 'Training needs a connection.'; return; }
  hist = []; $('#trainBtn').disabled = true;
  const es = new EventSource('/api/train?epochs=3');
  es.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.done) { es.close(); $('#trainBtn').disabled = false; $('#trainStat').textContent = ` final validation accuracy ${(m.val_acc * 100).toFixed(2)}%`; return; }
    hist.push(m);
    $('#trainStat').textContent = ` epoch ${m.epoch} · step ${m.step} · loss ${m.loss.toFixed(3)}` + (m.val_acc ? ` · val ${(m.val_acc * 100).toFixed(1)}%` : '');
    drawChart();
  };
  es.onerror = () => { es.close(); $('#trainBtn').disabled = false; };
};
$('#dreamBtn').onclick = async () => {
  $('#dreamBtn').disabled = true; $('#dreams').textContent = 'generating…';
  try {
    const r = await fetch('/api/dreams').then(r => r.json());
    $('#dreams').innerHTML = r.tiles.map((s, d) => `<figure><img src="${s}"><figcaption>${showDigit(String(d))}</figcaption></figure>`).join('');
  } catch { $('#dreams').textContent = 'needs a connection.'; }
  $('#dreamBtn').disabled = false;
};

const pad = $('#pad'), pc = pad.getContext('2d');
pc.lineWidth = 20; pc.lineCap = 'round'; pc.strokeStyle = '#fff';
const clearPad = () => { pc.fillStyle = '#000'; pc.fillRect(0, 0, 252, 252); $('#probs').innerHTML = ''; };
clearPad(); $('#clearBtn').onclick = clearPad;
let drawing = false;
const pos = e => { const r = pad.getBoundingClientRect(); return [(e.touches?.[0]?.clientX ?? e.clientX) - r.left, (e.touches?.[0]?.clientY ?? e.clientY) - r.top]; };
pad.addEventListener('pointerdown', e => { drawing = true; const [x, y] = pos(e); pc.beginPath(); pc.moveTo(x, y); });
pad.addEventListener('pointermove', e => { if (drawing) { const [x, y] = pos(e); pc.lineTo(x, y); pc.stroke(); } });
addEventListener('pointerup', () => { if (drawing) { drawing = false; predictPad(); } });

function padTo28() {
  const s = pc.getImageData(0, 0, 252, 252).data;
  let a = 252, b = 252, c = 0, d = 0;
  for (let y = 0; y < 252; y++) for (let x = 0; x < 252; x++) if (s[(y * 252 + x) * 4] > 20) { a = Math.min(a, x); b = Math.min(b, y); c = Math.max(c, x); d = Math.max(d, y); }
  if (c < a) return new Float32Array(784);
  const gw = c - a + 1, gh = d - b + 1, sc = 20 / Math.max(gw, gh);
  const t = document.createElement('canvas'); t.width = t.height = 28;
  const tc = t.getContext('2d'); tc.fillStyle = '#000'; tc.fillRect(0, 0, 28, 28);
  const dw = Math.round(gw * sc), dh = Math.round(gh * sc);
  tc.drawImage(pad, a, b, gw, gh, (28 - dw) / 2, (28 - dh) / 2, dw, dh);
  const px = tc.getImageData(0, 0, 28, 28).data, out = new Float32Array(784);
  let mx = 0, my = 0, tot = 0;
  for (let i = 0; i < 784; i++) { out[i] = px[i * 4] / 255; mx += (i % 28) * out[i]; my += ((i / 28) | 0) * out[i]; tot += out[i]; }
  if (tot > 0) {
    const shx = Math.round(14 - mx / tot), shy = Math.round(14 - my / tot), sh = new Float32Array(784);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) { const nx = x + shx, ny = y + shy; if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) sh[ny * 28 + nx] = out[y * 28 + x]; }
    return sh;
  }
  return out;
}
async function predictPad() {
  const px = padTo28();
  let digit, probs;
  try {
    if (!modelReady()) await loadModel('/static/numerals_cnn.json');
    ({ digit, probs } = classify(px));
  } catch {
    const r = await fetch('/api/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ pixels: [...px] }) }).then(r => r.json());
    digit = r.digit; probs = r.probs;
  }
  $('#probs').innerHTML = `<p><strong>Prediction: <span class="digits" style="font-size:1.4rem">${showDigit(String(digit))}</span></strong></p>` +
    probs.map((p, d) => `<div class="bar ${d === digit ? 'top' : ''}"><b>${showDigit(String(d))}</b><i style="width:${Math.round(p * 200)}px"></i><span>${(p * 100).toFixed(1)}%</span></div>`).join('');
}

/* ---------------- service worker ---------------- */
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
trySync();
