const META = {};
let lastDescriptorIdxs = [];
let lastSimilar = { effnet: [], clap: [] };
let lastTextIdxs = [];

// ---------- tabs ----------
document.querySelectorAll('nav button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    for (const tab of ['descriptors', 'similarity', 'text']) {
      document.getElementById(`tab-${tab}`).classList.toggle('hidden', tab !== btn.dataset.tab);
    }
  });
});

// ---------- util ----------
function toast(msg, ok = true) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = ok ? 'var(--accent)' : 'crimson';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

function resultCard(r, i) {
  return `
    <div class="result">
      <div class="result-head">
        <span class="tid">${i + 1}. ${r.track_id}</span>
        ${'score' in r ? `<span class="score">↑ ${r.score.toFixed(3)}</span>` : ''}
      </div>
      <div class="meta">
        <span class="meta-chip">🎵 ${r.bpm.toFixed(0)} BPM</span>
        <span class="meta-chip">🎹 ${r.key}</span>
        <span class="meta-chip">💃 ${r.danceability.toFixed(2)}</span>
        <span class="meta-chip">🎤 ${r.voice_prob.toFixed(2)}</span>
        <span class="meta-chip">📊 ${r.loudness_lufs.toFixed(1)} LUFS</span>
      </div>
      <audio controls preload="none" src="${r.audio_url}"></audio>
    </div>`;
}

async function exportM3U8(name, indices) {
  if (!indices.length) { toast('No tracks to export', false); return; }
  try {
    const r = await fetch('/api/m3u8', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, indices }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    const blob = new Blob([data.content], { type: 'audio/x-mpegurl' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = data.filename;
    a.click();
    URL.revokeObjectURL(url);
    toast(`Downloaded ${data.count} tracks as ${data.filename}`);
  } catch (e) { toast(`Export failed: ${e.message}`, false); }
}

// ---------- init ----------
async function init() {
  const meta = await (await fetch('/api/meta')).json();
  Object.assign(META, meta);

  const pSel = document.getElementById('f-parent');
  for (const p of meta.parent_genres) {
    const o = document.createElement('option'); o.value = p; o.textContent = p; pSel.appendChild(o);
  }
  pSel.addEventListener('change', () => {
    const styleSel = document.getElementById('f-style');
    styleSel.innerHTML = '<option value="any">(any)</option>';
    if (pSel.value !== 'any') {
      for (const s of meta.styles_by_parent[pSel.value] || []) {
        const o = document.createElement('option'); o.value = s; o.textContent = s;
        styleSel.appendChild(o);
      }
    }
  });

  const rSel = document.getElementById('f-root');
  for (const r of meta.key_roots) {
    const o = document.createElement('option'); o.value = r; o.textContent = r; rSel.appendChild(o);
  }
  document.getElementById('f-bpmmin').value = Math.floor(meta.bpm_min);
  document.getElementById('f-bpmmax').value = Math.ceil(meta.bpm_max);

  const tracks = await (await fetch('/api/tracks')).json();
  const tSel = document.getElementById('s-track');
  for (const t of tracks) {
    const o = document.createElement('option'); o.value = t.idx; o.textContent = t.track_id;
    tSel.appendChild(o);
  }
}

// ---------- descriptors ----------
document.getElementById('f-run').addEventListener('click', async () => {
  const body = {
    bpm_min: +document.getElementById('f-bpmmin').value,
    bpm_max: +document.getElementById('f-bpmmax').value,
    voice: document.getElementById('f-voice').value,
    dance_min: +document.getElementById('f-dmin').value,
    dance_max: +document.getElementById('f-dmax').value,
    key_root: document.getElementById('f-root').value,
    key_scale: document.getElementById('f-scale').value,
    key_min_conf: +document.getElementById('f-kconf').value,
    parent_genre: document.getElementById('f-parent').value,
    style: document.getElementById('f-style').value,
    style_min_act: +document.getElementById('f-actmin').value,
    top_k: +document.getElementById('f-k').value,
  };
  const btn = document.getElementById('f-run'); btn.disabled = true;
  const status = document.getElementById('f-status'); status.textContent = 'Filtering…';
  try {
    const r = await fetch('/api/filter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    lastDescriptorIdxs = data.results.map(x => x.idx);
    status.textContent = `${data.total} tracks match; showing top ${data.results.length}.`;
    document.getElementById('f-results').innerHTML = data.results.map((r, i) => resultCard(r, i)).join('');
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; }
});

document.getElementById('f-m3u8').addEventListener('click',
  () => exportM3U8('descriptors', lastDescriptorIdxs));

// ---------- similarity ----------
document.getElementById('s-run').addEventListener('click', async () => {
  const idx = +document.getElementById('s-track').value;
  const k = +document.getElementById('s-k').value;
  const btn = document.getElementById('s-run'); btn.disabled = true;
  try {
    const r = await fetch(`/api/similar/${idx}?k=${k}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    document.getElementById('s-query-box').classList.remove('hidden');
    document.getElementById('s-query-audio').src = data.query.audio_url;
    document.getElementById('s-effnet').innerHTML = data.effnet.map((r, i) => resultCard(r, i)).join('');
    document.getElementById('s-clap').innerHTML = data.clap.map((r, i) => resultCard(r, i)).join('');
    lastSimilar = {
      effnet: data.effnet.map(x => x.idx),
      clap: data.clap.map(x => x.idx),
      query_id: data.query.track_id,
    };
  } catch (e) { toast(`Error: ${e.message}`, false); }
  finally { btn.disabled = false; }
});

document.querySelectorAll('button[data-m3u8-src]').forEach(b => b.addEventListener('click', () => {
  const src = b.dataset.m3u8Src;
  const name = `similarity_${src}_${lastSimilar.query_id || 'query'}`.replace(/[^A-Za-z0-9_-]/g, '_');
  exportM3U8(name, lastSimilar[src] || []);
}));

// ---------- text ----------
document.getElementById('t-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('t-q').value.trim();
  const k = +document.getElementById('t-k').value;
  if (!q) return;
  const btn = document.getElementById('t-btn'); btn.disabled = true;
  const status = document.getElementById('t-status'); status.textContent = 'Searching…';
  try {
    const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&k=${k}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    lastTextIdxs = data.results.map(x => x.idx);
    status.textContent = `${data.results.length} results`;
    document.getElementById('t-results').innerHTML = data.results.map((r, i) => resultCard(r, i)).join('');
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; }
});

document.querySelectorAll('.examples span').forEach(el => el.addEventListener('click', () => {
  document.getElementById('t-q').value = el.textContent;
  document.getElementById('t-form').requestSubmit();
}));

document.getElementById('t-m3u8').addEventListener('click', () => {
  exportM3U8(`text_${Date.now()}`, lastTextIdxs);
});

init();
