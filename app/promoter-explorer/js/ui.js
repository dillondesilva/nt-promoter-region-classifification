import { EXAMPLES } from './examples.js';
import { renderHeatmapHtml } from './heatmap.js';

export function $(id) {
  return document.getElementById(id);
}

export function sanitizeSequence(raw) {
  return raw.toUpperCase().replace(/[^ATGC]/g, '');
}

export function renderExamples(onSelect) {
  const grid = $('examplesGrid');
  grid.innerHTML = EXAMPLES.map(
    (ex, i) => `
    <article class="example-card" data-idx="${i}" tabindex="0" role="button">
      <div class="card-title">${ex.title}</div>
      <div class="card-desc">${ex.desc}</div>
      <div class="card-seq">${ex.seq.slice(0, 48)}…</div>
    </article>`
  ).join('');

  grid.querySelectorAll('.example-card').forEach((card) => {
    const idx = Number(card.dataset.idx);
    const pick = () => onSelect(EXAMPLES[idx].seq);
    card.addEventListener('click', pick);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        pick();
      }
    });
  });
}

export function updateLength(seq) {
  $('seqLength').textContent = `${seq.length} bp`;
}

export function setLoading(on) {
  $('classifyBtn').disabled = on;
  $('classifyBtn').textContent = on ? 'Running…' : 'Classify';
  $('loadingBar').hidden = !on;
  $('waitHint').hidden = !on;
}

export function setError(msg) {
  const el = $('errorMsg');
  if (!msg) {
    el.hidden = true;
    el.textContent = '';
    return;
  }
  el.hidden = false;
  el.textContent = msg;
}

export function renderResults({ displayLogit, isPositive, attn_weights_to_tokens }) {
  const label = isPositive ? 'POSITIVE' : 'NEGATIVE';
  const cls = isPositive ? 'positive' : 'negative';

  $('results').innerHTML = `
    <div class="results-header">Classification result</div>
    <div class="result-card">
      <div class="prediction-row">
        <span class="prediction-label ${cls}">${label}</span>
        <span class="logit-badge">Logit: ${displayLogit.toFixed(4)}</span>
      </div>
      ${renderHeatmapHtml(attn_weights_to_tokens)}
    </div>`;

  $('results').hidden = false;
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

export function hideResults() {
  $('results').hidden = true;
}
