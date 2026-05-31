import { renderHeatmapHtml } from './heatmap.js';

export function $(id) {
  return document.getElementById(id);
}

export function sanitizeSequence(raw) {
  return raw.toUpperCase().replace(/[^ATGC]/g, '');
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function labelBadge(label) {
  const positive = label === 1;
  const text = positive ? 'Label 1 · positive' : 'Label 0 · negative';
  const cls = positive ? 'positive' : 'negative';
  return `<span class="label-badge ${cls}">${text}</span>`;
}

function renderExampleCard(ex) {
  return `
    <article class="example-card" data-seq="${escapeHtml(ex.seq)}" tabindex="0" role="button">
      <div class="card-head">
        <div class="card-title">${escapeHtml(ex.title)}</div>
        ${labelBadge(ex.label)}
      </div>
      <div class="card-name">${escapeHtml(ex.name)}</div>
      <div class="card-seq">${escapeHtml(ex.seq.slice(0, 48))}…</div>
    </article>`;
}

export function renderExamples(catalog, onSelect) {
  const root = $('examplesRoot');
  root.innerHTML = catalog.sections
    .map(
      (section) => `
    <section class="example-section" aria-labelledby="section-${section.task}">
      <div class="section-header">
        <h2 class="section-title" id="section-${section.task}">${escapeHtml(section.title)}</h2>
        <p class="section-blurb">${escapeHtml(section.blurb)}</p>
      </div>
      <div class="examples-grid">
        ${section.examples.map(renderExampleCard).join('')}
      </div>
    </section>`
    )
    .join('');

  root.querySelectorAll('.example-card').forEach((card) => {
    const pick = () => onSelect(card.dataset.seq);
    card.addEventListener('click', pick);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        pick();
      }
    });
  });
}

export function setExamplesMeta(catalog) {
  const link = $('datasetLink');
  link.href = catalog.dataset_url;
  link.textContent = catalog.source;
  $('examplesMeta').textContent = `${catalog.split} split · 300 bp · regenerate with scripts/generate_promoter_examples.py`;
}

export function setExamplesLoadError(msg) {
  const el = $('examplesLoadError');
  el.hidden = !msg;
  el.textContent = msg || '';
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
