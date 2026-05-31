import { HEATMAP_ROWS, HEATMAP_COLS } from './config.js';

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** Red–blue diverging scale; t in [0, 1] maps low→blue, high→red. */
export function weightToColor(t) {
  const x = Math.max(0, Math.min(1, t));
  if (x < 0.5) {
    const u = x * 2;
    return `rgb(${Math.round(lerp(49, 255, u))}, ${Math.round(lerp(54, 255, u))}, ${Math.round(lerp(149, 255, u))})`;
  }
  const u = (x - 0.5) * 2;
  return `rgb(${Math.round(lerp(255, 165, u))}, ${Math.round(lerp(255, 0, u))}, ${Math.round(lerp(255, 38, u))})`;
}

function isClsToken(token) {
  const t = token.trim().toLowerCase();
  return t === '<cls>' || t === '[cls]' || t === 'cls';
}

export function tokenEntries(attnWeights) {
  return Object.entries(attnWeights || {})
    .filter(([token]) => !isClsToken(token))
    .map(([token, weight]) => ({
      token,
      weight: Number(weight),
    }));
}

export function buildHeatmapRows(entries) {
  const slots = HEATMAP_ROWS * HEATMAP_COLS;
  if (entries.length === 0) {
    const empty = Array.from({ length: slots }, () => ({
      token: '',
      weight: null,
      norm: 0,
      empty: true,
    }));
    const rows = [];
    for (let r = 0; r < HEATMAP_ROWS; r++) {
      rows.push(empty.slice(r * HEATMAP_COLS, (r + 1) * HEATMAP_COLS));
    }
    return { rows, min: 0, max: 0 };
  }

  const weights = entries.map((e) => e.weight);
  const min = Math.min(...weights);
  const max = Math.max(...weights);
  const span = max - min || 1;

  const blocks = entries.slice(0, slots).map((e) => ({
    token: e.token,
    weight: e.weight,
    norm: (e.weight - min) / span,
  }));

  while (blocks.length < slots) {
    blocks.push({ token: '', weight: null, norm: 0, empty: true });
  }

  const rows = [];
  for (let r = 0; r < HEATMAP_ROWS; r++) {
    rows.push(blocks.slice(r * HEATMAP_COLS, (r + 1) * HEATMAP_COLS));
  }
  return { rows, min, max };
}

export function renderHeatmapHtml(attnWeights) {
  const entries = tokenEntries(attnWeights);
  const { rows } = buildHeatmapRows(entries);

  const grid = rows
    .map(
      (row) => `
    <div class="attn-row">
      ${row
        .map((b) => {
          if (b.empty) {
            return '<div class="attn-block empty">—</div>';
          }
          const bg = weightToColor(b.norm);
          return `<div class="attn-block" style="background:${bg}" title="${b.token}: ${b.weight.toFixed(4)}">
            <span>${b.token}<span class="weight">${b.weight.toFixed(3)}</span></span>
          </div>`;
        })
        .join('')}
    </div>`
    )
    .join('');

  return `
    <div class="heatmap-section">
      <div class="heatmap-title">Attention weights</div>
      <p class="heatmap-note">Sequence order: left to right, top to bottom.</p>
      <div class="attn-grid">${grid}</div>
      <div class="heatmap-legend">
        <span>Low</span>
        <div class="heatmap-legend-bar"></div>
        <span>High</span>
      </div>
    </div>`;
}
