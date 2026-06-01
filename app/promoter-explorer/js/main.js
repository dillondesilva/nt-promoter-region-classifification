import { runInference, parseLogit } from './api.js';
import { loadExampleCatalog } from './examples.js';
import {
  $,
  sanitizeSequence,
  renderExamples,
  setExamplesMeta,
  setExamplesLoadError,
  updateLength,
  fillSequenceFromExample,
  setLoading,
  setError,
  renderResults,
  hideResults,
} from './ui.js';

const MIN_LEN = 10;

async function classify() {
  const seq = sanitizeSequence($('seqInput').value);
  if (seq.length < MIN_LEN) {
    setError(`Enter at least ${MIN_LEN} nucleotides.`);
    return;
  }

  setError('');
  hideResults();
  setLoading(true);

  try {
    const data = await runInference(seq);
    const parsed = parseLogit(data);
    renderResults({
      ...parsed,
      attn_weights_to_tokens: data.attn_weights_to_tokens,
    });
  } catch (err) {
    setError(
      err.message ||
        'Inference failed. The API may be temporarily unavailable — try again shortly.'
    );
  } finally {
    setLoading(false);
  }
}

function loadSequence(seq, card) {
  const title = card?.querySelector('.card-title')?.textContent?.trim() || '';
  fillSequenceFromExample(seq, card, title);
}

async function init() {
  $('seqInput').addEventListener('input', () => {
    updateLength(sanitizeSequence($('seqInput').value));
    document.querySelectorAll('.example-card.is-selected').forEach((el) => {
      el.classList.remove('is-selected');
    });
    $('seqFillStatus').textContent = '';
  });
  $('classifyBtn').addEventListener('click', classify);
  updateLength('');

  try {
    const catalog = await loadExampleCatalog();
    renderExamples(catalog, loadSequence);
    setExamplesMeta(catalog);
  } catch (err) {
    setExamplesLoadError(
      err.message ||
        'Could not load examples. Serve this folder over HTTP (e.g. python -m http.server) so data/examples.json can be fetched.'
    );
  }
}

init();
