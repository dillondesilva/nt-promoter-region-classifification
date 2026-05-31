import { runInference, parseLogit } from './api.js';
import {
  $,
  sanitizeSequence,
  renderExamples,
  updateLength,
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

function loadSequence(seq) {
  $('seqInput').value = seq;
  updateLength(seq);
  hideResults();
  setError('');
}

function init() {
  renderExamples(loadSequence);

  $('seqInput').addEventListener('input', () => {
    updateLength(sanitizeSequence($('seqInput').value));
  });

  $('classifyBtn').addEventListener('click', classify);
  updateLength('');
}

init();
