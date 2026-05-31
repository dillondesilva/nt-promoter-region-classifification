import {
  API_URL,
  POSITIVE_THRESHOLD,
  REQUEST_TIMEOUT_MS,
  APPLY_SIGMOID_TO_RAW_LOGITS,
} from './config.js';

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export async function runInference(sequence) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(text || `Request failed (${res.status})`);
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(
        `Request timed out after ${REQUEST_TIMEOUT_MS / 1000} seconds. Please try again.`
      );
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

export function parseLogit(data) {
  const raw = data?.logits?.[0]?.[0];
  if (raw == null || Number.isNaN(Number(raw))) {
    throw new Error('Invalid response: missing logits');
  }
  const logit = Number(raw);
  const displayLogit = APPLY_SIGMOID_TO_RAW_LOGITS ? sigmoid(logit) : logit;
  return {
    displayLogit,
    isPositive: displayLogit > POSITIVE_THRESHOLD,
  };
}
