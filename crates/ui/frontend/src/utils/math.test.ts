import { describe, expect, it } from 'vitest';
import { sigmoid, softmax, topK } from './math';

describe('softmax', () => {
  it('sums to 1', () => {
    const probs = softmax([1, 2, 3, 4]);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it('is monotonic in input', () => {
    const probs = softmax([1, 2, 3, 4]);
    for (let i = 1; i < probs.length; i++) {
      expect(probs[i]).toBeGreaterThan(probs[i - 1]);
    }
  });

  it('is shift-invariant (numerical stability via max subtraction)', () => {
    const a = softmax([1000, 1001, 1002]);
    const b = softmax([0, 1, 2]);
    for (let i = 0; i < a.length; i++) {
      expect(a[i]).toBeCloseTo(b[i], 10);
    }
  });

  it('returns uniform on equal logits', () => {
    const probs = softmax([5, 5, 5, 5]);
    for (const p of probs) {
      expect(p).toBeCloseTo(0.25, 10);
    }
  });
});

describe('sigmoid', () => {
  it('maps 0 to 0.5', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5, 10);
  });

  it('saturates at large +/- inputs', () => {
    expect(sigmoid(50)).toBeCloseTo(1, 10);
    expect(sigmoid(-50)).toBeCloseTo(0, 10);
  });
});

describe('topK', () => {
  it('returns k items sorted by probability descending (default applies softmax)', () => {
    const top = topK([0, 5, 1, 4, 2, 3], 3);
    expect(top).toHaveLength(3);
    expect(top[0].id).toBe(1);
    expect(top[1].id).toBe(3);
    expect(top[2].id).toBe(5);
    expect(top[0].prob).toBeGreaterThan(top[1].prob);
    expect(top[1].prob).toBeGreaterThan(top[2].prob);
  });

  it('treats input as probabilities when applySoftmax: false', () => {
    const top = topK([0.1, 0.5, 0.4], 2, { applySoftmax: false });
    expect(top[0]).toEqual({ id: 1, prob: 0.5 });
    expect(top[1]).toEqual({ id: 2, prob: 0.4 });
  });
});
