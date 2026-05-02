import { describe, expect, it } from 'vitest';
import { humanizeError } from './error';

describe('humanizeError', () => {
  it('strips Connect [code] prefix from Error messages', () => {
    expect(humanizeError(new Error('[internal] BERT: cudaMalloc failed'))).toBe(
      'BERT: cudaMalloc failed',
    );
  });

  it('strips Connect [code] prefix from string messages', () => {
    expect(humanizeError('[unavailable] grpc connection refused')).toBe(
      'grpc connection refused',
    );
  });

  it('passes through messages without a bracket prefix', () => {
    expect(humanizeError('plain message')).toBe('plain message');
  });

  it('coerces non-Error / non-string thrown values', () => {
    expect(humanizeError(42)).toBe('42');
    expect(humanizeError(null)).toBe('null');
  });
});
