export function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function topK(
  values: number[],
  k: number,
  opts: { applySoftmax?: boolean } = {},
): { id: number; prob: number }[] {
  const applySoftmax = opts.applySoftmax ?? true;
  const probs = applySoftmax ? softmax(values) : values;
  const indexed = probs.map((p, i) => ({ id: i, prob: p }));
  indexed.sort((a, b) => b.prob - a.prob);
  return indexed.slice(0, k);
}
