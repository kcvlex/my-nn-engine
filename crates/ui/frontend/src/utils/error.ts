/**
 * Format an unknown thrown value as a user-facing string.
 *
 * Connect-Web reports gRPC errors as `[code] message` (e.g.
 * "[internal] BERT: cudaMalloc..."). The bracketed code is noise for end
 * users since the screen already conveys "this failed", so we strip it.
 */
export function humanizeError(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  return raw.replace(/^\[\w+\]\s+/, '');
}
