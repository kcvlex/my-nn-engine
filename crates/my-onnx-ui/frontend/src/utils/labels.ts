const cache = new Map<string, string[]>();

export async function loadLabels(
  url: string,
  fallbackCount: number,
  parseLine: (line: string) => string = (s) => s.trim(),
): Promise<string[]> {
  const cached = cache.get(url);
  if (cached) return cached;
  try {
    const resp = await fetch(url);
    const text = await resp.text();
    const labels = text.trim().split('\n').map(parseLine);
    cache.set(url, labels);
    return labels;
  } catch {
    return Array.from({ length: fallbackCount }, (_, i) => `Class ${i}`);
  }
}
