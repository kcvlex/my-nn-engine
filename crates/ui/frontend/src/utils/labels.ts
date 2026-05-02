const cache = new Map<string, string[]>();

export async function loadLabels(
  url: string,
  parseLine: (line: string) => string = (s) => s.trim(),
): Promise<string[]> {
  const cached = cache.get(url);
  if (cached) return cached;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to load labels from ${url}: HTTP ${resp.status}`);
  }
  const labels = (await resp.text()).trim().split('\n').map(parseLine);
  cache.set(url, labels);
  return labels;
}
