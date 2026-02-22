<script setup lang="ts">
// GPT-2 text generation model
// Tokenizer files: https://huggingface.co/openai-community/gpt2
// wte.weight extracted from the ONNX model as float16 for LM head projection
import { ref } from 'vue';
import { grpcClient } from '../api/grpc_client';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { TensorProto_DataType } from '../gen/onnx.proto3_pb';

const VOCAB_SIZE = 50257;
const HIDDEN_DIM = 768;
const CONTEXT_SIZE = 8; // Fixed: output shapes are baked to seq_len=8 in the ONNX model
const PAD_TOKEN = 50256; // <|endoftext|>

const props = defineProps<{
  backend: Backend;
}>();

const prompt = ref('The quick brown fox');
const maxTokens = ref(20);
const loading = ref(false);
const generatedText = ref('');
const result = ref<{
  type: 'success' | 'error';
  message?: string;
  totalTime?: number;
  tokens?: { text: string; topK: { token: string; prob: number }[] }[];
} | null>(null);

// --- GPT-2 BPE Tokenizer ---

let encoder: Record<string, number> | null = null;
let decoder: Record<number, string> | null = null;
let bpeMerges: [string, string][] | null = null;
let bpeRanks: Map<string, number> | null = null;

function bytesToUnicode(): [Map<number, string>, Map<string, number>] {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);    // '!' to '~'
  for (let i = 161; i <= 172; i++) bs.push(i);   // '¡' to '¬'
  for (let i = 174; i <= 255; i++) bs.push(i);   // '®' to 'ÿ'
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteToChar = new Map<number, string>();
  const charToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i]);
    byteToChar.set(bs[i], ch);
    charToByte.set(ch, bs[i]);
  }
  return [byteToChar, charToByte];
}

const [byteEncoder] = bytesToUnicode();

function getTextEncoder(): TextEncoder {
  return new TextEncoder();
}

function encodeStr(text: string): string {
  const bytes = getTextEncoder().encode(text);
  return Array.from(bytes).map(b => byteEncoder.get(b)!).join('');
}

function getPairs(word: string[]): Set<string> {
  const pairs = new Set<string>();
  for (let i = 0; i < word.length - 1; i++) {
    pairs.add(word[i] + '\0' + word[i + 1]);
  }
  return pairs;
}

function bpe(token: string): string[] {
  if (!bpeRanks) return [token];
  let word = token.split('');
  if (word.length <= 1) return word;

  while (true) {
    let pairs = getPairs(word);
    let minRank = Infinity;
    let bestPair = '';
    for (const p of pairs) {
      const rank = bpeRanks.get(p);
      if (rank !== undefined && rank < minRank) {
        minRank = rank;
        bestPair = p;
      }
    }
    if (minRank === Infinity) break;

    const [first, second] = bestPair.split('\0');
    const newWord: string[] = [];
    let i = 0;
    while (i < word.length) {
      const j = word.indexOf(first, i);
      if (j === -1) {
        newWord.push(...word.slice(i));
        break;
      }
      newWord.push(...word.slice(i, j));
      if (j < word.length - 1 && word[j] === first && word[j + 1] === second) {
        newWord.push(first + second);
        i = j + 2;
      } else {
        newWord.push(word[j]);
        i = j + 1;
      }
    }
    word = newWord;
    if (word.length === 1) break;
  }
  return word;
}

const GPT2_PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+/g;

function tokenize(text: string): number[] {
  if (!encoder) return [];
  const ids: number[] = [];
  const matches = text.match(GPT2_PAT) || [];
  for (const m of matches) {
    const encoded = encodeStr(m);
    const bpeTokens = bpe(encoded);
    for (const t of bpeTokens) {
      const id = encoder[t];
      if (id !== undefined) ids.push(id);
    }
  }
  return ids;
}

function decodeTokens(ids: number[]): string {
  if (!decoder) return '';
  const [, charToByte] = bytesToUnicode();
  const chars = ids.map(id => decoder![id] ?? '').join('');
  const bytes = new Uint8Array(chars.split('').map(c => charToByte.get(c) ?? 0));
  return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
}

function decodeToken(id: number): string {
  return decodeTokens([id]);
}

async function loadTokenizer(): Promise<void> {
  if (encoder) return;
  const [encResp, mergesResp] = await Promise.all([
    fetch('/gpt2-encoder.json'),
    fetch('/gpt2-merges.txt'),
  ]);
  encoder = await encResp.json();
  decoder = {};
  for (const [k, v] of Object.entries(encoder!)) {
    decoder[v as number] = k;
  }
  const mergesText = await mergesResp.text();
  const lines = mergesText.split('\n').slice(1); // skip header
  bpeMerges = [];
  bpeRanks = new Map();
  for (let i = 0; i < lines.length; i++) {
    const parts = lines[i].split(' ');
    if (parts.length === 2) {
      bpeMerges.push([parts[0], parts[1]]);
      bpeRanks.set(parts[0] + '\0' + parts[1], i);
    }
  }
}

// --- LM Head (wte.weight) ---

let wteWeight: Float32Array | null = null;

async function loadWteWeight(): Promise<void> {
  if (wteWeight) return;
  const resp = await fetch('/gpt2-wte-f16.bin');
  const buf = await resp.arrayBuffer();
  const f16 = new Uint16Array(buf);
  // Convert float16 to float32
  wteWeight = new Float32Array(f16.length);
  for (let i = 0; i < f16.length; i++) {
    wteWeight[i] = float16ToFloat32(f16[i]);
  }
}

function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // subnormal
    const val = frac / 1024 * Math.pow(2, -14);
    return sign ? -val : val;
  }
  if (exp === 31) {
    return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  const val = Math.pow(2, exp - 15) * (1 + frac / 1024);
  return sign ? -val : val;
}

function projectToLogits(hiddenState: number[]): number[] {
  if (!wteWeight) return [];
  // hiddenState: [HIDDEN_DIM], wteWeight: [VOCAB_SIZE, HIDDEN_DIM]
  // logits[v] = sum(hiddenState[i] * wteWeight[v * HIDDEN_DIM + i])
  const logits = new Float64Array(VOCAB_SIZE);
  for (let v = 0; v < VOCAB_SIZE; v++) {
    let sum = 0;
    const offset = v * HIDDEN_DIM;
    for (let i = 0; i < HIDDEN_DIM; i++) {
      sum += hiddenState[i] * wteWeight[offset + i];
    }
    logits[v] = sum;
  }
  return Array.from(logits);
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function topK(logits: number[], k: number): { id: number; prob: number }[] {
  const probs = softmax(logits);
  const indexed = probs.map((p, i) => ({ id: i, prob: p }));
  indexed.sort((a, b) => b.prob - a.prob);
  return indexed.slice(0, k);
}

async function runInference(backend?: Backend) {
  if (!prompt.value.trim()) return;

  loading.value = true;
  result.value = null;
  generatedText.value = '';

  try {
    await Promise.all([loadTokenizer(), loadWteWeight()]);

    const allIds = tokenize(prompt.value);
    const allTokenInfo: { text: string; topK: { token: string; prob: number }[] }[] = [];
    const startTime = performance.now();

    for (let step = 0; step < maxTokens.value; step++) {
      // Sliding window: take last CONTEXT_SIZE tokens, right-pad to CONTEXT_SIZE
      const window = allIds.length <= CONTEXT_SIZE
        ? allIds.slice()
        : allIds.slice(allIds.length - CONTEXT_SIZE);
      const realLen = window.length;
      while (window.length < CONTEXT_SIZE) {
        window.push(PAD_TOKEN);
      }

      const response = await grpcClient.runInference({
        modelId: ModelId.GPT2,
        inputs: [{
          name: 'input1',
          dims: [1n, 1n, BigInt(CONTEXT_SIZE)],
          dataType: TensorProto_DataType.INT64,
          int64Data: window.map(BigInt),
        }],
        backend: backend ?? props.backend,
      });

      // First output: [1, 1, CONTEXT_SIZE, 768] - hidden states
      const output = response.outputs[0];
      if (!output) throw new Error('No output found');

      const data = output.floatData.length > 0 ? output.floatData : output.doubleData;
      // Extract hidden state at last real token position
      const lastOffset = (realLen - 1) * HIDDEN_DIM;
      const hiddenState = Array.from(data.slice(lastOffset, lastOffset + HIDDEN_DIM));

      const logits = projectToLogits(hiddenState);
      const predictions = topK(logits, 5);
      const nextId = predictions[0].id;
      const nextToken = decodeToken(nextId);

      allTokenInfo.push({
        text: nextToken,
        topK: predictions.map(p => ({ token: decodeToken(p.id), prob: p.prob })),
      });

      generatedText.value += nextToken;
      allIds.push(nextId);

      // Stop on <|endoftext|> (token 50256)
      if (nextId === 50256) break;
    }

    const totalTime = performance.now() - startTime;

    result.value = {
      type: 'success',
      totalTime,
      tokens: allTokenInfo,
    };
  } catch (e) {
    result.value = {
      type: 'error',
      message: e instanceof Error ? e.message : 'Unknown error',
    };
  } finally {
    loading.value = false;
  }
}

defineExpose({ runInference });
</script>

<template>
  <div class="gpt2">
    <div class="form-group">
      <label for="gpt2-prompt">Prompt</label>
      <textarea
        id="gpt2-prompt"
        v-model="prompt"
        rows="3"
        placeholder="Enter a text prompt..."
      ></textarea>
    </div>

    <div class="form-group">
      <label for="gpt2-max-tokens">Max tokens</label>
      <input
        id="gpt2-max-tokens"
        type="number"
        v-model.number="maxTokens"
        min="1"
        max="100"
      />
    </div>

    <button type="button" class="run-btn" @click="runInference()" :disabled="loading || !prompt">
      {{ loading ? 'Generating...' : 'Generate' }}
    </button>

    <div v-if="generatedText || result" class="generation-output">
      <div v-if="generatedText || loading" class="generated-text">
        <span class="prompt-echo">{{ prompt }}</span><span class="completion">{{ generatedText }}<span v-if="loading" class="cursor">|</span></span>
      </div>

      <template v-if="result">
        <template v-if="result.type === 'success'">
          <p class="timing"><strong>Total time:</strong> {{ result.totalTime?.toFixed(0) }} ms ({{ result.tokens?.length }} tokens)</p>

          <details>
            <summary>Token details</summary>
            <ul class="token-list">
              <li v-for="(tok, i) in result.tokens" :key="i">
                <span class="step">#{{ i + 1 }}</span>
                <span class="chosen-token">{{ JSON.stringify(tok.text) }}</span>
                <span class="top-predictions">
                  <span v-for="(p, j) in tok.topK" :key="j" :class="['pred', { best: j === 0 }]">
                    {{ JSON.stringify(p.token) }} {{ (p.prob * 100).toFixed(1) }}%
                  </span>
                </span>
              </li>
            </ul>
          </details>
        </template>
        <template v-else>
          <p class="error-msg">{{ result.message }}</p>
        </template>
      </template>
    </div>
  </div>
</template>

<style scoped>
.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
  font-size: 0.9rem;
}

textarea,
input[type="number"] {
  width: 100%;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 1rem;
  font-family: inherit;
  transition: border 0.3s;
  box-sizing: border-box;
}

input[type="number"] {
  width: 100px;
}

textarea {
  resize: vertical;
}

textarea:focus,
input[type="number"]:focus {
  outline: none;
  border-color: #667eea;
}

.run-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 10px 24px;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.run-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.run-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.generation-output {
  margin-top: 20px;
}

.generated-text {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 20px;
  border-radius: 6px;
  font-family: 'Courier New', monospace;
  font-size: 1rem;
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.prompt-echo {
  color: #888;
}

.completion {
  color: #4ec9b0;
}

.cursor {
  animation: blink 1s step-end infinite;
  color: #fff;
}

@keyframes blink {
  50% { opacity: 0; }
}

.timing {
  margin-top: 12px;
  color: #333;
  font-size: 0.9rem;
}

.error-msg {
  margin-top: 12px;
  color: #c00;
  background: #fee;
  padding: 10px;
  border-radius: 6px;
  border-left: 4px solid #f44;
}

details {
  margin-top: 10px;
}

summary {
  cursor: pointer;
  font-weight: 600;
  margin-bottom: 8px;
}

.token-list {
  list-style: none;
  padding: 0;
  margin: 0;
  font-size: 0.85rem;
}

.token-list li {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 4px 0;
  border-bottom: 1px solid #eee;
}

.step {
  color: #999;
  width: 30px;
  flex-shrink: 0;
}

.chosen-token {
  font-weight: 700;
  color: #333;
  font-family: 'Courier New', monospace;
  min-width: 80px;
}

.top-predictions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.pred {
  background: #f0f0f0;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'Courier New', monospace;
  color: #666;
  font-size: 0.8rem;
}

.pred.best {
  background: #e0f0e0;
  color: #060;
  font-weight: 600;
}
</style>
