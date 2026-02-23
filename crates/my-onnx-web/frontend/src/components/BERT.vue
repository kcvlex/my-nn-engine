<script setup lang="ts">
// BERT-SQuAD question answering model (bertsquad-12)
// Vocab: https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt
// Model info: https://github.com/onnx/models/blob/main/validated/text/machine_comprehension/bert-squad/README.md
import { ref } from 'vue';
import { grpcClient } from '../api/grpc_client';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { TensorProto_DataType } from '../gen/onnx.proto3_pb';

const MAX_SEQ_LENGTH = 256;
const MAX_QUERY_LENGTH = 64;

const props = defineProps<{
  backend: Backend;
}>();

const context = ref('Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title.');
const question = ref('Which NFL team won Super Bowl 50?');
const loading = ref(false);
const result = ref<{
  type: 'success' | 'error';
  message?: string;
  inferenceTime?: number;
  answer?: string;
  score?: number;
  rawOutput?: string;
} | null>(null);

// WordPiece tokenizer
let vocab: Map<string, number> | null = null;
let inverseVocab: string[] | null = null;

async function loadVocab(): Promise<void> {
  if (vocab) return;
  const resp = await fetch('/bert-vocab.txt');
  const text = await resp.text();
  const tokens = text.split('\n');
  vocab = new Map();
  inverseVocab = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.length > 0) {
      vocab.set(t, i);
      inverseVocab[i] = t;
    }
  }
}

function basicTokenize(text: string): string[] {
  // Lowercase + split on whitespace and punctuation
  text = text.toLowerCase();
  const tokens: string[] = [];
  let current = '';
  for (const ch of text) {
    if (isPunctuation(ch) || isWhitespace(ch)) {
      if (current) tokens.push(current);
      if (isPunctuation(ch)) tokens.push(ch);
      current = '';
    } else {
      current += ch;
    }
  }
  if (current) tokens.push(current);
  return tokens;
}

function isPunctuation(ch: string): boolean {
  const code = ch.charCodeAt(0);
  return (code >= 33 && code <= 47) || (code >= 58 && code <= 64) ||
    (code >= 91 && code <= 96) || (code >= 123 && code <= 126);
}

function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r';
}

function wordpieceTokenize(token: string): string[] {
  if (!vocab) return [token];
  const subTokens: string[] = [];
  let start = 0;
  while (start < token.length) {
    let end = token.length;
    let found = false;
    while (start < end) {
      const substr = start === 0 ? token.slice(start, end) : '##' + token.slice(start, end);
      if (vocab.has(substr)) {
        subTokens.push(substr);
        found = true;
        break;
      }
      end--;
    }
    if (!found) {
      subTokens.push('[UNK]');
      break;
    }
    start = end;
  }
  return subTokens;
}

function tokenize(text: string): string[] {
  const basicTokens = basicTokenize(text);
  const wpTokens: string[] = [];
  for (const token of basicTokens) {
    wpTokens.push(...wordpieceTokenize(token));
  }
  return wpTokens;
}

function tokensToIds(tokens: string[]): number[] {
  if (!vocab) return [];
  return tokens.map(t => vocab!.get(t) ?? vocab!.get('[UNK]')!);
}

async function runInference(backend?: Backend) {
  if (!context.value.trim() || !question.value.trim()) return;

  loading.value = true;
  result.value = null;

  try {
    await loadVocab();

    // Tokenize question and context
    let qTokens = tokenize(question.value);
    if (qTokens.length > MAX_QUERY_LENGTH) {
      qTokens = qTokens.slice(0, MAX_QUERY_LENGTH);
    }
    const cTokens = tokenize(context.value);

    // Build input: [CLS] question [SEP] context [SEP]
    const tokens = ['[CLS]', ...qTokens, '[SEP]', ...cTokens, '[SEP]'];
    const contextStart = qTokens.length + 2; // index of first context token
    const contextEnd = contextStart + cTokens.length; // exclusive

    // Pad/truncate to MAX_SEQ_LENGTH
    const paddedTokens = tokens.slice(0, MAX_SEQ_LENGTH);
    const seqLen = paddedTokens.length;

    const inputIds = new Array(MAX_SEQ_LENGTH).fill(0);
    const inputMask = new Array(MAX_SEQ_LENGTH).fill(0);
    const segmentIds = new Array(MAX_SEQ_LENGTH).fill(0);

    const ids = tokensToIds(paddedTokens);
    for (let i = 0; i < seqLen; i++) {
      inputIds[i] = ids[i];
      inputMask[i] = 1;
      // segment 0 for [CLS]+question+[SEP], segment 1 for context+[SEP]
      if (i >= contextStart) segmentIds[i] = 1;
    }

    const response = await grpcClient.runInference({
      modelId: ModelId.BERT,
      inputs: [
        {
          name: 'unique_ids_raw_output___9:0',
          dims: [1n],
          dataType: TensorProto_DataType.INT64,
          int64Data: [0n],
        },
        {
          name: 'segment_ids:0',
          dims: [1n, BigInt(MAX_SEQ_LENGTH)],
          dataType: TensorProto_DataType.INT64,
          int64Data: segmentIds.map(BigInt),
        },
        {
          name: 'input_mask:0',
          dims: [1n, BigInt(MAX_SEQ_LENGTH)],
          dataType: TensorProto_DataType.INT64,
          int64Data: inputMask.map(BigInt),
        },
        {
          name: 'input_ids:0',
          dims: [1n, BigInt(MAX_SEQ_LENGTH)],
          dataType: TensorProto_DataType.INT64,
          int64Data: inputIds.map(BigInt),
        },
      ],
      backend: backend ?? props.backend,
    });

    // Find start/end logits outputs
    // unstack:0 = start logits, unstack:1 = end logits
    let startLogits: number[] = [];
    let endLogits: number[] = [];
    for (const t of response.outputs) {
      const data = t.floatData.length > 0 ? Array.from(t.floatData) : Array.from(t.doubleData);
      if (t.name === 'unstack:0') startLogits = data;
      else if (t.name === 'unstack:1') endLogits = data;
    }

    // Find best answer span within context tokens
    const effContextEnd = Math.min(contextEnd, seqLen);
    let bestScore = -Infinity;
    let bestStart = contextStart;
    let bestEnd = contextStart;

    for (let s = contextStart; s < effContextEnd; s++) {
      for (let e = s; e < effContextEnd && e - s < 30; e++) {
        const score = startLogits[s] + endLogits[e];
        if (score > bestScore) {
          bestScore = score;
          bestStart = s;
          bestEnd = e;
        }
      }
    }

    // Convert token indices back to text
    const answerTokens = paddedTokens.slice(bestStart, bestEnd + 1);
    const answer = detokenize(answerTokens);

    result.value = {
      type: 'success',
      inferenceTime: response.inferenceTimeMs,
      answer,
      score: bestScore,
      rawOutput: JSON.stringify(response.outputs.map(t => ({
        name: t.name,
        dims: t.dims.map(Number),
        dataType: t.dataType,
        floatDataLength: t.floatData.length,
        doubleDataLength: t.doubleData.length,
        int64Data: t.int64Data.map(Number),
      })), null, 2),
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

function detokenize(tokens: string[]): string {
  let text = '';
  for (const t of tokens) {
    if (t.startsWith('##')) {
      text += t.slice(2);
    } else {
      if (text) text += ' ';
      text += t;
    }
  }
  return text;
}

defineExpose({ runInference });
</script>

<template>
  <div class="bert">
    <div class="form-group">
      <label for="bert-context">Context</label>
      <textarea
        id="bert-context"
        v-model="context"
        rows="5"
        placeholder="Enter a paragraph..."
      ></textarea>
    </div>

    <div class="form-group">
      <label for="bert-question">Question</label>
      <input
        id="bert-question"
        type="text"
        v-model="question"
        placeholder="Ask a question about the context..."
      />
    </div>

    <button type="button" class="run-btn" @click="runInference()" :disabled="loading || !context || !question">
      {{ loading ? 'Running...' : 'Ask' }}
    </button>

    <div v-if="result" :class="['result', result.type]">
      <template v-if="result.type === 'success'">
        <p><strong>Time:</strong> {{ result.inferenceTime?.toFixed(2) }} ms</p>

        <div class="answer">
          <span class="answer-label">Answer:</span>
          <span class="answer-text">{{ result.answer }}</span>
        </div>

        <details>
          <summary>Raw output</summary>
          <pre class="output-json">{{ result.rawOutput }}</pre>
        </details>
      </template>
      <template v-else>
        <p>{{ result.message }}</p>
      </template>
    </div>
  </div>
</template>

<style scoped>
.answer {
  margin: 12px 0;
}

.answer-label {
  font-weight: 600;
  color: #333;
  margin-right: 8px;
}

.answer-text {
  font-size: 1.3rem;
  font-weight: 700;
  color: #333;
}
</style>
