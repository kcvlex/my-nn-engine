<script setup lang="ts">
// GPT-2 text generation model
// wte.weight (LM head) fetched via GetInitializer RPC from the ONNX model
import { ref } from 'vue';
import { encode, decode } from 'gpt-tokenizer/encoding/r50k_base';
import { useInference } from '../composables/useInference';
import { useWteWeight } from '../composables/useWteWeight';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { TensorProto_DataType } from '../gen/onnx.proto3_pb';
import { topK } from '../utils/math';

const VOCAB_SIZE = 50257;
const HIDDEN_DIM = 768;
const CONTEXT_SIZE = 8; // Fixed: output shapes are baked to seq_len=8 in the ONNX model
const PAD_TOKEN = 50256; // <|endoftext|>

const props = defineProps<{
  backend: Backend;
}>();

type TokenType = {
  text: string;
  topK: { token: string; prob: number }[];
};

const prompt = ref('The quick brown fox');
const requestedPrompt = ref<string>('');
const maxTokens = ref(20);
const generatedText = ref('');
const { loading, result, run } = useInference<{
  type: 'success' | 'error';
  message?: string;
  totalTime?: number;
  tokens?: TokenType[];
}>();

const { load: loadWteWeight, projectToLogits } = useWteWeight(
  VOCAB_SIZE,
  HIDDEN_DIM,
);

async function runInference(backend?: Backend) {
  if (!prompt.value.trim()) return;

  generatedText.value = '';
  requestedPrompt.value = prompt.value;

  await run(async (client) => {
    await loadWteWeight(client);

    const allIds = encode(prompt.value);
    const allTokenInfo: TokenType[] = [];
    const startTime = performance.now();

    for (let step = 0; step < maxTokens.value; step++) {
      const realLen = Math.min(allIds.length, CONTEXT_SIZE);
      const window = allIds.slice(-realLen);
      while (window.length < CONTEXT_SIZE) {
        window.push(PAD_TOKEN);
      }

      const response = await client.runInference({
        modelId: ModelId.GPT2,
        inputs: [
          {
            name: 'input1',
            dims: [1n, 1n, BigInt(CONTEXT_SIZE)],
            dataType: TensorProto_DataType.INT64,
            int64Data: window.map(BigInt),
          },
        ],
        backend: backend ?? props.backend,
      });

      // First output: [1, 1, CONTEXT_SIZE, 768] - hidden states
      const output = response.outputs[0];
      if (!output) throw new Error('No output found');

      // Extract hidden state at last real token position
      const lastOffset = (realLen - 1) * HIDDEN_DIM;
      const hiddenState = Array.from(
        output.floatData.slice(lastOffset, lastOffset + HIDDEN_DIM),
      );

      const logits = projectToLogits(hiddenState);
      const predictions = topK(logits, 5);
      const nextId = predictions[0].id;
      const nextToken = decode([nextId]);

      allTokenInfo.push({
        text: nextToken,
        topK: predictions.map((p) => ({ token: decode([p.id]), prob: p.prob })),
      });

      generatedText.value += nextToken;
      allIds.push(nextId);

      if (nextId === PAD_TOKEN) break;
    }

    const totalTime = performance.now() - startTime;

    return {
      type: 'success' as const,
      totalTime,
      tokens: allTokenInfo,
    };
  });
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
        v-model.number="maxTokens"
        type="number"
        min="1"
        max="100"
      />
    </div>

    <button
      type="button"
      class="run-btn"
      :disabled="loading || !prompt"
      @click="runInference()"
    >
      {{ loading ? 'Generating...' : 'Generate' }}
    </button>

    <div v-if="generatedText || result" class="generation-output">
      <div v-if="generatedText || loading" class="generated-text">
        <span class="prompt-echo">{{ requestedPrompt }}</span
        ><span class="completion"
          >{{ generatedText }}<span v-if="loading" class="cursor">|</span></span
        >
      </div>

      <template v-if="result">
        <template v-if="result.type === 'success'">
          <p class="timing">
            <strong>Total time:</strong> {{ result.totalTime?.toFixed(0) }} ms
            ({{ result.tokens?.length }} tokens)
          </p>

          <details>
            <summary>Token details</summary>
            <ul class="token-list">
              <li v-for="(tok, i) in result.tokens" :key="i">
                <span class="step">#{{ i + 1 }}</span>
                <span class="chosen-token">{{ JSON.stringify(tok.text) }}</span>
                <span class="top-predictions">
                  <span
                    v-for="(p, j) in tok.topK"
                    :key="j"
                    :class="['pred', { best: j === 0 }]"
                  >
                    {{ JSON.stringify(p.token) }}
                    {{ (p.prob * 100).toFixed(1) }}%
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
input[type='number'] {
  width: 100px;
}

.generation-output {
  margin-top: 22px;
}

.generated-text {
  background: var(--bg);
  color: var(--fg);
  padding: 16px 18px;
  border: 1px solid var(--border-strong);
  border-left: 2px solid var(--accent-dim);
  font-size: 13px;
  line-height: 1.65;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.prompt-echo {
  color: var(--fg-dim);
}

.completion {
  color: var(--accent);
}

.cursor {
  animation: blink 1.05s steps(2, end) infinite;
  color: var(--accent);
}

@keyframes blink {
  to {
    opacity: 0;
  }
}

.timing {
  margin-top: 14px;
  color: var(--fg-dim);
  font-size: 11px;
  letter-spacing: 0.04em;
}

.error-msg {
  margin-top: 12px;
  color: var(--danger);
  background: var(--bg-input);
  padding: 10px 12px;
  border-left: 2px solid var(--danger);
}

.token-list {
  list-style: none;
  padding: 0;
  margin: 6px 0 0;
  font-size: 11px;
}

.token-list li {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
}

.step {
  color: var(--fg-faint);
  width: 32px;
  flex-shrink: 0;
}

.chosen-token {
  font-weight: 500;
  color: var(--accent);
  min-width: 88px;
}

.top-predictions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.pred {
  background: var(--bg-input);
  padding: 2px 6px;
  color: var(--fg-dim);
  font-size: 10px;
  border: 1px solid var(--border);
}

.pred.best {
  border-color: var(--accent-dim);
  color: var(--accent);
}
</style>
