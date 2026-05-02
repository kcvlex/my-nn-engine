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
  50% {
    opacity: 0;
  }
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
