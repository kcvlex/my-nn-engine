<template>
  <div class="tab-content">
    <form @submit.prevent="handleSubmit" class="form">
      <div class="form-row">
        <div class="form-group">
          <label for="model-id">Model</label>
          <select id="model-id" v-model.number="modelId">
            <option :value="ModelId.MNIST">MNIST</option>
            <option :value="ModelId.RESNET">ResNet</option>
            <option :value="ModelId.YOLO">YOLO</option>
            <option :value="ModelId.BERT">BERT</option>
            <option :value="ModelId.GPT2">GPT-2</option>
          </select>
        </div>

        <div class="form-group">
          <label for="backend">Backend</label>
          <select id="backend" v-model.number="backend">
            <option :value="Backend.CPU">CPU</option>
            <option :value="Backend.CUDA">CUDA (GPU)</option>
          </select>
        </div>
      </div>

      <div class="form-group">
        <label for="input-tensor">Input Tensor (JSON)</label>
        <textarea
          id="input-tensor"
          v-model="inputJson"
          :placeholder="placeholder"
          rows="8"
        ></textarea>
      </div>

      <button type="submit" :disabled="loading || !inputJson">
        {{ loading ? 'Running...' : 'Run Inference' }}
      </button>
    </form>

    <div v-if="result" :class="['result', result.type]">
      <template v-if="result.type === 'success'">
        <h3>Inference Complete</h3>
        <p><strong>Time:</strong> {{ result.inferenceTime?.toFixed(2) }} ms</p>
        <details>
          <summary>Output Tensor</summary>
          <pre class="output-json">{{ result.output }}</pre>
        </details>
      </template>
      <template v-else>
        <h3>Error</h3>
        <p>{{ result.message }}</p>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { grpcClient } from '../api/grpc_client';
import { ModelId, Backend } from '../gen/onnx_service_pb';

const modelId = ref<ModelId>(ModelId.MNIST);
const backend = ref<Backend>(Backend.CPU);
const inputJson = ref('');
const loading = ref(false);
const result = ref<{
  type: 'success' | 'error';
  message?: string;
  inferenceTime?: number;
  output?: string;
} | null>(null);

const placeholder = `{
  "name": "input",
  "data": [0.0, 0.1, 0.2, ...],
  "dims": [1, 1, 28, 28],
  "dtype": "float32"
}`;

const handleSubmit = async () => {
  loading.value = true;
  result.value = null;

  try {
    const parsed = JSON.parse(inputJson.value);
    const response = await grpcClient.runInference({
      modelId: modelId.value,
      inputData: {
        name: parsed.name ?? '',
        dims: (parsed.dims ?? []).map((d: number) => BigInt(d)),
        dtype: parsed.dtype ?? '',
        data: parsed.data ?? [],
      },
      backend: backend.value,
    });

    const output = response.outputData;
    result.value = {
      type: 'success',
      inferenceTime: response.inferenceTimeMs,
      output: output ? JSON.stringify({
        name: output.name,
        dims: output.dims.map(Number),
        dtype: output.dtype,
        data: output.data,
      }, null, 2) : '(no output)',
    };
  } catch (error) {
    result.value = {
      type: 'error',
      message: error instanceof SyntaxError
        ? 'Invalid JSON format'
        : error instanceof Error ? error.message : 'Unknown error',
    };
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.tab-content {
  padding: 30px;
}

.form {
  max-width: 800px;
}

.form-row {
  display: flex;
  gap: 20px;
}

.form-row .form-group {
  flex: 1;
}

.form-group {
  margin-bottom: 20px;
}

label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
}

select,
textarea {
  width: 100%;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 1rem;
  transition: border 0.3s;
  font-family: inherit;
}

textarea {
  font-family: 'Courier New', monospace;
  resize: vertical;
}

select:focus,
textarea:focus {
  outline: none;
  border-color: #667eea;
}

button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 12px 30px;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.result {
  margin-top: 20px;
  padding: 15px;
  border-radius: 6px;
}

.result.success {
  background: #efe;
  border-left: 4px solid #4a4;
  color: #060;
}

.result.error {
  background: #fee;
  border-left: 4px solid #f44;
  color: #c00;
}

h3 {
  margin-bottom: 10px;
}

details {
  margin-top: 10px;
}

summary {
  cursor: pointer;
  font-weight: 600;
  margin-bottom: 8px;
}

.output-json {
  background: #2d2d2d;
  color: #f8f8f2;
  padding: 15px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.9rem;
  margin: 0;
}
</style>
