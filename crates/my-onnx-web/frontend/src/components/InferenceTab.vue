<template>
  <div class="tab-content">
    <h2>Run Inference</h2>

    <form @submit.prevent="handleSubmit" class="form">
      <div class="form-group">
        <label for="model-id">Model ID</label>
        <input
          id="model-id"
          type="text"
          v-model="modelId"
          placeholder="Paste model ID from upload"
          required
        />
      </div>

      <div class="form-group">
        <label for="input-tensors">Input Tensors (JSON)</label>
        <textarea
          id="input-tensors"
          v-model="inputJson"
          placeholder='[
  {
    "name": "input",
    "data": [0.1, 0.2, 0.3, ...],
    "dims": [1, 3, 224, 224],
    "dtype": "Float(F32)"
  }
]'
          rows="10"
        ></textarea>
        <small class="hint">Enter input tensors as JSON array. Data should be flattened.</small>
      </div>

      <button type="submit" :disabled="loading || !modelId || !inputJson">
        {{ loading ? 'Running...' : 'Run Inference' }}
      </button>
    </form>

    <div v-if="result" :class="['result', result.type]">
      <template v-if="result.type === 'success'">
        <h3>✓ Inference Complete</h3>
        <p><strong>Time:</strong> {{ result.inferenceTime?.toFixed(2) }} ms</p>
        <h4>Outputs:</h4>
        <pre class="output-json">{{ result.outputs }}</pre>
      </template>
      <p v-else>
        <strong>Error:</strong> {{ result.message }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { apiClient } from '../api/client';

const modelId = ref('');
const inputJson = ref('');
const loading = ref(false);
const result = ref<{
  type: 'success' | 'error';
  message?: string;
  inferenceTime?: number;
  outputs?: string;
} | null>(null);

const handleSubmit = async () => {
  loading.value = true;
  result.value = null;

  try {
    const inputs = JSON.parse(inputJson.value);

    const response = await apiClient.runInference(modelId.value, { inputs });

    result.value = {
      type: 'success',
      inferenceTime: response.inference_time_ms,
      outputs: JSON.stringify(response.outputs, null, 2),
    };
  } catch (error) {
    if (error instanceof SyntaxError) {
      result.value = {
        type: 'error',
        message: 'Invalid JSON format',
      };
    } else {
      result.value = {
        type: 'error',
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.tab-content {
  padding: 30px;
}

h2 {
  margin-bottom: 20px;
  color: #333;
}

.form {
  max-width: 800px;
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

input[type="text"],
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

input:focus,
textarea:focus {
  outline: none;
  border-color: #667eea;
}

.hint {
  color: #666;
  font-size: 0.9rem;
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
  border-left: 4px solid #667eea;
}

.result.success {
  background: #efe;
  border-left-color: #4a4;
  color: #060;
}

.result.error {
  background: #fee;
  border-left-color: #f44;
  color: #c00;
}

h4 {
  margin-top: 15px;
  margin-bottom: 10px;
  color: #333;
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
