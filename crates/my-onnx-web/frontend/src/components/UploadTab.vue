<template>
  <div class="tab-content">
    <h2>Upload ONNX Model</h2>

    <form @submit.prevent="handleSubmit" class="form">
      <div class="form-group">
        <label for="file">Model File (.onnx)</label>
        <input
          id="file"
          type="file"
          accept=".onnx"
          @change="handleFileChange"
          required
        />
      </div>

      <div class="form-group">
        <label for="target">Target</label>
        <select id="target" v-model="target">
          <option value="CPU">CPU</option>
          <option value="CUDA">CUDA (GPU)</option>
        </select>
      </div>

      <div class="form-group">
        <label class="checkbox-label">
          <input type="checkbox" v-model="optimize" />
          Enable Optimizations
        </label>
      </div>

      <button type="submit" :disabled="loading || !selectedFile">
        {{ loading ? 'Uploading...' : 'Upload & Compile' }}
      </button>
    </form>

    <div v-if="result" :class="['result', result.type]">
      <h3 v-if="result.type === 'success'">✓ {{ result.message }}</h3>
      <div v-if="result.modelId" class="model-id-section">
        <p><strong>Model ID:</strong></p>
        <code class="model-id">{{ result.modelId }}</code>
        <p class="hint">Copy this ID to use in the Inference tab.</p>
      </div>
      <p v-if="result.type === 'error'">
        <strong>Error:</strong> {{ result.message }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { apiClient } from '../api/client';
import type { Target } from '../types/api';

const selectedFile = ref<File | null>(null);
const target = ref<Target>('CPU');
const optimize = ref(true);
const loading = ref(false);
const result = ref<{
  type: 'success' | 'error';
  message: string;
  modelId?: string;
} | null>(null);

const handleFileChange = (event: Event) => {
  const input = event.target as HTMLInputElement;
  selectedFile.value = input.files?.[0] || null;
};

const handleSubmit = async () => {
  if (!selectedFile.value) return;

  loading.value = true;
  result.value = null;

  try {
    const response = await apiClient.uploadModel(
      selectedFile.value,
      target.value,
      optimize.value
    );

    result.value = {
      type: 'success',
      message: response.message,
      modelId: response.model_id,
    };
  } catch (error) {
    result.value = {
      type: 'error',
      message: error instanceof Error ? error.message : 'Unknown error',
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

h2 {
  margin-bottom: 20px;
  color: #333;
}

.form {
  max-width: 600px;
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

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
}

input[type="file"],
select {
  width: 100%;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 1rem;
  transition: border 0.3s;
}

input:focus,
select:focus {
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

.model-id-section {
  margin-top: 10px;
}

.model-id {
  display: block;
  padding: 10px;
  background: #f5f5f5;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  color: #667eea;
  font-weight: 600;
  margin: 8px 0;
  word-break: break-all;
}

.hint {
  font-size: 0.9rem;
  margin-top: 8px;
}
</style>
