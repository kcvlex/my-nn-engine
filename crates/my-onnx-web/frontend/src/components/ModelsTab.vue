<template>
  <div class="tab-content">
    <h2>Cached Models</h2>

    <button @click="loadModels" :disabled="loading" class="refresh-btn">
      {{ loading ? 'Loading...' : 'Refresh' }}
    </button>

    <div v-if="error" class="result error">
      <strong>Error:</strong> {{ error }}
    </div>

    <div v-else-if="models.length === 0 && !loading" class="empty-state">
      No models uploaded yet.
    </div>

    <ul v-else class="model-list">
      <li v-for="model in models" :key="model.model_id" class="model-item">
        <div class="model-info">
          <div class="model-id">{{ model.model_id }}</div>
          <small>Uploaded: {{ model.uploaded_at }}</small>
        </div>
        <span class="model-status">{{ model.status }}</span>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { apiClient } from '../api/client';
import type { ModelListItem } from '../types/api';

const models = ref<ModelListItem[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);

const loadModels = async () => {
  loading.value = true;
  error.value = null;

  try {
    models.value = await apiClient.listModels();
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load models';
  } finally {
    loading.value = false;
  }
};

onMounted(() => {
  loadModels();
});

defineExpose({
  loadModels,
});
</script>

<style scoped>
.tab-content {
  padding: 30px;
}

h2 {
  margin-bottom: 20px;
  color: #333;
}

.refresh-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  margin-bottom: 20px;
}

.refresh-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.result {
  padding: 15px;
  border-radius: 6px;
  border-left: 4px solid #f44;
}

.result.error {
  background: #fee;
  color: #c00;
}

.empty-state {
  color: #666;
  padding: 40px;
  text-align: center;
  font-size: 1.1rem;
}

.model-list {
  list-style: none;
  padding: 0;
}

.model-item {
  padding: 15px;
  margin-bottom: 10px;
  background: #f9f9f9;
  border-radius: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background 0.2s;
}

.model-item:hover {
  background: #f0f0f0;
}

.model-info {
  flex: 1;
}

.model-id {
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  color: #667eea;
  font-weight: 600;
  margin-bottom: 5px;
  word-break: break-all;
}

small {
  color: #666;
  font-size: 0.85rem;
}

.model-status {
  padding: 4px 12px;
  background: #4caf50;
  color: white;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
  white-space: nowrap;
}
</style>
