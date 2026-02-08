<template>
  <div class="app">
    <div class="container">
      <header>
        <h1>🚀 ONNX Inference Server</h1>
        <p class="subtitle">Upload ONNX models and run inference locally</p>
      </header>

      <div class="tabs">
        <button
          v-for="tab in tabs"
          :key="tab.id"
          :class="['tab', { active: activeTab === tab.id }]"
          @click="setActiveTab(tab.id)"
        >
          {{ tab.label }}
        </button>
      </div>

      <UploadTab v-show="activeTab === 'upload'" />
      <InferenceTab v-show="activeTab === 'inference'" />
      <ModelsTab ref="modelsTabRef" v-show="activeTab === 'models'" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import UploadTab from './components/UploadTab.vue';
import InferenceTab from './components/InferenceTab.vue';
import ModelsTab from './components/ModelsTab.vue';

type TabId = 'upload' | 'inference' | 'models';

interface Tab {
  id: TabId;
  label: string;
}

const tabs: Tab[] = [
  { id: 'upload', label: 'Upload Model' },
  { id: 'inference', label: 'Run Inference' },
  { id: 'models', label: 'My Models' },
];

const activeTab = ref<TabId>('upload');
const modelsTabRef = ref<InstanceType<typeof ModelsTab> | null>(null);

const setActiveTab = (tabId: TabId) => {
  activeTab.value = tabId;
};

watch(activeTab, (newTab) => {
  if (newTab === 'models' && modelsTabRef.value) {
    modelsTabRef.value.loadModels();
  }
});
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
    Ubuntu, Cantarell, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  padding: 20px;
}

#app {
  width: 100%;
}
</style>

<style scoped>
.app {
  max-width: 1200px;
  margin: 0 auto;
}

.container {
  background: white;
  border-radius: 12px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  overflow: hidden;
}

header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 30px;
  text-align: center;
}

h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
}

.subtitle {
  opacity: 0.9;
  font-size: 1.1rem;
}

.tabs {
  display: flex;
  background: #f5f5f5;
  border-bottom: 2px solid #e0e0e0;
}

.tab {
  flex: 1;
  padding: 15px;
  text-align: center;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s;
  border: none;
  background: transparent;
  font-size: 1rem;
  color: #333;
}

.tab:hover {
  background: rgba(102, 126, 234, 0.1);
}

.tab.active {
  background: white;
  border-bottom: 3px solid #667eea;
  color: #667eea;
}
</style>
