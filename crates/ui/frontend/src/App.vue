<template>
  <div class="app">
    <div class="frame">
      <header class="app-header">
        <div class="brand">
          <span class="prompt">$</span>
          <h1>my-nn-engine</h1>
        </div>
        <nav class="tabs" role="tablist">
          <button
            v-for="t in TABS"
            :key="t.id"
            :class="['tab', { active: activeTab === t.id }]"
            type="button"
            role="tab"
            :aria-selected="activeTab === t.id"
            @click="activeTab = t.id"
          >
            {{ t.label }}
          </button>
        </nav>
      </header>

      <main class="container">
        <InferenceTab v-if="activeTab === 'inference'" />
        <ChatTab v-else-if="activeTab === 'chat'" />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import InferenceTab from './components/InferenceTab.vue';
import ChatTab from './components/ChatTab.vue';

type TabId = 'inference' | 'chat';
const TABS: { id: TabId; label: string }[] = [
  { id: 'inference', label: 'inference' },
  { id: 'chat', label: 'chat' },
];

const activeTab = ref<TabId>('inference');
</script>

<style scoped>
.app {
  min-height: 100vh;
  padding: 28px 24px;
  display: flex;
  justify-content: center;
}

.frame {
  width: 100%;
  max-width: 1100px;
  border: 1px solid var(--border-strong);
  background: var(--bg-raised);
  display: flex;
  flex-direction: column;
}

.app-header {
  border-bottom: 1px solid var(--border-strong);
  padding: 16px 24px 0;
  background: var(--bg);
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.prompt {
  color: var(--accent);
  font-weight: 700;
  font-size: 18px;
}

h1 {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--fg);
}

.tabs {
  display: flex;
  gap: 0;
  margin-bottom: -1px;
}

.tab {
  background: transparent;
  border: 1px solid transparent;
  border-bottom: none;
  color: var(--fg-dim);
  padding: 8px 18px;
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: lowercase;
  cursor: pointer;
}

.tab::before {
  content: '[ ';
  color: transparent;
}

.tab::after {
  content: ' ]';
  color: transparent;
}

.tab:hover:not(.active) {
  color: var(--fg);
}

.tab.active {
  color: var(--accent);
  border-color: var(--border-strong);
  background: var(--bg-raised);
}

.tab.active::before,
.tab.active::after {
  color: var(--accent);
}

.container {
  background: var(--bg-raised);
}

@media (max-width: 600px) {
  .app {
    padding: 16px 12px;
  }
  .app-header {
    padding-left: 16px;
    padding-right: 16px;
  }
}
</style>
