<template>
  <div class="app">
    <div class="frame">
      <header class="app-header">
        <div class="brand">
          <span class="prompt">$</span>
          <h1>my-nn-engine</h1>
        </div>
      </header>

      <main class="container">
        <InferenceTab />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import InferenceTab from './components/InferenceTab.vue';
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
  padding: 16px 24px;
  background: var(--bg);
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
