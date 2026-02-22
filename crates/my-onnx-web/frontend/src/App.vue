<template>
  <div class="app">
    <div class="container">
      <header>
        <h1>My ONNX</h1>
        <p class="subtitle">Run inference on pre-loaded ONNX models via gRPC</p>
      </header>

      <InferenceTab />
    </div>
  </div>
</template>

<script setup lang="ts">
import InferenceTab from './components/InferenceTab.vue';
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
</style>
