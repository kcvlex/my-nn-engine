<script setup lang="ts">
defineProps<{
  visible: boolean;
  success: boolean;
  inferenceTime?: number;
  errorMessage?: string;
  rawOutput?: string;
}>();
</script>

<template>
  <div v-if="visible" :class="['result', success ? 'success' : 'error']">
    <template v-if="success">
      <p><strong>Time:</strong> {{ inferenceTime?.toFixed(2) }} ms</p>
      <slot />
      <details>
        <summary>Raw output</summary>
        <pre class="output-json">{{ rawOutput }}</pre>
      </details>
    </template>
    <template v-else>
      <p>{{ errorMessage }}</p>
    </template>
  </div>
</template>
