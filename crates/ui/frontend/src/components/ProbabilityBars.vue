<script setup lang="ts">
withDefaults(
  defineProps<{
    items: { label: string; probability: number; highlight: boolean }[];
    labelWidth?: string;
    truncateLabel?: boolean;
  }>(),
  {
    labelWidth: '20px',
    truncateLabel: false,
  },
);
</script>

<template>
  <ul class="probabilities">
    <li
      v-for="(item, i) in items"
      :key="i"
      :class="{ highlight: item.highlight }"
    >
      <span
        class="prob-label"
        :style="{
          width: labelWidth,
          overflow: truncateLabel ? 'hidden' : undefined,
          textOverflow: truncateLabel ? 'ellipsis' : undefined,
          whiteSpace: truncateLabel ? 'nowrap' : undefined,
        }"
        >{{ item.label }}</span
      >
      <div class="prob-bar-bg">
        <div
          class="prob-bar"
          :style="{ width: item.probability * 100 + '%' }"
        ></div>
      </div>
      <span class="prob-value">{{ (item.probability * 100).toFixed(1) }}%</span>
    </li>
  </ul>
</template>

<style scoped>
.prob-label {
  text-align: right;
}
</style>
