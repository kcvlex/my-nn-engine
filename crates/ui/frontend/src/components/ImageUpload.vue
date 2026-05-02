<script setup lang="ts">
import { computed, watch } from 'vue';
import { useFileDialog, useObjectUrl } from '@vueuse/core';

defineProps<{
  runLabel: string;
  loading: boolean;
}>();

const emit = defineEmits<{
  imageLoaded: [img: HTMLImageElement];
  run: [];
}>();

const { files, open } = useFileDialog({
  accept: 'image/*',
  multiple: false,
});

const file = computed<File | null>(() => files.value?.[0] ?? null);
const fileName = computed(() => file.value?.name ?? '');
const previewUrl = useObjectUrl(file);

watch(previewUrl, (url) => {
  if (!url) return;
  const img = new Image();
  img.onload = () => emit('imageLoaded', img);
  img.src = url;
});
</script>

<template>
  <div class="upload-area">
    <button type="button" class="upload-btn" @click="open()">
      Choose Image
    </button>
    <span v-if="fileName" class="file-name">{{ fileName }}</span>
  </div>

  <div v-if="previewUrl" class="preview-section">
    <div class="images">
      <div class="image-box">
        <label>Original</label>
        <img :src="previewUrl" class="preview-img" />
      </div>
      <slot name="canvas" />
    </div>

    <button
      type="button"
      class="run-btn"
      :disabled="loading"
      @click="$emit('run')"
    >
      {{ loading ? 'Running...' : runLabel }}
    </button>
  </div>
</template>
