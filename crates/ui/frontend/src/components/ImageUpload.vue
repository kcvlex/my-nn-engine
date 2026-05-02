<script setup lang="ts">
import { ref } from 'vue';

defineProps<{
  runLabel: string;
  loading: boolean;
}>();

const emit = defineEmits<{
  imageLoaded: [img: HTMLImageElement];
  run: [];
}>();

const fileInput = ref<HTMLInputElement>();
const fileName = ref('');
const previewUrl = ref('');

function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  fileName.value = file.name;

  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target?.result as string;
    previewUrl.value = dataUrl;

    const img = new Image();
    img.onload = () => emit('imageLoaded', img);
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}
</script>

<template>
  <div class="upload-area">
    <input
      ref="fileInput"
      type="file"
      accept="image/*"
      hidden
      @change="handleFileChange"
    />
    <button type="button" class="upload-btn" @click="fileInput?.click()">
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
