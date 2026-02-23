<script setup lang="ts">
import { ref } from 'vue';
import { grpcClient } from '../api/grpc_client';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { TensorProto_DataType } from '../gen/onnx.proto3_pb';

const props = defineProps<{
  backend: Backend;
}>();

const fileInput = ref<HTMLInputElement>();
const processedCanvas = ref<HTMLCanvasElement>();
const fileName = ref('');
const previewUrl = ref('');
const loading = ref(false);
const result = ref<{
  type: 'success' | 'error';
  message?: string;
  inferenceTime?: number;
  prediction?: number;
  probabilities?: number[];
  rawOutput?: string;
} | null>(null);

let tensorData: number[] = [];

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
    img.onload = () => processImage(img);
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}

function processImage(img: HTMLImageElement) {
  const canvas = processedCanvas.value;
  if (!canvas) return;

  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, 28, 28);
  ctx.drawImage(img, 0, 0, 28, 28);

  const imageData = ctx.getImageData(0, 0, 28, 28);
  const pixels = imageData.data;

  tensorData = [];
  for (let i = 0; i < pixels.length; i += 4) {
    const gray = (pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114) / 255.0;
    tensorData.push(gray);
  }
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

async function runInference(backend?: Backend) {
  if (tensorData.length === 0) return;

  loading.value = true;
  result.value = null;

  try {
    const response = await grpcClient.runInference({
      modelId: ModelId.MNIST,
      inputs: [{
        name: 'Input3',
        dims: [1n, 1n, 28n, 28n],
        dataType: TensorProto_DataType.FLOAT,
        floatData: tensorData,
      }],
      backend: backend ?? props.backend,
    });

    const output = response.outputs[0];
    const logits = output?.floatData ?? output?.doubleData ?? [];
    const probs = softmax(logits);
    const prediction = probs.indexOf(Math.max(...probs));

    result.value = {
      type: 'success',
      inferenceTime: response.inferenceTimeMs,
      prediction,
      probabilities: probs,
      rawOutput: JSON.stringify(response.outputs.map(t => ({
        name: t.name,
        dims: t.dims.map(Number),
        dataType: t.dataType,
        floatData: t.floatData,
        doubleData: t.doubleData,
        int32Data: t.int32Data,
        int64Data: t.int64Data.map(Number),
      })), null, 2),
    };
  } catch (e) {
    result.value = {
      type: 'error',
      message: e instanceof Error ? e.message : 'Unknown error',
    };
  } finally {
    loading.value = false;
  }
}

defineExpose({ runInference });
</script>

<template>
  <div class="mnist">
    <div class="upload-area">
      <input
        type="file"
        accept="image/*"
        @change="handleFileChange"
        ref="fileInput"
        hidden
      />
      <button type="button" class="upload-btn" @click="($refs.fileInput as HTMLInputElement).click()">
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
        <div class="image-box">
          <label>28x28 Grayscale</label>
          <canvas ref="processedCanvas" width="28" height="28" class="processed-canvas"></canvas>
        </div>
      </div>

      <button type="button" class="run-btn" @click="runInference()" :disabled="loading">
        {{ loading ? 'Running...' : 'Classify Digit' }}
      </button>
    </div>

    <div v-if="result" :class="['result', result.type]">
      <template v-if="result.type === 'success'">
        <p><strong>Time:</strong> {{ result.inferenceTime?.toFixed(2) }} ms</p>

        <div class="prediction">
          <span class="digit">{{ result.prediction }}</span>
          <span class="confidence">{{ ((result.probabilities?.[result.prediction!] ?? 0) * 100).toFixed(1) }}%</span>
        </div>

        <ul class="probabilities">
          <li v-for="(prob, i) in result.probabilities" :key="i" :class="{ highlight: i === result.prediction }">
            <span class="prob-label">{{ i }}</span>
            <div class="prob-bar-bg">
              <div class="prob-bar" :style="{ width: (prob * 100) + '%' }"></div>
            </div>
            <span class="prob-value">{{ (prob * 100).toFixed(1) }}%</span>
          </li>
        </ul>

        <details>
          <summary>Raw output</summary>
          <pre class="output-json">{{ result.rawOutput }}</pre>
        </details>
      </template>
      <template v-else>
        <p>{{ result.message }}</p>
      </template>
    </div>
  </div>
</template>


<style scoped>
.preview-img {
  width: 112px;
  height: 112px;
  background: #000;
  image-rendering: pixelated;
}

.processed-canvas {
  width: 112px;
  height: 112px;
  image-rendering: pixelated;
}

.digit {
  font-size: 3rem;
  font-weight: 700;
  color: #333;
}

.prob-label {
  width: 20px;
  text-align: right;
  color: #333;
}
</style>
