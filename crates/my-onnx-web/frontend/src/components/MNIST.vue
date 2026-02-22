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
.upload-area {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}

.upload-btn,
.run-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 10px 24px;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.upload-btn:hover,
.run-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.run-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.file-name {
  color: #666;
  font-size: 0.9rem;
}

.preview-section {
  margin-bottom: 20px;
}

.images {
  display: flex;
  gap: 30px;
  margin-bottom: 20px;
}

.image-box {
  text-align: center;
}

.image-box label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
  font-size: 0.9rem;
}

.preview-img {
  width: 112px;
  height: 112px;
  object-fit: contain;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  background: #000;
  image-rendering: pixelated;
}

.processed-canvas {
  width: 112px;
  height: 112px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  image-rendering: pixelated;
}

.result {
  margin-top: 20px;
  padding: 15px;
  border-radius: 6px;
}

.result.success {
  background: #efe;
  border-left: 4px solid #4a4;
  color: #060;
}

.result.error {
  background: #fee;
  border-left: 4px solid #f44;
  color: #c00;
}

.prediction {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin: 10px 0;
}

.digit {
  font-size: 3rem;
  font-weight: 700;
  color: #333;
}

.confidence {
  font-size: 1.1rem;
  color: #666;
}

.probabilities {
  list-style: none;
  padding: 0;
  margin: 10px 0;
}

.probabilities li {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
  font-size: 0.9rem;
}

.probabilities li.highlight {
  font-weight: 700;
}

.prob-label {
  width: 20px;
  text-align: right;
  color: #333;
}

.prob-bar-bg {
  flex: 1;
  height: 16px;
  background: #e0e0e0;
  border-radius: 3px;
  overflow: hidden;
}

.prob-bar {
  height: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 3px;
  transition: width 0.3s;
}

.prob-value {
  width: 50px;
  text-align: right;
  color: #666;
  font-family: 'Courier New', monospace;
}

details {
  margin-top: 10px;
}

summary {
  cursor: pointer;
  font-weight: 600;
  margin-bottom: 8px;
}

.output-json {
  background: #2d2d2d;
  color: #f8f8f2;
  padding: 15px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.9rem;
  margin: 8px 0 0;
}
</style>
