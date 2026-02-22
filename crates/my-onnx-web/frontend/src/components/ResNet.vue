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
  topK?: { index: number; label: string; probability: number }[];
  rawOutput?: string;
} | null>(null);

// ImageNet normalization constants (mean/std per channel)
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

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
  ctx.drawImage(img, 0, 0, 224, 224);

  const imageData = ctx.getImageData(0, 0, 224, 224);
  const pixels = imageData.data;

  // NCHW layout: [1, 3, 224, 224] with ImageNet normalization
  const r: number[] = [];
  const g: number[] = [];
  const b: number[] = [];
  for (let i = 0; i < pixels.length; i += 4) {
    r.push((pixels[i] / 255.0 - MEAN[0]) / STD[0]);
    g.push((pixels[i + 1] / 255.0 - MEAN[1]) / STD[1]);
    b.push((pixels[i + 2] / 255.0 - MEAN[2]) / STD[2]);
  }
  tensorData = [...r, ...g, ...b];
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
    const [labels, response] = await Promise.all([
      getImageNetLabels(),
      grpcClient.runInference({
        modelId: ModelId.RESNET,
        inputs: [{
          name: 'data',
          dims: [1n, 3n, 224n, 224n],
          dataType: TensorProto_DataType.FLOAT,
          floatData: tensorData,
        }],
        backend: backend ?? props.backend,
      }),
    ]);

    const output = response.outputs[0];
    const logits = output?.floatData ?? output?.doubleData ?? [];
    const probs = softmax(logits);

    // Top-5 predictions
    const indexed = probs.map((p, i) => ({ index: i, probability: p }));
    indexed.sort((a, b) => b.probability - a.probability);
    const topK = indexed.slice(0, 5).map(({ index, probability }) => ({
      index,
      label: labels[index] ?? `Class ${index}`,
      probability,
    }));

    result.value = {
      type: 'success',
      inferenceTime: response.inferenceTimeMs,
      topK,
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

let cachedLabels: string[] | null = null;

async function getImageNetLabels(): Promise<string[]> {
  if (cachedLabels) return cachedLabels;
  try {
    const resp = await fetch('/synset.txt');
    const text = await resp.text();
    cachedLabels = text.trim().split('\n').map(line => {
      // Format: "n01440764 tench, Tinca tinca" → "tench"
      const desc = line.substring(line.indexOf(' ') + 1);
      return desc.split(',')[0].trim();
    });
    return cachedLabels;
  } catch {
    return Array.from({ length: 1000 }, (_, i) => `Class ${i}`);
  }
}

defineExpose({ runInference });
</script>

<template>
  <div class="resnet">
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
          <label>224x224 RGB</label>
          <canvas ref="processedCanvas" width="224" height="224" class="processed-canvas"></canvas>
        </div>
      </div>

      <button type="button" class="run-btn" @click="runInference()" :disabled="loading">
        {{ loading ? 'Running...' : 'Classify Image' }}
      </button>
    </div>

    <div v-if="result" :class="['result', result.type]">
      <template v-if="result.type === 'success'">
        <p><strong>Time:</strong> {{ result.inferenceTime?.toFixed(2) }} ms</p>

        <div class="prediction">
          <span class="top-label">{{ result.topK?.[0]?.label }}</span>
          <span class="confidence">{{ ((result.topK?.[0]?.probability ?? 0) * 100).toFixed(1) }}%</span>
        </div>

        <ul class="probabilities">
          <li v-for="(entry, i) in result.topK" :key="i" :class="{ highlight: i === 0 }">
            <span class="prob-label">{{ entry.label }}</span>
            <div class="prob-bar-bg">
              <div class="prob-bar" :style="{ width: (entry.probability * 100) + '%' }"></div>
            </div>
            <span class="prob-value">{{ (entry.probability * 100).toFixed(1) }}%</span>
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
  width: 224px;
  height: 224px;
  object-fit: contain;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
}

.processed-canvas {
  width: 224px;
  height: 224px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
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

.top-label {
  font-size: 1.5rem;
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
  width: 180px;
  text-align: right;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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
