<script setup lang="ts">
// COCO labels from: https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/data/classes/coco.names
import { ref } from 'vue';
import { grpcClient } from '../api/grpc_client';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { TensorProto_DataType } from '../gen/onnx.proto3_pb';

const INPUT_SIZE = 416;

// YOLOv4 anchors (from hunglc007/tensorflow-yolov4-tflite)
const ANCHORS = [
  [[12, 16], [19, 36], [40, 28]],     // stride 8  (52x52)
  [[36, 75], [76, 55], [72, 146]],     // stride 16 (26x26)
  [[142, 110], [192, 243], [459, 401]], // stride 32 (13x13)
];
const STRIDES = [8, 16, 32];

const SCORE_THRESHOLD = 0.25;
const NMS_THRESHOLD = 0.45;

const props = defineProps<{
  backend: Backend;
}>();

const fileInput = ref<HTMLInputElement>();
const processedCanvas = ref<HTMLCanvasElement>();
const resultCanvas = ref<HTMLCanvasElement>();
const fileName = ref('');
const previewUrl = ref('');
const loading = ref(false);
const result = ref<{
  type: 'success' | 'error';
  message?: string;
  inferenceTime?: number;
  detections?: Detection[];
  rawOutput?: string;
} | null>(null);

interface Detection {
  x1: number; y1: number; x2: number; y2: number;
  score: number;
  classId: number;
  label: string;
}

let tensorData: number[] = [];
let originalImg: HTMLImageElement | null = null;

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
    img.onload = () => {
      originalImg = img;
      processImage(img);
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}

function processImage(img: HTMLImageElement) {
  const canvas = processedCanvas.value;
  if (!canvas) return;

  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);

  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = imageData.data;

  // NHWC layout: [1, 416, 416, 3], normalized to [0, 1]
  tensorData = [];
  for (let i = 0; i < pixels.length; i += 4) {
    tensorData.push(pixels[i] / 255.0);
    tensorData.push(pixels[i + 1] / 255.0);
    tensorData.push(pixels[i + 2] / 255.0);
  }
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function decodeDetections(
  outputs: { floatData: number[]; doubleData: number[] }[],
): Detection[] {
  const boxes: Detection[] = [];

  for (let scaleIdx = 0; scaleIdx < 3; scaleIdx++) {
    const output = outputs[scaleIdx];
    const data = output.floatData.length > 0
      ? Array.from(output.floatData)
      : Array.from(output.doubleData);
    const stride = STRIDES[scaleIdx];
    const gridSize = INPUT_SIZE / stride;
    const anchors = ANCHORS[scaleIdx];

    for (let cy = 0; cy < gridSize; cy++) {
      for (let cx = 0; cx < gridSize; cx++) {
        for (let a = 0; a < 3; a++) {
          const offset = ((cy * gridSize + cx) * 3 + a) * 85;

          const tx = data[offset];
          const ty = data[offset + 1];
          const tw = data[offset + 2];
          const th = data[offset + 3];
          const objectness = sigmoid(data[offset + 4]);

          if (objectness < SCORE_THRESHOLD) continue;

          // Decode box center and size
          const bx = (sigmoid(tx) + cx) * stride;
          const by = (sigmoid(ty) + cy) * stride;
          const bw = Math.exp(tw) * anchors[a][0];
          const bh = Math.exp(th) * anchors[a][1];

          // Find best class
          let bestClassId = 0;
          let bestClassScore = -Infinity;
          for (let c = 0; c < 80; c++) {
            const score = data[offset + 5 + c];
            if (score > bestClassScore) {
              bestClassScore = score;
              bestClassId = c;
            }
          }
          const classProb = sigmoid(bestClassScore);
          const finalScore = objectness * classProb;

          if (finalScore < SCORE_THRESHOLD) continue;

          boxes.push({
            x1: bx - bw / 2,
            y1: by - bh / 2,
            x2: bx + bw / 2,
            y2: by + bh / 2,
            score: finalScore,
            classId: bestClassId,
            label: '',
          });
        }
      }
    }
  }

  return nms(boxes);
}

function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function nms(boxes: Detection[]): Detection[] {
  boxes.sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < boxes.length; i++) {
    if (suppressed.has(i)) continue;
    keep.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (!suppressed.has(j) && iou(boxes[i], boxes[j]) > NMS_THRESHOLD) {
        suppressed.add(j);
      }
    }
  }
  return keep;
}

function drawDetections(detections: Detection[]) {
  const canvas = resultCanvas.value;
  if (!canvas || !originalImg) return;

  const img = originalImg;
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  const scaleX = img.width / INPUT_SIZE;
  const scaleY = img.height / INPUT_SIZE;

  for (const det of detections) {
    const x = det.x1 * scaleX;
    const y = det.y1 * scaleY;
    const w = (det.x2 - det.x1) * scaleX;
    const h = (det.y2 - det.y1) * scaleY;

    // Draw box
    const hue = (det.classId * 37) % 360;
    ctx.strokeStyle = `hsl(${hue}, 80%, 50%)`;
    ctx.lineWidth = Math.max(2, Math.min(img.width, img.height) / 200);
    ctx.strokeRect(x, y, w, h);

    // Draw label
    const label = `${det.label} ${(det.score * 100).toFixed(0)}%`;
    const fontSize = Math.max(12, Math.min(img.width, img.height) / 40);
    ctx.font = `bold ${fontSize}px sans-serif`;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
    ctx.fillRect(x, y - fontSize - 4, textWidth + 8, fontSize + 4);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 4, y - 4);
  }
}

let cachedLabels: string[] | null = null;

async function getCocoLabels(): Promise<string[]> {
  if (cachedLabels) return cachedLabels;
  try {
    const resp = await fetch('/coco.names');
    const text = await resp.text();
    cachedLabels = text.trim().split('\n').map(s => s.trim());
    return cachedLabels;
  } catch {
    return Array.from({ length: 80 }, (_, i) => `Class ${i}`);
  }
}

async function runInference(backend?: Backend) {
  if (tensorData.length === 0) return;

  loading.value = true;
  result.value = null;

  try {
    const [labels, response] = await Promise.all([
      getCocoLabels(),
      grpcClient.runInference({
        modelId: ModelId.YOLO,
        inputs: [{
          name: 'input_1:0',
          dims: [1n, BigInt(INPUT_SIZE), BigInt(INPUT_SIZE), 3n],
          dataType: TensorProto_DataType.FLOAT,
          floatData: tensorData,
        }],
        backend: backend ?? props.backend,
      }),
    ]);

    const outputs = response.outputs.map(t => ({
      floatData: Array.from(t.floatData),
      doubleData: Array.from(t.doubleData),
    }));

    const detections = decodeDetections(outputs);
    for (const det of detections) {
      det.label = labels[det.classId] ?? `Class ${det.classId}`;
    }

    drawDetections(detections);

    result.value = {
      type: 'success',
      inferenceTime: response.inferenceTimeMs,
      detections,
      rawOutput: JSON.stringify(response.outputs.map(t => ({
        name: t.name,
        dims: t.dims.map(Number),
        dataType: t.dataType,
        floatDataLength: t.floatData.length,
        doubleDataLength: t.doubleData.length,
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
  <div class="yolo">
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
          <label>416x416 RGB</label>
          <canvas ref="processedCanvas" :width="INPUT_SIZE" :height="INPUT_SIZE" class="processed-canvas"></canvas>
        </div>
      </div>

      <button type="button" class="run-btn" @click="runInference()" :disabled="loading">
        {{ loading ? 'Running...' : 'Detect Objects' }}
      </button>
    </div>

    <div v-if="result" :class="['result', result.type]">
      <template v-if="result.type === 'success'">
        <p><strong>Time:</strong> {{ result.inferenceTime?.toFixed(2) }} ms</p>
        <p><strong>Detections:</strong> {{ result.detections?.length ?? 0 }}</p>

        <canvas ref="resultCanvas" class="result-canvas"></canvas>

        <ul v-if="result.detections && result.detections.length > 0" class="detection-list">
          <li v-for="(det, i) in result.detections" :key="i">
            <span class="det-label">{{ det.label }}</span>
            <span class="det-score">{{ (det.score * 100).toFixed(1) }}%</span>
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
  max-width: 416px;
  max-height: 416px;
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

.result-canvas {
  max-width: 100%;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  margin: 10px 0;
}

.detection-list {
  list-style: none;
  padding: 0;
  margin: 10px 0;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.detection-list li {
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 4px 10px;
  font-size: 0.85rem;
}

.det-label {
  font-weight: 600;
  color: #333;
  margin-right: 6px;
}

.det-score {
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
