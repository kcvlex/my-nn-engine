<script setup lang="ts">
import { ref, nextTick } from 'vue';
import {
  useInference,
  type BaseInferenceResult,
} from '../composables/useInference';
import { useImageCanvas } from '../composables/useImageCanvas';
import { sigmoid } from '../utils/math';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { TensorProto_DataType } from '../gen/onnx.proto3_pb';
import ImageUpload from './ImageUpload.vue';
import ResultBox from './ResultBox.vue';
import { loadLabels } from '../utils/labels';
import { serializeOutputs } from '../utils/tensor';

const INPUT_SIZE = 416;

// YOLOv4 anchors (from hunglc007/tensorflow-yolov4-tflite)
const ANCHORS = [
  [
    [12, 16],
    [19, 36],
    [40, 28],
  ], // stride 8  (52x52)
  [
    [36, 75],
    [76, 55],
    [72, 146],
  ], // stride 16 (26x26)
  [
    [142, 110],
    [192, 243],
    [459, 401],
  ], // stride 32 (13x13)
];
const STRIDES = [8, 16, 32];

const SCORE_THRESHOLD = 0.5;
const NMS_THRESHOLD = 0.3;

type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
  label: string;
};

const props = defineProps<{
  backend: Backend;
}>();

// NHWC layout: [1, 416, 416, 3], normalized to [0, 1]
const {
  canvas: processedCanvas,
  processImage,
  tensorData,
} = useImageCanvas(INPUT_SIZE, INPUT_SIZE, (pixels) => {
  const data: number[] = [];
  for (let i = 0; i < pixels.length; i += 4) {
    data.push(pixels[i] / 255.0);
    data.push(pixels[i + 1] / 255.0);
    data.push(pixels[i + 2] / 255.0);
  }
  return data;
});

const resultCanvas = ref<HTMLCanvasElement>();
const { loading, result, run } = useInference<
  BaseInferenceResult & {
    detections?: Detection[];
  }
>();

let originalImg: HTMLImageElement | null = null;

function onImageLoaded(img: HTMLImageElement) {
  originalImg = img;
  processImage(img);
}

function decodeDetections(
  outputs: readonly { floatData: readonly number[] }[],
): Detection[] {
  const boxes: Detection[] = [];

  for (let scaleIdx = 0; scaleIdx < 3; scaleIdx++) {
    const data = outputs[scaleIdx].floatData;
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

// Compute Intersection over Union (IoU) between two boxes.
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

// Non-Maximum Suppression to filter overlapping boxes.
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

function getCocoLabels(): Promise<string[]> {
  return loadLabels('/coco.names');
}

async function runInference(backend?: Backend) {
  if (tensorData.value.length === 0) return;

  await run(async (client) => {
    const [labels, response] = await Promise.all([
      getCocoLabels(),
      client.runInference({
        modelId: ModelId.YOLO,
        inputs: [
          {
            name: 'input_1:0',
            dims: [1n, BigInt(INPUT_SIZE), BigInt(INPUT_SIZE), 3n],
            dataType: TensorProto_DataType.FLOAT,
            floatData: tensorData.value,
          },
        ],
        backend: backend ?? props.backend,
      }),
    ]);

    const detections = decodeDetections(response.outputs);
    for (const det of detections) {
      det.label = labels[det.classId] ?? `Class ${det.classId}`;
    }

    return {
      type: 'success' as const,
      inferenceTime: response.inferenceTimeMs,
      detections,
      rawOutput: serializeOutputs(response.outputs, { summarize: true }),
    };
  });

  if (result.value?.type === 'success' && result.value.detections) {
    await nextTick();
    drawDetections(result.value.detections);
  }
}

defineExpose({ runInference });
</script>

<template>
  <div class="yolo">
    <ImageUpload
      run-label="Detect Objects"
      :loading="loading"
      @image-loaded="onImageLoaded"
      @run="runInference()"
    >
      <template #canvas>
        <div class="image-box">
          <label>416x416 RGB</label>
          <canvas
            ref="processedCanvas"
            :width="INPUT_SIZE"
            :height="INPUT_SIZE"
            class="processed-canvas"
          ></canvas>
        </div>
      </template>
    </ImageUpload>

    <ResultBox
      :visible="result != null"
      :success="result?.type === 'success'"
      :inference-time="result?.inferenceTime"
      :error-message="result?.message"
      :raw-output="result?.rawOutput"
    >
      <p><strong>Detections:</strong> {{ result?.detections?.length ?? 0 }}</p>

      <canvas ref="resultCanvas" class="result-canvas"></canvas>

      <ul
        v-if="result?.detections && result.detections.length > 0"
        class="detection-list"
      >
        <li v-for="(det, i) in result.detections" :key="i">
          <span class="det-label">{{ det.label }}</span>
          <span class="det-score">{{ (det.score * 100).toFixed(1) }}%</span>
        </li>
      </ul>
    </ResultBox>
  </div>
</template>

<style scoped>
/* Display both Original and the 416x416 letterboxed input at the same
 * fixed size so the user can compare them side-by-side. */
:deep(.preview-img) {
  width: 320px;
  height: 320px;
}

.processed-canvas {
  width: 320px;
  height: 320px;
}

.result-canvas {
  max-width: 100%;
  border: 1px solid var(--border-strong);
  background: var(--bg-input);
  margin: 10px 0;
}

.detection-list {
  list-style: none;
  padding: 0;
  margin: 12px 0 0;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.detection-list li {
  background: var(--bg-input);
  border: 1px solid var(--border-strong);
  border-left: 2px solid var(--accent-dim);
  padding: 4px 10px;
  font-size: 11px;
  letter-spacing: 0.02em;
}

.det-label {
  font-weight: 500;
  color: var(--fg);
  margin-right: 6px;
  text-transform: lowercase;
}

.det-score {
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}
</style>
