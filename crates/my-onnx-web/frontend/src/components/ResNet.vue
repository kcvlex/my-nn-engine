<script setup lang="ts">
import { ref } from 'vue';
import { grpcClient } from '../api/grpc_client';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { TensorProto_DataType } from '../gen/onnx.proto3_pb';
import ImageUpload from './ImageUpload.vue';
import ResultBox from './ResultBox.vue';
import ProbabilityBars from './ProbabilityBars.vue';

const props = defineProps<{
  backend: Backend;
}>();

const processedCanvas = ref<HTMLCanvasElement>();
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

function onImageLoaded(img: HTMLImageElement) {
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
    <ImageUpload
      run-label="Classify Image"
      :loading="loading"
      @image-loaded="onImageLoaded"
      @run="runInference()"
    >
      <template #canvas>
        <div class="image-box">
          <label>224x224 RGB</label>
          <canvas ref="processedCanvas" width="224" height="224" class="processed-canvas"></canvas>
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
      <div class="prediction">
        <span class="top-label">{{ result?.topK?.[0]?.label }}</span>
        <span class="confidence">{{ ((result?.topK?.[0]?.probability ?? 0) * 100).toFixed(1) }}%</span>
      </div>

      <ProbabilityBars
        label-width="180px"
        truncate-label
        :items="(result?.topK ?? []).map((entry, i) => ({
          label: entry.label,
          probability: entry.probability,
          highlight: i === 0,
        }))"
      />
    </ResultBox>
  </div>
</template>

<style scoped>
:deep(.preview-img) {
  width: 224px;
  height: 224px;
}

.processed-canvas {
  width: 224px;
  height: 224px;
}

.top-label {
  font-size: 1.5rem;
  font-weight: 700;
  color: #333;
}
</style>
