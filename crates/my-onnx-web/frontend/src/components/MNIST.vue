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
  prediction?: number;
  probabilities?: number[];
  rawOutput?: string;
} | null>(null);

let tensorData: number[] = [];

function onImageLoaded(img: HTMLImageElement) {
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
    <ImageUpload
      run-label="Classify Digit"
      :loading="loading"
      @image-loaded="onImageLoaded"
      @run="runInference()"
    >
      <template #canvas>
        <div class="image-box">
          <label>28x28 Grayscale</label>
          <canvas ref="processedCanvas" width="28" height="28" class="processed-canvas"></canvas>
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
        <span class="digit">{{ result?.prediction }}</span>
        <span class="confidence">{{ ((result?.probabilities?.[result?.prediction!] ?? 0) * 100).toFixed(1) }}%</span>
      </div>

      <ProbabilityBars
        :items="(result?.probabilities ?? []).map((prob, i) => ({
          label: String(i),
          probability: prob,
          highlight: i === result?.prediction,
        }))"
      />
    </ResultBox>
  </div>
</template>


<style scoped>
:deep(.preview-img) {
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
</style>
