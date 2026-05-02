<script setup lang="ts">
import {
  useInference,
  type BaseInferenceResult,
} from '../composables/useInference';
import { useImageCanvas } from '../composables/useImageCanvas';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { TensorProto_DataType } from '../gen/onnx.proto3_pb';
import ImageUpload from './ImageUpload.vue';
import ResultBox from './ResultBox.vue';
import ProbabilityBars from './ProbabilityBars.vue';
import { topK } from '../utils/math';
import { loadLabels } from '../utils/labels';
import { serializeOutputs } from '../utils/tensor';

export type Layout = 'nchw' | 'nhwc';
export type Normalization = 'imagenet' | 'pm1';

const props = defineProps<{
  backend: Backend;
  modelId: ModelId;
  inputName: string;
  layout: Layout;
  normalize: Normalization;
  // True when the model already outputs softmax probabilities (e.g. EfficientNet-Lite4).
  outputIsProbability: boolean;
}>();

const INPUT_SIZE = 224;

// ImageNet normalization constants: per-channel mean/std computed over the ILSVRC2012 training set
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

function normalizeChannel(value: number, channelIdx: number): number {
  if (props.normalize === 'imagenet') {
    return (
      (value / 255.0 - IMAGENET_MEAN[channelIdx]) / IMAGENET_STD[channelIdx]
    );
  }
  // 'pm1': map [0, 255] → [-1, 1]
  return value / 127.5 - 1.0;
}

const {
  canvas: processedCanvas,
  processImage: onImageLoaded,
  getTensorData,
} = useImageCanvas(INPUT_SIZE, INPUT_SIZE, (pixels) => {
  if (props.layout === 'nchw') {
    const r: number[] = [];
    const g: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < pixels.length; i += 4) {
      r.push(normalizeChannel(pixels[i], 0));
      g.push(normalizeChannel(pixels[i + 1], 1));
      b.push(normalizeChannel(pixels[i + 2], 2));
    }
    return [...r, ...g, ...b];
  }
  // nhwc: interleaved RGB
  const data: number[] = [];
  for (let i = 0; i < pixels.length; i += 4) {
    data.push(normalizeChannel(pixels[i], 0));
    data.push(normalizeChannel(pixels[i + 1], 1));
    data.push(normalizeChannel(pixels[i + 2], 2));
  }
  return data;
});

const { loading, result, run } = useInference<
  BaseInferenceResult & {
    topK?: { label: string; probability: number }[];
  }
>();

async function runInference(backend?: Backend) {
  if (getTensorData().length === 0) return;

  await run(async (client) => {
    const dims =
      props.layout === 'nchw'
        ? [1n, 3n, BigInt(INPUT_SIZE), BigInt(INPUT_SIZE)]
        : [1n, BigInt(INPUT_SIZE), BigInt(INPUT_SIZE), 3n];

    const [labels, response] = await Promise.all([
      getImageNetLabels(),
      client.runInference({
        modelId: props.modelId,
        inputs: [
          {
            name: props.inputName,
            dims,
            dataType: TensorProto_DataType.FLOAT,
            floatData: getTensorData(),
          },
        ],
        backend: backend ?? props.backend,
      }),
    ]);

    const output = response.outputs[0];
    const raw = output?.floatData ?? output?.doubleData ?? [];
    const top5 = topK(raw, 5, {
      applySoftmax: !props.outputIsProbability,
    }).map(({ id, prob }) => ({
      label: labels[id] ?? `Class ${id}`,
      probability: prob,
    }));

    return {
      type: 'success' as const,
      inferenceTime: response.inferenceTimeMs,
      topK: top5,
      rawOutput: serializeOutputs(response.outputs),
    };
  });
}

function getImageNetLabels(): Promise<string[]> {
  return loadLabels('/synset.txt', 1000, (line) => {
    // Format: "n01440764 tench, Tinca tinca" → "tench"
    const desc = line.substring(line.indexOf(' ') + 1);
    return desc.split(',')[0].trim();
  });
}

defineExpose({ runInference });
</script>

<template>
  <div class="imagenet-classifier">
    <ImageUpload
      run-label="Classify Image"
      :loading="loading"
      @image-loaded="onImageLoaded"
      @run="runInference()"
    >
      <template #canvas>
        <div class="image-box">
          <label>{{ INPUT_SIZE }}x{{ INPUT_SIZE }} RGB</label>
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
      <div class="prediction">
        <span class="top-label">{{ result?.topK?.[0]?.label }}</span>
        <span class="confidence"
          >{{ ((result?.topK?.[0]?.probability ?? 0) * 100).toFixed(1) }}%</span
        >
      </div>

      <ProbabilityBars
        label-width="180px"
        truncate-label
        :items="
          (result?.topK ?? []).map((entry, i) => ({
            label: entry.label,
            probability: entry.probability,
            highlight: i === 0,
          }))
        "
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
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.01em;
}

.top-label::before {
  content: 'top1 ▸ ';
  color: var(--fg-dim);
  font-size: 0.85rem;
  font-weight: 400;
  letter-spacing: 0.08em;
}
</style>
