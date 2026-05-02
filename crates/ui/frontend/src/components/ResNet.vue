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

const props = defineProps<{
  backend: Backend;
  modelId: ModelId;
}>();

// ImageNet normalization constants: per-channel mean/std computed over the ILSVRC2012 training set
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// NCHW layout: [1, 3, 224, 224] with ImageNet normalization
const {
  canvas: processedCanvas,
  processImage: onImageLoaded,
  getTensorData,
} = useImageCanvas(224, 224, (pixels) => {
  const r: number[] = [];
  const g: number[] = [];
  const b: number[] = [];
  for (let i = 0; i < pixels.length; i += 4) {
    r.push((pixels[i] / 255.0 - MEAN[0]) / STD[0]);
    g.push((pixels[i + 1] / 255.0 - MEAN[1]) / STD[1]);
    b.push((pixels[i + 2] / 255.0 - MEAN[2]) / STD[2]);
  }
  return [...r, ...g, ...b];
});

const { loading, result, run } = useInference<
  BaseInferenceResult & {
    topK?: { label: string; probability: number }[];
  }
>();

async function runInference(backend?: Backend) {
  if (getTensorData().length === 0) return;

  await run(async (client) => {
    const [labels, response] = await Promise.all([
      getImageNetLabels(),
      client.runInference({
        modelId: props.modelId,
        inputs: [
          {
            name: 'data',
            dims: [1n, 3n, 224n, 224n],
            dataType: TensorProto_DataType.FLOAT,
            floatData: getTensorData(),
          },
        ],
        backend: backend ?? props.backend,
      }),
    ]);

    const output = response.outputs[0];
    const logits = output?.floatData ?? output?.doubleData ?? [];
    const top5 = topK(logits, 5).map(({ id, prob }) => ({
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
          <canvas
            ref="processedCanvas"
            width="224"
            height="224"
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
  font-size: 1.5rem;
  font-weight: 700;
  color: #333;
}
</style>
