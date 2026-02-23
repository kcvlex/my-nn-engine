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
import { softmax } from '../utils/math';
import { serializeOutputs } from '../utils/tensor';

const props = defineProps<{
  backend: Backend;
}>();

const {
  canvas: processedCanvas,
  processImage: onImageLoaded,
  getTensorData,
} = useImageCanvas(
  28,
  28,
  (pixels) => {
    // RGB to grayscale using ITU-R BT.601 luma coefficients (Y = 0.299R + 0.587G + 0.114B)
    const data: number[] = [];
    for (let i = 0; i < pixels.length; i += 4) {
      data.push(
        (pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114) /
          255.0,
      );
    }
    return data;
  },
  // MNIST expects white digits on a black background
  { fillStyle: 'black' },
);

const { loading, result, run } = useInference<
  BaseInferenceResult & {
    prediction?: number;
    probabilities?: number[];
  }
>();

async function runInference(backend?: Backend) {
  if (getTensorData().length === 0) return;

  await run(async (client) => {
    const response = await client.runInference({
      modelId: ModelId.MNIST,
      inputs: [
        {
          name: 'Input3',
          dims: [1n, 1n, 28n, 28n],
          dataType: TensorProto_DataType.FLOAT,
          floatData: getTensorData(),
        },
      ],
      backend: backend ?? props.backend,
    });

    const output = response.outputs[0];
    const logits = output?.floatData ?? output?.doubleData ?? [];
    const probs = softmax(logits);
    const prediction = probs.indexOf(Math.max(...probs));

    return {
      type: 'success' as const,
      inferenceTime: response.inferenceTimeMs,
      prediction,
      probabilities: probs,
      rawOutput: serializeOutputs(response.outputs),
    };
  });
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
          <canvas
            ref="processedCanvas"
            width="28"
            height="28"
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
        <span class="digit">{{ result?.prediction }}</span>
        <span class="confidence"
          >{{
            ((result?.probabilities?.[result?.prediction!] ?? 0) * 100).toFixed(
              1,
            )
          }}%</span
        >
      </div>

      <ProbabilityBars
        :items="
          (result?.probabilities ?? []).map((prob, i) => ({
            label: String(i),
            probability: prob,
            highlight: i === result?.prediction,
          }))
        "
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
