<template>
  <div class="tab-content">
    <form @submit.prevent="handleSubmit" class="form">
      <div class="form-row">
        <div class="form-group">
          <label for="model-id">Model</label>
          <select id="model-id" v-model.number="modelId">
            <option :value="ModelId.MNIST">MNIST</option>
            <option :value="ModelId.RESNET">ResNet</option>
            <option :value="ModelId.YOLO">YOLO</option>
            <option :value="ModelId.BERT">BERT</option>
            <option :value="ModelId.GPT2">GPT-2</option>
          </select>
        </div>

        <div class="form-group">
          <label for="backend">Backend</label>
          <select id="backend" v-model.number="backend">
            <option :value="Backend.CPU">CPU</option>
            <option :value="Backend.CUDA">CUDA (GPU)</option>
          </select>
        </div>
      </div>

      <!-- MNIST: image upload input -->
      <MNIST
        v-if="modelId === ModelId.MNIST"
        ref="mnistRef"
        :backend="backend"
      />

      <!-- ResNet: image classification -->
      <ResNet
        v-else-if="modelId === ModelId.RESNET"
        ref="resnetRef"
        :backend="backend"
      />

      <!-- YOLO: object detection -->
      <YOLO
        v-else-if="modelId === ModelId.YOLO"
        ref="yoloRef"
        :backend="backend"
      />

      <!-- BERT: question answering -->
      <BERT
        v-else-if="modelId === ModelId.BERT"
        ref="bertRef"
        :backend="backend"
      />

      <!-- GPT-2: text generation -->
      <GPT2
        v-else-if="modelId === ModelId.GPT2"
        ref="gpt2Ref"
        :backend="backend"
      />

      <!-- Other models: raw JSON input -->
      <template v-else>
        <div class="form-group">
          <label for="input-tensor">Input Tensors (JSON array)</label>
          <textarea
            id="input-tensor"
            v-model="inputJson"
            :placeholder="placeholder"
            rows="8"
          ></textarea>
        </div>

        <button type="submit" :disabled="loading || !inputJson">
          {{ loading ? 'Running...' : 'Run Inference' }}
        </button>
      </template>
    </form>

    <div v-if="result" :class="['result', result.type]">
      <template v-if="result.type === 'success'">
        <h3>Inference Complete</h3>
        <p><strong>Time:</strong> {{ result.inferenceTime?.toFixed(2) }} ms</p>
        <details>
          <summary>Output Tensor</summary>
          <pre class="output-json">{{ result.output }}</pre>
        </details>
      </template>
      <template v-else>
        <h3>Error</h3>
        <p>{{ result.message }}</p>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { grpcClient } from '../api/grpc_client';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import MNIST from './MNIST.vue';
import ResNet from './ResNet.vue';
import YOLO from './YOLO.vue';
import BERT from './BERT.vue';
import GPT2 from './GPT2.vue';

const modelId = ref<ModelId>(ModelId.MNIST);
const backend = ref<Backend>(Backend.CPU);
const inputJson = ref('');
const loading = ref(false);
const result = ref<{
  type: 'success' | 'error';
  message?: string;
  inferenceTime?: number;
  output?: string;
} | null>(null);

const mnistRef = ref<InstanceType<typeof MNIST>>();
const resnetRef = ref<InstanceType<typeof ResNet>>();
const yoloRef = ref<InstanceType<typeof YOLO>>();
const bertRef = ref<InstanceType<typeof BERT>>();
const gpt2Ref = ref<InstanceType<typeof GPT2>>();

const placeholder = `[
  {
    "name": "input",
    "floatData": [0.0, 0.1, 0.2, ...],
    "dims": [1, 1, 28, 28],
    "dataType": 1
  }
]`;

const handleSubmit = async () => {
  if (modelId.value === ModelId.MNIST) {
    mnistRef.value?.runInference(backend.value);
    return;
  }
  if (modelId.value === ModelId.RESNET) {
    resnetRef.value?.runInference(backend.value);
    return;
  }
  if (modelId.value === ModelId.YOLO) {
    yoloRef.value?.runInference(backend.value);
    return;
  }
  if (modelId.value === ModelId.BERT) {
    bertRef.value?.runInference(backend.value);
    return;
  }
  if (modelId.value === ModelId.GPT2) {
    gpt2Ref.value?.runInference(backend.value);
    return;
  }

  loading.value = true;
  result.value = null;

  try {
    const parsed: any[] = JSON.parse(inputJson.value);
    const inputs = parsed.map(t => ({
      name: t.name ?? '',
      dims: (t.dims ?? []).map((d: number) => BigInt(d)),
      dataType: t.dataType ?? 0,
      floatData: t.floatData ?? [],
      doubleData: t.doubleData ?? [],
      int32Data: t.int32Data ?? [],
      int64Data: (t.int64Data ?? []).map((d: number) => BigInt(d)),
    }));
    const response = await grpcClient.runInference({
      modelId: modelId.value,
      inputs,
      backend: backend.value,
    });

    result.value = {
      type: 'success',
      inferenceTime: response.inferenceTimeMs,
      output: JSON.stringify(response.outputs.map(t => ({
        name: t.name,
        dims: t.dims.map(Number),
        dataType: t.dataType,
        floatData: t.floatData,
        doubleData: t.doubleData,
        int32Data: t.int32Data,
        int64Data: t.int64Data.map(Number),
      })), null, 2),
    };
  } catch (error) {
    result.value = {
      type: 'error',
      message: error instanceof SyntaxError
        ? 'Invalid JSON format'
        : error instanceof Error ? error.message : 'Unknown error',
    };
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.tab-content {
  padding: 30px;
}

.form {
  max-width: 800px;
}

.form-row {
  display: flex;
  gap: 20px;
}

.form-row .form-group {
  flex: 1;
}

.form-group {
  margin-bottom: 20px;
}

textarea {
  font-family: 'Courier New', monospace;
}

button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 12px 30px;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

h3 {
  margin-bottom: 10px;
}

.output-json {
  margin: 0;
}
</style>
