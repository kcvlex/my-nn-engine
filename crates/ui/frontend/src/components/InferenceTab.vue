<template>
  <div class="tab-content">
    <form class="form" @submit.prevent="handleSubmit">
      <div class="form-row">
        <div class="form-group">
          <label for="model-id">Model</label>
          <select id="model-id" v-model.number="modelId">
            <option :value="ModelId.MNIST">MNIST</option>
            <option :value="ModelId.RESNET">ResNet18</option>
            <option :value="ModelId.RESNET152">ResNet152</option>
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

      <!-- ResNet: image classification (ResNet18 / ResNet152) -->
      <ResNet
        v-else-if="modelId === ModelId.RESNET || modelId === ModelId.RESNET152"
        ref="resnetRef"
        :key="modelId"
        :backend="backend"
        :model-id="modelId"
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

      <template v-else />
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
import { ModelId, Backend } from '../gen/onnx_service_pb';
import MNIST from './MNIST.vue';
import ResNet from './ResNet.vue';
import YOLO from './YOLO.vue';
import BERT from './BERT.vue';
import GPT2 from './GPT2.vue';

const modelId = ref<ModelId>(ModelId.MNIST);
const backend = ref<Backend>(Backend.CPU);
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

const handleSubmit = async () => {
  switch (modelId.value) {
    case ModelId.MNIST:
      await mnistRef.value?.runInference(backend.value);
      return;
    case ModelId.RESNET:
    case ModelId.RESNET152:
      await resnetRef.value?.runInference(backend.value);
      return;
    case ModelId.YOLO:
      await yoloRef.value?.runInference(backend.value);
      return;
    case ModelId.BERT:
      await bertRef.value?.runInference(backend.value);
      return;
    case ModelId.GPT2:
      await gpt2Ref.value?.runInference(backend.value);
      return;
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
  transition:
    transform 0.2s,
    box-shadow 0.2s;
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
