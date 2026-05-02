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
            <option :value="ModelId.MOBILENETV2">MobileNetV2</option>
            <option :value="ModelId.EFFICIENTNET_LITE4">
              EfficientNet-Lite4
            </option>
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

      <!-- ImageNet classifiers: ResNet18 / ResNet152 / MobileNetV2 / EfficientNet-Lite4 -->
      <ImageNetClassifier
        v-else-if="imageNetConfig"
        ref="imageNetRef"
        :key="modelId"
        :backend="backend"
        :model-id="modelId"
        :input-name="imageNetConfig.inputName"
        :layout="imageNetConfig.layout"
        :normalize="imageNetConfig.normalize"
        :output-is-probability="imageNetConfig.outputIsProbability"
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
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import MNIST from './MNIST.vue';
import ImageNetClassifier, {
  type Layout,
  type Normalization,
} from './ImageNetClassifier.vue';
import YOLO from './YOLO.vue';
import BERT from './BERT.vue';
import GPT2 from './GPT2.vue';

type ImageNetConfig = {
  inputName: string;
  layout: Layout;
  normalize: Normalization;
  outputIsProbability: boolean;
};

const IMAGENET_CONFIGS: Partial<Record<ModelId, ImageNetConfig>> = {
  [ModelId.RESNET]: {
    inputName: 'data',
    layout: 'nchw',
    normalize: 'imagenet',
    outputIsProbability: false,
  },
  [ModelId.RESNET152]: {
    inputName: 'data',
    layout: 'nchw',
    normalize: 'imagenet',
    outputIsProbability: false,
  },
  [ModelId.MOBILENETV2]: {
    inputName: 'input',
    layout: 'nchw',
    normalize: 'imagenet',
    outputIsProbability: false,
  },
  [ModelId.EFFICIENTNET_LITE4]: {
    inputName: 'images:0',
    layout: 'nhwc',
    normalize: 'pm1',
    outputIsProbability: true,
  },
};

const modelId = ref<ModelId>(ModelId.MNIST);
const backend = ref<Backend>(Backend.CPU);

const imageNetConfig = computed(() => IMAGENET_CONFIGS[modelId.value]);

const mnistRef = ref<InstanceType<typeof MNIST>>();
const imageNetRef = ref<InstanceType<typeof ImageNetClassifier>>();
const yoloRef = ref<InstanceType<typeof YOLO>>();
const bertRef = ref<InstanceType<typeof BERT>>();
const gpt2Ref = ref<InstanceType<typeof GPT2>>();

const handleSubmit = async () => {
  if (imageNetConfig.value) {
    await imageNetRef.value?.runInference(backend.value);
    return;
  }
  switch (modelId.value) {
    case ModelId.MNIST:
      await mnistRef.value?.runInference(backend.value);
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
  padding: 24px 28px 28px;
}

.form {
  max-width: 880px;
}

.form-row {
  display: flex;
  gap: 18px;
  border-bottom: 1px dashed var(--border-strong);
  padding-bottom: 18px;
  margin-bottom: 22px;
}

.form-row .form-group {
  flex: 1;
  margin-bottom: 0;
}

h3 {
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg-dim);
  margin-bottom: 8px;
}
</style>
