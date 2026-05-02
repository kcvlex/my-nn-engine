<template>
  <div class="tab-content">
    <form class="form" @submit.prevent="handleSubmit">
      <div class="form-row">
        <div class="form-group">
          <label for="model-id">Model</label>
          <select id="model-id" v-model.number="modelId">
            <option v-for="[id, entry] in MODELS" :key="id" :value="id">
              {{ entry.label }}
            </option>
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

      <div :class="['session-status', warmup.state]">
        <span class="status-dot"></span>
        <span class="status-text">{{ warmupStatusText }}</span>
        <span
          v-if="warmup.state === 'ready' && warmup.buildMs > 0"
          class="status-meta"
        >
          // built in {{ warmup.buildMs.toFixed(0) }} ms
        </span>
      </div>

      <component
        :is="activeEntry.component"
        ref="activeRef"
        :key="modelId"
        v-bind="activeEntry.props(backend)"
      />
    </form>
  </div>
</template>

<script setup lang="ts">
import { computed, defineAsyncComponent, ref, type Component } from 'vue';

import { useQuery } from '@tanstack/vue-query';
import { ModelId, Backend } from '../gen/onnx_service_pb';
import { grpcClient } from '../api/grpc_client';
import { humanizeError } from '../utils/error';
import { type Layout, type Normalization } from './ImageNetClassifier.vue';

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

type ModelEntry = {
  label: string;
  component: Component;
  props: (backend: Backend) => Record<string, unknown>;
};

const imageNetEntry = (modelId: ModelId, label: string): ModelEntry => ({
  label,
  component: defineAsyncComponent(() => import('./ImageNetClassifier.vue')),
  props: (backend) => ({ backend, modelId, ...IMAGENET_CONFIGS[modelId]! }),
});

// A Map (not a Record) so insertion order is the dropdown display order;
// Record keys with integer-like values get sorted numerically by JS.
const MODELS = new Map<ModelId, ModelEntry>([
  [
    ModelId.MNIST,
    {
      label: 'MNIST',
      component: defineAsyncComponent(() => import('./MNIST.vue')),
      props: (backend) => ({ backend }),
    },
  ],
  [ModelId.RESNET, imageNetEntry(ModelId.RESNET, 'ResNet18')],
  [ModelId.RESNET152, imageNetEntry(ModelId.RESNET152, 'ResNet152')],
  [ModelId.MOBILENETV2, imageNetEntry(ModelId.MOBILENETV2, 'MobileNetV2')],
  [
    ModelId.EFFICIENTNET_LITE4,
    imageNetEntry(ModelId.EFFICIENTNET_LITE4, 'EfficientNet-Lite4'),
  ],
  [
    ModelId.YOLO,
    {
      label: 'YOLO',
      component: defineAsyncComponent(() => import('./YOLO.vue')),
      props: (backend) => ({ backend }),
    },
  ],
  [
    ModelId.BERT,
    {
      label: 'BERT',
      component: defineAsyncComponent(() => import('./BERT.vue')),
      props: (backend) => ({ backend }),
    },
  ],
  [
    ModelId.GPT2,
    {
      label: 'GPT-2',
      component: defineAsyncComponent(() => import('./GPT2.vue')),
      props: (backend) => ({ backend }),
    },
  ],
]);

const modelId = ref<ModelId>(ModelId.MNIST);
const backend = ref<Backend>(Backend.CPU);

const activeEntry = computed(() => MODELS.get(modelId.value)!);

// Pre-build the session whenever (modelId, backend) changes. The server caches
// per (model, target), so the query is `staleTime: Infinity`. Vue Query
// auto-cancels in-flight requests when the queryKey changes, so we don't need
// a manual sequence ticket.
const warmupQuery = useQuery({
  queryKey: ['warmup', modelId, backend] as const,
  queryFn: () =>
    grpcClient.warmUp({ modelId: modelId.value, backend: backend.value }),
  staleTime: Infinity,
  retry: false,
});

const warmup = computed(() => {
  if (warmupQuery.isPending.value) return { state: 'compiling' as const };
  if (warmupQuery.error.value) {
    return {
      state: 'error' as const,
      message: humanizeError(warmupQuery.error.value),
    };
  }
  if (warmupQuery.data.value) {
    return {
      state: 'ready' as const,
      buildMs: warmupQuery.data.value.buildTimeMs,
    };
  }
  return { state: 'idle' as const };
});

const warmupStatusText = computed(() => {
  const w = warmup.value;
  switch (w.state) {
    case 'idle':
      return 'idle';
    case 'compiling':
      return 'compiling kernels...';
    case 'ready':
      return 'session ready';
    case 'error':
      return `error: ${w.message}`;
    default: {
      const _exhaustive: never = w;
      return _exhaustive;
    }
  }
});

type Runner = { runInference: (backend: Backend) => Promise<void> };
const activeRef = ref<Runner | null>(null);

const handleSubmit = async () => {
  await activeRef.value?.runInference(backend.value);
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

.session-status {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: lowercase;
  margin: -10px 0 22px;
  padding: 6px 10px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-left: 2px solid var(--fg-faint);
  color: var(--fg-dim);
}

.session-status .status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--fg-faint);
  flex-shrink: 0;
}

.session-status.compiling {
  border-left-color: var(--warn);
  color: var(--warn);
}

.session-status.compiling .status-dot {
  background: var(--warn);
  box-shadow: 0 0 6px var(--warn);
  animation: pulse 1.2s ease-in-out infinite;
}

.session-status.ready {
  border-left-color: var(--accent-dim);
  color: var(--fg-mid);
}

.session-status.ready .status-dot {
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent);
}

.session-status.error {
  border-left-color: var(--danger);
  color: var(--danger);
}

.session-status.error .status-dot {
  background: var(--danger);
  box-shadow: 0 0 6px var(--danger);
}

.status-meta {
  margin-left: auto;
  color: var(--fg-faint);
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.4;
  }
}
</style>
