<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { useQuery } from '@tanstack/vue-query';
import { Backend } from '../gen/onnx_service_pb';
import { grpcClient } from '../api/grpc_client';
import { useChatSession } from '../composables/useChatSession';

const modelDir = ref<string>('');
const backend = ref<Backend>(Backend.CUDA);

const modelsQuery = useQuery({
  queryKey: ['llm-models'],
  queryFn: async () => {
    const resp = await grpcClient.listLlmModels({});
    return resp.models.map((m) => m.dirName);
  },
  staleTime: Infinity,
});

// Auto-select the first model once the list arrives, if the user hasn't
// chosen anything yet.
watch(modelsQuery.data, (list) => {
  if (!modelDir.value && list && list.length > 0) {
    modelDir.value = list[0];
  }
});

const { status, messages, generating, lastError, send, reset } = useChatSession(
  modelDir,
  backend,
);

const input = ref('');
const maxTokens = ref(256);
const logEl = ref<HTMLDivElement | null>(null);

const sessionStatusText = computed(() => {
  const s = status.value;
  switch (s.state) {
    case 'idle':
      return 'no session — will compile on first send';
    case 'creating':
      return 'compiling kernels (may take a while)...';
    case 'ready':
      return `session ready :: built in ${s.buildMs.toFixed(0)} ms`;
    case 'error':
      return `error: ${s.message}`;
    default: {
      const _exhaustive: never = s;
      return _exhaustive;
    }
  }
});

async function handleSubmit() {
  const text = input.value.trim();
  if (!text || generating.value) return;
  input.value = '';
  await send(text, maxTokens.value);
  await scrollToBottom();
}

async function scrollToBottom() {
  await nextTick();
  if (logEl.value) {
    logEl.value.scrollTop = logEl.value.scrollHeight;
  }
}

watch(messages, scrollToBottom, { deep: false });

function onKeydown(event: KeyboardEvent) {
  // Enter sends; Shift+Enter inserts newline.
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    void handleSubmit();
  }
}
</script>

<template>
  <div class="chat-tab">
    <div class="meta-row">
      <div class="form-group model-select">
        <label for="chat-model">model</label>
        <select
          id="chat-model"
          v-model="modelDir"
          :disabled="modelsQuery.isPending.value || generating"
        >
          <option v-if="modelsQuery.isPending.value" value="">
            loading...
          </option>
          <option
            v-else-if="(modelsQuery.data.value?.length ?? 0) === 0"
            value=""
          >
            no models found
          </option>
          <option
            v-for="dir in modelsQuery.data.value ?? []"
            :key="dir"
            :value="dir"
          >
            {{ dir }}
          </option>
        </select>
      </div>
      <div class="form-group backend-select">
        <label for="chat-backend">backend</label>
        <select id="chat-backend" v-model="backend" :disabled="generating">
          <option :value="Backend.CUDA">cuda</option>
          <option :value="Backend.CPU">cpu</option>
        </select>
      </div>
      <div class="form-group max-tokens">
        <label for="chat-max-tokens">max tokens</label>
        <input
          id="chat-max-tokens"
          v-model.number="maxTokens"
          type="number"
          min="1"
          max="2048"
        />
      </div>
      <button
        type="button"
        class="reset-btn"
        :disabled="messages.length === 0 && status.state === 'idle'"
        @click="reset"
      >
        new chat
      </button>
    </div>

    <div :class="['session-status', status.state]">
      <span class="status-dot"></span>
      <span class="status-text">{{ sessionStatusText }}</span>
    </div>

    <div ref="logEl" class="chat-log">
      <div v-for="(msg, i) in messages" :key="i" :class="['msg', msg.role]">
        <span class="role"
          >{{ msg.role === 'user' ? '>' : '<' }} {{ msg.role }}</span
        >
        <pre class="content">{{ msg.content }}<span
          v-if="
            generating &&
            msg.role === 'assistant' &&
            i === messages.length - 1
          "
          class="generating-cursor"
        >&#x258c;</span></pre>
        <div
          v-if="msg.role === 'assistant' && msg.tokens !== undefined"
          class="msg-meta"
        >
          {{ msg.tokens }} tokens
          <span v-if="msg.generationMs">
            :: {{ msg.generationMs.toFixed(0) }} ms</span
          >
          <span v-if="msg.truncated" class="truncated">
            :: truncated (max_tokens)</span
          >
        </div>
      </div>

      <div v-if="lastError" class="error-banner">{{ lastError }}</div>
    </div>

    <form class="composer" @submit.prevent="handleSubmit">
      <textarea
        v-model="input"
        class="prompt-input"
        rows="3"
        placeholder="type a message — Enter to send, Shift+Enter for newline"
        :disabled="generating"
        @keydown="onKeydown"
      ></textarea>
      <button
        type="submit"
        class="run-btn send-btn"
        :disabled="generating || !input.trim()"
      >
        {{ generating ? 'generating...' : 'send' }}
      </button>
    </form>
  </div>
</template>

<style scoped>
.chat-tab {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 220px);
  min-height: 500px;
  padding: 22px 28px 24px;
  gap: 14px;
}

.meta-row {
  display: flex;
  align-items: end;
  gap: 22px;
  padding-bottom: 14px;
  border-bottom: 1px dashed var(--border-strong);
}

.model-select {
  margin-bottom: 0;
  flex: 0 0 auto;
}

.model-select select {
  min-width: 220px;
}

.backend-select {
  margin-bottom: 0;
  flex: 0 0 auto;
}

.backend-select select {
  min-width: 90px;
}

.max-tokens {
  margin-bottom: 0;
}

.max-tokens input {
  width: 110px;
}

.reset-btn {
  margin-left: auto;
}

.session-status {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: lowercase;
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

.session-status.creating {
  border-left-color: var(--warn);
  color: var(--warn);
}

.session-status.creating .status-dot {
  background: var(--warn);
  box-shadow: 0 0 6px var(--warn);
  animation: chat-pulse 1.2s ease-in-out infinite;
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

@keyframes chat-pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.4;
  }
}

.chat-log {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 12px 14px;
  border: 1px solid var(--border-strong);
  background: var(--bg);
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.msg {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.msg .role {
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--fg-dim);
}

.msg.user .role {
  color: var(--fg);
}

.msg.assistant .role {
  color: var(--accent);
}

.msg .content {
  font-family: var(--mono);
  font-size: 13px;
  white-space: pre-wrap;
  word-wrap: break-word;
  margin: 0;
  color: var(--fg);
  padding: 6px 10px;
  border-left: 2px solid var(--border-strong);
}

.msg.user .content {
  border-left-color: var(--fg-faint);
}

.msg.assistant .content {
  border-left-color: var(--accent-dim);
}

.msg-meta {
  font-size: 10px;
  letter-spacing: 0.06em;
  color: var(--fg-faint);
  padding-left: 10px;
}

.msg-meta .truncated {
  color: var(--warn);
}

.generating-cursor {
  display: inline-block;
  color: var(--accent);
  animation: chat-blink 1.05s steps(2, end) infinite;
  padding-left: 12px;
}

@keyframes chat-blink {
  to {
    opacity: 0;
  }
}

.error-banner {
  padding: 8px 12px;
  background: var(--bg-input);
  border-left: 2px solid var(--danger);
  color: var(--danger);
  font-size: 12px;
}

.composer {
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

.prompt-input {
  flex: 1;
  min-height: 60px;
  resize: vertical;
}

.send-btn {
  flex-shrink: 0;
}
</style>
