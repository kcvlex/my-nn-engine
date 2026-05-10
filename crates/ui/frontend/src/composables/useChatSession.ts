import { computed, onScopeDispose, shallowRef, watch, type Ref } from 'vue';
import { grpcClient } from '../api/grpc_client';
import { humanizeError } from '../utils/error';
import { Backend } from '../gen/onnx_service_pb';

type ChatRole = 'user' | 'assistant';

type ChatMessage = {
  role: ChatRole;
  content: string;
  /** Server-reported tokens generated for assistant turns. */
  tokens?: number;
  /** Server-reported wall time for assistant turns (ms). */
  generationMs?: number;
  /** False if the model hit max_tokens without emitting EOS. */
  truncated?: boolean;
};

type LifecycleStatus =
  | { state: 'idle' }
  | { state: 'creating' }
  | { state: 'ready'; buildMs: number }
  | { state: 'error'; message: string };

/**
 * Manages a single LLM chat session. The session is created lazily on the
 * first send (so the user doesn't pay the compile cost just by opening the
 * tab), then reused for subsequent messages. Calling `reset()` destroys the
 * server-side session and clears history. Switching `modelDir` auto-resets.
 */
export function useChatSession(modelDir: Ref<string>, backend: Backend) {
  const status = shallowRef<LifecycleStatus>({ state: 'idle' });
  const sessionId = shallowRef<string | null>(null);
  const messages = shallowRef<ChatMessage[]>([]);
  const generating = shallowRef(false);
  const lastError = shallowRef<string | null>(null);

  async function ensureSession(): Promise<string> {
    if (sessionId.value) return sessionId.value;
    if (!modelDir.value) {
      throw new Error('no model selected');
    }
    status.value = { state: 'creating' };
    try {
      const resp = await grpcClient.createChatSession({
        modelDir: modelDir.value,
        backend,
      });
      sessionId.value = resp.sessionId;
      status.value = { state: 'ready', buildMs: resp.buildTimeMs };
      return resp.sessionId;
    } catch (e) {
      status.value = { state: 'error', message: humanizeError(e) };
      throw e;
    }
  }

  function patchMessage(idx: number, patch: Partial<ChatMessage>): void {
    const cur = messages.value[idx];
    if (!cur) return;
    messages.value = [
      ...messages.value.slice(0, idx),
      { ...cur, ...patch },
      ...messages.value.slice(idx + 1),
    ];
  }

  async function send(userMessage: string, maxTokens = 256): Promise<void> {
    const trimmed = userMessage.trim();
    if (!trimmed) return;
    // Append user + an empty assistant placeholder we'll fill from the stream.
    messages.value = [
      ...messages.value,
      { role: 'user', content: trimmed },
      { role: 'assistant', content: '' },
    ];
    const assistantIdx = messages.value.length - 1;

    generating.value = true;
    lastError.value = null;
    try {
      const id = await ensureSession();
      const stream = grpcClient.chat({
        sessionId: id,
        userMessage: trimmed,
        maxTokens,
      });
      for await (const resp of stream) {
        const evt = resp.event;
        if (!evt) continue;
        if (evt.case === 'chunk') {
          const cur = messages.value[assistantIdx];
          if (cur) {
            patchMessage(assistantIdx, { content: cur.content + evt.value.delta });
          }
        } else if (evt.case === 'done') {
          patchMessage(assistantIdx, {
            tokens: evt.value.tokensGenerated,
            generationMs: evt.value.generationTimeMs,
            truncated: !evt.value.eosEmitted,
          });
        }
      }
    } catch (e) {
      lastError.value = humanizeError(e);
    } finally {
      generating.value = false;
    }
  }

  async function reset(): Promise<void> {
    const id = sessionId.value;
    sessionId.value = null;
    messages.value = [];
    status.value = { state: 'idle' };
    lastError.value = null;
    if (id) {
      try {
        await grpcClient.destroyChatSession({ sessionId: id });
      } catch {
        // Best-effort; server-side TTL evicts stale sessions anyway.
      }
    }
  }

  // Switching models invalidates the cached session.
  watch(modelDir, () => {
    void reset();
  });

  // Try to clean up the server-side session when the component goes away.
  onScopeDispose(() => {
    void reset();
  });

  const generatingComputed = computed(() => generating.value);
  const lastErrorComputed = computed(() => lastError.value);

  return {
    status,
    messages,
    generating: generatingComputed,
    lastError: lastErrorComputed,
    send,
    reset,
  };
}
