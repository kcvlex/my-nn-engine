import { computed, onScopeDispose, shallowRef } from 'vue';
import { useMutation } from '@tanstack/vue-query';
import { grpcClient } from '../api/grpc_client';
import { humanizeError } from '../utils/error';
import { Backend, LlmModelId } from '../gen/onnx_service_pb';

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
 * server-side session and clears history.
 */
export function useChatSession(modelId: LlmModelId, backend: Backend) {
  const status = shallowRef<LifecycleStatus>({ state: 'idle' });
  const sessionId = shallowRef<string | null>(null);
  const messages = shallowRef<ChatMessage[]>([]);

  async function ensureSession(): Promise<string> {
    if (sessionId.value) return sessionId.value;
    status.value = { state: 'creating' };
    try {
      const resp = await grpcClient.createChatSession({ modelId, backend });
      sessionId.value = resp.sessionId;
      status.value = { state: 'ready', buildMs: resp.buildTimeMs };
      return resp.sessionId;
    } catch (e) {
      status.value = { state: 'error', message: humanizeError(e) };
      throw e;
    }
  }

  const chatMutation = useMutation({
    mutationFn: async (input: { userMessage: string; maxTokens: number }) => {
      const id = await ensureSession();
      return grpcClient.chat({
        sessionId: id,
        userMessage: input.userMessage,
        maxTokens: input.maxTokens,
      });
    },
  });

  const generating = computed(() => chatMutation.isPending.value);

  async function send(userMessage: string, maxTokens = 256): Promise<void> {
    const trimmed = userMessage.trim();
    if (!trimmed) return;
    messages.value = [...messages.value, { role: 'user', content: trimmed }];
    try {
      const resp = await chatMutation.mutateAsync({
        userMessage: trimmed,
        maxTokens,
      });
      messages.value = [
        ...messages.value,
        {
          role: 'assistant',
          content: resp.assistantMessage,
          tokens: resp.tokensGenerated,
          generationMs: resp.generationTimeMs,
          truncated: !resp.eosEmitted,
        },
      ];
    } catch {
      // Error is exposed via chatMutation.error.value; humanize on demand.
    }
  }

  async function reset(): Promise<void> {
    const id = sessionId.value;
    sessionId.value = null;
    messages.value = [];
    status.value = { state: 'idle' };
    chatMutation.reset();
    if (id) {
      try {
        await grpcClient.destroyChatSession({ sessionId: id });
      } catch {
        // Best-effort; server-side TTL evicts stale sessions anyway.
      }
    }
  }

  // Try to clean up the server-side session when the component goes away.
  onScopeDispose(() => {
    void reset();
  });

  const lastError = computed(() =>
    chatMutation.error.value ? humanizeError(chatMutation.error.value) : null,
  );

  return {
    status,
    messages,
    generating,
    lastError,
    send,
    reset,
  };
}
