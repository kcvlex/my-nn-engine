import { computed } from 'vue';
import { useMutation } from '@tanstack/vue-query';
import { grpcClient } from '../api/grpc_client';
import type { Client } from '@connectrpc/connect';
import type { OnnxInferenceService } from '../gen/onnx_service_pb';

export type GrpcClient = Client<typeof OnnxInferenceService>;

export type BaseInferenceResult = {
  type: 'success' | 'error';
  message?: string;
  inferenceTime?: number;
  rawOutput?: string;
};

export function useInference<T extends BaseInferenceResult>() {
  const mutation = useMutation({
    mutationFn: (fn: (client: GrpcClient) => Promise<T>) => fn(grpcClient),
  });

  const loading = computed(() => mutation.isPending.value);

  const result = computed<T | null>(() => {
    if (mutation.isPending.value) return null;
    if (mutation.error.value) {
      const err = mutation.error.value;
      return {
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      } as T;
    }
    return (mutation.data.value as T | undefined) ?? null;
  });

  function run(fn: (client: GrpcClient) => Promise<T>): Promise<void> {
    return new Promise((resolve) => {
      mutation.mutate(fn, { onSettled: () => resolve() });
    });
  }

  return { loading, result, run };
}
