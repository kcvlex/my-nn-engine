import { shallowRef } from 'vue';
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
  const loading = shallowRef(false);
  const result = shallowRef<T | null>(null);

  async function run(fn: (client: GrpcClient) => Promise<T>): Promise<void> {
    loading.value = true;
    result.value = null;
    try {
      result.value = await fn(grpcClient);
    } catch (e) {
      result.value = {
        type: 'error',
        message: e instanceof Error ? e.message : 'Unknown error',
      } satisfies BaseInferenceResult as T;
    } finally {
      loading.value = false;
    }
  }

  return { loading, result, run };
}
