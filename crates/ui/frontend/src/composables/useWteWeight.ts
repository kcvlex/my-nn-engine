import type { GrpcClient } from './useInference';
import { ModelId } from '../gen/onnx_service_pb';

export function useWteWeight(vocabSize: number, hiddenDim: number) {
  let wteWeight: Float32Array | null = null;

  async function load(client: GrpcClient): Promise<void> {
    if (wteWeight) return;
    const resp = await client.getInitializer({
      modelId: ModelId.GPT2,
      name: 'wte.weight',
    });
    const rawView = new Uint8Array(resp.tensor!.rawData);
    const copiedBuffer = rawView.slice(0).buffer;
    wteWeight = new Float32Array(copiedBuffer);
  }

  function projectToLogits(hiddenState: number[]): number[] {
    if (!wteWeight) return [];
    const logits = new Float64Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      let sum = 0;
      const offset = v * hiddenDim;
      for (let i = 0; i < hiddenDim; i++) {
        sum += hiddenState[i] * wteWeight[offset + i];
      }
      logits[v] = sum;
    }
    return Array.from(logits);
  }

  return { load, projectToLogits };
}
