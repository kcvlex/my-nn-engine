import type {
  UploadResponse,
  InferenceRequest,
  InferenceResponse,
  ModelListItem,
  Target,
} from '../types/api';
import { encodeTensors, decodeTensors, type TensorProto } from './protobuf';

class ApiClient {
  private baseUrl = '';

  async uploadModel(
    file: File,
    target: Target,
    optimize: boolean
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target', target);
    formData.append('optimize', optimize.toString());

    const response = await fetch(`${this.baseUrl}/models/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Upload failed');
    }

    return response.json();
  }

  async runInference(
    modelId: string,
    request: InferenceRequest
  ): Promise<InferenceResponse> {
    const response = await fetch(`${this.baseUrl}/models/${modelId}/infer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Inference failed');
    }

    return response.json();
  }

  async listModels(): Promise<ModelListItem[]> {
    const response = await fetch(`${this.baseUrl}/models`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to load models');
    }

    return response.json();
  }

  async runInferenceProto(
    modelId: string,
    inputs: TensorProto[]
  ): Promise<TensorProto[]> {
    const requestBytes = encodeTensors(inputs);

    const response = await fetch(`${this.baseUrl}/models/${modelId}/infer/proto`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
      },
      body: requestBytes,
    });

    if (!response.ok) {
      // Try to parse JSON error if available
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        const error = await response.json();
        throw new Error(error.error || 'Inference failed');
      }
      throw new Error(`Inference failed: ${response.statusText}`);
    }

    const responseBytes = new Uint8Array(await response.arrayBuffer());
    return decodeTensors(responseBytes);
  }
}

export const apiClient = new ApiClient();
