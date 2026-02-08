import type {
  UploadResponse,
  InferenceRequest,
  InferenceResponse,
  ModelListItem,
  Target,
} from '../types/api';

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
}

export const apiClient = new ApiClient();
