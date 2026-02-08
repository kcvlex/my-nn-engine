export interface TensorData {
  name: string;
  data: number[];
  dims: number[];
  dtype: string;
}

export interface UploadResponse {
  model_id: string;
  message: string;
}

export interface InferenceRequest {
  inputs: TensorData[];
}

export interface InferenceResponse {
  outputs: TensorData[];
  inference_time_ms: number;
}

export interface ModelListItem {
  model_id: string;
  uploaded_at: string;
  status: string;
}

export interface ErrorResponse {
  error: string;
}

export type Target = 'CPU' | 'CUDA';
