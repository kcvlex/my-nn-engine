export interface TensorData {
  name: string;
  dims: number[];
  dtype: string;
  data: number[];
}

export enum ModelId {
  MNIST = 0,
  RESNET = 1,
  YOLO = 2,
  BERT = 3,
  GPT2 = 4,
}

export enum Backend {
  CPU = 0,
  CUDA = 1,
}

export interface InferenceRequest {
  model_id: ModelId;
  input_data: TensorData;
  backend: Backend;
}

export interface InferenceResponse {
  output_data: TensorData;
  inference_time_ms: number;
}
