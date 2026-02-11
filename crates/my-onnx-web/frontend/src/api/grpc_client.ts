import type { TensorData, ModelId, Backend, InferenceRequest, InferenceResponse } from '../types/api';

// Protobuf wire format encoding/decoding helpers
class ProtoEncoder {
  private buffer: number[] = [];

  writeVarint(value: number) {
    while (value > 127) {
      this.buffer.push((value & 127) | 128);
      value >>>= 7;
    }
    this.buffer.push(value);
  }

  writeString(fieldNumber: number, value: string) {
    this.writeVarint((fieldNumber << 3) | 2);
    const bytes = new TextEncoder().encode(value);
    this.writeVarint(bytes.length);
    this.buffer.push(...Array.from(bytes));
  }

  writeInt64(fieldNumber: number, value: number) {
    this.writeVarint((fieldNumber << 3) | 0);
    this.writeVarint(value);
  }

  writeEnum(fieldNumber: number, value: number) {
    this.writeVarint((fieldNumber << 3) | 0);
    this.writeVarint(value);
  }

  writeDouble(fieldNumber: number, value: number) {
    this.writeVarint((fieldNumber << 3) | 1);
    const view = new DataView(new ArrayBuffer(8));
    view.setFloat64(0, value, true);
    this.buffer.push(...Array.from(new Uint8Array(view.buffer)));
  }

  writeMessage(fieldNumber: number, value: Uint8Array) {
    this.writeVarint((fieldNumber << 3) | 2);
    this.writeVarint(value.length);
    this.buffer.push(...Array.from(value));
  }

  toUint8Array(): Uint8Array {
    return new Uint8Array(this.buffer);
  }
}

class ProtoDecoder {
  private view: DataView;
  private offset: number = 0;

  constructor(private data: Uint8Array) {
    this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  }

  readVarint(): number {
    let result = 0;
    let shift = 0;
    while (this.offset < this.data.length) {
      const byte = this.data[this.offset++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  }

  readString(): string {
    const length = this.readVarint();
    const bytes = this.data.slice(this.offset, this.offset + length);
    this.offset += length;
    return new TextDecoder().decode(bytes);
  }

  readDouble(): number {
    const value = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return value;
  }

  readMessage(): Uint8Array {
    const length = this.readVarint();
    const bytes = this.data.slice(this.offset, this.offset + length);
    this.offset += length;
    return bytes;
  }

  hasMore(): boolean {
    return this.offset < this.data.length;
  }

  readField(): { fieldNumber: number; wireType: number } {
    const tag = this.readVarint();
    return {
      fieldNumber: tag >>> 3,
      wireType: tag & 7,
    };
  }
}

// Encode TensorData to protobuf
function encodeTensorData(tensor: TensorData): Uint8Array {
  const encoder = new ProtoEncoder();

  // Field 1: name
  if (tensor.name) {
    encoder.writeString(1, tensor.name);
  }

  // Field 2: dims (repeated int64)
  for (const dim of tensor.dims) {
    encoder.writeInt64(2, dim);
  }

  // Field 3: dtype
  if (tensor.dtype) {
    encoder.writeString(3, tensor.dtype);
  }

  // Field 4: data (repeated double)
  for (const value of tensor.data) {
    encoder.writeDouble(4, value);
  }

  return encoder.toUint8Array();
}

// Decode TensorData from protobuf
function decodeTensorData(data: Uint8Array): TensorData {
  const decoder = new ProtoDecoder(data);
  const tensor: TensorData = {
    name: '',
    dims: [],
    dtype: '',
    data: [],
  };

  while (decoder.hasMore()) {
    const { fieldNumber, wireType } = decoder.readField();

    if (fieldNumber === 1 && wireType === 2) {
      // name
      tensor.name = decoder.readString();
    } else if (fieldNumber === 2 && wireType === 0) {
      // dims
      tensor.dims.push(decoder.readVarint());
    } else if (fieldNumber === 3 && wireType === 2) {
      // dtype
      tensor.dtype = decoder.readString();
    } else if (fieldNumber === 4 && wireType === 1) {
      // data
      tensor.data.push(decoder.readDouble());
    }
  }

  return tensor;
}

// Encode InferenceRequest to protobuf
function encodeInferenceRequest(req: InferenceRequest): Uint8Array {
  const encoder = new ProtoEncoder();

  // Field 1: model_id (enum)
  encoder.writeEnum(1, req.model_id);

  // Field 2: input_data (message)
  const inputDataBytes = encodeTensorData(req.input_data);
  encoder.writeMessage(2, inputDataBytes);

  // Field 3: backend (enum)
  encoder.writeEnum(3, req.backend);

  return encoder.toUint8Array();
}

// Decode InferenceResponse from protobuf
function decodeInferenceResponse(data: Uint8Array): InferenceResponse {
  const decoder = new ProtoDecoder(data);
  let outputData: TensorData | null = null;
  let inferenceTimeMs = 0;

  while (decoder.hasMore()) {
    const { fieldNumber, wireType } = decoder.readField();

    if (fieldNumber === 1 && wireType === 2) {
      // output_data
      const tensorBytes = decoder.readMessage();
      outputData = decodeTensorData(tensorBytes);
    } else if (fieldNumber === 2 && wireType === 1) {
      // inference_time_ms
      inferenceTimeMs = decoder.readDouble();
    }
  }

  if (!outputData) {
    throw new Error('Missing output_data in response');
  }

  return {
    output_data: outputData,
    inference_time_ms: inferenceTimeMs,
  };
}

class GrpcClient {
  private readonly serviceUrl = 'http://localhost:50051';

  private async unaryCall(
    method: string,
    requestData: Uint8Array
  ): Promise<Uint8Array> {
    const response = await fetch(`${this.serviceUrl}/${method}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/grpc-web+proto',
        'X-Grpc-Web': '1',
      },
      body: requestData as BodyInit,
    });

    if (!response.ok) {
      throw new Error(`gRPC call failed: ${response.statusText}`);
    }

    // Extract the message from gRPC-Web framing
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);

    // gRPC-Web frame format: [1 byte flags][4 bytes length][message]
    if (data.length < 5) {
      throw new Error('Invalid gRPC-Web response');
    }

    const messageLength = new DataView(data.buffer, 1, 4).getUint32(0, false);
    return data.slice(5, 5 + messageLength);
  }

  async runInference(
    modelId: ModelId,
    inputData: TensorData,
    backend: Backend
  ): Promise<InferenceResponse> {
    const request: InferenceRequest = {
      model_id: modelId,
      input_data: inputData,
      backend: backend,
    };

    const requestData = encodeInferenceRequest(request);
    const responseData = await this.unaryCall(
      'onnx_service.OnnxInferenceService/RunInference',
      requestData
    );

    return decodeInferenceResponse(responseData);
  }
}

export const grpcClient = new GrpcClient();
export { ModelId, Backend };
export type { TensorData, InferenceRequest, InferenceResponse };
