import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { OnnxInferenceService } from '../gen/onnx_service_pb';

const transport = createGrpcWebTransport({
  baseUrl: 'http://localhost:50051',
});

export const grpcClient = createClient(OnnxInferenceService, transport);
