import type { TensorProto } from '../gen/onnx.proto3_pb';

export function serializeOutputs(
  outputs: TensorProto[],
  { summarize = false } = {},
): string {
  return JSON.stringify(
    outputs.map((t) => ({
      name: t.name,
      dims: t.dims.map(Number),
      dataType: t.dataType,
      ...(summarize
        ? {
            floatDataLength: t.floatData.length,
            doubleDataLength: t.doubleData.length,
            int32DataLength: t.int32Data.length,
            int64DataLength: t.int64Data.length,
          }
        : {
            floatData: t.floatData,
            doubleData: t.doubleData,
            int32Data: t.int32Data,
            int64Data: t.int64Data.map(Number),
          }),
    })),
    null,
    2,
  );
}
