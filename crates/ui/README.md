# my-nn-engine-ui

A web UI for running inference on a fixed set of ONNX models against the
`my-nn-engine` runtime.

## Prerequisites

- Rust nightly (workspace already pins it via `rustfmt.toml`)
- `pnpm` and Node.js 22+
- `protoc` (`apt install protobuf-compiler` or equivalent)
- The validated ONNX model files. From the workspace root:

  ```bash
  cd models/validated
  make
  ```

  This downloads `mnist-12`, `resnet18-v2-7`, `resnet152-v2-7`,
  `mobilenetv2-12`, `efficientnet-lite4-11`, `bertsquad-12`, `yolov4`, and
  `gpt2-10` into `models/validated/`.

## Run

You need two terminals: one for the backend (gRPC server on `:50051`), one for
the frontend dev server (`:5173`).

### Backend

From the workspace root:

```bash
cargo run -p my-nn-engine-ui --bin web-server
```

CUDA is enabled by default. For a CPU-only build:

```bash
cargo run -p my-nn-engine-ui --bin web-server --no-default-features
```

Override the port with `PORT=...` if `:50051` is taken.

### Frontend

```bash
cd crates/ui/frontend
pnpm install
pnpm generate   # build TS bindings from proto/onnx_service.proto
pnpm dev        # http://localhost:5173
```

Open <http://localhost:5173> in a browser, pick a model and a backend.
