# ONNX Web Server

A modern web-based frontend for the my-onnx inference engine. Upload ONNX models through a Vue.js interface and run inference via REST API.

## Features

- 🚀 **Model Upload**: Upload `.onnx` files via web interface
- ⚡ **Session Caching**: Compiles models once, reuses compiled code
- 🖥️ **CPU & CUDA Support**: Choose between CPU or GPU execution
- 🌐 **REST API**: Simple HTTP API for integration
- 📊 **Modern Web UI**: Vue.js + TypeScript frontend with clean interface
- 🎨 **Responsive Design**: Works on desktop and mobile

## Prerequisites

### For Backend (Required)
```bash
sudo apt install clang libopenblas-dev
```

### For CUDA Backend (Optional)
```bash
# Install NVIDIA CUDA Toolkit (includes nvcc)
# Install cuDNN and cuBLAS libraries
# Ensure nvidia-smi is available
```

### For Frontend Development (Optional)
```bash
# Node.js 18+ and pnpm
# Only needed if you want to modify the frontend
npm install -g pnpm
# or: curl -fsSL https://get.pnpm.io/install.sh | sh -
```

## Building

The frontend is **automatically built** during the Cargo build process!

Simply build the web server:

```bash
cargo build -p my-onnx-web --release --bin web-server
```

Or from the workspace root:

```bash
cargo build -p my-onnx-web --release --bin web-server
```

The build script will:
1. Check if pnpm is installed
2. Run `pnpm install` if `node_modules` doesn't exist
3. Run `pnpm run build` to create optimized production files in `dist/`
4. Build the Rust web server

**Note:** If pnpm is not found, the build will continue with a warning, but you'll need to build the frontend manually:

```bash
cd frontend
pnpm install
pnpm run build
```

## Running

Start the server:

```bash
cargo run -p my-onnx-web --release --bin web-server
```

The server will start on `http://localhost:3000`

## Development

### Frontend Development

To work on the frontend with hot reload:

```bash
cd frontend
pnpm install  # First time only
pnpm run dev
```

This starts a Vite dev server at http://localhost:5173 that proxies API calls to the backend.

Make sure the backend server is also running:

```bash
cargo run -p my-onnx-web --bin web-server
```

## Directory Structure

```
crates/my-onnx-web/
├── frontend/           - Vue.js + TypeScript frontend
│   ├── src/
│   │   ├── components/ - Vue components
│   │   ├── api/        - API client
│   │   └── types/      - TypeScript types
│   └── dist/           - Built frontend (generated)
├── src/                - Rust backend
│   ├── main.rs         - Web server binary
│   ├── api.rs          - API handlers
│   └── cache.rs        - Session cache
├── cache/              - Compiled model cache (auto-created)
└── uploads/            - Uploaded ONNX files (auto-created)
```

## Usage

### Via Web UI

1. Open `http://localhost:3000` in your browser
2. **Upload Tab**:
   - Select an `.onnx` file
   - Choose target (CPU/CUDA)
   - Click "Upload & Compile"
   - Copy the generated Model ID
3. **Run Inference Tab**:
   - Paste the Model ID
   - Provide input tensors as JSON
   - Click "Run Inference"
4. **My Models Tab**:
   - View all cached models

### Via REST API

#### Upload Model
```bash
curl -X POST http://localhost:3000/models/upload \
  -F "file=@model.onnx" \
  -F "target=CPU" \
  -F "optimize=true"
```

Response:
```json
{
  "model_id": "abc123def456...",
  "message": "Model uploaded and compiled successfully"
}
```

#### Run Inference (JSON)
```bash
curl -X POST http://localhost:3000/models/{model_id}/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "data": [0.1, 0.2, 0.3, ...],
        "dims": [1, 3, 224, 224],
        "dtype": "Float(F32)"
      }
    ]
  }'
```

Response:
```json
{
  "outputs": [
    {
      "name": "output_0",
      "data": [...],
      "dims": [1, 1000],
      "dtype": "Float(F32)"
    }
  ],
  "inference_time_ms": 42.5
}
```

#### Run Inference (Protobuf - Recommended)

**More efficient**: Uses ONNX's native protobuf format for tensor serialization.

```bash
# Using binary protobuf tensors
curl -X POST http://localhost:3000/models/{model_id}/infer/proto \
  -H "Content-Type: application/octet-stream" \
  --data-binary @input_tensors.pb \
  -o output_tensors.pb
```

**Format**: Request and response are binary data with concatenated ONNX TensorProto messages:
- `[4 bytes length (LE)][TensorProto bytes][4 bytes length][TensorProto bytes]...`

**Advantages**:
- **Efficient**: Native binary format, no JSON parsing overhead
- **Standard**: Uses official ONNX TensorProto format
- **Type-safe**: Preserves exact data types (int32, int64, float32, float64, etc.)
- **Interoperable**: Compatible with ONNX ecosystem tools

The frontend automatically uses this endpoint when available for better performance.

#### List Models
```bash
curl http://localhost:3000/models
```

## Input Tensor Format

Input tensors should be provided as JSON with:
- `name`: Tensor name (string)
- `data`: Flattened array of values (array of numbers)
- `dims`: Shape/dimensions (array of integers)
- `dtype`: Data type (string, e.g., "Float(F32)")

Example for a 1x3x224x224 image tensor:
```json
{
  "name": "input",
  "data": [/* 150528 floats */],
  "dims": [1, 3, 224, 224],
  "dtype": "Float(F32)"
}
```

## How It Works

### Session Caching

1. When a model is uploaded, the system computes: `hash = SHA256(model_file + options)`
2. If the hash exists in cache, return the cached compiled session
3. If not, compile using `clang`/`nvcc` and store in cache
4. Subsequent requests reuse the compiled `.so` file (no recompilation!)

### Compilation Flow

**CPU Backend:**
```
ONNX Model → LLVM IR → clang → .so → libloading → Session
```

**CUDA Backend:**
```
ONNX Model → CUDA C++ → nvcc → .so → libloading → Session
```

### Architecture

```
Browser → HTTP → Axum Web Server → SessionCache → my-onnx::Session → clang/nvcc
```

## Configuration

Edit `examples/web_server.rs` to customize:
- Port number (default: 3000)
- Cache directory (default: ./cache)
- Upload directory (default: ./uploads)
- Logging level (default: info)

## Troubleshooting

### "clang not found"
Install clang: `sudo apt install clang`

### "nvcc not found"
Install CUDA Toolkit or switch to CPU target

### "libopenblas not found"
Install OpenBLAS: `sudo apt install libopenblas-dev`

### Model compilation fails
- Check the uploaded `.onnx` file is valid
- Review server logs for detailed error messages
- Ensure all required system libraries are installed

### Large models timeout
Increase the request timeout in your HTTP client. Initial compilation can take 30s-2min for large models.

## Performance Notes

- **First request**: Slow (compilation time)
- **Subsequent requests**: Fast (cached session)
- **Concurrent requests**: Safe (thread-safe cache)
- **Memory usage**: Models stay in memory until server restart

## Security Considerations

⚠️ **This is designed for local/trusted use only:**

- No authentication/authorization
- Executes arbitrary compilation commands (`clang`, `nvcc`)
- No rate limiting
- No input validation beyond basic checks

For production use, add:
- API authentication (tokens, OAuth)
- Input sanitization
- Resource limits (memory, CPU, disk)
- Sandboxed compilation (containers, namespaces)
- HTTPS/TLS encryption

## License

Same as the parent my-onnx project.
