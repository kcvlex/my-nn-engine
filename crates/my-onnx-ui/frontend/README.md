# ONNX Web Frontend

Vue.js + TypeScript frontend for the ONNX Inference Server.

## Development

1. Install dependencies:
```bash
pnpm install
```

2. Run development server:
```bash
pnpm run dev
```

This will start a Vite dev server at http://localhost:5173

## Building for Production

Build the frontend (outputs to `../dist`):
```bash
pnpm run build
```

The built files will be served by the Rust web server.

**Note:** The frontend is automatically built when you run `cargo build` on the parent crate!

## Stack

- **Vue 3** - Progressive JavaScript framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool and dev server
- **pnpm** - Fast, disk space efficient package manager
