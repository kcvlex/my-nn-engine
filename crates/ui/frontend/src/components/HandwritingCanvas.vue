<script setup lang="ts">
import { ref, onMounted, useTemplateRef } from 'vue';

const CANVAS_SIZE = 280;
const TARGET_SIZE = 28;
const BRUSH_PX = 16;

const drawCanvas = useTemplateRef<HTMLCanvasElement>('drawCanvas');
const previewCanvas = useTemplateRef<HTMLCanvasElement>('previewCanvas');

const tensorData = ref<number[]>([]);
const hasDrawing = ref(false);

let drawing = false;
let lastX = 0;
let lastY = 0;

function clientToCanvas(e: PointerEvent): { x: number; y: number } {
  const rect = drawCanvas.value!.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) * CANVAS_SIZE) / rect.width,
    y: ((e.clientY - rect.top) * CANVAS_SIZE) / rect.height,
  };
}

function onPointerDown(e: PointerEvent) {
  if (!drawCanvas.value) return;
  drawCanvas.value.setPointerCapture(e.pointerId);
  drawing = true;
  const p = clientToCanvas(e);
  lastX = p.x;
  lastY = p.y;
  // Drop a single dot so a tap registers.
  const ctx = drawCanvas.value.getContext('2d')!;
  ctx.fillStyle = 'white';
  ctx.beginPath();
  ctx.arc(p.x, p.y, BRUSH_PX / 2, 0, Math.PI * 2);
  ctx.fill();
  hasDrawing.value = true;
}

function onPointerMove(e: PointerEvent) {
  if (!drawing || !drawCanvas.value) return;
  const p = clientToCanvas(e);
  const ctx = drawCanvas.value.getContext('2d')!;
  ctx.strokeStyle = 'white';
  ctx.lineWidth = BRUSH_PX;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastX = p.x;
  lastY = p.y;
}

function onPointerUp() {
  if (!drawing) return;
  drawing = false;
  updatePreviewAndTensor();
}

function updatePreviewAndTensor() {
  if (!drawCanvas.value || !previewCanvas.value) return;
  const pCtx = previewCanvas.value.getContext('2d')!;
  pCtx.clearRect(0, 0, TARGET_SIZE, TARGET_SIZE);
  pCtx.drawImage(drawCanvas.value, 0, 0, TARGET_SIZE, TARGET_SIZE);

  const imageData = pCtx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE);
  const data: number[] = new Array(TARGET_SIZE * TARGET_SIZE);
  for (let i = 0, j = 0; i < imageData.data.length; i += 4, j++) {
    // White-on-black strokes, so any RGB channel works as luminance.
    data[j] = imageData.data[i] / 255.0;
  }
  tensorData.value = data;
}

function clear() {
  if (!drawCanvas.value || !previewCanvas.value) return;
  const ctx = drawCanvas.value.getContext('2d')!;
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const pCtx = previewCanvas.value.getContext('2d')!;
  pCtx.fillStyle = 'black';
  pCtx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
  tensorData.value = [];
  hasDrawing.value = false;
}

onMounted(() => {
  clear();
});

defineExpose({ tensorData, hasDrawing, clear });
</script>

<template>
  <div class="handwriting">
    <div class="canvases">
      <div class="canvas-box">
        <label>Draw a digit (0-9)</label>
        <canvas
          ref="drawCanvas"
          :width="CANVAS_SIZE"
          :height="CANVAS_SIZE"
          class="draw-canvas"
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @pointercancel="onPointerUp"
        ></canvas>
      </div>
      <div class="canvas-box">
        <label>28x28 Input</label>
        <canvas
          ref="previewCanvas"
          :width="TARGET_SIZE"
          :height="TARGET_SIZE"
          class="preview-canvas"
        ></canvas>
      </div>
    </div>
    <button type="button" class="clear-btn" @click="clear">Clear</button>
  </div>
</template>

<style scoped>
.handwriting {
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: flex-start;
}

.canvases {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

.canvas-box {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.canvas-box label {
  font-size: 0.85rem;
  color: var(--fg-dim);
}

.draw-canvas {
  width: 280px;
  height: 280px;
  background: black;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: crosshair;
  touch-action: none;
}

.preview-canvas {
  width: 112px;
  height: 112px;
  image-rendering: pixelated;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.clear-btn {
  align-self: flex-start;
}
</style>
