import { ref, shallowRef } from 'vue';

export function useImageCanvas(
  width: number,
  height: number,
  pixelTransform: (pixels: Uint8ClampedArray) => number[],
  options?: { fillStyle?: string },
) {
  const canvas = ref<HTMLCanvasElement>();
  const tensorData = shallowRef<number[]>([]);

  function processImage(img: HTMLImageElement) {
    const c = canvas.value;
    if (!c) return;
    const ctx = c.getContext('2d')!;
    if (options?.fillStyle) {
      ctx.fillStyle = options.fillStyle;
      ctx.fillRect(0, 0, width, height);
    }
    ctx.drawImage(img, 0, 0, width, height);
    tensorData.value = pixelTransform(
      ctx.getImageData(0, 0, width, height).data,
    );
  }

  return { canvas, processImage, tensorData };
}
