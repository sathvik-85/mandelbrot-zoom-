import init, { render_strip, render_strip_perturb } from './pkg/mandelbrot_wasm.js';

const wasmReady = init();

self.onmessage = async ({ data }) => {
  await wasmReady;
  const { cx_hi, cx_lo, cy_hi, cy_lo, zoom, maxIter, canvasW, canvasH, yStart, yEnd, jobId, pal } = data;

  // Above zoom 1e11 use perturbation theory: one DD reference orbit per strip,
  // then f64 perturbations per pixel — ~8× faster than full DD per pixel.
  const pixels = zoom > 1e11
    ? render_strip_perturb(cx_hi, cx_lo, cy_hi, cy_lo, zoom, maxIter, canvasW, canvasH, yStart, yEnd, pal)
    : render_strip(cx_hi, cy_hi, zoom, maxIter, canvasW, canvasH, yStart, yEnd, pal);

  self.postMessage({ pixels, yStart, yEnd, jobId }, [pixels.buffer]);
};
