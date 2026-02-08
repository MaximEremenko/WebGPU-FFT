import { createFftPlan, uploadComplex } from "../src/index.js";

async function getWebGpuDevice() {
  const gpu = globalThis.navigator?.gpu ?? globalThis.gpu;
  if (!gpu?.requestAdapter) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  return adapter.requestDevice();
}

function randomComplexInterleaved(n) {
  const out = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    out[2 * i] = (Math.random() * 2 - 1) * 0.5;
    out[2 * i + 1] = (Math.random() * 2 - 1) * 0.5;
  }
  return out;
}

const device = await getWebGpuDevice();
if (!device) {
  console.log("WebGPU unavailable; skipping benchmark.");
  process.exit(0);
}

const N = 1024;
const plan = createFftPlan(device, {
  shape: [N],
  direction: "forward",
  normalize: "none",
  inPlace: false,
  layout: "interleaved",
  precision: "f32",
});

const input = randomComplexInterleaved(N);
const inputBuf = uploadComplex(device, input);
const outputBuf = device.createBuffer({
  size: input.byteLength,
  usage:
    GPUBufferUsage.STORAGE |
    GPUBufferUsage.COPY_SRC |
    GPUBufferUsage.COPY_DST,
});

const warmup = 10;
for (let i = 0; i < warmup; i++) {
  const enc = device.createCommandEncoder();
  plan.exec(enc, { input: inputBuf, output: outputBuf });
  device.queue.submit([enc.finish()]);
}
await device.queue.onSubmittedWorkDone();

const iters = 200;
const t0 = (globalThis.performance?.now?.() ?? Date.now());
for (let i = 0; i < iters; i++) {
  const enc = device.createCommandEncoder();
  plan.exec(enc, { input: inputBuf, output: outputBuf });
  device.queue.submit([enc.finish()]);
}
await device.queue.onSubmittedWorkDone();
const t1 = (globalThis.performance?.now?.() ?? Date.now());

const avg = (t1 - t0) / iters;
console.log(`FFT N=${N} iters=${iters} avg=${avg.toFixed(3)} ms`);

plan.destroy();
inputBuf.destroy();
outputBuf.destroy();
