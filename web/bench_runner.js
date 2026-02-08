import { createPlan, uploadComplex } from "../src/index.js";

function el(id) {
  const x = document.getElementById(id);
  if (!x) throw new Error(`Missing element #${id}`);
  return x;
}

function appendLog(line) {
  const log = el("log");
  log.textContent += line + "\n";
  log.scrollTop = log.scrollHeight;
}

function setSummary(html) {
  el("summary").innerHTML = html;
}

function fmtMs(ms) {
  return `${ms.toFixed(2)} ms`;
}

function buildRequiredLimitsFromAdapter(adapter) {
  const lim = adapter?.limits;
  if (!lim) return null;
  const keys = [
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupSizeY",
    "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension",
  ];
  const out = {};
  for (const k of keys) {
    const v = lim[k];
    if (Number.isFinite(v) && v > 0) out[k] = Math.floor(v);
  }
  if (Number.isFinite(out.maxBufferSize) && Number.isFinite(out.maxStorageBufferBindingSize)) {
    out.maxStorageBufferBindingSize = Math.min(out.maxStorageBufferBindingSize, out.maxBufferSize);
  }
  return Object.keys(out).length ? out : null;
}

async function requestDeviceWithOptionalFeatures() {
  const gpu = globalThis.navigator?.gpu ?? globalThis.gpu;
  if (!gpu?.requestAdapter) throw new Error("WebGPU unavailable: navigator.gpu.requestAdapter missing");

  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Failed to acquire WebGPU adapter");

  const want = [];
  for (const f of ["shader-f16", "subgroups"]) {
    try {
      if (adapter.features?.has?.(f)) want.push(f);
    } catch {
      // ignore
    }
  }
  const requiredLimits = buildRequiredLimitsFromAdapter(adapter);
  const desc = want.length ? { requiredFeatures: want } : {};
  if (requiredLimits) desc.requiredLimits = requiredLimits;
  let device;
  try {
    device = await adapter.requestDevice(desc);
  } catch {
    device = await adapter.requestDevice(want.length ? { requiredFeatures: want } : {});
  }
  return { device, requiredFeatures: want };
}

function approxFftFlopsComplex(n, logN) {
  // Very rough FFT cost model for complex FFT:
  // ~5 * N * log2(N) floating-point ops (commonly used back-of-envelope).
  return 5 * n * logN;
}

async function runOnce(device, plan, input, output) {
  const enc = device.createCommandEncoder();
  plan.exec(enc, { input, output });
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function bench1dC2c1024(device) {
  const N = 1024;
  const plan = createPlan(device, {
    type: "c2c",
    shape: [N],
    direction: "forward",
    batch: 1,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  const inputData = new Float32Array(2 * N);
  for (let i = 0; i < inputData.length; i++) inputData[i] = (Math.random() * 2 - 1) * 0.1;
  const input = uploadComplex(device, inputData);
  const output = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const warmup = 3;
  for (let i = 0; i < warmup; i++) await runOnce(device, plan, input, output);

  const t0 = performance.now();
  await runOnce(device, plan, input, output);
  const coldMs = performance.now() - t0;

  const iters = 50;
  const t1 = performance.now();
  for (let i = 0; i < iters; i++) await runOnce(device, plan, input, output);
  const totalMs = performance.now() - t1;
  const avgMs = totalMs / iters;

  const flops = approxFftFlopsComplex(N, Math.log2(N));
  const gflops = (flops / (avgMs / 1000)) / 1e9;

  plan.destroy();
  input.destroy();
  output.destroy();
  return { name: "1D C2C N=1024", coldMs, avgMs, gflops };
}

function pick2dShapeWithinLimits(device, target = 1024) {
  const maxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
  let n = target;
  while (n >= 64) {
    const bytes = n * n * 8; // complex<f32>
    if (bytes <= maxBind) return [n, n];
    n = Math.floor(n / 2);
  }
  return [64, 64];
}

async function bench2dTranspose(device) {
  const [Nx, Ny] = pick2dShapeWithinLimits(device, 1024);
  const total = Nx * Ny;

  const plan = createPlan(device, {
    type: "c2c",
    shape: [Nx, Ny],
    direction: "forward",
    batch: 1,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  const inputData = new Float32Array(2 * total);
  for (let i = 0; i < inputData.length; i++) inputData[i] = (Math.random() * 2 - 1) * 0.1;
  const input = uploadComplex(device, inputData);
  const output = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const warmup = 2;
  for (let i = 0; i < warmup; i++) await runOnce(device, plan, input, output);

  const iters = Math.max(3, Math.floor(2000000 / total)); // fewer iters for bigger shapes
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) await runOnce(device, plan, input, output);
  const totalMs = performance.now() - t0;
  const avgMs = totalMs / iters;

  const flops = approxFftFlopsComplex(total, Math.log2(Nx) + Math.log2(Ny));
  const gflops = (flops / (avgMs / 1000)) / 1e9;

  const usedTranspose = !!plan.transpose; // internal hint (not part of stable API)

  plan.destroy();
  input.destroy();
  output.destroy();
  return { name: `2D C2C ${Nx}x${Ny}`, avgMs, gflops, usedTranspose };
}

async function benchMixedRadix2310(device) {
  const N = 2 * 3 * 5 * 7 * 11; // 2310
  const plan = createPlan(device, {
    type: "c2c",
    shape: [N],
    direction: "forward",
    batch: 1,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  const inputData = new Float32Array(2 * N);
  for (let i = 0; i < inputData.length; i++) inputData[i] = (Math.random() * 2 - 1) * 0.1;
  const input = uploadComplex(device, inputData);
  const output = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const warmup = 2;
  for (let i = 0; i < warmup; i++) await runOnce(device, plan, input, output);

  const iters = 40;
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) await runOnce(device, plan, input, output);
  const totalMs = performance.now() - t0;
  const avgMs = totalMs / iters;

  const flops = approxFftFlopsComplex(N, Math.log2(N));
  const gflops = (flops / (avgMs / 1000)) / 1e9;

  plan.destroy();
  input.destroy();
  output.destroy();
  return { name: "1D C2C N=2310 (mixed radix)", avgMs, gflops };
}

async function benchFftConv3dMultiKernel(device) {
  const shape = [16, 8, 4];
  const batch = 2;
  const kernelCount = 3;
  const n = shape[0] * shape[1] * shape[2];

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount, outputLayout: "kernel-major" },
  });

  const inputData = new Float32Array(2 * n * batch);
  for (let i = 0; i < inputData.length; i++) inputData[i] = (Math.random() * 2 - 1) * 0.1;
  const kernels = new Float32Array(2 * n * kernelCount);
  for (let i = 0; i < kernels.length; i++) kernels[i] = (Math.random() * 2 - 1) * 0.1;

  const input = uploadComplex(device, inputData);
  const output = device.createBuffer({
    size: n * batch * kernelCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const warmup = 2;
  for (let i = 0; i < warmup; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input, output, kernel: kernels });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const iters = 20;
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input, output, kernel: kernels });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const totalMs = performance.now() - t0;
  const avgMs = totalMs / iters;

  plan.destroy();
  input.destroy();
  output.destroy();
  return { name: `fftconv 3D ${shape.join("x")} batch=${batch} kernels=${kernelCount}`, avgMs };
}

export function initBenchUI() {
  const btnBench = el("btnBench");
  const btnInit = el("btnInit");

  let device = null;

  async function ensureDevice() {
    if (device) return device;
    const res = await requestDeviceWithOptionalFeatures();
    device = res.device;
    return device;
  }

  // If tests init ran first, don't fight it; best-effort share global.
  btnInit.addEventListener("click", async () => {
    try {
      await ensureDevice();
    } catch {
      // test_runner will surface init errors
    }
  });

  btnBench.addEventListener("click", async () => {
    el("log").textContent = "";
    setSummary("Status: running benchmarks...");
    try {
      const dev = await ensureDevice();
      appendLog(`Running on enabled features: ${[...dev.features].join(", ") || "(none)"}`);

      const a = await bench1dC2c1024(dev);
      appendLog(`${a.name}: cold=${fmtMs(a.coldMs)} avg=${fmtMs(a.avgMs)} approx=${a.gflops.toFixed(2)} GFLOP/s`);

      const b = await bench2dTranspose(dev);
      appendLog(`${b.name}: avg=${fmtMs(b.avgMs)} approx=${b.gflops.toFixed(2)} GFLOP/s transpose=${b.usedTranspose}`);

      const c = await benchMixedRadix2310(dev);
      appendLog(`${c.name}: avg=${fmtMs(c.avgMs)} approx=${c.gflops.toFixed(2)} GFLOP/s`);

      const d = await benchFftConv3dMultiKernel(dev);
      appendLog(`${d.name}: avg=${fmtMs(d.avgMs)}`);

      setSummary(`<span class="pass">PASS</span> – benchmarks completed`);
    } catch (e) {
      appendLog(String(e?.stack ?? e));
      setSummary(`<span class="fail">FAIL</span> – benchmark error`);
    }
  });
}
