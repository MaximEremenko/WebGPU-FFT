import { createPlan, uploadComplex } from "../src/index.js";

async function getWebGpuDevice() {
  const gpu = globalThis.navigator?.gpu ?? globalThis.gpu;
  if (!gpu?.requestAdapter) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  return adapter.requestDevice();
}

function nowMs() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function randomComplexInterleaved(n) {
  const out = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    out[2 * i] = (Math.random() * 2 - 1) * 0.5;
    out[2 * i + 1] = (Math.random() * 2 - 1) * 0.5;
  }
  return out;
}

function randomReal(n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (Math.random() * 2 - 1) * 0.5;
  return out;
}

async function benchC2c1D(device) {
  const N = 1024;
  const input = randomComplexInterleaved(N);
  const inputBuf = uploadComplex(device, input);
  const outputBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

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

  const warmup = 10;
  for (let i = 0; i < warmup; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const iters = 200;
  const t0 = nowMs();
  for (let i = 0; i < iters; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const t1 = nowMs();

  console.log(`c2c 1D N=${N} iters=${iters} avg=${((t1 - t0) / iters).toFixed(3)} ms`);
  plan.destroy();
  inputBuf.destroy();
  outputBuf.destroy();
}

async function benchC2c2DTranspose(device) {
  // This shape triggers the optional transpose path in current implementation (product >= 4096 and both axes mixed).
  const Nx = 64;
  const Ny = 64;
  const nTotal = Nx * Ny;
  const input = randomComplexInterleaved(nTotal);
  const inputBuf = uploadComplex(device, input);
  const outputBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

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

  const warmup = 10;
  for (let i = 0; i < warmup; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const iters = 100;
  const t0 = nowMs();
  for (let i = 0; i < iters; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const t1 = nowMs();

  console.log(`c2c 2D Nx=${Nx} Ny=${Ny} iters=${iters} avg=${((t1 - t0) / iters).toFixed(3)} ms`);
  plan.destroy();
  inputBuf.destroy();
  outputBuf.destroy();
}

async function benchFftConv3DMultiKernel(device) {
  const shape = [16, 8, 4];
  const batch = 2;
  const kernelCount = 3;
  const n = shape.reduce((a, b) => a * b, 1);

  const input = randomComplexInterleaved(n * batch);
  const kernels = randomComplexInterleaved(n * kernelCount);
  const inputBuf = uploadComplex(device, input);
  const outputBuf = device.createBuffer({
    size: n * batch * kernelCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount, outputLayout: "kernel-major" },
  });

  const warmup = 5;
  for (let i = 0; i < warmup; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf, kernel: kernels });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const iters = 40;
  const t0 = nowMs();
  for (let i = 0; i < iters; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf, kernel: kernels });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const t1 = nowMs();

  console.log(`fftconv 3D shape=${shape.join("x")} batch=${batch} kernels=${kernelCount} iters=${iters} avg=${((t1 - t0) / iters).toFixed(3)} ms`);

  plan.destroy();
  inputBuf.destroy();
  outputBuf.destroy();
}

async function benchR2cC2rOversizedLineLargeMode(device) {
  const shape = [32, 8];
  const n = shape.reduce((a, b) => a * b, 1);
  const packedN = (Math.floor(shape[0] / 2) + 1) * shape[1];
  const inputReal = randomReal(n);
  const inBuf = device.createBuffer({
    size: inputReal.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inBuf, 0, inputReal);
  const packedBuf = device.createBuffer({
    size: packedN * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outBuf = device.createBuffer({
    size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const r2c = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch: 1,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  const c2r = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch: 1,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  const warmup = 5;
  for (let i = 0; i < warmup; i++) {
    const enc = device.createCommandEncoder();
    r2c.exec(enc, { input: inBuf, output: packedBuf });
    c2r.exec(enc, { input: packedBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const iters = 50;
  const t0 = nowMs();
  for (let i = 0; i < iters; i++) {
    const enc = device.createCommandEncoder();
    r2c.exec(enc, { input: inBuf, output: packedBuf });
    c2r.exec(enc, { input: packedBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const t1 = nowMs();
  console.log(
    `r2c+c2r oversized-line large-mode shape=${shape.join("x")} iters=${iters} avg=${((t1 - t0) / iters).toFixed(3)} ms`
  );

  r2c.destroy();
  c2r.destroy();
  inBuf.destroy();
  packedBuf.destroy();
  outBuf.destroy();
}

async function benchFftConvForcedLargeLinearSame(device) {
  const shape = [32];
  const kernelShape = [5];
  const batch = 1;
  const inputData = randomComplexInterleaved(shape[0] * batch);
  const kernelData = randomComplexInterleaved(kernelShape[0]);
  const inputBuf = uploadComplex(device, inputData);
  const outputBuf = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 256 },
    fftConv: {
      mode: "convolution",
      boundary: "linear-same",
      kernelShape,
      kernelCount: 1,
      outputLayout: "kernel-major",
    },
  });

  const warmup = 5;
  for (let i = 0; i < warmup; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf, kernel: kernelData });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const iters = 60;
  const t0 = nowMs();
  for (let i = 0; i < iters; i++) {
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf, kernel: kernelData });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const t1 = nowMs();
  console.log(
    `fftconv forced-large linear-same shape=${shape[0]} kernel=${kernelShape[0]} iters=${iters} avg=${((t1 - t0) / iters).toFixed(3)} ms ` +
      `largeMode=${!!plan._largeMode} batchSliced=${!!plan._batchSlicedExecution}`
  );

  plan.destroy();
  inputBuf.destroy();
  outputBuf.destroy();
}

async function benchC2cOutOfCoreNonMixedAxis(device) {
  const cases = [
    {
      name: "bluestein",
      shape: [4, 17],
      tuning: { maxStorageBufferBindingSize: 320, forceBluesteinAxes: [1] },
    },
    {
      name: "rader",
      shape: [4, 29],
      tuning: { maxStorageBufferBindingSize: 480, forceRaderAxes: [1] },
    },
    {
      name: "bluestein-rank4",
      shape: [3, 2, 2, 34],
      tuning: { maxStorageBufferBindingSize: 512, forceBluesteinAxes: [3] },
    },
    {
      name: "rader-rank4",
      shape: [3, 2, 2, 29],
      tuning: { maxStorageBufferBindingSize: 512, forceRaderAxes: [3] },
    },
    {
      name: "bluestein-rank4-oversized-line",
      shape: [3, 2, 2, 34],
      tuning: { maxStorageBufferBindingSize: 160, forceBluesteinAxes: [3] },
    },
    {
      name: "rader-rank4-oversized-line",
      shape: [3, 2, 2, 29],
      tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [3] },
    },
    {
      name: "rader-rank5-oversized-line",
      shape: [2, 2, 2, 2, 29],
      tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [4] },
    },
  ];
  for (const cfg of cases) {
    const total = cfg.shape.reduce((a, b) => a * b, 1) * 2;
    const input = randomComplexInterleaved(total);
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const plan = createPlan(device, {
      type: "c2c",
      shape: cfg.shape,
      direction: "forward",
      batch: 2,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: cfg.tuning,
    });

    const warmup = 3;
    for (let i = 0; i < warmup; i++) {
      const enc = device.createCommandEncoder();
      plan.exec(enc, { input: inBuf, output: outBuf });
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    const iters = 30;
    const t0 = nowMs();
    for (let i = 0; i < iters; i++) {
      const enc = device.createCommandEncoder();
      plan.exec(enc, { input: inBuf, output: outBuf });
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const t1 = nowMs();
    console.log(`c2c out-of-core forced-${cfg.name} shape=${cfg.shape.join("x")} batch=2 avg=${((t1 - t0) / iters).toFixed(3)} ms`);

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  }
}

async function benchR2cC2rOutOfCoreNonMixedRank4(device) {
  const cases = [
    {
      name: "bluestein-rank4",
      shape: [3, 2, 2, 17],
      tuning: { maxStorageBufferBindingSize: 1024, forceBluesteinAxes: [3] },
    },
    {
      name: "rader-rank4",
      shape: [3, 2, 2, 29],
      tuning: { maxStorageBufferBindingSize: 1024, forceRaderAxes: [3] },
    },
    {
      name: "rader-rank4-oversized-line",
      shape: [3, 2, 2, 29],
      tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [3] },
    },
    {
      name: "rader-rank5-oversized-line",
      shape: [2, 2, 2, 2, 29],
      tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [4] },
    },
  ];

  for (const cfg of cases) {
    const batch = 2;
    const nReal = cfg.shape.reduce((a, b) => a * b, 1) * batch;
    const packedN = (Math.floor(cfg.shape[0] / 2) + 1) * cfg.shape.slice(1).reduce((a, b) => a * b, 1) * batch;
    const inputReal = randomReal(nReal);
    const inBuf = device.createBuffer({
      size: inputReal.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inBuf, 0, inputReal);
    const packedBuf = device.createBuffer({
      size: packedN * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBuf = device.createBuffer({
      size: nReal * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const r2c = createPlan(device, {
      type: "r2c",
      shape: cfg.shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: cfg.tuning,
    });
    const c2r = createPlan(device, {
      type: "c2r",
      shape: cfg.shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: cfg.tuning,
    });

    const warmup = 3;
    for (let i = 0; i < warmup; i++) {
      const enc = device.createCommandEncoder();
      r2c.exec(enc, { input: inBuf, output: packedBuf });
      c2r.exec(enc, { input: packedBuf, output: outBuf });
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    const iters = 20;
    const t0 = nowMs();
    for (let i = 0; i < iters; i++) {
      const enc = device.createCommandEncoder();
      r2c.exec(enc, { input: inBuf, output: packedBuf });
      c2r.exec(enc, { input: packedBuf, output: outBuf });
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const t1 = nowMs();
    console.log(
      `r2c+c2r forced-${cfg.name} shape=${cfg.shape.join("x")} batch=${batch} avg=${((t1 - t0) / iters).toFixed(3)} ms`
    );

    r2c.destroy();
    c2r.destroy();
    inBuf.destroy();
    packedBuf.destroy();
    outBuf.destroy();
  }
}

const device = await getWebGpuDevice();
if (!device) {
  console.log("WebGPU unavailable; skipping benchmark.");
  process.exit(0);
}

await benchC2c1D(device);
await benchC2c2DTranspose(device);
await benchFftConv3DMultiKernel(device);
await benchFftConvForcedLargeLinearSame(device);
await benchR2cC2rOversizedLineLargeMode(device);
await benchC2cOutOfCoreNonMixedAxis(device);
await benchR2cC2rOutOfCoreNonMixedRank4(device);

