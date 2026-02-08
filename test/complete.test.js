/* Legacy node:test file (disabled).
   The reusable definitions live in `test/complete.suite.js` and the Node runner is `test/complete.node.test.js`.

import test, { describe, before } from "node:test";
import assert from "node:assert/strict";

import { createPlan, BufferView, uploadComplex, downloadComplex } from "../src/index.js";
import {
  randomComplexInterleaved,
  fftNdRefAnySizeInterleaved,
  dct1Ref,
  dct2Ref,
  dct3Ref,
  dct4Ref,
  r2cRefPackedInterleaved,
  c2rRefFromPackedInterleaved,
  conv2dRef,
  normalizeScaleFactor as cpuNorm,
} from "../src/utils/math.js";

async function getWebGpuDevice() {
  const gpu = globalThis.navigator?.gpu ?? globalThis.gpu;
  if (!gpu?.requestAdapter) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  return adapter.requestDevice();
}

function assertAllClose(actual, expected, { atol = 1e-4, rtol = 1e-4 } = {}) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const tol = atol + rtol * Math.abs(e);
    const diff = Math.abs(a - e);
    if (diff > tol) {
      throw new assert.AssertionError({
        message: `Mismatch at [${i}]: actual=${a} expected=${e} diff=${diff} tol=${tol}`,
        actual: a,
        expected: e,
      });
    }
  }
}

function f32ToF16Bits(x) {
  // IEEE754 float32 -> float16 (approx round-to-nearest)
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = x;
  const v = u32[0];
  const sign = (v >>> 31) & 1;
  let exp = (v >>> 23) & 0xff;
  let mant = v & 0x7fffff;
  if (exp === 0xff) {
    const nan = mant !== 0;
    return (sign << 15) | (0x1f << 10) | (nan ? 0x200 : 0);
  }
  if (exp === 0) {
    if (mant === 0) return sign << 15;
    while ((mant & 0x800000) === 0) {
      mant <<= 1;
      exp--;
    }
    mant &= 0x7fffff;
    exp++;
  }
  const exp16 = exp - 127 + 15;
  if (exp16 >= 0x1f) return (sign << 15) | (0x1f << 10);
  if (exp16 <= 0) {
    if (exp16 < -10) return sign << 15;
    mant |= 0x800000;
    const shift = 14 - exp16;
    let m = mant >>> shift;
    if ((mant >>> (shift - 1)) & 1) m += 1;
    return (sign << 15) | m;
  }
  let m = mant >>> 13;
  if (mant & 0x1000) m += 1;
  if (m === 0x400) {
    m = 0;
    const e = exp16 + 1;
    if (e >= 0x1f) return (sign << 15) | (0x1f << 10);
    return (sign << 15) | (e << 10) | m;
  }
  return (sign << 15) | (exp16 << 10) | (m & 0x3ff);
}

function f16BitsToF32(h) {
  const sign = (h >>> 15) & 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  let v;
  if (exp === 0) {
    if (mant === 0) v = sign << 31;
    else {
      let m = mant;
      let e = -14;
      while ((m & 0x400) === 0) {
        m <<= 1;
        e--;
      }
      m &= 0x3ff;
      const exp32 = e + 127;
      v = (sign << 31) | (exp32 << 23) | (m << 13);
    }
  } else if (exp === 0x1f) {
    v = (sign << 31) | (0xff << 23) | (mant ? 0x200000 : 0);
  } else {
    const exp32 = exp - 15 + 127;
    v = (sign << 31) | (exp32 << 23) | (mant << 13);
  }
  const u32 = new Uint32Array(1);
  u32[0] = v >>> 0;
  return new Float32Array(u32.buffer)[0];
}

function splitBufferView(buffer, totalBytes, segments) {
  const segSize = Math.floor(totalBytes / segments / 4) * 4;
  const segs = [];
  let off = 0;
  for (let i = 0; i < segments; i++) {
    const size = i === segments - 1 ? totalBytes - off : segSize;
    segs.push({ buffer, offsetBytes: off, sizeBytes: size });
    off += size;
  }
  return new BufferView({ segments: segs, logicalByteOffset: 0, lengthBytes: totalBytes });
}

describe("createPlan complete (GPU)", () => {
  let device = null;
  before(async () => {
    device = await getWebGpuDevice();
  });

  test("skips when WebGPU unavailable", (t) => {
    if (!device) t.skip("WebGPU unavailable");
    assert.ok(device);
  });

  async function runC2cOnce({ shape, direction, normalize, inPlace, batch = 1, ioView, segmented = false }) {
    const nTotal = shape.reduce((a, b) => a * b, 1);
    const totalComplex = nTotal * batch;
    const inputData = randomComplexInterleaved(totalComplex);

    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({
      size: inputData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const input = segmented ? splitBufferView(inputBuf, inputData.byteLength, 4) : inputBuf;
    const output = segmented ? splitBufferView(outputBuf, inputData.byteLength, 4) : outputBuf;

    const plan = createPlan(device, {
      shape,
      type: "c2c",
      direction,
      batch,
      inPlace,
      normalize,
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView,
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input, output: inPlace ? undefined : output });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outBuf = inPlace ? inputBuf : outputBuf;
    const out = await downloadComplex(device, outBuf, totalComplex);
    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
    return { inputData, out };
  }

  test("c2c forward+inverse round-trip with backward normalization (N=210)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 210;
    const inputData = randomComplexInterleaved(N);
    const inputBuf = uploadComplex(device, inputData);
    const midBuf = device.createBuffer({ size: inputData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBuf = device.createBuffer({ size: inputData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const fwd = createPlan(device, {
      type: "c2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
    });
    const inv = createPlan(device, {
      type: "c2c",
      shape: [N],
      direction: "inverse",
      batch: 1,
      inPlace: false,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
    });

    const enc = device.createCommandEncoder();
    fwd.exec(enc, { input: inputBuf, output: midBuf });
    inv.exec(enc, { input: midBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await downloadComplex(device, outBuf, N);
    assertAllClose(out, inputData, { atol: 3e-3, rtol: 3e-3 });

    fwd.destroy();
    inv.destroy();
    inputBuf.destroy();
    midBuf.destroy();
    outBuf.destroy();
  });

  for (const N of [8, 16, 128, 210, 17, 29, 97, 2039]) {
    test(`c2c 1D forward matches CPU (N=${N})`, async (t) => {
      if (!device) t.skip("WebGPU unavailable");
      let res;
      try {
        res = await runC2cOnce({ shape: [N], direction: "forward", normalize: "none", inPlace: false });
      } catch (e) {
        t.skip(String(e));
        return;
      }
      const cpu = fftNdRefAnySizeInterleaved(res.inputData, [N], "forward", "none");
      assertAllClose(res.out, cpu, { atol: 3e-4, rtol: 3e-4 });
    });
  }

  test("c2c 2D mixed sizes forward matches CPU (96x105)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const shape = [96, 105];
    let res;
    try {
      res = await runC2cOnce({ shape, direction: "forward", normalize: "none", inPlace: false });
    } catch (e) {
      t.skip(String(e));
      return;
    }
    const cpu = fftNdRefAnySizeInterleaved(res.inputData, shape, "forward", "none");
    assertAllClose(res.out, cpu, { atol: 8e-4, rtol: 8e-4 });
  });

  test("c2c 3D mixed sizes forward matches CPU (24x25x27)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const shape = [24, 25, 27];
    let res;
    try {
      res = await runC2cOnce({ shape, direction: "forward", normalize: "none", inPlace: false });
    } catch (e) {
      t.skip(String(e));
      return;
    }
    const cpu = fftNdRefAnySizeInterleaved(res.inputData, shape, "forward", "none");
    assertAllClose(res.out, cpu, { atol: 1e-3, rtol: 1e-3 });
  });

  test("c2c batch>1 (N=32,batch=4) matches CPU", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 32;
    const batch = 4;
    const { inputData, out } = await runC2cOnce({ shape: [N], batch, direction: "forward", normalize: "none", inPlace: false });
    const expected = new Float32Array(out.length);
    for (let b = 0; b < batch; b++) {
      const slice = inputData.subarray(2 * b * N, 2 * (b + 1) * N);
      const cpu = fftNdRefAnySizeInterleaved(slice, [N], "forward", "none");
      expected.set(cpu, 2 * b * N);
    }
    assertAllClose(out, expected, { atol: 5e-4, rtol: 5e-4 });
  });

  test("c2c ioView pad-in-read center works (logical 16, input 8)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const logical = 16;
    const inputN = 8;
    const inputData = randomComplexInterleaved(inputN);
    const padded = new Float32Array(2 * logical);
    const off = Math.floor((logical - inputN) / 2);
    padded.set(inputData, 2 * off);

    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({
      size: padded.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      shape: [logical],
      type: "c2c",
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { input: { shape: [inputN], placement: "center" } },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const out = await downloadComplex(device, outputBuf, logical);
    const cpu = fftNdRefAnySizeInterleaved(padded, [logical], "forward", "none");
    assertAllClose(out, cpu, { atol: 5e-4, rtol: 5e-4 });
    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("c2c ioView pad-in-write (logical 16 -> output view 32 center clearOutside) matches CPU embed", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const logical = 16;
    const viewOut = 32;
    const off = Math.floor((logical - viewOut) / 2); // negative for centered embed
    const inputData = randomComplexInterleaved(logical);

    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({
      size: viewOut * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      shape: [logical],
      type: "c2c",
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { output: { shape: [viewOut], placement: "center", clearOutside: true } },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await downloadComplex(device, outputBuf, viewOut);
    const cpuLogical = fftNdRefAnySizeInterleaved(inputData, [logical], "forward", "none");
    const cpuView = new Float32Array(2 * viewOut);
    for (let i = 0; i < viewOut; i++) {
      const l = i + off;
      if (l >= 0 && l < logical) {
        cpuView[2 * i] = cpuLogical[2 * l];
        cpuView[2 * i + 1] = cpuLogical[2 * l + 1];
      }
    }
    assertAllClose(out, cpuView, { atol: 7e-4, rtol: 7e-4 });

    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("c2c ioView clearOutside=false preserves existing output outside logical region", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const logical = 16;
    const viewOut = 32;
    const off = Math.floor((logical - viewOut) / 2); // negative for centered embed
    const inputData = randomComplexInterleaved(logical);

    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({
      size: viewOut * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const sentinelRe = 123.0;
    const sentinelIm = -456.0;
    const sentinel = new Float32Array(2 * viewOut);
    for (let i = 0; i < viewOut; i++) {
      sentinel[2 * i] = sentinelRe;
      sentinel[2 * i + 1] = sentinelIm;
    }
    device.queue.writeBuffer(outputBuf, 0, sentinel);

    const plan = createPlan(device, {
      shape: [logical],
      type: "c2c",
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { output: { shape: [viewOut], placement: "center", clearOutside: false } },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await downloadComplex(device, outputBuf, viewOut);
    const cpuLogical = fftNdRefAnySizeInterleaved(inputData, [logical], "forward", "none");

    // Check inside mapped region (view i -> logical l = i + off)
    for (let i = 0; i < viewOut; i++) {
      const l = i + off;
      if (l >= 0 && l < logical) continue;
      assert.equal(out[2 * i], sentinelRe);
      assert.equal(out[2 * i + 1], sentinelIm);
    }
    const mappedStart = -off;
    const mapped = out.subarray(2 * mappedStart, 2 * (mappedStart + logical));
    assertAllClose(mapped, cpuLogical, { atol: 7e-4, rtol: 7e-4 });

    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("c2c segmented BufferView input/output works (N=128)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 128;
    let res;
    try {
      res = await runC2cOnce({ shape: [N], direction: "forward", normalize: "none", inPlace: false, segmented: true });
    } catch (e) {
      t.skip(String(e));
      return;
    }
    const cpu = fftNdRefAnySizeInterleaved(res.inputData, [N], "forward", "none");
    assertAllClose(res.out, cpu, { atol: 7e-4, rtol: 7e-4 });
  });

  test("BufferView Tier-B fallback (segmentCount > cap) works for c2c (N=32)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 32;
    const inputData = randomComplexInterleaved(N);
    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({ size: inputData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const maxStorage = device.limits?.maxStorageBuffersPerShaderStage ?? 8;
    const segCap = Math.max(0, Math.min(8, maxStorage - 1));
    const segments = Math.max(1, segCap + 1);

    const input = splitBufferView(inputBuf, inputData.byteLength, segments);
    const output = splitBufferView(outputBuf, inputData.byteLength, segments);

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
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input, output });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await downloadComplex(device, outputBuf, N);
    const cpu = fftNdRefAnySizeInterleaved(inputData, [N], "forward", "none");
    assertAllClose(out, cpu, { atol: 3e-4, rtol: 3e-4 });

    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("r2c/c2r packed conventions roundtrip (N=17)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 17;
    const inputReal = new Float32Array(N);
    for (let i = 0; i < N; i++) inputReal[i] = (Math.random() * 2 - 1) * 0.5;

    const inputBuf = device.createBuffer({
      size: inputReal.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(inputBuf, 0, inputReal);

    const outLen = Math.floor(N / 2) + 1;
    const packedBuf = device.createBuffer({
      size: outLen * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const r2c = createPlan(device, {
      type: "r2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
    });
    const enc1 = device.createCommandEncoder();
    r2c.exec(enc1, { input: inputBuf, output: packedBuf });
    device.queue.submit([enc1.finish()]);
    await device.queue.onSubmittedWorkDone();
    const packed = await downloadComplex(device, packedBuf, outLen);

    const cpuPacked = r2cRefPackedInterleaved(inputReal, N, "forward", "none");
    assertAllClose(packed, cpuPacked, { atol: 8e-4, rtol: 8e-4 });
    r2c.destroy();

    const outRealBuf = device.createBuffer({
      size: inputReal.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const c2r = createPlan(device, {
      type: "c2r",
      shape: [N],
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
    });
    const enc2 = device.createCommandEncoder();
    c2r.exec(enc2, { input: packedBuf, output: outRealBuf });
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();

    // downloadComplex reads vec2; for real buffer, map explicitly:
    const readback = device.createBuffer({ size: inputReal.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc3 = device.createCommandEncoder();
    enc3.copyBufferToBuffer(outRealBuf, 0, readback, 0, inputReal.byteLength);
    device.queue.submit([enc3.finish()]);
    await device.queue.onSubmittedWorkDone();
    const READ = globalThis.GPUMapMode?.READ ?? 1;
    await readback.mapAsync(READ);
    const outReal2 = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();

    assertAllClose(outReal2, inputReal, { atol: 2e-3, rtol: 2e-3 });
    c2r.destroy();
    inputBuf.destroy();
    packedBuf.destroy();
    outRealBuf.destroy();
  });

  test("r2c ioView pad-in-read (shape=16, input view=8 center) matches CPU", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 16;
    const V = 8;
    const off = Math.floor((N - V) / 2);
    const inputView = new Float32Array(V);
    for (let i = 0; i < V; i++) inputView[i] = (Math.random() * 2 - 1) * 0.5;
    const logical = new Float32Array(N);
    logical.set(inputView, off);

    const inBuf = device.createBuffer({ size: inputView.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, inputView);
    const outLen = Math.floor(N / 2) + 1;
    const outBuf = device.createBuffer({ size: outLen * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "r2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { input: { shape: [V], placement: "center" } },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const packed = await downloadComplex(device, outBuf, outLen);
    const cpu = r2cRefPackedInterleaved(logical, N, "forward", "none");
    assertAllClose(packed, cpu, { atol: 8e-4, rtol: 8e-4 });

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("r2c ioView pad-in-write (shape=17, packed output view=5 at offset=2) matches CPU extract", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 17;
    const outFull = Math.floor(N / 2) + 1; // 9
    const viewLen = 5;
    const off = 2;
    const input = new Float32Array(N);
    for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
    const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBuf = device.createBuffer({ size: viewLen * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "r2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { output: { shape: [viewLen], offset: [off] } },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const packedView = await downloadComplex(device, outBuf, viewLen);
    const cpuFull = r2cRefPackedInterleaved(input, N, "forward", "none");
    const cpuView = cpuFull.slice(off * 2, (off + viewLen) * 2);
    assertAllClose(packedView, cpuView, { atol: 8e-4, rtol: 8e-4 });

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
    assert.equal(outFull, 9);
  });

  test("c2r ioView pad-in-read (shape=17, packed input view=5 at offset=2) matches CPU embed+ifft", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 17;
    const outLen = Math.floor(N / 2) + 1; // 9
    const viewLen = 5;
    const off = 2;
    const packedView = randomComplexInterleaved(viewLen);
    const packedFull = new Float32Array(outLen * 2);
    packedFull.set(packedView, off * 2);

    const inBuf = uploadComplex(device, packedView);
    const outBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "c2r",
      shape: [N],
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { input: { shape: [viewLen], offset: [off] } },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const readback = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(outBuf, 0, readback, 0, N * 4);
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();
    const READ = globalThis.GPUMapMode?.READ ?? 1;
    await readback.mapAsync(READ);
    const gpu = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();

    const cpu = c2rRefFromPackedInterleaved(packedFull, N, "backward");
    assertAllClose(gpu, cpu, { atol: 2e-3, rtol: 2e-3 });

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  for (const type of ["dct1", "dct2", "dct3", "dct4"]) {
    test(`${type} forward matches CPU (N=16)`, async (t) => {
      if (!device) t.skip("WebGPU unavailable");
      const N = 16;
      const input = new Float32Array(N);
      for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
      const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      device.queue.writeBuffer(inBuf, 0, input);
      const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

      const plan = createPlan(device, {
        type,
        shape: [N],
        direction: "forward",
        batch: 1,
        inPlace: false,
        normalize: "none",
        layout: { interleavedComplex: false },
        precision: "f32",
      });
      const enc = device.createCommandEncoder();
      plan.exec(enc, { input: inBuf, output: outBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const read = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc2 = device.createCommandEncoder();
      enc2.copyBufferToBuffer(outBuf, 0, read, 0, input.byteLength);
      device.queue.submit([enc2.finish()]);
      await device.queue.onSubmittedWorkDone();
      const READ = globalThis.GPUMapMode?.READ ?? 1;
      await read.mapAsync(READ);
      const gpu = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      read.destroy();

      let cpu;
      if (type === "dct1") cpu = dct1Ref(input, N);
      else if (type === "dct2") cpu = dct2Ref(input, N, "forward");
      else if (type === "dct3") cpu = dct3Ref(input, N, "forward");
      else cpu = dct4Ref(input, N);
      assertAllClose(gpu, cpu, { atol: 2e-3, rtol: 2e-3 });
      plan.destroy();
      inBuf.destroy();
      outBuf.destroy();
    });
  }

  test("dct2 inverse matches CPU (N=16)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const N = 16;
    const input = new Float32Array(N);
    for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
    const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "dct2",
      shape: [N],
      direction: "inverse",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: false },
      precision: "f32",
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const read = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(outBuf, 0, read, 0, input.byteLength);
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();
    const READ = globalThis.GPUMapMode?.READ ?? 1;
    await read.mapAsync(READ);
    const gpu = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();

    const cpu = dct2Ref(input, N, "inverse");
    assertAllClose(gpu, cpu, { atol: 2e-3, rtol: 2e-3 });
    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c f16-storage forward matches CPU (N=64) when shader-f16 available", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    if (!device.features?.has?.("shader-f16")) t.skip("shader-f16 unavailable");
    const N = 64;
    const inputF32 = randomComplexInterleaved(N);
    const inputF16 = new Uint16Array(2 * N);
    for (let i = 0; i < 2 * N; i++) inputF16[i] = f32ToF16Bits(inputF32[i]);
    const inBuf = device.createBuffer({ size: inputF16.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, inputF16);
    const outBuf = device.createBuffer({ size: inputF16.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "c2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f16-storage",
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const read = device.createBuffer({ size: inputF16.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(outBuf, 0, read, 0, inputF16.byteLength);
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();
    const READ = globalThis.GPUMapMode?.READ ?? 1;
    await read.mapAsync(READ);
    const outU16 = new Uint16Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();

    const outF32 = new Float32Array(2 * N);
    for (let i = 0; i < 2 * N; i++) outF32[i] = f16BitsToF32(outU16[i]);

    const cpu = fftNdRefAnySizeInterleaved(inputF32, [N], "forward", "none");
    assertAllClose(outF32, cpu, { atol: 3e-2, rtol: 3e-2 });
    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c f16-storage with ioView input+output works (clearOutside=false preserves output)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    if (!device.features?.has?.("shader-f16")) t.skip("shader-f16 unavailable");

    const logical = 16;
    const inputN = 8;
    const viewOut = 32;
    const inOff = Math.floor((logical - inputN) / 2);
    const outOff = Math.floor((logical - viewOut) / 2); // negative for centered embed

    const inputViewF32 = randomComplexInterleaved(inputN);
    const inputViewF16 = new Uint16Array(2 * inputN);
    for (let i = 0; i < 2 * inputN; i++) inputViewF16[i] = f32ToF16Bits(inputViewF32[i]);
    const inBuf = device.createBuffer({ size: inputViewF16.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, inputViewF16);

    const outBytes = viewOut * 4; // vec2<f16>
    const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const sentinelRe = 1.5;
    const sentinelIm = -2.0;
    const sRe = f32ToF16Bits(sentinelRe);
    const sIm = f32ToF16Bits(sentinelIm);
    const sentinelU16 = new Uint16Array(2 * viewOut);
    for (let i = 0; i < viewOut; i++) {
      sentinelU16[2 * i] = sRe;
      sentinelU16[2 * i + 1] = sIm;
    }
    device.queue.writeBuffer(outBuf, 0, sentinelU16);

    const plan = createPlan(device, {
      type: "c2c",
      shape: [logical],
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f16-storage",
      ioView: {
        input: { shape: [inputN], placement: "center" },
        output: { shape: [viewOut], placement: "center", clearOutside: false },
      },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const read = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(outBuf, 0, read, 0, outBytes);
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();
    const READ = globalThis.GPUMapMode?.READ ?? 1;
    await read.mapAsync(READ);
    const outU16 = new Uint16Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();

    const outF32 = new Float32Array(2 * viewOut);
    for (let i = 0; i < 2 * viewOut; i++) outF32[i] = f16BitsToF32(outU16[i]);

    const logicalIn = new Float32Array(2 * logical);
    logicalIn.set(inputViewF32, 2 * inOff);
    const cpuLogical = fftNdRefAnySizeInterleaved(logicalIn, [logical], "forward", "none");

    for (let i = 0; i < viewOut; i++) {
      const l = i + outOff;
      if (l >= 0 && l < logical) continue;
      assert.equal(outF32[2 * i], f16BitsToF32(sRe));
      assert.equal(outF32[2 * i + 1], f16BitsToF32(sIm));
    }
    const mappedStart = -outOff;
    const mapped = outF32.subarray(2 * mappedStart, 2 * (mappedStart + logical));
    assertAllClose(mapped, cpuLogical, { atol: 3e-2, rtol: 3e-2 });

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("conv2d 3x3 same padding matches CPU (complex)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const H = 16, W = 17, k = 3;
    const input = randomComplexInterleaved(H * W);
    const kernel = new Float32Array(k * k);
    for (let i = 0; i < kernel.length; i++) kernel[i] = (Math.random() * 2 - 1) * 0.5;

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const plan = createPlan(device, {
      type: "conv2d",
      shape: [H, W],
      batch: 1,
      layout: { interleavedComplex: true },
      precision: "f32",
      conv: { kernelSize: 3, kernelType: "real", padding: "same", boundary: "zero" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const out = await downloadComplex(device, outBuf, H * W);

    const cpu = conv2dRef({
      input,
      kernel,
      Hout: H,
      Wout: W,
      Hin: H,
      Win: W,
      k,
      pad: [1, 1, 1, 1],
      complex: true,
      complexKernel: false,
    });
    assertAllClose(out, cpu, { atol: 3e-3, rtol: 3e-3 });
    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("conv2d 2x2 valid padding matches CPU (real)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const Hin = 11, Win = 9, k = 2;
    const Hout = Hin - k + 1;
    const Wout = Win - k + 1;
    const input = new Float32Array(Hin * Win);
    for (let i = 0; i < input.length; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
    const kernel = new Float32Array(k * k);
    for (let i = 0; i < kernel.length; i++) kernel[i] = (Math.random() * 2 - 1) * 0.5;

    const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBuf = device.createBuffer({ size: Hout * Wout * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "conv2d",
      shape: [Hout, Wout],
      batch: 1,
      layout: { interleavedComplex: false },
      precision: "f32",
      conv: { kernelSize: 2, kernelType: "real", padding: "valid", boundary: "zero" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const read = device.createBuffer({ size: Hout * Wout * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(outBuf, 0, read, 0, Hout * Wout * 4);
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();
    const READ = globalThis.GPUMapMode?.READ ?? 1;
    await read.mapAsync(READ);
    const gpu = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();

    const cpu = conv2dRef({ input, kernel, Hout, Wout, Hin, Win, k, pad: [0, 0, 0, 0], complex: false, complexKernel: false });
    assertAllClose(gpu, cpu, { atol: 3e-3, rtol: 3e-3 });
    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("conv2d 1x1 same padding matches CPU (complex kernel)", async (t) => {
    if (!device) t.skip("WebGPU unavailable");
    const H = 9, W = 7, k = 1;
    const input = randomComplexInterleaved(H * W);
    const kernel = new Float32Array(2 * k * k);
    kernel[0] = 0.5;
    kernel[1] = -0.25;

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const plan = createPlan(device, {
      type: "conv2d",
      shape: [H, W],
      batch: 1,
      layout: { interleavedComplex: true },
      precision: "f32",
      conv: { kernelSize: 1, kernelType: "complex", padding: "same", boundary: "zero" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const out = await downloadComplex(device, outBuf, H * W);

    const cpu = conv2dRef({ input, kernel, Hout: H, Wout: W, Hin: H, Win: W, k, pad: [0, 0, 0, 0], complex: true, complexKernel: true });
    assertAllClose(out, cpu, { atol: 3e-3, rtol: 3e-3 });
    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });
});
*/

