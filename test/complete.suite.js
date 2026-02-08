import { createPlan, BufferView, uploadComplex, downloadComplex, exportPipelineCacheSnapshot, importPipelineCacheSnapshot } from "../src/index.js";
import { createFftPlan } from "../src/plan.js";
import {
  randomComplexInterleaved,
  fftNdRefAnySizeInterleaved,
  dct1Ref,
  dct2Ref,
  dct3Ref,
  dct4Ref,
  dst1Ref,
  dst2Ref,
  dst3Ref,
  dst4Ref,
  r2cRefPackedInterleaved,
  c2rRefFromPackedInterleaved,
  conv2dRef,
  fftConvRef,
} from "../src/utils/math.js";

function defaultLog() {}

function f32ToF16Bits(x) {
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

function prod(arr) {
  return arr.reduce((a, b) => a * b, 1);
}

function strideSpanElements(shape, strides) {
  let span = 1;
  for (let d = 0; d < shape.length; d++) span += (shape[d] - 1) * strides[d];
  return span;
}

function requiredStridedElements(shape, strides, offsetElements, batch, batchStrideElements) {
  return offsetElements + (batch - 1) * batchStrideElements + strideSpanElements(shape, strides);
}

function linearToCoords(i, shape) {
  const c = new Array(shape.length);
  let rem = i;
  for (let d = 0; d < shape.length; d++) {
    c[d] = rem % shape[d];
    rem = Math.floor(rem / shape[d]);
  }
  return c;
}

function coordsToStridedIndex(coords, strides, offsetElements, b, batchStrideElements) {
  let idx = offsetElements + b * batchStrideElements;
  for (let d = 0; d < coords.length; d++) idx += coords[d] * strides[d];
  return idx;
}

function makeStridedPhysicalComplex({
  shape,
  batch,
  strides,
  offsetElements,
  batchStrideElements,
  logicalInterleaved,
  sentinelRe = 1234.5,
  sentinelIm = -987.25,
}) {
  const logicalTotal = prod(shape);
  const neededElements = requiredStridedElements(shape, strides, offsetElements, batch, batchStrideElements);
  const out = new Float32Array(2 * neededElements);
  for (let i = 0; i < neededElements; i++) {
    out[2 * i] = sentinelRe;
    out[2 * i + 1] = sentinelIm;
  }
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < logicalTotal; i++) {
      const coords = linearToCoords(i, shape);
      const pi = coordsToStridedIndex(coords, strides, offsetElements, b, batchStrideElements);
      const si = 2 * (b * logicalTotal + i);
      out[2 * pi] = logicalInterleaved[si];
      out[2 * pi + 1] = logicalInterleaved[si + 1];
    }
  }
  return out;
}

function extractStridedLogicalComplex({
  physicalInterleaved,
  shape,
  batch,
  strides,
  offsetElements,
  batchStrideElements,
}) {
  const logicalTotal = prod(shape);
  const out = new Float32Array(2 * logicalTotal * batch);
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < logicalTotal; i++) {
      const coords = linearToCoords(i, shape);
      const pi = coordsToStridedIndex(coords, strides, offsetElements, b, batchStrideElements);
      const di = 2 * (b * logicalTotal + i);
      out[di] = physicalInterleaved[2 * pi];
      out[di + 1] = physicalInterleaved[2 * pi + 1];
    }
  }
  return out;
}

function makeStridedPhysicalReal({
  shape,
  batch,
  strides,
  offsetElements,
  batchStrideElements,
  logical,
  sentinel = 1234.5,
}) {
  const logicalTotal = prod(shape);
  const neededElements = requiredStridedElements(shape, strides, offsetElements, batch, batchStrideElements);
  const out = new Float32Array(neededElements);
  out.fill(sentinel);
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < logicalTotal; i++) {
      const coords = linearToCoords(i, shape);
      const pi = coordsToStridedIndex(coords, strides, offsetElements, b, batchStrideElements);
      out[pi] = logical[b * logicalTotal + i];
    }
  }
  return out;
}

function extractStridedLogicalReal({
  physical,
  shape,
  batch,
  strides,
  offsetElements,
  batchStrideElements,
}) {
  const logicalTotal = prod(shape);
  const out = new Float32Array(logicalTotal * batch);
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < logicalTotal; i++) {
      const coords = linearToCoords(i, shape);
      const pi = coordsToStridedIndex(coords, strides, offsetElements, b, batchStrideElements);
      out[b * logicalTotal + i] = physical[pi];
    }
  }
  return out;
}

function coordsToLinear(coords, shape) {
  let idx = 0;
  let stride = 1;
  for (let d = 0; d < shape.length; d++) {
    idx += coords[d] * stride;
    stride *= shape[d];
  }
  return idx;
}

function embedRealViewIntoLogical({ logicalShape, viewShape, offset, view }) {
  const rank = logicalShape.length;
  const out = new Float32Array(prod(logicalShape));
  const viewTotal = prod(viewShape);
  for (let i = 0; i < viewTotal; i++) {
    const vc = linearToCoords(i, viewShape);
    const lc = new Array(rank);
    let inside = true;
    for (let d = 0; d < rank; d++) {
      lc[d] = vc[d] + offset[d];
      if (lc[d] < 0 || lc[d] >= logicalShape[d]) {
        inside = false;
        break;
      }
    }
    if (!inside) continue;
    out[coordsToLinear(lc, logicalShape)] = view[i];
  }
  return out;
}

function extractRealViewFromLogical({ logicalShape, viewShape, offset, logical }) {
  const rank = logicalShape.length;
  const out = new Float32Array(prod(viewShape));
  const viewTotal = prod(viewShape);
  for (let i = 0; i < viewTotal; i++) {
    const vc = linearToCoords(i, viewShape);
    const lc = new Array(rank);
    let inside = true;
    for (let d = 0; d < rank; d++) {
      lc[d] = vc[d] + offset[d];
      if (lc[d] < 0 || lc[d] >= logicalShape[d]) {
        inside = false;
        break;
      }
    }
    if (!inside) continue;
    out[i] = logical[coordsToLinear(lc, logicalShape)];
  }
  return out;
}

function embedComplexViewIntoLogical({ logicalShape, viewShape, offset, viewInterleaved }) {
  const rank = logicalShape.length;
  const out = new Float32Array(2 * prod(logicalShape));
  const viewTotal = prod(viewShape);
  for (let i = 0; i < viewTotal; i++) {
    const vc = linearToCoords(i, viewShape);
    const lc = new Array(rank);
    let inside = true;
    for (let d = 0; d < rank; d++) {
      lc[d] = vc[d] + offset[d];
      if (lc[d] < 0 || lc[d] >= logicalShape[d]) {
        inside = false;
        break;
      }
    }
    if (!inside) continue;
    const li = coordsToLinear(lc, logicalShape);
    out[2 * li] = viewInterleaved[2 * i];
    out[2 * li + 1] = viewInterleaved[2 * i + 1];
  }
  return out;
}

function extractComplexViewFromLogical({ logicalShape, viewShape, offset, logicalInterleaved }) {
  const rank = logicalShape.length;
  const out = new Float32Array(2 * prod(viewShape));
  const viewTotal = prod(viewShape);
  for (let i = 0; i < viewTotal; i++) {
    const vc = linearToCoords(i, viewShape);
    const lc = new Array(rank);
    let inside = true;
    for (let d = 0; d < rank; d++) {
      lc[d] = vc[d] + offset[d];
      if (lc[d] < 0 || lc[d] >= logicalShape[d]) {
        inside = false;
        break;
      }
    }
    if (!inside) continue;
    const li = coordsToLinear(lc, logicalShape);
    out[2 * i] = logicalInterleaved[2 * li];
    out[2 * i + 1] = logicalInterleaved[2 * li + 1];
  }
  return out;
}

function coordsInsideRange(coords, start, end) {
  for (let d = 0; d < coords.length; d++) {
    if (coords[d] < start[d] || coords[d] >= end[d]) return false;
  }
  return true;
}

function zeroOutsideRangeRealRef(data, shape, start, end) {
  const out = new Float32Array(data);
  const n = prod(shape);
  for (let i = 0; i < n; i++) {
    const c = linearToCoords(i, shape);
    if (!coordsInsideRange(c, start, end)) out[i] = 0;
  }
  return out;
}

function zeroOutsideRangeComplexRef(dataInterleaved, shape, start, end) {
  const out = new Float32Array(dataInterleaved);
  const n = prod(shape);
  for (let i = 0; i < n; i++) {
    const c = linearToCoords(i, shape);
    if (!coordsInsideRange(c, start, end)) {
      out[2 * i] = 0;
      out[2 * i + 1] = 0;
    }
  }
  return out;
}

function fftConvOutputSpec(shape, kernelShape, boundary) {
  const rank = shape.length;
  const fftShape = boundary === "circular" ? shape.slice() : shape.map((n, d) => n + kernelShape[d] - 1);
  if (boundary === "circular") {
    return {
      fftShape,
      outputShape: shape.slice(),
      outputOffset: new Array(rank).fill(0),
    };
  }
  if (boundary === "linear-full") {
    return {
      fftShape,
      outputShape: fftShape.slice(),
      outputOffset: new Array(rank).fill(0),
    };
  }
  if (boundary === "linear-same") {
    return {
      fftShape,
      outputShape: shape.slice(),
      outputOffset: kernelShape.map((n) => Math.floor((n - 1) / 2)),
    };
  }
  return {
    fftShape,
    outputShape: shape.map((n, d) => n - kernelShape[d] + 1),
    outputOffset: kernelShape.map((n) => n - 1),
  };
}

function applyZeroPadPerBatchComplexRef(dataInterleaved, logicalShape, batch, stage) {
  if (!stage) return new Float32Array(dataInterleaved);
  const logicalTotal = prod(logicalShape);
  const out = new Float32Array(dataInterleaved);
  for (let b = 0; b < batch; b++) {
    const start = 2 * b * logicalTotal;
    const end = start + 2 * logicalTotal;
    const masked = zeroOutsideRangeComplexRef(
      out.subarray(start, end),
      logicalShape,
      stage.start,
      stage.end
    );
    out.set(masked, start);
  }
  return out;
}

function fftConvRefWithZeroPad({
  input,
  kernel,
  shape,
  kernelShape,
  batch,
  mode,
  boundary,
  zeroPad,
}) {
  const rank = shape.length;
  const kShape = kernelShape ?? shape;
  const { fftShape, outputShape, outputOffset } = fftConvOutputSpec(shape, kShape, boundary);
  const fftTotal = prod(fftShape);
  const inputTotal = prod(shape);
  const outputTotal = prod(outputShape);

  const fftInput = new Float32Array(2 * fftTotal * batch);
  for (let b = 0; b < batch; b++) {
    const src = input.subarray(2 * b * inputTotal, 2 * (b + 1) * inputTotal);
    const embedded = embedComplexViewIntoLogical({
      logicalShape: fftShape,
      viewShape: shape,
      offset: new Array(rank).fill(0),
      viewInterleaved: src,
    });
    fftInput.set(embedded, 2 * b * fftTotal);
  }
  const maskedInput = applyZeroPadPerBatchComplexRef(fftInput, fftShape, batch, zeroPad?.read ?? null);

  const fftKernel = embedComplexViewIntoLogical({
    logicalShape: fftShape,
    viewShape: kShape,
    offset: new Array(rank).fill(0),
    viewInterleaved: kernel,
  });

  const full = fftConvRef({
    input: maskedInput,
    kernel: fftKernel,
    shape: fftShape,
    batch,
    mode,
    boundary: "circular",
  });
  const maskedFull = applyZeroPadPerBatchComplexRef(full, fftShape, batch, zeroPad?.write ?? null);

  const output = new Float32Array(2 * outputTotal * batch);
  for (let b = 0; b < batch; b++) {
    const batchLogical = maskedFull.subarray(2 * b * fftTotal, 2 * (b + 1) * fftTotal);
    const extracted = extractComplexViewFromLogical({
      logicalShape: fftShape,
      viewShape: outputShape,
      offset: outputOffset,
      logicalInterleaved: batchLogical,
    });
    output.set(extracted, 2 * b * outputTotal);
  }
  return output;
}

async function downloadF32(device, buffer, byteLength, offsetBytes = 0) {
  const readback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, offsetBytes, readback, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return out;
}

async function downloadU16(device, buffer, byteLength, offsetBytes = 0) {
  const readback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, offsetBytes, readback, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const out = new Uint16Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return out;
}

export function registerCompleteTests(ctx) {
  const {
    test,
    getDevice,
    assert,
    assertCloseArray,
    SkipError = class SkipError extends Error {
      constructor(msg) {
        super(msg);
        this.name = "SkipError";
      }
    },
    log = defaultLog,
    exportArtifact = null,
  } = ctx;

  const ensureDevice = async () => {
    const device = await getDevice?.();
    if (!device) throw new SkipError("WebGPU unavailable");
    return device;
  };

  const assertEq = (a, b, message) => assert(a === b, message ?? `Expected ${a} === ${b}`);

  async function runC2cOnce(device, { shape, direction, normalize, inPlace, batch = 1, ioView, segmented = false }) {
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

  test("WebGPU present", async () => {
    const device = await ensureDevice();
    assert(!!device, "device missing");
  });

  test("pipeline cache snapshot export/import roundtrip works", async () => {
    const device = await ensureDevice();
    const N = 17;
    const input = randomComplexInterleaved(N);
    const cpu = fftNdRefAnySizeInterleaved(input, [N], "forward", "none");

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

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
    {
      const enc = device.createCommandEncoder();
      plan.exec(enc, { input: inBuf, output: outBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    const out = await downloadComplex(device, outBuf, N);
    assertCloseArray(out, cpu, 3e-4, 3e-4, "cache snapshot warmup correctness");

    const snapshot = exportPipelineCacheSnapshot(device);
    assert(snapshot && typeof snapshot === "object", "snapshot object missing");
    assertEq(snapshot.schema, "webgpufft.pipeline-cache", "snapshot schema mismatch");
    assertEq(snapshot.version, 2, "snapshot version mismatch");
    assert(Number.isSafeInteger(snapshot.createdAtMs), "snapshot.createdAtMs missing");
    assert(Array.isArray(snapshot.shaderCodes), "snapshot.shaderCodes missing");
    assert(snapshot.shaderCodes.length > 0, "snapshot should contain shader sources after warmup");

    const imported = importPipelineCacheSnapshot(device, snapshot);
    assertEq(imported.schema, "webgpufft.pipeline-cache", "imported snapshot schema mismatch");
    assertEq(imported.version, 2, "imported snapshot version mismatch");
    assert(Array.isArray(imported.shaderCodes), "imported snapshot.shaderCodes missing");
    assert(imported.shaderCodes.length >= snapshot.shaderCodes.length, "import should preserve shader code list");

    const legacyImported = importPipelineCacheSnapshot(device, {
      version: 1,
      shaderCodes: snapshot.shaderCodes,
      pipelineKeys: snapshot.pipelineKeys,
    });
    assertEq(legacyImported.version, 2, "legacy import should upgrade to v2");
    assertEq(legacyImported.metadata?.fromVersion, 1, "legacy import metadata mismatch");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c forward+inverse round-trip with backward normalization (N=210)", async () => {
    const device = await ensureDevice();
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
    assertCloseArray(out, inputData, 3e-3, 3e-3, "c2c roundtrip");

    fwd.destroy();
    inv.destroy();
    inputBuf.destroy();
    midBuf.destroy();
    outBuf.destroy();
  });

  for (const N of [8, 16, 34, 128, 210, 17, 29, 97, 2039]) {
    test(`c2c 1D forward matches CPU (N=${N})`, async () => {
      const device = await ensureDevice();
      let res;
      try {
        res = await runC2cOnce(device, { shape: [N], direction: "forward", normalize: "none", inPlace: false });
      } catch (e) {
        throw new SkipError(String(e));
      }
      const cpu = fftNdRefAnySizeInterleaved(res.inputData, [N], "forward", "none");
      assertCloseArray(res.out, cpu, 3e-4, 3e-4, `c2c N=${N}`);
    });
  }

  test("c2c tuning knobs preserve correctness (force Bluestein + explicit workgroup)", async () => {
    const device = await ensureDevice();
    const N = 29;
    const input = randomComplexInterleaved(N);
    const cpu = fftNdRefAnySizeInterleaved(input, [N], "forward", "none");

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const wg = Math.max(1, Math.min(64, device.limits?.maxComputeWorkgroupSizeX ?? 64, device.limits?.maxComputeInvocationsPerWorkgroup ?? 64));
    const plan = createPlan(device, {
      type: "c2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: {
        workgroupSizeX: wg,
        raderMaxPrime: 17,
        forceBluesteinAxes: [0],
        disableTranspose: true,
      },
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, N);
    assertCloseArray(gpu, cpu, 1.2e-3, 1.2e-3, "c2c tuning correctness");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c tuning forceRaderAxes validates prime-only axes", async () => {
    const device = await ensureDevice();
    let threw = false;
    try {
      createPlan(device, {
        type: "c2c",
        shape: [16],
        direction: "forward",
        batch: 1,
        inPlace: false,
        normalize: "none",
        layout: { interleavedComplex: true },
        precision: "f32",
        tuning: { forceRaderAxes: [0] },
      });
    } catch (e) {
      threw = true;
      assert(String(e).includes("not prime"), `unexpected error for forceRaderAxes validation: ${String(e)}`);
    }
    assert(threw, "expected forceRaderAxes validation to throw for non-prime axis length");
  });

  test("c2c inPlace with non-zero inputOffset and BufferView temp works (N=34)", async () => {
    const device = await ensureDevice();
    const N = 34;
    const inputData = randomComplexInterleaved(N);
    const cpu = fftNdRefAnySizeInterleaved(inputData, [N], "forward", "none");

    const plan = createPlan(device, {
      type: "c2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      inPlace: true,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
    });

    const ws = plan.getWorkspaceSizeBytes();
    const inputOffsetBytes = 512;
    const tempOffsetBytes = 1024;
    const totalBytes = Math.max(inputOffsetBytes + inputData.byteLength, tempOffsetBytes + ws) + 256;
    const arena = device.createBuffer({
      size: totalBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(arena, inputOffsetBytes, inputData);

    const temp = new BufferView({
      segments: [{ buffer: arena, offsetBytes: tempOffsetBytes, sizeBytes: ws }],
      logicalByteOffset: 0,
      lengthBytes: ws,
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: arena, inputOffsetBytes, temp });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await downloadComplex(device, arena, N, inputOffsetBytes);
    assertCloseArray(out, cpu, 5e-4, 5e-4, "c2c nested temp+offset");

    plan.destroy();
    arena.destroy();
  });

  test("c2c inPlace mixed-radix with non-zero inputOffset and BufferView temp works (N=70)", async () => {
    const device = await ensureDevice();
    const N = 70;
    const inputData = randomComplexInterleaved(N);
    const cpu = fftNdRefAnySizeInterleaved(inputData, [N], "forward", "none");

    const plan = createPlan(device, {
      type: "c2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      inPlace: true,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
    });

    const ws = plan.getWorkspaceSizeBytes();
    const inputOffsetBytes = 512;
    const tempOffsetBytes = 2048;
    const totalBytes = Math.max(inputOffsetBytes + inputData.byteLength, tempOffsetBytes + ws) + 256;
    const arena = device.createBuffer({
      size: totalBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(arena, inputOffsetBytes, inputData);

    const temp = new BufferView({
      segments: [{ buffer: arena, offsetBytes: tempOffsetBytes, sizeBytes: ws }],
      logicalByteOffset: 0,
      lengthBytes: ws,
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: arena, inputOffsetBytes, temp });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await downloadComplex(device, arena, N, inputOffsetBytes);
    assertCloseArray(out, cpu, 5e-4, 5e-4, "c2c mixed nested temp+offset");

    plan.destroy();
    arena.destroy();
  });

  test("low-level fft inPlace with non-zero inputOffset and BufferView temp works (N=70)", async () => {
    const device = await ensureDevice();
    const N = 70;
    const inputData = randomComplexInterleaved(N);
    const cpu = fftNdRefAnySizeInterleaved(inputData, [N], "forward", "none");

    const plan = createFftPlan(device, {
      shape: [N],
      direction: "forward",
      normalize: "none",
      inPlace: true,
      layout: "interleaved",
      precision: "f32",
    });

    const bytes = N * 8;
    const inputOffsetBytes = 512;
    const tempOffsetBytes = 2048;
    const totalBytes = tempOffsetBytes + bytes + 256;
    const arena = device.createBuffer({
      size: totalBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(arena, inputOffsetBytes, inputData);

    const temp = new BufferView({
      segments: [{ buffer: arena, offsetBytes: tempOffsetBytes, sizeBytes: bytes }],
      logicalByteOffset: 0,
      lengthBytes: bytes,
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: arena, inputOffsetBytes, temp, batch: 1 });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await downloadComplex(device, arena, N, inputOffsetBytes);
    assertCloseArray(out, cpu, 5e-4, 5e-4, "low-level fft nested temp+offset");

    plan.destroy();
    arena.destroy();
  });

  test("c2c 2D mixed sizes forward matches CPU (96x105)", async () => {
    const device = await ensureDevice();
    const shape = [96, 105];
    let res;
    try {
      res = await runC2cOnce(device, { shape, direction: "forward", normalize: "none", inPlace: false });
    } catch (e) {
      throw new SkipError(String(e));
    }
    const cpu = fftNdRefAnySizeInterleaved(res.inputData, shape, "forward", "none");
    assertCloseArray(res.out, cpu, 8e-4, 8e-4, "c2c 2D");
  });

  test("c2c 3D mixed sizes forward matches CPU (24x25x27)", async () => {
    const device = await ensureDevice();
    const shape = [24, 25, 27];
    let res;
    try {
      res = await runC2cOnce(device, { shape, direction: "forward", normalize: "none", inPlace: false });
    } catch (e) {
      throw new SkipError(String(e));
    }
    const cpu = fftNdRefAnySizeInterleaved(res.inputData, shape, "forward", "none");
    assertCloseArray(res.out, cpu, 1e-3, 1e-3, "c2c 3D");
  });

  test("c2c 4D mixed sizes forward matches CPU (4x3x5x2)", async () => {
    const device = await ensureDevice();
    const shape = [4, 3, 5, 2];
    let res;
    try {
      res = await runC2cOnce(device, { shape, direction: "forward", normalize: "none", inPlace: false });
    } catch (e) {
      throw new SkipError(String(e));
    }
    const cpu = fftNdRefAnySizeInterleaved(res.inputData, shape, "forward", "none");
    assertCloseArray(res.out, cpu, 1.5e-3, 1.5e-3, "c2c 4D");
  });

  test("c2c out-of-core rank-3 forced via tuning maxStorageBufferBindingSize matches CPU", async () => {
    const device = await ensureDevice();
    const shape = [4, 3, 2];
    const total = prod(shape);
    const input = randomComplexInterleaved(total);
    const cpu = fftNdRefAnySizeInterleaved(input, shape, "forward", "none");

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    assert(plan?._outOfCoreFourStepMode === true, "expected forced out-of-core mode");

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, total);
    assertCloseArray(gpu, cpu, 2e-3, 2e-3, "c2c forced out-of-core rank3");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c out-of-core rank-3 forced path with ioView + zeroPad matches CPU", async () => {
    const device = await ensureDevice();
    const logicalShape = [2, 3, 4];
    const inputViewShape = [2, 1, 4];
    const readStart = [0, 1, 0];
    const readEnd = [2, 3, 3];
    const writeStart = [0, 0, 1];
    const writeEnd = [2, 3, 4];

    const inputView = randomComplexInterleaved(prod(inputViewShape));
    const inputBuf = uploadComplex(device, inputView);
    const outComplex = prod(logicalShape);
    const outputBuf = device.createBuffer({
      size: outComplex * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "c2c",
      shape: logicalShape,
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inputViewShape, placement: "center" },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    assert(plan?._outOfCoreFourStepMode === true, "expected forced out-of-core mode");

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inputBuf, output: outputBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outputBuf, outComplex);

    const logicalInput = embedComplexViewIntoLogical({
      logicalShape,
      viewShape: inputViewShape,
      offset: [0, 1, 0],
      viewInterleaved: inputView,
    });
    const afterReadPad = zeroOutsideRangeComplexRef(logicalInput, logicalShape, readStart, readEnd);
    const ffted = fftNdRefAnySizeInterleaved(afterReadPad, logicalShape, "forward", "none");
    const afterWritePad = zeroOutsideRangeComplexRef(ffted, logicalShape, writeStart, writeEnd);
    const expected = afterWritePad;

    assertCloseArray(gpu, expected, 2e-3, 2e-3, "c2c forced out-of-core ioView+zeroPad");

    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("c2c out-of-core rank-3 forced ioView input+output + zeroPad preserves clearOutside=false", async () => {
    const device = await ensureDevice();
    const logicalShape = [2, 3, 4];
    const inputViewShape = [2, 1, 4];
    const inputOffset = [0, 1, 0];
    const outputViewShape = [2, 5, 4];
    const outputOffset = [0, -1, 0];
    const readStart = [0, 1, 0];
    const readEnd = [2, 3, 3];
    const writeStart = [0, 0, 1];
    const writeEnd = [2, 3, 4];

    const inputView = randomComplexInterleaved(prod(inputViewShape));
    const inputBufRaw = uploadComplex(device, inputView);
    const input = splitBufferView(inputBufRaw, inputView.byteLength, 2);

    const outComplex = prod(outputViewShape);
    const outputBufRaw = device.createBuffer({
      size: outComplex * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const output = splitBufferView(outputBufRaw, outComplex * 8, 3);

    const sentinelRe = 321.5;
    const sentinelIm = -654.25;
    const sentinel = new Float32Array(2 * outComplex);
    for (let i = 0; i < outComplex; i++) {
      sentinel[2 * i] = sentinelRe;
      sentinel[2 * i + 1] = sentinelIm;
    }
    device.queue.writeBuffer(outputBufRaw, 0, sentinel);

    const plan = createPlan(device, {
      type: "c2c",
      shape: logicalShape,
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inputViewShape, placement: "center" },
        output: { shape: outputViewShape, placement: "center", clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    assert(plan?._outOfCoreFourStepMode === true, "expected forced out-of-core mode");

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input, output });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outputBufRaw, outComplex);

    const logicalInput = embedComplexViewIntoLogical({
      logicalShape,
      viewShape: inputViewShape,
      offset: inputOffset,
      viewInterleaved: inputView,
    });
    const afterReadPad = zeroOutsideRangeComplexRef(logicalInput, logicalShape, readStart, readEnd);
    const ffted = fftNdRefAnySizeInterleaved(afterReadPad, logicalShape, "forward", "none");
    const afterWritePad = zeroOutsideRangeComplexRef(ffted, logicalShape, writeStart, writeEnd);

    const expected = new Float32Array(sentinel);
    for (let i = 0; i < outComplex; i++) {
      const vc = linearToCoords(i, outputViewShape);
      const lc = new Array(outputViewShape.length);
      let inside = true;
      for (let d = 0; d < outputViewShape.length; d++) {
        lc[d] = vc[d] + outputOffset[d];
        if (lc[d] < 0 || lc[d] >= logicalShape[d]) {
          inside = false;
          break;
        }
      }
      if (!inside) continue;
      const li = coordsToLinear(lc, logicalShape);
      expected[2 * i] = afterWritePad[2 * li];
      expected[2 * i + 1] = afterWritePad[2 * li + 1];
    }

    assertCloseArray(gpu, expected, 2e-3, 2e-3, "c2c forced out-of-core ioView in+out + zeroPad");

    plan.destroy();
    inputBufRaw.destroy();
    outputBufRaw.destroy();
  });

  test("c2c out-of-core rank-4 forced ioView input+output + zeroPad preserves clearOutside=false", async () => {
    const device = await ensureDevice();
    const logicalShape = [2, 2, 2, 2];
    const inputViewShape = [1, 2, 2, 2];
    const inputOffset = [0, 0, 0, 0];
    const outputViewShape = [2, 3, 2, 2];
    const outputOffset = [0, -1, 0, 0];
    const readStart = [0, 0, 0, 1];
    const readEnd = [2, 2, 2, 2];
    const writeStart = [0, 0, 1, 0];
    const writeEnd = [2, 2, 2, 2];

    const inputView = randomComplexInterleaved(prod(inputViewShape));
    const inputBufRaw = uploadComplex(device, inputView);
    const input = splitBufferView(inputBufRaw, inputView.byteLength, 2);

    const outComplex = prod(outputViewShape);
    const outputBufRaw = device.createBuffer({
      size: outComplex * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const output = splitBufferView(outputBufRaw, outComplex * 8, 4);

    const sentinelRe = 77.125;
    const sentinelIm = -88.5;
    const sentinel = new Float32Array(2 * outComplex);
    for (let i = 0; i < outComplex; i++) {
      sentinel[2 * i] = sentinelRe;
      sentinel[2 * i + 1] = sentinelIm;
    }
    device.queue.writeBuffer(outputBufRaw, 0, sentinel);

    const plan = createPlan(device, {
      type: "c2c",
      shape: logicalShape,
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inputViewShape, placement: "center" },
        output: { shape: outputViewShape, placement: "center", clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    assert(plan?._outOfCoreFourStepMode === true, "expected forced out-of-core mode");

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input, output });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outputBufRaw, outComplex);

    const logicalInput = embedComplexViewIntoLogical({
      logicalShape,
      viewShape: inputViewShape,
      offset: inputOffset,
      viewInterleaved: inputView,
    });
    const afterReadPad = zeroOutsideRangeComplexRef(logicalInput, logicalShape, readStart, readEnd);
    const ffted = fftNdRefAnySizeInterleaved(afterReadPad, logicalShape, "forward", "none");
    const afterWritePad = zeroOutsideRangeComplexRef(ffted, logicalShape, writeStart, writeEnd);

    const expected = new Float32Array(sentinel);
    for (let i = 0; i < outComplex; i++) {
      const vc = linearToCoords(i, outputViewShape);
      const lc = new Array(outputViewShape.length);
      let inside = true;
      for (let d = 0; d < outputViewShape.length; d++) {
        lc[d] = vc[d] + outputOffset[d];
        if (lc[d] < 0 || lc[d] >= logicalShape[d]) {
          inside = false;
          break;
        }
      }
      if (!inside) continue;
      const li = coordsToLinear(lc, logicalShape);
      expected[2 * i] = afterWritePad[2 * li];
      expected[2 * i + 1] = afterWritePad[2 * li + 1];
    }

    assertCloseArray(gpu, expected, 2e-3, 2e-3, "c2c forced out-of-core rank4 ioView in+out + zeroPad");

    plan.destroy();
    inputBufRaw.destroy();
    outputBufRaw.destroy();
  });

  test("c2c batch>1 (N=32,batch=4) matches CPU", async () => {
    const device = await ensureDevice();
    const N = 32;
    const batch = 4;
    const { inputData, out } = await runC2cOnce(device, { shape: [N], batch, direction: "forward", normalize: "none", inPlace: false });
    const expected = new Float32Array(out.length);
    for (let b = 0; b < batch; b++) {
      const slice = inputData.subarray(2 * b * N, 2 * (b + 1) * N);
      const cpu = fftNdRefAnySizeInterleaved(slice, [N], "forward", "none");
      expected.set(cpu, 2 * b * N);
    }
    assertCloseArray(out, expected, 5e-4, 5e-4, "c2c batch");
  });

  test("c2c forced out-of-core path matches baseline with custom input/output strides (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const logicalTotal = prod(shape);
    const logicalInput = randomComplexInterleaved(logicalTotal * batch);

    const inStrides = [2, 11, 37];
    const outStrides = [3, 17, 61];
    const inOffset = 3;
    const outOffset = 4;
    const inBatchStride = strideSpanElements(shape, inStrides) + 17;
    const outBatchStride = strideSpanElements(shape, outStrides) + 19;

    const inPhys = makeStridedPhysicalComplex({
      shape,
      batch,
      strides: inStrides,
      offsetElements: inOffset,
      batchStrideElements: inBatchStride,
      logicalInterleaved: logicalInput,
    });
    const outElems = requiredStridedElements(shape, outStrides, outOffset, batch, outBatchStride);
    const outBytes = outElems * 8;

    const inBuf = uploadComplex(device, inPhys);
    const outForced = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBase = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const forced = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        outputStrides: outStrides,
        inputOffsetElements: inOffset,
        outputOffsetElements: outOffset,
        inputBatchStrideElements: inBatchStride,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    const baseline = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        outputStrides: outStrides,
        inputOffsetElements: inOffset,
        outputOffsetElements: outOffset,
        inputBatchStrideElements: inBatchStride,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
    });
    assert(forced?._outOfCoreFourStepMode === true, "expected forced c2c out-of-core mode");
    assert(baseline?._outOfCoreFourStepMode !== true, "baseline should use non-out-of-core path");

    const enc = device.createCommandEncoder();
    forced.exec(enc, { input: inBuf, output: outForced });
    baseline.exec(enc, { input: inBuf, output: outBase });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outForcedPhys = await downloadF32(device, outForced, outBytes);
    const outBasePhys = await downloadF32(device, outBase, outBytes);
    const got = extractStridedLogicalComplex({
      physicalInterleaved: outForcedPhys,
      shape,
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });
    const expected = extractStridedLogicalComplex({
      physicalInterleaved: outBasePhys,
      shape,
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });
    assertCloseArray(got, expected, 2e-3, 2e-3, "c2c forced out-of-core custom strides");

    forced.destroy();
    baseline.destroy();
    inBuf.destroy();
    outForced.destroy();
    outBase.destroy();
  });

  test("c2c with custom input/output strides matches CPU (out-of-place, batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [7, 5];
    const batch = 2;
    const logicalTotal = prod(shape);
    const logicalInput = randomComplexInterleaved(logicalTotal * batch);

    const inStrides = [2, 19];
    const outStrides = [3, 31];
    const inOffset = 5;
    const outOffset = 9;
    const inBatchStride = strideSpanElements(shape, inStrides) + 17;
    const outBatchStride = strideSpanElements(shape, outStrides) + 11;

    const inPhys = makeStridedPhysicalComplex({
      shape,
      batch,
      strides: inStrides,
      offsetElements: inOffset,
      batchStrideElements: inBatchStride,
      logicalInterleaved: logicalInput,
    });
    const outElems = requiredStridedElements(shape, outStrides, outOffset, batch, outBatchStride);
    const outBytes = outElems * 8;

    const inBuf = uploadComplex(device, inPhys);
    const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        outputStrides: outStrides,
        inputOffsetElements: inOffset,
        outputOffsetElements: outOffset,
        inputBatchStrideElements: inBatchStride,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outPhys = await downloadF32(device, outBuf, outBytes);
    const outLogical = extractStridedLogicalComplex({
      physicalInterleaved: outPhys,
      shape,
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });

    const expected = new Float32Array(2 * logicalTotal * batch);
    for (let b = 0; b < batch; b++) {
      const slice = logicalInput.subarray(2 * b * logicalTotal, 2 * (b + 1) * logicalTotal);
      const cpu = fftNdRefAnySizeInterleaved(slice, shape, "forward", "none");
      expected.set(cpu, 2 * b * logicalTotal);
    }
    assertCloseArray(outLogical, expected, 9e-4, 9e-4, "c2c strided out-of-place");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c with shared custom strides works in-place", async () => {
    const device = await ensureDevice();
    const shape = [8, 4];
    const batch = 1;
    const logicalTotal = prod(shape);
    const logicalInput = randomComplexInterleaved(logicalTotal);

    const strides = [2, 21];
    const offset = 3;
    const batchStride = strideSpanElements(shape, strides) + 5;

    const phys = makeStridedPhysicalComplex({
      shape,
      batch,
      strides,
      offsetElements: offset,
      batchStrideElements: batchStride,
      logicalInterleaved: logicalInput,
      sentinelRe: 77,
      sentinelIm: -88,
    });

    const buf = uploadComplex(device, phys);
    const plan = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: true,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        strides,
        offsetElements: offset,
        batchStrideElements: batchStride,
      },
      precision: "f32",
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: buf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outPhys = await downloadF32(device, buf, phys.byteLength);
    const outLogical = extractStridedLogicalComplex({
      physicalInterleaved: outPhys,
      shape,
      batch,
      strides,
      offsetElements: offset,
      batchStrideElements: batchStride,
    });
    const cpu = fftNdRefAnySizeInterleaved(logicalInput, shape, "forward", "none");
    assertCloseArray(outLogical, cpu, 9e-4, 9e-4, "c2c strided in-place");

    plan.destroy();
    buf.destroy();
  });

  test("c2c ioView pad-in-read center works (logical 16, input 8)", async () => {
    const device = await ensureDevice();
    const logical = 16;
    const inputN = 8;
    const inputData = randomComplexInterleaved(inputN);
    const padded = new Float32Array(2 * logical);
    const off = Math.floor((logical - inputN) / 2);
    padded.set(inputData, 2 * off);

    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({ size: padded.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

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
    assertCloseArray(out, cpu, 5e-4, 5e-4, "c2c ioView input");

    if (exportArtifact) {
      exportArtifact({
        name: "c2c_f32_ioview_input_center_N16_V8",
        plan: {
          type: "c2c",
          shape: [logical],
          direction: "forward",
          normalize: "none",
          batch: 1,
          inPlace: false,
          layout: { interleavedComplex: true },
          precision: "f32",
          ioView: { input: { shape: [inputN], placement: "center" } },
        },
        input: { kind: "complex-f32", data: Array.from(inputData) },
        cpu: { kind: "complex-f32", data: Array.from(cpu) },
        gpu: { kind: "complex-f32", data: Array.from(out) },
        tol: { atol: 5e-4, rtol: 5e-4 },
      });
    }

    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("c2c ioView pad-in-write (logical 16 -> output view 32 center clearOutside) matches CPU embed", async () => {
    const device = await ensureDevice();
    const logical = 16;
    const viewOut = 32;
    const off = Math.floor((logical - viewOut) / 2);
    const inputData = randomComplexInterleaved(logical);

    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({ size: viewOut * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

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
    assertCloseArray(out, cpuView, 7e-4, 7e-4, "c2c ioView output");

    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("c2c ioView clearOutside=false preserves existing output outside logical region", async () => {
    const device = await ensureDevice();
    const logical = 16;
    const viewOut = 32;
    const off = Math.floor((logical - viewOut) / 2);
    const inputData = randomComplexInterleaved(logical);

    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({ size: viewOut * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

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
    for (let i = 0; i < viewOut; i++) {
      const l = i + off;
      if (l >= 0 && l < logical) continue;
      assertEq(out[2 * i], sentinelRe, "clearOutside=false (re)");
      assertEq(out[2 * i + 1], sentinelIm, "clearOutside=false (im)");
    }
    const mappedStart = -off;
    const mapped = out.subarray(2 * mappedStart, 2 * (mappedStart + logical));
    assertCloseArray(mapped, cpuLogical, 7e-4, 7e-4, "clearOutside=false region");

    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("c2c ioView rank4 input+output mapping matches baseline", async () => {
    const device = await ensureDevice();
    const shape = [4, 3, 2, 2];
    const inViewShape = [2, 3, 2, 2];
    const inOff = [1, 0, 0, 0];
    const outViewShape = [3, 3, 2, 2];
    const outOff = [1, 0, 0, 0];

    const inputView = randomComplexInterleaved(prod(inViewShape));
    const logicalInput = embedComplexViewIntoLogical({
      logicalShape: shape,
      viewShape: inViewShape,
      offset: inOff,
      viewInterleaved: inputView,
    });

    const inViewBuf = uploadComplex(device, inputView);
    const outViewBuf = device.createBuffer({
      size: prod(outViewShape) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const inFullBuf = uploadComplex(device, logicalInput);
    const outFullBuf = device.createBuffer({
      size: prod(shape) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const pIo = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inOff },
        output: { shape: outViewShape, offset: outOff },
      },
    });
    const pBase = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
    });

    {
      const enc = device.createCommandEncoder();
      pBase.exec(enc, { input: inFullBuf, output: outFullBuf });
      pIo.exec(enc, { input: inViewBuf, output: outViewBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const full = await downloadComplex(device, outFullBuf, prod(shape));
    const expectedView = extractComplexViewFromLogical({
      logicalShape: shape,
      viewShape: outViewShape,
      offset: outOff,
      logicalInterleaved: full,
    });
    const got = await downloadComplex(device, outViewBuf, prod(outViewShape));
    assertCloseArray(got, expectedView, 1e-3, 1e-3, "c2c ioView rank4");

    pIo.destroy();
    pBase.destroy();
    inViewBuf.destroy();
    outViewBuf.destroy();
    inFullBuf.destroy();
    outFullBuf.destroy();
  });

  test("c2c zeroPad read+write ranges match CPU (N=16)", async () => {
    const device = await ensureDevice();
    const N = 16;
    const input = randomComplexInterleaved(N);
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const readStart = [3];
    const readEnd = [13];
    const writeStart = [2];
    const writeEnd = [12];

    const plan = createPlan(device, {
      type: "c2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const got = await downloadComplex(device, outBuf, N);
    const cpuRead = zeroOutsideRangeComplexRef(input, [N], readStart, readEnd);
    const cpuFft = fftNdRefAnySizeInterleaved(cpuRead, [N], "forward", "none");
    const expected = zeroOutsideRangeComplexRef(cpuFft, [N], writeStart, writeEnd);
    assertCloseArray(got, expected, 1e-3, 1e-3, "c2c zeroPad");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c segmented BufferView input/output works (N=128)", async () => {
    const device = await ensureDevice();
    const N = 128;
    let res;
    try {
      res = await runC2cOnce(device, { shape: [N], direction: "forward", normalize: "none", inPlace: false, segmented: true });
    } catch (e) {
      throw new SkipError(String(e));
    }
    const cpu = fftNdRefAnySizeInterleaved(res.inputData, [N], "forward", "none");
    assertCloseArray(res.out, cpu, 7e-4, 7e-4, "c2c BufferView");
  });

  test("BufferView Tier-B fallback (segmentCount > cap) works for c2c (N=32)", async () => {
    const device = await ensureDevice();
    const N = 32;
    const inputData = randomComplexInterleaved(N);
    const inputBuf = uploadComplex(device, inputData);
    const outputBuf = device.createBuffer({ size: inputData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const maxStorage = device.limits?.maxStorageBuffersPerShaderStage ?? 8;
    const segCap = Math.max(0, Math.min(8, maxStorage - 1));
    const segments = Math.max(1, segCap + 1);
    log(`Tier-B forcing segments=${segments} (cap=${segCap})`);

    const input = splitBufferView(inputBuf, inputData.byteLength, segments);
    const output = splitBufferView(outputBuf, inputData.byteLength, segments);

    const plan = createPlan(device, { type: "c2c", shape: [N], direction: "forward", batch: 1, inPlace: false, normalize: "none", layout: { interleavedComplex: true }, precision: "f32" });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input, output });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await downloadComplex(device, outputBuf, N);
    const cpu = fftNdRefAnySizeInterleaved(inputData, [N], "forward", "none");
    assertCloseArray(out, cpu, 3e-4, 3e-4, "Tier-B c2c");

    plan.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
  });

  test("r2c/c2r packed conventions roundtrip (N=17)", async () => {
    const device = await ensureDevice();
    const N = 17;
    const inputReal = new Float32Array(N);
    for (let i = 0; i < N; i++) inputReal[i] = (Math.random() * 2 - 1) * 0.5;

    const inputBuf = device.createBuffer({ size: inputReal.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inputBuf, 0, inputReal);

    const outLen = Math.floor(N / 2) + 1;
    const packedBuf = device.createBuffer({ size: outLen * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const r2c = createPlan(device, { type: "r2c", shape: [N], direction: "forward", batch: 1, normalize: "none", layout: { interleavedComplex: true }, precision: "f32" });
    const enc1 = device.createCommandEncoder();
    r2c.exec(enc1, { input: inputBuf, output: packedBuf });
    device.queue.submit([enc1.finish()]);
    await device.queue.onSubmittedWorkDone();
    const packed = await downloadComplex(device, packedBuf, outLen);

    const cpuPacked = r2cRefPackedInterleaved(inputReal, N, "forward", "none");
    assertCloseArray(packed, cpuPacked, 8e-4, 8e-4, "r2c packed");
    r2c.destroy();

    const outRealBuf = device.createBuffer({ size: inputReal.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const c2r = createPlan(device, { type: "c2r", shape: [N], direction: "inverse", batch: 1, normalize: "backward", layout: { interleavedComplex: true }, precision: "f32" });
    const enc2 = device.createCommandEncoder();
    c2r.exec(enc2, { input: packedBuf, output: outRealBuf });
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outReal2 = await downloadF32(device, outRealBuf, inputReal.byteLength);
    assertCloseArray(outReal2, inputReal, 2e-3, 2e-3, "c2r roundtrip");

    c2r.destroy();
    inputBuf.destroy();
    packedBuf.destroy();
    outRealBuf.destroy();
  });

  test("r2c/c2r 4D roundtrip with backward normalization (9x3x2x2)", async () => {
    const device = await ensureDevice();
    const shape = [9, 3, 2, 2];
    const nTotal = shape.reduce((a, b) => a * b, 1);
    const packedTotal = (Math.floor(shape[0] / 2) + 1) * shape.slice(1).reduce((a, b) => a * b, 1);

    const inputReal = new Float32Array(nTotal);
    for (let i = 0; i < nTotal; i++) inputReal[i] = (Math.random() * 2 - 1) * 0.5;

    const inputBuf = device.createBuffer({ size: inputReal.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inputBuf, 0, inputReal);

    const packedBuf = device.createBuffer({ size: packedTotal * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outRealBuf = device.createBuffer({ size: inputReal.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const r2c = createPlan(device, { type: "r2c", shape, direction: "forward", batch: 1, normalize: "none", layout: { interleavedComplex: true }, precision: "f32" });
    const c2r = createPlan(device, { type: "c2r", shape, direction: "inverse", batch: 1, normalize: "backward", layout: { interleavedComplex: true }, precision: "f32" });

    const enc = device.createCommandEncoder();
    r2c.exec(enc, { input: inputBuf, output: packedBuf });
    c2r.exec(enc, { input: packedBuf, output: outRealBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outReal = await downloadF32(device, outRealBuf, inputReal.byteLength);
    assertCloseArray(outReal, inputReal, 3e-3, 3e-3, "r2c/c2r 4D roundtrip");

    r2c.destroy();
    c2r.destroy();
    inputBuf.destroy();
    packedBuf.destroy();
    outRealBuf.destroy();
  });

  test("r2c forced large-shape path matches baseline with ioView+zeroPad (rank-3, segmented, clearOutside=false)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];
    const inViewShape = [3, 3, 2];
    const inOffset = [1, 0, 0];
    const outViewShape = [4, 3, 2];
    const outOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [5, 3, 2];
    const writeStart = [0, 0, 0];
    const writeEnd = [2, 3, 2];

    const inputView = new Float32Array(prod(inViewShape));
    for (let i = 0; i < inputView.length; i++) inputView[i] = (Math.random() * 2 - 1) * 0.5;

    const inBufRaw = device.createBuffer({
      size: inputView.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(inBufRaw, 0, inputView);
    const inView = splitBufferView(inBufRaw, inputView.byteLength, 2);

    const outBytes = prod(outViewShape) * 8;
    const outForcedRaw = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBaseRaw = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outForced = splitBufferView(outForcedRaw, outBytes, 3);
    const outBase = splitBufferView(outBaseRaw, outBytes, 2);

    const sentinel = new Float32Array(2 * prod(outViewShape));
    for (let i = 0; i < sentinel.length / 2; i++) {
      sentinel[2 * i] = 111.25;
      sentinel[2 * i + 1] = -222.5;
    }
    device.queue.writeBuffer(outForcedRaw, 0, sentinel);
    device.queue.writeBuffer(outBaseRaw, 0, sentinel);

    const forced = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inOffset },
        output: { shape: outViewShape, offset: outOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    assert(forced?._largeShapeMode === true, "expected forced r2c large-shape mode");

    const baseline = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inOffset },
        output: { shape: outViewShape, offset: outOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    assert(baseline?._largeShapeMode !== true, "baseline should use regular path");

    {
      const enc = device.createCommandEncoder();
      forced.exec(enc, { input: inView, output: outForced });
      baseline.exec(enc, { input: inView, output: outBase });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const got = await downloadComplex(device, outForcedRaw, prod(outViewShape));
    const expected = await downloadComplex(device, outBaseRaw, prod(outViewShape));
    assertCloseArray(got, expected, 2e-3, 2e-3, "r2c forced large-shape fallback");

    forced.destroy();
    baseline.destroy();
    inBufRaw.destroy();
    outForcedRaw.destroy();
    outBaseRaw.destroy();
    assertEq(packedShape[0], 3, "packed-shape sanity");
  });

  test("c2r forced large-shape path matches baseline with ioView+zeroPad (rank-3, segmented, clearOutside=false)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];
    const inViewShape = [2, 3, 2];
    const inOffset = [1, 0, 0];
    const outViewShape = [7, 3, 2];
    const outOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [3, 3, 2];
    const writeStart = [1, 0, 0];
    const writeEnd = [5, 3, 2];

    const packedView = randomComplexInterleaved(prod(inViewShape));
    const inBufRaw = uploadComplex(device, packedView);
    const inView = splitBufferView(inBufRaw, packedView.byteLength, 2);

    const outBytes = prod(outViewShape) * 4;
    const outForcedRaw = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBaseRaw = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outForced = splitBufferView(outForcedRaw, outBytes, 3);
    const outBase = splitBufferView(outBaseRaw, outBytes, 2);

    const sentinel = new Float32Array(prod(outViewShape));
    for (let i = 0; i < sentinel.length; i++) sentinel[i] = 77.375;
    device.queue.writeBuffer(outForcedRaw, 0, sentinel);
    device.queue.writeBuffer(outBaseRaw, 0, sentinel);

    const forced = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inOffset },
        output: { shape: outViewShape, offset: outOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    assert(forced?._largeShapeMode === true, "expected forced c2r large-shape mode");

    const baseline = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inOffset },
        output: { shape: outViewShape, offset: outOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    assert(baseline?._largeShapeMode !== true, "baseline should use regular path");

    {
      const enc = device.createCommandEncoder();
      forced.exec(enc, { input: inView, output: outForced });
      baseline.exec(enc, { input: inView, output: outBase });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const got = await downloadF32(device, outForcedRaw, outBytes);
    const expected = await downloadF32(device, outBaseRaw, outBytes);
    assertCloseArray(got, expected, 3e-3, 3e-3, "c2r forced large-shape fallback");

    forced.destroy();
    baseline.destroy();
    inBufRaw.destroy();
    outForcedRaw.destroy();
    outBaseRaw.destroy();
    assertEq(packedShape[0], 3, "packed-shape sanity");
  });

  test("c2r forced large-shape path matches baseline (rank-3, no ioView/zeroPad)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];
    const packed = randomComplexInterleaved(prod(packedShape));

    const inBuf = uploadComplex(device, packed);
    const outForced = device.createBuffer({
      size: prod(shape) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBase = device.createBuffer({
      size: prod(shape) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const forced = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    const baseline = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
    });
    assert(forced?._largeShapeMode === true, "expected forced c2r large mode");

    const enc = device.createCommandEncoder();
    forced.exec(enc, { input: inBuf, output: outForced });
    baseline.exec(enc, { input: inBuf, output: outBase });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const got = await downloadF32(device, outForced, prod(shape) * 4);
    const expected = await downloadF32(device, outBase, prod(shape) * 4);
    assertCloseArray(got, expected, 3e-3, 3e-3, "c2r forced large no-io baseline");

    forced.destroy();
    baseline.destroy();
    inBuf.destroy();
    outForced.destroy();
    outBase.destroy();
  });

  test("r2c forced large-shape oversized-axis line path matches baseline (32x3)", async () => {
    const device = await ensureDevice();
    const shape = [32, 3];
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];
    const real = new Float32Array(prod(shape));
    for (let i = 0; i < real.length; i++) real[i] = (Math.random() * 2 - 1) * 0.5;

    const inBuf = device.createBuffer({ size: real.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, real);
    const outForced = device.createBuffer({
      size: prod(packedShape) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBase = device.createBuffer({
      size: prod(packedShape) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const forced = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    const baseline = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
    });
    assert(forced?._largeShapeMode === true, "expected forced r2c large-shape mode");
    assert(forced?._oversizedLineMode === true, "expected oversized-line mode");
    assert(baseline?._largeShapeMode !== true, "baseline should use regular path");

    const enc = device.createCommandEncoder();
    forced.exec(enc, { input: inBuf, output: outForced });
    baseline.exec(enc, { input: inBuf, output: outBase });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const got = await downloadComplex(device, outForced, prod(packedShape));
    const expected = await downloadComplex(device, outBase, prod(packedShape));
    assertCloseArray(got, expected, 2e-3, 2e-3, "r2c oversized-axis large-shape");

    forced.destroy();
    baseline.destroy();
    inBuf.destroy();
    outForced.destroy();
    outBase.destroy();
  });

  test("c2r forced large-shape oversized-axis line path matches baseline (32x3)", async () => {
    const device = await ensureDevice();
    const shape = [32, 3];
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];
    const packed = randomComplexInterleaved(prod(packedShape));

    const inBuf = uploadComplex(device, packed);
    const outForced = device.createBuffer({
      size: prod(shape) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBase = device.createBuffer({
      size: prod(shape) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const forced = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    const baseline = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
    });
    assert(forced?._largeShapeMode === true, "expected forced c2r large-shape mode");
    assert(forced?._oversizedLineMode === true, "expected oversized-line mode");
    assert(baseline?._largeShapeMode !== true, "baseline should use regular path");

    const enc = device.createCommandEncoder();
    forced.exec(enc, { input: inBuf, output: outForced });
    baseline.exec(enc, { input: inBuf, output: outBase });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const got = await downloadF32(device, outForced, prod(shape) * 4);
    const expected = await downloadF32(device, outBase, prod(shape) * 4);
    assertCloseArray(got, expected, 3e-3, 3e-3, "c2r oversized-axis large-shape");

    forced.destroy();
    baseline.destroy();
    inBuf.destroy();
    outForced.destroy();
    outBase.destroy();
  });

  test("r2c forced large-shape path matches baseline with custom input/output strides (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];

    const inStrides = [2, 8, 24];
    const inOffset = 3;
    const inBatchStride = strideSpanElements(shape, inStrides) + 11;
    const logicalIn = new Float32Array(prod(shape) * batch);
    for (let i = 0; i < logicalIn.length; i++) logicalIn[i] = (Math.random() * 2 - 1) * 0.5;
    const inPhys = makeStridedPhysicalReal({
      shape,
      batch,
      strides: inStrides,
      offsetElements: inOffset,
      batchStrideElements: inBatchStride,
      logical: logicalIn,
    });
    const inBuf = device.createBuffer({ size: inPhys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, inPhys);

    const outStrides = [3, 9, 27];
    const outOffset = 4;
    const outBatchStride = strideSpanElements(packedShape, outStrides) + 11;
    const outElems = requiredStridedElements(packedShape, outStrides, outOffset, batch, outBatchStride);
    const outBytes = outElems * 8;
    const outForced = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBase = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const forced = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffset,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffset,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    const baseline = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffset,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffset,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
    });
    assert(forced?._largeShapeMode === true, "expected forced r2c large-shape mode");
    assert(baseline?._largeShapeMode !== true, "baseline should use regular path");

    {
      const enc = device.createCommandEncoder();
      forced.exec(enc, { input: inBuf, output: outForced });
      baseline.exec(enc, { input: inBuf, output: outBase });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const forcedPhys = await downloadF32(device, outForced, outBytes);
    const basePhys = await downloadF32(device, outBase, outBytes);
    const got = extractStridedLogicalComplex({
      physicalInterleaved: forcedPhys,
      shape: packedShape,
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });
    const expected = extractStridedLogicalComplex({
      physicalInterleaved: basePhys,
      shape: packedShape,
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });
    assertCloseArray(got, expected, 2e-3, 2e-3, "r2c large-shape custom strides baseline");

    forced.destroy();
    baseline.destroy();
    inBuf.destroy();
    outForced.destroy();
    outBase.destroy();
  });

  test("c2r forced large-shape path matches baseline with custom input/output strides (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];

    const inStrides = [2, 7, 21];
    const inOffset = 5;
    const inBatchStride = strideSpanElements(packedShape, inStrides) + 13;
    const logicalInPacked = randomComplexInterleaved(prod(packedShape) * batch);
    const inPhys = makeStridedPhysicalComplex({
      shape: packedShape,
      batch,
      strides: inStrides,
      offsetElements: inOffset,
      batchStrideElements: inBatchStride,
      logicalInterleaved: logicalInPacked,
    });
    const inBuf = uploadComplex(device, inPhys);

    const outStrides = [3, 10, 30];
    const outOffset = 2;
    const outBatchStride = strideSpanElements(shape, outStrides) + 11;
    const outElems = requiredStridedElements(shape, outStrides, outOffset, batch, outBatchStride);
    const outBytes = outElems * 4;
    const outForced = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBase = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const forced = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffset,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffset,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    const baseline = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffset,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffset,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
    });
    assert(forced?._largeShapeMode === true, "expected forced c2r large-shape mode");
    assert(baseline?._largeShapeMode !== true, "baseline should use regular path");

    {
      const enc = device.createCommandEncoder();
      forced.exec(enc, { input: inBuf, output: outForced });
      baseline.exec(enc, { input: inBuf, output: outBase });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const forcedPhys = await downloadF32(device, outForced, outBytes);
    const basePhys = await downloadF32(device, outBase, outBytes);
    const got = extractStridedLogicalReal({
      physical: forcedPhys,
      shape,
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });
    const expected = extractStridedLogicalReal({
      physical: basePhys,
      shape,
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });
    assertCloseArray(got, expected, 3e-3, 3e-3, "c2r large-shape custom strides baseline");

    forced.destroy();
    baseline.destroy();
    inBuf.destroy();
    outForced.destroy();
    outBase.destroy();
  });

  test("r2c forced large-shape path matches baseline with custom strides + ioView+zeroPad (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const inViewShape = [3, 3, 2];
    const inViewOffset = [1, 0, 0];
    const outViewShape = [4, 3, 2];
    const outViewOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [5, 3, 2];
    const writeStart = [0, 0, 0];
    const writeEnd = [2, 3, 2];

    const inViewLogical = new Float32Array(prod(inViewShape) * batch);
    for (let i = 0; i < inViewLogical.length; i++) inViewLogical[i] = (Math.random() * 2 - 1) * 0.5;

    const inStrides = [2, 9, 30];
    const inOffsetElements = 3;
    const inBatchStride = strideSpanElements(inViewShape, inStrides) + 17;
    const inPhys = makeStridedPhysicalReal({
      shape: inViewShape,
      batch,
      strides: inStrides,
      offsetElements: inOffsetElements,
      batchStrideElements: inBatchStride,
      logical: inViewLogical,
    });
    const inForced = device.createBuffer({ size: inPhys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const inBase = device.createBuffer({ size: inViewLogical.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inForced, 0, inPhys);
    device.queue.writeBuffer(inBase, 0, inViewLogical);

    const outStrides = [3, 14, 40];
    const outOffsetElements = 5;
    const outBatchStride = strideSpanElements(outViewShape, outStrides) + 19;
    const outElems = requiredStridedElements(outViewShape, outStrides, outOffsetElements, batch, outBatchStride);
    const outForcedBytes = outElems * 8;
    const outBaseBytes = prod(outViewShape) * batch * 8;
    const outForced = device.createBuffer({ size: outForcedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBase = device.createBuffer({ size: outBaseBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const outViewSentinel = new Float32Array(2 * prod(outViewShape) * batch);
    for (let i = 0; i < outViewSentinel.length / 2; i++) {
      outViewSentinel[2 * i] = 111.75;
      outViewSentinel[2 * i + 1] = -222.125;
    }
    const outForcedSentinel = makeStridedPhysicalComplex({
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
      logicalInterleaved: outViewSentinel,
    });
    device.queue.writeBuffer(outForced, 0, outForcedSentinel);
    device.queue.writeBuffer(outBase, 0, outViewSentinel);

    const forced = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffsetElements,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffsetElements,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    const baseline = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    assert(forced?._largeShapeMode === true, "expected forced r2c large-shape mode");
    assert(baseline?._largeShapeMode !== true, "baseline should use regular path");

    {
      const enc = device.createCommandEncoder();
      forced.exec(enc, { input: inForced, output: outForced });
      baseline.exec(enc, { input: inBase, output: outBase });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const forcedPhys = await downloadF32(device, outForced, outForcedBytes);
    const got = extractStridedLogicalComplex({
      physicalInterleaved: forcedPhys,
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
    });
    const expected = await downloadComplex(device, outBase, prod(outViewShape) * batch);
    assertCloseArray(got, expected, 2e-3, 2e-3, "r2c large-shape custom strides + ioView+zeroPad");

    forced.destroy();
    baseline.destroy();
    inForced.destroy();
    inBase.destroy();
    outForced.destroy();
    outBase.destroy();
  });

  test("c2r forced large-shape path matches baseline with custom strides + ioView+zeroPad (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const inViewShape = [2, 3, 2];
    const inViewOffset = [1, 0, 0];
    const outViewShape = [7, 3, 2];
    const outViewOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [3, 3, 2];
    const writeStart = [1, 0, 0];
    const writeEnd = [5, 3, 2];

    const inViewLogical = randomComplexInterleaved(prod(inViewShape) * batch);
    const inStrides = [2, 7, 25];
    const inOffsetElements = 4;
    const inBatchStride = strideSpanElements(inViewShape, inStrides) + 13;
    const inPhys = makeStridedPhysicalComplex({
      shape: inViewShape,
      batch,
      strides: inStrides,
      offsetElements: inOffsetElements,
      batchStrideElements: inBatchStride,
      logicalInterleaved: inViewLogical,
    });
    const inForced = uploadComplex(device, inPhys);
    const inBase = uploadComplex(device, inViewLogical);

    const outStrides = [2, 16, 48];
    const outOffsetElements = 3;
    const outBatchStride = strideSpanElements(outViewShape, outStrides) + 17;
    const outElems = requiredStridedElements(outViewShape, outStrides, outOffsetElements, batch, outBatchStride);
    const outForcedBytes = outElems * 4;
    const outBaseBytes = prod(outViewShape) * batch * 4;
    const outForced = device.createBuffer({ size: outForcedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBase = device.createBuffer({ size: outBaseBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const outViewSentinel = new Float32Array(prod(outViewShape) * batch);
    for (let i = 0; i < outViewSentinel.length; i++) outViewSentinel[i] = 77.625;
    const outForcedSentinel = makeStridedPhysicalReal({
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
      logical: outViewSentinel,
    });
    device.queue.writeBuffer(outForced, 0, outForcedSentinel);
    device.queue.writeBuffer(outBase, 0, outViewSentinel);

    const forced = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffsetElements,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffsetElements,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
      tuning: { maxStorageBufferBindingSize: 64 },
    });
    const baseline = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    assert(forced?._largeShapeMode === true, "expected forced c2r large-shape mode");
    assert(baseline?._largeShapeMode !== true, "baseline should use regular path");

    {
      const enc = device.createCommandEncoder();
      forced.exec(enc, { input: inForced, output: outForced });
      baseline.exec(enc, { input: inBase, output: outBase });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const forcedPhys = await downloadF32(device, outForced, outForcedBytes);
    const got = extractStridedLogicalReal({
      physical: forcedPhys,
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
    });
    const expected = await downloadF32(device, outBase, outBaseBytes);
    assertCloseArray(got, expected, 3e-3, 3e-3, "c2r large-shape custom strides + ioView+zeroPad");

    forced.destroy();
    baseline.destroy();
    inForced.destroy();
    inBase.destroy();
    outForced.destroy();
    outBase.destroy();
  });

  test("c2c regular path matches baseline with custom strides + ioView+zeroPad (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const inViewShape = [3, 3, 2];
    const inViewOffset = [1, 0, 0];
    const outViewShape = [7, 3, 2];
    const outViewOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [4, 3, 2];
    const writeStart = [0, 1, 0];
    const writeEnd = [5, 3, 2];

    const inViewLogical = randomComplexInterleaved(prod(inViewShape) * batch);
    const inStrides = [2, 11, 37];
    const inOffsetElements = 3;
    const inBatchStride = strideSpanElements(inViewShape, inStrides) + 17;
    const inPhys = makeStridedPhysicalComplex({
      shape: inViewShape,
      batch,
      strides: inStrides,
      offsetElements: inOffsetElements,
      batchStrideElements: inBatchStride,
      logicalInterleaved: inViewLogical,
    });
    const inStridedBuf = uploadComplex(device, inPhys);
    const inBaseBuf = uploadComplex(device, inViewLogical);

    const outStrides = [3, 17, 61];
    const outOffsetElements = 4;
    const outBatchStride = strideSpanElements(outViewShape, outStrides) + 19;
    const outElems = requiredStridedElements(outViewShape, outStrides, outOffsetElements, batch, outBatchStride);
    const outStridedBytes = outElems * 8;
    const outBaseComplex = prod(outViewShape) * batch;
    const outBaseBytes = outBaseComplex * 8;
    const outStridedBuf = device.createBuffer({ size: outStridedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBaseBuf = device.createBuffer({ size: outBaseBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const outViewSentinel = randomComplexInterleaved(outBaseComplex);
    const outStridedSentinel = makeStridedPhysicalComplex({
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
      logicalInterleaved: outViewSentinel,
    });
    device.queue.writeBuffer(outStridedBuf, 0, outStridedSentinel);
    device.queue.writeBuffer(outBaseBuf, 0, outViewSentinel);

    const stridedPlan = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffsetElements,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffsetElements,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    const baselinePlan = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    assert(stridedPlan?._outOfCoreFourStepMode !== true, "expected regular (non-out-of-core) c2c path");
    assert(baselinePlan?._outOfCoreFourStepMode !== true, "expected regular baseline path");

    {
      const enc = device.createCommandEncoder();
      stridedPlan.exec(enc, { input: inStridedBuf, output: outStridedBuf });
      baselinePlan.exec(enc, { input: inBaseBuf, output: outBaseBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const stridedPhys = await downloadF32(device, outStridedBuf, outStridedBytes);
    const got = extractStridedLogicalComplex({
      physicalInterleaved: stridedPhys,
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
    });
    const expected = await downloadComplex(device, outBaseBuf, outBaseComplex);
    assertCloseArray(got, expected, 2e-3, 2e-3, "c2c regular custom strides + ioView+zeroPad");

    stridedPlan.destroy();
    baselinePlan.destroy();
    inStridedBuf.destroy();
    inBaseBuf.destroy();
    outStridedBuf.destroy();
    outBaseBuf.destroy();
  });

  test("c2c layout.whdcn shorthand matches explicit strided layout (rank-2,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [4, 3];
    const batch = 2;

    const inStrides = [1, 4];
    const inOffsetElements = 12;
    const inBatchStride = 36;
    const logicalIn = randomComplexInterleaved(prod(shape) * batch);
    const inPhys = makeStridedPhysicalComplex({
      shape,
      batch,
      strides: inStrides,
      offsetElements: inOffsetElements,
      batchStrideElements: inBatchStride,
      logicalInterleaved: logicalIn,
    });
    const inExplicit = uploadComplex(device, inPhys);
    const inWhdcn = uploadComplex(device, inPhys);

    const outStrides = [1, 4];
    const outOffsetElements = 12;
    const outBatchStride = 24;
    const outElems = requiredStridedElements(shape, outStrides, outOffsetElements, batch, outBatchStride);
    const outBytes = outElems * 8;
    const outSentinel = new Float32Array(2 * outElems);
    for (let i = 0; i < outElems; i++) {
      outSentinel[2 * i] = 77.0;
      outSentinel[2 * i + 1] = -55.0;
    }
    const outExplicit = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outWhdcn = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(outExplicit, 0, outSentinel);
    device.queue.writeBuffer(outWhdcn, 0, outSentinel);

    const explicitPlan = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffsetElements,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffsetElements,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
    });
    const whdcnPlan = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        whdcn: {
          input: { channels: 3, channelIndex: 1 },
          output: { channels: 2, channelIndex: 1 },
        },
      },
      precision: "f32",
    });

    {
      const enc = device.createCommandEncoder();
      explicitPlan.exec(enc, { input: inExplicit, output: outExplicit });
      whdcnPlan.exec(enc, { input: inWhdcn, output: outWhdcn });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const gotExplicit = await downloadF32(device, outExplicit, outBytes);
    const gotWhdcn = await downloadF32(device, outWhdcn, outBytes);
    assertCloseArray(gotWhdcn, gotExplicit, 2e-3, 2e-3, "c2c whdcn parity");

    explicitPlan.destroy();
    whdcnPlan.destroy();
    inExplicit.destroy();
    inWhdcn.destroy();
    outExplicit.destroy();
    outWhdcn.destroy();
  });

  test("r2c/c2r layout.whdcn shorthand matches explicit strided layouts (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [9, 3, 2];
    const packedShape = [Math.floor(shape[0] / 2) + 1, shape[1], shape[2]];
    const batch = 2;

    const r2cInStrides = [1, 9, 27];
    const r2cInOffset = 54;
    const r2cInBatchStride = 108;
    const inputLogical = new Float32Array(prod(shape) * batch);
    for (let i = 0; i < inputLogical.length; i++) inputLogical[i] = (Math.random() * 2 - 1) * 0.5;
    const inPhys = makeStridedPhysicalReal({
      shape,
      batch,
      strides: r2cInStrides,
      offsetElements: r2cInOffset,
      batchStrideElements: r2cInBatchStride,
      logical: inputLogical,
    });
    const r2cInExplicit = device.createBuffer({ size: inPhys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const r2cInWhdcn = device.createBuffer({ size: inPhys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(r2cInExplicit, 0, inPhys);
    device.queue.writeBuffer(r2cInWhdcn, 0, inPhys);

    const r2cOutStrides = [1, 5, 15];
    const r2cOutOffset = 60;
    const r2cOutBatchStride = 90;
    const packedElems = requiredStridedElements(packedShape, r2cOutStrides, r2cOutOffset, batch, r2cOutBatchStride);
    const packedBytes = packedElems * 8;
    const packedExplicit = device.createBuffer({ size: packedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const packedWhdcn = device.createBuffer({ size: packedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const r2cExplicit = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: r2cInStrides,
        inputOffsetElements: r2cInOffset,
        inputBatchStrideElements: r2cInBatchStride,
        outputStrides: r2cOutStrides,
        outputOffsetElements: r2cOutOffset,
        outputBatchStrideElements: r2cOutBatchStride,
      },
      precision: "f32",
    });
    const r2cWhdcn = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        whdcn: {
          input: { channels: 2, channelIndex: 1 },
          output: { channels: 3, channelIndex: 2 },
        },
      },
      precision: "f32",
    });
    {
      const enc = device.createCommandEncoder();
      r2cExplicit.exec(enc, { input: r2cInExplicit, output: packedExplicit });
      r2cWhdcn.exec(enc, { input: r2cInWhdcn, output: packedWhdcn });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    const packedRef = await downloadF32(device, packedExplicit, packedBytes);
    const packedGot = await downloadF32(device, packedWhdcn, packedBytes);
    assertCloseArray(packedGot, packedRef, 2e-3, 2e-3, "r2c whdcn parity");

    const c2rOutStrides = [1, 9, 27];
    const c2rOutOffset = 54;
    const c2rOutBatchStride = 108;
    const realElems = requiredStridedElements(shape, c2rOutStrides, c2rOutOffset, batch, c2rOutBatchStride);
    const realBytes = realElems * 4;
    const realExplicit = device.createBuffer({ size: realBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const realWhdcn = device.createBuffer({ size: realBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const c2rExplicit = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: r2cOutStrides,
        inputOffsetElements: r2cOutOffset,
        inputBatchStrideElements: r2cOutBatchStride,
        outputStrides: c2rOutStrides,
        outputOffsetElements: c2rOutOffset,
        outputBatchStrideElements: c2rOutBatchStride,
      },
      precision: "f32",
    });
    const c2rWhdcn = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        whdcn: {
          input: { channels: 3, channelIndex: 2 },
          output: { channels: 2, channelIndex: 1 },
        },
      },
      precision: "f32",
    });
    {
      const enc = device.createCommandEncoder();
      c2rExplicit.exec(enc, { input: packedExplicit, output: realExplicit });
      c2rWhdcn.exec(enc, { input: packedWhdcn, output: realWhdcn });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    const realRef = await downloadF32(device, realExplicit, realBytes);
    const realGot = await downloadF32(device, realWhdcn, realBytes);
    assertCloseArray(realGot, realRef, 3e-3, 3e-3, "c2r whdcn parity");

    r2cExplicit.destroy();
    r2cWhdcn.destroy();
    c2rExplicit.destroy();
    c2rWhdcn.destroy();
    r2cInExplicit.destroy();
    r2cInWhdcn.destroy();
    packedExplicit.destroy();
    packedWhdcn.destroy();
    realExplicit.destroy();
    realWhdcn.destroy();
  });

  test("r2c regular path matches baseline with custom strides + ioView+zeroPad (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const inViewShape = [3, 3, 2];
    const inViewOffset = [1, 0, 0];
    const outViewShape = [4, 3, 2];
    const outViewOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [5, 3, 2];
    const writeStart = [0, 0, 0];
    const writeEnd = [2, 3, 2];

    const inViewLogical = new Float32Array(prod(inViewShape) * batch);
    for (let i = 0; i < inViewLogical.length; i++) inViewLogical[i] = (Math.random() * 2 - 1) * 0.5;

    const inStrides = [2, 9, 30];
    const inOffsetElements = 3;
    const inBatchStride = strideSpanElements(inViewShape, inStrides) + 17;
    const inPhys = makeStridedPhysicalReal({
      shape: inViewShape,
      batch,
      strides: inStrides,
      offsetElements: inOffsetElements,
      batchStrideElements: inBatchStride,
      logical: inViewLogical,
    });
    const inStridedBuf = device.createBuffer({ size: inPhys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const inBaseBuf = device.createBuffer({ size: inViewLogical.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inStridedBuf, 0, inPhys);
    device.queue.writeBuffer(inBaseBuf, 0, inViewLogical);

    const outStrides = [3, 14, 40];
    const outOffsetElements = 5;
    const outBatchStride = strideSpanElements(outViewShape, outStrides) + 19;
    const outElems = requiredStridedElements(outViewShape, outStrides, outOffsetElements, batch, outBatchStride);
    const outStridedBytes = outElems * 8;
    const outBaseBytes = prod(outViewShape) * batch * 8;
    const outStridedBuf = device.createBuffer({ size: outStridedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBaseBuf = device.createBuffer({ size: outBaseBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const outViewSentinel = new Float32Array(2 * prod(outViewShape) * batch);
    for (let i = 0; i < outViewSentinel.length / 2; i++) {
      outViewSentinel[2 * i] = 135.5;
      outViewSentinel[2 * i + 1] = -246.25;
    }
    const outStridedSentinel = makeStridedPhysicalComplex({
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
      logicalInterleaved: outViewSentinel,
    });
    device.queue.writeBuffer(outStridedBuf, 0, outStridedSentinel);
    device.queue.writeBuffer(outBaseBuf, 0, outViewSentinel);

    const stridedPlan = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffsetElements,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffsetElements,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    const baselinePlan = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    assert(stridedPlan?._largeShapeMode !== true, "expected regular (non-large) r2c path");
    assert(baselinePlan?._largeShapeMode !== true, "expected regular baseline path");

    {
      const enc = device.createCommandEncoder();
      stridedPlan.exec(enc, { input: inStridedBuf, output: outStridedBuf });
      baselinePlan.exec(enc, { input: inBaseBuf, output: outBaseBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const stridedPhys = await downloadF32(device, outStridedBuf, outStridedBytes);
    const got = extractStridedLogicalComplex({
      physicalInterleaved: stridedPhys,
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
    });
    const expected = await downloadComplex(device, outBaseBuf, prod(outViewShape) * batch);
    assertCloseArray(got, expected, 2e-3, 2e-3, "r2c regular custom strides + ioView+zeroPad");

    stridedPlan.destroy();
    baselinePlan.destroy();
    inStridedBuf.destroy();
    inBaseBuf.destroy();
    outStridedBuf.destroy();
    outBaseBuf.destroy();
  });

  test("c2r regular path matches baseline with custom strides + ioView+zeroPad (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const inViewShape = [2, 3, 2];
    const inViewOffset = [1, 0, 0];
    const outViewShape = [7, 3, 2];
    const outViewOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [3, 3, 2];
    const writeStart = [1, 0, 0];
    const writeEnd = [5, 3, 2];

    const inViewLogical = randomComplexInterleaved(prod(inViewShape) * batch);
    const inStrides = [2, 7, 25];
    const inOffsetElements = 4;
    const inBatchStride = strideSpanElements(inViewShape, inStrides) + 13;
    const inPhys = makeStridedPhysicalComplex({
      shape: inViewShape,
      batch,
      strides: inStrides,
      offsetElements: inOffsetElements,
      batchStrideElements: inBatchStride,
      logicalInterleaved: inViewLogical,
    });
    const inStridedBuf = uploadComplex(device, inPhys);
    const inBaseBuf = uploadComplex(device, inViewLogical);

    const outStrides = [2, 16, 48];
    const outOffsetElements = 3;
    const outBatchStride = strideSpanElements(outViewShape, outStrides) + 17;
    const outElems = requiredStridedElements(outViewShape, outStrides, outOffsetElements, batch, outBatchStride);
    const outStridedBytes = outElems * 4;
    const outBaseBytes = prod(outViewShape) * batch * 4;
    const outStridedBuf = device.createBuffer({ size: outStridedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBaseBuf = device.createBuffer({ size: outBaseBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const outViewSentinel = new Float32Array(prod(outViewShape) * batch);
    for (let i = 0; i < outViewSentinel.length; i++) outViewSentinel[i] = 66.875;
    const outStridedSentinel = makeStridedPhysicalReal({
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
      logical: outViewSentinel,
    });
    device.queue.writeBuffer(outStridedBuf, 0, outStridedSentinel);
    device.queue.writeBuffer(outBaseBuf, 0, outViewSentinel);

    const stridedPlan = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffsetElements,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffsetElements,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    const baselinePlan = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    assert(stridedPlan?._largeShapeMode !== true, "expected regular (non-large) c2r path");
    assert(baselinePlan?._largeShapeMode !== true, "expected regular baseline path");

    {
      const enc = device.createCommandEncoder();
      stridedPlan.exec(enc, { input: inStridedBuf, output: outStridedBuf });
      baselinePlan.exec(enc, { input: inBaseBuf, output: outBaseBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const stridedPhys = await downloadF32(device, outStridedBuf, outStridedBytes);
    const got = extractStridedLogicalReal({
      physical: stridedPhys,
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
    });
    const expected = await downloadF32(device, outBaseBuf, outBaseBytes);
    assertCloseArray(got, expected, 3e-3, 3e-3, "c2r regular custom strides + ioView+zeroPad");

    stridedPlan.destroy();
    baselinePlan.destroy();
    inStridedBuf.destroy();
    inBaseBuf.destroy();
    outStridedBuf.destroy();
    outBaseBuf.destroy();
  });

  test("r2c ioView pad-in-read (shape=16, input view=8 center) matches CPU", async () => {
    const device = await ensureDevice();
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

    const plan = createPlan(device, { type: "r2c", shape: [N], direction: "forward", batch: 1, normalize: "none", layout: { interleavedComplex: true }, precision: "f32", ioView: { input: { shape: [V], placement: "center" } } });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const packed = await downloadComplex(device, outBuf, outLen);
    const cpu = r2cRefPackedInterleaved(logical, N, "forward", "none");
    assertCloseArray(packed, cpu, 8e-4, 8e-4, "r2c ioView input");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("r2c ioView pad-in-write (shape=17, packed output view=5 at offset=2) matches CPU extract", async () => {
    const device = await ensureDevice();
    const N = 17;
    const outFull = Math.floor(N / 2) + 1; // 9
    const viewLen = 5;
    const off = 2;
    const input = new Float32Array(N);
    for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
    const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBuf = device.createBuffer({ size: viewLen * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, { type: "r2c", shape: [N], direction: "forward", batch: 1, normalize: "none", layout: { interleavedComplex: true }, precision: "f32", ioView: { output: { shape: [viewLen], offset: [off] } } });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const packedView = await downloadComplex(device, outBuf, viewLen);
    const cpuFull = r2cRefPackedInterleaved(input, N, "forward", "none");
    const cpuView = cpuFull.slice(off * 2, (off + viewLen) * 2);
    assertCloseArray(packedView, cpuView, 8e-4, 8e-4, "r2c ioView output");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
    assertEq(outFull, 9, "sanity");
  });

  test("c2r ioView pad-in-read (shape=17, packed input view=5 at offset=2) matches CPU embed+ifft", async () => {
    const device = await ensureDevice();
    const N = 17;
    const outLen = Math.floor(N / 2) + 1; // 9
    const viewLen = 5;
    const off = 2;
    const packedView = randomComplexInterleaved(viewLen);
    const packedFull = new Float32Array(outLen * 2);
    packedFull.set(packedView, off * 2);

    const inBuf = uploadComplex(device, packedView);
    const outBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, { type: "c2r", shape: [N], direction: "inverse", batch: 1, normalize: "backward", layout: { interleavedComplex: true }, precision: "f32", ioView: { input: { shape: [viewLen], offset: [off] } } });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadF32(device, outBuf, N * 4);
    const cpu = c2rRefFromPackedInterleaved(packedFull, N, "backward");
    assertCloseArray(gpu, cpu, 2e-3, 2e-3, "c2r ioView input");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("r2c ioView rank4 input mapping matches baseline", async () => {
    const device = await ensureDevice();
    const shape = [9, 3, 2, 2];
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];
    const inViewShape = [5, 3, 2, 2];
    const inOff = [2, 0, 0, 0];

    const inputView = new Float32Array(prod(inViewShape));
    for (let i = 0; i < inputView.length; i++) inputView[i] = (Math.random() * 2 - 1) * 0.5;
    const inputLogical = embedRealViewIntoLogical({
      logicalShape: shape,
      viewShape: inViewShape,
      offset: inOff,
      view: inputView,
    });

    const inViewBuf = device.createBuffer({
      size: inputView.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(inViewBuf, 0, inputView);
    const inFullBuf = device.createBuffer({
      size: inputLogical.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(inFullBuf, 0, inputLogical);

    const outIoBuf = device.createBuffer({
      size: prod(packedShape) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBaseBuf = device.createBuffer({
      size: prod(packedShape) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const pIo = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { input: { shape: inViewShape, offset: inOff } },
    });
    const pBase = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
    });

    {
      const enc = device.createCommandEncoder();
      pIo.exec(enc, { input: inViewBuf, output: outIoBuf });
      pBase.exec(enc, { input: inFullBuf, output: outBaseBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const got = await downloadComplex(device, outIoBuf, prod(packedShape));
    const expected = await downloadComplex(device, outBaseBuf, prod(packedShape));
    assertCloseArray(got, expected, 1e-3, 1e-3, "r2c ioView rank4 input");

    pIo.destroy();
    pBase.destroy();
    inViewBuf.destroy();
    inFullBuf.destroy();
    outIoBuf.destroy();
    outBaseBuf.destroy();
  });

  test("r2c ioView rank4 output mapping matches baseline", async () => {
    const device = await ensureDevice();
    const shape = [9, 3, 2, 2];
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];
    const outViewShape = [3, 3, 2, 2];
    const outOff = [1, 0, 0, 0];

    const input = new Float32Array(prod(shape));
    for (let i = 0; i < input.length; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
    const inBuf = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(inBuf, 0, input);

    const outViewBuf = device.createBuffer({
      size: prod(outViewShape) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outFullBuf = device.createBuffer({
      size: prod(packedShape) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const pIo = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { output: { shape: outViewShape, offset: outOff } },
    });
    const pBase = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
    });

    {
      const enc = device.createCommandEncoder();
      pIo.exec(enc, { input: inBuf, output: outViewBuf });
      pBase.exec(enc, { input: inBuf, output: outFullBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const full = await downloadComplex(device, outFullBuf, prod(packedShape));
    const expected = extractComplexViewFromLogical({
      logicalShape: packedShape,
      viewShape: outViewShape,
      offset: outOff,
      logicalInterleaved: full,
    });
    const got = await downloadComplex(device, outViewBuf, prod(outViewShape));
    assertCloseArray(got, expected, 1e-3, 1e-3, "r2c ioView rank4 output");

    pIo.destroy();
    pBase.destroy();
    inBuf.destroy();
    outViewBuf.destroy();
    outFullBuf.destroy();
  });

  test("c2r ioView rank4 input mapping matches baseline", async () => {
    const device = await ensureDevice();
    const shape = [9, 3, 2, 2];
    const packedShape = [Math.floor(shape[0] / 2) + 1, ...shape.slice(1)];
    const inViewShape = [3, 3, 2, 2];
    const inOff = [1, 0, 0, 0];

    const packedView = randomComplexInterleaved(prod(inViewShape));
    const packedLogical = embedComplexViewIntoLogical({
      logicalShape: packedShape,
      viewShape: inViewShape,
      offset: inOff,
      viewInterleaved: packedView,
    });

    const inViewBuf = uploadComplex(device, packedView);
    const inFullBuf = uploadComplex(device, packedLogical);
    const outIoBuf = device.createBuffer({
      size: prod(shape) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBaseBuf = device.createBuffer({
      size: prod(shape) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const pIo = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { input: { shape: inViewShape, offset: inOff } },
    });
    const pBase = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
    });

    {
      const enc = device.createCommandEncoder();
      pIo.exec(enc, { input: inViewBuf, output: outIoBuf });
      pBase.exec(enc, { input: inFullBuf, output: outBaseBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const got = await downloadF32(device, outIoBuf, prod(shape) * 4);
    const expected = await downloadF32(device, outBaseBuf, prod(shape) * 4);
    assertCloseArray(got, expected, 2e-3, 2e-3, "c2r ioView rank4 input");

    pIo.destroy();
    pBase.destroy();
    inViewBuf.destroy();
    inFullBuf.destroy();
    outIoBuf.destroy();
    outBaseBuf.destroy();
  });

  test("r2c zeroPad read+write ranges match CPU (N=17)", async () => {
    const device = await ensureDevice();
    const N = 17;
    const outLen = Math.floor(N / 2) + 1;
    const input = new Float32Array(N);
    for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;

    const inBuf = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBuf = device.createBuffer({
      size: outLen * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const readStart = [3];
    const readEnd = [14];
    const writeStart = [2];
    const writeEnd = [7];

    const plan = createPlan(device, {
      type: "r2c",
      shape: [N],
      direction: "forward",
      batch: 1,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const got = await downloadComplex(device, outBuf, outLen);
    const cpuRead = zeroOutsideRangeRealRef(input, [N], readStart, readEnd);
    const cpuPacked = r2cRefPackedInterleaved(cpuRead, N, "forward", "none");
    const expected = zeroOutsideRangeComplexRef(cpuPacked, [outLen], writeStart, writeEnd);
    assertCloseArray(got, expected, 1e-3, 1e-3, "r2c zeroPad");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2r zeroPad read+write ranges match CPU (N=17)", async () => {
    const device = await ensureDevice();
    const N = 17;
    const outLen = Math.floor(N / 2) + 1;
    const packed = randomComplexInterleaved(outLen);
    const inBuf = uploadComplex(device, packed);
    const outBuf = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const readStart = [2];
    const readEnd = [7];
    const writeStart = [3];
    const writeEnd = [14];

    const plan = createPlan(device, {
      type: "c2r",
      shape: [N],
      direction: "inverse",
      batch: 1,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const got = await downloadF32(device, outBuf, N * 4);
    const cpuRead = zeroOutsideRangeComplexRef(packed, [outLen], readStart, readEnd);
    const cpuReal = c2rRefFromPackedInterleaved(cpuRead, N, "backward");
    const expected = zeroOutsideRangeRealRef(cpuReal, [N], writeStart, writeEnd);
    assertCloseArray(got, expected, 2e-3, 2e-3, "c2r zeroPad");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("r2c with custom input strides matches CPU (N=17,batch=2)", async () => {
    const device = await ensureDevice();
    const N = 17;
    const batch = 2;
    const outLen = Math.floor(N / 2) + 1;
    const logical = new Float32Array(N * batch);
    for (let i = 0; i < logical.length; i++) logical[i] = (Math.random() * 2 - 1) * 0.5;

    const inStrides = [3];
    const inOffset = 4;
    const inBatchStride = strideSpanElements([N], inStrides) + 7;
    const inPhys = makeStridedPhysicalReal({
      shape: [N],
      batch,
      strides: inStrides,
      offsetElements: inOffset,
      batchStrideElements: inBatchStride,
      logical,
    });

    const inBuf = device.createBuffer({ size: inPhys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, inPhys);
    const outBuf = device.createBuffer({ size: outLen * batch * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "r2c",
      shape: [N],
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffset,
        inputBatchStrideElements: inBatchStride,
      },
      precision: "f32",
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, outLen * batch);
    const cpu = new Float32Array(2 * outLen * batch);
    for (let b = 0; b < batch; b++) {
      const s = logical.subarray(b * N, (b + 1) * N);
      cpu.set(r2cRefPackedInterleaved(s, N, "forward", "none"), b * outLen * 2);
    }
    assertCloseArray(gpu, cpu, 1e-3, 1e-3, "r2c input strides");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("r2c with custom packed output strides matches CPU (N=17,batch=2)", async () => {
    const device = await ensureDevice();
    const N = 17;
    const batch = 2;
    const outLen = Math.floor(N / 2) + 1;
    const logical = new Float32Array(N * batch);
    for (let i = 0; i < logical.length; i++) logical[i] = (Math.random() * 2 - 1) * 0.5;

    const inBuf = device.createBuffer({ size: logical.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, logical);

    const outStrides = [2];
    const outOffset = 5;
    const outBatchStride = strideSpanElements([outLen], outStrides) + 11;
    const outElems = requiredStridedElements([outLen], outStrides, outOffset, batch, outBatchStride);
    const outBytes = outElems * 8;
    const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "r2c",
      shape: [N],
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        outputStrides: outStrides,
        outputOffsetElements: outOffset,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outPhys = await downloadF32(device, outBuf, outBytes);
    const gpu = extractStridedLogicalComplex({
      physicalInterleaved: outPhys,
      shape: [outLen],
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });

    const cpu = new Float32Array(2 * outLen * batch);
    for (let b = 0; b < batch; b++) {
      const s = logical.subarray(b * N, (b + 1) * N);
      cpu.set(r2cRefPackedInterleaved(s, N, "forward", "none"), b * outLen * 2);
    }
    assertCloseArray(gpu, cpu, 1e-3, 1e-3, "r2c output strides");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2r with custom packed input strides matches CPU (N=17,batch=2)", async () => {
    const device = await ensureDevice();
    const N = 17;
    const batch = 2;
    const packedLen = Math.floor(N / 2) + 1;
    const logicalPacked = randomComplexInterleaved(packedLen * batch);

    const inStrides = [2];
    const inOffset = 3;
    const inBatchStride = strideSpanElements([packedLen], inStrides) + 9;
    const inPhys = makeStridedPhysicalComplex({
      shape: [packedLen],
      batch,
      strides: inStrides,
      offsetElements: inOffset,
      batchStrideElements: inBatchStride,
      logicalInterleaved: logicalPacked,
      sentinelRe: 11,
      sentinelIm: -13,
    });
    const inBuf = uploadComplex(device, inPhys);
    const outBuf = device.createBuffer({ size: N * batch * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "c2r",
      shape: [N],
      direction: "inverse",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: inStrides,
        inputOffsetElements: inOffset,
        inputBatchStrideElements: inBatchStride,
      },
      precision: "f32",
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadF32(device, outBuf, N * batch * 4);
    const cpu = new Float32Array(N * batch);
    for (let b = 0; b < batch; b++) {
      const s = logicalPacked.subarray(2 * b * packedLen, 2 * (b + 1) * packedLen);
      cpu.set(c2rRefFromPackedInterleaved(s, N, "none"), b * N);
    }
    assertCloseArray(gpu, cpu, 2e-3, 2e-3, "c2r input strides");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2r with custom real output strides matches CPU (N=17,batch=2)", async () => {
    const device = await ensureDevice();
    const N = 17;
    const batch = 2;
    const packedLen = Math.floor(N / 2) + 1;
    const logicalPacked = randomComplexInterleaved(packedLen * batch);

    const inBuf = uploadComplex(device, logicalPacked);
    const outStrides = [3];
    const outOffset = 4;
    const outBatchStride = strideSpanElements([N], outStrides) + 8;
    const outElems = requiredStridedElements([N], outStrides, outOffset, batch, outBatchStride);
    const outBytes = outElems * 4;
    const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, {
      type: "c2r",
      shape: [N],
      direction: "inverse",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        outputStrides: outStrides,
        outputOffsetElements: outOffset,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outPhys = await downloadF32(device, outBuf, outBytes);
    const gpu = extractStridedLogicalReal({
      physical: outPhys,
      shape: [N],
      batch,
      strides: outStrides,
      offsetElements: outOffset,
      batchStrideElements: outBatchStride,
    });

    const cpu = new Float32Array(N * batch);
    for (let b = 0; b < batch; b++) {
      const s = logicalPacked.subarray(2 * b * packedLen, 2 * (b + 1) * packedLen);
      cpu.set(c2rRefFromPackedInterleaved(s, N, "none"), b * N);
    }
    assertCloseArray(gpu, cpu, 2e-3, 2e-3, "c2r output strides");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  for (const type of ["dct1", "dct2", "dct3", "dct4"]) {
    test(`${type} forward matches CPU (N=16)`, async () => {
      const device = await ensureDevice();
      const N = 16;
      const input = new Float32Array(N);
      for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
      const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      device.queue.writeBuffer(inBuf, 0, input);
      const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

      const plan = createPlan(device, { type, shape: [N], direction: "forward", batch: 1, inPlace: false, normalize: "none", layout: { interleavedComplex: false }, precision: "f32" });
      const enc = device.createCommandEncoder();
      plan.exec(enc, { input: inBuf, output: outBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const gpu = await downloadF32(device, outBuf, input.byteLength);
      let cpu;
      if (type === "dct1") cpu = dct1Ref(input, N);
      else if (type === "dct2") cpu = dct2Ref(input, N, "forward");
      else if (type === "dct3") cpu = dct3Ref(input, N, "forward");
      else cpu = dct4Ref(input, N);
      assertCloseArray(gpu, cpu, 2e-3, 2e-3, `${type} fwd`);

      plan.destroy();
      inBuf.destroy();
      outBuf.destroy();
    });
  }

  test("dct2 inverse matches CPU (N=16)", async () => {
    const device = await ensureDevice();
    const N = 16;
    const input = new Float32Array(N);
    for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
    const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, { type: "dct2", shape: [N], direction: "inverse", batch: 1, normalize: "none", layout: { interleavedComplex: false }, precision: "f32" });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadF32(device, outBuf, input.byteLength);
    const cpu = dct2Ref(input, N, "inverse");
    assertCloseArray(gpu, cpu, 2e-3, 2e-3, "dct2 inv");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("dct2 zeroPad read+write ranges match CPU (N=16)", async () => {
    const device = await ensureDevice();
    const N = 16;
    const input = new Float32Array(N);
    for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
    const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const readStart = [4];
    const readEnd = [12];
    const writeStart = [2];
    const writeEnd = [14];

    const plan = createPlan(device, {
      type: "dct2",
      shape: [N],
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: false },
      precision: "f32",
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadF32(device, outBuf, input.byteLength);
    const cpuRead = zeroOutsideRangeRealRef(input, [N], readStart, readEnd);
    const cpuDct = dct2Ref(cpuRead, N, "forward");
    const expected = zeroOutsideRangeRealRef(cpuDct, [N], writeStart, writeEnd);
    assertCloseArray(gpu, expected, 2e-3, 2e-3, "dct2 zeroPad");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("dct2 ioView rank4 input+output mapping matches baseline", async () => {
    const device = await ensureDevice();
    const shape = [4, 3, 2, 2];
    const inViewShape = [2, 3, 2, 2];
    const inOff = [1, 0, 0, 0];
    const outViewShape = [2, 3, 2, 2];
    const outOff = [1, 0, 0, 0];

    const inputView = new Float32Array(prod(inViewShape));
    for (let i = 0; i < inputView.length; i++) inputView[i] = (Math.random() * 2 - 1) * 0.5;
    const inputLogical = embedRealViewIntoLogical({
      logicalShape: shape,
      viewShape: inViewShape,
      offset: inOff,
      view: inputView,
    });

    const inViewBuf = device.createBuffer({
      size: inputView.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(inViewBuf, 0, inputView);
    const inFullBuf = device.createBuffer({
      size: inputLogical.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(inFullBuf, 0, inputLogical);

    const outViewBuf = device.createBuffer({
      size: prod(outViewShape) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outFullBuf = device.createBuffer({
      size: prod(shape) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const pIo = createPlan(device, {
      type: "dct2",
      shape,
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: false },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inOff },
        output: { shape: outViewShape, offset: outOff },
      },
    });
    const pBase = createPlan(device, {
      type: "dct2",
      shape,
      direction: "forward",
      batch: 1,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: false },
      precision: "f32",
    });

    {
      const enc = device.createCommandEncoder();
      pIo.exec(enc, { input: inViewBuf, output: outViewBuf });
      pBase.exec(enc, { input: inFullBuf, output: outFullBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const full = await downloadF32(device, outFullBuf, prod(shape) * 4);
    const expectedView = extractRealViewFromLogical({
      logicalShape: shape,
      viewShape: outViewShape,
      offset: outOff,
      logical: full,
    });
    const got = await downloadF32(device, outViewBuf, prod(outViewShape) * 4);
    assertCloseArray(got, expectedView, 2e-3, 2e-3, "dct2 ioView rank4");

    pIo.destroy();
    pBase.destroy();
    inViewBuf.destroy();
    inFullBuf.destroy();
    outViewBuf.destroy();
    outFullBuf.destroy();
  });

  test("dct2 regular path matches baseline with custom strides + ioView+zeroPad (rank-3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const inViewShape = [3, 3, 2];
    const inViewOffset = [1, 0, 0];
    const outViewShape = [7, 3, 2];
    const outViewOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [5, 3, 2];
    const writeStart = [0, 1, 0];
    const writeEnd = [5, 3, 2];

    const inViewLogical = new Float32Array(prod(inViewShape) * batch);
    for (let i = 0; i < inViewLogical.length; i++) inViewLogical[i] = (Math.random() * 2 - 1) * 0.5;

    const inStrides = [2, 9, 30];
    const inOffsetElements = 3;
    const inBatchStride = strideSpanElements(inViewShape, inStrides) + 17;
    const inPhys = makeStridedPhysicalReal({
      shape: inViewShape,
      batch,
      strides: inStrides,
      offsetElements: inOffsetElements,
      batchStrideElements: inBatchStride,
      logical: inViewLogical,
    });
    const inStridedBuf = device.createBuffer({ size: inPhys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const inBaseBuf = device.createBuffer({ size: inViewLogical.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inStridedBuf, 0, inPhys);
    device.queue.writeBuffer(inBaseBuf, 0, inViewLogical);

    const outStrides = [3, 14, 44];
    const outOffsetElements = 5;
    const outBatchStride = strideSpanElements(outViewShape, outStrides) + 19;
    const outElems = requiredStridedElements(outViewShape, outStrides, outOffsetElements, batch, outBatchStride);
    const outStridedBytes = outElems * 4;
    const outBaseBytes = prod(outViewShape) * batch * 4;
    const outStridedBuf = device.createBuffer({ size: outStridedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBaseBuf = device.createBuffer({ size: outBaseBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const stridedPlan = createPlan(device, {
      type: "dct2",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: {
        interleavedComplex: false,
        inputStrides: inStrides,
        inputOffsetElements: inOffsetElements,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffsetElements,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: true },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    const baselinePlan = createPlan(device, {
      type: "dct2",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: false },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: true },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });

    {
      const enc = device.createCommandEncoder();
      stridedPlan.exec(enc, { input: inStridedBuf, output: outStridedBuf });
      baselinePlan.exec(enc, { input: inBaseBuf, output: outBaseBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const outPhys = await downloadF32(device, outStridedBuf, outStridedBytes);
    const got = extractStridedLogicalReal({
      physical: outPhys,
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
    });
    const expected = await downloadF32(device, outBaseBuf, outBaseBytes);
    assertCloseArray(got, expected, 2e-3, 2e-3, "dct2 regular custom strides + ioView+zeroPad");

    stridedPlan.destroy();
    baselinePlan.destroy();
    inStridedBuf.destroy();
    inBaseBuf.destroy();
    outStridedBuf.destroy();
    outBaseBuf.destroy();
  });

  test("dct2 regular path supports custom-strided output with ioView.clearOutside=false", async () => {
    const device = await ensureDevice();
    const shape = [5, 3, 2];
    const batch = 2;
    const inViewShape = [3, 3, 2];
    const inViewOffset = [1, 0, 0];
    const outViewShape = [7, 3, 2];
    const outViewOffset = [-1, 0, 0];
    const readStart = [1, 0, 0];
    const readEnd = [5, 3, 2];
    const writeStart = [0, 1, 0];
    const writeEnd = [5, 3, 2];

    const inViewLogical = new Float32Array(prod(inViewShape) * batch);
    for (let i = 0; i < inViewLogical.length; i++) inViewLogical[i] = (Math.random() * 2 - 1) * 0.5;

    const inStrides = [2, 9, 30];
    const inOffsetElements = 3;
    const inBatchStride = strideSpanElements(inViewShape, inStrides) + 17;
    const inPhys = makeStridedPhysicalReal({
      shape: inViewShape,
      batch,
      strides: inStrides,
      offsetElements: inOffsetElements,
      batchStrideElements: inBatchStride,
      logical: inViewLogical,
    });
    const inStridedBuf = device.createBuffer({ size: inPhys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const inBaseBuf = device.createBuffer({ size: inViewLogical.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inStridedBuf, 0, inPhys);
    device.queue.writeBuffer(inBaseBuf, 0, inViewLogical);

    const outStrides = [3, 14, 44];
    const outOffsetElements = 5;
    const outBatchStride = strideSpanElements(outViewShape, outStrides) + 19;
    const outElems = requiredStridedElements(outViewShape, outStrides, outOffsetElements, batch, outBatchStride);
    const outStridedBytes = outElems * 4;
    const outBaseBytes = prod(outViewShape) * batch * 4;
    const outStridedBuf = device.createBuffer({ size: outStridedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const outBaseBuf = device.createBuffer({ size: outBaseBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const outInitLogical = new Float32Array(prod(outViewShape) * batch);
    for (let i = 0; i < outInitLogical.length; i++) outInitLogical[i] = -0.45 + i * 1e-4;
    const outInitStrided = makeStridedPhysicalReal({
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
      logical: outInitLogical,
    });
    device.queue.writeBuffer(outStridedBuf, 0, outInitStrided);
    device.queue.writeBuffer(outBaseBuf, 0, outInitLogical);

    const stridedPlan = createPlan(device, {
      type: "dct2",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: {
        interleavedComplex: false,
        inputStrides: inStrides,
        inputOffsetElements: inOffsetElements,
        inputBatchStrideElements: inBatchStride,
        outputStrides: outStrides,
        outputOffsetElements: outOffsetElements,
        outputBatchStrideElements: outBatchStride,
      },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });
    const baselinePlan = createPlan(device, {
      type: "dct2",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: false },
      precision: "f32",
      ioView: {
        input: { shape: inViewShape, offset: inViewOffset },
        output: { shape: outViewShape, offset: outViewOffset, clearOutside: false },
      },
      zeroPad: {
        read: { start: readStart, end: readEnd },
        write: { start: writeStart, end: writeEnd },
      },
    });

    {
      const enc = device.createCommandEncoder();
      stridedPlan.exec(enc, { input: inStridedBuf, output: outStridedBuf });
      baselinePlan.exec(enc, { input: inBaseBuf, output: outBaseBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const outPhys = await downloadF32(device, outStridedBuf, outStridedBytes);
    const got = extractStridedLogicalReal({
      physical: outPhys,
      shape: outViewShape,
      batch,
      strides: outStrides,
      offsetElements: outOffsetElements,
      batchStrideElements: outBatchStride,
    });
    const expected = await downloadF32(device, outBaseBuf, outBaseBytes);
    assertCloseArray(got, expected, 2e-3, 2e-3, "dct2 custom-strided output clearOutside=false");

    stridedPlan.destroy();
    baselinePlan.destroy();
    inStridedBuf.destroy();
    inBaseBuf.destroy();
    outStridedBuf.destroy();
    outBaseBuf.destroy();
  });

  for (const type of ["dst1", "dst2", "dst3", "dst4"]) {
    test(`${type} forward matches CPU (N=16)`, async () => {
      const device = await ensureDevice();
      const N = 16;
      const input = new Float32Array(N);
      for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
      const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      device.queue.writeBuffer(inBuf, 0, input);
      const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

      const plan = createPlan(device, { type, shape: [N], direction: "forward", batch: 1, inPlace: false, normalize: "none", layout: { interleavedComplex: false }, precision: "f32" });
      const enc = device.createCommandEncoder();
      plan.exec(enc, { input: inBuf, output: outBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const gpu = await downloadF32(device, outBuf, input.byteLength);
      let cpu;
      if (type === "dst1") cpu = dst1Ref(input, N);
      else if (type === "dst2") cpu = dst2Ref(input, N, "forward");
      else if (type === "dst3") cpu = dst3Ref(input, N, "forward");
      else cpu = dst4Ref(input, N);
      assertCloseArray(gpu, cpu, 2e-3, 2e-3, `${type} fwd`);

      plan.destroy();
      inBuf.destroy();
      outBuf.destroy();
    });
  }

  test("dst2 inverse matches CPU (N=16)", async () => {
    const device = await ensureDevice();
    const N = 16;
    const input = new Float32Array(N);
    for (let i = 0; i < N; i++) input[i] = (Math.random() * 2 - 1) * 0.5;
    const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, { type: "dst2", shape: [N], direction: "inverse", batch: 1, normalize: "none", layout: { interleavedComplex: false }, precision: "f32" });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadF32(device, outBuf, input.byteLength);
    const cpu = dst2Ref(input, N, "inverse");
    assertCloseArray(gpu, cpu, 2e-3, 2e-3, "dst2 inv");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c f16-storage forward matches CPU (N=64) when shader-f16 available", async () => {
    const device = await ensureDevice();
    if (!device.features?.has?.("shader-f16")) throw new SkipError("shader-f16 unavailable");
    const N = 64;
    const inputF32 = randomComplexInterleaved(N);
    const inputF16 = new Uint16Array(2 * N);
    for (let i = 0; i < 2 * N; i++) inputF16[i] = f32ToF16Bits(inputF32[i]);
    const inBuf = device.createBuffer({ size: inputF16.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(inBuf, 0, inputF16);
    const outBuf = device.createBuffer({ size: inputF16.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const plan = createPlan(device, { type: "c2c", shape: [N], direction: "forward", batch: 1, inPlace: false, normalize: "none", layout: { interleavedComplex: true }, precision: "f16-storage" });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const outU16 = await downloadU16(device, outBuf, inputF16.byteLength);
    const outF32 = new Float32Array(2 * N);
    for (let i = 0; i < 2 * N; i++) outF32[i] = f16BitsToF32(outU16[i]);

    const cpu = fftNdRefAnySizeInterleaved(inputF32, [N], "forward", "none");
    assertCloseArray(outF32, cpu, 3e-2, 3e-2, "c2c f16-storage");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("c2c f16-storage with ioView input+output works (clearOutside=false preserves output)", async () => {
    const device = await ensureDevice();
    if (!device.features?.has?.("shader-f16")) throw new SkipError("shader-f16 unavailable");

    const logical = 16;
    const inputN = 8;
    const viewOut = 32;
    const inOff = Math.floor((logical - inputN) / 2);
    const outOff = Math.floor((logical - viewOut) / 2);

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

    const outU16 = await downloadU16(device, outBuf, outBytes);
    const outF32 = new Float32Array(2 * viewOut);
    for (let i = 0; i < 2 * viewOut; i++) outF32[i] = f16BitsToF32(outU16[i]);

    const logicalIn = new Float32Array(2 * logical);
    logicalIn.set(inputViewF32, 2 * inOff);
    const cpuLogical = fftNdRefAnySizeInterleaved(logicalIn, [logical], "forward", "none");

    for (let i = 0; i < viewOut; i++) {
      const l = i + outOff;
      if (l >= 0 && l < logical) continue;
      assertEq(outF32[2 * i], f16BitsToF32(sRe), "f16 clearOutside=false (re)");
      assertEq(outF32[2 * i + 1], f16BitsToF32(sIm), "f16 clearOutside=false (im)");
    }
    const mappedStart = -outOff;
    const mapped = outF32.subarray(2 * mappedStart, 2 * (mappedStart + logical));
    assertCloseArray(mapped, cpuLogical, 3e-2, 3e-2, "c2c f16 ioView mapped");

    if (exportArtifact) {
      exportArtifact({
        name: "c2c_f16_ioview_in8_out32_center",
        plan: {
          type: "c2c",
          shape: [logical],
          direction: "forward",
          batch: 1,
          inPlace: false,
          normalize: "none",
          layout: { interleavedComplex: true },
          precision: "f16-storage",
          ioView: { input: { shape: [inputN], placement: "center" }, output: { shape: [viewOut], placement: "center", clearOutside: false } },
        },
        input: { kind: "complex-f16-u16", data: Array.from(inputViewF16), meta: { inputViewF32: Array.from(inputViewF32) } },
        cpu: { kind: "complex-f32", data: Array.from(cpuLogical) },
        gpu: { kind: "complex-f32", data: Array.from(mapped) },
        tol: { atol: 3e-2, rtol: 3e-2 },
      });
    }

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv 1D convolution matches CPU (N=17)", async () => {
    const device = await ensureDevice();
    const N = 17;
    const input = randomComplexInterleaved(N);
    const kernel = randomComplexInterleaved(N);
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: N * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape: [N],
      batch: 1,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "convolution" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, N);
    const cpu = fftConvRef({ input, kernel, shape: [N], batch: 1, mode: "convolution" });
    assertCloseArray(gpu, cpu, 2e-3, 2e-3, "fftconv 1D convolution");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv 1D correlation matches CPU (N=29,batch=2)", async () => {
    const device = await ensureDevice();
    const N = 29;
    const batch = 2;
    const input = randomComplexInterleaved(N * batch);
    const kernel = randomComplexInterleaved(N);
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: N * batch * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const kernelBuf = uploadComplex(device, kernel);

    const plan = createPlan(device, {
      type: "fftconv",
      shape: [N],
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "correlation" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernelBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, N * batch);
    const cpu = fftConvRef({ input, kernel, shape: [N], batch, mode: "correlation" });
    assertCloseArray(gpu, cpu, 3e-3, 3e-3, "fftconv 1D correlation");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
    kernelBuf.destroy();
  });

  test("fftconv 2D convolution batched matches CPU (8x9,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [8, 9];
    const batch = 2;
    const n = prod(shape);
    const input = randomComplexInterleaved(n * batch);
    const kernel = randomComplexInterleaved(n);
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: n * batch * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "convolution" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, n * batch);
    const cpu = fftConvRef({ input, kernel, shape, batch, mode: "convolution" });
    assertCloseArray(gpu, cpu, 3e-3, 3e-3, "fftconv 2D convolution");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv 3D convolution batched matches CPU (4x5x3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [4, 5, 3];
    const batch = 2;
    const n = prod(shape);
    const input = randomComplexInterleaved(n * batch);
    const kernel = randomComplexInterleaved(n);
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: n * batch * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "convolution" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, n * batch);
    const cpu = fftConvRef({ input, kernel, shape, batch, mode: "convolution" });
    assertCloseArray(gpu, cpu, 4e-3, 4e-3, "fftconv 3D convolution");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv multi-kernel kernel-major output matches CPU (N=21,batch=2,kernels=3)", async () => {
    const device = await ensureDevice();
    const shape = [21];
    const batch = 2;
    const kernelCount = 3;
    const n = prod(shape);
    const input = randomComplexInterleaved(n * batch);
    const kernels = Array.from({ length: kernelCount }, () => randomComplexInterleaved(n));
    const packedKernels = new Float32Array(2 * n * kernelCount);
    for (let k = 0; k < kernelCount; k++) packedKernels.set(kernels[k], 2 * k * n);

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: n * batch * kernelCount * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "convolution", kernelCount, outputLayout: "kernel-major" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: packedKernels });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, n * batch * kernelCount);
    const expected = new Float32Array(gpu.length);
    for (let k = 0; k < kernelCount; k++) {
      const cpu = fftConvRef({ input, kernel: kernels[k], shape, batch, mode: "convolution" });
      expected.set(cpu, 2 * k * n * batch);
    }
    assertCloseArray(gpu, expected, 4e-3, 4e-3, "fftconv multi-kernel kernel-major");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv multi-kernel batch-major output matches CPU (N=19,batch=2,kernels=2)", async () => {
    const device = await ensureDevice();
    const shape = [19];
    const batch = 2;
    const kernelCount = 2;
    const n = prod(shape);
    const input = randomComplexInterleaved(n * batch);
    const kernels = Array.from({ length: kernelCount }, () => randomComplexInterleaved(n));

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: n * batch * kernelCount * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "correlation", kernelCount, outputLayout: "batch-major" },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernels });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, n * batch * kernelCount);
    const expected = new Float32Array(gpu.length);
    for (let b = 0; b < batch; b++) {
      for (let k = 0; k < kernelCount; k++) {
        const cpu = fftConvRef({ input, kernel: kernels[k], shape, batch, mode: "correlation" });
        const src = cpu.subarray(2 * b * n, 2 * (b + 1) * n);
        const dstOffset = 2 * (b * kernelCount + k) * n;
        expected.set(src, dstOffset);
      }
    }
    assertCloseArray(gpu, expected, 4e-3, 4e-3, "fftconv multi-kernel batch-major");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv channelPolicy maps multi-kernel outputs into channel lanes (N=12,batch=2,kernels=2)", async () => {
    const device = await ensureDevice();
    const shape = [12];
    const batch = 2;
    const kernelCount = 2;
    const n = prod(shape);

    const inputChannels = 3;
    const inputChannelIndex = 1;
    const inputChannelStride = 16;
    const inputBatchStride = 80;
    const inputOffsetElements = inputChannelIndex * inputChannelStride;

    const outputChannels = 5;
    const outputChannelIndex = 1;
    const outputChannelStride = 20;
    const outputBatchStride = 160;
    const kernelStepChannels = 1;
    const outputLastOffsetElements =
      (outputChannelIndex + (kernelCount - 1) * kernelStepChannels) * outputChannelStride;
    const outputElems = requiredStridedElements(shape, [1], outputLastOffsetElements, batch, outputBatchStride);
    const outputBytes = outputElems * 8;

    const logicalInput = randomComplexInterleaved(n * batch);
    const inputPhys = makeStridedPhysicalComplex({
      shape,
      batch,
      strides: [1],
      offsetElements: inputOffsetElements,
      batchStrideElements: inputBatchStride,
      logicalInterleaved: logicalInput,
    });
    const kernels = Array.from({ length: kernelCount }, () => randomComplexInterleaved(n));

    const inBuf = uploadComplex(device, inputPhys);
    const outBuf = device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outSentinel = new Float32Array(2 * outputElems);
    for (let i = 0; i < outputElems; i++) {
      outSentinel[2 * i] = 77.0;
      outSentinel[2 * i + 1] = -55.0;
    }
    device.queue.writeBuffer(outBuf, 0, outSentinel);

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: {
        mode: "convolution",
        kernelCount,
        outputLayout: "batch-major",
        channelPolicy: {
          input: {
            channels: inputChannels,
            channelIndex: inputChannelIndex,
            channelStrideElements: inputChannelStride,
            batchStrideElements: inputBatchStride,
          },
          output: {
            channels: outputChannels,
            channelIndex: outputChannelIndex,
            channelStrideElements: outputChannelStride,
            batchStrideElements: outputBatchStride,
            kernelStepChannels,
          },
        },
      },
    });
    if (plan._usesStridedInput !== true) throw new Error("expected strided input policy routing");
    if (plan._usesStridedOutput !== true) throw new Error("expected strided output policy routing");
    if (plan._stridedOutputKernelStrideElements !== outputChannelStride * kernelStepChannels) {
      throw new Error(
        `expected output kernel stride ${outputChannelStride * kernelStepChannels}, got ${plan._stridedOutputKernelStrideElements}`
      );
    }
    if (plan._outputOffsetElements !== outputChannelIndex * outputChannelStride) {
      throw new Error(
        `expected output offset ${outputChannelIndex * outputChannelStride}, got ${plan._outputOffsetElements}`
      );
    }
    if (plan._outputBatchStrideElements !== outputBatchStride) {
      throw new Error(`expected output batch stride ${outputBatchStride}, got ${plan._outputBatchStrideElements}`);
    }
    if (!Array.isArray(plan._outputStrides) || plan._outputStrides[0] !== 1) {
      throw new Error(`unexpected output strides ${JSON.stringify(plan._outputStrides)}`);
    }
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernels });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpuPhys = await downloadF32(device, outBuf, outputBytes);
    const expectedPhys = new Float32Array(outSentinel);
    for (let k = 0; k < kernelCount; k++) {
      const cpu = fftConvRef({
        input: logicalInput,
        kernel: kernels[k],
        shape,
        batch,
        mode: "convolution",
      });
      for (let b = 0; b < batch; b++) {
        const laneOffset = (outputChannelIndex + k * kernelStepChannels) * outputChannelStride + b * outputBatchStride;
        for (let i = 0; i < n; i++) {
          const src = 2 * (b * n + i);
          const dst = 2 * (laneOffset + i);
          expectedPhys[dst] = cpu[src];
          expectedPhys[dst + 1] = cpu[src + 1];
        }
      }
    }
    assertCloseArray(gpuPhys, expectedPhys, 5e-3, 5e-3, "fftconv channelPolicy multi-kernel lanes");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv linear-full convolution matches CPU (signal=17,kernel=7)", async () => {
    const device = await ensureDevice();
    const shape = [17];
    const kernelShape = [7];
    const outShape = [shape[0] + kernelShape[0] - 1];
    const nIn = prod(shape);
    const nOut = prod(outShape);
    const input = randomComplexInterleaved(nIn);
    const kernel = randomComplexInterleaved(prod(kernelShape));
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: nOut * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch: 1,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "convolution", boundary: "linear-full", kernelShape },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, nOut);
    const cpu = fftConvRef({ input, kernel, shape, kernelShape, batch: 1, mode: "convolution", boundary: "linear-full" });
    assertCloseArray(gpu, cpu, 4e-3, 4e-3, "fftconv linear-full");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv linear-same correlation matches CPU (8x7 kernel 3x2 batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [8, 7];
    const kernelShape = [3, 2];
    const batch = 2;
    const nIn = prod(shape);
    const nOut = nIn;
    const input = randomComplexInterleaved(nIn * batch);
    const kernel = randomComplexInterleaved(prod(kernelShape));
    const inBuf = uploadComplex(device, input);
    const kernelBuf = uploadComplex(device, kernel);
    const outBuf = device.createBuffer({
      size: nOut * batch * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "correlation", boundary: "linear-same", kernelShape },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernelBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, nOut * batch);
    const cpu = fftConvRef({ input, kernel, shape, kernelShape, batch, mode: "correlation", boundary: "linear-same" });
    assertCloseArray(gpu, cpu, 5e-3, 5e-3, "fftconv linear-same");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
    kernelBuf.destroy();
  });

  test("fftconv linear-valid convolution matches CPU (N=18,k=5,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [18];
    const kernelShape = [5];
    const batch = 2;
    const nIn = prod(shape);
    const nOut = shape[0] - kernelShape[0] + 1;
    const input = randomComplexInterleaved(nIn * batch);
    const kernel = randomComplexInterleaved(prod(kernelShape));
    const inBuf = uploadComplex(device, input);
    const kernelBuf = uploadComplex(device, kernel);
    const outBuf = device.createBuffer({
      size: nOut * batch * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: { mode: "convolution", boundary: "linear-valid", kernelShape },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernelBuf });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, nOut * batch);
    const cpu = fftConvRef({ input, kernel, shape, kernelShape, batch, mode: "convolution", boundary: "linear-valid" });
    assertCloseArray(gpu, cpu, 5e-3, 5e-3, "fftconv linear-valid");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
    kernelBuf.destroy();
  });

  test("fftconv linear-valid multi-kernel kernel-major matches CPU (N=18,k=5,batch=2,kernels=2)", async () => {
    const device = await ensureDevice();
    const shape = [18];
    const kernelShape = [5];
    const batch = 2;
    const kernelCount = 2;
    const nIn = prod(shape);
    const nOut = shape[0] - kernelShape[0] + 1;
    const input = randomComplexInterleaved(nIn * batch);
    const kernels = Array.from({ length: kernelCount }, () => randomComplexInterleaved(prod(kernelShape)));
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: nOut * batch * kernelCount * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: {
        mode: "convolution",
        boundary: "linear-valid",
        kernelShape,
        kernelCount,
        outputLayout: "kernel-major",
      },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernels });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, nOut * batch * kernelCount);
    const expected = new Float32Array(gpu.length);
    for (let k = 0; k < kernelCount; k++) {
      const cpu = fftConvRef({
        input,
        kernel: kernels[k],
        shape,
        kernelShape,
        batch,
        mode: "convolution",
        boundary: "linear-valid",
      });
      expected.set(cpu, 2 * k * nOut * batch);
    }
    assertCloseArray(gpu, expected, 5e-3, 5e-3, "fftconv linear-valid kernel-major");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv linear-valid multi-kernel batch-major matches CPU (N=18,k=5,batch=2,kernels=2)", async () => {
    const device = await ensureDevice();
    const shape = [18];
    const kernelShape = [5];
    const batch = 2;
    const kernelCount = 2;
    const nIn = prod(shape);
    const nOut = shape[0] - kernelShape[0] + 1;
    const input = randomComplexInterleaved(nIn * batch);
    const kernels = Array.from({ length: kernelCount }, () => randomComplexInterleaved(prod(kernelShape)));
    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: nOut * batch * kernelCount * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: {
        mode: "convolution",
        boundary: "linear-valid",
        kernelShape,
        kernelCount,
        outputLayout: "batch-major",
      },
    });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernels });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, nOut * batch * kernelCount);
    const expected = new Float32Array(gpu.length);
    for (let b = 0; b < batch; b++) {
      for (let k = 0; k < kernelCount; k++) {
        const cpu = fftConvRef({
          input,
          kernel: kernels[k],
          shape,
          kernelShape,
          batch,
          mode: "convolution",
          boundary: "linear-valid",
        });
        const src = cpu.subarray(2 * b * nOut, 2 * (b + 1) * nOut);
        const dstOffset = 2 * (b * kernelCount + k) * nOut;
        expected.set(src, dstOffset);
      }
    }
    assertCloseArray(gpu, expected, 5e-3, 5e-3, "fftconv linear-valid batch-major");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("fftconv zeroPad read+write ranges match CPU across boundary modes (N=8,k=3,batch=2)", async () => {
    const device = await ensureDevice();
    const shape = [8];
    const kernelShape = [3];
    const batch = 2;
    const inputTotal = prod(shape);
    const input = randomComplexInterleaved(inputTotal * batch);
    const kernel = randomComplexInterleaved(prod(kernelShape));
    const inBuf = uploadComplex(device, input);
    const kernelBuf = uploadComplex(device, kernel);

    const boundaries = ["circular", "linear-full", "linear-same", "linear-valid"];
    for (const boundary of boundaries) {
      const { fftShape, outputShape } = fftConvOutputSpec(shape, kernelShape, boundary);
      const zeroPad = {
        read: { start: [1], end: [fftShape[0] - 1] },
        write: { start: [2], end: [fftShape[0] - 1] },
      };
      const outBuf = device.createBuffer({
        size: prod(outputShape) * batch * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const plan = createPlan(device, {
        type: "fftconv",
        shape,
        batch,
        layout: { interleavedComplex: true },
        precision: "f32",
        zeroPad,
        fftConv: { mode: "convolution", boundary, kernelShape, kernelCount: 1, outputLayout: "kernel-major" },
      });

      const enc = device.createCommandEncoder();
      plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernelBuf });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const gpu = await downloadComplex(device, outBuf, prod(outputShape) * batch);
      const cpu = fftConvRefWithZeroPad({
        input,
        kernel,
        shape,
        kernelShape,
        batch,
        mode: "convolution",
        boundary,
        zeroPad,
      });
      assertCloseArray(gpu, cpu, 5e-3, 5e-3, `fftconv zeroPad ${boundary}`);

      plan.destroy();
      outBuf.destroy();
    }

    inBuf.destroy();
    kernelBuf.destroy();
  });

  test("fftconv zeroPad read+write ranges match CPU for linear-valid multi-kernel batch-major (N=18,k=5,batch=2,kernels=2)", async () => {
    const device = await ensureDevice();
    const shape = [18];
    const kernelShape = [5];
    const batch = 2;
    const kernelCount = 2;
    const inputTotal = prod(shape);
    const outputTotal = shape[0] - kernelShape[0] + 1;
    const fftShape = [shape[0] + kernelShape[0] - 1];
    const input = randomComplexInterleaved(inputTotal * batch);
    const kernels = Array.from({ length: kernelCount }, () => randomComplexInterleaved(prod(kernelShape)));
    const zeroPad = {
      read: { start: [1], end: [fftShape[0] - 1] },
      write: { start: [2], end: [fftShape[0] - 2] },
    };

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({
      size: outputTotal * batch * kernelCount * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      layout: { interleavedComplex: true },
      precision: "f32",
      zeroPad,
      fftConv: {
        mode: "convolution",
        boundary: "linear-valid",
        kernelShape,
        kernelCount,
        outputLayout: "batch-major",
      },
    });

    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel: kernels });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadComplex(device, outBuf, outputTotal * batch * kernelCount);
    const expected = new Float32Array(gpu.length);
    for (let b = 0; b < batch; b++) {
      for (let k = 0; k < kernelCount; k++) {
        const cpu = fftConvRefWithZeroPad({
          input,
          kernel: kernels[k],
          shape,
          kernelShape,
          batch,
          mode: "convolution",
          boundary: "linear-valid",
          zeroPad,
        });
        const src = cpu.subarray(2 * b * outputTotal, 2 * (b + 1) * outputTotal);
        const dstOffset = 2 * (b * kernelCount + k) * outputTotal;
        expected.set(src, dstOffset);
      }
    }
    assertCloseArray(gpu, expected, 5e-3, 5e-3, "fftconv zeroPad linear-valid batch-major");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("conv2d 3x3 same padding matches CPU (complex)", async () => {
    const device = await ensureDevice();
    const H = 16, W = 17, k = 3;
    const input = randomComplexInterleaved(H * W);
    const kernel = new Float32Array(k * k);
    for (let i = 0; i < kernel.length; i++) kernel[i] = (Math.random() * 2 - 1) * 0.5;

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const plan = createPlan(device, { type: "conv2d", shape: [H, W], batch: 1, layout: { interleavedComplex: true }, precision: "f32", conv: { kernelSize: 3, kernelType: "real", padding: "same", boundary: "zero" } });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const out = await downloadComplex(device, outBuf, H * W);

    const cpu = conv2dRef({ input, kernel, Hout: H, Wout: W, Hin: H, Win: W, k, pad: [1, 1, 1, 1], complex: true, complexKernel: false });
    assertCloseArray(out, cpu, 3e-3, 3e-3, "conv2d 3x3 complex");

    if (exportArtifact) {
      exportArtifact({
        name: "conv2d_complex_3x3_same_16x17",
        plan: { type: "conv2d", shape: [H, W], batch: 1, layout: { interleavedComplex: true }, precision: "f32", conv: { kernelSize: 3, kernelType: "real", padding: "same", boundary: "zero" } },
        input: { kind: "complex-f32", data: Array.from(input) },
        kernel: { kind: "real-f32", data: Array.from(kernel) },
        cpu: { kind: "complex-f32", data: Array.from(cpu) },
        gpu: { kind: "complex-f32", data: Array.from(out) },
        tol: { atol: 3e-3, rtol: 3e-3 },
      });
    }

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("conv2d 2x2 valid padding matches CPU (real)", async () => {
    const device = await ensureDevice();
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

    const plan = createPlan(device, { type: "conv2d", shape: [Hout, Wout], batch: 1, layout: { interleavedComplex: false }, precision: "f32", conv: { kernelSize: 2, kernelType: "real", padding: "valid", boundary: "zero" } });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = await downloadF32(device, outBuf, Hout * Wout * 4);
    const cpu = conv2dRef({ input, kernel, Hout, Wout, Hin, Win, k, pad: [0, 0, 0, 0], complex: false, complexKernel: false });
    assertCloseArray(gpu, cpu, 3e-3, 3e-3, "conv2d 2x2 real");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });

  test("conv2d 1x1 same padding matches CPU (complex kernel)", async () => {
    const device = await ensureDevice();
    const H = 9, W = 7, k = 1;
    const input = randomComplexInterleaved(H * W);
    const kernel = new Float32Array(2 * k * k);
    kernel[0] = 0.5;
    kernel[1] = -0.25;

    const inBuf = uploadComplex(device, input);
    const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const plan = createPlan(device, { type: "conv2d", shape: [H, W], batch: 1, layout: { interleavedComplex: true }, precision: "f32", conv: { kernelSize: 1, kernelType: "complex", padding: "same", boundary: "zero" } });
    const enc = device.createCommandEncoder();
    plan.exec(enc, { input: inBuf, output: outBuf, kernel });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const out = await downloadComplex(device, outBuf, H * W);

    const cpu = conv2dRef({ input, kernel, Hout: H, Wout: W, Hin: H, Win: W, k, pad: [0, 0, 0, 0], complex: true, complexKernel: true });
    assertCloseArray(out, cpu, 3e-3, 3e-3, "conv2d 1x1 complex kernel");

    plan.destroy();
    inBuf.destroy();
    outBuf.destroy();
  });
}

export function compareGoldenVectors(ctx) {
  const { test, getDevice, assertCloseArray, SkipError, golden, log = defaultLog } = ctx;
  const prod = (arr) => arr.reduce((a, b) => a * b, 1);
  const rankOf = (shape) => (Array.isArray(shape) ? shape.length : 0);
  const computeOffset1d = (logicalN, viewN, spec) => {
    if (!spec) return 0;
    if (spec.offset && Array.isArray(spec.offset)) return spec.offset[0] | 0;
    const placement = spec.placement ?? "start";
    if (placement === "center") return Math.floor((logicalN - viewN) / 2);
    return 0;
  };
  const ensureDevice = async () => {
    const device = await getDevice?.();
    if (!device) throw new SkipError("WebGPU unavailable");
    return device;
  };

  test("golden JSON schema supported", async () => {
    if (!golden || golden.schema !== "webgpufft-golden") {
      throw new Error(`Unsupported golden schema: ${golden?.schema}`);
    }
  });

  const artifacts = Array.isArray(golden?.artifacts) ? golden.artifacts : [];
  for (const a of artifacts) {
    test(`golden: ${a.name}`, async () => {
      const device = await ensureDevice();
      const planOpts = a.plan;
      if (!planOpts?.type) throw new Error("golden artifact missing plan.type");
      if (planOpts.precision === "f16-storage" && !device.features?.has?.("shader-f16")) {
        throw new SkipError("shader-f16 unavailable");
      }

      if (planOpts.type === "c2c") {
        const plan = createPlan(device, planOpts);
        const rank = rankOf(planOpts.shape);
        if (rank !== 1) {
          plan.destroy();
          throw new SkipError(`golden c2c compare supports rank=1 only (got rank=${rank})`);
        }

        const logicalN = planOpts.shape[0] | 0;
        const batch = planOpts.batch ?? 1;
        const precision = planOpts.precision ?? "f32";
        const bytesPerComplex = precision === "f16-storage" ? 4 : 8;

        const inView = planOpts.ioView?.input?.shape?.[0] ?? logicalN;
        const outView = planOpts.ioView?.output?.shape?.[0] ?? logicalN;
        const inComplex = inView * batch;
        const outComplex = outView * batch;

        let inputBuf = null;
        let outputBuf = null;

        if (a.input.kind === "complex-f16-u16") {
          const u16 = new Uint16Array(a.input.data);
          if (u16.length !== 2 * inComplex) throw new Error(`golden input length mismatch: have ${u16.length} expected ${2 * inComplex}`);
          inputBuf = device.createBuffer({ size: u16.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
          device.queue.writeBuffer(inputBuf, 0, u16);
        } else if (a.input.kind === "complex-f32") {
          const f32 = new Float32Array(a.input.data);
          if (f32.length !== 2 * inComplex) throw new Error(`golden input length mismatch: have ${f32.length} expected ${2 * inComplex}`);
          inputBuf = uploadComplex(device, f32);
        } else {
          plan.destroy();
          throw new SkipError(`unsupported c2c input kind: ${a.input.kind}`);
        }

        outputBuf = device.createBuffer({
          size: outComplex * bytesPerComplex,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const enc = device.createCommandEncoder();
        plan.exec(enc, { input: inputBuf, output: outputBuf });
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();

        let outF32;
        if (precision === "f16-storage") {
          const outU16 = await downloadU16(device, outputBuf, outComplex * 4);
          outF32 = new Float32Array(outU16.length);
          for (let i = 0; i < outU16.length; i++) outF32[i] = f16BitsToF32(outU16[i]);
        } else {
          outF32 = await downloadComplex(device, outputBuf, outComplex);
        }

        const tol = a.tol ?? { atol: 1e-3, rtol: 1e-3 };
        const cpu = new Float32Array(a.cpu.data);

        // If the plan wrote into an output view larger than the logical domain, compare only the logical region.
        if (outView !== logicalN && planOpts.ioView?.output) {
          const off = computeOffset1d(logicalN, outView, planOpts.ioView.output);
          const start = -off;
          const mapped = outF32.subarray(2 * start, 2 * (start + logicalN));
          assertCloseArray(mapped, cpu, tol.atol, tol.rtol, `golden cpu compare (mapped): ${a.name}`);
        } else {
          assertCloseArray(outF32, cpu, tol.atol, tol.rtol, `golden cpu compare: ${a.name}`);
        }

        plan.destroy();
        inputBuf.destroy();
        outputBuf.destroy();
        return;
      }

      if (planOpts.type === "conv2d") {
        const plan = createPlan(device, planOpts);
        const kernel = a.kernel ? new Float32Array(a.kernel.data) : null;
        if (!kernel) throw new Error("golden conv2d missing kernel");
        if (a.input.kind !== "complex-f32") throw new SkipError(`unsupported conv2d input kind: ${a.input.kind}`);

        const input = new Float32Array(a.input.data);
        const inputBuf = uploadComplex(device, input);
        const outputBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const enc = device.createCommandEncoder();
        plan.exec(enc, { input: inputBuf, output: outputBuf, kernel });
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const out = await downloadComplex(device, outputBuf, input.length / 2);
        const cpu = new Float32Array(a.cpu.data);
        const tol = a.tol ?? { atol: 3e-3, rtol: 3e-3 };
        assertCloseArray(out, cpu, tol.atol, tol.rtol, `golden conv2d cpu compare: ${a.name}`);

        plan.destroy();
        inputBuf.destroy();
        outputBuf.destroy();
        return;
      }

      log(`Skipping golden artifact ${a.name}: unsupported type=${planOpts.type}`);
      throw new SkipError(`unsupported golden type: ${planOpts.type}`);
    });
  }
}

