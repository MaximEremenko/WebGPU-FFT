// Copyright (c) 2026 Maksim Eremenko

import { generateStockhamRadixStageWGSL } from "./kernels/stockham_stage.js";
import { generateSubgroupPow2FftWGSL } from "./kernels/subgroup_pow2_fft.js";

function assertOneOf(value, allowed, name) {
  if (!allowed.includes(value)) {
    throw new Error(`${name} must be one of ${allowed.map((v) => JSON.stringify(v)).join(", ")}; got ${JSON.stringify(value)}`);
  }
}

function isPositiveInt(x) {
  return Number.isInteger(x) && x > 0;
}

function isPowerOfTwo(n) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

function factorizeRadices(n) {
  // Greedy decomposition for current mixed-radix kernels.
  // NOTE: For best performance this could be a cost-based search; keep simple for now.
  const allowed = [13, 11, 8, 7, 5, 4, 3, 2];
  const out = [];
  let x = n;
  for (const r of allowed) {
    while (x % r === 0) {
      out.push(r);
      x = x / r;
    }
  }
  return x === 1 ? out : null;
}

function product(arr) {
  let p = 1;
  for (const v of arr) p *= v;
  return p;
}

function strideForAxis(dims, axis) {
  let s = 1;
  for (let i = 0; i < axis; i++) s *= dims[i];
  return s;
}

function formatLimits(limits) {
  const m = limits?.maxComputeWorkgroupsPerDimension;
  let maxDispatch = undefined;
  if (Array.isArray(m) || ArrayBuffer.isView(m)) {
    maxDispatch = [m[0], m[1], m[2]];
  } else if (Number.isFinite(m)) {
    maxDispatch = Math.floor(m);
  }
  return JSON.stringify(
    {
      maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize,
      maxBufferSize: limits?.maxBufferSize,
      maxComputeWorkgroupSizeX: limits?.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: limits?.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupSizeZ: limits?.maxComputeWorkgroupSizeZ,
      maxComputeInvocationsPerWorkgroup: limits?.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupStorageSize: limits?.maxComputeWorkgroupStorageSize,
      minStorageBufferOffsetAlignment: limits?.minStorageBufferOffsetAlignment,
      maxComputeWorkgroupsPerDimension: maxDispatch,
    },
    null,
    2
  );
}

function maxComputeWorkgroupsX(limits) {
  const raw = limits?.maxComputeWorkgroupsPerDimension;
  if (Array.isArray(raw) || ArrayBuffer.isView(raw)) {
    const v = raw[0];
    return Number.isFinite(v) ? Math.floor(v) : null;
  }
  if (Number.isFinite(raw)) return Math.floor(raw);
  return null;
}

function normalizeScaleFactor({ normalize, direction, nTotal }) {
  if (normalize === "none") return 1.0;
  if (normalize === "unitary") return 1.0 / Math.sqrt(nTotal);
  if (normalize === "backward") return direction === "inverse" ? 1.0 / nTotal : 1.0;
  throw new Error(`Unknown normalize mode: ${normalize}`);
}

function chooseAxis0TwoStepFactors(axisLen, maxAxisElems) {
  if (!Number.isInteger(maxAxisElems) || maxAxisElems < 2) return null;
  let best = null;
  const consider = (n1, n2) => {
    if (!Number.isInteger(n1) || !Number.isInteger(n2)) return;
    if (n1 < 2 || n2 < 2) return;
    if (n1 > maxAxisElems || n2 > maxAxisElems) return;
    if (!factorizeRadices(n1) || !factorizeRadices(n2)) return;
    const score = Math.max(n1, n2);
    const balance = Math.abs(n1 - n2);
    if (!best || score < best.score || (score === best.score && balance < best.balance)) {
      best = { n1, n2, score, balance };
    }
  };
  const root = Math.floor(Math.sqrt(axisLen));
  for (let d = 1; d <= root; d++) {
    if (axisLen % d !== 0) continue;
    const q = axisLen / d;
    consider(d, q);
    consider(q, d);
  }
  if (!best) return null;
  return { n1: best.n1, n2: best.n2 };
}

function generateAxis0TwoStepTwiddleWGSL({ n1, n, sign, workgroupSize }) {
  return /* wgsl */ `
struct Params {
  total: u32,
  baseIndex: u32,
  lineOffset: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;
const N1: u32 = ${n1}u;
const N: u32 = ${n}u;
const SIGN: f32 = ${sign};

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cis(theta: f32) -> vec2<f32> {
  return vec2<f32>(cos(theta), sin(theta));
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = params.baseIndex + gid.x;
  if (idx >= params.total) {
    return;
  }
  let lineLocal: u32 = idx / N1;
  let k1: u32 = idx - lineLocal * N1;
  let m2: u32 = params.lineOffset + lineLocal;
  let angle: f32 = SIGN * (2.0 * PI) * (f32(k1) * f32(m2)) / f32(N);
  let w: vec2<f32> = cis(angle);
  data[idx] = c_mul(data[idx], w);
}
`;
}

function generateAxis0TwoStepScaleWGSL({ scaleFactor, workgroupSize }) {
  return /* wgsl */ `
struct Params {
  total: u32,
  baseIndex: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

const SCALE: f32 = ${scaleFactor};

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = params.baseIndex + gid.x;
  if (idx >= params.total) {
    return;
  }
  data[idx] = data[idx] * vec2<f32>(SCALE, SCALE);
}
`;
}

function pickWorkgroupSizeX(limits) {
  const maxX = limits?.maxComputeWorkgroupSizeX ?? 256;
  const maxInvocations = limits?.maxComputeInvocationsPerWorkgroup ?? 256;
  return Math.max(1, Math.min(256, maxX, maxInvocations));
}

function isGpuBufferLike(x) {
  return !!x && !x?.segments && typeof x?.size === "number" && typeof x?.destroy === "function";
}

class FftPlan {
  constructor(device, config, compiled) {
    this.device = device;
    this.config = config;
    this._axesList = Array.isArray(compiled.axesList) ? compiled.axesList.slice() : [];
    this._pipelines = compiled.pipelines;
    this._workgroupSizeX = compiled.workgroupSizeX;
    this._paramsBuffer = compiled.paramsBuffer;
    this._bindGroupLayout = compiled.bindGroupLayout;
    this._paramsUpload = null;
    this._paramsUploadBytes = 0;
    this._scratch = null;
    this._scratchBytes = 0;
    this._packIn = null;
    this._packOut = null;
    this._packBytes = 0;
    this._axis0TwoStep = compiled.axis0TwoStep ?? null;
    this._twoStepLineA = null;
    this._twoStepLineB = null;
    this._twoStepLineBytes = 0;
    this._axis0PointwiseParams = null;
    this._destroyed = false;

    if (this._axis0TwoStep) {
      const { n1, n2 } = this._axis0TwoStep;
      this._axis0TwoStep.stage1 = createFftPlan(this.device, {
        shape: [n1, n2],
        direction: this.config.direction,
        normalize: "none",
        inPlace: true,
        layout: "interleaved",
        precision: "f32",
        axes: [0],
      });
      this._axis0TwoStep.stage2 = createFftPlan(this.device, {
        shape: [n2, n1],
        direction: this.config.direction,
        normalize: "none",
        inPlace: true,
        layout: "interleaved",
        precision: "f32",
        axes: [0],
      });
      const bgl = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pl = this.device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const twiddleCode = generateAxis0TwoStepTwiddleWGSL({
        n1,
        n: n1 * n2,
        sign: this.config.direction === "forward" ? -1.0 : 1.0,
        workgroupSize: this._workgroupSizeX,
      });
      const twiddlePipeline = this.device.createComputePipeline({
        layout: pl,
        compute: { module: this.device.createShaderModule({ code: twiddleCode }), entryPoint: "main" },
      });
      let scalePipeline = null;
      if (this._axis0TwoStep.applyScale) {
        const scaleCode = generateAxis0TwoStepScaleWGSL({
          scaleFactor: this._axis0TwoStep.scaleFactor,
          workgroupSize: this._workgroupSizeX,
        });
        scalePipeline = this.device.createComputePipeline({
          layout: pl,
          compute: { module: this.device.createShaderModule({ code: scaleCode }), entryPoint: "main" },
        });
      }
      this._axis0TwoStep.pointwise = { bgl, pl, twiddlePipeline, scalePipeline };
      this._axis0PointwiseParams = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    if (this._scratch) this._scratch.destroy();
    this._scratch = null;
    this._scratchBytes = 0;
    if (this._packIn) this._packIn.destroy();
    if (this._packOut) this._packOut.destroy();
    this._packIn = null;
    this._packOut = null;
    this._packBytes = 0;
    if (this._paramsUpload) this._paramsUpload.destroy();
    this._paramsUpload = null;
    this._paramsUploadBytes = 0;
    if (this._axis0PointwiseParams) this._axis0PointwiseParams.destroy();
    this._axis0PointwiseParams = null;
    if (this._twoStepLineA) this._twoStepLineA.destroy();
    if (this._twoStepLineB) this._twoStepLineB.destroy();
    this._twoStepLineA = null;
    this._twoStepLineB = null;
    this._twoStepLineBytes = 0;
    if (this._axis0TwoStep?.stage1) this._axis0TwoStep.stage1.destroy();
    if (this._axis0TwoStep?.stage2) this._axis0TwoStep.stage2.destroy();
    this._axis0TwoStep = null;
  }

  _ensureScratch(bytes) {
    if (this._scratch && this._scratchBytes >= bytes) return;
    if (this._scratch) this._scratch.destroy();
    this._scratch = this.device.createBuffer({
      size: bytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    this._scratchBytes = bytes;
  }

  _ensurePack(bytes) {
    if (this._packIn && this._packBytes >= bytes) return;
    if (this._packIn) this._packIn.destroy();
    if (this._packOut) this._packOut.destroy();
    this._packIn = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._packOut = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._packBytes = bytes;
  }

  _ensureTwoStepLineBuffers(bytes) {
    if (this._twoStepLineA && this._twoStepLineB && this._twoStepLineBytes >= bytes) return;
    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    if (bytes > maxBufferSize) {
      throw new Error(
        `Axis-0 two-step fallback requires ${bytes} bytes line staging, exceeding device.limits.maxBufferSize=${maxBufferSize}`
      );
    }
    if (this._twoStepLineA) this._twoStepLineA.destroy();
    if (this._twoStepLineB) this._twoStepLineB.destroy();
    this._twoStepLineA = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._twoStepLineB = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._twoStepLineBytes = bytes;
  }

  _ensureParamsUpload(bytes) {
    if (this._paramsUpload && this._paramsUploadBytes >= bytes) return;
    if (this._paramsUpload) this._paramsUpload.destroy();
    this._paramsUpload = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._paramsUploadBytes = bytes;
  }

  _buildDispatchChunks(totalComplex, shape, batch) {
    const workgroupsX = Math.ceil(totalComplex / this._workgroupSizeX);
    const maxWgXRaw = maxComputeWorkgroupsX(this.device.limits);
    const maxWgX = Number.isFinite(maxWgXRaw) ? Math.floor(maxWgXRaw) : null;
    if (maxWgX != null && maxWgX < 1) {
      throw new Error(
        [
          `Invalid device limit: maxComputeWorkgroupsPerDimension=${maxWgXRaw}`,
          `shape=${JSON.stringify(shape)} batch=${batch} totalComplex=${totalComplex} workgroupSizeX=${this._workgroupSizeX} workgroupsX=${workgroupsX}`,
        ].join("\n")
      );
    }
    const maxChunkWgX = maxWgX == null ? workgroupsX : Math.max(1, Math.min(workgroupsX, maxWgX));
    const dispatchChunks = [];
    for (let wgStart = 0; wgStart < workgroupsX; wgStart += maxChunkWgX) {
      const wgCount = Math.min(maxChunkWgX, workgroupsX - wgStart);
      dispatchChunks.push({ wgCount, baseIndex: wgStart * this._workgroupSizeX });
    }
    return { workgroupsX, dispatchChunks };
  }

  _transposeLineMatrixCopies(commandEncoder, { srcBuffer, dstBuffer, nx, ny }) {
    const elemBytes = 8;
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const src = (y * nx + x) * elemBytes;
        const dst = (x * ny + y) * elemBytes;
        commandEncoder.copyBufferToBuffer(srcBuffer, src, dstBuffer, dst, elemBytes);
      }
    }
  }

  _runAxis0PointwiseWindowed(commandEncoder, { buffer, axisLen, lineCount, pipeline, maxBind }) {
    const lineStrideBytes = axisLen * 8;
    const maxLinesByBind = Math.max(1, Math.floor(maxBind / lineStrideBytes));
    const windows = [];
    let maxChunkBytes = 0;
    let totalParamRecords = 0;
    for (let lineStart = 0; lineStart < lineCount; lineStart += maxLinesByBind) {
      const lines = Math.min(maxLinesByBind, lineCount - lineStart);
      const chunkComplex = lines * axisLen;
      const chunkBytes = chunkComplex * 8;
      const dispatch = this._buildDispatchChunks(chunkComplex, [axisLen], lines);
      windows.push({
        lineStart,
        lineBaseBytes: lineStart * lineStrideBytes,
        chunkComplex,
        chunkBytes,
        dispatchChunks: dispatch.dispatchChunks,
        paramBaseIndex: totalParamRecords,
      });
      maxChunkBytes = Math.max(maxChunkBytes, chunkBytes);
      totalParamRecords += dispatch.dispatchChunks.length;
    }

    this._ensurePack(maxChunkBytes);
    this._ensureParamsUpload(totalParamRecords * 16);
    const paramsUpload = new Uint32Array(totalParamRecords * 4);
    let p = 0;
    for (const w of windows) {
      for (const dc of w.dispatchChunks) {
        paramsUpload[p++] = w.chunkComplex;
        paramsUpload[p++] = dc.baseIndex;
        paramsUpload[p++] = w.lineStart;
        paramsUpload[p++] = 0;
      }
    }
    this.device.queue.writeBuffer(this._paramsUpload, 0, paramsUpload);

    const bgCache = [];
    const getBindGroup = (sizeBytes) => {
      for (const e of bgCache) {
        if (e.sz === sizeBytes) return e.bg;
      }
      const bg = this.device.createBindGroup({
        layout: this._axis0TwoStep.pointwise.bgl,
        entries: [
          { binding: 0, resource: { buffer: this._packIn, offset: 0, size: sizeBytes } },
          { binding: 1, resource: { buffer: this._axis0PointwiseParams, offset: 0, size: 16 } },
        ],
      });
      bgCache.push({ sz: sizeBytes, bg });
      return bg;
    };

    for (const w of windows) {
      commandEncoder.copyBufferToBuffer(buffer, w.lineBaseBytes, this._packIn, 0, w.chunkBytes);
      const bg = getBindGroup(w.chunkBytes);
      let paramsRecordIndex = w.paramBaseIndex;
      for (const dc of w.dispatchChunks) {
        commandEncoder.copyBufferToBuffer(this._paramsUpload, paramsRecordIndex * 16, this._axis0PointwiseParams, 0, 16);
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(dc.wgCount, 1, 1);
        pass.end();
        paramsRecordIndex++;
      }
      commandEncoder.copyBufferToBuffer(this._packIn, 0, buffer, w.lineBaseBytes, w.chunkBytes);
    }
  }

  _execAxis0TwoStep(commandEncoder, opts) {
    const {
      input,
      output,
      shape,
      batch,
      totalComplex,
      totalBytes,
      inputOffsetBytes,
      outputOffsetBytes,
      maxBind,
    } = opts;
    const isGpuBuffer = isGpuBufferLike;
    const axisLen = shape[0];
    const lineStrideBytes = axisLen * 8;
    const linesTotal = totalComplex / axisLen;
    if (!Number.isInteger(linesTotal) || linesTotal < 1) {
      throw new Error(`Internal error: invalid axis-0 line count (${linesTotal})`);
    }
    const ts = this._axis0TwoStep;
    if (!ts) throw new Error("Internal error: axis-0 two-step state is missing");
    if (ts.n1 * ts.n2 !== axisLen) {
      throw new Error(`Internal error: axis-0 two-step factors mismatch axis length (${ts.n1}*${ts.n2} != ${axisLen})`);
    }

    const normalizeView = (bufOrView, extraOffsetBytes) => {
      if (!bufOrView) return null;
      if (isGpuBuffer(bufOrView)) {
        if (extraOffsetBytes + totalBytes > bufOrView.size) {
          throw new Error(`GPUBuffer too small: need ${extraOffsetBytes + totalBytes} bytes, have ${bufOrView.size}`);
        }
        return {
          kind: "buffer",
          buffer: bufOrView,
          startBytes: extraOffsetBytes,
          lengthBytes: bufOrView.size - extraOffsetBytes,
        };
      }
      const segments = bufOrView?.segments;
      if (!Array.isArray(segments) || segments.length === 0) {
        throw new Error("Expected GPUBuffer or BufferView");
      }
      const logicalByteOffset = bufOrView.logicalByteOffset ?? 0;
      const lengthBytes = bufOrView.lengthBytes ?? segments.reduce((a, s) => a + s.sizeBytes, 0);
      const start = logicalByteOffset + extraOffsetBytes;
      if (extraOffsetBytes + totalBytes > lengthBytes) {
        throw new Error(`BufferView too small: need ${totalBytes} bytes at offset ${extraOffsetBytes}, have ${lengthBytes}`);
      }
      return {
        kind: "view",
        segments,
        logicalByteOffset,
        lengthBytes,
        startBytes: start,
      };
    };

    const iterViewRanges = (view, relativeStartBytes, bytesWanted) => {
      if (view.kind === "buffer") {
        return [{ buffer: view.buffer, offsetBytes: view.startBytes + relativeStartBytes, sizeBytes: bytesWanted }];
      }
      const start = view.startBytes + relativeStartBytes;
      if (start + bytesWanted > view.logicalByteOffset + view.lengthBytes) {
        throw new Error(`BufferView window out of range: need ${bytesWanted} bytes at relative start ${relativeStartBytes}`);
      }
      const out = [];
      let remaining = bytesWanted;
      let logicalPos = 0;
      let cursor = start;
      for (const seg of view.segments) {
        const segStart = logicalPos;
        const segEnd = logicalPos + seg.sizeBytes;
        if (cursor >= segEnd) {
          logicalPos = segEnd;
          continue;
        }
        if (cursor < segStart) throw new Error("BufferView segments must be contiguous in logical space");
        const within = cursor - segStart;
        const take = Math.min(remaining, seg.sizeBytes - within);
        out.push({ buffer: seg.buffer, offsetBytes: seg.offsetBytes + within, sizeBytes: take });
        remaining -= take;
        cursor += take;
        logicalPos = segEnd;
        if (remaining === 0) break;
      }
      if (remaining !== 0) throw new Error("BufferView did not cover requested window");
      return out;
    };

    const inView = normalizeView(input, inputOffsetBytes);
    const outView = normalizeView(output, outputOffsetBytes);
    this._ensureTwoStepLineBuffers(lineStrideBytes);
    const lineA = this._twoStepLineA;
    const lineB = this._twoStepLineB;

    for (let line = 0; line < linesTotal; line++) {
      const lineOffsetBytes = line * lineStrideBytes;
      const inRanges = iterViewRanges(inView, lineOffsetBytes, lineStrideBytes);
      let dst = 0;
      for (const r of inRanges) {
        if ((r.offsetBytes | dst | r.sizeBytes) % 4 !== 0) {
          throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
        }
        commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, lineA, dst, r.sizeBytes);
        dst += r.sizeBytes;
      }

      ts.stage1.exec(commandEncoder, { input: lineA, batch: 1 });
      this._runAxis0PointwiseWindowed(commandEncoder, {
        buffer: lineA,
        axisLen: ts.n1,
        lineCount: ts.n2,
        pipeline: ts.pointwise.twiddlePipeline,
        maxBind,
      });
      this._transposeLineMatrixCopies(commandEncoder, { srcBuffer: lineA, dstBuffer: lineB, nx: ts.n1, ny: ts.n2 });
      ts.stage2.exec(commandEncoder, { input: lineB, batch: 1 });
      this._transposeLineMatrixCopies(commandEncoder, { srcBuffer: lineB, dstBuffer: lineA, nx: ts.n2, ny: ts.n1 });

      if (ts.applyScale && ts.pointwise.scalePipeline) {
        this._runAxis0PointwiseWindowed(commandEncoder, {
          buffer: lineA,
          axisLen: ts.n1,
          lineCount: ts.n2,
          pipeline: ts.pointwise.scalePipeline,
          maxBind,
        });
      }

      const outRanges = iterViewRanges(outView, lineOffsetBytes, lineStrideBytes);
      let src = 0;
      for (const r of outRanges) {
        if ((r.offsetBytes | src | r.sizeBytes) % 4 !== 0) {
          throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
        }
        commandEncoder.copyBufferToBuffer(lineA, src, r.buffer, r.offsetBytes, r.sizeBytes);
        src += r.sizeBytes;
      }
    }
  }

  _execAxis0Windowed(commandEncoder, opts) {
    const {
      input,
      output,
      shape,
      batch,
      totalComplex,
      totalBytes,
      inputOffsetBytes,
      outputOffsetBytes,
      maxBind,
    } = opts;
    const isGpuBuffer = isGpuBufferLike;

    const axisLen = shape[0];
    const lineStrideBytes = axisLen * 8;
    const linesTotal = totalComplex / axisLen;
    if (!Number.isInteger(linesTotal) || linesTotal < 1) {
      throw new Error(`Internal error: invalid axis-0 line count (${linesTotal})`);
    }

    const normalizeView = (bufOrView, extraOffsetBytes) => {
      if (!bufOrView) return null;
      if (isGpuBuffer(bufOrView)) {
        if (extraOffsetBytes + totalBytes > bufOrView.size) {
          throw new Error(`GPUBuffer too small: need ${extraOffsetBytes + totalBytes} bytes, have ${bufOrView.size}`);
        }
        return {
          kind: "buffer",
          buffer: bufOrView,
          startBytes: extraOffsetBytes,
          lengthBytes: bufOrView.size - extraOffsetBytes,
        };
      }
      const segments = bufOrView?.segments;
      if (!Array.isArray(segments) || segments.length === 0) {
        throw new Error("Expected GPUBuffer or BufferView");
      }
      const logicalByteOffset = bufOrView.logicalByteOffset ?? 0;
      const lengthBytes = bufOrView.lengthBytes ?? segments.reduce((a, s) => a + s.sizeBytes, 0);
      const start = logicalByteOffset + extraOffsetBytes;
      if (extraOffsetBytes + totalBytes > lengthBytes) {
        throw new Error(`BufferView too small: need ${totalBytes} bytes at offset ${extraOffsetBytes}, have ${lengthBytes}`);
      }
      return {
        kind: "view",
        segments,
        logicalByteOffset,
        lengthBytes,
        startBytes: start,
      };
    };

    const iterViewRanges = (view, relativeStartBytes, bytesWanted) => {
      if (view.kind === "buffer") {
        return [{ buffer: view.buffer, offsetBytes: view.startBytes + relativeStartBytes, sizeBytes: bytesWanted }];
      }
      const start = view.startBytes + relativeStartBytes;
      if (start + bytesWanted > view.logicalByteOffset + view.lengthBytes) {
        throw new Error(`BufferView window out of range: need ${bytesWanted} bytes at relative start ${relativeStartBytes}`);
      }
      const out = [];
      let remaining = bytesWanted;
      let logicalPos = 0;
      let cursor = start;
      for (const seg of view.segments) {
        const segStart = logicalPos;
        const segEnd = logicalPos + seg.sizeBytes;
        if (cursor >= segEnd) {
          logicalPos = segEnd;
          continue;
        }
        if (cursor < segStart) throw new Error("BufferView segments must be contiguous in logical space");
        const within = cursor - segStart;
        const take = Math.min(remaining, seg.sizeBytes - within);
        out.push({ buffer: seg.buffer, offsetBytes: seg.offsetBytes + within, sizeBytes: take });
        remaining -= take;
        cursor += take;
        logicalPos = segEnd;
        if (remaining === 0) break;
      }
      if (remaining !== 0) throw new Error("BufferView did not cover requested window");
      return out;
    };

    const inView = normalizeView(input, inputOffsetBytes);
    const outView = normalizeView(output, outputOffsetBytes);

    const maxLinesByBind = Math.max(1, Math.floor(maxBind / lineStrideBytes));
    const windows = [];
    let maxChunkBytes = 0;
    let totalParamRecords = 0;
    for (let lineStart = 0; lineStart < linesTotal; lineStart += maxLinesByBind) {
      const lineCount = Math.min(maxLinesByBind, linesTotal - lineStart);
      const chunkComplex = lineCount * axisLen;
      const chunkBytes = chunkComplex * 8;
      const dispatch = this._buildDispatchChunks(chunkComplex, shape, batch);
      windows.push({
        lineStart,
        lineBaseComplex: lineStart * axisLen,
        chunkComplex,
        chunkBytes,
        dispatchChunks: dispatch.dispatchChunks,
        paramBaseIndex: totalParamRecords,
      });
      maxChunkBytes = Math.max(maxChunkBytes, chunkBytes);
      totalParamRecords += dispatch.dispatchChunks.length;
    }

    if (this._pipelines.length !== 1) {
      throw new Error("Internal error: axis-0 window fallback expects a single-axis pipeline");
    }
    this._ensurePack(maxChunkBytes);
    const winA = this._packIn;
    const winB = this._packOut;

    this._ensureParamsUpload(totalParamRecords * 16);
    const paramsUpload = new Uint32Array(totalParamRecords * 4);
    let p = 0;
    for (const w of windows) {
      for (const dc of w.dispatchChunks) {
        paramsUpload[p++] = w.chunkComplex;
        paramsUpload[p++] = dc.baseIndex;
        paramsUpload[p++] = w.lineStart;
        paramsUpload[p++] = w.lineBaseComplex;
      }
    }
    this.device.queue.writeBuffer(this._paramsUpload, 0, paramsUpload);

    const bgCache = [];
    const getBindGroup = (srcBuf, dstBuf, sizeBytes, srcOffset = 0, dstOffset = 0) => {
      for (const e of bgCache) {
        if (e.src === srcBuf && e.dst === dstBuf && e.sz === sizeBytes && e.srcOff === srcOffset && e.dstOff === dstOffset) {
          return e.bg;
        }
      }
      const bg = this.device.createBindGroup({
        layout: this._bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: srcBuf, offset: srcOffset, size: sizeBytes } },
          { binding: 1, resource: { buffer: dstBuf, offset: dstOffset, size: sizeBytes } },
          { binding: 2, resource: { buffer: this._paramsBuffer, offset: 0, size: 16 } },
        ],
      });
      bgCache.push({ src: srcBuf, dst: dstBuf, sz: sizeBytes, srcOff: srcOffset, dstOff: dstOffset, bg });
      return bg;
    };

    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const canDirectInPlaceWindowed =
      inView.kind === "buffer" &&
      outView.kind === "buffer" &&
      inView.buffer === outView.buffer &&
      inView.startBytes === outView.startBytes &&
      inView.startBytes % storageAlign === 0 &&
      windows.every((w) => ((inView.startBytes + w.lineBaseComplex * 8) % storageAlign) === 0);

    if (canDirectInPlaceWindowed) {
      for (let axis = 0; axis < this._pipelines.length; axis++) {
        const stages = this._pipelines[axis];
        for (const w of windows) {
          const lineOffsetBytes = inView.startBytes + w.lineBaseComplex * 8;
          let srcBuf = inView.buffer;
          let srcOff = lineOffsetBytes;
          let dstBuf = winB;
          let dstOff = 0;

          for (let s = 0; s < stages.length; s++) {
            const pipeline = stages[s];
            const bg = getBindGroup(srcBuf, dstBuf, w.chunkBytes, srcOff, dstOff);
            let paramsRecordIndex = w.paramBaseIndex;
            for (const dc of w.dispatchChunks) {
              commandEncoder.copyBufferToBuffer(this._paramsUpload, paramsRecordIndex * 16, this._paramsBuffer, 0, 16);
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(dc.wgCount, 1, 1);
              pass.end();
              paramsRecordIndex++;
            }
            const tmpBuf = srcBuf;
            srcBuf = dstBuf;
            dstBuf = tmpBuf;
            const tmpOff = srcOff;
            srcOff = dstOff;
            dstOff = tmpOff;
          }

          if (srcBuf !== inView.buffer || srcOff !== lineOffsetBytes) {
            commandEncoder.copyBufferToBuffer(srcBuf, srcOff, inView.buffer, lineOffsetBytes, w.chunkBytes);
          }
        }
      }
      return;
    }

    for (let axis = 0; axis < this._pipelines.length; axis++) {
      const stages = this._pipelines[axis];
      for (const w of windows) {
        const lineOffsetBytes = w.lineBaseComplex * 8;
        const inRanges = iterViewRanges(inView, lineOffsetBytes, w.chunkBytes);
        let winWrite = 0;
        for (const r of inRanges) {
          if ((r.offsetBytes | winWrite | r.sizeBytes) % 4 !== 0) {
            throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
          }
          commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, winA, winWrite, r.sizeBytes);
          winWrite += r.sizeBytes;
        }

        let srcBuf = winA;
        let dstBuf = winB;
        for (let s = 0; s < stages.length; s++) {
          const pipeline = stages[s];
          const bg = getBindGroup(srcBuf, dstBuf, w.chunkBytes);
          let paramsRecordIndex = w.paramBaseIndex;
          for (const dc of w.dispatchChunks) {
            commandEncoder.copyBufferToBuffer(this._paramsUpload, paramsRecordIndex * 16, this._paramsBuffer, 0, 16);
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(dc.wgCount, 1, 1);
            pass.end();
            paramsRecordIndex++;
          }
          const tmp = srcBuf;
          srcBuf = dstBuf;
          dstBuf = tmp;
        }

        const outRanges = iterViewRanges(outView, lineOffsetBytes, w.chunkBytes);
        let winRead = 0;
        for (const r of outRanges) {
          if ((r.offsetBytes | winRead | r.sizeBytes) % 4 !== 0) {
            throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
          }
          commandEncoder.copyBufferToBuffer(srcBuf, winRead, r.buffer, r.offsetBytes, r.sizeBytes);
          winRead += r.sizeBytes;
        }
      }
    }
  }

  exec(commandEncoder, opts) {
    if (this._destroyed) throw new Error("FftPlan is destroyed");
    if (!commandEncoder) throw new Error("exec requires a commandEncoder");
    if (!opts?.input) throw new Error("exec requires opts.input (GPUBuffer)");

    const {
      input,
      output,
      temp,
      batch = 1,
      inputOffsetBytes = 0,
      outputOffsetBytes = 0,
    } = opts;

    if (!isPositiveInt(batch)) throw new Error(`batch must be a positive integer; got ${batch}`);
    if (!Number.isInteger(inputOffsetBytes) || inputOffsetBytes < 0) {
      throw new Error(`inputOffsetBytes must be a non-negative integer; got ${inputOffsetBytes}`);
    }
    if (!Number.isInteger(outputOffsetBytes) || outputOffsetBytes < 0) {
      throw new Error(`outputOffsetBytes must be a non-negative integer; got ${outputOffsetBytes}`);
    }
    if (inputOffsetBytes % 8 !== 0) throw new Error(`inputOffsetBytes must be a multiple of 8 (complex<f32>); got ${inputOffsetBytes}`);
    if (outputOffsetBytes % 8 !== 0) throw new Error(`outputOffsetBytes must be a multiple of 8 (complex<f32>); got ${outputOffsetBytes}`);

    const { shape, inPlace } = this.config;
    const nTotal = product(shape);
    const totalComplex = batch * nTotal;
    const totalBytes = totalComplex * 8;
    const axis0Only = this._axesList.length === 1 && this._axesList[0] === 0;

    if (!inPlace && !output) throw new Error("exec requires opts.output when plan.inPlace=false");
    if (inPlace && output && output !== input) {
      throw new Error("plan.inPlace=true requires opts.output to be omitted or equal to opts.input");
    }
    if (!inPlace && output && isGpuBufferLike(input) && isGpuBufferLike(output) && input === output) {
      throw new Error("plan.inPlace=false does not allow opts.input === opts.output");
    }

    const limits = this.device.limits;
    const storageAlign = limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxBind = this.config.maxStorageBufferBindingSize ?? limits?.maxStorageBufferBindingSize ?? Infinity;
    const isBufferOrView = (v) =>
      isGpuBufferLike(v) || (v && Array.isArray(v.segments) && v.segments.length > 0);
    if (totalBytes > maxBind) {
      const bytesPerBatch = nTotal * 8;
      // Out-of-core fallback (batch chunking): when one logical batch fits the binding limit,
      // execute multiple smaller batch slices with adjusted byte offsets.
      if (batch > 1 && bytesPerBatch <= maxBind) {
        const maxBatchPerChunk = Math.max(1, Math.floor(maxBind / bytesPerBatch));
        if (inPlace) {
          // In-place path binds the FFT data buffer directly and requires aligned offsets.
          if (bytesPerBatch % storageAlign !== 0) {
            throw new Error(
              [
                `Requested binding size exceeds device.limits.maxStorageBufferBindingSize.`,
                `Batch-chunk fallback is available but requires bytesPerBatch (${bytesPerBatch}) to be aligned to minStorageBufferOffsetAlignment (${storageAlign}) for inPlace plans.`,
                `shape=${JSON.stringify(shape)} batch=${batch} totalBytes=${totalBytes}`,
                `maxStorageBufferBindingSize=${maxBind}`,
              ].join("\n")
            );
          }
          if (inputOffsetBytes % storageAlign !== 0) {
            throw new Error(
              [
                `Requested binding size exceeds device.limits.maxStorageBufferBindingSize.`,
                `Batch-chunk fallback is available but requires inputOffsetBytes (${inputOffsetBytes}) aligned to minStorageBufferOffsetAlignment (${storageAlign}) for inPlace plans.`,
                `shape=${JSON.stringify(shape)} batch=${batch} totalBytes=${totalBytes}`,
                `maxStorageBufferBindingSize=${maxBind}`,
              ].join("\n")
            );
          }
        }

        for (let b0 = 0; b0 < batch; b0 += maxBatchPerChunk) {
          const bCount = Math.min(maxBatchPerChunk, batch - b0);
          const chunkOffsetBytes = b0 * bytesPerBatch;
          this.exec(commandEncoder, {
            input,
            output,
            temp,
            batch: bCount,
            inputOffsetBytes: inputOffsetBytes + chunkOffsetBytes,
            outputOffsetBytes: outputOffsetBytes + chunkOffsetBytes,
          });
        }
        return;
      }

      // Windowed fallback for axis-0-only plans: process contiguous line chunks that fit the binding limit.
      if (axis0Only) {
        const axisLen = shape[0];
        const lineStrideBytes = axisLen * 8;
        const outTarget = inPlace ? input : output;
        const outOffset = inPlace ? inputOffsetBytes : outputOffsetBytes;
        const canWindow =
          isBufferOrView(input) &&
          isBufferOrView(outTarget) &&
          lineStrideBytes <= maxBind;
        if (canWindow) {
          if (isGpuBufferLike(input) && inputOffsetBytes + totalBytes > input.size) {
            throw new Error(`Input buffer too small: need ${inputOffsetBytes + totalBytes} bytes, have ${input.size}`);
          }
          if (isGpuBufferLike(outTarget) && outOffset + totalBytes > outTarget.size) {
            throw new Error(`Output buffer too small: need ${outOffset + totalBytes} bytes, have ${outTarget.size}`);
          }
          if (temp && isGpuBufferLike(temp) && totalBytes > temp.size) {
            throw new Error(`temp buffer too small: need ${totalBytes} bytes, have ${temp.size}`);
          }
          this._execAxis0Windowed(commandEncoder, {
            input,
            output: outTarget,
            temp,
            shape,
            batch,
            totalComplex,
            totalBytes,
            inputOffsetBytes,
            outputOffsetBytes: outOffset,
            storageAlign,
            maxBind,
          });
          return;
        }
        const canTwoStep =
          !!this._axis0TwoStep &&
          isBufferOrView(input) &&
          isBufferOrView(outTarget) &&
          lineStrideBytes > maxBind;
        if (canTwoStep) {
          if (isGpuBufferLike(input) && inputOffsetBytes + totalBytes > input.size) {
            throw new Error(`Input buffer too small: need ${inputOffsetBytes + totalBytes} bytes, have ${input.size}`);
          }
          if (isGpuBufferLike(outTarget) && outOffset + totalBytes > outTarget.size) {
            throw new Error(`Output buffer too small: need ${outOffset + totalBytes} bytes, have ${outTarget.size}`);
          }
          this._execAxis0TwoStep(commandEncoder, {
            input,
            output: outTarget,
            shape,
            batch,
            totalComplex,
            totalBytes,
            inputOffsetBytes,
            outputOffsetBytes: outOffset,
            maxBind,
          });
          return;
        }
      }

      throw new Error(
        [
          `Requested binding size exceeds device.limits.maxStorageBufferBindingSize.`,
          `shape=${JSON.stringify(shape)} batch=${batch} totalBytes=${totalBytes}`,
          `bytesPerBatch=${bytesPerBatch}`,
          `maxStorageBufferBindingSize=${maxBind}`,
        ].join("\n")
      );
    }

    const normalizeView = (bufOrView, extraOffsetBytes) => {
      if (!bufOrView) return null;
      if (isGpuBufferLike(bufOrView)) {
        return {
          kind: "buffer",
          buffer: bufOrView,
          offsetBytes: extraOffsetBytes,
          lengthBytes: bufOrView.size - extraOffsetBytes,
        };
      }
      const segments = bufOrView?.segments;
      if (!Array.isArray(segments) || segments.length === 0) {
        throw new Error("Expected GPUBuffer or BufferView");
      }
      return {
        kind: "view",
        segments,
        logicalByteOffset: bufOrView.logicalByteOffset ?? 0,
        lengthBytes: bufOrView.lengthBytes ?? segments.reduce((a, s) => a + s.sizeBytes, 0),
        extraOffsetBytes,
      };
    };

    const iterViewRanges = (view, totalBytesWanted) => {
      // Produces a list of physical ranges that cover [start, start+totalBytesWanted) within the logical view.
      const start = view.logicalByteOffset + view.extraOffsetBytes;
      if (start < 0) throw new Error("BufferView start is negative");
      if (view.extraOffsetBytes < 0) throw new Error("BufferView extraOffsetBytes is negative");
      if (start + totalBytesWanted > view.logicalByteOffset + view.lengthBytes) {
        const have = view.lengthBytes - view.extraOffsetBytes;
        throw new Error(`BufferView too small: need ${totalBytesWanted} bytes, have ${have}`);
      }

      const out = [];
      let remaining = totalBytesWanted;
      let logicalPos = 0;
      let cursor = start;

      for (const seg of view.segments) {
        const segStart = logicalPos;
        const segEnd = logicalPos + seg.sizeBytes;
        if (cursor >= segEnd) {
          logicalPos = segEnd;
          continue;
        }
        if (cursor < segStart) {
          throw new Error("BufferView segments must be contiguous in logical space");
        }
        const within = cursor - segStart;
        const take = Math.min(remaining, seg.sizeBytes - within);
        out.push({
          buffer: seg.buffer,
          offsetBytes: seg.offsetBytes + within,
          sizeBytes: take,
        });
        remaining -= take;
        cursor += take;
        logicalPos = segEnd;
        if (remaining === 0) break;
      }
      if (remaining !== 0) throw new Error("BufferView did not cover requested range");
      return out;
    };

    const inNorm = normalizeView(input, inputOffsetBytes);
    const outNorm = normalizeView(output, outputOffsetBytes);

    const singleBindOffset = (v) => {
      if (!v || v.kind !== "view" || v.segments.length !== 1) return null;
      const seg = v.segments[0];
      return seg.offsetBytes + v.logicalByteOffset + v.extraOffsetBytes;
    };

    const inNeedsPackForAlign =
      (inNorm.kind === "buffer" && inNorm.offsetBytes % storageAlign !== 0) ||
      (inNorm.kind === "view" && inNorm.segments.length === 1 && singleBindOffset(inNorm) % storageAlign !== 0);
    const outNeedsPackForAlign =
      outNorm &&
      ((outNorm.kind === "buffer" && outNorm.offsetBytes % storageAlign !== 0) ||
        (outNorm.kind === "view" && outNorm.segments.length === 1 && singleBindOffset(outNorm) % storageAlign !== 0));

    const doPackInput = (inNorm.kind === "view" && inNorm.segments.length > 1) || (!inPlace && inNeedsPackForAlign);
    const doPackOutput = (outNorm && outNorm.kind === "view" && outNorm.segments.length > 1) || (!inPlace && outNeedsPackForAlign);

    if (inPlace && (doPackInput || doPackOutput)) {
      throw new Error("inPlace with non-contiguous or misaligned views is not supported");
    }

    if (doPackInput || doPackOutput) {
      this._ensurePack(totalBytes);
    }

    const inResolved = (() => {
      if (inNorm.kind === "buffer") {
        return { buffer: inNorm.buffer, offsetBytes: inNorm.offsetBytes, viewLimitBytes: inNorm.lengthBytes };
      }
      if (doPackInput) {
        const ranges = iterViewRanges(inNorm, totalBytes);
        let dst = 0;
        for (const r of ranges) {
          if ((r.offsetBytes | dst | r.sizeBytes) % 4 !== 0) throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
          commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, this._packIn, dst, r.sizeBytes);
          dst += r.sizeBytes;
        }
        return { buffer: this._packIn, offsetBytes: 0, viewLimitBytes: totalBytes };
      }
      if (inNorm.segments.length !== 1) throw new Error("Unexpected BufferView segment count");
      const seg = inNorm.segments[0];
      const off = seg.offsetBytes + inNorm.logicalByteOffset + inNorm.extraOffsetBytes;
      return { buffer: seg.buffer, offsetBytes: off, viewLimitBytes: inNorm.lengthBytes - inNorm.extraOffsetBytes };
    })();

    const outResolved = (() => {
      if (!outNorm) return null;
      if (outNorm.kind === "buffer") {
        return { buffer: outNorm.buffer, offsetBytes: outNorm.offsetBytes, viewLimitBytes: outNorm.lengthBytes };
      }
      if (doPackOutput) {
        // We will scatter after compute.
        return { buffer: this._packOut, offsetBytes: 0, viewLimitBytes: totalBytes, _scatterTo: outNorm };
      }
      if (outNorm.segments.length !== 1) throw new Error("Unexpected BufferView segment count");
      const seg = outNorm.segments[0];
      const off = seg.offsetBytes + outNorm.logicalByteOffset + outNorm.extraOffsetBytes;
      return { buffer: seg.buffer, offsetBytes: off, viewLimitBytes: outNorm.lengthBytes - outNorm.extraOffsetBytes };
    })();

    const primary = inPlace ? inResolved : outResolved;
    if (!primary) throw new Error("exec requires opts.output when plan.inPlace=false");

    if (inResolved.viewLimitBytes < totalBytes) {
      throw new Error(`Input view too small: need ${totalBytes} bytes, have ${inResolved.viewLimitBytes}`);
    }
    if (primary.viewLimitBytes < totalBytes) {
      throw new Error(`Output view too small: need ${totalBytes} bytes, have ${primary.viewLimitBytes}`);
    }

    const { dispatchChunks } = this._buildDispatchChunks(totalComplex, shape, batch);
    const useChunkedDispatch = dispatchChunks.length > 1;

    const totalPasses = this._pipelines.reduce((acc, stages) => acc + stages.length, 0);

    let scratch = null;
    let scratchOffset = 0;
    if (temp) {
      const tNorm = normalizeView(temp, 0);
      if (!tNorm) throw new Error("temp must be GPUBuffer or BufferView");
      const t =
        tNorm.kind === "buffer"
          ? { buffer: tNorm.buffer, offsetBytes: tNorm.offsetBytes, viewLimitBytes: tNorm.lengthBytes }
          : (() => {
              if (tNorm.segments.length !== 1) throw new Error("temp must be a GPUBuffer or a single-segment BufferView");
              const seg = tNorm.segments[0];
              const off = seg.offsetBytes + tNorm.logicalByteOffset + tNorm.extraOffsetBytes;
              return { buffer: seg.buffer, offsetBytes: off, viewLimitBytes: tNorm.lengthBytes - tNorm.extraOffsetBytes };
            })();
      if (t.offsetBytes % storageAlign !== 0) {
        throw new Error(`temp binding offset must be multiple of device.limits.minStorageBufferOffsetAlignment=${storageAlign}; got ${t.offsetBytes}`);
      }
      if (t.viewLimitBytes < totalBytes) {
        throw new Error(`temp view too small: need ${totalBytes} bytes, have ${t.viewLimitBytes}`);
      }

      // If temp aliases the data buffer (same GPUBuffer), WebGPU validation gets messy and copyBufferToBuffer
      // within the same buffer is rejected. for correctness, use an internal scratch buffer instead.
      const tempAliasesData = t.buffer === inResolved.buffer || (primary && t.buffer === primary.buffer);
      if (!tempAliasesData) {
        scratch = t.buffer;
        scratchOffset = t.offsetBytes;
      } else {
        this._ensureScratch(totalBytes);
        scratch = this._scratch;
        scratchOffset = 0;
      }
    } else {
      this._ensureScratch(totalBytes);
      scratch = this._scratch;
      scratchOffset = 0;
    }

    // Update params for bounds-checking in kernels.
    if (!useChunkedDispatch) {
      this.device.queue.writeBuffer(this._paramsBuffer, 0, new Uint32Array([totalComplex, 0, 0, 0]));
    } else {
      const bytes = dispatchChunks.length * 16;
      this._ensureParamsUpload(bytes);
      const upload = new Uint32Array(dispatchChunks.length * 4);
      for (let i = 0; i < dispatchChunks.length; i++) {
        const base = i * 4;
        upload[base] = totalComplex;
        upload[base + 1] = dispatchChunks[i].baseIndex;
      }
      this.device.queue.writeBuffer(this._paramsUpload, 0, upload);
    }

    const bgCache = [];
    const getBindGroup = (srcBuf, srcOffset, dstBuf, dstOffset) => {
      for (const e of bgCache) {
        if (e.src === srcBuf && e.dst === dstBuf && e.so === srcOffset && e.do === dstOffset) return e.bg;
      }
      const bg = this.device.createBindGroup({
        layout: this._bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: srcBuf, offset: srcOffset, size: totalBytes } },
          { binding: 1, resource: { buffer: dstBuf, offset: dstOffset, size: totalBytes } },
          { binding: 2, resource: { buffer: this._paramsBuffer, offset: 0, size: 16 } },
        ],
      });
      bgCache.push({ src: srcBuf, dst: dstBuf, so: srcOffset, do: dstOffset, bg });
      return bg;
    };

    const secondary = scratch;

    // Global pass order: X then Y then Z axes, stage 0..log2-1 per axis.
    // Buffer ping-pongs between `primary` and `secondary` after the first stage.
    let srcBuf = inResolved.buffer;
    let srcOffset = inResolved.offsetBytes;

    // Choose initial destination to avoid a final copy when possible (out-of-place only).
    // For inPlace, the first dispatch must not write back into the same region we're reading.
    let dstBuf;
    let dstOffset;
    if (inPlace) {
      dstBuf = secondary;
      dstOffset = scratchOffset;
    } else {
      const wantFirstDstIsPrimary = totalPasses % 2 === 1;
      dstBuf = wantFirstDstIsPrimary ? primary.buffer : secondary;
      dstOffset = dstBuf === primary.buffer ? primary.offsetBytes : scratchOffset;
    }

    let passIndex = 0;
    for (let axis = 0; axis < this._pipelines.length; axis++) {
      const stages = this._pipelines[axis];
      for (let s = 0; s < stages.length; s++) {
        const pipeline = stages[s];
        const bg = getBindGroup(srcBuf, srcOffset, dstBuf, dstOffset);
        if (srcOffset % storageAlign !== 0) {
          throw new Error(`source binding offset must be aligned to ${storageAlign}; got ${srcOffset}`);
        }
        if (dstOffset % storageAlign !== 0) {
          throw new Error(`dest binding offset must be aligned to ${storageAlign}; got ${dstOffset}`);
        }
        if (srcBuf === dstBuf && srcOffset === dstOffset) {
          throw new Error("Internal error: src and dst bindings alias the same range");
        }

        // Important: each stage must be its own compute pass. Otherwise, WebGPU rejects the command buffer
        // because a buffer would be used as both read-only and read-write storage within the same usage scope.
        for (let chunk = 0; chunk < dispatchChunks.length; chunk++) {
          if (useChunkedDispatch) {
            commandEncoder.copyBufferToBuffer(this._paramsUpload, chunk * 16, this._paramsBuffer, 0, 16);
          }
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(dispatchChunks[chunk].wgCount, 1, 1);
          pass.end();
        }

        // Swap buffers for next pass.
        passIndex++;
        const nextSrc = dstBuf;
        const nextSrcOffset = dstOffset;
        const nextDst = nextSrc === primary.buffer ? secondary : primary.buffer;
        const nextDstOffset = nextDst === primary.buffer ? primary.offsetBytes : scratchOffset;

        srcBuf = nextSrc;
        srcOffset = nextSrcOffset;
        dstBuf = nextDst;
        dstOffset = nextDstOffset;
      }
    }

    // If the final result ended in secondary, copy to primary.
    const finalInPrimary = srcBuf === primary.buffer && srcOffset === primary.offsetBytes;
    if (!finalInPrimary) {
      if (secondary === primary.buffer) {
        throw new Error("Internal error: FFT final copy would alias within a single buffer");
      }
      commandEncoder.copyBufferToBuffer(srcBuf, srcOffset, primary.buffer, primary.offsetBytes, totalBytes);
    }

    // Scatter packed output back into a multi-segment output view if needed.
    if (outResolved?._scatterTo) {
      const view = outResolved._scatterTo;
      const ranges = iterViewRanges(view, totalBytes);
      let src = 0;
      for (const r of ranges) {
        if ((r.offsetBytes | src | r.sizeBytes) % 4 !== 0) throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
        commandEncoder.copyBufferToBuffer(primary.buffer, primary.offsetBytes + src, r.buffer, r.offsetBytes, r.sizeBytes);
        src += r.sizeBytes;
      }
    }
  }
}

export function createFftPlan(device, options) {
  if (!device) throw new Error("createFftPlan requires a WebGPU device");
  const {
    shape,
    direction,
    normalize = "none",
    inPlace = false,
    layout = "interleaved",
    precision = "f32",
    axes = null,
    maxStorageBufferBindingSize = null,
  } = options ?? {};

  if (!Array.isArray(shape) || shape.length < 1) {
    throw new Error(`shape must be an array of one or more dimensions; got ${JSON.stringify(shape)}`);
  }
  if (!shape.every((n) => isPositiveInt(n))) {
    throw new Error(`shape elements must be positive integers; got ${JSON.stringify(shape)}`);
  }
  if (shape.some((n) => n < 2)) {
    throw new Error(`All dimensions must be >= 2; got shape=${JSON.stringify(shape)}`);
  }

  assertOneOf(direction, ["forward", "inverse"], "direction");
  assertOneOf(normalize, ["none", "unitary", "backward"], "normalize");
  assertOneOf(layout, ["interleaved"], "layout");
  assertOneOf(precision, ["f32"], "precision");
  if (typeof inPlace !== "boolean") throw new Error(`inPlace must be boolean; got ${inPlace}`);
  if (maxStorageBufferBindingSize != null) {
    if (!Number.isInteger(maxStorageBufferBindingSize) || maxStorageBufferBindingSize <= 0) {
      throw new Error(`maxStorageBufferBindingSize must be a positive integer when provided; got ${maxStorageBufferBindingSize}`);
    }
  }

  const rank = shape.length;
  const dims = shape.slice();
  const nTotal = product(shape);
  const axesList = axes == null ? Array.from({ length: rank }, (_, i) => i) : axes;
  if (!Array.isArray(axesList) || axesList.length === 0) throw new Error("axes must be null or a non-empty array");
  for (const axis of axesList) {
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank) throw new Error(`Invalid axis ${axis} for rank ${rank}`);
  }

  const limits = device.limits;
  let workgroupSizeX = pickWorkgroupSizeX(limits);
  const baselineWorkgroupSizeX = workgroupSizeX;
  const deviceMaxBind = limits?.maxStorageBufferBindingSize ?? Infinity;
  const maxBind = maxStorageBufferBindingSize == null ? deviceMaxBind : Math.min(deviceMaxBind, maxStorageBufferBindingSize);
  const minBytes = nTotal * 8;
  const axis0Only = axesList.length === 1 && axesList[0] === 0;
  const axis0LineBytes = dims[0] * 8;
  const maxBufferSize = limits?.maxBufferSize ?? Infinity;
  let axis0TwoStep = null;
  if (minBytes > maxBind) {
    if (!axis0Only) {
      throw new Error(
        [
          `shape=${JSON.stringify(shape)} requires at least ${minBytes} bytes per buffer binding (batch=1).`,
          `This exceeds device.limits.maxStorageBufferBindingSize=${maxBind}.`,
          `Only axis-0-only plans can use out-of-core windowed/two-step execution when full bindings exceed the limit.`,
          `Device limits:\n${formatLimits(limits)}`,
        ].join("\n")
      );
    }
    if (axis0LineBytes > maxBind) {
      const maxAxisElems = Math.floor(maxBind / 8);
      const factors = chooseAxis0TwoStepFactors(dims[0], maxAxisElems);
      if (!factors || axis0LineBytes > maxBufferSize) {
        throw new Error(
          [
            `shape=${JSON.stringify(shape)} requires at least ${minBytes} bytes per buffer binding (batch=1).`,
            `This exceeds device.limits.maxStorageBufferBindingSize=${maxBind}.`,
            `axis-0 line-window fallback is unavailable because one axis-0 line needs ${axis0LineBytes} bytes.`,
            factors
              ? `Axis-0 two-step fallback requires line staging of ${axis0LineBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}.`
              : `Axis-0 two-step fallback is unavailable: shape[0]=${dims[0]} cannot be factorized into n1*n2 with n1,n2 <= floor(maxStorageBufferBindingSize/8)=${maxAxisElems} using supported radices.`,
            `Device limits:\n${formatLimits(limits)}`,
          ].join("\n")
        );
      }
      axis0TwoStep = { n1: factors.n1, n2: factors.n2 };
    }
  }

  const scale = normalizeScaleFactor({ normalize, direction, nTotal });
  const applyAnyScale = Math.abs(scale - 1.0) > 0;

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const pipelines = [];

  // Optional subgroup-accelerated path:
  // - only when the device advertises subgroups
  // - only when compiling a single-axis plan (axes=[axis])
  // - only for power-of-two axis lengths that fit within one workgroup (1 invocation per element)
  let useSubgroupAxis = false;
  if (device.features?.has?.("subgroups") && axesList.length === 1) {
    const axis = axesList[0];
    const axisLen = dims[axis];
    const maxWG = Math.min(limits?.maxComputeWorkgroupSizeX ?? 256, limits?.maxComputeInvocationsPerWorkgroup ?? 256);
    const maxWgMem = limits?.maxComputeWorkgroupStorageSize ?? (32 * 1024);
    const shmemBytes = axisLen * 8; // vec2<f32> per element
    if (isPowerOfTwo(axisLen) && axisLen >= 2 && axisLen <= maxWG && shmemBytes <= maxWgMem) {
      useSubgroupAxis = true;
      workgroupSizeX = axisLen;
    }
  }

  for (const axis of axis0TwoStep ? [] : axesList) {
    const axisLen = dims[axis];
    const strideComplex = strideForAxis(dims, axis);

    const axisPipelines = [];
    if (useSubgroupAxis) {
      try {
        const isLastAxis = axis === rank - 1;
        const applyScale = applyAnyScale && isLastAxis;
        const code = generateSubgroupPow2FftWGSL({
          rank,
          axis,
          dims,
          axisLength: axisLen,
          strideComplex,
          direction,
          applyScale,
          scaleFactor: scale,
        });
        const module = device.createShaderModule({ code });
        const pipeline = device.createComputePipeline({
          layout: pipelineLayout,
          compute: { module, entryPoint: "main" },
        });
        axisPipelines.push(pipeline);
      } catch {
        // Graceful fallback to baseline path if subgroup compilation fails.
        useSubgroupAxis = false;
        workgroupSizeX = baselineWorkgroupSizeX;
      }
    }
    if (!useSubgroupAxis) {
      const radices = factorizeRadices(axisLen);
      if (!radices) {
        throw new Error(
          `Axis length ${axisLen} is not factorable by supported radices {2,3,4,5,7,8,11,13} in the current implementation`
        );
      }
      let ns = 1;
      for (let s = 0; s < radices.length; s++) {
        const radix = radices[s];
        ns *= radix;
        const isLastAxis = axis === rank - 1;
        const isLastStage = s === radices.length - 1;
        const applyScale = applyAnyScale && isLastAxis && isLastStage;

        const code = generateStockhamRadixStageWGSL({
          rank,
          axis,
          dims,
          axisLength: axisLen,
          strideComplex,
          radix,
          ns,
          direction,
          workgroupSize: workgroupSizeX,
          applyScale,
          scaleFactor: scale,
        });
        const module = device.createShaderModule({ code });
        const pipeline = device.createComputePipeline({
          layout: pipelineLayout,
          compute: { module, entryPoint: "main" },
        });
        axisPipelines.push(pipeline);
      }
    }
    pipelines.push(axisPipelines);
  }

  const config = {
    shape: dims,
    direction,
    normalize,
    inPlace,
    layout,
    precision,
    maxStorageBufferBindingSize: Number.isFinite(maxBind) ? maxBind : null,
  };

  return new FftPlan(device, config, {
    axesList,
    pipelines,
    workgroupSizeX,
    paramsBuffer,
    bindGroupLayout,
    axis0TwoStep: axis0TwoStep
      ? {
          ...axis0TwoStep,
          applyScale: applyAnyScale,
          scaleFactor: scale,
        }
      : null,
  });
}

