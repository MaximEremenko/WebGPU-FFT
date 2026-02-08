// Copyright (c) 2026 Maksim Eremenko

import { createFftPlan } from "../../plan.js";
import { factorizeSupportedRadices, nextPow2, nextSmoothAtLeast } from "../../utils/factors.js";
import { ensureWithinBindingLimit, prod, alignBytes } from "../common.js";
import { normalizeToContiguousRanges } from "../segmented_io.js";

import { generateBluesteinPreWGSL, generateBluesteinMulBfftWGSL, generateBluesteinPostWGSL } from "../../kernels/bluestein.js";

function generateSliceMulWriteWGSL(workgroupSize) {
  return /* wgsl */ `
struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> lhs: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> rhs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> outv: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  let a = lhs[i];
  let b = rhs[i];
  outv[i] = vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
`;
}

function generateSliceMulInPlaceWGSL(workgroupSize) {
  return /* wgsl */ `
struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> lhs: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> rhs: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  let a = lhs[i];
  let b = rhs[i];
  lhs[i] = vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
`;
}

export class BluesteinAxis {
  constructor(device, cache, { shape, rank, batch, axis, direction, workgroupSize, maxWorkBytes = null }) {
    this.device = device;
    this.cache = cache;
    this.shape = shape;
    this.rank = rank;
    this.batch = batch;
    this.axis = axis;
    this.direction = direction;
    this.workgroupSize = workgroupSize;

    this.N = shape[axis];
    const sign = direction === "forward" ? -1.0 : 1.0;
    const M0 = nextSmoothAtLeast(2 * this.N - 1);
    this.M = factorizeSupportedRadices(M0) ? M0 : nextPow2(2 * this.N - 1);
    if (!factorizeSupportedRadices(this.M)) throw new Error(`Bluestein internal M=${this.M} not factorable by supported radices`);

    this.logicalTotal = prod(shape);
    this.lines = batch * (this.logicalTotal / this.N);
    let strideComplex = 1;
    for (let d = 0; d < axis; d++) strideComplex *= shape[d];
    this._strideComplex = strideComplex;
    this._workBytesPerLine = this.M * 8;
    const deviceMaxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
    const chunkBudget = maxWorkBytes == null ? deviceMaxBind : Math.min(deviceMaxBind, maxWorkBytes);
    this._bindBudgetBytes = chunkBudget;
    this._maxSliceElems = Math.max(1, Math.floor(chunkBudget / 8));
    this.maxChunkLines = Math.max(1, Math.floor(chunkBudget / this._workBytesPerLine));
    this.maxChunkLines = Math.min(this.maxChunkLines, this.lines);
    this.workBytes = this.maxChunkLines * this._workBytesPerLine;
    this._maxChunkCount = Math.max(1, Math.ceil(this.lines / this.maxChunkLines));
    this._paramStride = alignBytes(16, device.limits?.minUniformBufferOffsetAlignment ?? 256);
    this._paramCapacity = this._maxChunkCount;
    this._retiredParamBuffers = [];
    const needsSlicedWorkPath = this._strideComplex === 1 && (this.N * 8 > this._bindBudgetBytes || this.M * 8 > this._bindBudgetBytes);
    if (!needsSlicedWorkPath) {
      ensureWithinBindingLimit(device, this.workBytes, `Bluestein work buffer: N=${this.N} M=${this.M} lines=${this.lines}`);
    }

    // chirpA/C length N
    const chirpA = new Float32Array(2 * this.N);
    for (let n = 0; n < this.N; n++) {
      const ang = sign * (Math.PI * (n * n) / this.N);
      chirpA[2 * n] = Math.cos(ang);
      chirpA[2 * n + 1] = Math.sin(ang);
    }
    const b = new Float32Array(2 * this.M);
    b[0] = 1;
    b[1] = 0;
    for (let m = 1; m <= this.N - 1; m++) {
      const ang = -sign * (Math.PI * (m * m) / this.N);
      const re = Math.cos(ang);
      const im = Math.sin(ang);
      b[2 * m] = re;
      b[2 * m + 1] = im;
      b[2 * (this.M - m)] = re;
      b[2 * (this.M - m) + 1] = im;
    }

    this.chirpABuf = device.createBuffer({ size: chirpA.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.chirpCBuf = device.createBuffer({ size: chirpA.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.chirpABuf, 0, chirpA);
    device.queue.writeBuffer(this.chirpCBuf, 0, chirpA);

    this.bfftBuf = device.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(this.bfftBuf, 0, b);

    this.fftFwd = createFftPlan(device, { shape: [this.M], direction: "forward", normalize: "none", inPlace: true, layout: "interleaved", precision: "f32" });
    this.fftInv = createFftPlan(device, { shape: [this.M], direction: "inverse", normalize: "backward", inPlace: true, layout: "interleaved", precision: "f32" });

    // compute bfft once
    {
      const enc = device.createCommandEncoder();
      this.fftFwd.exec(enc, { input: this.bfftBuf, batch: 1 });
      device.queue.submit([enc.finish()]);
    }

    const preCode = generateBluesteinPreWGSL({
      rank,
      axis,
      dims: shape,
      axisLength: this.N,
      mLength: this.M,
      strideComplex,
      workgroupSize,
    });
    const mulCode = generateBluesteinMulBfftWGSL({ mLength: this.M, workgroupSize });
    const postCode = generateBluesteinPostWGSL({
      rank,
      axis,
      dims: shape,
      axisLength: this.N,
      mLength: this.M,
      strideComplex,
      workgroupSize,
    });

    this.bglPre = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.bglMul = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.bglPost = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.plPre = device.createPipelineLayout({ bindGroupLayouts: [this.bglPre] });
    this.plMul = device.createPipelineLayout({ bindGroupLayouts: [this.bglMul] });
    this.plPost = device.createPipelineLayout({ bindGroupLayouts: [this.bglPost] });
    this.prePipe = cache.getComputePipeline({ code: preCode, layout: this.plPre });
    this.mulPipe = cache.getComputePipeline({ code: mulCode, layout: this.plMul });
    this.postPipe = cache.getComputePipeline({ code: postCode, layout: this.plPost });

    this.paramsPre = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsMul = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsPost = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.paramsPre, 0, new Uint32Array([this.maxChunkLines, 0, 0, 0]));
    device.queue.writeBuffer(this.paramsPost, 0, new Uint32Array([this.maxChunkLines, 0, 0, 0]));
    device.queue.writeBuffer(this.paramsMul, 0, new Uint32Array([this.maxChunkLines * this.M, 0, 0, 0]));

    this.sliceWrite = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipeline = cache.getComputePipeline({ code: generateSliceMulWriteWGSL(workgroupSize), layout: pl });
      return { bgl, pl, pipeline };
    })();
    this.sliceMul = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipeline = cache.getComputePipeline({ code: generateSliceMulInPlaceWGSL(workgroupSize), layout: pl });
      return { bgl, pl, pipeline };
    })();
    this.sliceParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._sliceInputBuffer = null;
    this._sliceTwiddleBuffer = null;
    this._sliceOutputBuffer = null;
    this._sliceZeroBuffer = null;
    this._sliceBytes = 0;
    this._retiredSliceBuffers = [];
  }

  _ensureParamCapacity(requiredChunkCount) {
    if (requiredChunkCount <= this._paramCapacity) return;
    let nextCapacity = this._paramCapacity;
    while (nextCapacity < requiredChunkCount) nextCapacity *= 2;
    const nextBytes = nextCapacity * this._paramStride;
    this._retiredParamBuffers.push(this.paramsPre, this.paramsMul, this.paramsPost);
    this.paramsPre = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsMul = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsPost = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._paramCapacity = nextCapacity;
  }

  supportsBoundedLineSlicing() {
    return this._strideComplex === 1;
  }

  _ensureSliceBuffers(minBytes) {
    if (this._sliceInputBuffer && this._sliceBytes >= minBytes) return;
    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    if (minBytes > maxBufferSize) {
      throw new Error(`Bluestein sliced staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`);
    }
    const next = () =>
      this.device.createBuffer({
        size: minBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    if (this._sliceInputBuffer) this._retiredSliceBuffers.push(this._sliceInputBuffer);
    if (this._sliceTwiddleBuffer) this._retiredSliceBuffers.push(this._sliceTwiddleBuffer);
    if (this._sliceOutputBuffer) this._retiredSliceBuffers.push(this._sliceOutputBuffer);
    if (this._sliceZeroBuffer) this._retiredSliceBuffers.push(this._sliceZeroBuffer);
    this._sliceInputBuffer = next();
    this._sliceTwiddleBuffer = next();
    this._sliceOutputBuffer = next();
    this._sliceZeroBuffer = next();
    this._sliceBytes = minBytes;
  }

  _zeroWorkRange(commandEncoder, buffer, offsetBytes, sizeBytes) {
    if (typeof commandEncoder.clearBuffer === "function") {
      commandEncoder.clearBuffer(buffer, offsetBytes, sizeBytes);
      return;
    }
    this._ensureSliceBuffers(Math.min(this._maxSliceElems * 8, sizeBytes));
    let done = 0;
    while (done < sizeBytes) {
      const n = Math.min(this._sliceBytes, sizeBytes - done);
      commandEncoder.copyBufferToBuffer(this._sliceZeroBuffer, 0, buffer, offsetBytes + done, n);
      done += n;
    }
  }

  _runSliceWrite(commandEncoder, lhsBuf, rhsBuf, dstBuf, countElems) {
    this.device.queue.writeBuffer(this.sliceParams, 0, new Uint32Array([countElems, 0, 0, 0]));
    const bytes = countElems * 8;
    const bg = this.device.createBindGroup({
      layout: this.sliceWrite.bgl,
      entries: [
        { binding: 0, resource: { buffer: lhsBuf, offset: 0, size: bytes } },
        { binding: 1, resource: { buffer: rhsBuf, offset: 0, size: bytes } },
        { binding: 2, resource: { buffer: dstBuf, offset: 0, size: bytes } },
        { binding: 3, resource: { buffer: this.sliceParams, offset: 0, size: 16 } },
      ],
    });
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.sliceWrite.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(countElems / this.workgroupSize), 1, 1);
    pass.end();
  }

  _runSliceMulInPlace(commandEncoder, lhsBuf, rhsBuf, countElems) {
    this.device.queue.writeBuffer(this.sliceParams, 0, new Uint32Array([countElems, 0, 0, 0]));
    const bytes = countElems * 8;
    const bg = this.device.createBindGroup({
      layout: this.sliceMul.bgl,
      entries: [
        { binding: 0, resource: { buffer: lhsBuf, offset: 0, size: bytes } },
        { binding: 1, resource: { buffer: rhsBuf, offset: 0, size: bytes } },
        { binding: 2, resource: { buffer: this.sliceParams, offset: 0, size: 16 } },
      ],
    });
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.sliceMul.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(countElems / this.workgroupSize), 1, 1);
    pass.end();
  }

  _execLineSliced(commandEncoder, { dataBuf, lineDataOffsetBytes, workBuf, workOffsetBytes, scratch }) {
    const lineBytes = this.N * 8;
    const workLineBytes = this.M * 8;
    const maxSliceElems = Math.max(1, this._maxSliceElems);
    const maxSliceBytes = maxSliceElems * 8;
    this._ensureSliceBuffers(maxSliceBytes);
    this._zeroWorkRange(commandEncoder, workBuf, workOffsetBytes, workLineBytes);

    for (let t0 = 0; t0 < this.N; t0 += maxSliceElems) {
      const n = Math.min(maxSliceElems, this.N - t0);
      const bytes = n * 8;
      commandEncoder.copyBufferToBuffer(dataBuf, lineDataOffsetBytes + t0 * 8, this._sliceInputBuffer, 0, bytes);
      commandEncoder.copyBufferToBuffer(this.chirpABuf, t0 * 8, this._sliceTwiddleBuffer, 0, bytes);
      this._runSliceWrite(commandEncoder, this._sliceInputBuffer, this._sliceTwiddleBuffer, this._sliceOutputBuffer, n);
      commandEncoder.copyBufferToBuffer(this._sliceOutputBuffer, 0, workBuf, workOffsetBytes + t0 * 8, bytes);
    }

    this.fftFwd.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOffsetBytes, batch: 1, temp: scratch });

    for (let k0 = 0; k0 < this.M; k0 += maxSliceElems) {
      const n = Math.min(maxSliceElems, this.M - k0);
      const bytes = n * 8;
      commandEncoder.copyBufferToBuffer(workBuf, workOffsetBytes + k0 * 8, this._sliceInputBuffer, 0, bytes);
      commandEncoder.copyBufferToBuffer(this.bfftBuf, k0 * 8, this._sliceTwiddleBuffer, 0, bytes);
      this._runSliceMulInPlace(commandEncoder, this._sliceInputBuffer, this._sliceTwiddleBuffer, n);
      commandEncoder.copyBufferToBuffer(this._sliceInputBuffer, 0, workBuf, workOffsetBytes + k0 * 8, bytes);
    }

    this.fftInv.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOffsetBytes, batch: 1, temp: scratch });

    for (let t0 = 0; t0 < this.N; t0 += maxSliceElems) {
      const n = Math.min(maxSliceElems, this.N - t0);
      const bytes = n * 8;
      commandEncoder.copyBufferToBuffer(workBuf, workOffsetBytes + t0 * 8, this._sliceInputBuffer, 0, bytes);
      commandEncoder.copyBufferToBuffer(this.chirpCBuf, t0 * 8, this._sliceTwiddleBuffer, 0, bytes);
      this._runSliceWrite(commandEncoder, this._sliceInputBuffer, this._sliceTwiddleBuffer, this._sliceOutputBuffer, n);
      commandEncoder.copyBufferToBuffer(this._sliceOutputBuffer, 0, dataBuf, lineDataOffsetBytes + t0 * 8, bytes);
    }

    if (lineBytes > this._bindBudgetBytes || workLineBytes > this._bindBudgetBytes) {
      // Explicit marker for callers/tests to detect this bounded path.
      this._usedSlicedLinePath = true;
    }
  }

  destroy() {
    this.chirpABuf.destroy();
    this.chirpCBuf.destroy();
    this.bfftBuf.destroy();
    this.paramsPre.destroy();
    this.paramsMul.destroy();
    this.paramsPost.destroy();
    this.sliceParams.destroy();
    this._sliceInputBuffer?.destroy?.();
    this._sliceTwiddleBuffer?.destroy?.();
    this._sliceOutputBuffer?.destroy?.();
    this._sliceZeroBuffer?.destroy?.();
    for (const b of this._retiredSliceBuffers) b?.destroy?.();
    for (const b of this._retiredParamBuffers) b?.destroy?.();
    this.fftFwd.destroy();
    this.fftInv.destroy();
  }

  exec(commandEncoder, { dataBuf, dataOffsetBytes, axisWork, scratch, lineCount = this.lines, paramChunkBase = 0 }) {
    if (!Number.isInteger(lineCount) || lineCount < 1 || lineCount > this.lines) {
      throw new Error(`BluesteinAxis.exec lineCount must be in [1, ${this.lines}], got ${lineCount}`);
    }
    if (!Number.isInteger(paramChunkBase) || paramChunkBase < 0) {
      throw new Error(`BluesteinAxis.exec paramChunkBase must be a non-negative integer, got ${paramChunkBase}`);
    }
    // axisWork: GPUBuffer|BufferView, size >= workBytes
    const workRange = normalizeToContiguousRanges(axisWork, 0, this.workBytes)[0];
    const workBuf = workRange.buffer;
    const workOff = workRange.offsetBytes;
    const lineBytes = this.N * 8;
    const workLineBytes = this.M * 8;
    const needsSlicedLinePath = lineBytes > this._bindBudgetBytes || workLineBytes > this._bindBudgetBytes;
    if (needsSlicedLinePath) {
      if (!this.supportsBoundedLineSlicing()) {
        throw new Error(
          `Bluestein bounded-line slicing currently requires contiguous axis lines (strideComplex=1), got strideComplex=${this._strideComplex}`
        );
      }
      if (workRange.sizeBytes < workLineBytes) {
        throw new Error(`Bluestein axisWork is too small for sliced-line execution: need ${workLineBytes}, got ${workRange.sizeBytes}`);
      }
      this._usedSlicedLinePath = false;
      for (let line = 0; line < lineCount; line++) {
        const lineDataOffsetBytes = dataOffsetBytes + line * lineBytes;
        this._execLineSliced(commandEncoder, {
          dataBuf,
          lineDataOffsetBytes,
          workBuf,
          workOffsetBytes: workOff,
          scratch,
        });
      }
      return 0;
    }

    const dataSize = lineCount * this.N * 8;
    const chunkCount = Math.ceil(lineCount / this.maxChunkLines);
    this._ensureParamCapacity(paramChunkBase + chunkCount);
    let chunkIndex = 0;
    for (let line0 = 0; line0 < lineCount; line0 += this.maxChunkLines) {
      const lines = Math.min(this.maxChunkLines, lineCount - line0);
      const chunkWorkBytes = lines * this._workBytesPerLine;
      const paramOff = (paramChunkBase + chunkIndex) * this._paramStride;
      if (paramOff + 16 > this.paramsPre.size || paramOff + 16 > this.paramsPost.size || paramOff + 16 > this.paramsMul.size) {
        throw new Error("BluesteinAxis.exec parameter buffer overflow; increase chunk-parameter capacity");
      }
      this.device.queue.writeBuffer(this.paramsPre, paramOff, new Uint32Array([lines, line0, 0, 0]));
      this.device.queue.writeBuffer(this.paramsPost, paramOff, new Uint32Array([lines, line0, 0, 0]));
      this.device.queue.writeBuffer(this.paramsMul, paramOff, new Uint32Array([lines * this.M, 0, 0, 0]));

      // pre
      const bgPre = this.device.createBindGroup({
        layout: this.bglPre,
        entries: [
          { binding: 0, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
          { binding: 1, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
          { binding: 2, resource: { buffer: this.chirpABuf, offset: 0, size: this.N * 8 } },
          { binding: 3, resource: { buffer: this.paramsPre, offset: paramOff, size: 16 } },
        ],
      });
      {
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.prePipe);
        pass.setBindGroup(0, bgPre);
        pass.dispatchWorkgroups(Math.ceil((lines * this.M) / this.workgroupSize), 1, 1);
        pass.end();
      }

      // FFT fwd (in-place on workBuf)
      this.fftFwd.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOff, batch: lines, temp: scratch });

      // mul
      const bgMul = this.device.createBindGroup({
        layout: this.bglMul,
        entries: [
          { binding: 0, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
          { binding: 1, resource: { buffer: this.bfftBuf, offset: 0, size: this.M * 8 } },
          { binding: 2, resource: { buffer: this.paramsMul, offset: paramOff, size: 16 } },
        ],
      });
      {
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.mulPipe);
        pass.setBindGroup(0, bgMul);
        pass.dispatchWorkgroups(Math.ceil((lines * this.M) / this.workgroupSize), 1, 1);
        pass.end();
      }

      // FFT inv
      this.fftInv.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOff, batch: lines, temp: scratch });

      // post
      const bgPost = this.device.createBindGroup({
        layout: this.bglPost,
        entries: [
          { binding: 0, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
          { binding: 1, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
          { binding: 2, resource: { buffer: this.chirpCBuf, offset: 0, size: this.N * 8 } },
          { binding: 3, resource: { buffer: this.paramsPost, offset: paramOff, size: 16 } },
        ],
      });
      {
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.postPipe);
        pass.setBindGroup(0, bgPost);
        pass.dispatchWorkgroups(Math.ceil((lines * this.N) / this.workgroupSize), 1, 1);
        pass.end();
      }
      chunkIndex += 1;
    }
    return chunkIndex;
  }
}
