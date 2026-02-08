// Copyright (c) 2026 Maksim Eremenko

import { createFftPlan } from "../../plan.js";
import { isPrime, primitiveRootPrime, modPow, factorizeSupportedRadices, nextPow2, nextSmoothAtLeast } from "../../utils/factors.js";
import { ensureWithinBindingLimit, prod, alignBytes } from "../common.js";
import { normalizeToContiguousRanges } from "../segmented_io.js";

import {
  generateRaderSumWGSL,
  generateRaderPackARevWGSL,
  generateRaderMulBfftWGSL,
  generateRaderWriteY0WGSL,
  generateRaderPostWGSL,
} from "../../kernels/rader.js";

export class RaderAxis {
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
    if (!isPrime(this.N)) throw new Error(`Rader requires prime N, got ${this.N}`);
    // Forward DFT uses exp(-i*2π/N); inverse uses exp(+i*2π/N).
    // This sign drives the Rader "b" sequence twiddles.
    const sign = direction === "forward" ? 1.0 : -1.0;
    this.L = this.N - 1;
    const M0 = nextSmoothAtLeast(2 * this.L - 1);
    this.M = factorizeSupportedRadices(M0) ? M0 : nextPow2(2 * this.L - 1);
    if (!factorizeSupportedRadices(this.M)) throw new Error(`Rader internal M=${this.M} not factorable by supported radices`);

    this.logicalTotal = prod(shape);
    this.lines = batch * (this.logicalTotal / this.N);
    this._workBytesPerLine = this.M * 8;
    const deviceMaxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
    const chunkBudget = maxWorkBytes == null ? deviceMaxBind : Math.min(deviceMaxBind, maxWorkBytes);
    this.maxChunkLines = Math.max(1, Math.floor(chunkBudget / this._workBytesPerLine));
    this.maxChunkLines = Math.min(this.maxChunkLines, this.lines);
    this.workBytes = this.maxChunkLines * this._workBytesPerLine;
    this._maxChunkCount = Math.max(1, Math.ceil(this.lines / this.maxChunkLines));
    this._paramStride = alignBytes(16, device.limits?.minUniformBufferOffsetAlignment ?? 256);
    this._paramCapacity = this._maxChunkCount;
    this._retiredParamBuffers = [];
    ensureWithinBindingLimit(device, this.workBytes, `Rader work buffer: N=${this.N} M=${this.M} lines=${this.lines}`);

    const g = primitiveRootPrime(this.N);
    const perm = new Uint32Array(this.L);
    for (let k = 0; k < this.L; k++) perm[k] = modPow(g, k + 1, this.N);
    this.permBuf = device.createBuffer({ size: perm.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.permBuf, 0, perm);

    const b = new Float32Array(2 * this.M);
    for (let k = 0; k < this.L; k++) {
      const ang = sign * (-2.0 * Math.PI * perm[k] / this.N);
      b[2 * k] = Math.cos(ang);
      b[2 * k + 1] = Math.sin(ang);
    }
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

    let strideComplex = 1;
    for (let d = 0; d < axis; d++) strideComplex *= shape[d];

    this.sumBuf = device.createBuffer({ size: this.maxChunkLines * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.x0Buf = device.createBuffer({ size: this.maxChunkLines * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const sumCode = generateRaderSumWGSL({ rank, axis, dims: shape, axisLength: this.N, strideComplex, workgroupSize: 256 });
    const packCode = generateRaderPackARevWGSL({ rank, axis, dims: shape, axisLength: this.N, mLength: this.M, strideComplex, workgroupSize });
    const mulCode = generateRaderMulBfftWGSL({ mLength: this.M, workgroupSize });
    const y0Code = generateRaderWriteY0WGSL({ rank, axis, dims: shape, axisLength: this.N, strideComplex, workgroupSize });
    const postCode = generateRaderPostWGSL({ rank, axis, dims: shape, axisLength: this.N, mLength: this.M, strideComplex, workgroupSize });

    this.bglSum = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.bglPack = device.createBindGroupLayout({
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
    this.bglY0 = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.bglPost = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.plSum = device.createPipelineLayout({ bindGroupLayouts: [this.bglSum] });
    this.plPack = device.createPipelineLayout({ bindGroupLayouts: [this.bglPack] });
    this.plMul = device.createPipelineLayout({ bindGroupLayouts: [this.bglMul] });
    this.plY0 = device.createPipelineLayout({ bindGroupLayouts: [this.bglY0] });
    this.plPost = device.createPipelineLayout({ bindGroupLayouts: [this.bglPost] });

    this.sumPipe = cache.getComputePipeline({ code: sumCode, layout: this.plSum });
    this.packPipe = cache.getComputePipeline({ code: packCode, layout: this.plPack });
    this.mulPipe = cache.getComputePipeline({ code: mulCode, layout: this.plMul });
    this.y0Pipe = cache.getComputePipeline({ code: y0Code, layout: this.plY0 });
    this.postPipe = cache.getComputePipeline({ code: postCode, layout: this.plPost });

    this.paramsLines = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.paramsLines, 0, new Uint32Array([this.maxChunkLines, 0, 0, 0]));
    this.paramsMul = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.paramsMul, 0, new Uint32Array([this.maxChunkLines * this.M, 0, 0, 0]));
  }

  _ensureParamCapacity(requiredChunkCount) {
    if (requiredChunkCount <= this._paramCapacity) return;
    let nextCapacity = this._paramCapacity;
    while (nextCapacity < requiredChunkCount) nextCapacity *= 2;
    const nextBytes = nextCapacity * this._paramStride;
    this._retiredParamBuffers.push(this.paramsLines, this.paramsMul);
    this.paramsLines = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsMul = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._paramCapacity = nextCapacity;
  }

  destroy() {
    this.permBuf.destroy();
    this.bfftBuf.destroy();
    this.sumBuf.destroy();
    this.x0Buf.destroy();
    this.paramsLines.destroy();
    this.paramsMul.destroy();
    for (const b of this._retiredParamBuffers) b?.destroy?.();
    this.fftFwd.destroy();
    this.fftInv.destroy();
  }

  exec(commandEncoder, { dataBuf, dataOffsetBytes, axisWork, scratch, lineCount = this.lines, paramChunkBase = 0 }) {
    if (!Number.isInteger(lineCount) || lineCount < 1 || lineCount > this.lines) {
      throw new Error(`RaderAxis.exec lineCount must be in [1, ${this.lines}], got ${lineCount}`);
    }
    if (!Number.isInteger(paramChunkBase) || paramChunkBase < 0) {
      throw new Error(`RaderAxis.exec paramChunkBase must be a non-negative integer, got ${paramChunkBase}`);
    }
    const workRange = normalizeToContiguousRanges(axisWork, 0, this.workBytes)[0];
    const workBuf = workRange.buffer;
    const workOff = workRange.offsetBytes;
    const dataSize = lineCount * this.N * 8;
    const chunkCount = Math.ceil(lineCount / this.maxChunkLines);
    this._ensureParamCapacity(paramChunkBase + chunkCount);

    let chunkIndex = 0;
    for (let line0 = 0; line0 < lineCount; line0 += this.maxChunkLines) {
      const lines = Math.min(this.maxChunkLines, lineCount - line0);
      const chunkWorkBytes = lines * this._workBytesPerLine;
      const chunkLineBytes = lines * 8;
      const paramOff = (paramChunkBase + chunkIndex) * this._paramStride;
      if (paramOff + 16 > this.paramsLines.size || paramOff + 16 > this.paramsMul.size) {
        throw new Error("RaderAxis.exec parameter buffer overflow; increase chunk-parameter capacity");
      }
      this.device.queue.writeBuffer(this.paramsLines, paramOff, new Uint32Array([lines, line0, 0, 0]));
      this.device.queue.writeBuffer(this.paramsMul, paramOff, new Uint32Array([lines * this.M, 0, 0, 0]));

      // sum + x0
      {
        const bg = this.device.createBindGroup({
          layout: this.bglSum,
          entries: [
            { binding: 0, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
            { binding: 1, resource: { buffer: this.sumBuf, offset: 0, size: chunkLineBytes } },
            { binding: 2, resource: { buffer: this.x0Buf, offset: 0, size: chunkLineBytes } },
            { binding: 3, resource: { buffer: this.paramsLines, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.sumPipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(lines, 1, 1);
        pass.end();
      }

      // pack a_rev
      {
        const bg = this.device.createBindGroup({
          layout: this.bglPack,
          entries: [
            { binding: 0, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
            { binding: 1, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
            { binding: 2, resource: { buffer: this.permBuf, offset: 0, size: this.L * 4 } },
            { binding: 3, resource: { buffer: this.paramsLines, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.packPipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil((lines * this.M) / this.workgroupSize), 1, 1);
        pass.end();
      }

      // FFT fwd
      this.fftFwd.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOff, batch: lines, temp: scratch });

      // mul
      {
        const bg = this.device.createBindGroup({
          layout: this.bglMul,
          entries: [
            { binding: 0, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
            { binding: 1, resource: { buffer: this.bfftBuf, offset: 0, size: this.M * 8 } },
            { binding: 2, resource: { buffer: this.paramsMul, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.mulPipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil((lines * this.M) / this.workgroupSize), 1, 1);
        pass.end();
      }

      // FFT inv
      this.fftInv.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOff, batch: lines, temp: scratch });

      // y0
      {
        const bg = this.device.createBindGroup({
          layout: this.bglY0,
          entries: [
            { binding: 0, resource: { buffer: this.sumBuf, offset: 0, size: chunkLineBytes } },
            { binding: 1, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
            { binding: 2, resource: { buffer: this.paramsLines, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.y0Pipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(lines / this.workgroupSize), 1, 1);
        pass.end();
      }

      // post
      {
        const bg = this.device.createBindGroup({
          layout: this.bglPost,
          entries: [
            { binding: 0, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
            { binding: 1, resource: { buffer: this.x0Buf, offset: 0, size: chunkLineBytes } },
            { binding: 2, resource: { buffer: this.permBuf, offset: 0, size: this.L * 4 } },
            { binding: 3, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
            { binding: 4, resource: { buffer: this.paramsLines, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.postPipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil((lines * (this.N - 1)) / this.workgroupSize), 1, 1);
        pass.end();
      }
      chunkIndex += 1;
    }
    return chunkIndex;
  }
}
