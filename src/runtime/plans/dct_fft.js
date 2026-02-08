// Copyright (c) 2026 Maksim Eremenko

import { createFftPlan } from "../../plan.js";
import { BasePlan } from "../base_plan.js";
import { createInternalArena, viewFromArena } from "../workspace.js";
import { normalizeToContiguousRanges } from "../segmented_io.js";
import { normalizeIoView } from "../ioview.js";
import { normalizeZeroPad } from "../zero_pad.js";
import { mergeLargeRouteMetadata, resolveLargeRoutingPolicy } from "../large_policy.js";
import { resolveLayoutSemantics } from "../layout_semantics.js";
import { C2CPlan } from "./c2c.js";
import { factorizeSupportedRadices } from "../../utils/factors.js";
import {
  coordsFromLinear as tensorCoordsFromLinear,
  createTensorDescriptor,
  requiredBytesForBatchRange,
} from "../tensor_descriptor.js";

import {
  assertOneOf,
  isPositiveInt,
  prod,
  normalizeScaleFactor,
  ensureWithinBindingLimit,
  getBufferByteLength,
  alignBytes,
  align4Bytes,
  buffersAlias,
  isGpuBuffer,
} from "../common.js";

import { generateScaleRealWGSL } from "../../kernels/scale.js";
import { generateZeroOutsideRangeRealWGSL } from "../../kernels/zero_pad.js";
import { generateEmbedRealWGSL, generateExtractRealWGSL } from "../../kernels/ioview.js";
import { generateF16ToF32RealWGSL, generateF32ToF16RealWGSL } from "../../kernels/f16_storage.js";
import { generateGatherRealStridedWGSL, generateScatterRealStridedWGSL } from "../../kernels/strided_real.js";
import { dctFftDirection, dctWorkLength, generateDctFftBuildWGSL, generateDctFftPostWGSL } from "../../kernels/dct_fft.js";

function needsIoMapping(io, logicalShape) {
  if (!io) return false;
  for (let i = 0; i < logicalShape.length; i++) {
    if (io.shape[i] !== logicalShape[i]) return true;
    if (io.offset[i] !== 0) return true;
  }
  return false;
}

function typeKindFor(type, direction) {
  if (type === "dct1") return "dct1";
  if (type === "dct4") return "dct4";
  if (type === "dct2") return direction === "forward" ? "dct2_fwd" : "dct2_inv";
  if (type === "dct3") return direction === "forward" ? "dct2_inv" : "dct2_fwd";
  if (type === "dst1") return "dst1";
  if (type === "dst4") return "dst4";
  if (type === "dst2") return direction === "forward" ? "dst2_fwd" : "dst2_inv";
  if (type === "dst3") return direction === "forward" ? "dst2_inv" : "dst2_fwd";
  throw new Error(`Unknown DCT/DST type ${type}`);
}

function axisStride(dims, axis) {
  let s = 1;
  for (let i = 0; i < axis; i++) s *= dims[i];
  return s;
}

export class DctPlan extends BasePlan {
  constructor(device, opts) {
    super(device, opts);
    const {
      shape,
      type,
      direction = "forward",
      batch = 1,
      inPlace = false,
      normalize = "none",
      layout = { interleavedComplex: false },
      precision = "f32",
      ioView = null,
      zeroPad = null,
    } = opts ?? {};

    assertOneOf(type, ["dct1", "dct2", "dct3", "dct4", "dst1", "dst2", "dst3", "dst4"], "type");
    assertOneOf(direction, ["forward", "inverse"], "direction");
    if (!Array.isArray(shape) || shape.length < 1) throw new Error("shape must be rank >= 1");
    if (!shape.every(isPositiveInt)) throw new Error("shape must be positive ints");
    if (shape.some((n) => n < 2)) throw new Error(`All DCT/DST dimensions must be >= 2; got shape=${JSON.stringify(shape)}`);
    if (layout?.interleavedComplex !== false) throw new Error("DCT/DST uses real buffers; set layout.interleavedComplex=false");
    assertOneOf(normalize, ["none", "backward", "unitary"], "normalize");
    assertOneOf(precision, ["f32", "f16-storage"], "precision");
    if (precision === "f16-storage" && !device.features?.has?.("shader-f16")) throw new Error('precision="f16-storage" requires shader-f16');
    if (inPlace) throw new Error("DCT/DST inPlace is not supported in current implementation");
    if (!Number.isInteger(batch) || batch <= 0) throw new Error(`batch must be positive int; got ${batch}`);

    this.type = type;
    this.direction = direction;
    this.typeKind = typeKindFor(type, direction);

    this.shape = shape.slice();
    this.rank = shape.length;
    this.batch = batch;
    this.normalize = normalize;
    this.precision = precision;

    this.io = normalizeIoView(this.rank, this.shape, ioView ?? {});
    this.zeroPad = normalizeZeroPad(this.rank, this.shape, zeroPad ?? null, "zeroPad");
    this.inViewShape = (this.io.input?.shape ?? this.shape).slice();
    this.outViewShape = (this.io.output?.shape ?? this.shape).slice();
    this._inputLayoutShape = this.inViewShape.slice();
    this._outputLayoutShape = this.outViewShape.slice();

    this.logicalTotal = prod(this.shape);
    this.totalReal = this.logicalTotal * this.batch;
    this.logicalBytesF32 = this.totalReal * 4;
    this._logicalBytesPerBatchF32 = this.logicalTotal * 4;

    this._inViewPerBatch = prod(this.inViewShape);
    this._outViewPerBatch = prod(this.outViewShape);
    this.inViewTotal = this._inViewPerBatch * this.batch;
    this.outViewTotal = this._outViewPerBatch * this.batch;
    this._inBytesPerBatchRaw = precision === "f16-storage" ? this._inViewPerBatch * 2 : this._inViewPerBatch * 4;
    this._outBytesPerBatchRaw = precision === "f16-storage" ? this._outViewPerBatch * 2 : this._outViewPerBatch * 4;
    this._inBytesPerBatchBind = precision === "f16-storage" ? align4Bytes(this._inBytesPerBatchRaw) : this._inBytesPerBatchRaw;
    this._outBytesPerBatchBind = precision === "f16-storage" ? align4Bytes(this._outBytesPerBatchRaw) : this._outBytesPerBatchRaw;
    this.inBytes = precision === "f16-storage" ? align4Bytes(this.inViewTotal * 2) : this.inViewTotal * 4;
    this.outBytes = precision === "f16-storage" ? align4Bytes(this.outViewTotal * 2) : this.outViewTotal * 4;

    const resolvedLayout = resolveLayoutSemantics({
      layout,
      rank: this.rank,
      inputShape: this._inputLayoutShape,
      outputShape: this._outputLayoutShape,
    });
    this._inputStrides = resolvedLayout.inputStrides;
    this._outputStrides = resolvedLayout.outputStrides;
    this._inputOffsetElements = resolvedLayout.inputOffsetElements;
    this._outputOffsetElements = resolvedLayout.outputOffsetElements;
    this._inputBatchStrideElements = resolvedLayout.inputBatchStrideElements;
    this._outputBatchStrideElements = resolvedLayout.outputBatchStrideElements;
    this._usesStridedInput = resolvedLayout.usesStridedInput;
    this._usesStridedOutput = resolvedLayout.usesStridedOutput;
    this._usesWhdcnInput = resolvedLayout.usesWhdcnInput;
    this._usesWhdcnOutput = resolvedLayout.usesWhdcnOutput;
    this._inputTensorDesc = this._usesStridedInput
      ? createTensorDescriptor({
          name: "dct.input",
          shape: this._inputLayoutShape,
          strides: this._inputStrides,
          offsetElements: this._inputOffsetElements,
          batchStrideElements: this._inputBatchStrideElements,
        })
      : null;
    this._outputTensorDesc = this._usesStridedOutput
      ? createTensorDescriptor({
          name: "dct.output",
          shape: this._outputLayoutShape,
          strides: this._outputStrides,
          offsetElements: this._outputOffsetElements,
          batchStrideElements: this._outputBatchStrideElements,
        })
      : null;
    this._inputSpanElements = this._inputTensorDesc?.spanElements ?? 0;
    this._outputSpanElements = this._outputTensorDesc?.spanElements ?? 0;
    if ((this._usesStridedInput || this._usesStridedOutput) && this.precision !== "f32") {
      throw new Error('custom strides for dct/dst currently support precision:"f32" only');
    }

    // Complex work buffer max across axes.
    this.workBytesMax = 0;
    this._workBytesPerBatchMax = 0;

    // Axis ops: build -> FFT -> post
    this.axes = [];
    this.fftScratchBytes = 0;
    for (let axis = 0; axis < this.rank; axis++) {
      const axisLen = this.shape[axis];
      const stride = axisStride(this.shape, axis);
      const linesPerBatch = this.logicalTotal / axisLen;
      const lines = this.batch * linesPerBatch;
      if (!Number.isInteger(lines)) throw new Error("internal error: lines is not integer");
      const M = dctWorkLength(this.typeKind, axisLen);
      const workElems = lines * M;
      const workElemsPerBatch = linesPerBatch * M;
      const workBytes = workElems * 8;
      const workBytesPerBatch = workElemsPerBatch * 8;
      if (workBytes > this.workBytesMax) this.workBytesMax = workBytes;
      if (workBytesPerBatch > this._workBytesPerBatchMax) this._workBytesPerBatchMax = workBytesPerBatch;

      const buildBgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const buildPl = device.createPipelineLayout({ bindGroupLayouts: [buildBgl] });
      const buildCode = generateDctFftBuildWGSL({
        typeKind: this.typeKind,
        rank: this.rank,
        axis,
        dims: this.shape,
        axisLength: axisLen,
        workgroupSize: this.workgroupSize,
      });
      const buildPipe = this.cache.getComputePipeline({ code: buildCode, layout: buildPl });
      const buildParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buildParams, 0, new Uint32Array([workElems, 0, 0, 0]));

      const postBgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const postPl = device.createPipelineLayout({ bindGroupLayouts: [postBgl] });
      const postCode = generateDctFftPostWGSL({
        typeKind: this.typeKind,
        rank: this.rank,
        axis,
        dims: this.shape,
        axisLength: axisLen,
        workgroupSize: this.workgroupSize,
      });
      const postPipe = this.cache.getComputePipeline({ code: postCode, layout: postPl });
      const postParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(postParams, 0, new Uint32Array([this.totalReal, 0, 0, 0]));

      const fftDir = dctFftDirection(this.typeKind);
      let fft = null;
      let fftNeedsBatch = false;
      let fftRunsPerBatch = false;
      if (factorizeSupportedRadices(M)) {
        fftNeedsBatch = true;
        fft = createFftPlan(device, {
          shape: [M],
          direction: fftDir,
          normalize: "none",
          inPlace: true,
          layout: "interleaved",
          precision: "f32",
        });
      } else {
        // Fallback to high-level planner so arbitrary lengths (e.g. DST-I with M=2*(N+1))
        // can use Bluestein/Rader internally. Keep this as per-logical-batch so large-mode
        // chunk scheduling and regular per-batch fallback both remain binding-safe.
        fftRunsPerBatch = true;
        fft = new C2CPlan(device, {
          shape: [M],
          direction: fftDir,
          batch: linesPerBatch,
          inPlace: true,
          normalize: "none",
          layout: { interleavedComplex: true },
          precision: "f32",
          ioView: { input: null, output: null },
          tuning: opts?.tuning ?? null,
        });
      }

      const fftWorkspaceBytes =
        typeof fft.getWorkspaceSizeBytes === "function"
          ? fft.getWorkspaceSizeBytes()
          : workBytes;
      this.fftScratchBytes = Math.max(this.fftScratchBytes, fftWorkspaceBytes);

      this.axes.push({
        axis,
        axisLen,
        stride,
        lines,
        linesPerBatch,
        M,
        workElems,
        workElemsPerBatch,
        workBytes,
        workBytesPerBatch,
        build: { bgl: buildBgl, pl: buildPl, pipeline: buildPipe, params: buildParams },
        post: { bgl: postBgl, pl: postPl, pipeline: postPipe, params: postParams },
        fft,
        fftNeedsBatch,
        fftRunsPerBatch,
      });
    }
    const largePolicy = resolveLargeRoutingPolicy({
      device,
      tuning: opts?.tuning ?? null,
      requiredBindingBytes: [this.logicalBytesF32, this.inBytes, this.outBytes, this.workBytesMax],
      lineBytes: this.shape.map((n) => n * 4),
      precision: this.precision,
    });
    this._maxBindBytes = largePolicy.maxBindBytes;
    this._largeBatchChunkMode = largePolicy.needsLargeMode;
    const mergedRoute = mergeLargeRouteMetadata([
      largePolicy,
      ...this.axes.map((ax) => ({
        routeMode: ax.fft?._largeRouteMode,
        reasonCodes: ax.fft?._largeRouteReasons,
        attemptedRoutes: ax.fft?._largeRouteAttempts,
      })),
    ]);
    this._largeRouteMode = mergedRoute.routeMode;
    this._largeRouteReasons = mergedRoute.reasonCodes;
    this._largeRouteAttempts = mergedRoute.attemptedRoutes;
    this._storageAlignBytes = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    this._storageAlignElems = Math.max(1, Math.floor(this._storageAlignBytes / 4));

    if (this._largeBatchChunkMode) {
      const perBatchRequirements = [
        this._logicalBytesPerBatchF32,
        this._inBytesPerBatchBind,
        this._outBytesPerBatchBind,
        this._workBytesPerBatchMax,
      ];
      if (this._usesStridedInput) perBatchRequirements.push((this._inputSpanElements + this._storageAlignElems - 1) * 4);
      if (this._usesStridedOutput) perBatchRequirements.push((this._outputSpanElements + this._storageAlignElems - 1) * 4);
      if (perBatchRequirements.some((bytes) => bytes > this._maxBindBytes)) {
        throw new Error(
          `DCT/DST large-mode requires one-batch bindings to fit maxStorageBufferBindingSize=${this._maxBindBytes} ` +
            `(perBatch=${JSON.stringify(perBatchRequirements)}). ` +
            `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
        );
      }
      this.workBytesMax = this._workBytesPerBatchMax;
    } else {
      ensureWithinBindingLimit(device, this.logicalBytesF32, `DCT logical buffer: shape=${JSON.stringify(shape)} batch=${batch}`);
      ensureWithinBindingLimit(device, this.inBytes, "DCT input");
      ensureWithinBindingLimit(device, this.outBytes, "DCT output");
      ensureWithinBindingLimit(device, this.workBytesMax, "DCT complex work");
    }

    // scale (applied once after final axis)
    this.scale = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipeline = this.cache.getComputePipeline({ code: generateScaleRealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
      const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      return { bgl, pl: pipelineLayout, pipeline, params };
    })();

    // ioView mapping (input embed, output extract)
    this.ioEmbed = null;
    if (this.io.input && needsIoMapping(this.io.input, this.shape)) {
      const code = generateEmbedRealWGSL({
        rank: this.rank,
        logicalDims: this.shape,
        viewDims: this.io.input.shape,
        offset: this.io.input.offset,
        workgroupSize: this.workgroupSize,
      });
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.ioEmbed = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.io.input.shape) };
    }
    this.ioExtract = null;
    if (this.io.output && needsIoMapping(this.io.output, this.shape)) {
      const code = generateExtractRealWGSL({
        rank: this.rank,
        logicalDims: this.shape,
        viewDims: this.io.output.shape,
        offset: this.io.output.offset,
        clearOutside: this.io.output.clearOutside,
        workgroupSize: this.workgroupSize,
      });
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.ioExtract = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.io.output.shape) };
    }

    this.zeroRead = null;
    if (this.zeroPad.read) {
      const bgl = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateZeroOutsideRangeRealWGSL({
        shape: this.shape,
        start: this.zeroPad.read.start,
        end: this.zeroPad.read.end,
        batch: this.batch,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      this.zeroRead = { bgl, pl, pipeline };
    }

    this.zeroWrite = null;
    if (this.zeroPad.write) {
      const bgl = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateZeroOutsideRangeRealWGSL({
        shape: this.shape,
        start: this.zeroPad.write.start,
        end: this.zeroPad.write.end,
        batch: this.batch,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      this.zeroWrite = { bgl, pl, pipeline };
    }

    this.stridedIn = null;
    this.stridedOut = null;
    if (this._usesStridedInput) {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateGatherRealStridedWGSL({
        shape: this._inputLayoutShape,
        strides: this._inputStrides,
        baseOffsetElements: 0,
        batchStrideElements: this._inputBatchStrideElements,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.stridedIn = { bgl, pl, pipeline, params };
    }

    if (this._usesStridedOutput) {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateScatterRealStridedWGSL({
        shape: this._outputLayoutShape,
        strides: this._outputStrides,
        baseOffsetElements: 0,
        batchStrideElements: this._outputBatchStrideElements,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.stridedOut = { bgl, pl, pipeline, params };
    }

    // f16 I/O conversion (real)
    this.f16 = null;
    if (precision === "f16-storage") {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const toF32 = this.cache.getComputePipeline({ code: generateF16ToF32RealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
      const toF16 = this.cache.getComputePipeline({ code: generateF32ToF16RealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.f16 = { bgl, pl: pipelineLayout, toF32, toF16, params };
    }

    // Workspace layout:
    // [stage scratch] [dataA f32] [dataB f32] [workComplex f32] [fftScratch] [optional f16 out scratch]
    // When monolithic allocation exceeds maxBufferSize, this plan falls back to split internal section buffers.
    const stageInViewElems = this._largeBatchChunkMode ? this._inViewPerBatch : this.inViewTotal;
    const stageOutViewElems = this._largeBatchChunkMode ? this._outViewPerBatch : this.outViewTotal;
    this.stageInF32Bytes = this.ioEmbed ? stageInViewElems * 4 : 0;
    this.stageOutF32Bytes = this.ioExtract ? stageOutViewElems * 4 : 0;
    this.stageF16Bytes = precision === "f16-storage" ? (this._largeBatchChunkMode ? this._inBytesPerBatchBind : this.inBytes) : 0;
    const stageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    this.stageF16Offset = alignBytes(this.stageInF32Bytes, stageAlign);
    this.stageBytes = Math.max(this.stageOutF32Bytes, this.stageF16Offset + this.stageF16Bytes);

    this.dataBytes = this._largeBatchChunkMode ? this._logicalBytesPerBatchF32 : this.logicalBytesF32;
    this.fftScratchBytes = Math.max(this.fftScratchBytes, this.workBytesMax);

    // Storage bindings require offsets aligned to device.limits.minStorageBufferOffsetAlignment (usually 256).
    // Complex (vec2<f32>) also wants 8-byte alignment, which is implied by that limit on most devices.
    const a = Math.max(8, device.limits?.minStorageBufferOffsetAlignment ?? 256);
    let off = 0;
    this.stageOffset = 0;
    off = this.stageBytes;
    off = alignBytes(off, a);
    this.dataAOffset = off;
    off += this.dataBytes;
    off = alignBytes(off, a);
    this.dataBOffset = off;
    off += this.dataBytes;
    off = alignBytes(off, a);
    this.workOffset = off;
    off += this.workBytesMax;
    off = alignBytes(off, a);
    this.fftScratchOffset = off;
    off += this.fftScratchBytes;
    off = alignBytes(off, a);
    this.f16OutOffset = off;
    this.f16OutBytes = precision === "f16-storage" ? (this._largeBatchChunkMode ? this._outBytesPerBatchBind : this.outBytes) : 0;
    off += this.f16OutBytes;
    this.workspaceBytes = off;
    this._splitWorkspace = null;
    this._maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
    if (this.workspaceBytes <= this._maxBufferSize) {
      this._arena = createInternalArena(device, this.workspaceBytes);
    } else {
      const splitNeeds = [
        ["stage", this.stageBytes],
        ["dataA", this.dataBytes],
        ["dataB", this.dataBytes],
        ["work", this.workBytesMax],
        ["fftScratch", this.fftScratchBytes],
        ["f16Out", this.f16OutBytes],
      ];
      for (const [name, bytes] of splitNeeds) {
        if (bytes > 0 && bytes > this._maxBufferSize) {
          throw new Error(
            `dct/dst split workspace cannot allocate ${name} buffer: ${bytes} bytes exceeds device.limits.maxBufferSize=${this._maxBufferSize}. ` +
              `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
          );
        }
      }
      this._arena = null;
      this._splitWorkspace = {
        stage: this.stageBytes ? createInternalArena(device, this.stageBytes) : null,
        dataA: createInternalArena(device, this.dataBytes),
        dataB: createInternalArena(device, this.dataBytes),
        work: createInternalArena(device, this.workBytesMax),
        fftScratch: createInternalArena(device, this.fftScratchBytes),
        f16Out: this.f16OutBytes ? createInternalArena(device, this.f16OutBytes) : null,
      };
    }
  }

  getWorkspaceSizeBytes() {
    return this.workspaceBytes;
  }

  destroy() {
    if (this._destroyed) return;
    for (const ax of this.axes) {
      ax.build.params.destroy();
      ax.post.params.destroy();
      ax.fft.destroy();
    }
    this.scale.params.destroy();
    this.ioEmbed?.params?.destroy?.();
    this.ioExtract?.params?.destroy?.();
    this.stridedIn?.params?.destroy?.();
    this.stridedOut?.params?.destroy?.();
    this.f16?.params?.destroy?.();
    this._splitWorkspace?.stage?.destroy?.();
    this._splitWorkspace?.dataA?.destroy?.();
    this._splitWorkspace?.dataB?.destroy?.();
    this._splitWorkspace?.work?.destroy?.();
    this._splitWorkspace?.fftScratch?.destroy?.();
    this._splitWorkspace?.f16Out?.destroy?.();
    this._arena?.destroy?.();
    super.destroy();
  }

  _resolveWorkspaceViews(arenaLike) {
    if (arenaLike) {
      if (getBufferByteLength(arenaLike) < this.workspaceBytes) throw new Error("temp too small");
      return {
        stage: this.stageBytes ? viewFromArena(arenaLike, this.stageOffset, this.stageBytes) : null,
        dataA: viewFromArena(arenaLike, this.dataAOffset, this.dataBytes),
        dataB: viewFromArena(arenaLike, this.dataBOffset, this.dataBytes),
        work: viewFromArena(arenaLike, this.workOffset, this.workBytesMax),
        fftScratch: viewFromArena(arenaLike, this.fftScratchOffset, this.fftScratchBytes),
        f16OutScratch: this.f16OutBytes ? viewFromArena(arenaLike, this.f16OutOffset, this.f16OutBytes) : null,
      };
    }
    if (this._splitWorkspace) {
      return {
        stage: this.stageBytes ? viewFromArena(this._splitWorkspace.stage, 0, this.stageBytes) : null,
        dataA: viewFromArena(this._splitWorkspace.dataA, 0, this.dataBytes),
        dataB: viewFromArena(this._splitWorkspace.dataB, 0, this.dataBytes),
        work: viewFromArena(this._splitWorkspace.work, 0, this.workBytesMax),
        fftScratch: viewFromArena(this._splitWorkspace.fftScratch, 0, this.fftScratchBytes),
        f16OutScratch: this.f16OutBytes ? viewFromArena(this._splitWorkspace.f16Out, 0, this.f16OutBytes) : null,
      };
    }
    throw new Error("No workspace buffer");
  }

  _normalizeCopyView(x) {
    if (x && x.buffer && Number.isInteger(x.offsetBytes) && Number.isInteger(x.sizeBytes)) {
      return {
        segments: [{ buffer: x.buffer, offsetBytes: x.offsetBytes, sizeBytes: x.sizeBytes }],
        logicalByteOffset: 0,
        lengthBytes: x.sizeBytes,
      };
    }
    return x;
  }

  _copyAnySpan(commandEncoder, { src, srcOffsetBytes, dst, dstOffsetBytes, bytes }) {
    if (bytes <= 0) return;
    const srcRanges = normalizeToContiguousRanges(this._normalizeCopyView(src), srcOffsetBytes, bytes);
    const dstRanges = normalizeToContiguousRanges(this._normalizeCopyView(dst), dstOffsetBytes, bytes);
    if (srcRanges.length === 1 && dstRanges.length === 1) {
      const s = srcRanges[0];
      const d = dstRanges[0];
      if (s.buffer === d.buffer && s.offsetBytes === d.offsetBytes) return;
      commandEncoder.copyBufferToBuffer(s.buffer, s.offsetBytes, d.buffer, d.offsetBytes, bytes);
      return;
    }
    if (srcRanges.length > 1 && dstRanges.length === 1) {
      this.copier.pack(commandEncoder, srcRanges, dstRanges[0].buffer, dstRanges[0].offsetBytes);
      return;
    }
    if (srcRanges.length === 1 && dstRanges.length > 1) {
      this.copier.unpack(commandEncoder, srcRanges[0].buffer, srcRanges[0].offsetBytes, dstRanges);
      return;
    }
    let si = 0;
    let di = 0;
    let soff = srcRanges[0].offsetBytes;
    let doff = dstRanges[0].offsetBytes;
    let srem = srcRanges[0].sizeBytes;
    let drem = dstRanges[0].sizeBytes;
    while (si < srcRanges.length && di < dstRanges.length) {
      const n = Math.min(srem, drem);
      commandEncoder.copyBufferToBuffer(srcRanges[si].buffer, soff, dstRanges[di].buffer, doff, n);
      soff += n;
      doff += n;
      srem -= n;
      drem -= n;
      if (srem === 0) {
        si += 1;
        if (si < srcRanges.length) {
          soff = srcRanges[si].offsetBytes;
          srem = srcRanges[si].sizeBytes;
        }
      }
      if (drem === 0) {
        di += 1;
        if (di < dstRanges.length) {
          doff = dstRanges[di].offsetBytes;
          drem = dstRanges[di].sizeBytes;
        }
      }
    }
  }

  _coordsFromLinear(i, shape, outCoords) {
    tensorCoordsFromLinear(i, shape, outCoords);
  }

  _requiredStridedInputBytes(runtimeExtraElements, batchStart = 0, batchCount = this.batch) {
    if (!this._inputTensorDesc) {
      throw new Error("internal error: strided input descriptor is not initialized");
    }
    return requiredBytesForBatchRange(this._inputTensorDesc, {
      bytesPerElement: 4,
      runtimeExtraElements,
      batchStart,
      batchCount,
    });
  }

  _requiredStridedOutputBytes(runtimeExtraElements, batchStart = 0, batchCount = this.batch) {
    if (!this._outputTensorDesc) {
      throw new Error("internal error: strided output descriptor is not initialized");
    }
    return requiredBytesForBatchRange(this._outputTensorDesc, {
      bytesPerElement: 4,
      runtimeExtraElements,
      batchStart,
      batchCount,
    });
  }

  _validateStridedInputBounds(input, inputOffsetBytes, batchStart, batchCount) {
    if (inputOffsetBytes % 4 !== 0) {
      throw new Error(`inputOffsetBytes must be a multiple of 4 for real-strided input; got ${inputOffsetBytes}`);
    }
    const runtimeExtraElements = (inputOffsetBytes / 4) | 0;
    const neededBytes = this._requiredStridedInputBytes(runtimeExtraElements, batchStart, batchCount);
    const inputBytes = getBufferByteLength(input);
    if (inputBytes < neededBytes) {
      throw new Error(`input buffer/view too small for dct/dst strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
    }
    return runtimeExtraElements;
  }

  _validateStridedOutputBounds(output, outputOffsetBytes, batchStart, batchCount) {
    if (outputOffsetBytes % 4 !== 0) {
      throw new Error(`outputOffsetBytes must be a multiple of 4 for real-strided output; got ${outputOffsetBytes}`);
    }
    const runtimeExtraElements = (outputOffsetBytes / 4) | 0;
    const neededBytes = this._requiredStridedOutputBytes(runtimeExtraElements, batchStart, batchCount);
    const outputBytes = getBufferByteLength(output);
    if (outputBytes < neededBytes) {
      throw new Error(`output buffer/view too small for dct/dst strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
    }
    return runtimeExtraElements;
  }

  _copyStridedInputToContiguous(commandEncoder, { input, inputOffsetBytes, batchStart, batchCount, dstBuffer, dstOffsetBytes }) {
    const runtimeExtraElements = this._validateStridedInputBounds(input, inputOffsetBytes, batchStart, batchCount);
    const coords = new Array(this.rank).fill(0);
    for (let lb = 0; lb < batchCount; lb++) {
      const gb = batchStart + lb;
      const srcBatchBase = this._inputOffsetElements + runtimeExtraElements + gb * this._inputBatchStrideElements;
      const dstBase = dstOffsetBytes + lb * this._inViewPerBatch * 4;
      for (let vi = 0; vi < this._inViewPerBatch; vi++) {
        this._coordsFromLinear(vi, this._inputLayoutShape, coords);
        let srcElem = srcBatchBase;
        for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src: input,
          srcOffsetBytes: srcElem * 4,
          dst: dstBuffer,
          dstOffsetBytes: dstBase + vi * 4,
          bytes: 4,
        });
      }
    }
  }

  _copyStridedOutputToContiguous(commandEncoder, { output, outputOffsetBytes, batchStart, batchCount, dstBuffer, dstOffsetBytes }) {
    const runtimeExtraElements = this._validateStridedOutputBounds(output, outputOffsetBytes, batchStart, batchCount);
    const coords = new Array(this.rank).fill(0);
    for (let lb = 0; lb < batchCount; lb++) {
      const gb = batchStart + lb;
      const srcBatchBase = this._outputOffsetElements + runtimeExtraElements + gb * this._outputBatchStrideElements;
      const dstBase = dstOffsetBytes + lb * this._outViewPerBatch * 4;
      for (let vi = 0; vi < this._outViewPerBatch; vi++) {
        this._coordsFromLinear(vi, this._outputLayoutShape, coords);
        let srcElem = srcBatchBase;
        for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._outputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src: output,
          srcOffsetBytes: srcElem * 4,
          dst: dstBuffer,
          dstOffsetBytes: dstBase + vi * 4,
          bytes: 4,
        });
      }
    }
  }

  _copyContiguousToStridedOutput(commandEncoder, { srcBuffer, srcOffsetBytes, output, outputOffsetBytes, batchStart, batchCount }) {
    const runtimeExtraElements = this._validateStridedOutputBounds(output, outputOffsetBytes, batchStart, batchCount);
    const coords = new Array(this.rank).fill(0);
    for (let lb = 0; lb < batchCount; lb++) {
      const gb = batchStart + lb;
      const srcBase = srcOffsetBytes + lb * this._outViewPerBatch * 4;
      const dstBatchBase = this._outputOffsetElements + runtimeExtraElements + gb * this._outputBatchStrideElements;
      for (let vi = 0; vi < this._outViewPerBatch; vi++) {
        this._coordsFromLinear(vi, this._outputLayoutShape, coords);
        let dstElem = dstBatchBase;
        for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src: srcBuffer,
          srcOffsetBytes: srcBase + vi * 4,
          dst: output,
          dstOffsetBytes: dstElem * 4,
          bytes: 4,
        });
      }
    }
  }

  _resolveStridedInputBinding(input, inputOffsetBytes) {
    if (!isGpuBuffer(input)) {
      throw new Error("dct/dst custom-strided input currently requires GPUBuffer input");
    }
    const runtimeExtraElements = this._validateStridedInputBounds(input, inputOffsetBytes, 0, this.batch);
    const extraOffsetElements = runtimeExtraElements + this._inputOffsetElements;
    const neededBytes = this._requiredStridedInputBytes(runtimeExtraElements);
    ensureWithinBindingLimit(this.device, neededBytes, "dct/dst strided input binding");
    return { extraOffsetElements, neededBytes };
  }

  _resolveStridedOutputBinding(output, outputOffsetBytes) {
    if (!isGpuBuffer(output)) {
      throw new Error("dct/dst custom-strided output currently requires GPUBuffer output");
    }
    const runtimeExtraElements = this._validateStridedOutputBounds(output, outputOffsetBytes, 0, this.batch);
    const extraOffsetElements = runtimeExtraElements + this._outputOffsetElements;
    const neededBytes = this._requiredStridedOutputBytes(runtimeExtraElements);
    ensureWithinBindingLimit(this.device, neededBytes, "dct/dst strided output binding");
    return { extraOffsetElements, neededBytes };
  }

  _resolveStridedInputBatchWindow(input, inputOffsetBytes, batchIndex) {
    if (!isGpuBuffer(input)) {
      throw new Error("dct/dst custom-strided input currently requires GPUBuffer input");
    }
    if (inputOffsetBytes % 4 !== 0) {
      throw new Error(`inputOffsetBytes must be a multiple of 4 for real-strided input; got ${inputOffsetBytes}`);
    }
    const runtimeExtraElements = (inputOffsetBytes / 4) | 0;
    const baseElements = this._inputOffsetElements + runtimeExtraElements + batchIndex * this._inputBatchStrideElements;
    const windowStartElements = Math.floor(baseElements / this._storageAlignElems) * this._storageAlignElems;
    const extraOffsetElements = baseElements - windowStartElements;
    const neededElements = extraOffsetElements + this._inputSpanElements;
    const neededBytes = neededElements * 4;
    ensureWithinBindingLimit(this.device, neededBytes, "dct/dst strided input batch-window binding");
    const windowEndBytes = (windowStartElements + neededElements) * 4;
    if (windowEndBytes > input.size) {
      throw new Error(`input buffer too small for dct/dst strided batch window: need ${windowEndBytes} bytes, have ${input.size}`);
    }
    return {
      bindingOffsetBytes: windowStartElements * 4,
      bindingSizeBytes: neededBytes,
      extraOffsetElements,
    };
  }

  _resolveStridedOutputBatchWindow(output, outputOffsetBytes, batchIndex) {
    if (!isGpuBuffer(output)) {
      throw new Error("dct/dst custom-strided output currently requires GPUBuffer output");
    }
    if (outputOffsetBytes % 4 !== 0) {
      throw new Error(`outputOffsetBytes must be a multiple of 4 for real-strided output; got ${outputOffsetBytes}`);
    }
    const runtimeExtraElements = (outputOffsetBytes / 4) | 0;
    const baseElements = this._outputOffsetElements + runtimeExtraElements + batchIndex * this._outputBatchStrideElements;
    const windowStartElements = Math.floor(baseElements / this._storageAlignElems) * this._storageAlignElems;
    const extraOffsetElements = baseElements - windowStartElements;
    const neededElements = extraOffsetElements + this._outputSpanElements;
    const neededBytes = neededElements * 4;
    ensureWithinBindingLimit(this.device, neededBytes, "dct/dst strided output batch-window binding");
    const windowEndBytes = (windowStartElements + neededElements) * 4;
    if (windowEndBytes > output.size) {
      throw new Error(`output buffer too small for dct/dst strided batch window: need ${windowEndBytes} bytes, have ${output.size}`);
    }
    return {
      bindingOffsetBytes: windowStartElements * 4,
      bindingSizeBytes: neededBytes,
      extraOffsetElements,
    };
  }

  _execLargeBatchChunk(commandEncoder, { input, output, inputOffsetBytes, outputOffsetBytes, workspaceViews }) {
    const { stage, dataA, dataB, work, fftScratch, f16OutScratch } = workspaceViews;

    const dataARange = normalizeToContiguousRanges(dataA, 0, this.dataBytes)[0];
    const dataBRange = normalizeToContiguousRanges(dataB, 0, this.dataBytes)[0];
    const stageInputF32Range =
      this.ioEmbed && stage
        ? normalizeToContiguousRanges(stage, 0, this._inViewPerBatch * 4)[0]
        : null;
    const stageOutputF32Range =
      this.ioExtract && stage
        ? normalizeToContiguousRanges(stage, 0, this._outViewPerBatch * 4)[0]
        : null;
    const stageF16Range =
      this.precision === "f16-storage" && stage
        ? normalizeToContiguousRanges(stage, this.stageF16Offset, this._inBytesPerBatchBind)[0]
        : null;
    const f16OutRange =
      this.precision === "f16-storage" && f16OutScratch
        ? normalizeToContiguousRanges(f16OutScratch, 0, this._outBytesPerBatchBind)[0]
        : null;

    const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });

    for (let b = 0; b < this.batch; b++) {
      const inBatchOffset = inputOffsetBytes + b * this._inBytesPerBatchRaw;
      const outBatchOffset = outputOffsetBytes + b * this._outBytesPerBatchRaw;

      if (this.precision === "f16-storage") {
        const inRanges = normalizeToContiguousRanges(input, inBatchOffset, this._inBytesPerBatchRaw);
        if (inRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(
            inRanges[0].buffer,
            inRanges[0].offsetBytes,
            stageF16Range.buffer,
            stageF16Range.offsetBytes,
            this._inBytesPerBatchRaw
          );
        } else {
          this.copier.pack(commandEncoder, inRanges, stageF16Range.buffer, stageF16Range.offsetBytes);
        }

        const dstF32 = this.ioEmbed ? stageInputF32Range : dataARange;
        this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this._inViewPerBatch, 0, 0, 0]));
        const bg = this.device.createBindGroup({
          layout: this.f16.bgl,
          entries: [
            { binding: 0, resource: { buffer: stageF16Range.buffer, offset: stageF16Range.offsetBytes, size: this._inBytesPerBatchBind } },
            { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this._inViewPerBatch * 4 } },
            { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.f16.toF32);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this._inViewPerBatch / this.workgroupSize), 1, 1);
        pass.end();
      } else if (this._usesStridedInput) {
        const dstF32 = this.ioEmbed ? stageInputF32Range : dataARange;
        if (isGpuBuffer(input)) {
          const window = this._resolveStridedInputBatchWindow(input, inputOffsetBytes, b);
          this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([this._inViewPerBatch, 1, window.extraOffsetElements, 0]));
          const bg = this.device.createBindGroup({
            layout: this.stridedIn.bgl,
            entries: [
              { binding: 0, resource: { buffer: input, offset: window.bindingOffsetBytes, size: window.bindingSizeBytes } },
              { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this._inViewPerBatch * 4 } },
              { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.stridedIn.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this._inViewPerBatch / this.workgroupSize), 1, 1);
          pass.end();
        } else {
          this._copyStridedInputToContiguous(commandEncoder, {
            input,
            inputOffsetBytes,
            batchStart: b,
            batchCount: 1,
            dstBuffer: dstF32.buffer,
            dstOffsetBytes: dstF32.offsetBytes,
          });
        }
      } else {
        const inRanges = normalizeToContiguousRanges(input, inBatchOffset, this._inBytesPerBatchRaw);
        const dstF32 = this.ioEmbed ? stageInputF32Range : dataARange;
        if (inRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(
            inRanges[0].buffer,
            inRanges[0].offsetBytes,
            dstF32.buffer,
            dstF32.offsetBytes,
            this._inBytesPerBatchRaw
          );
        } else {
          this.copier.pack(commandEncoder, inRanges, dstF32.buffer, dstF32.offsetBytes);
        }
      }

      if (this.ioEmbed) {
        this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, this.ioEmbed.viewTotal, 1, 0]));
        const bg = this.device.createBindGroup({
          layout: this.ioEmbed.bgl,
          entries: [
            { binding: 0, resource: { buffer: stageInputF32Range.buffer, offset: stageInputF32Range.offsetBytes, size: this.ioEmbed.viewTotal * 4 } },
            { binding: 1, resource: { buffer: dataARange.buffer, offset: dataARange.offsetBytes, size: this._logicalBytesPerBatchF32 } },
            { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.ioEmbed.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
        pass.end();
      }

      if (this.zeroRead) {
        const bg = this.device.createBindGroup({
          layout: this.zeroRead.bgl,
          entries: [{ binding: 0, resource: { buffer: dataARange.buffer, offset: dataARange.offsetBytes, size: this._logicalBytesPerBatchF32 } }],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.zeroRead.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
        pass.end();
      }

      let srcBuf = dataARange.buffer;
      let srcOff = dataARange.offsetBytes;
      let dstBuf = dataBRange.buffer;
      let dstOff = dataBRange.offsetBytes;

      for (const ax of this.axes) {
        const workRange = normalizeToContiguousRanges(work, 0, ax.workBytesPerBatch)[0];
        this.device.queue.writeBuffer(ax.build.params, 0, new Uint32Array([ax.workElemsPerBatch, 0, 0, 0]));
        {
          const bg = this.device.createBindGroup({
            layout: ax.build.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this._logicalBytesPerBatchF32 } },
              { binding: 1, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes, size: ax.workBytesPerBatch } },
              { binding: 2, resource: { buffer: ax.build.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(ax.build.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(ax.workElemsPerBatch / this.workgroupSize), 1, 1);
          pass.end();
        }

        const fftExec = {
          input: workRange.buffer,
          inputOffsetBytes: workRange.offsetBytes,
          temp: fftScratch,
        };
        if (ax.fftNeedsBatch) fftExec.batch = ax.linesPerBatch;
        ax.fft.exec(commandEncoder, fftExec);

        this.device.queue.writeBuffer(ax.post.params, 0, new Uint32Array([this.logicalTotal, 0, 0, 0]));
        {
          const bg = this.device.createBindGroup({
            layout: ax.post.bgl,
            entries: [
              { binding: 0, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes, size: ax.workBytesPerBatch } },
              { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: this._logicalBytesPerBatchF32 } },
              { binding: 2, resource: { buffer: ax.post.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(ax.post.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
          pass.end();
        }

        [srcBuf, dstBuf] = [dstBuf, srcBuf];
        [srcOff, dstOff] = [dstOff, srcOff];
      }

      if (scale !== 1.0) {
        this.device.queue.writeBuffer(this.scale.params, 0, new Uint32Array([this.logicalTotal, 0, 0, 0]));
        this.device.queue.writeBuffer(this.scale.params, 16, new Float32Array([scale, 0, 0, 0]));
        const bg = this.device.createBindGroup({
          layout: this.scale.bgl,
          entries: [
            { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this._logicalBytesPerBatchF32 } },
            { binding: 1, resource: { buffer: this.scale.params, offset: 0, size: 32 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.scale.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
        pass.end();
      }

      if (this.zeroWrite) {
        const bg = this.device.createBindGroup({
          layout: this.zeroWrite.bgl,
          entries: [{ binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this._logicalBytesPerBatchF32 } }],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.zeroWrite.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
        pass.end();
      }

      let outF32Buf = srcBuf;
      let outF32Off = srcOff;

      if (this.ioExtract) {
        this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, this.ioExtract.viewTotal, 1, 0]));
        if (!this.io.output.clearOutside) {
          if (this._usesStridedOutput) {
            this._copyStridedOutputToContiguous(commandEncoder, {
              output,
              outputOffsetBytes,
              batchStart: b,
              batchCount: 1,
              dstBuffer: stageOutputF32Range.buffer,
              dstOffsetBytes: stageOutputF32Range.offsetBytes,
            });
          } else {
            const outRanges = normalizeToContiguousRanges(output, outBatchOffset, this._outBytesPerBatchRaw);
            if (this.precision === "f16-storage") {
              if (outRanges.length === 1) {
                commandEncoder.copyBufferToBuffer(
                  outRanges[0].buffer,
                  outRanges[0].offsetBytes,
                  f16OutRange.buffer,
                  f16OutRange.offsetBytes,
                  this._outBytesPerBatchRaw
                );
              } else {
                this.copier.pack(commandEncoder, outRanges, f16OutRange.buffer, f16OutRange.offsetBytes);
              }
              this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this._outViewPerBatch, 0, 0, 0]));
              const bg = this.device.createBindGroup({
                layout: this.f16.bgl,
                entries: [
                  { binding: 0, resource: { buffer: f16OutRange.buffer, offset: f16OutRange.offsetBytes, size: this._outBytesPerBatchBind } },
                  { binding: 1, resource: { buffer: stageOutputF32Range.buffer, offset: stageOutputF32Range.offsetBytes, size: this._outViewPerBatch * 4 } },
                  { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.f16.toF32);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this._outViewPerBatch / this.workgroupSize), 1, 1);
              pass.end();
            } else {
              if (outRanges.length === 1) {
                commandEncoder.copyBufferToBuffer(
                  outRanges[0].buffer,
                  outRanges[0].offsetBytes,
                  stageOutputF32Range.buffer,
                  stageOutputF32Range.offsetBytes,
                  this._outBytesPerBatchRaw
                );
              } else {
                this.copier.pack(commandEncoder, outRanges, stageOutputF32Range.buffer, stageOutputF32Range.offsetBytes);
              }
            }
          }
        }

        const bg = this.device.createBindGroup({
          layout: this.ioExtract.bgl,
          entries: [
            { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this._logicalBytesPerBatchF32 } },
            { binding: 1, resource: { buffer: stageOutputF32Range.buffer, offset: stageOutputF32Range.offsetBytes, size: this._outViewPerBatch * 4 } },
            { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.ioExtract.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.ioExtract.viewTotal / this.workgroupSize), 1, 1);
        pass.end();
        outF32Buf = stageOutputF32Range.buffer;
        outF32Off = stageOutputF32Range.offsetBytes;
      }

      if (this._usesStridedOutput) {
        if (isGpuBuffer(output)) {
          const window = this._resolveStridedOutputBatchWindow(output, outputOffsetBytes, b);
          this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([this._outViewPerBatch, 1, window.extraOffsetElements, 0]));
          const bg = this.device.createBindGroup({
            layout: this.stridedOut.bgl,
            entries: [
              { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this._outViewPerBatch * 4 } },
              { binding: 1, resource: { buffer: output, offset: window.bindingOffsetBytes, size: window.bindingSizeBytes } },
              { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.stridedOut.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this._outViewPerBatch / this.workgroupSize), 1, 1);
          pass.end();
        } else {
          this._copyContiguousToStridedOutput(commandEncoder, {
            srcBuffer: outF32Buf,
            srcOffsetBytes: outF32Off,
            output,
            outputOffsetBytes,
            batchStart: b,
            batchCount: 1,
          });
        }
        continue;
      }

      if (this.precision === "f16-storage") {
        const outRanges = normalizeToContiguousRanges(output, outBatchOffset, this._outBytesPerBatchRaw);
        this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this._outViewPerBatch, 0, 0, 0]));
        const bg = this.device.createBindGroup({
          layout: this.f16.bgl,
          entries: [
            { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this._outViewPerBatch * 4 } },
            { binding: 1, resource: { buffer: f16OutRange.buffer, offset: f16OutRange.offsetBytes, size: this._outBytesPerBatchBind } },
            { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.f16.toF16);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this._outViewPerBatch / this.workgroupSize), 1, 1);
        pass.end();

        if (outRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(
            f16OutRange.buffer,
            f16OutRange.offsetBytes,
            outRanges[0].buffer,
            outRanges[0].offsetBytes,
            this._outBytesPerBatchRaw
          );
        } else {
          this.copier.unpack(commandEncoder, f16OutRange.buffer, f16OutRange.offsetBytes, outRanges);
        }
        continue;
      }

      const outRanges = normalizeToContiguousRanges(output, outBatchOffset, this._outBytesPerBatchRaw);
      if (outRanges.length === 1) {
        commandEncoder.copyBufferToBuffer(outF32Buf, outF32Off, outRanges[0].buffer, outRanges[0].offsetBytes, this._outBytesPerBatchRaw);
      } else {
        this.copier.unpack(commandEncoder, outF32Buf, outF32Off, outRanges);
      }
    }
  }

  _arenaSlicesAreContiguous(arena, largeBatchMode) {
    const checks = [
      [this.stageOffset, this.stageBytes],
      [this.dataAOffset, this.dataBytes],
      [this.dataBOffset, this.dataBytes],
      [this.workOffset, this.workBytesMax],
      [this.fftScratchOffset, this.fftScratchBytes],
      [this.f16OutOffset, this.precision === "f16-storage" ? (largeBatchMode ? this._outBytesPerBatchBind : this.f16OutBytes) : 0],
    ];
    for (const [off, bytes] of checks) {
      if (!bytes) continue;
      if (normalizeToContiguousRanges(arena, off, bytes).length !== 1) return false;
    }
    return true;
  }

  exec(commandEncoder, execOpts) {
    if (this._destroyed) throw new Error("plan destroyed");
    const { input, output, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
    if (!input || !output) throw new Error("exec requires input and output");

    let arena = temp ?? this._arena;
    if (temp && (buffersAlias(temp, input) || buffersAlias(temp, output))) {
      arena = this._arena ?? null;
    }
    if (temp && arena === temp && getBufferByteLength(arena) < this.workspaceBytes) {
      arena = this._arena ?? null;
    }
    if (temp && arena === temp && !this._arenaSlicesAreContiguous(arena, this._largeBatchChunkMode)) {
      arena = this._arena ?? null;
    }
    const workspaceViews = this._resolveWorkspaceViews(arena);
    if (this._largeBatchChunkMode) {
      this._execLargeBatchChunk(commandEncoder, {
        input,
        output,
        inputOffsetBytes,
        outputOffsetBytes,
        workspaceViews,
      });
      return;
    }

    const { stage, dataA, dataB, work, fftScratch, f16OutScratch } = workspaceViews;

    const dataARange = normalizeToContiguousRanges(dataA, 0, this.dataBytes)[0];
    const dataBRange = normalizeToContiguousRanges(dataB, 0, this.dataBytes)[0];

    // Load physical input -> dataA (f32), with optional ioView embed.
    if (this.precision === "f16-storage") {
      const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
      let srcBuf = inRanges[0].buffer;
      let srcOff = inRanges[0].offsetBytes;
      if (inRanges.length > 1) {
        const scratchF16 = normalizeToContiguousRanges(stage, this.stageF16Offset, this.inBytes)[0];
        this.copier.pack(commandEncoder, inRanges, scratchF16.buffer, scratchF16.offsetBytes);
        srcBuf = scratchF16.buffer;
        srcOff = scratchF16.offsetBytes;
      }

      const dstF32 = this.ioEmbed
        ? normalizeToContiguousRanges(stage, 0, this.inViewTotal * 4)[0]
        : dataARange;

      this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this.inViewTotal, 0, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: this.f16.bgl,
        entries: [
          { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.inBytes } },
          { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.inViewTotal * 4 } },
          { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.f16.toF32);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.inViewTotal / this.workgroupSize), 1, 1);
      pass.end();
    } else if (this._usesStridedInput) {
      const dstF32 = this.ioEmbed ? normalizeToContiguousRanges(stage, 0, this.inViewTotal * 4)[0] : dataARange;
      if (isGpuBuffer(input)) {
        const { extraOffsetElements, neededBytes } = this._resolveStridedInputBinding(input, inputOffsetBytes);
        this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([this._inViewPerBatch, this.batch, extraOffsetElements, 0]));
        const bg = this.device.createBindGroup({
          layout: this.stridedIn.bgl,
          entries: [
            { binding: 0, resource: { buffer: input, offset: 0, size: neededBytes } },
            { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.inViewTotal * 4 } },
            { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.stridedIn.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.inViewTotal / this.workgroupSize), 1, 1);
        pass.end();
      } else {
        this._copyStridedInputToContiguous(commandEncoder, {
          input,
          inputOffsetBytes,
          batchStart: 0,
          batchCount: this.batch,
          dstBuffer: dstF32.buffer,
          dstOffsetBytes: dstF32.offsetBytes,
        });
      }
    } else {
      const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
      const dstF32 = this.ioEmbed ? normalizeToContiguousRanges(stage, 0, this.inBytes)[0] : dataARange;
      if (inRanges.length === 1) {
        commandEncoder.copyBufferToBuffer(inRanges[0].buffer, inRanges[0].offsetBytes, dstF32.buffer, dstF32.offsetBytes, this.inBytes);
      } else {
        this.copier.pack(commandEncoder, inRanges, dstF32.buffer, dstF32.offsetBytes);
      }
    }

    if (this.ioEmbed) {
      this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, this.ioEmbed.viewTotal, this.batch, 0]));
      const src = normalizeToContiguousRanges(stage, 0, this.ioEmbed.viewTotal * this.batch * 4)[0];
      const dst = dataARange;
      const bg = this.device.createBindGroup({
        layout: this.ioEmbed.bgl,
        entries: [
          { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.ioEmbed.viewTotal * this.batch * 4 } },
          { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.dataBytes } },
          { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.ioEmbed.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
    }

    if (this.zeroRead) {
      const bg = this.device.createBindGroup({
        layout: this.zeroRead.bgl,
        entries: [{ binding: 0, resource: { buffer: dataARange.buffer, offset: dataARange.offsetBytes, size: this.dataBytes } }],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.zeroRead.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
    }

    // Core ND separable DCT passes.
    let srcBuf = dataARange.buffer;
    let srcOff = dataARange.offsetBytes;
    let dstBuf = dataBRange.buffer;
    let dstOff = dataBRange.offsetBytes;

    for (const ax of this.axes) {
      const workRange = normalizeToContiguousRanges(work, 0, ax.workBytes)[0];
      if (!ax.fftRunsPerBatch) {
        // build -> work
        {
          const bg = this.device.createBindGroup({
            layout: ax.build.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } },
              { binding: 1, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes, size: ax.workBytes } },
              { binding: 2, resource: { buffer: ax.build.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(ax.build.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(ax.workElems / this.workgroupSize), 1, 1);
          pass.end();
        }

        // FFT on work (1D, batch=lines)
        const fftExec = {
          input: workRange.buffer,
          inputOffsetBytes: workRange.offsetBytes,
          temp: fftScratch,
        };
        if (ax.fftNeedsBatch) fftExec.batch = ax.lines;
        ax.fft.exec(commandEncoder, fftExec);

        // post -> dst
        {
          const bg = this.device.createBindGroup({
            layout: ax.post.bgl,
            entries: [
              { binding: 0, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes, size: ax.workBytes } },
              { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: this.dataBytes } },
              { binding: 2, resource: { buffer: ax.post.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(ax.post.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
      } else {
        // Non-factorable M fallback uses a per-logical-batch C2CPlan. Run batch slices explicitly.
        this.device.queue.writeBuffer(ax.build.params, 0, new Uint32Array([ax.workElemsPerBatch, 0, 0, 0]));
        this.device.queue.writeBuffer(ax.post.params, 0, new Uint32Array([this.logicalTotal, 0, 0, 0]));
        for (let b = 0; b < this.batch; b++) {
          const batchDataOffset = b * this._logicalBytesPerBatchF32;
          const batchWorkOffset = b * ax.workBytesPerBatch;

          {
            const bg = this.device.createBindGroup({
              layout: ax.build.bgl,
              entries: [
                { binding: 0, resource: { buffer: srcBuf, offset: srcOff + batchDataOffset, size: this._logicalBytesPerBatchF32 } },
                { binding: 1, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes + batchWorkOffset, size: ax.workBytesPerBatch } },
                { binding: 2, resource: { buffer: ax.build.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(ax.build.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(ax.workElemsPerBatch / this.workgroupSize), 1, 1);
            pass.end();
          }

          ax.fft.exec(commandEncoder, {
            input: workRange.buffer,
            inputOffsetBytes: workRange.offsetBytes + batchWorkOffset,
            temp: fftScratch,
          });

          {
            const bg = this.device.createBindGroup({
              layout: ax.post.bgl,
              entries: [
                { binding: 0, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes + batchWorkOffset, size: ax.workBytesPerBatch } },
                { binding: 1, resource: { buffer: dstBuf, offset: dstOff + batchDataOffset, size: this._logicalBytesPerBatchF32 } },
                { binding: 2, resource: { buffer: ax.post.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(ax.post.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
            pass.end();
          }
        }
      }

      // swap
      [srcBuf, dstBuf] = [dstBuf, srcBuf];
      [srcOff, dstOff] = [dstOff, srcOff];
    }

    // Normalize once after final axis
    const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
    if (scale !== 1.0) {
      this.device.queue.writeBuffer(this.scale.params, 0, new Uint32Array([this.totalReal, 0, 0, 0]));
      this.device.queue.writeBuffer(this.scale.params, 16, new Float32Array([scale, 0, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: this.scale.bgl,
        entries: [
          { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } },
          { binding: 1, resource: { buffer: this.scale.params, offset: 0, size: 32 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.scale.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
    }

    if (this.zeroWrite) {
      const bg = this.device.createBindGroup({
        layout: this.zeroWrite.bgl,
        entries: [{ binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } }],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.zeroWrite.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
    }

    // Optional output mapping (logical -> view)
    let outF32Buf = srcBuf;
    let outF32Off = srcOff;
    let outF32Bytes = this.dataBytes;
    if (this.ioExtract) {
      const viewTotal = this.ioExtract.viewTotal;
      outF32Bytes = viewTotal * this.batch * 4;
      ensureWithinBindingLimit(this.device, outF32Bytes, "DCT ioView.output");
      this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, viewTotal, this.batch, 0]));

      // f32 output can be written directly when contiguous (preserves clearOutside=false semantics).
      if (this.precision === "f32" && !this._usesStridedOutput) {
        const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
        if (outRanges.length === 1) {
          const bg = this.device.createBindGroup({
            layout: this.ioExtract.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } },
              { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
              { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioExtract.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
          pass.end();
          return;
        }
      }

      const dst = normalizeToContiguousRanges(stage, 0, outF32Bytes)[0];

      // For clearOutside=false with staged output, initialize dst from existing output values.
      if (!this.io.output.clearOutside) {
        if (this._usesStridedOutput) {
          this._copyStridedOutputToContiguous(commandEncoder, {
            output,
            outputOffsetBytes,
            batchStart: 0,
            batchCount: this.batch,
            dstBuffer: dst.buffer,
            dstOffsetBytes: dst.offsetBytes,
          });
        } else {
          const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
          if (this.precision === "f16-storage") {
            // output f16 -> dst f32
            let f16SrcBuf = outRanges[0].buffer;
            let f16SrcOff = outRanges[0].offsetBytes;
            if (outRanges.length > 1) {
              const tmpF16 = normalizeToContiguousRanges(f16OutScratch, 0, this.outBytes)[0];
              this.copier.pack(commandEncoder, outRanges, tmpF16.buffer, tmpF16.offsetBytes);
              f16SrcBuf = tmpF16.buffer;
              f16SrcOff = tmpF16.offsetBytes;
            }
            this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this.outViewTotal, 0, 0, 0]));
            const bg = this.device.createBindGroup({
              layout: this.f16.bgl,
              entries: [
                { binding: 0, resource: { buffer: f16SrcBuf, offset: f16SrcOff, size: this.outBytes } },
                { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: outF32Bytes } },
                { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.f16.toF32);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.outViewTotal / this.workgroupSize), 1, 1);
            pass.end();
          } else {
            // output f32 -> dst f32
            if (outRanges.length === 1) {
              commandEncoder.copyBufferToBuffer(outRanges[0].buffer, outRanges[0].offsetBytes, dst.buffer, dst.offsetBytes, this.outBytes);
            } else {
              this.copier.pack(commandEncoder, outRanges, dst.buffer, dst.offsetBytes);
            }
          }
        }
      }

      const bg = this.device.createBindGroup({
        layout: this.ioExtract.bgl,
        entries: [
          { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } },
          { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: outF32Bytes } },
          { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.ioExtract.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
      pass.end();
      outF32Buf = dst.buffer;
      outF32Off = dst.offsetBytes;
    }

    if (this._usesStridedOutput) {
      if (isGpuBuffer(output)) {
        const { extraOffsetElements, neededBytes } = this._resolveStridedOutputBinding(output, outputOffsetBytes);
        this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([this._outViewPerBatch, this.batch, extraOffsetElements, 0]));
        const bg = this.device.createBindGroup({
          layout: this.stridedOut.bgl,
          entries: [
            { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this.outViewTotal * 4 } },
            { binding: 1, resource: { buffer: output, offset: 0, size: neededBytes } },
            { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.stridedOut.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.outViewTotal / this.workgroupSize), 1, 1);
        pass.end();
      } else {
        this._copyContiguousToStridedOutput(commandEncoder, {
          srcBuffer: outF32Buf,
          srcOffsetBytes: outF32Off,
          output,
          outputOffsetBytes,
          batchStart: 0,
          batchCount: this.batch,
        });
      }
      return;
    }

    if (this.precision === "f16-storage") {
      const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
      this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this.outViewTotal, 0, 0, 0]));
      if (outRanges.length === 1) {
        const bg = this.device.createBindGroup({
          layout: this.f16.bgl,
          entries: [
            { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this.outViewTotal * 4 } },
            { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
            { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.f16.toF16);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.outViewTotal / this.workgroupSize), 1, 1);
        pass.end();
        return;
      }
      const tmp = normalizeToContiguousRanges(f16OutScratch, 0, this.outBytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.f16.bgl,
        entries: [
          { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this.outViewTotal * 4 } },
          { binding: 1, resource: { buffer: tmp.buffer, offset: tmp.offsetBytes, size: this.outBytes } },
          { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.f16.toF16);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.outViewTotal / this.workgroupSize), 1, 1);
      pass.end();
      this.copier.unpack(commandEncoder, tmp.buffer, tmp.offsetBytes, outRanges);
      return;
    }

    const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
    if (outRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(outF32Buf, outF32Off, outRanges[0].buffer, outRanges[0].offsetBytes, this.outBytes);
    } else {
      this.copier.unpack(commandEncoder, outF32Buf, outF32Off, outRanges);
    }
  }
}

