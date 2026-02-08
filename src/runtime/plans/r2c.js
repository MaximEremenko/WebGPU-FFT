// Copyright (c) 2026 Maksim Eremenko

import { BasePlan } from "../base_plan.js";
import {
  mergeLargeRouteMetadata,
  resolveAxisKindsForShape,
  resolveLargeRoutingPolicy,
  resolveOutOfCoreAxisWindowPolicy,
} from "../large_policy.js";
import { createInternalArena, viewFromArena } from "../workspace.js";
import { normalizeToContiguousRanges } from "../segmented_io.js";
import { resolveLayoutSemantics } from "../layout_semantics.js";
import {
  assertOneOf,
  isPositiveInt,
  prod,
  alignBytes,
  ensureWithinBindingLimit,
  getBufferByteLength,
  align4Bytes,
  isGpuBuffer,
  buffersAlias,
} from "../common.js";
import { normalizeIoView } from "../ioview.js";
import { normalizeZeroPad } from "../zero_pad.js";
import {
  contiguousStrides as tensorContiguousStrides,
  coordsFromLinear as tensorCoordsFromLinear,
  linearFromCoords as tensorLinearFromCoords,
  createTensorDescriptor,
  requiredBytesForBatchRange,
} from "../tensor_descriptor.js";

import { C2CPlan } from "./c2c.js";
import { generateRealToComplexWGSL, generatePackR2CWGSL } from "../../kernels/real_complex.js";
import { generateZeroOutsideRangeComplexWGSL, generateZeroOutsideRangeRealWGSL } from "../../kernels/zero_pad.js";
import { generateEmbedRealWGSL, generateExtractComplexWGSL, generateExtractComplexF32ToF16WGSL } from "../../kernels/ioview.js";
import { generateF16ToF32RealWGSL, generateF32ToF16ComplexWGSL } from "../../kernels/f16_storage.js";
import { generateGatherRealStridedWGSL } from "../../kernels/strided_real.js";
import { generateScatterComplexStridedWGSL } from "../../kernels/strided_complex.js";

function needsIoMapping(io, logicalShape) {
  if (!io) return false;
  for (let i = 0; i < logicalShape.length; i++) {
    if (io.shape[i] !== logicalShape[i]) return true;
    if (io.offset[i] !== 0) return true;
  }
  return false;
}

export class R2CPlan extends BasePlan {
  constructor(device, opts) {
    super(device, opts);
    const { shape, direction, batch = 1, inPlace = false, normalize = "none", layout = { interleavedComplex: true }, precision = "f32", ioView = null, zeroPad = null } = opts ?? {};
    if (inPlace) throw new Error("r2c inPlace is not supported in current implementation");
    if (direction !== "forward") throw new Error('r2c supports direction:"forward" only');
    if (!Array.isArray(shape) || shape.length < 1) throw new Error("shape must be rank >= 1");
    if (!shape.every(isPositiveInt)) throw new Error("shape must be positive ints");
    assertOneOf(normalize, ["none", "backward", "unitary"], "normalize");
    assertOneOf(precision, ["f32", "f16-storage"], "precision");
    if (layout?.interleavedComplex !== true) throw new Error("r2c output is packed complex interleaved; set layout.interleavedComplex=true");
    if (precision === "f16-storage" && !device.features?.has?.("shader-f16")) throw new Error('precision="f16-storage" requires shader-f16');

    this.shape = shape.slice();
    this.rank = shape.length;
    this.batch = batch;
    this.normalize = normalize;
    this.precision = precision;
    const Nx = this.shape[0];
    this.packedShape = [Math.floor(Nx / 2) + 1, ...this.shape.slice(1)];

    const iov = ioView ?? {};
    this.ioIn = normalizeIoView(this.rank, this.shape, { input: iov.input }).input;
    this.ioOut = normalizeIoView(this.rank, this.packedShape, { output: iov.output }).output;
    this._needsInputMapping = !!(this.ioIn && needsIoMapping(this.ioIn, this.shape));
    this._needsOutputMapping = !!(this.ioOut && needsIoMapping(this.ioOut, this.packedShape));
    this._inputLayoutShape = this._needsInputMapping ? this.ioIn.shape.slice() : this.shape.slice();
    this._outputLayoutShape = this._needsOutputMapping ? this.ioOut.shape.slice() : this.packedShape.slice();

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
    this._inputSpanElements = resolvedLayout.inputSpanElements;
    this._outputSpanElements = resolvedLayout.outputSpanElements;
    this._usesStridedInput = resolvedLayout.usesStridedInput;
    this._usesStridedOutput = resolvedLayout.usesStridedOutput;
    this._usesWhdcnInput = resolvedLayout.usesWhdcnInput;
    this._usesWhdcnOutput = resolvedLayout.usesWhdcnOutput;
    this._inputTensorDesc = this._usesStridedInput
      ? createTensorDescriptor({
          name: "r2c.input",
          shape: this._inputLayoutShape,
          strides: this._inputStrides,
          offsetElements: this._inputOffsetElements,
          batchStrideElements: this._inputBatchStrideElements,
        })
      : null;
    this._outputTensorDesc = this._usesStridedOutput
      ? createTensorDescriptor({
          name: "r2c.output",
          shape: this._outputLayoutShape,
          strides: this._outputStrides,
          offsetElements: this._outputOffsetElements,
          batchStrideElements: this._outputBatchStrideElements,
        })
      : null;
    this._inputSpanElements = this._inputTensorDesc?.spanElements ?? 0;
    this._outputSpanElements = this._outputTensorDesc?.spanElements ?? 0;
    if ((this._usesStridedInput || this._usesStridedOutput) && this.precision !== "f32") {
      throw new Error('custom strides currently support precision:"f32" only for r2c');
    }

    this.zeroPadRead = normalizeZeroPad(this.rank, this.shape, { read: zeroPad?.read ?? null }, "zeroPad").read;
    this.zeroPadWrite = normalizeZeroPad(this.rank, this.packedShape, { write: zeroPad?.write ?? null }, "zeroPad").write;
    this.logicalTotal = prod(this.shape);
    this.totalReal = this.logicalTotal * this.batch;

    this.inViewShape = (this.ioIn?.shape ?? this.shape).slice();
    this.inViewTotalReal = prod(this.inViewShape) * this.batch;
    this.inBytes = precision === "f16-storage" ? align4Bytes(this.inViewTotalReal * 2) : this.inViewTotalReal * 4;

    this.totalComplexFull = this.totalReal;
    this.fullBytes = this.totalComplexFull * 8;

    this.outTotalComplexLogical = prod(this.packedShape) * this.batch;
    this.packedF32Bytes = this.outTotalComplexLogical * 8;

    this.outViewShape = (this.ioOut?.shape ?? this.packedShape).slice();
    this.outViewTotalComplex = prod(this.outViewShape) * this.batch;
    this.outBytes = this.outViewTotalComplex * (precision === "f16-storage" ? 4 : 8);
    this._lineCount = this.batch * prod(this.shape.slice(1));
    this._realLineBytes = this.shape[0] * 4;
    this._complexLineBytes = this.shape[0] * 8;
    this._packedLineBytes = this.packedShape[0] * 8;
    const axisStrategy = resolveAxisKindsForShape({
      shape: this.shape,
      tuning: opts?.tuning ?? null,
    });
    const largePolicy = resolveLargeRoutingPolicy({
      device,
      tuning: opts?.tuning ?? null,
      requiredBindingBytes: [this.fullBytes, this.packedF32Bytes, this.inBytes, this.outBytes],
      lineBytes: [this._realLineBytes, this._complexLineBytes, this._packedLineBytes],
      axisKinds: axisStrategy.axisKinds,
      axisLengths: this.shape,
      allowNonMixedBoundedSlicing: true,
      allowOutOfCore: this.rank >= 2,
      rank: this.rank,
      bytesPerBatch: this.logicalTotal * 8,
      hasStridedIO: this._usesStridedInput || this._usesStridedOutput,
      preferOutOfCoreForStrided: true,
      precision: this.precision,
      requireLargePrecision: "f32",
      requireLargePrecisionError: 'r2c large-shape fallback currently supports precision:"f32" only',
    });
    this._maxBindBytes = largePolicy.maxBindBytes;
    this._largeShapeMode = largePolicy.needsLargeMode;
    this._largeRouteMode = largePolicy.routeMode;
    this._largeRouteReasons = largePolicy.reasonCodes;
    this._largeRouteAttempts = largePolicy.attemptedRoutes;
    this._largeRouteAxisKinds = axisStrategy.axisKinds.slice();
    this._largeRouteAxisSupported = Array.isArray(largePolicy.axisSupported) ? largePolicy.axisSupported.slice() : null;
    if (!this._largeShapeMode) {
      ensureWithinBindingLimit(device, this.fullBytes, "r2c full complex");
      ensureWithinBindingLimit(device, this.packedF32Bytes, "r2c packed logical (f32)");
      ensureWithinBindingLimit(device, this.inBytes, "r2c input");
      ensureWithinBindingLimit(device, this.outBytes, "r2c output");
    }
    this._oversizedLineMode = this._largeShapeMode && largePolicy.oversizedLineMode;
    this._outOfCoreAxisWindowPolicy = null;
    if (this._largeShapeMode) {
      const axisKind0 = this._largeRouteAxisKinds?.[0] ?? "mixed";
      const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
      const tuning = opts?.tuning ?? null;
      this._outOfCoreAxisWindowPolicy = {
        realToComplex: resolveOutOfCoreAxisWindowPolicy({
          axisLen: this.shape[0],
          lineBytes: Math.max(this._realLineBytes, this._complexLineBytes),
          linesTotal: this._lineCount,
          maxBindBytes: this._maxBindBytes,
          axisKind: axisKind0,
          tuning,
          axisIndex: 0,
          storageAlign,
        }),
        pack: resolveOutOfCoreAxisWindowPolicy({
          axisLen: this.shape[0],
          lineBytes: Math.max(this._complexLineBytes, this._packedLineBytes),
          linesTotal: this._lineCount,
          maxBindBytes: this._maxBindBytes,
          axisKind: axisKind0,
          tuning,
          axisIndex: 0,
          storageAlign,
        }),
      };
    }

    // Internal C2C forward on full complex
    this.c2c = new C2CPlan(device, {
      shape: this.shape,
      direction: "forward",
      batch,
      inPlace: true,
      normalize,
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { input: null, output: null },
      tuning: opts?.tuning ?? null,
    });
    const mergedRoute = mergeLargeRouteMetadata([
      {
        routeMode: this._largeRouteMode,
        reasonCodes: this._largeRouteReasons,
        attemptedRoutes: this._largeRouteAttempts,
      },
      {
        routeMode: this.c2c?._largeRouteMode,
        reasonCodes: this.c2c?._largeRouteReasons,
        attemptedRoutes: this.c2c?._largeRouteAttempts,
      },
    ]);
    this._largeRouteMode = mergedRoute.routeMode;
    this._largeRouteReasons = mergedRoute.reasonCodes;
    this._largeRouteAttempts = mergedRoute.attemptedRoutes;

    // real->complex kernel
    this.rtob = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateRealToComplexWGSL({ totalReal: this.totalReal, workgroupSize: this.workgroupSize });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([this.totalReal, 0, 0, 0]));
      return { bgl, pl: pipelineLayout, pipeline, params };
    })();

    // pack kernel
    this.pack = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generatePackR2CWGSL({ shape: this.shape, workgroupSize: this.workgroupSize });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([batch, 0, 0, 0]));
      return { bgl, pl: pipelineLayout, pipeline, params };
    })();
    this.packLine = null;
    if (this._largeShapeMode) {
      this.packLine = (() => {
        const bgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
        const code = generatePackR2CWGSL({ shape: [this.shape[0]], workgroupSize: this.workgroupSize });
        const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
        const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        return { bgl, pl: pipelineLayout, pipeline, params };
      })();
    }

    // f16 input conversion (real)
    this.f16in = null;
    if (precision === "f16-storage") {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipeline = this.cache.getComputePipeline({ code: generateF16ToF32RealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.f16in = { bgl, pl: pipelineLayout, pipeline, params };
      device.queue.writeBuffer(params, 0, new Uint32Array([this.inViewTotalReal, 0, 0, 0]));
    }

    this.f16out = null;
    if (precision === "f16-storage") {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipeline = this.cache.getComputePipeline({ code: generateF32ToF16ComplexWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([this.outViewTotalComplex, 0, 0, 0]));
      this.f16out = { bgl, pl: pipelineLayout, pipeline, params };
    }

    // ioView mapping pipelines
    this.ioEmbed = null;
    if (this.ioIn && needsIoMapping(this.ioIn, this.shape)) {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateEmbedRealWGSL({ rank: this.rank, logicalDims: this.shape, viewDims: this.ioIn.shape, offset: this.ioIn.offset, workgroupSize: this.workgroupSize });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.ioEmbed = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.ioIn.shape) };
    }

    this.ioExtract = null;
    if (this.ioOut && needsIoMapping(this.ioOut, this.packedShape)) {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code =
        this.precision === "f16-storage"
          ? generateExtractComplexF32ToF16WGSL({
              rank: this.rank,
              logicalDims: this.packedShape,
              viewDims: this.ioOut.shape,
              offset: this.ioOut.offset,
              clearOutside: this.ioOut.clearOutside,
              workgroupSize: this.workgroupSize,
            })
          : generateExtractComplexWGSL({
              rank: this.rank,
              logicalDims: this.packedShape,
              viewDims: this.ioOut.shape,
              offset: this.ioOut.offset,
              clearOutside: this.ioOut.clearOutside,
              workgroupSize: this.workgroupSize,
            });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.ioExtract = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.ioOut.shape), logicalTotal: prod(this.packedShape) };
    }

    this.zeroRead = null;
    if (this.zeroPadRead) {
      const bgl = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateZeroOutsideRangeRealWGSL({
        shape: this.shape,
        start: this.zeroPadRead.start,
        end: this.zeroPadRead.end,
        batch: this.batch,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      this.zeroRead = { bgl, pl, pipeline };
    }

    this.zeroWrite = null;
    if (this.zeroPadWrite) {
      const bgl = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateZeroOutsideRangeComplexWGSL({
        shape: this.packedShape,
        start: this.zeroPadWrite.start,
        end: this.zeroPadWrite.end,
        batch: this.batch,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      this.zeroWrite = { bgl, pl, pipeline };
    }

    // Optional strided gather/scatter for real input and packed-complex output (f32 only).
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
        shape: this.shape,
        strides: this._inputStrides,
        baseOffsetElements: this._inputOffsetElements,
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
      const code = generateScatterComplexStridedWGSL({
        shape: this.packedShape,
        strides: this._outputStrides,
        baseOffsetElements: this._outputOffsetElements,
        batchStrideElements: this._outputBatchStrideElements,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.stridedOut = { bgl, pl, pipeline, params };
    }

    // Workspace: [stage scratch][real logical f32][full complex f32][packed logical f32][optional packed f16 out]
    this.realF32Bytes = this.totalReal * 4;
    this.stageInF32Bytes = this.ioEmbed ? this.inViewTotalReal * 4 : 0;
    this.stageOutF32Bytes = this.ioExtract && this.precision === "f32" ? this.outViewTotalComplex * 8 : 0;
    this.stageF16Bytes = precision === "f16-storage" ? this.inBytes : 0;
    const stageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    this.stageF16Offset = alignBytes(this.stageInF32Bytes, stageAlign);
    this.stageBytes = Math.max(this.stageOutF32Bytes, this.stageF16Offset + this.stageF16Bytes);

    const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    let off = 0;
    this.stageOffset = 0;
    off += this.stageBytes;
    off = alignBytes(off, storageAlign);
    this.realOffset = off;
    off += this.realF32Bytes;
    off = alignBytes(off, storageAlign);
    this.fullOffset = off;
    off += this.fullBytes;
    off = alignBytes(off, storageAlign);
    this.packedOffset = off;
    off += this.packedF32Bytes;
    off = alignBytes(off, storageAlign);
    this.packedF16Offset = precision === "f16-storage" ? off : 0;
    off += precision === "f16-storage" ? this.outBytes : 0;

    this.workspaceBytes = off;
    this._splitWorkspace = null;
    const maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
    if (this.workspaceBytes <= maxBufferSize) {
      this._arena = createInternalArena(device, this.workspaceBytes);
    } else {
      const splitNeeds = [
        ["stage", this.stageBytes],
        ["real", this.realF32Bytes],
        ["full", this.fullBytes],
        ["packed", this.packedF32Bytes],
        ["packedF16", this.precision === "f16-storage" ? this.outBytes : 0],
      ];
      for (const [name, bytes] of splitNeeds) {
        if (bytes > 0 && bytes > maxBufferSize) {
          throw new Error(
            `r2c split workspace cannot allocate ${name} buffer: ${bytes} bytes exceeds device.limits.maxBufferSize=${maxBufferSize}. ` +
              `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
          );
        }
      }
      this._arena = null;
      this._splitWorkspace = {
        stage: this.stageBytes ? createInternalArena(device, this.stageBytes) : null,
        real: createInternalArena(device, this.realF32Bytes),
        full: createInternalArena(device, this.fullBytes),
        packed: createInternalArena(device, this.packedF32Bytes),
        packedF16: this.precision === "f16-storage" ? createInternalArena(device, this.outBytes) : null,
      };
    }
    this._largeChunkBuffer = null;
    this._largeChunkBytes = 0;
    this._retiredLargeChunkBuffers = [];
    this._zeroRealBuffer = null;
    this._zeroComplexBuffer = null;
    this._deferredUniformBuffers = [];
  }

  getWorkspaceSizeBytes() {
    return this.workspaceBytes;
  }

  _ensureLargeChunkBuffer(minBytes) {
    if (this._largeChunkBuffer && this._largeChunkBytes >= minBytes) return this._largeChunkBuffer;
    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    if (minBytes > maxBufferSize) {
      throw new Error(
        `r2c large-shape staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}. ` +
          `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
      );
    }
    if (this._largeChunkBuffer) this._retiredLargeChunkBuffers.push(this._largeChunkBuffer);
    this._largeChunkBuffer = this.device.createBuffer({
      size: minBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._largeChunkBytes = minBytes;
    return this._largeChunkBuffer;
  }

  _ensureZeroRealBuffer() {
    if (this._zeroRealBuffer) return this._zeroRealBuffer;
    this._zeroRealBuffer = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this._zeroRealBuffer, 0, new Float32Array([0]));
    return this._zeroRealBuffer;
  }

  _ensureZeroComplexBuffer() {
    if (this._zeroComplexBuffer) return this._zeroComplexBuffer;
    this._zeroComplexBuffer = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this._zeroComplexBuffer, 0, new Float32Array([0, 0]));
    return this._zeroComplexBuffer;
  }

  _copyRangesToContiguous(commandEncoder, ranges, dstBuffer, dstOffsetBytes) {
    let dst = dstOffsetBytes;
    for (const r of ranges) {
      commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
      dst += r.sizeBytes;
    }
  }

  _copyContiguousToRanges(commandEncoder, srcBuffer, srcOffsetBytes, ranges) {
    let src = srcOffsetBytes;
    for (const r of ranges) {
      commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
      src += r.sizeBytes;
    }
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

  _storageRef(x) {
    if (x && x.view) {
      return { store: x.view, baseOffsetBytes: x.offsetBytes ?? 0, sizeBytes: x.sizeBytes ?? null };
    }
    if (x && x.buffer && Number.isInteger(x.offsetBytes) && Number.isInteger(x.sizeBytes)) {
      return { store: x.buffer, baseOffsetBytes: x.offsetBytes, sizeBytes: x.sizeBytes };
    }
    return { store: x, baseOffsetBytes: 0, sizeBytes: null };
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

  _copyRealFromAny(commandEncoder, src, srcOffsetBytes, dstBuffer, dstOffsetBytes) {
    if (isGpuBuffer(src)) {
      commandEncoder.copyBufferToBuffer(src, srcOffsetBytes, dstBuffer, dstOffsetBytes, 4);
      return;
    }
    const srcRanges = normalizeToContiguousRanges(src, srcOffsetBytes, 4);
    if (srcRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, dstBuffer, dstOffsetBytes, 4);
      return;
    }
    const chunkBuf = this._ensureLargeChunkBuffer(4);
    this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
    commandEncoder.copyBufferToBuffer(chunkBuf, 0, dstBuffer, dstOffsetBytes, 4);
  }

  _copyComplexToAny(commandEncoder, srcBuffer, srcOffsetBytes, dst, dstOffsetBytes) {
    if (isGpuBuffer(dst)) {
      commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dst, dstOffsetBytes, 8);
      return;
    }
    const dstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes, 8);
    if (dstRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dstRanges[0].buffer, dstRanges[0].offsetBytes, 8);
      return;
    }
    const chunkBuf = this._ensureLargeChunkBuffer(8);
    commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, chunkBuf, 0, 8);
    this.copier.unpack(commandEncoder, chunkBuf, 0, dstRanges);
  }

  _shapeStrides(shape) {
    return tensorContiguousStrides(shape);
  }

  _coordsFromLinear(i, shape, outCoords) {
    tensorCoordsFromLinear(i, shape, outCoords);
  }

  _linearFromCoords(coords, strides) {
    return tensorLinearFromCoords(coords, strides);
  }

  _requiredStridedInputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
    if (!this._inputTensorDesc) {
      throw new Error("internal error: strided input descriptor is not initialized");
    }
    return requiredBytesForBatchRange(this._inputTensorDesc, {
      bytesPerElement: 4,
      runtimeExtraElements: extraOffsetElements,
      batchStart,
      batchCount,
    });
  }

  _requiredStridedOutputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
    if (!this._outputTensorDesc) {
      throw new Error("internal error: strided output descriptor is not initialized");
    }
    return requiredBytesForBatchRange(this._outputTensorDesc, {
      bytesPerElement: 8,
      runtimeExtraElements: extraOffsetElements,
      batchStart,
      batchCount,
    });
  }

  _copyStridedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange }) {
    if (inputOffsetBytes % 4 !== 0) {
      throw new Error(`inputOffsetBytes must be a multiple of 4 for real-strided input; got ${inputOffsetBytes}`);
    }
    const extraOffsetElements = (inputOffsetBytes / 4) | 0;
    const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
    const inputBytes = getBufferByteLength(input);
    if (inputBytes < neededBytes) {
      throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
    }

    const realRef = this._storageRef(realRange);

    if (!this._needsInputMapping) {
      const coords = new Array(this.rank).fill(0);
      for (let b = 0; b < this.batch; b++) {
        const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
        const dstBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
        for (let li = 0; li < this.logicalTotal; li++) {
          this._coordsFromLinear(li, this.shape, coords);
          let srcElem = srcBatchBase;
          for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
          this._copyAnySpan(commandEncoder, {
            src: input,
            srcOffsetBytes: srcElem * 4,
            dst: realRef.store,
            dstOffsetBytes: dstBase + li * 4,
            bytes: 4,
          });
        }
      }
      return;
    }

    const zeroBuf = this._ensureZeroRealBuffer();
    const viewShape = this.ioIn.shape;
    const viewOffset = this.ioIn.offset;
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
      const dstBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
      for (let li = 0; li < this.logicalTotal; li++) {
        this._coordsFromLinear(li, this.shape, logicalCoords);
        let inside = true;
        for (let d = 0; d < this.rank; d++) {
          const v = logicalCoords[d] - viewOffset[d];
          viewCoords[d] = v;
          if (v < 0 || v >= viewShape[d]) {
            inside = false;
            break;
          }
        }
        if (!inside) {
          this._copyAnySpan(commandEncoder, {
            src: zeroBuf,
            srcOffsetBytes: 0,
            dst: realRef.store,
            dstOffsetBytes: dstBase + li * 4,
            bytes: 4,
          });
          continue;
        }
        let srcElem = srcBatchBase;
        for (let d = 0; d < this.rank; d++) srcElem += viewCoords[d] * this._inputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src: input,
          srcOffsetBytes: srcElem * 4,
          dst: realRef.store,
          dstOffsetBytes: dstBase + li * 4,
          bytes: 4,
        });
      }
    }
  }

  _copyContiguousPackedToStridedOutputOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes }) {
    if (outputOffsetBytes % 8 !== 0) {
      throw new Error(`outputOffsetBytes must be a multiple of 8 for packed-complex strided output; got ${outputOffsetBytes}`);
    }
    const extraOffsetElements = (outputOffsetBytes / 8) | 0;
    const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
    const outputBytes = getBufferByteLength(output);
    if (outputBytes < neededBytes) {
      throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
    }

    const packedRef = this._storageRef(packedRange);
    const packedTotal = prod(this.packedShape);
    if (!this._needsOutputMapping) {
      const coords = new Array(this.rank).fill(0);
      for (let b = 0; b < this.batch; b++) {
        const srcBase = packedRef.baseOffsetBytes + b * packedTotal * 8;
        const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
        for (let li = 0; li < packedTotal; li++) {
          this._coordsFromLinear(li, this.packedShape, coords);
          let dstElem = dstBatchBase;
          for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
          this._copyAnySpan(commandEncoder, {
            src: packedRef.store,
            srcOffsetBytes: srcBase + li * 8,
            dst: output,
            dstOffsetBytes: dstElem * 8,
            bytes: 8,
          });
        }
      }
      return;
    }

    const viewShape = this.ioOut.shape;
    const viewOffset = this.ioOut.offset;
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);

    if (this.ioOut.clearOutside) {
      const zeroBuf = this._ensureZeroComplexBuffer();
      const viewTotal = prod(viewShape);
      for (let b = 0; b < this.batch; b++) {
        const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
        for (let vi = 0; vi < viewTotal; vi++) {
          this._coordsFromLinear(vi, viewShape, viewCoords);
          let dstElem = dstBatchBase;
          for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
          this._copyAnySpan(commandEncoder, {
            src: zeroBuf,
            srcOffsetBytes: 0,
            dst: output,
            dstOffsetBytes: dstElem * 8,
            bytes: 8,
          });
        }
      }
    }

    for (let b = 0; b < this.batch; b++) {
      const srcBase = packedRef.baseOffsetBytes + b * packedTotal * 8;
      const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
      for (let li = 0; li < packedTotal; li++) {
        this._coordsFromLinear(li, this.packedShape, logicalCoords);
        let inside = true;
        for (let d = 0; d < this.rank; d++) {
          const v = logicalCoords[d] - viewOffset[d];
          viewCoords[d] = v;
          if (v < 0 || v >= viewShape[d]) {
            inside = false;
            break;
          }
        }
        if (!inside) continue;
        let dstElem = dstBatchBase;
        for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src: packedRef.store,
          srcOffsetBytes: srcBase + li * 8,
          dst: output,
          dstOffsetBytes: dstElem * 8,
          bytes: 8,
        });
      }
    }
  }

  _zeroOutsideRangeRealLarge(commandEncoder, { dataRange, start, end }) {
    const dataRef = this._storageRef(dataRange);
    const zeroBuf = this._ensureZeroRealBuffer();
    const coords = new Array(this.rank).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const base = dataRef.baseOffsetBytes + b * this.logicalTotal * 4;
      for (let i = 0; i < this.logicalTotal; i++) {
        this._coordsFromLinear(i, this.shape, coords);
        let inside = true;
        for (let d = 0; d < this.rank; d++) {
          if (coords[d] < start[d] || coords[d] >= end[d]) {
            inside = false;
            break;
          }
        }
        if (!inside) {
          this._copyAnySpan(commandEncoder, {
            src: zeroBuf,
            srcOffsetBytes: 0,
            dst: dataRef.store,
            dstOffsetBytes: base + i * 4,
            bytes: 4,
          });
        }
      }
    }
  }

  _zeroOutsideRangeComplexLarge(commandEncoder, { dataRange, shape, start, end }) {
    const dataRef = this._storageRef(dataRange);
    const zeroBuf = this._ensureZeroComplexBuffer();
    const logicalTotal = prod(shape);
    const coords = new Array(shape.length).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const base = dataRef.baseOffsetBytes + b * logicalTotal * 8;
      for (let i = 0; i < logicalTotal; i++) {
        this._coordsFromLinear(i, shape, coords);
        let inside = true;
        for (let d = 0; d < shape.length; d++) {
          if (coords[d] < start[d] || coords[d] >= end[d]) {
            inside = false;
            break;
          }
        }
        if (!inside) {
          this._copyAnySpan(commandEncoder, {
            src: zeroBuf,
            srcOffsetBytes: 0,
            dst: dataRef.store,
            dstOffsetBytes: base + i * 8,
            bytes: 8,
          });
        }
      }
    }
  }

  _embedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange }) {
    const realRef = this._storageRef(realRange);
    if (!this.ioEmbed) {
      this._copyAnySpan(commandEncoder, {
        src: input,
        srcOffsetBytes: inputOffsetBytes,
        dst: realRef.store,
        dstOffsetBytes: realRef.baseOffsetBytes,
        bytes: this.realF32Bytes,
      });
      return;
    }

    const inBytes = this.inViewTotalReal * 4;
    const inBuf = this._ensureLargeChunkBuffer(inBytes);
    this._copyAnySpan(commandEncoder, {
      src: input,
      srcOffsetBytes: inputOffsetBytes,
      dst: inBuf,
      dstOffsetBytes: 0,
      bytes: inBytes,
    });

    const zeroBuf = this._ensureZeroRealBuffer();
    const viewShape = this.ioIn.shape;
    const viewOffset = this.ioIn.offset;
    const viewTotal = prod(viewShape);
    const viewStrides = this._shapeStrides(viewShape);
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const srcBase = b * viewTotal * 4;
      const dstBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
      for (let li = 0; li < this.logicalTotal; li++) {
        this._coordsFromLinear(li, this.shape, logicalCoords);
        let inside = true;
        for (let d = 0; d < this.rank; d++) {
          const v = logicalCoords[d] - viewOffset[d];
          viewCoords[d] = v;
          if (v < 0 || v >= viewShape[d]) {
            inside = false;
            break;
          }
        }
        if (!inside) {
          this._copyAnySpan(commandEncoder, {
            src: zeroBuf,
            srcOffsetBytes: 0,
            dst: realRef.store,
            dstOffsetBytes: dstBase + li * 4,
            bytes: 4,
          });
          continue;
        }
        const vi = this._linearFromCoords(viewCoords, viewStrides);
        this._copyAnySpan(commandEncoder, {
          src: inBuf,
          srcOffsetBytes: srcBase + vi * 4,
          dst: realRef.store,
          dstOffsetBytes: dstBase + li * 4,
          bytes: 4,
        });
      }
    }
  }

  _extractOutputComplexOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes }) {
    const packedRef = this._storageRef(packedRange);
    if (!this.ioExtract) {
      this._copyAnySpan(commandEncoder, {
        src: packedRef.store,
        srcOffsetBytes: packedRef.baseOffsetBytes,
        dst: output,
        dstOffsetBytes: outputOffsetBytes,
        bytes: this.outBytes,
      });
      return;
    }

    const outBytes = this.outViewTotalComplex * 8;
    const outBuf = this._ensureLargeChunkBuffer(outBytes);
    if (!this.ioOut.clearOutside) {
      this._copyAnySpan(commandEncoder, {
        src: output,
        srcOffsetBytes: outputOffsetBytes,
        dst: outBuf,
        dstOffsetBytes: 0,
        bytes: outBytes,
      });
    } else {
      const zeroBuf = this._ensureZeroComplexBuffer();
      for (let i = 0; i < this.outViewTotalComplex; i++) {
        commandEncoder.copyBufferToBuffer(zeroBuf, 0, outBuf, i * 8, 8);
      }
    }

    const viewShape = this.ioOut.shape;
    const viewOffset = this.ioOut.offset;
    const viewTotal = prod(viewShape);
    const logicalTotal = prod(this.packedShape);
    const viewStrides = this._shapeStrides(viewShape);
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const srcBase = packedRef.baseOffsetBytes + b * logicalTotal * 8;
      const dstBase = b * viewTotal * 8;
      for (let li = 0; li < logicalTotal; li++) {
        this._coordsFromLinear(li, this.packedShape, logicalCoords);
        let inside = true;
        for (let d = 0; d < this.rank; d++) {
          const v = logicalCoords[d] - viewOffset[d];
          viewCoords[d] = v;
          if (v < 0 || v >= viewShape[d]) {
            inside = false;
            break;
          }
        }
        if (!inside) continue;
        const vi = this._linearFromCoords(viewCoords, viewStrides);
        this._copyAnySpan(commandEncoder, {
          src: packedRef.store,
          srcOffsetBytes: srcBase + li * 8,
          dst: outBuf,
          dstOffsetBytes: dstBase + vi * 8,
          bytes: 8,
        });
      }
    }

    this._copyAnySpan(commandEncoder, {
      src: outBuf,
      srcOffsetBytes: 0,
      dst: output,
      dstOffsetBytes: outputOffsetBytes,
      bytes: outBytes,
    });
  }

  _resolveLargeStageLinesPerChunk(stageKey, lineBytes) {
    const maxLinesByBind = Math.max(1, Math.floor(this._maxBindBytes / lineBytes));
    const policy = this._outOfCoreAxisWindowPolicy?.[stageKey] ?? null;
    let linesPerChunk = maxLinesByBind;
    if (policy && Number.isInteger(policy.linesPerChunk) && policy.linesPerChunk > 0) {
      linesPerChunk = Math.max(1, Math.min(linesPerChunk, policy.linesPerChunk));
    }
    const alignedLineStep = policy?.alignedLineStep ?? 1;
    if (Number.isInteger(alignedLineStep) && alignedLineStep > 1 && linesPerChunk >= alignedLineStep) {
      linesPerChunk = Math.max(alignedLineStep, Math.floor(linesPerChunk / alignedLineStep) * alignedLineStep);
    }
    return Math.max(1, Math.min(linesPerChunk, this._lineCount));
  }

  _runRealToComplexLineChunks(commandEncoder, { realRange, complexRange }) {
    const realRef = this._storageRef(realRange);
    const complexRef = this._storageRef(complexRange);
    if (this._realLineBytes > this._maxBindBytes || this._complexLineBytes > this._maxBindBytes) {
      this._runRealToComplexElementChunks(commandEncoder, { realRange, complexRange });
      return;
    }
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const lineBytes = Math.max(this._realLineBytes, this._complexLineBytes);
    const maxLinesPerChunk = this._resolveLargeStageLinesPerChunk("realToComplex", lineBytes);
    const maxInBytes = maxLinesPerChunk * this._realLineBytes;
    const maxOutBytes = maxLinesPerChunk * this._complexLineBytes;
    const maxOutOffset = alignBytes(maxInBytes, storageAlign);
    const chunkBuf = this._ensureLargeChunkBuffer(maxOutOffset + maxOutBytes);
    const chunkCount = Math.ceil(this._lineCount / maxLinesPerChunk);
    const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    const paramStride = alignBytes(16, uniformAlign);
    const paramsBuf = this.device.createBuffer({
      size: chunkCount * paramStride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._deferredUniformBuffers.push(paramsBuf);

    let chunkIndex = 0;
    for (let line0 = 0; line0 < this._lineCount; line0 += maxLinesPerChunk) {
      const lines = Math.min(maxLinesPerChunk, this._lineCount - line0);
      const inBytes = lines * this._realLineBytes;
      const outBytes = lines * this._complexLineBytes;
      const outOff = alignBytes(inBytes, storageAlign);
      const srcOff = realRef.baseOffsetBytes + line0 * this._realLineBytes;
      const dstOff = complexRef.baseOffsetBytes + line0 * this._complexLineBytes;

      this._copyAnySpan(commandEncoder, {
        src: realRef.store,
        srcOffsetBytes: srcOff,
        dst: chunkBuf,
        dstOffsetBytes: 0,
        bytes: inBytes,
      });
      const paramOff = chunkIndex * paramStride;
      this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([lines * this.shape[0], 0, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: this.rtob.bgl,
        entries: [
          { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
          { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
          { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.rtob.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((lines * this.shape[0]) / this.workgroupSize), 1, 1);
      pass.end();
      this._copyAnySpan(commandEncoder, {
        src: chunkBuf,
        srcOffsetBytes: outOff,
        dst: complexRef.store,
        dstOffsetBytes: dstOff,
        bytes: outBytes,
      });
      chunkIndex += 1;
    }
  }

  _runRealToComplexElementChunks(commandEncoder, { realRange, complexRange }) {
    const realRef = this._storageRef(realRange);
    const complexRef = this._storageRef(complexRange);
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxElems = Math.max(1, Math.floor(this._maxBindBytes / 8));
    const maxInBytes = maxElems * 4;
    const maxOutBytes = maxElems * 8;
    const outOff = alignBytes(maxInBytes, storageAlign);
    const chunkBuf = this._ensureLargeChunkBuffer(outOff + maxOutBytes);

    const chunksPerLine = Math.ceil(this.shape[0] / maxElems);
    const chunkCount = this._lineCount * chunksPerLine;
    const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    const paramStride = alignBytes(16, uniformAlign);
    const paramsBuf = this.device.createBuffer({
      size: chunkCount * paramStride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._deferredUniformBuffers.push(paramsBuf);

    let chunkIndex = 0;
    for (let line = 0; line < this._lineCount; line++) {
      const srcLineBase = realRef.baseOffsetBytes + line * this._realLineBytes;
      const dstLineBase = complexRef.baseOffsetBytes + line * this._complexLineBytes;
      for (let x0 = 0; x0 < this.shape[0]; x0 += maxElems) {
        const elems = Math.min(maxElems, this.shape[0] - x0);
        const inBytes = elems * 4;
        const outBytes = elems * 8;
        this._copyAnySpan(commandEncoder, {
          src: realRef.store,
          srcOffsetBytes: srcLineBase + x0 * 4,
          dst: chunkBuf,
          dstOffsetBytes: 0,
          bytes: inBytes,
        });
        const paramOff = chunkIndex * paramStride;
        this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([elems, 0, 0, 0]));
        const bg = this.device.createBindGroup({
          layout: this.rtob.bgl,
          entries: [
            { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
            { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
            { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.rtob.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(elems / this.workgroupSize), 1, 1);
        pass.end();
        this._copyAnySpan(commandEncoder, {
          src: chunkBuf,
          srcOffsetBytes: outOff,
          dst: complexRef.store,
          dstOffsetBytes: dstLineBase + x0 * 8,
          bytes: outBytes,
        });
        chunkIndex += 1;
      }
    }
  }

  _runPackLineChunks(commandEncoder, { complexRange, packedRange }) {
    const complexRef = this._storageRef(complexRange);
    const packedRef = this._storageRef(packedRange);
    if (this._complexLineBytes > this._maxBindBytes || this._packedLineBytes > this._maxBindBytes) {
      for (let line = 0; line < this._lineCount; line++) {
        const srcOff = complexRef.baseOffsetBytes + line * this._complexLineBytes;
        const dstOff = packedRef.baseOffsetBytes + line * this._packedLineBytes;
        this._copyAnySpan(commandEncoder, {
          src: complexRef.store,
          srcOffsetBytes: srcOff,
          dst: packedRef.store,
          dstOffsetBytes: dstOff,
          bytes: this._packedLineBytes,
        });
      }
      return;
    }
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const lineBytes = Math.max(this._complexLineBytes, this._packedLineBytes);
    const maxLinesPerChunk = this._resolveLargeStageLinesPerChunk("pack", lineBytes);
    const maxInBytes = maxLinesPerChunk * this._complexLineBytes;
    const maxOutBytes = maxLinesPerChunk * this._packedLineBytes;
    const maxOutOffset = alignBytes(maxInBytes, storageAlign);
    const chunkBuf = this._ensureLargeChunkBuffer(maxOutOffset + maxOutBytes);
    const chunkCount = Math.ceil(this._lineCount / maxLinesPerChunk);
    const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    const paramStride = alignBytes(16, uniformAlign);
    const paramsBuf = this.device.createBuffer({
      size: chunkCount * paramStride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._deferredUniformBuffers.push(paramsBuf);

    let chunkIndex = 0;
    for (let line0 = 0; line0 < this._lineCount; line0 += maxLinesPerChunk) {
      const lines = Math.min(maxLinesPerChunk, this._lineCount - line0);
      const inBytes = lines * this._complexLineBytes;
      const outBytes = lines * this._packedLineBytes;
      const outOff = alignBytes(inBytes, storageAlign);
      const srcOff = complexRef.baseOffsetBytes + line0 * this._complexLineBytes;
      const dstOff = packedRef.baseOffsetBytes + line0 * this._packedLineBytes;

      this._copyAnySpan(commandEncoder, {
        src: complexRef.store,
        srcOffsetBytes: srcOff,
        dst: chunkBuf,
        dstOffsetBytes: 0,
        bytes: inBytes,
      });
      const paramOff = chunkIndex * paramStride;
      this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([lines, 0, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: this.packLine.bgl,
        entries: [
          { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
          { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
          { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.packLine.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((lines * this.packedShape[0]) / this.workgroupSize), 1, 1);
      pass.end();
      this._copyAnySpan(commandEncoder, {
        src: chunkBuf,
        srcOffsetBytes: outOff,
        dst: packedRef.store,
        dstOffsetBytes: dstOff,
        bytes: outBytes,
      });
      chunkIndex += 1;
    }
  }

  _resolveWorkspaceViews(temp) {
    const arena = temp ?? this._arena;
    if (arena) {
      if (getBufferByteLength(arena) < this.workspaceBytes) throw new Error("temp too small");
      return {
        stage: this.stageBytes ? viewFromArena(arena, this.stageOffset, this.stageBytes) : null,
        realView: viewFromArena(arena, this.realOffset, this.realF32Bytes),
        complexView: viewFromArena(arena, this.fullOffset, this.fullBytes),
        packedView: viewFromArena(arena, this.packedOffset, this.packedF32Bytes),
        packedF16View: this.precision === "f16-storage" ? viewFromArena(arena, this.packedF16Offset, this.outBytes) : null,
      };
    }
    if (this._splitWorkspace) {
      return {
        stage: this.stageBytes ? viewFromArena(this._splitWorkspace.stage, 0, this.stageBytes) : null,
        realView: viewFromArena(this._splitWorkspace.real, 0, this.realF32Bytes),
        complexView: viewFromArena(this._splitWorkspace.full, 0, this.fullBytes),
        packedView: viewFromArena(this._splitWorkspace.packed, 0, this.packedF32Bytes),
        packedF16View: this.precision === "f16-storage" ? viewFromArena(this._splitWorkspace.packedF16, 0, this.outBytes) : null,
      };
    }
    throw new Error("No workspace buffer");
  }

  _workspaceViewsAreContiguous(views) {
    const single = (view, bytes) => {
      if (!view || !bytes) return true;
      return normalizeToContiguousRanges(view, 0, bytes).length === 1;
    };
    return (
      single(views.stage, this.stageBytes) &&
      single(views.realView, this.realF32Bytes) &&
      single(views.complexView, this.fullBytes) &&
      single(views.packedView, this.packedF32Bytes) &&
      single(views.packedF16View, this.precision === "f16-storage" ? this.outBytes : 0)
    );
  }

  _resolveLargeWorkspaceRanges(temp) {
    const { realView, complexView, packedView } = this._resolveWorkspaceViews(temp);
    return {
      realRange: { view: realView, offsetBytes: 0, sizeBytes: this.realF32Bytes },
      complexRange: { view: complexView, offsetBytes: 0, sizeBytes: this.fullBytes },
      packedRange: { view: packedView, offsetBytes: 0, sizeBytes: this.packedF32Bytes },
    };
  }

  _execLargeShape(commandEncoder, { input, output, temp, inputOffsetBytes, outputOffsetBytes }) {
    const { realRange, complexRange, packedRange } = this._resolveLargeWorkspaceRanges(temp);

    if (this._usesStridedInput) {
      this._copyStridedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange });
    } else {
      this._embedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange });
    }

    if (this.zeroPadRead) {
      this._zeroOutsideRangeRealLarge(commandEncoder, {
        dataRange: realRange,
        start: this.zeroPadRead.start,
        end: this.zeroPadRead.end,
      });
    }

    this._runRealToComplexLineChunks(commandEncoder, { realRange, complexRange });
    {
      const complexRef = this._storageRef(complexRange);
      this.c2c.exec(commandEncoder, { input: complexRef.store, inputOffsetBytes: complexRef.baseOffsetBytes });
    }
    this._runPackLineChunks(commandEncoder, { complexRange, packedRange });

    if (this.zeroPadWrite) {
      this._zeroOutsideRangeComplexLarge(commandEncoder, {
        dataRange: packedRange,
        shape: this.packedShape,
        start: this.zeroPadWrite.start,
        end: this.zeroPadWrite.end,
      });
    }

    if (this._usesStridedOutput) {
      this._copyContiguousPackedToStridedOutputOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes });
    } else {
      this._extractOutputComplexOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes });
    }
  }

  destroy() {
    if (this._destroyed) return;
    this.c2c.destroy();
    this.rtob.params.destroy();
    this.pack.params.destroy();
    this.packLine?.params?.destroy?.();
    this.f16in?.params?.destroy?.();
    this.f16out?.params?.destroy?.();
    this.ioEmbed?.params?.destroy?.();
    this.ioExtract?.params?.destroy?.();
    this.stridedIn?.params?.destroy?.();
    this.stridedOut?.params?.destroy?.();
    this._largeChunkBuffer?.destroy?.();
    this._splitWorkspace?.stage?.destroy?.();
    this._splitWorkspace?.real?.destroy?.();
    this._splitWorkspace?.full?.destroy?.();
    this._splitWorkspace?.packed?.destroy?.();
    this._splitWorkspace?.packedF16?.destroy?.();
    for (const b of this._retiredLargeChunkBuffers) b?.destroy?.();
    for (const b of this._deferredUniformBuffers) b?.destroy?.();
    this._zeroRealBuffer?.destroy?.();
    this._zeroComplexBuffer?.destroy?.();
    this._arena?.destroy?.();
    super.destroy();
  }

  exec(commandEncoder, execOpts) {
    if (this._destroyed) throw new Error("plan destroyed");
    const { input, output, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
    if (!input || !output) throw new Error("r2c exec requires input and output");
    let workspaceTemp = temp;
    if (workspaceTemp && (buffersAlias(workspaceTemp, input) || buffersAlias(workspaceTemp, output))) {
      workspaceTemp = null;
    }
    if (this._largeShapeMode) {
      this._execLargeShape(commandEncoder, { input, output, temp: workspaceTemp, inputOffsetBytes, outputOffsetBytes });
      return;
    }

    let workspaceViews = this._resolveWorkspaceViews(workspaceTemp);
    if (workspaceTemp && !this._workspaceViewsAreContiguous(workspaceViews)) {
      workspaceTemp = null;
      workspaceViews = this._resolveWorkspaceViews(null);
    }
    const { stage, realView, complexView, packedView, packedF16View } = workspaceViews;

    // Load physical input into f32, then optional ioView embed into logical domain.
    if (this._usesStridedInput) {
      if (this._needsInputMapping) {
        const realRange = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
        this._copyStridedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange });
      } else {
        if (!isGpuBuffer(input)) {
          const realRange = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
          this._copyStridedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange });
        } else {
          if (inputOffsetBytes % 4 !== 0) {
            throw new Error(`inputOffsetBytes must be a multiple of 4 for real-strided input; got ${inputOffsetBytes}`);
          }
          const extraOffsetElements = (inputOffsetBytes / 4) | 0;
          const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
          ensureWithinBindingLimit(this.device, neededBytes, "r2c strided input binding");
          if (input.size < neededBytes) {
            throw new Error(`input buffer too small for strided layout: need ${neededBytes} bytes, have ${input.size}`);
          }

          const dstF32 = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
          this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([this.logicalTotal, this.batch, extraOffsetElements, 0]));
          const bg = this.device.createBindGroup({
            layout: this.stridedIn.bgl,
            entries: [
              { binding: 0, resource: { buffer: input, offset: 0, size: neededBytes } },
              { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.realF32Bytes } },
              { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.stridedIn.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
      }
    } else if (this.precision === "f16-storage") {
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
        ? normalizeToContiguousRanges(stage, 0, this.inViewTotalReal * 4)[0]
        : normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];

      const bg = this.device.createBindGroup({
        layout: this.f16in.bgl,
        entries: [
          { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.inBytes } },
          { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.inViewTotalReal * 4 } },
          { binding: 2, resource: { buffer: this.f16in.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.f16in.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.inViewTotalReal / this.workgroupSize), 1, 1);
      pass.end();
    } else {
      const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
      const dstF32 = this.ioEmbed
        ? normalizeToContiguousRanges(stage, 0, this.inBytes)[0]
        : normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];

      if (inRanges.length === 1) {
        commandEncoder.copyBufferToBuffer(inRanges[0].buffer, inRanges[0].offsetBytes, dstF32.buffer, dstF32.offsetBytes, this.inBytes);
      } else {
        this.copier.pack(commandEncoder, inRanges, dstF32.buffer, dstF32.offsetBytes);
      }
    }

    if (this.ioEmbed && !(this._usesStridedInput && this._needsInputMapping)) {
      this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, this.ioEmbed.viewTotal, this.batch, 0]));
      const src = normalizeToContiguousRanges(stage, 0, this.ioEmbed.viewTotal * this.batch * 4)[0];
      const dst = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.ioEmbed.bgl,
        entries: [
          { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.ioEmbed.viewTotal * this.batch * 4 } },
          { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.realF32Bytes } },
          { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.ioEmbed.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((this.logicalTotal * this.batch) / this.workgroupSize), 1, 1);
      pass.end();
    }

    if (this.zeroRead) {
      const r = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.zeroRead.bgl,
        entries: [{ binding: 0, resource: { buffer: r.buffer, offset: r.offsetBytes, size: this.realF32Bytes } }],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.zeroRead.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
    }

    // real->complex
    {
      const r = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
      const c = normalizeToContiguousRanges(complexView, 0, this.fullBytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.rtob.bgl,
        entries: [
          { binding: 0, resource: { buffer: r.buffer, offset: r.offsetBytes, size: this.realF32Bytes } },
          { binding: 1, resource: { buffer: c.buffer, offset: c.offsetBytes, size: this.fullBytes } },
          { binding: 2, resource: { buffer: this.rtob.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.rtob.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
    }

    // FFT full complex in-place
    const c = normalizeToContiguousRanges(complexView, 0, this.fullBytes)[0];
    this.c2c.exec(commandEncoder, { input: c.buffer, inputOffsetBytes: c.offsetBytes });

    // pack to packedView (f32 complex)
    {
      const dst = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.pack.bgl,
        entries: [
          { binding: 0, resource: { buffer: c.buffer, offset: c.offsetBytes, size: this.fullBytes } },
          { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.packedF32Bytes } },
          { binding: 2, resource: { buffer: this.pack.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.pack.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
      pass.end();
    }

    if (this.zeroWrite) {
      const dst = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.zeroWrite.bgl,
        entries: [{ binding: 0, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.packedF32Bytes } }],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.zeroWrite.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
      pass.end();
    }

    // Optional output view mapping (packed logical -> packed view shape).
    // If present, write directly to the final output when contiguous to preserve clearOutside=false semantics.
    if (this.ioExtract && !(this._usesStridedOutput && this._needsOutputMapping)) {
      const viewTotal = this.ioExtract.viewTotal;
      this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.ioExtract.logicalTotal, viewTotal, this.batch, 0]));

      const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
      if (outRanges.length === 1) {
        const src = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
        const bg = this.device.createBindGroup({
          layout: this.ioExtract.bgl,
          entries: [
            { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.packedF32Bytes } },
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

      let dstBuf = null;
      let dstOff = 0;
      if (this.precision === "f16-storage") {
        const dst = normalizeToContiguousRanges(packedF16View, 0, this.outBytes)[0];
        dstBuf = dst.buffer;
        dstOff = dst.offsetBytes;
      } else {
        const dst = normalizeToContiguousRanges(stage, 0, this.outBytes)[0];
        dstBuf = dst.buffer;
        dstOff = dst.offsetBytes;
      }

      if (!this.ioOut.clearOutside) {
        this.copier.pack(commandEncoder, outRanges, dstBuf, dstOff);
      }

      const src = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.ioExtract.bgl,
        entries: [
          { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.packedF32Bytes } },
          { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: this.outBytes } },
          { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.ioExtract.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
      pass.end();

      this.copier.unpack(commandEncoder, dstBuf, dstOff, outRanges);
      return;
    }

    if (this._usesStridedOutput) {
      if (this._needsOutputMapping) {
        const packedRange = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
        this._copyContiguousPackedToStridedOutputOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes });
        return;
      }
      const packedRange = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
      if (!isGpuBuffer(output)) {
        this._copyContiguousPackedToStridedOutputOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes });
        return;
      }
      if (outputOffsetBytes % 8 !== 0) {
        throw new Error(`outputOffsetBytes must be a multiple of 8 for packed-complex strided output; got ${outputOffsetBytes}`);
      }
      const extraOffsetElements = (outputOffsetBytes / 8) | 0;
      const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
      ensureWithinBindingLimit(this.device, neededBytes, "r2c strided output binding");
      if (output.size < neededBytes) {
        throw new Error(`output buffer too small for strided layout: need ${neededBytes} bytes, have ${output.size}`);
      }
      this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([prod(this.packedShape), this.batch, extraOffsetElements, 0]));
      const bg = this.device.createBindGroup({
        layout: this.stridedOut.bgl,
        entries: [
          { binding: 0, resource: { buffer: packedRange.buffer, offset: packedRange.offsetBytes, size: this.packedF32Bytes } },
          { binding: 1, resource: { buffer: output, offset: 0, size: neededBytes } },
          { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.stridedOut.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
      pass.end();
      return;
    }

    // No output view mapping: write packed logical output in requested precision.
    if (this.precision === "f16-storage") {
      const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
      if (outRanges.length === 1) {
        const outF32 = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
        const bg = this.device.createBindGroup({
          layout: this.f16out.bgl,
          entries: [
            { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.outTotalComplexLogical * 8 } },
            { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
            { binding: 2, resource: { buffer: this.f16out.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.f16out.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
        pass.end();
        return;
      }

      const outF32 = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
      const dst = normalizeToContiguousRanges(packedF16View, 0, this.outBytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.f16out.bgl,
        entries: [
          { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.outTotalComplexLogical * 8 } },
          { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.outBytes } },
          { binding: 2, resource: { buffer: this.f16out.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.f16out.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
      pass.end();
      this.copier.unpack(commandEncoder, dst.buffer, dst.offsetBytes, outRanges);
      return;
    }

    const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
    const outF32 = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
    if (outRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(outF32.buffer, outF32.offsetBytes, outRanges[0].buffer, outRanges[0].offsetBytes, this.outBytes);
    } else {
      this.copier.unpack(commandEncoder, outF32.buffer, outF32.offsetBytes, outRanges);
    }
  }
}

