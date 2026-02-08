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
import { generateUnpackC2RWGSL, generateComplexToRealWGSL } from "../../kernels/real_complex.js";
import { generateZeroOutsideRangeComplexWGSL, generateZeroOutsideRangeRealWGSL } from "../../kernels/zero_pad.js";
import { generateEmbedComplexWGSL, generateExtractRealWGSL } from "../../kernels/ioview.js";
import { generateF16ToF32ComplexWGSL, generateF16ToF32RealWGSL, generateF32ToF16RealWGSL } from "../../kernels/f16_storage.js";
import { generateGatherComplexStridedWGSL } from "../../kernels/strided_complex.js";
import { generateScatterRealStridedWGSL } from "../../kernels/strided_real.js";

function needsIoMapping(io, logicalShape) {
  if (!io) return false;
  for (let i = 0; i < logicalShape.length; i++) {
    if (io.shape[i] !== logicalShape[i]) return true;
    if (io.offset[i] !== 0) return true;
  }
  return false;
}

function generateFinalizeUnpackedHermitianWGSL({ shape, workgroupSize }) {
  const Nx = shape[0];
  const inNx = Math.floor(Nx / 2) + 1;
  const evenNx = Nx % 2 === 0;
  let remName = "line";
  let decode = "";
  for (let d = 1; d < shape.length; d++) {
    const c = `c${d}`;
    decode += `  let ${c}: u32 = ${remName} % ${shape[d]}u;\n`;
    if (d < shape.length - 1) {
      const rn = `rem${d}`;
      decode += `  let ${rn}: u32 = ${remName} / ${shape[d]}u;\n`;
      remName = rn;
    }
  }

  let selfExpr = "(x == 0u || (EVEN_NX && x == (NX / 2u)))";
  for (let d = 1; d < shape.length; d++) {
    if (shape[d] % 2 === 0) selfExpr += ` && (c${d} == 0u || c${d} == ${shape[d] / 2}u)`;
    else selfExpr += ` && (c${d} == 0u)`;
  }

  return /* wgsl */ `
struct Params {
  lineOffset: u32,
  lineCount: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

const NX: u32 = ${Nx}u;
const IN_NX: u32 = ${inNx}u;
const EVEN_NX: bool = ${evenNx ? "true" : "false"};

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  let total: u32 = params.lineCount * NX;
  if (i >= total) { return; }
  let lineLocal: u32 = i / NX;
  let x: u32 = i - lineLocal * NX;
  let line: u32 = params.lineOffset + lineLocal;
${decode}
  var v: vec2<f32> = data[i];
  if (x >= IN_NX) {
    v = vec2<f32>(v.x, -v.y);
  }
  if (${selfExpr}) {
    v = vec2<f32>(v.x, 0.0);
  }
  data[i] = v;
}
`;
}

function generateFinalizeUnpackedHermitianSegmentWGSL({ shape, workgroupSize }) {
  const Nx = shape[0];
  const inNx = Math.floor(Nx / 2) + 1;
  const evenNx = Nx % 2 === 0;
  return /* wgsl */ `
struct Params {
  count: u32,
  xOffset: u32,
  lineSelfConj: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

const NX: u32 = ${Nx}u;
const IN_NX: u32 = ${inNx}u;
const EVEN_NX: bool = ${evenNx ? "true" : "false"};

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count) { return; }
  let x: u32 = params.xOffset + i;
  var v: vec2<f32> = data[i];
  if (x >= IN_NX) {
    v = vec2<f32>(v.x, -v.y);
  }
  if (params.lineSelfConj != 0u && (x == 0u || (EVEN_NX && x == (NX / 2u)))) {
    v = vec2<f32>(v.x, 0.0);
  }
  data[i] = v;
}
`;
}

export class C2RPlan extends BasePlan {
  constructor(device, opts) {
    super(device, opts);
    const { shape, direction, batch = 1, inPlace = false, normalize = "none", layout = { interleavedComplex: true }, precision = "f32", ioView = null, zeroPad = null } = opts ?? {};
    if (inPlace) throw new Error("c2r inPlace is not supported in current implementation");
    if (direction !== "inverse") throw new Error('c2r supports direction:"inverse" only');
    if (!Array.isArray(shape) || shape.length < 1) throw new Error("shape must be rank >= 1");
    if (!shape.every(isPositiveInt)) throw new Error("shape must be positive ints");
    assertOneOf(normalize, ["none", "backward", "unitary"], "normalize");
    assertOneOf(precision, ["f32", "f16-storage"], "precision");
    if (layout?.interleavedComplex !== true) throw new Error("c2r input is packed complex interleaved; set layout.interleavedComplex=true");
    if (precision === "f16-storage" && !device.features?.has?.("shader-f16")) throw new Error('precision="f16-storage" requires shader-f16');

    this.shape = shape.slice();
    this.rank = shape.length;
    this.batch = batch;
    this.normalize = normalize;
    this.precision = precision;
    const Nx = this.shape[0];
    this.packedShape = [Math.floor(Nx / 2) + 1, ...this.shape.slice(1)];
    this.logicalTotal = prod(this.shape);
    this.totalReal = this.logicalTotal * this.batch;

    const iov = ioView ?? {};

    this.ioIn = normalizeIoView(this.rank, this.packedShape, { input: iov.input }).input;
    this.ioOut = normalizeIoView(this.rank, this.shape, { output: iov.output }).output;
    this._needsInputMapping = !!(this.ioIn && needsIoMapping(this.ioIn, this.packedShape));
    this._needsOutputMapping = !!(this.ioOut && needsIoMapping(this.ioOut, this.shape));
    this._inputLayoutShape = this._needsInputMapping ? this.ioIn.shape.slice() : this.packedShape.slice();
    this._outputLayoutShape = this._needsOutputMapping ? this.ioOut.shape.slice() : this.shape.slice();

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
          name: "c2r.input",
          shape: this._inputLayoutShape,
          strides: this._inputStrides,
          offsetElements: this._inputOffsetElements,
          batchStrideElements: this._inputBatchStrideElements,
        })
      : null;
    this._outputTensorDesc = this._usesStridedOutput
      ? createTensorDescriptor({
          name: "c2r.output",
          shape: this._outputLayoutShape,
          strides: this._outputStrides,
          offsetElements: this._outputOffsetElements,
          batchStrideElements: this._outputBatchStrideElements,
        })
      : null;
    this._inputSpanElements = this._inputTensorDesc?.spanElements ?? 0;
    this._outputSpanElements = this._outputTensorDesc?.spanElements ?? 0;
    if ((this._usesStridedInput || this._usesStridedOutput) && this.precision !== "f32") {
      throw new Error('custom strides currently support precision:"f32" only for c2r');
    }

    this.zeroPadRead = normalizeZeroPad(this.rank, this.packedShape, { read: zeroPad?.read ?? null }, "zeroPad").read;
    this.zeroPadWrite = normalizeZeroPad(this.rank, this.shape, { write: zeroPad?.write ?? null }, "zeroPad").write;
    this.inTotalComplexLogical = prod(this.packedShape) * this.batch;
    this.packedF32Bytes = this.inTotalComplexLogical * 8;

    this.inViewShape = (this.ioIn?.shape ?? this.packedShape).slice();
    this.inViewTotalComplex = prod(this.inViewShape) * this.batch;
    this.inBytes = this.inViewTotalComplex * (precision === "f16-storage" ? 4 : 8);

    this.outViewShape = (this.ioOut?.shape ?? this.shape).slice();
    this.outViewTotalReal = prod(this.outViewShape) * this.batch;
    this.outBytes = precision === "f16-storage" ? align4Bytes(this.outViewTotalReal * 2) : this.outViewTotalReal * 4;

    this.totalComplexFull = this.totalReal;
    this.fullBytes = this.totalComplexFull * 8;
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
      requireLargePrecisionError: 'c2r large-shape fallback currently supports precision:"f32" only',
    });
    this._maxBindBytes = largePolicy.maxBindBytes;
    this._largeShapeMode = largePolicy.needsLargeMode;
    this._largeRouteMode = largePolicy.routeMode;
    this._largeRouteReasons = largePolicy.reasonCodes;
    this._largeRouteAttempts = largePolicy.attemptedRoutes;
    this._largeRouteAxisKinds = axisStrategy.axisKinds.slice();
    this._largeRouteAxisSupported = Array.isArray(largePolicy.axisSupported) ? largePolicy.axisSupported.slice() : null;
    if (!this._largeShapeMode) {
      ensureWithinBindingLimit(device, this.fullBytes, "c2r full complex");
      ensureWithinBindingLimit(device, this.packedF32Bytes, "c2r packed logical (f32)");
      ensureWithinBindingLimit(device, this.inBytes, "c2r input");
      ensureWithinBindingLimit(device, this.outBytes, "c2r output");
    }
    this._oversizedLineMode = this._largeShapeMode && largePolicy.oversizedLineMode;
    this._outOfCoreAxisWindowPolicy = null;
    if (this._largeShapeMode) {
      const axisKind0 = this._largeRouteAxisKinds?.[0] ?? "mixed";
      const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
      const tuning = opts?.tuning ?? null;
      this._outOfCoreAxisWindowPolicy = {
        unpack: resolveOutOfCoreAxisWindowPolicy({
          axisLen: this.shape[0],
          lineBytes: Math.max(this._complexLineBytes, this._packedLineBytes),
          linesTotal: this._lineCount,
          maxBindBytes: this._maxBindBytes,
          axisKind: axisKind0,
          tuning,
          axisIndex: 0,
          storageAlign,
        }),
        complexToReal: resolveOutOfCoreAxisWindowPolicy({
          axisLen: this.shape[0],
          lineBytes: Math.max(this._complexLineBytes, this._realLineBytes),
          linesTotal: this._lineCount,
          maxBindBytes: this._maxBindBytes,
          axisKind: axisKind0,
          tuning,
          axisIndex: 0,
          storageAlign,
        }),
      };
    }

    this.c2c = new C2CPlan(device, {
      shape: this.shape,
      direction: "inverse",
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

    // unpack packed spectrum to full complex
    this.unpack = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateUnpackC2RWGSL({ shape: this.shape, workgroupSize: this.workgroupSize });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([batch, 0, 0, 0]));
      return { bgl, pl: pipelineLayout, pipeline, params };
    })();
    this.unpackLine = null;
    if (this._largeShapeMode) {
      this.unpackLine = (() => {
        const bgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
        const code = generateUnpackC2RWGSL({ shape: [this.shape[0]], workgroupSize: this.workgroupSize });
        const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
        const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        return { bgl, pl: pipelineLayout, pipeline, params };
      })();
    }
    this.unpackFinalize = null;
    if (this._largeShapeMode) {
      this.unpackFinalize = (() => {
        const bgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
        const code = generateFinalizeUnpackedHermitianWGSL({ shape: this.shape, workgroupSize: this.workgroupSize });
        const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
        const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        return { bgl, pl: pipelineLayout, pipeline, params };
      })();
    }
    this.unpackFinalizeSegment = null;
    if (this._largeShapeMode) {
      this.unpackFinalizeSegment = (() => {
        const bgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
        const code = generateFinalizeUnpackedHermitianSegmentWGSL({ shape: this.shape, workgroupSize: this.workgroupSize });
        const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
        const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        return { bgl, pl: pipelineLayout, pipeline, params };
      })();
    }

    // complex->real
    this.c2r = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateComplexToRealWGSL({ workgroupSize: this.workgroupSize });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([this.totalReal, 0, 0, 0]));
      return { bgl, pl: pipelineLayout, pipeline, params };
    })();

    // f16 input conversion for packed complex
    this.f16In = null;
    if (precision === "f16-storage") {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipeline = this.cache.getComputePipeline({ code: generateF16ToF32ComplexWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([this.inViewTotalComplex, 0, 0, 0]));
      this.f16In = { bgl, pl: pipelineLayout, pipeline, params };
    }

    // f16 output conversion
    this.f16Out = null;
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
      device.queue.writeBuffer(params, 0, new Uint32Array([this.outViewTotalReal, 0, 0, 0]));
      this.f16Out = { bgl, pl: pipelineLayout, toF32, toF16, params };
    }

    // ioView mapping pipelines
    this.ioEmbed = null;
    if (this.ioIn && needsIoMapping(this.ioIn, this.packedShape)) {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateEmbedComplexWGSL({
        rank: this.rank,
        logicalDims: this.packedShape,
        viewDims: this.ioIn.shape,
        offset: this.ioIn.offset,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.ioEmbed = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.ioIn.shape), logicalTotal: prod(this.packedShape) };
    }

    this.ioExtract = null;
    if (this.ioOut && needsIoMapping(this.ioOut, this.shape)) {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateExtractRealWGSL({
        rank: this.rank,
        logicalDims: this.shape,
        viewDims: this.ioOut.shape,
        offset: this.ioOut.offset,
        clearOutside: this.ioOut.clearOutside,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.ioExtract = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.ioOut.shape), logicalTotal: this.logicalTotal };
    }

    this.zeroRead = null;
    if (this.zeroPadRead) {
      const bgl = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateZeroOutsideRangeComplexWGSL({
        shape: this.packedShape,
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
      const code = generateZeroOutsideRangeRealWGSL({
        shape: this.shape,
        start: this.zeroPadWrite.start,
        end: this.zeroPadWrite.end,
        batch: this.batch,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      this.zeroWrite = { bgl, pl, pipeline };
    }

    // Optional strided gather/scatter for packed-complex input and real output (f32 only).
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
      const code = generateGatherComplexStridedWGSL({
        shape: this.packedShape,
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
      const code = generateScatterRealStridedWGSL({
        shape: this.shape,
        strides: this._outputStrides,
        baseOffsetElements: this._outputOffsetElements,
        batchStrideElements: this._outputBatchStrideElements,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.stridedOut = { bgl, pl, pipeline, params };
    }

    // Workspace: [stage scratch][packed logical f32][full complex][real logical f32][optional f16 out scratch]
    this.realF32Bytes = this.totalReal * 4;
    this.stageInF32Bytes = this.ioEmbed ? this.inViewTotalComplex * 8 : 0;
    this.stageOutF32Bytes = this.ioExtract ? this.outViewTotalReal * 4 : 0;
    this.stageF16Bytes = precision === "f16-storage" ? this.inBytes : 0;
    const stageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    this.stageF16Offset = alignBytes(this.stageInF32Bytes, stageAlign);
    this.stageBytes = Math.max(this.stageOutF32Bytes, this.stageF16Offset + this.stageF16Bytes);

    const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    let off = 0;
    this.stageOffset = 0;
    off += this.stageBytes;
    off = alignBytes(off, storageAlign);
    this.packedOffset = off;
    off += this.packedF32Bytes;
    off = alignBytes(off, storageAlign);
    this.fullOffset = off;
    off += this.fullBytes;
    off = alignBytes(off, storageAlign);
    this.realOffset = off;
    off += this.realF32Bytes;
    off = alignBytes(off, storageAlign);
    this.outF16Offset = precision === "f16-storage" ? off : 0;
    off += precision === "f16-storage" ? this.outBytes : 0;

    this.workspaceBytes = off;
    this._splitWorkspace = null;
    const maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
    if (this.workspaceBytes <= maxBufferSize) {
      this._arena = createInternalArena(device, this.workspaceBytes);
    } else {
      const splitNeeds = [
        ["stage", this.stageBytes],
        ["packed", this.packedF32Bytes],
        ["full", this.fullBytes],
        ["real", this.realF32Bytes],
        ["outF16", this.precision === "f16-storage" ? this.outBytes : 0],
      ];
      for (const [name, bytes] of splitNeeds) {
        if (bytes > 0 && bytes > maxBufferSize) {
          throw new Error(
            `c2r split workspace cannot allocate ${name} buffer: ${bytes} bytes exceeds device.limits.maxBufferSize=${maxBufferSize}. ` +
              `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
          );
        }
      }
      this._arena = null;
      this._splitWorkspace = {
        stage: this.stageBytes ? createInternalArena(device, this.stageBytes) : null,
        packed: createInternalArena(device, this.packedF32Bytes),
        full: createInternalArena(device, this.fullBytes),
        real: createInternalArena(device, this.realF32Bytes),
        outF16: this.precision === "f16-storage" ? createInternalArena(device, this.outBytes) : null,
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
        `c2r large-shape staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}. ` +
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

  _copyComplexFromAny(commandEncoder, src, srcOffsetBytes, dstBuffer, dstOffsetBytes) {
    if (isGpuBuffer(src)) {
      commandEncoder.copyBufferToBuffer(src, srcOffsetBytes, dstBuffer, dstOffsetBytes, 8);
      return;
    }
    const srcRanges = normalizeToContiguousRanges(src, srcOffsetBytes, 8);
    if (srcRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, dstBuffer, dstOffsetBytes, 8);
      return;
    }
    const chunkBuf = this._ensureLargeChunkBuffer(8);
    this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
    commandEncoder.copyBufferToBuffer(chunkBuf, 0, dstBuffer, dstOffsetBytes, 8);
  }

  _copyRealToAny(commandEncoder, srcBuffer, srcOffsetBytes, dst, dstOffsetBytes) {
    if (isGpuBuffer(dst)) {
      commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dst, dstOffsetBytes, 4);
      return;
    }
    const dstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes, 4);
    if (dstRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dstRanges[0].buffer, dstRanges[0].offsetBytes, 4);
      return;
    }
    const chunkBuf = this._ensureLargeChunkBuffer(4);
    commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, chunkBuf, 0, 4);
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
      bytesPerElement: 8,
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
      bytesPerElement: 4,
      runtimeExtraElements: extraOffsetElements,
      batchStart,
      batchCount,
    });
  }

  _copyStridedInputPackedOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange }) {
    if (inputOffsetBytes % 8 !== 0) {
      throw new Error(`inputOffsetBytes must be a multiple of 8 for packed-complex strided input; got ${inputOffsetBytes}`);
    }
    const extraOffsetElements = (inputOffsetBytes / 8) | 0;
    const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
    const inputBytes = getBufferByteLength(input);
    if (inputBytes < neededBytes) {
      throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
    }

    const packedRef = this._storageRef(packedRange);
    const packedTotal = prod(this.packedShape);
    if (!this._needsInputMapping) {
      const coords = new Array(this.rank).fill(0);
      for (let b = 0; b < this.batch; b++) {
        const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
        const dstBase = packedRef.baseOffsetBytes + b * packedTotal * 8;
        for (let li = 0; li < packedTotal; li++) {
          this._coordsFromLinear(li, this.packedShape, coords);
          let srcElem = srcBatchBase;
          for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
          this._copyAnySpan(commandEncoder, {
            src: input,
            srcOffsetBytes: srcElem * 8,
            dst: packedRef.store,
            dstOffsetBytes: dstBase + li * 8,
            bytes: 8,
          });
        }
      }
      return;
    }

    const zeroBuf = this._ensureZeroComplexBuffer();
    const viewShape = this.ioIn.shape;
    const viewOffset = this.ioIn.offset;
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
      const dstBase = packedRef.baseOffsetBytes + b * packedTotal * 8;
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
        if (!inside) {
          this._copyAnySpan(commandEncoder, {
            src: zeroBuf,
            srcOffsetBytes: 0,
            dst: packedRef.store,
            dstOffsetBytes: dstBase + li * 8,
            bytes: 8,
          });
          continue;
        }
        let srcElem = srcBatchBase;
        for (let d = 0; d < this.rank; d++) srcElem += viewCoords[d] * this._inputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src: input,
          srcOffsetBytes: srcElem * 8,
          dst: packedRef.store,
          dstOffsetBytes: dstBase + li * 8,
          bytes: 8,
        });
      }
    }
  }

  _copyContiguousRealToStridedOutputOutOfCore(commandEncoder, { realRange, output, outputOffsetBytes }) {
    if (outputOffsetBytes % 4 !== 0) {
      throw new Error(`outputOffsetBytes must be a multiple of 4 for real-strided output; got ${outputOffsetBytes}`);
    }
    const extraOffsetElements = (outputOffsetBytes / 4) | 0;
    const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
    const outputBytes = getBufferByteLength(output);
    if (outputBytes < neededBytes) {
      throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
    }

    const realRef = this._storageRef(realRange);
    if (!this._needsOutputMapping) {
      const coords = new Array(this.rank).fill(0);
      for (let b = 0; b < this.batch; b++) {
        const srcBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
        const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
        for (let li = 0; li < this.logicalTotal; li++) {
          this._coordsFromLinear(li, this.shape, coords);
          let dstElem = dstBatchBase;
          for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
          this._copyAnySpan(commandEncoder, {
            src: realRef.store,
            srcOffsetBytes: srcBase + li * 4,
            dst: output,
            dstOffsetBytes: dstElem * 4,
            bytes: 4,
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
      const zeroBuf = this._ensureZeroRealBuffer();
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
            dstOffsetBytes: dstElem * 4,
            bytes: 4,
          });
        }
      }
    }

    for (let b = 0; b < this.batch; b++) {
      const srcBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
      const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
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
        if (!inside) continue;
        let dstElem = dstBatchBase;
        for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src: realRef.store,
          srcOffsetBytes: srcBase + li * 4,
          dst: output,
          dstOffsetBytes: dstElem * 4,
          bytes: 4,
        });
      }
    }
  }

  _zeroOutsideRangeRealLarge(commandEncoder, { dataRange, shape, start, end }) {
    const dataRef = this._storageRef(dataRange);
    const zeroBuf = this._ensureZeroRealBuffer();
    const logicalTotal = prod(shape);
    const coords = new Array(shape.length).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const base = dataRef.baseOffsetBytes + b * logicalTotal * 4;
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

  _embedInputComplexOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange }) {
    const packedRef = this._storageRef(packedRange);
    if (!this.ioEmbed) {
      this._copyAnySpan(commandEncoder, {
        src: input,
        srcOffsetBytes: inputOffsetBytes,
        dst: packedRef.store,
        dstOffsetBytes: packedRef.baseOffsetBytes,
        bytes: this.packedF32Bytes,
      });
      return;
    }

    const inBytes = this.inViewTotalComplex * 8;
    const inBuf = this._ensureLargeChunkBuffer(inBytes);
    this._copyAnySpan(commandEncoder, {
      src: input,
      srcOffsetBytes: inputOffsetBytes,
      dst: inBuf,
      dstOffsetBytes: 0,
      bytes: inBytes,
    });

    const zeroBuf = this._ensureZeroComplexBuffer();
    const viewShape = this.ioIn.shape;
    const viewOffset = this.ioIn.offset;
    const viewTotal = prod(viewShape);
    const viewStrides = this._shapeStrides(viewShape);
    const logicalTotal = prod(this.packedShape);
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const srcBase = b * viewTotal * 8;
      const dstBase = packedRef.baseOffsetBytes + b * logicalTotal * 8;
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
        if (!inside) {
          this._copyAnySpan(commandEncoder, {
            src: zeroBuf,
            srcOffsetBytes: 0,
            dst: packedRef.store,
            dstOffsetBytes: dstBase + li * 8,
            bytes: 8,
          });
          continue;
        }
        const vi = this._linearFromCoords(viewCoords, viewStrides);
        this._copyAnySpan(commandEncoder, {
          src: inBuf,
          srcOffsetBytes: srcBase + vi * 8,
          dst: packedRef.store,
          dstOffsetBytes: dstBase + li * 8,
          bytes: 8,
        });
      }
    }
  }

  _extractOutputRealOutOfCore(commandEncoder, { realRange, output, outputOffsetBytes }) {
    const realRef = this._storageRef(realRange);
    if (!this.ioExtract) {
      this._copyAnySpan(commandEncoder, {
        src: realRef.store,
        srcOffsetBytes: realRef.baseOffsetBytes,
        dst: output,
        dstOffsetBytes: outputOffsetBytes,
        bytes: this.outBytes,
      });
      return;
    }

    const outBytes = this.outViewTotalReal * 4;
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
      const zeroBuf = this._ensureZeroRealBuffer();
      for (let i = 0; i < this.outViewTotalReal; i++) {
        commandEncoder.copyBufferToBuffer(zeroBuf, 0, outBuf, i * 4, 4);
      }
    }

    const viewShape = this.ioOut.shape;
    const viewOffset = this.ioOut.offset;
    const viewTotal = prod(viewShape);
    const viewStrides = this._shapeStrides(viewShape);
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const srcBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
      const dstBase = b * viewTotal * 4;
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
        if (!inside) continue;
        const vi = this._linearFromCoords(viewCoords, viewStrides);
        this._copyAnySpan(commandEncoder, {
          src: realRef.store,
          srcOffsetBytes: srcBase + li * 4,
          dst: outBuf,
          dstOffsetBytes: dstBase + vi * 4,
          bytes: 4,
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

  _runUnpackLineChunks(commandEncoder, { packedRange, fullRange }) {
    const packedRef = this._storageRef(packedRange);
    const fullRef = this._storageRef(fullRange);
    if (this._complexLineBytes > this._maxBindBytes || this._packedLineBytes > this._maxBindBytes) {
      this._runUnpackLineElementChunks(commandEncoder, { packedRange, fullRange });
      return;
    }

    const lineBytes = Math.max(this._complexLineBytes, this._packedLineBytes);
    const linesPerChunk = this._resolveLargeStageLinesPerChunk("unpack", lineBytes);
    const chunkBuf = this._ensureLargeChunkBuffer(linesPerChunk * this._complexLineBytes);
    const packedStrides = this._shapeStrides(this.packedShape);
    const packedPerBatch = prod(this.packedShape);
    const fullPerBatch = prod(this.shape);
    const linesPerBatch = prod(this.shape.slice(1));
    const packedCoords = new Array(this.rank).fill(0);
    const fullCoords = new Array(this.rank).fill(0);
    const inNx = this.packedShape[0];
    const chunkCountPerBatch = Math.ceil(linesPerBatch / linesPerChunk);
    const chunkCount = this.batch * chunkCountPerBatch;
    const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    const paramStride = alignBytes(16, uniformAlign);
    const paramsBuf = this.device.createBuffer({
      size: chunkCount * paramStride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._deferredUniformBuffers.push(paramsBuf);
    let chunkIndex = 0;

    for (let b = 0; b < this.batch; b++) {
      const packedBase = packedRef.baseOffsetBytes + b * packedPerBatch * 8;
      const fullBase = fullRef.baseOffsetBytes + b * fullPerBatch * 8;
      for (let line0 = 0; line0 < linesPerBatch; line0 += linesPerChunk) {
        const lines = Math.min(linesPerChunk, linesPerBatch - line0);
        const chunkElems = lines * this.shape[0];
        const chunkBytes = chunkElems * 8;

        for (let lineLocal = 0; lineLocal < lines; lineLocal++) {
          const lineInBatch = line0 + lineLocal;
          this._decodeLineCoordsFromIndex(lineInBatch, fullCoords);
          for (let x = 0; x < this.shape[0]; x++) {
            const mirrored = x >= inNx;
            packedCoords[0] = mirrored ? this.shape[0] - x : x;
            for (let d = 1; d < this.rank; d++) {
              const c = fullCoords[d];
              packedCoords[d] = mirrored ? (c === 0 ? 0 : this.shape[d] - c) : c;
            }
            const srcIdx = this._linearFromCoords(packedCoords, packedStrides);
            const dstIdx = lineLocal * this.shape[0] + x;
            this._copyAnySpan(commandEncoder, {
              src: packedRef.store,
              srcOffsetBytes: packedBase + srcIdx * 8,
              dst: chunkBuf,
              dstOffsetBytes: dstIdx * 8,
              bytes: 8,
            });
          }
        }

        const paramOff = chunkIndex * paramStride;
        this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([line0, lines, 0, 0]));
        const bg = this.device.createBindGroup({
          layout: this.unpackFinalize.bgl,
          entries: [
            { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: chunkBytes } },
            { binding: 1, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.unpackFinalize.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(chunkElems / this.workgroupSize), 1, 1);
        pass.end();

        this._copyAnySpan(commandEncoder, {
          src: chunkBuf,
          srcOffsetBytes: 0,
          dst: fullRef.store,
          dstOffsetBytes: fullBase + line0 * this._complexLineBytes,
          bytes: chunkBytes,
        });
        chunkIndex += 1;
      }
    }
  }

  _runUnpackLineElementChunks(commandEncoder, { packedRange, fullRange }) {
    const packedRef = this._storageRef(packedRange);
    const fullRef = this._storageRef(fullRange);
    const maxElems = Math.max(1, Math.floor(this._maxBindBytes / 8));
    const chunkBuf = this._ensureLargeChunkBuffer(maxElems * 8);
    const packedStrides = this._shapeStrides(this.packedShape);
    const packedPerBatch = prod(this.packedShape);
    const fullPerBatch = prod(this.shape);
    const linesPerBatch = prod(this.shape.slice(1));
    const inNx = this.packedShape[0];
    const packedCoords = new Array(this.rank).fill(0);
    const fullCoords = new Array(this.rank).fill(0);

    for (let lineGlobal = 0; lineGlobal < this._lineCount; lineGlobal++) {
      const b = Math.floor(lineGlobal / linesPerBatch);
      const lineInBatch = lineGlobal - b * linesPerBatch;
      const packedBase = packedRef.baseOffsetBytes + b * packedPerBatch * 8;
      const fullBase = fullRef.baseOffsetBytes + b * fullPerBatch * 8;
      this._decodeLineCoordsFromIndex(lineInBatch, fullCoords);
      const lineSelfConj = this._isSelfConjugateLineCoords(fullCoords) ? 1 : 0;

      for (let x0 = 0; x0 < this.shape[0]; x0 += maxElems) {
        const count = Math.min(maxElems, this.shape[0] - x0);
        const chunkBytes = count * 8;
        for (let i = 0; i < count; i++) {
          const x = x0 + i;
          const mirrored = x >= inNx;
          packedCoords[0] = mirrored ? this.shape[0] - x : x;
          for (let d = 1; d < this.rank; d++) {
            const c = fullCoords[d];
            packedCoords[d] = mirrored ? (c === 0 ? 0 : this.shape[d] - c) : c;
          }
          const srcIdx = this._linearFromCoords(packedCoords, packedStrides);
          this._copyAnySpan(commandEncoder, {
            src: packedRef.store,
            srcOffsetBytes: packedBase + srcIdx * 8,
            dst: chunkBuf,
            dstOffsetBytes: i * 8,
            bytes: 8,
          });
        }

        this.device.queue.writeBuffer(this.unpackFinalizeSegment.params, 0, new Uint32Array([count, x0, lineSelfConj, 0]));
        const bg = this.device.createBindGroup({
          layout: this.unpackFinalizeSegment.bgl,
          entries: [
            { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: chunkBytes } },
            { binding: 1, resource: { buffer: this.unpackFinalizeSegment.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.unpackFinalizeSegment.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
        pass.end();

        this._copyAnySpan(commandEncoder, {
          src: chunkBuf,
          srcOffsetBytes: 0,
          dst: fullRef.store,
          dstOffsetBytes: fullBase + lineInBatch * this._complexLineBytes + x0 * 8,
          bytes: chunkBytes,
        });
      }
    }
  }

  _runComplexToRealLineChunks(commandEncoder, { fullRange, realRange }) {
    const fullRef = this._storageRef(fullRange);
    const realRef = this._storageRef(realRange);
    if (this._complexLineBytes > this._maxBindBytes || this._realLineBytes > this._maxBindBytes) {
      this._runComplexToRealElementChunks(commandEncoder, { fullRange, realRange });
      return;
    }
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const lineBytes = Math.max(this._complexLineBytes, this._realLineBytes);
    const maxLinesPerChunk = this._resolveLargeStageLinesPerChunk("complexToReal", lineBytes);
    const maxInBytes = maxLinesPerChunk * this._complexLineBytes;
    const maxOutBytes = maxLinesPerChunk * this._realLineBytes;
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
      const outBytes = lines * this._realLineBytes;
      const outOff = alignBytes(inBytes, storageAlign);
      const srcOff = fullRef.baseOffsetBytes + line0 * this._complexLineBytes;
      const dstOff = realRef.baseOffsetBytes + line0 * this._realLineBytes;

      this._copyAnySpan(commandEncoder, {
        src: fullRef.store,
        srcOffsetBytes: srcOff,
        dst: chunkBuf,
        dstOffsetBytes: 0,
        bytes: inBytes,
      });
      const paramOff = chunkIndex * paramStride;
      this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([lines * this.shape[0], 0, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: this.c2r.bgl,
        entries: [
          { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
          { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
          { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.c2r.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((lines * this.shape[0]) / this.workgroupSize), 1, 1);
      pass.end();
      this._copyAnySpan(commandEncoder, {
        src: chunkBuf,
        srcOffsetBytes: outOff,
        dst: realRef.store,
        dstOffsetBytes: dstOff,
        bytes: outBytes,
      });
      chunkIndex += 1;
    }
  }

  _runComplexToRealElementChunks(commandEncoder, { fullRange, realRange }) {
    const fullRef = this._storageRef(fullRange);
    const realRef = this._storageRef(realRange);
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxElems = Math.max(1, Math.floor(this._maxBindBytes / 8));
    const maxInBytes = maxElems * 8;
    const maxOutBytes = maxElems * 4;
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
      const srcLineBase = fullRef.baseOffsetBytes + line * this._complexLineBytes;
      const dstLineBase = realRef.baseOffsetBytes + line * this._realLineBytes;
      for (let x0 = 0; x0 < this.shape[0]; x0 += maxElems) {
        const elems = Math.min(maxElems, this.shape[0] - x0);
        const inBytes = elems * 8;
        const outBytes = elems * 4;
        this._copyAnySpan(commandEncoder, {
          src: fullRef.store,
          srcOffsetBytes: srcLineBase + x0 * 8,
          dst: chunkBuf,
          dstOffsetBytes: 0,
          bytes: inBytes,
        });
        const paramOff = chunkIndex * paramStride;
        this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([elems, 0, 0, 0]));
        const bg = this.device.createBindGroup({
          layout: this.c2r.bgl,
          entries: [
            { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
            { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
            { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.c2r.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(elems / this.workgroupSize), 1, 1);
        pass.end();
        this._copyAnySpan(commandEncoder, {
          src: chunkBuf,
          srcOffsetBytes: outOff,
          dst: realRef.store,
          dstOffsetBytes: dstLineBase + x0 * 4,
          bytes: outBytes,
        });
        chunkIndex += 1;
      }
    }
  }

  _decodeLineCoordsFromIndex(lineInBatch, outCoords) {
    let rem = lineInBatch;
    for (let d = 1; d < this.rank; d++) {
      const dim = this.shape[d];
      const c = rem % dim;
      outCoords[d] = c;
      rem = (rem - c) / dim;
    }
  }

  _isSelfConjugateLineCoords(coords) {
    for (let d = 1; d < this.rank; d++) {
      const c = coords[d];
      const n = this.shape[d];
      if (n % 2 === 0) {
        if (c !== 0 && c !== (n / 2)) return false;
      } else {
        if (c !== 0) return false;
      }
    }
    return true;
  }

  _resolveWorkspaceViews(temp) {
    const arena = temp ?? this._arena;
    if (arena) {
      if (getBufferByteLength(arena) < this.workspaceBytes) throw new Error("temp too small");
      return {
        stage: this.stageBytes ? viewFromArena(arena, this.stageOffset, this.stageBytes) : null,
        packedView: viewFromArena(arena, this.packedOffset, this.packedF32Bytes),
        complexView: viewFromArena(arena, this.fullOffset, this.fullBytes),
        realView: viewFromArena(arena, this.realOffset, this.realF32Bytes),
        f16OutScratch: this.precision === "f16-storage" ? viewFromArena(arena, this.outF16Offset, this.outBytes) : null,
      };
    }
    if (this._splitWorkspace) {
      return {
        stage: this.stageBytes ? viewFromArena(this._splitWorkspace.stage, 0, this.stageBytes) : null,
        packedView: viewFromArena(this._splitWorkspace.packed, 0, this.packedF32Bytes),
        complexView: viewFromArena(this._splitWorkspace.full, 0, this.fullBytes),
        realView: viewFromArena(this._splitWorkspace.real, 0, this.realF32Bytes),
        f16OutScratch: this.precision === "f16-storage" ? viewFromArena(this._splitWorkspace.outF16, 0, this.outBytes) : null,
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
      single(views.packedView, this.packedF32Bytes) &&
      single(views.complexView, this.fullBytes) &&
      single(views.realView, this.realF32Bytes) &&
      single(views.f16OutScratch, this.precision === "f16-storage" ? this.outBytes : 0)
    );
  }

  _resolveLargeWorkspaceRanges(temp) {
    const { packedView, complexView, realView } = this._resolveWorkspaceViews(temp);
    return {
      packedRange: { view: packedView, offsetBytes: 0, sizeBytes: this.packedF32Bytes },
      fullRange: { view: complexView, offsetBytes: 0, sizeBytes: this.fullBytes },
      realRange: { view: realView, offsetBytes: 0, sizeBytes: this.realF32Bytes },
    };
  }

  _execLargeShape(commandEncoder, { input, output, temp, inputOffsetBytes, outputOffsetBytes }) {
    const { packedRange, fullRange, realRange } = this._resolveLargeWorkspaceRanges(temp);

    if (this._usesStridedInput) {
      this._copyStridedInputPackedOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange });
    } else {
      this._embedInputComplexOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange });
    }

    if (this.zeroPadRead) {
      this._zeroOutsideRangeComplexLarge(commandEncoder, {
        dataRange: packedRange,
        shape: this.packedShape,
        start: this.zeroPadRead.start,
        end: this.zeroPadRead.end,
      });
    }

    this._runUnpackLineChunks(commandEncoder, { packedRange, fullRange });
    {
      const fullRef = this._storageRef(fullRange);
      this.c2c.exec(commandEncoder, { input: fullRef.store, inputOffsetBytes: fullRef.baseOffsetBytes });
    }
    this._runComplexToRealLineChunks(commandEncoder, { fullRange, realRange });

    if (this.zeroPadWrite) {
      this._zeroOutsideRangeRealLarge(commandEncoder, {
        dataRange: realRange,
        shape: this.shape,
        start: this.zeroPadWrite.start,
        end: this.zeroPadWrite.end,
      });
    }

    if (this._usesStridedOutput) {
      this._copyContiguousRealToStridedOutputOutOfCore(commandEncoder, { realRange, output, outputOffsetBytes });
    } else {
      this._extractOutputRealOutOfCore(commandEncoder, { realRange, output, outputOffsetBytes });
    }
  }

  destroy() {
    if (this._destroyed) return;
    this.c2c.destroy();
    this.unpack.params.destroy();
    this.unpackLine?.params?.destroy?.();
    this.unpackFinalize?.params?.destroy?.();
    this.unpackFinalizeSegment?.params?.destroy?.();
    this.c2r.params.destroy();
    this.f16In?.params?.destroy?.();
    this.f16Out?.params?.destroy?.();
    this.ioEmbed?.params?.destroy?.();
    this.ioExtract?.params?.destroy?.();
    this.stridedIn?.params?.destroy?.();
    this.stridedOut?.params?.destroy?.();
    this._largeChunkBuffer?.destroy?.();
    this._splitWorkspace?.stage?.destroy?.();
    this._splitWorkspace?.packed?.destroy?.();
    this._splitWorkspace?.full?.destroy?.();
    this._splitWorkspace?.real?.destroy?.();
    this._splitWorkspace?.outF16?.destroy?.();
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
    if (!input || !output) throw new Error("c2r exec requires input and output");
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
    const { stage, packedView, complexView, realView, f16OutScratch } = workspaceViews;

    // Load physical packed spectrum into f32, then optional ioView embed into packed-logical domain.
    if (this._usesStridedInput) {
      if (this._needsInputMapping) {
        const packedRange = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
        this._copyStridedInputPackedOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange });
      } else {
        if (!isGpuBuffer(input)) {
          const packedRange = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          this._copyStridedInputPackedOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange });
        } else {
          if (inputOffsetBytes % 8 !== 0) {
            throw new Error(`inputOffsetBytes must be a multiple of 8 for packed-complex strided input; got ${inputOffsetBytes}`);
          }
          const extraOffsetElements = (inputOffsetBytes / 8) | 0;
          const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
          ensureWithinBindingLimit(this.device, neededBytes, "c2r strided input binding");
          if (input.size < neededBytes) {
            throw new Error(`input buffer too small for strided layout: need ${neededBytes} bytes, have ${input.size}`);
          }

          const dstF32 = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([prod(this.packedShape), this.batch, extraOffsetElements, 0]));
          const bg = this.device.createBindGroup({
            layout: this.stridedIn.bgl,
            entries: [
              { binding: 0, resource: { buffer: input, offset: 0, size: neededBytes } },
              { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.packedF32Bytes } },
              { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.stridedIn.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.inTotalComplexLogical / this.workgroupSize), 1, 1);
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
        ? normalizeToContiguousRanges(stage, 0, this.inViewTotalComplex * 8)[0]
        : normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];

      const bg = this.device.createBindGroup({
        layout: this.f16In.bgl,
        entries: [
          { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.inBytes } },
          { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.inViewTotalComplex * 8 } },
          { binding: 2, resource: { buffer: this.f16In.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.f16In.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.inViewTotalComplex / this.workgroupSize), 1, 1);
      pass.end();
    } else {
      const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
      const dstF32 = this.ioEmbed
        ? normalizeToContiguousRanges(stage, 0, this.inBytes)[0]
        : normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];

      if (inRanges.length === 1) {
        commandEncoder.copyBufferToBuffer(inRanges[0].buffer, inRanges[0].offsetBytes, dstF32.buffer, dstF32.offsetBytes, this.inBytes);
      } else {
        this.copier.pack(commandEncoder, inRanges, dstF32.buffer, dstF32.offsetBytes);
      }
    }

    if (this.ioEmbed && !(this._usesStridedInput && this._needsInputMapping)) {
      this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.ioEmbed.logicalTotal, this.ioEmbed.viewTotal, this.batch, 0]));
      const src = normalizeToContiguousRanges(stage, 0, this.ioEmbed.viewTotal * this.batch * 8)[0];
      const dst = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.ioEmbed.bgl,
        entries: [
          { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.ioEmbed.viewTotal * this.batch * 8 } },
          { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.packedF32Bytes } },
          { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.ioEmbed.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((this.ioEmbed.logicalTotal * this.batch) / this.workgroupSize), 1, 1);
      pass.end();
    }

    if (this.zeroRead) {
      const p = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.zeroRead.bgl,
        entries: [{ binding: 0, resource: { buffer: p.buffer, offset: p.offsetBytes, size: this.packedF32Bytes } }],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.zeroRead.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.inTotalComplexLogical / this.workgroupSize), 1, 1);
      pass.end();
    }

    // unpack to full complex spectrum
    const packed = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
    const full = normalizeToContiguousRanges(complexView, 0, this.fullBytes)[0];
    {
      const bg = this.device.createBindGroup({
        layout: this.unpack.bgl,
        entries: [
          { binding: 0, resource: { buffer: packed.buffer, offset: packed.offsetBytes, size: this.packedF32Bytes } },
          { binding: 1, resource: { buffer: full.buffer, offset: full.offsetBytes, size: this.fullBytes } },
          { binding: 2, resource: { buffer: this.unpack.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.unpack.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalComplexFull / this.workgroupSize), 1, 1);
      pass.end();
    }

    // inverse c2c
    this.c2c.exec(commandEncoder, { input: full.buffer, inputOffsetBytes: full.offsetBytes });

    // extract real part to logical realView
    const real = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
    {
      const bg = this.device.createBindGroup({
        layout: this.c2r.bgl,
        entries: [
          { binding: 0, resource: { buffer: full.buffer, offset: full.offsetBytes, size: this.fullBytes } },
          { binding: 1, resource: { buffer: real.buffer, offset: real.offsetBytes, size: this.realF32Bytes } },
          { binding: 2, resource: { buffer: this.c2r.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.c2r.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
    }

    if (this.zeroWrite) {
      const bg = this.device.createBindGroup({
        layout: this.zeroWrite.bgl,
        entries: [{ binding: 0, resource: { buffer: real.buffer, offset: real.offsetBytes, size: this.realF32Bytes } }],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.zeroWrite.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
    }

    // optional output view mapping (logical real -> output view)
    let outF32 = real;
    if (this.ioExtract && !(this._usesStridedOutput && this._needsOutputMapping)) {
      const viewTotal = this.ioExtract.viewTotal;
      const outBytesF32 = viewTotal * this.batch * 4;
      ensureWithinBindingLimit(this.device, outBytesF32, "c2r ioView.output");
      this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.ioExtract.logicalTotal, viewTotal, this.batch, 0]));

      // f32 output can be written directly when contiguous (preserves clearOutside=false semantics).
      if (this.precision === "f32") {
        const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
        if (outRanges.length === 1) {
          const bg = this.device.createBindGroup({
            layout: this.ioExtract.bgl,
            entries: [
              { binding: 0, resource: { buffer: real.buffer, offset: real.offsetBytes, size: this.realF32Bytes } },
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

      const dst = normalizeToContiguousRanges(stage, 0, outBytesF32)[0];

      // For clearOutside=false with staged output, initialize dst from the existing output so out-of-bounds
      // view elements preserve prior values.
      if (!this.ioOut.clearOutside) {
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
          this.device.queue.writeBuffer(this.f16Out.params, 0, new Uint32Array([this.outViewTotalReal, 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.f16Out.bgl,
            entries: [
              { binding: 0, resource: { buffer: f16SrcBuf, offset: f16SrcOff, size: this.outBytes } },
              { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: outBytesF32 } },
              { binding: 2, resource: { buffer: this.f16Out.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16Out.toF32);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.outViewTotalReal / this.workgroupSize), 1, 1);
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

      const bg = this.device.createBindGroup({
        layout: this.ioExtract.bgl,
        entries: [
          { binding: 0, resource: { buffer: real.buffer, offset: real.offsetBytes, size: this.realF32Bytes } },
          { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: outBytesF32 } },
          { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.ioExtract.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
      pass.end();
      outF32 = dst;
    }

    if (this._usesStridedOutput) {
      if (this._needsOutputMapping) {
        this._copyContiguousRealToStridedOutputOutOfCore(commandEncoder, {
          realRange: outF32,
          output,
          outputOffsetBytes,
        });
        return;
      }
      if (!isGpuBuffer(output)) {
        this._copyContiguousRealToStridedOutputOutOfCore(commandEncoder, {
          realRange: outF32,
          output,
          outputOffsetBytes,
        });
        return;
      }
      if (outputOffsetBytes % 4 !== 0) {
        throw new Error(`outputOffsetBytes must be a multiple of 4 for real-strided output; got ${outputOffsetBytes}`);
      }
      const extraOffsetElements = (outputOffsetBytes / 4) | 0;
      const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
      ensureWithinBindingLimit(this.device, neededBytes, "c2r strided output binding");
      if (output.size < neededBytes) {
        throw new Error(`output buffer too small for strided layout: need ${neededBytes} bytes, have ${output.size}`);
      }

      this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([this.logicalTotal, this.batch, extraOffsetElements, 0]));
      const bg = this.device.createBindGroup({
        layout: this.stridedOut.bgl,
        entries: [
          { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.realF32Bytes } },
          { binding: 1, resource: { buffer: output, offset: 0, size: neededBytes } },
          { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.stridedOut.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
      pass.end();
      return;
    }

    // write output
    if (this.precision === "f16-storage") {
      const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
      if (outRanges.length === 1) {
        const bg = this.device.createBindGroup({
          layout: this.f16Out.bgl,
          entries: [
            { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.outViewTotalReal * 4 } },
            { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
            { binding: 2, resource: { buffer: this.f16Out.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.f16Out.toF16);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.outViewTotalReal / this.workgroupSize), 1, 1);
        pass.end();
        return;
      }
      const tmp = normalizeToContiguousRanges(f16OutScratch, 0, this.outBytes)[0];
      const bg = this.device.createBindGroup({
        layout: this.f16Out.bgl,
        entries: [
          { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.outViewTotalReal * 4 } },
          { binding: 1, resource: { buffer: tmp.buffer, offset: tmp.offsetBytes, size: this.outBytes } },
          { binding: 2, resource: { buffer: this.f16Out.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.f16Out.toF16);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.outViewTotalReal / this.workgroupSize), 1, 1);
      pass.end();
      this.copier.unpack(commandEncoder, tmp.buffer, tmp.offsetBytes, outRanges);
      return;
    }

    const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
    if (outRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(outF32.buffer, outF32.offsetBytes, outRanges[0].buffer, outRanges[0].offsetBytes, this.outBytes);
    } else {
      this.copier.unpack(commandEncoder, outF32.buffer, outF32.offsetBytes, outRanges);
    }
  }
}

