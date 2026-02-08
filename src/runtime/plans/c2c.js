// Copyright (c) 2026 Maksim Eremenko

import { createFftPlan } from "../../plan.js";
import { BasePlan } from "../base_plan.js";
import { resolveLargeRoutingPolicy, resolveAxisKindsForShape, resolveOutOfCoreAxisWindowPolicy } from "../large_policy.js";
import { normalizeIoView } from "../ioview.js";
import { normalizeZeroPad } from "../zero_pad.js";
import { viewFromArena, createInternalArena } from "../workspace.js";
import { normalizeToContiguousRanges } from "../segmented_io.js";
import { resolveLayoutSemantics } from "../layout_semantics.js";
import {
  assertOneOf,
  isPositiveInt,
  prod,
  alignBytes,
  normalizeScaleFactor,
  ensureWithinBindingLimit,
  isGpuBuffer,
  getBufferByteLength,
  buffersAlias,
} from "../common.js";
import {
  contiguousStrides as tensorContiguousStrides,
  coordsFromLinear as tensorCoordsFromLinear,
  linearFromCoords as tensorLinearFromCoords,
  createTensorDescriptor,
  requiredBytesForBatchRange,
} from "../tensor_descriptor.js";

import { generateScaleComplexWGSL } from "../../kernels/scale.js";
import { generateZeroOutsideRangeComplexWGSL } from "../../kernels/zero_pad.js";
import {
  generateEmbedComplexWGSL,
  generateExtractComplexWGSL,
  generateEmbedComplexF16ToF32WGSL,
  generateExtractComplexF32ToF16WGSL,
} from "../../kernels/ioview.js";
import { generateTransposeComplex2DWGSL } from "../../kernels/transpose.js";
import { generateF16ToF32ComplexWGSL, generateF32ToF16ComplexWGSL } from "../../kernels/f16_storage.js";
import { generateGatherComplexStridedWGSL, generateScatterComplexStridedWGSL } from "../../kernels/strided_complex.js";

import { BluesteinAxis } from "../algorithms/bluestein_axis.js";
import { RaderAxis } from "../algorithms/rader_axis.js";

function needsIoMapping(io, logicalShape) {
  if (!io) return false;
  for (let i = 0; i < logicalShape.length; i++) {
    if (io.shape[i] !== logicalShape[i]) return true;
    if (io.offset[i] !== 0) return true;
  }
  return false;
}

function permutedShapeAxisFront(shape, axis) {
  const out = [shape[axis]];
  for (let d = 0; d < shape.length; d++) {
    if (d === axis) continue;
    out.push(shape[d]);
  }
  return out;
}

function alignDownBytes(v, alignment) {
  if (!Number.isFinite(alignment) || alignment <= 1) return Math.max(0, v | 0);
  return Math.max(0, Math.floor(v / alignment) * alignment);
}

function generatePermuteRank3Axis2ToFrontWGSL({ shape, workgroupSize }) {
  const X = shape[0] | 0;
  const Y = shape[1] | 0;
  const Z = shape[2] | 0;
  const XY = X * Y;
  const ZX = Z * X;
  return /* wgsl */ `
struct Params {
  count: u32,
  hz: u32,
  srcStartElements: u32,
  dstStartElements: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const X: u32 = ${X}u;
const Z: u32 = ${Z}u;
const XY: u32 = ${XY}u;
const ZX: u32 = ${ZX}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count || params.hz == 0u) { return; }
  let x: u32 = i % X;
  let rem: u32 = i / X;
  let z: u32 = rem % params.hz;
  let y: u32 = rem / params.hz;
  let srcIdx: u32 = params.srcStartElements + x + y * X + z * XY;
  let dstIdx: u32 = params.dstStartElements + z + x * Z + y * ZX;
  dst[dstIdx] = src[srcIdx];
}
`;
}

function generatePermuteRank3Axis2FromFrontWGSL({ shape, workgroupSize }) {
  const X = shape[0] | 0;
  const Y = shape[1] | 0;
  const Z = shape[2] | 0;
  const XY = X * Y;
  const ZX = Z * X;
  return /* wgsl */ `
struct Params {
  count: u32,
  hz: u32,
  srcStartElements: u32,
  dstStartElements: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const X: u32 = ${X}u;
const Z: u32 = ${Z}u;
const XY: u32 = ${XY}u;
const ZX: u32 = ${ZX}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count || params.hz == 0u) { return; }
  let x: u32 = i % X;
  let rem: u32 = i / X;
  let z: u32 = rem % params.hz;
  let y: u32 = rem / params.hz;
  let srcIdx: u32 = params.srcStartElements + z + x * Z + y * ZX;
  let dstIdx: u32 = params.dstStartElements + x + y * X + z * XY;
  dst[dstIdx] = src[srcIdx];
}
`;
}

function generatePermuteAxis1TailToFrontWGSL({ shape, workgroupSize }) {
  const X = shape[0] | 0;
  const Y = shape[1] | 0;
  const XY = X * Y;
  return /* wgsl */ `
struct Params {
  count: u32,
  srcStartElements: u32,
  dstStartElements: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const X: u32 = ${X}u;
const Y: u32 = ${Y}u;
const XY: u32 = ${XY}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count) { return; }
  let xy: u32 = i % XY;
  let t: u32 = i / XY;
  let x: u32 = xy % X;
  let y: u32 = xy / X;
  let srcIdx: u32 = params.srcStartElements + x + y * X + t * XY;
  let dstIdx: u32 = params.dstStartElements + y + x * Y + t * XY;
  dst[dstIdx] = src[srcIdx];
}
`;
}

function generatePermuteAxis1TailFromFrontWGSL({ shape, workgroupSize }) {
  const X = shape[0] | 0;
  const Y = shape[1] | 0;
  const XY = X * Y;
  return /* wgsl */ `
struct Params {
  count: u32,
  srcStartElements: u32,
  dstStartElements: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const X: u32 = ${X}u;
const Y: u32 = ${Y}u;
const XY: u32 = ${XY}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count) { return; }
  let xy: u32 = i % XY;
  let t: u32 = i / XY;
  let x: u32 = xy % X;
  let y: u32 = xy / X;
  let srcIdx: u32 = params.srcStartElements + y + x * Y + t * XY;
  let dstIdx: u32 = params.dstStartElements + x + y * X + t * XY;
  dst[dstIdx] = src[srcIdx];
}
`;
}

function generatePermuteAxis1TailTiledToFrontWGSL({ shape, workgroupSize }) {
  const X = shape[0] | 0;
  const Y = shape[1] | 0;
  const XY = X * Y;
  return /* wgsl */ `
struct Params {
  count: u32,
  hx: u32,
  htail: u32,
  srcStartElements: u32,
  dstStartElements: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const X: u32 = ${X}u;
const Y: u32 = ${Y}u;
const XY: u32 = ${XY}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count || params.hx == 0u || params.htail == 0u) { return; }
  let x: u32 = i % params.hx;
  let t: u32 = i / params.hx;
  let srcIdx: u32 = params.srcStartElements + x + t * XY;
  let dstIdx: u32 = params.dstStartElements + x * Y + t * XY;
  dst[dstIdx] = src[srcIdx];
}
`;
}

function generatePermuteAxisGenericWGSL({ shape, axis, toFront, workgroupSize }) {
  const rank = shape.length;
  const srcShape = toFront ? shape.slice() : permutedShapeAxisFront(shape, axis);
  const dstShape = toFront ? permutedShapeAxisFront(shape, axis) : shape.slice();
  const srcStrides = tensorContiguousStrides(srcShape);
  const dstStrides = tensorContiguousStrides(dstShape);
  const total = prod(srcShape);

  const decodeSrcCoords = [];
  let rem = "li";
  for (let d = 0; d < rank; d++) {
    const dim = srcShape[d] | 0;
    const v = `s${d}`;
    decodeSrcCoords.push(`  let ${v}: u32 = ${rem} % ${dim}u;`);
    if (d < rank - 1) {
      const next = `rem${d}`;
      decodeSrcCoords.push(`  let ${next}: u32 = ${rem} / ${dim}u;`);
      rem = next;
    }
  }

  const mapDstCoords = [];
  if (toFront) {
    mapDstCoords.push(`  let d0: u32 = s${axis};`);
    let p = 1;
    for (let d = 0; d < rank; d++) {
      if (d === axis) continue;
      mapDstCoords.push(`  let d${p}: u32 = s${d};`);
      p += 1;
    }
  } else {
    mapDstCoords.push(`  let d${axis}: u32 = s0;`);
    let p = 1;
    for (let d = 0; d < rank; d++) {
      if (d === axis) continue;
      mapDstCoords.push(`  let d${d}: u32 = s${p};`);
      p += 1;
    }
  }

  let dstIdxExpr = "0u";
  for (let d = 0; d < rank; d++) {
    const stride = dstStrides[d] | 0;
    dstIdxExpr += ` + d${d} * ${stride}u`;
  }
  let srcIdxExpr = "0u";
  for (let d = 0; d < rank; d++) {
    const stride = srcStrides[d] | 0;
    srcIdxExpr += ` + s${d} * ${stride}u`;
  }

  return /* wgsl */ `
struct Params {
  count: u32,
  batch: u32,
  srcStartElements: u32,
  dstStartElements: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const TOTAL: u32 = ${total}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count) { return; }
  let b: u32 = i / TOTAL;
  if (b >= params.batch) { return; }
  let li: u32 = i - b * TOTAL;
${decodeSrcCoords.join("\n")}
${mapDstCoords.join("\n")}
  let srcIdx: u32 = params.srcStartElements + b * TOTAL + (${srcIdxExpr});
  let dstIdx: u32 = params.dstStartElements + b * TOTAL + (${dstIdxExpr});
  dst[dstIdx] = src[srcIdx];
}
`;
}

function generateGatherAxis2SlabWGSL({ N, workgroupSize }) {
  return /* wgsl */ `
struct Params {
  count: u32,
  zStart: u32,
  zCount: u32,
  srcStartElements: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const N: u32 = ${N}u;
const PLANE_ELEMS: u32 = ${N * N}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count || params.zCount == 0u) { return; }
  let x: u32 = i % N;
  let zLocal: u32 = i / N;
  let srcIdx: u32 = params.srcStartElements + zLocal * PLANE_ELEMS + x;
  let dstIdx: u32 = (params.zStart + zLocal) * N + x;
  dst[dstIdx] = src[srcIdx];
}
`;
}

function generateScatterAxis2SlabWGSL({ N, workgroupSize }) {
  return /* wgsl */ `
struct Params {
  count: u32,
  zStart: u32,
  zCount: u32,
  dstStartElements: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const N: u32 = ${N}u;
const PLANE_ELEMS: u32 = ${N * N}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count || params.zCount == 0u) { return; }
  let x: u32 = i % N;
  let zLocal: u32 = i / N;
  let srcIdx: u32 = (params.zStart + zLocal) * N + x;
  let dstIdx: u32 = params.dstStartElements + zLocal * PLANE_ELEMS + x;
  dst[dstIdx] = src[srcIdx];
}
`;
}

function maxComputeWorkgroupsPerDimension(limits, axis) {
  const raw = limits?.maxComputeWorkgroupsPerDimension;
  if (raw == null) return Infinity;
  if (Array.isArray(raw) || ArrayBuffer.isView(raw)) {
    const value = raw[axis];
    if (!Number.isFinite(value) || value < 1) return 0;
    return Math.floor(value);
  }
  if (!Number.isFinite(raw) || raw < 1) return 0;
  return Math.floor(raw);
}

function parseOptionalAxisList(v, rank, name) {
  if (v == null) return [];
  if (!Array.isArray(v) || !v.every((x) => Number.isInteger(x) && x >= 0 && x < rank)) {
    throw new Error(`${name} must be an array of axis indices in [0, ${rank - 1}]`);
  }
  return [...new Set(v.map((x) => x | 0))];
}

function parseC2cTuning(tuning, rank) {
  if (tuning == null) {
    return {
      raderMaxPrime: 4096,
      transposeMinElements: 4096,
      disableTranspose: false,
      disableOutOfCoreFourStep: false,
      largeChunkMaxBatches: null,
      largeRoute: "auto",
      preferOutOfCoreForStrided: null,
      maxStorageBufferBindingSize: null,
      swapTo2Stage4Step: 0,
      swapTo3Stage4Step: 0,
      groupedBatch: null,
      outOfCoreBurstWindows: 1,
      forceBluesteinAxes: [],
      forceRaderAxes: [],
      forceBluesteinAxisSet: new Set(),
      forceRaderAxisSet: new Set(),
    };
  }
  if (typeof tuning !== "object") {
    throw new Error("tuning must be an object when provided");
  }

  const raderMaxPrime = tuning.raderMaxPrime ?? 4096;
  if (!Number.isInteger(raderMaxPrime) || raderMaxPrime < 2) {
    throw new Error(`tuning.raderMaxPrime must be an integer >= 2; got ${raderMaxPrime}`);
  }

  const transposeMinElements = tuning.transposeMinElements ?? 4096;
  if (!Number.isInteger(transposeMinElements) || transposeMinElements < 1) {
    throw new Error(`tuning.transposeMinElements must be a positive integer; got ${transposeMinElements}`);
  }

  const disableTranspose = tuning.disableTranspose ?? false;
  if (typeof disableTranspose !== "boolean") {
    throw new Error(`tuning.disableTranspose must be boolean; got ${disableTranspose}`);
  }
  const disableOutOfCoreFourStep = tuning.disableOutOfCoreFourStep ?? false;
  if (typeof disableOutOfCoreFourStep !== "boolean") {
    throw new Error(`tuning.disableOutOfCoreFourStep must be boolean; got ${disableOutOfCoreFourStep}`);
  }
  const largeChunkMaxBatches = tuning.largeChunkMaxBatches ?? null;
  if (largeChunkMaxBatches != null) {
    if (!Number.isInteger(largeChunkMaxBatches) || largeChunkMaxBatches <= 0) {
      throw new Error(`tuning.largeChunkMaxBatches must be a positive integer; got ${largeChunkMaxBatches}`);
    }
  }
  const largeRoute = tuning.largeRoute ?? "auto";
  if (largeRoute !== "auto" && largeRoute !== "chunk" && largeRoute !== "out-of-core") {
    throw new Error(`tuning.largeRoute must be one of "auto" | "chunk" | "out-of-core"; got ${largeRoute}`);
  }
  const preferOutOfCoreForStrided = tuning.preferOutOfCoreForStrided ?? null;
  if (preferOutOfCoreForStrided != null && typeof preferOutOfCoreForStrided !== "boolean") {
    throw new Error(
      `tuning.preferOutOfCoreForStrided must be boolean when provided; got ${preferOutOfCoreForStrided}`
    );
  }
  const maxStorageBufferBindingSize = tuning.maxStorageBufferBindingSize ?? null;
  if (maxStorageBufferBindingSize != null) {
    if (!Number.isInteger(maxStorageBufferBindingSize) || maxStorageBufferBindingSize <= 0) {
      throw new Error(`tuning.maxStorageBufferBindingSize must be a positive integer; got ${maxStorageBufferBindingSize}`);
    }
  }
  const swapTo2Stage4Step = tuning.swapTo2Stage4Step ?? 0;
  if (!Number.isInteger(swapTo2Stage4Step) || swapTo2Stage4Step < 0) {
    throw new Error(`tuning.swapTo2Stage4Step must be a non-negative integer; got ${swapTo2Stage4Step}`);
  }
  const swapTo3Stage4Step = tuning.swapTo3Stage4Step ?? 0;
  if (!Number.isInteger(swapTo3Stage4Step) || swapTo3Stage4Step < 0) {
    throw new Error(`tuning.swapTo3Stage4Step must be a non-negative integer; got ${swapTo3Stage4Step}`);
  }
  const groupedBatch = tuning.groupedBatch ?? null;
  if (groupedBatch != null) {
    if (Number.isInteger(groupedBatch)) {
      if (groupedBatch <= 0) {
        throw new Error(`tuning.groupedBatch must be a positive integer or axis-indexed array; got ${groupedBatch}`);
      }
    } else if (Array.isArray(groupedBatch)) {
      if (!groupedBatch.every((v) => v == null || (Number.isInteger(v) && v > 0))) {
        throw new Error("tuning.groupedBatch array entries must be positive integers or null");
      }
    } else {
      throw new Error(`tuning.groupedBatch must be a positive integer or axis-indexed array; got ${groupedBatch}`);
    }
  }
  const outOfCoreBurstWindows = tuning.outOfCoreBurstWindows ?? 1;
  if (!Number.isInteger(outOfCoreBurstWindows) || outOfCoreBurstWindows <= 0) {
    throw new Error(`tuning.outOfCoreBurstWindows must be a positive integer; got ${outOfCoreBurstWindows}`);
  }

  const forceBluesteinAxes = parseOptionalAxisList(tuning.forceBluesteinAxes ?? null, rank, "tuning.forceBluesteinAxes");
  const forceRaderAxes = parseOptionalAxisList(tuning.forceRaderAxes ?? null, rank, "tuning.forceRaderAxes");
  const forceBluesteinAxisSet = new Set(forceBluesteinAxes);
  const forceRaderAxisSet = new Set(forceRaderAxes);
  for (const axis of forceBluesteinAxisSet) {
    if (forceRaderAxisSet.has(axis)) {
      throw new Error(`Axis ${axis} cannot be forced to both Bluestein and Rader`);
    }
  }

  return {
    raderMaxPrime,
    transposeMinElements,
    disableTranspose,
    disableOutOfCoreFourStep,
    largeChunkMaxBatches,
    largeRoute,
    preferOutOfCoreForStrided,
    maxStorageBufferBindingSize,
    swapTo2Stage4Step,
    swapTo3Stage4Step,
    groupedBatch,
    outOfCoreBurstWindows,
    forceBluesteinAxes,
    forceRaderAxes,
    forceBluesteinAxisSet,
    forceRaderAxisSet,
  };
}

export class C2CPlan extends BasePlan {
  constructor(device, opts) {
    super(device, opts);
    const {
      shape,
      direction,
      batch = 1,
      inPlace = false,
      normalize = "none",
      layout = { interleavedComplex: true },
      precision = "f32",
      ioView = null,
      zeroPad = null,
    } = opts ?? {};

    if (!Array.isArray(shape) || shape.length < 1) {
      throw new Error(`shape must be an array of one or more positive dimensions; got ${JSON.stringify(shape)}`);
    }
    if (!shape.every(isPositiveInt)) throw new Error(`shape elements must be positive ints; got ${JSON.stringify(shape)}`);
    assertOneOf(direction, ["forward", "inverse"], "direction");
    assertOneOf(normalize, ["none", "backward", "unitary"], "normalize");
    if (!Number.isInteger(batch) || batch <= 0) throw new Error(`batch must be positive int; got ${batch}`);
    if (layout?.interleavedComplex !== true) throw new Error("c2c requires layout.interleavedComplex=true");
    assertOneOf(precision, ["f32", "f16-storage"], "precision");
    if (precision === "f16-storage" && !device.features?.has?.("shader-f16")) {
      throw new Error('precision="f16-storage" requires device.features.has("shader-f16")');
    }

    this.shape = shape.slice();
    this.rank = shape.length;
    this.direction = direction;
    this.batch = batch;
    this.inPlace = !!inPlace;
    this.normalize = normalize;
    this.precision = precision;
    this._axis01MatrixBatch = this.batch * (this.rank > 2 ? prod(this.shape.slice(2)) : 1);
    this.tuning = parseC2cTuning(opts?.tuning ?? null, this.rank);
    this.io = normalizeIoView(this.rank, this.shape, ioView ?? {});
    this.zeroPad = normalizeZeroPad(this.rank, this.shape, zeroPad ?? null, "zeroPad");
    this._needsInputMapping = !!(this.io.input && needsIoMapping(this.io.input, this.shape));
    this._needsOutputMapping = !!(this.io.output && needsIoMapping(this.io.output, this.shape));
    this._inputLayoutShape = this._needsInputMapping ? this.io.input.shape.slice() : this.shape.slice();
    this._outputLayoutShape = this._needsOutputMapping ? this.io.output.shape.slice() : this.shape.slice();

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
          name: "c2c.input",
          shape: this._inputLayoutShape,
          strides: this._inputStrides,
          offsetElements: this._inputOffsetElements,
          batchStrideElements: this._inputBatchStrideElements,
        })
      : null;
    this._outputTensorDesc = this._usesStridedOutput
      ? createTensorDescriptor({
          name: "c2c.output",
          shape: this._outputLayoutShape,
          strides: this._outputStrides,
          offsetElements: this._outputOffsetElements,
          batchStrideElements: this._outputBatchStrideElements,
        })
      : null;
    this._inputSpanElements = this._inputTensorDesc?.spanElements ?? 0;
    this._outputSpanElements = this._outputTensorDesc?.spanElements ?? 0;
    if ((this._usesStridedInput || this._usesStridedOutput) && this.precision !== "f32") {
      throw new Error('custom strides currently support precision:"f32" only');
    }

    // Axis algorithm choice
    const axisStrategy = resolveAxisKindsForShape({
      shape: this.shape,
      tuning: {
        raderMaxPrime: this.tuning.raderMaxPrime,
        forceBluesteinAxes: this.tuning.forceBluesteinAxes,
        forceRaderAxes: this.tuning.forceRaderAxes,
      },
    });
    this.axisKind = axisStrategy.axisKinds;
    this.logicalTotal = prod(this.shape);
    this.totalComplex = this.logicalTotal * this.batch;
    this.mainBytes = this.totalComplex * 8;
    this._bytesPerBatch = this.logicalTotal * 8;

    const largePolicy = resolveLargeRoutingPolicy({
      device,
      tuning: {
        maxStorageBufferBindingSize: this.tuning.maxStorageBufferBindingSize,
        largeRoute: this.tuning.largeRoute,
        preferOutOfCoreForStrided: this.tuning.preferOutOfCoreForStrided,
      },
      requiredBindingBytes: [this.mainBytes],
      lineBytes: this.shape.map((n) => n * 8),
      axisKinds: this.axisKind,
      axisLengths: this.shape,
      allowNonMixedBoundedSlicing: true,
      allowOutOfCore: this.rank >= 2,
      disableOutOfCore: this.tuning.disableOutOfCoreFourStep,
      rank: this.rank,
      bytesPerBatch: this._bytesPerBatch,
      precision: this.precision,
      hasStridedIO: this._usesStridedInput || this._usesStridedOutput,
      preferOutOfCoreForStrided: true,
      outOfCoreUnsupportedError: ({ maxBindBytes, axisSupported, attemptedRoutes, reasonCodes }) =>
        [
          `c2c shape=${JSON.stringify(shape)} batch=${batch} requires ${this.mainBytes} bytes total,`,
          `and one batch requires ${this._bytesPerBatch} bytes > maxStorageBufferBindingSize=${maxBindBytes}.`,
          `Out-of-core fallback is available only for rank>=2 precision:"f32",`,
          `and when each axis line is compatible with the active axis strategy`,
          `(mixed-radix: direct or two-step factorable; Bluestein/Rader: line fits maxBufferSize for multi-upload slicing).`,
          `(axisBytes=${JSON.stringify(this.shape.map((n) => n * 8))}, axisKind=${JSON.stringify(this.axisKind)}, axisSupported=${JSON.stringify(axisSupported)}).`,
          `(attemptedRoutes=${JSON.stringify(attemptedRoutes)}, reasonCodes=${JSON.stringify(reasonCodes)}).`,
        ].join(" "),
    });
    this._maxBindBytes = largePolicy.maxBindBytes;
    this._largeBatchChunkMode = largePolicy.needsLargeMode;
    this._outOfCoreFourStepMode = largePolicy.useOutOfCore;
    this._largeRouteMode = largePolicy.routeMode;
    this._largeRouteReasons = largePolicy.reasonCodes;
    this._largeRouteAttempts = largePolicy.attemptedRoutes;
    if (!this._largeBatchChunkMode) {
      ensureWithinBindingLimit(device, this.mainBytes, `c2c shape=${JSON.stringify(shape)} batch=${batch}`);
    }

    // mixed axis plans
    this.axisPlans = new Array(this.rank).fill(null);
    for (let axis = 0; axis < this.rank; axis++) {
      if (this.axisKind[axis] === "mixed") {
        if (this._outOfCoreFourStepMode && axis !== 0) continue;
        const axisNormalize = this._outOfCoreFourStepMode ? "none" : (this._largeBatchChunkMode ? this.normalize : "none");
        this.axisPlans[axis] = createFftPlan(device, {
          shape: this.shape,
          direction,
          normalize: axisNormalize,
          inPlace: true,
          layout: "interleaved",
          precision: "f32",
          axes: [axis],
        });
      }
    }

    // advanced axis ops
    this.axisAdvanced = new Array(this.rank).fill(null);
    this.maxAxisWorkBytes = 0;
    for (let axis = 0; axis < this.rank; axis++) {
      const kind = this.axisKind[axis];
      const axisLineBytes = this.shape[axis] * 8;
      if (this._outOfCoreFourStepMode && kind === "rader" && axisLineBytes > this._maxBindBytes) {
        // Oversized out-of-core Rader axes are routed to Bluestein fallback executors below.
        continue;
      }
      if (this.axisKind[axis] === "bluestein") {
        const ax = new BluesteinAxis(device, this.cache, {
          shape: this.shape,
          rank: this.rank,
          batch,
          axis,
          direction,
          workgroupSize: this.workgroupSize,
          maxWorkBytes: this._maxBindBytes,
        });
        this.axisAdvanced[axis] = ax;
        this.maxAxisWorkBytes = Math.max(this.maxAxisWorkBytes, ax.workBytes);
      } else if (this.axisKind[axis] === "rader") {
        const ax = new RaderAxis(device, this.cache, {
          shape: this.shape,
          rank: this.rank,
          batch,
          axis,
          direction,
          workgroupSize: this.workgroupSize,
          maxWorkBytes: this._maxBindBytes,
        });
        this.axisAdvanced[axis] = ax;
        this.maxAxisWorkBytes = Math.max(this.maxAxisWorkBytes, ax.workBytes);
      }
    }
    this._outOfCoreAxis0OnTransposed = null;
    this._outOfCoreAxisPlans = null;
    this._outOfCoreAxisPermShapes = null;
    this._outOfCoreAxisWindowPolicy = null;
    this._outOfCoreTranspose = null;
    this._outOfCoreTransposePipelines = null;
    this._outOfCoreAxis1TailPermute = null;
    this._outOfCoreAxis1TailChunk = null;
    this._outOfCoreRank3Axis2Permute = null;
    this._outOfCoreRank3Axis2Tile = null;
    this._outOfCoreGenericPermute = null;
    this._outOfCoreGenericPermutePipelines = null;
    this._outOfCoreAdjacentSwapPipelines = null;
    this._outOfCoreAdjacentSwapTiled = null;
    this._outOfCoreAdjacentSwapTiledPipelines = null;
    if (this._outOfCoreFourStepMode) {
      this._outOfCoreAxisPlans = new Array(this.rank).fill(null);
      this._outOfCoreAxisPermShapes = new Array(this.rank).fill(null);
      this._outOfCoreAxisEffectiveKind = new Array(this.rank).fill(null);
      this._outOfCoreAxisWindowPolicy = new Array(this.rank).fill(null);
      for (let axis = 0; axis < this.rank; axis++) {
        const kind = this.axisKind[axis];
        const permShape = axis === 0 ? this.shape.slice() : permutedShapeAxisFront(this.shape, axis);
        const axisLineBytes = permShape[0] * 8;
        const axisLinesTotal = this.batch * (this.logicalTotal / permShape[0]);
        const axisWindowPolicy = resolveOutOfCoreAxisWindowPolicy({
          axisLen: permShape[0],
          lineBytes: axisLineBytes,
          linesTotal: axisLinesTotal,
          maxBindBytes: this._maxBindBytes,
          axisKind: kind,
          tuning: this.tuning,
          axisIndex: axis,
          storageAlign: this.device.limits?.minStorageBufferOffsetAlignment ?? 256,
        });
        this._outOfCoreAxisWindowPolicy[axis] = axisWindowPolicy;
        this._outOfCoreAxisPermShapes[axis] = permShape;
        if (kind === "mixed") {
          this._outOfCoreAxisEffectiveKind[axis] = "mixed";
          const stagedBind = Math.max(8, Math.floor(this._maxBindBytes / axisWindowPolicy.numAxisUploads));
          const effectiveAxisBind = Math.min(this._maxBindBytes, stagedBind);
          this._outOfCoreAxisPlans[axis] = createFftPlan(device, {
            shape: permShape,
            direction,
            normalize: "none",
            inPlace: true,
            layout: "interleaved",
            precision: "f32",
            axes: [0],
            maxStorageBufferBindingSize: effectiveAxisBind,
          });
          continue;
        }
        if (kind === "bluestein") {
          this._outOfCoreAxisEffectiveKind[axis] = "bluestein";
          if (axis === 0) {
            this._outOfCoreAxisPlans[axis] = this.axisAdvanced[axis];
          } else {
            this._outOfCoreAxisPlans[axis] = new BluesteinAxis(device, this.cache, {
              shape: permShape,
              rank: this.rank,
              batch,
              axis: 0,
              direction,
              workgroupSize: this.workgroupSize,
              maxWorkBytes: this._maxBindBytes,
            });
          }
        } else if (axisLineBytes > this._maxBindBytes) {
          // Oversized Rader lines use Bluestein under out-of-core multi-upload so no single
          // storage binding needs to cover the full line.
          this._outOfCoreAxisEffectiveKind[axis] = "bluestein-fallback";
          this._outOfCoreAxisPlans[axis] = new BluesteinAxis(device, this.cache, {
            shape: permShape,
            rank: this.rank,
            batch,
            axis: 0,
            direction,
            workgroupSize: this.workgroupSize,
            maxWorkBytes: this._maxBindBytes,
          });
        } else {
          this._outOfCoreAxisEffectiveKind[axis] = "rader";
          if (axis === 0) {
            this._outOfCoreAxisPlans[axis] = this.axisAdvanced[axis];
          } else {
            this._outOfCoreAxisPlans[axis] = new RaderAxis(device, this.cache, {
              shape: permShape,
              rank: this.rank,
              batch,
              axis: 0,
              direction,
              workgroupSize: this.workgroupSize,
              maxWorkBytes: this._maxBindBytes,
            });
          }
        }
      }
      // Backward-compat alias used in unit tests for the axis-1 out-of-core path.
      this._outOfCoreAxis0OnTransposed = this.rank >= 2 ? this._outOfCoreAxisPlans[1] : null;
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this._outOfCoreTranspose = { bgl, pl, params, tile: 16 };
      this._outOfCoreTransposePipelines = new Map();

      if (this.precision === "f32" && this.rank >= 3) {
        const pbglAxis1 = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const pplAxis1 = device.createPipelineLayout({ bindGroupLayouts: [pbglAxis1] });
        const axis1ToFrontCode = generatePermuteAxis1TailToFrontWGSL({
          shape: this.shape,
          workgroupSize: this.workgroupSize,
        });
        const axis1FromFrontCode = generatePermuteAxis1TailFromFrontWGSL({
          shape: this.shape,
          workgroupSize: this.workgroupSize,
        });
        const axis1ToFront = this.cache.getComputePipeline({ code: axis1ToFrontCode, layout: pplAxis1 });
        const axis1FromFront = this.cache.getComputePipeline({ code: axis1FromFrontCode, layout: pplAxis1 });
        const axis1Params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._outOfCoreAxis1TailPermute = {
          bgl: pbglAxis1,
          pl: pplAxis1,
          toFront: axis1ToFront,
          fromFront: axis1FromFront,
          params: axis1Params,
        };
      }

      if (this.precision === "f32" && this.rank === 3) {
        const pbgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const ppl = device.createPipelineLayout({ bindGroupLayouts: [pbgl] });
        const toFrontCode = generatePermuteRank3Axis2ToFrontWGSL({
          shape: this.shape,
          workgroupSize: this.workgroupSize,
        });
        const fromFrontCode = generatePermuteRank3Axis2FromFrontWGSL({
          shape: this.shape,
          workgroupSize: this.workgroupSize,
        });
        const toFront = this.cache.getComputePipeline({ code: toFrontCode, layout: ppl });
        const fromFront = this.cache.getComputePipeline({ code: fromFrontCode, layout: ppl });
        const pparams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._outOfCoreRank3Axis2Permute = {
          bgl: pbgl,
          pl: ppl,
          toFront,
          fromFront,
          params: pparams,
        };
      }

      if (this.precision === "f32" && this.rank >= 2) {
        const pbgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const ppl = device.createPipelineLayout({ bindGroupLayouts: [pbgl] });
        const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._outOfCoreGenericPermute = { bgl: pbgl, pl: ppl, params };
        this._outOfCoreGenericPermutePipelines = new Map();
        const tiledParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._outOfCoreAdjacentSwapTiled = { bgl: pbgl, pl: ppl, params: tiledParams };
        this._outOfCoreAdjacentSwapTiledPipelines = new Map();
      }
    }

    // scale pipeline
    this.scale = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateScaleComplexWGSL({ workgroupSize: this.workgroupSize });
      const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
      const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      return { bgl, pl: pipelineLayout, pipeline, params };
    })();

    this.zeroRead = null;
    this.zeroWrite = null;
    const makeZeroPipeline = (stage) => {
      const bgl = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateZeroOutsideRangeComplexWGSL({
        shape: this.shape,
        start: stage.start,
        end: stage.end,
        batch: this.batch,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      return { bgl, pl, pipeline };
    };
    if (this.zeroPad.read) this.zeroRead = makeZeroPipeline(this.zeroPad.read);
    if (this.zeroPad.write) this.zeroWrite = makeZeroPipeline(this.zeroPad.write);

    // ioView mapping
    this.ioEmbed = null;
    this.ioExtract = null;
    this._bytesPerComplexIO = this.precision === "f16-storage" ? 4 : 8;
    this._inViewTotal = this.io.input ? prod(this.io.input.shape) : this.logicalTotal;
    this._outViewTotal = this.io.output ? prod(this.io.output.shape) : this.logicalTotal;
    this._inPhysComplexPerBatch = this._needsInputMapping ? this._inViewTotal : this.logicalTotal;
    this._outPhysComplexPerBatch = this._needsOutputMapping ? this._outViewTotal : this.logicalTotal;
    this._inPhysComplex = this._inPhysComplexPerBatch * this.batch;
    this._outPhysComplex = this._outPhysComplexPerBatch * this.batch;
    this._inPhysBytesPerBatch = this._inPhysComplexPerBatch * this._bytesPerComplexIO;
    this._outPhysBytesPerBatch = this._outPhysComplexPerBatch * this._bytesPerComplexIO;
    this._inPhysBytes = this._inPhysComplex * this._bytesPerComplexIO;
    this._outPhysBytes = this._outPhysComplex * this._bytesPerComplexIO;

    if (this._needsInputMapping && !this._outOfCoreFourStepMode) {
      const inBindBytes = this._largeBatchChunkMode ? this._inPhysBytesPerBatch : this._inPhysBytes;
      ensureWithinBindingLimit(device, inBindBytes, "c2c ioView.input");
      const code =
        this.precision === "f16-storage"
          ? generateEmbedComplexF16ToF32WGSL({
              rank: this.rank,
              logicalDims: this.shape,
              viewDims: this.io.input.shape,
              offset: this.io.input.offset,
              workgroupSize: this.workgroupSize,
            })
          : generateEmbedComplexWGSL({
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
      this.ioEmbed = { bgl, pl: pipelineLayout, pipeline, params, viewShape: this.io.input.shape };
    }
    if (this._needsOutputMapping && !this._outOfCoreFourStepMode) {
      const outBindBytes = this._largeBatchChunkMode ? this._outPhysBytesPerBatch : this._outPhysBytes;
      ensureWithinBindingLimit(device, outBindBytes, "c2c ioView.output");
      const code =
        this.precision === "f16-storage"
          ? generateExtractComplexF32ToF16WGSL({
              rank: this.rank,
              logicalDims: this.shape,
              viewDims: this.io.output.shape,
              offset: this.io.output.offset,
              clearOutside: this.io.output.clearOutside,
              workgroupSize: this.workgroupSize,
            })
          : generateExtractComplexWGSL({
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
      this.ioExtract = { bgl, pl: pipelineLayout, pipeline, params, viewShape: this.io.output.shape };
    }

    if (this._largeBatchChunkMode && this.precision !== "f32") {
      throw new Error('Large-batch chunk mode currently supports precision:"f32" only');
    }
    this.scratchBytes = this._largeBatchChunkMode
      ? 16
      : Math.max(this.mainBytes, this.maxAxisWorkBytes, this._inPhysBytes, this._outPhysBytes);
    ensureWithinBindingLimit(device, this.scratchBytes, "c2c scratch");

    // f16 storage conversion
    this.f16 = null;
    if (this.precision === "f16-storage") {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const toF32 = this.cache.getComputePipeline({ code: generateF16ToF32ComplexWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
      const toF16 = this.cache.getComputePipeline({ code: generateF32ToF16ComplexWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.f16 = { bgl, pl: pipelineLayout, toF32, toF16, params };
    }

    // Optional strided gather/scatter (layout.inputStrides/outputStrides)
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

    // Optional transpose fast path for axis 1 using batched 2D tiles over [axis0, axis1].
    this.transpose = null;
    this.transposeBytes = 0;
    this.axis0OnTransposed = null;
    const transposeDispatchLimitZ = maxComputeWorkgroupsPerDimension(this.device.limits, 2);
    if (
      !this._largeBatchChunkMode &&
      !this.tuning.disableTranspose &&
      this.rank >= 2 &&
      this.axisKind[0] === "mixed" &&
      this.axisKind[1] === "mixed" &&
      this.shape[0] * this.shape[1] >= this.tuning.transposeMinElements &&
      this._axis01MatrixBatch <= transposeDispatchLimitZ
    ) {
      const [Nx, Ny] = this.shape;
      const transposedShape = [Ny, Nx, ...this.shape.slice(2)];
      const tile = 16;
      const codeXY = generateTransposeComplex2DWGSL({ Nx, Ny, tile }); // (Nx,Ny) -> (Ny,Nx)
      const codeYX = generateTransposeComplex2DWGSL({ Nx: Ny, Ny: Nx, tile }); // (Ny,Nx) -> (Nx,Ny)
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipelineXY = this.cache.getComputePipeline({ code: codeXY, layout: pipelineLayout });
      const pipelineYX = this.cache.getComputePipeline({ code: codeYX, layout: pipelineLayout });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.transpose = {
        bgl,
        pl: pipelineLayout,
        pipelineXY,
        pipelineYX,
        params,
        tile,
        Nx,
        Ny,
        matrixBatch: this._axis01MatrixBatch,
      };
      this.transposeBytes = this.mainBytes;

      // Precompile "FFT along original axis 1" as axis0 FFT on [Ny, Nx, ...tail].
      this.axis0OnTransposed = createFftPlan(this.device, {
        shape: transposedShape,
        direction: this.direction,
        normalize: "none",
        inPlace: true,
        layout: "interleaved",
        precision: "f32",
        axes: [0],
      });
    }

    // Workspace layout: [mainStage?][scratch][axisWork][transpose]
    this.needsMainStage = !!this.ioEmbed || !!this.ioExtract || this.precision === "f16-storage";
    this.mainStageBytes = this.needsMainStage && !this._largeBatchChunkMode ? this.mainBytes : 0;
    this.axisWorkBytes = this.maxAxisWorkBytes;

    const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    let off = 0;
    this.mainStageOffset = 0;
    off += this.mainStageBytes;
    off = alignBytes(off, storageAlign);
    this.scratchOffset = off;
    off += this.scratchBytes;
    off = alignBytes(off, storageAlign);
    this.axisWorkOffset = off;
    off += this.axisWorkBytes;
    off = alignBytes(off, storageAlign);
    this.transposeOffset = off;
    off += this.transposeBytes;

    this.workspaceBytes = off;
    this._splitWorkspace = null;
    const maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
    if (this.workspaceBytes <= maxBufferSize) {
      this._arena = createInternalArena(device, this.workspaceBytes);
    } else {
      const splitNeeds = [
        ["mainStage", this.mainStageBytes],
        ["scratch", this.scratchBytes],
        ["axisWork", this.axisWorkBytes],
        ["transpose", this.transposeBytes],
      ];
      for (const [name, bytes] of splitNeeds) {
        if (bytes > 0 && bytes > maxBufferSize) {
          throw new Error(
            `c2c split workspace cannot allocate ${name} buffer: ${bytes} bytes exceeds device.limits.maxBufferSize=${maxBufferSize}`
          );
        }
      }
      this._arena = null;
      this._splitWorkspace = {
        mainStage: this.mainStageBytes ? createInternalArena(device, this.mainStageBytes) : null,
        scratch: this.scratchBytes ? createInternalArena(device, this.scratchBytes) : null,
        axisWork: this.axisWorkBytes ? createInternalArena(device, this.axisWorkBytes) : null,
        transpose: this.transposeBytes ? createInternalArena(device, this.transposeBytes) : null,
      };
    }
    this._largeStageBuffer = null;
    this._largeStageBytes = 0;
    this._largeChunkBuffer = null;
    this._largeChunkBytes = 0;
    this._largeAuxBuffer = null;
    this._largeAuxBytes = 0;
    this._retiredLargeStageBuffers = [];
    this._retiredLargeChunkBuffers = [];
    this._retiredLargeAuxBuffers = [];
    this._scaleChunkParamsBuffer = null;
    this._scaleChunkParamsBytes = 0;
    this._retiredScaleChunkParamsBuffers = [];
    this._zeroComplexBuffer = null;

    const maxBufferSizeForMode = this.device.limits?.maxBufferSize ?? Infinity;
    this._outOfCoreSegmentedFullVolumeMode =
      this._outOfCoreFourStepMode &&
      this.mainBytes > maxBufferSizeForMode &&
      this.precision === "f32" &&
      this.rank === 3 &&
      this.axisKind.every((k) => k === "mixed") &&
      !this._needsInputMapping &&
      !this._needsOutputMapping &&
      !this._usesStridedInput &&
      !this._usesStridedOutput &&
      !this.zeroPad.read &&
      !this.zeroPad.write;
    this._outOfCoreSegmentedFullVolumeState = null;
    this._segmentedFullVolumeMeta = null;
  }

  getWorkspaceSizeBytes() {
    return this.workspaceBytes;
  }

  _resolveWorkspaceViews(arenaLike) {
    if (arenaLike) {
      if (getBufferByteLength(arenaLike) < this.workspaceBytes) throw new Error(`temp too small: need ${this.workspaceBytes} bytes`);
      return {
        mainStage: this.mainStageBytes ? viewFromArena(arenaLike, this.mainStageOffset, this.mainStageBytes) : null,
        scratch: viewFromArena(arenaLike, this.scratchOffset, this.scratchBytes),
        axisWork: this.axisWorkBytes ? viewFromArena(arenaLike, this.axisWorkOffset, this.axisWorkBytes) : null,
        transpose: this.transposeBytes ? viewFromArena(arenaLike, this.transposeOffset, this.transposeBytes) : null,
      };
    }
    if (this._splitWorkspace) {
      return {
        mainStage: this.mainStageBytes ? viewFromArena(this._splitWorkspace.mainStage, 0, this.mainStageBytes) : null,
        scratch: viewFromArena(this._splitWorkspace.scratch, 0, this.scratchBytes),
        axisWork: this.axisWorkBytes ? viewFromArena(this._splitWorkspace.axisWork, 0, this.axisWorkBytes) : null,
        transpose: this.transposeBytes ? viewFromArena(this._splitWorkspace.transpose, 0, this.transposeBytes) : null,
      };
    }
    throw new Error("No workspace buffer");
  }

  _ensureLargeStageBuffer(minBytes) {
    if (this._largeStageBuffer && this._largeStageBytes >= minBytes) return this._largeStageBuffer;

    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    if (minBytes > maxBufferSize) {
      throw new Error(
        `Large-batch staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`
      );
    }

    if (this._largeStageBuffer) this._retiredLargeStageBuffers.push(this._largeStageBuffer);
    this._largeStageBuffer = this.device.createBuffer({
      size: minBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._largeStageBytes = minBytes;
    return this._largeStageBuffer;
  }

  _ensureLargeChunkBuffer(minBytes) {
    if (this._largeChunkBuffer && this._largeChunkBytes >= minBytes) return this._largeChunkBuffer;

    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    if (minBytes > maxBufferSize) {
      throw new Error(
        `Large-batch chunk scratch requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`
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

  _ensureLargeAuxBuffer(minBytes) {
    if (this._largeAuxBuffer && this._largeAuxBytes >= minBytes) return this._largeAuxBuffer;

    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    if (minBytes > maxBufferSize) {
      throw new Error(
        `Large-batch auxiliary staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`
      );
    }

    if (this._largeAuxBuffer) this._retiredLargeAuxBuffers.push(this._largeAuxBuffer);
    this._largeAuxBuffer = this.device.createBuffer({
      size: minBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._largeAuxBytes = minBytes;
    return this._largeAuxBuffer;
  }

  _ensureScaleChunkParamsBuffer(minBytes) {
    if (this._scaleChunkParamsBuffer && this._scaleChunkParamsBytes >= minBytes) return this._scaleChunkParamsBuffer;

    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    if (minBytes > maxBufferSize) {
      throw new Error(
        `Large-batch scale params require ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`
      );
    }

    if (this._scaleChunkParamsBuffer) this._retiredScaleChunkParamsBuffers.push(this._scaleChunkParamsBuffer);
    this._scaleChunkParamsBuffer = this.device.createBuffer({
      size: minBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._scaleChunkParamsBytes = minBytes;
    return this._scaleChunkParamsBuffer;
  }

  _ensureZeroComplexBuffer() {
    if (this._zeroComplexBuffer) return this._zeroComplexBuffer;
    this._zeroComplexBuffer = this.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this._zeroComplexBuffer, 0, new Float32Array([0, 0]));
    return this._zeroComplexBuffer;
  }

  _rangesOverlap(a, b) {
    if (!a || !b) return false;
    if (a.buffer !== b.buffer) return false;
    const a0 = a.offsetBytes;
    const a1 = a.offsetBytes + a.sizeBytes;
    const b0 = b.offsetBytes;
    const b1 = b.offsetBytes + b.sizeBytes;
    return !(a1 <= b0 || b1 <= a0);
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

  _copyAnyToAny(commandEncoder, { src, srcOffsetBytes, dst, dstOffsetBytes, bytes }) {
    if (!Number.isInteger(bytes) || bytes < 0) {
      throw new Error(`_copyAnyToAny expects non-negative integer bytes; got ${bytes}`);
    }
    if (bytes === 0) return;
    const wholeSrcRanges = normalizeToContiguousRanges(src, srcOffsetBytes, bytes);
    const wholeDstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes, bytes);
    if (wholeSrcRanges.length === 1 && wholeDstRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(
        wholeSrcRanges[0].buffer,
        wholeSrcRanges[0].offsetBytes,
        wholeDstRanges[0].buffer,
        wholeDstRanges[0].offsetBytes,
        bytes
      );
      return;
    }
    const maxBufferSize = this.device.limits?.maxBufferSize ?? bytes;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const preferredChunk = Number.isFinite(this._maxBindBytes)
      ? Math.min(maxBufferSize, Math.max(storageAlign, this._maxBindBytes))
      : maxBufferSize;
    let chunkBytes = alignDownBytes(preferredChunk, 4);
    if (!Number.isInteger(chunkBytes) || chunkBytes <= 0) chunkBytes = Math.max(4, Math.min(bytes, 1024 * 1024));
    const chunkBuf = this._ensureLargeChunkBuffer(chunkBytes);

    for (let off = 0; off < bytes; off += chunkBytes) {
      const n = Math.min(chunkBytes, bytes - off);
      const srcRanges = normalizeToContiguousRanges(src, srcOffsetBytes + off, n);
      const dstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes + off, n);
      if (srcRanges.length === 1 && dstRanges.length === 1) {
        commandEncoder.copyBufferToBuffer(
          srcRanges[0].buffer,
          srcRanges[0].offsetBytes,
          dstRanges[0].buffer,
          dstRanges[0].offsetBytes,
          n
        );
        continue;
      }
      if (srcRanges.length === 1) {
        commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, chunkBuf, 0, n);
        this.copier.unpack(commandEncoder, chunkBuf, 0, dstRanges);
        continue;
      }
      if (dstRanges.length === 1) {
        this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
        commandEncoder.copyBufferToBuffer(chunkBuf, 0, dstRanges[0].buffer, dstRanges[0].offsetBytes, n);
        continue;
      }
      this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
      this.copier.unpack(commandEncoder, chunkBuf, 0, dstRanges);
    }
  }

  _createInternalSegmentedView(totalBytes, preferredSegmentBytes) {
    if (!Number.isInteger(totalBytes) || totalBytes <= 0) {
      throw new Error(`_createInternalSegmentedView requires positive totalBytes; got ${totalBytes}`);
    }
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    let segmentBytes = Math.min(totalBytes, preferredSegmentBytes ?? totalBytes, maxBufferSize);
    segmentBytes = alignDownBytes(segmentBytes, storageAlign);
    if (!Number.isInteger(segmentBytes) || segmentBytes <= 0) {
      segmentBytes = alignBytes(Math.min(totalBytes, maxBufferSize), storageAlign);
    }
    if (segmentBytes <= 0 || segmentBytes > maxBufferSize) {
      throw new Error(
        `Unable to allocate segmented internal view: segmentBytes=${segmentBytes}, maxBufferSize=${maxBufferSize}, totalBytes=${totalBytes}`
      );
    }
    const segments = [];
    let remaining = totalBytes;
    while (remaining > 0) {
      const takeRaw = Math.min(remaining, segmentBytes);
      const take = remaining === takeRaw ? takeRaw : alignDownBytes(takeRaw, storageAlign);
      if (!Number.isInteger(take) || take <= 0) {
        throw new Error(`Failed to split segmented view with storage alignment ${storageAlign}`);
      }
      const buffer = this.device.createBuffer({
        size: take,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      segments.push({ buffer, offsetBytes: 0, sizeBytes: take });
      remaining -= take;
    }
    return {
      segmentBytes,
      view: {
        segments,
        logicalByteOffset: 0,
        lengthBytes: totalBytes,
      },
    };
  }

  _destroySegmentedView(view) {
    const segs = view?.segments;
    if (!Array.isArray(segs)) return;
    for (const s of segs) {
      s?.buffer?.destroy?.();
    }
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
      bytesPerElement: 8,
      runtimeExtraElements: extraOffsetElements,
      batchStart,
      batchCount,
    });
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

  _copyLogicalMap(commandEncoder, {
    srcBuffer,
    srcOffsetBytes,
    dstBuffer,
    dstOffsetBytes,
    srcShape,
    dstShape,
    mapCoordFn,
    batch,
  }) {
    const logicalTotal = prod(srcShape);
    const srcStrides = this._shapeStrides(srcShape);
    const dstStrides = this._shapeStrides(dstShape);
    const srcCoords = new Array(srcShape.length).fill(0);
    const dstCoords = new Array(dstShape.length).fill(0);
    const perSrcBatchBytes = logicalTotal * 8;
    const perDstBatchBytes = prod(dstShape) * 8;
    for (let b = 0; b < batch; b++) {
      const srcBase = srcOffsetBytes + b * perSrcBatchBytes;
      const dstBase = dstOffsetBytes + b * perDstBatchBytes;
      for (let i = 0; i < logicalTotal; i++) {
        this._coordsFromLinear(i, srcShape, srcCoords);
        mapCoordFn(srcCoords, dstCoords);
        const srcIdx = this._linearFromCoords(srcCoords, srcStrides);
        const dstIdx = this._linearFromCoords(dstCoords, dstStrides);
        commandEncoder.copyBufferToBuffer(srcBuffer, srcBase + srcIdx * 8, dstBuffer, dstBase + dstIdx * 8, 8);
      }
    }
  }

  _zeroLogicalOutsideRange(commandEncoder, { dataBuffer, dataOffsetBytes, start, end }) {
    const logicalTotal = this.logicalTotal;
    const coords = new Array(this.rank).fill(0);
    const zeroBuf = this._ensureZeroComplexBuffer();
    for (let b = 0; b < this.batch; b++) {
      const base = dataOffsetBytes + b * logicalTotal * 8;
      for (let i = 0; i < logicalTotal; i++) {
        this._coordsFromLinear(i, this.shape, coords);
        let inside = true;
        for (let d = 0; d < this.rank; d++) {
          if (coords[d] < start[d] || coords[d] >= end[d]) {
            inside = false;
            break;
          }
        }
        if (!inside) {
          commandEncoder.copyBufferToBuffer(zeroBuf, 0, dataBuffer, base + i * 8, 8);
        }
      }
    }
  }

  _transposeOutOfCore2dCopies(commandEncoder, {
    srcBuffer,
    srcOffsetBytes,
    dstBuffer,
    dstOffsetBytes,
    Nx,
    Ny,
    batch,
  }) {
    const elemBytes = 8;
    const perBatchBytes = Nx * Ny * elemBytes;
    for (let b = 0; b < batch; b++) {
      const srcBase = srcOffsetBytes + b * perBatchBytes;
      const dstBase = dstOffsetBytes + b * perBatchBytes;
      for (let y = 0; y < Ny; y++) {
        for (let x = 0; x < Nx; x++) {
          const src = srcBase + (y * Nx + x) * elemBytes;
          const dst = dstBase + (x * Ny + y) * elemBytes;
          commandEncoder.copyBufferToBuffer(srcBuffer, src, dstBuffer, dst, elemBytes);
        }
      }
    }
  }

  _getOutOfCoreTransposePipeline(stripeNx, stripeNy) {
    if (!this._outOfCoreTranspose) {
      throw new Error("Internal error: out-of-core transpose state is not initialized");
    }
    const key = `${stripeNx}x${stripeNy}`;
    const existing = this._outOfCoreTransposePipelines.get(key);
    if (existing) return existing;
    const code = generateTransposeComplex2DWGSL({ Nx: stripeNx, Ny: stripeNy, tile: this._outOfCoreTranspose.tile });
    const pipeline = this.cache.getComputePipeline({ code, layout: this._outOfCoreTranspose.pl });
    this._outOfCoreTransposePipelines.set(key, pipeline);
    return pipeline;
  }

  _transposeOutOfCore2dStripes(commandEncoder, {
    srcBuffer,
    srcOffsetBytes,
    dstBuffer,
    dstOffsetBytes,
    Nx,
    Ny,
    batch,
  }) {
    if (!this._outOfCoreTranspose) {
      this._transposeOutOfCore2dCopies(commandEncoder, {
        srcBuffer,
        srcOffsetBytes,
        dstBuffer,
        dstOffsetBytes,
        Nx,
        Ny,
        batch,
      });
      return;
    }

    if (Ny * 8 > this._maxBindBytes) {
      this._transposeOutOfCore2dCopies(commandEncoder, {
        srcBuffer,
        srcOffsetBytes,
        dstBuffer,
        dstOffsetBytes,
        Nx,
        Ny,
        batch,
      });
      return;
    }

    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxStripeNx = Math.max(1, Math.floor(this._maxBindBytes / (Ny * 8)));
    const maxStripeBytes = maxStripeNx * Ny * 8;
    const stripeDstOffset = alignBytes(maxStripeBytes, storageAlign);
    const stripeScratchBytes = stripeDstOffset + maxStripeBytes;
    const stripeBuf = this._ensureLargeChunkBuffer(stripeScratchBytes);
    const stripeWidths = new Set();
    for (let x0 = 0; x0 < Nx; x0 += maxStripeNx) {
      stripeWidths.add(Math.min(maxStripeNx, Nx - x0));
    }
    try {
      for (const bx of stripeWidths) this._getOutOfCoreTransposePipeline(bx, Ny);
    } catch {
      this._transposeOutOfCore2dCopies(commandEncoder, {
        srcBuffer,
        srcOffsetBytes,
        dstBuffer,
        dstOffsetBytes,
        Nx,
        Ny,
        batch,
      });
      return;
    }

    for (let b = 0; b < batch; b++) {
      const srcBase = srcOffsetBytes + b * Nx * Ny * 8;
      const dstBase = dstOffsetBytes + b * Nx * Ny * 8;
      for (let x0 = 0; x0 < Nx; x0 += maxStripeNx) {
        const bx = Math.min(maxStripeNx, Nx - x0);
        const stripeBytes = bx * Ny * 8;
        const pipeline = this._getOutOfCoreTransposePipeline(bx, Ny);

        for (let y = 0; y < Ny; y++) {
          const srcRow = srcBase + (y * Nx + x0) * 8;
          const dstRow = y * bx * 8;
          commandEncoder.copyBufferToBuffer(srcBuffer, srcRow, stripeBuf, dstRow, bx * 8);
        }

        this.device.queue.writeBuffer(this._outOfCoreTranspose.params, 0, new Uint32Array([1, 0, 0, 0]));
        const bg = this.device.createBindGroup({
          layout: this._outOfCoreTranspose.bgl,
          entries: [
            { binding: 0, resource: { buffer: stripeBuf, offset: 0, size: stripeBytes } },
            { binding: 1, resource: { buffer: stripeBuf, offset: stripeDstOffset, size: stripeBytes } },
            { binding: 2, resource: { buffer: this._outOfCoreTranspose.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(bx / this._outOfCoreTranspose.tile), Math.ceil(Ny / this._outOfCoreTranspose.tile), 1);
        pass.end();

        const dstBlock = dstBase + x0 * Ny * 8;
        commandEncoder.copyBufferToBuffer(stripeBuf, stripeDstOffset, dstBuffer, dstBlock, stripeBytes);
      }
    }
  }

  _resolveOutOfCoreAxis1TailChunk() {
    if (this._outOfCoreAxis1TailChunk !== null) return this._outOfCoreAxis1TailChunk || null;
    if (!this._outOfCoreAxis1TailPermute || this.precision !== "f32" || this.rank < 3) {
      this._outOfCoreAxis1TailChunk = false;
      return null;
    }
    const X = this.shape[0];
    const Y = this.shape[1];
    const tail = prod(this.shape.slice(2));
    const XY = X * Y;
    if (tail < 1 || XY < 1) {
      this._outOfCoreAxis1TailChunk = false;
      return null;
    }
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxBindBytes = this._maxBindBytes;
    if (!Number.isFinite(maxBindBytes) || maxBindBytes < 8) {
      this._outOfCoreAxis1TailChunk = false;
      return null;
    }
    const effectiveBindBytes = Math.max(0, Math.floor(maxBindBytes - storageAlign));
    const maxElemsByBind = Math.floor(effectiveBindBytes / 8);
    if (maxElemsByBind < XY) {
      this._outOfCoreAxis1TailChunk = false;
      return null;
    }
    const maxWgX = maxComputeWorkgroupsPerDimension(this.device.limits, 0);
    const maxElemsByDispatch = Number.isFinite(maxWgX) ? Math.max(1, Math.floor(maxWgX * this.workgroupSize)) : Infinity;
    const maxTailByBind = Math.floor(maxElemsByBind / XY);
    const maxTailByDispatch = Math.floor(maxElemsByDispatch / XY);
    const chunkTail = Math.min(tail, maxTailByBind, maxTailByDispatch);
    if (chunkTail < 1) {
      this._outOfCoreAxis1TailChunk = false;
      return null;
    }
    this._outOfCoreAxis1TailChunk = { tailPerChunk: chunkTail, XY };
    return this._outOfCoreAxis1TailChunk;
  }

  _tryPermuteAxis1TailWithKernel(commandEncoder, { srcRange, dstRange, toFront }) {
    if (!this._outOfCoreAxis1TailPermute || this.precision !== "f32" || this.rank < 3) return false;
    const chunkCfg = this._resolveOutOfCoreAxis1TailChunk();
    if (!chunkCfg) return false;

    const tail = prod(this.shape.slice(2));
    const XY = chunkCfg.XY;
    const perBatchElems = this.logicalTotal;
    const perBatchBytes = perBatchElems * 8;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const state = this._outOfCoreAxis1TailPermute;
    const pipeline = toFront ? state.toFront : state.fromFront;
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);

    for (let b = 0; b < this.batch; b++) {
      const srcBatchBase = srcRange.offsetBytes + b * perBatchBytes;
      const dstBatchBase = dstRange.offsetBytes + b * perBatchBytes;
      for (let t0 = 0; t0 < tail; t0 += chunkCfg.tailPerChunk) {
        const tailCount = Math.min(chunkCfg.tailPerChunk, tail - t0);
        const spanElems = tailCount * XY;
        const count = spanElems;
        const windowStartElems = t0 * XY;
        const srcWindowStartBytes = srcBatchBase + windowStartElems * 8;
        const dstWindowStartBytes = dstBatchBase + windowStartElems * 8;
        const srcBindOffset = alignDownBytes(srcWindowStartBytes, storageAlign);
        const dstBindOffset = alignDownBytes(dstWindowStartBytes, storageAlign);
        const srcStartElems = Math.floor((srcWindowStartBytes - srcBindOffset) / 8);
        const dstStartElems = Math.floor((dstWindowStartBytes - dstBindOffset) / 8);
        const srcBindBytes = srcStartElems * 8 + spanElems * 8;
        const dstBindBytes = dstStartElems * 8 + spanElems * 8;
        this.device.queue.writeBuffer(state.params, 0, new Uint32Array([count, srcStartElems, dstStartElems, 0]));
        const bg = this.device.createBindGroup({
          layout: state.bgl,
          entries: [
            { binding: 0, resource: { buffer: srcRange.buffer, offset: srcBindOffset, size: srcBindBytes } },
            { binding: 1, resource: { buffer: dstRange.buffer, offset: dstBindOffset, size: dstBindBytes } },
            { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
          ],
        });
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
      }
    }
    pass.end();
    return true;
  }

  _getOutOfCoreAdjacentSwapPipeline(X, Y) {
    if (!this._outOfCoreGenericPermute) {
      throw new Error("Internal error: adjacent swap pipeline requires generic out-of-core permute state");
    }
    if (!this._outOfCoreAdjacentSwapPipelines) this._outOfCoreAdjacentSwapPipelines = new Map();
    const key = `${X}x${Y}`;
    const existing = this._outOfCoreAdjacentSwapPipelines.get(key);
    if (existing) return existing;
    const code = generatePermuteAxis1TailToFrontWGSL({
      shape: [X, Y],
      workgroupSize: this.workgroupSize,
    });
    const pipeline = this.cache.getComputePipeline({ code, layout: this._outOfCoreGenericPermute.pl });
    this._outOfCoreAdjacentSwapPipelines.set(key, pipeline);
    return pipeline;
  }

  _getOutOfCoreAdjacentSwapTiledPipeline(X, Y) {
    if (!this._outOfCoreAdjacentSwapTiled) {
      throw new Error("Internal error: adjacent tiled swap pipeline requires out-of-core tiled state");
    }
    if (!this._outOfCoreAdjacentSwapTiledPipelines) this._outOfCoreAdjacentSwapTiledPipelines = new Map();
    const key = `${X}x${Y}`;
    const existing = this._outOfCoreAdjacentSwapTiledPipelines.get(key);
    if (existing) return existing;
    const code = generatePermuteAxis1TailTiledToFrontWGSL({
      shape: [X, Y],
      workgroupSize: this.workgroupSize,
    });
    const pipeline = this.cache.getComputePipeline({ code, layout: this._outOfCoreAdjacentSwapTiled.pl });
    this._outOfCoreAdjacentSwapTiledPipelines.set(key, pipeline);
    return pipeline;
  }

  _resolveOutOfCoreAdjacentSwapChunk({ X, Y, tail }) {
    if (!(X > 0 && Y > 0 && tail > 0)) return null;
    const XY = X * Y;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxBindBytes = this._maxBindBytes;
    if (!Number.isFinite(maxBindBytes) || maxBindBytes < 8) return null;
    const effectiveBindBytes = Math.max(0, Math.floor(maxBindBytes - storageAlign));
    const maxElemsByBind = Math.floor(effectiveBindBytes / 8);
    if (maxElemsByBind < XY) return null;
    const maxWgX = maxComputeWorkgroupsPerDimension(this.device.limits, 0);
    const maxElemsByDispatch = Number.isFinite(maxWgX) ? Math.max(1, Math.floor(maxWgX * this.workgroupSize)) : Infinity;
    const maxTailByBind = Math.floor(maxElemsByBind / XY);
    const maxTailByDispatch = Math.floor(maxElemsByDispatch / XY);
    const tailPerChunk = Math.min(tail, maxTailByBind, maxTailByDispatch);
    if (tailPerChunk < 1) return null;
    return { XY, tailPerChunk };
  }

  _dispatchOutOfCoreAdjacentSwap(commandEncoder, { srcRange, dstRange, outerGroups, X, Y, tail }) {
    const chunkCfg = this._resolveOutOfCoreAdjacentSwapChunk({ X, Y, tail });
    if (!chunkCfg) {
      return this._dispatchOutOfCoreAdjacentSwapTiled(commandEncoder, { srcRange, dstRange, outerGroups, X, Y, tail });
    }
    const state = this._outOfCoreGenericPermute;
    const pipeline = this._getOutOfCoreAdjacentSwapPipeline(X, Y);
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const XY = chunkCfg.XY;
    const elemsPerOuter = XY * tail;
    const bytesPerOuter = elemsPerOuter * 8;
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);

    for (let g = 0; g < outerGroups; g++) {
      const srcOuterBase = srcRange.offsetBytes + g * bytesPerOuter;
      const dstOuterBase = dstRange.offsetBytes + g * bytesPerOuter;
      for (let t0 = 0; t0 < tail; t0 += chunkCfg.tailPerChunk) {
        const tailCount = Math.min(chunkCfg.tailPerChunk, tail - t0);
        const spanElems = tailCount * XY;
        const count = spanElems;
        const windowStartElems = t0 * XY;
        const srcWindowStartBytes = srcOuterBase + windowStartElems * 8;
        const dstWindowStartBytes = dstOuterBase + windowStartElems * 8;
        const srcBindOffset = alignDownBytes(srcWindowStartBytes, storageAlign);
        const dstBindOffset = alignDownBytes(dstWindowStartBytes, storageAlign);
        const srcStartElems = Math.floor((srcWindowStartBytes - srcBindOffset) / 8);
        const dstStartElems = Math.floor((dstWindowStartBytes - dstBindOffset) / 8);
        const srcBindBytes = srcStartElems * 8 + spanElems * 8;
        const dstBindBytes = dstStartElems * 8 + spanElems * 8;
        this.device.queue.writeBuffer(state.params, 0, new Uint32Array([count, srcStartElems, dstStartElems, 0]));
        const bg = this.device.createBindGroup({
          layout: state.bgl,
          entries: [
            { binding: 0, resource: { buffer: srcRange.buffer, offset: srcBindOffset, size: srcBindBytes } },
            { binding: 1, resource: { buffer: dstRange.buffer, offset: dstBindOffset, size: dstBindBytes } },
            { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
          ],
        });
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
      }
    }

    pass.end();
    return true;
  }

  _dispatchOutOfCoreAdjacentSwapTiled(commandEncoder, { srcRange, dstRange, outerGroups, X, Y, tail }) {
    if (!this._outOfCoreAdjacentSwapTiled) return false;
    const maxBindBytes = this._maxBindBytes;
    if (!Number.isFinite(maxBindBytes) || maxBindBytes < 8) return false;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const effectiveBindBytes = Math.max(0, Math.floor(maxBindBytes - storageAlign));
    const maxElemsByBind = Math.floor(effectiveBindBytes / 8);
    if (maxElemsByBind < 1) return false;
    const maxWgX = maxComputeWorkgroupsPerDimension(this.device.limits, 0);
    const maxElemsByDispatch = Number.isFinite(maxWgX) ? Math.max(1, Math.floor(maxWgX * this.workgroupSize)) : Infinity;
    const XY = X * Y;
    const elemsPerOuter = XY * tail;
    const bytesPerOuter = elemsPerOuter * 8;
    const pipeline = this._getOutOfCoreAdjacentSwapTiledPipeline(X, Y);
    const state = this._outOfCoreAdjacentSwapTiled;
    const maxTailByBind = Math.max(1, Math.floor((maxElemsByBind - 1) / XY) + 1);
    const maxTailByDispatch = Math.max(1, Math.floor(maxElemsByDispatch));
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);

    for (let g = 0; g < outerGroups; g++) {
      const srcOuterBase = srcRange.offsetBytes + g * bytesPerOuter;
      const dstOuterBase = dstRange.offsetBytes + g * bytesPerOuter;
      for (let y0 = 0; y0 < Y; y0++) {
        for (let t0 = 0; t0 < tail; ) {
          const htail = Math.max(1, Math.min(tail - t0, maxTailByBind, maxTailByDispatch));
          const srcHxCap = maxElemsByBind - (htail - 1) * XY;
          const dstHxCap = Math.floor((maxElemsByBind - (htail - 1) * XY - 1) / Y) + 1;
          const dispatchHxCap = Math.floor(maxElemsByDispatch / htail);
          const maxHx = Math.min(X, srcHxCap, dstHxCap, dispatchHxCap);
          if (!Number.isFinite(maxHx) || maxHx < 1) return false;
          for (let x0 = 0; x0 < X; x0 += maxHx) {
            const hx = Math.min(maxHx, X - x0);
            const srcLocalStartElems = x0 + y0 * X + t0 * XY;
            const dstLocalStartElems = y0 + x0 * Y + t0 * XY;
            const srcWindowStartBytes = srcOuterBase + srcLocalStartElems * 8;
            const dstWindowStartBytes = dstOuterBase + dstLocalStartElems * 8;
            const srcBindOffset = alignDownBytes(srcWindowStartBytes, storageAlign);
            const dstBindOffset = alignDownBytes(dstWindowStartBytes, storageAlign);
            const srcStartElems = Math.floor((srcWindowStartBytes - srcBindOffset) / 8);
            const dstStartElems = Math.floor((dstWindowStartBytes - dstBindOffset) / 8);
            const srcSpanElems = (htail - 1) * XY + hx;
            const dstSpanElems = (htail - 1) * XY + (hx - 1) * Y + 1;
            const srcBindBytes = srcStartElems * 8 + srcSpanElems * 8;
            const dstBindBytes = dstStartElems * 8 + dstSpanElems * 8;
            const count = hx * htail;
            this.device.queue.writeBuffer(
              state.params,
              0,
              new Uint32Array([count, hx, htail, srcStartElems, dstStartElems, 0, 0, 0])
            );
            const bg = this.device.createBindGroup({
              layout: state.bgl,
              entries: [
                { binding: 0, resource: { buffer: srcRange.buffer, offset: srcBindOffset, size: srcBindBytes } },
                { binding: 1, resource: { buffer: dstRange.buffer, offset: dstBindOffset, size: dstBindBytes } },
                { binding: 2, resource: { buffer: state.params, offset: 0, size: 32 } },
              ],
            });
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
          }
          t0 += htail;
        }
      }
    }

    pass.end();
    return true;
  }

  _tryPermuteViaAdjacentSwaps(commandEncoder, { srcRange, dstRange, axis, toFront }) {
    if (!this._outOfCoreGenericPermute || this.precision !== "f32" || this.rank < 2) return false;
    if (!Number.isInteger(axis) || axis < 1 || axis >= this.rank) return false;
    const totalBytes = this.batch * this.logicalTotal * 8;
    if (srcRange.sizeBytes < totalBytes || dstRange.sizeBytes < totalBytes) return false;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    if (srcRange.offsetBytes % storageAlign !== 0 || dstRange.offsetBytes % storageAlign !== 0) return false;

    const steps = [];
    if (toFront) {
      for (let k = axis; k >= 1; k--) steps.push(k - 1);
    } else {
      for (let k = 0; k < axis; k++) steps.push(k);
    }
    if (steps.length === 0) return true;

    let currentShape = toFront ? this.shape.slice() : permutedShapeAxisFront(this.shape, axis);
    let readRange = srcRange;
    let writeRange = dstRange;

    for (const left of steps) {
      const right = left + 1;
      const X = currentShape[left];
      const Y = currentShape[right];
      const tail = prod(currentShape.slice(right + 1));
      const outerGroups = this.batch * prod(currentShape.slice(0, left));
      const ok = this._dispatchOutOfCoreAdjacentSwap(commandEncoder, {
        srcRange: readRange,
        dstRange: writeRange,
        outerGroups,
        X,
        Y,
        tail,
      });
      if (!ok) return false;
      const tmp = currentShape[left];
      currentShape[left] = currentShape[right];
      currentShape[right] = tmp;
      const r = readRange;
      readRange = writeRange;
      writeRange = r;
    }

    if (readRange.buffer !== dstRange.buffer || readRange.offsetBytes !== dstRange.offsetBytes) {
      commandEncoder.copyBufferToBuffer(readRange.buffer, readRange.offsetBytes, dstRange.buffer, dstRange.offsetBytes, totalBytes);
    }
    return true;
  }

  _rank3Axis2WindowSpansElems(hy, hz) {
    const X = this.shape[0];
    const Y = this.shape[1];
    const Z = this.shape[2];
    return {
      srcSpanElems: (hz - 1) * X * Y + hy * X,
      dstSpanElems: hy * Z * X - Z + hz,
    };
  }

  _resolveOutOfCoreRank3Axis2Tile() {
    if (this._outOfCoreRank3Axis2Tile !== null) {
      return this._outOfCoreRank3Axis2Tile || null;
    }
    if (!this._outOfCoreRank3Axis2Permute || this.precision !== "f32" || this.rank !== 3) {
      this._outOfCoreRank3Axis2Tile = false;
      return null;
    }

    const [X, Y, Z] = this.shape;
    if (X <= 0 || Y <= 0 || Z <= 0) {
      this._outOfCoreRank3Axis2Tile = false;
      return null;
    }
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxBindBytes = this._maxBindBytes;
    if (!Number.isFinite(maxBindBytes) || maxBindBytes < 8) {
      this._outOfCoreRank3Axis2Tile = false;
      return null;
    }
    const effectiveBindBytes = Math.max(0, Math.floor(maxBindBytes - storageAlign));
    const maxElemsByBind = Math.floor(effectiveBindBytes / 8);
    if (maxElemsByBind < 1) {
      this._outOfCoreRank3Axis2Tile = false;
      return null;
    }
    const maxWgX = maxComputeWorkgroupsPerDimension(this.device.limits, 0);
    const maxElemsByDispatch = Number.isFinite(maxWgX) ? Math.max(1, Math.floor(maxWgX * this.workgroupSize)) : Infinity;

    let bestHy = 0;
    let bestHz = 0;
    let bestArea = 0;
    for (let hy = 1; hy <= Y; hy++) {
      const hzByDstSpan = maxElemsByBind - (hy * Z * X - Z);
      if (hzByDstSpan < 1) break;
      const hzBySrcSpan = Math.floor((maxElemsByBind - hy * X) / (X * Y)) + 1;
      const hzByDispatch = Math.floor(maxElemsByDispatch / (X * hy));
      const hz = Math.min(Z, hzByDstSpan, hzBySrcSpan, hzByDispatch);
      if (hz < 1) continue;
      const area = hy * hz;
      if (area > bestArea) {
        bestArea = area;
        bestHy = hy;
        bestHz = hz;
      }
    }

    if (bestArea < 1) {
      this._outOfCoreRank3Axis2Tile = false;
      return null;
    }
    this._outOfCoreRank3Axis2Tile = { hy: bestHy, hz: bestHz };
    return this._outOfCoreRank3Axis2Tile;
  }

  _tryPermuteRank3Axis2WithKernel(commandEncoder, { srcRange, dstRange, toFront }) {
    if (!this._outOfCoreRank3Axis2Permute || this.precision !== "f32" || this.rank !== 3) return false;
    const tile = this._resolveOutOfCoreRank3Axis2Tile();
    if (!tile) return false;

    const [X, Y, Z] = this.shape;
    const perBatchElems = this.logicalTotal;
    const perBatchBytes = perBatchElems * 8;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const state = this._outOfCoreRank3Axis2Permute;
    const pipeline = toFront ? state.toFront : state.fromFront;
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);

    for (let b = 0; b < this.batch; b++) {
      const srcBatchBase = srcRange.offsetBytes + b * perBatchBytes;
      const dstBatchBase = dstRange.offsetBytes + b * perBatchBytes;
      for (let y0 = 0; y0 < Y; y0 += tile.hy) {
        const hy = Math.min(tile.hy, Y - y0);
        for (let z0 = 0; z0 < Z; z0 += tile.hz) {
          const hz = Math.min(tile.hz, Z - z0);
          const count = X * hy * hz;
          const { srcSpanElems, dstSpanElems } = this._rank3Axis2WindowSpansElems(hy, hz);
          const srcMinElems = toFront ? y0 * X + z0 * X * Y : z0 + y0 * Z * X;
          const dstMinElems = toFront ? z0 + y0 * Z * X : y0 * X + z0 * X * Y;
          const srcWindowStartBytes = srcBatchBase + srcMinElems * 8;
          const dstWindowStartBytes = dstBatchBase + dstMinElems * 8;
          const srcBindOffset = alignDownBytes(srcWindowStartBytes, storageAlign);
          const dstBindOffset = alignDownBytes(dstWindowStartBytes, storageAlign);
          const srcStartElems = Math.floor((srcWindowStartBytes - srcBindOffset) / 8);
          const dstStartElems = Math.floor((dstWindowStartBytes - dstBindOffset) / 8);
          const srcBindBytes = srcStartElems * 8 + srcSpanElems * 8;
          const dstBindBytes = dstStartElems * 8 + dstSpanElems * 8;
          this.device.queue.writeBuffer(state.params, 0, new Uint32Array([count, hz, srcStartElems, dstStartElems]));
          const bg = this.device.createBindGroup({
            layout: state.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcRange.buffer, offset: srcBindOffset, size: srcBindBytes } },
              { binding: 1, resource: { buffer: dstRange.buffer, offset: dstBindOffset, size: dstBindBytes } },
              { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
            ],
          });
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
        }
      }
    }

    pass.end();
    return true;
  }

  _getOutOfCoreGenericPermutePipeline(axis, toFront) {
    if (!this._outOfCoreGenericPermute) {
      throw new Error("Internal error: generic out-of-core permute state is not initialized");
    }
    const key = `${toFront ? "to" : "from"}:${axis}`;
    const existing = this._outOfCoreGenericPermutePipelines?.get(key);
    if (existing) return existing;
    const code = generatePermuteAxisGenericWGSL({
      shape: this.shape,
      axis,
      toFront,
      workgroupSize: this.workgroupSize,
    });
    const pipeline = this.cache.getComputePipeline({ code, layout: this._outOfCoreGenericPermute.pl });
    this._outOfCoreGenericPermutePipelines?.set(key, pipeline);
    return pipeline;
  }

  _tryPermuteGenericWithKernel(commandEncoder, { srcRange, dstRange, axis, toFront }) {
    if (!this._outOfCoreGenericPermute || this.precision !== "f32") return false;
    const totalElems = this.batch * this.logicalTotal;
    const totalBytes = totalElems * 8;
    if (srcRange.sizeBytes < totalBytes || dstRange.sizeBytes < totalBytes) return false;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    if (srcRange.offsetBytes % storageAlign !== 0 || dstRange.offsetBytes % storageAlign !== 0) return false;
    const maxBindBytes = this.device.limits?.maxStorageBufferBindingSize ?? Infinity;
    if (totalBytes > maxBindBytes) {
      const perBatchBytes = this.logicalTotal * 8;
      if (perBatchBytes > maxBindBytes) return false;
      const pipeline = this._getOutOfCoreGenericPermutePipeline(axis, toFront);
      const state = this._outOfCoreGenericPermute;
      this.device.queue.writeBuffer(state.params, 0, new Uint32Array([this.logicalTotal, 1, 0, 0]));
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(pipeline);
      for (let b = 0; b < this.batch; b++) {
        const srcOff = srcRange.offsetBytes + b * perBatchBytes;
        const dstOff = dstRange.offsetBytes + b * perBatchBytes;
        const bg = this.device.createBindGroup({
          layout: state.bgl,
          entries: [
            { binding: 0, resource: { buffer: srcRange.buffer, offset: srcOff, size: perBatchBytes } },
            { binding: 1, resource: { buffer: dstRange.buffer, offset: dstOff, size: perBatchBytes } },
            { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
          ],
        });
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
      }
      pass.end();
      return true;
    }

    const pipeline = this._getOutOfCoreGenericPermutePipeline(axis, toFront);
    const state = this._outOfCoreGenericPermute;
    this.device.queue.writeBuffer(state.params, 0, new Uint32Array([totalElems, this.batch, 0, 0]));
    const bg = this.device.createBindGroup({
      layout: state.bgl,
      entries: [
        { binding: 0, resource: { buffer: srcRange.buffer, offset: srcRange.offsetBytes, size: totalBytes } },
        { binding: 1, resource: { buffer: dstRange.buffer, offset: dstRange.offsetBytes, size: totalBytes } },
        { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
      ],
    });
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(totalElems / this.workgroupSize), 1, 1);
    pass.end();
    return true;
  }

  _permuteAxisToFront(commandEncoder, { srcRange, dstRange, axis }) {
    if (axis === 1 && this._tryPermuteAxis1TailWithKernel(commandEncoder, { srcRange, dstRange, toFront: true })) {
      return;
    }
    if (axis === 2 && this._tryPermuteRank3Axis2WithKernel(commandEncoder, { srcRange, dstRange, toFront: true })) {
      return;
    }
    if (this._tryPermuteGenericWithKernel(commandEncoder, { srcRange, dstRange, axis, toFront: true })) {
      return;
    }
    if (this._tryPermuteViaAdjacentSwaps(commandEncoder, { srcRange, dstRange, axis, toFront: true })) {
      return;
    }
    const rank = this.rank;
    const srcShape = this.shape;
    const dstShape = permutedShapeAxisFront(srcShape, axis);
    const srcCoords = new Array(rank).fill(0);
    const dstCoords = new Array(rank).fill(0);
    const srcStrides = this._shapeStrides(srcShape);
    const dstStrides = this._shapeStrides(dstShape);
    const logicalTotal = this.logicalTotal;
    const perBytes = logicalTotal * 8;
    for (let b = 0; b < this.batch; b++) {
      const srcBase = srcRange.offsetBytes + b * perBytes;
      const dstBase = dstRange.offsetBytes + b * perBytes;
      for (let i = 0; i < logicalTotal; i++) {
        this._coordsFromLinear(i, srcShape, srcCoords);
        dstCoords[0] = srcCoords[axis];
        let p = 1;
        for (let d = 0; d < rank; d++) {
          if (d === axis) continue;
          dstCoords[p++] = srcCoords[d];
        }
        const srcIdx = this._linearFromCoords(srcCoords, srcStrides);
        const dstIdx = this._linearFromCoords(dstCoords, dstStrides);
        commandEncoder.copyBufferToBuffer(srcRange.buffer, srcBase + srcIdx * 8, dstRange.buffer, dstBase + dstIdx * 8, 8);
      }
    }
  }

  _permuteAxisFromFront(commandEncoder, { srcRange, dstRange, axis }) {
    if (axis === 1 && this._tryPermuteAxis1TailWithKernel(commandEncoder, { srcRange, dstRange, toFront: false })) {
      return;
    }
    if (axis === 2 && this._tryPermuteRank3Axis2WithKernel(commandEncoder, { srcRange, dstRange, toFront: false })) {
      return;
    }
    if (this._tryPermuteGenericWithKernel(commandEncoder, { srcRange, dstRange, axis, toFront: false })) {
      return;
    }
    if (this._tryPermuteViaAdjacentSwaps(commandEncoder, { srcRange, dstRange, axis, toFront: false })) {
      return;
    }
    const rank = this.rank;
    const srcShape = permutedShapeAxisFront(this.shape, axis);
    const dstShape = this.shape;
    const srcCoords = new Array(rank).fill(0);
    const dstCoords = new Array(rank).fill(0);
    const srcStrides = this._shapeStrides(srcShape);
    const dstStrides = this._shapeStrides(dstShape);
    const logicalTotal = this.logicalTotal;
    const perBytes = logicalTotal * 8;
    for (let b = 0; b < this.batch; b++) {
      const srcBase = srcRange.offsetBytes + b * perBytes;
      const dstBase = dstRange.offsetBytes + b * perBytes;
      for (let i = 0; i < logicalTotal; i++) {
        this._coordsFromLinear(i, srcShape, srcCoords);
        dstCoords[axis] = srcCoords[0];
        let p = 1;
        for (let d = 0; d < rank; d++) {
          if (d === axis) continue;
          dstCoords[d] = srcCoords[p++];
        }
        const srcIdx = this._linearFromCoords(srcCoords, srcStrides);
        const dstIdx = this._linearFromCoords(dstCoords, dstStrides);
        commandEncoder.copyBufferToBuffer(srcRange.buffer, srcBase + srcIdx * 8, dstRange.buffer, dstBase + dstIdx * 8, 8);
      }
    }
  }

  _embedInputOutOfCore(commandEncoder, { inputRanges, dataRange }) {
    if (!this._needsInputMapping) {
      if (inputRanges.length === 1) {
        if (inputRanges[0].buffer === dataRange.buffer && inputRanges[0].offsetBytes === dataRange.offsetBytes) {
          return;
        }
        commandEncoder.copyBufferToBuffer(inputRanges[0].buffer, inputRanges[0].offsetBytes, dataRange.buffer, dataRange.offsetBytes, this.mainBytes);
      } else {
        this._copyRangesToContiguous(commandEncoder, inputRanges, dataRange.buffer, dataRange.offsetBytes);
      }
      return;
    }

    const zeroBuf = this._ensureZeroComplexBuffer();
    const dataUsesAux = this._largeAuxBuffer && dataRange.buffer === this._largeAuxBuffer;
    let inContigBuf = null;
    if (dataUsesAux && this._largeAuxBytes < this._inPhysBytes) {
      inContigBuf = this._ensureLargeStageBuffer(this._inPhysBytes);
    } else {
      inContigBuf = this._ensureLargeAuxBuffer(this._inPhysBytes);
    }
    let inContigOff = 0;
    const inStageRange = { buffer: inContigBuf, offsetBytes: inContigOff, sizeBytes: this._inPhysBytes };
    if (this._rangesOverlap(inStageRange, dataRange)) {
      inContigBuf = this._ensureLargeStageBuffer(this._inPhysBytes);
      inContigOff = 0;
    }
    this._copyRangesToContiguous(commandEncoder, inputRanges, inContigBuf, inContigOff);
    const viewShape = this.io.input.shape;
    const viewOffset = this.io.input.offset;
    const viewTotal = this._inViewTotal;
    const logicalTotal = this.logicalTotal;
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    const viewStrides = this._shapeStrides(viewShape);
    for (let b = 0; b < this.batch; b++) {
      const srcBase = inContigOff + b * viewTotal * 8;
      const dstBase = dataRange.offsetBytes + b * logicalTotal * 8;
      for (let li = 0; li < logicalTotal; li++) {
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
          commandEncoder.copyBufferToBuffer(zeroBuf, 0, dataRange.buffer, dstBase + li * 8, 8);
          continue;
        }
        const vi = this._linearFromCoords(viewCoords, viewStrides);
        commandEncoder.copyBufferToBuffer(inContigBuf, srcBase + vi * 8, dataRange.buffer, dstBase + li * 8, 8);
      }
    }
  }

  _embedStridedInputOutOfCore(commandEncoder, { input, inputOffsetBytes, dataRange }) {
    if (inputOffsetBytes % 8 !== 0) {
      throw new Error(`inputOffsetBytes must be a multiple of 8 for complex-strided input; got ${inputOffsetBytes}`);
    }
    const extraOffsetElements = (inputOffsetBytes / 8) | 0;
    const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
    const inputBytes = getBufferByteLength(input);
    if (inputBytes < neededBytes) {
      throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
    }

    const zeroBuf = this._ensureZeroComplexBuffer();
    if (!this._needsInputMapping) {
      const coords = new Array(this.rank).fill(0);
      for (let b = 0; b < this.batch; b++) {
        const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
        const dstBase = dataRange.offsetBytes + b * this.logicalTotal * 8;
        for (let li = 0; li < this.logicalTotal; li++) {
          this._coordsFromLinear(li, this.shape, coords);
          let srcElem = srcBatchBase;
          for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
          this._copyComplexFromAny(commandEncoder, input, srcElem * 8, dataRange.buffer, dstBase + li * 8);
        }
      }
      return;
    }

    const viewShape = this.io.input.shape;
    const viewOffset = this.io.input.offset;
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    for (let b = 0; b < this.batch; b++) {
      const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
      const dstBase = dataRange.offsetBytes + b * this.logicalTotal * 8;
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
          commandEncoder.copyBufferToBuffer(zeroBuf, 0, dataRange.buffer, dstBase + li * 8, 8);
          continue;
        }
        let srcElem = srcBatchBase;
        for (let d = 0; d < this.rank; d++) srcElem += viewCoords[d] * this._inputStrides[d];
        this._copyComplexFromAny(commandEncoder, input, srcElem * 8, dataRange.buffer, dstBase + li * 8);
      }
    }
  }

  _extractOutputOutOfCore(commandEncoder, { dataRange, outputRanges }) {
    if (!this._needsOutputMapping) {
      if (outputRanges.length === 1) {
        if (outputRanges[0].buffer === dataRange.buffer && outputRanges[0].offsetBytes === dataRange.offsetBytes) {
          return;
        }
        commandEncoder.copyBufferToBuffer(dataRange.buffer, dataRange.offsetBytes, outputRanges[0].buffer, outputRanges[0].offsetBytes, this.mainBytes);
      } else {
        this._copyContiguousToRanges(commandEncoder, dataRange.buffer, dataRange.offsetBytes, outputRanges);
      }
      return;
    }

    const viewShape = this.io.output.shape;
    const viewOffset = this.io.output.offset;
    const viewTotal = this._outViewTotal;
    const logicalTotal = this.logicalTotal;
    const outBytes = viewTotal * this.batch * 8;
    const dataUsesAux = this._largeAuxBuffer && dataRange.buffer === this._largeAuxBuffer;
    let outContigBuf = null;
    if (dataUsesAux && this._largeAuxBytes < outBytes) {
      outContigBuf = this._ensureLargeStageBuffer(outBytes);
    } else {
      outContigBuf = this._ensureLargeAuxBuffer(outBytes);
    }
    let outContigOff = 0;
    const outContigRange = { buffer: outContigBuf, offsetBytes: outContigOff, sizeBytes: outBytes };
    if (this._rangesOverlap(outContigRange, dataRange)) {
      outContigBuf = this._ensureLargeStageBuffer(outBytes);
      outContigOff = 0;
    }
    if (!this.io.output.clearOutside) {
      this._copyRangesToContiguous(commandEncoder, outputRanges, outContigBuf, outContigOff);
    } else {
      const zeroBuf = this._ensureZeroComplexBuffer();
      for (let i = 0; i < viewTotal * this.batch; i++) {
        commandEncoder.copyBufferToBuffer(zeroBuf, 0, outContigBuf, outContigOff + i * 8, 8);
      }
    }
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    const viewStrides = this._shapeStrides(viewShape);
    for (let b = 0; b < this.batch; b++) {
      const srcBase = dataRange.offsetBytes + b * logicalTotal * 8;
      const dstBase = b * viewTotal * 8;
      for (let li = 0; li < logicalTotal; li++) {
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
        commandEncoder.copyBufferToBuffer(dataRange.buffer, srcBase + li * 8, outContigBuf, outContigOff + dstBase + vi * 8, 8);
      }
    }
    if (outputRanges.length === 1 && outputRanges[0].buffer === outContigBuf && outputRanges[0].offsetBytes === outContigOff) {
      return;
    }
    this._copyContiguousToRanges(commandEncoder, outContigBuf, outContigOff, outputRanges);
  }

  _extractStridedOutputOutOfCore(commandEncoder, { dataRange, output, outputOffsetBytes }) {
    if (outputOffsetBytes % 8 !== 0) {
      throw new Error(`outputOffsetBytes must be a multiple of 8 for complex-strided output; got ${outputOffsetBytes}`);
    }
    const extraOffsetElements = (outputOffsetBytes / 8) | 0;
    const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
    const outputBytes = getBufferByteLength(output);
    if (outputBytes < neededBytes) {
      throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
    }

    if (!this._needsOutputMapping) {
      const coords = new Array(this.rank).fill(0);
      for (let b = 0; b < this.batch; b++) {
        const srcBase = dataRange.offsetBytes + b * this.logicalTotal * 8;
        const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
        for (let li = 0; li < this.logicalTotal; li++) {
          this._coordsFromLinear(li, this.shape, coords);
          let dstElem = dstBatchBase;
          for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
          this._copyComplexToAny(commandEncoder, dataRange.buffer, srcBase + li * 8, output, dstElem * 8);
        }
      }
      return;
    }

    const viewShape = this.io.output.shape;
    const viewOffset = this.io.output.offset;
    const zeroBuf = this._ensureZeroComplexBuffer();
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);

    if (this.io.output.clearOutside) {
      const viewTotal = this._outViewTotal;
      for (let b = 0; b < this.batch; b++) {
        const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
        for (let vi = 0; vi < viewTotal; vi++) {
          this._coordsFromLinear(vi, viewShape, viewCoords);
          let dstElem = dstBatchBase;
          for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
          this._copyComplexToAny(commandEncoder, zeroBuf, 0, output, dstElem * 8);
        }
      }
    }

    for (let b = 0; b < this.batch; b++) {
      const srcBase = dataRange.offsetBytes + b * this.logicalTotal * 8;
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
        this._copyComplexToAny(commandEncoder, dataRange.buffer, srcBase + li * 8, output, dstElem * 8);
      }
    }
  }

  _embedInputChunkLarge(commandEncoder, { input, inputOffsetBytes, batchStart, batchCount, dstBuffer, dstOffsetBytes }) {
    const chunkBytes = batchCount * this._bytesPerBatch;
    if (!this._needsInputMapping) {
      const srcOff = inputOffsetBytes + batchStart * this._inPhysBytesPerBatch;
      const srcRanges = normalizeToContiguousRanges(input, srcOff, chunkBytes);
      if (srcRanges.length === 1) {
        commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, dstBuffer, dstOffsetBytes, chunkBytes);
      } else {
        this.copier.pack(commandEncoder, srcRanges, dstBuffer, dstOffsetBytes);
      }
      return;
    }

    const viewShape = this.io.input.shape;
    const viewOffset = this.io.input.offset;
    const viewTotal = this._inViewTotal;
    const logicalTotal = this.logicalTotal;
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    const viewStrides = this._shapeStrides(viewShape);
    const zeroBuf = this._ensureZeroComplexBuffer();
    for (let lb = 0; lb < batchCount; lb++) {
      const gb = batchStart + lb;
      const srcBase = inputOffsetBytes + gb * viewTotal * 8;
      const dstBase = dstOffsetBytes + lb * logicalTotal * 8;
      for (let li = 0; li < logicalTotal; li++) {
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
          commandEncoder.copyBufferToBuffer(zeroBuf, 0, dstBuffer, dstBase + li * 8, 8);
          continue;
        }
        const vi = this._linearFromCoords(viewCoords, viewStrides);
        this._copyComplexFromAny(commandEncoder, input, srcBase + vi * 8, dstBuffer, dstBase + li * 8);
      }
    }
  }

  _embedStridedInputChunkLarge(commandEncoder, { input, inputOffsetBytes, batchStart, batchCount, dstBuffer, dstOffsetBytes }) {
    if (inputOffsetBytes % 8 !== 0) {
      throw new Error(`inputOffsetBytes must be a multiple of 8 for complex-strided input; got ${inputOffsetBytes}`);
    }
    const extraOffsetElements = (inputOffsetBytes / 8) | 0;
    const neededBytes = this._requiredStridedInputBytes(extraOffsetElements, batchStart, batchCount);
    const inputBytes = getBufferByteLength(input);
    if (inputBytes < neededBytes) {
      throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
    }
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    const zeroBuf = this._ensureZeroComplexBuffer();
    const viewShape = this.io.input?.shape ?? null;
    const viewOffset = this.io.input?.offset ?? null;
    for (let lb = 0; lb < batchCount; lb++) {
      const gb = batchStart + lb;
      const srcBatchBase = this._inputOffsetElements + extraOffsetElements + gb * this._inputBatchStrideElements;
      const dstBase = dstOffsetBytes + lb * this.logicalTotal * 8;
      for (let li = 0; li < this.logicalTotal; li++) {
        this._coordsFromLinear(li, this.shape, logicalCoords);
        if (!this._needsInputMapping) {
          let srcElem = srcBatchBase;
          for (let d = 0; d < this.rank; d++) srcElem += logicalCoords[d] * this._inputStrides[d];
          this._copyComplexFromAny(commandEncoder, input, srcElem * 8, dstBuffer, dstBase + li * 8);
          continue;
        }
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
          commandEncoder.copyBufferToBuffer(zeroBuf, 0, dstBuffer, dstBase + li * 8, 8);
          continue;
        }
        let srcElem = srcBatchBase;
        for (let d = 0; d < this.rank; d++) srcElem += viewCoords[d] * this._inputStrides[d];
        this._copyComplexFromAny(commandEncoder, input, srcElem * 8, dstBuffer, dstBase + li * 8);
      }
    }
  }

  _extractOutputChunkLarge(commandEncoder, { srcBuffer, srcOffsetBytes, output, outputOffsetBytes, batchStart, batchCount }) {
    const chunkBytes = batchCount * this._bytesPerBatch;
    if (!this._needsOutputMapping) {
      const dstOff = outputOffsetBytes + batchStart * this._outPhysBytesPerBatch;
      const outRanges = normalizeToContiguousRanges(output, dstOff, chunkBytes);
      if (outRanges.length === 1) {
        commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, outRanges[0].buffer, outRanges[0].offsetBytes, chunkBytes);
      } else {
        this.copier.unpack(commandEncoder, srcBuffer, srcOffsetBytes, outRanges);
      }
      return;
    }
    const viewShape = this.io.output.shape;
    const viewOffset = this.io.output.offset;
    const viewTotal = this._outViewTotal;
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    const viewStrides = this._shapeStrides(viewShape);
    const zeroBuf = this._ensureZeroComplexBuffer();
    for (let lb = 0; lb < batchCount; lb++) {
      const gb = batchStart + lb;
      const srcBase = srcOffsetBytes + lb * this.logicalTotal * 8;
      const dstBase = outputOffsetBytes + gb * viewTotal * 8;
      if (this.io.output.clearOutside) {
        for (let vi = 0; vi < viewTotal; vi++) {
          this._copyComplexToAny(commandEncoder, zeroBuf, 0, output, dstBase + vi * 8);
        }
      }
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
        this._copyComplexToAny(commandEncoder, srcBuffer, srcBase + li * 8, output, dstBase + vi * 8);
      }
    }
  }

  _extractStridedOutputChunkLarge(commandEncoder, { srcBuffer, srcOffsetBytes, output, outputOffsetBytes, batchStart, batchCount }) {
    if (outputOffsetBytes % 8 !== 0) {
      throw new Error(`outputOffsetBytes must be a multiple of 8 for complex-strided output; got ${outputOffsetBytes}`);
    }
    const extraOffsetElements = (outputOffsetBytes / 8) | 0;
    const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements, batchStart, batchCount);
    const outputBytes = getBufferByteLength(output);
    if (outputBytes < neededBytes) {
      throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
    }
    const logicalCoords = new Array(this.rank).fill(0);
    const viewCoords = new Array(this.rank).fill(0);
    const zeroBuf = this._ensureZeroComplexBuffer();
    const viewShape = this.io.output?.shape ?? null;
    const viewOffset = this.io.output?.offset ?? null;
    const viewTotal = this._outViewTotal;
    for (let lb = 0; lb < batchCount; lb++) {
      const gb = batchStart + lb;
      const srcBase = srcOffsetBytes + lb * this.logicalTotal * 8;
      const dstBatchBase = this._outputOffsetElements + extraOffsetElements + gb * this._outputBatchStrideElements;
      if (this._needsOutputMapping && this.io.output.clearOutside) {
        for (let vi = 0; vi < viewTotal; vi++) {
          this._coordsFromLinear(vi, viewShape, viewCoords);
          let dstElem = dstBatchBase;
          for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
          this._copyComplexToAny(commandEncoder, zeroBuf, 0, output, dstElem * 8);
        }
      }
      for (let li = 0; li < this.logicalTotal; li++) {
        this._coordsFromLinear(li, this.shape, logicalCoords);
        if (!this._needsOutputMapping) {
          let dstElem = dstBatchBase;
          for (let d = 0; d < this.rank; d++) dstElem += logicalCoords[d] * this._outputStrides[d];
          this._copyComplexToAny(commandEncoder, srcBuffer, srcBase + li * 8, output, dstElem * 8);
          continue;
        }
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
        this._copyComplexToAny(commandEncoder, srcBuffer, srcBase + li * 8, output, dstElem * 8);
      }
    }
  }

  _runZeroStageLargeChunk(commandEncoder, stage, { buffer, offsetBytes, chunkBytes, chunkComplex }) {
    if (!stage) return;
    const bg = this.device.createBindGroup({
      layout: stage.bgl,
      entries: [{ binding: 0, resource: { buffer, offset: offsetBytes, size: chunkBytes } }],
    });
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(stage.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(chunkComplex / this.workgroupSize), 1, 1);
    pass.end();
  }

  _resolveLargeChunkBatchCount(limitByBindings) {
    let maxBatchPerChunk = Math.max(1, Math.floor(limitByBindings));
    if (this.tuning.largeChunkMaxBatches != null) {
      maxBatchPerChunk = Math.min(maxBatchPerChunk, this.tuning.largeChunkMaxBatches);
    }
    return Math.max(1, maxBatchPerChunk);
  }

  _execLargeBatchSegmentedStaging(commandEncoder, { input, output, inputOffsetBytes, outputOffsetBytes }) {
    const outTarget = this.inPlace ? input : output;
    const outOffset = this.inPlace ? inputOffsetBytes : outputOffsetBytes;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxBatchPerChunk = this._resolveLargeChunkBatchCount(this._maxBindBytes / this._bytesPerBatch);
    const maxChunkBytes = maxBatchPerChunk * this._bytesPerBatch;
    const axisWorkOffset = alignBytes(maxChunkBytes, storageAlign);
    const chunkTotalBytes = axisWorkOffset + this.maxAxisWorkBytes;
    const chunkBuf = this._ensureLargeChunkBuffer(chunkTotalBytes);
    for (let b0 = 0; b0 < this.batch; b0 += maxBatchPerChunk) {
      const bCount = Math.min(maxBatchPerChunk, this.batch - b0);
      const chunkBytes = bCount * this._bytesPerBatch;
      const chunkComplex = bCount * this.logicalTotal;
      if (this._usesStridedInput) {
        this._embedStridedInputChunkLarge(commandEncoder, {
          input,
          inputOffsetBytes,
          batchStart: b0,
          batchCount: bCount,
          dstBuffer: chunkBuf,
          dstOffsetBytes: 0,
        });
      } else {
        this._embedInputChunkLarge(commandEncoder, {
          input,
          inputOffsetBytes,
          batchStart: b0,
          batchCount: bCount,
          dstBuffer: chunkBuf,
          dstOffsetBytes: 0,
        });
      }

      this._runZeroStageLargeChunk(commandEncoder, this.zeroRead, {
        buffer: chunkBuf,
        offsetBytes: 0,
        chunkBytes,
        chunkComplex,
      });

      for (let axis = 0; axis < this.rank; axis++) {
        const kind = this.axisKind[axis];
        if (kind === "mixed") {
          this.axisPlans[axis].exec(commandEncoder, { input: chunkBuf, inputOffsetBytes: 0, batch: bCount, temp: null });
          continue;
        }
        const axisPlan = this.axisAdvanced[axis];
        if (!axisPlan) throw new Error(`Internal error: missing advanced axis plan for axis=${axis}`);
        const axisLines = bCount * (this.logicalTotal / this.shape[axis]);
        const axisWorkView = viewFromArena(chunkBuf, alignBytes(chunkBytes, storageAlign), axisPlan.workBytes);
        axisPlan.exec(commandEncoder, {
          dataBuf: chunkBuf,
          dataOffsetBytes: 0,
          axisWork: axisWorkView,
          scratch: null,
          lineCount: axisLines,
          paramChunkBase: 0,
        });
      }

      this._runZeroStageLargeChunk(commandEncoder, this.zeroWrite, {
        buffer: chunkBuf,
        offsetBytes: 0,
        chunkBytes,
        chunkComplex,
      });

      if (this._usesStridedOutput) {
        this._extractStridedOutputChunkLarge(commandEncoder, {
          srcBuffer: chunkBuf,
          srcOffsetBytes: 0,
          output: outTarget,
          outputOffsetBytes: outOffset,
          batchStart: b0,
          batchCount: bCount,
        });
      } else {
        this._extractOutputChunkLarge(commandEncoder, {
          srcBuffer: chunkBuf,
          srcOffsetBytes: 0,
          output: outTarget,
          outputOffsetBytes: outOffset,
          batchStart: b0,
          batchCount: bCount,
        });
      }
    }
  }

  _ensureOutOfCoreSegmentedFullVolumeState() {
    if (this._outOfCoreSegmentedFullVolumeState) return this._outOfCoreSegmentedFullVolumeState;
    if (!this._outOfCoreSegmentedFullVolumeMode) {
      throw new Error("Internal error: segmented full-volume out-of-core mode is not enabled");
    }

    const N0 = this.shape[0];
    const N1 = this.shape[1];
    const N2 = this.shape[2];
    if (!(N0 === N1 && N1 === N2)) {
      throw new Error(
        `Segmented full-volume mode currently expects cubic rank-3 shape; got ${JSON.stringify(this.shape)}`
      );
    }
    const N = N0;
    const rowBytes = N * 8;
    const planeBytes = N * N * 8;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
    const maxBindBytes = this._maxBindBytes;
    if (!Number.isFinite(maxBindBytes) || maxBindBytes < rowBytes) {
      throw new Error(
        `Segmented full-volume mode requires maxStorageBufferBindingSize >= one row (${rowBytes} bytes); got ${maxBindBytes}`
      );
    }

    let preferredSegmentBytes = Math.min(this.mainBytes, maxBufferSize);
    preferredSegmentBytes = alignDownBytes(preferredSegmentBytes, planeBytes);
    preferredSegmentBytes = alignDownBytes(preferredSegmentBytes, storageAlign);
    if (!Number.isInteger(preferredSegmentBytes) || preferredSegmentBytes < planeBytes) {
      preferredSegmentBytes = alignBytes(planeBytes, storageAlign);
    }
    const segmented = this._createInternalSegmentedView(this.mainBytes, preferredSegmentBytes);
    const dataView = segmented.view;
    const segmentBytes = segmented.segmentBytes;

    const ringDepth = Math.max(1, Math.min(3, this.tuning.outOfCoreBurstWindows ?? 1));
    const axis0Policy = this._outOfCoreAxisWindowPolicy?.[0] ?? resolveOutOfCoreAxisWindowPolicy({
      axisLen: N,
      lineBytes: rowBytes,
      linesTotal: this.batch * (this.logicalTotal / N),
      maxBindBytes: this._maxBindBytes,
      axisKind: "mixed",
      tuning: this.tuning,
      axisIndex: 0,
      storageAlign,
    });
    const maxAxis0LinesByBind = Math.max(1, Math.floor(maxBindBytes / rowBytes));
    const maxAxis0LinesBySeg = Math.max(1, Math.floor(segmentBytes / rowBytes));
    let axis0LinesPerChunk = Math.max(1, Math.min(axis0Policy.linesPerChunk, maxAxis0LinesByBind, maxAxis0LinesBySeg));
    const axis0ChunkBytes = axis0LinesPerChunk * rowBytes;

    const axis0Ring = Array.from({ length: ringDepth }, () =>
      this.device.createBuffer({
        size: axis0ChunkBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
    );

    const planRows = createFftPlan(this.device, {
      shape: [N],
      direction: this.direction,
      normalize: "none",
      inPlace: true,
      layout: "interleaved",
      precision: "f32",
      batch: N,
      maxStorageBufferBindingSize: this._maxBindBytes,
    });
    const axis0PlanCache = new Map();
    const getAxis0Plan = (lines) => {
      const key = String(lines);
      if (axis0PlanCache.has(key)) return axis0PlanCache.get(key);
      const p = createFftPlan(this.device, {
        shape: [N],
        direction: this.direction,
        normalize: "none",
        inPlace: true,
        layout: "interleaved",
        precision: "f32",
        batch: lines,
        maxStorageBufferBindingSize: this._maxBindBytes,
      });
      axis0PlanCache.set(key, p);
      return p;
    };

    const transposePipeline = this._getOutOfCoreTransposePipeline(N, N);
    this.device.queue.writeBuffer(this._outOfCoreTranspose.params, 0, new Uint32Array([1, 0, 0, 0]));
    const transposeDispatch = Math.ceil(N / this._outOfCoreTranspose.tile);
    const slabs = [];
    for (let i = 0; i < ringDepth; i++) {
      const a = this.device.createBuffer({
        size: planeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const b = this.device.createBuffer({
        size: planeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const bgAB = this.device.createBindGroup({
        layout: this._outOfCoreTranspose.bgl,
        entries: [
          { binding: 0, resource: { buffer: a, offset: 0, size: planeBytes } },
          { binding: 1, resource: { buffer: b, offset: 0, size: planeBytes } },
          { binding: 2, resource: { buffer: this._outOfCoreTranspose.params, offset: 0, size: 16 } },
        ],
      });
      const bgBA = this.device.createBindGroup({
        layout: this._outOfCoreTranspose.bgl,
        entries: [
          { binding: 0, resource: { buffer: b, offset: 0, size: planeBytes } },
          { binding: 1, resource: { buffer: a, offset: 0, size: planeBytes } },
          { binding: 2, resource: { buffer: this._outOfCoreTranspose.params, offset: 0, size: 16 } },
        ],
      });
      slabs.push({ a, b, bgAB, bgBA });
    }

    const slabBgl = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const slabPl = this.device.createPipelineLayout({ bindGroupLayouts: [slabBgl] });
    const gatherCode = generateGatherAxis2SlabWGSL({ N, workgroupSize: this.workgroupSize });
    const scatterCode = generateScatterAxis2SlabWGSL({ N, workgroupSize: this.workgroupSize });
    const slabGather = this.cache.getComputePipeline({ code: gatherCode, layout: slabPl });
    const slabScatter = this.cache.getComputePipeline({ code: scatterCode, layout: slabPl });
    const slabParams = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const maxZChunkByBind = Math.max(1, Math.floor((maxBindBytes - rowBytes) / planeBytes) + 1);
    const maxZChunkSpanBytes = (maxZChunkByBind - 1) * planeBytes + rowBytes;

    const state = {
      N,
      rowBytes,
      planeBytes,
      dataView,
      segmentBytes,
      ringDepth,
      axis0LinesPerChunk,
      axis0ChunkBytes,
      axis0Ring,
      planRows,
      axis0PlanCache,
      getAxis0Plan,
      slabs,
      transposePipeline,
      transposeDispatch,
      slabKernel: {
        bgl: slabBgl,
        pl: slabPl,
        gather: slabGather,
        scatter: slabScatter,
        params: slabParams,
        maxZChunkByBind,
        maxZChunkSpanBytes,
      },
    };
    this._outOfCoreSegmentedFullVolumeState = state;
    this._segmentedFullVolumeMeta = {
      mode: "rank3-segmented-slab",
      segmentBytes,
      segmentCount: dataView.segments.length,
      ringDepth,
      axis0LinesPerChunk,
      axis0ChunkBytes,
      axis0ChunkUtilization: Number.isFinite(maxBindBytes) ? axis0ChunkBytes / maxBindBytes : null,
      axis2MaxZChunk: maxZChunkByBind,
      axis2ChunkSpanBytes: maxZChunkSpanBytes,
      axis2ChunkUtilization: Number.isFinite(maxBindBytes) ? maxZChunkSpanBytes / maxBindBytes : null,
      maxStorageBufferBindingSize: maxBindBytes,
      maxBufferSize,
    };
    return state;
  }

  _resolveAxis2RowChunk(dataView, logicalStartBytes, { planeBytes, rowBytes, zRemain, maxZChunkByBind }) {
    if (zRemain <= 0) return null;
    const first = normalizeToContiguousRanges(dataView, logicalStartBytes, rowBytes)[0];
    let segRemain = 0;
    for (const seg of dataView.segments) {
      if (seg.buffer !== first.buffer) continue;
      const segStart = seg.offsetBytes;
      const segEnd = seg.offsetBytes + seg.sizeBytes;
      if (first.offsetBytes < segStart || first.offsetBytes >= segEnd) continue;
      segRemain = segEnd - first.offsetBytes;
      break;
    }
    if (segRemain < rowBytes) {
      throw new Error("Segmented axis2 slab chunking failed to locate source segment capacity");
    }
    const maxZBySeg = Math.max(1, Math.floor((segRemain - rowBytes) / planeBytes) + 1);
    let zCount = Math.max(1, Math.min(zRemain, maxZChunkByBind, maxZBySeg));
    while (zCount >= 1) {
      const spanBytes = (zCount - 1) * planeBytes + rowBytes;
      const ranges = normalizeToContiguousRanges(dataView, logicalStartBytes, spanBytes);
      if (ranges.length === 1) {
        return { range: ranges[0], zCount, spanBytes };
      }
      zCount -= 1;
    }
    throw new Error("Segmented axis2 slab chunking could not resolve a single bindable source window");
  }

  _runTransposeSlab(commandEncoder, state, slab, toBA) {
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(state.transposePipeline);
    pass.setBindGroup(0, toBA ? slab.bgBA : slab.bgAB);
    pass.dispatchWorkgroups(state.transposeDispatch, state.transposeDispatch, 1);
    pass.end();
  }

  _gatherAxis2RowSlab(commandEncoder, state, slab, yIndex) {
    let z0 = 0;
    while (z0 < state.N) {
      const logicalStartBytes = (yIndex * state.N + z0 * state.N * state.N) * 8;
      const chunk = this._resolveAxis2RowChunk(state.dataView, logicalStartBytes, {
        planeBytes: state.planeBytes,
        rowBytes: state.rowBytes,
        zRemain: state.N - z0,
        maxZChunkByBind: state.slabKernel.maxZChunkByBind,
      });
      const count = chunk.zCount * state.N;
      this.device.queue.writeBuffer(state.slabKernel.params, 0, new Uint32Array([count, z0, chunk.zCount, 0]));
      const bg = this.device.createBindGroup({
        layout: state.slabKernel.bgl,
        entries: [
          { binding: 0, resource: { buffer: chunk.range.buffer, offset: chunk.range.offsetBytes, size: chunk.spanBytes } },
          { binding: 1, resource: { buffer: slab.a, offset: 0, size: state.planeBytes } },
          { binding: 2, resource: { buffer: state.slabKernel.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(state.slabKernel.gather);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
      pass.end();
      z0 += chunk.zCount;
    }
  }

  _scatterAxis2RowSlab(commandEncoder, state, slab, yIndex) {
    let z0 = 0;
    while (z0 < state.N) {
      const logicalStartBytes = (yIndex * state.N + z0 * state.N * state.N) * 8;
      const chunk = this._resolveAxis2RowChunk(state.dataView, logicalStartBytes, {
        planeBytes: state.planeBytes,
        rowBytes: state.rowBytes,
        zRemain: state.N - z0,
        maxZChunkByBind: state.slabKernel.maxZChunkByBind,
      });
      const count = chunk.zCount * state.N;
      this.device.queue.writeBuffer(state.slabKernel.params, 0, new Uint32Array([count, z0, chunk.zCount, 0]));
      const bg = this.device.createBindGroup({
        layout: state.slabKernel.bgl,
        entries: [
          { binding: 0, resource: { buffer: slab.a, offset: 0, size: state.planeBytes } },
          { binding: 1, resource: { buffer: chunk.range.buffer, offset: chunk.range.offsetBytes, size: chunk.spanBytes } },
          { binding: 2, resource: { buffer: state.slabKernel.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(state.slabKernel.scatter);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
      pass.end();
      z0 += chunk.zCount;
    }
  }

  _applyScaleLargeDataSegmented(commandEncoder, { dataView, totalComplex, scale }) {
    if (scale === 1.0) return;
    const maxChunkComplex = Math.max(1, Math.floor(this._maxBindBytes / 8));
    const maxChunkBytes = maxChunkComplex * 8;
    const chunkBuf = this._ensureLargeChunkBuffer(maxChunkBytes);
    const chunkCount = Math.ceil(totalComplex / maxChunkComplex);
    const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    const paramStride = alignBytes(32, uniformAlign);
    const paramsBuf = this._ensureScaleChunkParamsBuffer(chunkCount * paramStride);
    let chunkIndex = 0;
    for (let i0 = 0; i0 < totalComplex; i0 += maxChunkComplex) {
      const n = Math.min(maxChunkComplex, totalComplex - i0);
      const bytes = n * 8;
      const srcOff = i0 * 8;
      this._copyAnyToAny(commandEncoder, {
        src: dataView,
        srcOffsetBytes: srcOff,
        dst: chunkBuf,
        dstOffsetBytes: 0,
        bytes,
      });
      const paramOff = chunkIndex * paramStride;
      this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([n, 0, 0, 0]));
      this.device.queue.writeBuffer(paramsBuf, paramOff + 16, new Float32Array([scale, 0, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: this.scale.bgl,
        entries: [
          { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: bytes } },
          { binding: 1, resource: { buffer: paramsBuf, offset: paramOff, size: 32 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.scale.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(n / this.workgroupSize), 1, 1);
      pass.end();
      this._copyAnyToAny(commandEncoder, {
        src: chunkBuf,
        srcOffsetBytes: 0,
        dst: dataView,
        dstOffsetBytes: srcOff,
        bytes,
      });
      chunkIndex += 1;
    }
  }

  _execOutOfCoreFourStepSegmentedRank3(commandEncoder, { input, output, inputOffsetBytes, outputOffsetBytes }) {
    if (inputOffsetBytes !== 0 || outputOffsetBytes !== 0) {
      throw new Error(
        `Segmented full-volume mode currently requires zero input/output offsets; got inputOffsetBytes=${inputOffsetBytes}, outputOffsetBytes=${outputOffsetBytes}`
      );
    }
    const state = this._ensureOutOfCoreSegmentedFullVolumeState();
    const outTarget = this.inPlace ? input : output;
    if (!outTarget) throw new Error("Segmented full-volume mode requires an output target");

    this._copyAnyToAny(commandEncoder, {
      src: input,
      srcOffsetBytes: 0,
      dst: state.dataView,
      dstOffsetBytes: 0,
      bytes: this.mainBytes,
    });

    const totalAxis0Lines = this.batch * (this.logicalTotal / state.N);
    let axis0Line = 0;
    while (axis0Line < totalAxis0Lines) {
      const burstItems = [];
      for (let i = 0; i < state.axis0Ring.length && axis0Line < totalAxis0Lines; i++) {
        const lines = Math.min(state.axis0LinesPerChunk, totalAxis0Lines - axis0Line);
        const bytes = lines * state.rowBytes;
        const slot = state.axis0Ring[i];
        this._copyAnyToAny(commandEncoder, {
          src: state.dataView,
          srcOffsetBytes: axis0Line * state.rowBytes,
          dst: slot,
          dstOffsetBytes: 0,
          bytes,
        });
        burstItems.push({
          slot,
          lines,
          bytes,
          axis0Line,
          axis0Plan: state.getAxis0Plan(lines),
        });
        axis0Line += lines;
      }
      for (const item of burstItems) {
        item.axis0Plan.exec(commandEncoder, { input: item.slot, inputOffsetBytes: 0, batch: item.lines });
      }
      for (const item of burstItems) {
        this._copyAnyToAny(commandEncoder, {
          src: item.slot,
          srcOffsetBytes: 0,
          dst: state.dataView,
          dstOffsetBytes: item.axis0Line * state.rowBytes,
          bytes: item.bytes,
        });
      }
    }

    for (let z0 = 0; z0 < state.N; z0 += state.ringDepth) {
      const burst = Math.min(state.ringDepth, state.N - z0);
      for (let i = 0; i < burst; i++) {
        const z = z0 + i;
        const slab = state.slabs[i];
        this._copyAnyToAny(commandEncoder, {
          src: state.dataView,
          srcOffsetBytes: z * state.planeBytes,
          dst: slab.a,
          dstOffsetBytes: 0,
          bytes: state.planeBytes,
        });
      }
      for (let i = 0; i < burst; i++) {
        const slab = state.slabs[i];
        this._runTransposeSlab(commandEncoder, state, slab, false);
        state.planRows.exec(commandEncoder, { input: slab.b, inputOffsetBytes: 0, batch: state.N });
        this._runTransposeSlab(commandEncoder, state, slab, true);
      }
      for (let i = 0; i < burst; i++) {
        const z = z0 + i;
        const slab = state.slabs[i];
        this._copyAnyToAny(commandEncoder, {
          src: slab.a,
          srcOffsetBytes: 0,
          dst: state.dataView,
          dstOffsetBytes: z * state.planeBytes,
          bytes: state.planeBytes,
        });
      }
    }

    for (let y0 = 0; y0 < state.N; y0 += state.ringDepth) {
      const burst = Math.min(state.ringDepth, state.N - y0);
      for (let i = 0; i < burst; i++) {
        const y = y0 + i;
        this._gatherAxis2RowSlab(commandEncoder, state, state.slabs[i], y);
      }
      for (let i = 0; i < burst; i++) {
        const slab = state.slabs[i];
        this._runTransposeSlab(commandEncoder, state, slab, false);
        state.planRows.exec(commandEncoder, { input: slab.b, inputOffsetBytes: 0, batch: state.N });
        this._runTransposeSlab(commandEncoder, state, slab, true);
      }
      for (let i = 0; i < burst; i++) {
        const y = y0 + i;
        this._scatterAxis2RowSlab(commandEncoder, state, state.slabs[i], y);
      }
    }

    const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
    this._applyScaleLargeDataSegmented(commandEncoder, {
      dataView: state.dataView,
      totalComplex: this.totalComplex,
      scale,
    });

    this._copyAnyToAny(commandEncoder, {
      src: state.dataView,
      srcOffsetBytes: 0,
      dst: outTarget,
      dstOffsetBytes: 0,
      bytes: this.mainBytes,
    });
  }

  _applyScaleLargeData(commandEncoder, { dataBuffer, dataOffsetBytes, totalComplex, scale }) {
    if (scale === 1.0) return;
    const maxChunkComplex = Math.max(1, Math.floor(this._maxBindBytes / 8));
    const maxChunkBytes = maxChunkComplex * 8;
    const chunkBuf = this._ensureLargeChunkBuffer(maxChunkBytes);
    const chunkCount = Math.ceil(totalComplex / maxChunkComplex);
    const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    const paramStride = alignBytes(32, uniformAlign);
    const paramsBuf = this._ensureScaleChunkParamsBuffer(chunkCount * paramStride);
    let chunkIndex = 0;
    for (let i0 = 0; i0 < totalComplex; i0 += maxChunkComplex) {
      const n = Math.min(maxChunkComplex, totalComplex - i0);
      const bytes = n * 8;
      const srcOff = dataOffsetBytes + i0 * 8;
      commandEncoder.copyBufferToBuffer(dataBuffer, srcOff, chunkBuf, 0, bytes);
      const paramOff = chunkIndex * paramStride;
      this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([n, 0, 0, 0]));
      this.device.queue.writeBuffer(paramsBuf, paramOff + 16, new Float32Array([scale, 0, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: this.scale.bgl,
        entries: [
          { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: bytes } },
          { binding: 1, resource: { buffer: paramsBuf, offset: paramOff, size: 32 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.scale.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(n / this.workgroupSize), 1, 1);
      pass.end();

      commandEncoder.copyBufferToBuffer(chunkBuf, 0, dataBuffer, srcOff, bytes);
      chunkIndex += 1;
    }
  }

  _execOutOfCoreAdvancedAxisWindows(commandEncoder, { axisExecutor, permShape, dataRange, axisIndex = 0 }) {
    const axisLen = permShape[0];
    const linesTotal = this.batch * (this.logicalTotal / axisLen);
    const lineBytes = axisLen * 8;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const axisKind = this._outOfCoreAxisEffectiveKind?.[axisIndex] ?? this.axisKind?.[axisIndex] ?? "mixed";
    const schedule = resolveOutOfCoreAxisWindowPolicy({
      axisLen,
      lineBytes,
      linesTotal,
      maxBindBytes: this._maxBindBytes,
      axisKind,
      tuning: this.tuning,
      axisIndex,
      storageAlign,
    });
    this._outOfCoreAxisWindowPolicy[axisIndex] = schedule;
    let linesPerChunk = schedule.linesPerChunk;
    if (lineBytes <= this._maxBindBytes) {
      const maxLinesByBind = Math.floor(this._maxBindBytes / lineBytes);
      if (!Number.isInteger(maxLinesByBind) || maxLinesByBind < 1) {
        throw new Error(
          `Out-of-core advanced axis chunking failed to size bindable line windows: axisLen=${axisLen} lineBytes=${lineBytes} ` +
            `maxStorageBufferBindingSize=${this._maxBindBytes}.` +
            ` (routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)})`
        );
      }
      linesPerChunk = Math.max(1, Math.min(linesTotal, maxLinesByBind, linesPerChunk));
    } else if (!(axisExecutor?.supportsBoundedLineSlicing?.())) {
      throw new Error(
        `Out-of-core advanced axis requires one axis line to fit maxStorageBufferBindingSize ` +
          `or a bounded sliced-line executor; axisLen=${axisLen} lineBytes=${lineBytes} ` +
          `maxStorageBufferBindingSize=${this._maxBindBytes}.` +
          ` (routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)})`
      );
    }
    const maxChunkDataBytes = linesPerChunk * lineBytes;
    const axisWorkOffset = alignBytes(maxChunkDataBytes, storageAlign);
    const chunkBuf = this._ensureLargeChunkBuffer(axisWorkOffset + axisExecutor.workBytes);
    const axisWorkView = viewFromArena(chunkBuf, axisWorkOffset, axisExecutor.workBytes);
    let paramChunkBase = 0;

    for (let line0 = 0; line0 < linesTotal; line0 += linesPerChunk) {
      const lines = Math.min(linesPerChunk, linesTotal - line0);
      const bytes = lines * lineBytes;
      const srcOff = dataRange.offsetBytes + line0 * lineBytes;
      commandEncoder.copyBufferToBuffer(dataRange.buffer, srcOff, chunkBuf, 0, bytes);
      const usedChunks = axisExecutor.exec(commandEncoder, {
        dataBuf: chunkBuf,
        dataOffsetBytes: 0,
        axisWork: axisWorkView,
        scratch: null,
        lineCount: lines,
        paramChunkBase,
      });
      paramChunkBase += usedChunks;
      commandEncoder.copyBufferToBuffer(chunkBuf, 0, dataRange.buffer, srcOff, bytes);
    }
  }

  _execOutOfCoreFourStep(commandEncoder, { input, output, temp, inputOffsetBytes, outputOffsetBytes }) {
    if (this.precision !== "f32") {
      throw new Error('Out-of-core four-step mode currently supports precision:"f32" only');
    }
    const inBytes = this._needsInputMapping ? this._inPhysBytes : this.mainBytes;
    const outBytes = this._needsOutputMapping ? this._outPhysBytes : this.mainBytes;

    const outTarget = this.inPlace ? input : output;
    const outOffset = this.inPlace ? inputOffsetBytes : outputOffsetBytes;

    const inRanges = this._usesStridedInput ? null : normalizeToContiguousRanges(input, inputOffsetBytes, inBytes);
    const inRange = inRanges && inRanges.length === 1 ? inRanges[0] : null;
    const outRanges = this._usesStridedOutput ? null : normalizeToContiguousRanges(outTarget, outOffset, outBytes);
    const outRange = outRanges && outRanges.length === 1 ? outRanges[0] : null;

    let dataRange = null;
    if (!this._needsOutputMapping && !this._usesStridedOutput && outRange && outRange.sizeBytes >= this.mainBytes) {
      dataRange = outRange;
    } else if (!this._needsInputMapping && !this._usesStridedInput && inRange && inRange.sizeBytes >= this.mainBytes) {
      dataRange = inRange ?? { buffer: this._ensureLargeAuxBuffer(this.mainBytes), offsetBytes: 0, sizeBytes: this.mainBytes };
    } else {
      dataRange = { buffer: this._ensureLargeAuxBuffer(this.mainBytes), offsetBytes: 0, sizeBytes: this.mainBytes };
    }

    if (this._usesStridedInput) {
      this._embedStridedInputOutOfCore(commandEncoder, { input, inputOffsetBytes, dataRange });
    } else {
      this._embedInputOutOfCore(commandEncoder, { inputRanges: inRanges, dataRange });
    }

    let transRange = null;
    if (temp) {
      const tRanges = normalizeToContiguousRanges(temp, 0, this.mainBytes);
      if (tRanges.length === 1) {
        const candidate = tRanges[0];
        if (!this._rangesOverlap(candidate, dataRange)) {
          transRange = candidate;
        }
      }
    }
    if (!transRange && inRange && !this._rangesOverlap(inRange, dataRange) && inRange.sizeBytes >= this.mainBytes) {
      transRange = inRange;
    }
    if (!transRange && outRange && !this._rangesOverlap(outRange, dataRange) && outRange.sizeBytes >= this.mainBytes) {
      transRange = outRange;
    }
    if (!transRange) {
      const stage = this._ensureLargeStageBuffer(this.mainBytes);
      transRange = { buffer: stage, offsetBytes: 0, sizeBytes: this.mainBytes };
    }
    if (this._rangesOverlap(transRange, dataRange)) {
      const stage = this._ensureLargeStageBuffer(this.mainBytes);
      transRange = { buffer: stage, offsetBytes: 0, sizeBytes: this.mainBytes };
    }
    if (this._rangesOverlap(transRange, dataRange)) {
      throw new Error("Out-of-core four-step requires distinct data and transpose staging ranges");
    }

    if (this.zeroPad.read) {
      this._zeroLogicalOutsideRange(commandEncoder, {
        dataBuffer: dataRange.buffer,
        dataOffsetBytes: dataRange.offsetBytes,
        start: this.zeroPad.read.start,
        end: this.zeroPad.read.end,
      });
    }

    if (this.axisKind[0] === "mixed") {
      this._outOfCoreAxisPlans[0].exec(commandEncoder, {
        input: dataRange.buffer,
        inputOffsetBytes: dataRange.offsetBytes,
        batch: this.batch,
      });
    } else {
      this._execOutOfCoreAdvancedAxisWindows(commandEncoder, {
        axisExecutor: this._outOfCoreAxisPlans[0],
        permShape: this._outOfCoreAxisPermShapes[0],
        dataRange,
        axisIndex: 0,
      });
    }

    for (let axis = 1; axis < this.rank; axis++) {
      const axisPlan = this._outOfCoreAxisPlans[axis];
      if (!axisPlan) throw new Error(`Internal error: missing out-of-core axis plan for axis=${axis}`);
      const kind = this.axisKind[axis];
      // Keep rank-2 transpose stripes as an optimization; rank>2 uses generic axis permutation.
      if (axis === 1 && this.rank === 2) {
        const [Nx, Ny] = this.shape;
        this._transposeOutOfCore2dStripes(commandEncoder, {
          srcBuffer: dataRange.buffer,
          srcOffsetBytes: dataRange.offsetBytes,
          dstBuffer: transRange.buffer,
          dstOffsetBytes: transRange.offsetBytes,
          Nx,
          Ny,
          batch: this.batch,
        });
        if (kind === "mixed") {
          axisPlan.exec(commandEncoder, {
            input: transRange.buffer,
            inputOffsetBytes: transRange.offsetBytes,
            batch: this.batch,
          });
        } else {
          this._execOutOfCoreAdvancedAxisWindows(commandEncoder, {
            axisExecutor: axisPlan,
            permShape: this._outOfCoreAxisPermShapes[axis],
            dataRange: transRange,
            axisIndex: axis,
          });
        }
        this._transposeOutOfCore2dStripes(commandEncoder, {
          srcBuffer: transRange.buffer,
          srcOffsetBytes: transRange.offsetBytes,
          dstBuffer: dataRange.buffer,
          dstOffsetBytes: dataRange.offsetBytes,
          Nx: Ny,
          Ny: Nx,
          batch: this.batch,
        });
      } else {
        this._permuteAxisToFront(commandEncoder, { srcRange: dataRange, dstRange: transRange, axis });
        if (kind === "mixed") {
          axisPlan.exec(commandEncoder, {
            input: transRange.buffer,
            inputOffsetBytes: transRange.offsetBytes,
            batch: this.batch,
          });
        } else {
          this._execOutOfCoreAdvancedAxisWindows(commandEncoder, {
            axisExecutor: axisPlan,
            permShape: this._outOfCoreAxisPermShapes[axis],
            dataRange: transRange,
            axisIndex: axis,
          });
        }
        this._permuteAxisFromFront(commandEncoder, { srcRange: transRange, dstRange: dataRange, axis });
      }
    }

    const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
    this._applyScaleLargeData(commandEncoder, {
      dataBuffer: dataRange.buffer,
      dataOffsetBytes: dataRange.offsetBytes,
      totalComplex: this.totalComplex,
      scale,
    });

    if (this.zeroPad.write) {
      this._zeroLogicalOutsideRange(commandEncoder, {
        dataBuffer: dataRange.buffer,
        dataOffsetBytes: dataRange.offsetBytes,
        start: this.zeroPad.write.start,
        end: this.zeroPad.write.end,
      });
    }

    if (this._usesStridedOutput) {
      this._extractStridedOutputOutOfCore(commandEncoder, { dataRange, output: outTarget, outputOffsetBytes: outOffset });
    } else {
      this._extractOutputOutOfCore(commandEncoder, { dataRange, outputRanges: outRanges });
    }
  }

  destroy() {
    if (this._destroyed) return;
    for (const p of this.axisPlans) p?.destroy?.();
    for (const p of this._outOfCoreAxisPlans ?? []) {
      if (!p) continue;
      if (this.axisPlans.includes(p)) continue;
      if (this.axisAdvanced.includes(p)) continue;
      p.destroy?.();
    }
    for (const ax of this.axisAdvanced) ax?.destroy?.();
    this.axis0OnTransposed?.destroy?.();
    if (this._outOfCoreAxis0OnTransposed && !(this._outOfCoreAxisPlans ?? []).includes(this._outOfCoreAxis0OnTransposed)) {
      this._outOfCoreAxis0OnTransposed.destroy?.();
    }
    this._outOfCoreTranspose?.params?.destroy?.();
    this._outOfCoreAxis1TailPermute?.params?.destroy?.();
    this._outOfCoreRank3Axis2Permute?.params?.destroy?.();
    this._outOfCoreGenericPermute?.params?.destroy?.();
    this._outOfCoreAdjacentSwapTiled?.params?.destroy?.();
    if (this._outOfCoreSegmentedFullVolumeState) {
      const s = this._outOfCoreSegmentedFullVolumeState;
      for (const p of s.axis0PlanCache?.values?.() ?? []) p?.destroy?.();
      s.planRows?.destroy?.();
      for (const b of s.axis0Ring ?? []) b?.destroy?.();
      for (const slab of s.slabs ?? []) {
        slab?.a?.destroy?.();
        slab?.b?.destroy?.();
      }
      s.slabKernel?.params?.destroy?.();
      this._destroySegmentedView(s.dataView);
      this._outOfCoreSegmentedFullVolumeState = null;
    }
    this.scale.params.destroy();
    this.ioEmbed?.params?.destroy?.();
    this.ioExtract?.params?.destroy?.();
    this.f16?.params?.destroy?.();
    this.stridedIn?.params?.destroy?.();
    this.stridedOut?.params?.destroy?.();
    this.transpose?.params?.destroy?.();
    this._largeStageBuffer?.destroy?.();
    this._largeChunkBuffer?.destroy?.();
    this._largeAuxBuffer?.destroy?.();
    this._scaleChunkParamsBuffer?.destroy?.();
    for (const b of this._retiredLargeStageBuffers) b?.destroy?.();
    for (const b of this._retiredLargeChunkBuffers) b?.destroy?.();
    for (const b of this._retiredLargeAuxBuffers) b?.destroy?.();
    for (const b of this._retiredScaleChunkParamsBuffers) b?.destroy?.();
    this._zeroComplexBuffer?.destroy?.();
    this._splitWorkspace?.mainStage?.destroy?.();
    this._splitWorkspace?.scratch?.destroy?.();
    this._splitWorkspace?.axisWork?.destroy?.();
    this._splitWorkspace?.transpose?.destroy?.();
    this._arena?.destroy?.();
    super.destroy();
  }

  exec(commandEncoder, execOpts) {
    if (this._destroyed) throw new Error("plan destroyed");
    const { input, output, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
    if (!input) throw new Error("exec requires input");
    if (!this.inPlace && !output) throw new Error("exec requires output when inPlace=false");
    if (this.inPlace && output && output !== input) throw new Error("inPlace=true requires output omitted or equal to input");
    if (this._outOfCoreFourStepMode) {
      if (this._outOfCoreSegmentedFullVolumeMode) {
        this._execOutOfCoreFourStepSegmentedRank3(commandEncoder, {
          input,
          output,
          inputOffsetBytes,
          outputOffsetBytes,
        });
        return;
      }
      this._execOutOfCoreFourStep(commandEncoder, { input, output, temp, inputOffsetBytes, outputOffsetBytes });
      return;
    }

    // Determine compute buffer (vec2<f32> contiguous) and ensure input is in it.
    // Fast path: f32, no ioView, no staging needed, contiguous input/output.
    const canDirect =
      !this.needsMainStage &&
      this.precision === "f32" &&
      !this._usesStridedInput &&
      !this._usesStridedOutput &&
      isGpuBuffer(input) &&
      inputOffsetBytes === 0 &&
      (this.inPlace || (isGpuBuffer(output) && outputOffsetBytes === 0));
    const needsLargeStaging = this._largeBatchChunkMode && !canDirect;
    let largeStagingArena = null;
    if (needsLargeStaging) {
      if (temp) {
        const tempAliasesInput = buffersAlias(temp, input);
        const tempAliasesOutput = output ? buffersAlias(temp, output) : false;
        if (tempAliasesInput || tempAliasesOutput) {
          throw new Error("Large-batch chunk mode staging temp must not alias input/output buffers");
        }
        largeStagingArena = temp;
      } else {
        largeStagingArena = this._ensureLargeStageBuffer(this.mainBytes);
      }
    }

    let arena = needsLargeStaging ? this._arena : (temp ?? this._arena);
    if (temp && this._arena && !needsLargeStaging) {
      // Staged execution emits copyBufferToBuffer commands; using temp slices that alias input/output
      // (same underlying GPUBuffer, different offsets) is invalid on some WebGPU stacks.
      const tempAliasesInput = buffersAlias(temp, input);
      const tempAliasesOutput = output ? buffersAlias(temp, output) : false;
      if (tempAliasesInput || tempAliasesOutput) {
        arena = this._arena;
      }
    }
    if (temp && !this._arena && this._splitWorkspace && !needsLargeStaging) {
      const tempAliasesInput = buffersAlias(temp, input);
      const tempAliasesOutput = output ? buffersAlias(temp, output) : false;
      if (tempAliasesInput || tempAliasesOutput) {
        arena = null; // fall back to internal split workspace
      }
    }
    let workspaceViews = this._resolveWorkspaceViews(arena);
    let { mainStage, scratch, axisWork, transpose } = workspaceViews;

    let dataBuf = null;
    let dataOff = 0;
    if (canDirect) {
      dataBuf = this.inPlace ? input : output;
      dataOff = 0;
      if (!this.inPlace && input !== output) {
        commandEncoder.copyBufferToBuffer(input, 0, output, 0, this.mainBytes);
      }
    } else {
      let stagingArena = needsLargeStaging ? largeStagingArena : (mainStage ?? scratch);
      let stageRanges = normalizeToContiguousRanges(stagingArena, 0, this.mainBytes);
      if (stageRanges.length !== 1) {
        if (!needsLargeStaging && temp) {
          arena = this._arena ?? null;
          workspaceViews = this._resolveWorkspaceViews(arena);
          ({ mainStage, scratch, axisWork, transpose } = workspaceViews);
          stagingArena = mainStage ?? scratch;
          stageRanges = normalizeToContiguousRanges(stagingArena, 0, this.mainBytes);
        }
        if (stageRanges.length !== 1) {
          if (this._largeBatchChunkMode) {
            this._execLargeBatchSegmentedStaging(commandEncoder, { input, output, inputOffsetBytes, outputOffsetBytes });
            return;
          }
          throw new Error("staging workspace must expose a contiguous range covering batch*product(shape)*8 bytes");
        }
      }
      const stageRange = stageRanges[0];
      dataBuf = stageRange.buffer;
      dataOff = stageRange.offsetBytes;

      if (this._usesStridedInput) {
        if (this._needsInputMapping) {
          this._embedStridedInputOutOfCore(commandEncoder, {
            input,
            inputOffsetBytes,
            dataRange: { buffer: dataBuf, offsetBytes: dataOff, sizeBytes: this.mainBytes },
          });
        } else {
          if (this._largeBatchChunkMode || !isGpuBuffer(input)) {
            this._embedStridedInputOutOfCore(commandEncoder, {
              input,
              inputOffsetBytes,
              dataRange: { buffer: dataBuf, offsetBytes: dataOff, sizeBytes: this.mainBytes },
            });
          } else {
            const extraOffsetElements = (inputOffsetBytes / 8) | 0;
            const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
            ensureWithinBindingLimit(this.device, neededBytes, "c2c strided input binding");
            if (input.size < neededBytes) {
              throw new Error(`input buffer too small for strided layout: need ${neededBytes} bytes, have ${input.size}`);
            }

            this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([this.logicalTotal, this.batch, extraOffsetElements, 0]));
            const bg = this.device.createBindGroup({
              layout: this.stridedIn.bgl,
              entries: [
                { binding: 0, resource: { buffer: input, offset: 0, size: neededBytes } },
                { binding: 1, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.stridedIn.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
            pass.end();
          }
        }
      } else if (this.ioEmbed) {
        const viewTotal = prod(this.ioEmbed.viewShape);
        const viewBytesPerBatch = viewTotal * this._bytesPerComplexIO;
        if (!this._largeBatchChunkMode) {
          const viewBytes = viewBytesPerBatch * this.batch;

          // Use direct binding when contiguous; otherwise pack once into scratch.
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, viewBytes);
          let physBuf = inRanges[0].buffer;
          let physOff = inRanges[0].offsetBytes;
          if (inRanges.length > 1) {
            const phys = normalizeToContiguousRanges(scratch, 0, viewBytes)[0];
            this.copier.pack(commandEncoder, inRanges, phys.buffer, phys.offsetBytes);
            physBuf = phys.buffer;
            physOff = phys.offsetBytes;
          }

          this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, viewTotal, this.batch, 0]));
          const bg = this.device.createBindGroup({
            layout: this.ioEmbed.bgl,
            entries: [
              { binding: 0, resource: { buffer: physBuf, offset: physOff, size: viewBytes } },
              { binding: 1, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
              { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioEmbed.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
          pass.end();
        } else {
          const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
          const maxBatchPerChunk = this._resolveLargeChunkBatchCount(
            Math.min(
              Math.floor(this._maxBindBytes / this._bytesPerBatch),
              Math.floor(this._maxBindBytes / viewBytesPerBatch)
            )
          );
          const maxChunkMainBytes = maxBatchPerChunk * this._bytesPerBatch;
          const maxChunkViewBytes = maxBatchPerChunk * viewBytesPerBatch;
          const maxPairBytes = alignBytes(maxChunkViewBytes, storageAlign) + maxChunkMainBytes;
          const chunkBuf = this._ensureLargeChunkBuffer(maxPairBytes);

          for (let b0 = 0; b0 < this.batch; b0 += maxBatchPerChunk) {
            const bCount = Math.min(maxBatchPerChunk, this.batch - b0);
            const chunkMainBytes = bCount * this._bytesPerBatch;
            const chunkViewBytes = bCount * viewBytesPerBatch;
            const chunkComplex = bCount * this.logicalTotal;
            const chunkInputOffset = inputOffsetBytes + b0 * viewBytesPerBatch;
            const chunkDataOffset = dataOff + b0 * this._bytesPerBatch;

            const inRanges = normalizeToContiguousRanges(input, chunkInputOffset, chunkViewBytes);
            const canBindInputDirect = inRanges.length === 1 && inRanges[0].offsetBytes % storageAlign === 0;
            const canBindDataDirect = chunkDataOffset % storageAlign === 0;

            let srcBuf;
            let srcOff;
            if (canBindInputDirect) {
              srcBuf = inRanges[0].buffer;
              srcOff = inRanges[0].offsetBytes;
            } else {
              this.copier.pack(commandEncoder, inRanges, chunkBuf, 0);
              srcBuf = chunkBuf;
              srcOff = 0;
            }

            let dstBuf;
            let dstOff;
            if (canBindDataDirect) {
              dstBuf = dataBuf;
              dstOff = chunkDataOffset;
            } else {
              dstOff = canBindInputDirect ? 0 : alignBytes(chunkViewBytes, storageAlign);
              dstBuf = chunkBuf;
            }

            this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, viewTotal, bCount, 0]));
            const bg = this.device.createBindGroup({
              layout: this.ioEmbed.bgl,
              entries: [
                { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: chunkViewBytes } },
                { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: chunkMainBytes } },
                { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.ioEmbed.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(chunkComplex / this.workgroupSize), 1, 1);
            pass.end();

            if (!canBindDataDirect) {
              commandEncoder.copyBufferToBuffer(chunkBuf, dstOff, dataBuf, chunkDataOffset, chunkMainBytes);
            }
          }
        }
      } else {
        // bring input into stageRange (segmented ok)
        if (this.precision === "f16-storage") {
          const total = this.totalComplex;
          this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([total, 0, 0, 0]));
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, total * 4); // vec2<f16> is 4 bytes
          let srcBuf = inRanges[0].buffer;
          let srcOff = inRanges[0].offsetBytes;
          if (inRanges.length > 1) {
            const packed = normalizeToContiguousRanges(scratch, 0, total * 4)[0];
            this.copier.pack(commandEncoder, inRanges, packed.buffer, packed.offsetBytes);
            srcBuf = packed.buffer;
            srcOff = packed.offsetBytes;
          }
          const bg = this.device.createBindGroup({
            layout: this.f16.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: total * 4 } },
              { binding: 1, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
              { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16.toF32);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(total / this.workgroupSize), 1, 1);
          pass.end();
        } else {
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.mainBytes);
          if (inRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(inRanges[0].buffer, inRanges[0].offsetBytes, dataBuf, dataOff, this.mainBytes);
          } else {
            this.copier.pack(commandEncoder, inRanges, dataBuf, dataOff);
          }
        }
      }
    }

    const runZeroStage = (stage) => {
      if (!stage) return;
      if (!this._largeBatchChunkMode) {
        const bg = this.device.createBindGroup({
          layout: stage.bgl,
          entries: [{ binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } }],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(stage.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
        pass.end();
        return;
      }

      const maxBatchPerChunk = this._resolveLargeChunkBatchCount(this._maxBindBytes / this._bytesPerBatch);
      const maxChunkBytes = maxBatchPerChunk * this._bytesPerBatch;
      const chunkBuf = this._ensureLargeChunkBuffer(maxChunkBytes);

      for (let b0 = 0; b0 < this.batch; b0 += maxBatchPerChunk) {
        const bCount = Math.min(maxBatchPerChunk, this.batch - b0);
        const chunkBytes = bCount * this._bytesPerBatch;
        const chunkComplex = bCount * this.logicalTotal;
        const chunkOffset = dataOff + b0 * this._bytesPerBatch;

        commandEncoder.copyBufferToBuffer(dataBuf, chunkOffset, chunkBuf, 0, chunkBytes);

        const bg = this.device.createBindGroup({
          layout: stage.bgl,
          entries: [{ binding: 0, resource: { buffer: chunkBuf, offset: 0, size: chunkBytes } }],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(stage.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(chunkComplex / this.workgroupSize), 1, 1);
        pass.end();

        commandEncoder.copyBufferToBuffer(chunkBuf, 0, dataBuf, chunkOffset, chunkBytes);
      }
    };

    runZeroStage(this.zeroRead);

    const axisTemp = this._largeBatchChunkMode ? null : scratch;
    // FFT axes in order, with optional axis-1 transpose fast path.
    for (let axis = 0; axis < this.rank; axis++) {
      if (axis === 1 && this.transpose) {
        const tr = this.transpose;
        const trBatch = tr.matrixBatch ?? this.batch;
        const trRange = normalizeToContiguousRanges(transpose, 0, this.mainBytes)[0];
        this.device.queue.writeBuffer(tr.params, 0, new Uint32Array([trBatch, 0, 0, 0]));
        const bg1 = this.device.createBindGroup({
          layout: tr.bgl,
          entries: [
            { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
            { binding: 1, resource: { buffer: trRange.buffer, offset: trRange.offsetBytes, size: this.mainBytes } },
            { binding: 2, resource: { buffer: tr.params, offset: 0, size: 16 } },
          ],
        });
        const pass1 = commandEncoder.beginComputePass();
        pass1.setPipeline(tr.pipelineXY);
        pass1.setBindGroup(0, bg1);
        pass1.dispatchWorkgroups(Math.ceil(tr.Nx / tr.tile), Math.ceil(tr.Ny / tr.tile), trBatch);
        pass1.end();

        this.axis0OnTransposed.exec(commandEncoder, { input: trRange.buffer, inputOffsetBytes: trRange.offsetBytes, batch: this.batch, temp: axisTemp });

        const bg2 = this.device.createBindGroup({
          layout: tr.bgl,
          entries: [
            { binding: 0, resource: { buffer: trRange.buffer, offset: trRange.offsetBytes, size: this.mainBytes } },
            { binding: 1, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
            { binding: 2, resource: { buffer: tr.params, offset: 0, size: 16 } },
          ],
        });
        const pass2 = commandEncoder.beginComputePass();
        pass2.setPipeline(tr.pipelineYX);
        pass2.setBindGroup(0, bg2);
        pass2.dispatchWorkgroups(Math.ceil(tr.Ny / tr.tile), Math.ceil(tr.Nx / tr.tile), trBatch);
        pass2.end();
        continue;
      }

      const kind = this.axisKind[axis];
      if (kind === "mixed") {
        this.axisPlans[axis].exec(commandEncoder, { input: dataBuf, inputOffsetBytes: dataOff, batch: this.batch, temp: axisTemp });
      } else if (kind === "bluestein") {
        this.axisAdvanced[axis].exec(commandEncoder, { dataBuf, dataOffsetBytes: dataOff, axisWork, scratch: axisTemp });
      } else {
        this.axisAdvanced[axis].exec(commandEncoder, { dataBuf, dataOffsetBytes: dataOff, axisWork, scratch: axisTemp });
      }
    }

    // Normalize scale once after full transform
    const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
    if (!this._largeBatchChunkMode && scale !== 1.0) {
      this.device.queue.writeBuffer(this.scale.params, 0, new Uint32Array([this.totalComplex, 0, 0, 0]));
      this.device.queue.writeBuffer(this.scale.params, 16, new Float32Array([scale, 0, 0, 0]));
      const bg = this.device.createBindGroup({
        layout: this.scale.bgl,
        entries: [
          { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
          { binding: 1, resource: { buffer: this.scale.params, offset: 0, size: 32 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.scale.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
      pass.end();
    }

    runZeroStage(this.zeroWrite);

    const outTarget = this.inPlace ? input : output;
    const outOffset = this.inPlace ? inputOffsetBytes : outputOffsetBytes;

    if (this._usesStridedOutput) {
      if (this._needsOutputMapping) {
        this._extractStridedOutputOutOfCore(commandEncoder, {
          dataRange: { buffer: dataBuf, offsetBytes: dataOff, sizeBytes: this.mainBytes },
          output: outTarget,
          outputOffsetBytes: outOffset,
        });
        return;
      }
      if (this._largeBatchChunkMode || !isGpuBuffer(outTarget)) {
        this._extractStridedOutputOutOfCore(commandEncoder, {
          dataRange: { buffer: dataBuf, offsetBytes: dataOff, sizeBytes: this.mainBytes },
          output: outTarget,
          outputOffsetBytes: outOffset,
        });
        return;
      }
      const extraOffsetElements = (outOffset / 8) | 0;
      const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
      ensureWithinBindingLimit(this.device, neededBytes, "c2c strided output binding");
      if (outTarget.size < neededBytes) {
        throw new Error(`output buffer too small for strided layout: need ${neededBytes} bytes, have ${outTarget.size}`);
      }

      this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([this.logicalTotal, this.batch, extraOffsetElements, 0]));
      const bg = this.device.createBindGroup({
        layout: this.stridedOut.bgl,
        entries: [
          { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
          { binding: 1, resource: { buffer: outTarget, offset: 0, size: neededBytes } },
          { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.stridedOut.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
      pass.end();
      return;
    }

    // Optional output view mapping. If present, write directly to the final output when contiguous;
    // otherwise, stage once and scatter.
    if (this.ioExtract) {
      const viewTotal = prod(this.ioExtract.viewShape);
      const outBytesPerBatch = viewTotal * this._bytesPerComplexIO;
      if (!this._largeBatchChunkMode) {
        const outBytes = outBytesPerBatch * this.batch;
        const outRanges = normalizeToContiguousRanges(outTarget, outOffset, outBytes);

        // Contiguous output: write directly to preserve clearOutside=false semantics.
        if (outRanges.length === 1) {
          this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, viewTotal, this.batch, 0]));
          const bg = this.device.createBindGroup({
            layout: this.ioExtract.bgl,
            entries: [
              { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
              { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: outBytes } },
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

        const stageOut = normalizeToContiguousRanges(scratch, 0, outBytes)[0];
        if (!this.io.output.clearOutside) {
          this.copier.pack(commandEncoder, outRanges, stageOut.buffer, stageOut.offsetBytes);
        }

        this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, viewTotal, this.batch, 0]));
        const bg = this.device.createBindGroup({
          layout: this.ioExtract.bgl,
          entries: [
            { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
            { binding: 1, resource: { buffer: stageOut.buffer, offset: stageOut.offsetBytes, size: outBytes } },
            { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.ioExtract.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
        pass.end();

        this.copier.unpack(commandEncoder, stageOut.buffer, stageOut.offsetBytes, outRanges);
      } else {
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxBatchPerChunk = this._resolveLargeChunkBatchCount(
          Math.min(
            Math.floor(this._maxBindBytes / this._bytesPerBatch),
            Math.floor(this._maxBindBytes / outBytesPerBatch)
          )
        );
        const maxChunkMainBytes = maxBatchPerChunk * this._bytesPerBatch;
        const maxChunkViewBytes = maxBatchPerChunk * outBytesPerBatch;
        const maxPairBytes = alignBytes(maxChunkMainBytes, storageAlign) + maxChunkViewBytes;
        const chunkBuf = this._ensureLargeChunkBuffer(maxPairBytes);

        for (let b0 = 0; b0 < this.batch; b0 += maxBatchPerChunk) {
          const bCount = Math.min(maxBatchPerChunk, this.batch - b0);
          const chunkMainBytes = bCount * this._bytesPerBatch;
          const chunkViewBytes = bCount * outBytesPerBatch;
          const chunkViewComplex = bCount * viewTotal;
          const chunkDataOffset = dataOff + b0 * this._bytesPerBatch;
          const chunkOutOffset = outOffset + b0 * outBytesPerBatch;
          const outRanges = normalizeToContiguousRanges(outTarget, chunkOutOffset, chunkViewBytes);

          const canBindDataDirect = chunkDataOffset % storageAlign === 0;
          const canBindOutputDirect = outRanges.length === 1 && outRanges[0].offsetBytes % storageAlign === 0;

          let srcBuf;
          let srcOff;
          if (canBindDataDirect) {
            srcBuf = dataBuf;
            srcOff = chunkDataOffset;
          } else {
            commandEncoder.copyBufferToBuffer(dataBuf, chunkDataOffset, chunkBuf, 0, chunkMainBytes);
            srcBuf = chunkBuf;
            srcOff = 0;
          }

          let dstBuf;
          let dstOff;
          if (canBindOutputDirect) {
            dstBuf = outRanges[0].buffer;
            dstOff = outRanges[0].offsetBytes;
          } else {
            dstOff = canBindDataDirect ? 0 : alignBytes(chunkMainBytes, storageAlign);
            dstBuf = chunkBuf;
            if (!this.io.output.clearOutside) {
              if (outRanges.length === 1) {
                commandEncoder.copyBufferToBuffer(outRanges[0].buffer, outRanges[0].offsetBytes, chunkBuf, dstOff, chunkViewBytes);
              } else {
                this.copier.pack(commandEncoder, outRanges, chunkBuf, dstOff);
              }
            }
          }

          this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, viewTotal, bCount, 0]));
          const bg = this.device.createBindGroup({
            layout: this.ioExtract.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: chunkMainBytes } },
              { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: chunkViewBytes } },
              { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioExtract.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(chunkViewComplex / this.workgroupSize), 1, 1);
          pass.end();

          if (!canBindOutputDirect) {
            if (outRanges.length === 1) {
              commandEncoder.copyBufferToBuffer(chunkBuf, dstOff, outRanges[0].buffer, outRanges[0].offsetBytes, chunkViewBytes);
            } else {
              this.copier.unpack(commandEncoder, chunkBuf, dstOff, outRanges);
            }
          }
        }
      }
      return;
    }

    // No ioView mapping: write logical output in requested precision.
    if (this.precision === "f16-storage") {
      const total = this.totalComplex;
      const outBytesF16 = total * 4;
      this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([total, 0, 0, 0]));
      const outRanges = normalizeToContiguousRanges(outTarget, outOffset, outBytesF16);
      if (outRanges.length === 1) {
        const bg = this.device.createBindGroup({
          layout: this.f16.bgl,
          entries: [
            { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
            { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: outBytesF16 } },
            { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.f16.toF16);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(total / this.workgroupSize), 1, 1);
        pass.end();
        return;
      }
      const packed = normalizeToContiguousRanges(scratch, 0, outBytesF16)[0];
      const bg = this.device.createBindGroup({
        layout: this.f16.bgl,
        entries: [
          { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
          { binding: 1, resource: { buffer: packed.buffer, offset: packed.offsetBytes, size: outBytesF16 } },
          { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
        ],
      });
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.f16.toF16);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(total / this.workgroupSize), 1, 1);
      pass.end();
      this.copier.unpack(commandEncoder, packed.buffer, packed.offsetBytes, outRanges);
      return;
    }

    const outRanges = normalizeToContiguousRanges(outTarget, outOffset, this.mainBytes);
    if (outRanges.length === 1) {
      if (outRanges[0].buffer !== dataBuf || outRanges[0].offsetBytes !== dataOff) {
        commandEncoder.copyBufferToBuffer(dataBuf, dataOff, outRanges[0].buffer, outRanges[0].offsetBytes, this.mainBytes);
      }
    } else {
      this.copier.unpack(commandEncoder, dataBuf, dataOff, outRanges);
    }
  }
}
