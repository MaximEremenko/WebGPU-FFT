// Copyright (c) 2026 Maksim Eremenko

import { BasePlan } from "../base_plan.js";
import { createInternalArena, viewFromArena } from "../workspace.js";
import { normalizeToContiguousRanges } from "../segmented_io.js";
import { assertOneOf, isPositiveInt, prod, alignBytes, getBufferByteLength, buffersAlias } from "../common.js";
import { mergeLargeRouteMetadata, resolveLargeRoutingPolicy } from "../large_policy.js";
import { resolveLayoutSemantics } from "../layout_semantics.js";
import { normalizeZeroPad } from "../zero_pad.js";
import { createTensorDescriptor, requiredBytesForBatchRange } from "../tensor_descriptor.js";

import { C2CPlan } from "./c2c.js";
import { generatePointwiseMulSegmentWGSL } from "../../kernels/fft_conv.js";
import { generateExtractComplexWGSL } from "../../kernels/ioview.js";
import { BufferView } from "../../utils/buffer_view.js";
import {
  coordsFromLinear as tensorCoordsFromLinear,
  linearFromCoordsShape as tensorLinearFromCoordsShape,
  contiguousStrides as tensorContiguousStrides,
  linearFromCoords as tensorLinearFromCoords,
} from "../tensor_descriptor.js";

function copyContiguousToRanges(copier, commandEncoder, srcBuffer, srcOffsetBytes, byteLength, outRanges, maxBindBytes = Infinity) {
  if (outRanges.length === 1) {
    if (outRanges[0].buffer !== srcBuffer || outRanges[0].offsetBytes !== srcOffsetBytes) {
      commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, outRanges[0].buffer, outRanges[0].offsetBytes, byteLength);
    }
  } else {
    if (byteLength > maxBindBytes) {
      let src = srcOffsetBytes;
      for (const r of outRanges) {
        commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
        src += r.sizeBytes;
      }
    } else {
      copier.unpack(commandEncoder, srcBuffer, srcOffsetBytes, outRanges);
    }
  }
}

function copyRangesToRanges(copier, commandEncoder, srcRanges, dstRanges, byteLength, maxBindBytes = Infinity) {
  if (srcRanges.length === 1 && dstRanges.length === 1) {
    if (
      srcRanges[0].buffer !== dstRanges[0].buffer ||
      srcRanges[0].offsetBytes !== dstRanges[0].offsetBytes
    ) {
      commandEncoder.copyBufferToBuffer(
        srcRanges[0].buffer,
        srcRanges[0].offsetBytes,
        dstRanges[0].buffer,
        dstRanges[0].offsetBytes,
        byteLength
      );
    }
    return;
  }
  if (srcRanges.length === 1 && dstRanges.length > 1 && byteLength <= maxBindBytes) {
    copier.unpack(commandEncoder, srcRanges[0].buffer, srcRanges[0].offsetBytes, dstRanges);
    return;
  }
  if (srcRanges.length > 1 && dstRanges.length === 1 && byteLength <= maxBindBytes) {
    copier.pack(commandEncoder, srcRanges, dstRanges[0].buffer, dstRanges[0].offsetBytes);
    return;
  }

  // General segmented-to-segmented fallback: piecewise linear copies.
  let si = 0;
  let di = 0;
  let soff = 0;
  let doff = 0;
  let remaining = byteLength;
  while (remaining > 0) {
    const s = srcRanges[si];
    const d = dstRanges[di];
    const sAvail = s.sizeBytes - soff;
    const dAvail = d.sizeBytes - doff;
    const take = Math.min(remaining, sAvail, dAvail);
    commandEncoder.copyBufferToBuffer(
      s.buffer,
      s.offsetBytes + soff,
      d.buffer,
      d.offsetBytes + doff,
      take
    );
    remaining -= take;
    soff += take;
    doff += take;
    if (soff === s.sizeBytes) {
      si += 1;
      soff = 0;
    }
    if (doff === d.sizeBytes) {
      di += 1;
      doff = 0;
    }
  }
}

function arraysEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function hasOwn(obj, key) {
  return !!obj && Object.prototype.hasOwnProperty.call(obj, key);
}

function parseOptionalPositiveInt(v, name) {
  if (v == null) return null;
  if (!Number.isSafeInteger(v) || v <= 0) throw new Error(`${name} must be a positive integer`);
  return v;
}

function parseOptionalNonNegativeInt(v, name) {
  if (v == null) return null;
  if (!Number.isSafeInteger(v) || v < 0) throw new Error(`${name} must be a non-negative integer`);
  return v;
}

function checkedSafeNonNegativeInt(v, name) {
  if (!Number.isSafeInteger(v) || v < 0) throw new Error(`${name} must stay within non-negative safe integer range`);
  return v;
}

function sideFieldName(side, suffix) {
  if (side === "input") return `input${suffix}`;
  return `output${suffix}`;
}

function hasExplicitSideLayout(layout, side) {
  return (
    hasOwn(layout, sideFieldName(side, "Strides")) ||
    hasOwn(layout, sideFieldName(side, "OffsetElements")) ||
    hasOwn(layout, sideFieldName(side, "BatchStrideElements")) ||
    hasOwn(layout, "strides") ||
    hasOwn(layout, "offsetElements") ||
    hasOwn(layout, "batchStrideElements")
  );
}

function normalizeChannelPolicySide({
  desc,
  sidePath,
  defaultChannelStrideElements,
  allowKernelStep = false,
  kernelCount = 1,
}) {
  if (desc == null) return null;
  if (typeof desc !== "object" || Array.isArray(desc)) throw new Error(`${sidePath} must be an object`);

  const channels = parseOptionalPositiveInt(desc.channels ?? null, `${sidePath}.channels`);
  if (channels == null) throw new Error(`${sidePath}.channels is required`);

  const channelIndex = parseOptionalNonNegativeInt(desc.channelIndex ?? null, `${sidePath}.channelIndex`) ?? 0;
  if (channelIndex >= channels) {
    throw new Error(`${sidePath}.channelIndex (${channelIndex}) must be < ${sidePath}.channels (${channels})`);
  }

  const channelStrideElements =
    parseOptionalPositiveInt(desc.channelStrideElements ?? null, `${sidePath}.channelStrideElements`) ??
    defaultChannelStrideElements;
  if (channelStrideElements < defaultChannelStrideElements) {
    throw new Error(`${sidePath}.channelStrideElements must be >= logical span (${defaultChannelStrideElements})`);
  }

  const offsetElements = parseOptionalNonNegativeInt(desc.offsetElements ?? null, `${sidePath}.offsetElements`) ?? 0;
  const defaultBatchStride = checkedSafeNonNegativeInt(
    channels * channelStrideElements,
    `${sidePath}.batchStrideElements`
  );
  const batchStrideElements =
    parseOptionalNonNegativeInt(desc.batchStrideElements ?? null, `${sidePath}.batchStrideElements`) ?? defaultBatchStride;
  if (batchStrideElements < defaultBatchStride) {
    throw new Error(`${sidePath}.batchStrideElements must be >= channels*channelStrideElements (${defaultBatchStride})`);
  }

  const kernelStepChannels = allowKernelStep
    ? parseOptionalPositiveInt(desc.kernelStepChannels ?? null, `${sidePath}.kernelStepChannels`) ?? 1
    : 1;
  if (allowKernelStep && kernelCount > 1) {
    const maxChannelIndex = checkedSafeNonNegativeInt(
      channelIndex + (kernelCount - 1) * kernelStepChannels,
      `${sidePath}.kernelStepChannels`
    );
    if (maxChannelIndex >= channels) {
      throw new Error(
        `${sidePath} does not fit kernelCount=${kernelCount}: max channel index ${maxChannelIndex} exceeds channels=${channels} ` +
          `(channelIndex=${channelIndex}, kernelStepChannels=${kernelStepChannels})`
      );
    }
  }

  return {
    channels,
    channelIndex,
    channelStrideElements,
    batchStrideElements,
    offsetElements,
    kernelStepChannels,
    layoutDesc: {
      channels,
      channelIndex,
      channelStrideElements,
      batchStrideElements,
      offsetElements,
    },
  };
}

function resolveFftConvLayoutWithChannelPolicy({
  layout,
  channelPolicy,
  kernelCount,
  inputLogicalTotal,
  outputLogicalTotal,
}) {
  if (channelPolicy == null) {
    return {
      layout: layout ?? {},
      outputKernelStrideElements: 0,
      usesChannelPolicy: false,
    };
  }
  if (typeof channelPolicy !== "object" || Array.isArray(channelPolicy)) {
    throw new Error("fftConv.channelPolicy must be an object");
  }

  const inPolicyPresent = hasOwn(channelPolicy, "input") && channelPolicy.input != null;
  const outPolicyPresent = hasOwn(channelPolicy, "output") && channelPolicy.output != null;
  if (!inPolicyPresent && !outPolicyPresent) {
    throw new Error("fftConv.channelPolicy must provide input and/or output descriptors");
  }

  if (layout?.whdcn != null) {
    throw new Error("fftConv.channelPolicy cannot be combined with layout.whdcn");
  }
  if (inPolicyPresent && hasExplicitSideLayout(layout ?? {}, "input")) {
    throw new Error("fftConv.channelPolicy.input cannot be combined with explicit input stride fields");
  }
  if (outPolicyPresent && hasExplicitSideLayout(layout ?? {}, "output")) {
    throw new Error("fftConv.channelPolicy.output cannot be combined with explicit output stride fields");
  }

  const inputPolicy = normalizeChannelPolicySide({
    desc: channelPolicy.input ?? null,
    sidePath: "fftConv.channelPolicy.input",
    defaultChannelStrideElements: inputLogicalTotal,
    allowKernelStep: false,
    kernelCount,
  });
  const outputPolicy = normalizeChannelPolicySide({
    desc: channelPolicy.output ?? null,
    sidePath: "fftConv.channelPolicy.output",
    defaultChannelStrideElements: outputLogicalTotal,
    allowKernelStep: true,
    kernelCount,
  });

  const outputKernelStrideElements =
    outputPolicy && kernelCount > 1
      ? checkedSafeNonNegativeInt(
          outputPolicy.channelStrideElements * outputPolicy.kernelStepChannels,
          "fftConv.channelPolicy.output.kernelStepChannels"
        )
      : 0;

  return {
    layout: {
      ...(layout ?? {}),
      whdcn: {
        ...(inputPolicy ? { input: inputPolicy.layoutDesc } : {}),
        ...(outputPolicy ? { output: outputPolicy.layoutDesc } : {}),
      },
    },
    outputKernelStrideElements,
    usesChannelPolicy: true,
  };
}

function normalizeFftConvTuning(tuning) {
  if (tuning == null) {
    return {
      pointwiseChunkElements: null,
      extractCopyChunkElements: null,
    };
  }
  if (typeof tuning !== "object" || Array.isArray(tuning)) {
    throw new Error("fftConv.tuning must be an object when provided");
  }
  const pointwiseChunkElements = parseOptionalPositiveInt(
    tuning.pointwiseChunkElements ?? null,
    "fftConv.tuning.pointwiseChunkElements"
  );
  const extractCopyChunkElements = parseOptionalPositiveInt(
    tuning.extractCopyChunkElements ?? null,
    "fftConv.tuning.extractCopyChunkElements"
  );
  return {
    pointwiseChunkElements,
    extractCopyChunkElements,
  };
}

export class FftConvPlan extends BasePlan {
  constructor(device, opts) {
    super(device, opts);
    const {
      shape,
      batch = 1,
      inPlace = false,
      layout = { interleavedComplex: true },
      precision = "f32",
      fftConv = null,
      zeroPad = null,
    } = opts ?? {};

    if (!Array.isArray(shape) || shape.length < 1) {
      throw new Error(`fftconv shape must be rank >= 1; got ${JSON.stringify(shape)}`);
    }
    if (!shape.every(isPositiveInt)) throw new Error("fftconv shape must be positive ints");
    if (!Number.isInteger(batch) || batch <= 0) throw new Error(`batch must be positive int; got ${batch}`);
    if (inPlace) throw new Error("fftconv inPlace=true is not supported in current implementation");
    if (layout?.interleavedComplex !== true) throw new Error("fftconv requires layout.interleavedComplex=true");
    if (precision !== "f32") throw new Error('fftconv supports precision:"f32" only in current implementation');

    const mode = fftConv?.mode ?? "convolution";
    assertOneOf(mode, ["convolution", "correlation"], "fftConv.mode");
    const boundary = fftConv?.boundary ?? "circular";
    assertOneOf(boundary, ["circular", "linear-full", "linear-same", "linear-valid"], "fftConv.boundary");
    const kernelCount = fftConv?.kernelCount ?? 1;
    if (!Number.isInteger(kernelCount) || kernelCount <= 0) {
      throw new Error(`fftConv.kernelCount must be a positive integer; got ${kernelCount}`);
    }
    const outputLayout = fftConv?.outputLayout ?? "kernel-major";
    assertOneOf(outputLayout, ["kernel-major", "batch-major"], "fftConv.outputLayout");

    const rank = shape.length;
    const kernelShape = fftConv?.kernelShape ?? shape;
    if (!Array.isArray(kernelShape) || kernelShape.length !== rank || !kernelShape.every(isPositiveInt)) {
      throw new Error(`fftConv.kernelShape must be an array of ${rank} positive ints`);
    }
    if (boundary === "circular") {
      for (let d = 0; d < rank; d++) {
        if (kernelShape[d] > shape[d]) {
          throw new Error(`fftConv.kernelShape[${d}] must be <= shape[${d}] when fftConv.boundary="circular"`);
        }
      }
    }

    const fftShape = boundary === "circular" ? shape.slice() : shape.map((n, d) => n + kernelShape[d] - 1);
    let outputShape;
    let outputOffset;
    if (boundary === "circular") {
      outputShape = shape.slice();
      outputOffset = new Array(rank).fill(0);
    } else if (boundary === "linear-full") {
      outputShape = fftShape.slice();
      outputOffset = new Array(rank).fill(0);
    } else if (boundary === "linear-same") {
      outputShape = shape.slice();
      outputOffset = kernelShape.map((n) => Math.floor((n - 1) / 2));
    } else {
      outputShape = shape.map((n, d) => n - kernelShape[d] + 1);
      for (let d = 0; d < rank; d++) {
        if (outputShape[d] <= 0) {
          throw new Error(`fftConv.boundary="linear-valid" requires kernelShape[${d}] <= shape[${d}]`);
        }
      }
      outputOffset = kernelShape.map((n) => n - 1);
    }

    this.rank = rank;
    this.inputShape = shape.slice();
    this.shape = fftShape.slice();
    this.kernelShape = kernelShape.slice();
    this.outputShape = outputShape;
    this.outputOffset = outputOffset;
    this.batch = batch;
    this.mode = mode;
    this.boundary = boundary;
    this.kernelCount = kernelCount | 0;
    this.outputLayout = outputLayout;
    this.zeroPad = normalizeZeroPad(this.rank, this.shape, zeroPad ?? null, "zeroPad");
    this._fftConvTuning = normalizeFftConvTuning(fftConv?.tuning ?? null);

    this.inputLogicalTotal = prod(this.inputShape);
    this.logicalTotal = prod(this.shape);
    this.kernelLogicalTotal = prod(this.kernelShape);
    this.outputLogicalTotal = prod(this.outputShape);

    this.totalComplex = this.logicalTotal * this.batch;
    this.mainBytes = this.totalComplex * 8;
    this.bytesPerBatch = this.logicalTotal * 8;
    this.kernelBytes = this.logicalTotal * 8;
    this.kernelInputBytes = this.kernelLogicalTotal * 8;
    this.inputBytes = this.inputLogicalTotal * this.batch * 8;
    this.outputBytesPerBatch = this.outputLogicalTotal * 8;
    this.outputBytesPerKernel = this.outputBytesPerBatch * this.batch;
    this.totalOutputComplex = this.outputLogicalTotal * this.batch * this.kernelCount;
    this.totalOutputBytes = this.totalOutputComplex * 8;
    const explicitOutputKernelStrideElements =
      parseOptionalPositiveInt(fftConv?.outputKernelStrideElements ?? null, "fftConv.outputKernelStrideElements") ?? 0;
    const channelPolicyResolved = resolveFftConvLayoutWithChannelPolicy({
      layout,
      channelPolicy: fftConv?.channelPolicy ?? null,
      kernelCount: this.kernelCount,
      inputLogicalTotal: this.inputLogicalTotal,
      outputLogicalTotal: this.outputLogicalTotal,
    });
    const policyOutputKernelStrideElements = channelPolicyResolved.outputKernelStrideElements ?? 0;
    if (
      explicitOutputKernelStrideElements > 0 &&
      policyOutputKernelStrideElements > 0 &&
      explicitOutputKernelStrideElements !== policyOutputKernelStrideElements
    ) {
      throw new Error(
        "fftConv.outputKernelStrideElements conflicts with fftConv.channelPolicy.output kernel step mapping"
      );
    }
    this._usesFftConvChannelPolicy = !!channelPolicyResolved.usesChannelPolicy;
    this._stridedOutputKernelStrideElements =
      explicitOutputKernelStrideElements || policyOutputKernelStrideElements || 0;
    const resolvedLayout = resolveLayoutSemantics({
      layout: channelPolicyResolved.layout,
      rank: this.rank,
      inputShape: this.inputShape,
      outputShape: this.outputShape,
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
          shape: this.inputShape,
          strides: this._inputStrides,
          offsetElements: this._inputOffsetElements,
          batchStrideElements: this._inputBatchStrideElements,
          name: "fftconv.input",
        })
      : null;
    this._outputTensorDesc = this._usesStridedOutput
      ? createTensorDescriptor({
          shape: this.outputShape,
          strides: this._outputStrides,
          offsetElements: this._outputOffsetElements,
          batchStrideElements: this._outputBatchStrideElements,
          name: "fftconv.output",
        })
      : null;

    const largePolicy = resolveLargeRoutingPolicy({
      device,
      tuning: opts?.tuning ?? null,
      requiredBindingBytes: [this.mainBytes, this.kernelBytes, this.inputBytes, this.totalOutputBytes],
      lineBytes: this.shape.map((n) => n * 8),
      precision: "f32",
    });
    this._maxBindBytes = largePolicy.maxBindBytes;
    this._maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
    this._largeMode = largePolicy.needsLargeMode;
    this._largeRouteMode = largePolicy.routeMode;
    this._largeRouteReasons = largePolicy.reasonCodes;
    this._largeRouteAttempts = largePolicy.attemptedRoutes;
    const pointwiseMaxElems = Math.max(1, Math.floor(this._maxBindBytes / 8));
    if (
      this._fftConvTuning.pointwiseChunkElements != null &&
      this._fftConvTuning.pointwiseChunkElements > pointwiseMaxElems
    ) {
      throw new Error(
        `fftConv.tuning.pointwiseChunkElements=${this._fftConvTuning.pointwiseChunkElements} exceeds max supported ${pointwiseMaxElems} ` +
          `(maxStorageBufferBindingSize=${this._maxBindBytes})`
      );
    }
    this._pointwiseChunkElems = this._fftConvTuning.pointwiseChunkElements ?? pointwiseMaxElems;
    this._extractCopyChunkElems = this._fftConvTuning.extractCopyChunkElements ?? 1;
    this._batchSlicedExecution = false;
    if (this.mainBytes > this._maxBufferSize) {
      if (this.bytesPerBatch > this._maxBufferSize) {
        throw new Error(
          `fftconv requires ${this.mainBytes} bytes for batch=${this.batch}, and one batch requires ${this.bytesPerBatch} bytes > ` +
            `device.limits.maxBufferSize=${this._maxBufferSize}. ` +
            `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
        );
      }
      this._batchSlicedExecution = true;
    }
    if (this.kernelBytes > this._maxBufferSize) {
      throw new Error(
        `fftconv kernel workspace requires ${this.kernelBytes} bytes, exceeding device.limits.maxBufferSize=${this._maxBufferSize}. ` +
          `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
      );
    }
    if (!this._usesStridedOutput && this._stridedOutputKernelStrideElements > 0) {
      throw new Error("fftConv.outputKernelStrideElements requires strided/whdcn output layout");
    }
    if (this._usesStridedOutput && this.kernelCount > 1 && this._stridedOutputKernelStrideElements <= 0) {
      throw new Error(
        "fftconv multi-kernel strided/whdcn output requires fftConv.channelPolicy.output (with kernelStepChannels) " +
          "or fftConv.outputKernelStrideElements"
      );
    }

    this._needsInputEmbed = !arraysEqual(this.inputShape, this.shape);
    this._needsKernelEmbed = !arraysEqual(this.kernelShape, this.shape);
    this._needsOutputExtract =
      !arraysEqual(this.outputShape, this.shape) || this.outputOffset.some((x) => x !== 0);

    this.kernelBuffer = device.createBuffer({
      size: this.kernelBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._kernelEmbedInputBuffer = this._needsKernelEmbed
      ? device.createBuffer({
          size: this.kernelInputBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
      : null;
    this._kernelUploadBuffer = null;
    this._kernelUploadBytes = 0;
    this._retiredKernelUploadBuffers = [];

    const fftBatch = this._batchSlicedExecution ? 1 : this.batch;
    this.fftData = new C2CPlan(device, {
      shape: this.shape,
      direction: "forward",
      batch: fftBatch,
      inPlace: !this._needsInputEmbed,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: this._needsInputEmbed ? { input: { shape: this.inputShape, offset: new Array(this.rank).fill(0) } } : { input: null, output: null },
      zeroPad: { read: this.zeroPad.read, write: null },
      tuning: opts?.tuning ?? null,
    });
    this.fftKernel = new C2CPlan(device, {
      shape: this.shape,
      direction: "forward",
      batch: 1,
      inPlace: !this._needsKernelEmbed,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: this._needsKernelEmbed ? { input: { shape: this.kernelShape, offset: new Array(this.rank).fill(0) } } : { input: null, output: null },
      tuning: opts?.tuning ?? null,
    });
    this.ifftData = new C2CPlan(device, {
      shape: this.shape,
      direction: "inverse",
      batch: fftBatch,
      inPlace: true,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      ioView: { input: null, output: null },
      zeroPad: { read: null, write: this.zeroPad.write },
      tuning: opts?.tuning ?? null,
    });
    const mergedRoute = mergeLargeRouteMetadata([
      {
        routeMode: this._largeRouteMode,
        reasonCodes: this._largeRouteReasons,
        attemptedRoutes: this._largeRouteAttempts,
      },
      {
        routeMode: this.fftData?._largeRouteMode,
        reasonCodes: this.fftData?._largeRouteReasons,
        attemptedRoutes: this.fftData?._largeRouteAttempts,
      },
      {
        routeMode: this.fftKernel?._largeRouteMode,
        reasonCodes: this.fftKernel?._largeRouteReasons,
        attemptedRoutes: this.fftKernel?._largeRouteAttempts,
      },
      {
        routeMode: this.ifftData?._largeRouteMode,
        reasonCodes: this.ifftData?._largeRouteReasons,
        attemptedRoutes: this.ifftData?._largeRouteAttempts,
      },
    ]);
    this._largeRouteMode = mergedRoute.routeMode;
    this._largeRouteReasons = mergedRoute.reasonCodes;
    this._largeRouteAttempts = mergedRoute.attemptedRoutes;
    this.zeroRead = this.fftData?.zeroRead ?? null;
    this.zeroWrite = this.ifftData?.zeroWrite ?? null;

    this.pointwise = (() => {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generatePointwiseMulSegmentWGSL({
        correlate: this.mode === "correlation",
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      return { bgl, pl, pipeline };
    })();
    this._pointwiseParamsBuffer = null;
    this._pointwiseParamsBytes = 0;
    this._pointwiseParamStride = alignBytes(16, device.limits?.minUniformBufferOffsetAlignment ?? 256);
    this._retiredPointwiseParamsBuffers = [];

    this.outputExtract = null;
    if (this._needsOutputExtract) {
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const code = generateExtractComplexWGSL({
        rank: this.rank,
        logicalDims: this.shape,
        viewDims: this.outputShape,
        offset: this.outputOffset,
        clearOutside: true,
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: pl });
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.outputExtract = { bgl, pl, pipeline, params };
    }

    const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    this._dataBytesPerRun = this._batchSlicedExecution ? this.bytesPerBatch : this.mainBytes;
    this._outputBytesPerRun = this._batchSlicedExecution ? this.outputBytesPerBatch : this.outputBytesPerKernel;
    this._kernelEmbedScratchBytes = this._needsKernelEmbed ? this.kernelInputBytes : 0;
    this._outputExtractScratchBytes = this._needsOutputExtract ? this.outputBytesPerBatch : 0;
    this._subplanScratchBytes = Math.max(
      this.fftData.getWorkspaceSizeBytes(),
      this.fftKernel.getWorkspaceSizeBytes(),
      this.ifftData.getWorkspaceSizeBytes(),
      this._kernelEmbedScratchBytes,
      this._outputExtractScratchBytes
    );
    this._stridedInputStageBytesPerRun = this._usesStridedInput
      ? (this._batchSlicedExecution ? this.inputLogicalTotal * 8 : this.inputBytes)
      : 0;
    this._stridedOutputStageBytesPerRun =
      this._usesStridedOutput && this._needsOutputExtract ? this._outputBytesPerRun : 0;
    let scratchOff = 0;
    this._subplanScratchOffset = 0;
    scratchOff += this._subplanScratchBytes;
    if (scratchOff) scratchOff = alignBytes(scratchOff, storageAlign);
    this._stridedInputStageOffset = 0;
    if (this._stridedInputStageBytesPerRun) {
      this._stridedInputStageOffset = scratchOff;
      scratchOff += this._stridedInputStageBytesPerRun;
      scratchOff = alignBytes(scratchOff, storageAlign);
    }
    this._stridedOutputStageOffset = 0;
    if (this._stridedOutputStageBytesPerRun) {
      this._stridedOutputStageOffset = scratchOff;
      scratchOff += this._stridedOutputStageBytesPerRun;
      scratchOff = alignBytes(scratchOff, storageAlign);
    }
    this.scratchBytes = scratchOff;

    let off = 0;
    this.dataOffset = 0;
    off += this._dataBytesPerRun;
    off = alignBytes(off, storageAlign);
    this.scratchOffset = off;
    off += this.scratchBytes;
    this.workspaceBytes = off;
    this._splitWorkspace = null;
    const requireDisjointWorkspaceBuffers =
      !!this.fftData?._largeBatchChunkMode ||
      !!this.fftKernel?._largeBatchChunkMode ||
      !!this.ifftData?._largeBatchChunkMode ||
      (this._usesStridedOutput && this._needsOutputExtract);
    if (!requireDisjointWorkspaceBuffers && this.workspaceBytes <= this._maxBufferSize) {
      this._arena = createInternalArena(device, this.workspaceBytes);
    } else {
      if (this._dataBytesPerRun > this._maxBufferSize) {
        throw new Error(
          `fftconv split workspace cannot allocate data buffer: ${this._dataBytesPerRun} bytes exceeds ` +
            `device.limits.maxBufferSize=${this._maxBufferSize}`
        );
      }
      if (this.scratchBytes > this._maxBufferSize) {
        throw new Error(
          `fftconv split workspace cannot allocate scratch buffer: ${this.scratchBytes} bytes exceeds ` +
            `device.limits.maxBufferSize=${this._maxBufferSize}`
        );
      }
      this._arena = null;
      this._splitWorkspace = {
        data: createInternalArena(device, this._dataBytesPerRun),
        scratch: this.scratchBytes ? createInternalArena(device, this.scratchBytes) : null,
      };
    }
  }

  getWorkspaceSizeBytes() {
    return this.workspaceBytes;
  }

  _ensureKernelUploadBuffer(bytes) {
    if (this._kernelUploadBuffer && this._kernelUploadBytes >= bytes) return;
    if (bytes > this._maxBufferSize) {
      throw new Error(
        `fftconv kernel upload staging requires ${bytes} bytes, exceeding device.limits.maxBufferSize=${this._maxBufferSize}`
      );
    }
    if (this._kernelUploadBuffer) this._retiredKernelUploadBuffers.push(this._kernelUploadBuffer);
    this._kernelUploadBuffer = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._kernelUploadBytes = bytes;
  }

  _ensurePointwiseParamsBuffer(bytes) {
    if (this._pointwiseParamsBuffer && this._pointwiseParamsBytes >= bytes) return this._pointwiseParamsBuffer;
    if (bytes > this._maxBufferSize) {
      throw new Error(
        `fftconv pointwise chunk params require ${bytes} bytes, exceeding device.limits.maxBufferSize=${this._maxBufferSize}`
      );
    }
    if (this._pointwiseParamsBuffer) this._retiredPointwiseParamsBuffers.push(this._pointwiseParamsBuffer);
    this._pointwiseParamsBuffer = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._pointwiseParamsBytes = bytes;
    return this._pointwiseParamsBuffer;
  }

  _resolveWorkspaceViews(temp) {
    const arena = temp ?? this._arena;
    if (arena) {
      if (getBufferByteLength(arena) < this.workspaceBytes) {
        throw new Error(`temp too small: need ${this.workspaceBytes} bytes`);
      }
      return {
        dataView: viewFromArena(arena, this.dataOffset, this._dataBytesPerRun),
        scratchView: this.scratchBytes ? viewFromArena(arena, this.scratchOffset, this.scratchBytes) : null,
      };
    }
    if (this._splitWorkspace) {
      return {
        dataView: viewFromArena(this._splitWorkspace.data, 0, this._dataBytesPerRun),
        scratchView: this.scratchBytes ? viewFromArena(this._splitWorkspace.scratch, 0, this.scratchBytes) : null,
      };
    }
    throw new Error("No workspace buffer");
  }

  _sliceView(view, offsetBytes, lengthBytes) {
    if (!lengthBytes) return null;
    const ranges = normalizeToContiguousRanges(view, offsetBytes, lengthBytes);
    return new BufferView({
      segments: ranges.map((r) => ({
        buffer: r.buffer,
        offsetBytes: r.offsetBytes,
        sizeBytes: r.sizeBytes,
      })),
      logicalByteOffset: 0,
      lengthBytes,
    });
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
    copyRangesToRanges(this.copier, commandEncoder, srcRanges, dstRanges, bytes, this._maxBindBytes);
  }

  _shapeStrides(shape) {
    return tensorContiguousStrides(shape);
  }

  _linearFromCoordsStrides(coords, strides) {
    return tensorLinearFromCoords(coords, strides);
  }

  _requiredStridedInputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
    if (!this._inputTensorDesc) {
      throw new Error("internal error: strided input descriptor is not initialized");
    }
    return requiredBytesForBatchRange(this._inputTensorDesc, {
      bytesPerElement: 2 * 4,
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
      bytesPerElement: 2 * 4,
      runtimeExtraElements: extraOffsetElements,
      batchStart,
      batchCount,
    });
  }

  _copyStridedInputToContiguous(commandEncoder, { input, inputOffsetBytes, batchStart, batchCount, dst, dstOffsetBytes }) {
    if (inputOffsetBytes % 8 !== 0) {
      throw new Error(`inputOffsetBytes must be a multiple of 8 for complex-strided input; got ${inputOffsetBytes}`);
    }
    const extraOffsetElements = (inputOffsetBytes / 8) | 0;
    const neededBytes = this._requiredStridedInputBytes(extraOffsetElements, batchStart, batchCount);
    const inputBytes = getBufferByteLength(input);
    if (inputBytes < neededBytes) {
      throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
    }

    const coords = new Array(this.rank).fill(0);
    for (let b = 0; b < batchCount; b++) {
      const srcBatchBase = this._inputOffsetElements + extraOffsetElements + (batchStart + b) * this._inputBatchStrideElements;
      const dstBase = dstOffsetBytes + b * this.inputLogicalTotal * 8;
      for (let li = 0; li < this.inputLogicalTotal; li++) {
        this._coordsFromLinear(li, this.inputShape, coords);
        let srcElem = srcBatchBase;
        for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src: input,
          srcOffsetBytes: srcElem * 8,
          dst,
          dstOffsetBytes: dstBase + li * 8,
          bytes: 8,
        });
      }
    }
  }

  _copyContiguousToStridedOutput(commandEncoder, { src, srcOffsetBytes, batchStart, batchCount, output, outputOffsetBytes, kernelIndex = 0 }) {
    if (outputOffsetBytes % 8 !== 0) {
      throw new Error(`outputOffsetBytes must be a multiple of 8 for complex-strided output; got ${outputOffsetBytes}`);
    }
    if (!Number.isInteger(kernelIndex) || kernelIndex < 0 || kernelIndex >= this.kernelCount) {
      throw new Error(`kernelIndex must be within [0, ${this.kernelCount - 1}]; got ${kernelIndex}`);
    }
    const baseExtraOffsetElements = outputOffsetBytes / 8;
    const kernelExtraElements = checkedSafeNonNegativeInt(
      kernelIndex * this._stridedOutputKernelStrideElements,
      "fftconv.output kernel offset"
    );
    const extraOffsetElements = checkedSafeNonNegativeInt(
      baseExtraOffsetElements + kernelExtraElements,
      "fftconv.output runtime extra offset"
    );
    const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements, batchStart, batchCount);
    const outputBytes = getBufferByteLength(output);
    if (outputBytes < neededBytes) {
      throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
    }

    const coords = new Array(this.rank).fill(0);
    for (let b = 0; b < batchCount; b++) {
      const srcBase = srcOffsetBytes + b * this.outputLogicalTotal * 8;
      const dstBatchBase = this._outputOffsetElements + extraOffsetElements + (batchStart + b) * this._outputBatchStrideElements;
      for (let li = 0; li < this.outputLogicalTotal; li++) {
        this._coordsFromLinear(li, this.outputShape, coords);
        let dstElem = dstBatchBase;
        for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
        this._copyAnySpan(commandEncoder, {
          src,
          srcOffsetBytes: srcBase + li * 8,
          dst: output,
          dstOffsetBytes: dstElem * 8,
          bytes: 8,
        });
      }
    }
  }

  _prepareKernelSource(kernel) {
    const singleKernelComplex = 2 * this.kernelLogicalTotal;
    const packedKernelComplex = singleKernelComplex * this.kernelCount;

    if (kernel instanceof Float32Array) {
      if (this.kernelCount === 1) {
        if (kernel.length !== singleKernelComplex && kernel.length !== packedKernelComplex) {
          throw new Error(`kernel Float32Array length must be ${singleKernelComplex}; got ${kernel.length}`);
        }
      } else if (kernel.length !== packedKernelComplex) {
        throw new Error(`kernel Float32Array length must be ${packedKernelComplex} for kernelCount=${this.kernelCount}; got ${kernel.length}`);
      }
      const uploadBytes = this.kernelCount * this.kernelInputBytes;
      const payload = kernel.length === singleKernelComplex ? kernel : kernel.subarray(0, packedKernelComplex);
      this._ensureKernelUploadBuffer(uploadBytes);
      this.device.queue.writeBuffer(this._kernelUploadBuffer, 0, payload);
      return { kind: "packed-upload", buffer: this._kernelUploadBuffer };
    }

    if (Array.isArray(kernel)) {
      if (kernel.length !== this.kernelCount) {
        throw new Error(`kernel array length must equal fftConv.kernelCount=${this.kernelCount}; got ${kernel.length}`);
      }
      const allTyped = kernel.every((k) => k instanceof Float32Array);
      if (allTyped) {
        const packed = new Float32Array(packedKernelComplex);
        for (let i = 0; i < this.kernelCount; i++) {
          if (kernel[i].length !== singleKernelComplex) {
            throw new Error(`kernel[${i}] Float32Array length must be ${singleKernelComplex}; got ${kernel[i].length}`);
          }
          packed.set(kernel[i], i * singleKernelComplex);
        }
        const uploadBytes = this.kernelCount * this.kernelInputBytes;
        this._ensureKernelUploadBuffer(uploadBytes);
        this.device.queue.writeBuffer(this._kernelUploadBuffer, 0, packed);
        return { kind: "packed-upload", buffer: this._kernelUploadBuffer };
      }
      if (kernel.some((k) => k instanceof Float32Array)) {
        throw new Error("kernel array items must be all Float32Array or all GPUBuffer/BufferView values");
      }
      return { kind: "array-sources", sources: kernel };
    }

    return { kind: "packed-source", source: kernel };
  }

  _kernelSourceSpec(preparedKernel, kernelIndex) {
    if (preparedKernel.kind === "packed-upload") {
      return { source: preparedKernel.buffer, offsetBytes: kernelIndex * this.kernelInputBytes };
    }
    if (preparedKernel.kind === "array-sources") {
      return { source: preparedKernel.sources[kernelIndex], offsetBytes: 0 };
    }
    const packedOffset = this.kernelCount > 1 ? kernelIndex * this.kernelInputBytes : 0;
    return { source: preparedKernel.source, offsetBytes: packedOffset };
  }

  _locateRangeForLogicalByte(ranges, logicalByteOffset) {
    let acc = 0;
    for (let i = 0; i < ranges.length; i++) {
      const r = ranges[i];
      const end = acc + r.sizeBytes;
      if (logicalByteOffset < end) {
        const offsetInRange = logicalByteOffset - acc;
        return {
          range: r,
          rangeIndex: i,
          rangeStart: acc,
          rangeEnd: end,
          offsetInRange,
          buffer: r.buffer,
          offsetBytes: r.offsetBytes + offsetInRange,
          bytesLeft: end - logicalByteOffset,
        };
      }
      acc = end;
    }
    throw new Error(`logical byte offset ${logicalByteOffset} is out of range`);
  }

  _mapLogicalByteToRange(ranges, logicalByteOffset) {
    const loc = this._locateRangeForLogicalByte(ranges, logicalByteOffset);
    return { buffer: loc.buffer, offsetBytes: loc.offsetBytes };
  }

  _sourceLogicalIndexForViewLinear(vi, coordsOut) {
    this._coordsFromLinear(vi, this.outputShape, coordsOut);
    for (let d = 0; d < this.rank; d++) coordsOut[d] += this.outputOffset[d];
    return this._linearFromCoords(coordsOut, this.shape);
  }

  _copyInputToData(commandEncoder, input, inputOffsetBytes, dataView, byteLength = this.mainBytes) {
    const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, byteLength);
    const dstRanges = normalizeToContiguousRanges(dataView, 0, byteLength);
    copyRangesToRanges(this.copier, commandEncoder, inRanges, dstRanges, byteLength, this._maxBindBytes);
  }

  _copyKernelToRange(commandEncoder, source, sourceOffsetBytes, kernelRange) {
    const kernelRanges = normalizeToContiguousRanges(source, sourceOffsetBytes, this.kernelBytes);
    if (kernelRanges.length === 1) {
      commandEncoder.copyBufferToBuffer(kernelRanges[0].buffer, kernelRanges[0].offsetBytes, kernelRange.buffer, kernelRange.offsetBytes, this.kernelBytes);
    } else {
      if (this.kernelBytes > this._maxBindBytes) {
        let dst = kernelRange.offsetBytes;
        for (const r of kernelRanges) {
          commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, kernelRange.buffer, dst, r.sizeBytes);
          dst += r.sizeBytes;
        }
      } else {
        this.copier.pack(commandEncoder, kernelRanges, kernelRange.buffer, kernelRange.offsetBytes);
      }
    }
  }

  _runPointwiseChunked(commandEncoder, { dataBuffer, dataOffsetBytes, batchCount, kernelBuffer, kernelOffsetBytes = 0 }) {
    const maxElems = this._pointwiseChunkElems;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxPrefixBytes = Math.max(0, storageAlign - 8);
    const minChunkElems = Math.max(1, Math.floor((this._maxBindBytes - maxPrefixBytes) / 8));
    const chunksPerBatch = Math.ceil(this.logicalTotal / minChunkElems);
    const chunkCount = Math.max(1, batchCount * chunksPerBatch);
    const paramsBytes = chunkCount * this._pointwiseParamStride;
    const paramsBuf = this._ensurePointwiseParamsBuffer(paramsBytes);

    let chunkIndex = 0;
    for (let b = 0; b < batchCount; b++) {
      const batchDataBase = dataOffsetBytes + b * this.bytesPerBatch;
      for (let i0 = 0; i0 < this.logicalTotal; ) {
        const dataBaseByte = batchDataBase + i0 * 8;
        const kernelBaseByte = kernelOffsetBytes + i0 * 8;
        const dataBindOffset = Math.floor(dataBaseByte / storageAlign) * storageAlign;
        const kernelBindOffset = Math.floor(kernelBaseByte / storageAlign) * storageAlign;
        const dataBaseElems = (dataBaseByte - dataBindOffset) / 8;
        const kernelBaseElems = (kernelBaseByte - kernelBindOffset) / 8;
        const maxCountData = Math.floor(this._maxBindBytes / 8) - dataBaseElems;
        const maxCountKernel = Math.floor(this._maxBindBytes / 8) - kernelBaseElems;
        const count = Math.min(maxElems, this.logicalTotal - i0, maxCountData, maxCountKernel);
        if (count <= 0) {
          throw new Error(
            `fftconv pointwise chunking could not satisfy aligned bind window under maxStorageBufferBindingSize=${this._maxBindBytes}`
          );
        }
        const dataBindBytes = (dataBaseElems + count) * 8;
        const kernelBindBytes = (kernelBaseElems + count) * 8;
        const paramOff = chunkIndex * this._pointwiseParamStride;
        this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([count, dataBaseElems, kernelBaseElems, 0]));
        const bg = this.device.createBindGroup({
          layout: this.pointwise.bgl,
          entries: [
            { binding: 0, resource: { buffer: dataBuffer, offset: dataBindOffset, size: dataBindBytes } },
            { binding: 1, resource: { buffer: kernelBuffer, offset: kernelBindOffset, size: kernelBindBytes } },
            { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pointwise.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
        pass.end();
        chunkIndex += 1;
        i0 += count;
      }
    }
  }

  _runPointwiseChunkedFromRanges(commandEncoder, { dataRanges, batchCount, kernelBuffer, kernelOffsetBytes = 0 }) {
    const maxElems = this._pointwiseChunkElems;
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const maxPrefixBytes = Math.max(0, storageAlign - 8);
    const minChunkElems = Math.max(1, Math.floor((this._maxBindBytes - maxPrefixBytes) / 8));

    const rangePrefix = new Array(dataRanges.length);
    let totalRangeBytes = 0;
    for (let i = 0; i < dataRanges.length; i++) {
      rangePrefix[i] = totalRangeBytes;
      totalRangeBytes += dataRanges[i].sizeBytes;
    }
    const expectedBytes = batchCount * this.bytesPerBatch;
    if (totalRangeBytes !== expectedBytes) {
      throw new Error(`fftconv internal error: data workspace spans ${totalRangeBytes} bytes, expected ${expectedBytes}`);
    }

    let chunkIndex = 0;
    for (let b = 0; b < batchCount; b++) {
      const batchByteBase = b * this.bytesPerBatch;
      let globalByte = batchByteBase;
      let i0 = 0;
      let rangeIndex = 0;
      while (rangeIndex + 1 < dataRanges.length && globalByte >= rangePrefix[rangeIndex] + dataRanges[rangeIndex].sizeBytes) {
        rangeIndex += 1;
      }
      while (i0 < this.logicalTotal) {
        while (rangeIndex + 1 < dataRanges.length && globalByte >= rangePrefix[rangeIndex] + dataRanges[rangeIndex].sizeBytes) {
          rangeIndex += 1;
        }
        const range = dataRanges[rangeIndex];
        const rangeLogicalStart = rangePrefix[rangeIndex];
        const withinRange = globalByte - rangeLogicalStart;
        const bytesLeftInRange = range.sizeBytes - withinRange;
        const elemsLeftInRange = Math.floor(bytesLeftInRange / 8);
        if (elemsLeftInRange < 1) {
          rangeIndex += 1;
          continue;
        }

        const dataBaseByte = range.offsetBytes + withinRange;
        const kernelBaseByte = kernelOffsetBytes + i0 * 8;
        const dataBindOffset = Math.floor(dataBaseByte / storageAlign) * storageAlign;
        const kernelBindOffset = Math.floor(kernelBaseByte / storageAlign) * storageAlign;
        const dataBaseElems = (dataBaseByte - dataBindOffset) / 8;
        const kernelBaseElems = (kernelBaseByte - kernelBindOffset) / 8;
        const maxCountData = Math.floor(this._maxBindBytes / 8) - dataBaseElems;
        const maxCountKernel = Math.floor(this._maxBindBytes / 8) - kernelBaseElems;
        const count = Math.min(maxElems, this.logicalTotal - i0, elemsLeftInRange, maxCountData, maxCountKernel);
        if (count <= 0) {
          throw new Error(
            `fftconv segmented pointwise chunking could not satisfy aligned bind window under maxStorageBufferBindingSize=${this._maxBindBytes}`
          );
        }

        const dataBindBytes = (dataBaseElems + count) * 8;
        const kernelBindBytes = (kernelBaseElems + count) * 8;
        const paramOff = chunkIndex * this._pointwiseParamStride;
        this._ensurePointwiseParamsBuffer(paramOff + this._pointwiseParamStride);
        this.device.queue.writeBuffer(this._pointwiseParamsBuffer, paramOff, new Uint32Array([count, dataBaseElems, kernelBaseElems, 0]));
        const bg = this.device.createBindGroup({
          layout: this.pointwise.bgl,
          entries: [
            { binding: 0, resource: { buffer: range.buffer, offset: dataBindOffset, size: dataBindBytes } },
            { binding: 1, resource: { buffer: kernelBuffer, offset: kernelBindOffset, size: kernelBindBytes } },
            { binding: 2, resource: { buffer: this._pointwiseParamsBuffer, offset: paramOff, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pointwise.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
        pass.end();

        chunkIndex += 1;
        i0 += count;
        globalByte += count * 8;
      }
    }
  }

  _coordsFromLinear(i, shape, outCoords) {
    tensorCoordsFromLinear(i, shape, outCoords);
  }

  _linearFromCoords(coords, shape) {
    return tensorLinearFromCoordsShape(coords, shape);
  }

  _extractOutputByCopies(commandEncoder, { srcBuffer, srcOffsetBytes, batchCount, dstBuffer, dstOffsetBytes }) {
    const logicalCoords = new Array(this.rank).fill(0);
    const maxChunkElems = Math.max(1, this._extractCopyChunkElems | 0);
    for (let b = 0; b < batchCount; b++) {
      const srcBase = srcOffsetBytes + b * this.bytesPerBatch;
      const dstBase = dstOffsetBytes + b * this.outputBytesPerBatch;
      for (let vi = 0; vi < this.outputLogicalTotal; ) {
        const srcIndex = this._sourceLogicalIndexForViewLinear(vi, logicalCoords);
        let run = Math.min(maxChunkElems, this.outputLogicalTotal - vi);
        let prevSrcIndex = srcIndex;
        for (let r = 1; r < run; r++) {
          const nextSrcIndex = this._sourceLogicalIndexForViewLinear(vi + r, logicalCoords);
          if (nextSrcIndex !== prevSrcIndex + 1) {
            run = r;
            break;
          }
          prevSrcIndex = nextSrcIndex;
        }
        commandEncoder.copyBufferToBuffer(
          srcBuffer,
          srcBase + srcIndex * 8,
          dstBuffer,
          dstBase + vi * 8,
          run * 8
        );
        vi += run;
      }
    }
  }

  _extractOutputByCopiesToRanges(commandEncoder, { srcRanges, batchCount, outRanges }) {
    const logicalCoords = new Array(this.rank).fill(0);
    const maxChunkElems = Math.max(1, this._extractCopyChunkElems | 0);
    let rangeIndex = 0;
    let rangeStart = 0;
    let rangeEnd = outRanges.length ? outRanges[0].sizeBytes : 0;
    for (let b = 0; b < batchCount; b++) {
      const srcBase = b * this.bytesPerBatch;
      for (let vi = 0; vi < this.outputLogicalTotal; ) {
        const outElem = b * this.outputLogicalTotal + vi;
        const outByte = outElem * 8;
        while (rangeIndex < outRanges.length - 1 && outByte >= rangeEnd) {
          rangeStart = rangeEnd;
          rangeIndex += 1;
          rangeEnd += outRanges[rangeIndex].sizeBytes;
        }
        const r = outRanges[rangeIndex];
        const dstByte = r.offsetBytes + (outByte - rangeStart);

        const srcIndex = this._sourceLogicalIndexForViewLinear(vi, logicalCoords);
        const srcLogicalByte = srcBase + srcIndex * 8;
        const srcLoc = this._locateRangeForLogicalByte(srcRanges, srcLogicalByte);

        const maxDstElemsInRange = Math.max(1, Math.floor((rangeEnd - outByte) / 8));
        const maxSrcElemsInRange = Math.max(1, Math.floor(srcLoc.bytesLeft / 8));
        let run = Math.min(
          maxChunkElems,
          this.outputLogicalTotal - vi,
          maxDstElemsInRange,
          maxSrcElemsInRange
        );
        let prevSrcIndex = srcIndex;
        for (let rr = 1; rr < run; rr++) {
          const nextSrcIndex = this._sourceLogicalIndexForViewLinear(vi + rr, logicalCoords);
          if (nextSrcIndex !== prevSrcIndex + 1) {
            run = rr;
            break;
          }
          prevSrcIndex = nextSrcIndex;
        }

        commandEncoder.copyBufferToBuffer(
          srcLoc.buffer,
          srcLoc.offsetBytes,
          r.buffer,
          dstByte,
          run * 8
        );
        vi += run;
      }
    }
  }

  _runOutputExtract(commandEncoder, srcBuffer, srcOffsetBytes, batchCount, dstBuffer, dstOffsetBytes, dstBytes) {
    this.device.queue.writeBuffer(
      this.outputExtract.params,
      0,
      new Uint32Array([this.logicalTotal, this.outputLogicalTotal, batchCount, 0])
    );
    const bg = this.device.createBindGroup({
      layout: this.outputExtract.bgl,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: srcBuffer,
            offset: srcOffsetBytes,
            size: this.bytesPerBatch * batchCount,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: dstBuffer,
            offset: dstOffsetBytes,
            size: dstBytes,
          },
        },
        { binding: 2, resource: { buffer: this.outputExtract.params, offset: 0, size: 16 } },
      ],
    });
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.outputExtract.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil((this.outputLogicalTotal * batchCount) / this.workgroupSize), 1, 1);
    pass.end();
  }

  _writeExtractedOutput(commandEncoder, output, dstOffsetBytes, srcView, srcOffsetBytes, batchCount) {
    const outBytes = this.outputBytesPerBatch * batchCount;
    const outRanges = normalizeToContiguousRanges(output, dstOffsetBytes, outBytes);
    const srcBindBytes = this.bytesPerBatch * batchCount;
    const srcRanges = normalizeToContiguousRanges(srcView, srcOffsetBytes, srcBindBytes);
    const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
    const canDirectBindInput =
      srcRanges.length === 1 &&
      srcRanges[0].offsetBytes % storageAlign === 0;
    const canDirectBindOutput = outRanges.length === 1 && outRanges[0].offsetBytes % storageAlign === 0;
    const fitsBindLimits = srcBindBytes <= this._maxBindBytes && outBytes <= this._maxBindBytes;
    const allowDirectExtract = this.kernelCount === 1;

    if (allowDirectExtract && canDirectBindInput && canDirectBindOutput && fitsBindLimits) {
      this._runOutputExtract(
        commandEncoder,
        srcRanges[0].buffer,
        srcRanges[0].offsetBytes,
        batchCount,
        outRanges[0].buffer,
        outRanges[0].offsetBytes,
        outBytes
      );
      return;
    }

    // Fallback for unaligned/segmented/oversized binds: explicit element copies.
    // This avoids alias hazards between source and staging buffers on real browser GPU drivers.
    this._extractOutputByCopiesToRanges(commandEncoder, {
      srcRanges,
      batchCount,
      outRanges,
    });
  }

  _writeSingleKernelOutputContiguous(commandEncoder, output, outputOffsetBytes, dataView, batchCount = this.batch) {
    const runOutputBytes = this.outputBytesPerBatch * batchCount;
    if (!this._needsOutputExtract) {
      const srcRanges = normalizeToContiguousRanges(dataView, 0, runOutputBytes);
      const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, runOutputBytes);
      copyRangesToRanges(
        this.copier,
        commandEncoder,
        srcRanges,
        outRanges,
        runOutputBytes,
        this._maxBindBytes
      );
      return;
    }
    for (let b = 0; b < batchCount; b++) {
      this._writeExtractedOutput(
        commandEncoder,
        output,
        outputOffsetBytes + b * this.outputBytesPerBatch,
        dataView,
        b * this.bytesPerBatch,
        1
      );
    }
  }

  _writeKernelOutput(commandEncoder, output, outputOffsetBytes, kernelIndex, dataView, batchStart = 0, batchCount = this.batch) {
    const runOutputBytes = this.outputBytesPerBatch * batchCount;
    if (this.outputLayout === "kernel-major") {
      const dstOffset = outputOffsetBytes + kernelIndex * this.outputBytesPerKernel + batchStart * this.outputBytesPerBatch;
      if (!this._needsOutputExtract) {
        const srcRanges = normalizeToContiguousRanges(dataView, 0, runOutputBytes);
        const outRanges = normalizeToContiguousRanges(output, dstOffset, runOutputBytes);
        copyRangesToRanges(
          this.copier,
          commandEncoder,
          srcRanges,
          outRanges,
          runOutputBytes,
          this._maxBindBytes
        );
      } else {
        for (let b = 0; b < batchCount; b++) {
          this._writeExtractedOutput(
            commandEncoder,
            output,
            dstOffset + b * this.outputBytesPerBatch,
            dataView,
            b * this.bytesPerBatch,
            1
          );
        }
      }
      return;
    }

    if (!this._needsOutputExtract) {
      for (let b = 0; b < batchCount; b++) {
        const srcRanges = normalizeToContiguousRanges(dataView, b * this.bytesPerBatch, this.outputBytesPerBatch);
        const slot = (batchStart + b) * this.kernelCount + kernelIndex;
        const dstOffset = outputOffsetBytes + slot * this.outputBytesPerBatch;
        const outRanges = normalizeToContiguousRanges(output, dstOffset, this.outputBytesPerBatch);
        copyRangesToRanges(
          this.copier,
          commandEncoder,
          srcRanges,
          outRanges,
          this.outputBytesPerBatch,
          this._maxBindBytes
        );
      }
      return;
    }

    for (let b = 0; b < batchCount; b++) {
      const slot = (batchStart + b) * this.kernelCount + kernelIndex;
      const dstOffset = outputOffsetBytes + slot * this.outputBytesPerBatch;
      this._writeExtractedOutput(
        commandEncoder,
        output,
        dstOffset,
        dataView,
        b * this.bytesPerBatch,
        1
      );
    }
  }

  destroy() {
    if (this._destroyed) return;
    this.fftData.destroy();
    this.fftKernel.destroy();
    this.ifftData.destroy();
    this.outputExtract?.params?.destroy?.();
    this._pointwiseParamsBuffer?.destroy?.();
    for (const b of this._retiredPointwiseParamsBuffers) b?.destroy?.();
    this._kernelUploadBuffer?.destroy?.();
    for (const b of this._retiredKernelUploadBuffers) b?.destroy?.();
    this.kernelBuffer?.destroy?.();
    this._kernelEmbedInputBuffer?.destroy?.();
    this._splitWorkspace?.data?.destroy?.();
    this._splitWorkspace?.scratch?.destroy?.();
    this._arena?.destroy?.();
    super.destroy();
  }

  exec(commandEncoder, execOpts) {
    if (this._destroyed) throw new Error("plan destroyed");
    const {
      input,
      output,
      kernel,
      temp,
      inputOffsetBytes = 0,
      outputOffsetBytes = 0,
    } = execOpts ?? {};

    if (!input || !output) throw new Error("fftconv exec requires input and output");
    if (!kernel) throw new Error("fftconv exec requires kernel");
    if (inputOffsetBytes % 8 !== 0) throw new Error(`inputOffsetBytes must be a multiple of 8; got ${inputOffsetBytes}`);
    if (outputOffsetBytes % 8 !== 0) throw new Error(`outputOffsetBytes must be a multiple of 8; got ${outputOffsetBytes}`);

    let workspaceTemp = temp;
    if (
      workspaceTemp &&
      (buffersAlias(workspaceTemp, input) || buffersAlias(workspaceTemp, output) || buffersAlias(workspaceTemp, kernel))
    ) {
      workspaceTemp = null;
    }
    if (workspaceTemp && getBufferByteLength(workspaceTemp) < this.workspaceBytes) {
      workspaceTemp = null;
    }
    const { dataView, scratchView } = this._resolveWorkspaceViews(workspaceTemp);
    const subplanScratchView = this._subplanScratchBytes
      ? this._sliceView(scratchView, this._subplanScratchOffset, this._subplanScratchBytes)
      : null;
    const stridedInputStageView = this._stridedInputStageBytesPerRun
      ? this._sliceView(scratchView, this._stridedInputStageOffset, this._stridedInputStageBytesPerRun)
      : null;
    const stridedOutputStageView = this._stridedOutputStageBytesPerRun
      ? this._sliceView(scratchView, this._stridedOutputStageOffset, this._stridedOutputStageBytesPerRun)
      : null;
    const dataRanges = normalizeToContiguousRanges(dataView, 0, this._dataBytesPerRun);
    const kernelRange = { buffer: this.kernelBuffer, offsetBytes: 0 };
    const preparedKernel = this._prepareKernelSource(kernel);
    const inputBytesPerBatch = this.inputLogicalTotal * 8;
    const fftDataTemp = this.fftData._largeBatchChunkMode ? null : subplanScratchView;
    const ifftDataTemp = this.ifftData._largeBatchChunkMode ? null : subplanScratchView;
    const fftKernelTemp = this.fftKernel._largeBatchChunkMode ? null : subplanScratchView;

    if (this._usesStridedOutput) {
      if (outputOffsetBytes % 8 !== 0) {
        throw new Error(`outputOffsetBytes must be a multiple of 8 for complex-strided output; got ${outputOffsetBytes}`);
      }
      const extraOffsetElements = outputOffsetBytes / 8;
      const kernelExtraElements = checkedSafeNonNegativeInt(
        (this.kernelCount - 1) * this._stridedOutputKernelStrideElements,
        "fftconv.output kernel coverage"
      );
      const neededBytes = this._requiredStridedOutputBytes(
        checkedSafeNonNegativeInt(extraOffsetElements + kernelExtraElements, "fftconv.output runtime extra offset"),
        0,
        this.batch
      );
      if (getBufferByteLength(output) < neededBytes) {
        throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${getBufferByteLength(output)}`);
      }
    } else {
      normalizeToContiguousRanges(output, outputOffsetBytes, this.totalOutputBytes);
    }

    for (let k = 0; k < this.kernelCount; k++) {
      const kernelSpec = this._kernelSourceSpec(preparedKernel, k);
      if (this._needsKernelEmbed) {
        let kernelEmbedSource = kernelSpec.source;
        let kernelEmbedOffset = kernelSpec.offsetBytes;
        if (this._kernelEmbedInputBuffer) {
          this._copyAnySpan(commandEncoder, {
            src: kernelSpec.source,
            srcOffsetBytes: kernelSpec.offsetBytes,
            dst: this._kernelEmbedInputBuffer,
            dstOffsetBytes: 0,
            bytes: this.kernelInputBytes,
          });
          kernelEmbedSource = this._kernelEmbedInputBuffer;
          kernelEmbedOffset = 0;
        }
        this.fftKernel.exec(commandEncoder, {
          input: kernelEmbedSource,
          output: kernelRange.buffer,
          inputOffsetBytes: kernelEmbedOffset,
          outputOffsetBytes: kernelRange.offsetBytes,
          temp: fftKernelTemp,
        });
      } else {
        this._copyKernelToRange(commandEncoder, kernelSpec.source, kernelSpec.offsetBytes, kernelRange);
        this.fftKernel.exec(commandEncoder, {
          input: kernelRange.buffer,
          inputOffsetBytes: kernelRange.offsetBytes,
          temp: fftKernelTemp,
        });
      }

      if (this._batchSlicedExecution) {
        for (let b = 0; b < this.batch; b++) {
          if (this._needsInputEmbed) {
            if (this._usesStridedInput) {
              this._copyStridedInputToContiguous(commandEncoder, {
                input,
                inputOffsetBytes,
                batchStart: b,
                batchCount: 1,
                dst: stridedInputStageView,
                dstOffsetBytes: 0,
              });
              this.fftData.exec(commandEncoder, {
                input: stridedInputStageView,
                output: dataView,
                inputOffsetBytes: 0,
                outputOffsetBytes: 0,
                temp: fftDataTemp,
              });
            } else {
              this.fftData.exec(commandEncoder, {
                input,
                output: dataView,
                inputOffsetBytes: inputOffsetBytes + b * inputBytesPerBatch,
                outputOffsetBytes: 0,
                temp: fftDataTemp,
              });
            }
          } else {
            if (this._usesStridedInput) {
              this._copyStridedInputToContiguous(commandEncoder, {
                input,
                inputOffsetBytes,
                batchStart: b,
                batchCount: 1,
                dst: dataView,
                dstOffsetBytes: 0,
              });
            } else {
              this._copyInputToData(
                commandEncoder,
                input,
                inputOffsetBytes + b * this.bytesPerBatch,
                dataView,
                this.bytesPerBatch
              );
            }
            this.fftData.exec(commandEncoder, {
              input: dataView,
              inputOffsetBytes: 0,
              temp: fftDataTemp,
            });
          }

          if (dataRanges.length === 1) {
            this._runPointwiseChunked(commandEncoder, {
              dataBuffer: dataRanges[0].buffer,
              dataOffsetBytes: dataRanges[0].offsetBytes,
              batchCount: 1,
              kernelBuffer: kernelRange.buffer,
              kernelOffsetBytes: kernelRange.offsetBytes,
            });
          } else {
            this._runPointwiseChunkedFromRanges(commandEncoder, {
              dataRanges,
              batchCount: 1,
              kernelBuffer: kernelRange.buffer,
              kernelOffsetBytes: kernelRange.offsetBytes,
            });
          }

          this.ifftData.exec(commandEncoder, {
            input: dataView,
            inputOffsetBytes: 0,
            temp: ifftDataTemp,
          });

          if (this._usesStridedOutput) {
            let stridedOutputSource = dataView;
            if (this._needsOutputExtract) {
              if (!stridedOutputStageView) {
                throw new Error("internal error: missing strided output staging view for extracted output path");
              }
              this._writeSingleKernelOutputContiguous(commandEncoder, stridedOutputStageView, 0, dataView, 1);
              stridedOutputSource = stridedOutputStageView;
            }
            this._copyContiguousToStridedOutput(commandEncoder, {
              src: stridedOutputSource,
              srcOffsetBytes: 0,
              batchStart: b,
              batchCount: 1,
              output,
              outputOffsetBytes,
              kernelIndex: k,
            });
          } else {
            this._writeKernelOutput(
              commandEncoder,
              output,
              outputOffsetBytes,
              k,
              dataView,
              b,
              1
            );
          }
        }
      } else {
        if (this._needsInputEmbed) {
          if (this._usesStridedInput) {
            this._copyStridedInputToContiguous(commandEncoder, {
              input,
              inputOffsetBytes,
              batchStart: 0,
              batchCount: this.batch,
              dst: stridedInputStageView,
              dstOffsetBytes: 0,
            });
            this.fftData.exec(commandEncoder, {
              input: stridedInputStageView,
              output: dataView,
              inputOffsetBytes: 0,
              outputOffsetBytes: 0,
              temp: fftDataTemp,
            });
          } else {
            this.fftData.exec(commandEncoder, {
              input,
              output: dataView,
              inputOffsetBytes,
              outputOffsetBytes: 0,
              temp: fftDataTemp,
            });
          }
        } else {
          if (this._usesStridedInput) {
            this._copyStridedInputToContiguous(commandEncoder, {
              input,
              inputOffsetBytes,
              batchStart: 0,
              batchCount: this.batch,
              dst: dataView,
              dstOffsetBytes: 0,
            });
          } else {
            this._copyInputToData(commandEncoder, input, inputOffsetBytes, dataView);
          }
          this.fftData.exec(commandEncoder, {
            input: dataView,
            inputOffsetBytes: 0,
            temp: fftDataTemp,
          });
        }

        if (dataRanges.length === 1) {
          this._runPointwiseChunked(commandEncoder, {
            dataBuffer: dataRanges[0].buffer,
            dataOffsetBytes: dataRanges[0].offsetBytes,
            batchCount: this.batch,
            kernelBuffer: kernelRange.buffer,
            kernelOffsetBytes: kernelRange.offsetBytes,
          });
        } else {
          this._runPointwiseChunkedFromRanges(commandEncoder, {
            dataRanges,
            batchCount: this.batch,
            kernelBuffer: kernelRange.buffer,
            kernelOffsetBytes: kernelRange.offsetBytes,
          });
        }

        this.ifftData.exec(commandEncoder, {
          input: dataView,
          inputOffsetBytes: 0,
          temp: ifftDataTemp,
        });

        if (this._usesStridedOutput) {
          let stridedOutputSource = dataView;
          if (this._needsOutputExtract) {
            if (!stridedOutputStageView) {
              throw new Error("internal error: missing strided output staging view for extracted output path");
            }
            this._writeSingleKernelOutputContiguous(commandEncoder, stridedOutputStageView, 0, dataView, this.batch);
            stridedOutputSource = stridedOutputStageView;
          }
          this._copyContiguousToStridedOutput(commandEncoder, {
            src: stridedOutputSource,
            srcOffsetBytes: 0,
            batchStart: 0,
            batchCount: this.batch,
            output,
            outputOffsetBytes,
            kernelIndex: k,
          });
        } else {
          this._writeKernelOutput(commandEncoder, output, outputOffsetBytes, k, dataView);
        }
      }
    }
  }
}


