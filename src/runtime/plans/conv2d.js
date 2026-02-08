// Copyright (c) 2026 Maksim Eremenko

import { BasePlan } from "../base_plan.js";
import { createInternalArena, viewFromArena } from "../workspace.js";
import { normalizeToContiguousRanges } from "../segmented_io.js";
import { assertOneOf, isPositiveInt, alignBytes, ensureWithinBindingLimit, getBufferByteLength } from "../common.js";
import { hashFloat32Array } from "../../utils/hash.js";

import { generateConv2dRealWGSL, generateConv2dComplexRealKernelWGSL, generateConv2dComplexComplexKernelWGSL } from "../../kernels/conv2d.js";

function isGpuBuffer(x) {
  return x && !x?.segments && typeof x.size === "number";
}

export class Conv2dPlan extends BasePlan {
  constructor(device, opts) {
    super(device, opts);
    const { shape, batch = 1, layout = { interleavedComplex: true }, precision = "f32", conv } = opts ?? {};
    if (!Array.isArray(shape) || shape.length !== 2) throw new Error(`conv2d shape must be [H,W]`);
    if (!shape.every(isPositiveInt)) throw new Error("conv2d shape must be positive ints");
    if (!conv) throw new Error("conv2d requires conv object");
    if (!Number.isInteger(batch) || batch <= 0) throw new Error("batch must be positive int");

    const { kernelSize, kernelType = "real", padding = "same", pad = null, boundary = "zero" } = conv;
    if (![1, 2, 3].includes(kernelSize)) throw new Error("conv.kernelSize must be 1|2|3");
    assertOneOf(kernelType, ["real", "complex"], "conv.kernelType");
    assertOneOf(padding, ["valid", "same", "explicit"], "conv.padding");
    if (boundary !== "zero") throw new Error('conv.boundary currently supports only "zero"');
    assertOneOf(precision, ["f32", "f16-storage"], "precision");
    if (precision !== "f32") throw new Error('conv2d precision="f16-storage" is not implemented in current implementation');

    const complex = layout?.interleavedComplex === true;
    if (!complex && kernelType === "complex") throw new Error("real input/output does not support complex kernel in current implementation");

    this.shape = shape.slice();
    this.batch = batch;
    this.complex = complex;
    this.kernelSize = kernelSize;
    this.kernelType = kernelType;
    this.padding = padding;

    const [Hout, Wout] = shape;
    let pt = 0, pb = 0, padL = 0, padR = 0;
    if (padding === "same") {
      const p = Math.floor(kernelSize / 2);
      pt = p; pb = kernelSize - 1 - p;
      padL = p; padR = kernelSize - 1 - p;
    } else if (padding === "valid") {
      pt = pb = padL = padR = 0;
    } else {
      if (!Array.isArray(pad) || pad.length !== 4) throw new Error('conv.pad must be [top,bottom,left,right] when padding="explicit"');
      [pt, pb, padL, padR] = pad;
      if (![pt, pb, padL, padR].every((x) => Number.isInteger(x) && x >= 0)) throw new Error("conv.pad entries must be non-negative ints");
    }
    this.pad = [pt, pb, padL, padR];

    const Hin = Hout + (kernelSize - 1) - pt - pb;
    const Win = Wout + (kernelSize - 1) - padL - padR;
    if (Hin <= 0 || Win <= 0) throw new Error(`Derived input shape invalid: Hin=${Hin} Win=${Win}`);
    this.inShape = [Hin, Win];

    if (padding === "valid") {
      const expH = Hin - kernelSize + 1;
      const expW = Win - kernelSize + 1;
      if (expH !== Hout || expW !== Wout) {
        throw new Error(`padding="valid" requires output [Hin-k+1,Win-k+1]; got [${Hout},${Wout}]`);
      }
    }

    const inElems = Hin * Win * batch;
    const outElems = Hout * Wout * batch;
    this.inBytes = inElems * (complex ? 8 : 4);
    this.outBytes = outElems * (complex ? 8 : 4);
    ensureWithinBindingLimit(device, this.inBytes, "conv2d input");
    ensureWithinBindingLimit(device, this.outBytes, "conv2d output");

    const bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

    let code;
    if (!complex) {
      code = generateConv2dRealWGSL({ Hin, Win, Hout, Wout, k: kernelSize, pad: this.pad, workgroupSize: this.workgroupSize });
    } else if (kernelType === "real") {
      code = generateConv2dComplexRealKernelWGSL({ Hin, Win, Hout, Wout, k: kernelSize, pad: this.pad, workgroupSize: this.workgroupSize });
    } else {
      code = generateConv2dComplexComplexKernelWGSL({ Hin, Win, Hout, Wout, k: kernelSize, pad: this.pad, workgroupSize: this.workgroupSize });
    }
    const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([batch, 0, 0, 0]));

    this.pipeline = { bgl, pl: pipelineLayout, pipeline, params };
    this.kernelCache = new Map();

    const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    this.inStageOffset = 0;
    this.outStageOffset = alignBytes(this.inBytes, storageAlign);

    const kBytes = kernelType === "complex" ? kernelSize * kernelSize * 8 : kernelSize * kernelSize * 4;
    this.workspaceBytes = this.outStageOffset + this.outBytes + kBytes;
    this._arena = createInternalArena(device, this.workspaceBytes);
  }

  getWorkspaceSizeBytes() {
    return this.workspaceBytes;
  }

  destroy() {
    if (this._destroyed) return;
    this.pipeline.params.destroy();
    for (const b of this.kernelCache.values()) b.destroy();
    this._arena?.destroy?.();
    super.destroy();
  }

  _getOrUploadKernel(kernel) {
    if (isGpuBuffer(kernel)) return kernel;
    if (!(kernel instanceof Float32Array)) throw new Error("kernel must be GPUBuffer or Float32Array");
    const h = hashFloat32Array(kernel);
    const key = `${this.kernelType}:${this.kernelSize}:${h}:${kernel.length}`;
    let buf = this.kernelCache.get(key);
    if (!buf) {
      buf = this.device.createBuffer({ size: kernel.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buf, 0, kernel);
      this.kernelCache.set(key, buf);
    }
    return buf;
  }

  exec(commandEncoder, execOpts) {
    if (this._destroyed) throw new Error("plan destroyed");
    const { input, output, kernel, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
    if (!input || !output) throw new Error("conv2d exec requires input and output");
    if (!kernel) throw new Error("conv2d exec requires kernel");

    const arena = temp ?? this._arena;
    if (!arena) throw new Error("No workspace buffer");
    if (getBufferByteLength(arena) < this.workspaceBytes) throw new Error("temp too small");

    const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
    const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);

    const inStage = viewFromArena(arena, this.inStageOffset, this.inBytes);
    const outStage = viewFromArena(arena, this.outStageOffset, this.outBytes);
    const kBytes = this.kernelType === "complex" ? this.kernelSize * this.kernelSize * 8 : this.kernelSize * this.kernelSize * 4;

    let inBuf = null;
    let inOff = 0;
    if (inRanges.length === 1) {
      inBuf = inRanges[0].buffer;
      inOff = inRanges[0].offsetBytes;
    } else {
      inBuf = inStage.segments[0].buffer;
      inOff = inStage.segments[0].offsetBytes;
      this.copier.pack(commandEncoder, inRanges, inBuf, inOff);
    }

    let outBuf = null;
    let outOff = 0;
    const needsUnpack = outRanges.length > 1;
    if (!needsUnpack) {
      outBuf = outRanges[0].buffer;
      outOff = outRanges[0].offsetBytes;
    } else {
      outBuf = outStage.segments[0].buffer;
      outOff = outStage.segments[0].offsetBytes;
    }

    const kBuf = this._getOrUploadKernel(kernel);
    const bg = this.device.createBindGroup({
      layout: this.pipeline.bgl,
      entries: [
        { binding: 0, resource: { buffer: inBuf, offset: inOff, size: this.inBytes } },
        { binding: 1, resource: { buffer: outBuf, offset: outOff, size: this.outBytes } },
        { binding: 2, resource: { buffer: kBuf, offset: 0, size: kBytes } },
        { binding: 3, resource: { buffer: this.pipeline.params, offset: 0, size: 16 } },
      ],
    });
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline.pipeline);
    pass.setBindGroup(0, bg);
    const outElems = this.outBytes / (this.complex ? 8 : 4);
    pass.dispatchWorkgroups(Math.ceil(outElems / this.workgroupSize), 1, 1);
    pass.end();

    if (needsUnpack) {
      this.copier.unpack(commandEncoder, outBuf, outOff, outRanges);
    }
  }
}

