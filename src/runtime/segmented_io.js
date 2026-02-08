// Copyright (c) 2026 Maksim Eremenko

import { generateSegmentedCopyWGSL } from "../kernels/segmented_copy.js";
import { pickWorkgroupSizeX } from "../utils/limits.js";

function isGpuBuffer(x) {
  return x && !x.segments && typeof x.size === "number" && typeof x.destroy === "function";
}

export function computeSegCap(device, reservedBindings = 1) {
  const max = device.limits?.maxStorageBuffersPerShaderStage ?? 8;
  return Math.max(0, Math.min(8, max - reservedBindings));
}

export function normalizeToContiguousRanges(bufOrView, extraOffsetBytes, totalBytesWanted) {
  if (isGpuBuffer(bufOrView)) {
    if (extraOffsetBytes + totalBytesWanted > bufOrView.size) {
      throw new Error(`GPUBuffer too small: need ${extraOffsetBytes + totalBytesWanted} bytes, have ${bufOrView.size}`);
    }
    return [{ buffer: bufOrView, offsetBytes: extraOffsetBytes, sizeBytes: totalBytesWanted }];
  }
  const segments = bufOrView?.segments;
  if (!Array.isArray(segments) || segments.length === 0) throw new Error("Expected GPUBuffer or BufferView");
  const viewStart = bufOrView.logicalByteOffset ?? 0;
  const logicalByteOffset = viewStart + extraOffsetBytes;
  const lengthBytes = bufOrView.lengthBytes ?? segments.reduce((a, s) => a + s.sizeBytes, 0);
  // BufferView length is relative to its logicalByteOffset start.
  if (extraOffsetBytes + totalBytesWanted > lengthBytes) {
    throw new Error(
      `BufferView too small: need ${totalBytesWanted} bytes at offset ${extraOffsetBytes}, lengthBytes=${lengthBytes}, logicalByteOffset=${viewStart}`
    );
  }

  const out = [];
  let remaining = totalBytesWanted;
  let cursor = logicalByteOffset;
  let logicalPos = 0;
  for (const seg of segments) {
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
  if (remaining !== 0) throw new Error("BufferView did not cover requested range");
  return out;
}

export class SegmentedCopier {
  constructor(device, pipelineCache) {
    this.device = device;
    this.cache = pipelineCache;
    this.workgroupSize = pickWorkgroupSizeX(device.limits, 256);
    this.storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
    // Tier-A cap: bind up to SEG_CAP segments + 1 contiguous buffer per shader stage.
    this.segCap = computeSegCap(device, 1);

    this._layout = device.createBindGroupLayout({
      entries: [
        // segment bindings 0..cap-1 (storage), plus contiguous binding cap, plus uniform cap+1
        ...Array.from({ length: this.segCap }, (_, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        })),
        {
          binding: this.segCap,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: this.segCap + 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    this._layoutUnpack = device.createBindGroupLayout({
      entries: [
        ...Array.from({ length: this.segCap }, (_, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        })),
        {
          binding: this.segCap,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: this.segCap + 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    this._plPack = device.createPipelineLayout({ bindGroupLayouts: [this._layout] });
    this._plUnpack = device.createPipelineLayout({ bindGroupLayouts: [this._layoutUnpack] });

    this._uniform = device.createBuffer({
      size: 16 + 4 * this.segCap * 2,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._dummy = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // For unpack, missing segment bindings are writable storage; they must not alias each other.
    this._dummyRW = Array.from({ length: this.segCap }, () =>
      device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    );
    device.queue.writeBuffer(this._dummy, 0, new Uint32Array([0]));
    for (const b of this._dummyRW) device.queue.writeBuffer(b, 0, new Uint32Array([0]));
  }

  destroy() {
    this._uniform.destroy();
    this._dummy.destroy();
    for (const b of this._dummyRW) b.destroy();
  }

  pack(commandEncoder, srcRanges, dstBuffer, dstOffsetBytes) {
    // srcRanges cover contiguous logical byte range, in order, sum sizes == totalBytes.
    const totalBytes = srcRanges.reduce((a, r) => a + r.sizeBytes, 0);
    if (totalBytes % 4 !== 0) throw new Error("SegmentedCopier only supports totalBytes multiple of 4");
    const totalWords = totalBytes >>> 2;

    if (srcRanges.length <= this.segCap) {
      const canTierA =
        dstOffsetBytes % this.storageAlign === 0 && srcRanges.every((r) => r.offsetBytes % this.storageAlign === 0);
      if (!canTierA) {
        // Fall back to Tier-B copies if bindings would violate offset alignment.
        let dst = dstOffsetBytes;
        for (const r of srcRanges) {
          commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
          dst += r.sizeBytes;
        }
        return;
      }

      const segSizesWords = new Uint32Array(this.segCap);
      const segPrefixWords = new Uint32Array(this.segCap);
      let prefix = 0;
      for (let i = 0; i < srcRanges.length; i++) {
        if (srcRanges[i].sizeBytes % 4 !== 0) throw new Error("Segment size must be multiple of 4");
        segSizesWords[i] = srcRanges[i].sizeBytes >>> 2;
        segPrefixWords[i] = prefix;
        prefix += segSizesWords[i];
      }

      const header = new Uint32Array([srcRanges.length, totalWords, 0, 0]);
      const ub = new Uint32Array(4 + this.segCap * 2);
      ub.set(header, 0);
      ub.set(segSizesWords, 4);
      ub.set(segPrefixWords, 4 + this.segCap);
      this.device.queue.writeBuffer(this._uniform, 0, ub);

      const code = generateSegmentedCopyWGSL({
        cap: this.segCap,
        direction: "pack",
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: this._plPack });

      const entries = [];
      for (let i = 0; i < this.segCap; i++) {
        const r = srcRanges[i];
        if (!r) {
          entries.push({ binding: i, resource: { buffer: this._dummy, offset: 0, size: 4 } });
        } else {
          entries.push({ binding: i, resource: { buffer: r.buffer, offset: r.offsetBytes, size: r.sizeBytes } });
        }
      }
      entries.push({ binding: this.segCap, resource: { buffer: dstBuffer, offset: dstOffsetBytes, size: totalBytes } });
      entries.push({ binding: this.segCap + 1, resource: { buffer: this._uniform, offset: 0, size: this._uniform.size } });
      const bg = this.device.createBindGroup({ layout: this._layout, entries });

      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(totalWords / this.workgroupSize), 1, 1);
      pass.end();
      return;
    }

    // Tier B fallback: multiple GPU copy commands (still once per exec, not per stage)
    let dst = dstOffsetBytes;
    for (const r of srcRanges) {
      commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
      dst += r.sizeBytes;
    }
  }

  unpack(commandEncoder, srcBuffer, srcOffsetBytes, dstRanges) {
    const totalBytes = dstRanges.reduce((a, r) => a + r.sizeBytes, 0);
    if (totalBytes % 4 !== 0) throw new Error("SegmentedCopier only supports totalBytes multiple of 4");
    const totalWords = totalBytes >>> 2;

    if (dstRanges.length <= this.segCap) {
      const canTierA =
        srcOffsetBytes % this.storageAlign === 0 && dstRanges.every((r) => r.offsetBytes % this.storageAlign === 0);
      if (!canTierA) {
        // Fall back to Tier-B copies if bindings would violate offset alignment.
        let src = srcOffsetBytes;
        for (const r of dstRanges) {
          commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
          src += r.sizeBytes;
        }
        return;
      }

      const segSizesWords = new Uint32Array(this.segCap);
      const segPrefixWords = new Uint32Array(this.segCap);
      let prefix = 0;
      for (let i = 0; i < dstRanges.length; i++) {
        if (dstRanges[i].sizeBytes % 4 !== 0) throw new Error("Segment size must be multiple of 4");
        segSizesWords[i] = dstRanges[i].sizeBytes >>> 2;
        segPrefixWords[i] = prefix;
        prefix += segSizesWords[i];
      }

      const header = new Uint32Array([dstRanges.length, totalWords, 0, 0]);
      const ub = new Uint32Array(4 + this.segCap * 2);
      ub.set(header, 0);
      ub.set(segSizesWords, 4);
      ub.set(segPrefixWords, 4 + this.segCap);
      this.device.queue.writeBuffer(this._uniform, 0, ub);

      const code = generateSegmentedCopyWGSL({
        cap: this.segCap,
        direction: "unpack",
        workgroupSize: this.workgroupSize,
      });
      const pipeline = this.cache.getComputePipeline({ code, layout: this._plUnpack });

      const entries = [];
      for (let i = 0; i < this.segCap; i++) {
        const r = dstRanges[i];
        if (!r) {
          entries.push({ binding: i, resource: { buffer: this._dummyRW[i], offset: 0, size: 4 } });
        } else {
          entries.push({ binding: i, resource: { buffer: r.buffer, offset: r.offsetBytes, size: r.sizeBytes } });
        }
      }
      entries.push({ binding: this.segCap, resource: { buffer: srcBuffer, offset: srcOffsetBytes, size: totalBytes } });
      entries.push({ binding: this.segCap + 1, resource: { buffer: this._uniform, offset: 0, size: this._uniform.size } });
      const bg = this.device.createBindGroup({ layout: this._layoutUnpack, entries });

      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(totalWords / this.workgroupSize), 1, 1);
      pass.end();
      return;
    }

    // Tier B fallback: multiple copies
    let src = srcOffsetBytes;
    for (const r of dstRanges) {
      commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
      src += r.sizeBytes;
    }
  }
}
