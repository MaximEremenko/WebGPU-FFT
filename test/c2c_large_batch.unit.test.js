import test from "node:test";
import assert from "node:assert/strict";

import { createPlan } from "../src/runtime/create_plan.js";
import {
  createFftConvChannelLanePreset,
  createFftConvKernelMajorChannelLanePreset,
  createFftConvBatchMajorChannelLanePreset,
} from "../src/runtime/fftconv_channel_lane_presets.js";
import { BufferView } from "../src/utils/buffer_view.js";

if (!globalThis.GPUBufferUsage) {
  globalThis.GPUBufferUsage = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
  };
}
if (!globalThis.GPUShaderStage) {
  globalThis.GPUShaderStage = {
    VERTEX: 0x1,
    FRAGMENT: 0x2,
    COMPUTE: 0x4,
  };
}

class FakeDevice {
  constructor(limitOverrides = {}) {
    this.limits = {
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 30,
      maxComputeWorkgroupSizeX: 4,
      maxComputeWorkgroupSizeY: 4,
      maxComputeWorkgroupSizeZ: 4,
      maxComputeInvocationsPerWorkgroup: 4,
      maxComputeWorkgroupStorageSize: 32 * 1024,
      minStorageBufferOffsetAlignment: 1,
      maxComputeWorkgroupsPerDimension: [65535, 65535, 65535],
      ...limitOverrides,
    };
    this.features = { has: () => false };
    this._id = 1;
    this.queue = {
      writes: [],
      submits: [],
      writeBuffer: (buffer, offset, data) => {
        const bytes = new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
        this.queue.writes.push({ buffer, offset, bytes });
      },
      submit: (cmds) => {
        this.queue.submits.push(cmds);
      },
    };
  }

  createBindGroupLayout(desc) {
    return { kind: "bgl", id: this._id++, desc };
  }

  createPipelineLayout(desc) {
    return { kind: "pl", id: this._id++, desc };
  }

  createShaderModule(desc) {
    return { kind: "sm", id: this._id++, code: desc.code };
  }

  createComputePipeline(desc) {
    return { kind: "cp", id: this._id++, desc };
  }

  createBuffer(desc) {
    return {
      kind: "buf",
      id: this._id++,
      size: desc.size,
      usage: desc.usage,
      destroy() {},
    };
  }

  createBindGroup(desc) {
    return { kind: "bg", id: this._id++, desc };
  }

  createCommandEncoder() {
    return makeEncoder();
  }
}

function makeEncoder() {
  const ops = [];
  return {
    ops,
    copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
      ops.push({ type: "copy", src, srcOffset, dst, dstOffset, size });
    },
    beginComputePass() {
      const pass = { type: "compute", dispatches: [] };
      ops.push(pass);
      return {
        setPipeline(p) {
          pass.pipeline = p;
        },
        setBindGroup(i, bg) {
          pass.bindGroupIndex = i;
          pass.bindGroup = bg;
        },
        dispatchWorkgroups(x, y, z) {
          pass.dispatches.push({ x, y, z });
        },
        end() {},
      };
    },
    finish() {
      return { ops: [...ops] };
    },
  };
}

function makeSegmentedView(buffer, totalBytes, parts) {
  if ((totalBytes & 3) !== 0) {
    throw new Error(`makeSegmentedView requires totalBytes multiple of 4, got ${totalBytes}`);
  }
  const segs = [];
  let off = 0;
  const align = 4;
  for (let i = 0; i < parts; i++) {
    const remain = totalBytes - off;
    let chunk;
    if (i === parts - 1) {
      chunk = remain;
    } else {
      const denom = parts - i;
      const minLeft = align * (parts - i - 1);
      const target = Math.floor(remain / denom);
      chunk = Math.floor(target / align) * align;
      if (chunk < align) chunk = align;
      const maxChunk = remain - minLeft;
      if (chunk > maxChunk) chunk = maxChunk;
    }
    segs.push({ buffer, offsetBytes: off, sizeBytes: chunk });
    off += chunk;
  }
  return {
    segments: segs,
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };
}

function assertComputeBindingsWithinLimit(encoder, maxBytes) {
  for (const op of encoder.ops) {
    if (op.type !== "compute") continue;
    const entries = op.bindGroup?.desc?.entries ?? [];
    for (const e of entries) {
      const size = e?.resource?.size;
      if (typeof size === "number") {
        assert.ok(
          size <= maxBytes,
          `compute binding ${e.binding} exceeded maxBytes=${maxBytes}: got ${size}`
        );
      }
    }
  }
}

function computePassUsesBuffer(encoder, buffer) {
  for (const op of encoder.ops) {
    if (op.type !== "compute") continue;
    const entries = op.bindGroup?.desc?.entries ?? [];
    for (const e of entries) {
      if (e?.resource?.buffer === buffer) return true;
    }
  }
  return false;
}

test("c2c large-batch chunk mode executes when bytesPerBatch fits binding", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const totalBytes = shape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, true);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();

  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.equal(computePasses.length, 2); // one radix-8 stage, chunked as 4+1 batches
});

test("c2c large-batch chunk mode supports normalize!=none via low-level axis plans", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const totalBytes = shape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "inverse",
    batch,
    inPlace: false,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan.axisPlans[0].config.normalize, "backward");

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();

  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.equal(computePasses.length, 2);
});

test("c2c out-of-core four-step mode executes when one batch exceeds binding limit", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 3];
  const batch = 5;
  const totalBytes = shape[0] * shape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "inverse",
    batch,
    inPlace: false,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.ok(plan.axisPlans[0]);
  assert.equal(plan.axisPlans[1], null);
  assert.ok(plan._outOfCoreAxis0OnTransposed);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();

  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 2);
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(copyOps.length > 0);
  const scaleWrites = device.queue.writes.filter((w) => w.buffer === plan.scale.params || w.buffer === plan._scaleChunkParamsBuffer);
  assert.ok(scaleWrites.length >= 2);
});

test("c2c transpose fast path supports rank>2 via batched axis-0/axis-1 tiles", () => {
  const device = new FakeDevice();
  const shape = [48, 96, 2];
  const batch = 3;
  const totalBytes = shape[0] * shape[1] * shape[2] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, false);
  assert.ok(plan.transpose);
  assert.equal(plan.transpose.matrixBatch, batch * shape[2]);
  assert.deepEqual(plan.axis0OnTransposed.config.shape, [shape[1], shape[0], shape[2]]);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const dispatches = encoder.ops
    .filter((op) => op.type === "compute")
    .flatMap((op) => op.dispatches ?? []);
  assert.ok(dispatches.some((d) => d.z === batch * shape[2]));
});

test("c2c out-of-core four-step mode supports segmented input/output", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 3];
  const batch = 5;
  const totalBytes = shape[0] * shape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._outOfCoreFourStepMode, true);

  const inA = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const inB = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outA = device.createBuffer({
    size: totalBytes / 3,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outB = device.createBuffer({
    size: totalBytes / 3,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outC = device.createBuffer({
    size: totalBytes - 2 * Math.floor(totalBytes / 3),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const input = {
    segments: [
      { buffer: inA, offsetBytes: 0, sizeBytes: totalBytes / 2 },
      { buffer: inB, offsetBytes: 0, sizeBytes: totalBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };
  const output = {
    segments: [
      { buffer: outA, offsetBytes: 0, sizeBytes: Math.floor(totalBytes / 3) },
      { buffer: outB, offsetBytes: 0, sizeBytes: Math.floor(totalBytes / 3) },
      { buffer: outC, offsetBytes: 0, sizeBytes: totalBytes - 2 * Math.floor(totalBytes / 3) },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core four-step mode supports non-trivial ioView + zeroPad", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 3];
  const batch = 5;
  const inViewShape = [2, 3];
  const outViewShape = [2, 3];
  const inBytes = inViewShape[0] * inViewShape[1] * batch * 8;
  const outBytes = outViewShape[0] * outViewShape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    ioView: {
      input: { shape: inViewShape, placement: "center" },
      output: { shape: outViewShape, placement: "center", clearOutside: false },
    },
    zeroPad: {
      read: { start: [1, 0], end: [3, 3] },
      write: { start: [0, 1], end: [4, 2] },
    },
  });

  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.equal(plan._needsInputMapping, true);
  assert.equal(plan._needsOutputMapping, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const inA = device.createBuffer({
    size: inBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const inB = device.createBuffer({
    size: inBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outA = device.createBuffer({
    size: outBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outB = device.createBuffer({
    size: outBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const input = {
    segments: [
      { buffer: inA, offsetBytes: 0, sizeBytes: inBytes / 2 },
      { buffer: inB, offsetBytes: 0, sizeBytes: inBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: inBytes,
  };
  const output = {
    segments: [
      { buffer: outA, offsetBytes: 0, sizeBytes: outBytes / 2 },
      { buffer: outB, offsetBytes: 0, sizeBytes: outBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: outBytes,
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 2);
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core four-step mode supports rank-3 by axis permutation staging", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [2, 2, 5];
  const batch = 2;
  const totalBytes = shape[0] * shape[1] * shape[2] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.ok(plan._outOfCoreAxisPlans);
  assert.equal(plan._outOfCoreAxisPlans.length, 3);
  assert.ok(plan._outOfCoreAxisPlans[2]);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 3);
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core rank-3 axis-2 permutation uses tiled compute windows to avoid scalar copy loops", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 32768, minStorageBufferOffsetAlignment: 256 });
  const shape = [32, 32, 32];
  const batch = 1;
  const totalBytes = shape[0] * shape[1] * shape[2] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 32768 },
  });

  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.ok(plan._outOfCoreRank3Axis2Permute);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const scalarCopies = encoder.ops.filter((op) => op.type === "copy" && op.size === 8);
  assert.equal(scalarCopies.length, 0);
});

test("c2c out-of-core rank-generic permutation uses compute path when total bytes exceed bind but per-batch fits", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 600, minStorageBufferOffsetAlignment: 1 });
  const shape = [3, 2, 2, 5];
  const batch = 2;
  const totalBytes = shape[0] * shape[1] * shape[2] * shape[3] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: {
      largeRoute: "out-of-core",
      maxStorageBufferBindingSize: 600,
    },
  });

  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.equal(plan._outOfCoreSegmentedFullVolumeMode, false);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const scalarCopies = encoder.ops.filter((op) => op.type === "copy" && op.size === 8);
  assert.equal(scalarCopies.length, 0);
});

test("c2c out-of-core rank-generic axis permutation uses compute adjacent-swap path when generic bind fails", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 160, minStorageBufferOffsetAlignment: 1 });
  const shape = [2, 5, 3, 2];
  const batch = 1;
  const totalBytes = shape[0] * shape[1] * shape[2] * shape[3] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: {
      largeRoute: "out-of-core",
      maxStorageBufferBindingSize: 160,
    },
  });

  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.equal(plan._outOfCoreSegmentedFullVolumeMode, false);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const scalarCopies = encoder.ops.filter((op) => op.type === "copy" && op.size === 8);
  assert.equal(scalarCopies.length, 0);
});

test("c2c out-of-core ultra-low-bind axis permutation uses tiled compute fallback (no scalar copy loop)", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;
  const totalBytes = shape[0] * shape[1] * shape[2] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: {
      maxStorageBufferBindingSize: 64,
    },
  });

  assert.equal(plan._outOfCoreFourStepMode, true);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const scalarCopies = encoder.ops.filter((op) => op.type === "copy" && op.size === 8);
  assert.equal(scalarCopies.length, 0);
});

test("c2c tuning.maxStorageBufferBindingSize can force out-of-core scheduling", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 20, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 3, 2];
  const batch = 1;
  const totalBytes = shape[0] * shape[1] * shape[2] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  assert.equal(plan._maxBindBytes, 64);
  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 3);
});

test('c2c tuning.largeRoute="out-of-core" forces out-of-core when not otherwise required', () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 3];
  const batch = 3; // total requires large mode, per-batch still fits => normally chunk route
  const totalBytes = shape[0] * shape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { largeRoute: "out-of-core" },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.equal(plan._largeRouteMode, "large-out-of-core");
  assert.ok(plan._largeRouteReasons.includes("forced-route-out-of-core"));

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
});

test("c2c out-of-core advanced-axis window policy applies staged upload tuning", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 320, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 17];
  const batch = 2;
  const totalBytes = shape[0] * shape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: {
      maxStorageBufferBindingSize: 320,
      forceBluesteinAxes: [1],
      swapTo2Stage4Step: 16,
      groupedBatch: [null, 4],
      outOfCoreBurstWindows: 2,
    },
  });

  assert.equal(plan._outOfCoreFourStepMode, true);
  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const policy = plan._outOfCoreAxisWindowPolicy?.[1];
  assert.ok(policy, "expected axis-1 out-of-core window policy metadata");
  assert.equal(policy.numAxisUploads, 2);
  assert.equal(policy.groupedBatch, 4);
  assert.equal(policy.burstWindows, 2);
  assert.ok(policy.linesPerChunk >= 1);
});

test("c2c out-of-core mixed-axis plans apply staged bind cap from upload policy", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 20, minStorageBufferOffsetAlignment: 1 });
  const shape = [8, 8];
  const batch = 1;
  const totalBytes = shape[0] * shape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: {
      maxStorageBufferBindingSize: 256,
      swapTo2Stage4Step: 8,
    },
  });

  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.equal(plan._outOfCoreAxisWindowPolicy?.[0]?.numAxisUploads, 2);
  assert.equal(plan._outOfCoreAxisWindowPolicy?.[1]?.numAxisUploads, 2);
  assert.equal(plan._outOfCoreAxisPlans?.[0]?.config?.maxStorageBufferBindingSize, 128);
  assert.equal(plan._outOfCoreAxisPlans?.[1]?.config?.maxStorageBufferBindingSize, 128);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
});

test('c2c tuning.largeRoute="chunk" rejects when out-of-core is required', () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "c2c",
        shape: [4, 3],
        direction: "forward",
        batch: 5,
        inPlace: false,
        normalize: "none",
        layout: { interleavedComplex: true },
        precision: "f32",
        tuning: { largeRoute: "chunk" },
      }),
    /largeRoute="chunk" is incompatible/
  );
});

test("c2c tuning.largeChunkMaxBatches bounds large chunk batch size", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const inputViewShape = [10];
  const inputViewBytes = inputViewShape[0] * batch * 8;
  const outputBytes = shape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    ioView: {
      input: { shape: inputViewShape, placement: "center" },
    },
    tuning: { largeChunkMaxBatches: 2 },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan.tuning.largeChunkMaxBatches, 2);

  const input = device.createBuffer({
    size: inputViewBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const embedWrites = device.queue.writes
    .filter((w) => w.buffer === plan.ioEmbed.params)
    .map((w) => new Uint32Array(w.bytes.buffer, w.bytes.byteOffset, 4)[2]);
  // Expected chunk split: 2 + 2 + 1 batches.
  assert.deepEqual(embedWrites, [2, 2, 1]);
});

test("c2c out-of-core supports oversized axis line via low-level axis-0 two-step fallback", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [32, 4];
  const batch = 2;
  const totalBytes = shape[0] * shape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.ok(plan._outOfCoreAxisPlans?.[0]?._axis0TwoStep);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core four-step supports forced Bluestein on non-axis0 path", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 320, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 17];
  const batch = 2;
  const totalBytes = shape[0] * shape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { forceBluesteinAxes: [1] },
  });

  assert.equal(plan.axisKind[1], "bluestein");
  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.ok(plan._outOfCoreAxisPlans?.[1]);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core four-step supports forced Rader on non-axis0 path", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 480, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 29];
  const batch = 2;
  const totalBytes = shape[0] * shape[1] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { forceRaderAxes: [1] },
  });

  assert.equal(plan.axisKind[1], "rader");
  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.ok(plan._outOfCoreAxisPlans?.[1]);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core non-mixed oversized axis runs via bounded sliced-line uploads", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 200, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 31];
  const batch = 2;
  const totalBytes = shape[0] * shape[1] * batch * 8;
  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { forceRaderAxes: [1] },
  });
  assert.equal(plan.axisKind[1], "rader");
  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.equal(plan._outOfCoreAxisEffectiveKind?.[1], "bluestein-fallback");

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
  assertComputeBindingsWithinLimit(encoder, 200);
});

test("c2c out-of-core four-step mode can be disabled by tuning", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "c2c",
        shape: [4, 3],
        direction: "forward",
        batch: 5,
        inPlace: false,
        normalize: "none",
        layout: { interleavedComplex: true },
        precision: "f32",
        tuning: { disableOutOfCoreFourStep: true },
      }),
    /Out-of-core fallback is available only/
  );
});

test("c2c large-batch chunk mode supports ioView.input mapping", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const logicalBytes = shape[0] * batch * 8;
  const inputViewShape = [10];
  const inputViewBytes = inputViewShape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    ioView: {
      input: { shape: inputViewShape, placement: "center" },
    },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._needsInputMapping, true);
  assert.equal(plan._inPhysBytes, inputViewBytes);

  const input = device.createBuffer({
    size: inputViewBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: logicalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();

  plan.exec(encoder, { input, output });

  const embedWrites = device.queue.writes.filter((w) => w.buffer === plan.ioEmbed.params);
  assert.equal(embedWrites.length, 2);
  assert.deepEqual(
    embedWrites.map((w) => new Uint32Array(w.bytes.buffer, w.bytes.byteOffset, 4)[2]),
    [3, 2]
  );
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 4);
});

test("c2c large-batch chunk mode supports ioView.output mapping with segmented output", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const logicalBytes = shape[0] * batch * 8;
  const outputViewShape = [10];
  const outputViewBytes = outputViewShape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    ioView: {
      output: { shape: outputViewShape, placement: "center", clearOutside: false },
    },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._needsOutputMapping, true);
  assert.equal(plan._outPhysBytes, outputViewBytes);

  const input = device.createBuffer({
    size: logicalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outA = device.createBuffer({
    size: outputViewBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outB = device.createBuffer({
    size: outputViewBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = {
    segments: [
      { buffer: outA, offsetBytes: 0, sizeBytes: outputViewBytes / 2 },
      { buffer: outB, offsetBytes: 0, sizeBytes: outputViewBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: outputViewBytes,
  };
  const encoder = makeEncoder();

  plan.exec(encoder, { input, output });

  const extractWrites = device.queue.writes.filter((w) => w.buffer === plan.ioExtract.params);
  assert.equal(extractWrites.length, 2);
  assert.deepEqual(
    extractWrites.map((w) => new Uint32Array(w.bytes.buffer, w.bytes.byteOffset, 4)[2]),
    [3, 2]
  );
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 4);
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(copyOps.length > 0);
});

test("c2c large-batch chunk mode supports zeroPad.read/zeroPad.write", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const totalBytes = shape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    zeroPad: {
      read: { start: [1], end: [7] },
      write: { start: [2], end: [6] },
    },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 4);
});

test("c2c large-batch chunk mode supports segmented input/output with staging temp", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const totalBytes = shape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  const inA = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const inB = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outA = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outB = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const temp = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const input = {
    segments: [
      { buffer: inA, offsetBytes: 0, sizeBytes: totalBytes / 2 },
      { buffer: inB, offsetBytes: 0, sizeBytes: totalBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };
  const output = {
    segments: [
      { buffer: outA, offsetBytes: 0, sizeBytes: totalBytes / 2 },
      { buffer: outB, offsetBytes: 0, sizeBytes: totalBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, temp });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 2);
});

test("c2c large-batch chunk mode auto-allocates internal staging when temp is omitted", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const totalBytes = shape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  const inA = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const inB = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const out = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = {
    segments: [
      { buffer: inA, offsetBytes: 0, sizeBytes: totalBytes / 2 },
      { buffer: inB, offsetBytes: 0, sizeBytes: totalBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output: out });
  assert.ok(plan._largeStageBuffer);
  assert.equal(plan._largeStageBytes, totalBytes);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length >= 2);
});

test("c2c large-batch chunk mode supports segmented staging temp without internal contiguous fallback", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;
  const totalBytes = shape[0] * batch * 8;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  const inA = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const inB = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const out = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const tA = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const tB = device.createBuffer({
    size: totalBytes / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const input = {
    segments: [
      { buffer: inA, offsetBytes: 0, sizeBytes: totalBytes / 2 },
      { buffer: inB, offsetBytes: 0, sizeBytes: totalBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };
  const temp = {
    segments: [
      { buffer: tA, offsetBytes: 0, sizeBytes: totalBytes / 2 },
      { buffer: tB, offsetBytes: 0, sizeBytes: totalBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output: out, temp });
  assert.equal(plan._largeStageBuffer, null);
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
});

test("c2c out-of-core four-step mode supports custom input/output strides", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "inverse",
    batch,
    inPlace: false,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 11, 37],
      outputStrides: [3, 17, 61],
      inputOffsetElements: 3,
      outputOffsetElements: 4,
      inputBatchStrideElements: 128,
      outputBatchStrideElements: 176,
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 8, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core four-step mode supports custom strides with segmented input/output", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "inverse",
    batch,
    inPlace: false,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 11, 37],
      outputStrides: [3, 17, 61],
      inputOffsetElements: 3,
      outputOffsetElements: 4,
      inputBatchStrideElements: 128,
      outputBatchStrideElements: 176,
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);

  const inputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 4096, 7);
  const output = makeSegmentedView(outputBuf, 4096, 6);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 8, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core four-step mode falls back when temp aliases data buffer", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "inverse",
    batch,
    inPlace: false,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  // temp aliases output, which is selected as dataRange in this configuration.
  plan.exec(encoder, { input, output, temp: output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core four-step mode supports custom strides with non-trivial ioView", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 29],
      outputStrides: [3, 11, 43],
      inputOffsetElements: 3,
      outputOffsetElements: 5,
      inputBatchStrideElements: 80,
      outputBatchStrideElements: 128,
    },
    ioView: {
      input: { shape: [3, 3, 2], offset: [1, 0, 0] },
      output: { shape: [7, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 24 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c out-of-core four-step mode supports custom strides + ioView + zeroPad", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 29],
      outputStrides: [3, 11, 43],
      inputOffsetElements: 3,
      outputOffsetElements: 5,
      inputBatchStrideElements: 80,
      outputBatchStrideElements: 128,
    },
    ioView: {
      input: { shape: [3, 3, 2], offset: [1, 0, 0] },
      output: { shape: [7, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    zeroPad: {
      read: { start: [1, 0, 0], end: [4, 3, 2] },
      write: { start: [0, 0, 0], end: [4, 3, 2] },
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 24 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
  assert.ok(copyOps.some((op) => op.size === 4 || op.size === 8));
});

test("c2c regular mode supports custom input/output strides with non-trivial ioView", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 29],
      outputStrides: [3, 11, 43],
      inputOffsetElements: 3,
      outputOffsetElements: 5,
      inputBatchStrideElements: 80,
      outputBatchStrideElements: 128,
    },
    ioView: {
      input: { shape: [3, 3, 2], offset: [1, 0, 0] },
      output: { shape: [7, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, false);
  assert.equal(plan._outOfCoreFourStepMode, false);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 24 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c regular mode supports custom strides + ioView + zeroPad", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 29],
      outputStrides: [3, 11, 43],
      inputOffsetElements: 3,
      outputOffsetElements: 5,
      inputBatchStrideElements: 80,
      outputBatchStrideElements: 128,
    },
    ioView: {
      input: { shape: [3, 3, 2], offset: [1, 0, 0] },
      output: { shape: [7, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    zeroPad: {
      read: { start: [1, 0, 0], end: [4, 3, 2] },
      write: { start: [0, 0, 0], end: [4, 3, 2] },
    },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, false);
  assert.equal(plan._outOfCoreFourStepMode, false);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 24 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
  assert.ok(copyOps.some((op) => op.size === 4 || op.size === 8));
});

test("c2c regular mode supports custom input/output strides with segmented input/output", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 11, 37],
      outputStrides: [3, 13, 41],
      inputOffsetElements: 3,
      outputOffsetElements: 4,
      inputBatchStrideElements: 128,
      outputBatchStrideElements: 176,
    },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, false);
  assert.equal(plan._outOfCoreFourStepMode, false);
  assert.equal(plan._needsInputMapping, false);
  assert.equal(plan._needsOutputMapping, false);

  const inputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 4096, 7);
  const output = makeSegmentedView(outputBuf, 4096, 6);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 24 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c large-batch chunk mode supports custom strides when per-batch bytes fit (no out-of-core)", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "c2c",
    shape: [8],
    direction: "forward",
    batch: 5,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2],
      outputStrides: [3],
      inputOffsetElements: 1,
      outputOffsetElements: 2,
      inputBatchStrideElements: 20,
      outputBatchStrideElements: 24,
    },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, false);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, {
    input,
    output,
    inputOffsetBytes: 8,
    outputOffsetBytes: 16,
  });

  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
});

test("c2c large-batch chunk mode supports custom strides with segmented input/output when out-of-core is not active", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "c2c",
    shape: [8],
    direction: "forward",
    batch: 5,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2],
      outputStrides: [3],
      inputOffsetElements: 1,
      outputOffsetElements: 2,
      inputBatchStrideElements: 20,
      outputBatchStrideElements: 24,
    },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, false);

  const inputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 4096, 5);
  const output = makeSegmentedView(outputBuf, 4096, 6);

  const encoder = makeEncoder();
  plan.exec(encoder, {
    input,
    output,
    inputOffsetBytes: 8,
    outputOffsetBytes: 16,
  });

  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
});

test("c2c large-batch with custom strides auto-routes to out-of-core four-step when eligible", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 100, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "c2c",
    shape: [4, 3],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9],
      outputStrides: [3, 10],
      inputOffsetElements: 1,
      outputOffsetElements: 2,
      inputBatchStrideElements: 40,
      outputBatchStrideElements: 45,
    },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, true);

  const input = device.createBuffer({
    size: 2048,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 2048,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, {
    input,
    output,
    inputOffsetBytes: 16,
    outputOffsetBytes: 24,
  });

  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
});

test("r2c large-shape fallback executes when bindings exceed cap", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const inViewShape = [3, 3, 2];
  const outViewShape = [4, 3, 2];
  const inBytes = inViewShape.reduce((a, b) => a * b, 1) * 4;
  const outBytes = outViewShape.reduce((a, b) => a * b, 1) * 8;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch: 1,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    ioView: {
      input: { shape: inViewShape, offset: [1, 0, 0] },
      output: { shape: outViewShape, offset: [-1, 0, 0], clearOutside: false },
    },
    zeroPad: {
      read: { start: [1, 0, 0], end: [5, 3, 2] },
      write: { start: [0, 0, 0], end: [2, 3, 2] },
    },
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);

  const inA = device.createBuffer({ size: inBytes / 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const inB = device.createBuffer({ size: inBytes - inBytes / 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const outA = device.createBuffer({ size: Math.floor(outBytes / 3), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const outB = device.createBuffer({ size: Math.floor(outBytes / 3), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const outC = device.createBuffer({
    size: outBytes - 2 * Math.floor(outBytes / 3),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const input = {
    segments: [
      { buffer: inA, offsetBytes: 0, sizeBytes: inBytes / 2 },
      { buffer: inB, offsetBytes: 0, sizeBytes: inBytes - inBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: inBytes,
  };
  const output = {
    segments: [
      { buffer: outA, offsetBytes: 0, sizeBytes: Math.floor(outBytes / 3) },
      { buffer: outB, offsetBytes: 0, sizeBytes: Math.floor(outBytes / 3) },
      { buffer: outC, offsetBytes: 0, sizeBytes: outBytes - 2 * Math.floor(outBytes / 3) },
    ],
    logicalByteOffset: 0,
    lengthBytes: outBytes,
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r large-shape fallback executes when bindings exceed cap", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const inViewShape = [2, 3, 2];
  const outViewShape = [7, 3, 2];
  const inBytes = inViewShape.reduce((a, b) => a * b, 1) * 8;
  const outBytes = outViewShape.reduce((a, b) => a * b, 1) * 4;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch: 1,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
    ioView: {
      input: { shape: inViewShape, offset: [1, 0, 0] },
      output: { shape: outViewShape, offset: [-1, 0, 0], clearOutside: false },
    },
    zeroPad: {
      read: { start: [1, 0, 0], end: [3, 3, 2] },
      write: { start: [1, 0, 0], end: [5, 3, 2] },
    },
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);

  const inA = device.createBuffer({ size: inBytes / 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const inB = device.createBuffer({ size: inBytes - inBytes / 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const outA = device.createBuffer({ size: Math.floor(outBytes / 2), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const outB = device.createBuffer({ size: outBytes - Math.floor(outBytes / 2), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

  const input = {
    segments: [
      { buffer: inA, offsetBytes: 0, sizeBytes: inBytes / 2 },
      { buffer: inB, offsetBytes: 0, sizeBytes: inBytes - inBytes / 2 },
    ],
    logicalByteOffset: 0,
    lengthBytes: inBytes,
  };
  const output = {
    segments: [
      { buffer: outA, offsetBytes: 0, sizeBytes: Math.floor(outBytes / 2) },
      { buffer: outB, offsetBytes: 0, sizeBytes: outBytes - Math.floor(outBytes / 2) },
    ],
    logicalByteOffset: 0,
    lengthBytes: outBytes,
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c large-shape fallback supports oversized axis-0 line via segmented multi-upload staging", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [32, 3];
  const batch = 1;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  assert.equal(plan._largeShapeMode, true);
  assert.equal(plan._oversizedLineMode, true);

  const inRaw = device.createBuffer({
    size: plan.inBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outRaw = device.createBuffer({
    size: plan.outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inRaw, plan.inBytes, 7);
  const output = makeSegmentedView(outRaw, plan.outBytes, 6);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r large-shape fallback supports oversized axis-0 line via segmented multi-upload staging", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [32, 3];
  const batch = 1;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });

  assert.equal(plan._largeShapeMode, true);
  assert.equal(plan._oversizedLineMode, true);

  const inRaw = device.createBuffer({
    size: plan.inBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outRaw = device.createBuffer({
    size: plan.outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inRaw, plan.inBytes, 6);
  const output = makeSegmentedView(outRaw, plan.outBytes, 5);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c large-shape path exposes shared out-of-core axis window policy metadata", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1024, minStorageBufferOffsetAlignment: 256 });
  const plan = createPlan(device, {
    type: "r2c",
    shape: [16, 8],
    direction: "forward",
    batch: 2,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: {
      maxStorageBufferBindingSize: 1024,
      swapTo2Stage4Step: 8,
      groupedBatch: 4,
      outOfCoreBurstWindows: 3,
    },
  });
  assert.equal(plan._largeShapeMode, true);
  const realToComplex = plan._outOfCoreAxisWindowPolicy?.realToComplex;
  const pack = plan._outOfCoreAxisWindowPolicy?.pack;
  assert.ok(realToComplex, "expected realToComplex window policy");
  assert.ok(pack, "expected pack window policy");
  assert.equal(realToComplex.numAxisUploads, 2);
  assert.equal(pack.numAxisUploads, 2);
  assert.equal(realToComplex.groupedBatch, 4);
  assert.equal(pack.groupedBatch, 4);
  assert.equal(realToComplex.burstWindows, 3);
  assert.equal(pack.burstWindows, 3);
  assert.ok(realToComplex.linesPerChunk >= 1);
  assert.ok(pack.linesPerChunk >= 1);
});

test("c2r large-shape path exposes shared out-of-core axis window policy metadata", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1024, minStorageBufferOffsetAlignment: 256 });
  const plan = createPlan(device, {
    type: "c2r",
    shape: [16, 8],
    direction: "inverse",
    batch: 2,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
    tuning: {
      maxStorageBufferBindingSize: 1024,
      swapTo2Stage4Step: 8,
      groupedBatch: 4,
      outOfCoreBurstWindows: 3,
    },
  });
  assert.equal(plan._largeShapeMode, true);
  const unpack = plan._outOfCoreAxisWindowPolicy?.unpack;
  const complexToReal = plan._outOfCoreAxisWindowPolicy?.complexToReal;
  assert.ok(unpack, "expected unpack window policy");
  assert.ok(complexToReal, "expected complexToReal window policy");
  assert.equal(unpack.numAxisUploads, 2);
  assert.equal(complexToReal.numAxisUploads, 2);
  assert.equal(unpack.groupedBatch, 4);
  assert.equal(complexToReal.groupedBatch, 4);
  assert.equal(unpack.burstWindows, 3);
  assert.equal(complexToReal.burstWindows, 3);
  assert.ok(unpack.linesPerChunk >= 1);
  assert.ok(complexToReal.linesPerChunk >= 1);
});

test("r2c large-shape oversized-axis mode rejects when axis-0 is not two-step factorable", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "r2c",
        shape: [17, 3],
        direction: "forward",
        batch: 1,
        normalize: "none",
        layout: { interleavedComplex: true },
        precision: "f32",
        tuning: { maxStorageBufferBindingSize: 64 },
      }),
    /mixed-radix|Out-of-core fallback|axisSupported|work buffer|required binding|shape=|two-step fallback/
  );
});

test("c2r large-shape oversized-axis mode rejects when axis-0 is not two-step factorable", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "c2r",
        shape: [17, 3],
        direction: "inverse",
        batch: 1,
        normalize: "backward",
        layout: { interleavedComplex: true },
        precision: "f32",
        tuning: { maxStorageBufferBindingSize: 64 },
      }),
    /mixed-radix|Out-of-core fallback|axisSupported|work buffer|required binding|shape=|two-step fallback/
  );
});

test("r2c large-shape fallback uses split internal workspace when monolithic temp exceeds maxBufferSize", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 20000,
    maxBufferSize: 45000,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [9, 8, 8];
  const batch = 6;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeShapeMode, true);
  assert.equal(plan._arena, null);
  assert.ok(plan._splitWorkspace?.real);
  assert.ok(plan._splitWorkspace?.full);
  assert.ok(plan._splitWorkspace?.packed);

  const input = device.createBuffer({
    size: plan.inBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: plan.outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r large-shape fallback uses split internal workspace when monolithic temp exceeds maxBufferSize", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 20000,
    maxBufferSize: 45000,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [9, 8, 8];
  const batch = 6;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeShapeMode, true);
  assert.equal(plan._arena, null);
  assert.ok(plan._splitWorkspace?.packed);
  assert.ok(plan._splitWorkspace?.full);
  assert.ok(plan._splitWorkspace?.real);

  const input = device.createBuffer({
    size: plan.inBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: plan.outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c large-shape fallback executes with segmented temp without internal contiguous fallback", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 20000,
    maxBufferSize: 45000,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [9, 8, 8];
  const batch = 6;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeShapeMode, true);
  const input = device.createBuffer({
    size: plan.inBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: plan.outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const tempBuf = device.createBuffer({
    size: plan.workspaceBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const temp = new BufferView(makeSegmentedView(tempBuf, plan.workspaceBytes, 11));

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, temp });

  assert.equal(plan._largeWorkspaceFallback, undefined);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r large-shape fallback executes with segmented temp without internal contiguous fallback", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 20000,
    maxBufferSize: 45000,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [9, 8, 8];
  const batch = 6;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeShapeMode, true);
  const input = device.createBuffer({
    size: plan.inBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: plan.outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const tempBuf = device.createBuffer({
    size: plan.workspaceBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const temp = new BufferView(makeSegmentedView(tempBuf, plan.workspaceBytes, 11));

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, temp });

  assert.equal(plan._largeWorkspaceFallback, undefined);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c regular mode uses split internal workspace when monolithic temp exceeds maxBufferSize", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 100000,
    maxBufferSize: 45000,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [9, 8, 8];
  const batch = 6;
  const inputViewShape = [5, 8, 8];

  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
    ioView: {
      input: { shape: inputViewShape, offset: [2, 0, 0] },
    },
  });

  assert.equal(plan._largeBatchChunkMode, false);
  assert.equal(plan._outOfCoreFourStepMode, false);
  assert.equal(plan._arena, null);
  assert.ok(plan._splitWorkspace?.mainStage);
  assert.ok(plan._splitWorkspace?.scratch);

  const inputBytes = inputViewShape.reduce((a, b) => a * b, 1) * batch * 8;
  const outputBytes = shape.reduce((a, b) => a * b, 1) * batch * 8;
  const input = device.createBuffer({
    size: inputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c regular mode uses split internal workspace when monolithic temp exceeds maxBufferSize", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 100000,
    maxBufferSize: 45000,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [9, 8, 8];
  const batch = 6;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeShapeMode, false);
  assert.equal(plan._arena, null);
  assert.ok(plan._splitWorkspace?.real);
  assert.ok(plan._splitWorkspace?.full);
  assert.ok(plan._splitWorkspace?.packed);

  const input = device.createBuffer({
    size: plan.inBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: plan.outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r regular mode uses split internal workspace when monolithic temp exceeds maxBufferSize", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 100000,
    maxBufferSize: 45000,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [9, 8, 8];
  const batch = 6;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(plan._largeShapeMode, false);
  assert.equal(plan._arena, null);
  assert.ok(plan._splitWorkspace?.packed);
  assert.ok(plan._splitWorkspace?.full);
  assert.ok(plan._splitWorkspace?.real);

  const input = device.createBuffer({
    size: plan.inBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: plan.outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2c stress matrix runs prime-heavy/odd-even segmented cases", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1 << 20,
    maxBufferSize: 1 << 20,
    minStorageBufferOffsetAlignment: 1,
    maxComputeWorkgroupsPerDimension: [4, 65535, 65535],
  });

  const cases = [
    { shape: [97], tuning: {} },
    { shape: [29], tuning: { forceRaderAxes: [0] } },
    { shape: [34], tuning: { forceBluesteinAxes: [0] } },
  ];

  for (const cfg of cases) {
    const shape = cfg.shape;
    const batch = 3;
    const mainBytes = shape.reduce((a, b) => a * b, 1) * batch * 8;
    const plan = createPlan(device, {
      type: "c2c",
      shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: cfg.tuning,
    });

    const inputBuf = device.createBuffer({
      size: mainBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outputBuf = device.createBuffer({
      size: mainBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const input = makeSegmentedView(inputBuf, mainBytes, 5);
    const output = makeSegmentedView(outputBuf, mainBytes, 4);

    const encoder = makeEncoder();
    plan.exec(encoder, { input, output });
    const computePasses = encoder.ops.filter((op) => op.type === "compute");
    const copyOps = encoder.ops.filter((op) => op.type === "copy");
    assert.ok(computePasses.length > 0);
    assert.ok(copyOps.length > 0);
  }
});

test("r2c/c2r stress odd-even segmented near-limit large-shape runs", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 160,
    maxBufferSize: 1 << 20,
    minStorageBufferOffsetAlignment: 1,
  });

  const shapes = [
    [15, 3, 2],
    [18, 3, 2],
  ];
  const batch = 2;

  for (const shape of shapes) {
    const packed0 = Math.floor(shape[0] / 2) + 1;
    const inBytesR2c = shape.reduce((a, b) => a * b, 1) * batch * 4;
    const outBytesR2c = packed0 * shape[1] * shape[2] * batch * 8;
    const inBytesC2r = outBytesR2c;
    const outBytesC2r = inBytesR2c;

    const r2c = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 160 },
    });
    assert.equal(r2c._largeShapeMode, true);
    {
      const inputBuf = device.createBuffer({
        size: inBytesR2c,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const outputBuf = device.createBuffer({
        size: outBytesR2c,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const input = makeSegmentedView(inputBuf, inBytesR2c, 7);
      const output = makeSegmentedView(outputBuf, outBytesR2c, 6);
      const encoder = makeEncoder();
      r2c.exec(encoder, { input, output });
      assert.ok(encoder.ops.some((op) => op.type === "compute"));
      assert.ok(encoder.ops.some((op) => op.type === "copy"));
    }

    const c2r = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 160 },
    });
    assert.equal(c2r._largeShapeMode, true);
    {
      const inputBuf = device.createBuffer({
        size: inBytesC2r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const outputBuf = device.createBuffer({
        size: outBytesC2r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const input = makeSegmentedView(inputBuf, inBytesC2r, 6);
      const output = makeSegmentedView(outputBuf, outBytesC2r, 7);
      const encoder = makeEncoder();
      c2r.exec(encoder, { input, output });
      assert.ok(encoder.ops.some((op) => op.type === "compute"));
      assert.ok(encoder.ops.some((op) => op.type === "copy"));
    }
  }
});

test("c2c stress rank-4/5/6 forced non-mixed out-of-core runs with segmented endpoints", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1 << 20,
    maxBufferSize: 1 << 20,
    minStorageBufferOffsetAlignment: 1,
  });

  const cases = [
    { shape: [3, 2, 2, 34], tuning: { maxStorageBufferBindingSize: 512, forceBluesteinAxes: [3] } },
    { shape: [3, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 512, forceRaderAxes: [3] } },
    { shape: [3, 2, 2, 34], tuning: { maxStorageBufferBindingSize: 160, forceBluesteinAxes: [3] } },
    { shape: [3, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [3] } },
    { shape: [2, 2, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [4] } },
    { shape: [2, 2, 2, 2, 2, 34], tuning: { maxStorageBufferBindingSize: 160, forceBluesteinAxes: [5] } },
    { shape: [2, 2, 2, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [5] } },
  ];

  for (const cfg of cases) {
    const batch = 4;
    const bytes = cfg.shape.reduce((a, b) => a * b, 1) * batch * 8;
    const plan = createPlan(device, {
      type: "c2c",
      shape: cfg.shape,
      direction: "forward",
      batch,
      inPlace: false,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: cfg.tuning,
    });
    assert.equal(plan._outOfCoreFourStepMode, true);

    const inBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const input = makeSegmentedView(inBuf, bytes, 6);
    const output = makeSegmentedView(outBuf, bytes, 5);

    const encoder = makeEncoder();
    plan.exec(encoder, { input, output });
    assert.ok(encoder.ops.some((op) => op.type === "compute"));
    assert.ok(encoder.ops.some((op) => op.type === "copy"));
  }
});

test("r2c/c2r stress rank-4/5/6 forced non-mixed large-shape runs with segmented temp workspace", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1 << 20,
    maxBufferSize: 1 << 20,
    minStorageBufferOffsetAlignment: 1,
  });

  const cases = [
    { shape: [3, 2, 2, 17], tuning: { maxStorageBufferBindingSize: 1024, forceBluesteinAxes: [3] } },
    { shape: [3, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 1024, forceRaderAxes: [3] } },
    { shape: [3, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [3] } },
    { shape: [2, 2, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [4] } },
    { shape: [2, 2, 2, 2, 2, 17], tuning: { maxStorageBufferBindingSize: 160, forceBluesteinAxes: [5] } },
    { shape: [2, 2, 2, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [5] } },
  ];
  const batch = 2;

  for (const cfg of cases) {
    const r2c = createPlan(device, {
      type: "r2c",
      shape: cfg.shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: cfg.tuning,
    });
    assert.equal(r2c._largeShapeMode, true);
    assert.equal(r2c.c2c?._outOfCoreFourStepMode, true);

    const r2cInBuf = device.createBuffer({
      size: r2c.inBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const r2cOutBuf = device.createBuffer({
      size: r2c.outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const r2cTempBuf = device.createBuffer({
      size: r2c.workspaceBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const r2cTemp = new BufferView(makeSegmentedView(r2cTempBuf, r2c.workspaceBytes, 9));
    {
      const enc = makeEncoder();
      r2c.exec(enc, { input: r2cInBuf, output: r2cOutBuf, temp: r2cTemp });
      assert.ok(enc.ops.some((op) => op.type === "compute"));
      assert.ok(enc.ops.some((op) => op.type === "copy"));
    }

    const c2r = createPlan(device, {
      type: "c2r",
      shape: cfg.shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: { interleavedComplex: true },
      precision: "f32",
      tuning: cfg.tuning,
    });
    assert.equal(c2r._largeShapeMode, true);
    assert.equal(c2r.c2c?._outOfCoreFourStepMode, true);

    const c2rInBuf = device.createBuffer({
      size: c2r.inBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const c2rOutBuf = device.createBuffer({
      size: c2r.outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const c2rTempBuf = device.createBuffer({
      size: c2r.workspaceBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const c2rTemp = new BufferView(makeSegmentedView(c2rTempBuf, c2r.workspaceBytes, 9));
    {
      const enc = makeEncoder();
      c2r.exec(enc, { input: c2rInBuf, output: c2rOutBuf, temp: c2rTemp });
      assert.ok(enc.ops.some((op) => op.type === "compute"));
      assert.ok(enc.ops.some((op) => op.type === "copy"));
    }
  }
});

test("dct/dst/fftconv stress rank-5 large routes execute with segmented endpoints", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1 << 20,
    maxBufferSize: 1 << 20,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [2, 2, 2, 2, 17];
  const batch = 4;
  const dctDstTuning = { maxStorageBufferBindingSize: 7000 };

  const dct = createPlan(device, {
    type: "dct2",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: false },
    precision: "f32",
    tuning: dctDstTuning,
  });
  assert.equal(dct._largeBatchChunkMode, true);
  assert.equal(dct._largeRouteMode, "large-chunk");
  {
    const inBuf = device.createBuffer({
      size: dct.inBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBuf = device.createBuffer({
      size: dct.outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const tempBuf = device.createBuffer({
      size: dct.workspaceBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const enc = makeEncoder();
    dct.exec(enc, {
      input: makeSegmentedView(inBuf, dct.inBytes, 7),
      output: makeSegmentedView(outBuf, dct.outBytes, 6),
      temp: new BufferView(makeSegmentedView(tempBuf, dct.workspaceBytes, 9)),
    });
    assert.ok(enc.ops.some((op) => op.type === "compute"));
    assert.ok(enc.ops.some((op) => op.type === "copy"));
  }

  const dst = createPlan(device, {
    type: "dst1",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: false },
    precision: "f32",
    tuning: dctDstTuning,
  });
  assert.equal(dst._largeBatchChunkMode, true);
  assert.equal(dst._largeRouteMode, "large-chunk");
  {
    const inBuf = device.createBuffer({
      size: dst.inBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBuf = device.createBuffer({
      size: dst.outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const tempBuf = device.createBuffer({
      size: dst.workspaceBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const enc = makeEncoder();
    dst.exec(enc, {
      input: makeSegmentedView(inBuf, dst.inBytes, 7),
      output: makeSegmentedView(outBuf, dst.outBytes, 6),
      temp: new BufferView(makeSegmentedView(tempBuf, dst.workspaceBytes, 9)),
    });
    assert.ok(enc.ops.some((op) => op.type === "compute"));
    assert.ok(enc.ops.some((op) => op.type === "copy"));
  }

  const fftconv = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "convolution", boundary: "circular" },
    tuning: { maxStorageBufferBindingSize: 160, forceBluesteinAxes: [4] },
  });
  assert.equal(fftconv._largeMode, true);
  assert.equal(fftconv._largeRouteMode, "large-out-of-core");
  assert.ok(fftconv._largeRouteAttempts.includes("out-of-core-four-step"));
  {
    const inBuf = device.createBuffer({
      size: fftconv.inputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outBuf = device.createBuffer({
      size: fftconv.totalOutputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const kernelBuf = device.createBuffer({
      size: fftconv.kernelInputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const tempBuf = device.createBuffer({
      size: fftconv.workspaceBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const enc = makeEncoder();
    fftconv.exec(enc, {
      input: makeSegmentedView(inBuf, fftconv.inputBytes, 7),
      output: makeSegmentedView(outBuf, fftconv.totalOutputBytes, 6),
      kernel: makeSegmentedView(kernelBuf, fftconv.kernelInputBytes, 5),
      temp: new BufferView(makeSegmentedView(tempBuf, fftconv.workspaceBytes, 9)),
    });
    assert.ok(enc.ops.some((op) => op.type === "compute"));
    assert.ok(enc.ops.some((op) => op.type === "copy"));
  }
});

test("c2c near-limit threshold toggles large-batch mode without hard reject", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1024,
    maxBufferSize: 1 << 20,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [8, 4];

  const below = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch: 4, // 8*4*4*8 = 1024
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });
  const above = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch: 5, // 1280
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });

  assert.equal(below._largeBatchChunkMode, false);
  assert.equal(above._largeBatchChunkMode, true);
  assert.equal(above._outOfCoreFourStepMode, false);

  for (const p of [below, above]) {
    const bytes = p.mainBytes;
    const input = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const output = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const encoder = makeEncoder();
    p.exec(encoder, { input, output });
    assert.ok(encoder.ops.some((op) => op.type === "compute"));
  }
});

test("r2c/c2r stress non-contiguous strides near line-fit binding limit (odd/even)", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 160,
    maxBufferSize: 1 << 20,
    minStorageBufferOffsetAlignment: 1,
  });
  const shapes = [
    [15, 3, 2],
    [18, 3, 2],
  ];

  for (const shape of shapes) {
    const batch = 2;
    const r2c = createPlan(device, {
      type: "r2c",
      shape,
      direction: "forward",
      batch,
      normalize: "none",
      layout: {
        interleavedComplex: true,
        inputStrides: [2, 9, 27],
        inputOffsetElements: 3,
        inputBatchStrideElements: 96,
        outputStrides: [3, 11, 35],
        outputOffsetElements: 4,
        outputBatchStrideElements: 104,
      },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 160 },
    });
    assert.equal(r2c._largeShapeMode, true);

    const c2r = createPlan(device, {
      type: "c2r",
      shape,
      direction: "inverse",
      batch,
      normalize: "backward",
      layout: {
        interleavedComplex: true,
        inputStrides: [3, 11, 35],
        inputOffsetElements: 4,
        inputBatchStrideElements: 104,
        outputStrides: [2, 8, 30],
        outputOffsetElements: 2,
        outputBatchStrideElements: 100,
      },
      precision: "f32",
      tuning: { maxStorageBufferBindingSize: 160 },
    });
    assert.equal(c2r._largeShapeMode, true);

    {
      const input = device.createBuffer({
        size: 16384,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const output = device.createBuffer({
        size: 16384,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const encoder = makeEncoder();
      r2c.exec(encoder, { input, output, inputOffsetBytes: 4, outputOffsetBytes: 16 });
      assert.ok(encoder.ops.some((op) => op.type === "compute"));
      assert.ok(encoder.ops.some((op) => op.type === "copy"));
    }
    {
      const input = device.createBuffer({
        size: 16384,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const output = device.createBuffer({
        size: 16384,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const encoder = makeEncoder();
      c2r.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 4 });
      assert.ok(encoder.ops.some((op) => op.type === "compute"));
      assert.ok(encoder.ops.some((op) => op.type === "copy"));
    }
  }
});

test("r2c large-shape fallback supports custom input/output strides", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 27],
      inputOffsetElements: 3,
      inputBatchStrideElements: 64,
      outputStrides: [3, 10, 30],
      outputOffsetElements: 4,
      outputBatchStrideElements: 80,
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 8, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c large-shape fallback supports custom input/output strides with segmented input/output", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 27],
      inputOffsetElements: 3,
      inputBatchStrideElements: 64,
      outputStrides: [3, 10, 30],
      outputOffsetElements: 4,
      outputBatchStrideElements: 80,
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);

  const inputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 4096, 7);
  const output = makeSegmentedView(outputBuf, 4096, 6);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 8, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c large-shape fallback supports custom input/output strides with non-trivial ioView", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 30],
      inputOffsetElements: 3,
      inputBatchStrideElements: 80,
      outputStrides: [3, 14, 40],
      outputOffsetElements: 4,
      outputBatchStrideElements: 120,
    },
    ioView: {
      input: { shape: [3, 3, 2], offset: [1, 0, 0] },
      output: { shape: [4, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 4, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c large-shape fallback supports custom strides + ioView + zeroPad", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 30],
      inputOffsetElements: 3,
      inputBatchStrideElements: 80,
      outputStrides: [3, 14, 40],
      outputOffsetElements: 4,
      outputBatchStrideElements: 120,
    },
    ioView: {
      input: { shape: [3, 3, 2], offset: [1, 0, 0] },
      output: { shape: [4, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    zeroPad: {
      read: { start: [1, 0, 0], end: [4, 3, 2] },
      write: { start: [0, 0, 0], end: [2, 3, 2] },
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 4, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
  assert.ok(copyOps.some((op) => op.size === 4 || op.size === 8));
});

test("c2r large-shape fallback supports custom input/output strides", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 8, 24],
      inputOffsetElements: 5,
      inputBatchStrideElements: 70,
      outputStrides: [3, 11, 33],
      outputOffsetElements: 2,
      outputBatchStrideElements: 90,
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 4 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r large-shape fallback supports custom input/output strides with segmented input/output", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 8, 24],
      inputOffsetElements: 5,
      inputBatchStrideElements: 70,
      outputStrides: [3, 11, 33],
      outputOffsetElements: 2,
      outputBatchStrideElements: 90,
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);

  const inputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 4096, 6);
  const output = makeSegmentedView(outputBuf, 4096, 7);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 4 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r large-shape fallback supports custom input/output strides with non-trivial ioView", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 7, 25],
      inputOffsetElements: 5,
      inputBatchStrideElements: 72,
      outputStrides: [2, 16, 48],
      outputOffsetElements: 2,
      outputBatchStrideElements: 140,
    },
    ioView: {
      input: { shape: [2, 3, 2], offset: [1, 0, 0] },
      output: { shape: [7, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 4 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r large-shape fallback supports custom strides + ioView + zeroPad", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 7, 25],
      inputOffsetElements: 5,
      inputBatchStrideElements: 72,
      outputStrides: [2, 16, 48],
      outputOffsetElements: 2,
      outputBatchStrideElements: 140,
    },
    ioView: {
      input: { shape: [2, 3, 2], offset: [1, 0, 0] },
      output: { shape: [7, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    zeroPad: {
      read: { start: [0, 0, 0], end: [2, 3, 2] },
      write: { start: [1, 0, 0], end: [4, 3, 2] },
    },
    precision: "f32",
    tuning: { maxStorageBufferBindingSize: 64 },
  });
  assert.equal(plan._largeShapeMode, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 4 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
  assert.ok(copyOps.some((op) => op.size === 4 || op.size === 8));
});

test("r2c regular mode supports custom input/output strides with segmented input/output", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 27],
      inputOffsetElements: 3,
      inputBatchStrideElements: 64,
      outputStrides: [3, 10, 30],
      outputOffsetElements: 4,
      outputBatchStrideElements: 80,
    },
    precision: "f32",
  });
  assert.equal(plan._largeShapeMode, false);
  assert.equal(plan._needsInputMapping, false);
  assert.equal(plan._needsOutputMapping, false);

  const inputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 4096, 5);
  const output = makeSegmentedView(outputBuf, 4096, 6);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 8, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r regular mode supports custom input/output strides with segmented input/output", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 8, 24],
      inputOffsetElements: 5,
      inputBatchStrideElements: 70,
      outputStrides: [3, 11, 33],
      outputOffsetElements: 2,
      outputBatchStrideElements: 90,
    },
    precision: "f32",
  });
  assert.equal(plan._largeShapeMode, false);
  assert.equal(plan._needsInputMapping, false);
  assert.equal(plan._needsOutputMapping, false);

  const inputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 4096, 6);
  const output = makeSegmentedView(outputBuf, 4096, 7);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 4 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c regular mode supports custom input/output strides with non-trivial ioView", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 30],
      inputOffsetElements: 3,
      inputBatchStrideElements: 80,
      outputStrides: [3, 14, 40],
      outputOffsetElements: 4,
      outputBatchStrideElements: 120,
    },
    ioView: {
      input: { shape: [3, 3, 2], offset: [1, 0, 0] },
      output: { shape: [4, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    precision: "f32",
  });
  assert.equal(plan._largeShapeMode, false);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 4, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("r2c regular mode supports custom strides + ioView + zeroPad", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "r2c",
    shape,
    direction: "forward",
    batch,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9, 30],
      inputOffsetElements: 3,
      inputBatchStrideElements: 80,
      outputStrides: [3, 14, 40],
      outputOffsetElements: 4,
      outputBatchStrideElements: 120,
    },
    ioView: {
      input: { shape: [3, 3, 2], offset: [1, 0, 0] },
      output: { shape: [4, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    zeroPad: {
      read: { start: [1, 0, 0], end: [4, 3, 2] },
      write: { start: [0, 0, 0], end: [2, 3, 2] },
    },
    precision: "f32",
  });
  assert.equal(plan._largeShapeMode, false);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 4, outputOffsetBytes: 16 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroRead.pipeline));
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroWrite.pipeline));
});

test("c2r regular mode supports custom input/output strides with non-trivial ioView", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 7, 25],
      inputOffsetElements: 5,
      inputBatchStrideElements: 72,
      outputStrides: [2, 16, 48],
      outputOffsetElements: 2,
      outputBatchStrideElements: 140,
    },
    ioView: {
      input: { shape: [2, 3, 2], offset: [1, 0, 0] },
      output: { shape: [7, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    precision: "f32",
  });
  assert.equal(plan._largeShapeMode, false);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 4 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("c2r regular mode supports custom strides + ioView + zeroPad", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [5, 3, 2];
  const batch = 2;

  const plan = createPlan(device, {
    type: "c2r",
    shape,
    direction: "inverse",
    batch,
    normalize: "backward",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 7, 25],
      inputOffsetElements: 5,
      inputBatchStrideElements: 72,
      outputStrides: [2, 16, 48],
      outputOffsetElements: 2,
      outputBatchStrideElements: 140,
    },
    ioView: {
      input: { shape: [2, 3, 2], offset: [1, 0, 0] },
      output: { shape: [7, 3, 2], offset: [-1, 0, 0], clearOutside: false },
    },
    zeroPad: {
      read: { start: [0, 0, 0], end: [2, 3, 2] },
      write: { start: [1, 0, 0], end: [4, 3, 2] },
    },
    precision: "f32",
  });
  assert.equal(plan._largeShapeMode, false);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 16, outputOffsetBytes: 4 });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroRead.pipeline));
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroWrite.pipeline));
});

test("c2c regular mode falls back to internal workspace when segmented temp does not provide contiguous staging", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "c2c",
    shape: [16],
    direction: "forward",
    batch: 1,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    ioView: {
      input: { shape: [8], offset: [4] },
      output: null,
    },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, false);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const tempBuf = device.createBuffer({
    size: plan.workspaceBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const temp = new BufferView(makeSegmentedView(tempBuf, plan.workspaceBytes, 11));

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, temp });

  assert.equal(computePassUsesBuffer(encoder, tempBuf), false);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("r2c regular mode ignores aliasing temp workspace and runs via internal workspace", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "r2c",
    shape: [17, 3],
    direction: "forward",
    batch: 2,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });
  assert.equal(plan._largeShapeMode, false);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, temp: input });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("c2r regular mode ignores aliasing temp workspace and runs via internal workspace", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "c2r",
    shape: [17, 3],
    direction: "inverse",
    batch: 2,
    normalize: "backward",
    layout: { interleavedComplex: true },
    precision: "f32",
  });
  assert.equal(plan._largeShapeMode, false);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, temp: output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("dct2 regular mode falls back to internal arena when segmented temp is non-contiguous", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "dct2",
    shape: [16],
    direction: "forward",
    batch: 2,
    normalize: "none",
    layout: { interleavedComplex: false },
    precision: "f32",
  });

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const tempBuf = device.createBuffer({
    size: plan.workspaceBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const temp = new BufferView(makeSegmentedView(tempBuf, plan.workspaceBytes, 13));

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, temp });

  assert.equal(computePassUsesBuffer(encoder, tempBuf), false);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("fftconv supports WHD+CN-style layout descriptors for input/output (kernelCount=1)", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 2;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: {
      interleavedComplex: true,
      whdcn: {
        input: {
          channels: 3,
          channelIndex: 1,
          channelStrideElements: 16,
          batchStrideElements: 80,
        },
        output: {
          channels: 2,
          channelIndex: 1,
          channelStrideElements: 24,
          batchStrideElements: 96,
        },
      },
    },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount: 1, outputLayout: "kernel-major" },
  });

  assert.equal(plan._usesStridedInput, true);
  assert.equal(plan._usesStridedOutput, true);
  assert.equal(plan._usesWhdcnInput, true);
  assert.equal(plan._usesWhdcnOutput, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * shape[0]);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
});

test("fftconv explicit contiguous strides canonicalize to contiguous routing", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 2;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: {
      interleavedComplex: true,
      inputStrides: [1],
      inputOffsetElements: 0,
      inputBatchStrideElements: shape[0],
      outputStrides: [1],
      outputOffsetElements: 0,
      outputBatchStrideElements: shape[0],
    },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount: 1, outputLayout: "kernel-major" },
  });

  assert.equal(plan._usesStridedInput, false);
  assert.equal(plan._usesStridedOutput, false);
});

test("fftconv forced-large mode supports explicit strided input/output routing", () => {
  const maxBind = 64;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [32];
  const batch = 2;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: {
      interleavedComplex: true,
      inputStrides: [2],
      inputOffsetElements: 1,
      inputBatchStrideElements: 90,
      outputStrides: [3],
      outputOffsetElements: 2,
      outputBatchStrideElements: 120,
    },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount: 1, outputLayout: "kernel-major" },
  });

  assert.equal(plan._largeMode, true);
  assert.equal(plan._usesStridedInput, true);
  assert.equal(plan._usesStridedOutput, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * shape[0]);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
});

test("fftconv multi-kernel strided output requires kernel-lane stride policy", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "fftconv",
        shape: [8],
        batch: 2,
        inPlace: false,
        layout: {
          interleavedComplex: true,
          outputStrides: [2],
          outputOffsetElements: 1,
          outputBatchStrideElements: 40,
        },
        precision: "f32",
        fftConv: { mode: "convolution", kernelCount: 2, outputLayout: "kernel-major" },
      }),
    /outputKernelStrideElements|channelPolicy\.output/
  );
});

test("fftconv forced-large mode supports multi-kernel channelPolicy output with segmented views", () => {
  const maxBind = 64;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [32];
  const batch = 2;
  const kernelCount = 2;
  const outChannelStride = 48;
  const outChannelIndex = 1;
  const outBatchStride = 256;
  const maxKernelChannel = outChannelIndex + (kernelCount - 1);
  const outElems = maxKernelChannel * outChannelStride + (batch - 1) * outBatchStride + shape[0];
  const outBytes = outElems * 8;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: {
      mode: "convolution",
      kernelCount,
      outputLayout: "batch-major",
      channelPolicy: {
        input: {
          channels: 3,
          channelIndex: 1,
          channelStrideElements: 40,
          batchStrideElements: 160,
        },
        output: {
          channels: 5,
          channelIndex: outChannelIndex,
          channelStrideElements: outChannelStride,
          batchStrideElements: outBatchStride,
          kernelStepChannels: 1,
        },
      },
    },
  });

  assert.equal(plan._largeMode, true);
  assert.equal(plan._usesStridedInput, true);
  assert.equal(plan._usesStridedOutput, true);
  assert.equal(plan._usesWhdcnInput, true);
  assert.equal(plan._usesWhdcnOutput, true);
  assert.equal(plan._stridedOutputKernelStrideElements, outChannelStride);

  const inputElems = 1 * 40 + (batch - 1) * 160 + shape[0];
  const input = device.createBuffer({
    size: inputElems * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = makeSegmentedView(outputBuf, outBytes, 6);
  const kernel = new Float32Array(2 * shape[0] * kernelCount);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
});

test("fftconv channelPolicy output validates channel capacity for kernelCount", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "fftconv",
        shape: [16],
        batch: 1,
        inPlace: false,
        layout: { interleavedComplex: true },
        precision: "f32",
        fftConv: {
          mode: "convolution",
          kernelCount: 3,
          outputLayout: "batch-major",
          channelPolicy: {
            output: {
              channels: 2,
              channelIndex: 1,
              channelStrideElements: 24,
              kernelStepChannels: 1,
            },
          },
        },
      }),
    /does not fit kernelCount/
  );
});

test("fftconv channelPolicy cannot be combined with layout.whdcn", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 30, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "fftconv",
        shape: [8],
        batch: 1,
        inPlace: false,
        layout: {
          interleavedComplex: true,
          whdcn: {
            output: { channels: 2, channelIndex: 1 },
          },
        },
        precision: "f32",
        fftConv: {
          mode: "convolution",
          channelPolicy: {
            output: {
              channels: 2,
              channelIndex: 1,
              channelStrideElements: 8,
            },
          },
        },
      }),
    /cannot be combined with layout\.whdcn/
  );
});

test("fftconv tuning.pointwiseChunkElements validates binding-aware upper bound", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "fftconv",
        shape: [16],
        batch: 1,
        inPlace: false,
        layout: { interleavedComplex: true },
        precision: "f32",
        fftConv: {
          mode: "convolution",
          kernelCount: 1,
          outputLayout: "kernel-major",
          tuning: {
            pointwiseChunkElements: 16,
          },
        },
      }),
    /pointwiseChunkElements=.*exceeds max supported/
  );
});

test("fftconv tuning.pointwiseChunkElements controls pointwise chunking granularity", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 20, minStorageBufferOffsetAlignment: 1 });
  const shape = [32];
  const batch = 1;
  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: {
      mode: "convolution",
      kernelCount: 1,
      outputLayout: "kernel-major",
      tuning: {
        pointwiseChunkElements: 4,
      },
    },
  });

  assert.equal(plan._pointwiseChunkElems, 4);

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * shape[0]);
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  const pointwisePasses = encoder.ops.filter((op) => op.type === "compute" && op.pipeline === plan.pointwise.pipeline);
  assert.equal(pointwisePasses.length, 8);
});

test("fftconv tuning.extractCopyChunkElements coalesces fallback extraction copies", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 1 << 20, minStorageBufferOffsetAlignment: 1 });
  const shape = [16];
  const kernelShape = [5];
  const batch = 1;
  const outLen = shape[0] - kernelShape[0] + 1;
  const outBytes = outLen * batch * 8;

  const runWithChunk = (extractCopyChunkElements) => {
    const plan = createPlan(device, {
      type: "fftconv",
      shape,
      batch,
      inPlace: false,
      layout: { interleavedComplex: true },
      precision: "f32",
      fftConv: {
        mode: "convolution",
        boundary: "linear-valid",
        kernelShape,
        kernelCount: 1,
        outputLayout: "kernel-major",
        tuning: { extractCopyChunkElements },
      },
    });
    const input = device.createBuffer({
      size: shape[0] * batch * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outputBuf = device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const output = makeSegmentedView(outputBuf, outBytes, 3);
    const kernel = new Float32Array(2 * kernelShape[0]);
    const encoder = makeEncoder();
    plan.exec(encoder, { input, output, kernel });
    return encoder.ops.filter((op) => op.type === "copy" && op.dst === outputBuf).length;
  };

  const copiesChunk1 = runWithChunk(1);
  const copiesChunk4 = runWithChunk(4);
  assert.ok(copiesChunk4 < copiesChunk1);
});

test("fftconv forced-large mode ignores aliasing temp workspace and keeps execution valid", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [32];
  const batch = 2;
  const kernelCount = 2;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount, outputLayout: "kernel-major" },
  });
  assert.equal(plan._largeMode, true);

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: shape[0] * batch * kernelCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * shape[0] * kernelCount);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel, temp: output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("fftconv uses segmented temp workspace directly without internal contiguous fallback", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 64, minStorageBufferOffsetAlignment: 1 });
  const shape = [32];
  const batch = 2;
  const kernelCount = 1;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount, outputLayout: "kernel-major" },
  });
  assert.equal(plan._largeMode, true);

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: shape[0] * batch * kernelCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * shape[0] * kernelCount);
  const tempBuf = device.createBuffer({
    size: plan.workspaceBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const temp = new BufferView(makeSegmentedView(tempBuf, plan.workspaceBytes, 13));

  let usedProvidedWorkspace = false;
  let fellBackToInternal = false;
  const resolveWorkspaceViewsOrig = plan._resolveWorkspaceViews.bind(plan);
  plan._resolveWorkspaceViews = (arg) => {
    if (arg === temp) usedProvidedWorkspace = true;
    if (arg == null) fellBackToInternal = true;
    return resolveWorkspaceViewsOrig(arg);
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel, temp });

  assert.equal(usedProvidedWorkspace, true);
  assert.equal(fellBackToInternal, false);
  assert.equal(computePassUsesBuffer(encoder, tempBuf), true);
});

test("fftconv forced-large path keeps compute bindings within maxStorageBufferBindingSize", () => {
  const maxBind = 64;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [32];
  const batch = 4;
  const kernelCount = 2;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount, outputLayout: "kernel-major" },
  });
  assert.equal(plan._largeMode, true);
  assert.equal(plan._batchSlicedExecution, false);

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: shape[0] * batch * kernelCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * shape[0] * kernelCount);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  const pointwisePasses = encoder.ops.filter((op) => op.type === "compute" && op.pipeline === plan.pointwise.pipeline);
  assert.equal(pointwisePasses.length, kernelCount * batch * 4);
});

test("fftconv batch-sliced mode executes when main workspace exceeds maxBufferSize", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 256,
    maxBufferSize: 512,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [8];
  const batch = 16;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "correlation", kernelCount: 1, outputLayout: "kernel-major" },
  });
  assert.equal(plan._batchSlicedExecution, true);
  assert.equal(plan.fftData.batch, 1);
  assert.equal(plan.ifftData.batch, 1);

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * shape[0]);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  const pointwisePasses = encoder.ops.filter((op) => op.type === "compute" && op.pipeline === plan.pointwise.pipeline);
  assert.equal(pointwisePasses.length, batch);
});

test("fftconv large mode supports zeroPad.read/zeroPad.write in circular boundary", () => {
  const maxBind = 256;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 5;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    zeroPad: {
      read: { start: [1], end: [7] },
      write: { start: [2], end: [6] },
    },
    fftConv: { mode: "convolution", boundary: "circular", kernelCount: 1, outputLayout: "kernel-major" },
  });
  assert.equal(plan._largeMode, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * shape[0]);
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroRead.pipeline));
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroWrite.pipeline));
});

test("fftconv batch-sliced linear-valid path supports zeroPad.read/zeroPad.write with extracted output", () => {
  const maxBind = 256;
  const device = new FakeDevice({
    maxStorageBufferBindingSize: maxBind,
    maxBufferSize: 256,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [8];
  const kernelShape = [3];
  const batch = 4;
  const outLen = shape[0] - kernelShape[0] + 1;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    zeroPad: {
      read: { start: [1], end: [9] },
      write: { start: [2], end: [8] },
    },
    fftConv: { mode: "correlation", boundary: "linear-valid", kernelShape, kernelCount: 1, outputLayout: "batch-major" },
  });
  assert.equal(plan._batchSlicedExecution, true);
  assert.equal(plan._needsOutputExtract, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: outLen * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = new Float32Array(2 * kernelShape[0]);
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroRead.pipeline));
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroWrite.pipeline));
  assert.ok(computePasses.some((op) => op.pipeline === plan.outputExtract.pipeline));
});

test("fftconv linear-same extracted output supports segmented output under forced-large limits", () => {
  const maxBind = 64;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const kernelShape = [5];
  const batch = 1;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: {
      mode: "convolution",
      boundary: "linear-same",
      kernelShape,
      kernelCount: 1,
      outputLayout: "batch-major",
    },
  });
  assert.equal(plan._needsOutputExtract, true);
  assert.equal(plan._largeMode, true);

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = makeSegmentedView(outputBuf, shape[0] * batch * 8, 6);
  const kernel = new Float32Array(2 * kernelShape[0]);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(copyOps.some((op) => op.size === 8), "expected per-element copy fallback for extracted output");
});

test("fftconv forced-large segmented kernel source uses copy fallback (no oversized segmented-copy compute bind)", () => {
  const maxBind = 64;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [32];
  const batch = 2;

  const plan = createPlan(device, {
    type: "fftconv",
    shape,
    batch,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "convolution", kernelCount: 1, outputLayout: "kernel-major" },
  });

  const input = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: shape[0] * batch * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernelBuf = device.createBuffer({
    size: shape[0] * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const kernel = makeSegmentedView(kernelBuf, shape[0] * 8, 5);

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, kernel });

  assertComputeBindingsWithinLimit(encoder, maxBind);
});

test("dct2 large-batch chunk mode executes when one-batch bindings fit", () => {
  const maxBind = 256;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 10;
  const totalBytes = shape[0] * batch * 4;

  const plan = createPlan(device, {
    type: "dct2",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: false },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, true);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("dct2 regular mode uses split internal workspace when monolithic workspace exceeds maxBufferSize", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1 << 30,
    maxBufferSize: 10000,
    minStorageBufferOffsetAlignment: 1,
  });
  const shape = [8];
  const batch = 64;
  const totalBytes = shape[0] * batch * 4;

  const plan = createPlan(device, {
    type: "dct2",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: false },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, false);
  assert.equal(plan._arena, null);
  assert.ok(plan._splitWorkspace?.dataA);
  assert.ok(plan._splitWorkspace?.dataB);
  assert.ok(plan._splitWorkspace?.work);
  assert.ok(plan._splitWorkspace?.fftScratch);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
});

test("dct2 regular mode supports custom input/output strides (f32)", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1 << 30,
    maxBufferSize: 1 << 30,
    minStorageBufferOffsetAlignment: 1,
  });
  const plan = createPlan(device, {
    type: "dct2",
    shape: [5, 3],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      inputStrides: [2, 11],
      outputStrides: [3, 13],
      inputOffsetElements: 3,
      outputOffsetElements: 5,
      inputBatchStrideElements: 80,
      outputBatchStrideElements: 90,
    },
    ioView: {
      input: { shape: [3, 3], offset: [1, 0] },
      output: { shape: [7, 3], offset: [-1, 0], clearOutside: true },
    },
    zeroPad: {
      read: { start: [1, 0], end: [5, 3] },
      write: { start: [0, 0], end: [5, 3] },
    },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, false);
  assert.ok(plan._usesStridedInput);
  assert.ok(plan._usesStridedOutput);

  const input = device.createBuffer({
    size: 1 << 20,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 1 << 20,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("dct2 regular mode supports custom strides with segmented input/output", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1 << 30,
    maxBufferSize: 1 << 30,
    minStorageBufferOffsetAlignment: 1,
  });
  const plan = createPlan(device, {
    type: "dct2",
    shape: [5, 3],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      inputStrides: [2, 11],
      outputStrides: [3, 13],
      inputOffsetElements: 3,
      outputOffsetElements: 5,
      inputBatchStrideElements: 80,
      outputBatchStrideElements: 90,
    },
    ioView: {
      input: { shape: [3, 3], offset: [1, 0] },
      output: { shape: [7, 3], offset: [-1, 0], clearOutside: true },
    },
    zeroPad: {
      read: { start: [1, 0], end: [5, 3] },
      write: { start: [0, 0], end: [5, 3] },
    },
    precision: "f32",
  });

  const inputBuf = device.createBuffer({
    size: 1 << 20,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 1 << 20,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 1 << 20, 7);
  const output = makeSegmentedView(outputBuf, 1 << 20, 6);
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
  const stridedCompute = encoder.ops.filter(
    (op) => op.type === "compute" && (op.pipeline === plan.stridedIn?.pipeline || op.pipeline === plan.stridedOut?.pipeline)
  );
  assert.equal(stridedCompute.length, 0);
});

test("dct2 large-batch chunk mode supports custom strides when one-batch windows fit", () => {
  const maxBind = 256;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "dct2",
    shape: [8],
    direction: "forward",
    batch: 10,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      inputStrides: [2],
      outputStrides: [3],
      inputOffsetElements: 1,
      outputOffsetElements: 2,
      inputBatchStrideElements: 20,
      outputBatchStrideElements: 48,
    },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, true);
  assert.ok(plan._usesStridedInput);
  assert.ok(plan._usesStridedOutput);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, {
    input,
    output,
    inputOffsetBytes: 8,
    outputOffsetBytes: 12,
  });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
});

test("dct2 large-batch chunk mode supports custom strides with segmented input/output", () => {
  const maxBind = 256;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "dct2",
    shape: [8],
    direction: "forward",
    batch: 10,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      inputStrides: [2],
      outputStrides: [3],
      inputOffsetElements: 1,
      outputOffsetElements: 2,
      inputBatchStrideElements: 20,
      outputBatchStrideElements: 48,
    },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, true);

  const inputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const input = makeSegmentedView(inputBuf, 4096, 5);
  const output = makeSegmentedView(outputBuf, 4096, 6);
  const encoder = makeEncoder();
  plan.exec(encoder, {
    input,
    output,
    inputOffsetBytes: 8,
    outputOffsetBytes: 12,
  });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
  const stridedCompute = encoder.ops.filter(
    (op) => op.type === "compute" && (op.pipeline === plan.stridedIn?.pipeline || op.pipeline === plan.stridedOut?.pipeline)
  );
  assert.equal(stridedCompute.length, 0);
});

test("dct2 large-batch chunk mode supports custom strides + ioView + zeroPad", () => {
  const maxBind = 256;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "dct2",
    shape: [8],
    direction: "forward",
    batch: 10,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      inputStrides: [2],
      outputStrides: [3],
      inputOffsetElements: 1,
      outputOffsetElements: 2,
      inputBatchStrideElements: 20,
      outputBatchStrideElements: 48,
    },
    ioView: {
      input: { shape: [6], offset: [1] },
      output: { shape: [12], offset: [2], clearOutside: false },
    },
    zeroPad: {
      read: { start: [1], end: [7] },
      write: { start: [1], end: [7] },
    },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 8, outputOffsetBytes: 12 });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroRead.pipeline));
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroWrite.pipeline));
});

test("dst1 large-batch mode supports custom strides + ioView + zeroPad", () => {
  const maxBind = 256;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "dst1",
    shape: [8],
    direction: "forward",
    batch: 10,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      inputStrides: [2],
      outputStrides: [3],
      inputOffsetElements: 1,
      outputOffsetElements: 2,
      inputBatchStrideElements: 20,
      outputBatchStrideElements: 48,
    },
    ioView: {
      input: { shape: [6], offset: [1] },
      output: { shape: [12], offset: [2], clearOutside: false },
    },
    zeroPad: {
      read: { start: [1], end: [7] },
      write: { start: [1], end: [7] },
    },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, true);
  assert.ok(plan.zeroRead);
  assert.ok(plan.zeroWrite);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, inputOffsetBytes: 8, outputOffsetBytes: 12 });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  const copyOps = encoder.ops.filter((op) => op.type === "copy");
  assert.ok(computePasses.length > 0);
  assert.ok(copyOps.length > 0);
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroRead.pipeline));
  assert.ok(computePasses.some((op) => op.pipeline === plan.zeroWrite.pipeline));
});

test("dct2 custom-strided output supports ioView.output.clearOutside=false", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1 << 30,
    maxBufferSize: 1 << 30,
    minStorageBufferOffsetAlignment: 1,
  });
  const plan = createPlan(device, {
    type: "dct2",
    shape: [8],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      outputStrides: [2],
      outputOffsetElements: 1,
      outputBatchStrideElements: 40,
    },
    ioView: {
      output: { shape: [16], offset: [4], clearOutside: false },
    },
    precision: "f32",
  });

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = makeSegmentedView(outputBuf, 4096, 6);
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy" && op.size === 4));
});

test("dct2 large-batch custom-strided output supports ioView.output.clearOutside=false", () => {
  const maxBind = 256;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const plan = createPlan(device, {
    type: "dct2",
    shape: [8],
    direction: "forward",
    batch: 10,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      outputStrides: [2],
      outputOffsetElements: 1,
      outputBatchStrideElements: 40,
    },
    ioView: {
      output: { shape: [16], offset: [4], clearOutside: false },
    },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, true);

  const input = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: 4096,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = makeSegmentedView(outputBuf, 4096, 5);
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  assert.ok(encoder.ops.some((op) => op.type === "compute"));
  assert.ok(encoder.ops.some((op) => op.type === "copy" && op.size === 4));
});

test("dct2 large-batch mode ignores aliasing temp and falls back to internal workspace", () => {
  const maxBind = 256;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [8];
  const batch = 10;
  const totalBytes = shape[0] * batch * 4;

  const plan = createPlan(device, {
    type: "dct2",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: false },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, true);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, temp: input });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("dst1 large-batch mode executes for non-factorable per-axis work FFT length", () => {
  const maxBind = 600;
  const device = new FakeDevice({ maxStorageBufferBindingSize: maxBind, minStorageBufferOffsetAlignment: 1 });
  const shape = [16];
  const batch = 3;
  const totalBytes = shape[0] * batch * 4;

  const plan = createPlan(device, {
    type: "dst1",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: false },
    precision: "f32",
  });
  assert.equal(plan._largeBatchChunkMode, true);

  const input = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  assertComputeBindingsWithinLimit(encoder, maxBind);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
});

test("dst1 large-batch mode rejects when one-batch bindings exceed maxStorageBufferBindingSize", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 200, minStorageBufferOffsetAlignment: 1 });
  assert.throws(
    () =>
      createPlan(device, {
        type: "dst1",
        shape: [16],
        direction: "forward",
        batch: 2,
        inPlace: false,
        normalize: "none",
        layout: { interleavedComplex: false },
        precision: "f32",
      }),
    /requires one-batch bindings to fit/
  );
});

test("c2c layout.whdcn maps to strided CN batches", () => {
  const device = new FakeDevice();
  const plan = createPlan(device, {
    type: "c2c",
    shape: [4, 3],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      whdcn: { channels: 3, channelIndex: 1 },
    },
    precision: "f32",
  });
  assert.deepEqual(plan._inputStrides, [1, 4]);
  assert.deepEqual(plan._outputStrides, [1, 4]);
  assert.equal(plan._inputOffsetElements, 12);
  assert.equal(plan._outputOffsetElements, 12);
  assert.equal(plan._inputBatchStrideElements, 36);
  assert.equal(plan._outputBatchStrideElements, 36);
  assert.equal(plan._usesWhdcnInput, true);
  assert.equal(plan._usesWhdcnOutput, true);
});

test("c2c layout.whdcn is side-selective and explicit fields keep priority", () => {
  const device = new FakeDevice();
  const plan = createPlan(device, {
    type: "c2c",
    shape: [4, 3],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [2, 9],
      inputOffsetElements: 5,
      inputBatchStrideElements: 80,
      whdcn: {
        output: { channels: 2, channelIndex: 1 },
      },
    },
    precision: "f32",
  });
  assert.deepEqual(plan._inputStrides, [2, 9]);
  assert.equal(plan._inputOffsetElements, 5);
  assert.equal(plan._inputBatchStrideElements, 80);
  assert.equal(plan._usesWhdcnInput, false);

  assert.deepEqual(plan._outputStrides, [1, 4]);
  assert.equal(plan._outputOffsetElements, 12);
  assert.equal(plan._outputBatchStrideElements, 24);
  assert.equal(plan._usesWhdcnOutput, true);
});

test("c2c explicit contiguous strides keep contiguous large-chunk fast path", () => {
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });
  const shape = [4, 4];
  const batch = 5;
  const plan = createPlan(device, {
    type: "c2c",
    shape,
    direction: "forward",
    batch,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [1, 4],
      outputStrides: [1, 4],
      inputOffsetElements: 0,
      outputOffsetElements: 0,
      inputBatchStrideElements: 16,
      outputBatchStrideElements: 16,
    },
    precision: "f32",
  });

  assert.equal(plan._largeBatchChunkMode, true);
  assert.equal(plan._outOfCoreFourStepMode, false);
  assert.equal(plan._largeRouteMode, "large-chunk");
  assert.equal(plan._usesStridedInput, false);
  assert.equal(plan._usesStridedOutput, false);
  assert.equal(plan.stridedIn, null);
  assert.equal(plan.stridedOut, null);
});

test("explicit contiguous strides canonicalize to non-strided routing across r2c/c2r/dct", () => {
  const device = new FakeDevice();

  const r2cPlan = createPlan(device, {
    type: "r2c",
    shape: [8],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [1],
      inputOffsetElements: 0,
      inputBatchStrideElements: 8,
      outputStrides: [1],
      outputOffsetElements: 0,
      outputBatchStrideElements: 5,
    },
    precision: "f32",
  });
  assert.equal(r2cPlan._usesStridedInput, false);
  assert.equal(r2cPlan._usesStridedOutput, false);

  const c2rPlan = createPlan(device, {
    type: "c2r",
    shape: [8],
    direction: "inverse",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      inputStrides: [1],
      inputOffsetElements: 0,
      inputBatchStrideElements: 5,
      outputStrides: [1],
      outputOffsetElements: 0,
      outputBatchStrideElements: 8,
    },
    precision: "f32",
  });
  assert.equal(c2rPlan._usesStridedInput, false);
  assert.equal(c2rPlan._usesStridedOutput, false);

  const dctPlan = createPlan(device, {
    type: "dct2",
    shape: [8],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      inputStrides: [1],
      inputOffsetElements: 0,
      inputBatchStrideElements: 8,
      outputStrides: [1],
      outputOffsetElements: 0,
      outputBatchStrideElements: 8,
    },
    precision: "f32",
  });
  assert.equal(dctPlan._usesStridedInput, false);
  assert.equal(dctPlan._usesStridedOutput, false);
});

test("shared large-route metadata matrix covers c2c/r2c/c2r/dct/fftconv", () => {
  const outOfCoreDevice = new FakeDevice({
    maxStorageBufferBindingSize: 64,
    minStorageBufferOffsetAlignment: 1,
  });
  const device = new FakeDevice({ maxStorageBufferBindingSize: 256, minStorageBufferOffsetAlignment: 1 });

  const c2cOutOfCore = createPlan(outOfCoreDevice, {
    type: "c2c",
    shape: [4, 3],
    direction: "forward",
    batch: 5,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });
  assert.equal(c2cOutOfCore._largeRouteMode, "large-out-of-core");
  assert.ok(c2cOutOfCore._largeRouteReasons.includes("bytes-per-batch-exceeds-bind"));
  assert.ok(c2cOutOfCore._largeRouteAttempts.includes("out-of-core-four-step"));

  const r2cLarge = createPlan(device, {
    type: "r2c",
    shape: [16, 4],
    direction: "forward",
    batch: 4,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });
  assert.equal(r2cLarge._largeShapeMode, true);
  assert.equal(r2cLarge._largeRouteMode, "large-out-of-core");
  assert.ok(r2cLarge._largeRouteReasons.includes("requires-large-bindings"));
  assert.ok(r2cLarge._largeRouteReasons.includes("bytes-per-batch-exceeds-bind"));
  assert.ok(r2cLarge._largeRouteAttempts.includes("batch-chunk"));
  assert.ok(r2cLarge._largeRouteAttempts.includes("out-of-core-four-step"));
  assert.deepEqual(r2cLarge._largeRouteAxisKinds, ["mixed", "mixed"]);
  assert.deepEqual(r2cLarge._largeRouteAxisSupported, [true, true]);

  const c2rLarge = createPlan(device, {
    type: "c2r",
    shape: [16, 4],
    direction: "inverse",
    batch: 4,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });
  assert.equal(c2rLarge._largeShapeMode, true);
  assert.equal(c2rLarge._largeRouteMode, "large-out-of-core");
  assert.ok(c2rLarge._largeRouteReasons.includes("requires-large-bindings"));
  assert.ok(c2rLarge._largeRouteReasons.includes("bytes-per-batch-exceeds-bind"));
  assert.ok(c2rLarge._largeRouteAttempts.includes("batch-chunk"));
  assert.ok(c2rLarge._largeRouteAttempts.includes("out-of-core-four-step"));
  assert.deepEqual(c2rLarge._largeRouteAxisKinds, ["mixed", "mixed"]);
  assert.deepEqual(c2rLarge._largeRouteAxisSupported, [true, true]);

  const dctLarge = createPlan(device, {
    type: "dct2",
    shape: [4, 4],
    direction: "forward",
    batch: 5,
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: false },
    precision: "f32",
  });
  assert.equal(dctLarge._largeBatchChunkMode, true);
  assert.equal(dctLarge._largeRouteMode, "large-chunk");
  assert.ok(dctLarge._largeRouteReasons.includes("requires-large-bindings"));
  assert.ok(dctLarge._largeRouteAttempts.includes("batch-chunk"));

  const fftconvLarge = createPlan(device, {
    type: "fftconv",
    shape: [8, 8],
    batch: 4,
    inPlace: false,
    layout: { interleavedComplex: true },
    precision: "f32",
    fftConv: { mode: "convolution", boundary: "circular" },
  });
  assert.equal(fftconvLarge._largeMode, true);
  assert.equal(fftconvLarge._largeRouteMode, "large-out-of-core");
  assert.ok(fftconvLarge._largeRouteReasons.includes("requires-large-bindings"));
  assert.ok(fftconvLarge._largeRouteAttempts.includes("batch-chunk"));
  assert.ok(fftconvLarge._largeRouteAttempts.includes("out-of-core-four-step"));
});

test("r2c layout.whdcn resolves distinct real and packed-domain shapes", () => {
  const device = new FakeDevice();
  const plan = createPlan(device, {
    type: "r2c",
    shape: [9, 3, 2],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      whdcn: {
        input: { channels: 4, channelIndex: 2 },
        output: { channels: 2, channelIndex: 1 },
      },
    },
    precision: "f32",
  });
  assert.deepEqual(plan._inputStrides, [1, 9, 27]);
  assert.equal(plan._inputOffsetElements, 108);
  assert.equal(plan._inputBatchStrideElements, 216);
  assert.deepEqual(plan._outputStrides, [1, 5, 15]);
  assert.equal(plan._outputOffsetElements, 30);
  assert.equal(plan._outputBatchStrideElements, 60);
  assert.equal(plan._usesWhdcnInput, true);
  assert.equal(plan._usesWhdcnOutput, true);
});

test("c2r layout.whdcn resolves distinct packed and real-domain shapes", () => {
  const device = new FakeDevice();
  const plan = createPlan(device, {
    type: "c2r",
    shape: [9, 3, 2],
    direction: "inverse",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: true,
      whdcn: {
        input: { channels: 2, channelIndex: 1 },
        output: { channels: 3, channelIndex: 2 },
      },
    },
    precision: "f32",
  });
  assert.deepEqual(plan._inputStrides, [1, 5, 15]);
  assert.equal(plan._inputOffsetElements, 30);
  assert.equal(plan._inputBatchStrideElements, 60);
  assert.deepEqual(plan._outputStrides, [1, 9, 27]);
  assert.equal(plan._outputOffsetElements, 108);
  assert.equal(plan._outputBatchStrideElements, 162);
  assert.equal(plan._usesWhdcnInput, true);
  assert.equal(plan._usesWhdcnOutput, true);
});

test("dct2 layout.whdcn maps from ioView physical shapes", () => {
  const device = new FakeDevice();
  const plan = createPlan(device, {
    type: "dct2",
    shape: [8],
    direction: "forward",
    batch: 2,
    inPlace: false,
    normalize: "none",
    layout: {
      interleavedComplex: false,
      whdcn: {
        input: { channels: 2, channelIndex: 1 },
        output: { channels: 3, channelIndex: 0 },
      },
    },
    ioView: {
      input: { shape: [4], offset: [2] },
      output: { shape: [10], offset: [1] },
    },
    precision: "f32",
  });
  assert.deepEqual(plan._inputStrides, [1]);
  assert.equal(plan._inputOffsetElements, 4);
  assert.equal(plan._inputBatchStrideElements, 8);
  assert.deepEqual(plan._outputStrides, [1]);
  assert.equal(plan._outputOffsetElements, 0);
  assert.equal(plan._outputBatchStrideElements, 30);
  assert.equal(plan._usesWhdcnInput, true);
  assert.equal(plan._usesWhdcnOutput, true);
});

test("layout.whdcn rejects out-of-range channelIndex", () => {
  const device = new FakeDevice();
  assert.throws(
    () =>
      createPlan(device, {
        type: "c2c",
        shape: [8],
        direction: "forward",
        batch: 1,
        inPlace: false,
        normalize: "none",
        layout: {
          interleavedComplex: true,
          whdcn: { channels: 2, channelIndex: 2 },
        },
        precision: "f32",
      }),
    /channelIndex/
  );
});

test("fftconv channel-lane preset builds deterministic defaults from logical span", () => {
  const preset = createFftConvChannelLanePreset({
    shape: [8, 4],
    batch: 2,
    kernelCount: 3,
    input: { channels: 6 },
    output: { channels: 12, kernelStepChannels: 2 },
  });

  assert.deepEqual(preset.shape, [8, 4]);
  assert.equal(preset.batch, 2);
  assert.deepEqual(preset.layout, { interleavedComplex: true });

  assert.equal(preset.fftConv.mode, "convolution");
  assert.equal(preset.fftConv.boundary, "circular");
  assert.equal(preset.fftConv.outputLayout, "kernel-major");
  assert.equal(preset.fftConv.kernelCount, 3);

  assert.deepEqual(preset.fftConv.channelPolicy.input, {
    channels: 6,
    channelIndex: 0,
    channelStrideElements: 32,
    batchStrideElements: 192,
    offsetElements: 0,
  });
  assert.deepEqual(preset.fftConv.channelPolicy.output, {
    channels: 12,
    channelIndex: 0,
    channelStrideElements: 32,
    batchStrideElements: 384,
    offsetElements: 0,
    kernelStepChannels: 2,
  });
});

test("fftconv channel-lane wrappers enforce output layout", () => {
  const base = {
    shape: [16],
    batch: 1,
    outputLayout: "batch-major",
    input: { channels: 8, channelIndex: 1 },
    output: { channels: 8, channelIndex: 2, kernelStepChannels: 1 },
  };

  const kernelMajor = createFftConvKernelMajorChannelLanePreset(base);
  const batchMajor = createFftConvBatchMajorChannelLanePreset({ ...base, outputLayout: "kernel-major" });

  assert.equal(kernelMajor.fftConv.outputLayout, "kernel-major");
  assert.equal(batchMajor.fftConv.outputLayout, "batch-major");
});

test("fftconv channel-lane preset validates output kernel lane capacity", () => {
  assert.throws(
    () =>
      createFftConvChannelLanePreset({
        shape: [32],
        batch: 1,
        kernelCount: 3,
        input: { channels: 8 },
        output: {
          channels: 4,
          channelIndex: 1,
          kernelStepChannels: 2,
        },
      }),
    /does not fit kernelCount=3/
  );
});

test("fftconv channel-lane preset rejects conflicting layout descriptors", () => {
  assert.throws(
    () =>
      createFftConvChannelLanePreset({
        shape: [32],
        batch: 1,
        input: { channels: 4 },
        output: { channels: 4 },
        layout: { whdcn: { channels: 4 } },
      }),
    /layout\.whdcn cannot be combined/
  );

  assert.throws(
    () =>
      createFftConvChannelLanePreset({
        shape: [32],
        batch: 1,
        input: { channels: 4 },
        output: { channels: 4 },
        layout: { inputStrides: [1] },
      }),
    /layout\.inputStrides cannot be combined/
  );
});
