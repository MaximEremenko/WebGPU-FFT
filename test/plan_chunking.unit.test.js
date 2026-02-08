import test from "node:test";
import assert from "node:assert/strict";

import { createFftPlan } from "../src/plan.js";

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
      minStorageBufferOffsetAlignment: 256,
      maxComputeWorkgroupsPerDimension: [65535, 65535, 65535],
      ...limitOverrides,
    };
    this.features = { has: () => false };
    this._id = 1;
    this.queue = {
      writes: [],
      writeBuffer: (buffer, offset, data) => {
        const bytes = new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
        this.queue.writes.push({ buffer, offset, bytes });
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
  };
}

function bytesToU32(bytes) {
  return new Uint32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
}

test("plan exec chunks dispatches when workgroup-x limit is tight", () => {
  const device = new FakeDevice({ maxComputeWorkgroupsPerDimension: [2, 65535, 65535] });
  const plan = createFftPlan(device, {
    shape: [32],
    direction: "forward",
    normalize: "none",
    inPlace: false,
    layout: "interleaved",
    precision: "f32",
  });

  const totalComplex = 32;
  const totalBytes = totalComplex * 8;
  const input = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const totalPasses = plan._pipelines.reduce((a, axisStages) => a + axisStages.length, 0);
  const workgroupsX = Math.ceil(totalComplex / plan._workgroupSizeX);
  const chunkCount = Math.ceil(workgroupsX / device.limits.maxComputeWorkgroupsPerDimension[0]);
  const expectedDispatches = totalPasses * chunkCount;

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.equal(computePasses.length, expectedDispatches);
  for (const pass of computePasses) {
    assert.equal(pass.dispatches.length, 1);
    assert.ok(pass.dispatches[0].x <= device.limits.maxComputeWorkgroupsPerDimension[0]);
  }

  const copies = encoder.ops.filter((op) => op.type === "copy");
  assert.equal(copies.length, expectedDispatches);
  for (const c of copies) assert.equal(c.size, 16);

  assert.equal(device.queue.writes.length, 1);
  const u32 = bytesToU32(device.queue.writes[0].bytes);
  assert.equal(u32.length, chunkCount * 4);
  for (let i = 0; i < chunkCount; i++) {
    const base = i * 4;
    assert.equal(u32[base], totalComplex);
    assert.equal(u32[base + 1], i * device.limits.maxComputeWorkgroupsPerDimension[0] * plan._workgroupSizeX);
    assert.equal(u32[base + 2], 0);
    assert.equal(u32[base + 3], 0);
  }
});

test("plan exec keeps single-dispatch path when limits allow it", () => {
  const device = new FakeDevice({ maxComputeWorkgroupsPerDimension: [1024, 65535, 65535] });
  const plan = createFftPlan(device, {
    shape: [32],
    direction: "forward",
    normalize: "none",
    inPlace: false,
    layout: "interleaved",
    precision: "f32",
  });

  const totalComplex = 32;
  const totalBytes = totalComplex * 8;
  const input = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const totalPasses = plan._pipelines.reduce((a, axisStages) => a + axisStages.length, 0);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.equal(computePasses.length, totalPasses);

  const copies = encoder.ops.filter((op) => op.type === "copy");
  assert.equal(copies.length, 0);

  assert.equal(device.queue.writes.length, 1);
  const u32 = bytesToU32(device.queue.writes[0].bytes);
  assert.deepEqual(Array.from(u32), [totalComplex, 0, 0, 0]);
});

test("plan exec chunks by batch when total binding bytes exceed max but per-batch fits", () => {
  const N = 8;
  const batch = 4;
  const bytesPerBatch = N * 8;
  const maxBind = bytesPerBatch; // force one-batch chunks
  const device = new FakeDevice({
    maxStorageBufferBindingSize: maxBind,
    minStorageBufferOffsetAlignment: 1,
    maxComputeWorkgroupsPerDimension: [1024, 65535, 65535],
  });

  const plan = createFftPlan(device, {
    shape: [N],
    direction: "forward",
    normalize: "none",
    inPlace: false,
    layout: "interleaved",
    precision: "f32",
  });

  const totalBytes = N * batch * 8;
  const input = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output, batch });

  const totalPasses = plan._pipelines.reduce((a, axisStages) => a + axisStages.length, 0);
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.equal(computePasses.length, totalPasses * batch);

  for (const pass of computePasses) {
    const entries = pass.bindGroup.desc.entries;
    const src = entries.find((e) => e.binding === 0);
    const dst = entries.find((e) => e.binding === 1);
    assert.ok(src.resource.size <= maxBind);
    assert.ok(dst.resource.size <= maxBind);
  }

  // One params write per chunked batch invocation.
  assert.equal(device.queue.writes.length, batch);
  for (const w of device.queue.writes) {
    const u32 = bytesToU32(w.bytes);
    assert.deepEqual(Array.from(u32), [N, 0, 0, 0]);
  }
});

test("createFftPlan allows oversized axis-0-only plans when one line fits binding", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1024,
    minStorageBufferOffsetAlignment: 128,
  });

  assert.doesNotThrow(() =>
    createFftPlan(device, {
      shape: [16, 16],
      axes: [0],
      direction: "forward",
      normalize: "none",
      inPlace: false,
      layout: "interleaved",
      precision: "f32",
    })
  );

  assert.throws(
    () =>
      createFftPlan(device, {
        shape: [16, 16],
        axes: [1],
        direction: "forward",
        normalize: "none",
        inPlace: false,
        layout: "interleaved",
        precision: "f32",
      }),
    /maxStorageBufferBindingSize/
  );
});

test("axis-0 window fallback runs oversized plan with bounded per-window bindings", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1024,
    minStorageBufferOffsetAlignment: 128,
    maxComputeWorkgroupsPerDimension: [1024, 65535, 65535],
  });
  const plan = createFftPlan(device, {
    shape: [16, 16],
    axes: [0],
    direction: "forward",
    normalize: "none",
    inPlace: false,
    layout: "interleaved",
    precision: "f32",
  });

  const totalComplex = 16 * 16;
  const totalBytes = totalComplex * 8;
  const input = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const totalPasses = plan._pipelines.reduce((a, axisStages) => a + axisStages.length, 0);
  const windows = 2; // 2048 bytes total, 1024 bytes window limit
  const expectedComputePasses = totalPasses * windows;
  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.equal(computePasses.length, expectedComputePasses);

  for (const pass of computePasses) {
    const entries = pass.bindGroup.desc.entries;
    const src = entries.find((e) => e.binding === 0);
    const dst = entries.find((e) => e.binding === 1);
    assert.ok(src.resource.size <= device.limits.maxStorageBufferBindingSize);
    assert.ok(dst.resource.size <= device.limits.maxStorageBufferBindingSize);
  }

  const copies = encoder.ops.filter((op) => op.type === "copy");
  const paramCopies = copies.filter((c) => c.size === 16 && c.dst === plan._paramsBuffer);
  assert.equal(paramCopies.length, expectedComputePasses);

  // One write uploads all window/chunk params:
  // [total, baseIndex, lineOffset, elementBase] per window.
  assert.equal(device.queue.writes.length, 1);
  const u32 = bytesToU32(device.queue.writes[0].bytes);
  assert.deepEqual(Array.from(u32), [128, 0, 0, 0, 128, 0, 8, 128]);
});

test("axis-0 window fallback uses direct in-place path for aligned GPUBuffer windows", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1024,
    minStorageBufferOffsetAlignment: 128,
    maxComputeWorkgroupsPerDimension: [1024, 65535, 65535],
  });
  const plan = createFftPlan(device, {
    shape: [16, 16],
    axes: [0],
    direction: "forward",
    normalize: "none",
    inPlace: true,
    layout: "interleaved",
    precision: "f32",
  });

  const totalBytes = 16 * 16 * 8;
  const input = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const encoder = makeEncoder();
  plan.exec(encoder, { input });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.equal(computePasses.length, 4); // 2 stages * 2 windows
  for (const pass of computePasses) {
    const entries = pass.bindGroup.desc.entries;
    const src = entries.find((e) => e.binding === 0);
    const dst = entries.find((e) => e.binding === 1);
    assert.ok(src.resource.size <= device.limits.maxStorageBufferBindingSize);
    assert.ok(dst.resource.size <= device.limits.maxStorageBufferBindingSize);
  }

  const copies = encoder.ops.filter((op) => op.type === "copy");
  const paramCopies = copies.filter((c) => c.size === 16 && c.dst === plan._paramsBuffer);
  assert.equal(paramCopies.length, 4);
  const dataCopies = copies.filter((c) => !(c.size === 16 && c.dst === plan._paramsBuffer));
  assert.equal(dataCopies.length, 0);
});

test("axis-0 window fallback supports segmented BufferView input/output", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 1024,
    minStorageBufferOffsetAlignment: 128,
    maxComputeWorkgroupsPerDimension: [1024, 65535, 65535],
  });
  const plan = createFftPlan(device, {
    shape: [16, 16],
    axes: [0],
    direction: "forward",
    normalize: "none",
    inPlace: false,
    layout: "interleaved",
    precision: "f32",
  });

  const totalBytes = 16 * 16 * 8;
  const inA = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const inB = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const outA = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const outB = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const inputView = {
    segments: [
      { buffer: inA, offsetBytes: 0, sizeBytes: 1024 },
      { buffer: inB, offsetBytes: 0, sizeBytes: 1024 },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };
  const outputView = {
    segments: [
      { buffer: outA, offsetBytes: 0, sizeBytes: 1024 },
      { buffer: outB, offsetBytes: 0, sizeBytes: 1024 },
    ],
    logicalByteOffset: 0,
    lengthBytes: totalBytes,
  };

  const encoder = makeEncoder();
  plan.exec(encoder, { input: inputView, output: outputView });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.equal(computePasses.length, 4); // 2 stages * 2 windows

  const copies = encoder.ops.filter((op) => op.type === "copy");
  const gatherCopies = copies.filter((c) => c.src === inA || c.src === inB);
  assert.equal(gatherCopies.length, 2);
  const scatterCopies = copies.filter((c) => c.dst === outA || c.dst === outB);
  assert.equal(scatterCopies.length, 2);
  const paramCopies = copies.filter((c) => c.dst === plan._paramsBuffer && c.size === 16);
  assert.equal(paramCopies.length, 4);
});

test("createFftPlan enables axis-0 two-step fallback when one line exceeds binding limit", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 64, // max axis elems per direct binding = 8
    minStorageBufferOffsetAlignment: 1,
    maxComputeWorkgroupsPerDimension: [1024, 65535, 65535],
  });

  const plan = createFftPlan(device, {
    shape: [32, 4], // axis0 line is 256 bytes (>64), but 32 = 4 * 8 is factorable within bind cap
    axes: [0],
    direction: "forward",
    normalize: "none",
    inPlace: false,
    layout: "interleaved",
    precision: "f32",
  });

  assert.ok(plan._axis0TwoStep);
  assert.equal(plan._axis0TwoStep.n1 * plan._axis0TwoStep.n2, 32);
});

test("axis-0 two-step fallback executes with bounded per-pass binding sizes", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 64,
    minStorageBufferOffsetAlignment: 1,
    maxComputeWorkgroupsPerDimension: [1024, 65535, 65535],
  });
  const plan = createFftPlan(device, {
    shape: [32, 4],
    axes: [0],
    direction: "forward",
    normalize: "none",
    inPlace: false,
    layout: "interleaved",
    precision: "f32",
  });
  assert.ok(plan._axis0TwoStep);

  const totalBytes = 32 * 4 * 8;
  const input = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

  const encoder = makeEncoder();
  plan.exec(encoder, { input, output });

  const computePasses = encoder.ops.filter((op) => op.type === "compute");
  assert.ok(computePasses.length > 0);
  for (const pass of computePasses) {
    const entries = pass.bindGroup.desc.entries;
    for (const e of entries) {
      const sz = e.resource?.size;
      if (typeof sz === "number") {
        assert.ok(sz <= device.limits.maxStorageBufferBindingSize || e.binding === 1);
      }
    }
  }
  assert.ok(encoder.ops.some((op) => op.type === "copy"));
});

test("createFftPlan rejects axis-0 oversized line when two-step factorization is unavailable", () => {
  const device = new FakeDevice({
    maxStorageBufferBindingSize: 64, // max axis elems per direct binding = 8
    minStorageBufferOffsetAlignment: 1,
  });

  assert.throws(
    () =>
      createFftPlan(device, {
        shape: [17, 4], // prime > 8: cannot factor into n1*n2 <= 8
        axes: [0],
        direction: "forward",
        normalize: "none",
        inPlace: false,
        layout: "interleaved",
        precision: "f32",
      }),
    /two-step fallback is unavailable/
  );
});
