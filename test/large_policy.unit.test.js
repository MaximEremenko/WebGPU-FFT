import test from "node:test";
import assert from "node:assert/strict";

import {
  mergeLargeRouteMetadata,
  resolveAxisKindsForShape,
  resolveLargeRoutingPolicy,
  resolveOutOfCoreAxisWindowPolicy,
} from "../src/runtime/large_policy.js";

function makeDevice(limitOverrides = {}) {
  return {
    limits: {
      maxStorageBufferBindingSize: 256,
      maxBufferSize: 1 << 30,
      ...limitOverrides,
    },
  };
}

test("large_policy: normal route when all required bindings fit", () => {
  const p = resolveLargeRoutingPolicy({
    device: makeDevice({ maxStorageBufferBindingSize: 256 }),
    requiredBindingBytes: [64, 128, 256],
    lineBytes: [64],
    precision: "f32",
  });

  assert.equal(p.needsLargeMode, false);
  assert.equal(p.useOutOfCore, false);
  assert.equal(p.routeMode, "normal");
  assert.deepEqual(p.attemptedRoutes, ["direct"]);
  assert.ok(p.reasonCodes.includes("within-bindings"));
  assert.ok(p.reasonCodes.includes("normal"));
});

test("large_policy: large-chunk route when large mode is needed but out-of-core is not selected", () => {
  const p = resolveLargeRoutingPolicy({
    device: makeDevice({ maxStorageBufferBindingSize: 256 }),
    requiredBindingBytes: [1024],
    lineBytes: [64],
    allowOutOfCore: false,
    bytesPerBatch: 128,
    precision: "f32",
  });

  assert.equal(p.needsLargeMode, true);
  assert.equal(p.useOutOfCore, false);
  assert.equal(p.routeMode, "large-chunk");
  assert.ok(p.attemptedRoutes.includes("dispatch-split"));
  assert.ok(p.attemptedRoutes.includes("batch-chunk"));
  assert.ok(p.reasonCodes.includes("requires-large-bindings"));
  assert.ok(p.reasonCodes.includes("large-chunk"));
});

test("large_policy: large-out-of-core route when bytesPerBatch exceeds binding limit and out-of-core is eligible", () => {
  const p = resolveLargeRoutingPolicy({
    device: makeDevice({ maxStorageBufferBindingSize: 256 }),
    requiredBindingBytes: [4096],
    lineBytes: [512, 32],
    axisKinds: ["mixed", "mixed"],
    axisLengths: [64, 4],
    allowOutOfCore: true,
    rank: 2,
    bytesPerBatch: 1024,
    precision: "f32",
  });

  assert.equal(p.needsLargeMode, true);
  assert.equal(p.requiresOutOfCore, true);
  assert.equal(p.outOfCoreEligible, true);
  assert.equal(p.useOutOfCore, true);
  assert.equal(p.routeMode, "large-out-of-core");
  assert.ok(p.attemptedRoutes.includes("out-of-core-four-step"));
  assert.ok(p.reasonCodes.includes("bytes-per-batch-exceeds-bind"));
  assert.ok(p.reasonCodes.includes("out-of-core-eligible"));
  assert.ok(p.reasonCodes.includes("large-out-of-core"));
});

test("large_policy: strided preference can select out-of-core even when bytesPerBatch fits", () => {
  const p = resolveLargeRoutingPolicy({
    device: makeDevice({ maxStorageBufferBindingSize: 256 }),
    requiredBindingBytes: [1024],
    lineBytes: [128, 64],
    axisKinds: ["mixed", "mixed"],
    axisLengths: [16, 8],
    allowOutOfCore: true,
    rank: 2,
    bytesPerBatch: 128,
    hasStridedIO: true,
    preferOutOfCoreForStrided: true,
    precision: "f32",
  });

  assert.equal(p.needsLargeMode, true);
  assert.equal(p.requiresOutOfCore, false);
  assert.equal(p.prefersOutOfCore, true);
  assert.equal(p.useOutOfCore, true);
  assert.equal(p.routeMode, "large-out-of-core");
  assert.ok(p.reasonCodes.includes("strided-prefers-out-of-core"));
});

test("large_policy: throws when out-of-core is required but axis strategy support is unavailable", () => {
  assert.throws(
    () =>
      resolveLargeRoutingPolicy({
        device: makeDevice({ maxStorageBufferBindingSize: 64 }),
        requiredBindingBytes: [4096],
        lineBytes: [136, 32],
        axisKinds: ["mixed", "mixed"],
        axisLengths: [17, 4],
        allowOutOfCore: true,
        rank: 2,
        bytesPerBatch: 1024,
        precision: "f32",
      }),
    /out-of-core fallback/i
  );
});

test("large_policy: precision-restricted large mode throws with clear error", () => {
  assert.throws(
    () =>
      resolveLargeRoutingPolicy({
        device: makeDevice({ maxStorageBufferBindingSize: 64 }),
        requiredBindingBytes: [1024],
        lineBytes: [32],
        precision: "f16-storage",
        requireLargePrecision: "f32",
        requireLargePrecisionError: 'large path supports precision:"f32" only',
      }),
    /supports precision:"f32" only/
  );
});

test("large_policy: outOfCoreUnsupportedError callback receives attempted routes and reason codes", () => {
  let captured = null;
  assert.throws(
    () =>
      resolveLargeRoutingPolicy({
        device: makeDevice({ maxStorageBufferBindingSize: 64 }),
        requiredBindingBytes: [4096],
        lineBytes: [136, 32],
        axisKinds: ["mixed", "mixed"],
        axisLengths: [17, 4],
        allowOutOfCore: true,
        rank: 2,
        bytesPerBatch: 1024,
        precision: "f32",
        outOfCoreUnsupportedError: (ctx) => {
          captured = ctx;
          return "custom out-of-core unsupported";
        },
      }),
    /custom out-of-core unsupported/
  );
  assert.ok(captured);
  assert.ok(Array.isArray(captured.attemptedRoutes));
  assert.ok(Array.isArray(captured.reasonCodes));
  assert.ok(captured.attemptedRoutes.includes("out-of-core-four-step"));
  assert.ok(captured.reasonCodes.includes("axis-line-unsupported"));
});

test("large_policy: resolveAxisKindsForShape auto-selects mixed/rader/bluestein with forced overrides", () => {
  const a = resolveAxisKindsForShape({
    shape: [8, 29, 34],
    tuning: { raderMaxPrime: 64, forceBluesteinAxes: [0], forceRaderAxes: [1] },
  });
  assert.deepEqual(a.axisKinds, ["bluestein", "rader", "bluestein"]);
});

test("large_policy: resolveAxisKindsForShape rejects invalid forced Rader axis", () => {
  assert.throws(
    () =>
      resolveAxisKindsForShape({
        shape: [8, 16],
        tuning: { forceRaderAxes: [1] },
      }),
    /is not prime/
  );
});

test("large_policy: out-of-core axis window policy defaults to max bindable lines in single-upload mode", () => {
  const p = resolveOutOfCoreAxisWindowPolicy({
    axisLen: 256,
    lineBytes: 2048,
    linesTotal: 65536,
    maxBindBytes: 1 << 24,
    axisKind: "mixed",
    tuning: null,
    axisIndex: 0,
    storageAlign: 256,
  });
  assert.equal(p.numAxisUploads, 1);
  assert.equal(p.maxLinesByBind, (1 << 24) / 2048);
  assert.equal(p.linesPerChunk, p.maxLinesByBind);
  assert.equal(p.alignedLineStep, 1);
});

test("large_policy: out-of-core axis window policy applies staged upload/grouped-batch/alignment rules", () => {
  const p = resolveOutOfCoreAxisWindowPolicy({
    axisLen: 4096,
    lineBytes: 264,
    linesTotal: 4096,
    maxBindBytes: 65536,
    axisKind: "bluestein",
    tuning: {
      swapTo2Stage4Step: 1024,
      swapTo3Stage4Step: 4096,
      groupedBatch: [null, 8],
      outOfCoreBurstWindows: 3,
    },
    axisIndex: 1,
    storageAlign: 256,
  });
  assert.equal(p.numAxisUploads, 3);
  assert.equal(p.groupedBatch, 8);
  assert.equal(p.burstWindows, 3);
  assert.ok(p.linesPerChunk >= 1);
  assert.ok(p.linesPerChunk <= p.maxLinesByBind);
  assert.equal((p.linesPerChunk * p.lineBytes) % 256, 0);
});

test("large_policy: mergeLargeRouteMetadata promotes out-of-core route and merges arrays", () => {
  const merged = mergeLargeRouteMetadata([
    {
      routeMode: "large-chunk",
      reasonCodes: ["requires-large-bindings", "large-chunk"],
      attemptedRoutes: ["direct", "dispatch-split", "batch-chunk"],
    },
    {
      routeMode: "large-out-of-core",
      reasonCodes: ["bytes-per-batch-exceeds-bind", "out-of-core-eligible"],
      attemptedRoutes: ["out-of-core-four-step"],
    },
  ]);
  assert.equal(merged.routeMode, "large-out-of-core");
  assert.ok(merged.reasonCodes.includes("requires-large-bindings"));
  assert.ok(merged.reasonCodes.includes("bytes-per-batch-exceeds-bind"));
  assert.ok(merged.reasonCodes.includes("large-out-of-core"));
  assert.ok(merged.attemptedRoutes.includes("batch-chunk"));
  assert.ok(merged.attemptedRoutes.includes("out-of-core-four-step"));
});

test('large_policy: tuning.largeRoute="out-of-core" forces out-of-core when eligible', () => {
  const p = resolveLargeRoutingPolicy({
    device: makeDevice({ maxStorageBufferBindingSize: 256 }),
    tuning: { largeRoute: "out-of-core" },
    requiredBindingBytes: [1024],
    lineBytes: [128, 64],
    axisKinds: ["mixed", "mixed"],
    axisLengths: [16, 8],
    allowOutOfCore: true,
    rank: 2,
    bytesPerBatch: 128,
    precision: "f32",
  });
  assert.equal(p.routeMode, "large-out-of-core");
  assert.equal(p.useOutOfCore, true);
  assert.equal(p.requestedLargeRoute, "out-of-core");
  assert.ok(p.reasonCodes.includes("forced-route-out-of-core"));
});

test('large_policy: tuning.largeRoute="chunk" rejects when out-of-core is required', () => {
  assert.throws(
    () =>
      resolveLargeRoutingPolicy({
        device: makeDevice({ maxStorageBufferBindingSize: 64 }),
        tuning: { largeRoute: "chunk" },
        requiredBindingBytes: [4096],
        lineBytes: [32, 32],
        axisKinds: ["mixed", "mixed"],
        axisLengths: [4, 4],
        allowOutOfCore: true,
        rank: 2,
        bytesPerBatch: 1024,
        precision: "f32",
      }),
    /largeRoute="chunk" is incompatible/
  );
});

test("large_policy: tuning.preferOutOfCoreForStrided overrides call-site default", () => {
  const p = resolveLargeRoutingPolicy({
    device: makeDevice({ maxStorageBufferBindingSize: 256 }),
    tuning: { preferOutOfCoreForStrided: false },
    requiredBindingBytes: [1024],
    lineBytes: [128, 64],
    axisKinds: ["mixed", "mixed"],
    axisLengths: [16, 8],
    allowOutOfCore: true,
    rank: 2,
    bytesPerBatch: 128,
    hasStridedIO: true,
    preferOutOfCoreForStrided: true,
    precision: "f32",
  });
  assert.equal(p.prefersOutOfCore, false);
  assert.equal(p.effectivePreferOutOfCoreForStrided, false);
  assert.equal(p.routeMode, "large-chunk");
});
