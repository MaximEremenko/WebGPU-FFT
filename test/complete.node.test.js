import test, { describe, before } from "node:test";
import assert from "node:assert/strict";

import { registerCompleteTests } from "./complete.suite.js";

async function getWebGpuDevice() {
  const gpu = globalThis.navigator?.gpu ?? globalThis.gpu;
  if (!gpu?.requestAdapter) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  return adapter.requestDevice();
}

function assertCloseArray(actual, expected, tolAbs = 1e-4, tolRel = 1e-4, message = "arrays not close") {
  assert.equal(actual.length, expected.length, `${message}: length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const tol = tolAbs + tolRel * Math.abs(e);
    const diff = Math.abs(a - e);
    if (diff > tol) {
      throw new assert.AssertionError({ message: `${message}: i=${i} actual=${a} expected=${e} diff=${diff} tol=${tol}` });
    }
  }
}

class SkipError extends Error {
  constructor(message) {
    super(message);
    this.name = "SkipError";
  }
}

describe("createPlan complete (GPU) â€“ suite", () => {
  let device = null;
  before(async () => {
    device = await getWebGpuDevice();
  });

  const wrap = (name, fn) =>
    test(name, async (t) => {
      try {
        await fn();
      } catch (e) {
        if (e && e.name === "SkipError") t.skip(e.message);
        else throw e;
      }
    });

  registerCompleteTests({
    test: wrap,
    getDevice: async () => device,
    assert: (cond, msg) => assert.ok(cond, msg),
    assertCloseArray,
    SkipError,
    log: () => {},
    exportArtifact: null,
  });
});


