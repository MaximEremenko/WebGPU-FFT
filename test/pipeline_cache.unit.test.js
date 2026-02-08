import test from "node:test";
import assert from "node:assert/strict";

import { exportPipelineCacheSnapshot, importPipelineCacheSnapshot } from "../src/runtime/pipeline_cache.js";

class FakeDevice {
  constructor() {
    this._id = 1;
  }

  createShaderModule(desc) {
    return { kind: "sm", id: this._id++, code: desc.code };
  }

  createComputePipeline(desc) {
    return { kind: "cp", id: this._id++, desc };
  }
}

test("pipeline_cache: empty export uses v2 schema", () => {
  const device = new FakeDevice();
  const snapshot = exportPipelineCacheSnapshot(device);
  assert.equal(snapshot.schema, "webgpufft.pipeline-cache");
  assert.equal(snapshot.version, 2);
  assert.ok(Number.isSafeInteger(snapshot.createdAtMs));
  assert(Array.isArray(snapshot.shaderCodes));
  assert(Array.isArray(snapshot.pipelineKeys));
});

test("pipeline_cache: import accepts legacy snapshot and upgrades to v2", () => {
  const device = new FakeDevice();
  const imported = importPipelineCacheSnapshot(device, {
    version: 1,
    shaderCodes: ["code-a", "code-a", "code-b"],
    pipelineKeys: ["p0", "p0", "p1"],
  });
  assert.equal(imported.schema, "webgpufft.pipeline-cache");
  assert.equal(imported.version, 2);
  assert.deepEqual(imported.shaderCodes.sort(), ["code-a", "code-b"]);
  assert.deepEqual(imported.pipelineKeys.sort(), ["p0", "p1"]);
  assert.equal(imported.metadata?.fromVersion, 1);
});

test("pipeline_cache: import rejects unsupported future versions", () => {
  const device = new FakeDevice();
  assert.throws(
    () =>
      importPipelineCacheSnapshot(device, {
        schema: "webgpufft.pipeline-cache",
        version: 99,
        createdAtMs: Date.now(),
        shaderCodes: [],
        pipelineKeys: [],
      }),
    /Unsupported pipeline cache snapshot version: 99/
  );
});

test("pipeline_cache: import rejects mismatched schema for v2 snapshots", () => {
  const device = new FakeDevice();
  assert.throws(
    () =>
      importPipelineCacheSnapshot(device, {
        schema: "wrong.schema",
        version: 2,
        createdAtMs: Date.now(),
        shaderCodes: [],
        pipelineKeys: [],
      }),
    /snapshot\.schema must be "webgpufft\.pipeline-cache"/
  );
});

test("pipeline_cache: import validates metadata primitive values", () => {
  const device = new FakeDevice();
  assert.throws(
    () =>
      importPipelineCacheSnapshot(device, {
        schema: "webgpufft.pipeline-cache",
        version: 2,
        createdAtMs: Date.now(),
        metadata: { bad: { nested: true } },
        shaderCodes: [],
        pipelineKeys: [],
      }),
    /metadata\["bad"\] must be a primitive/
  );
});

