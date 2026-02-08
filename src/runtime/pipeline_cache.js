// Copyright (c) 2026 Maksim Eremenko

const CACHE_SNAPSHOT_SCHEMA = "webgpufft.pipeline-cache";
const CACHE_SNAPSHOT_VERSION = 2;
const MIN_SUPPORTED_SNAPSHOT_VERSION = 1;

function normalizeSnapshotMetadata(meta, { fromVersion }) {
  if (meta == null) {
    return {
      source: "webgpufft",
      fromVersion,
    };
  }
  if (typeof meta !== "object" || Array.isArray(meta)) {
    throw new Error("pipeline cache snapshot.metadata must be an object");
  }
  const out = {};
  for (const [k, v] of Object.entries(meta)) {
    if (typeof k !== "string" || k.length === 0) {
      throw new Error("pipeline cache snapshot.metadata keys must be non-empty strings");
    }
    if (v == null || typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
      out[k] = v;
      continue;
    }
    throw new Error(
      `pipeline cache snapshot.metadata["${k}"] must be a primitive (string/number/boolean/null)`
    );
  }
  if (!Object.prototype.hasOwnProperty.call(out, "source")) {
    out.source = "webgpufft";
  }
  if (!Object.prototype.hasOwnProperty.call(out, "fromVersion")) {
    out.fromVersion = fromVersion;
  }
  return out;
}

function makeSnapshot({
  shaderCodes = [],
  pipelineKeys = [],
  createdAtMs = Date.now(),
  metadata = null,
  fromVersion = CACHE_SNAPSHOT_VERSION,
} = {}) {
  if (!Number.isSafeInteger(createdAtMs) || createdAtMs < 0) {
    throw new Error(`pipeline cache snapshot.createdAtMs must be a non-negative safe integer; got ${createdAtMs}`);
  }
  return {
    schema: CACHE_SNAPSHOT_SCHEMA,
    version: CACHE_SNAPSHOT_VERSION,
    createdAtMs,
    metadata: normalizeSnapshotMetadata(metadata, { fromVersion }),
    shaderCodes: [...new Set(shaderCodes)],
    pipelineKeys: [...new Set(pipelineKeys)],
  };
}

function emptySnapshot() {
  return makeSnapshot();
}

function normalizeSnapshot(snapshot) {
  if (snapshot == null) return emptySnapshot();
  if (typeof snapshot !== "object" || Array.isArray(snapshot)) {
    throw new Error("pipeline cache snapshot must be an object");
  }
  if (!Number.isInteger(snapshot.version)) {
    throw new Error(`pipeline cache snapshot.version must be an integer; got ${snapshot.version}`);
  }
  if (snapshot.version > CACHE_SNAPSHOT_VERSION) {
    throw new Error(
      `Unsupported pipeline cache snapshot version: ${snapshot.version} (max supported=${CACHE_SNAPSHOT_VERSION})`
    );
  }
  if (snapshot.version < MIN_SUPPORTED_SNAPSHOT_VERSION) {
    throw new Error(
      `Unsupported pipeline cache snapshot version: ${snapshot.version} (min supported=${MIN_SUPPORTED_SNAPSHOT_VERSION})`
    );
  }

  if (snapshot.version >= 2) {
    if (snapshot.schema !== CACHE_SNAPSHOT_SCHEMA) {
      throw new Error(
        `pipeline cache snapshot.schema must be "${CACHE_SNAPSHOT_SCHEMA}" for version ${snapshot.version}; got ${snapshot.schema}`
      );
    }
  }

  const shaderCodes = snapshot.shaderCodes ?? snapshot.shaders ?? [];
  if (!Array.isArray(shaderCodes) || !shaderCodes.every((x) => typeof x === "string")) {
    throw new Error("pipeline cache snapshot.shaderCodes must be an array of strings");
  }

  const pipelineKeys = snapshot.pipelineKeys ?? snapshot.pipelines ?? [];
  if (!Array.isArray(pipelineKeys) || !pipelineKeys.every((x) => typeof x === "string")) {
    throw new Error("pipeline cache snapshot.pipelineKeys must be an array of strings");
  }

  const createdAtMs = snapshot.createdAtMs ?? Date.now();
  if (!Number.isSafeInteger(createdAtMs) || createdAtMs < 0) {
    throw new Error(
      `pipeline cache snapshot.createdAtMs must be a non-negative safe integer; got ${createdAtMs}`
    );
  }

  return makeSnapshot({
    shaderCodes,
    pipelineKeys,
    createdAtMs,
    metadata: snapshot.version >= 2 ? snapshot.metadata ?? null : null,
    fromVersion: snapshot.version,
  });
}

const DEVICE_PIPELINE_CACHES = new WeakMap();

function createSharedState() {
  return {
    modules: new Map(),
    pipelines: new Map(),
    layoutIds: new WeakMap(),
    nextLayoutId: 1,
    stablePipelineKeys: new Set(),
    lastImportedFromVersion: CACHE_SNAPSHOT_VERSION,
  };
}

export class PipelineCache {
  constructor(device, sharedState = null) {
    this.device = device;
    this._state = sharedState ?? createSharedState();
    this._modules = this._state.modules;
    this._pipelines = this._state.pipelines;
  }

  _getLayoutId(layout) {
    if (!layout || typeof layout !== "object") {
      throw new Error("getComputePipeline requires a valid pipeline layout object");
    }
    let id = this._state.layoutIds.get(layout);
    if (id == null) {
      id = this._state.nextLayoutId++;
      this._state.layoutIds.set(layout, id);
    }
    return id;
  }

  getShaderModule(code) {
    let m = this._modules.get(code);
    if (!m) {
      m = this.device.createShaderModule({ code });
      this._modules.set(code, m);
    }
    return m;
  }

  getComputePipeline({ code, layout, entryPoint = "main" }) {
    const layoutId = this._getLayoutId(layout);
    const key = `${layoutId}\n${entryPoint}\n${code}`;
    let p = this._pipelines.get(key);
    if (!p) {
      const module = this.getShaderModule(code);
      p = this.device.createComputePipeline({
        layout,
        compute: { module, entryPoint },
      });
      this._pipelines.set(key, p);
    }
    this._state.stablePipelineKeys.add(`${entryPoint}\n${code}`);
    return p;
  }

  exportSnapshot() {
    return makeSnapshot({
      createdAtMs: Date.now(),
      metadata: {
        source: "webgpufft",
        fromVersion: this._state.lastImportedFromVersion,
        modules: this._modules.size,
        pipelineKeys: this._state.stablePipelineKeys.size,
      },
      shaderCodes: Array.from(this._modules.keys()),
      pipelineKeys: Array.from(this._state.stablePipelineKeys),
    });
  }

  importSnapshot(snapshot) {
    const normalized = normalizeSnapshot(snapshot);
    this._state.lastImportedFromVersion = normalized.metadata?.fromVersion ?? CACHE_SNAPSHOT_VERSION;
    for (const code of normalized.shaderCodes) {
      this.getShaderModule(code);
    }
    for (const key of normalized.pipelineKeys) {
      this._state.stablePipelineKeys.add(key);
    }
    return this.exportSnapshot();
  }
}

export function getOrCreatePipelineCache(device, { snapshot = null } = {}) {
  let cache = DEVICE_PIPELINE_CACHES.get(device);
  if (!cache) {
    cache = new PipelineCache(device);
    DEVICE_PIPELINE_CACHES.set(device, cache);
  }
  if (snapshot != null) {
    cache.importSnapshot(snapshot);
  }
  return cache;
}

export function exportPipelineCacheSnapshot(device) {
  const cache = DEVICE_PIPELINE_CACHES.get(device);
  return cache ? cache.exportSnapshot() : emptySnapshot();
}

export function importPipelineCacheSnapshot(device, snapshot) {
  const cache = getOrCreatePipelineCache(device);
  return cache.importSnapshot(snapshot);
}
