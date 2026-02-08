// Copyright (c) 2026 Maksim Eremenko

import { getOrCreatePipelineCache } from "./pipeline_cache.js";
import { SegmentedCopier } from "./segmented_io.js";
import { pickWorkgroupSizeX } from "../utils/limits.js";

function parseRequestedWorkgroupSize(limits, tuning) {
  const fallback = pickWorkgroupSizeX(limits, 256);
  if (!tuning || typeof tuning !== "object") return fallback;

  const requested = tuning.workgroupSizeX ?? tuning.workgroupSize ?? null;
  if (requested == null) return fallback;
  if (!Number.isInteger(requested) || requested <= 0) {
    throw new Error(`tuning.workgroupSizeX/workgroupSize must be a positive integer; got ${requested}`);
  }

  const maxX = limits?.maxComputeWorkgroupSizeX ?? fallback;
  const maxInv = limits?.maxComputeInvocationsPerWorkgroup ?? fallback;
  if (requested > maxX || requested > maxInv) {
    throw new Error(
      [
        `Requested workgroup size ${requested} exceeds device limits.`,
        `maxComputeWorkgroupSizeX=${maxX}`,
        `maxComputeInvocationsPerWorkgroup=${maxInv}`,
      ].join("\n")
    );
  }
  return requested | 0;
}

export class BasePlan {
  constructor(device, opts = null) {
    this.device = device;
    const snapshot = opts?.cache?.snapshot ?? opts?.pipelineCacheSnapshot ?? null;
    this.cache = getOrCreatePipelineCache(device, { snapshot });
    this.copier = new SegmentedCopier(device, this.cache);
    this.workgroupSize = parseRequestedWorkgroupSize(device.limits, opts?.tuning ?? null);
    this._destroyed = false;
  }

  getWorkspaceSizeBytes() {
    return 0;
  }

  getPipelineCacheSnapshot() {
    return this.cache.exportSnapshot();
  }

  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    this.copier.destroy();
  }
}
