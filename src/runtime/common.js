// Copyright (c) 2026 Maksim Eremenko

import { formatDeviceLimits } from "../utils/limits.js";

export function assertOneOf(value, allowed, name) {
  if (!allowed.includes(value)) {
    throw new Error(
      `${name} must be one of ${allowed.map((v) => JSON.stringify(v)).join(", ")}; got ${JSON.stringify(value)}`
    );
  }
}

export function isPositiveInt(x) {
  return Number.isInteger(x) && x > 0;
}

export function prod(arr) {
  let p = 1;
  for (const v of arr) p *= v;
  return p;
}

export function align4Bytes(bytes) {
  if (!Number.isInteger(bytes) || bytes < 0) throw new Error(`align4Bytes expects a non-negative integer; got ${bytes}`);
  return (bytes + 3) & ~3;
}

export function alignBytes(bytes, alignment) {
  if (!Number.isInteger(bytes) || bytes < 0) throw new Error(`alignBytes expects a non-negative integer; got ${bytes}`);
  if (!Number.isInteger(alignment) || alignment <= 0) throw new Error(`alignBytes expects a positive integer alignment; got ${alignment}`);
  const rem = bytes % alignment;
  return rem === 0 ? bytes : bytes + (alignment - rem);
}

export function normalizeScaleFactor({ normalize, direction, nTotal }) {
  if (normalize === "none") return 1.0;
  if (normalize === "unitary") return 1.0 / Math.sqrt(nTotal);
  if (normalize === "backward") return direction === "inverse" ? 1.0 / nTotal : 1.0;
  throw new Error(`Unknown normalize mode: ${normalize}`);
}

export function ensureWithinBindingLimit(device, bytes, context) {
  const maxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
  if (bytes > maxBind) {
    throw new Error(
      [
        `Unsupported: required binding ${bytes} bytes exceeds device.limits.maxStorageBufferBindingSize=${maxBind}`,
        context ?? "",
        `limits:\n${formatDeviceLimits(device.limits)}`,
      ].join("\n")
    );
  }
}

export function isGpuBuffer(x) {
  return x && !x?.segments && typeof x.size === "number" && typeof x.destroy === "function";
}

export function getBufferByteLength(x) {
  if (x && typeof x.size === "number") return x.size;
  if (x && typeof x.lengthBytes === "number") return x.lengthBytes;
  throw new Error("Expected GPUBuffer or BufferView");
}

export function collectBackingBuffers(x, out = new Set()) {
  if (!x) return out;
  if (x.view) {
    collectBackingBuffers(x.view, out);
    return out;
  }
  if (x.buffer && typeof x.buffer.size === "number") {
    out.add(x.buffer);
    return out;
  }
  if (typeof x.size === "number" && typeof x.destroy === "function" && !x.segments) {
    out.add(x);
    return out;
  }
  const segs = x.segments;
  if (!Array.isArray(segs)) return out;
  for (const seg of segs) {
    if (seg?.buffer) out.add(seg.buffer);
  }
  return out;
}

export function buffersAlias(a, b) {
  if (!a || !b) return false;
  const aa = collectBackingBuffers(a);
  const bb = collectBackingBuffers(b);
  for (const buf of aa) {
    if (bb.has(buf)) return true;
  }
  return false;
}
