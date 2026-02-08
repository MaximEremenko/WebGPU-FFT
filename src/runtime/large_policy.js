// Copyright (c) 2026 Maksim Eremenko

import { factorizeSupportedRadices, isPrime } from "../utils/factors.js";

export function parseOptionalMaxStorageBufferBindingSize(tuning) {
  if (tuning == null || typeof tuning !== "object") return null;
  const v = tuning.maxStorageBufferBindingSize ?? null;
  if (v == null) return null;
  if (!Number.isInteger(v) || v <= 0) {
    throw new Error(`tuning.maxStorageBufferBindingSize must be a positive integer; got ${v}`);
  }
  return v;
}

export function resolveEffectiveMaxStorageBufferBindingSize(device, tuning) {
  const deviceMaxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
  const tuningMaxBind = parseOptionalMaxStorageBufferBindingSize(tuning);
  return tuningMaxBind != null ? Math.min(deviceMaxBind, tuningMaxBind) : deviceMaxBind;
}

function parseOptionalLargeRoutePreference(tuning) {
  if (tuning == null) return "auto";
  if (typeof tuning !== "object") {
    throw new Error("tuning must be an object when provided");
  }
  const value = tuning.largeRoute ?? "auto";
  if (typeof value !== "string") {
    throw new Error(`tuning.largeRoute must be one of "auto" | "chunk" | "out-of-core"; got ${value}`);
  }
  if (value !== "auto" && value !== "chunk" && value !== "out-of-core") {
    throw new Error(`tuning.largeRoute must be one of "auto" | "chunk" | "out-of-core"; got ${value}`);
  }
  return value;
}

function parseOptionalPreferOutOfCoreForStrided(tuning) {
  if (tuning == null || typeof tuning !== "object") return null;
  if (tuning.preferOutOfCoreForStrided == null) return null;
  if (typeof tuning.preferOutOfCoreForStrided !== "boolean") {
    throw new Error(
      `tuning.preferOutOfCoreForStrided must be boolean when provided; got ${tuning.preferOutOfCoreForStrided}`
    );
  }
  return tuning.preferOutOfCoreForStrided;
}

function parseOptionalNonNegativeInt(v, name) {
  if (v == null) return null;
  if (!Number.isInteger(v) || v < 0) {
    throw new Error(`${name} must be a non-negative integer when provided; got ${v}`);
  }
  return v;
}

function parseOptionalPositiveInt(v, name) {
  if (v == null) return null;
  if (!Number.isInteger(v) || v <= 0) {
    throw new Error(`${name} must be a positive integer when provided; got ${v}`);
  }
  return v;
}

function resolveGroupedBatchValue(tuning, axisIndex = 0) {
  if (tuning == null || typeof tuning !== "object") return null;
  const groupedBatch = tuning.groupedBatch ?? null;
  if (groupedBatch == null) return null;
  if (Number.isInteger(groupedBatch) && groupedBatch > 0) return groupedBatch;
  if (Array.isArray(groupedBatch)) {
    if (!Number.isInteger(axisIndex) || axisIndex < 0) {
      throw new Error(`axisIndex must be a non-negative integer; got ${axisIndex}`);
    }
    if (groupedBatch.length === 0) return null;
    const idx = Math.min(axisIndex, groupedBatch.length - 1);
    const v = groupedBatch[idx];
    if (v == null) return null;
    if (!Number.isInteger(v) || v <= 0) {
      throw new Error(`tuning.groupedBatch[${idx}] must be a positive integer when provided; got ${v}`);
    }
    return v;
  }
  throw new Error(`tuning.groupedBatch must be a positive integer or axis-indexed array; got ${groupedBatch}`);
}

function gcd(a, b) {
  let x = Math.abs(a | 0);
  let y = Math.abs(b | 0);
  while (y !== 0) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x || 1;
}

export function resolveOutOfCoreAxisWindowPolicy({
  axisLen,
  lineBytes,
  linesTotal,
  maxBindBytes,
  axisKind = "mixed",
  tuning = null,
  axisIndex = 0,
  storageAlign = 256,
}) {
  if (!Number.isInteger(axisLen) || axisLen <= 0) {
    throw new Error(`axisLen must be a positive integer; got ${axisLen}`);
  }
  if (!Number.isInteger(lineBytes) || lineBytes <= 0) {
    throw new Error(`lineBytes must be a positive integer; got ${lineBytes}`);
  }
  if (!Number.isInteger(linesTotal) || linesTotal <= 0) {
    throw new Error(`linesTotal must be a positive integer; got ${linesTotal}`);
  }
  const effectiveMaxBind = Number.isFinite(maxBindBytes) && maxBindBytes > 0 ? Math.floor(maxBindBytes) : Infinity;
  const maxLinesByBind = lineBytes <= effectiveMaxBind ? Math.max(1, Math.floor(effectiveMaxBind / lineBytes)) : 1;
  const swapTo2Stage4Step = parseOptionalNonNegativeInt(tuning?.swapTo2Stage4Step ?? null, "tuning.swapTo2Stage4Step") ?? 0;
  const swapTo3Stage4Step = parseOptionalNonNegativeInt(tuning?.swapTo3Stage4Step ?? null, "tuning.swapTo3Stage4Step") ?? 0;
  const burstWindows = parseOptionalPositiveInt(tuning?.outOfCoreBurstWindows ?? null, "tuning.outOfCoreBurstWindows") ?? 1;
  const groupedBatch = resolveGroupedBatchValue(tuning, axisIndex);

  let numAxisUploads = 1;
  if (swapTo3Stage4Step > 0 && axisLen >= swapTo3Stage4Step) {
    numAxisUploads = 3;
  } else if (swapTo2Stage4Step > 0 && axisLen >= swapTo2Stage4Step) {
    numAxisUploads = 2;
  } else {
    // Conservative auto policy: keep legacy behavior for small/typical sizes; only split windows for very large strided/non-mixed lines.
    if (axisKind !== "mixed" && axisLen >= 1024 && maxLinesByBind >= 8) numAxisUploads = 2;
    if (axisKind !== "mixed" && axisLen >= 4096 && maxLinesByBind >= 16) numAxisUploads = 3;
  }
  numAxisUploads = Math.max(1, Math.min(3, numAxisUploads, maxLinesByBind));

  let linesPerChunk = Math.max(1, Math.floor(maxLinesByBind / numAxisUploads));
  if (groupedBatch != null && linesPerChunk > 1) {
    if (linesPerChunk >= groupedBatch) {
      linesPerChunk = Math.max(groupedBatch, Math.floor(linesPerChunk / groupedBatch) * groupedBatch);
    } else {
      linesPerChunk = 1;
    }
  }

  let alignedLineStep = 1;
  if (Number.isFinite(storageAlign) && storageAlign > 1) {
    alignedLineStep = Math.max(1, Math.floor(storageAlign / gcd(storageAlign, lineBytes)));
    if (alignedLineStep > 1 && linesPerChunk >= alignedLineStep) {
      linesPerChunk = Math.max(alignedLineStep, Math.floor(linesPerChunk / alignedLineStep) * alignedLineStep);
    }
  }

  linesPerChunk = Math.max(1, Math.min(linesPerChunk, linesTotal));
  return {
    axisKind,
    axisLen,
    lineBytes,
    linesTotal,
    maxLinesByBind,
    groupedBatch,
    numAxisUploads,
    linesPerChunk,
    alignedLineStep,
    burstWindows,
  };
}

export function canAxisLenFitOrTwoStep(axisLen, maxBindBytes, maxBufferSize) {
  const lineBytes = axisLen * 8;
  if (lineBytes <= maxBindBytes) return true;
  if (lineBytes > maxBufferSize) return false;
  const maxAxisElems = Math.floor(maxBindBytes / 8);
  if (!Number.isInteger(maxAxisElems) || maxAxisElems < 2) return false;
  const root = Math.floor(Math.sqrt(axisLen));
  for (let d = 1; d <= root; d++) {
    if (axisLen % d !== 0) continue;
    const q = axisLen / d;
    if (d >= 2 && q >= 2 && d <= maxAxisElems && q <= maxAxisElems && factorizeSupportedRadices(d) && factorizeSupportedRadices(q)) {
      return true;
    }
    if (q >= 2 && d >= 2 && q <= maxAxisElems && d <= maxAxisElems && factorizeSupportedRadices(q) && factorizeSupportedRadices(d)) {
      return true;
    }
  }
  return false;
}

function parseOptionalAxisList(v, rank, name) {
  if (v == null) return [];
  if (!Array.isArray(v) || !v.every((x) => Number.isInteger(x) && x >= 0 && x < rank)) {
    throw new Error(`${name} must be an array of axis indices in [0, ${rank - 1}]`);
  }
  return [...new Set(v.map((x) => x | 0))];
}

export function resolveAxisKindsForShape({ shape, tuning = null, defaultRaderMaxPrime = 4096 }) {
  if (!Array.isArray(shape) || shape.length < 1 || !shape.every((x) => Number.isInteger(x) && x > 0)) {
    throw new Error(`shape must be an array of positive integers; got ${JSON.stringify(shape)}`);
  }
  if (tuning != null && typeof tuning !== "object") {
    throw new Error("tuning must be an object when provided");
  }
  const rank = shape.length;
  const raderMaxPrime = tuning?.raderMaxPrime ?? defaultRaderMaxPrime;
  if (!Number.isInteger(raderMaxPrime) || raderMaxPrime < 2) {
    throw new Error(`tuning.raderMaxPrime must be an integer >= 2; got ${raderMaxPrime}`);
  }
  const forceBluesteinAxes = parseOptionalAxisList(tuning?.forceBluesteinAxes ?? null, rank, "tuning.forceBluesteinAxes");
  const forceRaderAxes = parseOptionalAxisList(tuning?.forceRaderAxes ?? null, rank, "tuning.forceRaderAxes");
  const forceBluesteinAxisSet = new Set(forceBluesteinAxes);
  const forceRaderAxisSet = new Set(forceRaderAxes);
  for (const axis of forceBluesteinAxisSet) {
    if (forceRaderAxisSet.has(axis)) {
      throw new Error(`Axis ${axis} cannot be forced to both Bluestein and Rader`);
    }
  }
  const axisKinds = shape.map((N, axis) => {
    if (forceBluesteinAxisSet.has(axis)) return "bluestein";
    if (forceRaderAxisSet.has(axis)) {
      if (!isPrime(N)) {
        throw new Error(`tuning.forceRaderAxes contains axis ${axis}, but shape[${axis}]=${N} is not prime`);
      }
      if (N > raderMaxPrime) {
        throw new Error(`tuning.forceRaderAxes contains axis ${axis}, but shape[${axis}]=${N} exceeds tuning.raderMaxPrime=${raderMaxPrime}`);
      }
      return "rader";
    }
    if (factorizeSupportedRadices(N)) return "mixed";
    if (isPrime(N) && N <= raderMaxPrime) return "rader";
    return "bluestein";
  });
  return {
    axisKinds,
    raderMaxPrime,
    forceBluesteinAxes,
    forceRaderAxes,
    forceBluesteinAxisSet,
    forceRaderAxisSet,
  };
}

function evaluateAxisSupport({ axisKinds, axisLengths, maxBindBytes, maxBufferSize, allowNonMixedBoundedSlicing }) {
  if (!Array.isArray(axisKinds) || !Array.isArray(axisLengths) || axisKinds.length !== axisLengths.length) {
    return null;
  }
  const supported = new Array(axisKinds.length).fill(false);
  for (let axis = 0; axis < axisKinds.length; axis++) {
    const kind = axisKinds[axis];
    const n = axisLengths[axis];
    if (kind === "mixed") {
      supported[axis] = canAxisLenFitOrTwoStep(n, maxBindBytes, maxBufferSize);
      continue;
    }
    const lineBytes = n * 8;
    supported[axis] = allowNonMixedBoundedSlicing ? lineBytes <= maxBufferSize : lineBytes <= maxBindBytes;
  }
  return supported;
}

function pushReason(reasonCodes, code) {
  if (!reasonCodes.includes(code)) reasonCodes.push(code);
}

function routeModePriority(routeMode) {
  switch (routeMode) {
    case "large-out-of-core":
      return 2;
    case "large-chunk":
      return 1;
    default:
      return 0;
  }
}

function pushUnique(list, value) {
  if (!list.includes(value)) list.push(value);
}

export function mergeLargeRouteMetadata(entries) {
  const list = Array.isArray(entries) ? entries : [entries];
  const reasonCodes = [];
  const attemptedRoutes = [];
  let routeMode = "normal";
  for (const entry of list) {
    if (!entry || typeof entry !== "object") continue;
    const entryMode = typeof entry.routeMode === "string" ? entry.routeMode : null;
    if (entryMode && routeModePriority(entryMode) > routeModePriority(routeMode)) {
      routeMode = entryMode;
    }
    if (Array.isArray(entry.reasonCodes)) {
      for (const code of entry.reasonCodes) pushUnique(reasonCodes, code);
    }
    if (Array.isArray(entry.attemptedRoutes)) {
      for (const route of entry.attemptedRoutes) pushUnique(attemptedRoutes, route);
    }
  }
  if (routeMode === "large-out-of-core") {
    pushUnique(attemptedRoutes, "out-of-core-four-step");
  }
  pushUnique(reasonCodes, routeMode);
  return { routeMode, reasonCodes, attemptedRoutes };
}

export function resolveLargeRoutingPolicy({
  device,
  tuning = null,
  requiredBindingBytes = [],
  lineBytes = [],
  axisKinds = null,
  axisLengths = null,
  allowNonMixedBoundedSlicing = false,
  precision = "f32",
  requireLargePrecision = null,
  requireLargePrecisionError = null,
  allowOutOfCore = false,
  disableOutOfCore = false,
  rank = 1,
  minOutOfCoreRank = 2,
  bytesPerBatch = null,
  hasStridedIO = false,
  preferOutOfCoreForStrided = false,
  outOfCoreUnsupportedError = null,
}) {
  const maxBindBytes = resolveEffectiveMaxStorageBufferBindingSize(device, tuning);
  const maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
  const requestedLargeRoute = parseOptionalLargeRoutePreference(tuning);
  const tuningPreferOutOfCoreForStrided = parseOptionalPreferOutOfCoreForStrided(tuning);
  const effectivePreferOutOfCoreForStrided =
    tuningPreferOutOfCoreForStrided == null ? preferOutOfCoreForStrided : tuningPreferOutOfCoreForStrided;
  const reasonCodes = [];
  const attemptedRoutes = ["direct"];
  const needsLargeMode = requiredBindingBytes.some((bytes) => bytes > maxBindBytes);
  const oversizedLineMode = lineBytes.some((bytes) => bytes > maxBindBytes);
  if (!needsLargeMode) {
    pushReason(reasonCodes, "within-bindings");
  } else {
    pushReason(reasonCodes, "requires-large-bindings");
  }
  if (oversizedLineMode) pushReason(reasonCodes, "oversized-line-bindings");
  if (needsLargeMode) {
    attemptedRoutes.push("dispatch-split");
    attemptedRoutes.push("batch-chunk");
  }
  if (oversizedLineMode) attemptedRoutes.push("line-slice-or-two-step");
  const axisSupported = evaluateAxisSupport({
    axisKinds,
    axisLengths,
    maxBindBytes,
    maxBufferSize,
    allowNonMixedBoundedSlicing,
  });

  if (needsLargeMode && requireLargePrecision && precision !== requireLargePrecision) {
    pushReason(reasonCodes, "precision-restricted-large-mode");
    throw new Error(requireLargePrecisionError ?? `large-mode fallback currently supports precision:"${requireLargePrecision}" only`);
  }

  const outOfCoreEligible =
    allowOutOfCore &&
    !disableOutOfCore &&
    rank >= minOutOfCoreRank &&
    precision === "f32" &&
    (axisSupported == null || axisSupported.every((v) => v === true));
  if (axisSupported && axisSupported.some((v) => v === false)) {
    pushReason(reasonCodes, "axis-line-unsupported");
  }
  if (outOfCoreEligible) {
    pushReason(reasonCodes, "out-of-core-eligible");
  } else {
    pushReason(reasonCodes, "out-of-core-ineligible");
  }

  const requiresOutOfCore = allowOutOfCore && needsLargeMode && bytesPerBatch != null && bytesPerBatch > maxBindBytes;
  const prefersOutOfCore =
    allowOutOfCore &&
    needsLargeMode &&
    effectivePreferOutOfCoreForStrided &&
    hasStridedIO &&
    outOfCoreEligible;
  if (allowOutOfCore && needsLargeMode) attemptedRoutes.push("out-of-core-four-step");
  if (requiresOutOfCore) pushReason(reasonCodes, "bytes-per-batch-exceeds-bind");
  if (prefersOutOfCore) pushReason(reasonCodes, "strided-prefers-out-of-core");

  if (needsLargeMode && requestedLargeRoute === "chunk") {
    pushReason(reasonCodes, "forced-route-chunk");
  }
  if (needsLargeMode && requestedLargeRoute === "out-of-core") {
    pushReason(reasonCodes, "forced-route-out-of-core");
    attemptedRoutes.push("forced-out-of-core");
  }

  if (needsLargeMode && requestedLargeRoute === "chunk" && requiresOutOfCore) {
    throw new Error(
      `tuning.largeRoute="chunk" is incompatible: one batch requires ${bytesPerBatch} bytes > ` +
        `maxStorageBufferBindingSize=${maxBindBytes}, so out-of-core routing is required.`
    );
  }

  if (needsLargeMode && requestedLargeRoute === "out-of-core" && !allowOutOfCore) {
    throw new Error('tuning.largeRoute="out-of-core" requested, but out-of-core routing is not enabled for this plan');
  }

  if (needsLargeMode && requestedLargeRoute === "out-of-core" && !outOfCoreEligible) {
    throw new Error(
      `tuning.largeRoute="out-of-core" requested, but out-of-core eligibility constraints were not met ` +
        `(reasonCodes=${JSON.stringify(reasonCodes)}, axisSupported=${JSON.stringify(axisSupported)}).`
    );
  }

  if (requiresOutOfCore && !outOfCoreEligible) {
    throw new Error(
      typeof outOfCoreUnsupportedError === "function"
        ? outOfCoreUnsupportedError({
            maxBindBytes,
            maxBufferSize,
            axisSupported,
            outOfCoreEligible,
            reasonCodes: reasonCodes.slice(),
            attemptedRoutes: attemptedRoutes.slice(),
            requiresOutOfCore,
            prefersOutOfCore,
          })
        : outOfCoreUnsupportedError ??
            "large-mode requires out-of-core fallback, but out-of-core eligibility constraints were not met"
    );
  }

  const useOutOfCore =
    (needsLargeMode && requestedLargeRoute === "out-of-core") ||
    requiresOutOfCore ||
    (requestedLargeRoute !== "chunk" && prefersOutOfCore);
  const routeMode = needsLargeMode ? (useOutOfCore ? "large-out-of-core" : "large-chunk") : "normal";
  pushReason(reasonCodes, routeMode);

  return {
    maxBindBytes,
    maxBufferSize,
    needsLargeMode,
    oversizedLineMode,
    axisKinds: Array.isArray(axisKinds) ? axisKinds.slice() : null,
    axisLengths: Array.isArray(axisLengths) ? axisLengths.slice() : null,
    axisSupported,
    outOfCoreEligible,
    requiresOutOfCore,
    prefersOutOfCore,
    requestedLargeRoute,
    effectivePreferOutOfCoreForStrided,
    useOutOfCore,
    routeMode,
    reasonCodes,
    attemptedRoutes,
  };
}
