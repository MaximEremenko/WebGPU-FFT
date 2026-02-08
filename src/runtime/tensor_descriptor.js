// Copyright (c) 2026 Maksim Eremenko

import { prod } from "./common.js";

function assertPositiveIntArray(shape, name) {
  if (!Array.isArray(shape) || shape.length < 1 || !shape.every((x) => Number.isInteger(x) && x > 0)) {
    throw new Error(`${name} must be an array of positive integers`);
  }
}

export function contiguousStrides(shape) {
  assertPositiveIntArray(shape, "shape");
  const out = new Array(shape.length);
  let stride = 1;
  for (let d = 0; d < shape.length; d++) {
    out[d] = stride;
    stride *= shape[d];
  }
  return out;
}

export function spanElements(shape, strides) {
  assertPositiveIntArray(shape, "shape");
  if (!Array.isArray(strides) || strides.length !== shape.length || !strides.every((x) => Number.isInteger(x) && x > 0)) {
    throw new Error("strides must be an array of positive integers matching shape rank");
  }
  let span = 1;
  for (let d = 0; d < shape.length; d++) span += (shape[d] - 1) * strides[d];
  return span;
}

export function coordsFromLinear(i, shape, outCoords) {
  assertPositiveIntArray(shape, "shape");
  if (!Array.isArray(outCoords) || outCoords.length !== shape.length) {
    throw new Error("outCoords must be an array matching shape rank");
  }
  let rem = i;
  for (let d = 0; d < shape.length; d++) {
    const dim = shape[d];
    const c = rem % dim;
    outCoords[d] = c;
    rem = (rem - c) / dim;
  }
}

export function linearFromCoords(coords, strides) {
  if (!Array.isArray(coords) || !Array.isArray(strides) || coords.length !== strides.length) {
    throw new Error("coords and strides must be arrays with the same length");
  }
  let idx = 0;
  for (let d = 0; d < coords.length; d++) idx += coords[d] * strides[d];
  return idx;
}

export function linearFromCoordsShape(coords, shape) {
  return linearFromCoords(coords, contiguousStrides(shape));
}

export function createTensorDescriptor({
  shape,
  strides = null,
  offsetElements = 0,
  batchStrideElements = null,
  name = "tensor",
}) {
  assertPositiveIntArray(shape, `${name}.shape`);
  if (strides != null && (!Array.isArray(strides) || strides.length !== shape.length || !strides.every((x) => Number.isInteger(x) && x > 0))) {
    throw new Error(`${name}.strides must be null or an array of ${shape.length} positive integers`);
  }
  if (!Number.isInteger(offsetElements) || offsetElements < 0) {
    throw new Error(`${name}.offsetElements must be a non-negative integer`);
  }
  const resolvedStrides = strides ? strides.slice() : contiguousStrides(shape);
  const resolvedSpan = spanElements(shape, resolvedStrides);
  const defaultBatchStride = resolvedSpan;
  const resolvedBatchStride = batchStrideElements == null ? defaultBatchStride : batchStrideElements;
  if (!Number.isInteger(resolvedBatchStride) || resolvedBatchStride < resolvedSpan) {
    throw new Error(`${name}.batchStrideElements must be an integer >= ${resolvedSpan}`);
  }
  return {
    name,
    shape: shape.slice(),
    strides: resolvedStrides,
    spanElements: resolvedSpan,
    offsetElements: offsetElements | 0,
    batchStrideElements: resolvedBatchStride | 0,
    logicalElementsPerBatch: prod(shape),
    usesCustomStrides: !!strides,
    isContiguous:
      !strides &&
      offsetElements === 0 &&
      resolvedBatchStride === prod(shape),
  };
}

export function requiredElementsForBatchRange(desc, { runtimeExtraElements = 0, batchStart = 0, batchCount = 1 } = {}) {
  if (!Number.isInteger(runtimeExtraElements) || runtimeExtraElements < 0) {
    throw new Error("runtimeExtraElements must be a non-negative integer");
  }
  if (!Number.isInteger(batchStart) || batchStart < 0) {
    throw new Error("batchStart must be a non-negative integer");
  }
  if (!Number.isInteger(batchCount) || batchCount < 0) {
    throw new Error("batchCount must be a non-negative integer");
  }
  if (!desc || !Number.isInteger(desc.offsetElements) || !Number.isInteger(desc.batchStrideElements) || !Number.isInteger(desc.spanElements)) {
    throw new Error("desc must be a tensor descriptor from createTensorDescriptor");
  }
  const lastBatch = batchStart + Math.max(0, batchCount - 1);
  return desc.offsetElements + runtimeExtraElements + lastBatch * desc.batchStrideElements + desc.spanElements;
}

export function requiredBytesForBatchRange(
  desc,
  { bytesPerElement, runtimeExtraElements = 0, batchStart = 0, batchCount = 1 } = {}
) {
  if (!Number.isInteger(bytesPerElement) || bytesPerElement <= 0) {
    throw new Error("bytesPerElement must be a positive integer");
  }
  return requiredElementsForBatchRange(desc, { runtimeExtraElements, batchStart, batchCount }) * bytesPerElement;
}
