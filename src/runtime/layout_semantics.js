// Copyright (c) 2026 Maksim Eremenko

import { prod } from "./common.js";
import { contiguousStrides as tensorContiguousStrides, spanElements as tensorSpanElements } from "./tensor_descriptor.js";

function hasOwn(obj, key) {
  return !!obj && Object.prototype.hasOwnProperty.call(obj, key);
}

function arraysEqual(a, b) {
  if (a === b) return true;
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function parseOptionalPositiveIntArray(v, rank, name) {
  if (v == null) return null;
  if (!Array.isArray(v) || v.length !== rank || !v.every((x) => Number.isInteger(x) && x > 0)) {
    throw new Error(`${name} must be an array of ${rank} positive integers`);
  }
  return v.map((x) => x | 0);
}

export function parseOptionalNonNegativeInt(v, name) {
  if (v == null) return null;
  if (!Number.isInteger(v) || v < 0) throw new Error(`${name} must be a non-negative integer`);
  return v | 0;
}

function parseOptionalPositiveInt(v, name) {
  if (v == null) return null;
  if (!Number.isInteger(v) || v <= 0) throw new Error(`${name} must be a positive integer`);
  return v | 0;
}

function checkedSafeInt(v, name) {
  if (!Number.isSafeInteger(v) || v < 0) {
    throw new Error(`${name} must stay within non-negative safe integer range`);
  }
  return v;
}

export function contiguousStrides(shape) {
  return tensorContiguousStrides(shape);
}

export function stridedSpanElements(shape, strides) {
  return tensorSpanElements(shape, strides);
}

function sideFieldName(side, suffix) {
  if (side === "input") return `input${suffix}`;
  return `output${suffix}`;
}

function hasExplicitSideLayout(layout, side) {
  return (
    hasOwn(layout, sideFieldName(side, "Strides")) ||
    hasOwn(layout, sideFieldName(side, "OffsetElements")) ||
    hasOwn(layout, sideFieldName(side, "BatchStrideElements")) ||
    hasOwn(layout, "strides") ||
    hasOwn(layout, "offsetElements") ||
    hasOwn(layout, "batchStrideElements")
  );
}

function resolveExplicitSideLayout(layout, side, rank, layoutShape) {
  const stridesName = side === "input" ? "layout.inputStrides/layout.strides" : "layout.outputStrides/layout.strides";
  const offsetName = side === "input" ? "layout.inputOffsetElements/layout.offsetElements" : "layout.outputOffsetElements/layout.offsetElements";
  const batchName = side === "input" ? "layout.inputBatchStrideElements/layout.batchStrideElements" : "layout.outputBatchStrideElements/layout.batchStrideElements";
  const stridesValue = layout?.[sideFieldName(side, "Strides")] ?? layout?.strides ?? null;
  const offsetValue = layout?.[sideFieldName(side, "OffsetElements")] ?? layout?.offsetElements ?? null;
  const batchStrideValue = layout?.[sideFieldName(side, "BatchStrideElements")] ?? layout?.batchStrideElements ?? null;

  const strides = parseOptionalPositiveIntArray(stridesValue, rank, stridesName);
  const offsetElements = parseOptionalNonNegativeInt(offsetValue, offsetName) ?? 0;
  const spanElements = strides ? stridedSpanElements(layoutShape, strides) : 0;
  const defaultBatchStrideElements = strides ? spanElements : prod(layoutShape);
  const batchStrideElements = parseOptionalNonNegativeInt(batchStrideValue, batchName) ?? defaultBatchStrideElements;
  if (strides && batchStrideElements < spanElements) {
    throw new Error(side === "input" ? "layout.inputBatchStrideElements is too small for layout.inputStrides" : "layout.outputBatchStrideElements is too small for layout.outputStrides");
  }
  // Keep contiguous-equivalent explicit descriptors on the direct contiguous route.
  if (strides) {
    const contiguous = contiguousStrides(layoutShape);
    const contiguousBatch = prod(layoutShape);
    const noOp = arraysEqual(strides, contiguous) && offsetElements === 0 && batchStrideElements === contiguousBatch;
    if (noOp) {
      return {
        strides: null,
        offsetElements: 0,
        batchStrideElements: contiguousBatch,
        spanElements: 0,
      };
    }
  }
  return {
    strides,
    offsetElements,
    batchStrideElements,
    spanElements,
  };
}

function mergeWhdcnDescriptor(globalDesc, sideDesc, sideName) {
  const g = globalDesc ?? {};
  const s = sideDesc ?? {};
  if (typeof g !== "object" || Array.isArray(g)) {
    throw new Error("layout.whdcn must be an object");
  }
  if (sideDesc != null && (typeof s !== "object" || Array.isArray(s))) {
    throw new Error(`layout.whdcn.${sideName} must be an object`);
  }
  return { ...g, ...s };
}

function resolveWhdcnSideLayout(desc, side, rank, layoutShape) {
  if (!desc) return null;
  if (hasOwn(desc, "enabled") && typeof desc.enabled !== "boolean") {
    throw new Error(`layout.whdcn.${side}.enabled must be boolean when provided`);
  }
  if (desc.enabled === false) return null;

  const hasAnyControls =
    hasOwn(desc, "strides") ||
    hasOwn(desc, "offsetElements") ||
    hasOwn(desc, "batchStrideElements") ||
    hasOwn(desc, "channels") ||
    hasOwn(desc, "channelIndex") ||
    hasOwn(desc, "channelStrideElements");
  if (!hasAnyControls) return null;

  const sidePath = `layout.whdcn.${side}`;
  const strides = parseOptionalPositiveIntArray(desc.strides ?? null, rank, `${sidePath}.strides`) ?? contiguousStrides(layoutShape);
  const spanElements = stridedSpanElements(layoutShape, strides);
  const channels = parseOptionalPositiveInt(desc.channels ?? null, `${sidePath}.channels`) ?? 1;
  const channelIndex = parseOptionalNonNegativeInt(desc.channelIndex ?? null, `${sidePath}.channelIndex`) ?? 0;
  if (channelIndex >= channels) {
    throw new Error(`${sidePath}.channelIndex (${channelIndex}) must be < ${sidePath}.channels (${channels})`);
  }
  const channelStrideElements = parseOptionalPositiveInt(desc.channelStrideElements ?? null, `${sidePath}.channelStrideElements`) ?? spanElements;
  if (channelStrideElements < spanElements) {
    throw new Error(`${sidePath}.channelStrideElements must be >= addressed span (${spanElements})`);
  }

  const baseOffset = parseOptionalNonNegativeInt(desc.offsetElements ?? null, `${sidePath}.offsetElements`) ?? 0;
  const offsetElements = checkedSafeInt(baseOffset + channelIndex * channelStrideElements, `${sidePath}.offsetElements`);

  const defaultBatchStride = checkedSafeInt(channelStrideElements * channels, `${sidePath}.batchStrideElements`);
  const batchStrideElements =
    parseOptionalNonNegativeInt(desc.batchStrideElements ?? null, `${sidePath}.batchStrideElements`) ?? defaultBatchStride;
  if (batchStrideElements < defaultBatchStride) {
    throw new Error(`${sidePath}.batchStrideElements must be >= channels*channelStrideElements (${defaultBatchStride})`);
  }

  const contiguous = contiguousStrides(layoutShape);
  const contiguousBatch = prod(layoutShape);
  const noOp =
    arraysEqual(strides, contiguous) &&
    offsetElements === 0 &&
    batchStrideElements === contiguousBatch &&
    channels === 1 &&
    channelIndex === 0 &&
    channelStrideElements === spanElements;
  if (noOp) return null;

  return {
    strides,
    offsetElements,
    batchStrideElements,
    spanElements,
  };
}

export function resolveLayoutSemantics({ layout, rank, inputShape, outputShape }) {
  const l = layout ?? {};
  if (typeof l !== "object" || Array.isArray(l)) {
    throw new Error("layout must be an object");
  }

  const inputExplicit = resolveExplicitSideLayout(l, "input", rank, inputShape);
  const outputExplicit = resolveExplicitSideLayout(l, "output", rank, outputShape);

  let inputResolved = inputExplicit;
  let outputResolved = outputExplicit;
  let usesWhdcnInput = false;
  let usesWhdcnOutput = false;

  if (l.whdcn != null) {
    if (typeof l.whdcn !== "object" || Array.isArray(l.whdcn)) {
      throw new Error("layout.whdcn must be an object");
    }
    const whdcnGlobal = { ...l.whdcn };
    delete whdcnGlobal.input;
    delete whdcnGlobal.output;

    if (!hasExplicitSideLayout(l, "input")) {
      const desc = mergeWhdcnDescriptor(whdcnGlobal, l.whdcn.input ?? null, "input");
      const resolved = resolveWhdcnSideLayout(desc, "input", rank, inputShape);
      if (resolved) {
        inputResolved = resolved;
        usesWhdcnInput = true;
      }
    }
    if (!hasExplicitSideLayout(l, "output")) {
      const desc = mergeWhdcnDescriptor(whdcnGlobal, l.whdcn.output ?? null, "output");
      const resolved = resolveWhdcnSideLayout(desc, "output", rank, outputShape);
      if (resolved) {
        outputResolved = resolved;
        usesWhdcnOutput = true;
      }
    }
  }

  return {
    inputStrides: inputResolved.strides,
    outputStrides: outputResolved.strides,
    inputOffsetElements: inputResolved.offsetElements,
    outputOffsetElements: outputResolved.offsetElements,
    inputBatchStrideElements: inputResolved.batchStrideElements,
    outputBatchStrideElements: outputResolved.batchStrideElements,
    inputSpanElements: inputResolved.spanElements,
    outputSpanElements: outputResolved.spanElements,
    usesStridedInput: !!inputResolved.strides,
    usesStridedOutput: !!outputResolved.strides,
    usesWhdcnInput,
    usesWhdcnOutput,
  };
}
