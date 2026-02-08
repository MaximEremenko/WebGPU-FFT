// Copyright (c) 2026 Maksim Eremenko

const FFTCONV_MODES = new Set(["convolution", "correlation"]);
const FFTCONV_BOUNDARIES = new Set(["circular", "linear-full", "linear-same", "linear-valid"]);
const FFTCONV_OUTPUT_LAYOUTS = new Set(["kernel-major", "batch-major"]);

const CONFLICTING_LAYOUT_KEYS = new Set([
  "whdcn",
  "strides",
  "inputStrides",
  "outputStrides",
  "offsetElements",
  "inputOffsetElements",
  "outputOffsetElements",
  "batchStrideElements",
  "inputBatchStrideElements",
  "outputBatchStrideElements",
]);

function hasOwn(obj, key) {
  return Object.prototype.hasOwnProperty.call(obj, key);
}

function assertPlainObject(value, name) {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
}

function assertPositiveSafeInt(value, name) {
  if (!Number.isInteger(value) || value <= 0 || !Number.isSafeInteger(value)) {
    throw new Error(`${name} must be a positive safe integer`);
  }
}

function assertNonNegativeSafeInt(value, name) {
  if (!Number.isInteger(value) || value < 0 || !Number.isSafeInteger(value)) {
    throw new Error(`${name} must be a non-negative safe integer`);
  }
}

function assertEnum(value, allowed, name) {
  if (!allowed.has(value)) {
    throw new Error(`${name} must be one of ${Array.from(allowed).map((v) => JSON.stringify(v)).join(", ")}; got ${value}`);
  }
}

function checkedMul(a, b, name) {
  const v = a * b;
  if (!Number.isSafeInteger(v) || v < 0) {
    throw new Error(`${name} exceeds safe integer range`);
  }
  return v;
}

function resolveLogicalSpan(shape) {
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new Error("shape must be a non-empty array");
  }
  let span = 1;
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];
    assertPositiveSafeInt(dim, `shape[${i}]`);
    span = checkedMul(span, dim, "shape product");
  }
  return span;
}

function normalizeSideDescriptor(side, sideName, logicalSpan, { kernelCount, allowKernelStep }) {
  assertPlainObject(side, sideName);
  assertPositiveSafeInt(side.channels, `${sideName}.channels`);
  const channels = side.channels;

  const channelIndex = side.channelIndex ?? 0;
  assertNonNegativeSafeInt(channelIndex, `${sideName}.channelIndex`);
  if (channelIndex >= channels) {
    throw new Error(`${sideName}.channelIndex (${channelIndex}) must be < ${sideName}.channels (${channels})`);
  }

  const channelStrideElements = side.channelStrideElements ?? logicalSpan;
  assertPositiveSafeInt(channelStrideElements, `${sideName}.channelStrideElements`);
  if (channelStrideElements < logicalSpan) {
    throw new Error(`${sideName}.channelStrideElements must be >= logical span (${logicalSpan})`);
  }

  const defaultBatchStrideElements = checkedMul(channels, channelStrideElements, `${sideName}.batchStrideElements`);
  const batchStrideElements = side.batchStrideElements ?? defaultBatchStrideElements;
  assertPositiveSafeInt(batchStrideElements, `${sideName}.batchStrideElements`);
  if (batchStrideElements < defaultBatchStrideElements) {
    throw new Error(`${sideName}.batchStrideElements must be >= channels*channelStrideElements (${defaultBatchStrideElements})`);
  }

  const offsetElements = side.offsetElements ?? 0;
  assertNonNegativeSafeInt(offsetElements, `${sideName}.offsetElements`);

  let kernelStepChannels = 1;
  if (allowKernelStep) {
    kernelStepChannels = side.kernelStepChannels ?? 1;
    assertPositiveSafeInt(kernelStepChannels, `${sideName}.kernelStepChannels`);
    if (kernelCount > 1) {
      const maxChannelIndex = channelIndex + (kernelCount - 1) * kernelStepChannels;
      if (!Number.isSafeInteger(maxChannelIndex)) {
        throw new Error(`${sideName}.kernelStepChannels mapping exceeds safe integer range`);
      }
      if (maxChannelIndex >= channels) {
        throw new Error(
          `${sideName} does not fit kernelCount=${kernelCount}: max channel index ${maxChannelIndex} exceeds channels=${channels} ` +
            `(channelIndex=${channelIndex}, kernelStepChannels=${kernelStepChannels})`
        );
      }
    }
  }

  const desc = {
    channels,
    channelIndex,
    channelStrideElements,
    batchStrideElements,
    offsetElements,
  };
  if (allowKernelStep) {
    desc.kernelStepChannels = kernelStepChannels;
  }
  return desc;
}

function validateLayout(layout) {
  assertPlainObject(layout, "layout");
  if (hasOwn(layout, "interleavedComplex") && layout.interleavedComplex !== true) {
    throw new Error("layout.interleavedComplex must be true for fftconv channel-lane presets");
  }
  for (const key of CONFLICTING_LAYOUT_KEYS) {
    if (hasOwn(layout, key)) {
      throw new Error(`layout.${key} cannot be combined with fftConv.channelPolicy presets`);
    }
  }
}

function buildPreset(opts, forcedOutputLayout = null) {
  assertPlainObject(opts, "opts");

  const {
    shape,
    batch,
    kernelCount = 1,
    mode = "convolution",
    boundary = "circular",
    outputLayout = "kernel-major",
    input,
    output,
    layout = {},
  } = opts;

  const logicalSpan = resolveLogicalSpan(shape);
  assertPositiveSafeInt(batch, "batch");
  assertPositiveSafeInt(kernelCount, "kernelCount");
  assertEnum(mode, FFTCONV_MODES, "mode");
  assertEnum(boundary, FFTCONV_BOUNDARIES, "boundary");
  assertEnum(outputLayout, FFTCONV_OUTPUT_LAYOUTS, "outputLayout");
  validateLayout(layout);

  const finalOutputLayout = forcedOutputLayout ?? outputLayout;
  if (forcedOutputLayout != null) {
    assertEnum(forcedOutputLayout, FFTCONV_OUTPUT_LAYOUTS, "forcedOutputLayout");
  }

  const inputDesc = normalizeSideDescriptor(input, "input", logicalSpan, {
    kernelCount,
    allowKernelStep: false,
  });
  const outputDesc = normalizeSideDescriptor(output, "output", logicalSpan, {
    kernelCount,
    allowKernelStep: true,
  });

  return {
    shape: [...shape],
    batch,
    layout: {
      interleavedComplex: true,
      ...layout,
    },
    fftConv: {
      mode,
      boundary,
      kernelCount,
      outputLayout: finalOutputLayout,
      channelPolicy: {
        input: inputDesc,
        output: outputDesc,
      },
    },
  };
}

export function createFftConvChannelLanePreset(opts) {
  return buildPreset(opts, null);
}

export function createFftConvKernelMajorChannelLanePreset(opts) {
  return buildPreset(opts, "kernel-major");
}

export function createFftConvBatchMajorChannelLanePreset(opts) {
  return buildPreset(opts, "batch-major");
}

