// Copyright (c) 2026 Maksim Eremenko

function isPositiveInt(x) {
  return Number.isInteger(x) && x > 0;
}

export function normalizeIoView(rank, logicalShape, ioView = {}) {
  const normOne = (v, kind) => {
    if (!v) return null;
    const shape = v.shape ?? null;
    if (!Array.isArray(shape) || shape.length !== rank || !shape.every(isPositiveInt)) {
      throw new Error(`ioView.${kind}.shape must be an array of ${rank} positive ints`);
    }
    const placement = v.placement ?? "start";
    if (placement !== "start" && placement !== "center") {
      throw new Error(`ioView.${kind}.placement must be "start"|"center"`);
    }
    let offset = v.offset ?? null;
    if (offset != null) {
      if (!Array.isArray(offset) || offset.length !== rank || !offset.every((x) => Number.isInteger(x))) {
        throw new Error(`ioView.${kind}.offset must be an array of ${rank} integers`);
      }
    } else if (placement === "center") {
      offset = shape.map((s, d) => Math.floor((logicalShape[d] - s) / 2));
    } else {
      offset = new Array(rank).fill(0);
    }
    const clearOutside = kind === "output" ? !!v.clearOutside : false;
    return { shape, placement, offset, clearOutside };
  };

  return {
    input: normOne(ioView.input, "input"),
    output: normOne(ioView.output, "output"),
  };
}

