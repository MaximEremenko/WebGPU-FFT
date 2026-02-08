// Copyright (c) 2026 Maksim Eremenko

function parseOptionalBoundArray(v, rank, name, defaults) {
  if (v == null) return defaults.slice();
  if (!Array.isArray(v) || v.length !== rank || !v.every((x) => Number.isInteger(x))) {
    throw new Error(`${name} must be an array of ${rank} integers`);
  }
  return v.map((x) => x | 0);
}

function normalizeStage(rank, shape, stage, name) {
  if (!stage) return null;
  if (typeof stage !== "object") {
    throw new Error(`${name} must be an object with optional start/end arrays`);
  }
  const src = stage.range && typeof stage.range === "object" ? stage.range : stage;
  const start = parseOptionalBoundArray(src.start ?? null, rank, `${name}.start`, new Array(rank).fill(0));
  const end = parseOptionalBoundArray(src.end ?? null, rank, `${name}.end`, shape);

  for (let d = 0; d < rank; d++) {
    if (start[d] < 0) throw new Error(`${name}.start[${d}] must be >= 0; got ${start[d]}`);
    if (end[d] < 0) throw new Error(`${name}.end[${d}] must be >= 0; got ${end[d]}`);
    if (start[d] > end[d]) throw new Error(`${name}: start[${d}] must be <= end[${d}]`);
    if (end[d] > shape[d]) throw new Error(`${name}.end[${d}] must be <= shape[${d}] (${shape[d]}); got ${end[d]}`);
  }

  const full = start.every((s) => s === 0) && end.every((e, d) => e === shape[d]);
  return full ? null : { start, end };
}

export function normalizeZeroPad(rank, shape, zeroPad = null, name = "zeroPad") {
  if (!zeroPad) return { read: null, write: null };
  if (typeof zeroPad !== "object") {
    throw new Error(`${name} must be an object with optional read/write stage configs`);
  }
  return {
    read: normalizeStage(rank, shape, zeroPad.read ?? null, `${name}.read`),
    write: normalizeStage(rank, shape, zeroPad.write ?? null, `${name}.write`),
  };
}

