import test from "node:test";
import assert from "node:assert/strict";

import { generateStockhamStageWGSL } from "../src/kernels/stockham1d.js";

function makeStockhamCode({ rank, axis, dims }) {
  return generateStockhamStageWGSL({
    rank,
    axis,
    dims,
    axisLength: dims[axis],
    strideComplex: dims.slice(0, axis).reduce((a, b) => a * b, 1),
    stageIndex: 0,
    direction: "forward",
    workgroupSize: 64,
    applyScale: false,
    scaleFactor: 1,
  });
}

test("stockham1d kernel generator supports rank-4 axis indexing", () => {
  const dims = [5, 3, 4, 2];
  const code = makeStockhamCode({ rank: 4, axis: 2, dims });

  // rank-4 axis-2 => lines_per_batch = 5*3*2 = 30
  assert.match(code, /let lines_per_batch: u32 = 30u;/);
  // Decoding should include all non-axis coordinates.
  assert.match(code, /let c0: u32 =/);
  assert.match(code, /let c1: u32 =/);
  assert.match(code, /let c3: u32 =/);
  assert.doesNotMatch(code, /Unsupported rank/);
});

test("stockham1d kernel generator keeps rank-1 path valid", () => {
  const dims = [32];
  const code = makeStockhamCode({ rank: 1, axis: 0, dims });

  assert.match(code, /let lines_per_batch: u32 = 1u;/);
  assert.match(code, /axis-only line \(rank=1\)/);
});

test("stockham1d kernel generator validates rank/dims consistency", () => {
  assert.throws(
    () =>
      generateStockhamStageWGSL({
        rank: 4,
        axis: 0,
        dims: [8, 8, 8],
        axisLength: 8,
        strideComplex: 1,
        stageIndex: 0,
        direction: "forward",
        workgroupSize: 64,
        applyScale: false,
        scaleFactor: 1,
      }),
    /dims length .* must match rank/i
  );
});
