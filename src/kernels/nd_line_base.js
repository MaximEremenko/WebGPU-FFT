export function wgslLineBaseFn({ rank, axis, dims }) {
  if (!Number.isInteger(rank) || rank < 1) throw new Error(`rank must be >= 1, got ${rank}`);
  if (!Array.isArray(dims) || dims.length !== rank) throw new Error(`dims length (${dims?.length}) must match rank (${rank})`);
  if (!Number.isInteger(axis) || axis < 0 || axis >= rank) throw new Error(`axis=${axis} out of range for rank=${rank}`);

  const nTotal = dims.reduce((a, b) => a * b, 1);
  const linesPerBatch = dims.reduce((a, d, i) => (i === axis ? a : a * d), 1);

  const strides = new Array(rank);
  strides[0] = 1;
  for (let i = 1; i < rank; i++) strides[i] = strides[i - 1] * dims[i - 1];

  let decode = "";
  let remName = "rem";
  let remInit = true;
  for (let d = 0; d < rank; d++) {
    if (d === axis) continue;
    const dim = dims[d];
    const stride = strides[d];
    const coordName = `c${d}`;
    const nextRem = `${remName}_${d}`;
    if (remInit) {
      decode += `  var ${remName}: u32 = line - b * lines_per_batch;\n`;
      remInit = false;
    }
    decode += `  let ${coordName}: u32 = ${remName} % ${dim}u;\n`;
    decode += `  base = base + ${coordName} * ${stride}u;\n`;
    decode += `  var ${nextRem}: u32 = ${remName} / ${dim}u;\n`;
    remName = nextRem;
  }

  if (decode.length === 0) {
    decode = "  // axis-only line (rank=1): no non-axis coordinates\n";
  }

  return /* wgsl */ `
fn line_base(line: u32) -> u32 {
  let lines_per_batch: u32 = ${linesPerBatch}u;
  let b: u32 = line / lines_per_batch;
  var base: u32 = b * ${nTotal}u;
${decode}  return base;
}
`;
}
