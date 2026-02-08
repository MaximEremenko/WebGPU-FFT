function decodeCoordsWgsl(dims, indexName = "li", coordPrefix = "c") {
  let code = `  var rem: u32 = ${indexName};\n`;
  const coords = [];
  for (let d = 0; d < dims.length; d++) {
    const c = `${coordPrefix}${d}`;
    coords.push(c);
    code += `  let ${c}: u32 = rem % ${dims[d]}u;\n`;
    if (d < dims.length - 1) code += `  rem = rem / ${dims[d]}u;\n`;
  }
  return { code, coords };
}

function insideRangeExpr(start, end, coords) {
  const terms = [];
  for (let d = 0; d < coords.length; d++) {
    terms.push(`${coords[d]} >= ${start[d]}u && ${coords[d]} < ${end[d]}u`);
  }
  return terms.join(" && ");
}

export function generateZeroOutsideRangeComplexWGSL({
  shape,
  start,
  end,
  batch,
  workgroupSize,
}) {
  const nTotal = shape.reduce((a, b) => a * b, 1);
  const totalElems = nTotal * batch;
  const decoded = decodeCoordsWgsl(shape, "li", "c");
  const insideExpr = insideRangeExpr(start, end, decoded.coords);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;

const TOTAL_LOGICAL: u32 = ${nTotal}u;
const TOTAL_ELEMS: u32 = ${totalElems}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= TOTAL_ELEMS) { return; }
  let li: u32 = i % TOTAL_LOGICAL;
${decoded.code}
  if (!(${insideExpr})) {
    data[i] = vec2<f32>(0.0, 0.0);
  }
}
`;
}

export function generateZeroOutsideRangeRealWGSL({
  shape,
  start,
  end,
  batch,
  workgroupSize,
}) {
  const nTotal = shape.reduce((a, b) => a * b, 1);
  const totalElems = nTotal * batch;
  const decoded = decodeCoordsWgsl(shape, "li", "c");
  const insideExpr = insideRangeExpr(start, end, decoded.coords);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

const TOTAL_LOGICAL: u32 = ${nTotal}u;
const TOTAL_ELEMS: u32 = ${totalElems}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= TOTAL_ELEMS) { return; }
  let li: u32 = i % TOTAL_LOGICAL;
${decoded.code}
  if (!(${insideExpr})) {
    data[i] = 0.0;
  }
}
`;
}

