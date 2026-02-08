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

function physIndexExpr({ baseOffsetElements, batchStrideElements, strides, coords }) {
  let expr = `${baseOffsetElements}u + params.extraOffsetElements + b * ${batchStrideElements}u`;
  for (let d = 0; d < strides.length; d++) {
    if (strides[d] === 1) expr += ` + ${coords[d]}`;
    else expr += ` + ${coords[d]} * ${strides[d]}u`;
  }
  return expr;
}

export function generateGatherRealStridedWGSL({
  shape,
  strides,
  baseOffsetElements,
  batchStrideElements,
  workgroupSize,
}) {
  const nTotal = shape.reduce((a, b) => a * b, 1);
  const decoded = decodeCoordsWgsl(shape, "li", "c");
  const physExpr = physIndexExpr({
    baseOffsetElements,
    batchStrideElements,
    strides,
    coords: decoded.coords,
  });
  return /* wgsl */ `
struct Params {
  totalLogical: u32,
  batch: u32,
  extraOffsetElements: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const TOTAL_LOGICAL: u32 = ${nTotal}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  let total: u32 = params.totalLogical * params.batch;
  if (i >= total) { return; }
  let b: u32 = i / params.totalLogical;
  let li: u32 = i - b * params.totalLogical;
${decoded.code}
  let pi: u32 = ${physExpr};
  dst[i] = src[pi];
}
`;
}

export function generateScatterRealStridedWGSL({
  shape,
  strides,
  baseOffsetElements,
  batchStrideElements,
  workgroupSize,
}) {
  const nTotal = shape.reduce((a, b) => a * b, 1);
  const decoded = decodeCoordsWgsl(shape, "li", "c");
  const physExpr = physIndexExpr({
    baseOffsetElements,
    batchStrideElements,
    strides,
    coords: decoded.coords,
  });
  return /* wgsl */ `
struct Params {
  totalLogical: u32,
  batch: u32,
  extraOffsetElements: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const TOTAL_LOGICAL: u32 = ${nTotal}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  let total: u32 = params.totalLogical * params.batch;
  if (i >= total) { return; }
  let b: u32 = i / params.totalLogical;
  let li: u32 = i - b * params.totalLogical;
${decoded.code}
  let pi: u32 = ${physExpr};
  dst[pi] = src[i];
}
`;
}
