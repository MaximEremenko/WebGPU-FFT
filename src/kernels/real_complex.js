export function generateRealToComplexWGSL({ totalReal, workgroupSize }) {
  return /* wgsl */ `
struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  output[i] = vec2<f32>(input[i], 0.0);
}
`;
}

export function generateComplexToRealWGSL({ workgroupSize }) {
  return /* wgsl */ `
struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  output[i] = input[i].x;
}
`;
}

function prod(arr) {
  return arr.reduce((a, b) => a * b, 1);
}

function rowMajorStrides(dims) {
  const out = new Array(dims.length);
  out[0] = 1;
  for (let i = 1; i < dims.length; i++) out[i] = out[i - 1] * dims[i - 1];
  return out;
}

function decodeCoordsWGSL({ indexName, dims, coordPrefix }) {
  let rem = indexName;
  const coords = [];
  let code = "";
  for (let d = 0; d < dims.length; d++) {
    const c = `${coordPrefix}${d}`;
    coords.push(c);
    code += `  let ${c}: u32 = ${rem} % ${dims[d]}u;\n`;
    if (d < dims.length - 1) {
      const nextRem = `${coordPrefix}rem${d}`;
      code += `  let ${nextRem}: u32 = ${rem} / ${dims[d]}u;\n`;
      rem = nextRem;
    }
  }
  return { code, coords };
}

export function generatePackR2CWGSL({ shape, workgroupSize }) {
  const rank = shape.length;
  const inDims = shape.slice();
  const outDims = [((shape[0] >>> 1) + 1), ...shape.slice(1)];
  const inStrides = rowMajorStrides(inDims);
  const inTotal = prod(inDims);
  const outTotal = prod(outDims);
  const decoded = decodeCoordsWGSL({ indexName: "rem", dims: outDims, coordPrefix: "c" });

  let inIndexBody = `  var inIndex: u32 = b * ${inTotal}u;\n`;
  for (let d = 0; d < rank; d++) {
    if (inStrides[d] === 1) inIndexBody += `  inIndex = inIndex + ${decoded.coords[d]};\n`;
    else inIndexBody += `  inIndex = inIndex + ${decoded.coords[d]} * ${inStrides[d]}u;\n`;
  }

  return /* wgsl */ `
struct Params {
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const OUT_TOTAL_PER_BATCH: u32 = ${outTotal}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  let totalOut: u32 = OUT_TOTAL_PER_BATCH * params.batch;
  if (i >= totalOut) { return; }
  let b: u32 = i / OUT_TOTAL_PER_BATCH;
  let rem: u32 = i - b * OUT_TOTAL_PER_BATCH;
${decoded.code}
${inIndexBody}
  output[i] = input[inIndex];
}
`;
}

export function generateUnpackC2RWGSL({ shape, workgroupSize }) {
  const rank = shape.length;
  const fullDims = shape.slice();
  const inDims = [((shape[0] >>> 1) + 1), ...shape.slice(1)];
  const inStrides = rowMajorStrides(inDims);
  const fullTotal = prod(fullDims);
  const inTotal = prod(inDims);
  const Nx = shape[0];
  const inNx = inDims[0];
  const even = Nx % 2 === 0;
  const decoded = decodeCoordsWGSL({ indexName: "rem", dims: fullDims, coordPrefix: "c" });

  let mirrorCoordsCode = "";
  const coordForInIndex = new Array(rank);
  coordForInIndex[0] = "xPacked";
  for (let d = 1; d < rank; d++) {
    const cd = decoded.coords[d];
    const cm = `c${d}m`;
    const cp = `c${d}p`;
    mirrorCoordsCode += `  let ${cm}: u32 = select(0u, ${fullDims[d]}u - ${cd}, ${cd} != 0u);\n`;
    mirrorCoordsCode += `  let ${cp}: u32 = select(${cd}, ${cm}, x >= IN_NX);\n`;
    coordForInIndex[d] = cp;
  }

  let inIndexBody = `  var inIndex: u32 = b * ${inTotal}u;\n`;
  for (let d = 0; d < rank; d++) {
    const coord = coordForInIndex[d];
    if (inStrides[d] === 1) inIndexBody += `  inIndex = inIndex + ${coord};\n`;
    else inIndexBody += `  inIndex = inIndex + ${coord} * ${inStrides[d]}u;\n`;
  }

  let selfConjExpr = "(x == 0u || (EVEN_NX && x == (NX / 2u)))";
  for (let d = 1; d < rank; d++) {
    const cd = decoded.coords[d];
    if (fullDims[d] % 2 === 0) {
      selfConjExpr += ` && (${cd} == 0u || ${cd} == ${fullDims[d] / 2}u)`;
    } else {
      selfConjExpr += ` && (${cd} == 0u)`;
    }
  }

  return /* wgsl */ `
struct Params {
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const NX: u32 = ${Nx}u;
const IN_NX: u32 = ${inNx}u;
const EVEN_NX: bool = ${even ? "true" : "false"};
const OUT_TOTAL_PER_BATCH: u32 = ${fullTotal}u;

fn conj(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(v.x, -v.y); }

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  let totalOut: u32 = OUT_TOTAL_PER_BATCH * params.batch;
  if (i >= totalOut) { return; }
  let b: u32 = i / OUT_TOTAL_PER_BATCH;
  let rem: u32 = i - b * OUT_TOTAL_PER_BATCH;
${decoded.code}

  // Map full spectrum X[x] from packed [0..N/2].
  var v: vec2<f32>;
  let x: u32 = ${decoded.coords[0]};
  let xPacked: u32 = select(x, NX - x, x >= IN_NX);
${mirrorCoordsCode}
${inIndexBody}
  v = input[inIndex];
  if (x >= IN_NX) { v = conj(v); }

  // Only globally self-conjugate bins are guaranteed real.
  if (${selfConjExpr}) {
    v = vec2<f32>(v.x, 0.0);
  }
  output[i] = v;
}
`;
}
