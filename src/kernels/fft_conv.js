import { COMPLEX_WGSL } from "./utils_wgsl.js";

export function generatePointwiseMulWGSL({
  totalLogical,
  batch,
  correlate,
  workgroupSize,
}) {
  const totalElems = totalLogical * batch;
  return /* wgsl */ `
${COMPLEX_WGSL}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> kernel: array<vec2<f32>>;

const TOTAL_LOGICAL: u32 = ${totalLogical}u;
const TOTAL_ELEMS: u32 = ${totalElems}u;
const CORRELATE: bool = ${correlate ? "true" : "false"};

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= TOTAL_ELEMS) { return; }
  var k: vec2<f32> = kernel[i % TOTAL_LOGICAL];
  if (CORRELATE) {
    k = vec2<f32>(k.x, -k.y);
  }
data[i] = c_mul(data[i], k);
}
`;
}

export function generatePointwiseMulSegmentWGSL({
  correlate,
  workgroupSize,
}) {
  return /* wgsl */ `
${COMPLEX_WGSL}

struct Params {
  count: u32,
  dataBase: u32,
  kernelBase: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> kernel: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const CORRELATE: bool = ${correlate ? "true" : "false"};

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.count) { return; }
  let di: u32 = params.dataBase + i;
  let ki: u32 = params.kernelBase + i;
  var k: vec2<f32> = kernel[ki];
  if (CORRELATE) {
    k = vec2<f32>(k.x, -k.y);
  }
  data[di] = c_mul(data[di], k);
}
`;
}
