import { COMPLEX_WGSL } from "./utils_wgsl.js";
import { wgslLineBaseFn } from "./nd_line_base.js";

export function generateRaderSumWGSL({ rank, axis, dims, axisLength, strideComplex, workgroupSize }) {
  const N = axisLength;
  const lineBase = wgslLineBaseFn({ rank, axis, dims });
  return /* wgsl */ `
struct Params {
  lines: u32,
  lineOffset: u32,
  _pad1: u32,
  _pad2: u32,
}

  @group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> sumAll: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> x0: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

${COMPLEX_WGSL}
const N: u32 = ${N}u;
const STRIDE: u32 = ${strideComplex}u;

${lineBase}

var<workgroup> scratch: array<vec2<f32>, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lineLocal: u32 = wid.x;
  if (lineLocal >= params.lines) { return; }
  let line: u32 = params.lineOffset + lineLocal;
  let base: u32 = line_base(line);

  // parallel reduction across N items
  var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  var i: u32 = lid.x;
  while (i < N) {
    acc = c_add(acc, input[base + i * STRIDE]);
    i = i + ${workgroupSize}u;
  }
  scratch[lid.x] = acc;
  workgroupBarrier();

  var stride: u32 = ${workgroupSize}u / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      scratch[lid.x] = c_add(scratch[lid.x], scratch[lid.x + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    sumAll[lineLocal] = scratch[0];
    x0[lineLocal] = input[base + 0u];
  }
}
`;
}

export function generateRaderPackARevWGSL({
  rank,
  axis,
  dims,
  axisLength,
  mLength,
  strideComplex,
  workgroupSize,
}) {
  const N = axisLength;
  const M = mLength;
  const lineBase = wgslLineBaseFn({ rank, axis, dims });
  return /* wgsl */ `
struct Params {
  lines: u32,
  lineOffset: u32,
  _pad1: u32,
  _pad2: u32,
}

  @group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> a: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

${COMPLEX_WGSL}
const N: u32 = ${N}u;
const M: u32 = ${M}u;
const STRIDE: u32 = ${strideComplex}u;

${lineBase}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  let total: u32 = params.lines * M;
  if (i >= total) { return; }
  let lineLocal: u32 = i / M;
  let line: u32 = params.lineOffset + lineLocal;
  let t: u32 = i - lineLocal * M;
  if (t >= (N - 1u)) {
    a[i] = vec2<f32>(0.0, 0.0);
    return;
  }
  let base: u32 = line_base(line);
  // a_rev[t] = x[perm[(N-2)-t]]
  let idx: u32 = perm[(N - 2u) - t];
  a[i] = input[base + idx * STRIDE];
}
`;
}

export function generateRaderMulBfftWGSL({ mLength, workgroupSize }) {
  const M = mLength;
  return /* wgsl */ `
struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> bfft: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

${COMPLEX_WGSL}
const M: u32 = ${M}u;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  let k: u32 = i % M;
  a[i] = c_mul(a[i], bfft[k]);
}
`;
}

export function generateRaderWriteY0WGSL({ rank, axis, dims, axisLength, strideComplex, workgroupSize }) {
  const lineBase = wgslLineBaseFn({ rank, axis, dims });
  const STRIDE = strideComplex;
  const N = axisLength;
  return /* wgsl */ `
struct Params {
  lines: u32,
  lineOffset: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> sumAll: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const STRIDE: u32 = ${STRIDE}u;
${lineBase}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lineLocal: u32 = gid.x;
  if (lineLocal >= params.lines) { return; }
  let line: u32 = params.lineOffset + lineLocal;
  let base: u32 = line_base(line);
  output[base + 0u] = sumAll[lineLocal];
}
`;
}

export function generateRaderPostWGSL({
  rank,
  axis,
  dims,
  axisLength,
  mLength,
  strideComplex,
  workgroupSize,
}) {
  const N = axisLength;
  const M = mLength;
  const STRIDE = strideComplex;
  const lineBase = wgslLineBaseFn({ rank, axis, dims });
  return /* wgsl */ `
struct Params {
  lines: u32,
  lineOffset: u32,
  _pad1: u32,
  _pad2: u32,
}

  @group(0) @binding(0) var<storage, read_write> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> x0: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

${COMPLEX_WGSL}
const N: u32 = ${N}u;
const M: u32 = ${M}u;
const STRIDE: u32 = ${STRIDE}u;

${lineBase}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  let total: u32 = params.lines * (N - 1u);
  if (i >= total) { return; }
  let lineLocal: u32 = i / (N - 1u);
  let line: u32 = params.lineOffset + lineLocal;
  let t: u32 = i - lineLocal * (N - 1u); // 0..N-2
  let base: u32 = line_base(line);

  // Wrap linear convolution into cyclic length (N-1):
  var v: vec2<f32> = conv[lineLocal * M + t];
  let wrapIdx: u32 = t + (N - 1u);
  if (wrapIdx < M) {
    v = c_add(v, conv[lineLocal * M + wrapIdx]);
  }

  let outIdx: u32 = perm[t];
  output[base + outIdx * STRIDE] = c_add(x0[lineLocal], v);
}
`;
}
