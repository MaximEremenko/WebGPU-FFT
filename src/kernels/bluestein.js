import { COMPLEX_WGSL } from "./utils_wgsl.js";
import { wgslLineBaseFn } from "./nd_line_base.js";

export function generateBluesteinPreWGSL({
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
@group(0) @binding(2) var<storage, read> chirpA: array<vec2<f32>>;
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
  if (t >= N) {
    a[i] = vec2<f32>(0.0, 0.0);
    return;
  }
  let base: u32 = line_base(line);
  let x: vec2<f32> = input[base + t * STRIDE];
  a[i] = c_mul(x, chirpA[t]);
}
`;
}

export function generateBluesteinMulBfftWGSL({ mLength, workgroupSize }) {
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

export function generateBluesteinPostWGSL({
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

@group(0) @binding(0) var<storage, read_write> a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> chirpC: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

${COMPLEX_WGSL}
const N: u32 = ${N}u;
const M: u32 = ${M}u;
const STRIDE: u32 = ${strideComplex}u;

${lineBase}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  let total: u32 = params.lines * N;
  if (i >= total) { return; }
  let lineLocal: u32 = i / N;
  let line: u32 = params.lineOffset + lineLocal;
  let t: u32 = i - lineLocal * N;
  let base: u32 = line_base(line);
  let v: vec2<f32> = a[lineLocal * M + t];
  output[base + t * STRIDE] = c_mul(v, chirpC[t]);
}
`;
}
