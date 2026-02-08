import { COMPLEX_WGSL } from "./utils_wgsl.js";
import { wgslLineBaseFn } from "./nd_line_base.js";

function dimsToStride(axis, dims) {
  let s = 1;
  for (let i = 0; i < axis; i++) s *= dims[i];
  return s;
}

export function dctWorkLength(typeKind, axisLength) {
  if (typeKind === "dct1") return 2 * (axisLength - 1);
  if (typeKind === "dst1") return 2 * (axisLength + 1);
  return 2 * axisLength;
}

export function dctFftDirection(typeKind) {
  if (typeKind === "dct2_inv" || typeKind === "dst2_inv") return "inverse";
  return "forward";
}

export function generateDctFftBuildWGSL({ typeKind, rank, axis, dims, axisLength, workgroupSize }) {
  const N = axisLength >>> 0;
  const M = dctWorkLength(typeKind, axisLength) >>> 0;
  const STRIDE = dimsToStride(axis, dims) >>> 0;
  const lines = dims.reduce((a, b) => a * b, 1) / axisLength;
  if (!Number.isInteger(lines) || lines <= 0) throw new Error("invalid dims/axisLength");

  const lineBaseFn = wgslLineBaseFn({ rank, axis, dims });

  const body = (() => {
    if (typeKind === "dct1") {
      if (N < 2) throw new Error("dct1 requires N>=2");
      return /* wgsl */ `
  let xi: u32 = select(M - p, p, p < N);
  let x: f32 = src[base + xi * STRIDE];
  work[idx] = vec2<f32>(x, 0.0);
`;
    }
    if (typeKind === "dst1") {
      if (N < 2) throw new Error("dst1 requires N>=2");
      return /* wgsl */ `
  if (p == 0u || p == (N + 1u)) {
    work[idx] = vec2<f32>(0.0, 0.0);
    return;
  }
  if (p < (N + 1u)) {
    let x: f32 = src[base + (p - 1u) * STRIDE];
    work[idx] = vec2<f32>(x, 0.0);
    return;
  }
  let xi: u32 = (M - p) - 1u;
  let x: f32 = src[base + xi * STRIDE];
  work[idx] = vec2<f32>(-x, 0.0);
`;
    }
    if (typeKind === "dct2_fwd") {
      return /* wgsl */ `
  let xi: u32 = select((M - 1u) - p, p, p < N);
  let x: f32 = src[base + xi * STRIDE];
  work[idx] = vec2<f32>(x, 0.0);
`;
    }
    if (typeKind === "dst2_fwd") {
      return /* wgsl */ `
  let left: bool = p < N;
  let xi: u32 = select((M - 1u) - p, p, left);
  let x: f32 = src[base + xi * STRIDE];
  let sgn: f32 = select(-1.0, 1.0, left);
  work[idx] = vec2<f32>(sgn * x, 0.0);
`;
    }
    if (typeKind === "dct2_inv") {
      return /* wgsl */ `
  // Build packed spectrum for length M=2N (full spectrum in work[] with conjugate symmetry).
  if (p == N) {
    work[idx] = vec2<f32>(0.0, 0.0);
    return;
  }
  let conjSide: bool = p > N;
  let kk: u32 = select(p, M - p, conjSide);
  let c0: f32 = src[base + kk * STRIDE];
  let theta: f32 = (PI * f32(kk)) / (2.0 * f32(N));
  let w: vec2<f32> = cis(theta);
  var v: vec2<f32> = c_mul(vec2<f32>(2.0 * c0, 0.0), w);
  if (conjSide) { v = vec2<f32>(v.x, -v.y); }
  work[idx] = v;
`;
    }
    if (typeKind === "dst2_inv") {
      return /* wgsl */ `
  // Build packed spectrum for length M=2N (full spectrum in work[] with conjugate symmetry).
  if (p == 0u) {
    work[idx] = vec2<f32>(0.0, 0.0);
    return;
  }
  let conjSide: bool = p > N;
  let kk: u32 = select(p, M - p, conjSide);
  let c0: f32 = src[base + (kk - 1u) * STRIDE];
  let theta: f32 = (PI * f32(kk)) / (2.0 * f32(N));
  let w: vec2<f32> = cis(theta);
  var v: vec2<f32> = c_mul(vec2<f32>(0.0, -2.0 * c0), w);
  if (conjSide) { v = vec2<f32>(v.x, -v.y); }
  work[idx] = v;
`;
    }
    if (typeKind === "dct4") {
      return /* wgsl */ `
  if (p < N) {
    let x: f32 = src[base + p * STRIDE];
    let theta: f32 = -(PI * f32(p)) / (2.0 * f32(N));
    let w: vec2<f32> = cis(theta);
    work[idx] = vec2<f32>(x * w.x, x * w.y);
  } else {
    work[idx] = vec2<f32>(0.0, 0.0);
  }
`;
    }
    if (typeKind === "dst4") {
      return /* wgsl */ `
  if (p < N) {
    let x: f32 = src[base + p * STRIDE];
    let theta: f32 = -(PI * f32(p)) / (2.0 * f32(N));
    let w: vec2<f32> = cis(theta);
    work[idx] = vec2<f32>(x * w.x, x * w.y);
  } else {
    work[idx] = vec2<f32>(0.0, 0.0);
  }
`;
    }
    throw new Error(`unknown typeKind ${typeKind}`);
  })();

  return /* wgsl */ `
struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

${COMPLEX_WGSL}

const N: u32 = ${N}u;
const M: u32 = ${M}u;
const STRIDE: u32 = ${STRIDE}u;

${lineBaseFn}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx >= params.total) { return; }
  let line: u32 = idx / M;
  let p: u32 = idx - line * M;
  let base: u32 = line_base(line);
${body}
}
`;
}

export function generateDctFftPostWGSL({ typeKind, rank, axis, dims, axisLength, workgroupSize }) {
  const N = axisLength >>> 0;
  const M = dctWorkLength(typeKind, axisLength) >>> 0;
  const STRIDE = dimsToStride(axis, dims) >>> 0;
  const lineBaseFn = wgslLineBaseFn({ rank, axis, dims });

  const body = (() => {
    if (typeKind === "dct1") {
      return /* wgsl */ `
  let y: vec2<f32> = work[line * M + k];
  dst[base + k * STRIDE] = y.x;
`;
    }
    if (typeKind === "dst1") {
      return /* wgsl */ `
  let y: vec2<f32> = work[line * M + (k + 1u)];
  dst[base + k * STRIDE] = -0.5 * y.y;
`;
    }
    if (typeKind === "dct2_fwd") {
      return /* wgsl */ `
  let y: vec2<f32> = work[line * M + k];
  let theta: f32 = -(PI * f32(k)) / (2.0 * f32(N));
  let w: vec2<f32> = cis(theta);
  let z: vec2<f32> = c_mul(w, y);
  dst[base + k * STRIDE] = 0.5 * z.x;
`;
    }
    if (typeKind === "dst2_fwd") {
      return /* wgsl */ `
  let y: vec2<f32> = work[line * M + (k + 1u)];
  let theta: f32 = -(PI * f32(k + 1u)) / (2.0 * f32(N));
  let w: vec2<f32> = cis(theta);
  let z: vec2<f32> = c_mul(w, y);
  dst[base + k * STRIDE] = -0.5 * z.y;
`;
    }
    if (typeKind === "dct2_inv") {
      return /* wgsl */ `
  let y: vec2<f32> = work[line * M + k];
  dst[base + k * STRIDE] = 0.25 * y.x;
`;
    }
    if (typeKind === "dst2_inv") {
      return /* wgsl */ `
  let y: vec2<f32> = work[line * M + k];
  dst[base + k * STRIDE] = 0.25 * y.x;
`;
    }
    if (typeKind === "dct4") {
      return /* wgsl */ `
  let y: vec2<f32> = work[line * M + k];
  let theta: f32 = -(PI * (f32(k) + 0.5)) / (2.0 * f32(N));
  let w: vec2<f32> = cis(theta);
  let z: vec2<f32> = c_mul(w, y);
  dst[base + k * STRIDE] = z.x;
`;
    }
    if (typeKind === "dst4") {
      return /* wgsl */ `
  let y: vec2<f32> = work[line * M + k];
  let theta: f32 = -(PI * (f32(k) + 0.5)) / (2.0 * f32(N));
  let w: vec2<f32> = cis(theta);
  let z: vec2<f32> = c_mul(w, y);
  dst[base + k * STRIDE] = -z.y;
`;
    }
    throw new Error(`unknown typeKind ${typeKind}`);
  })();

  return /* wgsl */ `
struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

${COMPLEX_WGSL}

const N: u32 = ${N}u;
const M: u32 = ${M}u;
const STRIDE: u32 = ${STRIDE}u;

${lineBaseFn}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx >= params.total) { return; }
  let line: u32 = idx / N;
  let k: u32 = idx - line * N;
  let base: u32 = line_base(line);
${body}
}
`;
}
