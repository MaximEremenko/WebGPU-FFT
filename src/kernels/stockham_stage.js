import { COMPLEX_WGSL } from "./utils_wgsl.js";
import { wgslLineBaseFn } from "./nd_line_base.js";

/**
 * Generates a Stockham autosort stage kernel for an arbitrary radix (compile-time constant).
 *
 * Stage parameters:
 * - N: axis length
 * - RADIX: stage radix
 * - NS: cumulative subtransform length after this stage (product of radices up to stage)
 *
 * Mapping:
 * base = floor(p / NS) * (NS/RADIX) + (p mod (NS/RADIX))
 * in_q = base + q*(N/RADIX)
 * out[p] = sum_q in_q * exp(SIGN*i*2π*q*(p mod NS)/NS)
 */
export function generateStockhamRadixStageWGSL({
  rank,
  axis,
  dims,
  axisLength,
  strideComplex,
  radix,
  ns,
  direction,
  workgroupSize,
  applyScale,
  scaleFactor,
}) {
  const N = axisLength >>> 0;
  const R = radix >>> 0;
  const NS = ns >>> 0;
  const NS_DIV_R = NS / R;
  const N_DIV_R = N / R;
  const sign = direction === "forward" ? -1.0 : 1.0;

  const lineBaseFn = wgslLineBaseFn({ rank, axis, dims });

  const maybeScale = applyScale
    ? /* wgsl */ `
  out = out * vec2<f32>(${scaleFactor}, ${scaleFactor});
`
    : "";

  return /* wgsl */ `
struct Params {
  total: u32,
  baseIndex: u32,
  lineOffset: u32,
  elementBase: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

${COMPLEX_WGSL}

const N: u32 = ${N}u;
const RADIX: u32 = ${R}u;
const NS: u32 = ${NS}u;
const NS_DIV_R: u32 = ${NS_DIV_R}u;
const N_DIV_R: u32 = ${N_DIV_R}u;
const STRIDE: u32 = ${strideComplex}u;
const SIGN: f32 = ${sign};

${lineBaseFn}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = params.baseIndex + gid.x;
  if (idx >= params.total) {
    return;
  }

  let lineLocal: u32 = idx / N;
  let line: u32 = params.lineOffset + lineLocal;
  let p: u32 = idx - lineLocal * N;
  let baseLineGlobal: u32 = line_base(line);

  let block: u32 = p / NS;
  let p_in_block: u32 = p - block * NS;
  let offset: u32 = p - (p / NS_DIV_R) * NS_DIV_R;
  let base: u32 = block * NS_DIV_R + offset;

  let r: u32 = p_in_block;
  let angle: f32 = SIGN * (2.0 * PI) * (f32(r) / f32(NS));
  let w1: vec2<f32> = cis(angle);

  var w: vec2<f32> = vec2<f32>(1.0, 0.0);
  var out: vec2<f32> = vec2<f32>(0.0, 0.0);

  for (var q: u32 = 0u; q < RADIX; q = q + 1u) {
    let srcIdxGlobal: u32 = baseLineGlobal + (base + q * N_DIV_R) * STRIDE;
    let srcIdx: u32 = srcIdxGlobal - params.elementBase;
    let x: vec2<f32> = src[srcIdx];
    out = c_add(out, c_mul(w, x));
    w = c_mul(w, w1);
  }
${maybeScale}
  let dstIdxGlobal: u32 = baseLineGlobal + p * STRIDE;
  let dstIdx: u32 = dstIdxGlobal - params.elementBase;
  dst[dstIdx] = out;
}
`;
}
