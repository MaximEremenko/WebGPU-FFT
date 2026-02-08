import { COMPLEX_WGSL } from "./utils_wgsl.js";
import { wgslLineBaseFn } from "./nd_line_base.js";

/**
 * Generates a single Stockham radix-2 stage kernel for one axis.
 * The kernel processes all lines for the axis and writes one output element per invocation.
 */
export function generateStockhamStageWGSL({
  rank,
  axis,
  dims,
  axisLength,
  strideComplex,
  stageIndex,
  direction,
  workgroupSize,
  applyScale,
  scaleFactor,
}) {
  const N = axisLength >>> 0;
  const stage = stageIndex >>> 0;
  const Ns = 1 << (stage + 1); // subtransform size for this iteration
  const halfNs = Ns >>> 1;
  const halfN = N >>> 1;
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
const NS: u32 = ${Ns}u;
const HALF_NS: u32 = ${halfNs}u;
const HALF_N: u32 = ${halfN}u;
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
  let baseGlobal: u32 = line_base(line);

  // Radix-2 Stockham formulation (Govindaraju et al. / Lloyd et al. style):
  // base = floor(p / Ns) * (Ns/2)
  // offset = p mod (Ns/2)
  // x0 = base + offset
  // x1 = x0 + N/2
  // out[p] = in[x0] + exp(SIGN * i*2π*(p mod Ns)/Ns) * in[x1]
  let base2: u32 = (p / NS) * HALF_NS;
  let offset: u32 = p - (p / HALF_NS) * HALF_NS;
  let x0: u32 = base2 + offset;
  let x1: u32 = x0 + HALF_N;

  let aIdxGlobal: u32 = baseGlobal + x0 * STRIDE;
  let bIdxGlobal: u32 = baseGlobal + x1 * STRIDE;
  let a: vec2<f32> = src[aIdxGlobal - params.elementBase];
  let b: vec2<f32> = src[bIdxGlobal - params.elementBase];

  let r: u32 = p - (p / NS) * NS;
  let angle: f32 = SIGN * (2.0 * PI) * (f32(r) / f32(NS));
  let w: vec2<f32> = cis(angle);

  var out: vec2<f32> = c_add(a, c_mul(w, b));
${maybeScale}
  let dstIdxGlobal: u32 = baseGlobal + p * STRIDE;
  dst[dstIdxGlobal - params.elementBase] = out;
}
`;
}
