import { COMPLEX_WGSL } from "./utils_wgsl.js";
import { wgslLineBaseFn } from "./nd_line_base.js";

function isPowerOfTwo(n) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

export function generateSubgroupPow2FftWGSL({
  rank,
  axis,
  dims,
  axisLength,
  strideComplex,
  direction,
  applyScale,
  scaleFactor,
}) {
  if (!isPowerOfTwo(axisLength) || axisLength < 2) {
    throw new Error(`subgroup pow2 kernel requires axisLength power-of-two >=2; got ${axisLength}`);
  }
  const N = axisLength >>> 0;
  const LOGN = Math.round(Math.log2(axisLength)) >>> 0;
  const STRIDE = strideComplex >>> 0;
  const sign = direction === "forward" ? -1.0 : 1.0;

  const lineBaseFn = wgslLineBaseFn({ rank, axis, dims });

  const maybeScale = applyScale
    ? /* wgsl */ `
  val = val * vec2<f32>(${scaleFactor}, ${scaleFactor});
`
    : "";

  return /* wgsl */ `
enable subgroups;

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
const LOGN: u32 = ${LOGN}u;
const STRIDE: u32 = ${STRIDE}u;
const SIGN: f32 = ${sign};

${lineBaseFn}

fn bit_reverse(x0: u32) -> u32 {
  var x: u32 = x0;
  var y: u32 = 0u;
  for (var i: u32 = 0u; i < LOGN; i = i + 1u) {
    y = (y << 1u) | (x & 1u);
    x = x >> 1u;
  }
  return y;
}

var<workgroup> shmem: array<vec2<f32>, ${N}>;

@compute @workgroup_size(${N}, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sgSize: u32
) {
  let p: u32 = lid.x;
  let baseLineLocal: u32 = params.baseIndex / N;
  let lineLocal: u32 = baseLineLocal + wid.x;
  let line: u32 = params.lineOffset + lineLocal;
  let lineCount: u32 = params.total / N;
  let activeLine: bool = lineLocal < lineCount;

  var lineBaseGlobal: u32 = 0u;
  var val: vec2<f32> = vec2<f32>(0.0, 0.0);
  if (activeLine) {
    lineBaseGlobal = line_base(line);
    let inP: u32 = bit_reverse(p);
    let srcIdxGlobal: u32 = lineBaseGlobal + inP * STRIDE;
    let srcIdx: u32 = srcIdxGlobal - params.elementBase;
    val = src[srcIdx];
  }

  var m: u32 = 2u;
  var usingShared: bool = false;
  loop {
    if (m > N) { break; }
    let half: u32 = m >> 1u;
    let j: u32 = p & (half - 1u);
    let angle: f32 = SIGN * (2.0 * PI) * (f32(j) / f32(m));
    let w: vec2<f32> = cis(angle);

    if (half < sgSize) {
      let other: vec2<f32> = subgroupShuffleXor(val, half);
      if ((p & half) == 0u) {
        val = c_add(val, c_mul(w, other));
      } else {
        val = c_sub(other, c_mul(w, val));
      }
    } else {
      if (!usingShared) {
        usingShared = true;
      }
      shmem[p] = val;
      workgroupBarrier();
      let other: vec2<f32> = shmem[p ^ half];
      if ((p & half) == 0u) {
        val = c_add(shmem[p], c_mul(w, other));
      } else {
        val = c_sub(other, c_mul(w, shmem[p]));
      }
      workgroupBarrier();
    }
    m = m << 1u;
  }

${maybeScale}
  if (activeLine) {
    let dstIdxGlobal: u32 = lineBaseGlobal + p * STRIDE;
    let dstIdx: u32 = dstIdxGlobal - params.elementBase;
    dst[dstIdx] = val;
  }
}
`;
}
