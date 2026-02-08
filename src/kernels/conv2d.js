import { COMPLEX_WGSL } from "./utils_wgsl.js";

export function generateConv2dRealWGSL({ Hin, Win, Hout, Wout, k, pad, workgroupSize }) {
  const [pt, pb, pl, pr] = pad;
  return /* wgsl */ `
struct Params {
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const HIN: i32 = ${Hin};
const WIN: i32 = ${Win};
const HOUT: i32 = ${Hout};
const WOUT: i32 = ${Wout};
const K: i32 = ${k};
const PAD_T: i32 = ${pt};
const PAD_L: i32 = ${pl};

fn in_index(b: i32, y: i32, x: i32) -> i32 {
  return b * (HIN * WIN) + y * WIN + x;
}

fn out_index(b: i32, y: i32, x: i32) -> i32 {
  return b * (HOUT * WOUT) + y * WOUT + x;
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: i32 = i32(gid.x);
  let total: i32 = i32(params.batch) * (HOUT * WOUT);
  if (i >= total) { return; }
  let b: i32 = i / (HOUT * WOUT);
  let rem: i32 = i - b * (HOUT * WOUT);
  let y: i32 = rem / WOUT;
  let x: i32 = rem - y * WOUT;

  var acc: f32 = 0.0;
  for (var ky: i32 = 0; ky < K; ky = ky + 1) {
    for (var kx: i32 = 0; kx < K; kx = kx + 1) {
      let iy: i32 = y + ky - PAD_T;
      let ix: i32 = x + kx - PAD_L;
      if (iy < 0 || ix < 0 || iy >= HIN || ix >= WIN) { continue; }
      let inIdx: i32 = in_index(b, iy, ix);
      let kIdx: i32 = ky * K + kx;
      acc = acc + input[u32(inIdx)] * kernel[u32(kIdx)];
    }
  }
  output[u32(out_index(b, y, x))] = acc;
}
`;
}

export function generateConv2dComplexRealKernelWGSL({ Hin, Win, Hout, Wout, k, pad, workgroupSize }) {
  const [pt, pb, pl, pr] = pad;
  return /* wgsl */ `
struct Params {
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

${COMPLEX_WGSL}

const HIN: i32 = ${Hin};
const WIN: i32 = ${Win};
const HOUT: i32 = ${Hout};
const WOUT: i32 = ${Wout};
const K: i32 = ${k};
const PAD_T: i32 = ${pt};
const PAD_L: i32 = ${pl};

fn in_index(b: i32, y: i32, x: i32) -> i32 {
  return b * (HIN * WIN) + y * WIN + x;
}

fn out_index(b: i32, y: i32, x: i32) -> i32 {
  return b * (HOUT * WOUT) + y * WOUT + x;
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: i32 = i32(gid.x);
  let total: i32 = i32(params.batch) * (HOUT * WOUT);
  if (i >= total) { return; }
  let b: i32 = i / (HOUT * WOUT);
  let rem: i32 = i - b * (HOUT * WOUT);
  let y: i32 = rem / WOUT;
  let x: i32 = rem - y * WOUT;

  var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  for (var ky: i32 = 0; ky < K; ky = ky + 1) {
    for (var kx: i32 = 0; kx < K; kx = kx + 1) {
      let iy: i32 = y + ky - PAD_T;
      let ix: i32 = x + kx - PAD_L;
      if (iy < 0 || ix < 0 || iy >= HIN || ix >= WIN) { continue; }
      let inIdx: i32 = in_index(b, iy, ix);
      let kIdx: i32 = ky * K + kx;
      let w: f32 = kernel[u32(kIdx)];
      acc = c_add(acc, input[u32(inIdx)] * vec2<f32>(w, w));
    }
  }
  output[u32(out_index(b, y, x))] = acc;
}
`;
}

export function generateConv2dComplexComplexKernelWGSL({ Hin, Win, Hout, Wout, k, pad, workgroupSize }) {
  const [pt, pb, pl, pr] = pad;
  return /* wgsl */ `
struct Params {
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> kernel: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

${COMPLEX_WGSL}

const HIN: i32 = ${Hin};
const WIN: i32 = ${Win};
const HOUT: i32 = ${Hout};
const WOUT: i32 = ${Wout};
const K: i32 = ${k};
const PAD_T: i32 = ${pt};
const PAD_L: i32 = ${pl};

fn in_index(b: i32, y: i32, x: i32) -> i32 {
  return b * (HIN * WIN) + y * WIN + x;
}

fn out_index(b: i32, y: i32, x: i32) -> i32 {
  return b * (HOUT * WOUT) + y * WOUT + x;
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: i32 = i32(gid.x);
  let total: i32 = i32(params.batch) * (HOUT * WOUT);
  if (i >= total) { return; }
  let b: i32 = i / (HOUT * WOUT);
  let rem: i32 = i - b * (HOUT * WOUT);
  let y: i32 = rem / WOUT;
  let x: i32 = rem - y * WOUT;

  var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  for (var ky: i32 = 0; ky < K; ky = ky + 1) {
    for (var kx: i32 = 0; kx < K; kx = kx + 1) {
      let iy: i32 = y + ky - PAD_T;
      let ix: i32 = x + kx - PAD_L;
      if (iy < 0 || ix < 0 || iy >= HIN || ix >= WIN) { continue; }
      let inIdx: i32 = in_index(b, iy, ix);
      let kIdx: i32 = ky * K + kx;
      acc = c_add(acc, c_mul(input[u32(inIdx)], kernel[u32(kIdx)]));
    }
  }
  output[u32(out_index(b, y, x))] = acc;
}
`;
}

