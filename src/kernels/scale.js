import { COMPLEX_WGSL } from "./utils_wgsl.js";

export function generateScaleComplexWGSL({ workgroupSize }) {
  return /* wgsl */ `
struct Params {
  totalComplex: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  scale: f32,
  _pad3: f32,
  _pad4: f32,
  _pad5: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

${COMPLEX_WGSL}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.totalComplex) { return; }
  var v: vec2<f32> = data[i];
  v = v * vec2<f32>(params.scale, params.scale);
  data[i] = v;
}
`;
}

export function generateScaleRealWGSL({ workgroupSize }) {
  return /* wgsl */ `
struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  scale: f32,
  _pad3: f32,
  _pad4: f32,
  _pad5: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  data[i] = data[i] * params.scale;
}
`;
}
