export function generateF16ToF32ComplexWGSL({ workgroupSize }) {
  return /* wgsl */ `
enable f16;

struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  let v: vec2<f16> = input[i];
  output[i] = vec2<f32>(f32(v.x), f32(v.y));
}
`;
}

export function generateF32ToF16ComplexWGSL({ workgroupSize }) {
  return /* wgsl */ `
enable f16;

struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f16>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  let v: vec2<f32> = input[i];
  output[i] = vec2<f16>(f16(v.x), f16(v.y));
}
`;
}

export function generateF16ToF32RealWGSL({ workgroupSize }) {
  return /* wgsl */ `
enable f16;

struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  output[i] = f32(input[i]);
}
`;
}

export function generateF32ToF16RealWGSL({ workgroupSize }) {
  return /* wgsl */ `
enable f16;

struct Params {
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.total) { return; }
  output[i] = f16(input[i]);
}
`;
}
