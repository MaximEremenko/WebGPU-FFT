export function generateTransposeComplex2DWGSL({ Nx, Ny, tile = 16 }) {
  const T = tile;
  return /* wgsl */ `
struct Params {
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const NX: u32 = ${Nx}u;
const NY: u32 = ${Ny}u;
const TILE: u32 = ${T}u;

var<workgroup> tileData: array<vec2<f32>, ${T} * (${T} + 1)>;

fn tile_idx(x: u32, y: u32) -> u32 {
  // padded row stride TILE+1 to reduce bank conflicts
  return y * (TILE + 1u) + x;
}

@compute @workgroup_size(${T}, ${T}, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let b: u32 = wid.z;
  if (b >= params.batch) { return; }

  let x: u32 = wid.x * TILE + lid.x;
  let y: u32 = wid.y * TILE + lid.y;
  if (x < NX && y < NY) {
    let inIdx: u32 = b * (NX * NY) + y * NX + x;
    tileData[tile_idx(lid.x, lid.y)] = input[inIdx];
  }

  workgroupBarrier();

  let ox: u32 = wid.y * TILE + lid.x;
  let oy: u32 = wid.x * TILE + lid.y;
  if (ox < NY && oy < NX) {
    let outIdx: u32 = b * (NX * NY) + oy * NY + ox;
    output[outIdx] = tileData[tile_idx(lid.y, lid.x)];
  }
}
`;
}

