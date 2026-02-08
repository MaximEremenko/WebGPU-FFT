function dimsConstU32(dims) {
  return dims.map((d) => `${d | 0}u`).join(", ");
}

function dimsConstI32(dims) {
  return dims.map((d) => `${d | 0}`).join(", ");
}

function wgslCoordFromIndex(rank, indexName, dimsName, coordName) {
  let out = `  var ${coordName}: array<u32, ${rank}>;\n`;
  out += `  var rem: u32 = ${indexName};\n`;
  for (let d = 0; d < rank; d++) {
    out += `  ${coordName}[${d}] = rem % ${dimsName}[${d}];\n`;
    if (d < rank - 1) out += `  rem = rem / ${dimsName}[${d}];\n`;
  }
  out += `  return ${coordName};\n`;
  return out;
}

function wgslIndexFromCoord(rank, coordName, dimsName) {
  const terms = [];
  for (let d = 0; d < rank; d++) {
    if (d === 0) {
      terms.push(`${coordName}[0]`);
      continue;
    }
    const stride = [];
    for (let s = 0; s < d; s++) stride.push(`${dimsName}[${s}]`);
    terms.push(`${coordName}[${d}] * (${stride.join(" * ")})`);
  }
  return terms.join(" + ");
}

function wgslOffsetDecl(rank, srcCoordArrayName, offsetName, dstPrefix, sign) {
  let out = "";
  for (let d = 0; d < rank; d++) {
    out += `  let ${dstPrefix}${d}: i32 = i32(${srcCoordArrayName}[${d}]) ${sign} ${offsetName}[${d}];\n`;
  }
  return out;
}

function wgslBoundsExpr(rank, prefix, dimsName) {
  const terms = [];
  for (let d = 0; d < rank; d++) {
    terms.push(`${prefix}${d} >= 0 && ${prefix}${d} < i32(${dimsName}[${d}])`);
  }
  return terms.join(" && ");
}

function wgslBuildU32CoordArray(rank, arrayName, prefix) {
  let out = `  var ${arrayName}: array<u32, ${rank}>;\n`;
  for (let d = 0; d < rank; d++) out += `  ${arrayName}[${d}] = u32(${prefix}${d});\n`;
  return out;
}

export function generateEmbedComplexWGSL({ rank, logicalDims, viewDims, offset, workgroupSize }) {
  const LDIMS = logicalDims;
  const VDIMS = viewDims;
  const OFF = offset;
  const vcDecl = wgslOffsetDecl(rank, "c", "OFF", "vc", "-");
  const inBounds = wgslBoundsExpr(rank, "vc", "VDIMS");
  const buildVcu = wgslBuildU32CoordArray(rank, "vcu", "vc");
  return /* wgsl */ `
struct Params {
  totalLogical: u32,
  totalView: u32,
  batch: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const LDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(LDIMS)});
const VDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(VDIMS)});
const OFF: array<i32, ${rank}> = array<i32, ${rank}>(${dimsConstI32(OFF)});

fn coord_from_index(i: u32) -> array<u32, ${rank}> {
${wgslCoordFromIndex(rank, "i", "LDIMS", "c")}
}

fn view_index(v: array<u32, ${rank}>) -> u32 {
  return ${wgslIndexFromCoord(rank, "v", "VDIMS")};
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.totalLogical * params.batch) { return; }
  let b: u32 = i / params.totalLogical;
  let li: u32 = i - b * params.totalLogical;
  let c: array<u32, ${rank}> = coord_from_index(li);

${vcDecl}
  if (!(${inBounds})) {
    output[i] = vec2<f32>(0.0, 0.0);
    return;
  }
${buildVcu}
  let vi: u32 = view_index(vcu);
  output[i] = input[b * params.totalView + vi];
}
`;
}

export function generateEmbedComplexF16ToF32WGSL({ rank, logicalDims, viewDims, offset, workgroupSize }) {
  const LDIMS = logicalDims;
  const VDIMS = viewDims;
  const OFF = offset;
  const vcDecl = wgslOffsetDecl(rank, "c", "OFF", "vc", "-");
  const inBounds = wgslBoundsExpr(rank, "vc", "VDIMS");
  const buildVcu = wgslBuildU32CoordArray(rank, "vcu", "vc");
  return /* wgsl */ `
enable f16;

struct Params {
  totalLogical: u32,
  totalView: u32,
  batch: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const LDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(LDIMS)});
const VDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(VDIMS)});
const OFF: array<i32, ${rank}> = array<i32, ${rank}>(${dimsConstI32(OFF)});

fn coord_from_index(i: u32) -> array<u32, ${rank}> {
${wgslCoordFromIndex(rank, "i", "LDIMS", "c")}
}

fn view_index(v: array<u32, ${rank}>) -> u32 {
  return ${wgslIndexFromCoord(rank, "v", "VDIMS")};
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.totalLogical * params.batch) { return; }
  let b: u32 = i / params.totalLogical;
  let li: u32 = i - b * params.totalLogical;
  let c: array<u32, ${rank}> = coord_from_index(li);

${vcDecl}
  if (!(${inBounds})) {
    output[i] = vec2<f32>(0.0, 0.0);
    return;
  }
${buildVcu}
  let vi: u32 = view_index(vcu);
  let v: vec2<f16> = input[b * params.totalView + vi];
  output[i] = vec2<f32>(f32(v.x), f32(v.y));
}
`;
}

export function generateExtractComplexWGSL({ rank, logicalDims, viewDims, offset, clearOutside, workgroupSize }) {
  const LDIMS = logicalDims;
  const VDIMS = viewDims;
  const OFF = offset;
  const doClear = !!clearOutside;
  const lcDecl = wgslOffsetDecl(rank, "c", "OFF", "lc", "+");
  const inBounds = wgslBoundsExpr(rank, "lc", "LDIMS");
  const buildLcu = wgslBuildU32CoordArray(rank, "lcu", "lc");
  return /* wgsl */ `
struct Params {
  totalLogical: u32,
  totalView: u32,
  batch: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const LDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(LDIMS)});
const VDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(VDIMS)});
const OFF: array<i32, ${rank}> = array<i32, ${rank}>(${dimsConstI32(OFF)});
const CLEAR_OUTSIDE: bool = ${doClear ? "true" : "false"};

fn coord_from_index_view(i: u32) -> array<u32, ${rank}> {
${wgslCoordFromIndex(rank, "i", "VDIMS", "c")}
}

fn logical_index(v: array<u32, ${rank}>) -> u32 {
  return ${wgslIndexFromCoord(rank, "v", "LDIMS")};
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.totalView * params.batch) { return; }
  let b: u32 = i / params.totalView;
  let vi: u32 = i - b * params.totalView;
  let c: array<u32, ${rank}> = coord_from_index_view(vi);

  // output view coord -> logical coord
${lcDecl}
  if (!(${inBounds})) {
    if (CLEAR_OUTSIDE) {
      output[i] = vec2<f32>(0.0, 0.0);
    }
    return;
  }
${buildLcu}
  let li: u32 = logical_index(lcu);
  output[i] = input[b * params.totalLogical + li];
}
`;
}

export function generateExtractComplexF32ToF16WGSL({ rank, logicalDims, viewDims, offset, clearOutside, workgroupSize }) {
  const LDIMS = logicalDims;
  const VDIMS = viewDims;
  const OFF = offset;
  const doClear = !!clearOutside;
  const lcDecl = wgslOffsetDecl(rank, "c", "OFF", "lc", "+");
  const inBounds = wgslBoundsExpr(rank, "lc", "LDIMS");
  const buildLcu = wgslBuildU32CoordArray(rank, "lcu", "lc");
  return /* wgsl */ `
enable f16;

struct Params {
  totalLogical: u32,
  totalView: u32,
  batch: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f16>>;
@group(0) @binding(2) var<uniform> params: Params;

const LDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(LDIMS)});
const VDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(VDIMS)});
const OFF: array<i32, ${rank}> = array<i32, ${rank}>(${dimsConstI32(OFF)});
const CLEAR_OUTSIDE: bool = ${doClear ? "true" : "false"};

fn coord_from_index_view(i: u32) -> array<u32, ${rank}> {
${wgslCoordFromIndex(rank, "i", "VDIMS", "c")}
}

fn logical_index(v: array<u32, ${rank}>) -> u32 {
  return ${wgslIndexFromCoord(rank, "v", "LDIMS")};
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.totalView * params.batch) { return; }
  let b: u32 = i / params.totalView;
  let vi: u32 = i - b * params.totalView;
  let c: array<u32, ${rank}> = coord_from_index_view(vi);

  // output view coord -> logical coord
${lcDecl}
  if (!(${inBounds})) {
    if (CLEAR_OUTSIDE) {
      output[i] = vec2<f16>(f16(0.0), f16(0.0));
    }
    return;
  }
${buildLcu}
  let li: u32 = logical_index(lcu);
  let v: vec2<f32> = input[b * params.totalLogical + li];
  output[i] = vec2<f16>(f16(v.x), f16(v.y));
}
`;
}

export function generateEmbedRealWGSL({ rank, logicalDims, viewDims, offset, workgroupSize }) {
  const LDIMS = logicalDims;
  const VDIMS = viewDims;
  const OFF = offset;
  const vcDecl = wgslOffsetDecl(rank, "c", "OFF", "vc", "-");
  const inBounds = wgslBoundsExpr(rank, "vc", "VDIMS");
  const buildVcu = wgslBuildU32CoordArray(rank, "vcu", "vc");
  return /* wgsl */ `
struct Params {
  totalLogical: u32,
  totalView: u32,
  batch: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const LDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(LDIMS)});
const VDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(VDIMS)});
const OFF: array<i32, ${rank}> = array<i32, ${rank}>(${dimsConstI32(OFF)});

fn coord_from_index(i: u32) -> array<u32, ${rank}> {
${wgslCoordFromIndex(rank, "i", "LDIMS", "c")}
}

fn view_index(v: array<u32, ${rank}>) -> u32 {
  return ${wgslIndexFromCoord(rank, "v", "VDIMS")};
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.totalLogical * params.batch) { return; }
  let b: u32 = i / params.totalLogical;
  let li: u32 = i - b * params.totalLogical;
  let c: array<u32, ${rank}> = coord_from_index(li);

${vcDecl}
  if (!(${inBounds})) {
    output[i] = 0.0;
    return;
  }
${buildVcu}
  let vi: u32 = view_index(vcu);
  output[i] = input[b * params.totalView + vi];
}
`;
}

export function generateExtractRealWGSL({ rank, logicalDims, viewDims, offset, clearOutside, workgroupSize }) {
  const LDIMS = logicalDims;
  const VDIMS = viewDims;
  const OFF = offset;
  const doClear = !!clearOutside;
  const lcDecl = wgslOffsetDecl(rank, "c", "OFF", "lc", "+");
  const inBounds = wgslBoundsExpr(rank, "lc", "LDIMS");
  const buildLcu = wgslBuildU32CoordArray(rank, "lcu", "lc");
  return /* wgsl */ `
struct Params {
  totalLogical: u32,
  totalView: u32,
  batch: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const LDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(LDIMS)});
const VDIMS: array<u32, ${rank}> = array<u32, ${rank}>(${dimsConstU32(VDIMS)});
const OFF: array<i32, ${rank}> = array<i32, ${rank}>(${dimsConstI32(OFF)});
const CLEAR_OUTSIDE: bool = ${doClear ? "true" : "false"};

fn coord_from_index_view(i: u32) -> array<u32, ${rank}> {
${wgslCoordFromIndex(rank, "i", "VDIMS", "c")}
}

fn logical_index(v: array<u32, ${rank}>) -> u32 {
  return ${wgslIndexFromCoord(rank, "v", "LDIMS")};
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.totalView * params.batch) { return; }
  let b: u32 = i / params.totalView;
  let vi: u32 = i - b * params.totalView;
  let c: array<u32, ${rank}> = coord_from_index_view(vi);

${lcDecl}
  if (!(${inBounds})) {
    if (CLEAR_OUTSIDE) {
      output[i] = 0.0;
    }
    return;
  }
${buildLcu}
  let li: u32 = logical_index(lcu);
  output[i] = input[b * params.totalLogical + li];
}
`;
}
