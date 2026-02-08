function genBindings(prefix, cap, access) {
  const lines = [];
  for (let i = 0; i < cap; i++) {
    lines.push(`@group(0) @binding(${i}) var<storage, ${access}> ${prefix}${i}: array<u32>;`);
  }
  return lines.join("\n");
}

function genSwitchLoad(prefix, cap) {
  const lines = [];
  lines.push("fn load_seg(seg: u32, idx: u32) -> u32 {");
  for (let i = 0; i < cap; i++) {
    lines.push(`  if (seg == ${i}u) { return ${prefix}${i}[idx]; }`);
  }
  lines.push("  return 0u;");
  lines.push("}");
  return lines.join("\n");
}

function genSwitchStore(prefix, cap) {
  const lines = [];
  lines.push("fn store_seg(seg: u32, idx: u32, v: u32) {");
  for (let i = 0; i < cap; i++) {
    lines.push(`  if (seg == ${i}u) { ${prefix}${i}[idx] = v; return; }`);
  }
  lines.push("}");
  return lines.join("\n");
}

export function generateSegmentedCopyWGSL({ cap, direction, workgroupSize }) {
  // direction: "pack" => segmented src -> contiguous dst
  // direction: "unpack" => contiguous src -> segmented dst
  const isPack = direction === "pack";
  const segPrefix = isPack ? "src" : "dst";
  const contPrefix = isPack ? "dst" : "src";
  const segAccess = isPack ? "read" : "read_write";
  const contAccess = isPack ? "read_write" : "read";

  const segBindings = genBindings(segPrefix, cap, segAccess);
  const contBindingIndex = cap;
  const contDecl =
    direction === "pack"
      ? `@group(0) @binding(${contBindingIndex}) var<storage, read_write> dst: array<u32>;`
      : `@group(0) @binding(${contBindingIndex}) var<storage, read> src: array<u32>;`;

  const infoBindingIndex = cap + 1;

  const segLoadStore = isPack ? genSwitchLoad("src", cap) : genSwitchStore("dst", cap);
  const contLoadStore = isPack
    ? "fn store_cont(idx: u32, v: u32) { dst[idx] = v; }"
    : "fn load_cont(idx: u32) -> u32 { return src[idx]; }";

  // Uniform with segment metadata in 32-bit words.
  return /* wgsl */ `
struct SegInfo {
  segCount: u32,
  totalWords: u32,
  _pad0: u32,
  _pad1: u32,
  segSizeWords: array<u32, ${cap}>,
  segPrefixWords: array<u32, ${cap}>,
}

${segBindings}
${contDecl}
@group(0) @binding(${infoBindingIndex}) var<uniform> info: SegInfo;

${segLoadStore}
${contLoadStore}

fn find_seg(wordIndex: u32) -> vec2<u32> {
  // returns (segIndex, localWordIndex)
  for (var s: u32 = 0u; s < info.segCount; s = s + 1u) {
    let start: u32 = info.segPrefixWords[s];
    let size: u32 = info.segSizeWords[s];
    if (wordIndex >= start && wordIndex < (start + size)) {
      return vec2<u32>(s, wordIndex - start);
    }
  }
  return vec2<u32>(0u, 0u);
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= info.totalWords) {
    return;
  }

  let m: vec2<u32> = find_seg(i);
  let seg: u32 = m.x;
  let local: u32 = m.y;

  ${
    isPack
      ? "let v: u32 = load_seg(seg, local);\n  store_cont(i, v);"
      : "let v: u32 = load_cont(i);\n  store_seg(seg, local, v);"
  }
}
`;
}

