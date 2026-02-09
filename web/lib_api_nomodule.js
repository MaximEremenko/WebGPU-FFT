/* webgpufft API bundle (file:// compatible) */
(function(){
  'use strict';
  const __modules = Object.create(null);
  const __cache = Object.create(null);
  function __define(id, fn){ __modules[id]=fn; }
  function __require(id){
    if(__cache[id]) return __cache[id].exports;
    const fn=__modules[id];
    if(!fn) throw new Error('Missing module: '+id);
    const module={exports:{}};
    __cache[id]=module;
    fn(__require, module.exports, module);
    return module.exports;
  }

  __define('src/index.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const __reexport_1 = require('src/plan.js');
    exports['createFftPlan'] = __reexport_1['createFftPlan'];
    const __reexport_2 = require('src/public_api.js');
    exports['createPlan'] = __reexport_2['createPlan'];
    exports['exportPipelineCacheSnapshot'] = __reexport_2['exportPipelineCacheSnapshot'];
    exports['importPipelineCacheSnapshot'] = __reexport_2['importPipelineCacheSnapshot'];
    exports['createFftConvChannelLanePreset'] = __reexport_2['createFftConvChannelLanePreset'];
    exports['createFftConvKernelMajorChannelLanePreset'] = __reexport_2['createFftConvKernelMajorChannelLanePreset'];
    exports['createFftConvBatchMajorChannelLanePreset'] = __reexport_2['createFftConvBatchMajorChannelLanePreset'];
    const __reexport_3 = require('src/utils/buffer_view.js');
    exports['BufferView'] = __reexport_3['BufferView'];
    const __reexport_4 = require('src/utils/webgpu.js');
    exports['uploadComplex'] = __reexport_4['uploadComplex'];
    exports['downloadComplex'] = __reexport_4['downloadComplex'];
  });

  __define('src/kernels/bluestein.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    const { wgslLineBaseFn } = require('src/kernels/nd_line_base.js');
    
    function generateBluesteinPreWGSL({
      rank,
      axis,
      dims,
      axisLength,
      mLength,
      strideComplex,
      workgroupSize,
    }) {
      const N = axisLength;
      const M = mLength;
      const lineBase = wgslLineBaseFn({ rank, axis, dims });
      return /* wgsl */ `
    struct Params {
      lines: u32,
      lineOffset: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> a: array<vec2<f32>>;
    @group(0) @binding(2) var<storage, read> chirpA: array<vec2<f32>>;
    @group(0) @binding(3) var<uniform> params: Params;
    
    ${COMPLEX_WGSL}
    
    const N: u32 = ${N}u;
    const M: u32 = ${M}u;
    const STRIDE: u32 = ${strideComplex}u;
    
    ${lineBase}
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.lines * M;
      if (i >= total) { return; }
      let lineLocal: u32 = i / M;
      let line: u32 = params.lineOffset + lineLocal;
      let t: u32 = i - lineLocal * M;
      if (t >= N) {
        a[i] = vec2<f32>(0.0, 0.0);
        return;
      }
      let base: u32 = line_base(line);
      let x: vec2<f32> = input[base + t * STRIDE];
      a[i] = c_mul(x, chirpA[t]);
    }
    `;
    }
    
    function generateBluesteinMulBfftWGSL({ mLength, workgroupSize }) {
      const M = mLength;
      return /* wgsl */ `
    struct Params {
      total: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> a: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read> bfft: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    ${COMPLEX_WGSL}
    const M: u32 = ${M}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.total) { return; }
      let k: u32 = i % M;
      a[i] = c_mul(a[i], bfft[k]);
    }
    `;
    }
    
    function generateBluesteinPostWGSL({
      rank,
      axis,
      dims,
      axisLength,
      mLength,
      strideComplex,
      workgroupSize,
    }) {
      const N = axisLength;
      const M = mLength;
      const lineBase = wgslLineBaseFn({ rank, axis, dims });
      return /* wgsl */ `
    struct Params {
      lines: u32,
      lineOffset: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> a: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
    @group(0) @binding(2) var<storage, read> chirpC: array<vec2<f32>>;
    @group(0) @binding(3) var<uniform> params: Params;
    
    ${COMPLEX_WGSL}
    const N: u32 = ${N}u;
    const M: u32 = ${M}u;
    const STRIDE: u32 = ${strideComplex}u;
    
    ${lineBase}
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.lines * N;
      if (i >= total) { return; }
      let lineLocal: u32 = i / N;
      let line: u32 = params.lineOffset + lineLocal;
      let t: u32 = i - lineLocal * N;
      let base: u32 = line_base(line);
      let v: vec2<f32> = a[lineLocal * M + t];
      output[base + t * STRIDE] = c_mul(v, chirpC[t]);
    }
    `;
    }
    
    exports['generateBluesteinPreWGSL'] = generateBluesteinPreWGSL;
    exports['generateBluesteinMulBfftWGSL'] = generateBluesteinMulBfftWGSL;
    exports['generateBluesteinPostWGSL'] = generateBluesteinPostWGSL;
  });

  __define('src/kernels/conv2d.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    
    function generateConv2dRealWGSL({ Hin, Win, Hout, Wout, k, pad, workgroupSize }) {
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
    
    function generateConv2dComplexRealKernelWGSL({ Hin, Win, Hout, Wout, k, pad, workgroupSize }) {
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
    
    function generateConv2dComplexComplexKernelWGSL({ Hin, Win, Hout, Wout, k, pad, workgroupSize }) {
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
    
    
    exports['generateConv2dRealWGSL'] = generateConv2dRealWGSL;
    exports['generateConv2dComplexRealKernelWGSL'] = generateConv2dComplexRealKernelWGSL;
    exports['generateConv2dComplexComplexKernelWGSL'] = generateConv2dComplexComplexKernelWGSL;
  });

  __define('src/kernels/dct_fft.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    const { wgslLineBaseFn } = require('src/kernels/nd_line_base.js');
    
    function dimsToStride(axis, dims) {
      let s = 1;
      for (let i = 0; i < axis; i++) s *= dims[i];
      return s;
    }
    
    function dctWorkLength(typeKind, axisLength) {
      if (typeKind === "dct1") return 2 * (axisLength - 1);
      if (typeKind === "dst1") return 2 * (axisLength + 1);
      return 2 * axisLength;
    }
    
    function dctFftDirection(typeKind) {
      if (typeKind === "dct2_inv" || typeKind === "dst2_inv") return "inverse";
      return "forward";
    }
    
    function generateDctFftBuildWGSL({ typeKind, rank, axis, dims, axisLength, workgroupSize }) {
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
    
    function generateDctFftPostWGSL({ typeKind, rank, axis, dims, axisLength, workgroupSize }) {
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
    
    exports['dctWorkLength'] = dctWorkLength;
    exports['dctFftDirection'] = dctFftDirection;
    exports['generateDctFftBuildWGSL'] = generateDctFftBuildWGSL;
    exports['generateDctFftPostWGSL'] = generateDctFftPostWGSL;
  });

  __define('src/kernels/f16_storage.js', function(require, exports, module){
    function generateF16ToF32ComplexWGSL({ workgroupSize }) {
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
    
    function generateF32ToF16ComplexWGSL({ workgroupSize }) {
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
    
    function generateF16ToF32RealWGSL({ workgroupSize }) {
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
    
    function generateF32ToF16RealWGSL({ workgroupSize }) {
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
    
    exports['generateF16ToF32ComplexWGSL'] = generateF16ToF32ComplexWGSL;
    exports['generateF32ToF16ComplexWGSL'] = generateF32ToF16ComplexWGSL;
    exports['generateF16ToF32RealWGSL'] = generateF16ToF32RealWGSL;
    exports['generateF32ToF16RealWGSL'] = generateF32ToF16RealWGSL;
  });

  __define('src/kernels/fft_conv.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    
    function generatePointwiseMulWGSL({
      totalLogical,
      batch,
      correlate,
      workgroupSize,
    }) {
      const totalElems = totalLogical * batch;
      return /* wgsl */ `
    ${COMPLEX_WGSL}
    
    @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read> kernel: array<vec2<f32>>;
    
    const TOTAL_LOGICAL: u32 = ${totalLogical}u;
    const TOTAL_ELEMS: u32 = ${totalElems}u;
    const CORRELATE: bool = ${correlate ? "true" : "false"};
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= TOTAL_ELEMS) { return; }
      var k: vec2<f32> = kernel[i % TOTAL_LOGICAL];
      if (CORRELATE) {
        k = vec2<f32>(k.x, -k.y);
      }
    data[i] = c_mul(data[i], k);
    }
    `;
    }
    
    function generatePointwiseMulSegmentWGSL({
      correlate,
      workgroupSize,
    }) {
      return /* wgsl */ `
    ${COMPLEX_WGSL}
    
    struct Params {
      count: u32,
      dataBase: u32,
      kernelBase: u32,
      _pad0: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read> kernel: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const CORRELATE: bool = ${correlate ? "true" : "false"};
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count) { return; }
      let di: u32 = params.dataBase + i;
      let ki: u32 = params.kernelBase + i;
      var k: vec2<f32> = kernel[ki];
      if (CORRELATE) {
        k = vec2<f32>(k.x, -k.y);
      }
      data[di] = c_mul(data[di], k);
    }
    `;
    }
    
    exports['generatePointwiseMulWGSL'] = generatePointwiseMulWGSL;
    exports['generatePointwiseMulSegmentWGSL'] = generatePointwiseMulSegmentWGSL;
  });

  __define('src/kernels/ioview.js', function(require, exports, module){
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
    
    function generateEmbedComplexWGSL({ rank, logicalDims, viewDims, offset, workgroupSize }) {
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
    
    function generateEmbedComplexF16ToF32WGSL({ rank, logicalDims, viewDims, offset, workgroupSize }) {
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
    
    function generateExtractComplexWGSL({ rank, logicalDims, viewDims, offset, clearOutside, workgroupSize }) {
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
    
    function generateExtractComplexF32ToF16WGSL({ rank, logicalDims, viewDims, offset, clearOutside, workgroupSize }) {
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
    
    function generateEmbedRealWGSL({ rank, logicalDims, viewDims, offset, workgroupSize }) {
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
    
    function generateExtractRealWGSL({ rank, logicalDims, viewDims, offset, clearOutside, workgroupSize }) {
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
    
    exports['generateEmbedComplexWGSL'] = generateEmbedComplexWGSL;
    exports['generateEmbedComplexF16ToF32WGSL'] = generateEmbedComplexF16ToF32WGSL;
    exports['generateExtractComplexWGSL'] = generateExtractComplexWGSL;
    exports['generateExtractComplexF32ToF16WGSL'] = generateExtractComplexF32ToF16WGSL;
    exports['generateEmbedRealWGSL'] = generateEmbedRealWGSL;
    exports['generateExtractRealWGSL'] = generateExtractRealWGSL;
  });

  __define('src/kernels/nd_line_base.js', function(require, exports, module){
    function wgslLineBaseFn({ rank, axis, dims }) {
      if (!Number.isInteger(rank) || rank < 1) throw new Error(`rank must be >= 1, got ${rank}`);
      if (!Array.isArray(dims) || dims.length !== rank) throw new Error(`dims length (${dims?.length}) must match rank (${rank})`);
      if (!Number.isInteger(axis) || axis < 0 || axis >= rank) throw new Error(`axis=${axis} out of range for rank=${rank}`);
    
      const nTotal = dims.reduce((a, b) => a * b, 1);
      const linesPerBatch = dims.reduce((a, d, i) => (i === axis ? a : a * d), 1);
    
      const strides = new Array(rank);
      strides[0] = 1;
      for (let i = 1; i < rank; i++) strides[i] = strides[i - 1] * dims[i - 1];
    
      let decode = "";
      let remName = "rem";
      let remInit = true;
      for (let d = 0; d < rank; d++) {
        if (d === axis) continue;
        const dim = dims[d];
        const stride = strides[d];
        const coordName = `c${d}`;
        const nextRem = `${remName}_${d}`;
        if (remInit) {
          decode += `  var ${remName}: u32 = line - b * lines_per_batch;\n`;
          remInit = false;
        }
        decode += `  let ${coordName}: u32 = ${remName} % ${dim}u;\n`;
        decode += `  base = base + ${coordName} * ${stride}u;\n`;
        decode += `  var ${nextRem}: u32 = ${remName} / ${dim}u;\n`;
        remName = nextRem;
      }
    
      if (decode.length === 0) {
        decode = "  // axis-only line (rank=1): no non-axis coordinates\n";
      }
    
      return /* wgsl */ `
    fn line_base(line: u32) -> u32 {
      let lines_per_batch: u32 = ${linesPerBatch}u;
      let b: u32 = line / lines_per_batch;
      var base: u32 = b * ${nTotal}u;
    ${decode}  return base;
    }
    `;
    }
    
    exports['wgslLineBaseFn'] = wgslLineBaseFn;
  });

  __define('src/kernels/rader.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    const { wgslLineBaseFn } = require('src/kernels/nd_line_base.js');
    
    function generateRaderSumWGSL({ rank, axis, dims, axisLength, strideComplex, workgroupSize }) {
      const N = axisLength;
      const lineBase = wgslLineBaseFn({ rank, axis, dims });
      return /* wgsl */ `
    struct Params {
      lines: u32,
      lineOffset: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
      @group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> sumAll: array<vec2<f32>>;
    @group(0) @binding(2) var<storage, read_write> x0: array<vec2<f32>>;
    @group(0) @binding(3) var<uniform> params: Params;
    
    ${COMPLEX_WGSL}
    const N: u32 = ${N}u;
    const STRIDE: u32 = ${strideComplex}u;
    
    ${lineBase}
    
    var<workgroup> scratch: array<vec2<f32>, ${workgroupSize}>;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
      let lineLocal: u32 = wid.x;
      if (lineLocal >= params.lines) { return; }
      let line: u32 = params.lineOffset + lineLocal;
      let base: u32 = line_base(line);
    
      // parallel reduction across N items
      var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
      var i: u32 = lid.x;
      while (i < N) {
        acc = c_add(acc, input[base + i * STRIDE]);
        i = i + ${workgroupSize}u;
      }
      scratch[lid.x] = acc;
      workgroupBarrier();
    
      var stride: u32 = ${workgroupSize}u / 2u;
      loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
          scratch[lid.x] = c_add(scratch[lid.x], scratch[lid.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
      }
    
      if (lid.x == 0u) {
        sumAll[lineLocal] = scratch[0];
        x0[lineLocal] = input[base + 0u];
      }
    }
    `;
    }
    
    function generateRaderPackARevWGSL({
      rank,
      axis,
      dims,
      axisLength,
      mLength,
      strideComplex,
      workgroupSize,
    }) {
      const N = axisLength;
      const M = mLength;
      const lineBase = wgslLineBaseFn({ rank, axis, dims });
      return /* wgsl */ `
    struct Params {
      lines: u32,
      lineOffset: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
      @group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> a: array<vec2<f32>>;
    @group(0) @binding(2) var<storage, read> perm: array<u32>;
    @group(0) @binding(3) var<uniform> params: Params;
    
    ${COMPLEX_WGSL}
    const N: u32 = ${N}u;
    const M: u32 = ${M}u;
    const STRIDE: u32 = ${strideComplex}u;
    
    ${lineBase}
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.lines * M;
      if (i >= total) { return; }
      let lineLocal: u32 = i / M;
      let line: u32 = params.lineOffset + lineLocal;
      let t: u32 = i - lineLocal * M;
      if (t >= (N - 1u)) {
        a[i] = vec2<f32>(0.0, 0.0);
        return;
      }
      let base: u32 = line_base(line);
      // a_rev[t] = x[perm[(N-2)-t]]
      let idx: u32 = perm[(N - 2u) - t];
      a[i] = input[base + idx * STRIDE];
    }
    `;
    }
    
    function generateRaderMulBfftWGSL({ mLength, workgroupSize }) {
      const M = mLength;
      return /* wgsl */ `
    struct Params {
      total: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> a: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read> bfft: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    ${COMPLEX_WGSL}
    const M: u32 = ${M}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.total) { return; }
      let k: u32 = i % M;
      a[i] = c_mul(a[i], bfft[k]);
    }
    `;
    }
    
    function generateRaderWriteY0WGSL({ rank, axis, dims, axisLength, strideComplex, workgroupSize }) {
      const lineBase = wgslLineBaseFn({ rank, axis, dims });
      const STRIDE = strideComplex;
      const N = axisLength;
      return /* wgsl */ `
    struct Params {
      lines: u32,
      lineOffset: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read> sumAll: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const STRIDE: u32 = ${STRIDE}u;
    ${lineBase}
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let lineLocal: u32 = gid.x;
      if (lineLocal >= params.lines) { return; }
      let line: u32 = params.lineOffset + lineLocal;
      let base: u32 = line_base(line);
      output[base + 0u] = sumAll[lineLocal];
    }
    `;
    }
    
    function generateRaderPostWGSL({
      rank,
      axis,
      dims,
      axisLength,
      mLength,
      strideComplex,
      workgroupSize,
    }) {
      const N = axisLength;
      const M = mLength;
      const STRIDE = strideComplex;
      const lineBase = wgslLineBaseFn({ rank, axis, dims });
      return /* wgsl */ `
    struct Params {
      lines: u32,
      lineOffset: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
      @group(0) @binding(0) var<storage, read_write> conv: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read> x0: array<vec2<f32>>;
    @group(0) @binding(2) var<storage, read> perm: array<u32>;
    @group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;
    @group(0) @binding(4) var<uniform> params: Params;
    
    ${COMPLEX_WGSL}
    const N: u32 = ${N}u;
    const M: u32 = ${M}u;
    const STRIDE: u32 = ${STRIDE}u;
    
    ${lineBase}
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.lines * (N - 1u);
      if (i >= total) { return; }
      let lineLocal: u32 = i / (N - 1u);
      let line: u32 = params.lineOffset + lineLocal;
      let t: u32 = i - lineLocal * (N - 1u); // 0..N-2
      let base: u32 = line_base(line);
    
      // Wrap linear convolution into cyclic length (N-1):
      var v: vec2<f32> = conv[lineLocal * M + t];
      let wrapIdx: u32 = t + (N - 1u);
      if (wrapIdx < M) {
        v = c_add(v, conv[lineLocal * M + wrapIdx]);
      }
    
      let outIdx: u32 = perm[t];
      output[base + outIdx * STRIDE] = c_add(x0[lineLocal], v);
    }
    `;
    }
    
    exports['generateRaderSumWGSL'] = generateRaderSumWGSL;
    exports['generateRaderPackARevWGSL'] = generateRaderPackARevWGSL;
    exports['generateRaderMulBfftWGSL'] = generateRaderMulBfftWGSL;
    exports['generateRaderWriteY0WGSL'] = generateRaderWriteY0WGSL;
    exports['generateRaderPostWGSL'] = generateRaderPostWGSL;
  });

  __define('src/kernels/real_complex.js', function(require, exports, module){
    function generateRealToComplexWGSL({ totalReal, workgroupSize }) {
      return /* wgsl */ `
    struct Params {
      total: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.total) { return; }
      output[i] = vec2<f32>(input[i], 0.0);
    }
    `;
    }
    
    function generateComplexToRealWGSL({ workgroupSize }) {
      return /* wgsl */ `
    struct Params {
      total: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.total) { return; }
      output[i] = input[i].x;
    }
    `;
    }
    
    function prod(arr) {
      return arr.reduce((a, b) => a * b, 1);
    }
    
    function rowMajorStrides(dims) {
      const out = new Array(dims.length);
      out[0] = 1;
      for (let i = 1; i < dims.length; i++) out[i] = out[i - 1] * dims[i - 1];
      return out;
    }
    
    function decodeCoordsWGSL({ indexName, dims, coordPrefix }) {
      let rem = indexName;
      const coords = [];
      let code = "";
      for (let d = 0; d < dims.length; d++) {
        const c = `${coordPrefix}${d}`;
        coords.push(c);
        code += `  let ${c}: u32 = ${rem} % ${dims[d]}u;\n`;
        if (d < dims.length - 1) {
          const nextRem = `${coordPrefix}rem${d}`;
          code += `  let ${nextRem}: u32 = ${rem} / ${dims[d]}u;\n`;
          rem = nextRem;
        }
      }
      return { code, coords };
    }
    
    function generatePackR2CWGSL({ shape, workgroupSize }) {
      const rank = shape.length;
      const inDims = shape.slice();
      const outDims = [((shape[0] >>> 1) + 1), ...shape.slice(1)];
      const inStrides = rowMajorStrides(inDims);
      const inTotal = prod(inDims);
      const outTotal = prod(outDims);
      const decoded = decodeCoordsWGSL({ indexName: "rem", dims: outDims, coordPrefix: "c" });
    
      let inIndexBody = `  var inIndex: u32 = b * ${inTotal}u;\n`;
      for (let d = 0; d < rank; d++) {
        if (inStrides[d] === 1) inIndexBody += `  inIndex = inIndex + ${decoded.coords[d]};\n`;
        else inIndexBody += `  inIndex = inIndex + ${decoded.coords[d]} * ${inStrides[d]}u;\n`;
      }
    
      return /* wgsl */ `
    struct Params {
      batch: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const OUT_TOTAL_PER_BATCH: u32 = ${outTotal}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let totalOut: u32 = OUT_TOTAL_PER_BATCH * params.batch;
      if (i >= totalOut) { return; }
      let b: u32 = i / OUT_TOTAL_PER_BATCH;
      let rem: u32 = i - b * OUT_TOTAL_PER_BATCH;
    ${decoded.code}
    ${inIndexBody}
      output[i] = input[inIndex];
    }
    `;
    }
    
    function generateUnpackC2RWGSL({ shape, workgroupSize }) {
      const rank = shape.length;
      const fullDims = shape.slice();
      const inDims = [((shape[0] >>> 1) + 1), ...shape.slice(1)];
      const inStrides = rowMajorStrides(inDims);
      const fullTotal = prod(fullDims);
      const inTotal = prod(inDims);
      const Nx = shape[0];
      const inNx = inDims[0];
      const even = Nx % 2 === 0;
      const decoded = decodeCoordsWGSL({ indexName: "rem", dims: fullDims, coordPrefix: "c" });
    
      let mirrorCoordsCode = "";
      const coordForInIndex = new Array(rank);
      coordForInIndex[0] = "xPacked";
      for (let d = 1; d < rank; d++) {
        const cd = decoded.coords[d];
        const cm = `c${d}m`;
        const cp = `c${d}p`;
        mirrorCoordsCode += `  let ${cm}: u32 = select(0u, ${fullDims[d]}u - ${cd}, ${cd} != 0u);\n`;
        mirrorCoordsCode += `  let ${cp}: u32 = select(${cd}, ${cm}, x >= IN_NX);\n`;
        coordForInIndex[d] = cp;
      }
    
      let inIndexBody = `  var inIndex: u32 = b * ${inTotal}u;\n`;
      for (let d = 0; d < rank; d++) {
        const coord = coordForInIndex[d];
        if (inStrides[d] === 1) inIndexBody += `  inIndex = inIndex + ${coord};\n`;
        else inIndexBody += `  inIndex = inIndex + ${coord} * ${inStrides[d]}u;\n`;
      }
    
      let selfConjExpr = "(x == 0u || (EVEN_NX && x == (NX / 2u)))";
      for (let d = 1; d < rank; d++) {
        const cd = decoded.coords[d];
        if (fullDims[d] % 2 === 0) {
          selfConjExpr += ` && (${cd} == 0u || ${cd} == ${fullDims[d] / 2}u)`;
        } else {
          selfConjExpr += ` && (${cd} == 0u)`;
        }
      }
    
      return /* wgsl */ `
    struct Params {
      batch: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const NX: u32 = ${Nx}u;
    const IN_NX: u32 = ${inNx}u;
    const EVEN_NX: bool = ${even ? "true" : "false"};
    const OUT_TOTAL_PER_BATCH: u32 = ${fullTotal}u;
    
    fn conj(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(v.x, -v.y); }
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let totalOut: u32 = OUT_TOTAL_PER_BATCH * params.batch;
      if (i >= totalOut) { return; }
      let b: u32 = i / OUT_TOTAL_PER_BATCH;
      let rem: u32 = i - b * OUT_TOTAL_PER_BATCH;
    ${decoded.code}
    
      // Map full spectrum X[x] from packed [0..N/2].
      var v: vec2<f32>;
      let x: u32 = ${decoded.coords[0]};
      let xPacked: u32 = select(x, NX - x, x >= IN_NX);
    ${mirrorCoordsCode}
    ${inIndexBody}
      v = input[inIndex];
      if (x >= IN_NX) { v = conj(v); }
    
      // Only globally self-conjugate bins are guaranteed real.
      if (${selfConjExpr}) {
        v = vec2<f32>(v.x, 0.0);
      }
      output[i] = v;
    }
    `;
    }
    
    exports['generateRealToComplexWGSL'] = generateRealToComplexWGSL;
    exports['generateComplexToRealWGSL'] = generateComplexToRealWGSL;
    exports['generatePackR2CWGSL'] = generatePackR2CWGSL;
    exports['generateUnpackC2RWGSL'] = generateUnpackC2RWGSL;
  });

  __define('src/kernels/scale.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    
    function generateScaleComplexWGSL({ workgroupSize }) {
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
    
    function generateScaleRealWGSL({ workgroupSize }) {
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
    
    exports['generateScaleComplexWGSL'] = generateScaleComplexWGSL;
    exports['generateScaleRealWGSL'] = generateScaleRealWGSL;
  });

  __define('src/kernels/segmented_copy.js', function(require, exports, module){
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
    
    function generateSegmentedCopyWGSL({ cap, direction, workgroupSize }) {
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
    
    
    exports['generateSegmentedCopyWGSL'] = generateSegmentedCopyWGSL;
  });

  __define('src/kernels/stockham1d.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    const { wgslLineBaseFn } = require('src/kernels/nd_line_base.js');
    
    /**
     * Generates a single Stockham radix-2 stage kernel for one axis.
     * The kernel processes all lines for the axis and writes one output element per invocation.
     */
    function generateStockhamStageWGSL({
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
    
    exports['generateStockhamStageWGSL'] = generateStockhamStageWGSL;
  });

  __define('src/kernels/stockham_stage.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    const { wgslLineBaseFn } = require('src/kernels/nd_line_base.js');
    
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
    function generateStockhamRadixStageWGSL({
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
    
    exports['generateStockhamRadixStageWGSL'] = generateStockhamRadixStageWGSL;
  });

  __define('src/kernels/strided_complex.js', function(require, exports, module){
    function decodeCoordsWgsl(dims, indexName = "li", coordPrefix = "c") {
      let code = `  var rem: u32 = ${indexName};\n`;
      const coords = [];
      for (let d = 0; d < dims.length; d++) {
        const c = `${coordPrefix}${d}`;
        coords.push(c);
        code += `  let ${c}: u32 = rem % ${dims[d]}u;\n`;
        if (d < dims.length - 1) code += `  rem = rem / ${dims[d]}u;\n`;
      }
      return { code, coords };
    }
    
    function physIndexExpr({ baseOffsetElements, batchStrideElements, strides, coords }) {
      let expr = `${baseOffsetElements}u + params.extraOffsetElements + b * ${batchStrideElements}u`;
      for (let d = 0; d < strides.length; d++) {
        if (strides[d] === 1) expr += ` + ${coords[d]}`;
        else expr += ` + ${coords[d]} * ${strides[d]}u`;
      }
      return expr;
    }
    
    function generateGatherComplexStridedWGSL({
      shape,
      strides,
      baseOffsetElements,
      batchStrideElements,
      workgroupSize,
    }) {
      const nTotal = shape.reduce((a, b) => a * b, 1);
      const decoded = decodeCoordsWgsl(shape, "li", "c");
      const physExpr = physIndexExpr({
        baseOffsetElements,
        batchStrideElements,
        strides,
        coords: decoded.coords,
      });
      return /* wgsl */ `
    struct Params {
      totalLogical: u32,
      batch: u32,
      extraOffsetElements: u32,
      _pad0: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const TOTAL_LOGICAL: u32 = ${nTotal}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.totalLogical * params.batch;
      if (i >= total) { return; }
      let b: u32 = i / params.totalLogical;
      let li: u32 = i - b * params.totalLogical;
    ${decoded.code}
      let pi: u32 = ${physExpr};
      dst[i] = src[pi];
    }
    `;
    }
    
    function generateScatterComplexStridedWGSL({
      shape,
      strides,
      baseOffsetElements,
      batchStrideElements,
      workgroupSize,
    }) {
      const nTotal = shape.reduce((a, b) => a * b, 1);
      const decoded = decodeCoordsWgsl(shape, "li", "c");
      const physExpr = physIndexExpr({
        baseOffsetElements,
        batchStrideElements,
        strides,
        coords: decoded.coords,
      });
      return /* wgsl */ `
    struct Params {
      totalLogical: u32,
      batch: u32,
      extraOffsetElements: u32,
      _pad0: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const TOTAL_LOGICAL: u32 = ${nTotal}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.totalLogical * params.batch;
      if (i >= total) { return; }
      let b: u32 = i / params.totalLogical;
      let li: u32 = i - b * params.totalLogical;
    ${decoded.code}
      let pi: u32 = ${physExpr};
      dst[pi] = src[i];
    }
    `;
    }
    
    exports['generateGatherComplexStridedWGSL'] = generateGatherComplexStridedWGSL;
    exports['generateScatterComplexStridedWGSL'] = generateScatterComplexStridedWGSL;
  });

  __define('src/kernels/strided_real.js', function(require, exports, module){
    function decodeCoordsWgsl(dims, indexName = "li", coordPrefix = "c") {
      let code = `  var rem: u32 = ${indexName};\n`;
      const coords = [];
      for (let d = 0; d < dims.length; d++) {
        const c = `${coordPrefix}${d}`;
        coords.push(c);
        code += `  let ${c}: u32 = rem % ${dims[d]}u;\n`;
        if (d < dims.length - 1) code += `  rem = rem / ${dims[d]}u;\n`;
      }
      return { code, coords };
    }
    
    function physIndexExpr({ baseOffsetElements, batchStrideElements, strides, coords }) {
      let expr = `${baseOffsetElements}u + params.extraOffsetElements + b * ${batchStrideElements}u`;
      for (let d = 0; d < strides.length; d++) {
        if (strides[d] === 1) expr += ` + ${coords[d]}`;
        else expr += ` + ${coords[d]} * ${strides[d]}u`;
      }
      return expr;
    }
    
    function generateGatherRealStridedWGSL({
      shape,
      strides,
      baseOffsetElements,
      batchStrideElements,
      workgroupSize,
    }) {
      const nTotal = shape.reduce((a, b) => a * b, 1);
      const decoded = decodeCoordsWgsl(shape, "li", "c");
      const physExpr = physIndexExpr({
        baseOffsetElements,
        batchStrideElements,
        strides,
        coords: decoded.coords,
      });
      return /* wgsl */ `
    struct Params {
      totalLogical: u32,
      batch: u32,
      extraOffsetElements: u32,
      _pad0: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<f32>;
    @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const TOTAL_LOGICAL: u32 = ${nTotal}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.totalLogical * params.batch;
      if (i >= total) { return; }
      let b: u32 = i / params.totalLogical;
      let li: u32 = i - b * params.totalLogical;
    ${decoded.code}
      let pi: u32 = ${physExpr};
      dst[i] = src[pi];
    }
    `;
    }
    
    function generateScatterRealStridedWGSL({
      shape,
      strides,
      baseOffsetElements,
      batchStrideElements,
      workgroupSize,
    }) {
      const nTotal = shape.reduce((a, b) => a * b, 1);
      const decoded = decodeCoordsWgsl(shape, "li", "c");
      const physExpr = physIndexExpr({
        baseOffsetElements,
        batchStrideElements,
        strides,
        coords: decoded.coords,
      });
      return /* wgsl */ `
    struct Params {
      totalLogical: u32,
      batch: u32,
      extraOffsetElements: u32,
      _pad0: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<f32>;
    @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const TOTAL_LOGICAL: u32 = ${nTotal}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.totalLogical * params.batch;
      if (i >= total) { return; }
      let b: u32 = i / params.totalLogical;
      let li: u32 = i - b * params.totalLogical;
    ${decoded.code}
      let pi: u32 = ${physExpr};
      dst[pi] = src[i];
    }
    `;
    }
    
    exports['generateGatherRealStridedWGSL'] = generateGatherRealStridedWGSL;
    exports['generateScatterRealStridedWGSL'] = generateScatterRealStridedWGSL;
  });

  __define('src/kernels/subgroup_pow2_fft.js', function(require, exports, module){
    const { COMPLEX_WGSL } = require('src/kernels/utils_wgsl.js');
    const { wgslLineBaseFn } = require('src/kernels/nd_line_base.js');
    
    function isPowerOfTwo(n) {
      return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
    }
    
    function generateSubgroupPow2FftWGSL({
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
    
    exports['generateSubgroupPow2FftWGSL'] = generateSubgroupPow2FftWGSL;
  });

  __define('src/kernels/transpose.js', function(require, exports, module){
    function generateTransposeComplex2DWGSL({ Nx, Ny, tile = 16 }) {
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
    
    
    exports['generateTransposeComplex2DWGSL'] = generateTransposeComplex2DWGSL;
  });

  __define('src/kernels/utils_wgsl.js', function(require, exports, module){
    const COMPLEX_WGSL = /* wgsl */ `
    const PI: f32 = 3.1415926535897932384626433832795;
    
    fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
      return a + b;
    }
    
    fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
      return a - b;
    }
    
    fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
      return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
      );
    }
    
    fn cis(angle: f32) -> vec2<f32> {
      return vec2<f32>(cos(angle), sin(angle));
    }
    `;
    
    
    exports['COMPLEX_WGSL'] = COMPLEX_WGSL;
  });

  __define('src/kernels/zero_pad.js', function(require, exports, module){
    function decodeCoordsWgsl(dims, indexName = "li", coordPrefix = "c") {
      let code = `  var rem: u32 = ${indexName};\n`;
      const coords = [];
      for (let d = 0; d < dims.length; d++) {
        const c = `${coordPrefix}${d}`;
        coords.push(c);
        code += `  let ${c}: u32 = rem % ${dims[d]}u;\n`;
        if (d < dims.length - 1) code += `  rem = rem / ${dims[d]}u;\n`;
      }
      return { code, coords };
    }
    
    function insideRangeExpr(start, end, coords) {
      const terms = [];
      for (let d = 0; d < coords.length; d++) {
        terms.push(`${coords[d]} >= ${start[d]}u && ${coords[d]} < ${end[d]}u`);
      }
      return terms.join(" && ");
    }
    
    function generateZeroOutsideRangeComplexWGSL({
      shape,
      start,
      end,
      batch,
      workgroupSize,
    }) {
      const nTotal = shape.reduce((a, b) => a * b, 1);
      const totalElems = nTotal * batch;
      const decoded = decodeCoordsWgsl(shape, "li", "c");
      const insideExpr = insideRangeExpr(start, end, decoded.coords);
      return /* wgsl */ `
    @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
    
    const TOTAL_LOGICAL: u32 = ${nTotal}u;
    const TOTAL_ELEMS: u32 = ${totalElems}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= TOTAL_ELEMS) { return; }
      let li: u32 = i % TOTAL_LOGICAL;
    ${decoded.code}
      if (!(${insideExpr})) {
        data[i] = vec2<f32>(0.0, 0.0);
      }
    }
    `;
    }
    
    function generateZeroOutsideRangeRealWGSL({
      shape,
      start,
      end,
      batch,
      workgroupSize,
    }) {
      const nTotal = shape.reduce((a, b) => a * b, 1);
      const totalElems = nTotal * batch;
      const decoded = decodeCoordsWgsl(shape, "li", "c");
      const insideExpr = insideRangeExpr(start, end, decoded.coords);
      return /* wgsl */ `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    
    const TOTAL_LOGICAL: u32 = ${nTotal}u;
    const TOTAL_ELEMS: u32 = ${totalElems}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= TOTAL_ELEMS) { return; }
      let li: u32 = i % TOTAL_LOGICAL;
    ${decoded.code}
      if (!(${insideExpr})) {
        data[i] = 0.0;
      }
    }
    `;
    }
    
    
    exports['generateZeroOutsideRangeComplexWGSL'] = generateZeroOutsideRangeComplexWGSL;
    exports['generateZeroOutsideRangeRealWGSL'] = generateZeroOutsideRangeRealWGSL;
  });

  __define('src/plan.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { generateStockhamRadixStageWGSL } = require('src/kernels/stockham_stage.js');
    const { generateSubgroupPow2FftWGSL } = require('src/kernels/subgroup_pow2_fft.js');
    
    function assertOneOf(value, allowed, name) {
      if (!allowed.includes(value)) {
        throw new Error(`${name} must be one of ${allowed.map((v) => JSON.stringify(v)).join(", ")}; got ${JSON.stringify(value)}`);
      }
    }
    
    function isPositiveInt(x) {
      return Number.isInteger(x) && x > 0;
    }
    
    function isPowerOfTwo(n) {
      return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
    }
    
    function factorizeRadices(n) {
      // Greedy decomposition for current mixed-radix kernels.
      // NOTE: For best performance this could be a cost-based search; keep simple for now.
      const allowed = [13, 11, 8, 7, 5, 4, 3, 2];
      const out = [];
      let x = n;
      for (const r of allowed) {
        while (x % r === 0) {
          out.push(r);
          x = x / r;
        }
      }
      return x === 1 ? out : null;
    }
    
    function product(arr) {
      let p = 1;
      for (const v of arr) p *= v;
      return p;
    }
    
    function strideForAxis(dims, axis) {
      let s = 1;
      for (let i = 0; i < axis; i++) s *= dims[i];
      return s;
    }
    
    function formatLimits(limits) {
      const m = limits?.maxComputeWorkgroupsPerDimension;
      return JSON.stringify(
        {
          maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize,
          maxBufferSize: limits?.maxBufferSize,
          maxComputeWorkgroupSizeX: limits?.maxComputeWorkgroupSizeX,
          maxComputeWorkgroupSizeY: limits?.maxComputeWorkgroupSizeY,
          maxComputeWorkgroupSizeZ: limits?.maxComputeWorkgroupSizeZ,
          maxComputeInvocationsPerWorkgroup: limits?.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupStorageSize: limits?.maxComputeWorkgroupStorageSize,
          minStorageBufferOffsetAlignment: limits?.minStorageBufferOffsetAlignment,
          maxComputeWorkgroupsPerDimension: m ? [m[0], m[1], m[2]] : undefined,
        },
        null,
        2
      );
    }
    
    function normalizeScaleFactor({ normalize, direction, nTotal }) {
      if (normalize === "none") return 1.0;
      if (normalize === "unitary") return 1.0 / Math.sqrt(nTotal);
      if (normalize === "backward") return direction === "inverse" ? 1.0 / nTotal : 1.0;
      throw new Error(`Unknown normalize mode: ${normalize}`);
    }
    
    function chooseAxis0TwoStepFactors(axisLen, maxAxisElems) {
      if (!Number.isInteger(maxAxisElems) || maxAxisElems < 2) return null;
      let best = null;
      const consider = (n1, n2) => {
        if (!Number.isInteger(n1) || !Number.isInteger(n2)) return;
        if (n1 < 2 || n2 < 2) return;
        if (n1 > maxAxisElems || n2 > maxAxisElems) return;
        if (!factorizeRadices(n1) || !factorizeRadices(n2)) return;
        const score = Math.max(n1, n2);
        const balance = Math.abs(n1 - n2);
        if (!best || score < best.score || (score === best.score && balance < best.balance)) {
          best = { n1, n2, score, balance };
        }
      };
      const root = Math.floor(Math.sqrt(axisLen));
      for (let d = 1; d <= root; d++) {
        if (axisLen % d !== 0) continue;
        const q = axisLen / d;
        consider(d, q);
        consider(q, d);
      }
      if (!best) return null;
      return { n1: best.n1, n2: best.n2 };
    }
    
    function generateAxis0TwoStepTwiddleWGSL({ n1, n, sign, workgroupSize }) {
      return /* wgsl */ `
    struct Params {
      total: u32,
      baseIndex: u32,
      lineOffset: u32,
      _pad: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
    @group(0) @binding(1) var<uniform> params: Params;
    
    const PI: f32 = 3.14159265358979323846;
    const N1: u32 = ${n1}u;
    const N: u32 = ${n}u;
    const SIGN: f32 = ${sign};
    
    fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
      return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    
    fn cis(theta: f32) -> vec2<f32> {
      return vec2<f32>(cos(theta), sin(theta));
    }
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx: u32 = params.baseIndex + gid.x;
      if (idx >= params.total) {
        return;
      }
      let lineLocal: u32 = idx / N1;
      let k1: u32 = idx - lineLocal * N1;
      let m2: u32 = params.lineOffset + lineLocal;
      let angle: f32 = SIGN * (2.0 * PI) * (f32(k1) * f32(m2)) / f32(N);
      let w: vec2<f32> = cis(angle);
      data[idx] = c_mul(data[idx], w);
    }
    `;
    }
    
    function generateAxis0TwoStepScaleWGSL({ scaleFactor, workgroupSize }) {
      return /* wgsl */ `
    struct Params {
      total: u32,
      baseIndex: u32,
      _pad0: u32,
      _pad1: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
    @group(0) @binding(1) var<uniform> params: Params;
    
    const SCALE: f32 = ${scaleFactor};
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx: u32 = params.baseIndex + gid.x;
      if (idx >= params.total) {
        return;
      }
      data[idx] = data[idx] * vec2<f32>(SCALE, SCALE);
    }
    `;
    }
    
    function pickWorkgroupSizeX(limits) {
      const maxX = limits?.maxComputeWorkgroupSizeX ?? 256;
      const maxInvocations = limits?.maxComputeInvocationsPerWorkgroup ?? 256;
      return Math.max(1, Math.min(256, maxX, maxInvocations));
    }
    
    function isGpuBufferLike(x) {
      return !!x && !x?.segments && typeof x?.size === "number" && typeof x?.destroy === "function";
    }
    
    class FftPlan {
      constructor(device, config, compiled) {
        this.device = device;
        this.config = config;
        this._axesList = Array.isArray(compiled.axesList) ? compiled.axesList.slice() : [];
        this._pipelines = compiled.pipelines;
        this._workgroupSizeX = compiled.workgroupSizeX;
        this._paramsBuffer = compiled.paramsBuffer;
        this._bindGroupLayout = compiled.bindGroupLayout;
        this._paramsUpload = null;
        this._paramsUploadBytes = 0;
        this._scratch = null;
        this._scratchBytes = 0;
        this._packIn = null;
        this._packOut = null;
        this._packBytes = 0;
        this._axis0TwoStep = compiled.axis0TwoStep ?? null;
        this._twoStepLineA = null;
        this._twoStepLineB = null;
        this._twoStepLineBytes = 0;
        this._axis0PointwiseParams = null;
        this._destroyed = false;
    
        if (this._axis0TwoStep) {
          const { n1, n2 } = this._axis0TwoStep;
          this._axis0TwoStep.stage1 = createFftPlan(this.device, {
            shape: [n1, n2],
            direction: this.config.direction,
            normalize: "none",
            inPlace: true,
            layout: "interleaved",
            precision: "f32",
            axes: [0],
          });
          this._axis0TwoStep.stage2 = createFftPlan(this.device, {
            shape: [n2, n1],
            direction: this.config.direction,
            normalize: "none",
            inPlace: true,
            layout: "interleaved",
            precision: "f32",
            axes: [0],
          });
          const bgl = this.device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = this.device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const twiddleCode = generateAxis0TwoStepTwiddleWGSL({
            n1,
            n: n1 * n2,
            sign: this.config.direction === "forward" ? -1.0 : 1.0,
            workgroupSize: this._workgroupSizeX,
          });
          const twiddlePipeline = this.device.createComputePipeline({
            layout: pl,
            compute: { module: this.device.createShaderModule({ code: twiddleCode }), entryPoint: "main" },
          });
          let scalePipeline = null;
          if (this._axis0TwoStep.applyScale) {
            const scaleCode = generateAxis0TwoStepScaleWGSL({
              scaleFactor: this._axis0TwoStep.scaleFactor,
              workgroupSize: this._workgroupSizeX,
            });
            scalePipeline = this.device.createComputePipeline({
              layout: pl,
              compute: { module: this.device.createShaderModule({ code: scaleCode }), entryPoint: "main" },
            });
          }
          this._axis0TwoStep.pointwise = { bgl, pl, twiddlePipeline, scalePipeline };
          this._axis0PointwiseParams = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
        }
      }
    
      destroy() {
        if (this._destroyed) return;
        this._destroyed = true;
        if (this._scratch) this._scratch.destroy();
        this._scratch = null;
        this._scratchBytes = 0;
        if (this._packIn) this._packIn.destroy();
        if (this._packOut) this._packOut.destroy();
        this._packIn = null;
        this._packOut = null;
        this._packBytes = 0;
        if (this._paramsUpload) this._paramsUpload.destroy();
        this._paramsUpload = null;
        this._paramsUploadBytes = 0;
        if (this._axis0PointwiseParams) this._axis0PointwiseParams.destroy();
        this._axis0PointwiseParams = null;
        if (this._twoStepLineA) this._twoStepLineA.destroy();
        if (this._twoStepLineB) this._twoStepLineB.destroy();
        this._twoStepLineA = null;
        this._twoStepLineB = null;
        this._twoStepLineBytes = 0;
        if (this._axis0TwoStep?.stage1) this._axis0TwoStep.stage1.destroy();
        if (this._axis0TwoStep?.stage2) this._axis0TwoStep.stage2.destroy();
        this._axis0TwoStep = null;
      }
    
      _ensureScratch(bytes) {
        if (this._scratch && this._scratchBytes >= bytes) return;
        if (this._scratch) this._scratch.destroy();
        this._scratch = this.device.createBuffer({
          size: bytes,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
        });
        this._scratchBytes = bytes;
      }
    
      _ensurePack(bytes) {
        if (this._packIn && this._packBytes >= bytes) return;
        if (this._packIn) this._packIn.destroy();
        if (this._packOut) this._packOut.destroy();
        this._packIn = this.device.createBuffer({
          size: bytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._packOut = this.device.createBuffer({
          size: bytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._packBytes = bytes;
      }
    
      _ensureTwoStepLineBuffers(bytes) {
        if (this._twoStepLineA && this._twoStepLineB && this._twoStepLineBytes >= bytes) return;
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        if (bytes > maxBufferSize) {
          throw new Error(
            `Axis-0 two-step fallback requires ${bytes} bytes line staging, exceeding device.limits.maxBufferSize=${maxBufferSize}`
          );
        }
        if (this._twoStepLineA) this._twoStepLineA.destroy();
        if (this._twoStepLineB) this._twoStepLineB.destroy();
        this._twoStepLineA = this.device.createBuffer({
          size: bytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._twoStepLineB = this.device.createBuffer({
          size: bytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._twoStepLineBytes = bytes;
      }
    
      _ensureParamsUpload(bytes) {
        if (this._paramsUpload && this._paramsUploadBytes >= bytes) return;
        if (this._paramsUpload) this._paramsUpload.destroy();
        this._paramsUpload = this.device.createBuffer({
          size: bytes,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._paramsUploadBytes = bytes;
      }
    
      _buildDispatchChunks(totalComplex, shape, batch) {
        const workgroupsX = Math.ceil(totalComplex / this._workgroupSizeX);
        const maxWgXRaw = this.device.limits?.maxComputeWorkgroupsPerDimension?.[0];
        const maxWgX = Number.isFinite(maxWgXRaw) ? Math.floor(maxWgXRaw) : null;
        if (maxWgX != null && maxWgX < 1) {
          throw new Error(
            [
              `Invalid device limit: maxComputeWorkgroupsPerDimension[0]=${maxWgXRaw}`,
              `shape=${JSON.stringify(shape)} batch=${batch} totalComplex=${totalComplex} workgroupSizeX=${this._workgroupSizeX} workgroupsX=${workgroupsX}`,
            ].join("\n")
          );
        }
        const maxChunkWgX = maxWgX == null ? workgroupsX : Math.min(workgroupsX, maxWgX);
        const dispatchChunks = [];
        for (let wgStart = 0; wgStart < workgroupsX; wgStart += maxChunkWgX) {
          const wgCount = Math.min(maxChunkWgX, workgroupsX - wgStart);
          dispatchChunks.push({ wgCount, baseIndex: wgStart * this._workgroupSizeX });
        }
        return { workgroupsX, dispatchChunks };
      }
    
      _transposeLineMatrixCopies(commandEncoder, { srcBuffer, dstBuffer, nx, ny }) {
        const elemBytes = 8;
        for (let y = 0; y < ny; y++) {
          for (let x = 0; x < nx; x++) {
            const src = (y * nx + x) * elemBytes;
            const dst = (x * ny + y) * elemBytes;
            commandEncoder.copyBufferToBuffer(srcBuffer, src, dstBuffer, dst, elemBytes);
          }
        }
      }
    
      _runAxis0PointwiseWindowed(commandEncoder, { buffer, axisLen, lineCount, pipeline, maxBind }) {
        const lineStrideBytes = axisLen * 8;
        const maxLinesByBind = Math.max(1, Math.floor(maxBind / lineStrideBytes));
        const windows = [];
        let maxChunkBytes = 0;
        let totalParamRecords = 0;
        for (let lineStart = 0; lineStart < lineCount; lineStart += maxLinesByBind) {
          const lines = Math.min(maxLinesByBind, lineCount - lineStart);
          const chunkComplex = lines * axisLen;
          const chunkBytes = chunkComplex * 8;
          const dispatch = this._buildDispatchChunks(chunkComplex, [axisLen], lines);
          windows.push({
            lineStart,
            lineBaseBytes: lineStart * lineStrideBytes,
            chunkComplex,
            chunkBytes,
            dispatchChunks: dispatch.dispatchChunks,
            paramBaseIndex: totalParamRecords,
          });
          maxChunkBytes = Math.max(maxChunkBytes, chunkBytes);
          totalParamRecords += dispatch.dispatchChunks.length;
        }
    
        this._ensurePack(maxChunkBytes);
        this._ensureParamsUpload(totalParamRecords * 16);
        const paramsUpload = new Uint32Array(totalParamRecords * 4);
        let p = 0;
        for (const w of windows) {
          for (const dc of w.dispatchChunks) {
            paramsUpload[p++] = w.chunkComplex;
            paramsUpload[p++] = dc.baseIndex;
            paramsUpload[p++] = w.lineStart;
            paramsUpload[p++] = 0;
          }
        }
        this.device.queue.writeBuffer(this._paramsUpload, 0, paramsUpload);
    
        const bgCache = [];
        const getBindGroup = (sizeBytes) => {
          for (const e of bgCache) {
            if (e.sz === sizeBytes) return e.bg;
          }
          const bg = this.device.createBindGroup({
            layout: this._axis0TwoStep.pointwise.bgl,
            entries: [
              { binding: 0, resource: { buffer: this._packIn, offset: 0, size: sizeBytes } },
              { binding: 1, resource: { buffer: this._axis0PointwiseParams, offset: 0, size: 16 } },
            ],
          });
          bgCache.push({ sz: sizeBytes, bg });
          return bg;
        };
    
        for (const w of windows) {
          commandEncoder.copyBufferToBuffer(buffer, w.lineBaseBytes, this._packIn, 0, w.chunkBytes);
          const bg = getBindGroup(w.chunkBytes);
          let paramsRecordIndex = w.paramBaseIndex;
          for (const dc of w.dispatchChunks) {
            commandEncoder.copyBufferToBuffer(this._paramsUpload, paramsRecordIndex * 16, this._axis0PointwiseParams, 0, 16);
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(dc.wgCount, 1, 1);
            pass.end();
            paramsRecordIndex++;
          }
          commandEncoder.copyBufferToBuffer(this._packIn, 0, buffer, w.lineBaseBytes, w.chunkBytes);
        }
      }
    
      _execAxis0TwoStep(commandEncoder, opts) {
        const {
          input,
          output,
          shape,
          batch,
          totalComplex,
          totalBytes,
          inputOffsetBytes,
          outputOffsetBytes,
          maxBind,
        } = opts;
        const isGpuBuffer = isGpuBufferLike;
        const axisLen = shape[0];
        const lineStrideBytes = axisLen * 8;
        const linesTotal = totalComplex / axisLen;
        if (!Number.isInteger(linesTotal) || linesTotal < 1) {
          throw new Error(`Internal error: invalid axis-0 line count (${linesTotal})`);
        }
        const ts = this._axis0TwoStep;
        if (!ts) throw new Error("Internal error: axis-0 two-step state is missing");
        if (ts.n1 * ts.n2 !== axisLen) {
          throw new Error(`Internal error: axis-0 two-step factors mismatch axis length (${ts.n1}*${ts.n2} != ${axisLen})`);
        }
    
        const normalizeView = (bufOrView, extraOffsetBytes) => {
          if (!bufOrView) return null;
          if (isGpuBuffer(bufOrView)) {
            if (extraOffsetBytes + totalBytes > bufOrView.size) {
              throw new Error(`GPUBuffer too small: need ${extraOffsetBytes + totalBytes} bytes, have ${bufOrView.size}`);
            }
            return {
              kind: "buffer",
              buffer: bufOrView,
              startBytes: extraOffsetBytes,
              lengthBytes: bufOrView.size - extraOffsetBytes,
            };
          }
          const segments = bufOrView?.segments;
          if (!Array.isArray(segments) || segments.length === 0) {
            throw new Error("Expected GPUBuffer or BufferView");
          }
          const logicalByteOffset = bufOrView.logicalByteOffset ?? 0;
          const lengthBytes = bufOrView.lengthBytes ?? segments.reduce((a, s) => a + s.sizeBytes, 0);
          const start = logicalByteOffset + extraOffsetBytes;
          if (extraOffsetBytes + totalBytes > lengthBytes) {
            throw new Error(`BufferView too small: need ${totalBytes} bytes at offset ${extraOffsetBytes}, have ${lengthBytes}`);
          }
          return {
            kind: "view",
            segments,
            logicalByteOffset,
            lengthBytes,
            startBytes: start,
          };
        };
    
        const iterViewRanges = (view, relativeStartBytes, bytesWanted) => {
          if (view.kind === "buffer") {
            return [{ buffer: view.buffer, offsetBytes: view.startBytes + relativeStartBytes, sizeBytes: bytesWanted }];
          }
          const start = view.startBytes + relativeStartBytes;
          if (start + bytesWanted > view.logicalByteOffset + view.lengthBytes) {
            throw new Error(`BufferView window out of range: need ${bytesWanted} bytes at relative start ${relativeStartBytes}`);
          }
          const out = [];
          let remaining = bytesWanted;
          let logicalPos = 0;
          let cursor = start;
          for (const seg of view.segments) {
            const segStart = logicalPos;
            const segEnd = logicalPos + seg.sizeBytes;
            if (cursor >= segEnd) {
              logicalPos = segEnd;
              continue;
            }
            if (cursor < segStart) throw new Error("BufferView segments must be contiguous in logical space");
            const within = cursor - segStart;
            const take = Math.min(remaining, seg.sizeBytes - within);
            out.push({ buffer: seg.buffer, offsetBytes: seg.offsetBytes + within, sizeBytes: take });
            remaining -= take;
            cursor += take;
            logicalPos = segEnd;
            if (remaining === 0) break;
          }
          if (remaining !== 0) throw new Error("BufferView did not cover requested window");
          return out;
        };
    
        const inView = normalizeView(input, inputOffsetBytes);
        const outView = normalizeView(output, outputOffsetBytes);
        this._ensureTwoStepLineBuffers(lineStrideBytes);
        const lineA = this._twoStepLineA;
        const lineB = this._twoStepLineB;
    
        for (let line = 0; line < linesTotal; line++) {
          const lineOffsetBytes = line * lineStrideBytes;
          const inRanges = iterViewRanges(inView, lineOffsetBytes, lineStrideBytes);
          let dst = 0;
          for (const r of inRanges) {
            if ((r.offsetBytes | dst | r.sizeBytes) % 4 !== 0) {
              throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
            }
            commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, lineA, dst, r.sizeBytes);
            dst += r.sizeBytes;
          }
    
          ts.stage1.exec(commandEncoder, { input: lineA, batch: 1 });
          this._runAxis0PointwiseWindowed(commandEncoder, {
            buffer: lineA,
            axisLen: ts.n1,
            lineCount: ts.n2,
            pipeline: ts.pointwise.twiddlePipeline,
            maxBind,
          });
          this._transposeLineMatrixCopies(commandEncoder, { srcBuffer: lineA, dstBuffer: lineB, nx: ts.n1, ny: ts.n2 });
          ts.stage2.exec(commandEncoder, { input: lineB, batch: 1 });
          this._transposeLineMatrixCopies(commandEncoder, { srcBuffer: lineB, dstBuffer: lineA, nx: ts.n2, ny: ts.n1 });
    
          if (ts.applyScale && ts.pointwise.scalePipeline) {
            this._runAxis0PointwiseWindowed(commandEncoder, {
              buffer: lineA,
              axisLen: ts.n1,
              lineCount: ts.n2,
              pipeline: ts.pointwise.scalePipeline,
              maxBind,
            });
          }
    
          const outRanges = iterViewRanges(outView, lineOffsetBytes, lineStrideBytes);
          let src = 0;
          for (const r of outRanges) {
            if ((r.offsetBytes | src | r.sizeBytes) % 4 !== 0) {
              throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
            }
            commandEncoder.copyBufferToBuffer(lineA, src, r.buffer, r.offsetBytes, r.sizeBytes);
            src += r.sizeBytes;
          }
        }
      }
    
      _execAxis0Windowed(commandEncoder, opts) {
        const {
          input,
          output,
          shape,
          batch,
          totalComplex,
          totalBytes,
          inputOffsetBytes,
          outputOffsetBytes,
          maxBind,
        } = opts;
        const isGpuBuffer = isGpuBufferLike;
    
        const axisLen = shape[0];
        const lineStrideBytes = axisLen * 8;
        const linesTotal = totalComplex / axisLen;
        if (!Number.isInteger(linesTotal) || linesTotal < 1) {
          throw new Error(`Internal error: invalid axis-0 line count (${linesTotal})`);
        }
    
        const normalizeView = (bufOrView, extraOffsetBytes) => {
          if (!bufOrView) return null;
          if (isGpuBuffer(bufOrView)) {
            if (extraOffsetBytes + totalBytes > bufOrView.size) {
              throw new Error(`GPUBuffer too small: need ${extraOffsetBytes + totalBytes} bytes, have ${bufOrView.size}`);
            }
            return {
              kind: "buffer",
              buffer: bufOrView,
              startBytes: extraOffsetBytes,
              lengthBytes: bufOrView.size - extraOffsetBytes,
            };
          }
          const segments = bufOrView?.segments;
          if (!Array.isArray(segments) || segments.length === 0) {
            throw new Error("Expected GPUBuffer or BufferView");
          }
          const logicalByteOffset = bufOrView.logicalByteOffset ?? 0;
          const lengthBytes = bufOrView.lengthBytes ?? segments.reduce((a, s) => a + s.sizeBytes, 0);
          const start = logicalByteOffset + extraOffsetBytes;
          if (extraOffsetBytes + totalBytes > lengthBytes) {
            throw new Error(`BufferView too small: need ${totalBytes} bytes at offset ${extraOffsetBytes}, have ${lengthBytes}`);
          }
          return {
            kind: "view",
            segments,
            logicalByteOffset,
            lengthBytes,
            startBytes: start,
          };
        };
    
        const iterViewRanges = (view, relativeStartBytes, bytesWanted) => {
          if (view.kind === "buffer") {
            return [{ buffer: view.buffer, offsetBytes: view.startBytes + relativeStartBytes, sizeBytes: bytesWanted }];
          }
          const start = view.startBytes + relativeStartBytes;
          if (start + bytesWanted > view.logicalByteOffset + view.lengthBytes) {
            throw new Error(`BufferView window out of range: need ${bytesWanted} bytes at relative start ${relativeStartBytes}`);
          }
          const out = [];
          let remaining = bytesWanted;
          let logicalPos = 0;
          let cursor = start;
          for (const seg of view.segments) {
            const segStart = logicalPos;
            const segEnd = logicalPos + seg.sizeBytes;
            if (cursor >= segEnd) {
              logicalPos = segEnd;
              continue;
            }
            if (cursor < segStart) throw new Error("BufferView segments must be contiguous in logical space");
            const within = cursor - segStart;
            const take = Math.min(remaining, seg.sizeBytes - within);
            out.push({ buffer: seg.buffer, offsetBytes: seg.offsetBytes + within, sizeBytes: take });
            remaining -= take;
            cursor += take;
            logicalPos = segEnd;
            if (remaining === 0) break;
          }
          if (remaining !== 0) throw new Error("BufferView did not cover requested window");
          return out;
        };
    
        const inView = normalizeView(input, inputOffsetBytes);
        const outView = normalizeView(output, outputOffsetBytes);
    
        const maxLinesByBind = Math.max(1, Math.floor(maxBind / lineStrideBytes));
        const windows = [];
        let maxChunkBytes = 0;
        let totalParamRecords = 0;
        for (let lineStart = 0; lineStart < linesTotal; lineStart += maxLinesByBind) {
          const lineCount = Math.min(maxLinesByBind, linesTotal - lineStart);
          const chunkComplex = lineCount * axisLen;
          const chunkBytes = chunkComplex * 8;
          const dispatch = this._buildDispatchChunks(chunkComplex, shape, batch);
          windows.push({
            lineStart,
            lineBaseComplex: lineStart * axisLen,
            chunkComplex,
            chunkBytes,
            dispatchChunks: dispatch.dispatchChunks,
            paramBaseIndex: totalParamRecords,
          });
          maxChunkBytes = Math.max(maxChunkBytes, chunkBytes);
          totalParamRecords += dispatch.dispatchChunks.length;
        }
    
        if (this._pipelines.length !== 1) {
          throw new Error("Internal error: axis-0 window fallback expects a single-axis pipeline");
        }
        this._ensurePack(maxChunkBytes);
        const winA = this._packIn;
        const winB = this._packOut;
    
        this._ensureParamsUpload(totalParamRecords * 16);
        const paramsUpload = new Uint32Array(totalParamRecords * 4);
        let p = 0;
        for (const w of windows) {
          for (const dc of w.dispatchChunks) {
            paramsUpload[p++] = w.chunkComplex;
            paramsUpload[p++] = dc.baseIndex;
            paramsUpload[p++] = w.lineStart;
            paramsUpload[p++] = w.lineBaseComplex;
          }
        }
        this.device.queue.writeBuffer(this._paramsUpload, 0, paramsUpload);
    
        const bgCache = [];
        const getBindGroup = (srcBuf, dstBuf, sizeBytes, srcOffset = 0, dstOffset = 0) => {
          for (const e of bgCache) {
            if (e.src === srcBuf && e.dst === dstBuf && e.sz === sizeBytes && e.srcOff === srcOffset && e.dstOff === dstOffset) {
              return e.bg;
            }
          }
          const bg = this.device.createBindGroup({
            layout: this._bindGroupLayout,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOffset, size: sizeBytes } },
              { binding: 1, resource: { buffer: dstBuf, offset: dstOffset, size: sizeBytes } },
              { binding: 2, resource: { buffer: this._paramsBuffer, offset: 0, size: 16 } },
            ],
          });
          bgCache.push({ src: srcBuf, dst: dstBuf, sz: sizeBytes, srcOff: srcOffset, dstOff: dstOffset, bg });
          return bg;
        };
    
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const canDirectInPlaceWindowed =
          inView.kind === "buffer" &&
          outView.kind === "buffer" &&
          inView.buffer === outView.buffer &&
          inView.startBytes === outView.startBytes &&
          inView.startBytes % storageAlign === 0 &&
          windows.every((w) => ((inView.startBytes + w.lineBaseComplex * 8) % storageAlign) === 0);
    
        if (canDirectInPlaceWindowed) {
          for (let axis = 0; axis < this._pipelines.length; axis++) {
            const stages = this._pipelines[axis];
            for (const w of windows) {
              const lineOffsetBytes = inView.startBytes + w.lineBaseComplex * 8;
              let srcBuf = inView.buffer;
              let srcOff = lineOffsetBytes;
              let dstBuf = winB;
              let dstOff = 0;
    
              for (let s = 0; s < stages.length; s++) {
                const pipeline = stages[s];
                const bg = getBindGroup(srcBuf, dstBuf, w.chunkBytes, srcOff, dstOff);
                let paramsRecordIndex = w.paramBaseIndex;
                for (const dc of w.dispatchChunks) {
                  commandEncoder.copyBufferToBuffer(this._paramsUpload, paramsRecordIndex * 16, this._paramsBuffer, 0, 16);
                  const pass = commandEncoder.beginComputePass();
                  pass.setPipeline(pipeline);
                  pass.setBindGroup(0, bg);
                  pass.dispatchWorkgroups(dc.wgCount, 1, 1);
                  pass.end();
                  paramsRecordIndex++;
                }
                const tmpBuf = srcBuf;
                srcBuf = dstBuf;
                dstBuf = tmpBuf;
                const tmpOff = srcOff;
                srcOff = dstOff;
                dstOff = tmpOff;
              }
    
              if (srcBuf !== inView.buffer || srcOff !== lineOffsetBytes) {
                commandEncoder.copyBufferToBuffer(srcBuf, srcOff, inView.buffer, lineOffsetBytes, w.chunkBytes);
              }
            }
          }
          return;
        }
    
        for (let axis = 0; axis < this._pipelines.length; axis++) {
          const stages = this._pipelines[axis];
          for (const w of windows) {
            const lineOffsetBytes = w.lineBaseComplex * 8;
            const inRanges = iterViewRanges(inView, lineOffsetBytes, w.chunkBytes);
            let winWrite = 0;
            for (const r of inRanges) {
              if ((r.offsetBytes | winWrite | r.sizeBytes) % 4 !== 0) {
                throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
              }
              commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, winA, winWrite, r.sizeBytes);
              winWrite += r.sizeBytes;
            }
    
            let srcBuf = winA;
            let dstBuf = winB;
            for (let s = 0; s < stages.length; s++) {
              const pipeline = stages[s];
              const bg = getBindGroup(srcBuf, dstBuf, w.chunkBytes);
              let paramsRecordIndex = w.paramBaseIndex;
              for (const dc of w.dispatchChunks) {
                commandEncoder.copyBufferToBuffer(this._paramsUpload, paramsRecordIndex * 16, this._paramsBuffer, 0, 16);
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(dc.wgCount, 1, 1);
                pass.end();
                paramsRecordIndex++;
              }
              const tmp = srcBuf;
              srcBuf = dstBuf;
              dstBuf = tmp;
            }
    
            const outRanges = iterViewRanges(outView, lineOffsetBytes, w.chunkBytes);
            let winRead = 0;
            for (const r of outRanges) {
              if ((r.offsetBytes | winRead | r.sizeBytes) % 4 !== 0) {
                throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
              }
              commandEncoder.copyBufferToBuffer(srcBuf, winRead, r.buffer, r.offsetBytes, r.sizeBytes);
              winRead += r.sizeBytes;
            }
          }
        }
      }
    
      exec(commandEncoder, opts) {
        if (this._destroyed) throw new Error("FftPlan is destroyed");
        if (!commandEncoder) throw new Error("exec requires a commandEncoder");
        if (!opts?.input) throw new Error("exec requires opts.input (GPUBuffer)");
    
        const {
          input,
          output,
          temp,
          batch = 1,
          inputOffsetBytes = 0,
          outputOffsetBytes = 0,
        } = opts;
    
        if (!isPositiveInt(batch)) throw new Error(`batch must be a positive integer; got ${batch}`);
        if (!Number.isInteger(inputOffsetBytes) || inputOffsetBytes < 0) {
          throw new Error(`inputOffsetBytes must be a non-negative integer; got ${inputOffsetBytes}`);
        }
        if (!Number.isInteger(outputOffsetBytes) || outputOffsetBytes < 0) {
          throw new Error(`outputOffsetBytes must be a non-negative integer; got ${outputOffsetBytes}`);
        }
        if (inputOffsetBytes % 8 !== 0) throw new Error(`inputOffsetBytes must be a multiple of 8 (complex<f32>); got ${inputOffsetBytes}`);
        if (outputOffsetBytes % 8 !== 0) throw new Error(`outputOffsetBytes must be a multiple of 8 (complex<f32>); got ${outputOffsetBytes}`);
    
        const { shape, inPlace } = this.config;
        const nTotal = product(shape);
        const totalComplex = batch * nTotal;
        const totalBytes = totalComplex * 8;
        const axis0Only = this._axesList.length === 1 && this._axesList[0] === 0;
    
        if (!inPlace && !output) throw new Error("exec requires opts.output when plan.inPlace=false");
        if (inPlace && output && output !== input) {
          throw new Error("plan.inPlace=true requires opts.output to be omitted or equal to opts.input");
        }
        if (!inPlace && output && isGpuBufferLike(input) && isGpuBufferLike(output) && input === output) {
          throw new Error("plan.inPlace=false does not allow opts.input === opts.output");
        }
    
        const limits = this.device.limits;
        const storageAlign = limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxBind = this.config.maxStorageBufferBindingSize ?? limits?.maxStorageBufferBindingSize ?? Infinity;
        const isBufferOrView = (v) =>
          isGpuBufferLike(v) || (v && Array.isArray(v.segments) && v.segments.length > 0);
        if (totalBytes > maxBind) {
          const bytesPerBatch = nTotal * 8;
          // Out-of-core fallback (batch chunking): when one logical batch fits the binding limit,
          // execute multiple smaller batch slices with adjusted byte offsets.
          if (batch > 1 && bytesPerBatch <= maxBind) {
            const maxBatchPerChunk = Math.max(1, Math.floor(maxBind / bytesPerBatch));
            if (inPlace) {
              // In-place path binds the FFT data buffer directly and requires aligned offsets.
              if (bytesPerBatch % storageAlign !== 0) {
                throw new Error(
                  [
                    `Requested binding size exceeds device.limits.maxStorageBufferBindingSize.`,
                    `Batch-chunk fallback is available but requires bytesPerBatch (${bytesPerBatch}) to be aligned to minStorageBufferOffsetAlignment (${storageAlign}) for inPlace plans.`,
                    `shape=${JSON.stringify(shape)} batch=${batch} totalBytes=${totalBytes}`,
                    `maxStorageBufferBindingSize=${maxBind}`,
                  ].join("\n")
                );
              }
              if (inputOffsetBytes % storageAlign !== 0) {
                throw new Error(
                  [
                    `Requested binding size exceeds device.limits.maxStorageBufferBindingSize.`,
                    `Batch-chunk fallback is available but requires inputOffsetBytes (${inputOffsetBytes}) aligned to minStorageBufferOffsetAlignment (${storageAlign}) for inPlace plans.`,
                    `shape=${JSON.stringify(shape)} batch=${batch} totalBytes=${totalBytes}`,
                    `maxStorageBufferBindingSize=${maxBind}`,
                  ].join("\n")
                );
              }
            }
    
            for (let b0 = 0; b0 < batch; b0 += maxBatchPerChunk) {
              const bCount = Math.min(maxBatchPerChunk, batch - b0);
              const chunkOffsetBytes = b0 * bytesPerBatch;
              this.exec(commandEncoder, {
                input,
                output,
                temp,
                batch: bCount,
                inputOffsetBytes: inputOffsetBytes + chunkOffsetBytes,
                outputOffsetBytes: outputOffsetBytes + chunkOffsetBytes,
              });
            }
            return;
          }
    
          // Windowed fallback for axis-0-only plans: process contiguous line chunks that fit the binding limit.
          if (axis0Only) {
            const axisLen = shape[0];
            const lineStrideBytes = axisLen * 8;
            const outTarget = inPlace ? input : output;
            const outOffset = inPlace ? inputOffsetBytes : outputOffsetBytes;
            const canWindow =
              isBufferOrView(input) &&
              isBufferOrView(outTarget) &&
              lineStrideBytes <= maxBind;
            if (canWindow) {
              if (isGpuBufferLike(input) && inputOffsetBytes + totalBytes > input.size) {
                throw new Error(`Input buffer too small: need ${inputOffsetBytes + totalBytes} bytes, have ${input.size}`);
              }
              if (isGpuBufferLike(outTarget) && outOffset + totalBytes > outTarget.size) {
                throw new Error(`Output buffer too small: need ${outOffset + totalBytes} bytes, have ${outTarget.size}`);
              }
              if (temp && isGpuBufferLike(temp) && totalBytes > temp.size) {
                throw new Error(`temp buffer too small: need ${totalBytes} bytes, have ${temp.size}`);
              }
              this._execAxis0Windowed(commandEncoder, {
                input,
                output: outTarget,
                temp,
                shape,
                batch,
                totalComplex,
                totalBytes,
                inputOffsetBytes,
                outputOffsetBytes: outOffset,
                storageAlign,
                maxBind,
              });
              return;
            }
            const canTwoStep =
              !!this._axis0TwoStep &&
              isBufferOrView(input) &&
              isBufferOrView(outTarget) &&
              lineStrideBytes > maxBind;
            if (canTwoStep) {
              if (isGpuBufferLike(input) && inputOffsetBytes + totalBytes > input.size) {
                throw new Error(`Input buffer too small: need ${inputOffsetBytes + totalBytes} bytes, have ${input.size}`);
              }
              if (isGpuBufferLike(outTarget) && outOffset + totalBytes > outTarget.size) {
                throw new Error(`Output buffer too small: need ${outOffset + totalBytes} bytes, have ${outTarget.size}`);
              }
              this._execAxis0TwoStep(commandEncoder, {
                input,
                output: outTarget,
                shape,
                batch,
                totalComplex,
                totalBytes,
                inputOffsetBytes,
                outputOffsetBytes: outOffset,
                maxBind,
              });
              return;
            }
          }
    
          throw new Error(
            [
              `Requested binding size exceeds device.limits.maxStorageBufferBindingSize.`,
              `shape=${JSON.stringify(shape)} batch=${batch} totalBytes=${totalBytes}`,
              `bytesPerBatch=${bytesPerBatch}`,
              `maxStorageBufferBindingSize=${maxBind}`,
            ].join("\n")
          );
        }
    
        const normalizeView = (bufOrView, extraOffsetBytes) => {
          if (!bufOrView) return null;
          if (isGpuBufferLike(bufOrView)) {
            return {
              kind: "buffer",
              buffer: bufOrView,
              offsetBytes: extraOffsetBytes,
              lengthBytes: bufOrView.size - extraOffsetBytes,
            };
          }
          const segments = bufOrView?.segments;
          if (!Array.isArray(segments) || segments.length === 0) {
            throw new Error("Expected GPUBuffer or BufferView");
          }
          return {
            kind: "view",
            segments,
            logicalByteOffset: bufOrView.logicalByteOffset ?? 0,
            lengthBytes: bufOrView.lengthBytes ?? segments.reduce((a, s) => a + s.sizeBytes, 0),
            extraOffsetBytes,
          };
        };
    
        const iterViewRanges = (view, totalBytesWanted) => {
          // Produces a list of physical ranges that cover [start, start+totalBytesWanted) within the logical view.
          const start = view.logicalByteOffset + view.extraOffsetBytes;
          if (start < 0) throw new Error("BufferView start is negative");
          if (view.extraOffsetBytes < 0) throw new Error("BufferView extraOffsetBytes is negative");
          if (start + totalBytesWanted > view.logicalByteOffset + view.lengthBytes) {
            const have = view.lengthBytes - view.extraOffsetBytes;
            throw new Error(`BufferView too small: need ${totalBytesWanted} bytes, have ${have}`);
          }
    
          const out = [];
          let remaining = totalBytesWanted;
          let logicalPos = 0;
          let cursor = start;
    
          for (const seg of view.segments) {
            const segStart = logicalPos;
            const segEnd = logicalPos + seg.sizeBytes;
            if (cursor >= segEnd) {
              logicalPos = segEnd;
              continue;
            }
            if (cursor < segStart) {
              throw new Error("BufferView segments must be contiguous in logical space");
            }
            const within = cursor - segStart;
            const take = Math.min(remaining, seg.sizeBytes - within);
            out.push({
              buffer: seg.buffer,
              offsetBytes: seg.offsetBytes + within,
              sizeBytes: take,
            });
            remaining -= take;
            cursor += take;
            logicalPos = segEnd;
            if (remaining === 0) break;
          }
          if (remaining !== 0) throw new Error("BufferView did not cover requested range");
          return out;
        };
    
        const inNorm = normalizeView(input, inputOffsetBytes);
        const outNorm = normalizeView(output, outputOffsetBytes);
    
        const singleBindOffset = (v) => {
          if (!v || v.kind !== "view" || v.segments.length !== 1) return null;
          const seg = v.segments[0];
          return seg.offsetBytes + v.logicalByteOffset + v.extraOffsetBytes;
        };
    
        const inNeedsPackForAlign =
          (inNorm.kind === "buffer" && inNorm.offsetBytes % storageAlign !== 0) ||
          (inNorm.kind === "view" && inNorm.segments.length === 1 && singleBindOffset(inNorm) % storageAlign !== 0);
        const outNeedsPackForAlign =
          outNorm &&
          ((outNorm.kind === "buffer" && outNorm.offsetBytes % storageAlign !== 0) ||
            (outNorm.kind === "view" && outNorm.segments.length === 1 && singleBindOffset(outNorm) % storageAlign !== 0));
    
        const doPackInput = (inNorm.kind === "view" && inNorm.segments.length > 1) || (!inPlace && inNeedsPackForAlign);
        const doPackOutput = (outNorm && outNorm.kind === "view" && outNorm.segments.length > 1) || (!inPlace && outNeedsPackForAlign);
    
        if (inPlace && (doPackInput || doPackOutput)) {
          throw new Error("inPlace with non-contiguous or misaligned views is not supported");
        }
    
        if (doPackInput || doPackOutput) {
          this._ensurePack(totalBytes);
        }
    
        const inResolved = (() => {
          if (inNorm.kind === "buffer") {
            return { buffer: inNorm.buffer, offsetBytes: inNorm.offsetBytes, viewLimitBytes: inNorm.lengthBytes };
          }
          if (doPackInput) {
            const ranges = iterViewRanges(inNorm, totalBytes);
            let dst = 0;
            for (const r of ranges) {
              if ((r.offsetBytes | dst | r.sizeBytes) % 4 !== 0) throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
              commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, this._packIn, dst, r.sizeBytes);
              dst += r.sizeBytes;
            }
            return { buffer: this._packIn, offsetBytes: 0, viewLimitBytes: totalBytes };
          }
          if (inNorm.segments.length !== 1) throw new Error("Unexpected BufferView segment count");
          const seg = inNorm.segments[0];
          const off = seg.offsetBytes + inNorm.logicalByteOffset + inNorm.extraOffsetBytes;
          return { buffer: seg.buffer, offsetBytes: off, viewLimitBytes: inNorm.lengthBytes - inNorm.extraOffsetBytes };
        })();
    
        const outResolved = (() => {
          if (!outNorm) return null;
          if (outNorm.kind === "buffer") {
            return { buffer: outNorm.buffer, offsetBytes: outNorm.offsetBytes, viewLimitBytes: outNorm.lengthBytes };
          }
          if (doPackOutput) {
            // We will scatter after compute.
            return { buffer: this._packOut, offsetBytes: 0, viewLimitBytes: totalBytes, _scatterTo: outNorm };
          }
          if (outNorm.segments.length !== 1) throw new Error("Unexpected BufferView segment count");
          const seg = outNorm.segments[0];
          const off = seg.offsetBytes + outNorm.logicalByteOffset + outNorm.extraOffsetBytes;
          return { buffer: seg.buffer, offsetBytes: off, viewLimitBytes: outNorm.lengthBytes - outNorm.extraOffsetBytes };
        })();
    
        const primary = inPlace ? inResolved : outResolved;
        if (!primary) throw new Error("exec requires opts.output when plan.inPlace=false");
    
        if (inResolved.viewLimitBytes < totalBytes) {
          throw new Error(`Input view too small: need ${totalBytes} bytes, have ${inResolved.viewLimitBytes}`);
        }
        if (primary.viewLimitBytes < totalBytes) {
          throw new Error(`Output view too small: need ${totalBytes} bytes, have ${primary.viewLimitBytes}`);
        }
    
        const { dispatchChunks } = this._buildDispatchChunks(totalComplex, shape, batch);
        const useChunkedDispatch = dispatchChunks.length > 1;
    
        const totalPasses = this._pipelines.reduce((acc, stages) => acc + stages.length, 0);
    
        let scratch = null;
        let scratchOffset = 0;
        if (temp) {
          const tNorm = normalizeView(temp, 0);
          if (!tNorm) throw new Error("temp must be GPUBuffer or BufferView");
          const t =
            tNorm.kind === "buffer"
              ? { buffer: tNorm.buffer, offsetBytes: tNorm.offsetBytes, viewLimitBytes: tNorm.lengthBytes }
              : (() => {
                  if (tNorm.segments.length !== 1) throw new Error("temp must be a GPUBuffer or a single-segment BufferView");
                  const seg = tNorm.segments[0];
                  const off = seg.offsetBytes + tNorm.logicalByteOffset + tNorm.extraOffsetBytes;
                  return { buffer: seg.buffer, offsetBytes: off, viewLimitBytes: tNorm.lengthBytes - tNorm.extraOffsetBytes };
                })();
          if (t.offsetBytes % storageAlign !== 0) {
            throw new Error(`temp binding offset must be multiple of device.limits.minStorageBufferOffsetAlignment=${storageAlign}; got ${t.offsetBytes}`);
          }
          if (t.viewLimitBytes < totalBytes) {
            throw new Error(`temp view too small: need ${totalBytes} bytes, have ${t.viewLimitBytes}`);
          }
    
          // If temp aliases the data buffer (same GPUBuffer), WebGPU validation gets messy and copyBufferToBuffer
          // within the same buffer is rejected. for correctness, use an internal scratch buffer instead.
          const tempAliasesData = t.buffer === inResolved.buffer || (primary && t.buffer === primary.buffer);
          if (!tempAliasesData) {
            scratch = t.buffer;
            scratchOffset = t.offsetBytes;
          } else {
            this._ensureScratch(totalBytes);
            scratch = this._scratch;
            scratchOffset = 0;
          }
        } else {
          this._ensureScratch(totalBytes);
          scratch = this._scratch;
          scratchOffset = 0;
        }
    
        // Update params for bounds-checking in kernels.
        if (!useChunkedDispatch) {
          this.device.queue.writeBuffer(this._paramsBuffer, 0, new Uint32Array([totalComplex, 0, 0, 0]));
        } else {
          const bytes = dispatchChunks.length * 16;
          this._ensureParamsUpload(bytes);
          const upload = new Uint32Array(dispatchChunks.length * 4);
          for (let i = 0; i < dispatchChunks.length; i++) {
            const base = i * 4;
            upload[base] = totalComplex;
            upload[base + 1] = dispatchChunks[i].baseIndex;
          }
          this.device.queue.writeBuffer(this._paramsUpload, 0, upload);
        }
    
        const bgCache = [];
        const getBindGroup = (srcBuf, srcOffset, dstBuf, dstOffset) => {
          for (const e of bgCache) {
            if (e.src === srcBuf && e.dst === dstBuf && e.so === srcOffset && e.do === dstOffset) return e.bg;
          }
          const bg = this.device.createBindGroup({
            layout: this._bindGroupLayout,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOffset, size: totalBytes } },
              { binding: 1, resource: { buffer: dstBuf, offset: dstOffset, size: totalBytes } },
              { binding: 2, resource: { buffer: this._paramsBuffer, offset: 0, size: 16 } },
            ],
          });
          bgCache.push({ src: srcBuf, dst: dstBuf, so: srcOffset, do: dstOffset, bg });
          return bg;
        };
    
        const secondary = scratch;
    
        // Global pass order: X then Y then Z axes, stage 0..log2-1 per axis.
        // Buffer ping-pongs between `primary` and `secondary` after the first stage.
        let srcBuf = inResolved.buffer;
        let srcOffset = inResolved.offsetBytes;
    
        // Choose initial destination to avoid a final copy when possible (out-of-place only).
        // For inPlace, the first dispatch must not write back into the same region we're reading.
        let dstBuf;
        let dstOffset;
        if (inPlace) {
          dstBuf = secondary;
          dstOffset = scratchOffset;
        } else {
          const wantFirstDstIsPrimary = totalPasses % 2 === 1;
          dstBuf = wantFirstDstIsPrimary ? primary.buffer : secondary;
          dstOffset = dstBuf === primary.buffer ? primary.offsetBytes : scratchOffset;
        }
    
        let passIndex = 0;
        for (let axis = 0; axis < this._pipelines.length; axis++) {
          const stages = this._pipelines[axis];
          for (let s = 0; s < stages.length; s++) {
            const pipeline = stages[s];
            const bg = getBindGroup(srcBuf, srcOffset, dstBuf, dstOffset);
            if (srcOffset % storageAlign !== 0) {
              throw new Error(`source binding offset must be aligned to ${storageAlign}; got ${srcOffset}`);
            }
            if (dstOffset % storageAlign !== 0) {
              throw new Error(`dest binding offset must be aligned to ${storageAlign}; got ${dstOffset}`);
            }
            if (srcBuf === dstBuf && srcOffset === dstOffset) {
              throw new Error("Internal error: src and dst bindings alias the same range");
            }
    
            // Important: each stage must be its own compute pass. Otherwise, WebGPU rejects the command buffer
            // because a buffer would be used as both read-only and read-write storage within the same usage scope.
            for (let chunk = 0; chunk < dispatchChunks.length; chunk++) {
              if (useChunkedDispatch) {
                commandEncoder.copyBufferToBuffer(this._paramsUpload, chunk * 16, this._paramsBuffer, 0, 16);
              }
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(dispatchChunks[chunk].wgCount, 1, 1);
              pass.end();
            }
    
            // Swap buffers for next pass.
            passIndex++;
            const nextSrc = dstBuf;
            const nextSrcOffset = dstOffset;
            const nextDst = nextSrc === primary.buffer ? secondary : primary.buffer;
            const nextDstOffset = nextDst === primary.buffer ? primary.offsetBytes : scratchOffset;
    
            srcBuf = nextSrc;
            srcOffset = nextSrcOffset;
            dstBuf = nextDst;
            dstOffset = nextDstOffset;
          }
        }
    
        // If the final result ended in secondary, copy to primary.
        const finalInPrimary = srcBuf === primary.buffer && srcOffset === primary.offsetBytes;
        if (!finalInPrimary) {
          if (secondary === primary.buffer) {
            throw new Error("Internal error: FFT final copy would alias within a single buffer");
          }
          commandEncoder.copyBufferToBuffer(srcBuf, srcOffset, primary.buffer, primary.offsetBytes, totalBytes);
        }
    
        // Scatter packed output back into a multi-segment output view if needed.
        if (outResolved?._scatterTo) {
          const view = outResolved._scatterTo;
          const ranges = iterViewRanges(view, totalBytes);
          let src = 0;
          for (const r of ranges) {
            if ((r.offsetBytes | src | r.sizeBytes) % 4 !== 0) throw new Error("copyBufferToBuffer requires 4-byte aligned offsets/sizes");
            commandEncoder.copyBufferToBuffer(primary.buffer, primary.offsetBytes + src, r.buffer, r.offsetBytes, r.sizeBytes);
            src += r.sizeBytes;
          }
        }
      }
    }
    
    function createFftPlan(device, options) {
      if (!device) throw new Error("createFftPlan requires a WebGPU device");
      const {
        shape,
        direction,
        normalize = "none",
        inPlace = false,
        layout = "interleaved",
        precision = "f32",
        axes = null,
        maxStorageBufferBindingSize = null,
      } = options ?? {};
    
      if (!Array.isArray(shape) || shape.length < 1) {
        throw new Error(`shape must be an array of one or more dimensions; got ${JSON.stringify(shape)}`);
      }
      if (!shape.every((n) => isPositiveInt(n))) {
        throw new Error(`shape elements must be positive integers; got ${JSON.stringify(shape)}`);
      }
      if (shape.some((n) => n < 2)) {
        throw new Error(`All dimensions must be >= 2; got shape=${JSON.stringify(shape)}`);
      }
    
      assertOneOf(direction, ["forward", "inverse"], "direction");
      assertOneOf(normalize, ["none", "unitary", "backward"], "normalize");
      assertOneOf(layout, ["interleaved"], "layout");
      assertOneOf(precision, ["f32"], "precision");
      if (typeof inPlace !== "boolean") throw new Error(`inPlace must be boolean; got ${inPlace}`);
      if (maxStorageBufferBindingSize != null) {
        if (!Number.isInteger(maxStorageBufferBindingSize) || maxStorageBufferBindingSize <= 0) {
          throw new Error(`maxStorageBufferBindingSize must be a positive integer when provided; got ${maxStorageBufferBindingSize}`);
        }
      }
    
      const rank = shape.length;
      const dims = shape.slice();
      const nTotal = product(shape);
      const axesList = axes == null ? Array.from({ length: rank }, (_, i) => i) : axes;
      if (!Array.isArray(axesList) || axesList.length === 0) throw new Error("axes must be null or a non-empty array");
      for (const axis of axesList) {
        if (!Number.isInteger(axis) || axis < 0 || axis >= rank) throw new Error(`Invalid axis ${axis} for rank ${rank}`);
      }
    
      const limits = device.limits;
      let workgroupSizeX = pickWorkgroupSizeX(limits);
      const baselineWorkgroupSizeX = workgroupSizeX;
      const deviceMaxBind = limits?.maxStorageBufferBindingSize ?? Infinity;
      const maxBind = maxStorageBufferBindingSize == null ? deviceMaxBind : Math.min(deviceMaxBind, maxStorageBufferBindingSize);
      const minBytes = nTotal * 8;
      const axis0Only = axesList.length === 1 && axesList[0] === 0;
      const axis0LineBytes = dims[0] * 8;
      const maxBufferSize = limits?.maxBufferSize ?? Infinity;
      let axis0TwoStep = null;
      if (minBytes > maxBind) {
        if (!axis0Only) {
          throw new Error(
            [
              `shape=${JSON.stringify(shape)} requires at least ${minBytes} bytes per buffer binding (batch=1).`,
              `This exceeds device.limits.maxStorageBufferBindingSize=${maxBind}.`,
              `Only axis-0-only plans can use out-of-core windowed/two-step execution when full bindings exceed the limit.`,
              `Device limits:\n${formatLimits(limits)}`,
            ].join("\n")
          );
        }
        if (axis0LineBytes > maxBind) {
          const maxAxisElems = Math.floor(maxBind / 8);
          const factors = chooseAxis0TwoStepFactors(dims[0], maxAxisElems);
          if (!factors || axis0LineBytes > maxBufferSize) {
            throw new Error(
              [
                `shape=${JSON.stringify(shape)} requires at least ${minBytes} bytes per buffer binding (batch=1).`,
                `This exceeds device.limits.maxStorageBufferBindingSize=${maxBind}.`,
                `axis-0 line-window fallback is unavailable because one axis-0 line needs ${axis0LineBytes} bytes.`,
                factors
                  ? `Axis-0 two-step fallback requires line staging of ${axis0LineBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}.`
                  : `Axis-0 two-step fallback is unavailable: shape[0]=${dims[0]} cannot be factorized into n1*n2 with n1,n2 <= floor(maxStorageBufferBindingSize/8)=${maxAxisElems} using supported radices.`,
                `Device limits:\n${formatLimits(limits)}`,
              ].join("\n")
            );
          }
          axis0TwoStep = { n1: factors.n1, n2: factors.n2 };
        }
      }
    
      const scale = normalizeScaleFactor({ normalize, direction, nTotal });
      const applyAnyScale = Math.abs(scale - 1.0) > 0;
    
      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    
      const paramsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    
      const pipelines = [];
    
      // Optional subgroup-accelerated path:
      // - only when the device advertises subgroups
      // - only when compiling a single-axis plan (axes=[axis])
      // - only for power-of-two axis lengths that fit within one workgroup (1 invocation per element)
      let useSubgroupAxis = false;
      if (device.features?.has?.("subgroups") && axesList.length === 1) {
        const axis = axesList[0];
        const axisLen = dims[axis];
        const maxWG = Math.min(limits?.maxComputeWorkgroupSizeX ?? 256, limits?.maxComputeInvocationsPerWorkgroup ?? 256);
        const maxWgMem = limits?.maxComputeWorkgroupStorageSize ?? (32 * 1024);
        const shmemBytes = axisLen * 8; // vec2<f32> per element
        if (isPowerOfTwo(axisLen) && axisLen >= 2 && axisLen <= maxWG && shmemBytes <= maxWgMem) {
          useSubgroupAxis = true;
          workgroupSizeX = axisLen;
        }
      }
    
      for (const axis of axis0TwoStep ? [] : axesList) {
        const axisLen = dims[axis];
        const strideComplex = strideForAxis(dims, axis);
    
        const axisPipelines = [];
        if (useSubgroupAxis) {
          try {
            const isLastAxis = axis === rank - 1;
            const applyScale = applyAnyScale && isLastAxis;
            const code = generateSubgroupPow2FftWGSL({
              rank,
              axis,
              dims,
              axisLength: axisLen,
              strideComplex,
              direction,
              applyScale,
              scaleFactor: scale,
            });
            const module = device.createShaderModule({ code });
            const pipeline = device.createComputePipeline({
              layout: pipelineLayout,
              compute: { module, entryPoint: "main" },
            });
            axisPipelines.push(pipeline);
          } catch {
            // Graceful fallback to baseline path if subgroup compilation fails.
            useSubgroupAxis = false;
            workgroupSizeX = baselineWorkgroupSizeX;
          }
        }
        if (!useSubgroupAxis) {
          const radices = factorizeRadices(axisLen);
          if (!radices) {
            throw new Error(
              `Axis length ${axisLen} is not factorable by supported radices {2,3,4,5,7,8,11,13} in the current implementation`
            );
          }
          let ns = 1;
          for (let s = 0; s < radices.length; s++) {
            const radix = radices[s];
            ns *= radix;
            const isLastAxis = axis === rank - 1;
            const isLastStage = s === radices.length - 1;
            const applyScale = applyAnyScale && isLastAxis && isLastStage;
    
            const code = generateStockhamRadixStageWGSL({
              rank,
              axis,
              dims,
              axisLength: axisLen,
              strideComplex,
              radix,
              ns,
              direction,
              workgroupSize: workgroupSizeX,
              applyScale,
              scaleFactor: scale,
            });
            const module = device.createShaderModule({ code });
            const pipeline = device.createComputePipeline({
              layout: pipelineLayout,
              compute: { module, entryPoint: "main" },
            });
            axisPipelines.push(pipeline);
          }
        }
        pipelines.push(axisPipelines);
      }
    
      const config = {
        shape: dims,
        direction,
        normalize,
        inPlace,
        layout,
        precision,
        maxStorageBufferBindingSize: Number.isFinite(maxBind) ? maxBind : null,
      };
    
      return new FftPlan(device, config, {
        axesList,
        pipelines,
        workgroupSizeX,
        paramsBuffer,
        bindGroupLayout,
        axis0TwoStep: axis0TwoStep
          ? {
              ...axis0TwoStep,
              applyScale: applyAnyScale,
              scaleFactor: scale,
            }
          : null,
      });
    }
    
    
    exports['createFftPlan'] = createFftPlan;
  });

  __define('src/public_api.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const __reexport_1 = require('src/runtime/create_plan.js');
    exports['createPlan'] = __reexport_1['createPlan'];
    const __reexport_2 = require('src/runtime/pipeline_cache.js');
    exports['exportPipelineCacheSnapshot'] = __reexport_2['exportPipelineCacheSnapshot'];
    exports['importPipelineCacheSnapshot'] = __reexport_2['importPipelineCacheSnapshot'];
    const __reexport_3 = require('src/runtime/fftconv_channel_lane_presets.js');
    exports['createFftConvChannelLanePreset'] = __reexport_3['createFftConvChannelLanePreset'];
    exports['createFftConvKernelMajorChannelLanePreset'] = __reexport_3['createFftConvKernelMajorChannelLanePreset'];
    exports['createFftConvBatchMajorChannelLanePreset'] = __reexport_3['createFftConvBatchMajorChannelLanePreset'];
  });

  __define('src/runtime/algorithms/bluestein_axis.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { createFftPlan } = require('src/plan.js');
    const { factorizeSupportedRadices, nextPow2, nextSmoothAtLeast } = require('src/utils/factors.js');
    const { ensureWithinBindingLimit, prod, alignBytes } = require('src/runtime/common.js');
    const { normalizeToContiguousRanges } = require('src/runtime/segmented_io.js');
    
    const { generateBluesteinPreWGSL, generateBluesteinMulBfftWGSL, generateBluesteinPostWGSL } = require('src/kernels/bluestein.js');
    
    function generateSliceMulWriteWGSL(workgroupSize) {
      return /* wgsl */ `
    struct Params {
      total: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read> lhs: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read> rhs: array<vec2<f32>>;
    @group(0) @binding(2) var<storage, read_write> outv: array<vec2<f32>>;
    @group(0) @binding(3) var<uniform> params: Params;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.total) { return; }
      let a = lhs[i];
      let b = rhs[i];
      outv[i] = vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    `;
    }
    
    function generateSliceMulInPlaceWGSL(workgroupSize) {
      return /* wgsl */ `
    struct Params {
      total: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> lhs: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read> rhs: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.total) { return; }
      let a = lhs[i];
      let b = rhs[i];
      lhs[i] = vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    `;
    }
    
    class BluesteinAxis {
      constructor(device, cache, { shape, rank, batch, axis, direction, workgroupSize, maxWorkBytes = null }) {
        this.device = device;
        this.cache = cache;
        this.shape = shape;
        this.rank = rank;
        this.batch = batch;
        this.axis = axis;
        this.direction = direction;
        this.workgroupSize = workgroupSize;
    
        this.N = shape[axis];
        const sign = direction === "forward" ? -1.0 : 1.0;
        const M0 = nextSmoothAtLeast(2 * this.N - 1);
        this.M = factorizeSupportedRadices(M0) ? M0 : nextPow2(2 * this.N - 1);
        if (!factorizeSupportedRadices(this.M)) throw new Error(`Bluestein internal M=${this.M} not factorable by supported radices`);
    
        this.logicalTotal = prod(shape);
        this.lines = batch * (this.logicalTotal / this.N);
        let strideComplex = 1;
        for (let d = 0; d < axis; d++) strideComplex *= shape[d];
        this._strideComplex = strideComplex;
        this._workBytesPerLine = this.M * 8;
        const deviceMaxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
        const chunkBudget = maxWorkBytes == null ? deviceMaxBind : Math.min(deviceMaxBind, maxWorkBytes);
        this._bindBudgetBytes = chunkBudget;
        this._maxSliceElems = Math.max(1, Math.floor(chunkBudget / 8));
        this.maxChunkLines = Math.max(1, Math.floor(chunkBudget / this._workBytesPerLine));
        this.maxChunkLines = Math.min(this.maxChunkLines, this.lines);
        this.workBytes = this.maxChunkLines * this._workBytesPerLine;
        this._maxChunkCount = Math.max(1, Math.ceil(this.lines / this.maxChunkLines));
        this._paramStride = alignBytes(16, device.limits?.minUniformBufferOffsetAlignment ?? 256);
        this._paramCapacity = this._maxChunkCount;
        this._retiredParamBuffers = [];
        const needsSlicedWorkPath = this._strideComplex === 1 && (this.N * 8 > this._bindBudgetBytes || this.M * 8 > this._bindBudgetBytes);
        if (!needsSlicedWorkPath) {
          ensureWithinBindingLimit(device, this.workBytes, `Bluestein work buffer: N=${this.N} M=${this.M} lines=${this.lines}`);
        }
    
        // chirpA/C length N
        const chirpA = new Float32Array(2 * this.N);
        for (let n = 0; n < this.N; n++) {
          const ang = sign * (Math.PI * (n * n) / this.N);
          chirpA[2 * n] = Math.cos(ang);
          chirpA[2 * n + 1] = Math.sin(ang);
        }
        const b = new Float32Array(2 * this.M);
        b[0] = 1;
        b[1] = 0;
        for (let m = 1; m <= this.N - 1; m++) {
          const ang = -sign * (Math.PI * (m * m) / this.N);
          const re = Math.cos(ang);
          const im = Math.sin(ang);
          b[2 * m] = re;
          b[2 * m + 1] = im;
          b[2 * (this.M - m)] = re;
          b[2 * (this.M - m) + 1] = im;
        }
    
        this.chirpABuf = device.createBuffer({ size: chirpA.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.chirpCBuf = device.createBuffer({ size: chirpA.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.chirpABuf, 0, chirpA);
        device.queue.writeBuffer(this.chirpCBuf, 0, chirpA);
    
        this.bfftBuf = device.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        device.queue.writeBuffer(this.bfftBuf, 0, b);
    
        this.fftFwd = createFftPlan(device, { shape: [this.M], direction: "forward", normalize: "none", inPlace: true, layout: "interleaved", precision: "f32" });
        this.fftInv = createFftPlan(device, { shape: [this.M], direction: "inverse", normalize: "backward", inPlace: true, layout: "interleaved", precision: "f32" });
    
        // compute bfft once
        {
          const enc = device.createCommandEncoder();
          this.fftFwd.exec(enc, { input: this.bfftBuf, batch: 1 });
          device.queue.submit([enc.finish()]);
        }
    
        const preCode = generateBluesteinPreWGSL({
          rank,
          axis,
          dims: shape,
          axisLength: this.N,
          mLength: this.M,
          strideComplex,
          workgroupSize,
        });
        const mulCode = generateBluesteinMulBfftWGSL({ mLength: this.M, workgroupSize });
        const postCode = generateBluesteinPostWGSL({
          rank,
          axis,
          dims: shape,
          axisLength: this.N,
          mLength: this.M,
          strideComplex,
          workgroupSize,
        });
    
        this.bglPre = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        this.bglMul = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        this.bglPost = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        this.plPre = device.createPipelineLayout({ bindGroupLayouts: [this.bglPre] });
        this.plMul = device.createPipelineLayout({ bindGroupLayouts: [this.bglMul] });
        this.plPost = device.createPipelineLayout({ bindGroupLayouts: [this.bglPost] });
        this.prePipe = cache.getComputePipeline({ code: preCode, layout: this.plPre });
        this.mulPipe = cache.getComputePipeline({ code: mulCode, layout: this.plMul });
        this.postPipe = cache.getComputePipeline({ code: postCode, layout: this.plPost });
    
        this.paramsPre = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.paramsMul = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.paramsPost = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.paramsPre, 0, new Uint32Array([this.maxChunkLines, 0, 0, 0]));
        device.queue.writeBuffer(this.paramsPost, 0, new Uint32Array([this.maxChunkLines, 0, 0, 0]));
        device.queue.writeBuffer(this.paramsMul, 0, new Uint32Array([this.maxChunkLines * this.M, 0, 0, 0]));
    
        this.sliceWrite = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = cache.getComputePipeline({ code: generateSliceMulWriteWGSL(workgroupSize), layout: pl });
          return { bgl, pl, pipeline };
        })();
        this.sliceMul = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = cache.getComputePipeline({ code: generateSliceMulInPlaceWGSL(workgroupSize), layout: pl });
          return { bgl, pl, pipeline };
        })();
        this.sliceParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._sliceInputBuffer = null;
        this._sliceTwiddleBuffer = null;
        this._sliceOutputBuffer = null;
        this._sliceZeroBuffer = null;
        this._sliceBytes = 0;
        this._retiredSliceBuffers = [];
      }
    
      _ensureParamCapacity(requiredChunkCount) {
        if (requiredChunkCount <= this._paramCapacity) return;
        let nextCapacity = this._paramCapacity;
        while (nextCapacity < requiredChunkCount) nextCapacity *= 2;
        const nextBytes = nextCapacity * this._paramStride;
        this._retiredParamBuffers.push(this.paramsPre, this.paramsMul, this.paramsPost);
        this.paramsPre = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.paramsMul = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.paramsPost = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._paramCapacity = nextCapacity;
      }
    
      supportsBoundedLineSlicing() {
        return this._strideComplex === 1;
      }
    
      _ensureSliceBuffers(minBytes) {
        if (this._sliceInputBuffer && this._sliceBytes >= minBytes) return;
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        if (minBytes > maxBufferSize) {
          throw new Error(`Bluestein sliced staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`);
        }
        const next = () =>
          this.device.createBuffer({
            size: minBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
        if (this._sliceInputBuffer) this._retiredSliceBuffers.push(this._sliceInputBuffer);
        if (this._sliceTwiddleBuffer) this._retiredSliceBuffers.push(this._sliceTwiddleBuffer);
        if (this._sliceOutputBuffer) this._retiredSliceBuffers.push(this._sliceOutputBuffer);
        if (this._sliceZeroBuffer) this._retiredSliceBuffers.push(this._sliceZeroBuffer);
        this._sliceInputBuffer = next();
        this._sliceTwiddleBuffer = next();
        this._sliceOutputBuffer = next();
        this._sliceZeroBuffer = next();
        this._sliceBytes = minBytes;
      }
    
      _zeroWorkRange(commandEncoder, buffer, offsetBytes, sizeBytes) {
        if (typeof commandEncoder.clearBuffer === "function") {
          commandEncoder.clearBuffer(buffer, offsetBytes, sizeBytes);
          return;
        }
        this._ensureSliceBuffers(Math.min(this._maxSliceElems * 8, sizeBytes));
        let done = 0;
        while (done < sizeBytes) {
          const n = Math.min(this._sliceBytes, sizeBytes - done);
          commandEncoder.copyBufferToBuffer(this._sliceZeroBuffer, 0, buffer, offsetBytes + done, n);
          done += n;
        }
      }
    
      _runSliceWrite(commandEncoder, lhsBuf, rhsBuf, dstBuf, countElems) {
        this.device.queue.writeBuffer(this.sliceParams, 0, new Uint32Array([countElems, 0, 0, 0]));
        const bytes = countElems * 8;
        const bg = this.device.createBindGroup({
          layout: this.sliceWrite.bgl,
          entries: [
            { binding: 0, resource: { buffer: lhsBuf, offset: 0, size: bytes } },
            { binding: 1, resource: { buffer: rhsBuf, offset: 0, size: bytes } },
            { binding: 2, resource: { buffer: dstBuf, offset: 0, size: bytes } },
            { binding: 3, resource: { buffer: this.sliceParams, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.sliceWrite.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(countElems / this.workgroupSize), 1, 1);
        pass.end();
      }
    
      _runSliceMulInPlace(commandEncoder, lhsBuf, rhsBuf, countElems) {
        this.device.queue.writeBuffer(this.sliceParams, 0, new Uint32Array([countElems, 0, 0, 0]));
        const bytes = countElems * 8;
        const bg = this.device.createBindGroup({
          layout: this.sliceMul.bgl,
          entries: [
            { binding: 0, resource: { buffer: lhsBuf, offset: 0, size: bytes } },
            { binding: 1, resource: { buffer: rhsBuf, offset: 0, size: bytes } },
            { binding: 2, resource: { buffer: this.sliceParams, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.sliceMul.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(countElems / this.workgroupSize), 1, 1);
        pass.end();
      }
    
      _execLineSliced(commandEncoder, { dataBuf, lineDataOffsetBytes, workBuf, workOffsetBytes, scratch }) {
        const lineBytes = this.N * 8;
        const workLineBytes = this.M * 8;
        const maxSliceElems = Math.max(1, this._maxSliceElems);
        const maxSliceBytes = maxSliceElems * 8;
        this._ensureSliceBuffers(maxSliceBytes);
        this._zeroWorkRange(commandEncoder, workBuf, workOffsetBytes, workLineBytes);
    
        for (let t0 = 0; t0 < this.N; t0 += maxSliceElems) {
          const n = Math.min(maxSliceElems, this.N - t0);
          const bytes = n * 8;
          commandEncoder.copyBufferToBuffer(dataBuf, lineDataOffsetBytes + t0 * 8, this._sliceInputBuffer, 0, bytes);
          commandEncoder.copyBufferToBuffer(this.chirpABuf, t0 * 8, this._sliceTwiddleBuffer, 0, bytes);
          this._runSliceWrite(commandEncoder, this._sliceInputBuffer, this._sliceTwiddleBuffer, this._sliceOutputBuffer, n);
          commandEncoder.copyBufferToBuffer(this._sliceOutputBuffer, 0, workBuf, workOffsetBytes + t0 * 8, bytes);
        }
    
        this.fftFwd.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOffsetBytes, batch: 1, temp: scratch });
    
        for (let k0 = 0; k0 < this.M; k0 += maxSliceElems) {
          const n = Math.min(maxSliceElems, this.M - k0);
          const bytes = n * 8;
          commandEncoder.copyBufferToBuffer(workBuf, workOffsetBytes + k0 * 8, this._sliceInputBuffer, 0, bytes);
          commandEncoder.copyBufferToBuffer(this.bfftBuf, k0 * 8, this._sliceTwiddleBuffer, 0, bytes);
          this._runSliceMulInPlace(commandEncoder, this._sliceInputBuffer, this._sliceTwiddleBuffer, n);
          commandEncoder.copyBufferToBuffer(this._sliceInputBuffer, 0, workBuf, workOffsetBytes + k0 * 8, bytes);
        }
    
        this.fftInv.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOffsetBytes, batch: 1, temp: scratch });
    
        for (let t0 = 0; t0 < this.N; t0 += maxSliceElems) {
          const n = Math.min(maxSliceElems, this.N - t0);
          const bytes = n * 8;
          commandEncoder.copyBufferToBuffer(workBuf, workOffsetBytes + t0 * 8, this._sliceInputBuffer, 0, bytes);
          commandEncoder.copyBufferToBuffer(this.chirpCBuf, t0 * 8, this._sliceTwiddleBuffer, 0, bytes);
          this._runSliceWrite(commandEncoder, this._sliceInputBuffer, this._sliceTwiddleBuffer, this._sliceOutputBuffer, n);
          commandEncoder.copyBufferToBuffer(this._sliceOutputBuffer, 0, dataBuf, lineDataOffsetBytes + t0 * 8, bytes);
        }
    
        if (lineBytes > this._bindBudgetBytes || workLineBytes > this._bindBudgetBytes) {
          // Explicit marker for callers/tests to detect this bounded path.
          this._usedSlicedLinePath = true;
        }
      }
    
      destroy() {
        this.chirpABuf.destroy();
        this.chirpCBuf.destroy();
        this.bfftBuf.destroy();
        this.paramsPre.destroy();
        this.paramsMul.destroy();
        this.paramsPost.destroy();
        this.sliceParams.destroy();
        this._sliceInputBuffer?.destroy?.();
        this._sliceTwiddleBuffer?.destroy?.();
        this._sliceOutputBuffer?.destroy?.();
        this._sliceZeroBuffer?.destroy?.();
        for (const b of this._retiredSliceBuffers) b?.destroy?.();
        for (const b of this._retiredParamBuffers) b?.destroy?.();
        this.fftFwd.destroy();
        this.fftInv.destroy();
      }
    
      exec(commandEncoder, { dataBuf, dataOffsetBytes, axisWork, scratch, lineCount = this.lines, paramChunkBase = 0 }) {
        if (!Number.isInteger(lineCount) || lineCount < 1 || lineCount > this.lines) {
          throw new Error(`BluesteinAxis.exec lineCount must be in [1, ${this.lines}], got ${lineCount}`);
        }
        if (!Number.isInteger(paramChunkBase) || paramChunkBase < 0) {
          throw new Error(`BluesteinAxis.exec paramChunkBase must be a non-negative integer, got ${paramChunkBase}`);
        }
        // axisWork: GPUBuffer|BufferView, size >= workBytes
        const workRange = normalizeToContiguousRanges(axisWork, 0, this.workBytes)[0];
        const workBuf = workRange.buffer;
        const workOff = workRange.offsetBytes;
        const lineBytes = this.N * 8;
        const workLineBytes = this.M * 8;
        const needsSlicedLinePath = lineBytes > this._bindBudgetBytes || workLineBytes > this._bindBudgetBytes;
        if (needsSlicedLinePath) {
          if (!this.supportsBoundedLineSlicing()) {
            throw new Error(
              `Bluestein bounded-line slicing currently requires contiguous axis lines (strideComplex=1), got strideComplex=${this._strideComplex}`
            );
          }
          if (workRange.sizeBytes < workLineBytes) {
            throw new Error(`Bluestein axisWork is too small for sliced-line execution: need ${workLineBytes}, got ${workRange.sizeBytes}`);
          }
          this._usedSlicedLinePath = false;
          for (let line = 0; line < lineCount; line++) {
            const lineDataOffsetBytes = dataOffsetBytes + line * lineBytes;
            this._execLineSliced(commandEncoder, {
              dataBuf,
              lineDataOffsetBytes,
              workBuf,
              workOffsetBytes: workOff,
              scratch,
            });
          }
          return 0;
        }
    
        const dataSize = lineCount * this.N * 8;
        const chunkCount = Math.ceil(lineCount / this.maxChunkLines);
        this._ensureParamCapacity(paramChunkBase + chunkCount);
        let chunkIndex = 0;
        for (let line0 = 0; line0 < lineCount; line0 += this.maxChunkLines) {
          const lines = Math.min(this.maxChunkLines, lineCount - line0);
          const chunkWorkBytes = lines * this._workBytesPerLine;
          const paramOff = (paramChunkBase + chunkIndex) * this._paramStride;
          if (paramOff + 16 > this.paramsPre.size || paramOff + 16 > this.paramsPost.size || paramOff + 16 > this.paramsMul.size) {
            throw new Error("BluesteinAxis.exec parameter buffer overflow; increase chunk-parameter capacity");
          }
          this.device.queue.writeBuffer(this.paramsPre, paramOff, new Uint32Array([lines, line0, 0, 0]));
          this.device.queue.writeBuffer(this.paramsPost, paramOff, new Uint32Array([lines, line0, 0, 0]));
          this.device.queue.writeBuffer(this.paramsMul, paramOff, new Uint32Array([lines * this.M, 0, 0, 0]));
    
          // pre
          const bgPre = this.device.createBindGroup({
            layout: this.bglPre,
            entries: [
              { binding: 0, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
              { binding: 1, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
              { binding: 2, resource: { buffer: this.chirpABuf, offset: 0, size: this.N * 8 } },
              { binding: 3, resource: { buffer: this.paramsPre, offset: paramOff, size: 16 } },
            ],
          });
          {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.prePipe);
            pass.setBindGroup(0, bgPre);
            pass.dispatchWorkgroups(Math.ceil((lines * this.M) / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          // FFT fwd (in-place on workBuf)
          this.fftFwd.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOff, batch: lines, temp: scratch });
    
          // mul
          const bgMul = this.device.createBindGroup({
            layout: this.bglMul,
            entries: [
              { binding: 0, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
              { binding: 1, resource: { buffer: this.bfftBuf, offset: 0, size: this.M * 8 } },
              { binding: 2, resource: { buffer: this.paramsMul, offset: paramOff, size: 16 } },
            ],
          });
          {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.mulPipe);
            pass.setBindGroup(0, bgMul);
            pass.dispatchWorkgroups(Math.ceil((lines * this.M) / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          // FFT inv
          this.fftInv.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOff, batch: lines, temp: scratch });
    
          // post
          const bgPost = this.device.createBindGroup({
            layout: this.bglPost,
            entries: [
              { binding: 0, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
              { binding: 1, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
              { binding: 2, resource: { buffer: this.chirpCBuf, offset: 0, size: this.N * 8 } },
              { binding: 3, resource: { buffer: this.paramsPost, offset: paramOff, size: 16 } },
            ],
          });
          {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.postPipe);
            pass.setBindGroup(0, bgPost);
            pass.dispatchWorkgroups(Math.ceil((lines * this.N) / this.workgroupSize), 1, 1);
            pass.end();
          }
          chunkIndex += 1;
        }
        return chunkIndex;
      }
    }
    
    exports['BluesteinAxis'] = BluesteinAxis;
  });

  __define('src/runtime/algorithms/rader_axis.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { createFftPlan } = require('src/plan.js');
    const { isPrime, primitiveRootPrime, modPow, factorizeSupportedRadices, nextPow2, nextSmoothAtLeast } = require('src/utils/factors.js');
    const { ensureWithinBindingLimit, prod, alignBytes } = require('src/runtime/common.js');
    const { normalizeToContiguousRanges } = require('src/runtime/segmented_io.js');
    
    const { generateRaderSumWGSL, generateRaderPackARevWGSL, generateRaderMulBfftWGSL, generateRaderWriteY0WGSL, generateRaderPostWGSL } = require('src/kernels/rader.js');
    
    class RaderAxis {
      constructor(device, cache, { shape, rank, batch, axis, direction, workgroupSize, maxWorkBytes = null }) {
        this.device = device;
        this.cache = cache;
        this.shape = shape;
        this.rank = rank;
        this.batch = batch;
        this.axis = axis;
        this.direction = direction;
        this.workgroupSize = workgroupSize;
    
        this.N = shape[axis];
        if (!isPrime(this.N)) throw new Error(`Rader requires prime N, got ${this.N}`);
        // Forward DFT uses exp(-i*2π/N); inverse uses exp(+i*2π/N).
        // This sign drives the Rader "b" sequence twiddles.
        const sign = direction === "forward" ? 1.0 : -1.0;
        this.L = this.N - 1;
        const M0 = nextSmoothAtLeast(2 * this.L - 1);
        this.M = factorizeSupportedRadices(M0) ? M0 : nextPow2(2 * this.L - 1);
        if (!factorizeSupportedRadices(this.M)) throw new Error(`Rader internal M=${this.M} not factorable by supported radices`);
    
        this.logicalTotal = prod(shape);
        this.lines = batch * (this.logicalTotal / this.N);
        this._workBytesPerLine = this.M * 8;
        const deviceMaxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
        const chunkBudget = maxWorkBytes == null ? deviceMaxBind : Math.min(deviceMaxBind, maxWorkBytes);
        this.maxChunkLines = Math.max(1, Math.floor(chunkBudget / this._workBytesPerLine));
        this.maxChunkLines = Math.min(this.maxChunkLines, this.lines);
        this.workBytes = this.maxChunkLines * this._workBytesPerLine;
        this._maxChunkCount = Math.max(1, Math.ceil(this.lines / this.maxChunkLines));
        this._paramStride = alignBytes(16, device.limits?.minUniformBufferOffsetAlignment ?? 256);
        this._paramCapacity = this._maxChunkCount;
        this._retiredParamBuffers = [];
        ensureWithinBindingLimit(device, this.workBytes, `Rader work buffer: N=${this.N} M=${this.M} lines=${this.lines}`);
    
        const g = primitiveRootPrime(this.N);
        const perm = new Uint32Array(this.L);
        for (let k = 0; k < this.L; k++) perm[k] = modPow(g, k + 1, this.N);
        this.permBuf = device.createBuffer({ size: perm.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.permBuf, 0, perm);
    
        const b = new Float32Array(2 * this.M);
        for (let k = 0; k < this.L; k++) {
          const ang = sign * (-2.0 * Math.PI * perm[k] / this.N);
          b[2 * k] = Math.cos(ang);
          b[2 * k + 1] = Math.sin(ang);
        }
        this.bfftBuf = device.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        device.queue.writeBuffer(this.bfftBuf, 0, b);
    
        this.fftFwd = createFftPlan(device, { shape: [this.M], direction: "forward", normalize: "none", inPlace: true, layout: "interleaved", precision: "f32" });
        this.fftInv = createFftPlan(device, { shape: [this.M], direction: "inverse", normalize: "backward", inPlace: true, layout: "interleaved", precision: "f32" });
    
        // compute bfft once
        {
          const enc = device.createCommandEncoder();
          this.fftFwd.exec(enc, { input: this.bfftBuf, batch: 1 });
          device.queue.submit([enc.finish()]);
        }
    
        let strideComplex = 1;
        for (let d = 0; d < axis; d++) strideComplex *= shape[d];
    
        this.sumBuf = device.createBuffer({ size: this.maxChunkLines * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.x0Buf = device.createBuffer({ size: this.maxChunkLines * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    
        const sumCode = generateRaderSumWGSL({ rank, axis, dims: shape, axisLength: this.N, strideComplex, workgroupSize: 256 });
        const packCode = generateRaderPackARevWGSL({ rank, axis, dims: shape, axisLength: this.N, mLength: this.M, strideComplex, workgroupSize });
        const mulCode = generateRaderMulBfftWGSL({ mLength: this.M, workgroupSize });
        const y0Code = generateRaderWriteY0WGSL({ rank, axis, dims: shape, axisLength: this.N, strideComplex, workgroupSize });
        const postCode = generateRaderPostWGSL({ rank, axis, dims: shape, axisLength: this.N, mLength: this.M, strideComplex, workgroupSize });
    
        this.bglSum = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        this.bglPack = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        this.bglMul = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        this.bglY0 = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        this.bglPost = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        this.plSum = device.createPipelineLayout({ bindGroupLayouts: [this.bglSum] });
        this.plPack = device.createPipelineLayout({ bindGroupLayouts: [this.bglPack] });
        this.plMul = device.createPipelineLayout({ bindGroupLayouts: [this.bglMul] });
        this.plY0 = device.createPipelineLayout({ bindGroupLayouts: [this.bglY0] });
        this.plPost = device.createPipelineLayout({ bindGroupLayouts: [this.bglPost] });
    
        this.sumPipe = cache.getComputePipeline({ code: sumCode, layout: this.plSum });
        this.packPipe = cache.getComputePipeline({ code: packCode, layout: this.plPack });
        this.mulPipe = cache.getComputePipeline({ code: mulCode, layout: this.plMul });
        this.y0Pipe = cache.getComputePipeline({ code: y0Code, layout: this.plY0 });
        this.postPipe = cache.getComputePipeline({ code: postCode, layout: this.plPost });
    
        this.paramsLines = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.paramsLines, 0, new Uint32Array([this.maxChunkLines, 0, 0, 0]));
        this.paramsMul = device.createBuffer({ size: this._paramCapacity * this._paramStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.paramsMul, 0, new Uint32Array([this.maxChunkLines * this.M, 0, 0, 0]));
      }
    
      _ensureParamCapacity(requiredChunkCount) {
        if (requiredChunkCount <= this._paramCapacity) return;
        let nextCapacity = this._paramCapacity;
        while (nextCapacity < requiredChunkCount) nextCapacity *= 2;
        const nextBytes = nextCapacity * this._paramStride;
        this._retiredParamBuffers.push(this.paramsLines, this.paramsMul);
        this.paramsLines = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.paramsMul = this.device.createBuffer({ size: nextBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._paramCapacity = nextCapacity;
      }
    
      destroy() {
        this.permBuf.destroy();
        this.bfftBuf.destroy();
        this.sumBuf.destroy();
        this.x0Buf.destroy();
        this.paramsLines.destroy();
        this.paramsMul.destroy();
        for (const b of this._retiredParamBuffers) b?.destroy?.();
        this.fftFwd.destroy();
        this.fftInv.destroy();
      }
    
      exec(commandEncoder, { dataBuf, dataOffsetBytes, axisWork, scratch, lineCount = this.lines, paramChunkBase = 0 }) {
        if (!Number.isInteger(lineCount) || lineCount < 1 || lineCount > this.lines) {
          throw new Error(`RaderAxis.exec lineCount must be in [1, ${this.lines}], got ${lineCount}`);
        }
        if (!Number.isInteger(paramChunkBase) || paramChunkBase < 0) {
          throw new Error(`RaderAxis.exec paramChunkBase must be a non-negative integer, got ${paramChunkBase}`);
        }
        const workRange = normalizeToContiguousRanges(axisWork, 0, this.workBytes)[0];
        const workBuf = workRange.buffer;
        const workOff = workRange.offsetBytes;
        const dataSize = lineCount * this.N * 8;
        const chunkCount = Math.ceil(lineCount / this.maxChunkLines);
        this._ensureParamCapacity(paramChunkBase + chunkCount);
    
        let chunkIndex = 0;
        for (let line0 = 0; line0 < lineCount; line0 += this.maxChunkLines) {
          const lines = Math.min(this.maxChunkLines, lineCount - line0);
          const chunkWorkBytes = lines * this._workBytesPerLine;
          const chunkLineBytes = lines * 8;
          const paramOff = (paramChunkBase + chunkIndex) * this._paramStride;
          if (paramOff + 16 > this.paramsLines.size || paramOff + 16 > this.paramsMul.size) {
            throw new Error("RaderAxis.exec parameter buffer overflow; increase chunk-parameter capacity");
          }
          this.device.queue.writeBuffer(this.paramsLines, paramOff, new Uint32Array([lines, line0, 0, 0]));
          this.device.queue.writeBuffer(this.paramsMul, paramOff, new Uint32Array([lines * this.M, 0, 0, 0]));
    
          // sum + x0
          {
            const bg = this.device.createBindGroup({
              layout: this.bglSum,
              entries: [
                { binding: 0, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
                { binding: 1, resource: { buffer: this.sumBuf, offset: 0, size: chunkLineBytes } },
                { binding: 2, resource: { buffer: this.x0Buf, offset: 0, size: chunkLineBytes } },
                { binding: 3, resource: { buffer: this.paramsLines, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.sumPipe);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(lines, 1, 1);
            pass.end();
          }
    
          // pack a_rev
          {
            const bg = this.device.createBindGroup({
              layout: this.bglPack,
              entries: [
                { binding: 0, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
                { binding: 1, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
                { binding: 2, resource: { buffer: this.permBuf, offset: 0, size: this.L * 4 } },
                { binding: 3, resource: { buffer: this.paramsLines, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.packPipe);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil((lines * this.M) / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          // FFT fwd
          this.fftFwd.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOff, batch: lines, temp: scratch });
    
          // mul
          {
            const bg = this.device.createBindGroup({
              layout: this.bglMul,
              entries: [
                { binding: 0, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
                { binding: 1, resource: { buffer: this.bfftBuf, offset: 0, size: this.M * 8 } },
                { binding: 2, resource: { buffer: this.paramsMul, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.mulPipe);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil((lines * this.M) / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          // FFT inv
          this.fftInv.exec(commandEncoder, { input: workBuf, inputOffsetBytes: workOff, batch: lines, temp: scratch });
    
          // y0
          {
            const bg = this.device.createBindGroup({
              layout: this.bglY0,
              entries: [
                { binding: 0, resource: { buffer: this.sumBuf, offset: 0, size: chunkLineBytes } },
                { binding: 1, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
                { binding: 2, resource: { buffer: this.paramsLines, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.y0Pipe);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(lines / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          // post
          {
            const bg = this.device.createBindGroup({
              layout: this.bglPost,
              entries: [
                { binding: 0, resource: { buffer: workBuf, offset: workOff, size: chunkWorkBytes } },
                { binding: 1, resource: { buffer: this.x0Buf, offset: 0, size: chunkLineBytes } },
                { binding: 2, resource: { buffer: this.permBuf, offset: 0, size: this.L * 4 } },
                { binding: 3, resource: { buffer: dataBuf, offset: dataOffsetBytes, size: dataSize } },
                { binding: 4, resource: { buffer: this.paramsLines, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.postPipe);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil((lines * (this.N - 1)) / this.workgroupSize), 1, 1);
            pass.end();
          }
          chunkIndex += 1;
        }
        return chunkIndex;
      }
    }
    
    exports['RaderAxis'] = RaderAxis;
  });

  __define('src/runtime/base_plan.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { getOrCreatePipelineCache } = require('src/runtime/pipeline_cache.js');
    const { SegmentedCopier } = require('src/runtime/segmented_io.js');
    const { pickWorkgroupSizeX } = require('src/utils/limits.js');
    
    function parseRequestedWorkgroupSize(limits, tuning) {
      const fallback = pickWorkgroupSizeX(limits, 256);
      if (!tuning || typeof tuning !== "object") return fallback;
    
      const requested = tuning.workgroupSizeX ?? tuning.workgroupSize ?? null;
      if (requested == null) return fallback;
      if (!Number.isInteger(requested) || requested <= 0) {
        throw new Error(`tuning.workgroupSizeX/workgroupSize must be a positive integer; got ${requested}`);
      }
    
      const maxX = limits?.maxComputeWorkgroupSizeX ?? fallback;
      const maxInv = limits?.maxComputeInvocationsPerWorkgroup ?? fallback;
      if (requested > maxX || requested > maxInv) {
        throw new Error(
          [
            `Requested workgroup size ${requested} exceeds device limits.`,
            `maxComputeWorkgroupSizeX=${maxX}`,
            `maxComputeInvocationsPerWorkgroup=${maxInv}`,
          ].join("\n")
        );
      }
      return requested | 0;
    }
    
    class BasePlan {
      constructor(device, opts = null) {
        this.device = device;
        const snapshot = opts?.cache?.snapshot ?? opts?.pipelineCacheSnapshot ?? null;
        this.cache = getOrCreatePipelineCache(device, { snapshot });
        this.copier = new SegmentedCopier(device, this.cache);
        this.workgroupSize = parseRequestedWorkgroupSize(device.limits, opts?.tuning ?? null);
        this._destroyed = false;
      }
    
      getWorkspaceSizeBytes() {
        return 0;
      }
    
      getPipelineCacheSnapshot() {
        return this.cache.exportSnapshot();
      }
    
      destroy() {
        if (this._destroyed) return;
        this._destroyed = true;
        this.copier.destroy();
      }
    }
    
    exports['BasePlan'] = BasePlan;
  });

  __define('src/runtime/common.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { formatDeviceLimits } = require('src/utils/limits.js');
    
    function assertOneOf(value, allowed, name) {
      if (!allowed.includes(value)) {
        throw new Error(
          `${name} must be one of ${allowed.map((v) => JSON.stringify(v)).join(", ")}; got ${JSON.stringify(value)}`
        );
      }
    }
    
    function isPositiveInt(x) {
      return Number.isInteger(x) && x > 0;
    }
    
    function prod(arr) {
      let p = 1;
      for (const v of arr) p *= v;
      return p;
    }
    
    function align4Bytes(bytes) {
      if (!Number.isInteger(bytes) || bytes < 0) throw new Error(`align4Bytes expects a non-negative integer; got ${bytes}`);
      return (bytes + 3) & ~3;
    }
    
    function alignBytes(bytes, alignment) {
      if (!Number.isInteger(bytes) || bytes < 0) throw new Error(`alignBytes expects a non-negative integer; got ${bytes}`);
      if (!Number.isInteger(alignment) || alignment <= 0) throw new Error(`alignBytes expects a positive integer alignment; got ${alignment}`);
      const rem = bytes % alignment;
      return rem === 0 ? bytes : bytes + (alignment - rem);
    }
    
    function normalizeScaleFactor({ normalize, direction, nTotal }) {
      if (normalize === "none") return 1.0;
      if (normalize === "unitary") return 1.0 / Math.sqrt(nTotal);
      if (normalize === "backward") return direction === "inverse" ? 1.0 / nTotal : 1.0;
      throw new Error(`Unknown normalize mode: ${normalize}`);
    }
    
    function ensureWithinBindingLimit(device, bytes, context) {
      const maxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
      if (bytes > maxBind) {
        throw new Error(
          [
            `Unsupported: required binding ${bytes} bytes exceeds device.limits.maxStorageBufferBindingSize=${maxBind}`,
            context ?? "",
            `limits:\n${formatDeviceLimits(device.limits)}`,
          ].join("\n")
        );
      }
    }
    
    function isGpuBuffer(x) {
      return x && !x?.segments && typeof x.size === "number" && typeof x.destroy === "function";
    }
    
    function getBufferByteLength(x) {
      if (x && typeof x.size === "number") return x.size;
      if (x && typeof x.lengthBytes === "number") return x.lengthBytes;
      throw new Error("Expected GPUBuffer or BufferView");
    }
    
    function collectBackingBuffers(x, out = new Set()) {
      if (!x) return out;
      if (x.view) {
        collectBackingBuffers(x.view, out);
        return out;
      }
      if (x.buffer && typeof x.buffer.size === "number") {
        out.add(x.buffer);
        return out;
      }
      if (typeof x.size === "number" && typeof x.destroy === "function" && !x.segments) {
        out.add(x);
        return out;
      }
      const segs = x.segments;
      if (!Array.isArray(segs)) return out;
      for (const seg of segs) {
        if (seg?.buffer) out.add(seg.buffer);
      }
      return out;
    }
    
    function buffersAlias(a, b) {
      if (!a || !b) return false;
      const aa = collectBackingBuffers(a);
      const bb = collectBackingBuffers(b);
      for (const buf of aa) {
        if (bb.has(buf)) return true;
      }
      return false;
    }
    
    exports['assertOneOf'] = assertOneOf;
    exports['isPositiveInt'] = isPositiveInt;
    exports['prod'] = prod;
    exports['align4Bytes'] = align4Bytes;
    exports['alignBytes'] = alignBytes;
    exports['normalizeScaleFactor'] = normalizeScaleFactor;
    exports['ensureWithinBindingLimit'] = ensureWithinBindingLimit;
    exports['isGpuBuffer'] = isGpuBuffer;
    exports['getBufferByteLength'] = getBufferByteLength;
    exports['collectBackingBuffers'] = collectBackingBuffers;
    exports['buffersAlias'] = buffersAlias;
  });

  __define('src/runtime/create_plan.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { assertOneOf } = require('src/runtime/common.js');
    
    const { C2CPlan } = require('src/runtime/plans/c2c.js');
    const { R2CPlan } = require('src/runtime/plans/r2c.js');
    const { C2RPlan } = require('src/runtime/plans/c2r.js');
    const { DctPlan } = require('src/runtime/plans/dct_fft.js');
    const { Conv2dPlan } = require('src/runtime/plans/conv2d.js');
    const { FftConvPlan } = require('src/runtime/plans/fftconv.js');
    
    function createPlan(device, opts) {
      if (!device) throw new Error("createPlan requires a WebGPU device");
      const { type } = opts ?? {};
      assertOneOf(type, ["c2c", "r2c", "c2r", "dct1", "dct2", "dct3", "dct4", "dst1", "dst2", "dst3", "dst4", "conv2d", "fftconv"], "type");
    
      if (type === "c2c") return new C2CPlan(device, opts);
      if (type === "r2c") return new R2CPlan(device, opts);
      if (type === "c2r") return new C2RPlan(device, opts);
      if (type === "conv2d") return new Conv2dPlan(device, opts);
      if (type === "fftconv") return new FftConvPlan(device, opts);
      return new DctPlan(device, opts);
    }
    
    exports['createPlan'] = createPlan;
  });

  __define('src/runtime/fftconv_channel_lane_presets.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const FFTCONV_MODES = new Set(["convolution", "correlation"]);
    const FFTCONV_BOUNDARIES = new Set(["circular", "linear-full", "linear-same", "linear-valid"]);
    const FFTCONV_OUTPUT_LAYOUTS = new Set(["kernel-major", "batch-major"]);
    
    const CONFLICTING_LAYOUT_KEYS = new Set([
      "whdcn",
      "strides",
      "inputStrides",
      "outputStrides",
      "offsetElements",
      "inputOffsetElements",
      "outputOffsetElements",
      "batchStrideElements",
      "inputBatchStrideElements",
      "outputBatchStrideElements",
    ]);
    
    function hasOwn(obj, key) {
      return Object.prototype.hasOwnProperty.call(obj, key);
    }
    
    function assertPlainObject(value, name) {
      if (value == null || typeof value !== "object" || Array.isArray(value)) {
        throw new Error(`${name} must be an object`);
      }
    }
    
    function assertPositiveSafeInt(value, name) {
      if (!Number.isInteger(value) || value <= 0 || !Number.isSafeInteger(value)) {
        throw new Error(`${name} must be a positive safe integer`);
      }
    }
    
    function assertNonNegativeSafeInt(value, name) {
      if (!Number.isInteger(value) || value < 0 || !Number.isSafeInteger(value)) {
        throw new Error(`${name} must be a non-negative safe integer`);
      }
    }
    
    function assertEnum(value, allowed, name) {
      if (!allowed.has(value)) {
        throw new Error(`${name} must be one of ${Array.from(allowed).map((v) => JSON.stringify(v)).join(", ")}; got ${value}`);
      }
    }
    
    function checkedMul(a, b, name) {
      const v = a * b;
      if (!Number.isSafeInteger(v) || v < 0) {
        throw new Error(`${name} exceeds safe integer range`);
      }
      return v;
    }
    
    function resolveLogicalSpan(shape) {
      if (!Array.isArray(shape) || shape.length === 0) {
        throw new Error("shape must be a non-empty array");
      }
      let span = 1;
      for (let i = 0; i < shape.length; i++) {
        const dim = shape[i];
        assertPositiveSafeInt(dim, `shape[${i}]`);
        span = checkedMul(span, dim, "shape product");
      }
      return span;
    }
    
    function normalizeSideDescriptor(side, sideName, logicalSpan, { kernelCount, allowKernelStep }) {
      assertPlainObject(side, sideName);
      assertPositiveSafeInt(side.channels, `${sideName}.channels`);
      const channels = side.channels;
    
      const channelIndex = side.channelIndex ?? 0;
      assertNonNegativeSafeInt(channelIndex, `${sideName}.channelIndex`);
      if (channelIndex >= channels) {
        throw new Error(`${sideName}.channelIndex (${channelIndex}) must be < ${sideName}.channels (${channels})`);
      }
    
      const channelStrideElements = side.channelStrideElements ?? logicalSpan;
      assertPositiveSafeInt(channelStrideElements, `${sideName}.channelStrideElements`);
      if (channelStrideElements < logicalSpan) {
        throw new Error(`${sideName}.channelStrideElements must be >= logical span (${logicalSpan})`);
      }
    
      const defaultBatchStrideElements = checkedMul(channels, channelStrideElements, `${sideName}.batchStrideElements`);
      const batchStrideElements = side.batchStrideElements ?? defaultBatchStrideElements;
      assertPositiveSafeInt(batchStrideElements, `${sideName}.batchStrideElements`);
      if (batchStrideElements < defaultBatchStrideElements) {
        throw new Error(`${sideName}.batchStrideElements must be >= channels*channelStrideElements (${defaultBatchStrideElements})`);
      }
    
      const offsetElements = side.offsetElements ?? 0;
      assertNonNegativeSafeInt(offsetElements, `${sideName}.offsetElements`);
    
      let kernelStepChannels = 1;
      if (allowKernelStep) {
        kernelStepChannels = side.kernelStepChannels ?? 1;
        assertPositiveSafeInt(kernelStepChannels, `${sideName}.kernelStepChannels`);
        if (kernelCount > 1) {
          const maxChannelIndex = channelIndex + (kernelCount - 1) * kernelStepChannels;
          if (!Number.isSafeInteger(maxChannelIndex)) {
            throw new Error(`${sideName}.kernelStepChannels mapping exceeds safe integer range`);
          }
          if (maxChannelIndex >= channels) {
            throw new Error(
              `${sideName} does not fit kernelCount=${kernelCount}: max channel index ${maxChannelIndex} exceeds channels=${channels} ` +
                `(channelIndex=${channelIndex}, kernelStepChannels=${kernelStepChannels})`
            );
          }
        }
      }
    
      const desc = {
        channels,
        channelIndex,
        channelStrideElements,
        batchStrideElements,
        offsetElements,
      };
      if (allowKernelStep) {
        desc.kernelStepChannels = kernelStepChannels;
      }
      return desc;
    }
    
    function validateLayout(layout) {
      assertPlainObject(layout, "layout");
      if (hasOwn(layout, "interleavedComplex") && layout.interleavedComplex !== true) {
        throw new Error("layout.interleavedComplex must be true for fftconv channel-lane presets");
      }
      for (const key of CONFLICTING_LAYOUT_KEYS) {
        if (hasOwn(layout, key)) {
          throw new Error(`layout.${key} cannot be combined with fftConv.channelPolicy presets`);
        }
      }
    }
    
    function buildPreset(opts, forcedOutputLayout = null) {
      assertPlainObject(opts, "opts");
    
      const {
        shape,
        batch,
        kernelCount = 1,
        mode = "convolution",
        boundary = "circular",
        outputLayout = "kernel-major",
        input,
        output,
        layout = {},
      } = opts;
    
      const logicalSpan = resolveLogicalSpan(shape);
      assertPositiveSafeInt(batch, "batch");
      assertPositiveSafeInt(kernelCount, "kernelCount");
      assertEnum(mode, FFTCONV_MODES, "mode");
      assertEnum(boundary, FFTCONV_BOUNDARIES, "boundary");
      assertEnum(outputLayout, FFTCONV_OUTPUT_LAYOUTS, "outputLayout");
      validateLayout(layout);
    
      const finalOutputLayout = forcedOutputLayout ?? outputLayout;
      if (forcedOutputLayout != null) {
        assertEnum(forcedOutputLayout, FFTCONV_OUTPUT_LAYOUTS, "forcedOutputLayout");
      }
    
      const inputDesc = normalizeSideDescriptor(input, "input", logicalSpan, {
        kernelCount,
        allowKernelStep: false,
      });
      const outputDesc = normalizeSideDescriptor(output, "output", logicalSpan, {
        kernelCount,
        allowKernelStep: true,
      });
    
      return {
        shape: [...shape],
        batch,
        layout: {
          interleavedComplex: true,
          ...layout,
        },
        fftConv: {
          mode,
          boundary,
          kernelCount,
          outputLayout: finalOutputLayout,
          channelPolicy: {
            input: inputDesc,
            output: outputDesc,
          },
        },
      };
    }
    
    function createFftConvChannelLanePreset(opts) {
      return buildPreset(opts, null);
    }
    
    function createFftConvKernelMajorChannelLanePreset(opts) {
      return buildPreset(opts, "kernel-major");
    }
    
    function createFftConvBatchMajorChannelLanePreset(opts) {
      return buildPreset(opts, "batch-major");
    }
    
    
    exports['createFftConvChannelLanePreset'] = createFftConvChannelLanePreset;
    exports['createFftConvKernelMajorChannelLanePreset'] = createFftConvKernelMajorChannelLanePreset;
    exports['createFftConvBatchMajorChannelLanePreset'] = createFftConvBatchMajorChannelLanePreset;
  });

  __define('src/runtime/ioview.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    function isPositiveInt(x) {
      return Number.isInteger(x) && x > 0;
    }
    
    function normalizeIoView(rank, logicalShape, ioView = {}) {
      const normOne = (v, kind) => {
        if (!v) return null;
        const shape = v.shape ?? null;
        if (!Array.isArray(shape) || shape.length !== rank || !shape.every(isPositiveInt)) {
          throw new Error(`ioView.${kind}.shape must be an array of ${rank} positive ints`);
        }
        const placement = v.placement ?? "start";
        if (placement !== "start" && placement !== "center") {
          throw new Error(`ioView.${kind}.placement must be "start"|"center"`);
        }
        let offset = v.offset ?? null;
        if (offset != null) {
          if (!Array.isArray(offset) || offset.length !== rank || !offset.every((x) => Number.isInteger(x))) {
            throw new Error(`ioView.${kind}.offset must be an array of ${rank} integers`);
          }
        } else if (placement === "center") {
          offset = shape.map((s, d) => Math.floor((logicalShape[d] - s) / 2));
        } else {
          offset = new Array(rank).fill(0);
        }
        const clearOutside = kind === "output" ? !!v.clearOutside : false;
        return { shape, placement, offset, clearOutside };
      };
    
      return {
        input: normOne(ioView.input, "input"),
        output: normOne(ioView.output, "output"),
      };
    }
    
    
    exports['normalizeIoView'] = normalizeIoView;
  });

  __define('src/runtime/large_policy.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { factorizeSupportedRadices, isPrime } = require('src/utils/factors.js');
    
    function parseOptionalMaxStorageBufferBindingSize(tuning) {
      if (tuning == null || typeof tuning !== "object") return null;
      const v = tuning.maxStorageBufferBindingSize ?? null;
      if (v == null) return null;
      if (!Number.isInteger(v) || v <= 0) {
        throw new Error(`tuning.maxStorageBufferBindingSize must be a positive integer; got ${v}`);
      }
      return v;
    }
    
    function resolveEffectiveMaxStorageBufferBindingSize(device, tuning) {
      const deviceMaxBind = device.limits?.maxStorageBufferBindingSize ?? Infinity;
      const tuningMaxBind = parseOptionalMaxStorageBufferBindingSize(tuning);
      return tuningMaxBind != null ? Math.min(deviceMaxBind, tuningMaxBind) : deviceMaxBind;
    }
    
    function parseOptionalLargeRoutePreference(tuning) {
      if (tuning == null) return "auto";
      if (typeof tuning !== "object") {
        throw new Error("tuning must be an object when provided");
      }
      const value = tuning.largeRoute ?? "auto";
      if (typeof value !== "string") {
        throw new Error(`tuning.largeRoute must be one of "auto" | "chunk" | "out-of-core"; got ${value}`);
      }
      if (value !== "auto" && value !== "chunk" && value !== "out-of-core") {
        throw new Error(`tuning.largeRoute must be one of "auto" | "chunk" | "out-of-core"; got ${value}`);
      }
      return value;
    }
    
    function parseOptionalPreferOutOfCoreForStrided(tuning) {
      if (tuning == null || typeof tuning !== "object") return null;
      if (tuning.preferOutOfCoreForStrided == null) return null;
      if (typeof tuning.preferOutOfCoreForStrided !== "boolean") {
        throw new Error(
          `tuning.preferOutOfCoreForStrided must be boolean when provided; got ${tuning.preferOutOfCoreForStrided}`
        );
      }
      return tuning.preferOutOfCoreForStrided;
    }
    
    function parseOptionalNonNegativeInt(v, name) {
      if (v == null) return null;
      if (!Number.isInteger(v) || v < 0) {
        throw new Error(`${name} must be a non-negative integer when provided; got ${v}`);
      }
      return v;
    }
    
    function parseOptionalPositiveInt(v, name) {
      if (v == null) return null;
      if (!Number.isInteger(v) || v <= 0) {
        throw new Error(`${name} must be a positive integer when provided; got ${v}`);
      }
      return v;
    }
    
    function resolveGroupedBatchValue(tuning, axisIndex = 0) {
      if (tuning == null || typeof tuning !== "object") return null;
      const groupedBatch = tuning.groupedBatch ?? null;
      if (groupedBatch == null) return null;
      if (Number.isInteger(groupedBatch) && groupedBatch > 0) return groupedBatch;
      if (Array.isArray(groupedBatch)) {
        if (!Number.isInteger(axisIndex) || axisIndex < 0) {
          throw new Error(`axisIndex must be a non-negative integer; got ${axisIndex}`);
        }
        if (groupedBatch.length === 0) return null;
        const idx = Math.min(axisIndex, groupedBatch.length - 1);
        const v = groupedBatch[idx];
        if (v == null) return null;
        if (!Number.isInteger(v) || v <= 0) {
          throw new Error(`tuning.groupedBatch[${idx}] must be a positive integer when provided; got ${v}`);
        }
        return v;
      }
      throw new Error(`tuning.groupedBatch must be a positive integer or axis-indexed array; got ${groupedBatch}`);
    }
    
    function gcd(a, b) {
      let x = Math.abs(a | 0);
      let y = Math.abs(b | 0);
      while (y !== 0) {
        const t = x % y;
        x = y;
        y = t;
      }
      return x || 1;
    }
    
    function resolveOutOfCoreAxisWindowPolicy({
      axisLen,
      lineBytes,
      linesTotal,
      maxBindBytes,
      axisKind = "mixed",
      tuning = null,
      axisIndex = 0,
      storageAlign = 256,
    }) {
      if (!Number.isInteger(axisLen) || axisLen <= 0) {
        throw new Error(`axisLen must be a positive integer; got ${axisLen}`);
      }
      if (!Number.isInteger(lineBytes) || lineBytes <= 0) {
        throw new Error(`lineBytes must be a positive integer; got ${lineBytes}`);
      }
      if (!Number.isInteger(linesTotal) || linesTotal <= 0) {
        throw new Error(`linesTotal must be a positive integer; got ${linesTotal}`);
      }
      const effectiveMaxBind = Number.isFinite(maxBindBytes) && maxBindBytes > 0 ? Math.floor(maxBindBytes) : Infinity;
      const maxLinesByBind = lineBytes <= effectiveMaxBind ? Math.max(1, Math.floor(effectiveMaxBind / lineBytes)) : 1;
      const swapTo2Stage4Step = parseOptionalNonNegativeInt(tuning?.swapTo2Stage4Step ?? null, "tuning.swapTo2Stage4Step") ?? 0;
      const swapTo3Stage4Step = parseOptionalNonNegativeInt(tuning?.swapTo3Stage4Step ?? null, "tuning.swapTo3Stage4Step") ?? 0;
      const burstWindows = parseOptionalPositiveInt(tuning?.outOfCoreBurstWindows ?? null, "tuning.outOfCoreBurstWindows") ?? 1;
      const groupedBatch = resolveGroupedBatchValue(tuning, axisIndex);
    
      let numAxisUploads = 1;
      if (swapTo3Stage4Step > 0 && axisLen >= swapTo3Stage4Step) {
        numAxisUploads = 3;
      } else if (swapTo2Stage4Step > 0 && axisLen >= swapTo2Stage4Step) {
        numAxisUploads = 2;
      } else {
        // Conservative auto policy: keep legacy behavior for small/typical sizes; only split windows for very large strided/non-mixed lines.
        if (axisKind !== "mixed" && axisLen >= 1024 && maxLinesByBind >= 8) numAxisUploads = 2;
        if (axisKind !== "mixed" && axisLen >= 4096 && maxLinesByBind >= 16) numAxisUploads = 3;
      }
      numAxisUploads = Math.max(1, Math.min(3, numAxisUploads, maxLinesByBind));
    
      let linesPerChunk = Math.max(1, Math.floor(maxLinesByBind / numAxisUploads));
      if (groupedBatch != null && linesPerChunk > 1) {
        if (linesPerChunk >= groupedBatch) {
          linesPerChunk = Math.max(groupedBatch, Math.floor(linesPerChunk / groupedBatch) * groupedBatch);
        } else {
          linesPerChunk = 1;
        }
      }
    
      let alignedLineStep = 1;
      if (Number.isFinite(storageAlign) && storageAlign > 1) {
        alignedLineStep = Math.max(1, Math.floor(storageAlign / gcd(storageAlign, lineBytes)));
        if (alignedLineStep > 1 && linesPerChunk >= alignedLineStep) {
          linesPerChunk = Math.max(alignedLineStep, Math.floor(linesPerChunk / alignedLineStep) * alignedLineStep);
        }
      }
    
      linesPerChunk = Math.max(1, Math.min(linesPerChunk, linesTotal));
      return {
        axisKind,
        axisLen,
        lineBytes,
        linesTotal,
        maxLinesByBind,
        groupedBatch,
        numAxisUploads,
        linesPerChunk,
        alignedLineStep,
        burstWindows,
      };
    }
    
    function canAxisLenFitOrTwoStep(axisLen, maxBindBytes, maxBufferSize) {
      const lineBytes = axisLen * 8;
      if (lineBytes <= maxBindBytes) return true;
      if (lineBytes > maxBufferSize) return false;
      const maxAxisElems = Math.floor(maxBindBytes / 8);
      if (!Number.isInteger(maxAxisElems) || maxAxisElems < 2) return false;
      const root = Math.floor(Math.sqrt(axisLen));
      for (let d = 1; d <= root; d++) {
        if (axisLen % d !== 0) continue;
        const q = axisLen / d;
        if (d >= 2 && q >= 2 && d <= maxAxisElems && q <= maxAxisElems && factorizeSupportedRadices(d) && factorizeSupportedRadices(q)) {
          return true;
        }
        if (q >= 2 && d >= 2 && q <= maxAxisElems && d <= maxAxisElems && factorizeSupportedRadices(q) && factorizeSupportedRadices(d)) {
          return true;
        }
      }
      return false;
    }
    
    function parseOptionalAxisList(v, rank, name) {
      if (v == null) return [];
      if (!Array.isArray(v) || !v.every((x) => Number.isInteger(x) && x >= 0 && x < rank)) {
        throw new Error(`${name} must be an array of axis indices in [0, ${rank - 1}]`);
      }
      return [...new Set(v.map((x) => x | 0))];
    }
    
    function resolveAxisKindsForShape({ shape, tuning = null, defaultRaderMaxPrime = 4096 }) {
      if (!Array.isArray(shape) || shape.length < 1 || !shape.every((x) => Number.isInteger(x) && x > 0)) {
        throw new Error(`shape must be an array of positive integers; got ${JSON.stringify(shape)}`);
      }
      if (tuning != null && typeof tuning !== "object") {
        throw new Error("tuning must be an object when provided");
      }
      const rank = shape.length;
      const raderMaxPrime = tuning?.raderMaxPrime ?? defaultRaderMaxPrime;
      if (!Number.isInteger(raderMaxPrime) || raderMaxPrime < 2) {
        throw new Error(`tuning.raderMaxPrime must be an integer >= 2; got ${raderMaxPrime}`);
      }
      const forceBluesteinAxes = parseOptionalAxisList(tuning?.forceBluesteinAxes ?? null, rank, "tuning.forceBluesteinAxes");
      const forceRaderAxes = parseOptionalAxisList(tuning?.forceRaderAxes ?? null, rank, "tuning.forceRaderAxes");
      const forceBluesteinAxisSet = new Set(forceBluesteinAxes);
      const forceRaderAxisSet = new Set(forceRaderAxes);
      for (const axis of forceBluesteinAxisSet) {
        if (forceRaderAxisSet.has(axis)) {
          throw new Error(`Axis ${axis} cannot be forced to both Bluestein and Rader`);
        }
      }
      const axisKinds = shape.map((N, axis) => {
        if (forceBluesteinAxisSet.has(axis)) return "bluestein";
        if (forceRaderAxisSet.has(axis)) {
          if (!isPrime(N)) {
            throw new Error(`tuning.forceRaderAxes contains axis ${axis}, but shape[${axis}]=${N} is not prime`);
          }
          if (N > raderMaxPrime) {
            throw new Error(`tuning.forceRaderAxes contains axis ${axis}, but shape[${axis}]=${N} exceeds tuning.raderMaxPrime=${raderMaxPrime}`);
          }
          return "rader";
        }
        if (factorizeSupportedRadices(N)) return "mixed";
        if (isPrime(N) && N <= raderMaxPrime) return "rader";
        return "bluestein";
      });
      return {
        axisKinds,
        raderMaxPrime,
        forceBluesteinAxes,
        forceRaderAxes,
        forceBluesteinAxisSet,
        forceRaderAxisSet,
      };
    }
    
    function evaluateAxisSupport({ axisKinds, axisLengths, maxBindBytes, maxBufferSize, allowNonMixedBoundedSlicing }) {
      if (!Array.isArray(axisKinds) || !Array.isArray(axisLengths) || axisKinds.length !== axisLengths.length) {
        return null;
      }
      const supported = new Array(axisKinds.length).fill(false);
      for (let axis = 0; axis < axisKinds.length; axis++) {
        const kind = axisKinds[axis];
        const n = axisLengths[axis];
        if (kind === "mixed") {
          supported[axis] = canAxisLenFitOrTwoStep(n, maxBindBytes, maxBufferSize);
          continue;
        }
        const lineBytes = n * 8;
        supported[axis] = allowNonMixedBoundedSlicing ? lineBytes <= maxBufferSize : lineBytes <= maxBindBytes;
      }
      return supported;
    }
    
    function pushReason(reasonCodes, code) {
      if (!reasonCodes.includes(code)) reasonCodes.push(code);
    }
    
    function routeModePriority(routeMode) {
      switch (routeMode) {
        case "large-out-of-core":
          return 2;
        case "large-chunk":
          return 1;
        default:
          return 0;
      }
    }
    
    function pushUnique(list, value) {
      if (!list.includes(value)) list.push(value);
    }
    
    function mergeLargeRouteMetadata(entries) {
      const list = Array.isArray(entries) ? entries : [entries];
      const reasonCodes = [];
      const attemptedRoutes = [];
      let routeMode = "normal";
      for (const entry of list) {
        if (!entry || typeof entry !== "object") continue;
        const entryMode = typeof entry.routeMode === "string" ? entry.routeMode : null;
        if (entryMode && routeModePriority(entryMode) > routeModePriority(routeMode)) {
          routeMode = entryMode;
        }
        if (Array.isArray(entry.reasonCodes)) {
          for (const code of entry.reasonCodes) pushUnique(reasonCodes, code);
        }
        if (Array.isArray(entry.attemptedRoutes)) {
          for (const route of entry.attemptedRoutes) pushUnique(attemptedRoutes, route);
        }
      }
      if (routeMode === "large-out-of-core") {
        pushUnique(attemptedRoutes, "out-of-core-four-step");
      }
      pushUnique(reasonCodes, routeMode);
      return { routeMode, reasonCodes, attemptedRoutes };
    }
    
    function resolveLargeRoutingPolicy({
      device,
      tuning = null,
      requiredBindingBytes = [],
      lineBytes = [],
      axisKinds = null,
      axisLengths = null,
      allowNonMixedBoundedSlicing = false,
      precision = "f32",
      requireLargePrecision = null,
      requireLargePrecisionError = null,
      allowOutOfCore = false,
      disableOutOfCore = false,
      rank = 1,
      minOutOfCoreRank = 2,
      bytesPerBatch = null,
      hasStridedIO = false,
      preferOutOfCoreForStrided = false,
      outOfCoreUnsupportedError = null,
    }) {
      const maxBindBytes = resolveEffectiveMaxStorageBufferBindingSize(device, tuning);
      const maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
      const requestedLargeRoute = parseOptionalLargeRoutePreference(tuning);
      const tuningPreferOutOfCoreForStrided = parseOptionalPreferOutOfCoreForStrided(tuning);
      const effectivePreferOutOfCoreForStrided =
        tuningPreferOutOfCoreForStrided == null ? preferOutOfCoreForStrided : tuningPreferOutOfCoreForStrided;
      const reasonCodes = [];
      const attemptedRoutes = ["direct"];
      const needsLargeMode = requiredBindingBytes.some((bytes) => bytes > maxBindBytes);
      const oversizedLineMode = lineBytes.some((bytes) => bytes > maxBindBytes);
      if (!needsLargeMode) {
        pushReason(reasonCodes, "within-bindings");
      } else {
        pushReason(reasonCodes, "requires-large-bindings");
      }
      if (oversizedLineMode) pushReason(reasonCodes, "oversized-line-bindings");
      if (needsLargeMode) {
        attemptedRoutes.push("dispatch-split");
        attemptedRoutes.push("batch-chunk");
      }
      if (oversizedLineMode) attemptedRoutes.push("line-slice-or-two-step");
      const axisSupported = evaluateAxisSupport({
        axisKinds,
        axisLengths,
        maxBindBytes,
        maxBufferSize,
        allowNonMixedBoundedSlicing,
      });
    
      if (needsLargeMode && requireLargePrecision && precision !== requireLargePrecision) {
        pushReason(reasonCodes, "precision-restricted-large-mode");
        throw new Error(requireLargePrecisionError ?? `large-mode fallback currently supports precision:"${requireLargePrecision}" only`);
      }
    
      const outOfCoreEligible =
        allowOutOfCore &&
        !disableOutOfCore &&
        rank >= minOutOfCoreRank &&
        precision === "f32" &&
        (axisSupported == null || axisSupported.every((v) => v === true));
      if (axisSupported && axisSupported.some((v) => v === false)) {
        pushReason(reasonCodes, "axis-line-unsupported");
      }
      if (outOfCoreEligible) {
        pushReason(reasonCodes, "out-of-core-eligible");
      } else {
        pushReason(reasonCodes, "out-of-core-ineligible");
      }
    
      const requiresOutOfCore = allowOutOfCore && needsLargeMode && bytesPerBatch != null && bytesPerBatch > maxBindBytes;
      const prefersOutOfCore =
        allowOutOfCore &&
        needsLargeMode &&
        effectivePreferOutOfCoreForStrided &&
        hasStridedIO &&
        outOfCoreEligible;
      if (allowOutOfCore && needsLargeMode) attemptedRoutes.push("out-of-core-four-step");
      if (requiresOutOfCore) pushReason(reasonCodes, "bytes-per-batch-exceeds-bind");
      if (prefersOutOfCore) pushReason(reasonCodes, "strided-prefers-out-of-core");
    
      if (needsLargeMode && requestedLargeRoute === "chunk") {
        pushReason(reasonCodes, "forced-route-chunk");
      }
      if (needsLargeMode && requestedLargeRoute === "out-of-core") {
        pushReason(reasonCodes, "forced-route-out-of-core");
        attemptedRoutes.push("forced-out-of-core");
      }
    
      if (needsLargeMode && requestedLargeRoute === "chunk" && requiresOutOfCore) {
        throw new Error(
          `tuning.largeRoute="chunk" is incompatible: one batch requires ${bytesPerBatch} bytes > ` +
            `maxStorageBufferBindingSize=${maxBindBytes}, so out-of-core routing is required.`
        );
      }
    
      if (needsLargeMode && requestedLargeRoute === "out-of-core" && !allowOutOfCore) {
        throw new Error('tuning.largeRoute="out-of-core" requested, but out-of-core routing is not enabled for this plan');
      }
    
      if (needsLargeMode && requestedLargeRoute === "out-of-core" && !outOfCoreEligible) {
        throw new Error(
          `tuning.largeRoute="out-of-core" requested, but out-of-core eligibility constraints were not met ` +
            `(reasonCodes=${JSON.stringify(reasonCodes)}, axisSupported=${JSON.stringify(axisSupported)}).`
        );
      }
    
      if (requiresOutOfCore && !outOfCoreEligible) {
        throw new Error(
          typeof outOfCoreUnsupportedError === "function"
            ? outOfCoreUnsupportedError({
                maxBindBytes,
                maxBufferSize,
                axisSupported,
                outOfCoreEligible,
                reasonCodes: reasonCodes.slice(),
                attemptedRoutes: attemptedRoutes.slice(),
                requiresOutOfCore,
                prefersOutOfCore,
              })
            : outOfCoreUnsupportedError ??
                "large-mode requires out-of-core fallback, but out-of-core eligibility constraints were not met"
        );
      }
    
      const useOutOfCore =
        (needsLargeMode && requestedLargeRoute === "out-of-core") ||
        requiresOutOfCore ||
        (requestedLargeRoute !== "chunk" && prefersOutOfCore);
      const routeMode = needsLargeMode ? (useOutOfCore ? "large-out-of-core" : "large-chunk") : "normal";
      pushReason(reasonCodes, routeMode);
    
      return {
        maxBindBytes,
        maxBufferSize,
        needsLargeMode,
        oversizedLineMode,
        axisKinds: Array.isArray(axisKinds) ? axisKinds.slice() : null,
        axisLengths: Array.isArray(axisLengths) ? axisLengths.slice() : null,
        axisSupported,
        outOfCoreEligible,
        requiresOutOfCore,
        prefersOutOfCore,
        requestedLargeRoute,
        effectivePreferOutOfCoreForStrided,
        useOutOfCore,
        routeMode,
        reasonCodes,
        attemptedRoutes,
      };
    }
    
    exports['parseOptionalMaxStorageBufferBindingSize'] = parseOptionalMaxStorageBufferBindingSize;
    exports['resolveEffectiveMaxStorageBufferBindingSize'] = resolveEffectiveMaxStorageBufferBindingSize;
    exports['resolveOutOfCoreAxisWindowPolicy'] = resolveOutOfCoreAxisWindowPolicy;
    exports['canAxisLenFitOrTwoStep'] = canAxisLenFitOrTwoStep;
    exports['resolveAxisKindsForShape'] = resolveAxisKindsForShape;
    exports['mergeLargeRouteMetadata'] = mergeLargeRouteMetadata;
    exports['resolveLargeRoutingPolicy'] = resolveLargeRoutingPolicy;
  });

  __define('src/runtime/layout_semantics.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { prod } = require('src/runtime/common.js');
    const { contiguousStrides: tensorContiguousStrides, spanElements: tensorSpanElements } = require('src/runtime/tensor_descriptor.js');
    
    function hasOwn(obj, key) {
      return !!obj && Object.prototype.hasOwnProperty.call(obj, key);
    }
    
    function arraysEqual(a, b) {
      if (a === b) return true;
      if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
      }
      return true;
    }
    
    function parseOptionalPositiveIntArray(v, rank, name) {
      if (v == null) return null;
      if (!Array.isArray(v) || v.length !== rank || !v.every((x) => Number.isInteger(x) && x > 0)) {
        throw new Error(`${name} must be an array of ${rank} positive integers`);
      }
      return v.map((x) => x | 0);
    }
    
    function parseOptionalNonNegativeInt(v, name) {
      if (v == null) return null;
      if (!Number.isInteger(v) || v < 0) throw new Error(`${name} must be a non-negative integer`);
      return v | 0;
    }
    
    function parseOptionalPositiveInt(v, name) {
      if (v == null) return null;
      if (!Number.isInteger(v) || v <= 0) throw new Error(`${name} must be a positive integer`);
      return v | 0;
    }
    
    function checkedSafeInt(v, name) {
      if (!Number.isSafeInteger(v) || v < 0) {
        throw new Error(`${name} must stay within non-negative safe integer range`);
      }
      return v;
    }
    
    function contiguousStrides(shape) {
      return tensorContiguousStrides(shape);
    }
    
    function stridedSpanElements(shape, strides) {
      return tensorSpanElements(shape, strides);
    }
    
    function sideFieldName(side, suffix) {
      if (side === "input") return `input${suffix}`;
      return `output${suffix}`;
    }
    
    function hasExplicitSideLayout(layout, side) {
      return (
        hasOwn(layout, sideFieldName(side, "Strides")) ||
        hasOwn(layout, sideFieldName(side, "OffsetElements")) ||
        hasOwn(layout, sideFieldName(side, "BatchStrideElements")) ||
        hasOwn(layout, "strides") ||
        hasOwn(layout, "offsetElements") ||
        hasOwn(layout, "batchStrideElements")
      );
    }
    
    function resolveExplicitSideLayout(layout, side, rank, layoutShape) {
      const stridesName = side === "input" ? "layout.inputStrides/layout.strides" : "layout.outputStrides/layout.strides";
      const offsetName = side === "input" ? "layout.inputOffsetElements/layout.offsetElements" : "layout.outputOffsetElements/layout.offsetElements";
      const batchName = side === "input" ? "layout.inputBatchStrideElements/layout.batchStrideElements" : "layout.outputBatchStrideElements/layout.batchStrideElements";
      const stridesValue = layout?.[sideFieldName(side, "Strides")] ?? layout?.strides ?? null;
      const offsetValue = layout?.[sideFieldName(side, "OffsetElements")] ?? layout?.offsetElements ?? null;
      const batchStrideValue = layout?.[sideFieldName(side, "BatchStrideElements")] ?? layout?.batchStrideElements ?? null;
    
      const strides = parseOptionalPositiveIntArray(stridesValue, rank, stridesName);
      const offsetElements = parseOptionalNonNegativeInt(offsetValue, offsetName) ?? 0;
      const spanElements = strides ? stridedSpanElements(layoutShape, strides) : 0;
      const defaultBatchStrideElements = strides ? spanElements : prod(layoutShape);
      const batchStrideElements = parseOptionalNonNegativeInt(batchStrideValue, batchName) ?? defaultBatchStrideElements;
      if (strides && batchStrideElements < spanElements) {
        throw new Error(side === "input" ? "layout.inputBatchStrideElements is too small for layout.inputStrides" : "layout.outputBatchStrideElements is too small for layout.outputStrides");
      }
      // Keep contiguous-equivalent explicit descriptors on the direct contiguous route.
      if (strides) {
        const contiguous = contiguousStrides(layoutShape);
        const contiguousBatch = prod(layoutShape);
        const noOp = arraysEqual(strides, contiguous) && offsetElements === 0 && batchStrideElements === contiguousBatch;
        if (noOp) {
          return {
            strides: null,
            offsetElements: 0,
            batchStrideElements: contiguousBatch,
            spanElements: 0,
          };
        }
      }
      return {
        strides,
        offsetElements,
        batchStrideElements,
        spanElements,
      };
    }
    
    function mergeWhdcnDescriptor(globalDesc, sideDesc, sideName) {
      const g = globalDesc ?? {};
      const s = sideDesc ?? {};
      if (typeof g !== "object" || Array.isArray(g)) {
        throw new Error("layout.whdcn must be an object");
      }
      if (sideDesc != null && (typeof s !== "object" || Array.isArray(s))) {
        throw new Error(`layout.whdcn.${sideName} must be an object`);
      }
      return { ...g, ...s };
    }
    
    function resolveWhdcnSideLayout(desc, side, rank, layoutShape) {
      if (!desc) return null;
      if (hasOwn(desc, "enabled") && typeof desc.enabled !== "boolean") {
        throw new Error(`layout.whdcn.${side}.enabled must be boolean when provided`);
      }
      if (desc.enabled === false) return null;
    
      const hasAnyControls =
        hasOwn(desc, "strides") ||
        hasOwn(desc, "offsetElements") ||
        hasOwn(desc, "batchStrideElements") ||
        hasOwn(desc, "channels") ||
        hasOwn(desc, "channelIndex") ||
        hasOwn(desc, "channelStrideElements");
      if (!hasAnyControls) return null;
    
      const sidePath = `layout.whdcn.${side}`;
      const strides = parseOptionalPositiveIntArray(desc.strides ?? null, rank, `${sidePath}.strides`) ?? contiguousStrides(layoutShape);
      const spanElements = stridedSpanElements(layoutShape, strides);
      const channels = parseOptionalPositiveInt(desc.channels ?? null, `${sidePath}.channels`) ?? 1;
      const channelIndex = parseOptionalNonNegativeInt(desc.channelIndex ?? null, `${sidePath}.channelIndex`) ?? 0;
      if (channelIndex >= channels) {
        throw new Error(`${sidePath}.channelIndex (${channelIndex}) must be < ${sidePath}.channels (${channels})`);
      }
      const channelStrideElements = parseOptionalPositiveInt(desc.channelStrideElements ?? null, `${sidePath}.channelStrideElements`) ?? spanElements;
      if (channelStrideElements < spanElements) {
        throw new Error(`${sidePath}.channelStrideElements must be >= addressed span (${spanElements})`);
      }
    
      const baseOffset = parseOptionalNonNegativeInt(desc.offsetElements ?? null, `${sidePath}.offsetElements`) ?? 0;
      const offsetElements = checkedSafeInt(baseOffset + channelIndex * channelStrideElements, `${sidePath}.offsetElements`);
    
      const defaultBatchStride = checkedSafeInt(channelStrideElements * channels, `${sidePath}.batchStrideElements`);
      const batchStrideElements =
        parseOptionalNonNegativeInt(desc.batchStrideElements ?? null, `${sidePath}.batchStrideElements`) ?? defaultBatchStride;
      if (batchStrideElements < defaultBatchStride) {
        throw new Error(`${sidePath}.batchStrideElements must be >= channels*channelStrideElements (${defaultBatchStride})`);
      }
    
      const contiguous = contiguousStrides(layoutShape);
      const contiguousBatch = prod(layoutShape);
      const noOp =
        arraysEqual(strides, contiguous) &&
        offsetElements === 0 &&
        batchStrideElements === contiguousBatch &&
        channels === 1 &&
        channelIndex === 0 &&
        channelStrideElements === spanElements;
      if (noOp) return null;
    
      return {
        strides,
        offsetElements,
        batchStrideElements,
        spanElements,
      };
    }
    
    function resolveLayoutSemantics({ layout, rank, inputShape, outputShape }) {
      const l = layout ?? {};
      if (typeof l !== "object" || Array.isArray(l)) {
        throw new Error("layout must be an object");
      }
    
      const inputExplicit = resolveExplicitSideLayout(l, "input", rank, inputShape);
      const outputExplicit = resolveExplicitSideLayout(l, "output", rank, outputShape);
    
      let inputResolved = inputExplicit;
      let outputResolved = outputExplicit;
      let usesWhdcnInput = false;
      let usesWhdcnOutput = false;
    
      if (l.whdcn != null) {
        if (typeof l.whdcn !== "object" || Array.isArray(l.whdcn)) {
          throw new Error("layout.whdcn must be an object");
        }
        const whdcnGlobal = { ...l.whdcn };
        delete whdcnGlobal.input;
        delete whdcnGlobal.output;
    
        if (!hasExplicitSideLayout(l, "input")) {
          const desc = mergeWhdcnDescriptor(whdcnGlobal, l.whdcn.input ?? null, "input");
          const resolved = resolveWhdcnSideLayout(desc, "input", rank, inputShape);
          if (resolved) {
            inputResolved = resolved;
            usesWhdcnInput = true;
          }
        }
        if (!hasExplicitSideLayout(l, "output")) {
          const desc = mergeWhdcnDescriptor(whdcnGlobal, l.whdcn.output ?? null, "output");
          const resolved = resolveWhdcnSideLayout(desc, "output", rank, outputShape);
          if (resolved) {
            outputResolved = resolved;
            usesWhdcnOutput = true;
          }
        }
      }
    
      return {
        inputStrides: inputResolved.strides,
        outputStrides: outputResolved.strides,
        inputOffsetElements: inputResolved.offsetElements,
        outputOffsetElements: outputResolved.offsetElements,
        inputBatchStrideElements: inputResolved.batchStrideElements,
        outputBatchStrideElements: outputResolved.batchStrideElements,
        inputSpanElements: inputResolved.spanElements,
        outputSpanElements: outputResolved.spanElements,
        usesStridedInput: !!inputResolved.strides,
        usesStridedOutput: !!outputResolved.strides,
        usesWhdcnInput,
        usesWhdcnOutput,
      };
    }
    
    exports['parseOptionalPositiveIntArray'] = parseOptionalPositiveIntArray;
    exports['parseOptionalNonNegativeInt'] = parseOptionalNonNegativeInt;
    exports['contiguousStrides'] = contiguousStrides;
    exports['stridedSpanElements'] = stridedSpanElements;
    exports['resolveLayoutSemantics'] = resolveLayoutSemantics;
  });

  __define('src/runtime/pipeline_cache.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const CACHE_SNAPSHOT_SCHEMA = "webgpufft.pipeline-cache";
    const CACHE_SNAPSHOT_VERSION = 2;
    const MIN_SUPPORTED_SNAPSHOT_VERSION = 1;
    
    function normalizeSnapshotMetadata(meta, { fromVersion }) {
      if (meta == null) {
        return {
          source: "webgpufft",
          fromVersion,
        };
      }
      if (typeof meta !== "object" || Array.isArray(meta)) {
        throw new Error("pipeline cache snapshot.metadata must be an object");
      }
      const out = {};
      for (const [k, v] of Object.entries(meta)) {
        if (typeof k !== "string" || k.length === 0) {
          throw new Error("pipeline cache snapshot.metadata keys must be non-empty strings");
        }
        if (v == null || typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
          out[k] = v;
          continue;
        }
        throw new Error(
          `pipeline cache snapshot.metadata["${k}"] must be a primitive (string/number/boolean/null)`
        );
      }
      if (!Object.prototype.hasOwnProperty.call(out, "source")) {
        out.source = "webgpufft";
      }
      if (!Object.prototype.hasOwnProperty.call(out, "fromVersion")) {
        out.fromVersion = fromVersion;
      }
      return out;
    }
    
    function makeSnapshot({
      shaderCodes = [],
      pipelineKeys = [],
      createdAtMs = Date.now(),
      metadata = null,
      fromVersion = CACHE_SNAPSHOT_VERSION,
    } = {}) {
      if (!Number.isSafeInteger(createdAtMs) || createdAtMs < 0) {
        throw new Error(`pipeline cache snapshot.createdAtMs must be a non-negative safe integer; got ${createdAtMs}`);
      }
      return {
        schema: CACHE_SNAPSHOT_SCHEMA,
        version: CACHE_SNAPSHOT_VERSION,
        createdAtMs,
        metadata: normalizeSnapshotMetadata(metadata, { fromVersion }),
        shaderCodes: [...new Set(shaderCodes)],
        pipelineKeys: [...new Set(pipelineKeys)],
      };
    }
    
    function emptySnapshot() {
      return makeSnapshot();
    }
    
    function normalizeSnapshot(snapshot) {
      if (snapshot == null) return emptySnapshot();
      if (typeof snapshot !== "object" || Array.isArray(snapshot)) {
        throw new Error("pipeline cache snapshot must be an object");
      }
      if (!Number.isInteger(snapshot.version)) {
        throw new Error(`pipeline cache snapshot.version must be an integer; got ${snapshot.version}`);
      }
      if (snapshot.version > CACHE_SNAPSHOT_VERSION) {
        throw new Error(
          `Unsupported pipeline cache snapshot version: ${snapshot.version} (max supported=${CACHE_SNAPSHOT_VERSION})`
        );
      }
      if (snapshot.version < MIN_SUPPORTED_SNAPSHOT_VERSION) {
        throw new Error(
          `Unsupported pipeline cache snapshot version: ${snapshot.version} (min supported=${MIN_SUPPORTED_SNAPSHOT_VERSION})`
        );
      }
    
      if (snapshot.version >= 2) {
        if (snapshot.schema !== CACHE_SNAPSHOT_SCHEMA) {
          throw new Error(
            `pipeline cache snapshot.schema must be "${CACHE_SNAPSHOT_SCHEMA}" for version ${snapshot.version}; got ${snapshot.schema}`
          );
        }
      }
    
      const shaderCodes = snapshot.shaderCodes ?? snapshot.shaders ?? [];
      if (!Array.isArray(shaderCodes) || !shaderCodes.every((x) => typeof x === "string")) {
        throw new Error("pipeline cache snapshot.shaderCodes must be an array of strings");
      }
    
      const pipelineKeys = snapshot.pipelineKeys ?? snapshot.pipelines ?? [];
      if (!Array.isArray(pipelineKeys) || !pipelineKeys.every((x) => typeof x === "string")) {
        throw new Error("pipeline cache snapshot.pipelineKeys must be an array of strings");
      }
    
      const createdAtMs = snapshot.createdAtMs ?? Date.now();
      if (!Number.isSafeInteger(createdAtMs) || createdAtMs < 0) {
        throw new Error(
          `pipeline cache snapshot.createdAtMs must be a non-negative safe integer; got ${createdAtMs}`
        );
      }
    
      return makeSnapshot({
        shaderCodes,
        pipelineKeys,
        createdAtMs,
        metadata: snapshot.version >= 2 ? snapshot.metadata ?? null : null,
        fromVersion: snapshot.version,
      });
    }
    
    const DEVICE_PIPELINE_CACHES = new WeakMap();
    
    function createSharedState() {
      return {
        modules: new Map(),
        pipelines: new Map(),
        layoutIds: new WeakMap(),
        nextLayoutId: 1,
        stablePipelineKeys: new Set(),
        lastImportedFromVersion: CACHE_SNAPSHOT_VERSION,
      };
    }
    
    class PipelineCache {
      constructor(device, sharedState = null) {
        this.device = device;
        this._state = sharedState ?? createSharedState();
        this._modules = this._state.modules;
        this._pipelines = this._state.pipelines;
      }
    
      _getLayoutId(layout) {
        if (!layout || typeof layout !== "object") {
          throw new Error("getComputePipeline requires a valid pipeline layout object");
        }
        let id = this._state.layoutIds.get(layout);
        if (id == null) {
          id = this._state.nextLayoutId++;
          this._state.layoutIds.set(layout, id);
        }
        return id;
      }
    
      getShaderModule(code) {
        let m = this._modules.get(code);
        if (!m) {
          m = this.device.createShaderModule({ code });
          this._modules.set(code, m);
        }
        return m;
      }
    
      getComputePipeline({ code, layout, entryPoint = "main" }) {
        const layoutId = this._getLayoutId(layout);
        const key = `${layoutId}\n${entryPoint}\n${code}`;
        let p = this._pipelines.get(key);
        if (!p) {
          const module = this.getShaderModule(code);
          p = this.device.createComputePipeline({
            layout,
            compute: { module, entryPoint },
          });
          this._pipelines.set(key, p);
        }
        this._state.stablePipelineKeys.add(`${entryPoint}\n${code}`);
        return p;
      }
    
      exportSnapshot() {
        return makeSnapshot({
          createdAtMs: Date.now(),
          metadata: {
            source: "webgpufft",
            fromVersion: this._state.lastImportedFromVersion,
            modules: this._modules.size,
            pipelineKeys: this._state.stablePipelineKeys.size,
          },
          shaderCodes: Array.from(this._modules.keys()),
          pipelineKeys: Array.from(this._state.stablePipelineKeys),
        });
      }
    
      importSnapshot(snapshot) {
        const normalized = normalizeSnapshot(snapshot);
        this._state.lastImportedFromVersion = normalized.metadata?.fromVersion ?? CACHE_SNAPSHOT_VERSION;
        for (const code of normalized.shaderCodes) {
          this.getShaderModule(code);
        }
        for (const key of normalized.pipelineKeys) {
          this._state.stablePipelineKeys.add(key);
        }
        return this.exportSnapshot();
      }
    }
    
    function getOrCreatePipelineCache(device, { snapshot = null } = {}) {
      let cache = DEVICE_PIPELINE_CACHES.get(device);
      if (!cache) {
        cache = new PipelineCache(device);
        DEVICE_PIPELINE_CACHES.set(device, cache);
      }
      if (snapshot != null) {
        cache.importSnapshot(snapshot);
      }
      return cache;
    }
    
    function exportPipelineCacheSnapshot(device) {
      const cache = DEVICE_PIPELINE_CACHES.get(device);
      return cache ? cache.exportSnapshot() : emptySnapshot();
    }
    
    function importPipelineCacheSnapshot(device, snapshot) {
      const cache = getOrCreatePipelineCache(device);
      return cache.importSnapshot(snapshot);
    }
    
    exports['PipelineCache'] = PipelineCache;
    exports['getOrCreatePipelineCache'] = getOrCreatePipelineCache;
    exports['exportPipelineCacheSnapshot'] = exportPipelineCacheSnapshot;
    exports['importPipelineCacheSnapshot'] = importPipelineCacheSnapshot;
  });

  __define('src/runtime/plans/c2c.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { createFftPlan } = require('src/plan.js');
    const { BasePlan } = require('src/runtime/base_plan.js');
    const { resolveLargeRoutingPolicy, resolveAxisKindsForShape, resolveOutOfCoreAxisWindowPolicy } = require('src/runtime/large_policy.js');
    const { normalizeIoView } = require('src/runtime/ioview.js');
    const { normalizeZeroPad } = require('src/runtime/zero_pad.js');
    const { viewFromArena, createInternalArena } = require('src/runtime/workspace.js');
    const { normalizeToContiguousRanges } = require('src/runtime/segmented_io.js');
    const { resolveLayoutSemantics } = require('src/runtime/layout_semantics.js');
    const { assertOneOf, isPositiveInt, prod, alignBytes, normalizeScaleFactor, ensureWithinBindingLimit, isGpuBuffer, getBufferByteLength, buffersAlias } = require('src/runtime/common.js');
    const { contiguousStrides: tensorContiguousStrides, coordsFromLinear: tensorCoordsFromLinear, linearFromCoords: tensorLinearFromCoords, createTensorDescriptor, requiredBytesForBatchRange } = require('src/runtime/tensor_descriptor.js');
    
    const { generateScaleComplexWGSL } = require('src/kernels/scale.js');
    const { generateZeroOutsideRangeComplexWGSL } = require('src/kernels/zero_pad.js');
    const { generateEmbedComplexWGSL, generateExtractComplexWGSL, generateEmbedComplexF16ToF32WGSL, generateExtractComplexF32ToF16WGSL } = require('src/kernels/ioview.js');
    const { generateTransposeComplex2DWGSL } = require('src/kernels/transpose.js');
    const { generateF16ToF32ComplexWGSL, generateF32ToF16ComplexWGSL } = require('src/kernels/f16_storage.js');
    const { generateGatherComplexStridedWGSL, generateScatterComplexStridedWGSL } = require('src/kernels/strided_complex.js');
    
    const { BluesteinAxis } = require('src/runtime/algorithms/bluestein_axis.js');
    const { RaderAxis } = require('src/runtime/algorithms/rader_axis.js');
    
    function needsIoMapping(io, logicalShape) {
      if (!io) return false;
      for (let i = 0; i < logicalShape.length; i++) {
        if (io.shape[i] !== logicalShape[i]) return true;
        if (io.offset[i] !== 0) return true;
      }
      return false;
    }
    
    function permutedShapeAxisFront(shape, axis) {
      const out = [shape[axis]];
      for (let d = 0; d < shape.length; d++) {
        if (d === axis) continue;
        out.push(shape[d]);
      }
      return out;
    }
    
    function alignDownBytes(v, alignment) {
      if (!Number.isFinite(alignment) || alignment <= 1) return Math.max(0, v | 0);
      return Math.max(0, Math.floor(v / alignment) * alignment);
    }
    
    function generatePermuteRank3Axis2ToFrontWGSL({ shape, workgroupSize }) {
      const X = shape[0] | 0;
      const Y = shape[1] | 0;
      const Z = shape[2] | 0;
      const XY = X * Y;
      const ZX = Z * X;
      return /* wgsl */ `
    struct Params {
      count: u32,
      hz: u32,
      srcStartElements: u32,
      dstStartElements: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const X: u32 = ${X}u;
    const Z: u32 = ${Z}u;
    const XY: u32 = ${XY}u;
    const ZX: u32 = ${ZX}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count || params.hz == 0u) { return; }
      let x: u32 = i % X;
      let rem: u32 = i / X;
      let z: u32 = rem % params.hz;
      let y: u32 = rem / params.hz;
      let srcIdx: u32 = params.srcStartElements + x + y * X + z * XY;
      let dstIdx: u32 = params.dstStartElements + z + x * Z + y * ZX;
      dst[dstIdx] = src[srcIdx];
    }
    `;
    }
    
    function generatePermuteRank3Axis2FromFrontWGSL({ shape, workgroupSize }) {
      const X = shape[0] | 0;
      const Y = shape[1] | 0;
      const Z = shape[2] | 0;
      const XY = X * Y;
      const ZX = Z * X;
      return /* wgsl */ `
    struct Params {
      count: u32,
      hz: u32,
      srcStartElements: u32,
      dstStartElements: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const X: u32 = ${X}u;
    const Z: u32 = ${Z}u;
    const XY: u32 = ${XY}u;
    const ZX: u32 = ${ZX}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count || params.hz == 0u) { return; }
      let x: u32 = i % X;
      let rem: u32 = i / X;
      let z: u32 = rem % params.hz;
      let y: u32 = rem / params.hz;
      let srcIdx: u32 = params.srcStartElements + z + x * Z + y * ZX;
      let dstIdx: u32 = params.dstStartElements + x + y * X + z * XY;
      dst[dstIdx] = src[srcIdx];
    }
    `;
    }
    
    function generatePermuteAxis1TailToFrontWGSL({ shape, workgroupSize }) {
      const X = shape[0] | 0;
      const Y = shape[1] | 0;
      const XY = X * Y;
      return /* wgsl */ `
    struct Params {
      count: u32,
      srcStartElements: u32,
      dstStartElements: u32,
      _pad0: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const X: u32 = ${X}u;
    const Y: u32 = ${Y}u;
    const XY: u32 = ${XY}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count) { return; }
      let xy: u32 = i % XY;
      let t: u32 = i / XY;
      let x: u32 = xy % X;
      let y: u32 = xy / X;
      let srcIdx: u32 = params.srcStartElements + x + y * X + t * XY;
      let dstIdx: u32 = params.dstStartElements + y + x * Y + t * XY;
      dst[dstIdx] = src[srcIdx];
    }
    `;
    }
    
    function generatePermuteAxis1TailFromFrontWGSL({ shape, workgroupSize }) {
      const X = shape[0] | 0;
      const Y = shape[1] | 0;
      const XY = X * Y;
      return /* wgsl */ `
    struct Params {
      count: u32,
      srcStartElements: u32,
      dstStartElements: u32,
      _pad0: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const X: u32 = ${X}u;
    const Y: u32 = ${Y}u;
    const XY: u32 = ${XY}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count) { return; }
      let xy: u32 = i % XY;
      let t: u32 = i / XY;
      let x: u32 = xy % X;
      let y: u32 = xy / X;
      let srcIdx: u32 = params.srcStartElements + y + x * Y + t * XY;
      let dstIdx: u32 = params.dstStartElements + x + y * X + t * XY;
      dst[dstIdx] = src[srcIdx];
    }
    `;
    }
    
    function generatePermuteAxis1TailTiledToFrontWGSL({ shape, workgroupSize }) {
      const X = shape[0] | 0;
      const Y = shape[1] | 0;
      const XY = X * Y;
      return /* wgsl */ `
    struct Params {
      count: u32,
      hx: u32,
      htail: u32,
      srcStartElements: u32,
      dstStartElements: u32,
      _pad0: u32,
      _pad1: u32,
      _pad2: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const X: u32 = ${X}u;
    const Y: u32 = ${Y}u;
    const XY: u32 = ${XY}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count || params.hx == 0u || params.htail == 0u) { return; }
      let x: u32 = i % params.hx;
      let t: u32 = i / params.hx;
      let srcIdx: u32 = params.srcStartElements + x + t * XY;
      let dstIdx: u32 = params.dstStartElements + x * Y + t * XY;
      dst[dstIdx] = src[srcIdx];
    }
    `;
    }
    
    function generatePermuteAxisGenericWGSL({ shape, axis, toFront, workgroupSize }) {
      const rank = shape.length;
      const srcShape = toFront ? shape.slice() : permutedShapeAxisFront(shape, axis);
      const dstShape = toFront ? permutedShapeAxisFront(shape, axis) : shape.slice();
      const srcStrides = tensorContiguousStrides(srcShape);
      const dstStrides = tensorContiguousStrides(dstShape);
      const total = prod(srcShape);
    
      const decodeSrcCoords = [];
      let rem = "li";
      for (let d = 0; d < rank; d++) {
        const dim = srcShape[d] | 0;
        const v = `s${d}`;
        decodeSrcCoords.push(`  let ${v}: u32 = ${rem} % ${dim}u;`);
        if (d < rank - 1) {
          const next = `rem${d}`;
          decodeSrcCoords.push(`  let ${next}: u32 = ${rem} / ${dim}u;`);
          rem = next;
        }
      }
    
      const mapDstCoords = [];
      if (toFront) {
        mapDstCoords.push(`  let d0: u32 = s${axis};`);
        let p = 1;
        for (let d = 0; d < rank; d++) {
          if (d === axis) continue;
          mapDstCoords.push(`  let d${p}: u32 = s${d};`);
          p += 1;
        }
      } else {
        mapDstCoords.push(`  let d${axis}: u32 = s0;`);
        let p = 1;
        for (let d = 0; d < rank; d++) {
          if (d === axis) continue;
          mapDstCoords.push(`  let d${d}: u32 = s${p};`);
          p += 1;
        }
      }
    
      let dstIdxExpr = "0u";
      for (let d = 0; d < rank; d++) {
        const stride = dstStrides[d] | 0;
        dstIdxExpr += ` + d${d} * ${stride}u`;
      }
      let srcIdxExpr = "0u";
      for (let d = 0; d < rank; d++) {
        const stride = srcStrides[d] | 0;
        srcIdxExpr += ` + s${d} * ${stride}u`;
      }
    
      return /* wgsl */ `
    struct Params {
      count: u32,
      batch: u32,
      srcStartElements: u32,
      dstStartElements: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const TOTAL: u32 = ${total}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count) { return; }
      let b: u32 = i / TOTAL;
      if (b >= params.batch) { return; }
      let li: u32 = i - b * TOTAL;
    ${decodeSrcCoords.join("\n")}
    ${mapDstCoords.join("\n")}
      let srcIdx: u32 = params.srcStartElements + b * TOTAL + (${srcIdxExpr});
      let dstIdx: u32 = params.dstStartElements + b * TOTAL + (${dstIdxExpr});
      dst[dstIdx] = src[srcIdx];
    }
    `;
    }
    
    function generateGatherAxis2SlabWGSL({ N, workgroupSize }) {
      return /* wgsl */ `
    struct Params {
      count: u32,
      zStart: u32,
      zCount: u32,
      srcStartElements: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const N: u32 = ${N}u;
    const PLANE_ELEMS: u32 = ${N * N}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count || params.zCount == 0u) { return; }
      let x: u32 = i % N;
      let zLocal: u32 = i / N;
      let srcIdx: u32 = params.srcStartElements + zLocal * PLANE_ELEMS + x;
      let dstIdx: u32 = (params.zStart + zLocal) * N + x;
      dst[dstIdx] = src[srcIdx];
    }
    `;
    }
    
    function generateScatterAxis2SlabWGSL({ N, workgroupSize }) {
      return /* wgsl */ `
    struct Params {
      count: u32,
      zStart: u32,
      zCount: u32,
      dstStartElements: u32,
    }
    
    @group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
    @group(0) @binding(2) var<uniform> params: Params;
    
    const N: u32 = ${N}u;
    const PLANE_ELEMS: u32 = ${N * N}u;
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count || params.zCount == 0u) { return; }
      let x: u32 = i % N;
      let zLocal: u32 = i / N;
      let srcIdx: u32 = (params.zStart + zLocal) * N + x;
      let dstIdx: u32 = params.dstStartElements + zLocal * PLANE_ELEMS + x;
      dst[dstIdx] = src[srcIdx];
    }
    `;
    }
    
    function maxComputeWorkgroupsPerDimension(limits, axis) {
      const raw = limits?.maxComputeWorkgroupsPerDimension;
      if (raw == null) return Infinity;
      if (Array.isArray(raw) || ArrayBuffer.isView(raw)) {
        const value = raw[axis];
        if (!Number.isFinite(value) || value < 1) return 0;
        return Math.floor(value);
      }
      if (!Number.isFinite(raw) || raw < 1) return 0;
      return Math.floor(raw);
    }
    
    function parseOptionalAxisList(v, rank, name) {
      if (v == null) return [];
      if (!Array.isArray(v) || !v.every((x) => Number.isInteger(x) && x >= 0 && x < rank)) {
        throw new Error(`${name} must be an array of axis indices in [0, ${rank - 1}]`);
      }
      return [...new Set(v.map((x) => x | 0))];
    }
    
    function parseC2cTuning(tuning, rank) {
      if (tuning == null) {
        return {
          raderMaxPrime: 4096,
          transposeMinElements: 4096,
          disableTranspose: false,
          disableOutOfCoreFourStep: false,
          largeChunkMaxBatches: null,
          largeRoute: "auto",
          preferOutOfCoreForStrided: null,
          maxStorageBufferBindingSize: null,
          swapTo2Stage4Step: 0,
          swapTo3Stage4Step: 0,
          groupedBatch: null,
          outOfCoreBurstWindows: 1,
          forceBluesteinAxes: [],
          forceRaderAxes: [],
          forceBluesteinAxisSet: new Set(),
          forceRaderAxisSet: new Set(),
        };
      }
      if (typeof tuning !== "object") {
        throw new Error("tuning must be an object when provided");
      }
    
      const raderMaxPrime = tuning.raderMaxPrime ?? 4096;
      if (!Number.isInteger(raderMaxPrime) || raderMaxPrime < 2) {
        throw new Error(`tuning.raderMaxPrime must be an integer >= 2; got ${raderMaxPrime}`);
      }
    
      const transposeMinElements = tuning.transposeMinElements ?? 4096;
      if (!Number.isInteger(transposeMinElements) || transposeMinElements < 1) {
        throw new Error(`tuning.transposeMinElements must be a positive integer; got ${transposeMinElements}`);
      }
    
      const disableTranspose = tuning.disableTranspose ?? false;
      if (typeof disableTranspose !== "boolean") {
        throw new Error(`tuning.disableTranspose must be boolean; got ${disableTranspose}`);
      }
      const disableOutOfCoreFourStep = tuning.disableOutOfCoreFourStep ?? false;
      if (typeof disableOutOfCoreFourStep !== "boolean") {
        throw new Error(`tuning.disableOutOfCoreFourStep must be boolean; got ${disableOutOfCoreFourStep}`);
      }
      const largeChunkMaxBatches = tuning.largeChunkMaxBatches ?? null;
      if (largeChunkMaxBatches != null) {
        if (!Number.isInteger(largeChunkMaxBatches) || largeChunkMaxBatches <= 0) {
          throw new Error(`tuning.largeChunkMaxBatches must be a positive integer; got ${largeChunkMaxBatches}`);
        }
      }
      const largeRoute = tuning.largeRoute ?? "auto";
      if (largeRoute !== "auto" && largeRoute !== "chunk" && largeRoute !== "out-of-core") {
        throw new Error(`tuning.largeRoute must be one of "auto" | "chunk" | "out-of-core"; got ${largeRoute}`);
      }
      const preferOutOfCoreForStrided = tuning.preferOutOfCoreForStrided ?? null;
      if (preferOutOfCoreForStrided != null && typeof preferOutOfCoreForStrided !== "boolean") {
        throw new Error(
          `tuning.preferOutOfCoreForStrided must be boolean when provided; got ${preferOutOfCoreForStrided}`
        );
      }
      const maxStorageBufferBindingSize = tuning.maxStorageBufferBindingSize ?? null;
      if (maxStorageBufferBindingSize != null) {
        if (!Number.isInteger(maxStorageBufferBindingSize) || maxStorageBufferBindingSize <= 0) {
          throw new Error(`tuning.maxStorageBufferBindingSize must be a positive integer; got ${maxStorageBufferBindingSize}`);
        }
      }
      const swapTo2Stage4Step = tuning.swapTo2Stage4Step ?? 0;
      if (!Number.isInteger(swapTo2Stage4Step) || swapTo2Stage4Step < 0) {
        throw new Error(`tuning.swapTo2Stage4Step must be a non-negative integer; got ${swapTo2Stage4Step}`);
      }
      const swapTo3Stage4Step = tuning.swapTo3Stage4Step ?? 0;
      if (!Number.isInteger(swapTo3Stage4Step) || swapTo3Stage4Step < 0) {
        throw new Error(`tuning.swapTo3Stage4Step must be a non-negative integer; got ${swapTo3Stage4Step}`);
      }
      const groupedBatch = tuning.groupedBatch ?? null;
      if (groupedBatch != null) {
        if (Number.isInteger(groupedBatch)) {
          if (groupedBatch <= 0) {
            throw new Error(`tuning.groupedBatch must be a positive integer or axis-indexed array; got ${groupedBatch}`);
          }
        } else if (Array.isArray(groupedBatch)) {
          if (!groupedBatch.every((v) => v == null || (Number.isInteger(v) && v > 0))) {
            throw new Error("tuning.groupedBatch array entries must be positive integers or null");
          }
        } else {
          throw new Error(`tuning.groupedBatch must be a positive integer or axis-indexed array; got ${groupedBatch}`);
        }
      }
      const outOfCoreBurstWindows = tuning.outOfCoreBurstWindows ?? 1;
      if (!Number.isInteger(outOfCoreBurstWindows) || outOfCoreBurstWindows <= 0) {
        throw new Error(`tuning.outOfCoreBurstWindows must be a positive integer; got ${outOfCoreBurstWindows}`);
      }
    
      const forceBluesteinAxes = parseOptionalAxisList(tuning.forceBluesteinAxes ?? null, rank, "tuning.forceBluesteinAxes");
      const forceRaderAxes = parseOptionalAxisList(tuning.forceRaderAxes ?? null, rank, "tuning.forceRaderAxes");
      const forceBluesteinAxisSet = new Set(forceBluesteinAxes);
      const forceRaderAxisSet = new Set(forceRaderAxes);
      for (const axis of forceBluesteinAxisSet) {
        if (forceRaderAxisSet.has(axis)) {
          throw new Error(`Axis ${axis} cannot be forced to both Bluestein and Rader`);
        }
      }
    
      return {
        raderMaxPrime,
        transposeMinElements,
        disableTranspose,
        disableOutOfCoreFourStep,
        largeChunkMaxBatches,
        largeRoute,
        preferOutOfCoreForStrided,
        maxStorageBufferBindingSize,
        swapTo2Stage4Step,
        swapTo3Stage4Step,
        groupedBatch,
        outOfCoreBurstWindows,
        forceBluesteinAxes,
        forceRaderAxes,
        forceBluesteinAxisSet,
        forceRaderAxisSet,
      };
    }
    
    class C2CPlan extends BasePlan {
      constructor(device, opts) {
        super(device, opts);
        const {
          shape,
          direction,
          batch = 1,
          inPlace = false,
          normalize = "none",
          layout = { interleavedComplex: true },
          precision = "f32",
          ioView = null,
          zeroPad = null,
        } = opts ?? {};
    
        if (!Array.isArray(shape) || shape.length < 1) {
          throw new Error(`shape must be an array of one or more positive dimensions; got ${JSON.stringify(shape)}`);
        }
        if (!shape.every(isPositiveInt)) throw new Error(`shape elements must be positive ints; got ${JSON.stringify(shape)}`);
        assertOneOf(direction, ["forward", "inverse"], "direction");
        assertOneOf(normalize, ["none", "backward", "unitary"], "normalize");
        if (!Number.isInteger(batch) || batch <= 0) throw new Error(`batch must be positive int; got ${batch}`);
        if (layout?.interleavedComplex !== true) throw new Error("c2c requires layout.interleavedComplex=true");
        assertOneOf(precision, ["f32", "f16-storage"], "precision");
        if (precision === "f16-storage" && !device.features?.has?.("shader-f16")) {
          throw new Error('precision="f16-storage" requires device.features.has("shader-f16")');
        }
    
        this.shape = shape.slice();
        this.rank = shape.length;
        this.direction = direction;
        this.batch = batch;
        this.inPlace = !!inPlace;
        this.normalize = normalize;
        this.precision = precision;
        this._axis01MatrixBatch = this.batch * (this.rank > 2 ? prod(this.shape.slice(2)) : 1);
        this.tuning = parseC2cTuning(opts?.tuning ?? null, this.rank);
        this.io = normalizeIoView(this.rank, this.shape, ioView ?? {});
        this.zeroPad = normalizeZeroPad(this.rank, this.shape, zeroPad ?? null, "zeroPad");
        this._needsInputMapping = !!(this.io.input && needsIoMapping(this.io.input, this.shape));
        this._needsOutputMapping = !!(this.io.output && needsIoMapping(this.io.output, this.shape));
        this._inputLayoutShape = this._needsInputMapping ? this.io.input.shape.slice() : this.shape.slice();
        this._outputLayoutShape = this._needsOutputMapping ? this.io.output.shape.slice() : this.shape.slice();
    
        const resolvedLayout = resolveLayoutSemantics({
          layout,
          rank: this.rank,
          inputShape: this._inputLayoutShape,
          outputShape: this._outputLayoutShape,
        });
        this._inputStrides = resolvedLayout.inputStrides;
        this._outputStrides = resolvedLayout.outputStrides;
        this._inputOffsetElements = resolvedLayout.inputOffsetElements;
        this._outputOffsetElements = resolvedLayout.outputOffsetElements;
        this._inputBatchStrideElements = resolvedLayout.inputBatchStrideElements;
        this._outputBatchStrideElements = resolvedLayout.outputBatchStrideElements;
        this._usesStridedInput = resolvedLayout.usesStridedInput;
        this._usesStridedOutput = resolvedLayout.usesStridedOutput;
        this._usesWhdcnInput = resolvedLayout.usesWhdcnInput;
        this._usesWhdcnOutput = resolvedLayout.usesWhdcnOutput;
        this._inputTensorDesc = this._usesStridedInput
          ? createTensorDescriptor({
              name: "c2c.input",
              shape: this._inputLayoutShape,
              strides: this._inputStrides,
              offsetElements: this._inputOffsetElements,
              batchStrideElements: this._inputBatchStrideElements,
            })
          : null;
        this._outputTensorDesc = this._usesStridedOutput
          ? createTensorDescriptor({
              name: "c2c.output",
              shape: this._outputLayoutShape,
              strides: this._outputStrides,
              offsetElements: this._outputOffsetElements,
              batchStrideElements: this._outputBatchStrideElements,
            })
          : null;
        this._inputSpanElements = this._inputTensorDesc?.spanElements ?? 0;
        this._outputSpanElements = this._outputTensorDesc?.spanElements ?? 0;
        if ((this._usesStridedInput || this._usesStridedOutput) && this.precision !== "f32") {
          throw new Error('custom strides currently support precision:"f32" only');
        }
    
        // Axis algorithm choice
        const axisStrategy = resolveAxisKindsForShape({
          shape: this.shape,
          tuning: {
            raderMaxPrime: this.tuning.raderMaxPrime,
            forceBluesteinAxes: this.tuning.forceBluesteinAxes,
            forceRaderAxes: this.tuning.forceRaderAxes,
          },
        });
        this.axisKind = axisStrategy.axisKinds;
        this.logicalTotal = prod(this.shape);
        this.totalComplex = this.logicalTotal * this.batch;
        this.mainBytes = this.totalComplex * 8;
        this._bytesPerBatch = this.logicalTotal * 8;
    
        const largePolicy = resolveLargeRoutingPolicy({
          device,
          tuning: {
            maxStorageBufferBindingSize: this.tuning.maxStorageBufferBindingSize,
            largeRoute: this.tuning.largeRoute,
            preferOutOfCoreForStrided: this.tuning.preferOutOfCoreForStrided,
          },
          requiredBindingBytes: [this.mainBytes],
          lineBytes: this.shape.map((n) => n * 8),
          axisKinds: this.axisKind,
          axisLengths: this.shape,
          allowNonMixedBoundedSlicing: true,
          allowOutOfCore: this.rank >= 2,
          disableOutOfCore: this.tuning.disableOutOfCoreFourStep,
          rank: this.rank,
          bytesPerBatch: this._bytesPerBatch,
          precision: this.precision,
          hasStridedIO: this._usesStridedInput || this._usesStridedOutput,
          preferOutOfCoreForStrided: true,
          outOfCoreUnsupportedError: ({ maxBindBytes, axisSupported, attemptedRoutes, reasonCodes }) =>
            [
              `c2c shape=${JSON.stringify(shape)} batch=${batch} requires ${this.mainBytes} bytes total,`,
              `and one batch requires ${this._bytesPerBatch} bytes > maxStorageBufferBindingSize=${maxBindBytes}.`,
              `Out-of-core fallback is available only for rank>=2 precision:"f32",`,
              `and when each axis line is compatible with the active axis strategy`,
              `(mixed-radix: direct or two-step factorable; Bluestein/Rader: line fits maxBufferSize for multi-upload slicing).`,
              `(axisBytes=${JSON.stringify(this.shape.map((n) => n * 8))}, axisKind=${JSON.stringify(this.axisKind)}, axisSupported=${JSON.stringify(axisSupported)}).`,
              `(attemptedRoutes=${JSON.stringify(attemptedRoutes)}, reasonCodes=${JSON.stringify(reasonCodes)}).`,
            ].join(" "),
        });
        this._maxBindBytes = largePolicy.maxBindBytes;
        this._largeBatchChunkMode = largePolicy.needsLargeMode;
        this._outOfCoreFourStepMode = largePolicy.useOutOfCore;
        this._largeRouteMode = largePolicy.routeMode;
        this._largeRouteReasons = largePolicy.reasonCodes;
        this._largeRouteAttempts = largePolicy.attemptedRoutes;
        if (!this._largeBatchChunkMode) {
          ensureWithinBindingLimit(device, this.mainBytes, `c2c shape=${JSON.stringify(shape)} batch=${batch}`);
        }
    
        // mixed axis plans
        this.axisPlans = new Array(this.rank).fill(null);
        for (let axis = 0; axis < this.rank; axis++) {
          if (this.axisKind[axis] === "mixed") {
            if (this._outOfCoreFourStepMode && axis !== 0) continue;
            const axisNormalize = this._outOfCoreFourStepMode ? "none" : (this._largeBatchChunkMode ? this.normalize : "none");
            this.axisPlans[axis] = createFftPlan(device, {
              shape: this.shape,
              direction,
              normalize: axisNormalize,
              inPlace: true,
              layout: "interleaved",
              precision: "f32",
              axes: [axis],
            });
          }
        }
    
        // advanced axis ops
        this.axisAdvanced = new Array(this.rank).fill(null);
        this.maxAxisWorkBytes = 0;
        for (let axis = 0; axis < this.rank; axis++) {
          const kind = this.axisKind[axis];
          const axisLineBytes = this.shape[axis] * 8;
          if (this._outOfCoreFourStepMode && kind === "rader" && axisLineBytes > this._maxBindBytes) {
            // Oversized out-of-core Rader axes are routed to Bluestein fallback executors below.
            continue;
          }
          if (this.axisKind[axis] === "bluestein") {
            const ax = new BluesteinAxis(device, this.cache, {
              shape: this.shape,
              rank: this.rank,
              batch,
              axis,
              direction,
              workgroupSize: this.workgroupSize,
              maxWorkBytes: this._maxBindBytes,
            });
            this.axisAdvanced[axis] = ax;
            this.maxAxisWorkBytes = Math.max(this.maxAxisWorkBytes, ax.workBytes);
          } else if (this.axisKind[axis] === "rader") {
            const ax = new RaderAxis(device, this.cache, {
              shape: this.shape,
              rank: this.rank,
              batch,
              axis,
              direction,
              workgroupSize: this.workgroupSize,
              maxWorkBytes: this._maxBindBytes,
            });
            this.axisAdvanced[axis] = ax;
            this.maxAxisWorkBytes = Math.max(this.maxAxisWorkBytes, ax.workBytes);
          }
        }
        this._outOfCoreAxis0OnTransposed = null;
        this._outOfCoreAxisPlans = null;
        this._outOfCoreAxisPermShapes = null;
        this._outOfCoreAxisWindowPolicy = null;
        this._outOfCoreTranspose = null;
        this._outOfCoreTransposePipelines = null;
        this._outOfCoreAxis1TailPermute = null;
        this._outOfCoreAxis1TailChunk = null;
        this._outOfCoreRank3Axis2Permute = null;
        this._outOfCoreRank3Axis2Tile = null;
        this._outOfCoreGenericPermute = null;
        this._outOfCoreGenericPermutePipelines = null;
        this._outOfCoreAdjacentSwapPipelines = null;
        this._outOfCoreAdjacentSwapTiled = null;
        this._outOfCoreAdjacentSwapTiledPipelines = null;
        if (this._outOfCoreFourStepMode) {
          this._outOfCoreAxisPlans = new Array(this.rank).fill(null);
          this._outOfCoreAxisPermShapes = new Array(this.rank).fill(null);
          this._outOfCoreAxisEffectiveKind = new Array(this.rank).fill(null);
          this._outOfCoreAxisWindowPolicy = new Array(this.rank).fill(null);
          for (let axis = 0; axis < this.rank; axis++) {
            const kind = this.axisKind[axis];
            const permShape = axis === 0 ? this.shape.slice() : permutedShapeAxisFront(this.shape, axis);
            const axisLineBytes = permShape[0] * 8;
            const axisLinesTotal = this.batch * (this.logicalTotal / permShape[0]);
            const axisWindowPolicy = resolveOutOfCoreAxisWindowPolicy({
              axisLen: permShape[0],
              lineBytes: axisLineBytes,
              linesTotal: axisLinesTotal,
              maxBindBytes: this._maxBindBytes,
              axisKind: kind,
              tuning: this.tuning,
              axisIndex: axis,
              storageAlign: this.device.limits?.minStorageBufferOffsetAlignment ?? 256,
            });
            this._outOfCoreAxisWindowPolicy[axis] = axisWindowPolicy;
            this._outOfCoreAxisPermShapes[axis] = permShape;
            if (kind === "mixed") {
              this._outOfCoreAxisEffectiveKind[axis] = "mixed";
              const stagedBind = Math.max(8, Math.floor(this._maxBindBytes / axisWindowPolicy.numAxisUploads));
              const effectiveAxisBind = Math.min(this._maxBindBytes, stagedBind);
              this._outOfCoreAxisPlans[axis] = createFftPlan(device, {
                shape: permShape,
                direction,
                normalize: "none",
                inPlace: true,
                layout: "interleaved",
                precision: "f32",
                axes: [0],
                maxStorageBufferBindingSize: effectiveAxisBind,
              });
              continue;
            }
            if (kind === "bluestein") {
              this._outOfCoreAxisEffectiveKind[axis] = "bluestein";
              if (axis === 0) {
                this._outOfCoreAxisPlans[axis] = this.axisAdvanced[axis];
              } else {
                this._outOfCoreAxisPlans[axis] = new BluesteinAxis(device, this.cache, {
                  shape: permShape,
                  rank: this.rank,
                  batch,
                  axis: 0,
                  direction,
                  workgroupSize: this.workgroupSize,
                  maxWorkBytes: this._maxBindBytes,
                });
              }
            } else if (axisLineBytes > this._maxBindBytes) {
              // Oversized Rader lines use Bluestein under out-of-core multi-upload so no single
              // storage binding needs to cover the full line.
              this._outOfCoreAxisEffectiveKind[axis] = "bluestein-fallback";
              this._outOfCoreAxisPlans[axis] = new BluesteinAxis(device, this.cache, {
                shape: permShape,
                rank: this.rank,
                batch,
                axis: 0,
                direction,
                workgroupSize: this.workgroupSize,
                maxWorkBytes: this._maxBindBytes,
              });
            } else {
              this._outOfCoreAxisEffectiveKind[axis] = "rader";
              if (axis === 0) {
                this._outOfCoreAxisPlans[axis] = this.axisAdvanced[axis];
              } else {
                this._outOfCoreAxisPlans[axis] = new RaderAxis(device, this.cache, {
                  shape: permShape,
                  rank: this.rank,
                  batch,
                  axis: 0,
                  direction,
                  workgroupSize: this.workgroupSize,
                  maxWorkBytes: this._maxBindBytes,
                });
              }
            }
          }
          // Backward-compat alias used in unit tests for the axis-1 out-of-core path.
          this._outOfCoreAxis0OnTransposed = this.rank >= 2 ? this._outOfCoreAxisPlans[1] : null;
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this._outOfCoreTranspose = { bgl, pl, params, tile: 16 };
          this._outOfCoreTransposePipelines = new Map();
    
          if (this.precision === "f32" && this.rank >= 3) {
            const pbglAxis1 = device.createBindGroupLayout({
              entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
              ],
            });
            const pplAxis1 = device.createPipelineLayout({ bindGroupLayouts: [pbglAxis1] });
            const axis1ToFrontCode = generatePermuteAxis1TailToFrontWGSL({
              shape: this.shape,
              workgroupSize: this.workgroupSize,
            });
            const axis1FromFrontCode = generatePermuteAxis1TailFromFrontWGSL({
              shape: this.shape,
              workgroupSize: this.workgroupSize,
            });
            const axis1ToFront = this.cache.getComputePipeline({ code: axis1ToFrontCode, layout: pplAxis1 });
            const axis1FromFront = this.cache.getComputePipeline({ code: axis1FromFrontCode, layout: pplAxis1 });
            const axis1Params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this._outOfCoreAxis1TailPermute = {
              bgl: pbglAxis1,
              pl: pplAxis1,
              toFront: axis1ToFront,
              fromFront: axis1FromFront,
              params: axis1Params,
            };
          }
    
          if (this.precision === "f32" && this.rank === 3) {
            const pbgl = device.createBindGroupLayout({
              entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
              ],
            });
            const ppl = device.createPipelineLayout({ bindGroupLayouts: [pbgl] });
            const toFrontCode = generatePermuteRank3Axis2ToFrontWGSL({
              shape: this.shape,
              workgroupSize: this.workgroupSize,
            });
            const fromFrontCode = generatePermuteRank3Axis2FromFrontWGSL({
              shape: this.shape,
              workgroupSize: this.workgroupSize,
            });
            const toFront = this.cache.getComputePipeline({ code: toFrontCode, layout: ppl });
            const fromFront = this.cache.getComputePipeline({ code: fromFrontCode, layout: ppl });
            const pparams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this._outOfCoreRank3Axis2Permute = {
              bgl: pbgl,
              pl: ppl,
              toFront,
              fromFront,
              params: pparams,
            };
          }
    
          if (this.precision === "f32" && this.rank >= 2) {
            const pbgl = device.createBindGroupLayout({
              entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
              ],
            });
            const ppl = device.createPipelineLayout({ bindGroupLayouts: [pbgl] });
            const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this._outOfCoreGenericPermute = { bgl: pbgl, pl: ppl, params };
            this._outOfCoreGenericPermutePipelines = new Map();
            const tiledParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this._outOfCoreAdjacentSwapTiled = { bgl: pbgl, pl: ppl, params: tiledParams };
            this._outOfCoreAdjacentSwapTiledPipelines = new Map();
          }
        }
    
        // scale pipeline
        this.scale = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateScaleComplexWGSL({ workgroupSize: this.workgroupSize });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          return { bgl, pl: pipelineLayout, pipeline, params };
        })();
    
        this.zeroRead = null;
        this.zeroWrite = null;
        const makeZeroPipeline = (stage) => {
          const bgl = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateZeroOutsideRangeComplexWGSL({
            shape: this.shape,
            start: stage.start,
            end: stage.end,
            batch: this.batch,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          return { bgl, pl, pipeline };
        };
        if (this.zeroPad.read) this.zeroRead = makeZeroPipeline(this.zeroPad.read);
        if (this.zeroPad.write) this.zeroWrite = makeZeroPipeline(this.zeroPad.write);
    
        // ioView mapping
        this.ioEmbed = null;
        this.ioExtract = null;
        this._bytesPerComplexIO = this.precision === "f16-storage" ? 4 : 8;
        this._inViewTotal = this.io.input ? prod(this.io.input.shape) : this.logicalTotal;
        this._outViewTotal = this.io.output ? prod(this.io.output.shape) : this.logicalTotal;
        this._inPhysComplexPerBatch = this._needsInputMapping ? this._inViewTotal : this.logicalTotal;
        this._outPhysComplexPerBatch = this._needsOutputMapping ? this._outViewTotal : this.logicalTotal;
        this._inPhysComplex = this._inPhysComplexPerBatch * this.batch;
        this._outPhysComplex = this._outPhysComplexPerBatch * this.batch;
        this._inPhysBytesPerBatch = this._inPhysComplexPerBatch * this._bytesPerComplexIO;
        this._outPhysBytesPerBatch = this._outPhysComplexPerBatch * this._bytesPerComplexIO;
        this._inPhysBytes = this._inPhysComplex * this._bytesPerComplexIO;
        this._outPhysBytes = this._outPhysComplex * this._bytesPerComplexIO;
    
        if (this._needsInputMapping && !this._outOfCoreFourStepMode) {
          const inBindBytes = this._largeBatchChunkMode ? this._inPhysBytesPerBatch : this._inPhysBytes;
          ensureWithinBindingLimit(device, inBindBytes, "c2c ioView.input");
          const code =
            this.precision === "f16-storage"
              ? generateEmbedComplexF16ToF32WGSL({
                  rank: this.rank,
                  logicalDims: this.shape,
                  viewDims: this.io.input.shape,
                  offset: this.io.input.offset,
                  workgroupSize: this.workgroupSize,
                })
              : generateEmbedComplexWGSL({
                  rank: this.rank,
                  logicalDims: this.shape,
                  viewDims: this.io.input.shape,
                  offset: this.io.input.offset,
                  workgroupSize: this.workgroupSize,
                });
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.ioEmbed = { bgl, pl: pipelineLayout, pipeline, params, viewShape: this.io.input.shape };
        }
        if (this._needsOutputMapping && !this._outOfCoreFourStepMode) {
          const outBindBytes = this._largeBatchChunkMode ? this._outPhysBytesPerBatch : this._outPhysBytes;
          ensureWithinBindingLimit(device, outBindBytes, "c2c ioView.output");
          const code =
            this.precision === "f16-storage"
              ? generateExtractComplexF32ToF16WGSL({
                  rank: this.rank,
                  logicalDims: this.shape,
                  viewDims: this.io.output.shape,
                  offset: this.io.output.offset,
                  clearOutside: this.io.output.clearOutside,
                  workgroupSize: this.workgroupSize,
                })
              : generateExtractComplexWGSL({
                  rank: this.rank,
                  logicalDims: this.shape,
                  viewDims: this.io.output.shape,
                  offset: this.io.output.offset,
                  clearOutside: this.io.output.clearOutside,
                  workgroupSize: this.workgroupSize,
                });
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.ioExtract = { bgl, pl: pipelineLayout, pipeline, params, viewShape: this.io.output.shape };
        }
    
        if (this._largeBatchChunkMode && this.precision !== "f32") {
          throw new Error('Large-batch chunk mode currently supports precision:"f32" only');
        }
        this.scratchBytes = this._largeBatchChunkMode
          ? 16
          : Math.max(this.mainBytes, this.maxAxisWorkBytes, this._inPhysBytes, this._outPhysBytes);
        ensureWithinBindingLimit(device, this.scratchBytes, "c2c scratch");
    
        // f16 storage conversion
        this.f16 = null;
        if (this.precision === "f16-storage") {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const toF32 = this.cache.getComputePipeline({ code: generateF16ToF32ComplexWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const toF16 = this.cache.getComputePipeline({ code: generateF32ToF16ComplexWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.f16 = { bgl, pl: pipelineLayout, toF32, toF16, params };
        }
    
        // Optional strided gather/scatter (layout.inputStrides/outputStrides)
        this.stridedIn = null;
        this.stridedOut = null;
    
        if (this._usesStridedInput) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateGatherComplexStridedWGSL({
            shape: this.shape,
            strides: this._inputStrides,
            baseOffsetElements: this._inputOffsetElements,
            batchStrideElements: this._inputBatchStrideElements,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.stridedIn = { bgl, pl, pipeline, params };
        }
    
        if (this._usesStridedOutput) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateScatterComplexStridedWGSL({
            shape: this.shape,
            strides: this._outputStrides,
            baseOffsetElements: this._outputOffsetElements,
            batchStrideElements: this._outputBatchStrideElements,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.stridedOut = { bgl, pl, pipeline, params };
        }
    
        // Optional transpose fast path for axis 1 using batched 2D tiles over [axis0, axis1].
        this.transpose = null;
        this.transposeBytes = 0;
        this.axis0OnTransposed = null;
        const transposeDispatchLimitZ = maxComputeWorkgroupsPerDimension(this.device.limits, 2);
        if (
          !this._largeBatchChunkMode &&
          !this.tuning.disableTranspose &&
          this.rank >= 2 &&
          this.axisKind[0] === "mixed" &&
          this.axisKind[1] === "mixed" &&
          this.shape[0] * this.shape[1] >= this.tuning.transposeMinElements &&
          this._axis01MatrixBatch <= transposeDispatchLimitZ
        ) {
          const [Nx, Ny] = this.shape;
          const transposedShape = [Ny, Nx, ...this.shape.slice(2)];
          const tile = 16;
          const codeXY = generateTransposeComplex2DWGSL({ Nx, Ny, tile }); // (Nx,Ny) -> (Ny,Nx)
          const codeYX = generateTransposeComplex2DWGSL({ Nx: Ny, Ny: Nx, tile }); // (Ny,Nx) -> (Nx,Ny)
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipelineXY = this.cache.getComputePipeline({ code: codeXY, layout: pipelineLayout });
          const pipelineYX = this.cache.getComputePipeline({ code: codeYX, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.transpose = {
            bgl,
            pl: pipelineLayout,
            pipelineXY,
            pipelineYX,
            params,
            tile,
            Nx,
            Ny,
            matrixBatch: this._axis01MatrixBatch,
          };
          this.transposeBytes = this.mainBytes;
    
          // Precompile "FFT along original axis 1" as axis0 FFT on [Ny, Nx, ...tail].
          this.axis0OnTransposed = createFftPlan(this.device, {
            shape: transposedShape,
            direction: this.direction,
            normalize: "none",
            inPlace: true,
            layout: "interleaved",
            precision: "f32",
            axes: [0],
          });
        }
    
        // Workspace layout: [mainStage?][scratch][axisWork][transpose]
        this.needsMainStage = !!this.ioEmbed || !!this.ioExtract || this.precision === "f16-storage";
        this.mainStageBytes = this.needsMainStage && !this._largeBatchChunkMode ? this.mainBytes : 0;
        this.axisWorkBytes = this.maxAxisWorkBytes;
    
        const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        let off = 0;
        this.mainStageOffset = 0;
        off += this.mainStageBytes;
        off = alignBytes(off, storageAlign);
        this.scratchOffset = off;
        off += this.scratchBytes;
        off = alignBytes(off, storageAlign);
        this.axisWorkOffset = off;
        off += this.axisWorkBytes;
        off = alignBytes(off, storageAlign);
        this.transposeOffset = off;
        off += this.transposeBytes;
    
        this.workspaceBytes = off;
        this._splitWorkspace = null;
        const maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
        if (this.workspaceBytes <= maxBufferSize) {
          this._arena = createInternalArena(device, this.workspaceBytes);
        } else {
          const splitNeeds = [
            ["mainStage", this.mainStageBytes],
            ["scratch", this.scratchBytes],
            ["axisWork", this.axisWorkBytes],
            ["transpose", this.transposeBytes],
          ];
          for (const [name, bytes] of splitNeeds) {
            if (bytes > 0 && bytes > maxBufferSize) {
              throw new Error(
                `c2c split workspace cannot allocate ${name} buffer: ${bytes} bytes exceeds device.limits.maxBufferSize=${maxBufferSize}`
              );
            }
          }
          this._arena = null;
          this._splitWorkspace = {
            mainStage: this.mainStageBytes ? createInternalArena(device, this.mainStageBytes) : null,
            scratch: this.scratchBytes ? createInternalArena(device, this.scratchBytes) : null,
            axisWork: this.axisWorkBytes ? createInternalArena(device, this.axisWorkBytes) : null,
            transpose: this.transposeBytes ? createInternalArena(device, this.transposeBytes) : null,
          };
        }
        this._largeStageBuffer = null;
        this._largeStageBytes = 0;
        this._largeChunkBuffer = null;
        this._largeChunkBytes = 0;
        this._largeAuxBuffer = null;
        this._largeAuxBytes = 0;
        this._retiredLargeStageBuffers = [];
        this._retiredLargeChunkBuffers = [];
        this._retiredLargeAuxBuffers = [];
        this._scaleChunkParamsBuffer = null;
        this._scaleChunkParamsBytes = 0;
        this._retiredScaleChunkParamsBuffers = [];
        this._zeroComplexBuffer = null;
    
        const maxBufferSizeForMode = this.device.limits?.maxBufferSize ?? Infinity;
        this._outOfCoreSegmentedFullVolumeMode =
          this._outOfCoreFourStepMode &&
          this.mainBytes > maxBufferSizeForMode &&
          this.precision === "f32" &&
          this.rank === 3 &&
          this.axisKind.every((k) => k === "mixed") &&
          !this._needsInputMapping &&
          !this._needsOutputMapping &&
          !this._usesStridedInput &&
          !this._usesStridedOutput &&
          !this.zeroPad.read &&
          !this.zeroPad.write;
        this._outOfCoreSegmentedFullVolumeState = null;
        this._segmentedFullVolumeMeta = null;
      }
    
      getWorkspaceSizeBytes() {
        return this.workspaceBytes;
      }
    
      _resolveWorkspaceViews(arenaLike) {
        if (arenaLike) {
          if (getBufferByteLength(arenaLike) < this.workspaceBytes) throw new Error(`temp too small: need ${this.workspaceBytes} bytes`);
          return {
            mainStage: this.mainStageBytes ? viewFromArena(arenaLike, this.mainStageOffset, this.mainStageBytes) : null,
            scratch: viewFromArena(arenaLike, this.scratchOffset, this.scratchBytes),
            axisWork: this.axisWorkBytes ? viewFromArena(arenaLike, this.axisWorkOffset, this.axisWorkBytes) : null,
            transpose: this.transposeBytes ? viewFromArena(arenaLike, this.transposeOffset, this.transposeBytes) : null,
          };
        }
        if (this._splitWorkspace) {
          return {
            mainStage: this.mainStageBytes ? viewFromArena(this._splitWorkspace.mainStage, 0, this.mainStageBytes) : null,
            scratch: viewFromArena(this._splitWorkspace.scratch, 0, this.scratchBytes),
            axisWork: this.axisWorkBytes ? viewFromArena(this._splitWorkspace.axisWork, 0, this.axisWorkBytes) : null,
            transpose: this.transposeBytes ? viewFromArena(this._splitWorkspace.transpose, 0, this.transposeBytes) : null,
          };
        }
        throw new Error("No workspace buffer");
      }
    
      _ensureLargeStageBuffer(minBytes) {
        if (this._largeStageBuffer && this._largeStageBytes >= minBytes) return this._largeStageBuffer;
    
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        if (minBytes > maxBufferSize) {
          throw new Error(
            `Large-batch staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`
          );
        }
    
        if (this._largeStageBuffer) this._retiredLargeStageBuffers.push(this._largeStageBuffer);
        this._largeStageBuffer = this.device.createBuffer({
          size: minBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._largeStageBytes = minBytes;
        return this._largeStageBuffer;
      }
    
      _ensureLargeChunkBuffer(minBytes) {
        if (this._largeChunkBuffer && this._largeChunkBytes >= minBytes) return this._largeChunkBuffer;
    
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        if (minBytes > maxBufferSize) {
          throw new Error(
            `Large-batch chunk scratch requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`
          );
        }
    
        if (this._largeChunkBuffer) this._retiredLargeChunkBuffers.push(this._largeChunkBuffer);
        this._largeChunkBuffer = this.device.createBuffer({
          size: minBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._largeChunkBytes = minBytes;
        return this._largeChunkBuffer;
      }
    
      _ensureLargeAuxBuffer(minBytes) {
        if (this._largeAuxBuffer && this._largeAuxBytes >= minBytes) return this._largeAuxBuffer;
    
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        if (minBytes > maxBufferSize) {
          throw new Error(
            `Large-batch auxiliary staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`
          );
        }
    
        if (this._largeAuxBuffer) this._retiredLargeAuxBuffers.push(this._largeAuxBuffer);
        this._largeAuxBuffer = this.device.createBuffer({
          size: minBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._largeAuxBytes = minBytes;
        return this._largeAuxBuffer;
      }
    
      _ensureScaleChunkParamsBuffer(minBytes) {
        if (this._scaleChunkParamsBuffer && this._scaleChunkParamsBytes >= minBytes) return this._scaleChunkParamsBuffer;
    
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        if (minBytes > maxBufferSize) {
          throw new Error(
            `Large-batch scale params require ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}`
          );
        }
    
        if (this._scaleChunkParamsBuffer) this._retiredScaleChunkParamsBuffers.push(this._scaleChunkParamsBuffer);
        this._scaleChunkParamsBuffer = this.device.createBuffer({
          size: minBytes,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._scaleChunkParamsBytes = minBytes;
        return this._scaleChunkParamsBuffer;
      }
    
      _ensureZeroComplexBuffer() {
        if (this._zeroComplexBuffer) return this._zeroComplexBuffer;
        this._zeroComplexBuffer = this.device.createBuffer({
          size: 8,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this._zeroComplexBuffer, 0, new Float32Array([0, 0]));
        return this._zeroComplexBuffer;
      }
    
      _rangesOverlap(a, b) {
        if (!a || !b) return false;
        if (a.buffer !== b.buffer) return false;
        const a0 = a.offsetBytes;
        const a1 = a.offsetBytes + a.sizeBytes;
        const b0 = b.offsetBytes;
        const b1 = b.offsetBytes + b.sizeBytes;
        return !(a1 <= b0 || b1 <= a0);
      }
    
      _copyRangesToContiguous(commandEncoder, ranges, dstBuffer, dstOffsetBytes) {
        let dst = dstOffsetBytes;
        for (const r of ranges) {
          commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
          dst += r.sizeBytes;
        }
      }
    
      _copyContiguousToRanges(commandEncoder, srcBuffer, srcOffsetBytes, ranges) {
        let src = srcOffsetBytes;
        for (const r of ranges) {
          commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
          src += r.sizeBytes;
        }
      }
    
      _copyComplexFromAny(commandEncoder, src, srcOffsetBytes, dstBuffer, dstOffsetBytes) {
        if (isGpuBuffer(src)) {
          commandEncoder.copyBufferToBuffer(src, srcOffsetBytes, dstBuffer, dstOffsetBytes, 8);
          return;
        }
        const srcRanges = normalizeToContiguousRanges(src, srcOffsetBytes, 8);
        if (srcRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, dstBuffer, dstOffsetBytes, 8);
          return;
        }
        const chunkBuf = this._ensureLargeChunkBuffer(8);
        this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
        commandEncoder.copyBufferToBuffer(chunkBuf, 0, dstBuffer, dstOffsetBytes, 8);
      }
    
      _copyComplexToAny(commandEncoder, srcBuffer, srcOffsetBytes, dst, dstOffsetBytes) {
        if (isGpuBuffer(dst)) {
          commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dst, dstOffsetBytes, 8);
          return;
        }
        const dstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes, 8);
        if (dstRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dstRanges[0].buffer, dstRanges[0].offsetBytes, 8);
          return;
        }
        const chunkBuf = this._ensureLargeChunkBuffer(8);
        commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, chunkBuf, 0, 8);
        this.copier.unpack(commandEncoder, chunkBuf, 0, dstRanges);
      }
    
      _copyAnyToAny(commandEncoder, { src, srcOffsetBytes, dst, dstOffsetBytes, bytes }) {
        if (!Number.isInteger(bytes) || bytes < 0) {
          throw new Error(`_copyAnyToAny expects non-negative integer bytes; got ${bytes}`);
        }
        if (bytes === 0) return;
        const wholeSrcRanges = normalizeToContiguousRanges(src, srcOffsetBytes, bytes);
        const wholeDstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes, bytes);
        if (wholeSrcRanges.length === 1 && wholeDstRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(
            wholeSrcRanges[0].buffer,
            wholeSrcRanges[0].offsetBytes,
            wholeDstRanges[0].buffer,
            wholeDstRanges[0].offsetBytes,
            bytes
          );
          return;
        }
        const maxBufferSize = this.device.limits?.maxBufferSize ?? bytes;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const preferredChunk = Number.isFinite(this._maxBindBytes)
          ? Math.min(maxBufferSize, Math.max(storageAlign, this._maxBindBytes))
          : maxBufferSize;
        let chunkBytes = alignDownBytes(preferredChunk, 4);
        if (!Number.isInteger(chunkBytes) || chunkBytes <= 0) chunkBytes = Math.max(4, Math.min(bytes, 1024 * 1024));
        const chunkBuf = this._ensureLargeChunkBuffer(chunkBytes);
    
        for (let off = 0; off < bytes; off += chunkBytes) {
          const n = Math.min(chunkBytes, bytes - off);
          const srcRanges = normalizeToContiguousRanges(src, srcOffsetBytes + off, n);
          const dstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes + off, n);
          if (srcRanges.length === 1 && dstRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(
              srcRanges[0].buffer,
              srcRanges[0].offsetBytes,
              dstRanges[0].buffer,
              dstRanges[0].offsetBytes,
              n
            );
            continue;
          }
          if (srcRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, chunkBuf, 0, n);
            this.copier.unpack(commandEncoder, chunkBuf, 0, dstRanges);
            continue;
          }
          if (dstRanges.length === 1) {
            this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
            commandEncoder.copyBufferToBuffer(chunkBuf, 0, dstRanges[0].buffer, dstRanges[0].offsetBytes, n);
            continue;
          }
          this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
          this.copier.unpack(commandEncoder, chunkBuf, 0, dstRanges);
        }
      }
    
      _createInternalSegmentedView(totalBytes, preferredSegmentBytes) {
        if (!Number.isInteger(totalBytes) || totalBytes <= 0) {
          throw new Error(`_createInternalSegmentedView requires positive totalBytes; got ${totalBytes}`);
        }
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        let segmentBytes = Math.min(totalBytes, preferredSegmentBytes ?? totalBytes, maxBufferSize);
        segmentBytes = alignDownBytes(segmentBytes, storageAlign);
        if (!Number.isInteger(segmentBytes) || segmentBytes <= 0) {
          segmentBytes = alignBytes(Math.min(totalBytes, maxBufferSize), storageAlign);
        }
        if (segmentBytes <= 0 || segmentBytes > maxBufferSize) {
          throw new Error(
            `Unable to allocate segmented internal view: segmentBytes=${segmentBytes}, maxBufferSize=${maxBufferSize}, totalBytes=${totalBytes}`
          );
        }
        const segments = [];
        let remaining = totalBytes;
        while (remaining > 0) {
          const takeRaw = Math.min(remaining, segmentBytes);
          const take = remaining === takeRaw ? takeRaw : alignDownBytes(takeRaw, storageAlign);
          if (!Number.isInteger(take) || take <= 0) {
            throw new Error(`Failed to split segmented view with storage alignment ${storageAlign}`);
          }
          const buffer = this.device.createBuffer({
            size: take,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          segments.push({ buffer, offsetBytes: 0, sizeBytes: take });
          remaining -= take;
        }
        return {
          segmentBytes,
          view: {
            segments,
            logicalByteOffset: 0,
            lengthBytes: totalBytes,
          },
        };
      }
    
      _destroySegmentedView(view) {
        const segs = view?.segments;
        if (!Array.isArray(segs)) return;
        for (const s of segs) {
          s?.buffer?.destroy?.();
        }
      }
    
      _requiredStridedInputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
        if (!this._inputTensorDesc) {
          throw new Error("internal error: strided input descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._inputTensorDesc, {
          bytesPerElement: 8,
          runtimeExtraElements: extraOffsetElements,
          batchStart,
          batchCount,
        });
      }
    
      _requiredStridedOutputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
        if (!this._outputTensorDesc) {
          throw new Error("internal error: strided output descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._outputTensorDesc, {
          bytesPerElement: 8,
          runtimeExtraElements: extraOffsetElements,
          batchStart,
          batchCount,
        });
      }
    
      _shapeStrides(shape) {
        return tensorContiguousStrides(shape);
      }
    
      _coordsFromLinear(i, shape, outCoords) {
        tensorCoordsFromLinear(i, shape, outCoords);
      }
    
      _linearFromCoords(coords, strides) {
        return tensorLinearFromCoords(coords, strides);
      }
    
      _copyLogicalMap(commandEncoder, {
        srcBuffer,
        srcOffsetBytes,
        dstBuffer,
        dstOffsetBytes,
        srcShape,
        dstShape,
        mapCoordFn,
        batch,
      }) {
        const logicalTotal = prod(srcShape);
        const srcStrides = this._shapeStrides(srcShape);
        const dstStrides = this._shapeStrides(dstShape);
        const srcCoords = new Array(srcShape.length).fill(0);
        const dstCoords = new Array(dstShape.length).fill(0);
        const perSrcBatchBytes = logicalTotal * 8;
        const perDstBatchBytes = prod(dstShape) * 8;
        for (let b = 0; b < batch; b++) {
          const srcBase = srcOffsetBytes + b * perSrcBatchBytes;
          const dstBase = dstOffsetBytes + b * perDstBatchBytes;
          for (let i = 0; i < logicalTotal; i++) {
            this._coordsFromLinear(i, srcShape, srcCoords);
            mapCoordFn(srcCoords, dstCoords);
            const srcIdx = this._linearFromCoords(srcCoords, srcStrides);
            const dstIdx = this._linearFromCoords(dstCoords, dstStrides);
            commandEncoder.copyBufferToBuffer(srcBuffer, srcBase + srcIdx * 8, dstBuffer, dstBase + dstIdx * 8, 8);
          }
        }
      }
    
      _zeroLogicalOutsideRange(commandEncoder, { dataBuffer, dataOffsetBytes, start, end }) {
        const logicalTotal = this.logicalTotal;
        const coords = new Array(this.rank).fill(0);
        const zeroBuf = this._ensureZeroComplexBuffer();
        for (let b = 0; b < this.batch; b++) {
          const base = dataOffsetBytes + b * logicalTotal * 8;
          for (let i = 0; i < logicalTotal; i++) {
            this._coordsFromLinear(i, this.shape, coords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              if (coords[d] < start[d] || coords[d] >= end[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              commandEncoder.copyBufferToBuffer(zeroBuf, 0, dataBuffer, base + i * 8, 8);
            }
          }
        }
      }
    
      _transposeOutOfCore2dCopies(commandEncoder, {
        srcBuffer,
        srcOffsetBytes,
        dstBuffer,
        dstOffsetBytes,
        Nx,
        Ny,
        batch,
      }) {
        const elemBytes = 8;
        const perBatchBytes = Nx * Ny * elemBytes;
        for (let b = 0; b < batch; b++) {
          const srcBase = srcOffsetBytes + b * perBatchBytes;
          const dstBase = dstOffsetBytes + b * perBatchBytes;
          for (let y = 0; y < Ny; y++) {
            for (let x = 0; x < Nx; x++) {
              const src = srcBase + (y * Nx + x) * elemBytes;
              const dst = dstBase + (x * Ny + y) * elemBytes;
              commandEncoder.copyBufferToBuffer(srcBuffer, src, dstBuffer, dst, elemBytes);
            }
          }
        }
      }
    
      _getOutOfCoreTransposePipeline(stripeNx, stripeNy) {
        if (!this._outOfCoreTranspose) {
          throw new Error("Internal error: out-of-core transpose state is not initialized");
        }
        const key = `${stripeNx}x${stripeNy}`;
        const existing = this._outOfCoreTransposePipelines.get(key);
        if (existing) return existing;
        const code = generateTransposeComplex2DWGSL({ Nx: stripeNx, Ny: stripeNy, tile: this._outOfCoreTranspose.tile });
        const pipeline = this.cache.getComputePipeline({ code, layout: this._outOfCoreTranspose.pl });
        this._outOfCoreTransposePipelines.set(key, pipeline);
        return pipeline;
      }
    
      _transposeOutOfCore2dStripes(commandEncoder, {
        srcBuffer,
        srcOffsetBytes,
        dstBuffer,
        dstOffsetBytes,
        Nx,
        Ny,
        batch,
      }) {
        if (!this._outOfCoreTranspose) {
          this._transposeOutOfCore2dCopies(commandEncoder, {
            srcBuffer,
            srcOffsetBytes,
            dstBuffer,
            dstOffsetBytes,
            Nx,
            Ny,
            batch,
          });
          return;
        }
    
        if (Ny * 8 > this._maxBindBytes) {
          this._transposeOutOfCore2dCopies(commandEncoder, {
            srcBuffer,
            srcOffsetBytes,
            dstBuffer,
            dstOffsetBytes,
            Nx,
            Ny,
            batch,
          });
          return;
        }
    
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxStripeNx = Math.max(1, Math.floor(this._maxBindBytes / (Ny * 8)));
        const maxStripeBytes = maxStripeNx * Ny * 8;
        const stripeDstOffset = alignBytes(maxStripeBytes, storageAlign);
        const stripeScratchBytes = stripeDstOffset + maxStripeBytes;
        const stripeBuf = this._ensureLargeChunkBuffer(stripeScratchBytes);
        const stripeWidths = new Set();
        for (let x0 = 0; x0 < Nx; x0 += maxStripeNx) {
          stripeWidths.add(Math.min(maxStripeNx, Nx - x0));
        }
        try {
          for (const bx of stripeWidths) this._getOutOfCoreTransposePipeline(bx, Ny);
        } catch {
          this._transposeOutOfCore2dCopies(commandEncoder, {
            srcBuffer,
            srcOffsetBytes,
            dstBuffer,
            dstOffsetBytes,
            Nx,
            Ny,
            batch,
          });
          return;
        }
    
        for (let b = 0; b < batch; b++) {
          const srcBase = srcOffsetBytes + b * Nx * Ny * 8;
          const dstBase = dstOffsetBytes + b * Nx * Ny * 8;
          for (let x0 = 0; x0 < Nx; x0 += maxStripeNx) {
            const bx = Math.min(maxStripeNx, Nx - x0);
            const stripeBytes = bx * Ny * 8;
            const pipeline = this._getOutOfCoreTransposePipeline(bx, Ny);
    
            for (let y = 0; y < Ny; y++) {
              const srcRow = srcBase + (y * Nx + x0) * 8;
              const dstRow = y * bx * 8;
              commandEncoder.copyBufferToBuffer(srcBuffer, srcRow, stripeBuf, dstRow, bx * 8);
            }
    
            this.device.queue.writeBuffer(this._outOfCoreTranspose.params, 0, new Uint32Array([1, 0, 0, 0]));
            const bg = this.device.createBindGroup({
              layout: this._outOfCoreTranspose.bgl,
              entries: [
                { binding: 0, resource: { buffer: stripeBuf, offset: 0, size: stripeBytes } },
                { binding: 1, resource: { buffer: stripeBuf, offset: stripeDstOffset, size: stripeBytes } },
                { binding: 2, resource: { buffer: this._outOfCoreTranspose.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(bx / this._outOfCoreTranspose.tile), Math.ceil(Ny / this._outOfCoreTranspose.tile), 1);
            pass.end();
    
            const dstBlock = dstBase + x0 * Ny * 8;
            commandEncoder.copyBufferToBuffer(stripeBuf, stripeDstOffset, dstBuffer, dstBlock, stripeBytes);
          }
        }
      }
    
      _resolveOutOfCoreAxis1TailChunk() {
        if (this._outOfCoreAxis1TailChunk !== null) return this._outOfCoreAxis1TailChunk || null;
        if (!this._outOfCoreAxis1TailPermute || this.precision !== "f32" || this.rank < 3) {
          this._outOfCoreAxis1TailChunk = false;
          return null;
        }
        const X = this.shape[0];
        const Y = this.shape[1];
        const tail = prod(this.shape.slice(2));
        const XY = X * Y;
        if (tail < 1 || XY < 1) {
          this._outOfCoreAxis1TailChunk = false;
          return null;
        }
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxBindBytes = this._maxBindBytes;
        if (!Number.isFinite(maxBindBytes) || maxBindBytes < 8) {
          this._outOfCoreAxis1TailChunk = false;
          return null;
        }
        const effectiveBindBytes = Math.max(0, Math.floor(maxBindBytes - storageAlign));
        const maxElemsByBind = Math.floor(effectiveBindBytes / 8);
        if (maxElemsByBind < XY) {
          this._outOfCoreAxis1TailChunk = false;
          return null;
        }
        const maxWgX = maxComputeWorkgroupsPerDimension(this.device.limits, 0);
        const maxElemsByDispatch = Number.isFinite(maxWgX) ? Math.max(1, Math.floor(maxWgX * this.workgroupSize)) : Infinity;
        const maxTailByBind = Math.floor(maxElemsByBind / XY);
        const maxTailByDispatch = Math.floor(maxElemsByDispatch / XY);
        const chunkTail = Math.min(tail, maxTailByBind, maxTailByDispatch);
        if (chunkTail < 1) {
          this._outOfCoreAxis1TailChunk = false;
          return null;
        }
        this._outOfCoreAxis1TailChunk = { tailPerChunk: chunkTail, XY };
        return this._outOfCoreAxis1TailChunk;
      }
    
      _tryPermuteAxis1TailWithKernel(commandEncoder, { srcRange, dstRange, toFront }) {
        if (!this._outOfCoreAxis1TailPermute || this.precision !== "f32" || this.rank < 3) return false;
        const chunkCfg = this._resolveOutOfCoreAxis1TailChunk();
        if (!chunkCfg) return false;
    
        const tail = prod(this.shape.slice(2));
        const XY = chunkCfg.XY;
        const perBatchElems = this.logicalTotal;
        const perBatchBytes = perBatchElems * 8;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const state = this._outOfCoreAxis1TailPermute;
        const pipeline = toFront ? state.toFront : state.fromFront;
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
    
        for (let b = 0; b < this.batch; b++) {
          const srcBatchBase = srcRange.offsetBytes + b * perBatchBytes;
          const dstBatchBase = dstRange.offsetBytes + b * perBatchBytes;
          for (let t0 = 0; t0 < tail; t0 += chunkCfg.tailPerChunk) {
            const tailCount = Math.min(chunkCfg.tailPerChunk, tail - t0);
            const spanElems = tailCount * XY;
            const count = spanElems;
            const windowStartElems = t0 * XY;
            const srcWindowStartBytes = srcBatchBase + windowStartElems * 8;
            const dstWindowStartBytes = dstBatchBase + windowStartElems * 8;
            const srcBindOffset = alignDownBytes(srcWindowStartBytes, storageAlign);
            const dstBindOffset = alignDownBytes(dstWindowStartBytes, storageAlign);
            const srcStartElems = Math.floor((srcWindowStartBytes - srcBindOffset) / 8);
            const dstStartElems = Math.floor((dstWindowStartBytes - dstBindOffset) / 8);
            const srcBindBytes = srcStartElems * 8 + spanElems * 8;
            const dstBindBytes = dstStartElems * 8 + spanElems * 8;
            this.device.queue.writeBuffer(state.params, 0, new Uint32Array([count, srcStartElems, dstStartElems, 0]));
            const bg = this.device.createBindGroup({
              layout: state.bgl,
              entries: [
                { binding: 0, resource: { buffer: srcRange.buffer, offset: srcBindOffset, size: srcBindBytes } },
                { binding: 1, resource: { buffer: dstRange.buffer, offset: dstBindOffset, size: dstBindBytes } },
                { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
              ],
            });
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
          }
        }
        pass.end();
        return true;
      }
    
      _getOutOfCoreAdjacentSwapPipeline(X, Y) {
        if (!this._outOfCoreGenericPermute) {
          throw new Error("Internal error: adjacent swap pipeline requires generic out-of-core permute state");
        }
        if (!this._outOfCoreAdjacentSwapPipelines) this._outOfCoreAdjacentSwapPipelines = new Map();
        const key = `${X}x${Y}`;
        const existing = this._outOfCoreAdjacentSwapPipelines.get(key);
        if (existing) return existing;
        const code = generatePermuteAxis1TailToFrontWGSL({
          shape: [X, Y],
          workgroupSize: this.workgroupSize,
        });
        const pipeline = this.cache.getComputePipeline({ code, layout: this._outOfCoreGenericPermute.pl });
        this._outOfCoreAdjacentSwapPipelines.set(key, pipeline);
        return pipeline;
      }
    
      _getOutOfCoreAdjacentSwapTiledPipeline(X, Y) {
        if (!this._outOfCoreAdjacentSwapTiled) {
          throw new Error("Internal error: adjacent tiled swap pipeline requires out-of-core tiled state");
        }
        if (!this._outOfCoreAdjacentSwapTiledPipelines) this._outOfCoreAdjacentSwapTiledPipelines = new Map();
        const key = `${X}x${Y}`;
        const existing = this._outOfCoreAdjacentSwapTiledPipelines.get(key);
        if (existing) return existing;
        const code = generatePermuteAxis1TailTiledToFrontWGSL({
          shape: [X, Y],
          workgroupSize: this.workgroupSize,
        });
        const pipeline = this.cache.getComputePipeline({ code, layout: this._outOfCoreAdjacentSwapTiled.pl });
        this._outOfCoreAdjacentSwapTiledPipelines.set(key, pipeline);
        return pipeline;
      }
    
      _resolveOutOfCoreAdjacentSwapChunk({ X, Y, tail }) {
        if (!(X > 0 && Y > 0 && tail > 0)) return null;
        const XY = X * Y;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxBindBytes = this._maxBindBytes;
        if (!Number.isFinite(maxBindBytes) || maxBindBytes < 8) return null;
        const effectiveBindBytes = Math.max(0, Math.floor(maxBindBytes - storageAlign));
        const maxElemsByBind = Math.floor(effectiveBindBytes / 8);
        if (maxElemsByBind < XY) return null;
        const maxWgX = maxComputeWorkgroupsPerDimension(this.device.limits, 0);
        const maxElemsByDispatch = Number.isFinite(maxWgX) ? Math.max(1, Math.floor(maxWgX * this.workgroupSize)) : Infinity;
        const maxTailByBind = Math.floor(maxElemsByBind / XY);
        const maxTailByDispatch = Math.floor(maxElemsByDispatch / XY);
        const tailPerChunk = Math.min(tail, maxTailByBind, maxTailByDispatch);
        if (tailPerChunk < 1) return null;
        return { XY, tailPerChunk };
      }
    
      _dispatchOutOfCoreAdjacentSwap(commandEncoder, { srcRange, dstRange, outerGroups, X, Y, tail }) {
        const chunkCfg = this._resolveOutOfCoreAdjacentSwapChunk({ X, Y, tail });
        if (!chunkCfg) {
          return this._dispatchOutOfCoreAdjacentSwapTiled(commandEncoder, { srcRange, dstRange, outerGroups, X, Y, tail });
        }
        const state = this._outOfCoreGenericPermute;
        const pipeline = this._getOutOfCoreAdjacentSwapPipeline(X, Y);
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const XY = chunkCfg.XY;
        const elemsPerOuter = XY * tail;
        const bytesPerOuter = elemsPerOuter * 8;
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
    
        for (let g = 0; g < outerGroups; g++) {
          const srcOuterBase = srcRange.offsetBytes + g * bytesPerOuter;
          const dstOuterBase = dstRange.offsetBytes + g * bytesPerOuter;
          for (let t0 = 0; t0 < tail; t0 += chunkCfg.tailPerChunk) {
            const tailCount = Math.min(chunkCfg.tailPerChunk, tail - t0);
            const spanElems = tailCount * XY;
            const count = spanElems;
            const windowStartElems = t0 * XY;
            const srcWindowStartBytes = srcOuterBase + windowStartElems * 8;
            const dstWindowStartBytes = dstOuterBase + windowStartElems * 8;
            const srcBindOffset = alignDownBytes(srcWindowStartBytes, storageAlign);
            const dstBindOffset = alignDownBytes(dstWindowStartBytes, storageAlign);
            const srcStartElems = Math.floor((srcWindowStartBytes - srcBindOffset) / 8);
            const dstStartElems = Math.floor((dstWindowStartBytes - dstBindOffset) / 8);
            const srcBindBytes = srcStartElems * 8 + spanElems * 8;
            const dstBindBytes = dstStartElems * 8 + spanElems * 8;
            this.device.queue.writeBuffer(state.params, 0, new Uint32Array([count, srcStartElems, dstStartElems, 0]));
            const bg = this.device.createBindGroup({
              layout: state.bgl,
              entries: [
                { binding: 0, resource: { buffer: srcRange.buffer, offset: srcBindOffset, size: srcBindBytes } },
                { binding: 1, resource: { buffer: dstRange.buffer, offset: dstBindOffset, size: dstBindBytes } },
                { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
              ],
            });
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
          }
        }
    
        pass.end();
        return true;
      }
    
      _dispatchOutOfCoreAdjacentSwapTiled(commandEncoder, { srcRange, dstRange, outerGroups, X, Y, tail }) {
        if (!this._outOfCoreAdjacentSwapTiled) return false;
        const maxBindBytes = this._maxBindBytes;
        if (!Number.isFinite(maxBindBytes) || maxBindBytes < 8) return false;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const effectiveBindBytes = Math.max(0, Math.floor(maxBindBytes - storageAlign));
        const maxElemsByBind = Math.floor(effectiveBindBytes / 8);
        if (maxElemsByBind < 1) return false;
        const maxWgX = maxComputeWorkgroupsPerDimension(this.device.limits, 0);
        const maxElemsByDispatch = Number.isFinite(maxWgX) ? Math.max(1, Math.floor(maxWgX * this.workgroupSize)) : Infinity;
        const XY = X * Y;
        const elemsPerOuter = XY * tail;
        const bytesPerOuter = elemsPerOuter * 8;
        const pipeline = this._getOutOfCoreAdjacentSwapTiledPipeline(X, Y);
        const state = this._outOfCoreAdjacentSwapTiled;
        const maxTailByBind = Math.max(1, Math.floor((maxElemsByBind - 1) / XY) + 1);
        const maxTailByDispatch = Math.max(1, Math.floor(maxElemsByDispatch));
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
    
        for (let g = 0; g < outerGroups; g++) {
          const srcOuterBase = srcRange.offsetBytes + g * bytesPerOuter;
          const dstOuterBase = dstRange.offsetBytes + g * bytesPerOuter;
          for (let y0 = 0; y0 < Y; y0++) {
            for (let t0 = 0; t0 < tail; ) {
              const htail = Math.max(1, Math.min(tail - t0, maxTailByBind, maxTailByDispatch));
              const srcHxCap = maxElemsByBind - (htail - 1) * XY;
              const dstHxCap = Math.floor((maxElemsByBind - (htail - 1) * XY - 1) / Y) + 1;
              const dispatchHxCap = Math.floor(maxElemsByDispatch / htail);
              const maxHx = Math.min(X, srcHxCap, dstHxCap, dispatchHxCap);
              if (!Number.isFinite(maxHx) || maxHx < 1) return false;
              for (let x0 = 0; x0 < X; x0 += maxHx) {
                const hx = Math.min(maxHx, X - x0);
                const srcLocalStartElems = x0 + y0 * X + t0 * XY;
                const dstLocalStartElems = y0 + x0 * Y + t0 * XY;
                const srcWindowStartBytes = srcOuterBase + srcLocalStartElems * 8;
                const dstWindowStartBytes = dstOuterBase + dstLocalStartElems * 8;
                const srcBindOffset = alignDownBytes(srcWindowStartBytes, storageAlign);
                const dstBindOffset = alignDownBytes(dstWindowStartBytes, storageAlign);
                const srcStartElems = Math.floor((srcWindowStartBytes - srcBindOffset) / 8);
                const dstStartElems = Math.floor((dstWindowStartBytes - dstBindOffset) / 8);
                const srcSpanElems = (htail - 1) * XY + hx;
                const dstSpanElems = (htail - 1) * XY + (hx - 1) * Y + 1;
                const srcBindBytes = srcStartElems * 8 + srcSpanElems * 8;
                const dstBindBytes = dstStartElems * 8 + dstSpanElems * 8;
                const count = hx * htail;
                this.device.queue.writeBuffer(
                  state.params,
                  0,
                  new Uint32Array([count, hx, htail, srcStartElems, dstStartElems, 0, 0, 0])
                );
                const bg = this.device.createBindGroup({
                  layout: state.bgl,
                  entries: [
                    { binding: 0, resource: { buffer: srcRange.buffer, offset: srcBindOffset, size: srcBindBytes } },
                    { binding: 1, resource: { buffer: dstRange.buffer, offset: dstBindOffset, size: dstBindBytes } },
                    { binding: 2, resource: { buffer: state.params, offset: 0, size: 32 } },
                  ],
                });
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
              }
              t0 += htail;
            }
          }
        }
    
        pass.end();
        return true;
      }
    
      _tryPermuteViaAdjacentSwaps(commandEncoder, { srcRange, dstRange, axis, toFront }) {
        if (!this._outOfCoreGenericPermute || this.precision !== "f32" || this.rank < 2) return false;
        if (!Number.isInteger(axis) || axis < 1 || axis >= this.rank) return false;
        const totalBytes = this.batch * this.logicalTotal * 8;
        if (srcRange.sizeBytes < totalBytes || dstRange.sizeBytes < totalBytes) return false;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        if (srcRange.offsetBytes % storageAlign !== 0 || dstRange.offsetBytes % storageAlign !== 0) return false;
    
        const steps = [];
        if (toFront) {
          for (let k = axis; k >= 1; k--) steps.push(k - 1);
        } else {
          for (let k = 0; k < axis; k++) steps.push(k);
        }
        if (steps.length === 0) return true;
    
        let currentShape = toFront ? this.shape.slice() : permutedShapeAxisFront(this.shape, axis);
        let readRange = srcRange;
        let writeRange = dstRange;
    
        for (const left of steps) {
          const right = left + 1;
          const X = currentShape[left];
          const Y = currentShape[right];
          const tail = prod(currentShape.slice(right + 1));
          const outerGroups = this.batch * prod(currentShape.slice(0, left));
          const ok = this._dispatchOutOfCoreAdjacentSwap(commandEncoder, {
            srcRange: readRange,
            dstRange: writeRange,
            outerGroups,
            X,
            Y,
            tail,
          });
          if (!ok) return false;
          const tmp = currentShape[left];
          currentShape[left] = currentShape[right];
          currentShape[right] = tmp;
          const r = readRange;
          readRange = writeRange;
          writeRange = r;
        }
    
        if (readRange.buffer !== dstRange.buffer || readRange.offsetBytes !== dstRange.offsetBytes) {
          commandEncoder.copyBufferToBuffer(readRange.buffer, readRange.offsetBytes, dstRange.buffer, dstRange.offsetBytes, totalBytes);
        }
        return true;
      }
    
      _rank3Axis2WindowSpansElems(hy, hz) {
        const X = this.shape[0];
        const Y = this.shape[1];
        const Z = this.shape[2];
        return {
          srcSpanElems: (hz - 1) * X * Y + hy * X,
          dstSpanElems: hy * Z * X - Z + hz,
        };
      }
    
      _resolveOutOfCoreRank3Axis2Tile() {
        if (this._outOfCoreRank3Axis2Tile !== null) {
          return this._outOfCoreRank3Axis2Tile || null;
        }
        if (!this._outOfCoreRank3Axis2Permute || this.precision !== "f32" || this.rank !== 3) {
          this._outOfCoreRank3Axis2Tile = false;
          return null;
        }
    
        const [X, Y, Z] = this.shape;
        if (X <= 0 || Y <= 0 || Z <= 0) {
          this._outOfCoreRank3Axis2Tile = false;
          return null;
        }
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxBindBytes = this._maxBindBytes;
        if (!Number.isFinite(maxBindBytes) || maxBindBytes < 8) {
          this._outOfCoreRank3Axis2Tile = false;
          return null;
        }
        const effectiveBindBytes = Math.max(0, Math.floor(maxBindBytes - storageAlign));
        const maxElemsByBind = Math.floor(effectiveBindBytes / 8);
        if (maxElemsByBind < 1) {
          this._outOfCoreRank3Axis2Tile = false;
          return null;
        }
        const maxWgX = maxComputeWorkgroupsPerDimension(this.device.limits, 0);
        const maxElemsByDispatch = Number.isFinite(maxWgX) ? Math.max(1, Math.floor(maxWgX * this.workgroupSize)) : Infinity;
    
        let bestHy = 0;
        let bestHz = 0;
        let bestArea = 0;
        for (let hy = 1; hy <= Y; hy++) {
          const hzByDstSpan = maxElemsByBind - (hy * Z * X - Z);
          if (hzByDstSpan < 1) break;
          const hzBySrcSpan = Math.floor((maxElemsByBind - hy * X) / (X * Y)) + 1;
          const hzByDispatch = Math.floor(maxElemsByDispatch / (X * hy));
          const hz = Math.min(Z, hzByDstSpan, hzBySrcSpan, hzByDispatch);
          if (hz < 1) continue;
          const area = hy * hz;
          if (area > bestArea) {
            bestArea = area;
            bestHy = hy;
            bestHz = hz;
          }
        }
    
        if (bestArea < 1) {
          this._outOfCoreRank3Axis2Tile = false;
          return null;
        }
        this._outOfCoreRank3Axis2Tile = { hy: bestHy, hz: bestHz };
        return this._outOfCoreRank3Axis2Tile;
      }
    
      _tryPermuteRank3Axis2WithKernel(commandEncoder, { srcRange, dstRange, toFront }) {
        if (!this._outOfCoreRank3Axis2Permute || this.precision !== "f32" || this.rank !== 3) return false;
        const tile = this._resolveOutOfCoreRank3Axis2Tile();
        if (!tile) return false;
    
        const [X, Y, Z] = this.shape;
        const perBatchElems = this.logicalTotal;
        const perBatchBytes = perBatchElems * 8;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const state = this._outOfCoreRank3Axis2Permute;
        const pipeline = toFront ? state.toFront : state.fromFront;
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
    
        for (let b = 0; b < this.batch; b++) {
          const srcBatchBase = srcRange.offsetBytes + b * perBatchBytes;
          const dstBatchBase = dstRange.offsetBytes + b * perBatchBytes;
          for (let y0 = 0; y0 < Y; y0 += tile.hy) {
            const hy = Math.min(tile.hy, Y - y0);
            for (let z0 = 0; z0 < Z; z0 += tile.hz) {
              const hz = Math.min(tile.hz, Z - z0);
              const count = X * hy * hz;
              const { srcSpanElems, dstSpanElems } = this._rank3Axis2WindowSpansElems(hy, hz);
              const srcMinElems = toFront ? y0 * X + z0 * X * Y : z0 + y0 * Z * X;
              const dstMinElems = toFront ? z0 + y0 * Z * X : y0 * X + z0 * X * Y;
              const srcWindowStartBytes = srcBatchBase + srcMinElems * 8;
              const dstWindowStartBytes = dstBatchBase + dstMinElems * 8;
              const srcBindOffset = alignDownBytes(srcWindowStartBytes, storageAlign);
              const dstBindOffset = alignDownBytes(dstWindowStartBytes, storageAlign);
              const srcStartElems = Math.floor((srcWindowStartBytes - srcBindOffset) / 8);
              const dstStartElems = Math.floor((dstWindowStartBytes - dstBindOffset) / 8);
              const srcBindBytes = srcStartElems * 8 + srcSpanElems * 8;
              const dstBindBytes = dstStartElems * 8 + dstSpanElems * 8;
              this.device.queue.writeBuffer(state.params, 0, new Uint32Array([count, hz, srcStartElems, dstStartElems]));
              const bg = this.device.createBindGroup({
                layout: state.bgl,
                entries: [
                  { binding: 0, resource: { buffer: srcRange.buffer, offset: srcBindOffset, size: srcBindBytes } },
                  { binding: 1, resource: { buffer: dstRange.buffer, offset: dstBindOffset, size: dstBindBytes } },
                  { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
                ],
              });
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
            }
          }
        }
    
        pass.end();
        return true;
      }
    
      _getOutOfCoreGenericPermutePipeline(axis, toFront) {
        if (!this._outOfCoreGenericPermute) {
          throw new Error("Internal error: generic out-of-core permute state is not initialized");
        }
        const key = `${toFront ? "to" : "from"}:${axis}`;
        const existing = this._outOfCoreGenericPermutePipelines?.get(key);
        if (existing) return existing;
        const code = generatePermuteAxisGenericWGSL({
          shape: this.shape,
          axis,
          toFront,
          workgroupSize: this.workgroupSize,
        });
        const pipeline = this.cache.getComputePipeline({ code, layout: this._outOfCoreGenericPermute.pl });
        this._outOfCoreGenericPermutePipelines?.set(key, pipeline);
        return pipeline;
      }
    
      _tryPermuteGenericWithKernel(commandEncoder, { srcRange, dstRange, axis, toFront }) {
        if (!this._outOfCoreGenericPermute || this.precision !== "f32") return false;
        const totalElems = this.batch * this.logicalTotal;
        const totalBytes = totalElems * 8;
        if (srcRange.sizeBytes < totalBytes || dstRange.sizeBytes < totalBytes) return false;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        if (srcRange.offsetBytes % storageAlign !== 0 || dstRange.offsetBytes % storageAlign !== 0) return false;
        const maxBindBytes = this.device.limits?.maxStorageBufferBindingSize ?? Infinity;
        if (totalBytes > maxBindBytes) {
          const perBatchBytes = this.logicalTotal * 8;
          if (perBatchBytes > maxBindBytes) return false;
          const pipeline = this._getOutOfCoreGenericPermutePipeline(axis, toFront);
          const state = this._outOfCoreGenericPermute;
          this.device.queue.writeBuffer(state.params, 0, new Uint32Array([this.logicalTotal, 1, 0, 0]));
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(pipeline);
          for (let b = 0; b < this.batch; b++) {
            const srcOff = srcRange.offsetBytes + b * perBatchBytes;
            const dstOff = dstRange.offsetBytes + b * perBatchBytes;
            const bg = this.device.createBindGroup({
              layout: state.bgl,
              entries: [
                { binding: 0, resource: { buffer: srcRange.buffer, offset: srcOff, size: perBatchBytes } },
                { binding: 1, resource: { buffer: dstRange.buffer, offset: dstOff, size: perBatchBytes } },
                { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
              ],
            });
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
          }
          pass.end();
          return true;
        }
    
        const pipeline = this._getOutOfCoreGenericPermutePipeline(axis, toFront);
        const state = this._outOfCoreGenericPermute;
        this.device.queue.writeBuffer(state.params, 0, new Uint32Array([totalElems, this.batch, 0, 0]));
        const bg = this.device.createBindGroup({
          layout: state.bgl,
          entries: [
            { binding: 0, resource: { buffer: srcRange.buffer, offset: srcRange.offsetBytes, size: totalBytes } },
            { binding: 1, resource: { buffer: dstRange.buffer, offset: dstRange.offsetBytes, size: totalBytes } },
            { binding: 2, resource: { buffer: state.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(totalElems / this.workgroupSize), 1, 1);
        pass.end();
        return true;
      }
    
      _permuteAxisToFront(commandEncoder, { srcRange, dstRange, axis }) {
        if (axis === 1 && this._tryPermuteAxis1TailWithKernel(commandEncoder, { srcRange, dstRange, toFront: true })) {
          return;
        }
        if (axis === 2 && this._tryPermuteRank3Axis2WithKernel(commandEncoder, { srcRange, dstRange, toFront: true })) {
          return;
        }
        if (this._tryPermuteGenericWithKernel(commandEncoder, { srcRange, dstRange, axis, toFront: true })) {
          return;
        }
        if (this._tryPermuteViaAdjacentSwaps(commandEncoder, { srcRange, dstRange, axis, toFront: true })) {
          return;
        }
        const rank = this.rank;
        const srcShape = this.shape;
        const dstShape = permutedShapeAxisFront(srcShape, axis);
        const srcCoords = new Array(rank).fill(0);
        const dstCoords = new Array(rank).fill(0);
        const srcStrides = this._shapeStrides(srcShape);
        const dstStrides = this._shapeStrides(dstShape);
        const logicalTotal = this.logicalTotal;
        const perBytes = logicalTotal * 8;
        for (let b = 0; b < this.batch; b++) {
          const srcBase = srcRange.offsetBytes + b * perBytes;
          const dstBase = dstRange.offsetBytes + b * perBytes;
          for (let i = 0; i < logicalTotal; i++) {
            this._coordsFromLinear(i, srcShape, srcCoords);
            dstCoords[0] = srcCoords[axis];
            let p = 1;
            for (let d = 0; d < rank; d++) {
              if (d === axis) continue;
              dstCoords[p++] = srcCoords[d];
            }
            const srcIdx = this._linearFromCoords(srcCoords, srcStrides);
            const dstIdx = this._linearFromCoords(dstCoords, dstStrides);
            commandEncoder.copyBufferToBuffer(srcRange.buffer, srcBase + srcIdx * 8, dstRange.buffer, dstBase + dstIdx * 8, 8);
          }
        }
      }
    
      _permuteAxisFromFront(commandEncoder, { srcRange, dstRange, axis }) {
        if (axis === 1 && this._tryPermuteAxis1TailWithKernel(commandEncoder, { srcRange, dstRange, toFront: false })) {
          return;
        }
        if (axis === 2 && this._tryPermuteRank3Axis2WithKernel(commandEncoder, { srcRange, dstRange, toFront: false })) {
          return;
        }
        if (this._tryPermuteGenericWithKernel(commandEncoder, { srcRange, dstRange, axis, toFront: false })) {
          return;
        }
        if (this._tryPermuteViaAdjacentSwaps(commandEncoder, { srcRange, dstRange, axis, toFront: false })) {
          return;
        }
        const rank = this.rank;
        const srcShape = permutedShapeAxisFront(this.shape, axis);
        const dstShape = this.shape;
        const srcCoords = new Array(rank).fill(0);
        const dstCoords = new Array(rank).fill(0);
        const srcStrides = this._shapeStrides(srcShape);
        const dstStrides = this._shapeStrides(dstShape);
        const logicalTotal = this.logicalTotal;
        const perBytes = logicalTotal * 8;
        for (let b = 0; b < this.batch; b++) {
          const srcBase = srcRange.offsetBytes + b * perBytes;
          const dstBase = dstRange.offsetBytes + b * perBytes;
          for (let i = 0; i < logicalTotal; i++) {
            this._coordsFromLinear(i, srcShape, srcCoords);
            dstCoords[axis] = srcCoords[0];
            let p = 1;
            for (let d = 0; d < rank; d++) {
              if (d === axis) continue;
              dstCoords[d] = srcCoords[p++];
            }
            const srcIdx = this._linearFromCoords(srcCoords, srcStrides);
            const dstIdx = this._linearFromCoords(dstCoords, dstStrides);
            commandEncoder.copyBufferToBuffer(srcRange.buffer, srcBase + srcIdx * 8, dstRange.buffer, dstBase + dstIdx * 8, 8);
          }
        }
      }
    
      _embedInputOutOfCore(commandEncoder, { inputRanges, dataRange }) {
        if (!this._needsInputMapping) {
          if (inputRanges.length === 1) {
            if (inputRanges[0].buffer === dataRange.buffer && inputRanges[0].offsetBytes === dataRange.offsetBytes) {
              return;
            }
            commandEncoder.copyBufferToBuffer(inputRanges[0].buffer, inputRanges[0].offsetBytes, dataRange.buffer, dataRange.offsetBytes, this.mainBytes);
          } else {
            this._copyRangesToContiguous(commandEncoder, inputRanges, dataRange.buffer, dataRange.offsetBytes);
          }
          return;
        }
    
        const zeroBuf = this._ensureZeroComplexBuffer();
        const dataUsesAux = this._largeAuxBuffer && dataRange.buffer === this._largeAuxBuffer;
        let inContigBuf = null;
        if (dataUsesAux && this._largeAuxBytes < this._inPhysBytes) {
          inContigBuf = this._ensureLargeStageBuffer(this._inPhysBytes);
        } else {
          inContigBuf = this._ensureLargeAuxBuffer(this._inPhysBytes);
        }
        let inContigOff = 0;
        const inStageRange = { buffer: inContigBuf, offsetBytes: inContigOff, sizeBytes: this._inPhysBytes };
        if (this._rangesOverlap(inStageRange, dataRange)) {
          inContigBuf = this._ensureLargeStageBuffer(this._inPhysBytes);
          inContigOff = 0;
        }
        this._copyRangesToContiguous(commandEncoder, inputRanges, inContigBuf, inContigOff);
        const viewShape = this.io.input.shape;
        const viewOffset = this.io.input.offset;
        const viewTotal = this._inViewTotal;
        const logicalTotal = this.logicalTotal;
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        const viewStrides = this._shapeStrides(viewShape);
        for (let b = 0; b < this.batch; b++) {
          const srcBase = inContigOff + b * viewTotal * 8;
          const dstBase = dataRange.offsetBytes + b * logicalTotal * 8;
          for (let li = 0; li < logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              commandEncoder.copyBufferToBuffer(zeroBuf, 0, dataRange.buffer, dstBase + li * 8, 8);
              continue;
            }
            const vi = this._linearFromCoords(viewCoords, viewStrides);
            commandEncoder.copyBufferToBuffer(inContigBuf, srcBase + vi * 8, dataRange.buffer, dstBase + li * 8, 8);
          }
        }
      }
    
      _embedStridedInputOutOfCore(commandEncoder, { input, inputOffsetBytes, dataRange }) {
        if (inputOffsetBytes % 8 !== 0) {
          throw new Error(`inputOffsetBytes must be a multiple of 8 for complex-strided input; got ${inputOffsetBytes}`);
        }
        const extraOffsetElements = (inputOffsetBytes / 8) | 0;
        const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
        const inputBytes = getBufferByteLength(input);
        if (inputBytes < neededBytes) {
          throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
        }
    
        const zeroBuf = this._ensureZeroComplexBuffer();
        if (!this._needsInputMapping) {
          const coords = new Array(this.rank).fill(0);
          for (let b = 0; b < this.batch; b++) {
            const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
            const dstBase = dataRange.offsetBytes + b * this.logicalTotal * 8;
            for (let li = 0; li < this.logicalTotal; li++) {
              this._coordsFromLinear(li, this.shape, coords);
              let srcElem = srcBatchBase;
              for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
              this._copyComplexFromAny(commandEncoder, input, srcElem * 8, dataRange.buffer, dstBase + li * 8);
            }
          }
          return;
        }
    
        const viewShape = this.io.input.shape;
        const viewOffset = this.io.input.offset;
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
          const dstBase = dataRange.offsetBytes + b * this.logicalTotal * 8;
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              commandEncoder.copyBufferToBuffer(zeroBuf, 0, dataRange.buffer, dstBase + li * 8, 8);
              continue;
            }
            let srcElem = srcBatchBase;
            for (let d = 0; d < this.rank; d++) srcElem += viewCoords[d] * this._inputStrides[d];
            this._copyComplexFromAny(commandEncoder, input, srcElem * 8, dataRange.buffer, dstBase + li * 8);
          }
        }
      }
    
      _extractOutputOutOfCore(commandEncoder, { dataRange, outputRanges }) {
        if (!this._needsOutputMapping) {
          if (outputRanges.length === 1) {
            if (outputRanges[0].buffer === dataRange.buffer && outputRanges[0].offsetBytes === dataRange.offsetBytes) {
              return;
            }
            commandEncoder.copyBufferToBuffer(dataRange.buffer, dataRange.offsetBytes, outputRanges[0].buffer, outputRanges[0].offsetBytes, this.mainBytes);
          } else {
            this._copyContiguousToRanges(commandEncoder, dataRange.buffer, dataRange.offsetBytes, outputRanges);
          }
          return;
        }
    
        const viewShape = this.io.output.shape;
        const viewOffset = this.io.output.offset;
        const viewTotal = this._outViewTotal;
        const logicalTotal = this.logicalTotal;
        const outBytes = viewTotal * this.batch * 8;
        const dataUsesAux = this._largeAuxBuffer && dataRange.buffer === this._largeAuxBuffer;
        let outContigBuf = null;
        if (dataUsesAux && this._largeAuxBytes < outBytes) {
          outContigBuf = this._ensureLargeStageBuffer(outBytes);
        } else {
          outContigBuf = this._ensureLargeAuxBuffer(outBytes);
        }
        let outContigOff = 0;
        const outContigRange = { buffer: outContigBuf, offsetBytes: outContigOff, sizeBytes: outBytes };
        if (this._rangesOverlap(outContigRange, dataRange)) {
          outContigBuf = this._ensureLargeStageBuffer(outBytes);
          outContigOff = 0;
        }
        if (!this.io.output.clearOutside) {
          this._copyRangesToContiguous(commandEncoder, outputRanges, outContigBuf, outContigOff);
        } else {
          const zeroBuf = this._ensureZeroComplexBuffer();
          for (let i = 0; i < viewTotal * this.batch; i++) {
            commandEncoder.copyBufferToBuffer(zeroBuf, 0, outContigBuf, outContigOff + i * 8, 8);
          }
        }
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        const viewStrides = this._shapeStrides(viewShape);
        for (let b = 0; b < this.batch; b++) {
          const srcBase = dataRange.offsetBytes + b * logicalTotal * 8;
          const dstBase = b * viewTotal * 8;
          for (let li = 0; li < logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) continue;
            const vi = this._linearFromCoords(viewCoords, viewStrides);
            commandEncoder.copyBufferToBuffer(dataRange.buffer, srcBase + li * 8, outContigBuf, outContigOff + dstBase + vi * 8, 8);
          }
        }
        if (outputRanges.length === 1 && outputRanges[0].buffer === outContigBuf && outputRanges[0].offsetBytes === outContigOff) {
          return;
        }
        this._copyContiguousToRanges(commandEncoder, outContigBuf, outContigOff, outputRanges);
      }
    
      _extractStridedOutputOutOfCore(commandEncoder, { dataRange, output, outputOffsetBytes }) {
        if (outputOffsetBytes % 8 !== 0) {
          throw new Error(`outputOffsetBytes must be a multiple of 8 for complex-strided output; got ${outputOffsetBytes}`);
        }
        const extraOffsetElements = (outputOffsetBytes / 8) | 0;
        const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
        const outputBytes = getBufferByteLength(output);
        if (outputBytes < neededBytes) {
          throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
        }
    
        if (!this._needsOutputMapping) {
          const coords = new Array(this.rank).fill(0);
          for (let b = 0; b < this.batch; b++) {
            const srcBase = dataRange.offsetBytes + b * this.logicalTotal * 8;
            const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
            for (let li = 0; li < this.logicalTotal; li++) {
              this._coordsFromLinear(li, this.shape, coords);
              let dstElem = dstBatchBase;
              for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
              this._copyComplexToAny(commandEncoder, dataRange.buffer, srcBase + li * 8, output, dstElem * 8);
            }
          }
          return;
        }
    
        const viewShape = this.io.output.shape;
        const viewOffset = this.io.output.offset;
        const zeroBuf = this._ensureZeroComplexBuffer();
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
    
        if (this.io.output.clearOutside) {
          const viewTotal = this._outViewTotal;
          for (let b = 0; b < this.batch; b++) {
            const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
            for (let vi = 0; vi < viewTotal; vi++) {
              this._coordsFromLinear(vi, viewShape, viewCoords);
              let dstElem = dstBatchBase;
              for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
              this._copyComplexToAny(commandEncoder, zeroBuf, 0, output, dstElem * 8);
            }
          }
        }
    
        for (let b = 0; b < this.batch; b++) {
          const srcBase = dataRange.offsetBytes + b * this.logicalTotal * 8;
          const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) continue;
            let dstElem = dstBatchBase;
            for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
            this._copyComplexToAny(commandEncoder, dataRange.buffer, srcBase + li * 8, output, dstElem * 8);
          }
        }
      }
    
      _embedInputChunkLarge(commandEncoder, { input, inputOffsetBytes, batchStart, batchCount, dstBuffer, dstOffsetBytes }) {
        const chunkBytes = batchCount * this._bytesPerBatch;
        if (!this._needsInputMapping) {
          const srcOff = inputOffsetBytes + batchStart * this._inPhysBytesPerBatch;
          const srcRanges = normalizeToContiguousRanges(input, srcOff, chunkBytes);
          if (srcRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, dstBuffer, dstOffsetBytes, chunkBytes);
          } else {
            this.copier.pack(commandEncoder, srcRanges, dstBuffer, dstOffsetBytes);
          }
          return;
        }
    
        const viewShape = this.io.input.shape;
        const viewOffset = this.io.input.offset;
        const viewTotal = this._inViewTotal;
        const logicalTotal = this.logicalTotal;
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        const viewStrides = this._shapeStrides(viewShape);
        const zeroBuf = this._ensureZeroComplexBuffer();
        for (let lb = 0; lb < batchCount; lb++) {
          const gb = batchStart + lb;
          const srcBase = inputOffsetBytes + gb * viewTotal * 8;
          const dstBase = dstOffsetBytes + lb * logicalTotal * 8;
          for (let li = 0; li < logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              commandEncoder.copyBufferToBuffer(zeroBuf, 0, dstBuffer, dstBase + li * 8, 8);
              continue;
            }
            const vi = this._linearFromCoords(viewCoords, viewStrides);
            this._copyComplexFromAny(commandEncoder, input, srcBase + vi * 8, dstBuffer, dstBase + li * 8);
          }
        }
      }
    
      _embedStridedInputChunkLarge(commandEncoder, { input, inputOffsetBytes, batchStart, batchCount, dstBuffer, dstOffsetBytes }) {
        if (inputOffsetBytes % 8 !== 0) {
          throw new Error(`inputOffsetBytes must be a multiple of 8 for complex-strided input; got ${inputOffsetBytes}`);
        }
        const extraOffsetElements = (inputOffsetBytes / 8) | 0;
        const neededBytes = this._requiredStridedInputBytes(extraOffsetElements, batchStart, batchCount);
        const inputBytes = getBufferByteLength(input);
        if (inputBytes < neededBytes) {
          throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
        }
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        const zeroBuf = this._ensureZeroComplexBuffer();
        const viewShape = this.io.input?.shape ?? null;
        const viewOffset = this.io.input?.offset ?? null;
        for (let lb = 0; lb < batchCount; lb++) {
          const gb = batchStart + lb;
          const srcBatchBase = this._inputOffsetElements + extraOffsetElements + gb * this._inputBatchStrideElements;
          const dstBase = dstOffsetBytes + lb * this.logicalTotal * 8;
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            if (!this._needsInputMapping) {
              let srcElem = srcBatchBase;
              for (let d = 0; d < this.rank; d++) srcElem += logicalCoords[d] * this._inputStrides[d];
              this._copyComplexFromAny(commandEncoder, input, srcElem * 8, dstBuffer, dstBase + li * 8);
              continue;
            }
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              commandEncoder.copyBufferToBuffer(zeroBuf, 0, dstBuffer, dstBase + li * 8, 8);
              continue;
            }
            let srcElem = srcBatchBase;
            for (let d = 0; d < this.rank; d++) srcElem += viewCoords[d] * this._inputStrides[d];
            this._copyComplexFromAny(commandEncoder, input, srcElem * 8, dstBuffer, dstBase + li * 8);
          }
        }
      }
    
      _extractOutputChunkLarge(commandEncoder, { srcBuffer, srcOffsetBytes, output, outputOffsetBytes, batchStart, batchCount }) {
        const chunkBytes = batchCount * this._bytesPerBatch;
        if (!this._needsOutputMapping) {
          const dstOff = outputOffsetBytes + batchStart * this._outPhysBytesPerBatch;
          const outRanges = normalizeToContiguousRanges(output, dstOff, chunkBytes);
          if (outRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, outRanges[0].buffer, outRanges[0].offsetBytes, chunkBytes);
          } else {
            this.copier.unpack(commandEncoder, srcBuffer, srcOffsetBytes, outRanges);
          }
          return;
        }
        const viewShape = this.io.output.shape;
        const viewOffset = this.io.output.offset;
        const viewTotal = this._outViewTotal;
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        const viewStrides = this._shapeStrides(viewShape);
        const zeroBuf = this._ensureZeroComplexBuffer();
        for (let lb = 0; lb < batchCount; lb++) {
          const gb = batchStart + lb;
          const srcBase = srcOffsetBytes + lb * this.logicalTotal * 8;
          const dstBase = outputOffsetBytes + gb * viewTotal * 8;
          if (this.io.output.clearOutside) {
            for (let vi = 0; vi < viewTotal; vi++) {
              this._copyComplexToAny(commandEncoder, zeroBuf, 0, output, dstBase + vi * 8);
            }
          }
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) continue;
            const vi = this._linearFromCoords(viewCoords, viewStrides);
            this._copyComplexToAny(commandEncoder, srcBuffer, srcBase + li * 8, output, dstBase + vi * 8);
          }
        }
      }
    
      _extractStridedOutputChunkLarge(commandEncoder, { srcBuffer, srcOffsetBytes, output, outputOffsetBytes, batchStart, batchCount }) {
        if (outputOffsetBytes % 8 !== 0) {
          throw new Error(`outputOffsetBytes must be a multiple of 8 for complex-strided output; got ${outputOffsetBytes}`);
        }
        const extraOffsetElements = (outputOffsetBytes / 8) | 0;
        const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements, batchStart, batchCount);
        const outputBytes = getBufferByteLength(output);
        if (outputBytes < neededBytes) {
          throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
        }
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        const zeroBuf = this._ensureZeroComplexBuffer();
        const viewShape = this.io.output?.shape ?? null;
        const viewOffset = this.io.output?.offset ?? null;
        const viewTotal = this._outViewTotal;
        for (let lb = 0; lb < batchCount; lb++) {
          const gb = batchStart + lb;
          const srcBase = srcOffsetBytes + lb * this.logicalTotal * 8;
          const dstBatchBase = this._outputOffsetElements + extraOffsetElements + gb * this._outputBatchStrideElements;
          if (this._needsOutputMapping && this.io.output.clearOutside) {
            for (let vi = 0; vi < viewTotal; vi++) {
              this._coordsFromLinear(vi, viewShape, viewCoords);
              let dstElem = dstBatchBase;
              for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
              this._copyComplexToAny(commandEncoder, zeroBuf, 0, output, dstElem * 8);
            }
          }
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            if (!this._needsOutputMapping) {
              let dstElem = dstBatchBase;
              for (let d = 0; d < this.rank; d++) dstElem += logicalCoords[d] * this._outputStrides[d];
              this._copyComplexToAny(commandEncoder, srcBuffer, srcBase + li * 8, output, dstElem * 8);
              continue;
            }
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) continue;
            let dstElem = dstBatchBase;
            for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
            this._copyComplexToAny(commandEncoder, srcBuffer, srcBase + li * 8, output, dstElem * 8);
          }
        }
      }
    
      _runZeroStageLargeChunk(commandEncoder, stage, { buffer, offsetBytes, chunkBytes, chunkComplex }) {
        if (!stage) return;
        const bg = this.device.createBindGroup({
          layout: stage.bgl,
          entries: [{ binding: 0, resource: { buffer, offset: offsetBytes, size: chunkBytes } }],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(stage.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(chunkComplex / this.workgroupSize), 1, 1);
        pass.end();
      }
    
      _resolveLargeChunkBatchCount(limitByBindings) {
        let maxBatchPerChunk = Math.max(1, Math.floor(limitByBindings));
        if (this.tuning.largeChunkMaxBatches != null) {
          maxBatchPerChunk = Math.min(maxBatchPerChunk, this.tuning.largeChunkMaxBatches);
        }
        return Math.max(1, maxBatchPerChunk);
      }
    
      _execLargeBatchSegmentedStaging(commandEncoder, { input, output, inputOffsetBytes, outputOffsetBytes }) {
        const outTarget = this.inPlace ? input : output;
        const outOffset = this.inPlace ? inputOffsetBytes : outputOffsetBytes;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxBatchPerChunk = this._resolveLargeChunkBatchCount(this._maxBindBytes / this._bytesPerBatch);
        const maxChunkBytes = maxBatchPerChunk * this._bytesPerBatch;
        const axisWorkOffset = alignBytes(maxChunkBytes, storageAlign);
        const chunkTotalBytes = axisWorkOffset + this.maxAxisWorkBytes;
        const chunkBuf = this._ensureLargeChunkBuffer(chunkTotalBytes);
        for (let b0 = 0; b0 < this.batch; b0 += maxBatchPerChunk) {
          const bCount = Math.min(maxBatchPerChunk, this.batch - b0);
          const chunkBytes = bCount * this._bytesPerBatch;
          const chunkComplex = bCount * this.logicalTotal;
          if (this._usesStridedInput) {
            this._embedStridedInputChunkLarge(commandEncoder, {
              input,
              inputOffsetBytes,
              batchStart: b0,
              batchCount: bCount,
              dstBuffer: chunkBuf,
              dstOffsetBytes: 0,
            });
          } else {
            this._embedInputChunkLarge(commandEncoder, {
              input,
              inputOffsetBytes,
              batchStart: b0,
              batchCount: bCount,
              dstBuffer: chunkBuf,
              dstOffsetBytes: 0,
            });
          }
    
          this._runZeroStageLargeChunk(commandEncoder, this.zeroRead, {
            buffer: chunkBuf,
            offsetBytes: 0,
            chunkBytes,
            chunkComplex,
          });
    
          for (let axis = 0; axis < this.rank; axis++) {
            const kind = this.axisKind[axis];
            if (kind === "mixed") {
              this.axisPlans[axis].exec(commandEncoder, { input: chunkBuf, inputOffsetBytes: 0, batch: bCount, temp: null });
              continue;
            }
            const axisPlan = this.axisAdvanced[axis];
            if (!axisPlan) throw new Error(`Internal error: missing advanced axis plan for axis=${axis}`);
            const axisLines = bCount * (this.logicalTotal / this.shape[axis]);
            const axisWorkView = viewFromArena(chunkBuf, alignBytes(chunkBytes, storageAlign), axisPlan.workBytes);
            axisPlan.exec(commandEncoder, {
              dataBuf: chunkBuf,
              dataOffsetBytes: 0,
              axisWork: axisWorkView,
              scratch: null,
              lineCount: axisLines,
              paramChunkBase: 0,
            });
          }
    
          this._runZeroStageLargeChunk(commandEncoder, this.zeroWrite, {
            buffer: chunkBuf,
            offsetBytes: 0,
            chunkBytes,
            chunkComplex,
          });
    
          if (this._usesStridedOutput) {
            this._extractStridedOutputChunkLarge(commandEncoder, {
              srcBuffer: chunkBuf,
              srcOffsetBytes: 0,
              output: outTarget,
              outputOffsetBytes: outOffset,
              batchStart: b0,
              batchCount: bCount,
            });
          } else {
            this._extractOutputChunkLarge(commandEncoder, {
              srcBuffer: chunkBuf,
              srcOffsetBytes: 0,
              output: outTarget,
              outputOffsetBytes: outOffset,
              batchStart: b0,
              batchCount: bCount,
            });
          }
        }
      }
    
      _ensureOutOfCoreSegmentedFullVolumeState() {
        if (this._outOfCoreSegmentedFullVolumeState) return this._outOfCoreSegmentedFullVolumeState;
        if (!this._outOfCoreSegmentedFullVolumeMode) {
          throw new Error("Internal error: segmented full-volume out-of-core mode is not enabled");
        }
    
        const N0 = this.shape[0];
        const N1 = this.shape[1];
        const N2 = this.shape[2];
        if (!(N0 === N1 && N1 === N2)) {
          throw new Error(
            `Segmented full-volume mode currently expects cubic rank-3 shape; got ${JSON.stringify(this.shape)}`
          );
        }
        const N = N0;
        const rowBytes = N * 8;
        const planeBytes = N * N * 8;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        const maxBindBytes = this._maxBindBytes;
        if (!Number.isFinite(maxBindBytes) || maxBindBytes < rowBytes) {
          throw new Error(
            `Segmented full-volume mode requires maxStorageBufferBindingSize >= one row (${rowBytes} bytes); got ${maxBindBytes}`
          );
        }
    
        let preferredSegmentBytes = Math.min(this.mainBytes, maxBufferSize);
        preferredSegmentBytes = alignDownBytes(preferredSegmentBytes, planeBytes);
        preferredSegmentBytes = alignDownBytes(preferredSegmentBytes, storageAlign);
        if (!Number.isInteger(preferredSegmentBytes) || preferredSegmentBytes < planeBytes) {
          preferredSegmentBytes = alignBytes(planeBytes, storageAlign);
        }
        const segmented = this._createInternalSegmentedView(this.mainBytes, preferredSegmentBytes);
        const dataView = segmented.view;
        const segmentBytes = segmented.segmentBytes;
    
        const ringDepth = Math.max(1, Math.min(3, this.tuning.outOfCoreBurstWindows ?? 1));
        const axis0Policy = this._outOfCoreAxisWindowPolicy?.[0] ?? resolveOutOfCoreAxisWindowPolicy({
          axisLen: N,
          lineBytes: rowBytes,
          linesTotal: this.batch * (this.logicalTotal / N),
          maxBindBytes: this._maxBindBytes,
          axisKind: "mixed",
          tuning: this.tuning,
          axisIndex: 0,
          storageAlign,
        });
        const maxAxis0LinesByBind = Math.max(1, Math.floor(maxBindBytes / rowBytes));
        const maxAxis0LinesBySeg = Math.max(1, Math.floor(segmentBytes / rowBytes));
        let axis0LinesPerChunk = Math.max(1, Math.min(axis0Policy.linesPerChunk, maxAxis0LinesByBind, maxAxis0LinesBySeg));
        const axis0ChunkBytes = axis0LinesPerChunk * rowBytes;
    
        const axis0Ring = Array.from({ length: ringDepth }, () =>
          this.device.createBuffer({
            size: axis0ChunkBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          })
        );
    
        const planRows = createFftPlan(this.device, {
          shape: [N],
          direction: this.direction,
          normalize: "none",
          inPlace: true,
          layout: "interleaved",
          precision: "f32",
          batch: N,
          maxStorageBufferBindingSize: this._maxBindBytes,
        });
        const axis0PlanCache = new Map();
        const getAxis0Plan = (lines) => {
          const key = String(lines);
          if (axis0PlanCache.has(key)) return axis0PlanCache.get(key);
          const p = createFftPlan(this.device, {
            shape: [N],
            direction: this.direction,
            normalize: "none",
            inPlace: true,
            layout: "interleaved",
            precision: "f32",
            batch: lines,
            maxStorageBufferBindingSize: this._maxBindBytes,
          });
          axis0PlanCache.set(key, p);
          return p;
        };
    
        const transposePipeline = this._getOutOfCoreTransposePipeline(N, N);
        this.device.queue.writeBuffer(this._outOfCoreTranspose.params, 0, new Uint32Array([1, 0, 0, 0]));
        const transposeDispatch = Math.ceil(N / this._outOfCoreTranspose.tile);
        const slabs = [];
        for (let i = 0; i < ringDepth; i++) {
          const a = this.device.createBuffer({
            size: planeBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          const b = this.device.createBuffer({
            size: planeBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          const bgAB = this.device.createBindGroup({
            layout: this._outOfCoreTranspose.bgl,
            entries: [
              { binding: 0, resource: { buffer: a, offset: 0, size: planeBytes } },
              { binding: 1, resource: { buffer: b, offset: 0, size: planeBytes } },
              { binding: 2, resource: { buffer: this._outOfCoreTranspose.params, offset: 0, size: 16 } },
            ],
          });
          const bgBA = this.device.createBindGroup({
            layout: this._outOfCoreTranspose.bgl,
            entries: [
              { binding: 0, resource: { buffer: b, offset: 0, size: planeBytes } },
              { binding: 1, resource: { buffer: a, offset: 0, size: planeBytes } },
              { binding: 2, resource: { buffer: this._outOfCoreTranspose.params, offset: 0, size: 16 } },
            ],
          });
          slabs.push({ a, b, bgAB, bgBA });
        }
    
        const slabBgl = this.device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const slabPl = this.device.createPipelineLayout({ bindGroupLayouts: [slabBgl] });
        const gatherCode = generateGatherAxis2SlabWGSL({ N, workgroupSize: this.workgroupSize });
        const scatterCode = generateScatterAxis2SlabWGSL({ N, workgroupSize: this.workgroupSize });
        const slabGather = this.cache.getComputePipeline({ code: gatherCode, layout: slabPl });
        const slabScatter = this.cache.getComputePipeline({ code: scatterCode, layout: slabPl });
        const slabParams = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const maxZChunkByBind = Math.max(1, Math.floor((maxBindBytes - rowBytes) / planeBytes) + 1);
        const maxZChunkSpanBytes = (maxZChunkByBind - 1) * planeBytes + rowBytes;
    
        const state = {
          N,
          rowBytes,
          planeBytes,
          dataView,
          segmentBytes,
          ringDepth,
          axis0LinesPerChunk,
          axis0ChunkBytes,
          axis0Ring,
          planRows,
          axis0PlanCache,
          getAxis0Plan,
          slabs,
          transposePipeline,
          transposeDispatch,
          slabKernel: {
            bgl: slabBgl,
            pl: slabPl,
            gather: slabGather,
            scatter: slabScatter,
            params: slabParams,
            maxZChunkByBind,
            maxZChunkSpanBytes,
          },
        };
        this._outOfCoreSegmentedFullVolumeState = state;
        this._segmentedFullVolumeMeta = {
          mode: "rank3-segmented-slab",
          segmentBytes,
          segmentCount: dataView.segments.length,
          ringDepth,
          axis0LinesPerChunk,
          axis0ChunkBytes,
          axis0ChunkUtilization: Number.isFinite(maxBindBytes) ? axis0ChunkBytes / maxBindBytes : null,
          axis2MaxZChunk: maxZChunkByBind,
          axis2ChunkSpanBytes: maxZChunkSpanBytes,
          axis2ChunkUtilization: Number.isFinite(maxBindBytes) ? maxZChunkSpanBytes / maxBindBytes : null,
          maxStorageBufferBindingSize: maxBindBytes,
          maxBufferSize,
        };
        return state;
      }
    
      _resolveAxis2RowChunk(dataView, logicalStartBytes, { planeBytes, rowBytes, zRemain, maxZChunkByBind }) {
        if (zRemain <= 0) return null;
        const first = normalizeToContiguousRanges(dataView, logicalStartBytes, rowBytes)[0];
        let segRemain = 0;
        for (const seg of dataView.segments) {
          if (seg.buffer !== first.buffer) continue;
          const segStart = seg.offsetBytes;
          const segEnd = seg.offsetBytes + seg.sizeBytes;
          if (first.offsetBytes < segStart || first.offsetBytes >= segEnd) continue;
          segRemain = segEnd - first.offsetBytes;
          break;
        }
        if (segRemain < rowBytes) {
          throw new Error("Segmented axis2 slab chunking failed to locate source segment capacity");
        }
        const maxZBySeg = Math.max(1, Math.floor((segRemain - rowBytes) / planeBytes) + 1);
        let zCount = Math.max(1, Math.min(zRemain, maxZChunkByBind, maxZBySeg));
        while (zCount >= 1) {
          const spanBytes = (zCount - 1) * planeBytes + rowBytes;
          const ranges = normalizeToContiguousRanges(dataView, logicalStartBytes, spanBytes);
          if (ranges.length === 1) {
            return { range: ranges[0], zCount, spanBytes };
          }
          zCount -= 1;
        }
        throw new Error("Segmented axis2 slab chunking could not resolve a single bindable source window");
      }
    
      _runTransposeSlab(commandEncoder, state, slab, toBA) {
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(state.transposePipeline);
        pass.setBindGroup(0, toBA ? slab.bgBA : slab.bgAB);
        pass.dispatchWorkgroups(state.transposeDispatch, state.transposeDispatch, 1);
        pass.end();
      }
    
      _gatherAxis2RowSlab(commandEncoder, state, slab, yIndex) {
        let z0 = 0;
        while (z0 < state.N) {
          const logicalStartBytes = (yIndex * state.N + z0 * state.N * state.N) * 8;
          const chunk = this._resolveAxis2RowChunk(state.dataView, logicalStartBytes, {
            planeBytes: state.planeBytes,
            rowBytes: state.rowBytes,
            zRemain: state.N - z0,
            maxZChunkByBind: state.slabKernel.maxZChunkByBind,
          });
          const count = chunk.zCount * state.N;
          this.device.queue.writeBuffer(state.slabKernel.params, 0, new Uint32Array([count, z0, chunk.zCount, 0]));
          const bg = this.device.createBindGroup({
            layout: state.slabKernel.bgl,
            entries: [
              { binding: 0, resource: { buffer: chunk.range.buffer, offset: chunk.range.offsetBytes, size: chunk.spanBytes } },
              { binding: 1, resource: { buffer: slab.a, offset: 0, size: state.planeBytes } },
              { binding: 2, resource: { buffer: state.slabKernel.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(state.slabKernel.gather);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
          pass.end();
          z0 += chunk.zCount;
        }
      }
    
      _scatterAxis2RowSlab(commandEncoder, state, slab, yIndex) {
        let z0 = 0;
        while (z0 < state.N) {
          const logicalStartBytes = (yIndex * state.N + z0 * state.N * state.N) * 8;
          const chunk = this._resolveAxis2RowChunk(state.dataView, logicalStartBytes, {
            planeBytes: state.planeBytes,
            rowBytes: state.rowBytes,
            zRemain: state.N - z0,
            maxZChunkByBind: state.slabKernel.maxZChunkByBind,
          });
          const count = chunk.zCount * state.N;
          this.device.queue.writeBuffer(state.slabKernel.params, 0, new Uint32Array([count, z0, chunk.zCount, 0]));
          const bg = this.device.createBindGroup({
            layout: state.slabKernel.bgl,
            entries: [
              { binding: 0, resource: { buffer: slab.a, offset: 0, size: state.planeBytes } },
              { binding: 1, resource: { buffer: chunk.range.buffer, offset: chunk.range.offsetBytes, size: chunk.spanBytes } },
              { binding: 2, resource: { buffer: state.slabKernel.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(state.slabKernel.scatter);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
          pass.end();
          z0 += chunk.zCount;
        }
      }
    
      _applyScaleLargeDataSegmented(commandEncoder, { dataView, totalComplex, scale }) {
        if (scale === 1.0) return;
        const maxChunkComplex = Math.max(1, Math.floor(this._maxBindBytes / 8));
        const maxChunkBytes = maxChunkComplex * 8;
        const chunkBuf = this._ensureLargeChunkBuffer(maxChunkBytes);
        const chunkCount = Math.ceil(totalComplex / maxChunkComplex);
        const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
        const paramStride = alignBytes(32, uniformAlign);
        const paramsBuf = this._ensureScaleChunkParamsBuffer(chunkCount * paramStride);
        let chunkIndex = 0;
        for (let i0 = 0; i0 < totalComplex; i0 += maxChunkComplex) {
          const n = Math.min(maxChunkComplex, totalComplex - i0);
          const bytes = n * 8;
          const srcOff = i0 * 8;
          this._copyAnyToAny(commandEncoder, {
            src: dataView,
            srcOffsetBytes: srcOff,
            dst: chunkBuf,
            dstOffsetBytes: 0,
            bytes,
          });
          const paramOff = chunkIndex * paramStride;
          this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([n, 0, 0, 0]));
          this.device.queue.writeBuffer(paramsBuf, paramOff + 16, new Float32Array([scale, 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.scale.bgl,
            entries: [
              { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: bytes } },
              { binding: 1, resource: { buffer: paramsBuf, offset: paramOff, size: 32 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.scale.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(n / this.workgroupSize), 1, 1);
          pass.end();
          this._copyAnyToAny(commandEncoder, {
            src: chunkBuf,
            srcOffsetBytes: 0,
            dst: dataView,
            dstOffsetBytes: srcOff,
            bytes,
          });
          chunkIndex += 1;
        }
      }
    
      _execOutOfCoreFourStepSegmentedRank3(commandEncoder, { input, output, inputOffsetBytes, outputOffsetBytes }) {
        if (inputOffsetBytes !== 0 || outputOffsetBytes !== 0) {
          throw new Error(
            `Segmented full-volume mode currently requires zero input/output offsets; got inputOffsetBytes=${inputOffsetBytes}, outputOffsetBytes=${outputOffsetBytes}`
          );
        }
        const state = this._ensureOutOfCoreSegmentedFullVolumeState();
        const outTarget = this.inPlace ? input : output;
        if (!outTarget) throw new Error("Segmented full-volume mode requires an output target");
    
        this._copyAnyToAny(commandEncoder, {
          src: input,
          srcOffsetBytes: 0,
          dst: state.dataView,
          dstOffsetBytes: 0,
          bytes: this.mainBytes,
        });
    
        const totalAxis0Lines = this.batch * (this.logicalTotal / state.N);
        let axis0Line = 0;
        while (axis0Line < totalAxis0Lines) {
          const burstItems = [];
          for (let i = 0; i < state.axis0Ring.length && axis0Line < totalAxis0Lines; i++) {
            const lines = Math.min(state.axis0LinesPerChunk, totalAxis0Lines - axis0Line);
            const bytes = lines * state.rowBytes;
            const slot = state.axis0Ring[i];
            this._copyAnyToAny(commandEncoder, {
              src: state.dataView,
              srcOffsetBytes: axis0Line * state.rowBytes,
              dst: slot,
              dstOffsetBytes: 0,
              bytes,
            });
            burstItems.push({
              slot,
              lines,
              bytes,
              axis0Line,
              axis0Plan: state.getAxis0Plan(lines),
            });
            axis0Line += lines;
          }
          for (const item of burstItems) {
            item.axis0Plan.exec(commandEncoder, { input: item.slot, inputOffsetBytes: 0, batch: item.lines });
          }
          for (const item of burstItems) {
            this._copyAnyToAny(commandEncoder, {
              src: item.slot,
              srcOffsetBytes: 0,
              dst: state.dataView,
              dstOffsetBytes: item.axis0Line * state.rowBytes,
              bytes: item.bytes,
            });
          }
        }
    
        for (let z0 = 0; z0 < state.N; z0 += state.ringDepth) {
          const burst = Math.min(state.ringDepth, state.N - z0);
          for (let i = 0; i < burst; i++) {
            const z = z0 + i;
            const slab = state.slabs[i];
            this._copyAnyToAny(commandEncoder, {
              src: state.dataView,
              srcOffsetBytes: z * state.planeBytes,
              dst: slab.a,
              dstOffsetBytes: 0,
              bytes: state.planeBytes,
            });
          }
          for (let i = 0; i < burst; i++) {
            const slab = state.slabs[i];
            this._runTransposeSlab(commandEncoder, state, slab, false);
            state.planRows.exec(commandEncoder, { input: slab.b, inputOffsetBytes: 0, batch: state.N });
            this._runTransposeSlab(commandEncoder, state, slab, true);
          }
          for (let i = 0; i < burst; i++) {
            const z = z0 + i;
            const slab = state.slabs[i];
            this._copyAnyToAny(commandEncoder, {
              src: slab.a,
              srcOffsetBytes: 0,
              dst: state.dataView,
              dstOffsetBytes: z * state.planeBytes,
              bytes: state.planeBytes,
            });
          }
        }
    
        for (let y0 = 0; y0 < state.N; y0 += state.ringDepth) {
          const burst = Math.min(state.ringDepth, state.N - y0);
          for (let i = 0; i < burst; i++) {
            const y = y0 + i;
            this._gatherAxis2RowSlab(commandEncoder, state, state.slabs[i], y);
          }
          for (let i = 0; i < burst; i++) {
            const slab = state.slabs[i];
            this._runTransposeSlab(commandEncoder, state, slab, false);
            state.planRows.exec(commandEncoder, { input: slab.b, inputOffsetBytes: 0, batch: state.N });
            this._runTransposeSlab(commandEncoder, state, slab, true);
          }
          for (let i = 0; i < burst; i++) {
            const y = y0 + i;
            this._scatterAxis2RowSlab(commandEncoder, state, state.slabs[i], y);
          }
        }
    
        const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
        this._applyScaleLargeDataSegmented(commandEncoder, {
          dataView: state.dataView,
          totalComplex: this.totalComplex,
          scale,
        });
    
        this._copyAnyToAny(commandEncoder, {
          src: state.dataView,
          srcOffsetBytes: 0,
          dst: outTarget,
          dstOffsetBytes: 0,
          bytes: this.mainBytes,
        });
      }
    
      _applyScaleLargeData(commandEncoder, { dataBuffer, dataOffsetBytes, totalComplex, scale }) {
        if (scale === 1.0) return;
        const maxChunkComplex = Math.max(1, Math.floor(this._maxBindBytes / 8));
        const maxChunkBytes = maxChunkComplex * 8;
        const chunkBuf = this._ensureLargeChunkBuffer(maxChunkBytes);
        const chunkCount = Math.ceil(totalComplex / maxChunkComplex);
        const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
        const paramStride = alignBytes(32, uniformAlign);
        const paramsBuf = this._ensureScaleChunkParamsBuffer(chunkCount * paramStride);
        let chunkIndex = 0;
        for (let i0 = 0; i0 < totalComplex; i0 += maxChunkComplex) {
          const n = Math.min(maxChunkComplex, totalComplex - i0);
          const bytes = n * 8;
          const srcOff = dataOffsetBytes + i0 * 8;
          commandEncoder.copyBufferToBuffer(dataBuffer, srcOff, chunkBuf, 0, bytes);
          const paramOff = chunkIndex * paramStride;
          this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([n, 0, 0, 0]));
          this.device.queue.writeBuffer(paramsBuf, paramOff + 16, new Float32Array([scale, 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.scale.bgl,
            entries: [
              { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: bytes } },
              { binding: 1, resource: { buffer: paramsBuf, offset: paramOff, size: 32 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.scale.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(n / this.workgroupSize), 1, 1);
          pass.end();
    
          commandEncoder.copyBufferToBuffer(chunkBuf, 0, dataBuffer, srcOff, bytes);
          chunkIndex += 1;
        }
      }
    
      _execOutOfCoreAdvancedAxisWindows(commandEncoder, { axisExecutor, permShape, dataRange, axisIndex = 0 }) {
        const axisLen = permShape[0];
        const linesTotal = this.batch * (this.logicalTotal / axisLen);
        const lineBytes = axisLen * 8;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const axisKind = this._outOfCoreAxisEffectiveKind?.[axisIndex] ?? this.axisKind?.[axisIndex] ?? "mixed";
        const schedule = resolveOutOfCoreAxisWindowPolicy({
          axisLen,
          lineBytes,
          linesTotal,
          maxBindBytes: this._maxBindBytes,
          axisKind,
          tuning: this.tuning,
          axisIndex,
          storageAlign,
        });
        this._outOfCoreAxisWindowPolicy[axisIndex] = schedule;
        let linesPerChunk = schedule.linesPerChunk;
        if (lineBytes <= this._maxBindBytes) {
          const maxLinesByBind = Math.floor(this._maxBindBytes / lineBytes);
          if (!Number.isInteger(maxLinesByBind) || maxLinesByBind < 1) {
            throw new Error(
              `Out-of-core advanced axis chunking failed to size bindable line windows: axisLen=${axisLen} lineBytes=${lineBytes} ` +
                `maxStorageBufferBindingSize=${this._maxBindBytes}.` +
                ` (routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)})`
            );
          }
          linesPerChunk = Math.max(1, Math.min(linesTotal, maxLinesByBind, linesPerChunk));
        } else if (!(axisExecutor?.supportsBoundedLineSlicing?.())) {
          throw new Error(
            `Out-of-core advanced axis requires one axis line to fit maxStorageBufferBindingSize ` +
              `or a bounded sliced-line executor; axisLen=${axisLen} lineBytes=${lineBytes} ` +
              `maxStorageBufferBindingSize=${this._maxBindBytes}.` +
              ` (routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)})`
          );
        }
        const maxChunkDataBytes = linesPerChunk * lineBytes;
        const axisWorkOffset = alignBytes(maxChunkDataBytes, storageAlign);
        const chunkBuf = this._ensureLargeChunkBuffer(axisWorkOffset + axisExecutor.workBytes);
        const axisWorkView = viewFromArena(chunkBuf, axisWorkOffset, axisExecutor.workBytes);
        let paramChunkBase = 0;
    
        for (let line0 = 0; line0 < linesTotal; line0 += linesPerChunk) {
          const lines = Math.min(linesPerChunk, linesTotal - line0);
          const bytes = lines * lineBytes;
          const srcOff = dataRange.offsetBytes + line0 * lineBytes;
          commandEncoder.copyBufferToBuffer(dataRange.buffer, srcOff, chunkBuf, 0, bytes);
          const usedChunks = axisExecutor.exec(commandEncoder, {
            dataBuf: chunkBuf,
            dataOffsetBytes: 0,
            axisWork: axisWorkView,
            scratch: null,
            lineCount: lines,
            paramChunkBase,
          });
          paramChunkBase += usedChunks;
          commandEncoder.copyBufferToBuffer(chunkBuf, 0, dataRange.buffer, srcOff, bytes);
        }
      }
    
      _execOutOfCoreFourStep(commandEncoder, { input, output, temp, inputOffsetBytes, outputOffsetBytes }) {
        if (this.precision !== "f32") {
          throw new Error('Out-of-core four-step mode currently supports precision:"f32" only');
        }
        const inBytes = this._needsInputMapping ? this._inPhysBytes : this.mainBytes;
        const outBytes = this._needsOutputMapping ? this._outPhysBytes : this.mainBytes;
    
        const outTarget = this.inPlace ? input : output;
        const outOffset = this.inPlace ? inputOffsetBytes : outputOffsetBytes;
    
        const inRanges = this._usesStridedInput ? null : normalizeToContiguousRanges(input, inputOffsetBytes, inBytes);
        const inRange = inRanges && inRanges.length === 1 ? inRanges[0] : null;
        const outRanges = this._usesStridedOutput ? null : normalizeToContiguousRanges(outTarget, outOffset, outBytes);
        const outRange = outRanges && outRanges.length === 1 ? outRanges[0] : null;
    
        let dataRange = null;
        if (!this._needsOutputMapping && !this._usesStridedOutput && outRange && outRange.sizeBytes >= this.mainBytes) {
          dataRange = outRange;
        } else if (!this._needsInputMapping && !this._usesStridedInput && inRange && inRange.sizeBytes >= this.mainBytes) {
          dataRange = inRange ?? { buffer: this._ensureLargeAuxBuffer(this.mainBytes), offsetBytes: 0, sizeBytes: this.mainBytes };
        } else {
          dataRange = { buffer: this._ensureLargeAuxBuffer(this.mainBytes), offsetBytes: 0, sizeBytes: this.mainBytes };
        }
    
        if (this._usesStridedInput) {
          this._embedStridedInputOutOfCore(commandEncoder, { input, inputOffsetBytes, dataRange });
        } else {
          this._embedInputOutOfCore(commandEncoder, { inputRanges: inRanges, dataRange });
        }
    
        let transRange = null;
        if (temp) {
          const tRanges = normalizeToContiguousRanges(temp, 0, this.mainBytes);
          if (tRanges.length === 1) {
            const candidate = tRanges[0];
            if (!this._rangesOverlap(candidate, dataRange)) {
              transRange = candidate;
            }
          }
        }
        if (!transRange && inRange && !this._rangesOverlap(inRange, dataRange) && inRange.sizeBytes >= this.mainBytes) {
          transRange = inRange;
        }
        if (!transRange && outRange && !this._rangesOverlap(outRange, dataRange) && outRange.sizeBytes >= this.mainBytes) {
          transRange = outRange;
        }
        if (!transRange) {
          const stage = this._ensureLargeStageBuffer(this.mainBytes);
          transRange = { buffer: stage, offsetBytes: 0, sizeBytes: this.mainBytes };
        }
        if (this._rangesOverlap(transRange, dataRange)) {
          const stage = this._ensureLargeStageBuffer(this.mainBytes);
          transRange = { buffer: stage, offsetBytes: 0, sizeBytes: this.mainBytes };
        }
        if (this._rangesOverlap(transRange, dataRange)) {
          throw new Error("Out-of-core four-step requires distinct data and transpose staging ranges");
        }
    
        if (this.zeroPad.read) {
          this._zeroLogicalOutsideRange(commandEncoder, {
            dataBuffer: dataRange.buffer,
            dataOffsetBytes: dataRange.offsetBytes,
            start: this.zeroPad.read.start,
            end: this.zeroPad.read.end,
          });
        }
    
        if (this.axisKind[0] === "mixed") {
          this._outOfCoreAxisPlans[0].exec(commandEncoder, {
            input: dataRange.buffer,
            inputOffsetBytes: dataRange.offsetBytes,
            batch: this.batch,
          });
        } else {
          this._execOutOfCoreAdvancedAxisWindows(commandEncoder, {
            axisExecutor: this._outOfCoreAxisPlans[0],
            permShape: this._outOfCoreAxisPermShapes[0],
            dataRange,
            axisIndex: 0,
          });
        }
    
        for (let axis = 1; axis < this.rank; axis++) {
          const axisPlan = this._outOfCoreAxisPlans[axis];
          if (!axisPlan) throw new Error(`Internal error: missing out-of-core axis plan for axis=${axis}`);
          const kind = this.axisKind[axis];
          // Keep rank-2 transpose stripes as an optimization; rank>2 uses generic axis permutation.
          if (axis === 1 && this.rank === 2) {
            const [Nx, Ny] = this.shape;
            this._transposeOutOfCore2dStripes(commandEncoder, {
              srcBuffer: dataRange.buffer,
              srcOffsetBytes: dataRange.offsetBytes,
              dstBuffer: transRange.buffer,
              dstOffsetBytes: transRange.offsetBytes,
              Nx,
              Ny,
              batch: this.batch,
            });
            if (kind === "mixed") {
              axisPlan.exec(commandEncoder, {
                input: transRange.buffer,
                inputOffsetBytes: transRange.offsetBytes,
                batch: this.batch,
              });
            } else {
              this._execOutOfCoreAdvancedAxisWindows(commandEncoder, {
                axisExecutor: axisPlan,
                permShape: this._outOfCoreAxisPermShapes[axis],
                dataRange: transRange,
                axisIndex: axis,
              });
            }
            this._transposeOutOfCore2dStripes(commandEncoder, {
              srcBuffer: transRange.buffer,
              srcOffsetBytes: transRange.offsetBytes,
              dstBuffer: dataRange.buffer,
              dstOffsetBytes: dataRange.offsetBytes,
              Nx: Ny,
              Ny: Nx,
              batch: this.batch,
            });
          } else {
            this._permuteAxisToFront(commandEncoder, { srcRange: dataRange, dstRange: transRange, axis });
            if (kind === "mixed") {
              axisPlan.exec(commandEncoder, {
                input: transRange.buffer,
                inputOffsetBytes: transRange.offsetBytes,
                batch: this.batch,
              });
            } else {
              this._execOutOfCoreAdvancedAxisWindows(commandEncoder, {
                axisExecutor: axisPlan,
                permShape: this._outOfCoreAxisPermShapes[axis],
                dataRange: transRange,
                axisIndex: axis,
              });
            }
            this._permuteAxisFromFront(commandEncoder, { srcRange: transRange, dstRange: dataRange, axis });
          }
        }
    
        const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
        this._applyScaleLargeData(commandEncoder, {
          dataBuffer: dataRange.buffer,
          dataOffsetBytes: dataRange.offsetBytes,
          totalComplex: this.totalComplex,
          scale,
        });
    
        if (this.zeroPad.write) {
          this._zeroLogicalOutsideRange(commandEncoder, {
            dataBuffer: dataRange.buffer,
            dataOffsetBytes: dataRange.offsetBytes,
            start: this.zeroPad.write.start,
            end: this.zeroPad.write.end,
          });
        }
    
        if (this._usesStridedOutput) {
          this._extractStridedOutputOutOfCore(commandEncoder, { dataRange, output: outTarget, outputOffsetBytes: outOffset });
        } else {
          this._extractOutputOutOfCore(commandEncoder, { dataRange, outputRanges: outRanges });
        }
      }
    
      destroy() {
        if (this._destroyed) return;
        for (const p of this.axisPlans) p?.destroy?.();
        for (const p of this._outOfCoreAxisPlans ?? []) {
          if (!p) continue;
          if (this.axisPlans.includes(p)) continue;
          if (this.axisAdvanced.includes(p)) continue;
          p.destroy?.();
        }
        for (const ax of this.axisAdvanced) ax?.destroy?.();
        this.axis0OnTransposed?.destroy?.();
        if (this._outOfCoreAxis0OnTransposed && !(this._outOfCoreAxisPlans ?? []).includes(this._outOfCoreAxis0OnTransposed)) {
          this._outOfCoreAxis0OnTransposed.destroy?.();
        }
        this._outOfCoreTranspose?.params?.destroy?.();
        this._outOfCoreAxis1TailPermute?.params?.destroy?.();
        this._outOfCoreRank3Axis2Permute?.params?.destroy?.();
        this._outOfCoreGenericPermute?.params?.destroy?.();
        this._outOfCoreAdjacentSwapTiled?.params?.destroy?.();
        if (this._outOfCoreSegmentedFullVolumeState) {
          const s = this._outOfCoreSegmentedFullVolumeState;
          for (const p of s.axis0PlanCache?.values?.() ?? []) p?.destroy?.();
          s.planRows?.destroy?.();
          for (const b of s.axis0Ring ?? []) b?.destroy?.();
          for (const slab of s.slabs ?? []) {
            slab?.a?.destroy?.();
            slab?.b?.destroy?.();
          }
          s.slabKernel?.params?.destroy?.();
          this._destroySegmentedView(s.dataView);
          this._outOfCoreSegmentedFullVolumeState = null;
        }
        this.scale.params.destroy();
        this.ioEmbed?.params?.destroy?.();
        this.ioExtract?.params?.destroy?.();
        this.f16?.params?.destroy?.();
        this.stridedIn?.params?.destroy?.();
        this.stridedOut?.params?.destroy?.();
        this.transpose?.params?.destroy?.();
        this._largeStageBuffer?.destroy?.();
        this._largeChunkBuffer?.destroy?.();
        this._largeAuxBuffer?.destroy?.();
        this._scaleChunkParamsBuffer?.destroy?.();
        for (const b of this._retiredLargeStageBuffers) b?.destroy?.();
        for (const b of this._retiredLargeChunkBuffers) b?.destroy?.();
        for (const b of this._retiredLargeAuxBuffers) b?.destroy?.();
        for (const b of this._retiredScaleChunkParamsBuffers) b?.destroy?.();
        this._zeroComplexBuffer?.destroy?.();
        this._splitWorkspace?.mainStage?.destroy?.();
        this._splitWorkspace?.scratch?.destroy?.();
        this._splitWorkspace?.axisWork?.destroy?.();
        this._splitWorkspace?.transpose?.destroy?.();
        this._arena?.destroy?.();
        super.destroy();
      }
    
      exec(commandEncoder, execOpts) {
        if (this._destroyed) throw new Error("plan destroyed");
        const { input, output, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
        if (!input) throw new Error("exec requires input");
        if (!this.inPlace && !output) throw new Error("exec requires output when inPlace=false");
        if (this.inPlace && output && output !== input) throw new Error("inPlace=true requires output omitted or equal to input");
        if (this._outOfCoreFourStepMode) {
          if (this._outOfCoreSegmentedFullVolumeMode) {
            this._execOutOfCoreFourStepSegmentedRank3(commandEncoder, {
              input,
              output,
              inputOffsetBytes,
              outputOffsetBytes,
            });
            return;
          }
          this._execOutOfCoreFourStep(commandEncoder, { input, output, temp, inputOffsetBytes, outputOffsetBytes });
          return;
        }
    
        // Determine compute buffer (vec2<f32> contiguous) and ensure input is in it.
        // Fast path: f32, no ioView, no staging needed, contiguous input/output.
        const canDirect =
          !this.needsMainStage &&
          this.precision === "f32" &&
          !this._usesStridedInput &&
          !this._usesStridedOutput &&
          isGpuBuffer(input) &&
          inputOffsetBytes === 0 &&
          (this.inPlace || (isGpuBuffer(output) && outputOffsetBytes === 0));
        const needsLargeStaging = this._largeBatchChunkMode && !canDirect;
        let largeStagingArena = null;
        if (needsLargeStaging) {
          if (temp) {
            const tempAliasesInput = buffersAlias(temp, input);
            const tempAliasesOutput = output ? buffersAlias(temp, output) : false;
            if (tempAliasesInput || tempAliasesOutput) {
              throw new Error("Large-batch chunk mode staging temp must not alias input/output buffers");
            }
            largeStagingArena = temp;
          } else {
            largeStagingArena = this._ensureLargeStageBuffer(this.mainBytes);
          }
        }
    
        let arena = needsLargeStaging ? this._arena : (temp ?? this._arena);
        if (temp && this._arena && !needsLargeStaging) {
          // Staged execution emits copyBufferToBuffer commands; using temp slices that alias input/output
          // (same underlying GPUBuffer, different offsets) is invalid on some WebGPU stacks.
          const tempAliasesInput = buffersAlias(temp, input);
          const tempAliasesOutput = output ? buffersAlias(temp, output) : false;
          if (tempAliasesInput || tempAliasesOutput) {
            arena = this._arena;
          }
        }
        if (temp && !this._arena && this._splitWorkspace && !needsLargeStaging) {
          const tempAliasesInput = buffersAlias(temp, input);
          const tempAliasesOutput = output ? buffersAlias(temp, output) : false;
          if (tempAliasesInput || tempAliasesOutput) {
            arena = null; // fall back to internal split workspace
          }
        }
        let workspaceViews = this._resolveWorkspaceViews(arena);
        let { mainStage, scratch, axisWork, transpose } = workspaceViews;
    
        let dataBuf = null;
        let dataOff = 0;
        if (canDirect) {
          dataBuf = this.inPlace ? input : output;
          dataOff = 0;
          if (!this.inPlace && input !== output) {
            commandEncoder.copyBufferToBuffer(input, 0, output, 0, this.mainBytes);
          }
        } else {
          let stagingArena = needsLargeStaging ? largeStagingArena : (mainStage ?? scratch);
          let stageRanges = normalizeToContiguousRanges(stagingArena, 0, this.mainBytes);
          if (stageRanges.length !== 1) {
            if (!needsLargeStaging && temp) {
              arena = this._arena ?? null;
              workspaceViews = this._resolveWorkspaceViews(arena);
              ({ mainStage, scratch, axisWork, transpose } = workspaceViews);
              stagingArena = mainStage ?? scratch;
              stageRanges = normalizeToContiguousRanges(stagingArena, 0, this.mainBytes);
            }
            if (stageRanges.length !== 1) {
              if (this._largeBatchChunkMode) {
                this._execLargeBatchSegmentedStaging(commandEncoder, { input, output, inputOffsetBytes, outputOffsetBytes });
                return;
              }
              throw new Error("staging workspace must expose a contiguous range covering batch*product(shape)*8 bytes");
            }
          }
          const stageRange = stageRanges[0];
          dataBuf = stageRange.buffer;
          dataOff = stageRange.offsetBytes;
    
          if (this._usesStridedInput) {
            if (this._needsInputMapping) {
              this._embedStridedInputOutOfCore(commandEncoder, {
                input,
                inputOffsetBytes,
                dataRange: { buffer: dataBuf, offsetBytes: dataOff, sizeBytes: this.mainBytes },
              });
            } else {
              if (this._largeBatchChunkMode || !isGpuBuffer(input)) {
                this._embedStridedInputOutOfCore(commandEncoder, {
                  input,
                  inputOffsetBytes,
                  dataRange: { buffer: dataBuf, offsetBytes: dataOff, sizeBytes: this.mainBytes },
                });
              } else {
                const extraOffsetElements = (inputOffsetBytes / 8) | 0;
                const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
                ensureWithinBindingLimit(this.device, neededBytes, "c2c strided input binding");
                if (input.size < neededBytes) {
                  throw new Error(`input buffer too small for strided layout: need ${neededBytes} bytes, have ${input.size}`);
                }
    
                this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([this.logicalTotal, this.batch, extraOffsetElements, 0]));
                const bg = this.device.createBindGroup({
                  layout: this.stridedIn.bgl,
                  entries: [
                    { binding: 0, resource: { buffer: input, offset: 0, size: neededBytes } },
                    { binding: 1, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                    { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
                  ],
                });
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(this.stridedIn.pipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
                pass.end();
              }
            }
          } else if (this.ioEmbed) {
            const viewTotal = prod(this.ioEmbed.viewShape);
            const viewBytesPerBatch = viewTotal * this._bytesPerComplexIO;
            if (!this._largeBatchChunkMode) {
              const viewBytes = viewBytesPerBatch * this.batch;
    
              // Use direct binding when contiguous; otherwise pack once into scratch.
              const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, viewBytes);
              let physBuf = inRanges[0].buffer;
              let physOff = inRanges[0].offsetBytes;
              if (inRanges.length > 1) {
                const phys = normalizeToContiguousRanges(scratch, 0, viewBytes)[0];
                this.copier.pack(commandEncoder, inRanges, phys.buffer, phys.offsetBytes);
                physBuf = phys.buffer;
                physOff = phys.offsetBytes;
              }
    
              this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, viewTotal, this.batch, 0]));
              const bg = this.device.createBindGroup({
                layout: this.ioEmbed.bgl,
                entries: [
                  { binding: 0, resource: { buffer: physBuf, offset: physOff, size: viewBytes } },
                  { binding: 1, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                  { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.ioEmbed.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
              pass.end();
            } else {
              const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
              const maxBatchPerChunk = this._resolveLargeChunkBatchCount(
                Math.min(
                  Math.floor(this._maxBindBytes / this._bytesPerBatch),
                  Math.floor(this._maxBindBytes / viewBytesPerBatch)
                )
              );
              const maxChunkMainBytes = maxBatchPerChunk * this._bytesPerBatch;
              const maxChunkViewBytes = maxBatchPerChunk * viewBytesPerBatch;
              const maxPairBytes = alignBytes(maxChunkViewBytes, storageAlign) + maxChunkMainBytes;
              const chunkBuf = this._ensureLargeChunkBuffer(maxPairBytes);
    
              for (let b0 = 0; b0 < this.batch; b0 += maxBatchPerChunk) {
                const bCount = Math.min(maxBatchPerChunk, this.batch - b0);
                const chunkMainBytes = bCount * this._bytesPerBatch;
                const chunkViewBytes = bCount * viewBytesPerBatch;
                const chunkComplex = bCount * this.logicalTotal;
                const chunkInputOffset = inputOffsetBytes + b0 * viewBytesPerBatch;
                const chunkDataOffset = dataOff + b0 * this._bytesPerBatch;
    
                const inRanges = normalizeToContiguousRanges(input, chunkInputOffset, chunkViewBytes);
                const canBindInputDirect = inRanges.length === 1 && inRanges[0].offsetBytes % storageAlign === 0;
                const canBindDataDirect = chunkDataOffset % storageAlign === 0;
    
                let srcBuf;
                let srcOff;
                if (canBindInputDirect) {
                  srcBuf = inRanges[0].buffer;
                  srcOff = inRanges[0].offsetBytes;
                } else {
                  this.copier.pack(commandEncoder, inRanges, chunkBuf, 0);
                  srcBuf = chunkBuf;
                  srcOff = 0;
                }
    
                let dstBuf;
                let dstOff;
                if (canBindDataDirect) {
                  dstBuf = dataBuf;
                  dstOff = chunkDataOffset;
                } else {
                  dstOff = canBindInputDirect ? 0 : alignBytes(chunkViewBytes, storageAlign);
                  dstBuf = chunkBuf;
                }
    
                this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, viewTotal, bCount, 0]));
                const bg = this.device.createBindGroup({
                  layout: this.ioEmbed.bgl,
                  entries: [
                    { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: chunkViewBytes } },
                    { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: chunkMainBytes } },
                    { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
                  ],
                });
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(this.ioEmbed.pipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(chunkComplex / this.workgroupSize), 1, 1);
                pass.end();
    
                if (!canBindDataDirect) {
                  commandEncoder.copyBufferToBuffer(chunkBuf, dstOff, dataBuf, chunkDataOffset, chunkMainBytes);
                }
              }
            }
          } else {
            // bring input into stageRange (segmented ok)
            if (this.precision === "f16-storage") {
              const total = this.totalComplex;
              this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([total, 0, 0, 0]));
              const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, total * 4); // vec2<f16> is 4 bytes
              let srcBuf = inRanges[0].buffer;
              let srcOff = inRanges[0].offsetBytes;
              if (inRanges.length > 1) {
                const packed = normalizeToContiguousRanges(scratch, 0, total * 4)[0];
                this.copier.pack(commandEncoder, inRanges, packed.buffer, packed.offsetBytes);
                srcBuf = packed.buffer;
                srcOff = packed.offsetBytes;
              }
              const bg = this.device.createBindGroup({
                layout: this.f16.bgl,
                entries: [
                  { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: total * 4 } },
                  { binding: 1, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                  { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.f16.toF32);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(total / this.workgroupSize), 1, 1);
              pass.end();
            } else {
              const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.mainBytes);
              if (inRanges.length === 1) {
                commandEncoder.copyBufferToBuffer(inRanges[0].buffer, inRanges[0].offsetBytes, dataBuf, dataOff, this.mainBytes);
              } else {
                this.copier.pack(commandEncoder, inRanges, dataBuf, dataOff);
              }
            }
          }
        }
    
        const runZeroStage = (stage) => {
          if (!stage) return;
          if (!this._largeBatchChunkMode) {
            const bg = this.device.createBindGroup({
              layout: stage.bgl,
              entries: [{ binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } }],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(stage.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
            pass.end();
            return;
          }
    
          const maxBatchPerChunk = this._resolveLargeChunkBatchCount(this._maxBindBytes / this._bytesPerBatch);
          const maxChunkBytes = maxBatchPerChunk * this._bytesPerBatch;
          const chunkBuf = this._ensureLargeChunkBuffer(maxChunkBytes);
    
          for (let b0 = 0; b0 < this.batch; b0 += maxBatchPerChunk) {
            const bCount = Math.min(maxBatchPerChunk, this.batch - b0);
            const chunkBytes = bCount * this._bytesPerBatch;
            const chunkComplex = bCount * this.logicalTotal;
            const chunkOffset = dataOff + b0 * this._bytesPerBatch;
    
            commandEncoder.copyBufferToBuffer(dataBuf, chunkOffset, chunkBuf, 0, chunkBytes);
    
            const bg = this.device.createBindGroup({
              layout: stage.bgl,
              entries: [{ binding: 0, resource: { buffer: chunkBuf, offset: 0, size: chunkBytes } }],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(stage.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(chunkComplex / this.workgroupSize), 1, 1);
            pass.end();
    
            commandEncoder.copyBufferToBuffer(chunkBuf, 0, dataBuf, chunkOffset, chunkBytes);
          }
        };
    
        runZeroStage(this.zeroRead);
    
        const axisTemp = this._largeBatchChunkMode ? null : scratch;
        // FFT axes in order, with optional axis-1 transpose fast path.
        for (let axis = 0; axis < this.rank; axis++) {
          if (axis === 1 && this.transpose) {
            const tr = this.transpose;
            const trBatch = tr.matrixBatch ?? this.batch;
            const trRange = normalizeToContiguousRanges(transpose, 0, this.mainBytes)[0];
            this.device.queue.writeBuffer(tr.params, 0, new Uint32Array([trBatch, 0, 0, 0]));
            const bg1 = this.device.createBindGroup({
              layout: tr.bgl,
              entries: [
                { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                { binding: 1, resource: { buffer: trRange.buffer, offset: trRange.offsetBytes, size: this.mainBytes } },
                { binding: 2, resource: { buffer: tr.params, offset: 0, size: 16 } },
              ],
            });
            const pass1 = commandEncoder.beginComputePass();
            pass1.setPipeline(tr.pipelineXY);
            pass1.setBindGroup(0, bg1);
            pass1.dispatchWorkgroups(Math.ceil(tr.Nx / tr.tile), Math.ceil(tr.Ny / tr.tile), trBatch);
            pass1.end();
    
            this.axis0OnTransposed.exec(commandEncoder, { input: trRange.buffer, inputOffsetBytes: trRange.offsetBytes, batch: this.batch, temp: axisTemp });
    
            const bg2 = this.device.createBindGroup({
              layout: tr.bgl,
              entries: [
                { binding: 0, resource: { buffer: trRange.buffer, offset: trRange.offsetBytes, size: this.mainBytes } },
                { binding: 1, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                { binding: 2, resource: { buffer: tr.params, offset: 0, size: 16 } },
              ],
            });
            const pass2 = commandEncoder.beginComputePass();
            pass2.setPipeline(tr.pipelineYX);
            pass2.setBindGroup(0, bg2);
            pass2.dispatchWorkgroups(Math.ceil(tr.Ny / tr.tile), Math.ceil(tr.Nx / tr.tile), trBatch);
            pass2.end();
            continue;
          }
    
          const kind = this.axisKind[axis];
          if (kind === "mixed") {
            this.axisPlans[axis].exec(commandEncoder, { input: dataBuf, inputOffsetBytes: dataOff, batch: this.batch, temp: axisTemp });
          } else if (kind === "bluestein") {
            this.axisAdvanced[axis].exec(commandEncoder, { dataBuf, dataOffsetBytes: dataOff, axisWork, scratch: axisTemp });
          } else {
            this.axisAdvanced[axis].exec(commandEncoder, { dataBuf, dataOffsetBytes: dataOff, axisWork, scratch: axisTemp });
          }
        }
    
        // Normalize scale once after full transform
        const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
        if (!this._largeBatchChunkMode && scale !== 1.0) {
          this.device.queue.writeBuffer(this.scale.params, 0, new Uint32Array([this.totalComplex, 0, 0, 0]));
          this.device.queue.writeBuffer(this.scale.params, 16, new Float32Array([scale, 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.scale.bgl,
            entries: [
              { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
              { binding: 1, resource: { buffer: this.scale.params, offset: 0, size: 32 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.scale.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        runZeroStage(this.zeroWrite);
    
        const outTarget = this.inPlace ? input : output;
        const outOffset = this.inPlace ? inputOffsetBytes : outputOffsetBytes;
    
        if (this._usesStridedOutput) {
          if (this._needsOutputMapping) {
            this._extractStridedOutputOutOfCore(commandEncoder, {
              dataRange: { buffer: dataBuf, offsetBytes: dataOff, sizeBytes: this.mainBytes },
              output: outTarget,
              outputOffsetBytes: outOffset,
            });
            return;
          }
          if (this._largeBatchChunkMode || !isGpuBuffer(outTarget)) {
            this._extractStridedOutputOutOfCore(commandEncoder, {
              dataRange: { buffer: dataBuf, offsetBytes: dataOff, sizeBytes: this.mainBytes },
              output: outTarget,
              outputOffsetBytes: outOffset,
            });
            return;
          }
          const extraOffsetElements = (outOffset / 8) | 0;
          const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
          ensureWithinBindingLimit(this.device, neededBytes, "c2c strided output binding");
          if (outTarget.size < neededBytes) {
            throw new Error(`output buffer too small for strided layout: need ${neededBytes} bytes, have ${outTarget.size}`);
          }
    
          this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([this.logicalTotal, this.batch, extraOffsetElements, 0]));
          const bg = this.device.createBindGroup({
            layout: this.stridedOut.bgl,
            entries: [
              { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
              { binding: 1, resource: { buffer: outTarget, offset: 0, size: neededBytes } },
              { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.stridedOut.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalComplex / this.workgroupSize), 1, 1);
          pass.end();
          return;
        }
    
        // Optional output view mapping. If present, write directly to the final output when contiguous;
        // otherwise, stage once and scatter.
        if (this.ioExtract) {
          const viewTotal = prod(this.ioExtract.viewShape);
          const outBytesPerBatch = viewTotal * this._bytesPerComplexIO;
          if (!this._largeBatchChunkMode) {
            const outBytes = outBytesPerBatch * this.batch;
            const outRanges = normalizeToContiguousRanges(outTarget, outOffset, outBytes);
    
            // Contiguous output: write directly to preserve clearOutside=false semantics.
            if (outRanges.length === 1) {
              this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, viewTotal, this.batch, 0]));
              const bg = this.device.createBindGroup({
                layout: this.ioExtract.bgl,
                entries: [
                  { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                  { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: outBytes } },
                  { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.ioExtract.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
              pass.end();
              return;
            }
    
            const stageOut = normalizeToContiguousRanges(scratch, 0, outBytes)[0];
            if (!this.io.output.clearOutside) {
              this.copier.pack(commandEncoder, outRanges, stageOut.buffer, stageOut.offsetBytes);
            }
    
            this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, viewTotal, this.batch, 0]));
            const bg = this.device.createBindGroup({
              layout: this.ioExtract.bgl,
              entries: [
                { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                { binding: 1, resource: { buffer: stageOut.buffer, offset: stageOut.offsetBytes, size: outBytes } },
                { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.ioExtract.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
            pass.end();
    
            this.copier.unpack(commandEncoder, stageOut.buffer, stageOut.offsetBytes, outRanges);
          } else {
            const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
            const maxBatchPerChunk = this._resolveLargeChunkBatchCount(
              Math.min(
                Math.floor(this._maxBindBytes / this._bytesPerBatch),
                Math.floor(this._maxBindBytes / outBytesPerBatch)
              )
            );
            const maxChunkMainBytes = maxBatchPerChunk * this._bytesPerBatch;
            const maxChunkViewBytes = maxBatchPerChunk * outBytesPerBatch;
            const maxPairBytes = alignBytes(maxChunkMainBytes, storageAlign) + maxChunkViewBytes;
            const chunkBuf = this._ensureLargeChunkBuffer(maxPairBytes);
    
            for (let b0 = 0; b0 < this.batch; b0 += maxBatchPerChunk) {
              const bCount = Math.min(maxBatchPerChunk, this.batch - b0);
              const chunkMainBytes = bCount * this._bytesPerBatch;
              const chunkViewBytes = bCount * outBytesPerBatch;
              const chunkViewComplex = bCount * viewTotal;
              const chunkDataOffset = dataOff + b0 * this._bytesPerBatch;
              const chunkOutOffset = outOffset + b0 * outBytesPerBatch;
              const outRanges = normalizeToContiguousRanges(outTarget, chunkOutOffset, chunkViewBytes);
    
              const canBindDataDirect = chunkDataOffset % storageAlign === 0;
              const canBindOutputDirect = outRanges.length === 1 && outRanges[0].offsetBytes % storageAlign === 0;
    
              let srcBuf;
              let srcOff;
              if (canBindDataDirect) {
                srcBuf = dataBuf;
                srcOff = chunkDataOffset;
              } else {
                commandEncoder.copyBufferToBuffer(dataBuf, chunkDataOffset, chunkBuf, 0, chunkMainBytes);
                srcBuf = chunkBuf;
                srcOff = 0;
              }
    
              let dstBuf;
              let dstOff;
              if (canBindOutputDirect) {
                dstBuf = outRanges[0].buffer;
                dstOff = outRanges[0].offsetBytes;
              } else {
                dstOff = canBindDataDirect ? 0 : alignBytes(chunkMainBytes, storageAlign);
                dstBuf = chunkBuf;
                if (!this.io.output.clearOutside) {
                  if (outRanges.length === 1) {
                    commandEncoder.copyBufferToBuffer(outRanges[0].buffer, outRanges[0].offsetBytes, chunkBuf, dstOff, chunkViewBytes);
                  } else {
                    this.copier.pack(commandEncoder, outRanges, chunkBuf, dstOff);
                  }
                }
              }
    
              this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, viewTotal, bCount, 0]));
              const bg = this.device.createBindGroup({
                layout: this.ioExtract.bgl,
                entries: [
                  { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: chunkMainBytes } },
                  { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: chunkViewBytes } },
                  { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.ioExtract.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(chunkViewComplex / this.workgroupSize), 1, 1);
              pass.end();
    
              if (!canBindOutputDirect) {
                if (outRanges.length === 1) {
                  commandEncoder.copyBufferToBuffer(chunkBuf, dstOff, outRanges[0].buffer, outRanges[0].offsetBytes, chunkViewBytes);
                } else {
                  this.copier.unpack(commandEncoder, chunkBuf, dstOff, outRanges);
                }
              }
            }
          }
          return;
        }
    
        // No ioView mapping: write logical output in requested precision.
        if (this.precision === "f16-storage") {
          const total = this.totalComplex;
          const outBytesF16 = total * 4;
          this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([total, 0, 0, 0]));
          const outRanges = normalizeToContiguousRanges(outTarget, outOffset, outBytesF16);
          if (outRanges.length === 1) {
            const bg = this.device.createBindGroup({
              layout: this.f16.bgl,
              entries: [
                { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
                { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: outBytesF16 } },
                { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.f16.toF16);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(total / this.workgroupSize), 1, 1);
            pass.end();
            return;
          }
          const packed = normalizeToContiguousRanges(scratch, 0, outBytesF16)[0];
          const bg = this.device.createBindGroup({
            layout: this.f16.bgl,
            entries: [
              { binding: 0, resource: { buffer: dataBuf, offset: dataOff, size: this.mainBytes } },
              { binding: 1, resource: { buffer: packed.buffer, offset: packed.offsetBytes, size: outBytesF16 } },
              { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16.toF16);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(total / this.workgroupSize), 1, 1);
          pass.end();
          this.copier.unpack(commandEncoder, packed.buffer, packed.offsetBytes, outRanges);
          return;
        }
    
        const outRanges = normalizeToContiguousRanges(outTarget, outOffset, this.mainBytes);
        if (outRanges.length === 1) {
          if (outRanges[0].buffer !== dataBuf || outRanges[0].offsetBytes !== dataOff) {
            commandEncoder.copyBufferToBuffer(dataBuf, dataOff, outRanges[0].buffer, outRanges[0].offsetBytes, this.mainBytes);
          }
        } else {
          this.copier.unpack(commandEncoder, dataBuf, dataOff, outRanges);
        }
      }
    }
    
    exports['C2CPlan'] = C2CPlan;
  });

  __define('src/runtime/plans/c2r.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { BasePlan } = require('src/runtime/base_plan.js');
    const { mergeLargeRouteMetadata, resolveAxisKindsForShape, resolveLargeRoutingPolicy, resolveOutOfCoreAxisWindowPolicy } = require('src/runtime/large_policy.js');
    const { createInternalArena, viewFromArena } = require('src/runtime/workspace.js');
    const { normalizeToContiguousRanges } = require('src/runtime/segmented_io.js');
    const { resolveLayoutSemantics } = require('src/runtime/layout_semantics.js');
    const { assertOneOf, isPositiveInt, prod, alignBytes, ensureWithinBindingLimit, getBufferByteLength, align4Bytes, isGpuBuffer, buffersAlias } = require('src/runtime/common.js');
    const { normalizeIoView } = require('src/runtime/ioview.js');
    const { normalizeZeroPad } = require('src/runtime/zero_pad.js');
    const { contiguousStrides: tensorContiguousStrides, coordsFromLinear: tensorCoordsFromLinear, linearFromCoords: tensorLinearFromCoords, createTensorDescriptor, requiredBytesForBatchRange } = require('src/runtime/tensor_descriptor.js');
    
    const { C2CPlan } = require('src/runtime/plans/c2c.js');
    const { generateUnpackC2RWGSL, generateComplexToRealWGSL } = require('src/kernels/real_complex.js');
    const { generateZeroOutsideRangeComplexWGSL, generateZeroOutsideRangeRealWGSL } = require('src/kernels/zero_pad.js');
    const { generateEmbedComplexWGSL, generateExtractRealWGSL } = require('src/kernels/ioview.js');
    const { generateF16ToF32ComplexWGSL, generateF16ToF32RealWGSL, generateF32ToF16RealWGSL } = require('src/kernels/f16_storage.js');
    const { generateGatherComplexStridedWGSL } = require('src/kernels/strided_complex.js');
    const { generateScatterRealStridedWGSL } = require('src/kernels/strided_real.js');
    
    function needsIoMapping(io, logicalShape) {
      if (!io) return false;
      for (let i = 0; i < logicalShape.length; i++) {
        if (io.shape[i] !== logicalShape[i]) return true;
        if (io.offset[i] !== 0) return true;
      }
      return false;
    }
    
    function generateFinalizeUnpackedHermitianWGSL({ shape, workgroupSize }) {
      const Nx = shape[0];
      const inNx = Math.floor(Nx / 2) + 1;
      const evenNx = Nx % 2 === 0;
      let remName = "line";
      let decode = "";
      for (let d = 1; d < shape.length; d++) {
        const c = `c${d}`;
        decode += `  let ${c}: u32 = ${remName} % ${shape[d]}u;\n`;
        if (d < shape.length - 1) {
          const rn = `rem${d}`;
          decode += `  let ${rn}: u32 = ${remName} / ${shape[d]}u;\n`;
          remName = rn;
        }
      }
    
      let selfExpr = "(x == 0u || (EVEN_NX && x == (NX / 2u)))";
      for (let d = 1; d < shape.length; d++) {
        if (shape[d] % 2 === 0) selfExpr += ` && (c${d} == 0u || c${d} == ${shape[d] / 2}u)`;
        else selfExpr += ` && (c${d} == 0u)`;
      }
    
      return /* wgsl */ `
    struct Params {
      lineOffset: u32,
      lineCount: u32,
      _pad0: u32,
      _pad1: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
    @group(0) @binding(1) var<uniform> params: Params;
    
    const NX: u32 = ${Nx}u;
    const IN_NX: u32 = ${inNx}u;
    const EVEN_NX: bool = ${evenNx ? "true" : "false"};
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      let total: u32 = params.lineCount * NX;
      if (i >= total) { return; }
      let lineLocal: u32 = i / NX;
      let x: u32 = i - lineLocal * NX;
      let line: u32 = params.lineOffset + lineLocal;
    ${decode}
      var v: vec2<f32> = data[i];
      if (x >= IN_NX) {
        v = vec2<f32>(v.x, -v.y);
      }
      if (${selfExpr}) {
        v = vec2<f32>(v.x, 0.0);
      }
      data[i] = v;
    }
    `;
    }
    
    function generateFinalizeUnpackedHermitianSegmentWGSL({ shape, workgroupSize }) {
      const Nx = shape[0];
      const inNx = Math.floor(Nx / 2) + 1;
      const evenNx = Nx % 2 === 0;
      return /* wgsl */ `
    struct Params {
      count: u32,
      xOffset: u32,
      lineSelfConj: u32,
      _pad0: u32,
    }
    
    @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
    @group(0) @binding(1) var<uniform> params: Params;
    
    const NX: u32 = ${Nx}u;
    const IN_NX: u32 = ${inNx}u;
    const EVEN_NX: bool = ${evenNx ? "true" : "false"};
    
    @compute @workgroup_size(${workgroupSize}, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= params.count) { return; }
      let x: u32 = params.xOffset + i;
      var v: vec2<f32> = data[i];
      if (x >= IN_NX) {
        v = vec2<f32>(v.x, -v.y);
      }
      if (params.lineSelfConj != 0u && (x == 0u || (EVEN_NX && x == (NX / 2u)))) {
        v = vec2<f32>(v.x, 0.0);
      }
      data[i] = v;
    }
    `;
    }
    
    class C2RPlan extends BasePlan {
      constructor(device, opts) {
        super(device, opts);
        const { shape, direction, batch = 1, inPlace = false, normalize = "none", layout = { interleavedComplex: true }, precision = "f32", ioView = null, zeroPad = null } = opts ?? {};
        if (inPlace) throw new Error("c2r inPlace is not supported in current implementation");
        if (direction !== "inverse") throw new Error('c2r supports direction:"inverse" only');
        if (!Array.isArray(shape) || shape.length < 1) throw new Error("shape must be rank >= 1");
        if (!shape.every(isPositiveInt)) throw new Error("shape must be positive ints");
        assertOneOf(normalize, ["none", "backward", "unitary"], "normalize");
        assertOneOf(precision, ["f32", "f16-storage"], "precision");
        if (layout?.interleavedComplex !== true) throw new Error("c2r input is packed complex interleaved; set layout.interleavedComplex=true");
        if (precision === "f16-storage" && !device.features?.has?.("shader-f16")) throw new Error('precision="f16-storage" requires shader-f16');
    
        this.shape = shape.slice();
        this.rank = shape.length;
        this.batch = batch;
        this.normalize = normalize;
        this.precision = precision;
        const Nx = this.shape[0];
        this.packedShape = [Math.floor(Nx / 2) + 1, ...this.shape.slice(1)];
        this.logicalTotal = prod(this.shape);
        this.totalReal = this.logicalTotal * this.batch;
    
        const iov = ioView ?? {};
    
        this.ioIn = normalizeIoView(this.rank, this.packedShape, { input: iov.input }).input;
        this.ioOut = normalizeIoView(this.rank, this.shape, { output: iov.output }).output;
        this._needsInputMapping = !!(this.ioIn && needsIoMapping(this.ioIn, this.packedShape));
        this._needsOutputMapping = !!(this.ioOut && needsIoMapping(this.ioOut, this.shape));
        this._inputLayoutShape = this._needsInputMapping ? this.ioIn.shape.slice() : this.packedShape.slice();
        this._outputLayoutShape = this._needsOutputMapping ? this.ioOut.shape.slice() : this.shape.slice();
    
        const resolvedLayout = resolveLayoutSemantics({
          layout,
          rank: this.rank,
          inputShape: this._inputLayoutShape,
          outputShape: this._outputLayoutShape,
        });
        this._inputStrides = resolvedLayout.inputStrides;
        this._outputStrides = resolvedLayout.outputStrides;
        this._inputOffsetElements = resolvedLayout.inputOffsetElements;
        this._outputOffsetElements = resolvedLayout.outputOffsetElements;
        this._inputBatchStrideElements = resolvedLayout.inputBatchStrideElements;
        this._outputBatchStrideElements = resolvedLayout.outputBatchStrideElements;
        this._inputSpanElements = resolvedLayout.inputSpanElements;
        this._outputSpanElements = resolvedLayout.outputSpanElements;
        this._usesStridedInput = resolvedLayout.usesStridedInput;
        this._usesStridedOutput = resolvedLayout.usesStridedOutput;
        this._usesWhdcnInput = resolvedLayout.usesWhdcnInput;
        this._usesWhdcnOutput = resolvedLayout.usesWhdcnOutput;
        this._inputTensorDesc = this._usesStridedInput
          ? createTensorDescriptor({
              name: "c2r.input",
              shape: this._inputLayoutShape,
              strides: this._inputStrides,
              offsetElements: this._inputOffsetElements,
              batchStrideElements: this._inputBatchStrideElements,
            })
          : null;
        this._outputTensorDesc = this._usesStridedOutput
          ? createTensorDescriptor({
              name: "c2r.output",
              shape: this._outputLayoutShape,
              strides: this._outputStrides,
              offsetElements: this._outputOffsetElements,
              batchStrideElements: this._outputBatchStrideElements,
            })
          : null;
        this._inputSpanElements = this._inputTensorDesc?.spanElements ?? 0;
        this._outputSpanElements = this._outputTensorDesc?.spanElements ?? 0;
        if ((this._usesStridedInput || this._usesStridedOutput) && this.precision !== "f32") {
          throw new Error('custom strides currently support precision:"f32" only for c2r');
        }
    
        this.zeroPadRead = normalizeZeroPad(this.rank, this.packedShape, { read: zeroPad?.read ?? null }, "zeroPad").read;
        this.zeroPadWrite = normalizeZeroPad(this.rank, this.shape, { write: zeroPad?.write ?? null }, "zeroPad").write;
        this.inTotalComplexLogical = prod(this.packedShape) * this.batch;
        this.packedF32Bytes = this.inTotalComplexLogical * 8;
    
        this.inViewShape = (this.ioIn?.shape ?? this.packedShape).slice();
        this.inViewTotalComplex = prod(this.inViewShape) * this.batch;
        this.inBytes = this.inViewTotalComplex * (precision === "f16-storage" ? 4 : 8);
    
        this.outViewShape = (this.ioOut?.shape ?? this.shape).slice();
        this.outViewTotalReal = prod(this.outViewShape) * this.batch;
        this.outBytes = precision === "f16-storage" ? align4Bytes(this.outViewTotalReal * 2) : this.outViewTotalReal * 4;
    
        this.totalComplexFull = this.totalReal;
        this.fullBytes = this.totalComplexFull * 8;
        this._lineCount = this.batch * prod(this.shape.slice(1));
        this._realLineBytes = this.shape[0] * 4;
        this._complexLineBytes = this.shape[0] * 8;
        this._packedLineBytes = this.packedShape[0] * 8;
        const axisStrategy = resolveAxisKindsForShape({
          shape: this.shape,
          tuning: opts?.tuning ?? null,
        });
        const largePolicy = resolveLargeRoutingPolicy({
          device,
          tuning: opts?.tuning ?? null,
          requiredBindingBytes: [this.fullBytes, this.packedF32Bytes, this.inBytes, this.outBytes],
          lineBytes: [this._realLineBytes, this._complexLineBytes, this._packedLineBytes],
          axisKinds: axisStrategy.axisKinds,
          axisLengths: this.shape,
          allowNonMixedBoundedSlicing: true,
          allowOutOfCore: this.rank >= 2,
          rank: this.rank,
          bytesPerBatch: this.logicalTotal * 8,
          hasStridedIO: this._usesStridedInput || this._usesStridedOutput,
          preferOutOfCoreForStrided: true,
          precision: this.precision,
          requireLargePrecision: "f32",
          requireLargePrecisionError: 'c2r large-shape fallback currently supports precision:"f32" only',
        });
        this._maxBindBytes = largePolicy.maxBindBytes;
        this._largeShapeMode = largePolicy.needsLargeMode;
        this._largeRouteMode = largePolicy.routeMode;
        this._largeRouteReasons = largePolicy.reasonCodes;
        this._largeRouteAttempts = largePolicy.attemptedRoutes;
        this._largeRouteAxisKinds = axisStrategy.axisKinds.slice();
        this._largeRouteAxisSupported = Array.isArray(largePolicy.axisSupported) ? largePolicy.axisSupported.slice() : null;
        if (!this._largeShapeMode) {
          ensureWithinBindingLimit(device, this.fullBytes, "c2r full complex");
          ensureWithinBindingLimit(device, this.packedF32Bytes, "c2r packed logical (f32)");
          ensureWithinBindingLimit(device, this.inBytes, "c2r input");
          ensureWithinBindingLimit(device, this.outBytes, "c2r output");
        }
        this._oversizedLineMode = this._largeShapeMode && largePolicy.oversizedLineMode;
        this._outOfCoreAxisWindowPolicy = null;
        if (this._largeShapeMode) {
          const axisKind0 = this._largeRouteAxisKinds?.[0] ?? "mixed";
          const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
          const tuning = opts?.tuning ?? null;
          this._outOfCoreAxisWindowPolicy = {
            unpack: resolveOutOfCoreAxisWindowPolicy({
              axisLen: this.shape[0],
              lineBytes: Math.max(this._complexLineBytes, this._packedLineBytes),
              linesTotal: this._lineCount,
              maxBindBytes: this._maxBindBytes,
              axisKind: axisKind0,
              tuning,
              axisIndex: 0,
              storageAlign,
            }),
            complexToReal: resolveOutOfCoreAxisWindowPolicy({
              axisLen: this.shape[0],
              lineBytes: Math.max(this._complexLineBytes, this._realLineBytes),
              linesTotal: this._lineCount,
              maxBindBytes: this._maxBindBytes,
              axisKind: axisKind0,
              tuning,
              axisIndex: 0,
              storageAlign,
            }),
          };
        }
    
        this.c2c = new C2CPlan(device, {
          shape: this.shape,
          direction: "inverse",
          batch,
          inPlace: true,
          normalize,
          layout: { interleavedComplex: true },
          precision: "f32",
          ioView: { input: null, output: null },
          tuning: opts?.tuning ?? null,
        });
        const mergedRoute = mergeLargeRouteMetadata([
          {
            routeMode: this._largeRouteMode,
            reasonCodes: this._largeRouteReasons,
            attemptedRoutes: this._largeRouteAttempts,
          },
          {
            routeMode: this.c2c?._largeRouteMode,
            reasonCodes: this.c2c?._largeRouteReasons,
            attemptedRoutes: this.c2c?._largeRouteAttempts,
          },
        ]);
        this._largeRouteMode = mergedRoute.routeMode;
        this._largeRouteReasons = mergedRoute.reasonCodes;
        this._largeRouteAttempts = mergedRoute.attemptedRoutes;
    
        // unpack packed spectrum to full complex
        this.unpack = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateUnpackC2RWGSL({ shape: this.shape, workgroupSize: this.workgroupSize });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(params, 0, new Uint32Array([batch, 0, 0, 0]));
          return { bgl, pl: pipelineLayout, pipeline, params };
        })();
        this.unpackLine = null;
        if (this._largeShapeMode) {
          this.unpackLine = (() => {
            const bgl = device.createBindGroupLayout({
              entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
              ],
            });
            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
            const code = generateUnpackC2RWGSL({ shape: [this.shape[0]], workgroupSize: this.workgroupSize });
            const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
            const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            return { bgl, pl: pipelineLayout, pipeline, params };
          })();
        }
        this.unpackFinalize = null;
        if (this._largeShapeMode) {
          this.unpackFinalize = (() => {
            const bgl = device.createBindGroupLayout({
              entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
              ],
            });
            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
            const code = generateFinalizeUnpackedHermitianWGSL({ shape: this.shape, workgroupSize: this.workgroupSize });
            const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
            const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            return { bgl, pl: pipelineLayout, pipeline, params };
          })();
        }
        this.unpackFinalizeSegment = null;
        if (this._largeShapeMode) {
          this.unpackFinalizeSegment = (() => {
            const bgl = device.createBindGroupLayout({
              entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
              ],
            });
            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
            const code = generateFinalizeUnpackedHermitianSegmentWGSL({ shape: this.shape, workgroupSize: this.workgroupSize });
            const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
            const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            return { bgl, pl: pipelineLayout, pipeline, params };
          })();
        }
    
        // complex->real
        this.c2r = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateComplexToRealWGSL({ workgroupSize: this.workgroupSize });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(params, 0, new Uint32Array([this.totalReal, 0, 0, 0]));
          return { bgl, pl: pipelineLayout, pipeline, params };
        })();
    
        // f16 input conversion for packed complex
        this.f16In = null;
        if (precision === "f16-storage") {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = this.cache.getComputePipeline({ code: generateF16ToF32ComplexWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(params, 0, new Uint32Array([this.inViewTotalComplex, 0, 0, 0]));
          this.f16In = { bgl, pl: pipelineLayout, pipeline, params };
        }
    
        // f16 output conversion
        this.f16Out = null;
        if (precision === "f16-storage") {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const toF32 = this.cache.getComputePipeline({ code: generateF16ToF32RealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const toF16 = this.cache.getComputePipeline({ code: generateF32ToF16RealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(params, 0, new Uint32Array([this.outViewTotalReal, 0, 0, 0]));
          this.f16Out = { bgl, pl: pipelineLayout, toF32, toF16, params };
        }
    
        // ioView mapping pipelines
        this.ioEmbed = null;
        if (this.ioIn && needsIoMapping(this.ioIn, this.packedShape)) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateEmbedComplexWGSL({
            rank: this.rank,
            logicalDims: this.packedShape,
            viewDims: this.ioIn.shape,
            offset: this.ioIn.offset,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.ioEmbed = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.ioIn.shape), logicalTotal: prod(this.packedShape) };
        }
    
        this.ioExtract = null;
        if (this.ioOut && needsIoMapping(this.ioOut, this.shape)) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateExtractRealWGSL({
            rank: this.rank,
            logicalDims: this.shape,
            viewDims: this.ioOut.shape,
            offset: this.ioOut.offset,
            clearOutside: this.ioOut.clearOutside,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.ioExtract = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.ioOut.shape), logicalTotal: this.logicalTotal };
        }
    
        this.zeroRead = null;
        if (this.zeroPadRead) {
          const bgl = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateZeroOutsideRangeComplexWGSL({
            shape: this.packedShape,
            start: this.zeroPadRead.start,
            end: this.zeroPadRead.end,
            batch: this.batch,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          this.zeroRead = { bgl, pl, pipeline };
        }
    
        this.zeroWrite = null;
        if (this.zeroPadWrite) {
          const bgl = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateZeroOutsideRangeRealWGSL({
            shape: this.shape,
            start: this.zeroPadWrite.start,
            end: this.zeroPadWrite.end,
            batch: this.batch,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          this.zeroWrite = { bgl, pl, pipeline };
        }
    
        // Optional strided gather/scatter for packed-complex input and real output (f32 only).
        this.stridedIn = null;
        this.stridedOut = null;
        if (this._usesStridedInput) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateGatherComplexStridedWGSL({
            shape: this.packedShape,
            strides: this._inputStrides,
            baseOffsetElements: this._inputOffsetElements,
            batchStrideElements: this._inputBatchStrideElements,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.stridedIn = { bgl, pl, pipeline, params };
        }
    
        if (this._usesStridedOutput) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateScatterRealStridedWGSL({
            shape: this.shape,
            strides: this._outputStrides,
            baseOffsetElements: this._outputOffsetElements,
            batchStrideElements: this._outputBatchStrideElements,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.stridedOut = { bgl, pl, pipeline, params };
        }
    
        // Workspace: [stage scratch][packed logical f32][full complex][real logical f32][optional f16 out scratch]
        this.realF32Bytes = this.totalReal * 4;
        this.stageInF32Bytes = this.ioEmbed ? this.inViewTotalComplex * 8 : 0;
        this.stageOutF32Bytes = this.ioExtract ? this.outViewTotalReal * 4 : 0;
        this.stageF16Bytes = precision === "f16-storage" ? this.inBytes : 0;
        const stageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        this.stageF16Offset = alignBytes(this.stageInF32Bytes, stageAlign);
        this.stageBytes = Math.max(this.stageOutF32Bytes, this.stageF16Offset + this.stageF16Bytes);
    
        const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        let off = 0;
        this.stageOffset = 0;
        off += this.stageBytes;
        off = alignBytes(off, storageAlign);
        this.packedOffset = off;
        off += this.packedF32Bytes;
        off = alignBytes(off, storageAlign);
        this.fullOffset = off;
        off += this.fullBytes;
        off = alignBytes(off, storageAlign);
        this.realOffset = off;
        off += this.realF32Bytes;
        off = alignBytes(off, storageAlign);
        this.outF16Offset = precision === "f16-storage" ? off : 0;
        off += precision === "f16-storage" ? this.outBytes : 0;
    
        this.workspaceBytes = off;
        this._splitWorkspace = null;
        const maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
        if (this.workspaceBytes <= maxBufferSize) {
          this._arena = createInternalArena(device, this.workspaceBytes);
        } else {
          const splitNeeds = [
            ["stage", this.stageBytes],
            ["packed", this.packedF32Bytes],
            ["full", this.fullBytes],
            ["real", this.realF32Bytes],
            ["outF16", this.precision === "f16-storage" ? this.outBytes : 0],
          ];
          for (const [name, bytes] of splitNeeds) {
            if (bytes > 0 && bytes > maxBufferSize) {
              throw new Error(
                `c2r split workspace cannot allocate ${name} buffer: ${bytes} bytes exceeds device.limits.maxBufferSize=${maxBufferSize}. ` +
                  `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
              );
            }
          }
          this._arena = null;
          this._splitWorkspace = {
            stage: this.stageBytes ? createInternalArena(device, this.stageBytes) : null,
            packed: createInternalArena(device, this.packedF32Bytes),
            full: createInternalArena(device, this.fullBytes),
            real: createInternalArena(device, this.realF32Bytes),
            outF16: this.precision === "f16-storage" ? createInternalArena(device, this.outBytes) : null,
          };
        }
        this._largeChunkBuffer = null;
        this._largeChunkBytes = 0;
        this._retiredLargeChunkBuffers = [];
        this._zeroRealBuffer = null;
        this._zeroComplexBuffer = null;
        this._deferredUniformBuffers = [];
      }
    
      getWorkspaceSizeBytes() {
        return this.workspaceBytes;
      }
    
      _ensureLargeChunkBuffer(minBytes) {
        if (this._largeChunkBuffer && this._largeChunkBytes >= minBytes) return this._largeChunkBuffer;
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        if (minBytes > maxBufferSize) {
          throw new Error(
            `c2r large-shape staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}. ` +
              `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
          );
        }
        if (this._largeChunkBuffer) this._retiredLargeChunkBuffers.push(this._largeChunkBuffer);
        this._largeChunkBuffer = this.device.createBuffer({
          size: minBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._largeChunkBytes = minBytes;
        return this._largeChunkBuffer;
      }
    
      _ensureZeroRealBuffer() {
        if (this._zeroRealBuffer) return this._zeroRealBuffer;
        this._zeroRealBuffer = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this._zeroRealBuffer, 0, new Float32Array([0]));
        return this._zeroRealBuffer;
      }
    
      _ensureZeroComplexBuffer() {
        if (this._zeroComplexBuffer) return this._zeroComplexBuffer;
        this._zeroComplexBuffer = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this._zeroComplexBuffer, 0, new Float32Array([0, 0]));
        return this._zeroComplexBuffer;
      }
    
      _copyRangesToContiguous(commandEncoder, ranges, dstBuffer, dstOffsetBytes) {
        let dst = dstOffsetBytes;
        for (const r of ranges) {
          commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
          dst += r.sizeBytes;
        }
      }
    
      _copyContiguousToRanges(commandEncoder, srcBuffer, srcOffsetBytes, ranges) {
        let src = srcOffsetBytes;
        for (const r of ranges) {
          commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
          src += r.sizeBytes;
        }
      }
    
      _normalizeCopyView(x) {
        if (x && x.buffer && Number.isInteger(x.offsetBytes) && Number.isInteger(x.sizeBytes)) {
          return {
            segments: [{ buffer: x.buffer, offsetBytes: x.offsetBytes, sizeBytes: x.sizeBytes }],
            logicalByteOffset: 0,
            lengthBytes: x.sizeBytes,
          };
        }
        return x;
      }
    
      _storageRef(x) {
        if (x && x.view) {
          return { store: x.view, baseOffsetBytes: x.offsetBytes ?? 0, sizeBytes: x.sizeBytes ?? null };
        }
        if (x && x.buffer && Number.isInteger(x.offsetBytes) && Number.isInteger(x.sizeBytes)) {
          return { store: x.buffer, baseOffsetBytes: x.offsetBytes, sizeBytes: x.sizeBytes };
        }
        return { store: x, baseOffsetBytes: 0, sizeBytes: null };
      }
    
      _copyAnySpan(commandEncoder, { src, srcOffsetBytes, dst, dstOffsetBytes, bytes }) {
        if (bytes <= 0) return;
        const srcRanges = normalizeToContiguousRanges(this._normalizeCopyView(src), srcOffsetBytes, bytes);
        const dstRanges = normalizeToContiguousRanges(this._normalizeCopyView(dst), dstOffsetBytes, bytes);
        if (srcRanges.length === 1 && dstRanges.length === 1) {
          const s = srcRanges[0];
          const d = dstRanges[0];
          if (s.buffer === d.buffer && s.offsetBytes === d.offsetBytes) return;
          commandEncoder.copyBufferToBuffer(s.buffer, s.offsetBytes, d.buffer, d.offsetBytes, bytes);
          return;
        }
        if (srcRanges.length > 1 && dstRanges.length === 1) {
          this.copier.pack(commandEncoder, srcRanges, dstRanges[0].buffer, dstRanges[0].offsetBytes);
          return;
        }
        if (srcRanges.length === 1 && dstRanges.length > 1) {
          this.copier.unpack(commandEncoder, srcRanges[0].buffer, srcRanges[0].offsetBytes, dstRanges);
          return;
        }
        let si = 0;
        let di = 0;
        let soff = srcRanges[0].offsetBytes;
        let doff = dstRanges[0].offsetBytes;
        let srem = srcRanges[0].sizeBytes;
        let drem = dstRanges[0].sizeBytes;
        while (si < srcRanges.length && di < dstRanges.length) {
          const n = Math.min(srem, drem);
          commandEncoder.copyBufferToBuffer(srcRanges[si].buffer, soff, dstRanges[di].buffer, doff, n);
          soff += n;
          doff += n;
          srem -= n;
          drem -= n;
          if (srem === 0) {
            si += 1;
            if (si < srcRanges.length) {
              soff = srcRanges[si].offsetBytes;
              srem = srcRanges[si].sizeBytes;
            }
          }
          if (drem === 0) {
            di += 1;
            if (di < dstRanges.length) {
              doff = dstRanges[di].offsetBytes;
              drem = dstRanges[di].sizeBytes;
            }
          }
        }
      }
    
      _copyComplexFromAny(commandEncoder, src, srcOffsetBytes, dstBuffer, dstOffsetBytes) {
        if (isGpuBuffer(src)) {
          commandEncoder.copyBufferToBuffer(src, srcOffsetBytes, dstBuffer, dstOffsetBytes, 8);
          return;
        }
        const srcRanges = normalizeToContiguousRanges(src, srcOffsetBytes, 8);
        if (srcRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, dstBuffer, dstOffsetBytes, 8);
          return;
        }
        const chunkBuf = this._ensureLargeChunkBuffer(8);
        this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
        commandEncoder.copyBufferToBuffer(chunkBuf, 0, dstBuffer, dstOffsetBytes, 8);
      }
    
      _copyRealToAny(commandEncoder, srcBuffer, srcOffsetBytes, dst, dstOffsetBytes) {
        if (isGpuBuffer(dst)) {
          commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dst, dstOffsetBytes, 4);
          return;
        }
        const dstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes, 4);
        if (dstRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dstRanges[0].buffer, dstRanges[0].offsetBytes, 4);
          return;
        }
        const chunkBuf = this._ensureLargeChunkBuffer(4);
        commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, chunkBuf, 0, 4);
        this.copier.unpack(commandEncoder, chunkBuf, 0, dstRanges);
      }
    
      _shapeStrides(shape) {
        return tensorContiguousStrides(shape);
      }
    
      _coordsFromLinear(i, shape, outCoords) {
        tensorCoordsFromLinear(i, shape, outCoords);
      }
    
      _linearFromCoords(coords, strides) {
        return tensorLinearFromCoords(coords, strides);
      }
    
      _requiredStridedInputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
        if (!this._inputTensorDesc) {
          throw new Error("internal error: strided input descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._inputTensorDesc, {
          bytesPerElement: 8,
          runtimeExtraElements: extraOffsetElements,
          batchStart,
          batchCount,
        });
      }
    
      _requiredStridedOutputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
        if (!this._outputTensorDesc) {
          throw new Error("internal error: strided output descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._outputTensorDesc, {
          bytesPerElement: 4,
          runtimeExtraElements: extraOffsetElements,
          batchStart,
          batchCount,
        });
      }
    
      _copyStridedInputPackedOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange }) {
        if (inputOffsetBytes % 8 !== 0) {
          throw new Error(`inputOffsetBytes must be a multiple of 8 for packed-complex strided input; got ${inputOffsetBytes}`);
        }
        const extraOffsetElements = (inputOffsetBytes / 8) | 0;
        const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
        const inputBytes = getBufferByteLength(input);
        if (inputBytes < neededBytes) {
          throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
        }
    
        const packedRef = this._storageRef(packedRange);
        const packedTotal = prod(this.packedShape);
        if (!this._needsInputMapping) {
          const coords = new Array(this.rank).fill(0);
          for (let b = 0; b < this.batch; b++) {
            const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
            const dstBase = packedRef.baseOffsetBytes + b * packedTotal * 8;
            for (let li = 0; li < packedTotal; li++) {
              this._coordsFromLinear(li, this.packedShape, coords);
              let srcElem = srcBatchBase;
              for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
              this._copyAnySpan(commandEncoder, {
                src: input,
                srcOffsetBytes: srcElem * 8,
                dst: packedRef.store,
                dstOffsetBytes: dstBase + li * 8,
                bytes: 8,
              });
            }
          }
          return;
        }
    
        const zeroBuf = this._ensureZeroComplexBuffer();
        const viewShape = this.ioIn.shape;
        const viewOffset = this.ioIn.offset;
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
          const dstBase = packedRef.baseOffsetBytes + b * packedTotal * 8;
          for (let li = 0; li < packedTotal; li++) {
            this._coordsFromLinear(li, this.packedShape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: packedRef.store,
                dstOffsetBytes: dstBase + li * 8,
                bytes: 8,
              });
              continue;
            }
            let srcElem = srcBatchBase;
            for (let d = 0; d < this.rank; d++) srcElem += viewCoords[d] * this._inputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src: input,
              srcOffsetBytes: srcElem * 8,
              dst: packedRef.store,
              dstOffsetBytes: dstBase + li * 8,
              bytes: 8,
            });
          }
        }
      }
    
      _copyContiguousRealToStridedOutputOutOfCore(commandEncoder, { realRange, output, outputOffsetBytes }) {
        if (outputOffsetBytes % 4 !== 0) {
          throw new Error(`outputOffsetBytes must be a multiple of 4 for real-strided output; got ${outputOffsetBytes}`);
        }
        const extraOffsetElements = (outputOffsetBytes / 4) | 0;
        const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
        const outputBytes = getBufferByteLength(output);
        if (outputBytes < neededBytes) {
          throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
        }
    
        const realRef = this._storageRef(realRange);
        if (!this._needsOutputMapping) {
          const coords = new Array(this.rank).fill(0);
          for (let b = 0; b < this.batch; b++) {
            const srcBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
            const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
            for (let li = 0; li < this.logicalTotal; li++) {
              this._coordsFromLinear(li, this.shape, coords);
              let dstElem = dstBatchBase;
              for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
              this._copyAnySpan(commandEncoder, {
                src: realRef.store,
                srcOffsetBytes: srcBase + li * 4,
                dst: output,
                dstOffsetBytes: dstElem * 4,
                bytes: 4,
              });
            }
          }
          return;
        }
    
        const viewShape = this.ioOut.shape;
        const viewOffset = this.ioOut.offset;
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
    
        if (this.ioOut.clearOutside) {
          const zeroBuf = this._ensureZeroRealBuffer();
          const viewTotal = prod(viewShape);
          for (let b = 0; b < this.batch; b++) {
            const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
            for (let vi = 0; vi < viewTotal; vi++) {
              this._coordsFromLinear(vi, viewShape, viewCoords);
              let dstElem = dstBatchBase;
              for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: output,
                dstOffsetBytes: dstElem * 4,
                bytes: 4,
              });
            }
          }
        }
    
        for (let b = 0; b < this.batch; b++) {
          const srcBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
          const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) continue;
            let dstElem = dstBatchBase;
            for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src: realRef.store,
              srcOffsetBytes: srcBase + li * 4,
              dst: output,
              dstOffsetBytes: dstElem * 4,
              bytes: 4,
            });
          }
        }
      }
    
      _zeroOutsideRangeRealLarge(commandEncoder, { dataRange, shape, start, end }) {
        const dataRef = this._storageRef(dataRange);
        const zeroBuf = this._ensureZeroRealBuffer();
        const logicalTotal = prod(shape);
        const coords = new Array(shape.length).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const base = dataRef.baseOffsetBytes + b * logicalTotal * 4;
          for (let i = 0; i < logicalTotal; i++) {
            this._coordsFromLinear(i, shape, coords);
            let inside = true;
            for (let d = 0; d < shape.length; d++) {
              if (coords[d] < start[d] || coords[d] >= end[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: dataRef.store,
                dstOffsetBytes: base + i * 4,
                bytes: 4,
              });
            }
          }
        }
      }
    
      _zeroOutsideRangeComplexLarge(commandEncoder, { dataRange, shape, start, end }) {
        const dataRef = this._storageRef(dataRange);
        const zeroBuf = this._ensureZeroComplexBuffer();
        const logicalTotal = prod(shape);
        const coords = new Array(shape.length).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const base = dataRef.baseOffsetBytes + b * logicalTotal * 8;
          for (let i = 0; i < logicalTotal; i++) {
            this._coordsFromLinear(i, shape, coords);
            let inside = true;
            for (let d = 0; d < shape.length; d++) {
              if (coords[d] < start[d] || coords[d] >= end[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: dataRef.store,
                dstOffsetBytes: base + i * 8,
                bytes: 8,
              });
            }
          }
        }
      }
    
      _embedInputComplexOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange }) {
        const packedRef = this._storageRef(packedRange);
        if (!this.ioEmbed) {
          this._copyAnySpan(commandEncoder, {
            src: input,
            srcOffsetBytes: inputOffsetBytes,
            dst: packedRef.store,
            dstOffsetBytes: packedRef.baseOffsetBytes,
            bytes: this.packedF32Bytes,
          });
          return;
        }
    
        const inBytes = this.inViewTotalComplex * 8;
        const inBuf = this._ensureLargeChunkBuffer(inBytes);
        this._copyAnySpan(commandEncoder, {
          src: input,
          srcOffsetBytes: inputOffsetBytes,
          dst: inBuf,
          dstOffsetBytes: 0,
          bytes: inBytes,
        });
    
        const zeroBuf = this._ensureZeroComplexBuffer();
        const viewShape = this.ioIn.shape;
        const viewOffset = this.ioIn.offset;
        const viewTotal = prod(viewShape);
        const viewStrides = this._shapeStrides(viewShape);
        const logicalTotal = prod(this.packedShape);
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const srcBase = b * viewTotal * 8;
          const dstBase = packedRef.baseOffsetBytes + b * logicalTotal * 8;
          for (let li = 0; li < logicalTotal; li++) {
            this._coordsFromLinear(li, this.packedShape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: packedRef.store,
                dstOffsetBytes: dstBase + li * 8,
                bytes: 8,
              });
              continue;
            }
            const vi = this._linearFromCoords(viewCoords, viewStrides);
            this._copyAnySpan(commandEncoder, {
              src: inBuf,
              srcOffsetBytes: srcBase + vi * 8,
              dst: packedRef.store,
              dstOffsetBytes: dstBase + li * 8,
              bytes: 8,
            });
          }
        }
      }
    
      _extractOutputRealOutOfCore(commandEncoder, { realRange, output, outputOffsetBytes }) {
        const realRef = this._storageRef(realRange);
        if (!this.ioExtract) {
          this._copyAnySpan(commandEncoder, {
            src: realRef.store,
            srcOffsetBytes: realRef.baseOffsetBytes,
            dst: output,
            dstOffsetBytes: outputOffsetBytes,
            bytes: this.outBytes,
          });
          return;
        }
    
        const outBytes = this.outViewTotalReal * 4;
        const outBuf = this._ensureLargeChunkBuffer(outBytes);
        if (!this.ioOut.clearOutside) {
          this._copyAnySpan(commandEncoder, {
            src: output,
            srcOffsetBytes: outputOffsetBytes,
            dst: outBuf,
            dstOffsetBytes: 0,
            bytes: outBytes,
          });
        } else {
          const zeroBuf = this._ensureZeroRealBuffer();
          for (let i = 0; i < this.outViewTotalReal; i++) {
            commandEncoder.copyBufferToBuffer(zeroBuf, 0, outBuf, i * 4, 4);
          }
        }
    
        const viewShape = this.ioOut.shape;
        const viewOffset = this.ioOut.offset;
        const viewTotal = prod(viewShape);
        const viewStrides = this._shapeStrides(viewShape);
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const srcBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
          const dstBase = b * viewTotal * 4;
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) continue;
            const vi = this._linearFromCoords(viewCoords, viewStrides);
            this._copyAnySpan(commandEncoder, {
              src: realRef.store,
              srcOffsetBytes: srcBase + li * 4,
              dst: outBuf,
              dstOffsetBytes: dstBase + vi * 4,
              bytes: 4,
            });
          }
        }
    
        this._copyAnySpan(commandEncoder, {
          src: outBuf,
          srcOffsetBytes: 0,
          dst: output,
          dstOffsetBytes: outputOffsetBytes,
          bytes: outBytes,
        });
      }
    
      _resolveLargeStageLinesPerChunk(stageKey, lineBytes) {
        const maxLinesByBind = Math.max(1, Math.floor(this._maxBindBytes / lineBytes));
        const policy = this._outOfCoreAxisWindowPolicy?.[stageKey] ?? null;
        let linesPerChunk = maxLinesByBind;
        if (policy && Number.isInteger(policy.linesPerChunk) && policy.linesPerChunk > 0) {
          linesPerChunk = Math.max(1, Math.min(linesPerChunk, policy.linesPerChunk));
        }
        const alignedLineStep = policy?.alignedLineStep ?? 1;
        if (Number.isInteger(alignedLineStep) && alignedLineStep > 1 && linesPerChunk >= alignedLineStep) {
          linesPerChunk = Math.max(alignedLineStep, Math.floor(linesPerChunk / alignedLineStep) * alignedLineStep);
        }
        return Math.max(1, Math.min(linesPerChunk, this._lineCount));
      }
    
      _runUnpackLineChunks(commandEncoder, { packedRange, fullRange }) {
        const packedRef = this._storageRef(packedRange);
        const fullRef = this._storageRef(fullRange);
        if (this._complexLineBytes > this._maxBindBytes || this._packedLineBytes > this._maxBindBytes) {
          this._runUnpackLineElementChunks(commandEncoder, { packedRange, fullRange });
          return;
        }
    
        const lineBytes = Math.max(this._complexLineBytes, this._packedLineBytes);
        const linesPerChunk = this._resolveLargeStageLinesPerChunk("unpack", lineBytes);
        const chunkBuf = this._ensureLargeChunkBuffer(linesPerChunk * this._complexLineBytes);
        const packedStrides = this._shapeStrides(this.packedShape);
        const packedPerBatch = prod(this.packedShape);
        const fullPerBatch = prod(this.shape);
        const linesPerBatch = prod(this.shape.slice(1));
        const packedCoords = new Array(this.rank).fill(0);
        const fullCoords = new Array(this.rank).fill(0);
        const inNx = this.packedShape[0];
        const chunkCountPerBatch = Math.ceil(linesPerBatch / linesPerChunk);
        const chunkCount = this.batch * chunkCountPerBatch;
        const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
        const paramStride = alignBytes(16, uniformAlign);
        const paramsBuf = this.device.createBuffer({
          size: chunkCount * paramStride,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._deferredUniformBuffers.push(paramsBuf);
        let chunkIndex = 0;
    
        for (let b = 0; b < this.batch; b++) {
          const packedBase = packedRef.baseOffsetBytes + b * packedPerBatch * 8;
          const fullBase = fullRef.baseOffsetBytes + b * fullPerBatch * 8;
          for (let line0 = 0; line0 < linesPerBatch; line0 += linesPerChunk) {
            const lines = Math.min(linesPerChunk, linesPerBatch - line0);
            const chunkElems = lines * this.shape[0];
            const chunkBytes = chunkElems * 8;
    
            for (let lineLocal = 0; lineLocal < lines; lineLocal++) {
              const lineInBatch = line0 + lineLocal;
              this._decodeLineCoordsFromIndex(lineInBatch, fullCoords);
              for (let x = 0; x < this.shape[0]; x++) {
                const mirrored = x >= inNx;
                packedCoords[0] = mirrored ? this.shape[0] - x : x;
                for (let d = 1; d < this.rank; d++) {
                  const c = fullCoords[d];
                  packedCoords[d] = mirrored ? (c === 0 ? 0 : this.shape[d] - c) : c;
                }
                const srcIdx = this._linearFromCoords(packedCoords, packedStrides);
                const dstIdx = lineLocal * this.shape[0] + x;
                this._copyAnySpan(commandEncoder, {
                  src: packedRef.store,
                  srcOffsetBytes: packedBase + srcIdx * 8,
                  dst: chunkBuf,
                  dstOffsetBytes: dstIdx * 8,
                  bytes: 8,
                });
              }
            }
    
            const paramOff = chunkIndex * paramStride;
            this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([line0, lines, 0, 0]));
            const bg = this.device.createBindGroup({
              layout: this.unpackFinalize.bgl,
              entries: [
                { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: chunkBytes } },
                { binding: 1, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.unpackFinalize.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(chunkElems / this.workgroupSize), 1, 1);
            pass.end();
    
            this._copyAnySpan(commandEncoder, {
              src: chunkBuf,
              srcOffsetBytes: 0,
              dst: fullRef.store,
              dstOffsetBytes: fullBase + line0 * this._complexLineBytes,
              bytes: chunkBytes,
            });
            chunkIndex += 1;
          }
        }
      }
    
      _runUnpackLineElementChunks(commandEncoder, { packedRange, fullRange }) {
        const packedRef = this._storageRef(packedRange);
        const fullRef = this._storageRef(fullRange);
        const maxElems = Math.max(1, Math.floor(this._maxBindBytes / 8));
        const chunkBuf = this._ensureLargeChunkBuffer(maxElems * 8);
        const packedStrides = this._shapeStrides(this.packedShape);
        const packedPerBatch = prod(this.packedShape);
        const fullPerBatch = prod(this.shape);
        const linesPerBatch = prod(this.shape.slice(1));
        const inNx = this.packedShape[0];
        const packedCoords = new Array(this.rank).fill(0);
        const fullCoords = new Array(this.rank).fill(0);
    
        for (let lineGlobal = 0; lineGlobal < this._lineCount; lineGlobal++) {
          const b = Math.floor(lineGlobal / linesPerBatch);
          const lineInBatch = lineGlobal - b * linesPerBatch;
          const packedBase = packedRef.baseOffsetBytes + b * packedPerBatch * 8;
          const fullBase = fullRef.baseOffsetBytes + b * fullPerBatch * 8;
          this._decodeLineCoordsFromIndex(lineInBatch, fullCoords);
          const lineSelfConj = this._isSelfConjugateLineCoords(fullCoords) ? 1 : 0;
    
          for (let x0 = 0; x0 < this.shape[0]; x0 += maxElems) {
            const count = Math.min(maxElems, this.shape[0] - x0);
            const chunkBytes = count * 8;
            for (let i = 0; i < count; i++) {
              const x = x0 + i;
              const mirrored = x >= inNx;
              packedCoords[0] = mirrored ? this.shape[0] - x : x;
              for (let d = 1; d < this.rank; d++) {
                const c = fullCoords[d];
                packedCoords[d] = mirrored ? (c === 0 ? 0 : this.shape[d] - c) : c;
              }
              const srcIdx = this._linearFromCoords(packedCoords, packedStrides);
              this._copyAnySpan(commandEncoder, {
                src: packedRef.store,
                srcOffsetBytes: packedBase + srcIdx * 8,
                dst: chunkBuf,
                dstOffsetBytes: i * 8,
                bytes: 8,
              });
            }
    
            this.device.queue.writeBuffer(this.unpackFinalizeSegment.params, 0, new Uint32Array([count, x0, lineSelfConj, 0]));
            const bg = this.device.createBindGroup({
              layout: this.unpackFinalizeSegment.bgl,
              entries: [
                { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: chunkBytes } },
                { binding: 1, resource: { buffer: this.unpackFinalizeSegment.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.unpackFinalizeSegment.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
            pass.end();
    
            this._copyAnySpan(commandEncoder, {
              src: chunkBuf,
              srcOffsetBytes: 0,
              dst: fullRef.store,
              dstOffsetBytes: fullBase + lineInBatch * this._complexLineBytes + x0 * 8,
              bytes: chunkBytes,
            });
          }
        }
      }
    
      _runComplexToRealLineChunks(commandEncoder, { fullRange, realRange }) {
        const fullRef = this._storageRef(fullRange);
        const realRef = this._storageRef(realRange);
        if (this._complexLineBytes > this._maxBindBytes || this._realLineBytes > this._maxBindBytes) {
          this._runComplexToRealElementChunks(commandEncoder, { fullRange, realRange });
          return;
        }
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const lineBytes = Math.max(this._complexLineBytes, this._realLineBytes);
        const maxLinesPerChunk = this._resolveLargeStageLinesPerChunk("complexToReal", lineBytes);
        const maxInBytes = maxLinesPerChunk * this._complexLineBytes;
        const maxOutBytes = maxLinesPerChunk * this._realLineBytes;
        const maxOutOffset = alignBytes(maxInBytes, storageAlign);
        const chunkBuf = this._ensureLargeChunkBuffer(maxOutOffset + maxOutBytes);
        const chunkCount = Math.ceil(this._lineCount / maxLinesPerChunk);
        const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
        const paramStride = alignBytes(16, uniformAlign);
        const paramsBuf = this.device.createBuffer({
          size: chunkCount * paramStride,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._deferredUniformBuffers.push(paramsBuf);
    
        let chunkIndex = 0;
        for (let line0 = 0; line0 < this._lineCount; line0 += maxLinesPerChunk) {
          const lines = Math.min(maxLinesPerChunk, this._lineCount - line0);
          const inBytes = lines * this._complexLineBytes;
          const outBytes = lines * this._realLineBytes;
          const outOff = alignBytes(inBytes, storageAlign);
          const srcOff = fullRef.baseOffsetBytes + line0 * this._complexLineBytes;
          const dstOff = realRef.baseOffsetBytes + line0 * this._realLineBytes;
    
          this._copyAnySpan(commandEncoder, {
            src: fullRef.store,
            srcOffsetBytes: srcOff,
            dst: chunkBuf,
            dstOffsetBytes: 0,
            bytes: inBytes,
          });
          const paramOff = chunkIndex * paramStride;
          this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([lines * this.shape[0], 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.c2r.bgl,
            entries: [
              { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
              { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
              { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.c2r.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((lines * this.shape[0]) / this.workgroupSize), 1, 1);
          pass.end();
          this._copyAnySpan(commandEncoder, {
            src: chunkBuf,
            srcOffsetBytes: outOff,
            dst: realRef.store,
            dstOffsetBytes: dstOff,
            bytes: outBytes,
          });
          chunkIndex += 1;
        }
      }
    
      _runComplexToRealElementChunks(commandEncoder, { fullRange, realRange }) {
        const fullRef = this._storageRef(fullRange);
        const realRef = this._storageRef(realRange);
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxElems = Math.max(1, Math.floor(this._maxBindBytes / 8));
        const maxInBytes = maxElems * 8;
        const maxOutBytes = maxElems * 4;
        const outOff = alignBytes(maxInBytes, storageAlign);
        const chunkBuf = this._ensureLargeChunkBuffer(outOff + maxOutBytes);
        const chunksPerLine = Math.ceil(this.shape[0] / maxElems);
        const chunkCount = this._lineCount * chunksPerLine;
        const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
        const paramStride = alignBytes(16, uniformAlign);
        const paramsBuf = this.device.createBuffer({
          size: chunkCount * paramStride,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._deferredUniformBuffers.push(paramsBuf);
    
        let chunkIndex = 0;
        for (let line = 0; line < this._lineCount; line++) {
          const srcLineBase = fullRef.baseOffsetBytes + line * this._complexLineBytes;
          const dstLineBase = realRef.baseOffsetBytes + line * this._realLineBytes;
          for (let x0 = 0; x0 < this.shape[0]; x0 += maxElems) {
            const elems = Math.min(maxElems, this.shape[0] - x0);
            const inBytes = elems * 8;
            const outBytes = elems * 4;
            this._copyAnySpan(commandEncoder, {
              src: fullRef.store,
              srcOffsetBytes: srcLineBase + x0 * 8,
              dst: chunkBuf,
              dstOffsetBytes: 0,
              bytes: inBytes,
            });
            const paramOff = chunkIndex * paramStride;
            this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([elems, 0, 0, 0]));
            const bg = this.device.createBindGroup({
              layout: this.c2r.bgl,
              entries: [
                { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
                { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
                { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.c2r.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(elems / this.workgroupSize), 1, 1);
            pass.end();
            this._copyAnySpan(commandEncoder, {
              src: chunkBuf,
              srcOffsetBytes: outOff,
              dst: realRef.store,
              dstOffsetBytes: dstLineBase + x0 * 4,
              bytes: outBytes,
            });
            chunkIndex += 1;
          }
        }
      }
    
      _decodeLineCoordsFromIndex(lineInBatch, outCoords) {
        let rem = lineInBatch;
        for (let d = 1; d < this.rank; d++) {
          const dim = this.shape[d];
          const c = rem % dim;
          outCoords[d] = c;
          rem = (rem - c) / dim;
        }
      }
    
      _isSelfConjugateLineCoords(coords) {
        for (let d = 1; d < this.rank; d++) {
          const c = coords[d];
          const n = this.shape[d];
          if (n % 2 === 0) {
            if (c !== 0 && c !== (n / 2)) return false;
          } else {
            if (c !== 0) return false;
          }
        }
        return true;
      }
    
      _resolveWorkspaceViews(temp) {
        const arena = temp ?? this._arena;
        if (arena) {
          if (getBufferByteLength(arena) < this.workspaceBytes) throw new Error("temp too small");
          return {
            stage: this.stageBytes ? viewFromArena(arena, this.stageOffset, this.stageBytes) : null,
            packedView: viewFromArena(arena, this.packedOffset, this.packedF32Bytes),
            complexView: viewFromArena(arena, this.fullOffset, this.fullBytes),
            realView: viewFromArena(arena, this.realOffset, this.realF32Bytes),
            f16OutScratch: this.precision === "f16-storage" ? viewFromArena(arena, this.outF16Offset, this.outBytes) : null,
          };
        }
        if (this._splitWorkspace) {
          return {
            stage: this.stageBytes ? viewFromArena(this._splitWorkspace.stage, 0, this.stageBytes) : null,
            packedView: viewFromArena(this._splitWorkspace.packed, 0, this.packedF32Bytes),
            complexView: viewFromArena(this._splitWorkspace.full, 0, this.fullBytes),
            realView: viewFromArena(this._splitWorkspace.real, 0, this.realF32Bytes),
            f16OutScratch: this.precision === "f16-storage" ? viewFromArena(this._splitWorkspace.outF16, 0, this.outBytes) : null,
          };
        }
        throw new Error("No workspace buffer");
      }
    
      _workspaceViewsAreContiguous(views) {
        const single = (view, bytes) => {
          if (!view || !bytes) return true;
          return normalizeToContiguousRanges(view, 0, bytes).length === 1;
        };
        return (
          single(views.stage, this.stageBytes) &&
          single(views.packedView, this.packedF32Bytes) &&
          single(views.complexView, this.fullBytes) &&
          single(views.realView, this.realF32Bytes) &&
          single(views.f16OutScratch, this.precision === "f16-storage" ? this.outBytes : 0)
        );
      }
    
      _resolveLargeWorkspaceRanges(temp) {
        const { packedView, complexView, realView } = this._resolveWorkspaceViews(temp);
        return {
          packedRange: { view: packedView, offsetBytes: 0, sizeBytes: this.packedF32Bytes },
          fullRange: { view: complexView, offsetBytes: 0, sizeBytes: this.fullBytes },
          realRange: { view: realView, offsetBytes: 0, sizeBytes: this.realF32Bytes },
        };
      }
    
      _execLargeShape(commandEncoder, { input, output, temp, inputOffsetBytes, outputOffsetBytes }) {
        const { packedRange, fullRange, realRange } = this._resolveLargeWorkspaceRanges(temp);
    
        if (this._usesStridedInput) {
          this._copyStridedInputPackedOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange });
        } else {
          this._embedInputComplexOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange });
        }
    
        if (this.zeroPadRead) {
          this._zeroOutsideRangeComplexLarge(commandEncoder, {
            dataRange: packedRange,
            shape: this.packedShape,
            start: this.zeroPadRead.start,
            end: this.zeroPadRead.end,
          });
        }
    
        this._runUnpackLineChunks(commandEncoder, { packedRange, fullRange });
        {
          const fullRef = this._storageRef(fullRange);
          this.c2c.exec(commandEncoder, { input: fullRef.store, inputOffsetBytes: fullRef.baseOffsetBytes });
        }
        this._runComplexToRealLineChunks(commandEncoder, { fullRange, realRange });
    
        if (this.zeroPadWrite) {
          this._zeroOutsideRangeRealLarge(commandEncoder, {
            dataRange: realRange,
            shape: this.shape,
            start: this.zeroPadWrite.start,
            end: this.zeroPadWrite.end,
          });
        }
    
        if (this._usesStridedOutput) {
          this._copyContiguousRealToStridedOutputOutOfCore(commandEncoder, { realRange, output, outputOffsetBytes });
        } else {
          this._extractOutputRealOutOfCore(commandEncoder, { realRange, output, outputOffsetBytes });
        }
      }
    
      destroy() {
        if (this._destroyed) return;
        this.c2c.destroy();
        this.unpack.params.destroy();
        this.unpackLine?.params?.destroy?.();
        this.unpackFinalize?.params?.destroy?.();
        this.unpackFinalizeSegment?.params?.destroy?.();
        this.c2r.params.destroy();
        this.f16In?.params?.destroy?.();
        this.f16Out?.params?.destroy?.();
        this.ioEmbed?.params?.destroy?.();
        this.ioExtract?.params?.destroy?.();
        this.stridedIn?.params?.destroy?.();
        this.stridedOut?.params?.destroy?.();
        this._largeChunkBuffer?.destroy?.();
        this._splitWorkspace?.stage?.destroy?.();
        this._splitWorkspace?.packed?.destroy?.();
        this._splitWorkspace?.full?.destroy?.();
        this._splitWorkspace?.real?.destroy?.();
        this._splitWorkspace?.outF16?.destroy?.();
        for (const b of this._retiredLargeChunkBuffers) b?.destroy?.();
        for (const b of this._deferredUniformBuffers) b?.destroy?.();
        this._zeroRealBuffer?.destroy?.();
        this._zeroComplexBuffer?.destroy?.();
        this._arena?.destroy?.();
        super.destroy();
      }
    
      exec(commandEncoder, execOpts) {
        if (this._destroyed) throw new Error("plan destroyed");
        const { input, output, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
        if (!input || !output) throw new Error("c2r exec requires input and output");
        let workspaceTemp = temp;
        if (workspaceTemp && (buffersAlias(workspaceTemp, input) || buffersAlias(workspaceTemp, output))) {
          workspaceTemp = null;
        }
        if (this._largeShapeMode) {
          this._execLargeShape(commandEncoder, { input, output, temp: workspaceTemp, inputOffsetBytes, outputOffsetBytes });
          return;
        }
        let workspaceViews = this._resolveWorkspaceViews(workspaceTemp);
        if (workspaceTemp && !this._workspaceViewsAreContiguous(workspaceViews)) {
          workspaceTemp = null;
          workspaceViews = this._resolveWorkspaceViews(null);
        }
        const { stage, packedView, complexView, realView, f16OutScratch } = workspaceViews;
    
        // Load physical packed spectrum into f32, then optional ioView embed into packed-logical domain.
        if (this._usesStridedInput) {
          if (this._needsInputMapping) {
            const packedRange = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
            this._copyStridedInputPackedOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange });
          } else {
            if (!isGpuBuffer(input)) {
              const packedRange = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
              this._copyStridedInputPackedOutOfCore(commandEncoder, { input, inputOffsetBytes, packedRange });
            } else {
              if (inputOffsetBytes % 8 !== 0) {
                throw new Error(`inputOffsetBytes must be a multiple of 8 for packed-complex strided input; got ${inputOffsetBytes}`);
              }
              const extraOffsetElements = (inputOffsetBytes / 8) | 0;
              const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
              ensureWithinBindingLimit(this.device, neededBytes, "c2r strided input binding");
              if (input.size < neededBytes) {
                throw new Error(`input buffer too small for strided layout: need ${neededBytes} bytes, have ${input.size}`);
              }
    
              const dstF32 = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
              this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([prod(this.packedShape), this.batch, extraOffsetElements, 0]));
              const bg = this.device.createBindGroup({
                layout: this.stridedIn.bgl,
                entries: [
                  { binding: 0, resource: { buffer: input, offset: 0, size: neededBytes } },
                  { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.packedF32Bytes } },
                  { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.stridedIn.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this.inTotalComplexLogical / this.workgroupSize), 1, 1);
              pass.end();
            }
          }
        } else if (this.precision === "f16-storage") {
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
          let srcBuf = inRanges[0].buffer;
          let srcOff = inRanges[0].offsetBytes;
          if (inRanges.length > 1) {
            const scratchF16 = normalizeToContiguousRanges(stage, this.stageF16Offset, this.inBytes)[0];
            this.copier.pack(commandEncoder, inRanges, scratchF16.buffer, scratchF16.offsetBytes);
            srcBuf = scratchF16.buffer;
            srcOff = scratchF16.offsetBytes;
          }
    
          const dstF32 = this.ioEmbed
            ? normalizeToContiguousRanges(stage, 0, this.inViewTotalComplex * 8)[0]
            : normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
    
          const bg = this.device.createBindGroup({
            layout: this.f16In.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.inBytes } },
              { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.inViewTotalComplex * 8 } },
              { binding: 2, resource: { buffer: this.f16In.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16In.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.inViewTotalComplex / this.workgroupSize), 1, 1);
          pass.end();
        } else {
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
          const dstF32 = this.ioEmbed
            ? normalizeToContiguousRanges(stage, 0, this.inBytes)[0]
            : normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
    
          if (inRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(inRanges[0].buffer, inRanges[0].offsetBytes, dstF32.buffer, dstF32.offsetBytes, this.inBytes);
          } else {
            this.copier.pack(commandEncoder, inRanges, dstF32.buffer, dstF32.offsetBytes);
          }
        }
    
        if (this.ioEmbed && !(this._usesStridedInput && this._needsInputMapping)) {
          this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.ioEmbed.logicalTotal, this.ioEmbed.viewTotal, this.batch, 0]));
          const src = normalizeToContiguousRanges(stage, 0, this.ioEmbed.viewTotal * this.batch * 8)[0];
          const dst = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.ioEmbed.bgl,
            entries: [
              { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.ioEmbed.viewTotal * this.batch * 8 } },
              { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.packedF32Bytes } },
              { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioEmbed.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((this.ioEmbed.logicalTotal * this.batch) / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        if (this.zeroRead) {
          const p = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.zeroRead.bgl,
            entries: [{ binding: 0, resource: { buffer: p.buffer, offset: p.offsetBytes, size: this.packedF32Bytes } }],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.zeroRead.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.inTotalComplexLogical / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        // unpack to full complex spectrum
        const packed = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
        const full = normalizeToContiguousRanges(complexView, 0, this.fullBytes)[0];
        {
          const bg = this.device.createBindGroup({
            layout: this.unpack.bgl,
            entries: [
              { binding: 0, resource: { buffer: packed.buffer, offset: packed.offsetBytes, size: this.packedF32Bytes } },
              { binding: 1, resource: { buffer: full.buffer, offset: full.offsetBytes, size: this.fullBytes } },
              { binding: 2, resource: { buffer: this.unpack.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.unpack.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalComplexFull / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        // inverse c2c
        this.c2c.exec(commandEncoder, { input: full.buffer, inputOffsetBytes: full.offsetBytes });
    
        // extract real part to logical realView
        const real = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
        {
          const bg = this.device.createBindGroup({
            layout: this.c2r.bgl,
            entries: [
              { binding: 0, resource: { buffer: full.buffer, offset: full.offsetBytes, size: this.fullBytes } },
              { binding: 1, resource: { buffer: real.buffer, offset: real.offsetBytes, size: this.realF32Bytes } },
              { binding: 2, resource: { buffer: this.c2r.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.c2r.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        if (this.zeroWrite) {
          const bg = this.device.createBindGroup({
            layout: this.zeroWrite.bgl,
            entries: [{ binding: 0, resource: { buffer: real.buffer, offset: real.offsetBytes, size: this.realF32Bytes } }],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.zeroWrite.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        // optional output view mapping (logical real -> output view)
        let outF32 = real;
        if (this.ioExtract && !(this._usesStridedOutput && this._needsOutputMapping)) {
          const viewTotal = this.ioExtract.viewTotal;
          const outBytesF32 = viewTotal * this.batch * 4;
          ensureWithinBindingLimit(this.device, outBytesF32, "c2r ioView.output");
          this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.ioExtract.logicalTotal, viewTotal, this.batch, 0]));
    
          // f32 output can be written directly when contiguous (preserves clearOutside=false semantics).
          if (this.precision === "f32") {
            const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
            if (outRanges.length === 1) {
              const bg = this.device.createBindGroup({
                layout: this.ioExtract.bgl,
                entries: [
                  { binding: 0, resource: { buffer: real.buffer, offset: real.offsetBytes, size: this.realF32Bytes } },
                  { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
                  { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.ioExtract.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
              pass.end();
              return;
            }
          }
    
          const dst = normalizeToContiguousRanges(stage, 0, outBytesF32)[0];
    
          // For clearOutside=false with staged output, initialize dst from the existing output so out-of-bounds
          // view elements preserve prior values.
          if (!this.ioOut.clearOutside) {
            const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
            if (this.precision === "f16-storage") {
              // output f16 -> dst f32
              let f16SrcBuf = outRanges[0].buffer;
              let f16SrcOff = outRanges[0].offsetBytes;
              if (outRanges.length > 1) {
                const tmpF16 = normalizeToContiguousRanges(f16OutScratch, 0, this.outBytes)[0];
                this.copier.pack(commandEncoder, outRanges, tmpF16.buffer, tmpF16.offsetBytes);
                f16SrcBuf = tmpF16.buffer;
                f16SrcOff = tmpF16.offsetBytes;
              }
              this.device.queue.writeBuffer(this.f16Out.params, 0, new Uint32Array([this.outViewTotalReal, 0, 0, 0]));
              const bg = this.device.createBindGroup({
                layout: this.f16Out.bgl,
                entries: [
                  { binding: 0, resource: { buffer: f16SrcBuf, offset: f16SrcOff, size: this.outBytes } },
                  { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: outBytesF32 } },
                  { binding: 2, resource: { buffer: this.f16Out.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.f16Out.toF32);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this.outViewTotalReal / this.workgroupSize), 1, 1);
              pass.end();
            } else {
              // output f32 -> dst f32
              if (outRanges.length === 1) {
                commandEncoder.copyBufferToBuffer(outRanges[0].buffer, outRanges[0].offsetBytes, dst.buffer, dst.offsetBytes, this.outBytes);
              } else {
                this.copier.pack(commandEncoder, outRanges, dst.buffer, dst.offsetBytes);
              }
            }
          }
    
          const bg = this.device.createBindGroup({
            layout: this.ioExtract.bgl,
            entries: [
              { binding: 0, resource: { buffer: real.buffer, offset: real.offsetBytes, size: this.realF32Bytes } },
              { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: outBytesF32 } },
              { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioExtract.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
          pass.end();
          outF32 = dst;
        }
    
        if (this._usesStridedOutput) {
          if (this._needsOutputMapping) {
            this._copyContiguousRealToStridedOutputOutOfCore(commandEncoder, {
              realRange: outF32,
              output,
              outputOffsetBytes,
            });
            return;
          }
          if (!isGpuBuffer(output)) {
            this._copyContiguousRealToStridedOutputOutOfCore(commandEncoder, {
              realRange: outF32,
              output,
              outputOffsetBytes,
            });
            return;
          }
          if (outputOffsetBytes % 4 !== 0) {
            throw new Error(`outputOffsetBytes must be a multiple of 4 for real-strided output; got ${outputOffsetBytes}`);
          }
          const extraOffsetElements = (outputOffsetBytes / 4) | 0;
          const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
          ensureWithinBindingLimit(this.device, neededBytes, "c2r strided output binding");
          if (output.size < neededBytes) {
            throw new Error(`output buffer too small for strided layout: need ${neededBytes} bytes, have ${output.size}`);
          }
    
          this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([this.logicalTotal, this.batch, extraOffsetElements, 0]));
          const bg = this.device.createBindGroup({
            layout: this.stridedOut.bgl,
            entries: [
              { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.realF32Bytes } },
              { binding: 1, resource: { buffer: output, offset: 0, size: neededBytes } },
              { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.stridedOut.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
          return;
        }
    
        // write output
        if (this.precision === "f16-storage") {
          const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
          if (outRanges.length === 1) {
            const bg = this.device.createBindGroup({
              layout: this.f16Out.bgl,
              entries: [
                { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.outViewTotalReal * 4 } },
                { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
                { binding: 2, resource: { buffer: this.f16Out.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.f16Out.toF16);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.outViewTotalReal / this.workgroupSize), 1, 1);
            pass.end();
            return;
          }
          const tmp = normalizeToContiguousRanges(f16OutScratch, 0, this.outBytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.f16Out.bgl,
            entries: [
              { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.outViewTotalReal * 4 } },
              { binding: 1, resource: { buffer: tmp.buffer, offset: tmp.offsetBytes, size: this.outBytes } },
              { binding: 2, resource: { buffer: this.f16Out.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16Out.toF16);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.outViewTotalReal / this.workgroupSize), 1, 1);
          pass.end();
          this.copier.unpack(commandEncoder, tmp.buffer, tmp.offsetBytes, outRanges);
          return;
        }
    
        const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
        if (outRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(outF32.buffer, outF32.offsetBytes, outRanges[0].buffer, outRanges[0].offsetBytes, this.outBytes);
        } else {
          this.copier.unpack(commandEncoder, outF32.buffer, outF32.offsetBytes, outRanges);
        }
      }
    }
    
    
    exports['C2RPlan'] = C2RPlan;
  });

  __define('src/runtime/plans/conv2d.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { BasePlan } = require('src/runtime/base_plan.js');
    const { createInternalArena, viewFromArena } = require('src/runtime/workspace.js');
    const { normalizeToContiguousRanges } = require('src/runtime/segmented_io.js');
    const { assertOneOf, isPositiveInt, alignBytes, ensureWithinBindingLimit, getBufferByteLength } = require('src/runtime/common.js');
    const { hashFloat32Array } = require('src/utils/hash.js');
    
    const { generateConv2dRealWGSL, generateConv2dComplexRealKernelWGSL, generateConv2dComplexComplexKernelWGSL } = require('src/kernels/conv2d.js');
    
    function isGpuBuffer(x) {
      return x && !x?.segments && typeof x.size === "number";
    }
    
    class Conv2dPlan extends BasePlan {
      constructor(device, opts) {
        super(device, opts);
        const { shape, batch = 1, layout = { interleavedComplex: true }, precision = "f32", conv } = opts ?? {};
        if (!Array.isArray(shape) || shape.length !== 2) throw new Error(`conv2d shape must be [H,W]`);
        if (!shape.every(isPositiveInt)) throw new Error("conv2d shape must be positive ints");
        if (!conv) throw new Error("conv2d requires conv object");
        if (!Number.isInteger(batch) || batch <= 0) throw new Error("batch must be positive int");
    
        const { kernelSize, kernelType = "real", padding = "same", pad = null, boundary = "zero" } = conv;
        if (![1, 2, 3].includes(kernelSize)) throw new Error("conv.kernelSize must be 1|2|3");
        assertOneOf(kernelType, ["real", "complex"], "conv.kernelType");
        assertOneOf(padding, ["valid", "same", "explicit"], "conv.padding");
        if (boundary !== "zero") throw new Error('conv.boundary currently supports only "zero"');
        assertOneOf(precision, ["f32", "f16-storage"], "precision");
        if (precision !== "f32") throw new Error('conv2d precision="f16-storage" is not implemented in current implementation');
    
        const complex = layout?.interleavedComplex === true;
        if (!complex && kernelType === "complex") throw new Error("real input/output does not support complex kernel in current implementation");
    
        this.shape = shape.slice();
        this.batch = batch;
        this.complex = complex;
        this.kernelSize = kernelSize;
        this.kernelType = kernelType;
        this.padding = padding;
    
        const [Hout, Wout] = shape;
        let pt = 0, pb = 0, padL = 0, padR = 0;
        if (padding === "same") {
          const p = Math.floor(kernelSize / 2);
          pt = p; pb = kernelSize - 1 - p;
          padL = p; padR = kernelSize - 1 - p;
        } else if (padding === "valid") {
          pt = pb = padL = padR = 0;
        } else {
          if (!Array.isArray(pad) || pad.length !== 4) throw new Error('conv.pad must be [top,bottom,left,right] when padding="explicit"');
          [pt, pb, padL, padR] = pad;
          if (![pt, pb, padL, padR].every((x) => Number.isInteger(x) && x >= 0)) throw new Error("conv.pad entries must be non-negative ints");
        }
        this.pad = [pt, pb, padL, padR];
    
        const Hin = Hout + (kernelSize - 1) - pt - pb;
        const Win = Wout + (kernelSize - 1) - padL - padR;
        if (Hin <= 0 || Win <= 0) throw new Error(`Derived input shape invalid: Hin=${Hin} Win=${Win}`);
        this.inShape = [Hin, Win];
    
        if (padding === "valid") {
          const expH = Hin - kernelSize + 1;
          const expW = Win - kernelSize + 1;
          if (expH !== Hout || expW !== Wout) {
            throw new Error(`padding="valid" requires output [Hin-k+1,Win-k+1]; got [${Hout},${Wout}]`);
          }
        }
    
        const inElems = Hin * Win * batch;
        const outElems = Hout * Wout * batch;
        this.inBytes = inElems * (complex ? 8 : 4);
        this.outBytes = outElems * (complex ? 8 : 4);
        ensureWithinBindingLimit(device, this.inBytes, "conv2d input");
        ensureWithinBindingLimit(device, this.outBytes, "conv2d output");
    
        const bgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          ],
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    
        let code;
        if (!complex) {
          code = generateConv2dRealWGSL({ Hin, Win, Hout, Wout, k: kernelSize, pad: this.pad, workgroupSize: this.workgroupSize });
        } else if (kernelType === "real") {
          code = generateConv2dComplexRealKernelWGSL({ Hin, Win, Hout, Wout, k: kernelSize, pad: this.pad, workgroupSize: this.workgroupSize });
        } else {
          code = generateConv2dComplexComplexKernelWGSL({ Hin, Win, Hout, Wout, k: kernelSize, pad: this.pad, workgroupSize: this.workgroupSize });
        }
        const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
        const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(params, 0, new Uint32Array([batch, 0, 0, 0]));
    
        this.pipeline = { bgl, pl: pipelineLayout, pipeline, params };
        this.kernelCache = new Map();
    
        const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        this.inStageOffset = 0;
        this.outStageOffset = alignBytes(this.inBytes, storageAlign);
    
        const kBytes = kernelType === "complex" ? kernelSize * kernelSize * 8 : kernelSize * kernelSize * 4;
        this.workspaceBytes = this.outStageOffset + this.outBytes + kBytes;
        this._arena = createInternalArena(device, this.workspaceBytes);
      }
    
      getWorkspaceSizeBytes() {
        return this.workspaceBytes;
      }
    
      destroy() {
        if (this._destroyed) return;
        this.pipeline.params.destroy();
        for (const b of this.kernelCache.values()) b.destroy();
        this._arena?.destroy?.();
        super.destroy();
      }
    
      _getOrUploadKernel(kernel) {
        if (isGpuBuffer(kernel)) return kernel;
        if (!(kernel instanceof Float32Array)) throw new Error("kernel must be GPUBuffer or Float32Array");
        const h = hashFloat32Array(kernel);
        const key = `${this.kernelType}:${this.kernelSize}:${h}:${kernel.length}`;
        let buf = this.kernelCache.get(key);
        if (!buf) {
          buf = this.device.createBuffer({ size: kernel.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          this.device.queue.writeBuffer(buf, 0, kernel);
          this.kernelCache.set(key, buf);
        }
        return buf;
      }
    
      exec(commandEncoder, execOpts) {
        if (this._destroyed) throw new Error("plan destroyed");
        const { input, output, kernel, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
        if (!input || !output) throw new Error("conv2d exec requires input and output");
        if (!kernel) throw new Error("conv2d exec requires kernel");
    
        const arena = temp ?? this._arena;
        if (!arena) throw new Error("No workspace buffer");
        if (getBufferByteLength(arena) < this.workspaceBytes) throw new Error("temp too small");
    
        const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
        const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
    
        const inStage = viewFromArena(arena, this.inStageOffset, this.inBytes);
        const outStage = viewFromArena(arena, this.outStageOffset, this.outBytes);
        const kBytes = this.kernelType === "complex" ? this.kernelSize * this.kernelSize * 8 : this.kernelSize * this.kernelSize * 4;
    
        let inBuf = null;
        let inOff = 0;
        if (inRanges.length === 1) {
          inBuf = inRanges[0].buffer;
          inOff = inRanges[0].offsetBytes;
        } else {
          inBuf = inStage.segments[0].buffer;
          inOff = inStage.segments[0].offsetBytes;
          this.copier.pack(commandEncoder, inRanges, inBuf, inOff);
        }
    
        let outBuf = null;
        let outOff = 0;
        const needsUnpack = outRanges.length > 1;
        if (!needsUnpack) {
          outBuf = outRanges[0].buffer;
          outOff = outRanges[0].offsetBytes;
        } else {
          outBuf = outStage.segments[0].buffer;
          outOff = outStage.segments[0].offsetBytes;
        }
    
        const kBuf = this._getOrUploadKernel(kernel);
        const bg = this.device.createBindGroup({
          layout: this.pipeline.bgl,
          entries: [
            { binding: 0, resource: { buffer: inBuf, offset: inOff, size: this.inBytes } },
            { binding: 1, resource: { buffer: outBuf, offset: outOff, size: this.outBytes } },
            { binding: 2, resource: { buffer: kBuf, offset: 0, size: kBytes } },
            { binding: 3, resource: { buffer: this.pipeline.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline.pipeline);
        pass.setBindGroup(0, bg);
        const outElems = this.outBytes / (this.complex ? 8 : 4);
        pass.dispatchWorkgroups(Math.ceil(outElems / this.workgroupSize), 1, 1);
        pass.end();
    
        if (needsUnpack) {
          this.copier.unpack(commandEncoder, outBuf, outOff, outRanges);
        }
      }
    }
    
    
    exports['Conv2dPlan'] = Conv2dPlan;
  });

  __define('src/runtime/plans/dct_fft.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { createFftPlan } = require('src/plan.js');
    const { BasePlan } = require('src/runtime/base_plan.js');
    const { createInternalArena, viewFromArena } = require('src/runtime/workspace.js');
    const { normalizeToContiguousRanges } = require('src/runtime/segmented_io.js');
    const { normalizeIoView } = require('src/runtime/ioview.js');
    const { normalizeZeroPad } = require('src/runtime/zero_pad.js');
    const { mergeLargeRouteMetadata, resolveLargeRoutingPolicy } = require('src/runtime/large_policy.js');
    const { resolveLayoutSemantics } = require('src/runtime/layout_semantics.js');
    const { C2CPlan } = require('src/runtime/plans/c2c.js');
    const { factorizeSupportedRadices } = require('src/utils/factors.js');
    const { coordsFromLinear: tensorCoordsFromLinear, createTensorDescriptor, requiredBytesForBatchRange } = require('src/runtime/tensor_descriptor.js');
    
    const { assertOneOf, isPositiveInt, prod, normalizeScaleFactor, ensureWithinBindingLimit, getBufferByteLength, alignBytes, align4Bytes, buffersAlias, isGpuBuffer } = require('src/runtime/common.js');
    
    const { generateScaleRealWGSL } = require('src/kernels/scale.js');
    const { generateZeroOutsideRangeRealWGSL } = require('src/kernels/zero_pad.js');
    const { generateEmbedRealWGSL, generateExtractRealWGSL } = require('src/kernels/ioview.js');
    const { generateF16ToF32RealWGSL, generateF32ToF16RealWGSL } = require('src/kernels/f16_storage.js');
    const { generateGatherRealStridedWGSL, generateScatterRealStridedWGSL } = require('src/kernels/strided_real.js');
    const { dctFftDirection, dctWorkLength, generateDctFftBuildWGSL, generateDctFftPostWGSL } = require('src/kernels/dct_fft.js');
    
    function needsIoMapping(io, logicalShape) {
      if (!io) return false;
      for (let i = 0; i < logicalShape.length; i++) {
        if (io.shape[i] !== logicalShape[i]) return true;
        if (io.offset[i] !== 0) return true;
      }
      return false;
    }
    
    function typeKindFor(type, direction) {
      if (type === "dct1") return "dct1";
      if (type === "dct4") return "dct4";
      if (type === "dct2") return direction === "forward" ? "dct2_fwd" : "dct2_inv";
      if (type === "dct3") return direction === "forward" ? "dct2_inv" : "dct2_fwd";
      if (type === "dst1") return "dst1";
      if (type === "dst4") return "dst4";
      if (type === "dst2") return direction === "forward" ? "dst2_fwd" : "dst2_inv";
      if (type === "dst3") return direction === "forward" ? "dst2_inv" : "dst2_fwd";
      throw new Error(`Unknown DCT/DST type ${type}`);
    }
    
    function axisStride(dims, axis) {
      let s = 1;
      for (let i = 0; i < axis; i++) s *= dims[i];
      return s;
    }
    
    class DctPlan extends BasePlan {
      constructor(device, opts) {
        super(device, opts);
        const {
          shape,
          type,
          direction = "forward",
          batch = 1,
          inPlace = false,
          normalize = "none",
          layout = { interleavedComplex: false },
          precision = "f32",
          ioView = null,
          zeroPad = null,
        } = opts ?? {};
    
        assertOneOf(type, ["dct1", "dct2", "dct3", "dct4", "dst1", "dst2", "dst3", "dst4"], "type");
        assertOneOf(direction, ["forward", "inverse"], "direction");
        if (!Array.isArray(shape) || shape.length < 1) throw new Error("shape must be rank >= 1");
        if (!shape.every(isPositiveInt)) throw new Error("shape must be positive ints");
        if (shape.some((n) => n < 2)) throw new Error(`All DCT/DST dimensions must be >= 2; got shape=${JSON.stringify(shape)}`);
        if (layout?.interleavedComplex !== false) throw new Error("DCT/DST uses real buffers; set layout.interleavedComplex=false");
        assertOneOf(normalize, ["none", "backward", "unitary"], "normalize");
        assertOneOf(precision, ["f32", "f16-storage"], "precision");
        if (precision === "f16-storage" && !device.features?.has?.("shader-f16")) throw new Error('precision="f16-storage" requires shader-f16');
        if (inPlace) throw new Error("DCT/DST inPlace is not supported in current implementation");
        if (!Number.isInteger(batch) || batch <= 0) throw new Error(`batch must be positive int; got ${batch}`);
    
        this.type = type;
        this.direction = direction;
        this.typeKind = typeKindFor(type, direction);
    
        this.shape = shape.slice();
        this.rank = shape.length;
        this.batch = batch;
        this.normalize = normalize;
        this.precision = precision;
    
        this.io = normalizeIoView(this.rank, this.shape, ioView ?? {});
        this.zeroPad = normalizeZeroPad(this.rank, this.shape, zeroPad ?? null, "zeroPad");
        this.inViewShape = (this.io.input?.shape ?? this.shape).slice();
        this.outViewShape = (this.io.output?.shape ?? this.shape).slice();
        this._inputLayoutShape = this.inViewShape.slice();
        this._outputLayoutShape = this.outViewShape.slice();
    
        this.logicalTotal = prod(this.shape);
        this.totalReal = this.logicalTotal * this.batch;
        this.logicalBytesF32 = this.totalReal * 4;
        this._logicalBytesPerBatchF32 = this.logicalTotal * 4;
    
        this._inViewPerBatch = prod(this.inViewShape);
        this._outViewPerBatch = prod(this.outViewShape);
        this.inViewTotal = this._inViewPerBatch * this.batch;
        this.outViewTotal = this._outViewPerBatch * this.batch;
        this._inBytesPerBatchRaw = precision === "f16-storage" ? this._inViewPerBatch * 2 : this._inViewPerBatch * 4;
        this._outBytesPerBatchRaw = precision === "f16-storage" ? this._outViewPerBatch * 2 : this._outViewPerBatch * 4;
        this._inBytesPerBatchBind = precision === "f16-storage" ? align4Bytes(this._inBytesPerBatchRaw) : this._inBytesPerBatchRaw;
        this._outBytesPerBatchBind = precision === "f16-storage" ? align4Bytes(this._outBytesPerBatchRaw) : this._outBytesPerBatchRaw;
        this.inBytes = precision === "f16-storage" ? align4Bytes(this.inViewTotal * 2) : this.inViewTotal * 4;
        this.outBytes = precision === "f16-storage" ? align4Bytes(this.outViewTotal * 2) : this.outViewTotal * 4;
    
        const resolvedLayout = resolveLayoutSemantics({
          layout,
          rank: this.rank,
          inputShape: this._inputLayoutShape,
          outputShape: this._outputLayoutShape,
        });
        this._inputStrides = resolvedLayout.inputStrides;
        this._outputStrides = resolvedLayout.outputStrides;
        this._inputOffsetElements = resolvedLayout.inputOffsetElements;
        this._outputOffsetElements = resolvedLayout.outputOffsetElements;
        this._inputBatchStrideElements = resolvedLayout.inputBatchStrideElements;
        this._outputBatchStrideElements = resolvedLayout.outputBatchStrideElements;
        this._usesStridedInput = resolvedLayout.usesStridedInput;
        this._usesStridedOutput = resolvedLayout.usesStridedOutput;
        this._usesWhdcnInput = resolvedLayout.usesWhdcnInput;
        this._usesWhdcnOutput = resolvedLayout.usesWhdcnOutput;
        this._inputTensorDesc = this._usesStridedInput
          ? createTensorDescriptor({
              name: "dct.input",
              shape: this._inputLayoutShape,
              strides: this._inputStrides,
              offsetElements: this._inputOffsetElements,
              batchStrideElements: this._inputBatchStrideElements,
            })
          : null;
        this._outputTensorDesc = this._usesStridedOutput
          ? createTensorDescriptor({
              name: "dct.output",
              shape: this._outputLayoutShape,
              strides: this._outputStrides,
              offsetElements: this._outputOffsetElements,
              batchStrideElements: this._outputBatchStrideElements,
            })
          : null;
        this._inputSpanElements = this._inputTensorDesc?.spanElements ?? 0;
        this._outputSpanElements = this._outputTensorDesc?.spanElements ?? 0;
        if ((this._usesStridedInput || this._usesStridedOutput) && this.precision !== "f32") {
          throw new Error('custom strides for dct/dst currently support precision:"f32" only');
        }
    
        // Complex work buffer max across axes.
        this.workBytesMax = 0;
        this._workBytesPerBatchMax = 0;
    
        // Axis ops: build -> FFT -> post
        this.axes = [];
        this.fftScratchBytes = 0;
        for (let axis = 0; axis < this.rank; axis++) {
          const axisLen = this.shape[axis];
          const stride = axisStride(this.shape, axis);
          const linesPerBatch = this.logicalTotal / axisLen;
          const lines = this.batch * linesPerBatch;
          if (!Number.isInteger(lines)) throw new Error("internal error: lines is not integer");
          const M = dctWorkLength(this.typeKind, axisLen);
          const workElems = lines * M;
          const workElemsPerBatch = linesPerBatch * M;
          const workBytes = workElems * 8;
          const workBytesPerBatch = workElemsPerBatch * 8;
          if (workBytes > this.workBytesMax) this.workBytesMax = workBytes;
          if (workBytesPerBatch > this._workBytesPerBatchMax) this._workBytesPerBatchMax = workBytesPerBatch;
    
          const buildBgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const buildPl = device.createPipelineLayout({ bindGroupLayouts: [buildBgl] });
          const buildCode = generateDctFftBuildWGSL({
            typeKind: this.typeKind,
            rank: this.rank,
            axis,
            dims: this.shape,
            axisLength: axisLen,
            workgroupSize: this.workgroupSize,
          });
          const buildPipe = this.cache.getComputePipeline({ code: buildCode, layout: buildPl });
          const buildParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(buildParams, 0, new Uint32Array([workElems, 0, 0, 0]));
    
          const postBgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const postPl = device.createPipelineLayout({ bindGroupLayouts: [postBgl] });
          const postCode = generateDctFftPostWGSL({
            typeKind: this.typeKind,
            rank: this.rank,
            axis,
            dims: this.shape,
            axisLength: axisLen,
            workgroupSize: this.workgroupSize,
          });
          const postPipe = this.cache.getComputePipeline({ code: postCode, layout: postPl });
          const postParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(postParams, 0, new Uint32Array([this.totalReal, 0, 0, 0]));
    
          const fftDir = dctFftDirection(this.typeKind);
          let fft = null;
          let fftNeedsBatch = false;
          let fftRunsPerBatch = false;
          if (factorizeSupportedRadices(M)) {
            fftNeedsBatch = true;
            fft = createFftPlan(device, {
              shape: [M],
              direction: fftDir,
              normalize: "none",
              inPlace: true,
              layout: "interleaved",
              precision: "f32",
            });
          } else {
            // Fallback to high-level planner so arbitrary lengths (e.g. DST-I with M=2*(N+1))
            // can use Bluestein/Rader internally. Keep this as per-logical-batch so large-mode
            // chunk scheduling and regular per-batch fallback both remain binding-safe.
            fftRunsPerBatch = true;
            fft = new C2CPlan(device, {
              shape: [M],
              direction: fftDir,
              batch: linesPerBatch,
              inPlace: true,
              normalize: "none",
              layout: { interleavedComplex: true },
              precision: "f32",
              ioView: { input: null, output: null },
              tuning: opts?.tuning ?? null,
            });
          }
    
          const fftWorkspaceBytes =
            typeof fft.getWorkspaceSizeBytes === "function"
              ? fft.getWorkspaceSizeBytes()
              : workBytes;
          this.fftScratchBytes = Math.max(this.fftScratchBytes, fftWorkspaceBytes);
    
          this.axes.push({
            axis,
            axisLen,
            stride,
            lines,
            linesPerBatch,
            M,
            workElems,
            workElemsPerBatch,
            workBytes,
            workBytesPerBatch,
            build: { bgl: buildBgl, pl: buildPl, pipeline: buildPipe, params: buildParams },
            post: { bgl: postBgl, pl: postPl, pipeline: postPipe, params: postParams },
            fft,
            fftNeedsBatch,
            fftRunsPerBatch,
          });
        }
        const largePolicy = resolveLargeRoutingPolicy({
          device,
          tuning: opts?.tuning ?? null,
          requiredBindingBytes: [this.logicalBytesF32, this.inBytes, this.outBytes, this.workBytesMax],
          lineBytes: this.shape.map((n) => n * 4),
          precision: this.precision,
        });
        this._maxBindBytes = largePolicy.maxBindBytes;
        this._largeBatchChunkMode = largePolicy.needsLargeMode;
        const mergedRoute = mergeLargeRouteMetadata([
          largePolicy,
          ...this.axes.map((ax) => ({
            routeMode: ax.fft?._largeRouteMode,
            reasonCodes: ax.fft?._largeRouteReasons,
            attemptedRoutes: ax.fft?._largeRouteAttempts,
          })),
        ]);
        this._largeRouteMode = mergedRoute.routeMode;
        this._largeRouteReasons = mergedRoute.reasonCodes;
        this._largeRouteAttempts = mergedRoute.attemptedRoutes;
        this._storageAlignBytes = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        this._storageAlignElems = Math.max(1, Math.floor(this._storageAlignBytes / 4));
    
        if (this._largeBatchChunkMode) {
          const perBatchRequirements = [
            this._logicalBytesPerBatchF32,
            this._inBytesPerBatchBind,
            this._outBytesPerBatchBind,
            this._workBytesPerBatchMax,
          ];
          if (this._usesStridedInput) perBatchRequirements.push((this._inputSpanElements + this._storageAlignElems - 1) * 4);
          if (this._usesStridedOutput) perBatchRequirements.push((this._outputSpanElements + this._storageAlignElems - 1) * 4);
          if (perBatchRequirements.some((bytes) => bytes > this._maxBindBytes)) {
            throw new Error(
              `DCT/DST large-mode requires one-batch bindings to fit maxStorageBufferBindingSize=${this._maxBindBytes} ` +
                `(perBatch=${JSON.stringify(perBatchRequirements)}). ` +
                `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
            );
          }
          this.workBytesMax = this._workBytesPerBatchMax;
        } else {
          ensureWithinBindingLimit(device, this.logicalBytesF32, `DCT logical buffer: shape=${JSON.stringify(shape)} batch=${batch}`);
          ensureWithinBindingLimit(device, this.inBytes, "DCT input");
          ensureWithinBindingLimit(device, this.outBytes, "DCT output");
          ensureWithinBindingLimit(device, this.workBytesMax, "DCT complex work");
        }
    
        // scale (applied once after final axis)
        this.scale = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = this.cache.getComputePipeline({ code: generateScaleRealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          return { bgl, pl: pipelineLayout, pipeline, params };
        })();
    
        // ioView mapping (input embed, output extract)
        this.ioEmbed = null;
        if (this.io.input && needsIoMapping(this.io.input, this.shape)) {
          const code = generateEmbedRealWGSL({
            rank: this.rank,
            logicalDims: this.shape,
            viewDims: this.io.input.shape,
            offset: this.io.input.offset,
            workgroupSize: this.workgroupSize,
          });
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.ioEmbed = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.io.input.shape) };
        }
        this.ioExtract = null;
        if (this.io.output && needsIoMapping(this.io.output, this.shape)) {
          const code = generateExtractRealWGSL({
            rank: this.rank,
            logicalDims: this.shape,
            viewDims: this.io.output.shape,
            offset: this.io.output.offset,
            clearOutside: this.io.output.clearOutside,
            workgroupSize: this.workgroupSize,
          });
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.ioExtract = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.io.output.shape) };
        }
    
        this.zeroRead = null;
        if (this.zeroPad.read) {
          const bgl = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateZeroOutsideRangeRealWGSL({
            shape: this.shape,
            start: this.zeroPad.read.start,
            end: this.zeroPad.read.end,
            batch: this.batch,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          this.zeroRead = { bgl, pl, pipeline };
        }
    
        this.zeroWrite = null;
        if (this.zeroPad.write) {
          const bgl = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateZeroOutsideRangeRealWGSL({
            shape: this.shape,
            start: this.zeroPad.write.start,
            end: this.zeroPad.write.end,
            batch: this.batch,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          this.zeroWrite = { bgl, pl, pipeline };
        }
    
        this.stridedIn = null;
        this.stridedOut = null;
        if (this._usesStridedInput) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateGatherRealStridedWGSL({
            shape: this._inputLayoutShape,
            strides: this._inputStrides,
            baseOffsetElements: 0,
            batchStrideElements: this._inputBatchStrideElements,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.stridedIn = { bgl, pl, pipeline, params };
        }
    
        if (this._usesStridedOutput) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateScatterRealStridedWGSL({
            shape: this._outputLayoutShape,
            strides: this._outputStrides,
            baseOffsetElements: 0,
            batchStrideElements: this._outputBatchStrideElements,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.stridedOut = { bgl, pl, pipeline, params };
        }
    
        // f16 I/O conversion (real)
        this.f16 = null;
        if (precision === "f16-storage") {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const toF32 = this.cache.getComputePipeline({ code: generateF16ToF32RealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const toF16 = this.cache.getComputePipeline({ code: generateF32ToF16RealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.f16 = { bgl, pl: pipelineLayout, toF32, toF16, params };
        }
    
        // Workspace layout:
        // [stage scratch] [dataA f32] [dataB f32] [workComplex f32] [fftScratch] [optional f16 out scratch]
        // When monolithic allocation exceeds maxBufferSize, this plan falls back to split internal section buffers.
        const stageInViewElems = this._largeBatchChunkMode ? this._inViewPerBatch : this.inViewTotal;
        const stageOutViewElems = this._largeBatchChunkMode ? this._outViewPerBatch : this.outViewTotal;
        this.stageInF32Bytes = this.ioEmbed ? stageInViewElems * 4 : 0;
        this.stageOutF32Bytes = this.ioExtract ? stageOutViewElems * 4 : 0;
        this.stageF16Bytes = precision === "f16-storage" ? (this._largeBatchChunkMode ? this._inBytesPerBatchBind : this.inBytes) : 0;
        const stageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        this.stageF16Offset = alignBytes(this.stageInF32Bytes, stageAlign);
        this.stageBytes = Math.max(this.stageOutF32Bytes, this.stageF16Offset + this.stageF16Bytes);
    
        this.dataBytes = this._largeBatchChunkMode ? this._logicalBytesPerBatchF32 : this.logicalBytesF32;
        this.fftScratchBytes = Math.max(this.fftScratchBytes, this.workBytesMax);
    
        // Storage bindings require offsets aligned to device.limits.minStorageBufferOffsetAlignment (usually 256).
        // Complex (vec2<f32>) also wants 8-byte alignment, which is implied by that limit on most devices.
        const a = Math.max(8, device.limits?.minStorageBufferOffsetAlignment ?? 256);
        let off = 0;
        this.stageOffset = 0;
        off = this.stageBytes;
        off = alignBytes(off, a);
        this.dataAOffset = off;
        off += this.dataBytes;
        off = alignBytes(off, a);
        this.dataBOffset = off;
        off += this.dataBytes;
        off = alignBytes(off, a);
        this.workOffset = off;
        off += this.workBytesMax;
        off = alignBytes(off, a);
        this.fftScratchOffset = off;
        off += this.fftScratchBytes;
        off = alignBytes(off, a);
        this.f16OutOffset = off;
        this.f16OutBytes = precision === "f16-storage" ? (this._largeBatchChunkMode ? this._outBytesPerBatchBind : this.outBytes) : 0;
        off += this.f16OutBytes;
        this.workspaceBytes = off;
        this._splitWorkspace = null;
        this._maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
        if (this.workspaceBytes <= this._maxBufferSize) {
          this._arena = createInternalArena(device, this.workspaceBytes);
        } else {
          const splitNeeds = [
            ["stage", this.stageBytes],
            ["dataA", this.dataBytes],
            ["dataB", this.dataBytes],
            ["work", this.workBytesMax],
            ["fftScratch", this.fftScratchBytes],
            ["f16Out", this.f16OutBytes],
          ];
          for (const [name, bytes] of splitNeeds) {
            if (bytes > 0 && bytes > this._maxBufferSize) {
              throw new Error(
                `dct/dst split workspace cannot allocate ${name} buffer: ${bytes} bytes exceeds device.limits.maxBufferSize=${this._maxBufferSize}. ` +
                  `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
              );
            }
          }
          this._arena = null;
          this._splitWorkspace = {
            stage: this.stageBytes ? createInternalArena(device, this.stageBytes) : null,
            dataA: createInternalArena(device, this.dataBytes),
            dataB: createInternalArena(device, this.dataBytes),
            work: createInternalArena(device, this.workBytesMax),
            fftScratch: createInternalArena(device, this.fftScratchBytes),
            f16Out: this.f16OutBytes ? createInternalArena(device, this.f16OutBytes) : null,
          };
        }
      }
    
      getWorkspaceSizeBytes() {
        return this.workspaceBytes;
      }
    
      destroy() {
        if (this._destroyed) return;
        for (const ax of this.axes) {
          ax.build.params.destroy();
          ax.post.params.destroy();
          ax.fft.destroy();
        }
        this.scale.params.destroy();
        this.ioEmbed?.params?.destroy?.();
        this.ioExtract?.params?.destroy?.();
        this.stridedIn?.params?.destroy?.();
        this.stridedOut?.params?.destroy?.();
        this.f16?.params?.destroy?.();
        this._splitWorkspace?.stage?.destroy?.();
        this._splitWorkspace?.dataA?.destroy?.();
        this._splitWorkspace?.dataB?.destroy?.();
        this._splitWorkspace?.work?.destroy?.();
        this._splitWorkspace?.fftScratch?.destroy?.();
        this._splitWorkspace?.f16Out?.destroy?.();
        this._arena?.destroy?.();
        super.destroy();
      }
    
      _resolveWorkspaceViews(arenaLike) {
        if (arenaLike) {
          if (getBufferByteLength(arenaLike) < this.workspaceBytes) throw new Error("temp too small");
          return {
            stage: this.stageBytes ? viewFromArena(arenaLike, this.stageOffset, this.stageBytes) : null,
            dataA: viewFromArena(arenaLike, this.dataAOffset, this.dataBytes),
            dataB: viewFromArena(arenaLike, this.dataBOffset, this.dataBytes),
            work: viewFromArena(arenaLike, this.workOffset, this.workBytesMax),
            fftScratch: viewFromArena(arenaLike, this.fftScratchOffset, this.fftScratchBytes),
            f16OutScratch: this.f16OutBytes ? viewFromArena(arenaLike, this.f16OutOffset, this.f16OutBytes) : null,
          };
        }
        if (this._splitWorkspace) {
          return {
            stage: this.stageBytes ? viewFromArena(this._splitWorkspace.stage, 0, this.stageBytes) : null,
            dataA: viewFromArena(this._splitWorkspace.dataA, 0, this.dataBytes),
            dataB: viewFromArena(this._splitWorkspace.dataB, 0, this.dataBytes),
            work: viewFromArena(this._splitWorkspace.work, 0, this.workBytesMax),
            fftScratch: viewFromArena(this._splitWorkspace.fftScratch, 0, this.fftScratchBytes),
            f16OutScratch: this.f16OutBytes ? viewFromArena(this._splitWorkspace.f16Out, 0, this.f16OutBytes) : null,
          };
        }
        throw new Error("No workspace buffer");
      }
    
      _normalizeCopyView(x) {
        if (x && x.buffer && Number.isInteger(x.offsetBytes) && Number.isInteger(x.sizeBytes)) {
          return {
            segments: [{ buffer: x.buffer, offsetBytes: x.offsetBytes, sizeBytes: x.sizeBytes }],
            logicalByteOffset: 0,
            lengthBytes: x.sizeBytes,
          };
        }
        return x;
      }
    
      _copyAnySpan(commandEncoder, { src, srcOffsetBytes, dst, dstOffsetBytes, bytes }) {
        if (bytes <= 0) return;
        const srcRanges = normalizeToContiguousRanges(this._normalizeCopyView(src), srcOffsetBytes, bytes);
        const dstRanges = normalizeToContiguousRanges(this._normalizeCopyView(dst), dstOffsetBytes, bytes);
        if (srcRanges.length === 1 && dstRanges.length === 1) {
          const s = srcRanges[0];
          const d = dstRanges[0];
          if (s.buffer === d.buffer && s.offsetBytes === d.offsetBytes) return;
          commandEncoder.copyBufferToBuffer(s.buffer, s.offsetBytes, d.buffer, d.offsetBytes, bytes);
          return;
        }
        if (srcRanges.length > 1 && dstRanges.length === 1) {
          this.copier.pack(commandEncoder, srcRanges, dstRanges[0].buffer, dstRanges[0].offsetBytes);
          return;
        }
        if (srcRanges.length === 1 && dstRanges.length > 1) {
          this.copier.unpack(commandEncoder, srcRanges[0].buffer, srcRanges[0].offsetBytes, dstRanges);
          return;
        }
        let si = 0;
        let di = 0;
        let soff = srcRanges[0].offsetBytes;
        let doff = dstRanges[0].offsetBytes;
        let srem = srcRanges[0].sizeBytes;
        let drem = dstRanges[0].sizeBytes;
        while (si < srcRanges.length && di < dstRanges.length) {
          const n = Math.min(srem, drem);
          commandEncoder.copyBufferToBuffer(srcRanges[si].buffer, soff, dstRanges[di].buffer, doff, n);
          soff += n;
          doff += n;
          srem -= n;
          drem -= n;
          if (srem === 0) {
            si += 1;
            if (si < srcRanges.length) {
              soff = srcRanges[si].offsetBytes;
              srem = srcRanges[si].sizeBytes;
            }
          }
          if (drem === 0) {
            di += 1;
            if (di < dstRanges.length) {
              doff = dstRanges[di].offsetBytes;
              drem = dstRanges[di].sizeBytes;
            }
          }
        }
      }
    
      _coordsFromLinear(i, shape, outCoords) {
        tensorCoordsFromLinear(i, shape, outCoords);
      }
    
      _requiredStridedInputBytes(runtimeExtraElements, batchStart = 0, batchCount = this.batch) {
        if (!this._inputTensorDesc) {
          throw new Error("internal error: strided input descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._inputTensorDesc, {
          bytesPerElement: 4,
          runtimeExtraElements,
          batchStart,
          batchCount,
        });
      }
    
      _requiredStridedOutputBytes(runtimeExtraElements, batchStart = 0, batchCount = this.batch) {
        if (!this._outputTensorDesc) {
          throw new Error("internal error: strided output descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._outputTensorDesc, {
          bytesPerElement: 4,
          runtimeExtraElements,
          batchStart,
          batchCount,
        });
      }
    
      _validateStridedInputBounds(input, inputOffsetBytes, batchStart, batchCount) {
        if (inputOffsetBytes % 4 !== 0) {
          throw new Error(`inputOffsetBytes must be a multiple of 4 for real-strided input; got ${inputOffsetBytes}`);
        }
        const runtimeExtraElements = (inputOffsetBytes / 4) | 0;
        const neededBytes = this._requiredStridedInputBytes(runtimeExtraElements, batchStart, batchCount);
        const inputBytes = getBufferByteLength(input);
        if (inputBytes < neededBytes) {
          throw new Error(`input buffer/view too small for dct/dst strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
        }
        return runtimeExtraElements;
      }
    
      _validateStridedOutputBounds(output, outputOffsetBytes, batchStart, batchCount) {
        if (outputOffsetBytes % 4 !== 0) {
          throw new Error(`outputOffsetBytes must be a multiple of 4 for real-strided output; got ${outputOffsetBytes}`);
        }
        const runtimeExtraElements = (outputOffsetBytes / 4) | 0;
        const neededBytes = this._requiredStridedOutputBytes(runtimeExtraElements, batchStart, batchCount);
        const outputBytes = getBufferByteLength(output);
        if (outputBytes < neededBytes) {
          throw new Error(`output buffer/view too small for dct/dst strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
        }
        return runtimeExtraElements;
      }
    
      _copyStridedInputToContiguous(commandEncoder, { input, inputOffsetBytes, batchStart, batchCount, dstBuffer, dstOffsetBytes }) {
        const runtimeExtraElements = this._validateStridedInputBounds(input, inputOffsetBytes, batchStart, batchCount);
        const coords = new Array(this.rank).fill(0);
        for (let lb = 0; lb < batchCount; lb++) {
          const gb = batchStart + lb;
          const srcBatchBase = this._inputOffsetElements + runtimeExtraElements + gb * this._inputBatchStrideElements;
          const dstBase = dstOffsetBytes + lb * this._inViewPerBatch * 4;
          for (let vi = 0; vi < this._inViewPerBatch; vi++) {
            this._coordsFromLinear(vi, this._inputLayoutShape, coords);
            let srcElem = srcBatchBase;
            for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src: input,
              srcOffsetBytes: srcElem * 4,
              dst: dstBuffer,
              dstOffsetBytes: dstBase + vi * 4,
              bytes: 4,
            });
          }
        }
      }
    
      _copyStridedOutputToContiguous(commandEncoder, { output, outputOffsetBytes, batchStart, batchCount, dstBuffer, dstOffsetBytes }) {
        const runtimeExtraElements = this._validateStridedOutputBounds(output, outputOffsetBytes, batchStart, batchCount);
        const coords = new Array(this.rank).fill(0);
        for (let lb = 0; lb < batchCount; lb++) {
          const gb = batchStart + lb;
          const srcBatchBase = this._outputOffsetElements + runtimeExtraElements + gb * this._outputBatchStrideElements;
          const dstBase = dstOffsetBytes + lb * this._outViewPerBatch * 4;
          for (let vi = 0; vi < this._outViewPerBatch; vi++) {
            this._coordsFromLinear(vi, this._outputLayoutShape, coords);
            let srcElem = srcBatchBase;
            for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._outputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src: output,
              srcOffsetBytes: srcElem * 4,
              dst: dstBuffer,
              dstOffsetBytes: dstBase + vi * 4,
              bytes: 4,
            });
          }
        }
      }
    
      _copyContiguousToStridedOutput(commandEncoder, { srcBuffer, srcOffsetBytes, output, outputOffsetBytes, batchStart, batchCount }) {
        const runtimeExtraElements = this._validateStridedOutputBounds(output, outputOffsetBytes, batchStart, batchCount);
        const coords = new Array(this.rank).fill(0);
        for (let lb = 0; lb < batchCount; lb++) {
          const gb = batchStart + lb;
          const srcBase = srcOffsetBytes + lb * this._outViewPerBatch * 4;
          const dstBatchBase = this._outputOffsetElements + runtimeExtraElements + gb * this._outputBatchStrideElements;
          for (let vi = 0; vi < this._outViewPerBatch; vi++) {
            this._coordsFromLinear(vi, this._outputLayoutShape, coords);
            let dstElem = dstBatchBase;
            for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src: srcBuffer,
              srcOffsetBytes: srcBase + vi * 4,
              dst: output,
              dstOffsetBytes: dstElem * 4,
              bytes: 4,
            });
          }
        }
      }
    
      _resolveStridedInputBinding(input, inputOffsetBytes) {
        if (!isGpuBuffer(input)) {
          throw new Error("dct/dst custom-strided input currently requires GPUBuffer input");
        }
        const runtimeExtraElements = this._validateStridedInputBounds(input, inputOffsetBytes, 0, this.batch);
        const extraOffsetElements = runtimeExtraElements + this._inputOffsetElements;
        const neededBytes = this._requiredStridedInputBytes(runtimeExtraElements);
        ensureWithinBindingLimit(this.device, neededBytes, "dct/dst strided input binding");
        return { extraOffsetElements, neededBytes };
      }
    
      _resolveStridedOutputBinding(output, outputOffsetBytes) {
        if (!isGpuBuffer(output)) {
          throw new Error("dct/dst custom-strided output currently requires GPUBuffer output");
        }
        const runtimeExtraElements = this._validateStridedOutputBounds(output, outputOffsetBytes, 0, this.batch);
        const extraOffsetElements = runtimeExtraElements + this._outputOffsetElements;
        const neededBytes = this._requiredStridedOutputBytes(runtimeExtraElements);
        ensureWithinBindingLimit(this.device, neededBytes, "dct/dst strided output binding");
        return { extraOffsetElements, neededBytes };
      }
    
      _resolveStridedInputBatchWindow(input, inputOffsetBytes, batchIndex) {
        if (!isGpuBuffer(input)) {
          throw new Error("dct/dst custom-strided input currently requires GPUBuffer input");
        }
        if (inputOffsetBytes % 4 !== 0) {
          throw new Error(`inputOffsetBytes must be a multiple of 4 for real-strided input; got ${inputOffsetBytes}`);
        }
        const runtimeExtraElements = (inputOffsetBytes / 4) | 0;
        const baseElements = this._inputOffsetElements + runtimeExtraElements + batchIndex * this._inputBatchStrideElements;
        const windowStartElements = Math.floor(baseElements / this._storageAlignElems) * this._storageAlignElems;
        const extraOffsetElements = baseElements - windowStartElements;
        const neededElements = extraOffsetElements + this._inputSpanElements;
        const neededBytes = neededElements * 4;
        ensureWithinBindingLimit(this.device, neededBytes, "dct/dst strided input batch-window binding");
        const windowEndBytes = (windowStartElements + neededElements) * 4;
        if (windowEndBytes > input.size) {
          throw new Error(`input buffer too small for dct/dst strided batch window: need ${windowEndBytes} bytes, have ${input.size}`);
        }
        return {
          bindingOffsetBytes: windowStartElements * 4,
          bindingSizeBytes: neededBytes,
          extraOffsetElements,
        };
      }
    
      _resolveStridedOutputBatchWindow(output, outputOffsetBytes, batchIndex) {
        if (!isGpuBuffer(output)) {
          throw new Error("dct/dst custom-strided output currently requires GPUBuffer output");
        }
        if (outputOffsetBytes % 4 !== 0) {
          throw new Error(`outputOffsetBytes must be a multiple of 4 for real-strided output; got ${outputOffsetBytes}`);
        }
        const runtimeExtraElements = (outputOffsetBytes / 4) | 0;
        const baseElements = this._outputOffsetElements + runtimeExtraElements + batchIndex * this._outputBatchStrideElements;
        const windowStartElements = Math.floor(baseElements / this._storageAlignElems) * this._storageAlignElems;
        const extraOffsetElements = baseElements - windowStartElements;
        const neededElements = extraOffsetElements + this._outputSpanElements;
        const neededBytes = neededElements * 4;
        ensureWithinBindingLimit(this.device, neededBytes, "dct/dst strided output batch-window binding");
        const windowEndBytes = (windowStartElements + neededElements) * 4;
        if (windowEndBytes > output.size) {
          throw new Error(`output buffer too small for dct/dst strided batch window: need ${windowEndBytes} bytes, have ${output.size}`);
        }
        return {
          bindingOffsetBytes: windowStartElements * 4,
          bindingSizeBytes: neededBytes,
          extraOffsetElements,
        };
      }
    
      _execLargeBatchChunk(commandEncoder, { input, output, inputOffsetBytes, outputOffsetBytes, workspaceViews }) {
        const { stage, dataA, dataB, work, fftScratch, f16OutScratch } = workspaceViews;
    
        const dataARange = normalizeToContiguousRanges(dataA, 0, this.dataBytes)[0];
        const dataBRange = normalizeToContiguousRanges(dataB, 0, this.dataBytes)[0];
        const stageInputF32Range =
          this.ioEmbed && stage
            ? normalizeToContiguousRanges(stage, 0, this._inViewPerBatch * 4)[0]
            : null;
        const stageOutputF32Range =
          this.ioExtract && stage
            ? normalizeToContiguousRanges(stage, 0, this._outViewPerBatch * 4)[0]
            : null;
        const stageF16Range =
          this.precision === "f16-storage" && stage
            ? normalizeToContiguousRanges(stage, this.stageF16Offset, this._inBytesPerBatchBind)[0]
            : null;
        const f16OutRange =
          this.precision === "f16-storage" && f16OutScratch
            ? normalizeToContiguousRanges(f16OutScratch, 0, this._outBytesPerBatchBind)[0]
            : null;
    
        const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
    
        for (let b = 0; b < this.batch; b++) {
          const inBatchOffset = inputOffsetBytes + b * this._inBytesPerBatchRaw;
          const outBatchOffset = outputOffsetBytes + b * this._outBytesPerBatchRaw;
    
          if (this.precision === "f16-storage") {
            const inRanges = normalizeToContiguousRanges(input, inBatchOffset, this._inBytesPerBatchRaw);
            if (inRanges.length === 1) {
              commandEncoder.copyBufferToBuffer(
                inRanges[0].buffer,
                inRanges[0].offsetBytes,
                stageF16Range.buffer,
                stageF16Range.offsetBytes,
                this._inBytesPerBatchRaw
              );
            } else {
              this.copier.pack(commandEncoder, inRanges, stageF16Range.buffer, stageF16Range.offsetBytes);
            }
    
            const dstF32 = this.ioEmbed ? stageInputF32Range : dataARange;
            this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this._inViewPerBatch, 0, 0, 0]));
            const bg = this.device.createBindGroup({
              layout: this.f16.bgl,
              entries: [
                { binding: 0, resource: { buffer: stageF16Range.buffer, offset: stageF16Range.offsetBytes, size: this._inBytesPerBatchBind } },
                { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this._inViewPerBatch * 4 } },
                { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.f16.toF32);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this._inViewPerBatch / this.workgroupSize), 1, 1);
            pass.end();
          } else if (this._usesStridedInput) {
            const dstF32 = this.ioEmbed ? stageInputF32Range : dataARange;
            if (isGpuBuffer(input)) {
              const window = this._resolveStridedInputBatchWindow(input, inputOffsetBytes, b);
              this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([this._inViewPerBatch, 1, window.extraOffsetElements, 0]));
              const bg = this.device.createBindGroup({
                layout: this.stridedIn.bgl,
                entries: [
                  { binding: 0, resource: { buffer: input, offset: window.bindingOffsetBytes, size: window.bindingSizeBytes } },
                  { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this._inViewPerBatch * 4 } },
                  { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.stridedIn.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this._inViewPerBatch / this.workgroupSize), 1, 1);
              pass.end();
            } else {
              this._copyStridedInputToContiguous(commandEncoder, {
                input,
                inputOffsetBytes,
                batchStart: b,
                batchCount: 1,
                dstBuffer: dstF32.buffer,
                dstOffsetBytes: dstF32.offsetBytes,
              });
            }
          } else {
            const inRanges = normalizeToContiguousRanges(input, inBatchOffset, this._inBytesPerBatchRaw);
            const dstF32 = this.ioEmbed ? stageInputF32Range : dataARange;
            if (inRanges.length === 1) {
              commandEncoder.copyBufferToBuffer(
                inRanges[0].buffer,
                inRanges[0].offsetBytes,
                dstF32.buffer,
                dstF32.offsetBytes,
                this._inBytesPerBatchRaw
              );
            } else {
              this.copier.pack(commandEncoder, inRanges, dstF32.buffer, dstF32.offsetBytes);
            }
          }
    
          if (this.ioEmbed) {
            this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, this.ioEmbed.viewTotal, 1, 0]));
            const bg = this.device.createBindGroup({
              layout: this.ioEmbed.bgl,
              entries: [
                { binding: 0, resource: { buffer: stageInputF32Range.buffer, offset: stageInputF32Range.offsetBytes, size: this.ioEmbed.viewTotal * 4 } },
                { binding: 1, resource: { buffer: dataARange.buffer, offset: dataARange.offsetBytes, size: this._logicalBytesPerBatchF32 } },
                { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.ioEmbed.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          if (this.zeroRead) {
            const bg = this.device.createBindGroup({
              layout: this.zeroRead.bgl,
              entries: [{ binding: 0, resource: { buffer: dataARange.buffer, offset: dataARange.offsetBytes, size: this._logicalBytesPerBatchF32 } }],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.zeroRead.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          let srcBuf = dataARange.buffer;
          let srcOff = dataARange.offsetBytes;
          let dstBuf = dataBRange.buffer;
          let dstOff = dataBRange.offsetBytes;
    
          for (const ax of this.axes) {
            const workRange = normalizeToContiguousRanges(work, 0, ax.workBytesPerBatch)[0];
            this.device.queue.writeBuffer(ax.build.params, 0, new Uint32Array([ax.workElemsPerBatch, 0, 0, 0]));
            {
              const bg = this.device.createBindGroup({
                layout: ax.build.bgl,
                entries: [
                  { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this._logicalBytesPerBatchF32 } },
                  { binding: 1, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes, size: ax.workBytesPerBatch } },
                  { binding: 2, resource: { buffer: ax.build.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(ax.build.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(ax.workElemsPerBatch / this.workgroupSize), 1, 1);
              pass.end();
            }
    
            const fftExec = {
              input: workRange.buffer,
              inputOffsetBytes: workRange.offsetBytes,
              temp: fftScratch,
            };
            if (ax.fftNeedsBatch) fftExec.batch = ax.linesPerBatch;
            ax.fft.exec(commandEncoder, fftExec);
    
            this.device.queue.writeBuffer(ax.post.params, 0, new Uint32Array([this.logicalTotal, 0, 0, 0]));
            {
              const bg = this.device.createBindGroup({
                layout: ax.post.bgl,
                entries: [
                  { binding: 0, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes, size: ax.workBytesPerBatch } },
                  { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: this._logicalBytesPerBatchF32 } },
                  { binding: 2, resource: { buffer: ax.post.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(ax.post.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
              pass.end();
            }
    
            [srcBuf, dstBuf] = [dstBuf, srcBuf];
            [srcOff, dstOff] = [dstOff, srcOff];
          }
    
          if (scale !== 1.0) {
            this.device.queue.writeBuffer(this.scale.params, 0, new Uint32Array([this.logicalTotal, 0, 0, 0]));
            this.device.queue.writeBuffer(this.scale.params, 16, new Float32Array([scale, 0, 0, 0]));
            const bg = this.device.createBindGroup({
              layout: this.scale.bgl,
              entries: [
                { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this._logicalBytesPerBatchF32 } },
                { binding: 1, resource: { buffer: this.scale.params, offset: 0, size: 32 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.scale.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          if (this.zeroWrite) {
            const bg = this.device.createBindGroup({
              layout: this.zeroWrite.bgl,
              entries: [{ binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this._logicalBytesPerBatchF32 } }],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.zeroWrite.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
            pass.end();
          }
    
          let outF32Buf = srcBuf;
          let outF32Off = srcOff;
    
          if (this.ioExtract) {
            this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, this.ioExtract.viewTotal, 1, 0]));
            if (!this.io.output.clearOutside) {
              if (this._usesStridedOutput) {
                this._copyStridedOutputToContiguous(commandEncoder, {
                  output,
                  outputOffsetBytes,
                  batchStart: b,
                  batchCount: 1,
                  dstBuffer: stageOutputF32Range.buffer,
                  dstOffsetBytes: stageOutputF32Range.offsetBytes,
                });
              } else {
                const outRanges = normalizeToContiguousRanges(output, outBatchOffset, this._outBytesPerBatchRaw);
                if (this.precision === "f16-storage") {
                  if (outRanges.length === 1) {
                    commandEncoder.copyBufferToBuffer(
                      outRanges[0].buffer,
                      outRanges[0].offsetBytes,
                      f16OutRange.buffer,
                      f16OutRange.offsetBytes,
                      this._outBytesPerBatchRaw
                    );
                  } else {
                    this.copier.pack(commandEncoder, outRanges, f16OutRange.buffer, f16OutRange.offsetBytes);
                  }
                  this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this._outViewPerBatch, 0, 0, 0]));
                  const bg = this.device.createBindGroup({
                    layout: this.f16.bgl,
                    entries: [
                      { binding: 0, resource: { buffer: f16OutRange.buffer, offset: f16OutRange.offsetBytes, size: this._outBytesPerBatchBind } },
                      { binding: 1, resource: { buffer: stageOutputF32Range.buffer, offset: stageOutputF32Range.offsetBytes, size: this._outViewPerBatch * 4 } },
                      { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
                    ],
                  });
                  const pass = commandEncoder.beginComputePass();
                  pass.setPipeline(this.f16.toF32);
                  pass.setBindGroup(0, bg);
                  pass.dispatchWorkgroups(Math.ceil(this._outViewPerBatch / this.workgroupSize), 1, 1);
                  pass.end();
                } else {
                  if (outRanges.length === 1) {
                    commandEncoder.copyBufferToBuffer(
                      outRanges[0].buffer,
                      outRanges[0].offsetBytes,
                      stageOutputF32Range.buffer,
                      stageOutputF32Range.offsetBytes,
                      this._outBytesPerBatchRaw
                    );
                  } else {
                    this.copier.pack(commandEncoder, outRanges, stageOutputF32Range.buffer, stageOutputF32Range.offsetBytes);
                  }
                }
              }
            }
    
            const bg = this.device.createBindGroup({
              layout: this.ioExtract.bgl,
              entries: [
                { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this._logicalBytesPerBatchF32 } },
                { binding: 1, resource: { buffer: stageOutputF32Range.buffer, offset: stageOutputF32Range.offsetBytes, size: this._outViewPerBatch * 4 } },
                { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.ioExtract.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.ioExtract.viewTotal / this.workgroupSize), 1, 1);
            pass.end();
            outF32Buf = stageOutputF32Range.buffer;
            outF32Off = stageOutputF32Range.offsetBytes;
          }
    
          if (this._usesStridedOutput) {
            if (isGpuBuffer(output)) {
              const window = this._resolveStridedOutputBatchWindow(output, outputOffsetBytes, b);
              this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([this._outViewPerBatch, 1, window.extraOffsetElements, 0]));
              const bg = this.device.createBindGroup({
                layout: this.stridedOut.bgl,
                entries: [
                  { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this._outViewPerBatch * 4 } },
                  { binding: 1, resource: { buffer: output, offset: window.bindingOffsetBytes, size: window.bindingSizeBytes } },
                  { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.stridedOut.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this._outViewPerBatch / this.workgroupSize), 1, 1);
              pass.end();
            } else {
              this._copyContiguousToStridedOutput(commandEncoder, {
                srcBuffer: outF32Buf,
                srcOffsetBytes: outF32Off,
                output,
                outputOffsetBytes,
                batchStart: b,
                batchCount: 1,
              });
            }
            continue;
          }
    
          if (this.precision === "f16-storage") {
            const outRanges = normalizeToContiguousRanges(output, outBatchOffset, this._outBytesPerBatchRaw);
            this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this._outViewPerBatch, 0, 0, 0]));
            const bg = this.device.createBindGroup({
              layout: this.f16.bgl,
              entries: [
                { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this._outViewPerBatch * 4 } },
                { binding: 1, resource: { buffer: f16OutRange.buffer, offset: f16OutRange.offsetBytes, size: this._outBytesPerBatchBind } },
                { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.f16.toF16);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this._outViewPerBatch / this.workgroupSize), 1, 1);
            pass.end();
    
            if (outRanges.length === 1) {
              commandEncoder.copyBufferToBuffer(
                f16OutRange.buffer,
                f16OutRange.offsetBytes,
                outRanges[0].buffer,
                outRanges[0].offsetBytes,
                this._outBytesPerBatchRaw
              );
            } else {
              this.copier.unpack(commandEncoder, f16OutRange.buffer, f16OutRange.offsetBytes, outRanges);
            }
            continue;
          }
    
          const outRanges = normalizeToContiguousRanges(output, outBatchOffset, this._outBytesPerBatchRaw);
          if (outRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(outF32Buf, outF32Off, outRanges[0].buffer, outRanges[0].offsetBytes, this._outBytesPerBatchRaw);
          } else {
            this.copier.unpack(commandEncoder, outF32Buf, outF32Off, outRanges);
          }
        }
      }
    
      _arenaSlicesAreContiguous(arena, largeBatchMode) {
        const checks = [
          [this.stageOffset, this.stageBytes],
          [this.dataAOffset, this.dataBytes],
          [this.dataBOffset, this.dataBytes],
          [this.workOffset, this.workBytesMax],
          [this.fftScratchOffset, this.fftScratchBytes],
          [this.f16OutOffset, this.precision === "f16-storage" ? (largeBatchMode ? this._outBytesPerBatchBind : this.f16OutBytes) : 0],
        ];
        for (const [off, bytes] of checks) {
          if (!bytes) continue;
          if (normalizeToContiguousRanges(arena, off, bytes).length !== 1) return false;
        }
        return true;
      }
    
      exec(commandEncoder, execOpts) {
        if (this._destroyed) throw new Error("plan destroyed");
        const { input, output, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
        if (!input || !output) throw new Error("exec requires input and output");
    
        let arena = temp ?? this._arena;
        if (temp && (buffersAlias(temp, input) || buffersAlias(temp, output))) {
          arena = this._arena ?? null;
        }
        if (temp && arena === temp && getBufferByteLength(arena) < this.workspaceBytes) {
          arena = this._arena ?? null;
        }
        if (temp && arena === temp && !this._arenaSlicesAreContiguous(arena, this._largeBatchChunkMode)) {
          arena = this._arena ?? null;
        }
        const workspaceViews = this._resolveWorkspaceViews(arena);
        if (this._largeBatchChunkMode) {
          this._execLargeBatchChunk(commandEncoder, {
            input,
            output,
            inputOffsetBytes,
            outputOffsetBytes,
            workspaceViews,
          });
          return;
        }
    
        const { stage, dataA, dataB, work, fftScratch, f16OutScratch } = workspaceViews;
    
        const dataARange = normalizeToContiguousRanges(dataA, 0, this.dataBytes)[0];
        const dataBRange = normalizeToContiguousRanges(dataB, 0, this.dataBytes)[0];
    
        // Load physical input -> dataA (f32), with optional ioView embed.
        if (this.precision === "f16-storage") {
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
          let srcBuf = inRanges[0].buffer;
          let srcOff = inRanges[0].offsetBytes;
          if (inRanges.length > 1) {
            const scratchF16 = normalizeToContiguousRanges(stage, this.stageF16Offset, this.inBytes)[0];
            this.copier.pack(commandEncoder, inRanges, scratchF16.buffer, scratchF16.offsetBytes);
            srcBuf = scratchF16.buffer;
            srcOff = scratchF16.offsetBytes;
          }
    
          const dstF32 = this.ioEmbed
            ? normalizeToContiguousRanges(stage, 0, this.inViewTotal * 4)[0]
            : dataARange;
    
          this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this.inViewTotal, 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.f16.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.inBytes } },
              { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.inViewTotal * 4 } },
              { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16.toF32);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.inViewTotal / this.workgroupSize), 1, 1);
          pass.end();
        } else if (this._usesStridedInput) {
          const dstF32 = this.ioEmbed ? normalizeToContiguousRanges(stage, 0, this.inViewTotal * 4)[0] : dataARange;
          if (isGpuBuffer(input)) {
            const { extraOffsetElements, neededBytes } = this._resolveStridedInputBinding(input, inputOffsetBytes);
            this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([this._inViewPerBatch, this.batch, extraOffsetElements, 0]));
            const bg = this.device.createBindGroup({
              layout: this.stridedIn.bgl,
              entries: [
                { binding: 0, resource: { buffer: input, offset: 0, size: neededBytes } },
                { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.inViewTotal * 4 } },
                { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.stridedIn.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.inViewTotal / this.workgroupSize), 1, 1);
            pass.end();
          } else {
            this._copyStridedInputToContiguous(commandEncoder, {
              input,
              inputOffsetBytes,
              batchStart: 0,
              batchCount: this.batch,
              dstBuffer: dstF32.buffer,
              dstOffsetBytes: dstF32.offsetBytes,
            });
          }
        } else {
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
          const dstF32 = this.ioEmbed ? normalizeToContiguousRanges(stage, 0, this.inBytes)[0] : dataARange;
          if (inRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(inRanges[0].buffer, inRanges[0].offsetBytes, dstF32.buffer, dstF32.offsetBytes, this.inBytes);
          } else {
            this.copier.pack(commandEncoder, inRanges, dstF32.buffer, dstF32.offsetBytes);
          }
        }
    
        if (this.ioEmbed) {
          this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, this.ioEmbed.viewTotal, this.batch, 0]));
          const src = normalizeToContiguousRanges(stage, 0, this.ioEmbed.viewTotal * this.batch * 4)[0];
          const dst = dataARange;
          const bg = this.device.createBindGroup({
            layout: this.ioEmbed.bgl,
            entries: [
              { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.ioEmbed.viewTotal * this.batch * 4 } },
              { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.dataBytes } },
              { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioEmbed.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        if (this.zeroRead) {
          const bg = this.device.createBindGroup({
            layout: this.zeroRead.bgl,
            entries: [{ binding: 0, resource: { buffer: dataARange.buffer, offset: dataARange.offsetBytes, size: this.dataBytes } }],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.zeroRead.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        // Core ND separable DCT passes.
        let srcBuf = dataARange.buffer;
        let srcOff = dataARange.offsetBytes;
        let dstBuf = dataBRange.buffer;
        let dstOff = dataBRange.offsetBytes;
    
        for (const ax of this.axes) {
          const workRange = normalizeToContiguousRanges(work, 0, ax.workBytes)[0];
          if (!ax.fftRunsPerBatch) {
            // build -> work
            {
              const bg = this.device.createBindGroup({
                layout: ax.build.bgl,
                entries: [
                  { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } },
                  { binding: 1, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes, size: ax.workBytes } },
                  { binding: 2, resource: { buffer: ax.build.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(ax.build.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(ax.workElems / this.workgroupSize), 1, 1);
              pass.end();
            }
    
            // FFT on work (1D, batch=lines)
            const fftExec = {
              input: workRange.buffer,
              inputOffsetBytes: workRange.offsetBytes,
              temp: fftScratch,
            };
            if (ax.fftNeedsBatch) fftExec.batch = ax.lines;
            ax.fft.exec(commandEncoder, fftExec);
    
            // post -> dst
            {
              const bg = this.device.createBindGroup({
                layout: ax.post.bgl,
                entries: [
                  { binding: 0, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes, size: ax.workBytes } },
                  { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: this.dataBytes } },
                  { binding: 2, resource: { buffer: ax.post.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(ax.post.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
              pass.end();
            }
          } else {
            // Non-factorable M fallback uses a per-logical-batch C2CPlan. Run batch slices explicitly.
            this.device.queue.writeBuffer(ax.build.params, 0, new Uint32Array([ax.workElemsPerBatch, 0, 0, 0]));
            this.device.queue.writeBuffer(ax.post.params, 0, new Uint32Array([this.logicalTotal, 0, 0, 0]));
            for (let b = 0; b < this.batch; b++) {
              const batchDataOffset = b * this._logicalBytesPerBatchF32;
              const batchWorkOffset = b * ax.workBytesPerBatch;
    
              {
                const bg = this.device.createBindGroup({
                  layout: ax.build.bgl,
                  entries: [
                    { binding: 0, resource: { buffer: srcBuf, offset: srcOff + batchDataOffset, size: this._logicalBytesPerBatchF32 } },
                    { binding: 1, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes + batchWorkOffset, size: ax.workBytesPerBatch } },
                    { binding: 2, resource: { buffer: ax.build.params, offset: 0, size: 16 } },
                  ],
                });
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(ax.build.pipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(ax.workElemsPerBatch / this.workgroupSize), 1, 1);
                pass.end();
              }
    
              ax.fft.exec(commandEncoder, {
                input: workRange.buffer,
                inputOffsetBytes: workRange.offsetBytes + batchWorkOffset,
                temp: fftScratch,
              });
    
              {
                const bg = this.device.createBindGroup({
                  layout: ax.post.bgl,
                  entries: [
                    { binding: 0, resource: { buffer: workRange.buffer, offset: workRange.offsetBytes + batchWorkOffset, size: ax.workBytesPerBatch } },
                    { binding: 1, resource: { buffer: dstBuf, offset: dstOff + batchDataOffset, size: this._logicalBytesPerBatchF32 } },
                    { binding: 2, resource: { buffer: ax.post.params, offset: 0, size: 16 } },
                  ],
                });
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(ax.post.pipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(this.logicalTotal / this.workgroupSize), 1, 1);
                pass.end();
              }
            }
          }
    
          // swap
          [srcBuf, dstBuf] = [dstBuf, srcBuf];
          [srcOff, dstOff] = [dstOff, srcOff];
        }
    
        // Normalize once after final axis
        const scale = normalizeScaleFactor({ normalize: this.normalize, direction: this.direction, nTotal: this.logicalTotal });
        if (scale !== 1.0) {
          this.device.queue.writeBuffer(this.scale.params, 0, new Uint32Array([this.totalReal, 0, 0, 0]));
          this.device.queue.writeBuffer(this.scale.params, 16, new Float32Array([scale, 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.scale.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } },
              { binding: 1, resource: { buffer: this.scale.params, offset: 0, size: 32 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.scale.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        if (this.zeroWrite) {
          const bg = this.device.createBindGroup({
            layout: this.zeroWrite.bgl,
            entries: [{ binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } }],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.zeroWrite.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        // Optional output mapping (logical -> view)
        let outF32Buf = srcBuf;
        let outF32Off = srcOff;
        let outF32Bytes = this.dataBytes;
        if (this.ioExtract) {
          const viewTotal = this.ioExtract.viewTotal;
          outF32Bytes = viewTotal * this.batch * 4;
          ensureWithinBindingLimit(this.device, outF32Bytes, "DCT ioView.output");
          this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.logicalTotal, viewTotal, this.batch, 0]));
    
          // f32 output can be written directly when contiguous (preserves clearOutside=false semantics).
          if (this.precision === "f32" && !this._usesStridedOutput) {
            const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
            if (outRanges.length === 1) {
              const bg = this.device.createBindGroup({
                layout: this.ioExtract.bgl,
                entries: [
                  { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } },
                  { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
                  { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.ioExtract.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
              pass.end();
              return;
            }
          }
    
          const dst = normalizeToContiguousRanges(stage, 0, outF32Bytes)[0];
    
          // For clearOutside=false with staged output, initialize dst from existing output values.
          if (!this.io.output.clearOutside) {
            if (this._usesStridedOutput) {
              this._copyStridedOutputToContiguous(commandEncoder, {
                output,
                outputOffsetBytes,
                batchStart: 0,
                batchCount: this.batch,
                dstBuffer: dst.buffer,
                dstOffsetBytes: dst.offsetBytes,
              });
            } else {
              const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
              if (this.precision === "f16-storage") {
                // output f16 -> dst f32
                let f16SrcBuf = outRanges[0].buffer;
                let f16SrcOff = outRanges[0].offsetBytes;
                if (outRanges.length > 1) {
                  const tmpF16 = normalizeToContiguousRanges(f16OutScratch, 0, this.outBytes)[0];
                  this.copier.pack(commandEncoder, outRanges, tmpF16.buffer, tmpF16.offsetBytes);
                  f16SrcBuf = tmpF16.buffer;
                  f16SrcOff = tmpF16.offsetBytes;
                }
                this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this.outViewTotal, 0, 0, 0]));
                const bg = this.device.createBindGroup({
                  layout: this.f16.bgl,
                  entries: [
                    { binding: 0, resource: { buffer: f16SrcBuf, offset: f16SrcOff, size: this.outBytes } },
                    { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: outF32Bytes } },
                    { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
                  ],
                });
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(this.f16.toF32);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(this.outViewTotal / this.workgroupSize), 1, 1);
                pass.end();
              } else {
                // output f32 -> dst f32
                if (outRanges.length === 1) {
                  commandEncoder.copyBufferToBuffer(outRanges[0].buffer, outRanges[0].offsetBytes, dst.buffer, dst.offsetBytes, this.outBytes);
                } else {
                  this.copier.pack(commandEncoder, outRanges, dst.buffer, dst.offsetBytes);
                }
              }
            }
          }
    
          const bg = this.device.createBindGroup({
            layout: this.ioExtract.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.dataBytes } },
              { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: outF32Bytes } },
              { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioExtract.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
          pass.end();
          outF32Buf = dst.buffer;
          outF32Off = dst.offsetBytes;
        }
    
        if (this._usesStridedOutput) {
          if (isGpuBuffer(output)) {
            const { extraOffsetElements, neededBytes } = this._resolveStridedOutputBinding(output, outputOffsetBytes);
            this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([this._outViewPerBatch, this.batch, extraOffsetElements, 0]));
            const bg = this.device.createBindGroup({
              layout: this.stridedOut.bgl,
              entries: [
                { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this.outViewTotal * 4 } },
                { binding: 1, resource: { buffer: output, offset: 0, size: neededBytes } },
                { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.stridedOut.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.outViewTotal / this.workgroupSize), 1, 1);
            pass.end();
          } else {
            this._copyContiguousToStridedOutput(commandEncoder, {
              srcBuffer: outF32Buf,
              srcOffsetBytes: outF32Off,
              output,
              outputOffsetBytes,
              batchStart: 0,
              batchCount: this.batch,
            });
          }
          return;
        }
    
        if (this.precision === "f16-storage") {
          const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
          this.device.queue.writeBuffer(this.f16.params, 0, new Uint32Array([this.outViewTotal, 0, 0, 0]));
          if (outRanges.length === 1) {
            const bg = this.device.createBindGroup({
              layout: this.f16.bgl,
              entries: [
                { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this.outViewTotal * 4 } },
                { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
                { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.f16.toF16);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.outViewTotal / this.workgroupSize), 1, 1);
            pass.end();
            return;
          }
          const tmp = normalizeToContiguousRanges(f16OutScratch, 0, this.outBytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.f16.bgl,
            entries: [
              { binding: 0, resource: { buffer: outF32Buf, offset: outF32Off, size: this.outViewTotal * 4 } },
              { binding: 1, resource: { buffer: tmp.buffer, offset: tmp.offsetBytes, size: this.outBytes } },
              { binding: 2, resource: { buffer: this.f16.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16.toF16);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.outViewTotal / this.workgroupSize), 1, 1);
          pass.end();
          this.copier.unpack(commandEncoder, tmp.buffer, tmp.offsetBytes, outRanges);
          return;
        }
    
        const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
        if (outRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(outF32Buf, outF32Off, outRanges[0].buffer, outRanges[0].offsetBytes, this.outBytes);
        } else {
          this.copier.unpack(commandEncoder, outF32Buf, outF32Off, outRanges);
        }
      }
    }
    
    
    exports['DctPlan'] = DctPlan;
  });

  __define('src/runtime/plans/fftconv.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { BasePlan } = require('src/runtime/base_plan.js');
    const { createInternalArena, viewFromArena } = require('src/runtime/workspace.js');
    const { normalizeToContiguousRanges } = require('src/runtime/segmented_io.js');
    const { assertOneOf, isPositiveInt, prod, alignBytes, getBufferByteLength, buffersAlias } = require('src/runtime/common.js');
    const { mergeLargeRouteMetadata, resolveLargeRoutingPolicy } = require('src/runtime/large_policy.js');
    const { resolveLayoutSemantics } = require('src/runtime/layout_semantics.js');
    const { normalizeZeroPad } = require('src/runtime/zero_pad.js');
    const { createTensorDescriptor, requiredBytesForBatchRange } = require('src/runtime/tensor_descriptor.js');
    
    const { C2CPlan } = require('src/runtime/plans/c2c.js');
    const { generatePointwiseMulSegmentWGSL } = require('src/kernels/fft_conv.js');
    const { generateExtractComplexWGSL } = require('src/kernels/ioview.js');
    const { BufferView } = require('src/utils/buffer_view.js');
    const { coordsFromLinear: tensorCoordsFromLinear, linearFromCoordsShape: tensorLinearFromCoordsShape, contiguousStrides: tensorContiguousStrides, linearFromCoords: tensorLinearFromCoords } = require('src/runtime/tensor_descriptor.js');
    
    function copyContiguousToRanges(copier, commandEncoder, srcBuffer, srcOffsetBytes, byteLength, outRanges, maxBindBytes = Infinity) {
      if (outRanges.length === 1) {
        if (outRanges[0].buffer !== srcBuffer || outRanges[0].offsetBytes !== srcOffsetBytes) {
          commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, outRanges[0].buffer, outRanges[0].offsetBytes, byteLength);
        }
      } else {
        if (byteLength > maxBindBytes) {
          let src = srcOffsetBytes;
          for (const r of outRanges) {
            commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
            src += r.sizeBytes;
          }
        } else {
          copier.unpack(commandEncoder, srcBuffer, srcOffsetBytes, outRanges);
        }
      }
    }
    
    function copyRangesToRanges(copier, commandEncoder, srcRanges, dstRanges, byteLength, maxBindBytes = Infinity) {
      if (srcRanges.length === 1 && dstRanges.length === 1) {
        if (
          srcRanges[0].buffer !== dstRanges[0].buffer ||
          srcRanges[0].offsetBytes !== dstRanges[0].offsetBytes
        ) {
          commandEncoder.copyBufferToBuffer(
            srcRanges[0].buffer,
            srcRanges[0].offsetBytes,
            dstRanges[0].buffer,
            dstRanges[0].offsetBytes,
            byteLength
          );
        }
        return;
      }
      if (srcRanges.length === 1 && dstRanges.length > 1 && byteLength <= maxBindBytes) {
        copier.unpack(commandEncoder, srcRanges[0].buffer, srcRanges[0].offsetBytes, dstRanges);
        return;
      }
      if (srcRanges.length > 1 && dstRanges.length === 1 && byteLength <= maxBindBytes) {
        copier.pack(commandEncoder, srcRanges, dstRanges[0].buffer, dstRanges[0].offsetBytes);
        return;
      }
    
      // General segmented-to-segmented fallback: piecewise linear copies.
      let si = 0;
      let di = 0;
      let soff = 0;
      let doff = 0;
      let remaining = byteLength;
      while (remaining > 0) {
        const s = srcRanges[si];
        const d = dstRanges[di];
        const sAvail = s.sizeBytes - soff;
        const dAvail = d.sizeBytes - doff;
        const take = Math.min(remaining, sAvail, dAvail);
        commandEncoder.copyBufferToBuffer(
          s.buffer,
          s.offsetBytes + soff,
          d.buffer,
          d.offsetBytes + doff,
          take
        );
        remaining -= take;
        soff += take;
        doff += take;
        if (soff === s.sizeBytes) {
          si += 1;
          soff = 0;
        }
        if (doff === d.sizeBytes) {
          di += 1;
          doff = 0;
        }
      }
    }
    
    function arraysEqual(a, b) {
      if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
      }
      return true;
    }
    
    function hasOwn(obj, key) {
      return !!obj && Object.prototype.hasOwnProperty.call(obj, key);
    }
    
    function parseOptionalPositiveInt(v, name) {
      if (v == null) return null;
      if (!Number.isSafeInteger(v) || v <= 0) throw new Error(`${name} must be a positive integer`);
      return v;
    }
    
    function parseOptionalNonNegativeInt(v, name) {
      if (v == null) return null;
      if (!Number.isSafeInteger(v) || v < 0) throw new Error(`${name} must be a non-negative integer`);
      return v;
    }
    
    function checkedSafeNonNegativeInt(v, name) {
      if (!Number.isSafeInteger(v) || v < 0) throw new Error(`${name} must stay within non-negative safe integer range`);
      return v;
    }
    
    function sideFieldName(side, suffix) {
      if (side === "input") return `input${suffix}`;
      return `output${suffix}`;
    }
    
    function hasExplicitSideLayout(layout, side) {
      return (
        hasOwn(layout, sideFieldName(side, "Strides")) ||
        hasOwn(layout, sideFieldName(side, "OffsetElements")) ||
        hasOwn(layout, sideFieldName(side, "BatchStrideElements")) ||
        hasOwn(layout, "strides") ||
        hasOwn(layout, "offsetElements") ||
        hasOwn(layout, "batchStrideElements")
      );
    }
    
    function normalizeChannelPolicySide({
      desc,
      sidePath,
      defaultChannelStrideElements,
      allowKernelStep = false,
      kernelCount = 1,
    }) {
      if (desc == null) return null;
      if (typeof desc !== "object" || Array.isArray(desc)) throw new Error(`${sidePath} must be an object`);
    
      const channels = parseOptionalPositiveInt(desc.channels ?? null, `${sidePath}.channels`);
      if (channels == null) throw new Error(`${sidePath}.channels is required`);
    
      const channelIndex = parseOptionalNonNegativeInt(desc.channelIndex ?? null, `${sidePath}.channelIndex`) ?? 0;
      if (channelIndex >= channels) {
        throw new Error(`${sidePath}.channelIndex (${channelIndex}) must be < ${sidePath}.channels (${channels})`);
      }
    
      const channelStrideElements =
        parseOptionalPositiveInt(desc.channelStrideElements ?? null, `${sidePath}.channelStrideElements`) ??
        defaultChannelStrideElements;
      if (channelStrideElements < defaultChannelStrideElements) {
        throw new Error(`${sidePath}.channelStrideElements must be >= logical span (${defaultChannelStrideElements})`);
      }
    
      const offsetElements = parseOptionalNonNegativeInt(desc.offsetElements ?? null, `${sidePath}.offsetElements`) ?? 0;
      const defaultBatchStride = checkedSafeNonNegativeInt(
        channels * channelStrideElements,
        `${sidePath}.batchStrideElements`
      );
      const batchStrideElements =
        parseOptionalNonNegativeInt(desc.batchStrideElements ?? null, `${sidePath}.batchStrideElements`) ?? defaultBatchStride;
      if (batchStrideElements < defaultBatchStride) {
        throw new Error(`${sidePath}.batchStrideElements must be >= channels*channelStrideElements (${defaultBatchStride})`);
      }
    
      const kernelStepChannels = allowKernelStep
        ? parseOptionalPositiveInt(desc.kernelStepChannels ?? null, `${sidePath}.kernelStepChannels`) ?? 1
        : 1;
      if (allowKernelStep && kernelCount > 1) {
        const maxChannelIndex = checkedSafeNonNegativeInt(
          channelIndex + (kernelCount - 1) * kernelStepChannels,
          `${sidePath}.kernelStepChannels`
        );
        if (maxChannelIndex >= channels) {
          throw new Error(
            `${sidePath} does not fit kernelCount=${kernelCount}: max channel index ${maxChannelIndex} exceeds channels=${channels} ` +
              `(channelIndex=${channelIndex}, kernelStepChannels=${kernelStepChannels})`
          );
        }
      }
    
      return {
        channels,
        channelIndex,
        channelStrideElements,
        batchStrideElements,
        offsetElements,
        kernelStepChannels,
        layoutDesc: {
          channels,
          channelIndex,
          channelStrideElements,
          batchStrideElements,
          offsetElements,
        },
      };
    }
    
    function resolveFftConvLayoutWithChannelPolicy({
      layout,
      channelPolicy,
      kernelCount,
      inputLogicalTotal,
      outputLogicalTotal,
    }) {
      if (channelPolicy == null) {
        return {
          layout: layout ?? {},
          outputKernelStrideElements: 0,
          usesChannelPolicy: false,
        };
      }
      if (typeof channelPolicy !== "object" || Array.isArray(channelPolicy)) {
        throw new Error("fftConv.channelPolicy must be an object");
      }
    
      const inPolicyPresent = hasOwn(channelPolicy, "input") && channelPolicy.input != null;
      const outPolicyPresent = hasOwn(channelPolicy, "output") && channelPolicy.output != null;
      if (!inPolicyPresent && !outPolicyPresent) {
        throw new Error("fftConv.channelPolicy must provide input and/or output descriptors");
      }
    
      if (layout?.whdcn != null) {
        throw new Error("fftConv.channelPolicy cannot be combined with layout.whdcn");
      }
      if (inPolicyPresent && hasExplicitSideLayout(layout ?? {}, "input")) {
        throw new Error("fftConv.channelPolicy.input cannot be combined with explicit input stride fields");
      }
      if (outPolicyPresent && hasExplicitSideLayout(layout ?? {}, "output")) {
        throw new Error("fftConv.channelPolicy.output cannot be combined with explicit output stride fields");
      }
    
      const inputPolicy = normalizeChannelPolicySide({
        desc: channelPolicy.input ?? null,
        sidePath: "fftConv.channelPolicy.input",
        defaultChannelStrideElements: inputLogicalTotal,
        allowKernelStep: false,
        kernelCount,
      });
      const outputPolicy = normalizeChannelPolicySide({
        desc: channelPolicy.output ?? null,
        sidePath: "fftConv.channelPolicy.output",
        defaultChannelStrideElements: outputLogicalTotal,
        allowKernelStep: true,
        kernelCount,
      });
    
      const outputKernelStrideElements =
        outputPolicy && kernelCount > 1
          ? checkedSafeNonNegativeInt(
              outputPolicy.channelStrideElements * outputPolicy.kernelStepChannels,
              "fftConv.channelPolicy.output.kernelStepChannels"
            )
          : 0;
    
      return {
        layout: {
          ...(layout ?? {}),
          whdcn: {
            ...(inputPolicy ? { input: inputPolicy.layoutDesc } : {}),
            ...(outputPolicy ? { output: outputPolicy.layoutDesc } : {}),
          },
        },
        outputKernelStrideElements,
        usesChannelPolicy: true,
      };
    }
    
    function normalizeFftConvTuning(tuning) {
      if (tuning == null) {
        return {
          pointwiseChunkElements: null,
          extractCopyChunkElements: null,
        };
      }
      if (typeof tuning !== "object" || Array.isArray(tuning)) {
        throw new Error("fftConv.tuning must be an object when provided");
      }
      const pointwiseChunkElements = parseOptionalPositiveInt(
        tuning.pointwiseChunkElements ?? null,
        "fftConv.tuning.pointwiseChunkElements"
      );
      const extractCopyChunkElements = parseOptionalPositiveInt(
        tuning.extractCopyChunkElements ?? null,
        "fftConv.tuning.extractCopyChunkElements"
      );
      return {
        pointwiseChunkElements,
        extractCopyChunkElements,
      };
    }
    
    class FftConvPlan extends BasePlan {
      constructor(device, opts) {
        super(device, opts);
        const {
          shape,
          batch = 1,
          inPlace = false,
          layout = { interleavedComplex: true },
          precision = "f32",
          fftConv = null,
          zeroPad = null,
        } = opts ?? {};
    
        if (!Array.isArray(shape) || shape.length < 1) {
          throw new Error(`fftconv shape must be rank >= 1; got ${JSON.stringify(shape)}`);
        }
        if (!shape.every(isPositiveInt)) throw new Error("fftconv shape must be positive ints");
        if (!Number.isInteger(batch) || batch <= 0) throw new Error(`batch must be positive int; got ${batch}`);
        if (inPlace) throw new Error("fftconv inPlace=true is not supported in current implementation");
        if (layout?.interleavedComplex !== true) throw new Error("fftconv requires layout.interleavedComplex=true");
        if (precision !== "f32") throw new Error('fftconv supports precision:"f32" only in current implementation');
    
        const mode = fftConv?.mode ?? "convolution";
        assertOneOf(mode, ["convolution", "correlation"], "fftConv.mode");
        const boundary = fftConv?.boundary ?? "circular";
        assertOneOf(boundary, ["circular", "linear-full", "linear-same", "linear-valid"], "fftConv.boundary");
        const kernelCount = fftConv?.kernelCount ?? 1;
        if (!Number.isInteger(kernelCount) || kernelCount <= 0) {
          throw new Error(`fftConv.kernelCount must be a positive integer; got ${kernelCount}`);
        }
        const outputLayout = fftConv?.outputLayout ?? "kernel-major";
        assertOneOf(outputLayout, ["kernel-major", "batch-major"], "fftConv.outputLayout");
    
        const rank = shape.length;
        const kernelShape = fftConv?.kernelShape ?? shape;
        if (!Array.isArray(kernelShape) || kernelShape.length !== rank || !kernelShape.every(isPositiveInt)) {
          throw new Error(`fftConv.kernelShape must be an array of ${rank} positive ints`);
        }
        if (boundary === "circular") {
          for (let d = 0; d < rank; d++) {
            if (kernelShape[d] > shape[d]) {
              throw new Error(`fftConv.kernelShape[${d}] must be <= shape[${d}] when fftConv.boundary="circular"`);
            }
          }
        }
    
        const fftShape = boundary === "circular" ? shape.slice() : shape.map((n, d) => n + kernelShape[d] - 1);
        let outputShape;
        let outputOffset;
        if (boundary === "circular") {
          outputShape = shape.slice();
          outputOffset = new Array(rank).fill(0);
        } else if (boundary === "linear-full") {
          outputShape = fftShape.slice();
          outputOffset = new Array(rank).fill(0);
        } else if (boundary === "linear-same") {
          outputShape = shape.slice();
          outputOffset = kernelShape.map((n) => Math.floor((n - 1) / 2));
        } else {
          outputShape = shape.map((n, d) => n - kernelShape[d] + 1);
          for (let d = 0; d < rank; d++) {
            if (outputShape[d] <= 0) {
              throw new Error(`fftConv.boundary="linear-valid" requires kernelShape[${d}] <= shape[${d}]`);
            }
          }
          outputOffset = kernelShape.map((n) => n - 1);
        }
    
        this.rank = rank;
        this.inputShape = shape.slice();
        this.shape = fftShape.slice();
        this.kernelShape = kernelShape.slice();
        this.outputShape = outputShape;
        this.outputOffset = outputOffset;
        this.batch = batch;
        this.mode = mode;
        this.boundary = boundary;
        this.kernelCount = kernelCount | 0;
        this.outputLayout = outputLayout;
        this.zeroPad = normalizeZeroPad(this.rank, this.shape, zeroPad ?? null, "zeroPad");
        this._fftConvTuning = normalizeFftConvTuning(fftConv?.tuning ?? null);
    
        this.inputLogicalTotal = prod(this.inputShape);
        this.logicalTotal = prod(this.shape);
        this.kernelLogicalTotal = prod(this.kernelShape);
        this.outputLogicalTotal = prod(this.outputShape);
    
        this.totalComplex = this.logicalTotal * this.batch;
        this.mainBytes = this.totalComplex * 8;
        this.bytesPerBatch = this.logicalTotal * 8;
        this.kernelBytes = this.logicalTotal * 8;
        this.kernelInputBytes = this.kernelLogicalTotal * 8;
        this.inputBytes = this.inputLogicalTotal * this.batch * 8;
        this.outputBytesPerBatch = this.outputLogicalTotal * 8;
        this.outputBytesPerKernel = this.outputBytesPerBatch * this.batch;
        this.totalOutputComplex = this.outputLogicalTotal * this.batch * this.kernelCount;
        this.totalOutputBytes = this.totalOutputComplex * 8;
        const explicitOutputKernelStrideElements =
          parseOptionalPositiveInt(fftConv?.outputKernelStrideElements ?? null, "fftConv.outputKernelStrideElements") ?? 0;
        const channelPolicyResolved = resolveFftConvLayoutWithChannelPolicy({
          layout,
          channelPolicy: fftConv?.channelPolicy ?? null,
          kernelCount: this.kernelCount,
          inputLogicalTotal: this.inputLogicalTotal,
          outputLogicalTotal: this.outputLogicalTotal,
        });
        const policyOutputKernelStrideElements = channelPolicyResolved.outputKernelStrideElements ?? 0;
        if (
          explicitOutputKernelStrideElements > 0 &&
          policyOutputKernelStrideElements > 0 &&
          explicitOutputKernelStrideElements !== policyOutputKernelStrideElements
        ) {
          throw new Error(
            "fftConv.outputKernelStrideElements conflicts with fftConv.channelPolicy.output kernel step mapping"
          );
        }
        this._usesFftConvChannelPolicy = !!channelPolicyResolved.usesChannelPolicy;
        this._stridedOutputKernelStrideElements =
          explicitOutputKernelStrideElements || policyOutputKernelStrideElements || 0;
        const resolvedLayout = resolveLayoutSemantics({
          layout: channelPolicyResolved.layout,
          rank: this.rank,
          inputShape: this.inputShape,
          outputShape: this.outputShape,
        });
        this._inputStrides = resolvedLayout.inputStrides;
        this._outputStrides = resolvedLayout.outputStrides;
        this._inputOffsetElements = resolvedLayout.inputOffsetElements;
        this._outputOffsetElements = resolvedLayout.outputOffsetElements;
        this._inputBatchStrideElements = resolvedLayout.inputBatchStrideElements;
        this._outputBatchStrideElements = resolvedLayout.outputBatchStrideElements;
        this._usesStridedInput = resolvedLayout.usesStridedInput;
        this._usesStridedOutput = resolvedLayout.usesStridedOutput;
        this._usesWhdcnInput = resolvedLayout.usesWhdcnInput;
        this._usesWhdcnOutput = resolvedLayout.usesWhdcnOutput;
        this._inputTensorDesc = this._usesStridedInput
          ? createTensorDescriptor({
              shape: this.inputShape,
              strides: this._inputStrides,
              offsetElements: this._inputOffsetElements,
              batchStrideElements: this._inputBatchStrideElements,
              name: "fftconv.input",
            })
          : null;
        this._outputTensorDesc = this._usesStridedOutput
          ? createTensorDescriptor({
              shape: this.outputShape,
              strides: this._outputStrides,
              offsetElements: this._outputOffsetElements,
              batchStrideElements: this._outputBatchStrideElements,
              name: "fftconv.output",
            })
          : null;
    
        const largePolicy = resolveLargeRoutingPolicy({
          device,
          tuning: opts?.tuning ?? null,
          requiredBindingBytes: [this.mainBytes, this.kernelBytes, this.inputBytes, this.totalOutputBytes],
          lineBytes: this.shape.map((n) => n * 8),
          precision: "f32",
        });
        this._maxBindBytes = largePolicy.maxBindBytes;
        this._maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
        this._largeMode = largePolicy.needsLargeMode;
        this._largeRouteMode = largePolicy.routeMode;
        this._largeRouteReasons = largePolicy.reasonCodes;
        this._largeRouteAttempts = largePolicy.attemptedRoutes;
        const pointwiseMaxElems = Math.max(1, Math.floor(this._maxBindBytes / 8));
        if (
          this._fftConvTuning.pointwiseChunkElements != null &&
          this._fftConvTuning.pointwiseChunkElements > pointwiseMaxElems
        ) {
          throw new Error(
            `fftConv.tuning.pointwiseChunkElements=${this._fftConvTuning.pointwiseChunkElements} exceeds max supported ${pointwiseMaxElems} ` +
              `(maxStorageBufferBindingSize=${this._maxBindBytes})`
          );
        }
        this._pointwiseChunkElems = this._fftConvTuning.pointwiseChunkElements ?? pointwiseMaxElems;
        this._extractCopyChunkElems = this._fftConvTuning.extractCopyChunkElements ?? 1;
        this._batchSlicedExecution = false;
        if (this.mainBytes > this._maxBufferSize) {
          if (this.bytesPerBatch > this._maxBufferSize) {
            throw new Error(
              `fftconv requires ${this.mainBytes} bytes for batch=${this.batch}, and one batch requires ${this.bytesPerBatch} bytes > ` +
                `device.limits.maxBufferSize=${this._maxBufferSize}. ` +
                `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
            );
          }
          this._batchSlicedExecution = true;
        }
        if (this.kernelBytes > this._maxBufferSize) {
          throw new Error(
            `fftconv kernel workspace requires ${this.kernelBytes} bytes, exceeding device.limits.maxBufferSize=${this._maxBufferSize}. ` +
              `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
          );
        }
        if (!this._usesStridedOutput && this._stridedOutputKernelStrideElements > 0) {
          throw new Error("fftConv.outputKernelStrideElements requires strided/whdcn output layout");
        }
        if (this._usesStridedOutput && this.kernelCount > 1 && this._stridedOutputKernelStrideElements <= 0) {
          throw new Error(
            "fftconv multi-kernel strided/whdcn output requires fftConv.channelPolicy.output (with kernelStepChannels) " +
              "or fftConv.outputKernelStrideElements"
          );
        }
    
        this._needsInputEmbed = !arraysEqual(this.inputShape, this.shape);
        this._needsKernelEmbed = !arraysEqual(this.kernelShape, this.shape);
        this._needsOutputExtract =
          !arraysEqual(this.outputShape, this.shape) || this.outputOffset.some((x) => x !== 0);
    
        this.kernelBuffer = device.createBuffer({
          size: this.kernelBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._kernelEmbedInputBuffer = this._needsKernelEmbed
          ? device.createBuffer({
              size: this.kernelInputBytes,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            })
          : null;
        this._kernelUploadBuffer = null;
        this._kernelUploadBytes = 0;
        this._retiredKernelUploadBuffers = [];
    
        const fftBatch = this._batchSlicedExecution ? 1 : this.batch;
        this.fftData = new C2CPlan(device, {
          shape: this.shape,
          direction: "forward",
          batch: fftBatch,
          inPlace: !this._needsInputEmbed,
          normalize: "none",
          layout: { interleavedComplex: true },
          precision: "f32",
          ioView: this._needsInputEmbed ? { input: { shape: this.inputShape, offset: new Array(this.rank).fill(0) } } : { input: null, output: null },
          zeroPad: { read: this.zeroPad.read, write: null },
          tuning: opts?.tuning ?? null,
        });
        this.fftKernel = new C2CPlan(device, {
          shape: this.shape,
          direction: "forward",
          batch: 1,
          inPlace: !this._needsKernelEmbed,
          normalize: "none",
          layout: { interleavedComplex: true },
          precision: "f32",
          ioView: this._needsKernelEmbed ? { input: { shape: this.kernelShape, offset: new Array(this.rank).fill(0) } } : { input: null, output: null },
          tuning: opts?.tuning ?? null,
        });
        this.ifftData = new C2CPlan(device, {
          shape: this.shape,
          direction: "inverse",
          batch: fftBatch,
          inPlace: true,
          normalize: "backward",
          layout: { interleavedComplex: true },
          precision: "f32",
          ioView: { input: null, output: null },
          zeroPad: { read: null, write: this.zeroPad.write },
          tuning: opts?.tuning ?? null,
        });
        const mergedRoute = mergeLargeRouteMetadata([
          {
            routeMode: this._largeRouteMode,
            reasonCodes: this._largeRouteReasons,
            attemptedRoutes: this._largeRouteAttempts,
          },
          {
            routeMode: this.fftData?._largeRouteMode,
            reasonCodes: this.fftData?._largeRouteReasons,
            attemptedRoutes: this.fftData?._largeRouteAttempts,
          },
          {
            routeMode: this.fftKernel?._largeRouteMode,
            reasonCodes: this.fftKernel?._largeRouteReasons,
            attemptedRoutes: this.fftKernel?._largeRouteAttempts,
          },
          {
            routeMode: this.ifftData?._largeRouteMode,
            reasonCodes: this.ifftData?._largeRouteReasons,
            attemptedRoutes: this.ifftData?._largeRouteAttempts,
          },
        ]);
        this._largeRouteMode = mergedRoute.routeMode;
        this._largeRouteReasons = mergedRoute.reasonCodes;
        this._largeRouteAttempts = mergedRoute.attemptedRoutes;
        this.zeroRead = this.fftData?.zeroRead ?? null;
        this.zeroWrite = this.ifftData?.zeroWrite ?? null;
    
        this.pointwise = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generatePointwiseMulSegmentWGSL({
            correlate: this.mode === "correlation",
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          return { bgl, pl, pipeline };
        })();
        this._pointwiseParamsBuffer = null;
        this._pointwiseParamsBytes = 0;
        this._pointwiseParamStride = alignBytes(16, device.limits?.minUniformBufferOffsetAlignment ?? 256);
        this._retiredPointwiseParamsBuffers = [];
    
        this.outputExtract = null;
        if (this._needsOutputExtract) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateExtractComplexWGSL({
            rank: this.rank,
            logicalDims: this.shape,
            viewDims: this.outputShape,
            offset: this.outputOffset,
            clearOutside: true,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.outputExtract = { bgl, pl, pipeline, params };
        }
    
        const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        this._dataBytesPerRun = this._batchSlicedExecution ? this.bytesPerBatch : this.mainBytes;
        this._outputBytesPerRun = this._batchSlicedExecution ? this.outputBytesPerBatch : this.outputBytesPerKernel;
        this._kernelEmbedScratchBytes = this._needsKernelEmbed ? this.kernelInputBytes : 0;
        this._outputExtractScratchBytes = this._needsOutputExtract ? this.outputBytesPerBatch : 0;
        this._subplanScratchBytes = Math.max(
          this.fftData.getWorkspaceSizeBytes(),
          this.fftKernel.getWorkspaceSizeBytes(),
          this.ifftData.getWorkspaceSizeBytes(),
          this._kernelEmbedScratchBytes,
          this._outputExtractScratchBytes
        );
        this._stridedInputStageBytesPerRun = this._usesStridedInput
          ? (this._batchSlicedExecution ? this.inputLogicalTotal * 8 : this.inputBytes)
          : 0;
        this._stridedOutputStageBytesPerRun =
          this._usesStridedOutput && this._needsOutputExtract ? this._outputBytesPerRun : 0;
        let scratchOff = 0;
        this._subplanScratchOffset = 0;
        scratchOff += this._subplanScratchBytes;
        if (scratchOff) scratchOff = alignBytes(scratchOff, storageAlign);
        this._stridedInputStageOffset = 0;
        if (this._stridedInputStageBytesPerRun) {
          this._stridedInputStageOffset = scratchOff;
          scratchOff += this._stridedInputStageBytesPerRun;
          scratchOff = alignBytes(scratchOff, storageAlign);
        }
        this._stridedOutputStageOffset = 0;
        if (this._stridedOutputStageBytesPerRun) {
          this._stridedOutputStageOffset = scratchOff;
          scratchOff += this._stridedOutputStageBytesPerRun;
          scratchOff = alignBytes(scratchOff, storageAlign);
        }
        this.scratchBytes = scratchOff;
    
        let off = 0;
        this.dataOffset = 0;
        off += this._dataBytesPerRun;
        off = alignBytes(off, storageAlign);
        this.scratchOffset = off;
        off += this.scratchBytes;
        this.workspaceBytes = off;
        this._splitWorkspace = null;
        const requireDisjointWorkspaceBuffers =
          !!this.fftData?._largeBatchChunkMode ||
          !!this.fftKernel?._largeBatchChunkMode ||
          !!this.ifftData?._largeBatchChunkMode ||
          (this._usesStridedOutput && this._needsOutputExtract);
        if (!requireDisjointWorkspaceBuffers && this.workspaceBytes <= this._maxBufferSize) {
          this._arena = createInternalArena(device, this.workspaceBytes);
        } else {
          if (this._dataBytesPerRun > this._maxBufferSize) {
            throw new Error(
              `fftconv split workspace cannot allocate data buffer: ${this._dataBytesPerRun} bytes exceeds ` +
                `device.limits.maxBufferSize=${this._maxBufferSize}`
            );
          }
          if (this.scratchBytes > this._maxBufferSize) {
            throw new Error(
              `fftconv split workspace cannot allocate scratch buffer: ${this.scratchBytes} bytes exceeds ` +
                `device.limits.maxBufferSize=${this._maxBufferSize}`
            );
          }
          this._arena = null;
          this._splitWorkspace = {
            data: createInternalArena(device, this._dataBytesPerRun),
            scratch: this.scratchBytes ? createInternalArena(device, this.scratchBytes) : null,
          };
        }
      }
    
      getWorkspaceSizeBytes() {
        return this.workspaceBytes;
      }
    
      _ensureKernelUploadBuffer(bytes) {
        if (this._kernelUploadBuffer && this._kernelUploadBytes >= bytes) return;
        if (bytes > this._maxBufferSize) {
          throw new Error(
            `fftconv kernel upload staging requires ${bytes} bytes, exceeding device.limits.maxBufferSize=${this._maxBufferSize}`
          );
        }
        if (this._kernelUploadBuffer) this._retiredKernelUploadBuffers.push(this._kernelUploadBuffer);
        this._kernelUploadBuffer = this.device.createBuffer({
          size: bytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._kernelUploadBytes = bytes;
      }
    
      _ensurePointwiseParamsBuffer(bytes) {
        if (this._pointwiseParamsBuffer && this._pointwiseParamsBytes >= bytes) return this._pointwiseParamsBuffer;
        if (bytes > this._maxBufferSize) {
          throw new Error(
            `fftconv pointwise chunk params require ${bytes} bytes, exceeding device.limits.maxBufferSize=${this._maxBufferSize}`
          );
        }
        if (this._pointwiseParamsBuffer) this._retiredPointwiseParamsBuffers.push(this._pointwiseParamsBuffer);
        this._pointwiseParamsBuffer = this.device.createBuffer({
          size: bytes,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._pointwiseParamsBytes = bytes;
        return this._pointwiseParamsBuffer;
      }
    
      _resolveWorkspaceViews(temp) {
        const arena = temp ?? this._arena;
        if (arena) {
          if (getBufferByteLength(arena) < this.workspaceBytes) {
            throw new Error(`temp too small: need ${this.workspaceBytes} bytes`);
          }
          return {
            dataView: viewFromArena(arena, this.dataOffset, this._dataBytesPerRun),
            scratchView: this.scratchBytes ? viewFromArena(arena, this.scratchOffset, this.scratchBytes) : null,
          };
        }
        if (this._splitWorkspace) {
          return {
            dataView: viewFromArena(this._splitWorkspace.data, 0, this._dataBytesPerRun),
            scratchView: this.scratchBytes ? viewFromArena(this._splitWorkspace.scratch, 0, this.scratchBytes) : null,
          };
        }
        throw new Error("No workspace buffer");
      }
    
      _sliceView(view, offsetBytes, lengthBytes) {
        if (!lengthBytes) return null;
        const ranges = normalizeToContiguousRanges(view, offsetBytes, lengthBytes);
        return new BufferView({
          segments: ranges.map((r) => ({
            buffer: r.buffer,
            offsetBytes: r.offsetBytes,
            sizeBytes: r.sizeBytes,
          })),
          logicalByteOffset: 0,
          lengthBytes,
        });
      }
    
      _normalizeCopyView(x) {
        if (x && x.buffer && Number.isInteger(x.offsetBytes) && Number.isInteger(x.sizeBytes)) {
          return {
            segments: [{ buffer: x.buffer, offsetBytes: x.offsetBytes, sizeBytes: x.sizeBytes }],
            logicalByteOffset: 0,
            lengthBytes: x.sizeBytes,
          };
        }
        return x;
      }
    
      _copyAnySpan(commandEncoder, { src, srcOffsetBytes, dst, dstOffsetBytes, bytes }) {
        if (bytes <= 0) return;
        const srcRanges = normalizeToContiguousRanges(this._normalizeCopyView(src), srcOffsetBytes, bytes);
        const dstRanges = normalizeToContiguousRanges(this._normalizeCopyView(dst), dstOffsetBytes, bytes);
        copyRangesToRanges(this.copier, commandEncoder, srcRanges, dstRanges, bytes, this._maxBindBytes);
      }
    
      _shapeStrides(shape) {
        return tensorContiguousStrides(shape);
      }
    
      _linearFromCoordsStrides(coords, strides) {
        return tensorLinearFromCoords(coords, strides);
      }
    
      _requiredStridedInputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
        if (!this._inputTensorDesc) {
          throw new Error("internal error: strided input descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._inputTensorDesc, {
          bytesPerElement: 2 * 4,
          runtimeExtraElements: extraOffsetElements,
          batchStart,
          batchCount,
        });
      }
    
      _requiredStridedOutputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
        if (!this._outputTensorDesc) {
          throw new Error("internal error: strided output descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._outputTensorDesc, {
          bytesPerElement: 2 * 4,
          runtimeExtraElements: extraOffsetElements,
          batchStart,
          batchCount,
        });
      }
    
      _copyStridedInputToContiguous(commandEncoder, { input, inputOffsetBytes, batchStart, batchCount, dst, dstOffsetBytes }) {
        if (inputOffsetBytes % 8 !== 0) {
          throw new Error(`inputOffsetBytes must be a multiple of 8 for complex-strided input; got ${inputOffsetBytes}`);
        }
        const extraOffsetElements = (inputOffsetBytes / 8) | 0;
        const neededBytes = this._requiredStridedInputBytes(extraOffsetElements, batchStart, batchCount);
        const inputBytes = getBufferByteLength(input);
        if (inputBytes < neededBytes) {
          throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
        }
    
        const coords = new Array(this.rank).fill(0);
        for (let b = 0; b < batchCount; b++) {
          const srcBatchBase = this._inputOffsetElements + extraOffsetElements + (batchStart + b) * this._inputBatchStrideElements;
          const dstBase = dstOffsetBytes + b * this.inputLogicalTotal * 8;
          for (let li = 0; li < this.inputLogicalTotal; li++) {
            this._coordsFromLinear(li, this.inputShape, coords);
            let srcElem = srcBatchBase;
            for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src: input,
              srcOffsetBytes: srcElem * 8,
              dst,
              dstOffsetBytes: dstBase + li * 8,
              bytes: 8,
            });
          }
        }
      }
    
      _copyContiguousToStridedOutput(commandEncoder, { src, srcOffsetBytes, batchStart, batchCount, output, outputOffsetBytes, kernelIndex = 0 }) {
        if (outputOffsetBytes % 8 !== 0) {
          throw new Error(`outputOffsetBytes must be a multiple of 8 for complex-strided output; got ${outputOffsetBytes}`);
        }
        if (!Number.isInteger(kernelIndex) || kernelIndex < 0 || kernelIndex >= this.kernelCount) {
          throw new Error(`kernelIndex must be within [0, ${this.kernelCount - 1}]; got ${kernelIndex}`);
        }
        const baseExtraOffsetElements = outputOffsetBytes / 8;
        const kernelExtraElements = checkedSafeNonNegativeInt(
          kernelIndex * this._stridedOutputKernelStrideElements,
          "fftconv.output kernel offset"
        );
        const extraOffsetElements = checkedSafeNonNegativeInt(
          baseExtraOffsetElements + kernelExtraElements,
          "fftconv.output runtime extra offset"
        );
        const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements, batchStart, batchCount);
        const outputBytes = getBufferByteLength(output);
        if (outputBytes < neededBytes) {
          throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
        }
    
        const coords = new Array(this.rank).fill(0);
        for (let b = 0; b < batchCount; b++) {
          const srcBase = srcOffsetBytes + b * this.outputLogicalTotal * 8;
          const dstBatchBase = this._outputOffsetElements + extraOffsetElements + (batchStart + b) * this._outputBatchStrideElements;
          for (let li = 0; li < this.outputLogicalTotal; li++) {
            this._coordsFromLinear(li, this.outputShape, coords);
            let dstElem = dstBatchBase;
            for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src,
              srcOffsetBytes: srcBase + li * 8,
              dst: output,
              dstOffsetBytes: dstElem * 8,
              bytes: 8,
            });
          }
        }
      }
    
      _prepareKernelSource(kernel) {
        const singleKernelComplex = 2 * this.kernelLogicalTotal;
        const packedKernelComplex = singleKernelComplex * this.kernelCount;
    
        if (kernel instanceof Float32Array) {
          if (this.kernelCount === 1) {
            if (kernel.length !== singleKernelComplex && kernel.length !== packedKernelComplex) {
              throw new Error(`kernel Float32Array length must be ${singleKernelComplex}; got ${kernel.length}`);
            }
          } else if (kernel.length !== packedKernelComplex) {
            throw new Error(`kernel Float32Array length must be ${packedKernelComplex} for kernelCount=${this.kernelCount}; got ${kernel.length}`);
          }
          const uploadBytes = this.kernelCount * this.kernelInputBytes;
          const payload = kernel.length === singleKernelComplex ? kernel : kernel.subarray(0, packedKernelComplex);
          this._ensureKernelUploadBuffer(uploadBytes);
          this.device.queue.writeBuffer(this._kernelUploadBuffer, 0, payload);
          return { kind: "packed-upload", buffer: this._kernelUploadBuffer };
        }
    
        if (Array.isArray(kernel)) {
          if (kernel.length !== this.kernelCount) {
            throw new Error(`kernel array length must equal fftConv.kernelCount=${this.kernelCount}; got ${kernel.length}`);
          }
          const allTyped = kernel.every((k) => k instanceof Float32Array);
          if (allTyped) {
            const packed = new Float32Array(packedKernelComplex);
            for (let i = 0; i < this.kernelCount; i++) {
              if (kernel[i].length !== singleKernelComplex) {
                throw new Error(`kernel[${i}] Float32Array length must be ${singleKernelComplex}; got ${kernel[i].length}`);
              }
              packed.set(kernel[i], i * singleKernelComplex);
            }
            const uploadBytes = this.kernelCount * this.kernelInputBytes;
            this._ensureKernelUploadBuffer(uploadBytes);
            this.device.queue.writeBuffer(this._kernelUploadBuffer, 0, packed);
            return { kind: "packed-upload", buffer: this._kernelUploadBuffer };
          }
          if (kernel.some((k) => k instanceof Float32Array)) {
            throw new Error("kernel array items must be all Float32Array or all GPUBuffer/BufferView values");
          }
          return { kind: "array-sources", sources: kernel };
        }
    
        return { kind: "packed-source", source: kernel };
      }
    
      _kernelSourceSpec(preparedKernel, kernelIndex) {
        if (preparedKernel.kind === "packed-upload") {
          return { source: preparedKernel.buffer, offsetBytes: kernelIndex * this.kernelInputBytes };
        }
        if (preparedKernel.kind === "array-sources") {
          return { source: preparedKernel.sources[kernelIndex], offsetBytes: 0 };
        }
        const packedOffset = this.kernelCount > 1 ? kernelIndex * this.kernelInputBytes : 0;
        return { source: preparedKernel.source, offsetBytes: packedOffset };
      }
    
      _locateRangeForLogicalByte(ranges, logicalByteOffset) {
        let acc = 0;
        for (let i = 0; i < ranges.length; i++) {
          const r = ranges[i];
          const end = acc + r.sizeBytes;
          if (logicalByteOffset < end) {
            const offsetInRange = logicalByteOffset - acc;
            return {
              range: r,
              rangeIndex: i,
              rangeStart: acc,
              rangeEnd: end,
              offsetInRange,
              buffer: r.buffer,
              offsetBytes: r.offsetBytes + offsetInRange,
              bytesLeft: end - logicalByteOffset,
            };
          }
          acc = end;
        }
        throw new Error(`logical byte offset ${logicalByteOffset} is out of range`);
      }
    
      _mapLogicalByteToRange(ranges, logicalByteOffset) {
        const loc = this._locateRangeForLogicalByte(ranges, logicalByteOffset);
        return { buffer: loc.buffer, offsetBytes: loc.offsetBytes };
      }
    
      _sourceLogicalIndexForViewLinear(vi, coordsOut) {
        this._coordsFromLinear(vi, this.outputShape, coordsOut);
        for (let d = 0; d < this.rank; d++) coordsOut[d] += this.outputOffset[d];
        return this._linearFromCoords(coordsOut, this.shape);
      }
    
      _copyInputToData(commandEncoder, input, inputOffsetBytes, dataView, byteLength = this.mainBytes) {
        const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, byteLength);
        const dstRanges = normalizeToContiguousRanges(dataView, 0, byteLength);
        copyRangesToRanges(this.copier, commandEncoder, inRanges, dstRanges, byteLength, this._maxBindBytes);
      }
    
      _copyKernelToRange(commandEncoder, source, sourceOffsetBytes, kernelRange) {
        const kernelRanges = normalizeToContiguousRanges(source, sourceOffsetBytes, this.kernelBytes);
        if (kernelRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(kernelRanges[0].buffer, kernelRanges[0].offsetBytes, kernelRange.buffer, kernelRange.offsetBytes, this.kernelBytes);
        } else {
          if (this.kernelBytes > this._maxBindBytes) {
            let dst = kernelRange.offsetBytes;
            for (const r of kernelRanges) {
              commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, kernelRange.buffer, dst, r.sizeBytes);
              dst += r.sizeBytes;
            }
          } else {
            this.copier.pack(commandEncoder, kernelRanges, kernelRange.buffer, kernelRange.offsetBytes);
          }
        }
      }
    
      _runPointwiseChunked(commandEncoder, { dataBuffer, dataOffsetBytes, batchCount, kernelBuffer, kernelOffsetBytes = 0 }) {
        const maxElems = this._pointwiseChunkElems;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxPrefixBytes = Math.max(0, storageAlign - 8);
        const minChunkElems = Math.max(1, Math.floor((this._maxBindBytes - maxPrefixBytes) / 8));
        const chunksPerBatch = Math.ceil(this.logicalTotal / minChunkElems);
        const chunkCount = Math.max(1, batchCount * chunksPerBatch);
        const paramsBytes = chunkCount * this._pointwiseParamStride;
        const paramsBuf = this._ensurePointwiseParamsBuffer(paramsBytes);
    
        let chunkIndex = 0;
        for (let b = 0; b < batchCount; b++) {
          const batchDataBase = dataOffsetBytes + b * this.bytesPerBatch;
          for (let i0 = 0; i0 < this.logicalTotal; ) {
            const dataBaseByte = batchDataBase + i0 * 8;
            const kernelBaseByte = kernelOffsetBytes + i0 * 8;
            const dataBindOffset = Math.floor(dataBaseByte / storageAlign) * storageAlign;
            const kernelBindOffset = Math.floor(kernelBaseByte / storageAlign) * storageAlign;
            const dataBaseElems = (dataBaseByte - dataBindOffset) / 8;
            const kernelBaseElems = (kernelBaseByte - kernelBindOffset) / 8;
            const maxCountData = Math.floor(this._maxBindBytes / 8) - dataBaseElems;
            const maxCountKernel = Math.floor(this._maxBindBytes / 8) - kernelBaseElems;
            const count = Math.min(maxElems, this.logicalTotal - i0, maxCountData, maxCountKernel);
            if (count <= 0) {
              throw new Error(
                `fftconv pointwise chunking could not satisfy aligned bind window under maxStorageBufferBindingSize=${this._maxBindBytes}`
              );
            }
            const dataBindBytes = (dataBaseElems + count) * 8;
            const kernelBindBytes = (kernelBaseElems + count) * 8;
            const paramOff = chunkIndex * this._pointwiseParamStride;
            this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([count, dataBaseElems, kernelBaseElems, 0]));
            const bg = this.device.createBindGroup({
              layout: this.pointwise.bgl,
              entries: [
                { binding: 0, resource: { buffer: dataBuffer, offset: dataBindOffset, size: dataBindBytes } },
                { binding: 1, resource: { buffer: kernelBuffer, offset: kernelBindOffset, size: kernelBindBytes } },
                { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.pointwise.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
            pass.end();
            chunkIndex += 1;
            i0 += count;
          }
        }
      }
    
      _runPointwiseChunkedFromRanges(commandEncoder, { dataRanges, batchCount, kernelBuffer, kernelOffsetBytes = 0 }) {
        const maxElems = this._pointwiseChunkElems;
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxPrefixBytes = Math.max(0, storageAlign - 8);
        const minChunkElems = Math.max(1, Math.floor((this._maxBindBytes - maxPrefixBytes) / 8));
    
        const rangePrefix = new Array(dataRanges.length);
        let totalRangeBytes = 0;
        for (let i = 0; i < dataRanges.length; i++) {
          rangePrefix[i] = totalRangeBytes;
          totalRangeBytes += dataRanges[i].sizeBytes;
        }
        const expectedBytes = batchCount * this.bytesPerBatch;
        if (totalRangeBytes !== expectedBytes) {
          throw new Error(`fftconv internal error: data workspace spans ${totalRangeBytes} bytes, expected ${expectedBytes}`);
        }
    
        let chunkIndex = 0;
        for (let b = 0; b < batchCount; b++) {
          const batchByteBase = b * this.bytesPerBatch;
          let globalByte = batchByteBase;
          let i0 = 0;
          let rangeIndex = 0;
          while (rangeIndex + 1 < dataRanges.length && globalByte >= rangePrefix[rangeIndex] + dataRanges[rangeIndex].sizeBytes) {
            rangeIndex += 1;
          }
          while (i0 < this.logicalTotal) {
            while (rangeIndex + 1 < dataRanges.length && globalByte >= rangePrefix[rangeIndex] + dataRanges[rangeIndex].sizeBytes) {
              rangeIndex += 1;
            }
            const range = dataRanges[rangeIndex];
            const rangeLogicalStart = rangePrefix[rangeIndex];
            const withinRange = globalByte - rangeLogicalStart;
            const bytesLeftInRange = range.sizeBytes - withinRange;
            const elemsLeftInRange = Math.floor(bytesLeftInRange / 8);
            if (elemsLeftInRange < 1) {
              rangeIndex += 1;
              continue;
            }
    
            const dataBaseByte = range.offsetBytes + withinRange;
            const kernelBaseByte = kernelOffsetBytes + i0 * 8;
            const dataBindOffset = Math.floor(dataBaseByte / storageAlign) * storageAlign;
            const kernelBindOffset = Math.floor(kernelBaseByte / storageAlign) * storageAlign;
            const dataBaseElems = (dataBaseByte - dataBindOffset) / 8;
            const kernelBaseElems = (kernelBaseByte - kernelBindOffset) / 8;
            const maxCountData = Math.floor(this._maxBindBytes / 8) - dataBaseElems;
            const maxCountKernel = Math.floor(this._maxBindBytes / 8) - kernelBaseElems;
            const count = Math.min(maxElems, this.logicalTotal - i0, elemsLeftInRange, maxCountData, maxCountKernel);
            if (count <= 0) {
              throw new Error(
                `fftconv segmented pointwise chunking could not satisfy aligned bind window under maxStorageBufferBindingSize=${this._maxBindBytes}`
              );
            }
    
            const dataBindBytes = (dataBaseElems + count) * 8;
            const kernelBindBytes = (kernelBaseElems + count) * 8;
            const paramOff = chunkIndex * this._pointwiseParamStride;
            this._ensurePointwiseParamsBuffer(paramOff + this._pointwiseParamStride);
            this.device.queue.writeBuffer(this._pointwiseParamsBuffer, paramOff, new Uint32Array([count, dataBaseElems, kernelBaseElems, 0]));
            const bg = this.device.createBindGroup({
              layout: this.pointwise.bgl,
              entries: [
                { binding: 0, resource: { buffer: range.buffer, offset: dataBindOffset, size: dataBindBytes } },
                { binding: 1, resource: { buffer: kernelBuffer, offset: kernelBindOffset, size: kernelBindBytes } },
                { binding: 2, resource: { buffer: this._pointwiseParamsBuffer, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.pointwise.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(count / this.workgroupSize), 1, 1);
            pass.end();
    
            chunkIndex += 1;
            i0 += count;
            globalByte += count * 8;
          }
        }
      }
    
      _coordsFromLinear(i, shape, outCoords) {
        tensorCoordsFromLinear(i, shape, outCoords);
      }
    
      _linearFromCoords(coords, shape) {
        return tensorLinearFromCoordsShape(coords, shape);
      }
    
      _extractOutputByCopies(commandEncoder, { srcBuffer, srcOffsetBytes, batchCount, dstBuffer, dstOffsetBytes }) {
        const logicalCoords = new Array(this.rank).fill(0);
        const maxChunkElems = Math.max(1, this._extractCopyChunkElems | 0);
        for (let b = 0; b < batchCount; b++) {
          const srcBase = srcOffsetBytes + b * this.bytesPerBatch;
          const dstBase = dstOffsetBytes + b * this.outputBytesPerBatch;
          for (let vi = 0; vi < this.outputLogicalTotal; ) {
            const srcIndex = this._sourceLogicalIndexForViewLinear(vi, logicalCoords);
            let run = Math.min(maxChunkElems, this.outputLogicalTotal - vi);
            let prevSrcIndex = srcIndex;
            for (let r = 1; r < run; r++) {
              const nextSrcIndex = this._sourceLogicalIndexForViewLinear(vi + r, logicalCoords);
              if (nextSrcIndex !== prevSrcIndex + 1) {
                run = r;
                break;
              }
              prevSrcIndex = nextSrcIndex;
            }
            commandEncoder.copyBufferToBuffer(
              srcBuffer,
              srcBase + srcIndex * 8,
              dstBuffer,
              dstBase + vi * 8,
              run * 8
            );
            vi += run;
          }
        }
      }
    
      _extractOutputByCopiesToRanges(commandEncoder, { srcRanges, batchCount, outRanges }) {
        const logicalCoords = new Array(this.rank).fill(0);
        const maxChunkElems = Math.max(1, this._extractCopyChunkElems | 0);
        let rangeIndex = 0;
        let rangeStart = 0;
        let rangeEnd = outRanges.length ? outRanges[0].sizeBytes : 0;
        for (let b = 0; b < batchCount; b++) {
          const srcBase = b * this.bytesPerBatch;
          for (let vi = 0; vi < this.outputLogicalTotal; ) {
            const outElem = b * this.outputLogicalTotal + vi;
            const outByte = outElem * 8;
            while (rangeIndex < outRanges.length - 1 && outByte >= rangeEnd) {
              rangeStart = rangeEnd;
              rangeIndex += 1;
              rangeEnd += outRanges[rangeIndex].sizeBytes;
            }
            const r = outRanges[rangeIndex];
            const dstByte = r.offsetBytes + (outByte - rangeStart);
    
            const srcIndex = this._sourceLogicalIndexForViewLinear(vi, logicalCoords);
            const srcLogicalByte = srcBase + srcIndex * 8;
            const srcLoc = this._locateRangeForLogicalByte(srcRanges, srcLogicalByte);
    
            const maxDstElemsInRange = Math.max(1, Math.floor((rangeEnd - outByte) / 8));
            const maxSrcElemsInRange = Math.max(1, Math.floor(srcLoc.bytesLeft / 8));
            let run = Math.min(
              maxChunkElems,
              this.outputLogicalTotal - vi,
              maxDstElemsInRange,
              maxSrcElemsInRange
            );
            let prevSrcIndex = srcIndex;
            for (let rr = 1; rr < run; rr++) {
              const nextSrcIndex = this._sourceLogicalIndexForViewLinear(vi + rr, logicalCoords);
              if (nextSrcIndex !== prevSrcIndex + 1) {
                run = rr;
                break;
              }
              prevSrcIndex = nextSrcIndex;
            }
    
            commandEncoder.copyBufferToBuffer(
              srcLoc.buffer,
              srcLoc.offsetBytes,
              r.buffer,
              dstByte,
              run * 8
            );
            vi += run;
          }
        }
      }
    
      _runOutputExtract(commandEncoder, srcBuffer, srcOffsetBytes, batchCount, dstBuffer, dstOffsetBytes, dstBytes) {
        this.device.queue.writeBuffer(
          this.outputExtract.params,
          0,
          new Uint32Array([this.logicalTotal, this.outputLogicalTotal, batchCount, 0])
        );
        const bg = this.device.createBindGroup({
          layout: this.outputExtract.bgl,
          entries: [
            {
              binding: 0,
              resource: {
                buffer: srcBuffer,
                offset: srcOffsetBytes,
                size: this.bytesPerBatch * batchCount,
              },
            },
            {
              binding: 1,
              resource: {
                buffer: dstBuffer,
                offset: dstOffsetBytes,
                size: dstBytes,
              },
            },
            { binding: 2, resource: { buffer: this.outputExtract.params, offset: 0, size: 16 } },
          ],
        });
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.outputExtract.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil((this.outputLogicalTotal * batchCount) / this.workgroupSize), 1, 1);
        pass.end();
      }
    
      _writeExtractedOutput(commandEncoder, output, dstOffsetBytes, srcView, srcOffsetBytes, batchCount) {
        const outBytes = this.outputBytesPerBatch * batchCount;
        const outRanges = normalizeToContiguousRanges(output, dstOffsetBytes, outBytes);
        const srcBindBytes = this.bytesPerBatch * batchCount;
        const srcRanges = normalizeToContiguousRanges(srcView, srcOffsetBytes, srcBindBytes);
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const canDirectBindInput =
          srcRanges.length === 1 &&
          srcRanges[0].offsetBytes % storageAlign === 0;
        const canDirectBindOutput = outRanges.length === 1 && outRanges[0].offsetBytes % storageAlign === 0;
        const fitsBindLimits = srcBindBytes <= this._maxBindBytes && outBytes <= this._maxBindBytes;
        const allowDirectExtract = this.kernelCount === 1;
    
        if (allowDirectExtract && canDirectBindInput && canDirectBindOutput && fitsBindLimits) {
          this._runOutputExtract(
            commandEncoder,
            srcRanges[0].buffer,
            srcRanges[0].offsetBytes,
            batchCount,
            outRanges[0].buffer,
            outRanges[0].offsetBytes,
            outBytes
          );
          return;
        }
    
        // Fallback for unaligned/segmented/oversized binds: explicit element copies.
        // This avoids alias hazards between source and staging buffers on real browser GPU drivers.
        this._extractOutputByCopiesToRanges(commandEncoder, {
          srcRanges,
          batchCount,
          outRanges,
        });
      }
    
      _writeSingleKernelOutputContiguous(commandEncoder, output, outputOffsetBytes, dataView, batchCount = this.batch) {
        const runOutputBytes = this.outputBytesPerBatch * batchCount;
        if (!this._needsOutputExtract) {
          const srcRanges = normalizeToContiguousRanges(dataView, 0, runOutputBytes);
          const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, runOutputBytes);
          copyRangesToRanges(
            this.copier,
            commandEncoder,
            srcRanges,
            outRanges,
            runOutputBytes,
            this._maxBindBytes
          );
          return;
        }
        for (let b = 0; b < batchCount; b++) {
          this._writeExtractedOutput(
            commandEncoder,
            output,
            outputOffsetBytes + b * this.outputBytesPerBatch,
            dataView,
            b * this.bytesPerBatch,
            1
          );
        }
      }
    
      _writeKernelOutput(commandEncoder, output, outputOffsetBytes, kernelIndex, dataView, batchStart = 0, batchCount = this.batch) {
        const runOutputBytes = this.outputBytesPerBatch * batchCount;
        if (this.outputLayout === "kernel-major") {
          const dstOffset = outputOffsetBytes + kernelIndex * this.outputBytesPerKernel + batchStart * this.outputBytesPerBatch;
          if (!this._needsOutputExtract) {
            const srcRanges = normalizeToContiguousRanges(dataView, 0, runOutputBytes);
            const outRanges = normalizeToContiguousRanges(output, dstOffset, runOutputBytes);
            copyRangesToRanges(
              this.copier,
              commandEncoder,
              srcRanges,
              outRanges,
              runOutputBytes,
              this._maxBindBytes
            );
          } else {
            for (let b = 0; b < batchCount; b++) {
              this._writeExtractedOutput(
                commandEncoder,
                output,
                dstOffset + b * this.outputBytesPerBatch,
                dataView,
                b * this.bytesPerBatch,
                1
              );
            }
          }
          return;
        }
    
        if (!this._needsOutputExtract) {
          for (let b = 0; b < batchCount; b++) {
            const srcRanges = normalizeToContiguousRanges(dataView, b * this.bytesPerBatch, this.outputBytesPerBatch);
            const slot = (batchStart + b) * this.kernelCount + kernelIndex;
            const dstOffset = outputOffsetBytes + slot * this.outputBytesPerBatch;
            const outRanges = normalizeToContiguousRanges(output, dstOffset, this.outputBytesPerBatch);
            copyRangesToRanges(
              this.copier,
              commandEncoder,
              srcRanges,
              outRanges,
              this.outputBytesPerBatch,
              this._maxBindBytes
            );
          }
          return;
        }
    
        for (let b = 0; b < batchCount; b++) {
          const slot = (batchStart + b) * this.kernelCount + kernelIndex;
          const dstOffset = outputOffsetBytes + slot * this.outputBytesPerBatch;
          this._writeExtractedOutput(
            commandEncoder,
            output,
            dstOffset,
            dataView,
            b * this.bytesPerBatch,
            1
          );
        }
      }
    
      destroy() {
        if (this._destroyed) return;
        this.fftData.destroy();
        this.fftKernel.destroy();
        this.ifftData.destroy();
        this.outputExtract?.params?.destroy?.();
        this._pointwiseParamsBuffer?.destroy?.();
        for (const b of this._retiredPointwiseParamsBuffers) b?.destroy?.();
        this._kernelUploadBuffer?.destroy?.();
        for (const b of this._retiredKernelUploadBuffers) b?.destroy?.();
        this.kernelBuffer?.destroy?.();
        this._kernelEmbedInputBuffer?.destroy?.();
        this._splitWorkspace?.data?.destroy?.();
        this._splitWorkspace?.scratch?.destroy?.();
        this._arena?.destroy?.();
        super.destroy();
      }
    
      exec(commandEncoder, execOpts) {
        if (this._destroyed) throw new Error("plan destroyed");
        const {
          input,
          output,
          kernel,
          temp,
          inputOffsetBytes = 0,
          outputOffsetBytes = 0,
        } = execOpts ?? {};
    
        if (!input || !output) throw new Error("fftconv exec requires input and output");
        if (!kernel) throw new Error("fftconv exec requires kernel");
        if (inputOffsetBytes % 8 !== 0) throw new Error(`inputOffsetBytes must be a multiple of 8; got ${inputOffsetBytes}`);
        if (outputOffsetBytes % 8 !== 0) throw new Error(`outputOffsetBytes must be a multiple of 8; got ${outputOffsetBytes}`);
    
        let workspaceTemp = temp;
        if (
          workspaceTemp &&
          (buffersAlias(workspaceTemp, input) || buffersAlias(workspaceTemp, output) || buffersAlias(workspaceTemp, kernel))
        ) {
          workspaceTemp = null;
        }
        if (workspaceTemp && getBufferByteLength(workspaceTemp) < this.workspaceBytes) {
          workspaceTemp = null;
        }
        const { dataView, scratchView } = this._resolveWorkspaceViews(workspaceTemp);
        const subplanScratchView = this._subplanScratchBytes
          ? this._sliceView(scratchView, this._subplanScratchOffset, this._subplanScratchBytes)
          : null;
        const stridedInputStageView = this._stridedInputStageBytesPerRun
          ? this._sliceView(scratchView, this._stridedInputStageOffset, this._stridedInputStageBytesPerRun)
          : null;
        const stridedOutputStageView = this._stridedOutputStageBytesPerRun
          ? this._sliceView(scratchView, this._stridedOutputStageOffset, this._stridedOutputStageBytesPerRun)
          : null;
        const dataRanges = normalizeToContiguousRanges(dataView, 0, this._dataBytesPerRun);
        const kernelRange = { buffer: this.kernelBuffer, offsetBytes: 0 };
        const preparedKernel = this._prepareKernelSource(kernel);
        const inputBytesPerBatch = this.inputLogicalTotal * 8;
        const fftDataTemp = this.fftData._largeBatchChunkMode ? null : subplanScratchView;
        const ifftDataTemp = this.ifftData._largeBatchChunkMode ? null : subplanScratchView;
        const fftKernelTemp = this.fftKernel._largeBatchChunkMode ? null : subplanScratchView;
    
        if (this._usesStridedOutput) {
          if (outputOffsetBytes % 8 !== 0) {
            throw new Error(`outputOffsetBytes must be a multiple of 8 for complex-strided output; got ${outputOffsetBytes}`);
          }
          const extraOffsetElements = outputOffsetBytes / 8;
          const kernelExtraElements = checkedSafeNonNegativeInt(
            (this.kernelCount - 1) * this._stridedOutputKernelStrideElements,
            "fftconv.output kernel coverage"
          );
          const neededBytes = this._requiredStridedOutputBytes(
            checkedSafeNonNegativeInt(extraOffsetElements + kernelExtraElements, "fftconv.output runtime extra offset"),
            0,
            this.batch
          );
          if (getBufferByteLength(output) < neededBytes) {
            throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${getBufferByteLength(output)}`);
          }
        } else {
          normalizeToContiguousRanges(output, outputOffsetBytes, this.totalOutputBytes);
        }
    
        for (let k = 0; k < this.kernelCount; k++) {
          const kernelSpec = this._kernelSourceSpec(preparedKernel, k);
          if (this._needsKernelEmbed) {
            let kernelEmbedSource = kernelSpec.source;
            let kernelEmbedOffset = kernelSpec.offsetBytes;
            if (this._kernelEmbedInputBuffer) {
              this._copyAnySpan(commandEncoder, {
                src: kernelSpec.source,
                srcOffsetBytes: kernelSpec.offsetBytes,
                dst: this._kernelEmbedInputBuffer,
                dstOffsetBytes: 0,
                bytes: this.kernelInputBytes,
              });
              kernelEmbedSource = this._kernelEmbedInputBuffer;
              kernelEmbedOffset = 0;
            }
            this.fftKernel.exec(commandEncoder, {
              input: kernelEmbedSource,
              output: kernelRange.buffer,
              inputOffsetBytes: kernelEmbedOffset,
              outputOffsetBytes: kernelRange.offsetBytes,
              temp: fftKernelTemp,
            });
          } else {
            this._copyKernelToRange(commandEncoder, kernelSpec.source, kernelSpec.offsetBytes, kernelRange);
            this.fftKernel.exec(commandEncoder, {
              input: kernelRange.buffer,
              inputOffsetBytes: kernelRange.offsetBytes,
              temp: fftKernelTemp,
            });
          }
    
          if (this._batchSlicedExecution) {
            for (let b = 0; b < this.batch; b++) {
              if (this._needsInputEmbed) {
                if (this._usesStridedInput) {
                  this._copyStridedInputToContiguous(commandEncoder, {
                    input,
                    inputOffsetBytes,
                    batchStart: b,
                    batchCount: 1,
                    dst: stridedInputStageView,
                    dstOffsetBytes: 0,
                  });
                  this.fftData.exec(commandEncoder, {
                    input: stridedInputStageView,
                    output: dataView,
                    inputOffsetBytes: 0,
                    outputOffsetBytes: 0,
                    temp: fftDataTemp,
                  });
                } else {
                  this.fftData.exec(commandEncoder, {
                    input,
                    output: dataView,
                    inputOffsetBytes: inputOffsetBytes + b * inputBytesPerBatch,
                    outputOffsetBytes: 0,
                    temp: fftDataTemp,
                  });
                }
              } else {
                if (this._usesStridedInput) {
                  this._copyStridedInputToContiguous(commandEncoder, {
                    input,
                    inputOffsetBytes,
                    batchStart: b,
                    batchCount: 1,
                    dst: dataView,
                    dstOffsetBytes: 0,
                  });
                } else {
                  this._copyInputToData(
                    commandEncoder,
                    input,
                    inputOffsetBytes + b * this.bytesPerBatch,
                    dataView,
                    this.bytesPerBatch
                  );
                }
                this.fftData.exec(commandEncoder, {
                  input: dataView,
                  inputOffsetBytes: 0,
                  temp: fftDataTemp,
                });
              }
    
              if (dataRanges.length === 1) {
                this._runPointwiseChunked(commandEncoder, {
                  dataBuffer: dataRanges[0].buffer,
                  dataOffsetBytes: dataRanges[0].offsetBytes,
                  batchCount: 1,
                  kernelBuffer: kernelRange.buffer,
                  kernelOffsetBytes: kernelRange.offsetBytes,
                });
              } else {
                this._runPointwiseChunkedFromRanges(commandEncoder, {
                  dataRanges,
                  batchCount: 1,
                  kernelBuffer: kernelRange.buffer,
                  kernelOffsetBytes: kernelRange.offsetBytes,
                });
              }
    
              this.ifftData.exec(commandEncoder, {
                input: dataView,
                inputOffsetBytes: 0,
                temp: ifftDataTemp,
              });
    
              if (this._usesStridedOutput) {
                let stridedOutputSource = dataView;
                if (this._needsOutputExtract) {
                  if (!stridedOutputStageView) {
                    throw new Error("internal error: missing strided output staging view for extracted output path");
                  }
                  this._writeSingleKernelOutputContiguous(commandEncoder, stridedOutputStageView, 0, dataView, 1);
                  stridedOutputSource = stridedOutputStageView;
                }
                this._copyContiguousToStridedOutput(commandEncoder, {
                  src: stridedOutputSource,
                  srcOffsetBytes: 0,
                  batchStart: b,
                  batchCount: 1,
                  output,
                  outputOffsetBytes,
                  kernelIndex: k,
                });
              } else {
                this._writeKernelOutput(
                  commandEncoder,
                  output,
                  outputOffsetBytes,
                  k,
                  dataView,
                  b,
                  1
                );
              }
            }
          } else {
            if (this._needsInputEmbed) {
              if (this._usesStridedInput) {
                this._copyStridedInputToContiguous(commandEncoder, {
                  input,
                  inputOffsetBytes,
                  batchStart: 0,
                  batchCount: this.batch,
                  dst: stridedInputStageView,
                  dstOffsetBytes: 0,
                });
                this.fftData.exec(commandEncoder, {
                  input: stridedInputStageView,
                  output: dataView,
                  inputOffsetBytes: 0,
                  outputOffsetBytes: 0,
                  temp: fftDataTemp,
                });
              } else {
                this.fftData.exec(commandEncoder, {
                  input,
                  output: dataView,
                  inputOffsetBytes,
                  outputOffsetBytes: 0,
                  temp: fftDataTemp,
                });
              }
            } else {
              if (this._usesStridedInput) {
                this._copyStridedInputToContiguous(commandEncoder, {
                  input,
                  inputOffsetBytes,
                  batchStart: 0,
                  batchCount: this.batch,
                  dst: dataView,
                  dstOffsetBytes: 0,
                });
              } else {
                this._copyInputToData(commandEncoder, input, inputOffsetBytes, dataView);
              }
              this.fftData.exec(commandEncoder, {
                input: dataView,
                inputOffsetBytes: 0,
                temp: fftDataTemp,
              });
            }
    
            if (dataRanges.length === 1) {
              this._runPointwiseChunked(commandEncoder, {
                dataBuffer: dataRanges[0].buffer,
                dataOffsetBytes: dataRanges[0].offsetBytes,
                batchCount: this.batch,
                kernelBuffer: kernelRange.buffer,
                kernelOffsetBytes: kernelRange.offsetBytes,
              });
            } else {
              this._runPointwiseChunkedFromRanges(commandEncoder, {
                dataRanges,
                batchCount: this.batch,
                kernelBuffer: kernelRange.buffer,
                kernelOffsetBytes: kernelRange.offsetBytes,
              });
            }
    
            this.ifftData.exec(commandEncoder, {
              input: dataView,
              inputOffsetBytes: 0,
              temp: ifftDataTemp,
            });
    
            if (this._usesStridedOutput) {
              let stridedOutputSource = dataView;
              if (this._needsOutputExtract) {
                if (!stridedOutputStageView) {
                  throw new Error("internal error: missing strided output staging view for extracted output path");
                }
                this._writeSingleKernelOutputContiguous(commandEncoder, stridedOutputStageView, 0, dataView, this.batch);
                stridedOutputSource = stridedOutputStageView;
              }
              this._copyContiguousToStridedOutput(commandEncoder, {
                src: stridedOutputSource,
                srcOffsetBytes: 0,
                batchStart: 0,
                batchCount: this.batch,
                output,
                outputOffsetBytes,
                kernelIndex: k,
              });
            } else {
              this._writeKernelOutput(commandEncoder, output, outputOffsetBytes, k, dataView);
            }
          }
        }
      }
    }
    
    
    
    exports['FftConvPlan'] = FftConvPlan;
  });

  __define('src/runtime/plans/r2c.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { BasePlan } = require('src/runtime/base_plan.js');
    const { mergeLargeRouteMetadata, resolveAxisKindsForShape, resolveLargeRoutingPolicy, resolveOutOfCoreAxisWindowPolicy } = require('src/runtime/large_policy.js');
    const { createInternalArena, viewFromArena } = require('src/runtime/workspace.js');
    const { normalizeToContiguousRanges } = require('src/runtime/segmented_io.js');
    const { resolveLayoutSemantics } = require('src/runtime/layout_semantics.js');
    const { assertOneOf, isPositiveInt, prod, alignBytes, ensureWithinBindingLimit, getBufferByteLength, align4Bytes, isGpuBuffer, buffersAlias } = require('src/runtime/common.js');
    const { normalizeIoView } = require('src/runtime/ioview.js');
    const { normalizeZeroPad } = require('src/runtime/zero_pad.js');
    const { contiguousStrides: tensorContiguousStrides, coordsFromLinear: tensorCoordsFromLinear, linearFromCoords: tensorLinearFromCoords, createTensorDescriptor, requiredBytesForBatchRange } = require('src/runtime/tensor_descriptor.js');
    
    const { C2CPlan } = require('src/runtime/plans/c2c.js');
    const { generateRealToComplexWGSL, generatePackR2CWGSL } = require('src/kernels/real_complex.js');
    const { generateZeroOutsideRangeComplexWGSL, generateZeroOutsideRangeRealWGSL } = require('src/kernels/zero_pad.js');
    const { generateEmbedRealWGSL, generateExtractComplexWGSL, generateExtractComplexF32ToF16WGSL } = require('src/kernels/ioview.js');
    const { generateF16ToF32RealWGSL, generateF32ToF16ComplexWGSL } = require('src/kernels/f16_storage.js');
    const { generateGatherRealStridedWGSL } = require('src/kernels/strided_real.js');
    const { generateScatterComplexStridedWGSL } = require('src/kernels/strided_complex.js');
    
    function needsIoMapping(io, logicalShape) {
      if (!io) return false;
      for (let i = 0; i < logicalShape.length; i++) {
        if (io.shape[i] !== logicalShape[i]) return true;
        if (io.offset[i] !== 0) return true;
      }
      return false;
    }
    
    class R2CPlan extends BasePlan {
      constructor(device, opts) {
        super(device, opts);
        const { shape, direction, batch = 1, inPlace = false, normalize = "none", layout = { interleavedComplex: true }, precision = "f32", ioView = null, zeroPad = null } = opts ?? {};
        if (inPlace) throw new Error("r2c inPlace is not supported in current implementation");
        if (direction !== "forward") throw new Error('r2c supports direction:"forward" only');
        if (!Array.isArray(shape) || shape.length < 1) throw new Error("shape must be rank >= 1");
        if (!shape.every(isPositiveInt)) throw new Error("shape must be positive ints");
        assertOneOf(normalize, ["none", "backward", "unitary"], "normalize");
        assertOneOf(precision, ["f32", "f16-storage"], "precision");
        if (layout?.interleavedComplex !== true) throw new Error("r2c output is packed complex interleaved; set layout.interleavedComplex=true");
        if (precision === "f16-storage" && !device.features?.has?.("shader-f16")) throw new Error('precision="f16-storage" requires shader-f16');
    
        this.shape = shape.slice();
        this.rank = shape.length;
        this.batch = batch;
        this.normalize = normalize;
        this.precision = precision;
        const Nx = this.shape[0];
        this.packedShape = [Math.floor(Nx / 2) + 1, ...this.shape.slice(1)];
    
        const iov = ioView ?? {};
        this.ioIn = normalizeIoView(this.rank, this.shape, { input: iov.input }).input;
        this.ioOut = normalizeIoView(this.rank, this.packedShape, { output: iov.output }).output;
        this._needsInputMapping = !!(this.ioIn && needsIoMapping(this.ioIn, this.shape));
        this._needsOutputMapping = !!(this.ioOut && needsIoMapping(this.ioOut, this.packedShape));
        this._inputLayoutShape = this._needsInputMapping ? this.ioIn.shape.slice() : this.shape.slice();
        this._outputLayoutShape = this._needsOutputMapping ? this.ioOut.shape.slice() : this.packedShape.slice();
    
        const resolvedLayout = resolveLayoutSemantics({
          layout,
          rank: this.rank,
          inputShape: this._inputLayoutShape,
          outputShape: this._outputLayoutShape,
        });
        this._inputStrides = resolvedLayout.inputStrides;
        this._outputStrides = resolvedLayout.outputStrides;
        this._inputOffsetElements = resolvedLayout.inputOffsetElements;
        this._outputOffsetElements = resolvedLayout.outputOffsetElements;
        this._inputBatchStrideElements = resolvedLayout.inputBatchStrideElements;
        this._outputBatchStrideElements = resolvedLayout.outputBatchStrideElements;
        this._inputSpanElements = resolvedLayout.inputSpanElements;
        this._outputSpanElements = resolvedLayout.outputSpanElements;
        this._usesStridedInput = resolvedLayout.usesStridedInput;
        this._usesStridedOutput = resolvedLayout.usesStridedOutput;
        this._usesWhdcnInput = resolvedLayout.usesWhdcnInput;
        this._usesWhdcnOutput = resolvedLayout.usesWhdcnOutput;
        this._inputTensorDesc = this._usesStridedInput
          ? createTensorDescriptor({
              name: "r2c.input",
              shape: this._inputLayoutShape,
              strides: this._inputStrides,
              offsetElements: this._inputOffsetElements,
              batchStrideElements: this._inputBatchStrideElements,
            })
          : null;
        this._outputTensorDesc = this._usesStridedOutput
          ? createTensorDescriptor({
              name: "r2c.output",
              shape: this._outputLayoutShape,
              strides: this._outputStrides,
              offsetElements: this._outputOffsetElements,
              batchStrideElements: this._outputBatchStrideElements,
            })
          : null;
        this._inputSpanElements = this._inputTensorDesc?.spanElements ?? 0;
        this._outputSpanElements = this._outputTensorDesc?.spanElements ?? 0;
        if ((this._usesStridedInput || this._usesStridedOutput) && this.precision !== "f32") {
          throw new Error('custom strides currently support precision:"f32" only for r2c');
        }
    
        this.zeroPadRead = normalizeZeroPad(this.rank, this.shape, { read: zeroPad?.read ?? null }, "zeroPad").read;
        this.zeroPadWrite = normalizeZeroPad(this.rank, this.packedShape, { write: zeroPad?.write ?? null }, "zeroPad").write;
        this.logicalTotal = prod(this.shape);
        this.totalReal = this.logicalTotal * this.batch;
    
        this.inViewShape = (this.ioIn?.shape ?? this.shape).slice();
        this.inViewTotalReal = prod(this.inViewShape) * this.batch;
        this.inBytes = precision === "f16-storage" ? align4Bytes(this.inViewTotalReal * 2) : this.inViewTotalReal * 4;
    
        this.totalComplexFull = this.totalReal;
        this.fullBytes = this.totalComplexFull * 8;
    
        this.outTotalComplexLogical = prod(this.packedShape) * this.batch;
        this.packedF32Bytes = this.outTotalComplexLogical * 8;
    
        this.outViewShape = (this.ioOut?.shape ?? this.packedShape).slice();
        this.outViewTotalComplex = prod(this.outViewShape) * this.batch;
        this.outBytes = this.outViewTotalComplex * (precision === "f16-storage" ? 4 : 8);
        this._lineCount = this.batch * prod(this.shape.slice(1));
        this._realLineBytes = this.shape[0] * 4;
        this._complexLineBytes = this.shape[0] * 8;
        this._packedLineBytes = this.packedShape[0] * 8;
        const axisStrategy = resolveAxisKindsForShape({
          shape: this.shape,
          tuning: opts?.tuning ?? null,
        });
        const largePolicy = resolveLargeRoutingPolicy({
          device,
          tuning: opts?.tuning ?? null,
          requiredBindingBytes: [this.fullBytes, this.packedF32Bytes, this.inBytes, this.outBytes],
          lineBytes: [this._realLineBytes, this._complexLineBytes, this._packedLineBytes],
          axisKinds: axisStrategy.axisKinds,
          axisLengths: this.shape,
          allowNonMixedBoundedSlicing: true,
          allowOutOfCore: this.rank >= 2,
          rank: this.rank,
          bytesPerBatch: this.logicalTotal * 8,
          hasStridedIO: this._usesStridedInput || this._usesStridedOutput,
          preferOutOfCoreForStrided: true,
          precision: this.precision,
          requireLargePrecision: "f32",
          requireLargePrecisionError: 'r2c large-shape fallback currently supports precision:"f32" only',
        });
        this._maxBindBytes = largePolicy.maxBindBytes;
        this._largeShapeMode = largePolicy.needsLargeMode;
        this._largeRouteMode = largePolicy.routeMode;
        this._largeRouteReasons = largePolicy.reasonCodes;
        this._largeRouteAttempts = largePolicy.attemptedRoutes;
        this._largeRouteAxisKinds = axisStrategy.axisKinds.slice();
        this._largeRouteAxisSupported = Array.isArray(largePolicy.axisSupported) ? largePolicy.axisSupported.slice() : null;
        if (!this._largeShapeMode) {
          ensureWithinBindingLimit(device, this.fullBytes, "r2c full complex");
          ensureWithinBindingLimit(device, this.packedF32Bytes, "r2c packed logical (f32)");
          ensureWithinBindingLimit(device, this.inBytes, "r2c input");
          ensureWithinBindingLimit(device, this.outBytes, "r2c output");
        }
        this._oversizedLineMode = this._largeShapeMode && largePolicy.oversizedLineMode;
        this._outOfCoreAxisWindowPolicy = null;
        if (this._largeShapeMode) {
          const axisKind0 = this._largeRouteAxisKinds?.[0] ?? "mixed";
          const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
          const tuning = opts?.tuning ?? null;
          this._outOfCoreAxisWindowPolicy = {
            realToComplex: resolveOutOfCoreAxisWindowPolicy({
              axisLen: this.shape[0],
              lineBytes: Math.max(this._realLineBytes, this._complexLineBytes),
              linesTotal: this._lineCount,
              maxBindBytes: this._maxBindBytes,
              axisKind: axisKind0,
              tuning,
              axisIndex: 0,
              storageAlign,
            }),
            pack: resolveOutOfCoreAxisWindowPolicy({
              axisLen: this.shape[0],
              lineBytes: Math.max(this._complexLineBytes, this._packedLineBytes),
              linesTotal: this._lineCount,
              maxBindBytes: this._maxBindBytes,
              axisKind: axisKind0,
              tuning,
              axisIndex: 0,
              storageAlign,
            }),
          };
        }
    
        // Internal C2C forward on full complex
        this.c2c = new C2CPlan(device, {
          shape: this.shape,
          direction: "forward",
          batch,
          inPlace: true,
          normalize,
          layout: { interleavedComplex: true },
          precision: "f32",
          ioView: { input: null, output: null },
          tuning: opts?.tuning ?? null,
        });
        const mergedRoute = mergeLargeRouteMetadata([
          {
            routeMode: this._largeRouteMode,
            reasonCodes: this._largeRouteReasons,
            attemptedRoutes: this._largeRouteAttempts,
          },
          {
            routeMode: this.c2c?._largeRouteMode,
            reasonCodes: this.c2c?._largeRouteReasons,
            attemptedRoutes: this.c2c?._largeRouteAttempts,
          },
        ]);
        this._largeRouteMode = mergedRoute.routeMode;
        this._largeRouteReasons = mergedRoute.reasonCodes;
        this._largeRouteAttempts = mergedRoute.attemptedRoutes;
    
        // real->complex kernel
        this.rtob = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateRealToComplexWGSL({ totalReal: this.totalReal, workgroupSize: this.workgroupSize });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(params, 0, new Uint32Array([this.totalReal, 0, 0, 0]));
          return { bgl, pl: pipelineLayout, pipeline, params };
        })();
    
        // pack kernel
        this.pack = (() => {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generatePackR2CWGSL({ shape: this.shape, workgroupSize: this.workgroupSize });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(params, 0, new Uint32Array([batch, 0, 0, 0]));
          return { bgl, pl: pipelineLayout, pipeline, params };
        })();
        this.packLine = null;
        if (this._largeShapeMode) {
          this.packLine = (() => {
            const bgl = device.createBindGroupLayout({
              entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
              ],
            });
            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
            const code = generatePackR2CWGSL({ shape: [this.shape[0]], workgroupSize: this.workgroupSize });
            const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
            const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            return { bgl, pl: pipelineLayout, pipeline, params };
          })();
        }
    
        // f16 input conversion (real)
        this.f16in = null;
        if (precision === "f16-storage") {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = this.cache.getComputePipeline({ code: generateF16ToF32RealWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.f16in = { bgl, pl: pipelineLayout, pipeline, params };
          device.queue.writeBuffer(params, 0, new Uint32Array([this.inViewTotalReal, 0, 0, 0]));
        }
    
        this.f16out = null;
        if (precision === "f16-storage") {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipeline = this.cache.getComputePipeline({ code: generateF32ToF16ComplexWGSL({ workgroupSize: this.workgroupSize }), layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(params, 0, new Uint32Array([this.outViewTotalComplex, 0, 0, 0]));
          this.f16out = { bgl, pl: pipelineLayout, pipeline, params };
        }
    
        // ioView mapping pipelines
        this.ioEmbed = null;
        if (this.ioIn && needsIoMapping(this.ioIn, this.shape)) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateEmbedRealWGSL({ rank: this.rank, logicalDims: this.shape, viewDims: this.ioIn.shape, offset: this.ioIn.offset, workgroupSize: this.workgroupSize });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.ioEmbed = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.ioIn.shape) };
        }
    
        this.ioExtract = null;
        if (this.ioOut && needsIoMapping(this.ioOut, this.packedShape)) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code =
            this.precision === "f16-storage"
              ? generateExtractComplexF32ToF16WGSL({
                  rank: this.rank,
                  logicalDims: this.packedShape,
                  viewDims: this.ioOut.shape,
                  offset: this.ioOut.offset,
                  clearOutside: this.ioOut.clearOutside,
                  workgroupSize: this.workgroupSize,
                })
              : generateExtractComplexWGSL({
                  rank: this.rank,
                  logicalDims: this.packedShape,
                  viewDims: this.ioOut.shape,
                  offset: this.ioOut.offset,
                  clearOutside: this.ioOut.clearOutside,
                  workgroupSize: this.workgroupSize,
                });
          const pipeline = this.cache.getComputePipeline({ code, layout: pipelineLayout });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.ioExtract = { bgl, pl: pipelineLayout, pipeline, params, viewTotal: prod(this.ioOut.shape), logicalTotal: prod(this.packedShape) };
        }
    
        this.zeroRead = null;
        if (this.zeroPadRead) {
          const bgl = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateZeroOutsideRangeRealWGSL({
            shape: this.shape,
            start: this.zeroPadRead.start,
            end: this.zeroPadRead.end,
            batch: this.batch,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          this.zeroRead = { bgl, pl, pipeline };
        }
    
        this.zeroWrite = null;
        if (this.zeroPadWrite) {
          const bgl = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateZeroOutsideRangeComplexWGSL({
            shape: this.packedShape,
            start: this.zeroPadWrite.start,
            end: this.zeroPadWrite.end,
            batch: this.batch,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          this.zeroWrite = { bgl, pl, pipeline };
        }
    
        // Optional strided gather/scatter for real input and packed-complex output (f32 only).
        this.stridedIn = null;
        this.stridedOut = null;
        if (this._usesStridedInput) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateGatherRealStridedWGSL({
            shape: this.shape,
            strides: this._inputStrides,
            baseOffsetElements: this._inputOffsetElements,
            batchStrideElements: this._inputBatchStrideElements,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.stridedIn = { bgl, pl, pipeline, params };
        }
    
        if (this._usesStridedOutput) {
          const bgl = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
          });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const code = generateScatterComplexStridedWGSL({
            shape: this.packedShape,
            strides: this._outputStrides,
            baseOffsetElements: this._outputOffsetElements,
            batchStrideElements: this._outputBatchStrideElements,
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: pl });
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.stridedOut = { bgl, pl, pipeline, params };
        }
    
        // Workspace: [stage scratch][real logical f32][full complex f32][packed logical f32][optional packed f16 out]
        this.realF32Bytes = this.totalReal * 4;
        this.stageInF32Bytes = this.ioEmbed ? this.inViewTotalReal * 4 : 0;
        this.stageOutF32Bytes = this.ioExtract && this.precision === "f32" ? this.outViewTotalComplex * 8 : 0;
        this.stageF16Bytes = precision === "f16-storage" ? this.inBytes : 0;
        const stageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        this.stageF16Offset = alignBytes(this.stageInF32Bytes, stageAlign);
        this.stageBytes = Math.max(this.stageOutF32Bytes, this.stageF16Offset + this.stageF16Bytes);
    
        const storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        let off = 0;
        this.stageOffset = 0;
        off += this.stageBytes;
        off = alignBytes(off, storageAlign);
        this.realOffset = off;
        off += this.realF32Bytes;
        off = alignBytes(off, storageAlign);
        this.fullOffset = off;
        off += this.fullBytes;
        off = alignBytes(off, storageAlign);
        this.packedOffset = off;
        off += this.packedF32Bytes;
        off = alignBytes(off, storageAlign);
        this.packedF16Offset = precision === "f16-storage" ? off : 0;
        off += precision === "f16-storage" ? this.outBytes : 0;
    
        this.workspaceBytes = off;
        this._splitWorkspace = null;
        const maxBufferSize = device.limits?.maxBufferSize ?? Infinity;
        if (this.workspaceBytes <= maxBufferSize) {
          this._arena = createInternalArena(device, this.workspaceBytes);
        } else {
          const splitNeeds = [
            ["stage", this.stageBytes],
            ["real", this.realF32Bytes],
            ["full", this.fullBytes],
            ["packed", this.packedF32Bytes],
            ["packedF16", this.precision === "f16-storage" ? this.outBytes : 0],
          ];
          for (const [name, bytes] of splitNeeds) {
            if (bytes > 0 && bytes > maxBufferSize) {
              throw new Error(
                `r2c split workspace cannot allocate ${name} buffer: ${bytes} bytes exceeds device.limits.maxBufferSize=${maxBufferSize}. ` +
                  `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
              );
            }
          }
          this._arena = null;
          this._splitWorkspace = {
            stage: this.stageBytes ? createInternalArena(device, this.stageBytes) : null,
            real: createInternalArena(device, this.realF32Bytes),
            full: createInternalArena(device, this.fullBytes),
            packed: createInternalArena(device, this.packedF32Bytes),
            packedF16: this.precision === "f16-storage" ? createInternalArena(device, this.outBytes) : null,
          };
        }
        this._largeChunkBuffer = null;
        this._largeChunkBytes = 0;
        this._retiredLargeChunkBuffers = [];
        this._zeroRealBuffer = null;
        this._zeroComplexBuffer = null;
        this._deferredUniformBuffers = [];
      }
    
      getWorkspaceSizeBytes() {
        return this.workspaceBytes;
      }
    
      _ensureLargeChunkBuffer(minBytes) {
        if (this._largeChunkBuffer && this._largeChunkBytes >= minBytes) return this._largeChunkBuffer;
        const maxBufferSize = this.device.limits?.maxBufferSize ?? Infinity;
        if (minBytes > maxBufferSize) {
          throw new Error(
            `r2c large-shape staging requires ${minBytes} bytes, exceeding device.limits.maxBufferSize=${maxBufferSize}. ` +
              `(routeMode=${this._largeRouteMode}, attempts=${JSON.stringify(this._largeRouteAttempts)}, reasonCodes=${JSON.stringify(this._largeRouteReasons)}).`
          );
        }
        if (this._largeChunkBuffer) this._retiredLargeChunkBuffers.push(this._largeChunkBuffer);
        this._largeChunkBuffer = this.device.createBuffer({
          size: minBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._largeChunkBytes = minBytes;
        return this._largeChunkBuffer;
      }
    
      _ensureZeroRealBuffer() {
        if (this._zeroRealBuffer) return this._zeroRealBuffer;
        this._zeroRealBuffer = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this._zeroRealBuffer, 0, new Float32Array([0]));
        return this._zeroRealBuffer;
      }
    
      _ensureZeroComplexBuffer() {
        if (this._zeroComplexBuffer) return this._zeroComplexBuffer;
        this._zeroComplexBuffer = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this._zeroComplexBuffer, 0, new Float32Array([0, 0]));
        return this._zeroComplexBuffer;
      }
    
      _copyRangesToContiguous(commandEncoder, ranges, dstBuffer, dstOffsetBytes) {
        let dst = dstOffsetBytes;
        for (const r of ranges) {
          commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
          dst += r.sizeBytes;
        }
      }
    
      _copyContiguousToRanges(commandEncoder, srcBuffer, srcOffsetBytes, ranges) {
        let src = srcOffsetBytes;
        for (const r of ranges) {
          commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
          src += r.sizeBytes;
        }
      }
    
      _normalizeCopyView(x) {
        if (x && x.buffer && Number.isInteger(x.offsetBytes) && Number.isInteger(x.sizeBytes)) {
          return {
            segments: [{ buffer: x.buffer, offsetBytes: x.offsetBytes, sizeBytes: x.sizeBytes }],
            logicalByteOffset: 0,
            lengthBytes: x.sizeBytes,
          };
        }
        return x;
      }
    
      _storageRef(x) {
        if (x && x.view) {
          return { store: x.view, baseOffsetBytes: x.offsetBytes ?? 0, sizeBytes: x.sizeBytes ?? null };
        }
        if (x && x.buffer && Number.isInteger(x.offsetBytes) && Number.isInteger(x.sizeBytes)) {
          return { store: x.buffer, baseOffsetBytes: x.offsetBytes, sizeBytes: x.sizeBytes };
        }
        return { store: x, baseOffsetBytes: 0, sizeBytes: null };
      }
    
      _copyAnySpan(commandEncoder, { src, srcOffsetBytes, dst, dstOffsetBytes, bytes }) {
        if (bytes <= 0) return;
        const srcRanges = normalizeToContiguousRanges(this._normalizeCopyView(src), srcOffsetBytes, bytes);
        const dstRanges = normalizeToContiguousRanges(this._normalizeCopyView(dst), dstOffsetBytes, bytes);
        if (srcRanges.length === 1 && dstRanges.length === 1) {
          const s = srcRanges[0];
          const d = dstRanges[0];
          if (s.buffer === d.buffer && s.offsetBytes === d.offsetBytes) return;
          commandEncoder.copyBufferToBuffer(s.buffer, s.offsetBytes, d.buffer, d.offsetBytes, bytes);
          return;
        }
        if (srcRanges.length > 1 && dstRanges.length === 1) {
          this.copier.pack(commandEncoder, srcRanges, dstRanges[0].buffer, dstRanges[0].offsetBytes);
          return;
        }
        if (srcRanges.length === 1 && dstRanges.length > 1) {
          this.copier.unpack(commandEncoder, srcRanges[0].buffer, srcRanges[0].offsetBytes, dstRanges);
          return;
        }
        let si = 0;
        let di = 0;
        let soff = srcRanges[0].offsetBytes;
        let doff = dstRanges[0].offsetBytes;
        let srem = srcRanges[0].sizeBytes;
        let drem = dstRanges[0].sizeBytes;
        while (si < srcRanges.length && di < dstRanges.length) {
          const n = Math.min(srem, drem);
          commandEncoder.copyBufferToBuffer(srcRanges[si].buffer, soff, dstRanges[di].buffer, doff, n);
          soff += n;
          doff += n;
          srem -= n;
          drem -= n;
          if (srem === 0) {
            si += 1;
            if (si < srcRanges.length) {
              soff = srcRanges[si].offsetBytes;
              srem = srcRanges[si].sizeBytes;
            }
          }
          if (drem === 0) {
            di += 1;
            if (di < dstRanges.length) {
              doff = dstRanges[di].offsetBytes;
              drem = dstRanges[di].sizeBytes;
            }
          }
        }
      }
    
      _copyRealFromAny(commandEncoder, src, srcOffsetBytes, dstBuffer, dstOffsetBytes) {
        if (isGpuBuffer(src)) {
          commandEncoder.copyBufferToBuffer(src, srcOffsetBytes, dstBuffer, dstOffsetBytes, 4);
          return;
        }
        const srcRanges = normalizeToContiguousRanges(src, srcOffsetBytes, 4);
        if (srcRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(srcRanges[0].buffer, srcRanges[0].offsetBytes, dstBuffer, dstOffsetBytes, 4);
          return;
        }
        const chunkBuf = this._ensureLargeChunkBuffer(4);
        this.copier.pack(commandEncoder, srcRanges, chunkBuf, 0);
        commandEncoder.copyBufferToBuffer(chunkBuf, 0, dstBuffer, dstOffsetBytes, 4);
      }
    
      _copyComplexToAny(commandEncoder, srcBuffer, srcOffsetBytes, dst, dstOffsetBytes) {
        if (isGpuBuffer(dst)) {
          commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dst, dstOffsetBytes, 8);
          return;
        }
        const dstRanges = normalizeToContiguousRanges(dst, dstOffsetBytes, 8);
        if (dstRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, dstRanges[0].buffer, dstRanges[0].offsetBytes, 8);
          return;
        }
        const chunkBuf = this._ensureLargeChunkBuffer(8);
        commandEncoder.copyBufferToBuffer(srcBuffer, srcOffsetBytes, chunkBuf, 0, 8);
        this.copier.unpack(commandEncoder, chunkBuf, 0, dstRanges);
      }
    
      _shapeStrides(shape) {
        return tensorContiguousStrides(shape);
      }
    
      _coordsFromLinear(i, shape, outCoords) {
        tensorCoordsFromLinear(i, shape, outCoords);
      }
    
      _linearFromCoords(coords, strides) {
        return tensorLinearFromCoords(coords, strides);
      }
    
      _requiredStridedInputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
        if (!this._inputTensorDesc) {
          throw new Error("internal error: strided input descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._inputTensorDesc, {
          bytesPerElement: 4,
          runtimeExtraElements: extraOffsetElements,
          batchStart,
          batchCount,
        });
      }
    
      _requiredStridedOutputBytes(extraOffsetElements, batchStart = 0, batchCount = this.batch) {
        if (!this._outputTensorDesc) {
          throw new Error("internal error: strided output descriptor is not initialized");
        }
        return requiredBytesForBatchRange(this._outputTensorDesc, {
          bytesPerElement: 8,
          runtimeExtraElements: extraOffsetElements,
          batchStart,
          batchCount,
        });
      }
    
      _copyStridedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange }) {
        if (inputOffsetBytes % 4 !== 0) {
          throw new Error(`inputOffsetBytes must be a multiple of 4 for real-strided input; got ${inputOffsetBytes}`);
        }
        const extraOffsetElements = (inputOffsetBytes / 4) | 0;
        const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
        const inputBytes = getBufferByteLength(input);
        if (inputBytes < neededBytes) {
          throw new Error(`input buffer/view too small for strided layout: need ${neededBytes} bytes, have ${inputBytes}`);
        }
    
        const realRef = this._storageRef(realRange);
    
        if (!this._needsInputMapping) {
          const coords = new Array(this.rank).fill(0);
          for (let b = 0; b < this.batch; b++) {
            const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
            const dstBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
            for (let li = 0; li < this.logicalTotal; li++) {
              this._coordsFromLinear(li, this.shape, coords);
              let srcElem = srcBatchBase;
              for (let d = 0; d < this.rank; d++) srcElem += coords[d] * this._inputStrides[d];
              this._copyAnySpan(commandEncoder, {
                src: input,
                srcOffsetBytes: srcElem * 4,
                dst: realRef.store,
                dstOffsetBytes: dstBase + li * 4,
                bytes: 4,
              });
            }
          }
          return;
        }
    
        const zeroBuf = this._ensureZeroRealBuffer();
        const viewShape = this.ioIn.shape;
        const viewOffset = this.ioIn.offset;
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const srcBatchBase = this._inputOffsetElements + extraOffsetElements + b * this._inputBatchStrideElements;
          const dstBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: realRef.store,
                dstOffsetBytes: dstBase + li * 4,
                bytes: 4,
              });
              continue;
            }
            let srcElem = srcBatchBase;
            for (let d = 0; d < this.rank; d++) srcElem += viewCoords[d] * this._inputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src: input,
              srcOffsetBytes: srcElem * 4,
              dst: realRef.store,
              dstOffsetBytes: dstBase + li * 4,
              bytes: 4,
            });
          }
        }
      }
    
      _copyContiguousPackedToStridedOutputOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes }) {
        if (outputOffsetBytes % 8 !== 0) {
          throw new Error(`outputOffsetBytes must be a multiple of 8 for packed-complex strided output; got ${outputOffsetBytes}`);
        }
        const extraOffsetElements = (outputOffsetBytes / 8) | 0;
        const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
        const outputBytes = getBufferByteLength(output);
        if (outputBytes < neededBytes) {
          throw new Error(`output buffer/view too small for strided layout: need ${neededBytes} bytes, have ${outputBytes}`);
        }
    
        const packedRef = this._storageRef(packedRange);
        const packedTotal = prod(this.packedShape);
        if (!this._needsOutputMapping) {
          const coords = new Array(this.rank).fill(0);
          for (let b = 0; b < this.batch; b++) {
            const srcBase = packedRef.baseOffsetBytes + b * packedTotal * 8;
            const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
            for (let li = 0; li < packedTotal; li++) {
              this._coordsFromLinear(li, this.packedShape, coords);
              let dstElem = dstBatchBase;
              for (let d = 0; d < this.rank; d++) dstElem += coords[d] * this._outputStrides[d];
              this._copyAnySpan(commandEncoder, {
                src: packedRef.store,
                srcOffsetBytes: srcBase + li * 8,
                dst: output,
                dstOffsetBytes: dstElem * 8,
                bytes: 8,
              });
            }
          }
          return;
        }
    
        const viewShape = this.ioOut.shape;
        const viewOffset = this.ioOut.offset;
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
    
        if (this.ioOut.clearOutside) {
          const zeroBuf = this._ensureZeroComplexBuffer();
          const viewTotal = prod(viewShape);
          for (let b = 0; b < this.batch; b++) {
            const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
            for (let vi = 0; vi < viewTotal; vi++) {
              this._coordsFromLinear(vi, viewShape, viewCoords);
              let dstElem = dstBatchBase;
              for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: output,
                dstOffsetBytes: dstElem * 8,
                bytes: 8,
              });
            }
          }
        }
    
        for (let b = 0; b < this.batch; b++) {
          const srcBase = packedRef.baseOffsetBytes + b * packedTotal * 8;
          const dstBatchBase = this._outputOffsetElements + extraOffsetElements + b * this._outputBatchStrideElements;
          for (let li = 0; li < packedTotal; li++) {
            this._coordsFromLinear(li, this.packedShape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) continue;
            let dstElem = dstBatchBase;
            for (let d = 0; d < this.rank; d++) dstElem += viewCoords[d] * this._outputStrides[d];
            this._copyAnySpan(commandEncoder, {
              src: packedRef.store,
              srcOffsetBytes: srcBase + li * 8,
              dst: output,
              dstOffsetBytes: dstElem * 8,
              bytes: 8,
            });
          }
        }
      }
    
      _zeroOutsideRangeRealLarge(commandEncoder, { dataRange, start, end }) {
        const dataRef = this._storageRef(dataRange);
        const zeroBuf = this._ensureZeroRealBuffer();
        const coords = new Array(this.rank).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const base = dataRef.baseOffsetBytes + b * this.logicalTotal * 4;
          for (let i = 0; i < this.logicalTotal; i++) {
            this._coordsFromLinear(i, this.shape, coords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              if (coords[d] < start[d] || coords[d] >= end[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: dataRef.store,
                dstOffsetBytes: base + i * 4,
                bytes: 4,
              });
            }
          }
        }
      }
    
      _zeroOutsideRangeComplexLarge(commandEncoder, { dataRange, shape, start, end }) {
        const dataRef = this._storageRef(dataRange);
        const zeroBuf = this._ensureZeroComplexBuffer();
        const logicalTotal = prod(shape);
        const coords = new Array(shape.length).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const base = dataRef.baseOffsetBytes + b * logicalTotal * 8;
          for (let i = 0; i < logicalTotal; i++) {
            this._coordsFromLinear(i, shape, coords);
            let inside = true;
            for (let d = 0; d < shape.length; d++) {
              if (coords[d] < start[d] || coords[d] >= end[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: dataRef.store,
                dstOffsetBytes: base + i * 8,
                bytes: 8,
              });
            }
          }
        }
      }
    
      _embedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange }) {
        const realRef = this._storageRef(realRange);
        if (!this.ioEmbed) {
          this._copyAnySpan(commandEncoder, {
            src: input,
            srcOffsetBytes: inputOffsetBytes,
            dst: realRef.store,
            dstOffsetBytes: realRef.baseOffsetBytes,
            bytes: this.realF32Bytes,
          });
          return;
        }
    
        const inBytes = this.inViewTotalReal * 4;
        const inBuf = this._ensureLargeChunkBuffer(inBytes);
        this._copyAnySpan(commandEncoder, {
          src: input,
          srcOffsetBytes: inputOffsetBytes,
          dst: inBuf,
          dstOffsetBytes: 0,
          bytes: inBytes,
        });
    
        const zeroBuf = this._ensureZeroRealBuffer();
        const viewShape = this.ioIn.shape;
        const viewOffset = this.ioIn.offset;
        const viewTotal = prod(viewShape);
        const viewStrides = this._shapeStrides(viewShape);
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const srcBase = b * viewTotal * 4;
          const dstBase = realRef.baseOffsetBytes + b * this.logicalTotal * 4;
          for (let li = 0; li < this.logicalTotal; li++) {
            this._coordsFromLinear(li, this.shape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) {
              this._copyAnySpan(commandEncoder, {
                src: zeroBuf,
                srcOffsetBytes: 0,
                dst: realRef.store,
                dstOffsetBytes: dstBase + li * 4,
                bytes: 4,
              });
              continue;
            }
            const vi = this._linearFromCoords(viewCoords, viewStrides);
            this._copyAnySpan(commandEncoder, {
              src: inBuf,
              srcOffsetBytes: srcBase + vi * 4,
              dst: realRef.store,
              dstOffsetBytes: dstBase + li * 4,
              bytes: 4,
            });
          }
        }
      }
    
      _extractOutputComplexOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes }) {
        const packedRef = this._storageRef(packedRange);
        if (!this.ioExtract) {
          this._copyAnySpan(commandEncoder, {
            src: packedRef.store,
            srcOffsetBytes: packedRef.baseOffsetBytes,
            dst: output,
            dstOffsetBytes: outputOffsetBytes,
            bytes: this.outBytes,
          });
          return;
        }
    
        const outBytes = this.outViewTotalComplex * 8;
        const outBuf = this._ensureLargeChunkBuffer(outBytes);
        if (!this.ioOut.clearOutside) {
          this._copyAnySpan(commandEncoder, {
            src: output,
            srcOffsetBytes: outputOffsetBytes,
            dst: outBuf,
            dstOffsetBytes: 0,
            bytes: outBytes,
          });
        } else {
          const zeroBuf = this._ensureZeroComplexBuffer();
          for (let i = 0; i < this.outViewTotalComplex; i++) {
            commandEncoder.copyBufferToBuffer(zeroBuf, 0, outBuf, i * 8, 8);
          }
        }
    
        const viewShape = this.ioOut.shape;
        const viewOffset = this.ioOut.offset;
        const viewTotal = prod(viewShape);
        const logicalTotal = prod(this.packedShape);
        const viewStrides = this._shapeStrides(viewShape);
        const logicalCoords = new Array(this.rank).fill(0);
        const viewCoords = new Array(this.rank).fill(0);
        for (let b = 0; b < this.batch; b++) {
          const srcBase = packedRef.baseOffsetBytes + b * logicalTotal * 8;
          const dstBase = b * viewTotal * 8;
          for (let li = 0; li < logicalTotal; li++) {
            this._coordsFromLinear(li, this.packedShape, logicalCoords);
            let inside = true;
            for (let d = 0; d < this.rank; d++) {
              const v = logicalCoords[d] - viewOffset[d];
              viewCoords[d] = v;
              if (v < 0 || v >= viewShape[d]) {
                inside = false;
                break;
              }
            }
            if (!inside) continue;
            const vi = this._linearFromCoords(viewCoords, viewStrides);
            this._copyAnySpan(commandEncoder, {
              src: packedRef.store,
              srcOffsetBytes: srcBase + li * 8,
              dst: outBuf,
              dstOffsetBytes: dstBase + vi * 8,
              bytes: 8,
            });
          }
        }
    
        this._copyAnySpan(commandEncoder, {
          src: outBuf,
          srcOffsetBytes: 0,
          dst: output,
          dstOffsetBytes: outputOffsetBytes,
          bytes: outBytes,
        });
      }
    
      _resolveLargeStageLinesPerChunk(stageKey, lineBytes) {
        const maxLinesByBind = Math.max(1, Math.floor(this._maxBindBytes / lineBytes));
        const policy = this._outOfCoreAxisWindowPolicy?.[stageKey] ?? null;
        let linesPerChunk = maxLinesByBind;
        if (policy && Number.isInteger(policy.linesPerChunk) && policy.linesPerChunk > 0) {
          linesPerChunk = Math.max(1, Math.min(linesPerChunk, policy.linesPerChunk));
        }
        const alignedLineStep = policy?.alignedLineStep ?? 1;
        if (Number.isInteger(alignedLineStep) && alignedLineStep > 1 && linesPerChunk >= alignedLineStep) {
          linesPerChunk = Math.max(alignedLineStep, Math.floor(linesPerChunk / alignedLineStep) * alignedLineStep);
        }
        return Math.max(1, Math.min(linesPerChunk, this._lineCount));
      }
    
      _runRealToComplexLineChunks(commandEncoder, { realRange, complexRange }) {
        const realRef = this._storageRef(realRange);
        const complexRef = this._storageRef(complexRange);
        if (this._realLineBytes > this._maxBindBytes || this._complexLineBytes > this._maxBindBytes) {
          this._runRealToComplexElementChunks(commandEncoder, { realRange, complexRange });
          return;
        }
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const lineBytes = Math.max(this._realLineBytes, this._complexLineBytes);
        const maxLinesPerChunk = this._resolveLargeStageLinesPerChunk("realToComplex", lineBytes);
        const maxInBytes = maxLinesPerChunk * this._realLineBytes;
        const maxOutBytes = maxLinesPerChunk * this._complexLineBytes;
        const maxOutOffset = alignBytes(maxInBytes, storageAlign);
        const chunkBuf = this._ensureLargeChunkBuffer(maxOutOffset + maxOutBytes);
        const chunkCount = Math.ceil(this._lineCount / maxLinesPerChunk);
        const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
        const paramStride = alignBytes(16, uniformAlign);
        const paramsBuf = this.device.createBuffer({
          size: chunkCount * paramStride,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._deferredUniformBuffers.push(paramsBuf);
    
        let chunkIndex = 0;
        for (let line0 = 0; line0 < this._lineCount; line0 += maxLinesPerChunk) {
          const lines = Math.min(maxLinesPerChunk, this._lineCount - line0);
          const inBytes = lines * this._realLineBytes;
          const outBytes = lines * this._complexLineBytes;
          const outOff = alignBytes(inBytes, storageAlign);
          const srcOff = realRef.baseOffsetBytes + line0 * this._realLineBytes;
          const dstOff = complexRef.baseOffsetBytes + line0 * this._complexLineBytes;
    
          this._copyAnySpan(commandEncoder, {
            src: realRef.store,
            srcOffsetBytes: srcOff,
            dst: chunkBuf,
            dstOffsetBytes: 0,
            bytes: inBytes,
          });
          const paramOff = chunkIndex * paramStride;
          this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([lines * this.shape[0], 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.rtob.bgl,
            entries: [
              { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
              { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
              { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.rtob.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((lines * this.shape[0]) / this.workgroupSize), 1, 1);
          pass.end();
          this._copyAnySpan(commandEncoder, {
            src: chunkBuf,
            srcOffsetBytes: outOff,
            dst: complexRef.store,
            dstOffsetBytes: dstOff,
            bytes: outBytes,
          });
          chunkIndex += 1;
        }
      }
    
      _runRealToComplexElementChunks(commandEncoder, { realRange, complexRange }) {
        const realRef = this._storageRef(realRange);
        const complexRef = this._storageRef(complexRange);
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const maxElems = Math.max(1, Math.floor(this._maxBindBytes / 8));
        const maxInBytes = maxElems * 4;
        const maxOutBytes = maxElems * 8;
        const outOff = alignBytes(maxInBytes, storageAlign);
        const chunkBuf = this._ensureLargeChunkBuffer(outOff + maxOutBytes);
    
        const chunksPerLine = Math.ceil(this.shape[0] / maxElems);
        const chunkCount = this._lineCount * chunksPerLine;
        const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
        const paramStride = alignBytes(16, uniformAlign);
        const paramsBuf = this.device.createBuffer({
          size: chunkCount * paramStride,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._deferredUniformBuffers.push(paramsBuf);
    
        let chunkIndex = 0;
        for (let line = 0; line < this._lineCount; line++) {
          const srcLineBase = realRef.baseOffsetBytes + line * this._realLineBytes;
          const dstLineBase = complexRef.baseOffsetBytes + line * this._complexLineBytes;
          for (let x0 = 0; x0 < this.shape[0]; x0 += maxElems) {
            const elems = Math.min(maxElems, this.shape[0] - x0);
            const inBytes = elems * 4;
            const outBytes = elems * 8;
            this._copyAnySpan(commandEncoder, {
              src: realRef.store,
              srcOffsetBytes: srcLineBase + x0 * 4,
              dst: chunkBuf,
              dstOffsetBytes: 0,
              bytes: inBytes,
            });
            const paramOff = chunkIndex * paramStride;
            this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([elems, 0, 0, 0]));
            const bg = this.device.createBindGroup({
              layout: this.rtob.bgl,
              entries: [
                { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
                { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
                { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.rtob.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(elems / this.workgroupSize), 1, 1);
            pass.end();
            this._copyAnySpan(commandEncoder, {
              src: chunkBuf,
              srcOffsetBytes: outOff,
              dst: complexRef.store,
              dstOffsetBytes: dstLineBase + x0 * 8,
              bytes: outBytes,
            });
            chunkIndex += 1;
          }
        }
      }
    
      _runPackLineChunks(commandEncoder, { complexRange, packedRange }) {
        const complexRef = this._storageRef(complexRange);
        const packedRef = this._storageRef(packedRange);
        if (this._complexLineBytes > this._maxBindBytes || this._packedLineBytes > this._maxBindBytes) {
          for (let line = 0; line < this._lineCount; line++) {
            const srcOff = complexRef.baseOffsetBytes + line * this._complexLineBytes;
            const dstOff = packedRef.baseOffsetBytes + line * this._packedLineBytes;
            this._copyAnySpan(commandEncoder, {
              src: complexRef.store,
              srcOffsetBytes: srcOff,
              dst: packedRef.store,
              dstOffsetBytes: dstOff,
              bytes: this._packedLineBytes,
            });
          }
          return;
        }
        const storageAlign = this.device.limits?.minStorageBufferOffsetAlignment ?? 256;
        const lineBytes = Math.max(this._complexLineBytes, this._packedLineBytes);
        const maxLinesPerChunk = this._resolveLargeStageLinesPerChunk("pack", lineBytes);
        const maxInBytes = maxLinesPerChunk * this._complexLineBytes;
        const maxOutBytes = maxLinesPerChunk * this._packedLineBytes;
        const maxOutOffset = alignBytes(maxInBytes, storageAlign);
        const chunkBuf = this._ensureLargeChunkBuffer(maxOutOffset + maxOutBytes);
        const chunkCount = Math.ceil(this._lineCount / maxLinesPerChunk);
        const uniformAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
        const paramStride = alignBytes(16, uniformAlign);
        const paramsBuf = this.device.createBuffer({
          size: chunkCount * paramStride,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._deferredUniformBuffers.push(paramsBuf);
    
        let chunkIndex = 0;
        for (let line0 = 0; line0 < this._lineCount; line0 += maxLinesPerChunk) {
          const lines = Math.min(maxLinesPerChunk, this._lineCount - line0);
          const inBytes = lines * this._complexLineBytes;
          const outBytes = lines * this._packedLineBytes;
          const outOff = alignBytes(inBytes, storageAlign);
          const srcOff = complexRef.baseOffsetBytes + line0 * this._complexLineBytes;
          const dstOff = packedRef.baseOffsetBytes + line0 * this._packedLineBytes;
    
          this._copyAnySpan(commandEncoder, {
            src: complexRef.store,
            srcOffsetBytes: srcOff,
            dst: chunkBuf,
            dstOffsetBytes: 0,
            bytes: inBytes,
          });
          const paramOff = chunkIndex * paramStride;
          this.device.queue.writeBuffer(paramsBuf, paramOff, new Uint32Array([lines, 0, 0, 0]));
          const bg = this.device.createBindGroup({
            layout: this.packLine.bgl,
            entries: [
              { binding: 0, resource: { buffer: chunkBuf, offset: 0, size: inBytes } },
              { binding: 1, resource: { buffer: chunkBuf, offset: outOff, size: outBytes } },
              { binding: 2, resource: { buffer: paramsBuf, offset: paramOff, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.packLine.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((lines * this.packedShape[0]) / this.workgroupSize), 1, 1);
          pass.end();
          this._copyAnySpan(commandEncoder, {
            src: chunkBuf,
            srcOffsetBytes: outOff,
            dst: packedRef.store,
            dstOffsetBytes: dstOff,
            bytes: outBytes,
          });
          chunkIndex += 1;
        }
      }
    
      _resolveWorkspaceViews(temp) {
        const arena = temp ?? this._arena;
        if (arena) {
          if (getBufferByteLength(arena) < this.workspaceBytes) throw new Error("temp too small");
          return {
            stage: this.stageBytes ? viewFromArena(arena, this.stageOffset, this.stageBytes) : null,
            realView: viewFromArena(arena, this.realOffset, this.realF32Bytes),
            complexView: viewFromArena(arena, this.fullOffset, this.fullBytes),
            packedView: viewFromArena(arena, this.packedOffset, this.packedF32Bytes),
            packedF16View: this.precision === "f16-storage" ? viewFromArena(arena, this.packedF16Offset, this.outBytes) : null,
          };
        }
        if (this._splitWorkspace) {
          return {
            stage: this.stageBytes ? viewFromArena(this._splitWorkspace.stage, 0, this.stageBytes) : null,
            realView: viewFromArena(this._splitWorkspace.real, 0, this.realF32Bytes),
            complexView: viewFromArena(this._splitWorkspace.full, 0, this.fullBytes),
            packedView: viewFromArena(this._splitWorkspace.packed, 0, this.packedF32Bytes),
            packedF16View: this.precision === "f16-storage" ? viewFromArena(this._splitWorkspace.packedF16, 0, this.outBytes) : null,
          };
        }
        throw new Error("No workspace buffer");
      }
    
      _workspaceViewsAreContiguous(views) {
        const single = (view, bytes) => {
          if (!view || !bytes) return true;
          return normalizeToContiguousRanges(view, 0, bytes).length === 1;
        };
        return (
          single(views.stage, this.stageBytes) &&
          single(views.realView, this.realF32Bytes) &&
          single(views.complexView, this.fullBytes) &&
          single(views.packedView, this.packedF32Bytes) &&
          single(views.packedF16View, this.precision === "f16-storage" ? this.outBytes : 0)
        );
      }
    
      _resolveLargeWorkspaceRanges(temp) {
        const { realView, complexView, packedView } = this._resolveWorkspaceViews(temp);
        return {
          realRange: { view: realView, offsetBytes: 0, sizeBytes: this.realF32Bytes },
          complexRange: { view: complexView, offsetBytes: 0, sizeBytes: this.fullBytes },
          packedRange: { view: packedView, offsetBytes: 0, sizeBytes: this.packedF32Bytes },
        };
      }
    
      _execLargeShape(commandEncoder, { input, output, temp, inputOffsetBytes, outputOffsetBytes }) {
        const { realRange, complexRange, packedRange } = this._resolveLargeWorkspaceRanges(temp);
    
        if (this._usesStridedInput) {
          this._copyStridedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange });
        } else {
          this._embedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange });
        }
    
        if (this.zeroPadRead) {
          this._zeroOutsideRangeRealLarge(commandEncoder, {
            dataRange: realRange,
            start: this.zeroPadRead.start,
            end: this.zeroPadRead.end,
          });
        }
    
        this._runRealToComplexLineChunks(commandEncoder, { realRange, complexRange });
        {
          const complexRef = this._storageRef(complexRange);
          this.c2c.exec(commandEncoder, { input: complexRef.store, inputOffsetBytes: complexRef.baseOffsetBytes });
        }
        this._runPackLineChunks(commandEncoder, { complexRange, packedRange });
    
        if (this.zeroPadWrite) {
          this._zeroOutsideRangeComplexLarge(commandEncoder, {
            dataRange: packedRange,
            shape: this.packedShape,
            start: this.zeroPadWrite.start,
            end: this.zeroPadWrite.end,
          });
        }
    
        if (this._usesStridedOutput) {
          this._copyContiguousPackedToStridedOutputOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes });
        } else {
          this._extractOutputComplexOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes });
        }
      }
    
      destroy() {
        if (this._destroyed) return;
        this.c2c.destroy();
        this.rtob.params.destroy();
        this.pack.params.destroy();
        this.packLine?.params?.destroy?.();
        this.f16in?.params?.destroy?.();
        this.f16out?.params?.destroy?.();
        this.ioEmbed?.params?.destroy?.();
        this.ioExtract?.params?.destroy?.();
        this.stridedIn?.params?.destroy?.();
        this.stridedOut?.params?.destroy?.();
        this._largeChunkBuffer?.destroy?.();
        this._splitWorkspace?.stage?.destroy?.();
        this._splitWorkspace?.real?.destroy?.();
        this._splitWorkspace?.full?.destroy?.();
        this._splitWorkspace?.packed?.destroy?.();
        this._splitWorkspace?.packedF16?.destroy?.();
        for (const b of this._retiredLargeChunkBuffers) b?.destroy?.();
        for (const b of this._deferredUniformBuffers) b?.destroy?.();
        this._zeroRealBuffer?.destroy?.();
        this._zeroComplexBuffer?.destroy?.();
        this._arena?.destroy?.();
        super.destroy();
      }
    
      exec(commandEncoder, execOpts) {
        if (this._destroyed) throw new Error("plan destroyed");
        const { input, output, temp, inputOffsetBytes = 0, outputOffsetBytes = 0 } = execOpts ?? {};
        if (!input || !output) throw new Error("r2c exec requires input and output");
        let workspaceTemp = temp;
        if (workspaceTemp && (buffersAlias(workspaceTemp, input) || buffersAlias(workspaceTemp, output))) {
          workspaceTemp = null;
        }
        if (this._largeShapeMode) {
          this._execLargeShape(commandEncoder, { input, output, temp: workspaceTemp, inputOffsetBytes, outputOffsetBytes });
          return;
        }
    
        let workspaceViews = this._resolveWorkspaceViews(workspaceTemp);
        if (workspaceTemp && !this._workspaceViewsAreContiguous(workspaceViews)) {
          workspaceTemp = null;
          workspaceViews = this._resolveWorkspaceViews(null);
        }
        const { stage, realView, complexView, packedView, packedF16View } = workspaceViews;
    
        // Load physical input into f32, then optional ioView embed into logical domain.
        if (this._usesStridedInput) {
          if (this._needsInputMapping) {
            const realRange = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
            this._copyStridedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange });
          } else {
            if (!isGpuBuffer(input)) {
              const realRange = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
              this._copyStridedInputRealOutOfCore(commandEncoder, { input, inputOffsetBytes, realRange });
            } else {
              if (inputOffsetBytes % 4 !== 0) {
                throw new Error(`inputOffsetBytes must be a multiple of 4 for real-strided input; got ${inputOffsetBytes}`);
              }
              const extraOffsetElements = (inputOffsetBytes / 4) | 0;
              const neededBytes = this._requiredStridedInputBytes(extraOffsetElements);
              ensureWithinBindingLimit(this.device, neededBytes, "r2c strided input binding");
              if (input.size < neededBytes) {
                throw new Error(`input buffer too small for strided layout: need ${neededBytes} bytes, have ${input.size}`);
              }
    
              const dstF32 = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
              this.device.queue.writeBuffer(this.stridedIn.params, 0, new Uint32Array([this.logicalTotal, this.batch, extraOffsetElements, 0]));
              const bg = this.device.createBindGroup({
                layout: this.stridedIn.bgl,
                entries: [
                  { binding: 0, resource: { buffer: input, offset: 0, size: neededBytes } },
                  { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.realF32Bytes } },
                  { binding: 2, resource: { buffer: this.stridedIn.params, offset: 0, size: 16 } },
                ],
              });
              const pass = commandEncoder.beginComputePass();
              pass.setPipeline(this.stridedIn.pipeline);
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
              pass.end();
            }
          }
        } else if (this.precision === "f16-storage") {
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
          let srcBuf = inRanges[0].buffer;
          let srcOff = inRanges[0].offsetBytes;
          if (inRanges.length > 1) {
            const scratchF16 = normalizeToContiguousRanges(stage, this.stageF16Offset, this.inBytes)[0];
            this.copier.pack(commandEncoder, inRanges, scratchF16.buffer, scratchF16.offsetBytes);
            srcBuf = scratchF16.buffer;
            srcOff = scratchF16.offsetBytes;
          }
    
          const dstF32 = this.ioEmbed
            ? normalizeToContiguousRanges(stage, 0, this.inViewTotalReal * 4)[0]
            : normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
    
          const bg = this.device.createBindGroup({
            layout: this.f16in.bgl,
            entries: [
              { binding: 0, resource: { buffer: srcBuf, offset: srcOff, size: this.inBytes } },
              { binding: 1, resource: { buffer: dstF32.buffer, offset: dstF32.offsetBytes, size: this.inViewTotalReal * 4 } },
              { binding: 2, resource: { buffer: this.f16in.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16in.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.inViewTotalReal / this.workgroupSize), 1, 1);
          pass.end();
        } else {
          const inRanges = normalizeToContiguousRanges(input, inputOffsetBytes, this.inBytes);
          const dstF32 = this.ioEmbed
            ? normalizeToContiguousRanges(stage, 0, this.inBytes)[0]
            : normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
    
          if (inRanges.length === 1) {
            commandEncoder.copyBufferToBuffer(inRanges[0].buffer, inRanges[0].offsetBytes, dstF32.buffer, dstF32.offsetBytes, this.inBytes);
          } else {
            this.copier.pack(commandEncoder, inRanges, dstF32.buffer, dstF32.offsetBytes);
          }
        }
    
        if (this.ioEmbed && !(this._usesStridedInput && this._needsInputMapping)) {
          this.device.queue.writeBuffer(this.ioEmbed.params, 0, new Uint32Array([this.logicalTotal, this.ioEmbed.viewTotal, this.batch, 0]));
          const src = normalizeToContiguousRanges(stage, 0, this.ioEmbed.viewTotal * this.batch * 4)[0];
          const dst = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.ioEmbed.bgl,
            entries: [
              { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.ioEmbed.viewTotal * this.batch * 4 } },
              { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.realF32Bytes } },
              { binding: 2, resource: { buffer: this.ioEmbed.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioEmbed.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((this.logicalTotal * this.batch) / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        if (this.zeroRead) {
          const r = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.zeroRead.bgl,
            entries: [{ binding: 0, resource: { buffer: r.buffer, offset: r.offsetBytes, size: this.realF32Bytes } }],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.zeroRead.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        // real->complex
        {
          const r = normalizeToContiguousRanges(realView, 0, this.realF32Bytes)[0];
          const c = normalizeToContiguousRanges(complexView, 0, this.fullBytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.rtob.bgl,
            entries: [
              { binding: 0, resource: { buffer: r.buffer, offset: r.offsetBytes, size: this.realF32Bytes } },
              { binding: 1, resource: { buffer: c.buffer, offset: c.offsetBytes, size: this.fullBytes } },
              { binding: 2, resource: { buffer: this.rtob.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.rtob.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.totalReal / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        // FFT full complex in-place
        const c = normalizeToContiguousRanges(complexView, 0, this.fullBytes)[0];
        this.c2c.exec(commandEncoder, { input: c.buffer, inputOffsetBytes: c.offsetBytes });
    
        // pack to packedView (f32 complex)
        {
          const dst = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.pack.bgl,
            entries: [
              { binding: 0, resource: { buffer: c.buffer, offset: c.offsetBytes, size: this.fullBytes } },
              { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.packedF32Bytes } },
              { binding: 2, resource: { buffer: this.pack.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.pack.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        if (this.zeroWrite) {
          const dst = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.zeroWrite.bgl,
            entries: [{ binding: 0, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.packedF32Bytes } }],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.zeroWrite.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
          pass.end();
        }
    
        // Optional output view mapping (packed logical -> packed view shape).
        // If present, write directly to the final output when contiguous to preserve clearOutside=false semantics.
        if (this.ioExtract && !(this._usesStridedOutput && this._needsOutputMapping)) {
          const viewTotal = this.ioExtract.viewTotal;
          this.device.queue.writeBuffer(this.ioExtract.params, 0, new Uint32Array([this.ioExtract.logicalTotal, viewTotal, this.batch, 0]));
    
          const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
          if (outRanges.length === 1) {
            const src = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
            const bg = this.device.createBindGroup({
              layout: this.ioExtract.bgl,
              entries: [
                { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.packedF32Bytes } },
                { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
                { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.ioExtract.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
            pass.end();
            return;
          }
    
          let dstBuf = null;
          let dstOff = 0;
          if (this.precision === "f16-storage") {
            const dst = normalizeToContiguousRanges(packedF16View, 0, this.outBytes)[0];
            dstBuf = dst.buffer;
            dstOff = dst.offsetBytes;
          } else {
            const dst = normalizeToContiguousRanges(stage, 0, this.outBytes)[0];
            dstBuf = dst.buffer;
            dstOff = dst.offsetBytes;
          }
    
          if (!this.ioOut.clearOutside) {
            this.copier.pack(commandEncoder, outRanges, dstBuf, dstOff);
          }
    
          const src = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.ioExtract.bgl,
            entries: [
              { binding: 0, resource: { buffer: src.buffer, offset: src.offsetBytes, size: this.packedF32Bytes } },
              { binding: 1, resource: { buffer: dstBuf, offset: dstOff, size: this.outBytes } },
              { binding: 2, resource: { buffer: this.ioExtract.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.ioExtract.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil((viewTotal * this.batch) / this.workgroupSize), 1, 1);
          pass.end();
    
          this.copier.unpack(commandEncoder, dstBuf, dstOff, outRanges);
          return;
        }
    
        if (this._usesStridedOutput) {
          if (this._needsOutputMapping) {
            const packedRange = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
            this._copyContiguousPackedToStridedOutputOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes });
            return;
          }
          const packedRange = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          if (!isGpuBuffer(output)) {
            this._copyContiguousPackedToStridedOutputOutOfCore(commandEncoder, { packedRange, output, outputOffsetBytes });
            return;
          }
          if (outputOffsetBytes % 8 !== 0) {
            throw new Error(`outputOffsetBytes must be a multiple of 8 for packed-complex strided output; got ${outputOffsetBytes}`);
          }
          const extraOffsetElements = (outputOffsetBytes / 8) | 0;
          const neededBytes = this._requiredStridedOutputBytes(extraOffsetElements);
          ensureWithinBindingLimit(this.device, neededBytes, "r2c strided output binding");
          if (output.size < neededBytes) {
            throw new Error(`output buffer too small for strided layout: need ${neededBytes} bytes, have ${output.size}`);
          }
          this.device.queue.writeBuffer(this.stridedOut.params, 0, new Uint32Array([prod(this.packedShape), this.batch, extraOffsetElements, 0]));
          const bg = this.device.createBindGroup({
            layout: this.stridedOut.bgl,
            entries: [
              { binding: 0, resource: { buffer: packedRange.buffer, offset: packedRange.offsetBytes, size: this.packedF32Bytes } },
              { binding: 1, resource: { buffer: output, offset: 0, size: neededBytes } },
              { binding: 2, resource: { buffer: this.stridedOut.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.stridedOut.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
          pass.end();
          return;
        }
    
        // No output view mapping: write packed logical output in requested precision.
        if (this.precision === "f16-storage") {
          const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
          if (outRanges.length === 1) {
            const outF32 = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
            const bg = this.device.createBindGroup({
              layout: this.f16out.bgl,
              entries: [
                { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.outTotalComplexLogical * 8 } },
                { binding: 1, resource: { buffer: outRanges[0].buffer, offset: outRanges[0].offsetBytes, size: this.outBytes } },
                { binding: 2, resource: { buffer: this.f16out.params, offset: 0, size: 16 } },
              ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.f16out.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
            pass.end();
            return;
          }
    
          const outF32 = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
          const dst = normalizeToContiguousRanges(packedF16View, 0, this.outBytes)[0];
          const bg = this.device.createBindGroup({
            layout: this.f16out.bgl,
            entries: [
              { binding: 0, resource: { buffer: outF32.buffer, offset: outF32.offsetBytes, size: this.outTotalComplexLogical * 8 } },
              { binding: 1, resource: { buffer: dst.buffer, offset: dst.offsetBytes, size: this.outBytes } },
              { binding: 2, resource: { buffer: this.f16out.params, offset: 0, size: 16 } },
            ],
          });
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(this.f16out.pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(this.outTotalComplexLogical / this.workgroupSize), 1, 1);
          pass.end();
          this.copier.unpack(commandEncoder, dst.buffer, dst.offsetBytes, outRanges);
          return;
        }
    
        const outRanges = normalizeToContiguousRanges(output, outputOffsetBytes, this.outBytes);
        const outF32 = normalizeToContiguousRanges(packedView, 0, this.packedF32Bytes)[0];
        if (outRanges.length === 1) {
          commandEncoder.copyBufferToBuffer(outF32.buffer, outF32.offsetBytes, outRanges[0].buffer, outRanges[0].offsetBytes, this.outBytes);
        } else {
          this.copier.unpack(commandEncoder, outF32.buffer, outF32.offsetBytes, outRanges);
        }
      }
    }
    
    
    exports['R2CPlan'] = R2CPlan;
  });

  __define('src/runtime/segmented_io.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { generateSegmentedCopyWGSL } = require('src/kernels/segmented_copy.js');
    const { pickWorkgroupSizeX } = require('src/utils/limits.js');
    
    function isGpuBuffer(x) {
      return x && !x.segments && typeof x.size === "number" && typeof x.destroy === "function";
    }
    
    function computeSegCap(device, reservedBindings = 1) {
      const max = device.limits?.maxStorageBuffersPerShaderStage ?? 8;
      return Math.max(0, Math.min(8, max - reservedBindings));
    }
    
    function normalizeToContiguousRanges(bufOrView, extraOffsetBytes, totalBytesWanted) {
      if (isGpuBuffer(bufOrView)) {
        if (extraOffsetBytes + totalBytesWanted > bufOrView.size) {
          throw new Error(`GPUBuffer too small: need ${extraOffsetBytes + totalBytesWanted} bytes, have ${bufOrView.size}`);
        }
        return [{ buffer: bufOrView, offsetBytes: extraOffsetBytes, sizeBytes: totalBytesWanted }];
      }
      const segments = bufOrView?.segments;
      if (!Array.isArray(segments) || segments.length === 0) throw new Error("Expected GPUBuffer or BufferView");
      const viewStart = bufOrView.logicalByteOffset ?? 0;
      const logicalByteOffset = viewStart + extraOffsetBytes;
      const lengthBytes = bufOrView.lengthBytes ?? segments.reduce((a, s) => a + s.sizeBytes, 0);
      // BufferView length is relative to its logicalByteOffset start.
      if (extraOffsetBytes + totalBytesWanted > lengthBytes) {
        throw new Error(
          `BufferView too small: need ${totalBytesWanted} bytes at offset ${extraOffsetBytes}, lengthBytes=${lengthBytes}, logicalByteOffset=${viewStart}`
        );
      }
    
      const out = [];
      let remaining = totalBytesWanted;
      let cursor = logicalByteOffset;
      let logicalPos = 0;
      for (const seg of segments) {
        const segStart = logicalPos;
        const segEnd = logicalPos + seg.sizeBytes;
        if (cursor >= segEnd) {
          logicalPos = segEnd;
          continue;
        }
        if (cursor < segStart) throw new Error("BufferView segments must be contiguous in logical space");
        const within = cursor - segStart;
        const take = Math.min(remaining, seg.sizeBytes - within);
        out.push({ buffer: seg.buffer, offsetBytes: seg.offsetBytes + within, sizeBytes: take });
        remaining -= take;
        cursor += take;
        logicalPos = segEnd;
        if (remaining === 0) break;
      }
      if (remaining !== 0) throw new Error("BufferView did not cover requested range");
      return out;
    }
    
    class SegmentedCopier {
      constructor(device, pipelineCache) {
        this.device = device;
        this.cache = pipelineCache;
        this.workgroupSize = pickWorkgroupSizeX(device.limits, 256);
        this.storageAlign = device.limits?.minStorageBufferOffsetAlignment ?? 256;
        // Tier-A cap: bind up to SEG_CAP segments + 1 contiguous buffer per shader stage.
        this.segCap = computeSegCap(device, 1);
    
        this._layout = device.createBindGroupLayout({
          entries: [
            // segment bindings 0..cap-1 (storage), plus contiguous binding cap, plus uniform cap+1
            ...Array.from({ length: this.segCap }, (_, i) => ({
              binding: i,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "read-only-storage" },
            })),
            {
              binding: this.segCap,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "storage" },
            },
            {
              binding: this.segCap + 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "uniform" },
            },
          ],
        });
        this._layoutUnpack = device.createBindGroupLayout({
          entries: [
            ...Array.from({ length: this.segCap }, (_, i) => ({
              binding: i,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "storage" },
            })),
            {
              binding: this.segCap,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "read-only-storage" },
            },
            {
              binding: this.segCap + 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "uniform" },
            },
          ],
        });
        this._plPack = device.createPipelineLayout({ bindGroupLayouts: [this._layout] });
        this._plUnpack = device.createPipelineLayout({ bindGroupLayouts: [this._layoutUnpack] });
    
        this._uniform = device.createBuffer({
          size: 16 + 4 * this.segCap * 2,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._dummy = device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // For unpack, missing segment bindings are writable storage; they must not alias each other.
        this._dummyRW = Array.from({ length: this.segCap }, () =>
          device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
        );
        device.queue.writeBuffer(this._dummy, 0, new Uint32Array([0]));
        for (const b of this._dummyRW) device.queue.writeBuffer(b, 0, new Uint32Array([0]));
      }
    
      destroy() {
        this._uniform.destroy();
        this._dummy.destroy();
        for (const b of this._dummyRW) b.destroy();
      }
    
      pack(commandEncoder, srcRanges, dstBuffer, dstOffsetBytes) {
        // srcRanges cover contiguous logical byte range, in order, sum sizes == totalBytes.
        const totalBytes = srcRanges.reduce((a, r) => a + r.sizeBytes, 0);
        if (totalBytes % 4 !== 0) throw new Error("SegmentedCopier only supports totalBytes multiple of 4");
        const totalWords = totalBytes >>> 2;
    
        if (srcRanges.length <= this.segCap) {
          const canTierA =
            dstOffsetBytes % this.storageAlign === 0 && srcRanges.every((r) => r.offsetBytes % this.storageAlign === 0);
          if (!canTierA) {
            // Fall back to Tier-B copies if bindings would violate offset alignment.
            let dst = dstOffsetBytes;
            for (const r of srcRanges) {
              commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
              dst += r.sizeBytes;
            }
            return;
          }
    
          const segSizesWords = new Uint32Array(this.segCap);
          const segPrefixWords = new Uint32Array(this.segCap);
          let prefix = 0;
          for (let i = 0; i < srcRanges.length; i++) {
            if (srcRanges[i].sizeBytes % 4 !== 0) throw new Error("Segment size must be multiple of 4");
            segSizesWords[i] = srcRanges[i].sizeBytes >>> 2;
            segPrefixWords[i] = prefix;
            prefix += segSizesWords[i];
          }
    
          const header = new Uint32Array([srcRanges.length, totalWords, 0, 0]);
          const ub = new Uint32Array(4 + this.segCap * 2);
          ub.set(header, 0);
          ub.set(segSizesWords, 4);
          ub.set(segPrefixWords, 4 + this.segCap);
          this.device.queue.writeBuffer(this._uniform, 0, ub);
    
          const code = generateSegmentedCopyWGSL({
            cap: this.segCap,
            direction: "pack",
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: this._plPack });
    
          const entries = [];
          for (let i = 0; i < this.segCap; i++) {
            const r = srcRanges[i];
            if (!r) {
              entries.push({ binding: i, resource: { buffer: this._dummy, offset: 0, size: 4 } });
            } else {
              entries.push({ binding: i, resource: { buffer: r.buffer, offset: r.offsetBytes, size: r.sizeBytes } });
            }
          }
          entries.push({ binding: this.segCap, resource: { buffer: dstBuffer, offset: dstOffsetBytes, size: totalBytes } });
          entries.push({ binding: this.segCap + 1, resource: { buffer: this._uniform, offset: 0, size: this._uniform.size } });
          const bg = this.device.createBindGroup({ layout: this._layout, entries });
    
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(totalWords / this.workgroupSize), 1, 1);
          pass.end();
          return;
        }
    
        // Tier B fallback: multiple GPU copy commands (still once per exec, not per stage)
        let dst = dstOffsetBytes;
        for (const r of srcRanges) {
          commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
          dst += r.sizeBytes;
        }
      }
    
      unpack(commandEncoder, srcBuffer, srcOffsetBytes, dstRanges) {
        const totalBytes = dstRanges.reduce((a, r) => a + r.sizeBytes, 0);
        if (totalBytes % 4 !== 0) throw new Error("SegmentedCopier only supports totalBytes multiple of 4");
        const totalWords = totalBytes >>> 2;
    
        if (dstRanges.length <= this.segCap) {
          const canTierA =
            srcOffsetBytes % this.storageAlign === 0 && dstRanges.every((r) => r.offsetBytes % this.storageAlign === 0);
          if (!canTierA) {
            // Fall back to Tier-B copies if bindings would violate offset alignment.
            let src = srcOffsetBytes;
            for (const r of dstRanges) {
              commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
              src += r.sizeBytes;
            }
            return;
          }
    
          const segSizesWords = new Uint32Array(this.segCap);
          const segPrefixWords = new Uint32Array(this.segCap);
          let prefix = 0;
          for (let i = 0; i < dstRanges.length; i++) {
            if (dstRanges[i].sizeBytes % 4 !== 0) throw new Error("Segment size must be multiple of 4");
            segSizesWords[i] = dstRanges[i].sizeBytes >>> 2;
            segPrefixWords[i] = prefix;
            prefix += segSizesWords[i];
          }
    
          const header = new Uint32Array([dstRanges.length, totalWords, 0, 0]);
          const ub = new Uint32Array(4 + this.segCap * 2);
          ub.set(header, 0);
          ub.set(segSizesWords, 4);
          ub.set(segPrefixWords, 4 + this.segCap);
          this.device.queue.writeBuffer(this._uniform, 0, ub);
    
          const code = generateSegmentedCopyWGSL({
            cap: this.segCap,
            direction: "unpack",
            workgroupSize: this.workgroupSize,
          });
          const pipeline = this.cache.getComputePipeline({ code, layout: this._plUnpack });
    
          const entries = [];
          for (let i = 0; i < this.segCap; i++) {
            const r = dstRanges[i];
            if (!r) {
              entries.push({ binding: i, resource: { buffer: this._dummyRW[i], offset: 0, size: 4 } });
            } else {
              entries.push({ binding: i, resource: { buffer: r.buffer, offset: r.offsetBytes, size: r.sizeBytes } });
            }
          }
          entries.push({ binding: this.segCap, resource: { buffer: srcBuffer, offset: srcOffsetBytes, size: totalBytes } });
          entries.push({ binding: this.segCap + 1, resource: { buffer: this._uniform, offset: 0, size: this._uniform.size } });
          const bg = this.device.createBindGroup({ layout: this._layoutUnpack, entries });
    
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(totalWords / this.workgroupSize), 1, 1);
          pass.end();
          return;
        }
    
        // Tier B fallback: multiple copies
        let src = srcOffsetBytes;
        for (const r of dstRanges) {
          commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
          src += r.sizeBytes;
        }
      }
    }
    
    exports['computeSegCap'] = computeSegCap;
    exports['normalizeToContiguousRanges'] = normalizeToContiguousRanges;
    exports['SegmentedCopier'] = SegmentedCopier;
  });

  __define('src/runtime/tensor_descriptor.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { prod } = require('src/runtime/common.js');
    
    function assertPositiveIntArray(shape, name) {
      if (!Array.isArray(shape) || shape.length < 1 || !shape.every((x) => Number.isInteger(x) && x > 0)) {
        throw new Error(`${name} must be an array of positive integers`);
      }
    }
    
    function contiguousStrides(shape) {
      assertPositiveIntArray(shape, "shape");
      const out = new Array(shape.length);
      let stride = 1;
      for (let d = 0; d < shape.length; d++) {
        out[d] = stride;
        stride *= shape[d];
      }
      return out;
    }
    
    function spanElements(shape, strides) {
      assertPositiveIntArray(shape, "shape");
      if (!Array.isArray(strides) || strides.length !== shape.length || !strides.every((x) => Number.isInteger(x) && x > 0)) {
        throw new Error("strides must be an array of positive integers matching shape rank");
      }
      let span = 1;
      for (let d = 0; d < shape.length; d++) span += (shape[d] - 1) * strides[d];
      return span;
    }
    
    function coordsFromLinear(i, shape, outCoords) {
      assertPositiveIntArray(shape, "shape");
      if (!Array.isArray(outCoords) || outCoords.length !== shape.length) {
        throw new Error("outCoords must be an array matching shape rank");
      }
      let rem = i;
      for (let d = 0; d < shape.length; d++) {
        const dim = shape[d];
        const c = rem % dim;
        outCoords[d] = c;
        rem = (rem - c) / dim;
      }
    }
    
    function linearFromCoords(coords, strides) {
      if (!Array.isArray(coords) || !Array.isArray(strides) || coords.length !== strides.length) {
        throw new Error("coords and strides must be arrays with the same length");
      }
      let idx = 0;
      for (let d = 0; d < coords.length; d++) idx += coords[d] * strides[d];
      return idx;
    }
    
    function linearFromCoordsShape(coords, shape) {
      return linearFromCoords(coords, contiguousStrides(shape));
    }
    
    function createTensorDescriptor({
      shape,
      strides = null,
      offsetElements = 0,
      batchStrideElements = null,
      name = "tensor",
    }) {
      assertPositiveIntArray(shape, `${name}.shape`);
      if (strides != null && (!Array.isArray(strides) || strides.length !== shape.length || !strides.every((x) => Number.isInteger(x) && x > 0))) {
        throw new Error(`${name}.strides must be null or an array of ${shape.length} positive integers`);
      }
      if (!Number.isInteger(offsetElements) || offsetElements < 0) {
        throw new Error(`${name}.offsetElements must be a non-negative integer`);
      }
      const resolvedStrides = strides ? strides.slice() : contiguousStrides(shape);
      const resolvedSpan = spanElements(shape, resolvedStrides);
      const defaultBatchStride = resolvedSpan;
      const resolvedBatchStride = batchStrideElements == null ? defaultBatchStride : batchStrideElements;
      if (!Number.isInteger(resolvedBatchStride) || resolvedBatchStride < resolvedSpan) {
        throw new Error(`${name}.batchStrideElements must be an integer >= ${resolvedSpan}`);
      }
      return {
        name,
        shape: shape.slice(),
        strides: resolvedStrides,
        spanElements: resolvedSpan,
        offsetElements: offsetElements | 0,
        batchStrideElements: resolvedBatchStride | 0,
        logicalElementsPerBatch: prod(shape),
        usesCustomStrides: !!strides,
        isContiguous:
          !strides &&
          offsetElements === 0 &&
          resolvedBatchStride === prod(shape),
      };
    }
    
    function requiredElementsForBatchRange(desc, { runtimeExtraElements = 0, batchStart = 0, batchCount = 1 } = {}) {
      if (!Number.isInteger(runtimeExtraElements) || runtimeExtraElements < 0) {
        throw new Error("runtimeExtraElements must be a non-negative integer");
      }
      if (!Number.isInteger(batchStart) || batchStart < 0) {
        throw new Error("batchStart must be a non-negative integer");
      }
      if (!Number.isInteger(batchCount) || batchCount < 0) {
        throw new Error("batchCount must be a non-negative integer");
      }
      if (!desc || !Number.isInteger(desc.offsetElements) || !Number.isInteger(desc.batchStrideElements) || !Number.isInteger(desc.spanElements)) {
        throw new Error("desc must be a tensor descriptor from createTensorDescriptor");
      }
      const lastBatch = batchStart + Math.max(0, batchCount - 1);
      return desc.offsetElements + runtimeExtraElements + lastBatch * desc.batchStrideElements + desc.spanElements;
    }
    
    function requiredBytesForBatchRange(
      desc,
      { bytesPerElement, runtimeExtraElements = 0, batchStart = 0, batchCount = 1 } = {}
    ) {
      if (!Number.isInteger(bytesPerElement) || bytesPerElement <= 0) {
        throw new Error("bytesPerElement must be a positive integer");
      }
      return requiredElementsForBatchRange(desc, { runtimeExtraElements, batchStart, batchCount }) * bytesPerElement;
    }
    
    exports['contiguousStrides'] = contiguousStrides;
    exports['spanElements'] = spanElements;
    exports['coordsFromLinear'] = coordsFromLinear;
    exports['linearFromCoords'] = linearFromCoords;
    exports['linearFromCoordsShape'] = linearFromCoordsShape;
    exports['createTensorDescriptor'] = createTensorDescriptor;
    exports['requiredElementsForBatchRange'] = requiredElementsForBatchRange;
    exports['requiredBytesForBatchRange'] = requiredBytesForBatchRange;
  });

  __define('src/runtime/workspace.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    const { BufferView } = require('src/utils/buffer_view.js');
    
    function viewFromArena(arena, offsetBytes, lengthBytes) {
      if (arena instanceof BufferView) {
        return new BufferView({
          segments: arena.segments,
          logicalByteOffset: arena.logicalByteOffset + offsetBytes,
          lengthBytes,
        });
      }
      return BufferView.fromBuffer(arena, offsetBytes, lengthBytes);
    }
    
    function createInternalArena(device, sizeBytes) {
      if (sizeBytes <= 0) return null;
      return device.createBuffer({
        size: sizeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
    
    
    exports['viewFromArena'] = viewFromArena;
    exports['createInternalArena'] = createInternalArena;
  });

  __define('src/runtime/zero_pad.js', function(require, exports, module){
    // Copyright (c) 2026 Maksim Eremenko
    
    function parseOptionalBoundArray(v, rank, name, defaults) {
      if (v == null) return defaults.slice();
      if (!Array.isArray(v) || v.length !== rank || !v.every((x) => Number.isInteger(x))) {
        throw new Error(`${name} must be an array of ${rank} integers`);
      }
      return v.map((x) => x | 0);
    }
    
    function normalizeStage(rank, shape, stage, name) {
      if (!stage) return null;
      if (typeof stage !== "object") {
        throw new Error(`${name} must be an object with optional start/end arrays`);
      }
      const src = stage.range && typeof stage.range === "object" ? stage.range : stage;
      const start = parseOptionalBoundArray(src.start ?? null, rank, `${name}.start`, new Array(rank).fill(0));
      const end = parseOptionalBoundArray(src.end ?? null, rank, `${name}.end`, shape);
    
      for (let d = 0; d < rank; d++) {
        if (start[d] < 0) throw new Error(`${name}.start[${d}] must be >= 0; got ${start[d]}`);
        if (end[d] < 0) throw new Error(`${name}.end[${d}] must be >= 0; got ${end[d]}`);
        if (start[d] > end[d]) throw new Error(`${name}: start[${d}] must be <= end[${d}]`);
        if (end[d] > shape[d]) throw new Error(`${name}.end[${d}] must be <= shape[${d}] (${shape[d]}); got ${end[d]}`);
      }
    
      const full = start.every((s) => s === 0) && end.every((e, d) => e === shape[d]);
      return full ? null : { start, end };
    }
    
    function normalizeZeroPad(rank, shape, zeroPad = null, name = "zeroPad") {
      if (!zeroPad) return { read: null, write: null };
      if (typeof zeroPad !== "object") {
        throw new Error(`${name} must be an object with optional read/write stage configs`);
      }
      return {
        read: normalizeStage(rank, shape, zeroPad.read ?? null, `${name}.read`),
        write: normalizeStage(rank, shape, zeroPad.write ?? null, `${name}.write`),
      };
    }
    
    
    exports['normalizeZeroPad'] = normalizeZeroPad;
  });

  __define('src/utils/buffer_view.js', function(require, exports, module){
    /**
     * BufferView describes a logical byte range potentially split across segments.
     * This is the public abstraction requested for buffer splitting.
     *
     * NOTE:
     * - Plans may operate on a contiguous internal workspace. When input/output are multi-segment, the library
     *   will pack/unpack once per exec (Tier A: WGSL segmented copy for small segment counts; Tier B: multiple
     *   GPU copyBufferToBuffer ops for larger segment counts). If the logical transform domain exceeds
     *   device.limits.maxStorageBufferBindingSize, plan creation rejects it (no out-of-core in current implementation).
     */
    class BufferView {
      /**
       * @param {object} opts
       * @param {{buffer: GPUBuffer, offsetBytes: number, sizeBytes: number}[]} opts.segments
       * @param {number} [opts.logicalByteOffset=0]
       * @param {number} opts.lengthBytes
       */
      constructor({ segments, logicalByteOffset = 0, lengthBytes }) {
        if (!Array.isArray(segments) || segments.length === 0) throw new Error("BufferView.segments must be a non-empty array");
        if (!Number.isInteger(logicalByteOffset) || logicalByteOffset < 0) throw new Error("BufferView.logicalByteOffset must be a non-negative integer");
        if (!Number.isInteger(lengthBytes) || lengthBytes <= 0) throw new Error("BufferView.lengthBytes must be a positive integer");
        for (const s of segments) {
          if (!s?.buffer) throw new Error("BufferView segment missing buffer");
          if (!Number.isInteger(s.offsetBytes) || s.offsetBytes < 0) throw new Error("BufferView segment offsetBytes must be non-negative integer");
          if (!Number.isInteger(s.sizeBytes) || s.sizeBytes <= 0) throw new Error("BufferView segment sizeBytes must be positive integer");
          if (s.offsetBytes + s.sizeBytes > s.buffer.size) {
            throw new Error(`BufferView segment out of bounds: offsetBytes+sizeBytes=${s.offsetBytes + s.sizeBytes} > buffer.size=${s.buffer.size}`);
          }
        }
        this.segments = segments;
        this.logicalByteOffset = logicalByteOffset;
        this.lengthBytes = lengthBytes;
      }
    
      static fromBuffer(buffer, offsetBytes = 0, lengthBytes = buffer.size - offsetBytes) {
        return new BufferView({
          segments: [{ buffer, offsetBytes, sizeBytes: lengthBytes }],
          logicalByteOffset: 0,
          lengthBytes,
        });
      }
    }
    
    
    exports['BufferView'] = BufferView;
  });

  __define('src/utils/factors.js', function(require, exports, module){
    const SUPPORTED_RADICES = [2, 3, 4, 5, 7, 8, 11, 13];
    
    function factorizeSupportedRadices(n) {
      if (!Number.isInteger(n) || n <= 0) throw new Error(`factorizeSupportedRadices: n must be positive int, got ${n}`);
      const allowed = [13, 11, 8, 7, 5, 4, 3, 2];
      const out = [];
      let x = n;
      for (const r of allowed) {
        while (x % r === 0) {
          out.push(r);
          x = x / r;
        }
      }
      return x === 1 ? out : null;
    }
    
    function isPrime(n) {
      if (!Number.isInteger(n) || n < 2) return false;
      if (n % 2 === 0) return n === 2;
      for (let d = 3; d * d <= n; d += 2) {
        if (n % d === 0) return false;
      }
      return true;
    }
    
    function gcd(a, b) {
      let x = Math.abs(a | 0);
      let y = Math.abs(b | 0);
      while (y !== 0) {
        const t = x % y;
        x = y;
        y = t;
      }
      return x;
    }
    
    function modPow(base, exp, mod) {
      let b = base % mod;
      let e = exp;
      let res = 1;
      while (e > 0) {
        if (e & 1) res = (res * b) % mod;
        b = (b * b) % mod;
        e >>= 1;
      }
      return res;
    }
    
    function primeFactors(n) {
      const out = [];
      let x = n;
      for (let d = 2; d * d <= x; d += d === 2 ? 1 : 2) {
        if (x % d === 0) {
          out.push(d);
          while (x % d === 0) x = x / d;
        }
      }
      if (x > 1) out.push(x);
      return out;
    }
    
    function primitiveRootPrime(p) {
      if (!isPrime(p)) throw new Error(`primitiveRootPrime: p must be prime, got ${p}`);
      const phi = p - 1;
      const factors = primeFactors(phi);
      for (let g = 2; g < p; g++) {
        let ok = true;
        for (const q of factors) {
          if (modPow(g, phi / q, p) === 1) {
            ok = false;
            break;
          }
        }
        if (ok) return g;
      }
      throw new Error(`primitiveRootPrime: failed for p=${p}`);
    }
    
    function nextSmoothAtLeast(minN) {
      // Find smallest n >= minN with factorization by supported radices.
      // For the sizes in tests this is fast enough.
      if (!Number.isInteger(minN) || minN <= 0) throw new Error(`nextSmoothAtLeast: minN must be positive int, got ${minN}`);
      let n = minN;
      for (;;) {
        if (factorizeSupportedRadices(n)) return n;
        n++;
        if (n - minN > 1_000_000) {
          // Hard stop to avoid accidental infinite work.
          // In practice we can always fall back to nextPow2, but keep this bounded.
          return nextPow2(minN);
        }
      }
    }
    
    function nextPow2(n) {
      if (!Number.isInteger(n) || n <= 0) throw new Error(`nextPow2: n must be positive int, got ${n}`);
      let x = 1;
      while (x < n) x <<= 1;
      return x;
    }
    
    
    exports['SUPPORTED_RADICES'] = SUPPORTED_RADICES;
    exports['factorizeSupportedRadices'] = factorizeSupportedRadices;
    exports['isPrime'] = isPrime;
    exports['gcd'] = gcd;
    exports['modPow'] = modPow;
    exports['primeFactors'] = primeFactors;
    exports['primitiveRootPrime'] = primitiveRootPrime;
    exports['nextSmoothAtLeast'] = nextSmoothAtLeast;
    exports['nextPow2'] = nextPow2;
  });

  __define('src/utils/hash.js', function(require, exports, module){
    function fnv1a32Bytes(u8) {
      let h = 0x811c9dc5;
      for (let i = 0; i < u8.length; i++) {
        h ^= u8[i];
        h = Math.imul(h, 0x01000193);
      }
      // unsigned
      return h >>> 0;
    }
    
    function hashFloat32Array(a) {
      if (!(a instanceof Float32Array)) throw new Error("hashFloat32Array expects Float32Array");
      return fnv1a32Bytes(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
    }
    
    
    exports['fnv1a32Bytes'] = fnv1a32Bytes;
    exports['hashFloat32Array'] = hashFloat32Array;
  });

  __define('src/utils/limits.js', function(require, exports, module){
    function formatDeviceLimits(limits) {
      const m = limits?.maxComputeWorkgroupsPerDimension;
      return JSON.stringify(
        {
          maxBufferSize: limits?.maxBufferSize,
          maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize,
          maxStorageBuffersPerShaderStage: limits?.maxStorageBuffersPerShaderStage,
          minStorageBufferOffsetAlignment: limits?.minStorageBufferOffsetAlignment,
          maxComputeInvocationsPerWorkgroup: limits?.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupSizeX: limits?.maxComputeWorkgroupSizeX,
          maxComputeWorkgroupSizeY: limits?.maxComputeWorkgroupSizeY,
          maxComputeWorkgroupSizeZ: limits?.maxComputeWorkgroupSizeZ,
          maxComputeWorkgroupStorageSize: limits?.maxComputeWorkgroupStorageSize,
          maxComputeWorkgroupsPerDimension: m ? [m[0], m[1], m[2]] : undefined,
        },
        null,
        2
      );
    }
    
    function pickWorkgroupSizeX(limits, preferred = 256) {
      const maxX = limits?.maxComputeWorkgroupSizeX ?? preferred;
      const maxInv = limits?.maxComputeInvocationsPerWorkgroup ?? preferred;
      return Math.max(1, Math.min(preferred, maxX, maxInv));
    }
    
    
    exports['formatDeviceLimits'] = formatDeviceLimits;
    exports['pickWorkgroupSizeX'] = pickWorkgroupSizeX;
  });

  __define('src/utils/math.js', function(require, exports, module){
    function isPowerOfTwo(n) {
      return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
    }
    
    function reverseBits(x, bits) {
      let y = 0;
      for (let i = 0; i < bits; i++) {
        y = (y << 1) | (x & 1);
        x >>>= 1;
      }
      return y;
    }
    
    function normalizeScaleFactor({ normalize, direction, nTotal }) {
      if (normalize === "none") return 1.0;
      if (normalize === "unitary") return 1.0 / Math.sqrt(nTotal);
      if (normalize === "backward") return direction === "inverse" ? 1.0 / nTotal : 1.0;
      throw new Error(`Unknown normalize mode: ${normalize}`);
    }
    
    /**
     * Reference radix-2 FFT (DIT with bit reversal), interleaved complex<f32>.
     * Returns a new Float32Array of length 2*N.
     */
    function fft1dRefInterleaved(input, N, direction) {
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (!isPowerOfTwo(N) || N < 2) throw new Error(`N must be power-of-two >= 2; got ${N}`);
      if (input.length !== 2 * N) throw new Error(`input length must be 2*N; got ${input.length}, expected ${2 * N}`);
      if (direction !== "forward" && direction !== "inverse") throw new Error(`direction must be forward|inverse; got ${direction}`);
    
      const out = new Float32Array(input); // copy
      const bits = Math.round(Math.log2(N));
    
      // Bit-reversal permutation.
      for (let i = 0; i < N; i++) {
        const j = reverseBits(i, bits);
        if (j > i) {
          const ia = 2 * i;
          const ja = 2 * j;
          const tr = out[ia];
          const ti = out[ia + 1];
          out[ia] = out[ja];
          out[ia + 1] = out[ja + 1];
          out[ja] = tr;
          out[ja + 1] = ti;
        }
      }
    
      const sign = direction === "forward" ? -1.0 : 1.0;
    
      for (let len = 2; len <= N; len <<= 1) {
        const half = len >>> 1;
        const ang = (sign * 2.0 * Math.PI) / len;
        const wlenRe = Math.cos(ang);
        const wlenIm = Math.sin(ang);
    
        for (let i = 0; i < N; i += len) {
          let wRe = 1.0;
          let wIm = 0.0;
          for (let j = 0; j < half; j++) {
            const a = 2 * (i + j);
            const b = 2 * (i + j + half);
    
            const uRe = out[a];
            const uIm = out[a + 1];
            const vRe0 = out[b];
            const vIm0 = out[b + 1];
    
            // v = v0 * w
            const vRe = vRe0 * wRe - vIm0 * wIm;
            const vIm = vRe0 * wIm + vIm0 * wRe;
    
            out[a] = uRe + vRe;
            out[a + 1] = uIm + vIm;
            out[b] = uRe - vRe;
            out[b + 1] = uIm - vIm;
    
            // w *= wlen
            const nextWRe = wRe * wlenRe - wIm * wlenIm;
            const nextWIm = wRe * wlenIm + wIm * wlenRe;
            wRe = nextWRe;
            wIm = nextWIm;
          }
        }
      }
    
      return out;
    }
    
    function fftNdRefInterleaved(input, shape, direction, normalize = "none") {
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (!Array.isArray(shape) || shape.length < 1) throw new Error(`shape must have length >=1; got ${shape}`);
      if (!shape.every((n) => Number.isInteger(n) && n > 0)) throw new Error(`shape must contain positive integers; got ${JSON.stringify(shape)}`);
      if (direction !== "forward" && direction !== "inverse") throw new Error(`direction must be forward|inverse; got ${direction}`);
    
      const nTotal = shape.reduce((a, b) => a * b, 1);
      if (input.length !== 2 * nTotal) throw new Error(`input length must be 2*product(shape); got ${input.length}, expected ${2 * nTotal}`);
      if (!shape.every((n) => isPowerOfTwo(n))) {
        throw new Error(`shape must be power-of-two per dim; got ${JSON.stringify(shape)}`);
      }
    
      let data = new Float32Array(input);
      const rank = shape.length;
      const strides = new Array(rank);
      strides[0] = 1;
      for (let i = 1; i < rank; i++) strides[i] = strides[i - 1] * shape[i - 1];
    
      const lineBase = (line, axis) => {
        let rem = line;
        let base = 0;
        for (let d = 0; d < rank; d++) {
          if (d === axis) continue;
          const c = rem % shape[d];
          rem = Math.floor(rem / shape[d]);
          base += c * strides[d];
        }
        return base;
      };
    
      for (let axis = 0; axis < rank; axis++) {
        const N = shape[axis];
        const stride = strides[axis];
        const lineCount = nTotal / N;
        const line = new Float32Array(2 * N);
    
        for (let l = 0; l < lineCount; l++) {
          const base = lineBase(l, axis);
          for (let p = 0; p < N; p++) {
            const idx = 2 * (base + p * stride);
            line[2 * p] = data[idx];
            line[2 * p + 1] = data[idx + 1];
          }
          const out = fft1dRefInterleaved(line, N, direction);
          for (let p = 0; p < N; p++) {
            const idx = 2 * (base + p * stride);
            data[idx] = out[2 * p];
            data[idx + 1] = out[2 * p + 1];
          }
        }
      }
    
      const scale = normalizeScaleFactor({ normalize, direction, nTotal });
      if (scale !== 1.0) {
        for (let i = 0; i < data.length; i++) data[i] = data[i] * scale;
      }
    
      return data;
    }
    
    function randomComplexInterleaved(lengthComplex, rng = Math.random) {
      const out = new Float32Array(2 * lengthComplex);
      for (let i = 0; i < lengthComplex; i++) {
        // centered-ish small range to avoid overflow and to keep errors readable
        out[2 * i] = (rng() * 2 - 1) * 0.5;
        out[2 * i + 1] = (rng() * 2 - 1) * 0.5;
      }
      return out;
    }
    
    function dft1dRefInterleaved(input, N, direction) {
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (!Number.isInteger(N) || N <= 0) throw new Error(`N must be positive int; got ${N}`);
      if (input.length !== 2 * N) throw new Error(`input length must be 2*N; got ${input.length}, expected ${2 * N}`);
      if (direction !== "forward" && direction !== "inverse") throw new Error(`direction must be forward|inverse; got ${direction}`);
    
      const out = new Float32Array(2 * N);
      const sign = direction === "forward" ? -1.0 : 1.0;
      for (let k = 0; k < N; k++) {
        let re = 0;
        let im = 0;
        for (let n = 0; n < N; n++) {
          const ang = (sign * 2.0 * Math.PI * n * k) / N;
          const c = Math.cos(ang);
          const s = Math.sin(ang);
          const xr = input[2 * n];
          const xi = input[2 * n + 1];
          re += xr * c - xi * s;
          im += xr * s + xi * c;
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
      }
      return out;
    }
    
    function fftNdRefAnySizeInterleaved(input, shape, direction, normalize = "none") {
      if (!Array.isArray(shape) || shape.length < 1) throw new Error(`shape must have length >=1; got ${shape}`);
      if (!shape.every((n) => Number.isInteger(n) && n > 0)) throw new Error(`shape must contain positive integers; got ${JSON.stringify(shape)}`);
      const nTotal = shape.reduce((a, b) => a * b, 1);
      if (input.length !== 2 * nTotal) throw new Error(`input length must be 2*product(shape)`);
      let data = new Float32Array(input);
      const rank = shape.length;
      const strides = new Array(rank);
      strides[0] = 1;
      for (let i = 1; i < rank; i++) strides[i] = strides[i - 1] * shape[i - 1];
    
      const lineBase = (line, axis) => {
        let rem = line;
        let base = 0;
        for (let d = 0; d < rank; d++) {
          if (d === axis) continue;
          const c = rem % shape[d];
          rem = Math.floor(rem / shape[d]);
          base += c * strides[d];
        }
        return base;
      };
    
      for (let axis = 0; axis < rank; axis++) {
        const N = shape[axis];
        const stride = strides[axis];
        const lineCount = nTotal / N;
        const line = new Float32Array(2 * N);
    
        for (let l = 0; l < lineCount; l++) {
          const base = lineBase(l, axis);
          for (let p = 0; p < N; p++) {
            const idx = 2 * (base + p * stride);
            line[2 * p] = data[idx];
            line[2 * p + 1] = data[idx + 1];
          }
          const out = dft1dRefInterleaved(line, N, direction);
          for (let p = 0; p < N; p++) {
            const idx = 2 * (base + p * stride);
            data[idx] = out[2 * p];
            data[idx + 1] = out[2 * p + 1];
          }
        }
      }
    
      const scale = normalizeScaleFactor({ normalize, direction, nTotal });
      if (scale !== 1.0) {
        for (let i = 0; i < data.length; i++) data[i] *= scale;
      }
      return data;
    }
    
    function r2cRefPackedInterleaved(inputReal, N, direction = "forward", normalize = "none") {
      if (!(inputReal instanceof Float32Array)) throw new Error("inputReal must be Float32Array");
      if (!Number.isInteger(N) || N <= 1) throw new Error(`N must be int>=2; got ${N}`);
      if (inputReal.length !== N) throw new Error(`inputReal length must be N`);
      if (direction !== "forward") throw new Error("r2cRefPackedInterleaved is forward-only");
    
      const complex = new Float32Array(2 * N);
      for (let i = 0; i < N; i++) complex[2 * i] = inputReal[i];
      const full = fftNdRefAnySizeInterleaved(complex, [N], "forward", "none");
      const outLen = Math.floor(N / 2) + 1;
      const out = new Float32Array(2 * outLen);
      for (let k = 0; k < outLen; k++) {
        out[2 * k] = full[2 * k];
        out[2 * k + 1] = full[2 * k + 1];
      }
      const scale = normalizeScaleFactor({ normalize, direction: "forward", nTotal: N });
      if (scale !== 1.0) {
        for (let i = 0; i < out.length; i++) out[i] *= scale;
      }
      return out;
    }
    
    function c2rRefFromPackedInterleaved(inputPacked, N, normalize = "none") {
      if (!(inputPacked instanceof Float32Array)) throw new Error("inputPacked must be Float32Array");
      const outLen = Math.floor(N / 2) + 1;
      if (inputPacked.length !== 2 * outLen) throw new Error(`inputPacked length must be 2*(N/2+1)`);
    
      const full = new Float32Array(2 * N);
      for (let k = 0; k < outLen; k++) {
        full[2 * k] = inputPacked[2 * k];
        full[2 * k + 1] = inputPacked[2 * k + 1];
      }
      const kMaxMirror = N % 2 === 0 ? (N / 2) - 1 : Math.floor(N / 2);
      for (let k = 1; k <= kMaxMirror; k++) {
        // Hermitian: X[N-k] = conj(X[k])
        full[2 * (N - k)] = full[2 * k];
        full[2 * (N - k) + 1] = -full[2 * k + 1];
      }
      // k=0 and k=N/2 are self-conjugate; imag should be ~0.
    
      const time = fftNdRefAnySizeInterleaved(full, [N], "inverse", "none");
      const out = new Float32Array(N);
      for (let n = 0; n < N; n++) out[n] = time[2 * n];
    
      const scale = normalizeScaleFactor({ normalize, direction: "inverse", nTotal: N });
      // Our normalizeScaleFactor matches library policy; for inverse "backward" is 1/N.
      // The inverse DFT itself is unnormalized, so apply scale (including 1/N for backward).
      if (scale !== 1.0) {
        for (let i = 0; i < out.length; i++) out[i] *= scale;
      }
      return out;
    }
    
    function dct2Ref(input, N, direction = "forward") {
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (input.length !== N) throw new Error("length mismatch");
      const out = new Float32Array(N);
      if (direction === "forward") {
        for (let k = 0; k < N; k++) {
          let sum = 0;
          for (let n = 0; n < N; n++) {
            sum += input[n] * Math.cos((Math.PI / N) * (n + 0.5) * k);
          }
          out[k] = sum;
        }
        return out;
      }
      // inverse (DCT-III up to scale): x[n] = X[0]/2 + sum_{k=1}^{N-1} X[k] cos(pi/N k (n+0.5))
      for (let n = 0; n < N; n++) {
        let sum = input[0] * 0.5;
        for (let k = 1; k < N; k++) {
          sum += input[k] * Math.cos((Math.PI / N) * (n + 0.5) * k);
        }
        out[n] = sum;
      }
      return out;
    }
    
    function dct3Ref(input, N, direction = "forward") {
      // DCT-III forward is inverse of DCT-II up to scale; we provide both directions.
      return dct2Ref(input, N, direction === "forward" ? "inverse" : "forward");
    }
    
    function dct4Ref(input, N) {
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (input.length !== N) throw new Error("length mismatch");
      const out = new Float32Array(N);
      for (let k = 0; k < N; k++) {
        let sum = 0;
        for (let n = 0; n < N; n++) {
          sum += input[n] * Math.cos((Math.PI / N) * (n + 0.5) * (k + 0.5));
        }
        out[k] = sum;
      }
      return out;
    }
    
    function dst1Ref(input, N) {
      // N-point DST-I (N>=2): X[k] = sum_{n=0..N-1} x[n] sin(pi*(n+1)*(k+1)/(N+1))
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (input.length !== N) throw new Error("length mismatch");
      if (N < 2) throw new Error("N must be >= 2");
      const out = new Float32Array(N);
      for (let k = 0; k < N; k++) {
        let sum = 0;
        for (let n = 0; n < N; n++) {
          sum += input[n] * Math.sin((Math.PI * (n + 1) * (k + 1)) / (N + 1));
        }
        out[k] = sum;
      }
      return out;
    }
    
    function dst2Ref(input, N, direction = "forward") {
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (input.length !== N) throw new Error("length mismatch");
      const out = new Float32Array(N);
      if (direction === "forward") {
        for (let k = 0; k < N; k++) {
          let sum = 0;
          for (let n = 0; n < N; n++) {
            sum += input[n] * Math.sin((Math.PI / N) * (n + 0.5) * (k + 1));
          }
          out[k] = sum;
        }
        return out;
      }
      // inverse (DST-III up to scale): x[n] = 0.5*(-1)^n*X[N-1] + sum_{k=0..N-2} X[k] sin(pi/N*(n+0.5)*(k+1))
      for (let n = 0; n < N; n++) {
        let sum = (n % 2 === 0 ? 0.5 : -0.5) * input[N - 1];
        for (let k = 0; k < N - 1; k++) {
          sum += input[k] * Math.sin((Math.PI / N) * (n + 0.5) * (k + 1));
        }
        out[n] = sum;
      }
      return out;
    }
    
    function dst3Ref(input, N, direction = "forward") {
      // DST-III forward is inverse of DST-II up to scale; we provide both directions.
      return dst2Ref(input, N, direction === "forward" ? "inverse" : "forward");
    }
    
    function dst4Ref(input, N) {
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (input.length !== N) throw new Error("length mismatch");
      const out = new Float32Array(N);
      for (let k = 0; k < N; k++) {
        let sum = 0;
        for (let n = 0; n < N; n++) {
          sum += input[n] * Math.sin((Math.PI / N) * (n + 0.5) * (k + 0.5));
        }
        out[k] = sum;
      }
      return out;
    }
    
    function dct1Ref(input, N) {
      // N-point DCT-I (N>=2): X[k] = x[0] + (-1)^k x[N-1] + 2*sum_{n=1..N-2} x[n] cos(pi*n*k/(N-1))
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (input.length !== N) throw new Error("length mismatch");
      if (N < 2) throw new Error("N must be >= 2");
      const out = new Float32Array(N);
      for (let k = 0; k < N; k++) {
        let sum = input[0] + (k % 2 === 0 ? 1 : -1) * input[N - 1];
        for (let n = 1; n <= N - 2; n++) {
          sum += 2.0 * input[n] * Math.cos((Math.PI * n * k) / (N - 1));
        }
        out[k] = sum;
      }
      return out;
    }
    
    function conv2dRef({
      input,
      kernel,
      Hout,
      Wout,
      Hin,
      Win,
      k,
      pad,
      complex,
      complexKernel,
    }) {
      const out = complex ? new Float32Array(2 * Hout * Wout) : new Float32Array(Hout * Wout);
      for (let y = 0; y < Hout; y++) {
        for (let x = 0; x < Wout; x++) {
          let accRe = 0;
          let accIm = 0;
          for (let ky = 0; ky < k; ky++) {
            for (let kx = 0; kx < k; kx++) {
              const iy = y + ky - pad[0];
              const ix = x + kx - pad[2];
              if (iy < 0 || ix < 0 || iy >= Hin || ix >= Win) continue;
    
              const inIdx = iy * Win + ix;
              const krnIdx = ky * k + kx;
    
              if (complex) {
                const xr = input[2 * inIdx];
                const xi = input[2 * inIdx + 1];
                if (complexKernel) {
                  const wr = kernel[2 * krnIdx];
                  const wi = kernel[2 * krnIdx + 1];
                  accRe += xr * wr - xi * wi;
                  accIm += xr * wi + xi * wr;
                } else {
                  const w = kernel[krnIdx];
                  accRe += xr * w;
                  accIm += xi * w;
                }
              } else {
                const x0 = input[inIdx];
                const w = kernel[krnIdx];
                accRe += x0 * w;
              }
            }
          }
          const outIdx = y * Wout + x;
          if (complex) {
            out[2 * outIdx] = accRe;
            out[2 * outIdx + 1] = accIm;
          } else {
            out[outIdx] = accRe;
          }
        }
      }
      return out;
    }
    
    function fftConvRef({
      input,
      kernel,
      shape,
      batch = 1,
      mode = "convolution",
      boundary = "circular",
      kernelShape = null,
    }) {
      if (!(input instanceof Float32Array)) throw new Error("input must be Float32Array");
      if (!(kernel instanceof Float32Array)) throw new Error("kernel must be Float32Array");
      if (!Array.isArray(shape) || shape.length < 1) throw new Error("shape must be rank >= 1");
      if (!shape.every((x) => Number.isInteger(x) && x > 0)) throw new Error("shape must contain positive integers");
      if (!Number.isInteger(batch) || batch <= 0) throw new Error("batch must be positive int");
      if (mode !== "convolution" && mode !== "correlation") throw new Error("mode must be convolution|correlation");
      if (!["circular", "linear-full", "linear-same", "linear-valid"].includes(boundary)) {
        throw new Error('boundary must be "circular"|"linear-full"|"linear-same"|"linear-valid"');
      }
    
      const rank = shape.length;
      const kShape = kernelShape == null ? shape.slice() : kernelShape;
      if (!Array.isArray(kShape) || kShape.length !== rank || !kShape.every((x) => Number.isInteger(x) && x > 0)) {
        throw new Error(`kernelShape must be an array of ${rank} positive integers`);
      }
    
      const prodShape = (s) => s.reduce((a, b) => a * b, 1);
      const makeStrides = (s) => {
        const strides = new Array(s.length);
        let acc = 1;
        for (let d = 0; d < s.length; d++) {
          strides[d] = acc;
          acc *= s[d] | 0;
        }
        return strides;
      };
      const embedAtOffset = (dst, dstShape, src, srcShape, offset) => {
        const srcN = prodShape(srcShape);
        const dstStrides = makeStrides(dstShape);
        for (let i = 0; i < srcN; i++) {
          let rem = i;
          let dstIdx = 0;
          for (let d = 0; d < srcShape.length; d++) {
            const dim = srcShape[d];
            const coord = rem % dim;
            rem = (rem - coord) / dim;
            dstIdx += (offset[d] + coord) * dstStrides[d];
          }
          dst[2 * dstIdx] = src[2 * i];
          dst[2 * dstIdx + 1] = src[2 * i + 1];
        }
      };
      const extractAtOffset = (src, srcShape, dstShape, offset) => {
        const dstN = prodShape(dstShape);
        const out = new Float32Array(2 * dstN);
        const srcStrides = makeStrides(srcShape);
        for (let i = 0; i < dstN; i++) {
          let rem = i;
          let srcIdx = 0;
          for (let d = 0; d < dstShape.length; d++) {
            const dim = dstShape[d];
            const coord = rem % dim;
            rem = (rem - coord) / dim;
            srcIdx += (offset[d] + coord) * srcStrides[d];
          }
          out[2 * i] = src[2 * srcIdx];
          out[2 * i + 1] = src[2 * srcIdx + 1];
        }
        return out;
      };
    
      const inputN = prodShape(shape);
      const kernelN = prodShape(kShape);
      if (input.length !== 2 * inputN * batch) throw new Error(`input length must be ${2 * inputN * batch}`);
      if (kernel.length !== 2 * kernelN) throw new Error(`kernel length must be ${2 * kernelN}`);
    
      if (boundary === "circular") {
        for (let d = 0; d < rank; d++) {
          if (kShape[d] > shape[d]) {
            throw new Error(`kernelShape[${d}] must be <= shape[${d}] for circular boundary`);
          }
        }
      }
    
      const fftShape = boundary === "circular" ? shape.slice() : shape.map((n, d) => n + kShape[d] - 1);
      let outShape;
      let outOffset;
      if (boundary === "circular") {
        outShape = shape.slice();
        outOffset = new Array(rank).fill(0);
      } else if (boundary === "linear-full") {
        outShape = fftShape.slice();
        outOffset = new Array(rank).fill(0);
      } else if (boundary === "linear-same") {
        outShape = shape.slice();
        outOffset = kShape.map((n) => Math.floor((n - 1) / 2));
      } else {
        // linear-valid
        outShape = shape.map((n, d) => n - kShape[d] + 1);
        for (let d = 0; d < rank; d++) {
          if (outShape[d] <= 0) {
            throw new Error(`linear-valid requires kernelShape[${d}] <= shape[${d}]`);
          }
        }
        outOffset = kShape.map((n) => n - 1);
      }
    
      const fftN = prodShape(fftShape);
      const outN = prodShape(outShape);
    
      const kPad = new Float32Array(2 * fftN);
      embedAtOffset(kPad, fftShape, kernel, kShape, new Array(rank).fill(0));
      const kf = fftNdRefAnySizeInterleaved(kPad, fftShape, "forward", "none");
      const out = new Float32Array(2 * outN * batch);
    
      for (let b = 0; b < batch; b++) {
        const x = input.subarray(2 * b * inputN, 2 * (b + 1) * inputN);
        const xPad = new Float32Array(2 * fftN);
        embedAtOffset(xPad, fftShape, x, shape, new Array(rank).fill(0));
        const xf = fftNdRefAnySizeInterleaved(xPad, fftShape, "forward", "none");
        const yf = new Float32Array(2 * fftN);
        for (let i = 0; i < fftN; i++) {
          const ar = xf[2 * i];
          const ai = xf[2 * i + 1];
          const br = kf[2 * i];
          const bi = mode === "correlation" ? -kf[2 * i + 1] : kf[2 * i + 1];
          yf[2 * i] = ar * br - ai * bi;
          yf[2 * i + 1] = ar * bi + ai * br;
        }
        const yFull = fftNdRefAnySizeInterleaved(yf, fftShape, "inverse", "backward");
        const yOut = extractAtOffset(yFull, fftShape, outShape, outOffset);
        out.set(yOut, 2 * b * outN);
      }
    
      return out;
    }
    
    exports['normalizeScaleFactor'] = normalizeScaleFactor;
    exports['fft1dRefInterleaved'] = fft1dRefInterleaved;
    exports['fftNdRefInterleaved'] = fftNdRefInterleaved;
    exports['randomComplexInterleaved'] = randomComplexInterleaved;
    exports['dft1dRefInterleaved'] = dft1dRefInterleaved;
    exports['fftNdRefAnySizeInterleaved'] = fftNdRefAnySizeInterleaved;
    exports['r2cRefPackedInterleaved'] = r2cRefPackedInterleaved;
    exports['c2rRefFromPackedInterleaved'] = c2rRefFromPackedInterleaved;
    exports['dct2Ref'] = dct2Ref;
    exports['dct3Ref'] = dct3Ref;
    exports['dct4Ref'] = dct4Ref;
    exports['dst1Ref'] = dst1Ref;
    exports['dst2Ref'] = dst2Ref;
    exports['dst3Ref'] = dst3Ref;
    exports['dst4Ref'] = dst4Ref;
    exports['dct1Ref'] = dct1Ref;
    exports['conv2dRef'] = conv2dRef;
    exports['fftConvRef'] = fftConvRef;
  });

  __define('src/utils/webgpu.js', function(require, exports, module){
    function assertDevice(device) {
      if (!device) throw new Error("Expected a WebGPU device");
    }
    
    /**
     * Uploads interleaved complex<f32> data `[re, im, re, im, ...]` into a GPUBuffer.
     * The returned buffer is usable as STORAGE and COPY_SRC for readback.
     */
    function uploadComplex(device, data) {
      assertDevice(device);
      if (!(data instanceof Float32Array)) {
        throw new Error("uploadComplex expects a Float32Array");
      }
      const buffer = device.createBuffer({
        size: data.byteLength,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    }
    
    /**
     * Downloads interleaved complex<f32> data from a GPUBuffer into a Float32Array.
     * Returns a Promise because GPUBuffer.mapAsync is asynchronous.
     */
    async function downloadComplex(device, buffer, lengthComplex, offsetBytes = 0) {
      assertDevice(device);
      if (!buffer) throw new Error("downloadComplex expects a GPUBuffer");
      if (!Number.isInteger(lengthComplex) || lengthComplex <= 0) {
        throw new Error(`lengthComplex must be a positive integer; got ${lengthComplex}`);
      }
      if (!Number.isInteger(offsetBytes) || offsetBytes < 0 || offsetBytes % 8 !== 0) {
        throw new Error(`offsetBytes must be a non-negative multiple of 8; got ${offsetBytes}`);
      }
    
      const byteLength = lengthComplex * 8;
      const readback = device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, offsetBytes, readback, 0, byteLength);
      device.queue.submit([encoder.finish()]);
    
      await readback.mapAsync(GPUMapMode.READ);
      const copy = readback.getMappedRange();
      const out = new Float32Array(copy.slice(0));
      readback.unmap();
      readback.destroy();
      return out;
    }
    
    
    exports['uploadComplex'] = uploadComplex;
    exports['downloadComplex'] = downloadComplex;
  });

  const lib = __require('src/index.js');
  const api = Object.assign({}, lib);
  globalThis.webgpufft = api;
  globalThis.webgpufftReady = Promise.resolve(api);
  try { globalThis.dispatchEvent(new Event('webgpufft-ready')); } catch (_) {}
})();
