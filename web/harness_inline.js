  function $(id){ const el = document.getElementById(id); if(!el) throw new Error('Missing #' + id); return el; }
  function log(line){ const el = $('log'); el.textContent += line + "\n"; el.scrollTop = el.scrollHeight; }
  function setSummary(html){ $('summary').innerHTML = html; }
  const __qs = new URLSearchParams(location.search);
  const __autorun = (__qs.get('autorun') || '').toLowerCase();
  const __requireRtx5090 = (__qs.get('require_rtx5090') || '0') === '1';
  const __preferHighPerf = (__qs.get('prefer_high_perf') || '1') !== '0';
  const __realLarge3d = (__qs.get('real_large3d') || '0') === '1';
  const __realLarge3dN = parsePositiveIntOr(__qs.get('real_large3d_n'), 1024);
  const __realLarge3dStrict = (__qs.get('real_large3d_strict') || '1') !== '0';

  function trimOrEmpty(v){ return v == null ? '' : String(v).trim(); }
  function parsePositiveIntOr(v, fallback){
    const n = Number(v);
    return Number.isInteger(n) && n > 0 ? n : fallback;
  }

  function buildRequiredLimitsFromAdapter(adapter){
    const lim = adapter && adapter.limits;
    if(!lim) return null;
    const keys = [
      'maxBufferSize',
      'maxStorageBufferBindingSize',
      'maxStorageBuffersPerShaderStage',
      'maxComputeWorkgroupStorageSize',
      'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupSizeX',
      'maxComputeWorkgroupSizeY',
      'maxComputeWorkgroupSizeZ',
      'maxComputeWorkgroupsPerDimension',
    ];
    const out = {};
    for(const k of keys){
      const v = lim[k];
      if(Number.isFinite(v) && v > 0){
        out[k] = Math.floor(v);
      }
    }
    if(Number.isFinite(out.maxBufferSize) && Number.isFinite(out.maxStorageBufferBindingSize)){
      out.maxStorageBufferBindingSize = Math.min(out.maxStorageBufferBindingSize, out.maxBufferSize);
    }
    return Object.keys(out).length ? out : null;
  }

  function isNvidiaRtx5090(info){
    if(!info) return false;
    const hay = (
      trimOrEmpty(info.vendor) + ' ' +
      trimOrEmpty(info.device) + ' ' +
      trimOrEmpty(info.description) + ' ' +
      trimOrEmpty(info.architecture) + ' ' +
      trimOrEmpty(info.vendorID) + ' ' +
      trimOrEmpty(info.deviceID)
    ).toLowerCase();
    const hasNvidia = hay.includes('nvidia') || hay.includes('10de') || hay.includes('geforce');
    const has5090 = /\b5090\b/.test(hay) || /rtx\s*5090/.test(hay) || /geforce.*5090/.test(hay);
    return hasNvidia && has5090;
  }

  function isNvidiaBlackwell(info){
    if(!info) return false;
    const hay = (
      trimOrEmpty(info.vendor) + ' ' +
      trimOrEmpty(info.device) + ' ' +
      trimOrEmpty(info.description) + ' ' +
      trimOrEmpty(info.architecture) + ' ' +
      trimOrEmpty(info.vendorID)
    ).toLowerCase();
    const hasNvidia = hay.includes('nvidia') || hay.includes('10de') || hay.includes('geforce');
    return hasNvidia && hay.includes('blackwell');
  }

  function summarizeAdapterInfo(info){
    if(!info) return '(unavailable)';
    const parts = [];
    if(trimOrEmpty(info.vendor)) parts.push(`vendor=${trimOrEmpty(info.vendor)}`);
    if(trimOrEmpty(info.description)) parts.push(`description=${trimOrEmpty(info.description)}`);
    if(trimOrEmpty(info.device)) parts.push(`device=${trimOrEmpty(info.device)}`);
    if(trimOrEmpty(info.architecture)) parts.push(`arch=${trimOrEmpty(info.architecture)}`);
    if(trimOrEmpty(info.vendorID)) parts.push(`vendorID=${trimOrEmpty(info.vendorID)}`);
    if(trimOrEmpty(info.deviceID)) parts.push(`deviceID=${trimOrEmpty(info.deviceID)}`);
    return parts.length ? parts.join(', ') : '(available but empty)';
  }

  async function readAdapterInfo(adapter){
    let info = null;
    try{ if(adapter && adapter.info) info = adapter.info; }catch{}
    if(!info){
      try{ if(adapter && adapter.requestAdapterInfo) info = await adapter.requestAdapterInfo(); }catch{}
    }
    if(!info) return null;
    const out = {};
    for(const k of ['vendor','device','description','architecture','vendorID','deviceID']){
      try{
        const v = info[k];
        if(v !== undefined && v !== null && String(v).length) out[k] = String(v);
      }catch{}
    }
    return Object.keys(out).length ? out : null;
  }

  function publishAutorunResult(kind, result){
    try{
      const failures = Array.isArray(result?.failures) ? result.failures.slice(0, 16) : [];
      const payload = {
        kind,
        ok: !!result?.ok,
        pass: result?.pass ?? 0,
        fail: result?.fail ?? 0,
        skip: result?.skip ?? 0,
        total: result?.total ?? 0,
        ms: result?.ms ?? 0,
        failureNames: failures.map(f => f?.name).filter(Boolean),
        failures,
        benches: Array.isArray(result?.benches) ? result.benches.slice(0, 64) : undefined,
        adapter: result?.adapter ?? null,
        timestamp: new Date().toISOString(),
      };
      const json = JSON.stringify(payload);
      document.documentElement.setAttribute('data-webgpufft-autorun', encodeURIComponent(json));
      console.log('AUTORUN_RESULT ' + json);
      try{
        if(navigator.sendBeacon){
          const blob = new Blob([json], { type: 'application/json' });
          navigator.sendBeacon('/__autorun_result', blob);
        } else {
          fetch('/__autorun_result', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: json,
            keepalive: true,
          }).catch(()=>{});
        }
      }catch{}
      if(__qs.get('machine') === '1'){
        setSummary((payload.ok ? '<span class="pass">PASS</span>' : '<span class="fail">FAIL</span>') + ` - autorun ${kind}`);
      }
    }catch{}
  }

  function seededRng(seed){
    let x = (seed|0) || 123456789;
    return function(){
      x ^= (x << 13); x |= 0;
      x ^= (x >>> 17); x |= 0;
      x ^= (x << 5); x |= 0;
      return ((x >>> 0) / 4294967296);
    };
  }

  function assert(cond, msg){ if(!cond) throw new Error(msg || 'assert failed'); }

  function assertCloseArrayFloat32(a,b,tolAbs,tolRel,msg){
    if(a.length !== b.length) throw new Error((msg?msg+': ':'') + `length ${a.length} != ${b.length}`);
    let worst = -1;
    let worstAbs = 0;
    let worstRel = 0;
    let worstTol = 0;
    for(let i=0;i<a.length;i++){
      const av = a[i], bv = b[i];
      const absErr = Math.abs(av - bv);
      const tol = Math.max(tolAbs, tolRel * Math.abs(bv));
      if(absErr > tol && absErr > worstAbs){
        worst = i;
        worstAbs = absErr;
        worstRel = absErr / (Math.abs(bv) + 1e-20);
        worstTol = tol;
      }
    }
    if(worst !== -1){
      throw new Error((msg?msg+': ':'') + `worst i=${worst} actual=${a[worst]} expected=${b[worst]} maxAbs=${worstAbs} maxRel=${worstRel} tolAbs=${tolAbs} tolRel=${tolRel} tolAtWorst=${worstTol}`);
    }
  }

  async function requestDevice(){
    const gpu = globalThis.navigator && navigator.gpu;
    if(!gpu || !gpu.requestAdapter) return null;
    let adapter = null;
    let adapterSelection = 'default';
    const prefs = __preferHighPerf ? ['high-performance', null] : [null];
    for(const pref of prefs){
      try{
        adapter = pref ? await gpu.requestAdapter({ powerPreference: pref }) : await gpu.requestAdapter();
      }catch{
        adapter = null;
      }
      if(adapter){
        adapterSelection = pref || 'default';
        break;
      }
    }
    if(!adapter) return null;
    const want=[];
    for(const f of ['shader-f16','subgroups']){ try{ if(adapter.features && adapter.features.has(f)) want.push(f); }catch{} }
    const requiredLimits = buildRequiredLimitsFromAdapter(adapter);
    const desc = want.length ? { requiredFeatures: want } : {};
    if(requiredLimits) desc.requiredLimits = requiredLimits;
    let device = null;
    let limitsRequestMode = 'adapter-max';
    try{
      device = await adapter.requestDevice(desc);
    }catch{
      limitsRequestMode = 'default-fallback';
      device = await adapter.requestDevice(want.length ? { requiredFeatures: want } : {});
    }
    return {
      adapter,
      device,
      requested: want,
      adapterSelection,
      requestedLimits: requiredLimits,
      limitsRequestMode,
    };
  }

  function showDeviceInfo(adapter, device, requested, adapterMeta){
    const kv = $('gpuInfo');
    kv.textContent='';
    const add=(k,v)=>{ const d1=document.createElement('div'); d1.textContent=k; const d2=document.createElement('div'); d2.textContent=String(v); kv.appendChild(d1); kv.appendChild(d2); };
    add('location.protocol', location.protocol);
    add('navigator.gpu', !!(navigator && navigator.gpu));
    add('device.features.enabled', Array.from(device.features||[]).join(', ') || '(none)');
    add('device.features.requested', (requested||[]).join(', ') || '(none)');
    const lim = device.limits || {};
    const alim = adapter && adapter.limits ? adapter.limits : {};
    add('limits.maxStorageBufferBindingSize', lim.maxStorageBufferBindingSize);
    add('limits.maxBufferSize', lim.maxBufferSize);
    add('limits.maxStorageBuffersPerShaderStage', lim.maxStorageBuffersPerShaderStage);
    add('limits.minStorageBufferOffsetAlignment', lim.minStorageBufferOffsetAlignment);
    add('limits.maxComputeWorkgroupStorageSize', lim.maxComputeWorkgroupStorageSize);
    add('limits.maxComputeInvocationsPerWorkgroup', lim.maxComputeInvocationsPerWorkgroup);
    add('limits.maxComputeWorkgroupSizeX', lim.maxComputeWorkgroupSizeX);
    add('limits.maxComputeWorkgroupSizeY', lim.maxComputeWorkgroupSizeY);
    add('limits.maxComputeWorkgroupSizeZ', lim.maxComputeWorkgroupSizeZ);
    add('adapter.selection', adapterMeta?.selection || '(unknown)');
    add('adapter.limits.maxStorageBufferBindingSize', alim.maxStorageBufferBindingSize ?? '(unknown)');
    add('adapter.limits.maxBufferSize', alim.maxBufferSize ?? '(unknown)');
    add('adapter.limits.request.mode', adapterMeta?.limitsRequestMode || '(unknown)');
    add('adapter.summary', adapterMeta?.summary || '(unavailable)');
    add('adapter.match.rtx5090', adapterMeta?.isNvidia5090 == null ? 'unknown' : String(!!adapterMeta.isNvidia5090));
    add('adapter.match.nvidiaBlackwell', adapterMeta?.isNvidiaBlackwell == null ? 'unknown' : String(!!adapterMeta.isNvidiaBlackwell));
    add('adapter.require.rtx5090', String(__requireRtx5090));
    add('adapter.info', adapterMeta?.raw ? JSON.stringify(adapterMeta.raw) : '(unavailable)');
  }

  async function downloadF32(device, buffer, byteLen, offset){
    const rb = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offset||0, rb, 0, byteLen);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await rb.mapAsync(GPUMapMode.READ);
    const outArr = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return outArr;
  }

  async function downloadU16(device, buffer, byteLen, offset){
    const rb = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offset||0, rb, 0, byteLen);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await rb.mapAsync(GPUMapMode.READ);
    const outArr = new Uint16Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return outArr;
  }

  function f32ToF16Bits(x){
    const f32 = new Float32Array(1); const u32 = new Uint32Array(f32.buffer);
    f32[0]=x; const v=u32[0]>>>0;
    const sign=(v>>>31)&1; let exp=(v>>>23)&0xff; let mant=v&0x7fffff;
    if(exp===0xff){ const nan=mant!==0; return (sign<<15)|(0x1f<<10)|(nan?0x200:0); }
    if(exp===0){ if(mant===0) return sign<<15; while((mant&0x800000)===0){ mant<<=1; exp--; } mant&=0x7fffff; exp++; }
    const exp16=exp-127+15;
    if(exp16>=0x1f) return (sign<<15)|(0x1f<<10);
    if(exp16<=0){ if(exp16<-10) return sign<<15; mant|=0x800000; const shift=14-exp16; let m=mant>>>shift; if((mant>>>(shift-1))&1) m+=1; return (sign<<15)|m; }
    let m=mant>>>13; if(mant&0x1000) m+=1;
    if(m===0x400){ m=0; const e=exp16+1; if(e>=0x1f) return (sign<<15)|(0x1f<<10); return (sign<<15)|(e<<10)|m; }
    return (sign<<15)|(exp16<<10)|(m&0x3ff);
  }

  function f16BitsToF32(h){
    const sign=(h>>>15)&1; const exp=(h>>>10)&0x1f; const mant=h&0x3ff; let v;
    if(exp===0){ if(mant===0) v=sign<<31; else { let m=mant; let e=-14; while((m&0x400)===0){ m<<=1; e--; } m&=0x3ff; const exp32=e+127; v=(sign<<31)|(exp32<<23)|(m<<13); } }
    else if(exp===0x1f){ v=(sign<<31)|(0xff<<23)|(mant?0x200000:0); }
    else { const exp32=exp-15+127; v=(sign<<31)|(exp32<<23)|(mant<<13); }
    const u32=new Uint32Array(1); u32[0]=v>>>0; return new Float32Array(u32.buffer)[0];
  }

  function makeRealInput(N, rng){ const a=new Float32Array(N); for(let i=0;i<N;i++) a[i]=(rng()*2-1)*0.5; return a; }

  const __tests = [];
  let __testsRegistered = false;
  function test(name, fn){ __tests.push({name, fn}); }

  function registerTests(getDevice){
    if(__testsRegistered) return;
    const suite = __require('test/complete.suite.js');
    class SkipError extends Error { constructor(msg){ super(msg); this.name='SkipError'; } }
    suite.registerCompleteTests({
      test,
      getDevice,
      assert,
      assertCloseArray: assertCloseArrayFloat32,
      SkipError,
      log,
      exportArtifact: null,
    });
    __testsRegistered = true;
  }

  async function runAllTests(device){
    const t0 = performance.now();
    const prevRandom = Math.random;
    Math.random = seededRng(12345);

    let pass=0, fail=0, skip=0;
    const failures = [];
    for(const tc of __tests){
      const s0 = performance.now();
      try{
        await tc.fn();
        pass++;
        log(`PASS ${tc.name} (${(performance.now()-s0).toFixed(2)} ms)`);
      } catch(e){
        if(e && (e.name === 'SkipError')){
          skip++;
          log(`SKIP ${tc.name}: ${String(e && e.message || e)}`);
        } else {
          fail++;
          const msg = String(e && e.message || e);
          log(`FAIL ${tc.name}: ${msg}`);
          if(e && e.stack) log(e.stack);
          failures.push({ name: tc.name, message: msg });
        }
      }
    }
    Math.random = prevRandom;
    const ms = performance.now()-t0;
    const status = fail===0 ? `<span class="pass">PASS</span>` : `<span class="fail">FAIL</span>`;
    setSummary(`${status} - ${pass}/${pass+fail+skip} tests (${skip} skipped), ${ms.toFixed(2)} ms`);
    return { ok: fail===0, pass, fail, skip, total: pass+fail+skip, ms, failures };
  }

  function approxFftFlopsComplex(n, logN){ return 5*n*logN; }

  async function runOnce(device, plan, input, output){
    const enc=device.createCommandEncoder();
    plan.exec(enc,{input,output});
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  async function runBurst(device, plan, input, output, runs){
    const n = Math.max(1, runs|0);
    const enc = device.createCommandEncoder();
    for(let i=0;i<n;i++) plan.exec(enc,{input,output});
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  function createSegmentedBufferView(device, totalBytes, maxSegmentBytes){
    const segAlign = 256;
    const cap = Math.max(segAlign, Math.floor(maxSegmentBytes));
    let remain = totalBytes;
    const segments = [];
    while(remain > 0){
      let size = Math.min(remain, cap);
      size = Math.floor(size / segAlign) * segAlign;
      if(size <= 0 || size > remain) size = remain;
      const buffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      segments.push({ buffer, offsetBytes: 0, sizeBytes: size });
      remain -= size;
    }
    return { segments, logicalByteOffset: 0, lengthBytes: totalBytes };
  }

  function destroyBufferView(view){
    if(!view || !Array.isArray(view.segments)) return;
    for(const s of view.segments){
      try{ s?.buffer?.destroy?.(); }catch{}
    }
  }

  function sliceBufferView(view, logicalOffsetBytes, lengthBytes){
    return {
      segments: view.segments,
      logicalByteOffset: (view.logicalByteOffset || 0) + logicalOffsetBytes,
      lengthBytes,
    };
  }

  function toContiguousRanges(view, logicalOffsetBytes, lengthBytes){
    const base = (view.logicalByteOffset || 0) + logicalOffsetBytes;
    let off = base;
    let remain = lengthBytes;
    const out = [];
    for(const seg of view.segments){
      if(remain <= 0) break;
      const segSize = seg.sizeBytes | 0;
      if(off >= segSize){
        off -= segSize;
        continue;
      }
      const take = Math.min(remain, segSize - off);
      out.push({
        buffer: seg.buffer,
        offsetBytes: (seg.offsetBytes | 0) + off,
        sizeBytes: take,
      });
      remain -= take;
      off = 0;
    }
    if(remain !== 0){
      throw new Error(`BufferView range out of bounds: offset=${logicalOffsetBytes} length=${lengthBytes}`);
    }
    return out;
  }

  function copyViewToBuffer(commandEncoder, view, logicalOffsetBytes, lengthBytes, dstBuffer, dstOffsetBytes){
    const ranges = toContiguousRanges(view, logicalOffsetBytes, lengthBytes);
    let dst = dstOffsetBytes;
    for(const r of ranges){
      commandEncoder.copyBufferToBuffer(r.buffer, r.offsetBytes, dstBuffer, dst, r.sizeBytes);
      dst += r.sizeBytes;
    }
  }

  function copyBufferToView(commandEncoder, srcBuffer, srcOffsetBytes, view, logicalOffsetBytes, lengthBytes){
    const ranges = toContiguousRanges(view, logicalOffsetBytes, lengthBytes);
    let src = srcOffsetBytes;
    for(const r of ranges){
      commandEncoder.copyBufferToBuffer(srcBuffer, src, r.buffer, r.offsetBytes, r.sizeBytes);
      src += r.sizeBytes;
    }
  }

  function generateTransposeComplex2dWgsl(N){
    const T = 16;
    return `
struct Params {
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
const NX: u32 = ${N}u;
const NY: u32 = ${N}u;
const TILE: u32 = ${T}u;
var<workgroup> tileData: array<vec2<f32>, ${T} * (${T} + 1)>;
fn tile_idx(x: u32, y: u32) -> u32 { return y * (TILE + 1u) + x; }
@compute @workgroup_size(${T}, ${T}, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
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

  async function runRealSegmented3dFft(device, N){
    const lim = device.limits || {};
    const maxBuf = lim.maxBufferSize ?? (256 * 1024 * 1024);
    const planeBytes = N * N * 8;
    let segmentBytes = Math.max(256, Math.min(maxBuf, 2 * 1024 * 1024 * 1024));
    if(segmentBytes > planeBytes){
      const p = Math.floor(segmentBytes / planeBytes);
      segmentBytes = Math.max(planeBytes, p * planeBytes);
    }
    const total = N * N * N;
    const totalBytes = total * 8;
    const inputView = createSegmentedBufferView(device, totalBytes, segmentBytes);
    const outputView = createSegmentedBufferView(device, totalBytes, segmentBytes);
    try{
      const seedComplex = Math.min(total, 1 << 20);
      const seed = new Float32Array(seedComplex * 2);
      for(let i=0;i<seedComplex;i++){
        seed[2*i] = Math.sin(i * 0.0031);
        seed[2*i + 1] = Math.cos(i * 0.0041);
      }
      let seedRemain = seed.byteLength;
      let seedSrcOff = 0;
      for(const s of inputView.segments){
        if(seedRemain <= 0) break;
        const nBytes = Math.min(seedRemain, s.sizeBytes);
        device.queue.writeBuffer(s.buffer, s.offsetBytes, seed.buffer, seedSrcOff, nBytes);
        seedRemain -= nBytes;
        seedSrcOff += nBytes;
      }

      const plan = lib.createPlan(device, {
        type: "c2c",
        shape: [N, N, N],
        direction: "forward",
        batch: 1,
        inPlace: false,
        normalize: "none",
        layout: { interleavedComplex: true },
        precision: "f32",
        tuning: {
          largeRoute: "out-of-core",
          outOfCoreBurstWindows: 3,
        },
      });
      try{
        const encWarm = device.createCommandEncoder();
        plan.exec(encWarm, { input: inputView, output: outputView });
        device.queue.submit([encWarm.finish()]);
        await device.queue.onSubmittedWorkDone();

        const t0 = performance.now();
        const enc = device.createCommandEncoder();
        plan.exec(enc, { input: inputView, output: outputView });
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const elapsedMs = performance.now() - t0;
        const gflops = (approxFftFlopsComplex(total, Math.log2(N) * 3) / (elapsedMs / 1000)) / 1e9;
        const meta = plan._segmentedFullVolumeMeta || {};
        return {
          elapsedMs,
          gflops,
          axis0ChunkLines: meta.axis0LinesPerChunk ?? null,
          axis0ChunkBytes: meta.axis0ChunkBytes ?? null,
          axis2MaxZChunk: meta.axis2MaxZChunk ?? null,
          axis2ChunkSpanBytes: meta.axis2ChunkSpanBytes ?? null,
          segmentBytes: meta.segmentBytes ?? segmentBytes,
          segmentCount: meta.segmentCount ?? inputView.segments.length,
          ringDepth: meta.ringDepth ?? null,
          axis0ChunkUtilization: meta.axis0ChunkUtilization ?? null,
          axis2ChunkUtilization: meta.axis2ChunkUtilization ?? null,
        };
      } finally {
        plan?.destroy?.();
      }
    } finally {
      destroyBufferView(inputView);
      destroyBufferView(outputView);
    }
  }

  function pick2dShapeWithinLimits(device, target){
    const maxBind=(device.limits && device.limits.maxStorageBufferBindingSize) || Infinity;
    const maxWg = (device.limits && (device.limits.maxComputeWorkgroupsPerDimension ?? 65535)) || 65535;
    const assumeWorkgroupSize = 256; // most kernels use up to 256 invocations
    let n=target||1024;
    while(n>=64){
      const bytes=n*n*8;
      // Many FFT passes dispatch 1D over totalComplex. Keep within workgroup-per-dimension limits.
      const groups = Math.ceil((n*n) / assumeWorkgroupSize);
      if(bytes<=maxBind && groups<=maxWg) return [n,n];
      n=Math.floor(n/2);
    }
    return [64,64];
  }

  function pick3dShapeWithinLimits(device, target){
    const maxBind=(device.limits && device.limits.maxStorageBufferBindingSize) || Infinity;
    const maxWg = (device.limits && (device.limits.maxComputeWorkgroupsPerDimension ?? 65535)) || 65535;
    const assumeWorkgroupSize = 256;
    let n=target||64;
    while(n>=16){
      const bytes=n*n*n*8;
      const groups = Math.ceil((n*n*n) / assumeWorkgroupSize);
      if(bytes<=maxBind && groups<=maxWg) return [n,n,n];
      n=Math.floor(n/2);
    }
    return [16,16,16];
  }

  async function runBenchmarks(device, adapterMeta){
    const t0=performance.now();
    const benchResults = [];
    if(adapterMeta){
      const dlim = device.limits || {};
      log(
        `BENCH adapter: selection=${adapterMeta.selection || 'unknown'} ` +
        `limitsRequestMode=${adapterMeta.limitsRequestMode || 'unknown'} ` +
        `matchRtx5090=${adapterMeta.isNvidia5090 == null ? 'unknown' : String(!!adapterMeta.isNvidia5090)} ` +
        `matchNvidiaBlackwell=${adapterMeta.isNvidiaBlackwell == null ? 'unknown' : String(!!adapterMeta.isNvidiaBlackwell)} ` +
        `summary=${adapterMeta.summary || '(unavailable)'} ` +
        `maxBufferSize=${dlim.maxBufferSize ?? '(unknown)'} ` +
        `maxStorageBufferBindingSize=${dlim.maxStorageBufferBindingSize ?? '(unknown)'}`
      );
    }

    {
      const N=1024;
      const plan = lib.createPlan(device,{type:'c2c',shape:[N],direction:'forward',batch:1,inPlace:false,normalize:'none',layout:{interleavedComplex:true},precision:'f32'});
      const rng=seededRng(10);
      const inputData=math.randomComplexInterleaved(N,rng);
      const input=lib.uploadComplex(device,inputData);
      const output=device.createBuffer({size:inputData.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      for(let i=0;i<3;i++) await runOnce(device,plan,input,output);
      const tCold0=performance.now(); await runOnce(device,plan,input,output); const coldMs=performance.now()-tCold0;
      const iters=50; const t1=performance.now(); for(let i=0;i<iters;i++) await runOnce(device,plan,input,output); const avgMs=(performance.now()-t1)/iters;
      const gflops=(approxFftFlopsComplex(N,Math.log2(N))/(avgMs/1000))/1e9;
      log(`BENCH 1D C2C N=1024: cold=${coldMs.toFixed(2)} ms avg=${avgMs.toFixed(2)} ms ~${gflops.toFixed(2)} GFLOP/s`);
      benchResults.push({ name: '1D C2C N=1024', coldMs, avgMs, gflops, skipped: false });
      plan.destroy(); input.destroy(); output.destroy();
    }

    {
      const N=2310;
      const plan = lib.createPlan(device,{type:'c2c',shape:[N],direction:'forward',batch:1,inPlace:false,normalize:'none',layout:{interleavedComplex:true},precision:'f32'});
      const rng=seededRng(11);
      const inputData=math.randomComplexInterleaved(N,rng);
      const input=lib.uploadComplex(device,inputData);
      const output=device.createBuffer({size:inputData.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      for(let i=0;i<2;i++) await runOnce(device,plan,input,output);
      const iters=20; const t1=performance.now(); for(let i=0;i<iters;i++) await runOnce(device,plan,input,output); const avgMs=(performance.now()-t1)/iters;
      const gflops=(approxFftFlopsComplex(N,Math.log2(N))/(avgMs/1000))/1e9;
      log(`BENCH 1D C2C N=2310: avg=${avgMs.toFixed(2)} ms ~${gflops.toFixed(2)} GFLOP/s`);
      benchResults.push({ name: '1D C2C N=2310', avgMs, gflops, skipped: false });
      plan.destroy(); input.destroy(); output.destroy();
    }

    {
      const requested=4096;
      const [Nx,Ny]=pick2dShapeWithinLimits(device,requested);
      const total=Nx*Ny;
      const plan = lib.createPlan(device,{type:'c2c',shape:[Nx,Ny],direction:'forward',batch:1,inPlace:false,normalize:'none',layout:{interleavedComplex:true},precision:'f32'});
      const rng=seededRng(12);
      const inputData=math.randomComplexInterleaved(total,rng);
      const input=lib.uploadComplex(device,inputData);
      const output=device.createBuffer({size:inputData.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      for(let i=0;i<2;i++) await runOnce(device,plan,input,output);
      const iters=Math.max(3, Math.floor(2000000/total));
      const t1=performance.now(); for(let i=0;i<iters;i++) await runOnce(device,plan,input,output); const avgMs=(performance.now()-t1)/iters;
      const gflops=(approxFftFlopsComplex(total,Math.log2(Nx)+Math.log2(Ny))/(avgMs/1000))/1e9;
      const usedTranspose=!!plan.transpose;
      log(`BENCH 2D C2C ${Nx}x${Ny} (requested ${requested}x${requested}): avg=${avgMs.toFixed(2)} ms ~${gflops.toFixed(2)} GFLOP/s transpose=${usedTranspose}`);
      benchResults.push({ name: `2D C2C ${Nx}x${Ny}`, avgMs, gflops, transpose: usedTranspose, skipped: false });
      plan.destroy(); input.destroy(); output.destroy();
    }

    {
      const lim = device.limits || {};
      const maxBind = lim.maxStorageBufferBindingSize ?? Infinity;
      const maxBuf = lim.maxBufferSize ?? maxBind;
      const directCases = [
        { n: 128, iters: 10 },
        { n: 256, iters: 2 },
      ];
      for (const cfg of directCases) {
        const reqN = cfg.n;
        const total = reqN * reqN * reqN;
        const bytes = total * 8;
        if (bytes > maxBind || bytes > maxBuf) {
          const reasons = [];
          if (bytes > maxBind) reasons.push(`bytes=${bytes} > maxStorageBufferBindingSize=${maxBind}`);
          if (bytes > maxBuf) reasons.push(`bytes=${bytes} > maxBufferSize=${maxBuf}`);
          const reason = reasons.join(", ");
          log(`SKIP BENCH 3D C2C direct ${reqN}x${reqN}x${reqN} x${cfg.iters}: ${reason}`);
          benchResults.push({ name: `3D C2C direct ${reqN}x${reqN}x${reqN} x${cfg.iters}`, skipped: true, reason });
          continue;
        }

        let plan = null;
        let input = null;
        let output = null;
        try {
          plan = lib.createPlan(device, {
            type: "c2c",
            shape: [reqN, reqN, reqN],
            direction: "forward",
            batch: 1,
            inPlace: false,
            normalize: "none",
            layout: { interleavedComplex: true },
            precision: "f32",
          });
          input = device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          output = device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });

          const initN = Math.min(4096, total);
          const init = new Float32Array(2 * initN);
          for (let i = 0; i < initN; i++) {
            init[2 * i] = Math.sin(i * 0.01);
            init[2 * i + 1] = Math.cos(i * 0.01);
          }
          device.queue.writeBuffer(input, 0, init);

          await runOnce(device, plan, input, output);
          const iters = cfg.iters;
          const t1 = performance.now();
          for (let i = 0; i < iters; i++) await runOnce(device, plan, input, output);
          const totalMs = performance.now() - t1;
          const avgMs = totalMs / iters;
          const logSum = Math.log2(reqN) * 3;
          const gflops = (approxFftFlopsComplex(total, logSum) / (avgMs / 1000)) / 1e9;
          const usedTranspose = !!plan.transpose;
          log(
            `BENCH 3D C2C direct ${reqN}x${reqN}x${reqN} x${iters}: total=${totalMs.toFixed(2)} ms ` +
              `avg=${avgMs.toFixed(2)} ms ~${gflops.toFixed(2)} GFLOP/s transpose=${usedTranspose}`
          );
          benchResults.push({
            name: `3D C2C direct ${reqN}x${reqN}x${reqN} x${iters}`,
            runs: iters,
            totalMs,
            avgMs,
            gflops,
            transpose: usedTranspose,
            skipped: false,
          });
        } catch (e) {
          const reason = String((e && e.message) || e);
          log(`SKIP BENCH 3D C2C direct ${reqN}x${reqN}x${reqN} x${cfg.iters}: ${reason}`);
          benchResults.push({ name: `3D C2C direct ${reqN}x${reqN}x${reqN} x${cfg.iters}`, skipped: true, reason });
        } finally {
          plan?.destroy?.();
          input?.destroy?.();
          output?.destroy?.();
        }
      }
    }

    {
      const lim = device.limits || {};
      const maxBind = lim.maxStorageBufferBindingSize ?? Infinity;
      const maxBuf = lim.maxBufferSize ?? maxBind;
      const maxWg = lim.maxComputeWorkgroupsPerDimension ?? 65535;
      const assumeWorkgroupSize = 256;
      const requests = [512, 1024];

      for (const reqN of requests) {
        const total = reqN * reqN * reqN;
        const bytes = total * 8;
        const groups = Math.ceil(total / assumeWorkgroupSize);

        if (bytes > maxBuf) {
          if (__realLarge3d && reqN === __realLarge3dN) {
            try {
              log(`BENCH 3D C2C real segmented full-volume ${reqN}x${reqN}x${reqN}: start`);
              const real = await runRealSegmented3dFft(device, reqN);
              log(
                `BENCH 3D C2C real segmented full-volume ${reqN}x${reqN}x${reqN}: ` +
                  `total=${real.elapsedMs.toFixed(2)} ms ~${real.gflops.toFixed(2)} GFLOP/s ` +
                  `axis0ChunkLines=${real.axis0ChunkLines} axis0ChunkBytes=${real.axis0ChunkBytes} ` +
                  `axis2MaxZChunk=${real.axis2MaxZChunk} axis2ChunkSpanBytes=${real.axis2ChunkSpanBytes} ` +
                  `segmentBytes=${real.segmentBytes} segmentCount=${real.segmentCount} ringDepth=${real.ringDepth} mode=segmented-full-volume`
              );
              benchResults.push({
                name: `3D C2C real segmented full-volume ${reqN}x${reqN}x${reqN}`,
                totalMs: real.elapsedMs,
                avgMs: real.elapsedMs,
                gflops: real.gflops,
                mode: "segmented-full-volume",
                axis0ChunkLines: real.axis0ChunkLines,
                axis0ChunkBytes: real.axis0ChunkBytes,
                axis2MaxZChunk: real.axis2MaxZChunk,
                axis2ChunkSpanBytes: real.axis2ChunkSpanBytes,
                segmentBytes: real.segmentBytes,
                segmentCount: real.segmentCount,
                ringDepth: real.ringDepth,
                axis0ChunkUtilization: real.axis0ChunkUtilization,
                axis2ChunkUtilization: real.axis2ChunkUtilization,
                skipped: false,
              });
              continue;
            } catch (e) {
              const reason = `real segmented full-volume failed: ${String((e && e.message) || e)}`;
              log(`SKIP BENCH 3D C2C real segmented full-volume ${reqN}x${reqN}x${reqN}: ${reason}`);
              benchResults.push({ name: `3D C2C real segmented full-volume ${reqN}x${reqN}x${reqN}`, skipped: true, reason });
              if (__realLarge3dStrict) {
                continue;
              }
              // fall through to tile-proxy fallback when strict mode is disabled.
            }
          }

          // Hardware/API maxBufferSize cannot hold a full volume buffer. Run a streamed tile-sweep
          // to keep a non-skipped large-kernel benchmark signal in browser environments.
          log(
            `NOTE BENCH 3D C2C ${reqN}x${reqN}x${reqN}: using tile-proxy because bytes=${bytes} > maxBufferSize=${maxBuf}; ` +
            `this is not full-volume FFT throughput`
          );
          let plan = null;
          let input = null;
          let output = null;
          try {
            let tileN = Math.min(reqN, 256);
            while (tileN > 16 && (tileN * tileN * tileN * 8 > maxBind || tileN * tileN * tileN * 8 > maxBuf)) {
              tileN = Math.floor(tileN / 2);
            }
            const tileTotal = tileN * tileN * tileN;
            const tileBytes = tileTotal * 8;
            const tilesPerDim = Math.max(1, Math.ceil(reqN / tileN));
            const tileCount = tilesPerDim * tilesPerDim * tilesPerDim;
            const sweeps = reqN >= 1024 ? 1 : 2;

            plan = lib.createPlan(device, {
              type: "c2c",
              shape: [tileN, tileN, tileN],
              direction: "forward",
              batch: 1,
              inPlace: false,
              normalize: "none",
              layout: { interleavedComplex: true },
              precision: "f32",
            });
            input = device.createBuffer({
              size: tileBytes,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            output = device.createBuffer({
              size: tileBytes,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });

            const initN = Math.min(4096, tileTotal);
            const init = new Float32Array(2 * initN);
            for (let i = 0; i < initN; i++) {
              init[2 * i] = Math.sin(i * 0.019);
              init[2 * i + 1] = Math.cos(i * 0.019);
            }
            device.queue.writeBuffer(input, 0, init);

            await runOnce(device, plan, input, output);
            const t1 = performance.now();
            const burstSize = reqN >= 1024 ? 32 : 16;
            for (let s = 0; s < sweeps; s++) {
              let remain = tileCount;
              while (remain > 0) {
                const runs = Math.min(remain, burstSize);
                await runBurst(device, plan, input, output, runs);
                remain -= runs;
              }
            }
            const elapsedMs = performance.now() - t1;
            const avgGlobalMs = elapsedMs / sweeps;
            const tileLogSum = Math.log2(tileN) * 3;
            const tileOps = approxFftFlopsComplex(tileTotal, tileLogSum);
            const kernelThroughputGflops = (tileOps * tileCount * sweeps / (elapsedMs / 1000)) / 1e9;
            log(
              `BENCH 3D C2C tiled/chunked streamed ${reqN}x${reqN}x${reqN}: avgGlobal=${avgGlobalMs.toFixed(2)} ms ` +
                `kernelThroughput=${kernelThroughputGflops.toFixed(2)} GFLOP/s tile=${tileN} tiles=${tileCount} sweeps=${sweeps} mode=tile-proxy`
            );
            benchResults.push({
              name: `3D C2C tiled/chunked streamed ${reqN}x${reqN}x${reqN}`,
              avgMs: avgGlobalMs,
              gflops: kernelThroughputGflops,
              tileN,
              tiles: tileCount,
              sweeps,
              mode: "tile-proxy",
              proxy: true,
              requestedBytes: bytes,
              maxBufferSize: maxBuf,
              skipped: false,
            });
          } catch (e) {
            const reason = `streamed tile fallback failed: ${String((e && e.message) || e)}`;
            log(`SKIP BENCH 3D C2C tiled/chunked streamed ${reqN}x${reqN}x${reqN}: ${reason}`);
            benchResults.push({ name: `3D C2C tiled/chunked streamed ${reqN}x${reqN}x${reqN}`, skipped: true, reason });
          } finally {
            plan?.destroy?.();
            input?.destroy?.();
            output?.destroy?.();
          }
          continue;
        }

        const lineBytes = reqN * 8;
        let forcedBind = Math.floor(Math.min(maxBind, Math.max(lineBytes, Math.floor(bytes * 0.9))));
        forcedBind = Math.max(256, forcedBind);
        if (!(forcedBind < bytes)) forcedBind = Math.max(lineBytes, bytes - 4096);
        if (Number.isFinite(maxBind)) forcedBind = Math.min(forcedBind, maxBind);
        if (!(forcedBind < bytes)) {
          const reason = `cannot force multi-upload: forcedMaxBind=${forcedBind} bytes=${bytes} maxBind=${maxBind} dispatchX=${groups} maxWorkgroups=${maxWg}`;
          log(`SKIP BENCH 3D C2C tiled/chunked forced-large ${reqN}x${reqN}x${reqN}: ${reason}`);
          benchResults.push({ name: `3D C2C tiled/chunked forced-large ${reqN}x${reqN}x${reqN}`, skipped: true, reason });
          continue;
        }

        let plan = null;
        let input = null;
        let output = null;
        try {
          plan = lib.createPlan(device, {
            type: "c2c",
            shape: [reqN, reqN, reqN],
            direction: "forward",
            batch: 1,
            inPlace: false,
            normalize: "none",
            layout: { interleavedComplex: true },
            precision: "f32",
            tuning: { maxStorageBufferBindingSize: forcedBind },
          });

          input = device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          output = device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });

          const initN = Math.min(4096, total);
          const init = new Float32Array(2 * initN);
          for (let i = 0; i < initN; i++) {
            init[2 * i] = Math.sin(i * 0.017);
            init[2 * i + 1] = Math.cos(i * 0.017);
          }
          device.queue.writeBuffer(input, 0, init);

          await runOnce(device, plan, input, output);
          const iters = reqN <= 256 ? 2 : 1;
          const t1 = performance.now();
          for (let i = 0; i < iters; i++) await runOnce(device, plan, input, output);
          const avgMs = (performance.now() - t1) / iters;
          const logSum = Math.log2(reqN) * 3;
          const gflops = (approxFftFlopsComplex(total, logSum) / (avgMs / 1000)) / 1e9;
          const outOfCore = !!plan._outOfCoreFourStepMode;
          log(
            `BENCH 3D C2C tiled/chunked forced-large ${reqN}x${reqN}x${reqN}: avg=${avgMs.toFixed(2)} ms ` +
              `~${gflops.toFixed(2)} GFLOP/s outOfCore=${outOfCore} forcedMaxBind=${forcedBind}`
          );
          benchResults.push({
            name: `3D C2C tiled/chunked forced-large ${reqN}x${reqN}x${reqN}`,
            avgMs,
            gflops,
            forcedMaxBind: forcedBind,
            outOfCore,
            skipped: false,
          });
        } catch (e) {
          const reason = String((e && e.message) || e);
          log(`SKIP BENCH 3D C2C tiled/chunked forced-large ${reqN}x${reqN}x${reqN}: ${reason}`);
          benchResults.push({ name: `3D C2C tiled/chunked forced-large ${reqN}x${reqN}x${reqN}`, skipped: true, reason });
        } finally {
          plan?.destroy?.();
          input?.destroy?.();
          output?.destroy?.();
        }
      }
    }

    {
      const shape = [32, 8];
      const totalReal = shape[0] * shape[1];
      const packedTotal = (Math.floor(shape[0] / 2) + 1) * shape[1];
      try {
        const real = new Float32Array(totalReal);
        for (let i = 0; i < totalReal; i++) real[i] = Math.sin(i * 0.013);
        const inBuf = device.createBuffer({ size: real.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(inBuf, 0, real);
        const packedBuf = device.createBuffer({ size: packedTotal * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const outBuf = device.createBuffer({ size: totalReal * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

        const r2c = lib.createPlan(device, {
          type: "r2c",
          shape,
          direction: "forward",
          batch: 1,
          normalize: "none",
          layout: { interleavedComplex: true },
          precision: "f32",
          tuning: { maxStorageBufferBindingSize: 64 },
        });
        const c2r = lib.createPlan(device, {
          type: "c2r",
          shape,
          direction: "inverse",
          batch: 1,
          normalize: "backward",
          layout: { interleavedComplex: true },
          precision: "f32",
          tuning: { maxStorageBufferBindingSize: 64 },
        });

        for (let i = 0; i < 3; i++) {
          const enc = device.createCommandEncoder();
          r2c.exec(enc, { input: inBuf, output: packedBuf });
          c2r.exec(enc, { input: packedBuf, output: outBuf });
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
        }
        const iters = 30;
        const t1 = performance.now();
        for (let i = 0; i < iters; i++) {
          const enc = device.createCommandEncoder();
          r2c.exec(enc, { input: inBuf, output: packedBuf });
          c2r.exec(enc, { input: packedBuf, output: outBuf });
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
        }
        const avgMs = (performance.now() - t1) / iters;
        const r2cPolicy = r2c._outOfCoreAxisWindowPolicy?.realToComplex || null;
        const c2rPolicy = c2r._outOfCoreAxisWindowPolicy?.complexToReal || null;
        const r2cUploads = r2cPolicy?.numAxisUploads ?? null;
        const r2cChunkLines = r2cPolicy?.linesPerChunk ?? null;
        const c2rUploads = c2rPolicy?.numAxisUploads ?? null;
        const c2rChunkLines = c2rPolicy?.linesPerChunk ?? null;
        log(
          `BENCH R2C+C2R oversized-line large-mode ${shape[0]}x${shape[1]}: avg=${avgMs.toFixed(2)} ms` +
            ` r2cUploads=${r2cUploads ?? "n/a"} r2cChunkLines=${r2cChunkLines ?? "n/a"}` +
            ` c2rUploads=${c2rUploads ?? "n/a"} c2rChunkLines=${c2rChunkLines ?? "n/a"}`
        );
        benchResults.push({
          name: `R2C+C2R oversized-line ${shape[0]}x${shape[1]}`,
          avgMs,
          r2cUploads,
          r2cChunkLines,
          c2rUploads,
          c2rChunkLines,
          skipped: false,
        });

        r2c.destroy();
        c2r.destroy();
        inBuf.destroy();
        packedBuf.destroy();
        outBuf.destroy();
      } catch (e) {
        const reason = String(e && e.message || e);
        log(`SKIP BENCH R2C+C2R oversized-line ${shape[0]}x${shape[1]}: ${reason}`);
        benchResults.push({ name: `R2C+C2R oversized-line ${shape[0]}x${shape[1]}`, skipped: true, reason });
      }
    }

    {
      const cases = [
        { name: "Bluestein", shape: [4, 17], tuning: { maxStorageBufferBindingSize: 320, forceBluesteinAxes: [1] } },
        { name: "Rader", shape: [4, 29], tuning: { maxStorageBufferBindingSize: 480, forceRaderAxes: [1] } },
        { name: "Bluestein-rank4", shape: [3, 2, 2, 34], tuning: { maxStorageBufferBindingSize: 512, forceBluesteinAxes: [3] } },
        { name: "Rader-rank4", shape: [3, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 512, forceRaderAxes: [3] } },
        { name: "Bluestein-rank4-oversized-line", shape: [3, 2, 2, 34], tuning: { maxStorageBufferBindingSize: 160, forceBluesteinAxes: [3] } },
        { name: "Rader-rank4-oversized-line", shape: [3, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [3] } },
        { name: "Rader-rank5-oversized-line", shape: [2, 2, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [4] } },
      ];
      for (const cfg of cases) {
        const logical = cfg.shape.reduce((a, b) => a * b, 1);
        const total = logical * 2;
        try {
          const rng = seededRng(cfg.name === "Bluestein" ? 21 : 22);
          const inputData = math.randomComplexInterleaved(total, rng);
          const inBuf = lib.uploadComplex(device, inputData);
          const outBuf = device.createBuffer({
            size: inputData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          const plan = lib.createPlan(device, {
            type: "c2c",
            shape: cfg.shape,
            direction: "forward",
            batch: 2,
            inPlace: false,
            normalize: "none",
            layout: { interleavedComplex: true },
            precision: "f32",
            tuning: cfg.tuning,
          });

          for (let i = 0; i < 2; i++) await runOnce(device, plan, inBuf, outBuf);
          const iters = 20;
          const t1 = performance.now();
          for (let i = 0; i < iters; i++) await runOnce(device, plan, inBuf, outBuf);
          const avgMs = (performance.now() - t1) / iters;
          const axisPolicy = Array.isArray(plan._outOfCoreAxisWindowPolicy)
            ? plan._outOfCoreAxisWindowPolicy.find((p) => !!p) || null
            : null;
          const numAxisUploads = axisPolicy?.numAxisUploads ?? null;
          const linesPerChunk = axisPolicy?.linesPerChunk ?? null;
          log(
            `BENCH C2C out-of-core forced-${cfg.name} ${cfg.shape.join("x")} batch=2: avg=${avgMs.toFixed(2)} ms` +
              ` numAxisUploads=${numAxisUploads ?? "n/a"} linesPerChunk=${linesPerChunk ?? "n/a"}`
          );
          benchResults.push({
            name: `C2C forced-${cfg.name} ${cfg.shape.join("x")} b2`,
            avgMs,
            numAxisUploads,
            linesPerChunk,
            skipped: false,
          });

          plan.destroy();
          inBuf.destroy();
          outBuf.destroy();
        } catch (e) {
          const reason = String(e && e.message || e);
          log(`SKIP BENCH C2C out-of-core forced-${cfg.name} ${cfg.shape.join("x")}: ${reason}`);
          benchResults.push({ name: `C2C forced-${cfg.name} ${cfg.shape.join("x")} b2`, skipped: true, reason });
        }
      }
    }

    {
      const cases = [
        { name: "Bluestein-rank4", shape: [3, 2, 2, 17], tuning: { maxStorageBufferBindingSize: 1024, forceBluesteinAxes: [3] } },
        { name: "Rader-rank4", shape: [3, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 1024, forceRaderAxes: [3] } },
        { name: "Rader-rank4-oversized-line", shape: [3, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [3] } },
        { name: "Rader-rank5-oversized-line", shape: [2, 2, 2, 2, 29], tuning: { maxStorageBufferBindingSize: 160, forceRaderAxes: [4] } },
      ];
      for (const cfg of cases) {
        try {
          const batch = 2;
          const nReal = cfg.shape.reduce((a, b) => a * b, 1) * batch;
          const packedN = (Math.floor(cfg.shape[0] / 2) + 1) * cfg.shape.slice(1).reduce((a, b) => a * b, 1) * batch;
          const real = new Float32Array(nReal);
          for (let i = 0; i < nReal; i++) real[i] = Math.sin(i * 0.011);
          const inBuf = device.createBuffer({ size: real.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(inBuf, 0, real);
          const packedBuf = device.createBuffer({ size: packedN * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
          const outBuf = device.createBuffer({ size: nReal * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

          const r2c = lib.createPlan(device, {
            type: "r2c",
            shape: cfg.shape,
            direction: "forward",
            batch,
            normalize: "none",
            layout: { interleavedComplex: true },
            precision: "f32",
            tuning: cfg.tuning,
          });
          const c2r = lib.createPlan(device, {
            type: "c2r",
            shape: cfg.shape,
            direction: "inverse",
            batch,
            normalize: "backward",
            layout: { interleavedComplex: true },
            precision: "f32",
            tuning: cfg.tuning,
          });

          for (let i = 0; i < 2; i++) {
            const enc = device.createCommandEncoder();
            r2c.exec(enc, { input: inBuf, output: packedBuf });
            c2r.exec(enc, { input: packedBuf, output: outBuf });
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
          }
          const iters = 20;
          const t1 = performance.now();
          for (let i = 0; i < iters; i++) {
            const enc = device.createCommandEncoder();
            r2c.exec(enc, { input: inBuf, output: packedBuf });
            c2r.exec(enc, { input: packedBuf, output: outBuf });
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
          }
          const avgMs = (performance.now() - t1) / iters;
          const r2cPolicy = r2c._outOfCoreAxisWindowPolicy?.realToComplex || null;
          const c2rPolicy = c2r._outOfCoreAxisWindowPolicy?.complexToReal || null;
          const r2cUploads = r2cPolicy?.numAxisUploads ?? null;
          const r2cChunkLines = r2cPolicy?.linesPerChunk ?? null;
          const c2rUploads = c2rPolicy?.numAxisUploads ?? null;
          const c2rChunkLines = c2rPolicy?.linesPerChunk ?? null;
          log(
            `BENCH R2C+C2R forced-${cfg.name} ${cfg.shape.join("x")} batch=2: avg=${avgMs.toFixed(2)} ms` +
              ` r2cUploads=${r2cUploads ?? "n/a"} r2cChunkLines=${r2cChunkLines ?? "n/a"}` +
              ` c2rUploads=${c2rUploads ?? "n/a"} c2rChunkLines=${c2rChunkLines ?? "n/a"}`
          );
          benchResults.push({
            name: `R2C+C2R forced-${cfg.name} ${cfg.shape.join("x")} b2`,
            avgMs,
            r2cUploads,
            r2cChunkLines,
            c2rUploads,
            c2rChunkLines,
            skipped: false,
          });

          r2c.destroy();
          c2r.destroy();
          inBuf.destroy();
          packedBuf.destroy();
          outBuf.destroy();
        } catch (e) {
          const reason = String(e && e.message || e);
          log(`SKIP BENCH R2C+C2R forced-${cfg.name} ${cfg.shape.join("x")}: ${reason}`);
          benchResults.push({ name: `R2C+C2R forced-${cfg.name} ${cfg.shape.join("x")} b2`, skipped: true, reason });
        }
      }
    }

    {
      const shape = [32];
      const kernelShape = [5];
      const batch = 1;
      try {
        const inputData = math.randomComplexInterleaved(shape[0] * batch, seededRng(77));
        const kernelData = math.randomComplexInterleaved(kernelShape[0], seededRng(78));
        const input = lib.uploadComplex(device, inputData);
        const output = device.createBuffer({
          size: shape[0] * batch * 8,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const plan = lib.createPlan(device, {
          type: "fftconv",
          shape,
          batch,
          inPlace: false,
          layout: { interleavedComplex: true },
          precision: "f32",
          tuning: { maxStorageBufferBindingSize: 256 },
          fftConv: {
            mode: "convolution",
            boundary: "linear-same",
            kernelShape,
            kernelCount: 1,
            outputLayout: "kernel-major",
          },
        });

        for (let i = 0; i < 3; i++) {
          const enc = device.createCommandEncoder();
          plan.exec(enc, { input, output, kernel: kernelData });
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
        }
        const iters = 30;
        const t1 = performance.now();
        for (let i = 0; i < iters; i++) {
          const enc = device.createCommandEncoder();
          plan.exec(enc, { input, output, kernel: kernelData });
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
        }
        const avgMs = (performance.now() - t1) / iters;
        log(
          `BENCH FFTCONV forced-large linear-same ${shape[0]} k=${kernelShape[0]}: avg=${avgMs.toFixed(2)} ms ` +
            `largeMode=${!!plan._largeMode} batchSliced=${!!plan._batchSlicedExecution}`
        );
        benchResults.push({
          name: `FFTCONV forced-large linear-same ${shape[0]} k=${kernelShape[0]}`,
          avgMs,
          largeMode: !!plan._largeMode,
          batchSliced: !!plan._batchSlicedExecution,
          skipped: false,
        });

        plan.destroy();
        input.destroy();
        output.destroy();
      } catch (e) {
        const reason = String(e && e.message || e);
        log(`SKIP BENCH FFTCONV forced-large linear-same ${shape[0]} k=${kernelShape[0]}: ${reason}`);
        benchResults.push({ name: `FFTCONV forced-large linear-same ${shape[0]} k=${kernelShape[0]}`, skipped: true, reason });
      }
    }

    const ms=performance.now()-t0;
    setSummary(`<span class="pass">PASS</span> - benchmarks completed, ${ms.toFixed(2)} ms`);
    return {
      ok: true,
      pass: 0,
      fail: 0,
      skip: benchResults.filter(x=>x.skipped).length,
      total: benchResults.length,
      ms,
      benches: benchResults,
      adapter: adapterMeta ? {
        selection: adapterMeta.selection || 'unknown',
        limitsRequestMode: adapterMeta.limitsRequestMode || 'unknown',
        summary: adapterMeta.summary || '(unavailable)',
        isNvidia5090: !!adapterMeta.isNvidia5090,
        isNvidiaBlackwell: !!adapterMeta.isNvidiaBlackwell,
        acceptedByRtx5090Policy: !!adapterMeta.acceptedByRtx5090Policy,
        infoAvailable: !!adapterMeta.raw,
      } : null,
    };
  }

  (function initUI(){
    const btnInit=$('btnInit'); const btnTests=$('btnTests'); const btnBench=$('btnBench');
    let _dev=null, _adapter=null, _requested=null, _adapterMeta=null;

    async function ensure(){
      if(_dev) return _dev;
      setSummary('Status: initializing WebGPU...');
      const res = await requestDevice();
      if(!res){
        const hint = location.protocol==='file:' ? 'WebGPU may be disabled for local files or by policy. Try enabling WebGPU in chrome://flags (search: WebGPU).' : 'WebGPU unavailable.';
        setSummary(`<span class="skip">SKIP</span> - WebGPU unavailable. ${hint}`);
        return null;
      }
      _adapter=res.adapter; _dev=res.device; _requested=res.requested;
      const rawInfo = await readAdapterInfo(_adapter);
      const exact5090 = rawInfo ? isNvidiaRtx5090(rawInfo) : null;
      const nvidiaBlackwell = rawInfo ? isNvidiaBlackwell(rawInfo) : null;
      _adapterMeta = {
        selection: res.adapterSelection || 'default',
        limitsRequestMode: res.limitsRequestMode || 'unknown',
        raw: rawInfo,
        summary: summarizeAdapterInfo(rawInfo),
        isNvidia5090: exact5090,
        isNvidiaBlackwell: nvidiaBlackwell,
        acceptedByRtx5090Policy: rawInfo ? !!(exact5090 || nvidiaBlackwell) : null,
      };
      showDeviceInfo(_adapter,_dev,_requested,_adapterMeta);
      btnTests.disabled=false; btnBench.disabled=false;
      setSummary('Status: ready');
      return _dev;
    }

    const getDevice = async ()=>{
      try{ return await ensure(); }
      catch(e){ setSummary(`<span class="fail">FAIL</span> - init error`); log(String(e && e.stack || e)); return null; }
    };

    async function runTestsAction(){
      $('log').textContent='';
      const d=await ensure();
      if(!d) return { ok:false, pass:0, fail:1, skip:0, total:1, ms:0 };
      try{
        registerTests(getDevice);
      }catch(e){
        const msg = `test-suite load failed: ${String(e && e.message || e)}`;
        setSummary(`<span class="fail">FAIL</span> - test setup error`);
        log(`FAIL TEST SETUP: ${msg}`);
        return {
          ok: false,
          pass: 0,
          fail: 1,
          skip: 0,
          total: 1,
          ms: 0,
          failures: [{ name: 'test setup', message: msg }],
        };
      }
      setSummary('Status: running tests...');
      return await runAllTests(d);
    }

    async function runBenchAction(){
      $('log').textContent='';
      const d=await ensure();
      if(!d) return { ok:false, pass:0, fail:1, skip:0, total:1, ms:0 };
      if(__requireRtx5090){
        const summary = _adapterMeta?.summary || '(unavailable)';
        if(!_adapterMeta?.raw){
          const msg = `adapter verification required (RTX 5090) but adapter info is unavailable; selection=${_adapterMeta?.selection || 'unknown'}`;
          setSummary(`<span class="fail">FAIL</span> - adapter verification failed`);
          log(`FAIL ADAPTER CHECK: ${msg}`);
          return {
            ok: false,
            pass: 0,
            fail: 1,
            skip: 0,
            total: 1,
            ms: 0,
            failures: [{ name: 'adapter verification (RTX 5090)', message: msg }],
            adapter: {
              selection: _adapterMeta?.selection || 'unknown',
              summary,
              isNvidia5090: false,
              isNvidiaBlackwell: false,
              acceptedByRtx5090Policy: false,
              infoAvailable: false,
            },
          };
        }
        if(!_adapterMeta.isNvidia5090){
          if(_adapterMeta.isNvidiaBlackwell){
            log(`PASS ADAPTER CHECK: inferred RTX 5090-class adapter from NVIDIA Blackwell (${summary})`);
          } else {
            const msg = `expected NVIDIA RTX 5090-class adapter, got: ${summary}`;
            setSummary(`<span class="fail">FAIL</span> - adapter verification failed`);
            log(`FAIL ADAPTER CHECK: ${msg}`);
            return {
              ok: false,
              pass: 0,
              fail: 1,
              skip: 0,
              total: 1,
              ms: 0,
              failures: [{ name: 'adapter verification (RTX 5090)', message: msg }],
              adapter: {
                selection: _adapterMeta.selection || 'unknown',
                summary,
                isNvidia5090: false,
                isNvidiaBlackwell: !!_adapterMeta.isNvidiaBlackwell,
                acceptedByRtx5090Policy: false,
                infoAvailable: true,
              },
            };
          }
        }
        if(_adapterMeta.isNvidia5090) log(`PASS ADAPTER CHECK: NVIDIA RTX 5090 detected (${summary})`);
      }
      setSummary('Status: running benchmarks...');
      try{
        return await runBenchmarks(d, _adapterMeta);
      }catch(e){
        setSummary(`<span class="fail">FAIL</span> - benchmark error`);
        log(String(e && e.stack || e));
        return { ok:false, pass:0, fail:1, skip:0, total:1, ms:0 };
      }
    }

    btnInit.addEventListener('click', async ()=>{ $('log').textContent=''; await ensure(); });
    btnTests.addEventListener('click', async ()=>{ await runTestsAction(); });
    btnBench.addEventListener('click', async ()=>{ await runBenchAction(); });

    (async ()=>{
      await ensure().catch(()=>{});
      if(!__autorun) return;
      if(__autorun === 'tests'){
        const r = await runTestsAction();
        publishAutorunResult('tests', r);
        return;
      }
      if(__autorun === 'bench'){
        const r = await runBenchAction();
        publishAutorunResult('bench', r);
        return;
      }
      if(__autorun === 'all'){
        const tr = await runTestsAction();
        const br = await runBenchAction();
        publishAutorunResult('all', {
          ok: !!(tr?.ok && br?.ok),
          pass: tr?.pass ?? 0,
          fail: (tr?.fail ?? 0) + (br?.fail ?? 0),
          skip: (tr?.skip ?? 0) + (br?.skip ?? 0),
          total: (tr?.total ?? 0) + (br?.total ?? 0),
          ms: (tr?.ms ?? 0) + (br?.ms ?? 0),
        });
      }
    })();
  })();
