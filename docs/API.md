# API Reference

Detailed API and behavior notes for `webgpufft`.

Source: extracted from the repository README to keep the root README concise.

## Public API

```js
createPlan(device, {
  shape: number[], // rank>=1
  type: "c2c"|"r2c"|"c2r"|"dct1"|"dct2"|"dct3"|"dct4"|"dst1"|"dst2"|"dst3"|"dst4"|"fftconv"|"conv2d",
  direction: "forward"|"inverse", // required for c2c/r2c/c2r/dct*/dst*; ignored by fftconv/conv2d
  batch: 1,
  inPlace: false, // only c2c supports inPlace:true in current implementation
  normalize: "none"|"backward"|"unitary",
  layout: {
    interleavedComplex: true|false,
    strides?: number[],
    inputStrides?: number[],
    outputStrides?: number[],
    offsetElements?: number,
    inputOffsetElements?: number,
    outputOffsetElements?: number,
    batchStrideElements?: number,
    inputBatchStrideElements?: number,
    outputBatchStrideElements?: number,
    whdcn?: {
      channels?: number,
      channelIndex?: number,
      channelStrideElements?: number,
      batchStrideElements?: number,
      offsetElements?: number,
      input?: {
        channels?: number,
        channelIndex?: number,
        channelStrideElements?: number,
        batchStrideElements?: number,
        offsetElements?: number
      },
      output?: {
        channels?: number,
        channelIndex?: number,
        channelStrideElements?: number,
        batchStrideElements?: number,
        offsetElements?: number
      }
    }
  },
  precision: "f32"|"f16-storage", // f16-storage: c2c/r2c/c2r/dct*/dst* only
  ioView?: {
    input?:  { shape: number[], offset?: number[], placement?: "start"|"center" },
    output?: { shape: number[], offset?: number[], placement?: "start"|"center", clearOutside?: boolean }
  },
  zeroPad?: {
    read?:  { start?: number[], end?: number[] },
    write?: { start?: number[], end?: number[] }
  },
  fftConv?: {
    mode?: "convolution"|"correlation",
    boundary?: "circular"|"linear-full"|"linear-same"|"linear-valid", // default "circular"
    kernelShape?: number[], // default shape
    kernelCount?: number, // default 1
    outputLayout?: "kernel-major"|"batch-major", // default "kernel-major"
    channelPolicy?: {
      input?: {
        channels: number,
        channelIndex?: number,
        channelStrideElements?: number,
        batchStrideElements?: number,
        offsetElements?: number
      },
      output?: {
        channels: number,
        channelIndex?: number,
        channelStrideElements?: number,
        batchStrideElements?: number,
        offsetElements?: number,
        kernelStepChannels?: number // default 1
      }
    },
    outputKernelStrideElements?: number, // optional explicit per-kernel output lane stride for multi-kernel strided output
    tuning?: {
      pointwiseChunkElements?: number, // max complex elements processed per pointwise chunk
      extractCopyChunkElements?: number // max complex elements per fallback extraction copy chunk
    }
  }, // only for type:"fftconv"
  tuning?: {
    workgroupSizeX?: number, // global compute workgroup X override
    raderMaxPrime?: number, // c2c axis selection threshold (default 4096)
    forceBluesteinAxes?: number[], // c2c axis override
    forceRaderAxes?: number[], // c2c axis override (prime axes only)
    transposeMinElements?: number, // c2c 2D transpose threshold (default 4096)
    disableTranspose?: boolean, // c2c 2D transpose off switch
    disableOutOfCoreFourStep?: boolean, // c2c out-of-core fallback off switch
    largeRoute?: "auto"|"chunk"|"out-of-core", // shared large-mode routing preference (default "auto")
    preferOutOfCoreForStrided?: boolean, // shared override for strided large-route preference
    largeChunkMaxBatches?: number, // c2c large-batch per-chunk cap
    swapTo2Stage4Step?: number, // staged large-window scheduler threshold (>= this axis length -> 2 uploads)
    swapTo3Stage4Step?: number, // staged large-window scheduler threshold (>= this axis length -> 3 uploads)
    groupedBatch?: number|Array<number|null>, // line-window grouping hint for staged large scheduling
    outOfCoreBurstWindows?: number, // scheduler metadata knob for staged out-of-core window bursts
    maxStorageBufferBindingSize?: number // effective bind-size cap for large-route scheduler decisions (advanced/testing)
  },
  cache?: {
    snapshot?: PipelineCacheSnapshot // optional snapshot to prewarm shader modules
  },
  conv?: { ... } // only for type:"conv2d"
}) -> plan

plan.exec(commandEncoder, {
  input: GPUBuffer | BufferView,
  output?: GPUBuffer | BufferView,
  temp?: GPUBuffer | BufferView,
  inputOffsetBytes?: number,
  outputOffsetBytes?: number,
  kernel?: GPUBuffer | Float32Array // required for conv2d and fftconv
})

plan.getWorkspaceSizeBytes()
plan.getPipelineCacheSnapshot()
plan.destroy()
```

```js
exportPipelineCacheSnapshot(device) -> PipelineCacheSnapshot
importPipelineCacheSnapshot(device, snapshot) -> PipelineCacheSnapshot

createFftConvChannelLanePreset(opts) -> FftConvPreset
createFftConvKernelMajorChannelLanePreset(opts) -> FftConvPreset
createFftConvBatchMajorChannelLanePreset(opts) -> FftConvPreset
```

Notes:
- `inPlace:true` is currently supported only on `c2c`.
- `r2c` supports `direction:"forward"` only; `c2r` supports `direction:"inverse"` only.
- Strided/offset layout in `precision:"f32"` is supported on `c2c`, `r2c`, `c2r`, `dct*`, and `dst*`:
  - `layout.strides` (shared) or side-specific `layout.inputStrides` / `layout.outputStrides`,
  - optional side/shared offsets and batch strides (`*OffsetElements`, `*BatchStrideElements`).
- `layout.whdcn` is a shorthand for channel-lane WHD+CN-style mappings on the same families:
  - supports `channels`, `channelIndex`, `channelStrideElements`, `batchStrideElements`, `offsetElements`,
  - supports side-specific `layout.whdcn.input` / `layout.whdcn.output`,
  - resolves per-side against the physical side shape (including packed-domain side for `r2c`/`c2r`),
  - explicit stride/offset/batch fields keep priority when provided for a side.
- Custom strides and `layout.whdcn` can be combined with non-trivial `ioView` and `zeroPad` flows on these families.
- `ioView` mapping supports rank>3 on `c2c`, `r2c`, `c2r`, `dct*`, and `dst*`.
- `zeroPad.read` and `zeroPad.write` apply range-based native zeroing in the logical domain:
  - `c2c`: complex logical domain (`shape`)
  - `r2c`: read on real logical domain (`shape`), write on packed domain (`[floor(Nx/2)+1, ...]`)
  - `c2r`: read on packed domain, write on real logical domain (`shape`)
  - `dct*`/`dst*`: real logical domain (`shape`)
  - `fftconv`: FFT logical domain (`fftShape`; `shape` for circular, `shape+kernelShape-1` for linear modes)
- `fftconv` supports complex convolution/correlation for rank >=1 with:
  - `fftConv.boundary:"circular"` (default), or
  - linear crop modes: `"linear-full"|"linear-same"|"linear-valid"` (with optional `fftConv.kernelShape`).
- `fftconv` supports `precision:"f32"` only in current implementation.
- `fftconv` multi-kernel workflows:
  - set `fftConv.kernelCount > 1`
  - provide `kernel` at exec as either:
    - a packed `Float32Array` / `GPUBuffer` / `BufferView` of `kernelCount` kernels, or
    - an array of `kernelCount` kernel payloads (`Float32Array` or GPU buffer/view)
  - output layout:
    - `"kernel-major"`: `[kernel][batch][logical]`
    - `"batch-major"`: `[batch][kernel][logical]`
- `conv2d` currently supports `precision:"f32"` only.
- `conv2d` current scope is fixed-rank spatial mode: `shape:[H,W]`, `conv.boundary:"zero"`, `kernelSize in {1,2,3}`, and no `ioView`/`zeroPad`/custom-stride routing.
- `c2c` now supports a large-batch chunk mode when:
  - total bytes exceed `maxStorageBufferBindingSize`,
  - one batch (`product(shape)*8`) still fits the binding limit.
  - non-trivial `ioView.input` / `ioView.output` mappings are supported in this mode via safe per-batch chunking.
  - `zeroPad.read` and `zeroPad.write` are supported in large-batch mode (applied per safe batch chunk).
  - normalization modes (`"none"|"backward"|"unitary"`) are supported.
  - custom strides are supported in this mode through staged gather/scatter paths.
  - if direct GPUBuffer execution is unavailable (for example segmented views), execution uses staging:
    - caller-provided `temp` may be used when it is non-aliasing and exposes one contiguous range of at least `batch*product(shape)*8` bytes
    - if `temp` is omitted, an internal staging buffer is allocated on demand (subject to `device.limits.maxBufferSize`).
- `c2c` also has an out-of-core four-step fallback when one full batch itself exceeds `maxStorageBufferBindingSize`:
  - current scope: rank>=2 `precision:"f32"` with mixed and non-mixed axis support under bounded line/window constraints.
  - supports contiguous or segmented input/output, custom strides, non-trivial `ioView`, and `zeroPad.read`/`zeroPad.write`.
  - rank-2 uses staged transpose + axis-window FFT passes; rank>2 uses staged axis-permutation passes + axis-window FFT
  - normalization is applied with chunked scaling after all axis passes
  - supports segmented/multi-buffer input/output via staged pack/unpack in this mode
  - advanced/testing: `tuning.maxStorageBufferBindingSize` can set a lower effective cap to force scheduler decisions on capable hardware
  - advanced: `tuning.largeRoute` can prefer `"chunk"` vs `"out-of-core"` (with safe validation when impossible)
  - advanced: `tuning.largeChunkMaxBatches` can reduce per-chunk batch size for memory/latency tuning
  - advanced: staged window policy knobs `tuning.swapTo2Stage4Step`, `tuning.swapTo3Stage4Step`, `tuning.groupedBatch`, `tuning.outOfCoreBurstWindows`
  - can be disabled with `tuning.disableOutOfCoreFourStep:true`.
- `r2c`/`c2r` large-shape execution also uses the shared staged line-window policy metadata for real<->complex and pack/unpack chunk scheduling:
  - exposed on plans via `plan._outOfCoreAxisWindowPolicy` (`numAxisUploads`, `linesPerChunk`, `groupedBatch`, `burstWindows`).

## Normalization

Normalization is applied once per plan:

- `normalize: "none"`: no scaling
- `normalize: "backward"`: scale by `1/Ntotal` only for `direction:"inverse"`
- `normalize: "unitary"`: scale by `1/sqrt(Ntotal)` for forward and inverse

`Ntotal = product(shape)` (logical domain size; not including `batch`).

## R2C/C2R packing

For a logical real transform of size `N` along axis 0, the packed spectrum stores bins `k = 0..floor(N/2)`:

- packed length along axis 0 is `Npacked = floor(N/2) + 1`
- complex bins are interleaved `[re0, im0, re1, im1, ...]`

For ND, packing is applied along axis 0 only; other axes are unchanged.

## ioView (padding / embedding)

`shape` in `createPlan` is always the logical transform domain.

`ioView` maps a physical input/output view into that logical domain:

- pad-in-read: if `ioView.input.shape < shape`, reads outside the physical view are treated as zero
- pad-in-write:
  - if `ioView.output.shape < shape`, only the subregion is written
  - if `ioView.output.shape > shape`, the logical output is embedded into the larger physical output; if `clearOutside:true`, the rest is zeroed
- `placement:"center"` with omitted `offset` uses `offset[d] = floor((shape[d] - viewShape[d]) / 2)`

For `r2c`/`c2r`, `ioView.input` / `ioView.output` refer to the physical domains of the real input or packed spectrum, respectively.

## zeroPad (range-based native zeroing)

`zeroPad` lets you zero values outside a logical hyper-rectangle with per-stage control:

- `zeroPad.read`: applied after input load/embedding and before transform passes
- `zeroPad.write`: applied after transform/normalization and before output extraction/write

Each stage accepts:

- `start`: inclusive logical start per axis (default `0`)
- `end`: exclusive logical end per axis (default `shape[d]`)

Example:

```js
zeroPad: {
  read:  { start: [4], end: [12] },
  write: { start: [2], end: [14] }
}
```

For rank `R`, `start` and `end` must be arrays of length `R`.

## fftconv

`type:"fftconv"` implements FFT-based complex convolution/correlation on interleaved f32 buffers:

- rank >= 1 (`shape.length >= 1`)
- batched inputs (`batch >= 1`)
- modes:
  - `fftConv.mode:"convolution"` (default)
  - `fftConv.mode:"correlation"` (conjugates kernel spectrum)
- boundary/output domain:
  - `fftConv.boundary:"circular"` (default): output shape is `shape`
  - `fftConv.boundary:"linear-full"`: output shape is `shape + kernelShape - 1` (per axis)
  - `fftConv.boundary:"linear-same"`: output shape is `shape` (center crop of full linear result)
  - `fftConv.boundary:"linear-valid"`: output shape is `shape - kernelShape + 1` (per axis, requires positive result)
  - `fftConv.kernelShape` defaults to `shape` when omitted
- optional multi-kernel workflows:
  - `fftConv.kernelCount` (default `1`)
  - `fftConv.outputLayout` in `"kernel-major"` (default) or `"batch-major"`
  - channelized feature workflows can be expressed directly with `fftConv.channelPolicy`:
    - `channelPolicy.input` / `channelPolicy.output` use WHD+CN-style fields (`channels`, `channelIndex`, `channelStrideElements`, `batchStrideElements`, `offsetElements`)
    - `channelPolicy.output.kernelStepChannels` maps kernel `k` to output lane `channelIndex + k*kernelStepChannels`
  - for multi-kernel strided output, either:
    - provide `fftConv.channelPolicy.output` (recommended), or
    - provide explicit `fftConv.outputKernelStrideElements`
  - optional tuning:
    - `fftConv.tuning.pointwiseChunkElements` controls pointwise multiply chunk granularity (must fit bind limits)
    - `fftConv.tuning.extractCopyChunkElements` controls fallback extraction copy chunk size
- native `zeroPad` stages are supported:
  - `zeroPad.read` is applied after input embedding and before forward FFT passes
  - `zeroPad.write` is applied after inverse FFT/normalization and before boundary extraction/output write
  - ranges are interpreted in the FFT logical domain (`fftShape`)

`kernel` is required at execution.

- For `kernelCount=1`:
  - one logical kernel of size `product(kernelShape)` as interleaved complex f32 (`GPUBuffer`, `BufferView`, or `Float32Array`).
- For `kernelCount>1`:
  - either a packed payload of `kernelCount * product(kernelShape)` complex values (`GPUBuffer`/`BufferView`/`Float32Array`),
  - or an array with one payload per kernel.

Output size for multi-kernel workflows is `batch * kernelCount * product(outputShape)` complex values, where `outputShape` is determined by `fftConv.boundary` (layout controlled by `fftConv.outputLayout`).

When `fftConv.channelPolicy` is used, do not also set `layout.whdcn` for `fftconv`; use one policy surface.

### FFTConv Channel-Lane Preset Helpers

The API exports helper builders for `fftConv.channelPolicy` presets:

- `createFftConvChannelLanePreset(opts)` (generic output layout)
- `createFftConvKernelMajorChannelLanePreset(opts)` (`outputLayout:"kernel-major"`)
- `createFftConvBatchMajorChannelLanePreset(opts)` (`outputLayout:"batch-major"`)

`opts`:

- `shape: number[]` and `batch: number`
- `kernelCount?: number` (default `1`)
- `mode?: "convolution"|"correlation"` (default `"convolution"`)
- `boundary?: "circular"|"linear-full"|"linear-same"|"linear-valid"` (default `"circular"`)
- `outputLayout?: "kernel-major"|"batch-major"` (generic helper only)
- `layout?: { interleavedComplex?: true }` (must not include stride/`whdcn` fields)
- `input` descriptor:
  - `channels` (required)
  - `channelIndex?`, `channelStrideElements?`, `batchStrideElements?`, `offsetElements?`
- `output` descriptor:
  - same fields as input
  - `kernelStepChannels?` (default `1`) for multi-kernel lane stepping

Each helper returns a plan fragment compatible with `createPlan(device, { type:"fftconv", ...preset })`.

## Pipeline Cache Snapshot

`webgpufft` keeps a shared per-device in-memory pipeline cache in current implementation.

- `exportPipelineCacheSnapshot(device)` returns a versioned snapshot (`schema`, `version`, `createdAtMs`, `metadata`, `shaderCodes`, `pipelineKeys`).
- `importPipelineCacheSnapshot(device, snapshot)` validates and prewarms shader modules from a prior snapshot.
- v2 snapshots use `schema: "webgpufft.pipeline-cache"` and `version: 2`; legacy snapshots are accepted and upgraded on import.
- You can also pass `cache.snapshot` into `createPlan(...)` to import during plan construction.

## conv2d

`type:"conv2d"` implements small spatial convolution (stride=1, dilation=1) on:

- real f32 arrays, or
- complex interleaved f32 arrays (re,im,re,im,...)

At execution time, weights are provided via `kernel` (GPUBuffer or Float32Array). Float32Array kernels are uploaded and cached per plan.

## Buffer splitting (BufferView)

Use `BufferView` to represent a logical byte range split across multiple GPUBuffer segments:

- Tier A: if `segmentCount <= SEG_CAP`, a WGSL segmented-copy kernel binds all segments and packs/unpacks once per exec
- Tier B: if `segmentCount > SEG_CAP`, pack/unpack is done once per exec via multiple `copyBufferToBuffer` calls
- Tier C: if one transform instance itself exceeds `device.limits.maxStorageBufferBindingSize`, execution uses supported large-mode fallbacks (dispatch chunking, batch slicing, axis-window line chunking, and out-of-core/four-step where implemented: `c2c` rank>=2 `precision:"f32"` and large-shape `r2c`/`c2r` rank>=2 `precision:"f32"`). Paths without a compatible large strategy reject with explicit limit diagnostics.

## Tests

```bash
npm test
```

GPU tests auto-skip if WebGPU is unavailable.


