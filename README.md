# webgpufft

Standalone WebGPU FFT/DCT/convolution library in pure JavaScript (ESM) with WGSL compute shaders.

`webgpufft` follows a plan-based execution model (create plan -> bind resources -> execute) inspired by reference-style workflows.

## Author

Maksim Eremenko

## Quick Start

```js
import { createPlan, uploadComplex, downloadComplex } from "webgpufft";

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const N = 1024;
const plan = createPlan(device, {
  type: "c2c",
  shape: [N],
  direction: "forward",
  batch: 1,
  normalize: "none",
  layout: { interleavedComplex: true },
  precision: "f32",
});

const input = new Float32Array(2 * N);
const inputBuf = uploadComplex(device, input);
const outputBuf = device.createBuffer({
  size: input.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

const encoder = device.createCommandEncoder();
plan.exec(encoder, { input: inputBuf, output: outputBuf });
device.queue.submit([encoder.finish()]);

const out = await downloadComplex(device, outputBuf, N);
plan.destroy();
```

## FFTConv Channel-Lane Preset Helpers

Use preset builders to generate valid `fftConv.channelPolicy` descriptors for multi-kernel channel-lane routing:

```js
import {
  createPlan,
  createFftConvKernelMajorChannelLanePreset,
} from "webgpufft";

const preset = createFftConvKernelMajorChannelLanePreset({
  shape: [256],
  batch: 4,
  kernelCount: 3,
  input: { channels: 64 },
  output: { channels: 128, kernelStepChannels: 16 },
});

const plan = createPlan(device, { type: "fftconv", ...preset });
```

## Documentation

- Documentation index: `docs/README.md`
- API reference: `docs/API.md`
- Performance and browser validation: `docs/PERFORMANCE.md`
- Port status (authoritative progress tracker): `PORT_STATUS.md`
- reference parity comparison and remaining gaps: `FEATURE_GAP.md`
- Contribution workflow and quality gate: `CONTRIBUTING.md`
- Internal implementation process docs: `docs/internal/`

## Supported in current implementation (Summary)

- Transforms:
  - `c2c`, `r2c`, `c2r`
  - `dct1`, `dct2`, `dct3`, `dct4`
  - `dst1`, `dst2`, `dst3`, `dst4`
  - `fftconv` (FFT-based convolution/correlation)
  - `conv2d` (small spatial convolution)
- Precision:
  - baseline `f32`
  - optional `f16-storage` on `c2c`/`r2c`/`c2r`/`dct*`/`dst*`
  - `fftconv` and `conv2d` are `f32`-only in current implementation
- Large-mode behavior:
  - dispatch chunking and batch slicing
  - out-of-core/four-step support where implemented (see `docs/API.md` and `PORT_STATUS.md`)
- Segmented buffers through `BufferView` (Tier A/B/C policy)

## Important Constraints

- `inPlace:true` is supported only for `c2c` in current implementation.
- `r2c` is forward-only and `c2r` is inverse-only.
- Some large fallback routes are currently `precision:"f32"` only.
- See `PORT_STATUS.md` for current partial gaps and deferred items.

## Test And Validate

```bash
npm test
npm run bench
```

For browser validation, see `docs/PERFORMANCE.md`.

## Repository Layout

- Runtime code: `src/`
- Kernels (WGSL generators): `src/kernels/`
- Canonical runtime planners: `src/runtime/plans/`
- Canonical algorithm executors: `src/runtime/algorithms/`
- Tests: `test/` and `web/`
- User docs: root `.md` + `docs/`
- Internal process docs: `docs/internal/`
- Architecture and structure guide: `docs/ARCHITECTURE.md`
