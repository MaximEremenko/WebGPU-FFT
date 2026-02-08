# Architecture

This document defines the canonical folder layout for `webgpufft` and how new code should be organized.

## Top-level layout

- `src/`: runtime library source.
- `test/`: node/unit/integration tests.
- `web/`: browser harness, in-browser tests, and benchmark runner.
- `bench/`: node benchmark entrypoints.
- `docs/`: user and maintainer documentation.

## Runtime structure (`src/`)

- `src/index.js`: public package exports.
- `src/public_api.js`: stable public API entrypoints.
- `src/plan.js`: low-level FFT execution core used by runtime plans.
- `src/runtime/`: canonical high-level planning and routing layer.

Inside `src/runtime/`:

- `plans/`: user-facing transform plans (`c2c`, `r2c`, `c2r`, `dct/dst`, `conv2d`, `fftconv`).
- `algorithms/`: specialized FFT algorithm executors (Bluestein, Rader).
- shared helpers:
  - `large_policy.js`
  - `layout_semantics.js`
  - `tensor_descriptor.js`
  - `workspace.js`
  - `segmented_io.js`

## Kernel structure (`src/kernels/`)

- WGSL generator modules only.
- One file per kernel family (`stockham1d`, `transpose`, `strided_*`, `zero_pad`, etc).
- Keep kernel codegen pure: no WebGPU device creation or side-effectful runtime state.

## Naming conventions

- File names: `snake_case.js`.
- Plan classes: `PascalCase` (`C2CPlan`, `R2CPlan`).
- Kernel generators: `generate*WGSL`.
- Shared policy/descriptor modules must stay family-agnostic.

## Legacy/unused code policy

- Do not introduce parallel duplicate trees for the same runtime logic.
- Remove deprecated shims once all imports have migrated to canonical paths.
- Any planned removals should include:
  - import search proof (`rg` results),
  - test pass (`node --test`),
  - doc update in `PORT_STATUS.md` or `docs/README.md` when behavior changes.

