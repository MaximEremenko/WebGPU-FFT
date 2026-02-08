import test, { describe, before } from "node:test";
import assert from "node:assert/strict";

import { createFftPlan, uploadComplex, downloadComplex } from "../src/index.js";
import {
  fftNdRefInterleaved,
  randomComplexInterleaved,
} from "../src/utils/math.js";

async function getWebGpuDevice() {
  const gpu = globalThis.navigator?.gpu ?? globalThis.gpu;
  if (!gpu?.requestAdapter) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  return adapter.requestDevice();
}

function assertAllCloseComplex(actual, expected, { atol = 1e-4, rtol = 1e-4 } = {}) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const tol = atol + rtol * Math.abs(e);
    const diff = Math.abs(a - e);
    if (diff > tol) {
      throw new assert.AssertionError({
        message: `Mismatch at f32[${i}]: actual=${a} expected=${e} diff=${diff} tol=${tol}`,
        actual: a,
        expected: e,
      });
    }
  }
}

async function runFftOnce(device, planOpts, execOpts, inputData) {
  const nTotal = planOpts.shape.reduce((a, b) => a * b, 1);
  const batch = execOpts.batch ?? 1;
  const totalComplex = nTotal * batch;

  const inputBuf = uploadComplex(device, inputData);
  let outputBuf = null;
  if (!planOpts.inPlace) {
    outputBuf = device.createBuffer({
      size: inputData.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
  }

  const plan = createFftPlan(device, {
    ...planOpts,
    layout: "interleaved",
    precision: "f32",
  });

  const encoder = device.createCommandEncoder();
  plan.exec(encoder, { input: inputBuf, output: outputBuf, ...execOpts });
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  const out = await downloadComplex(
    device,
    planOpts.inPlace ? inputBuf : outputBuf,
    totalComplex
  );
  plan.destroy();
  inputBuf.destroy();
  if (outputBuf) outputBuf.destroy();
  return out;
}

describe("webgpufft correctness", () => {
  let device = null;

  before(async () => {
    device = await getWebGpuDevice();
  });

  test("skips when WebGPU unavailable", (t) => {
    if (!device) return t.skip("WebGPU unavailable in this runtime");
    assert.ok(device);
  });

  for (const N of [8, 16, 32, 64, 128]) {
    test(`1D forward C2C matches CPU (N=${N})`, async (t) => {
      if (!device) return t.skip("WebGPU unavailable in this runtime");
      const input = randomComplexInterleaved(N);

      const gpuOut = await runFftOnce(
        device,
        { shape: [N], direction: "forward", normalize: "none", inPlace: false },
        {},
        input
      );

      const cpuOut = fftNdRefInterleaved(input, [N], "forward", "none");
      assertAllCloseComplex(gpuOut, cpuOut);
    });
  }

  test("1D inverse roundtrip obeys normalization modes (N=64)", async (t) => {
    if (!device) return t.skip("WebGPU unavailable in this runtime");
    const N = 64;
    const input = randomComplexInterleaved(N);

    for (const normalize of ["none", "backward", "unitary"]) {
      const fwd = await runFftOnce(
        device,
        { shape: [N], direction: "forward", normalize, inPlace: false },
        {},
        input
      );
      const inv = await runFftOnce(
        device,
        { shape: [N], direction: "inverse", normalize, inPlace: false },
        {},
        fwd
      );

      const scale = normalize === "none" ? N : 1;
      const expected = new Float32Array(inv.length);
      for (let i = 0; i < input.length; i++) expected[i] = input[i] * scale;
      assertAllCloseComplex(inv, expected, { atol: 2e-4, rtol: 2e-4 });
    }
  });

  test("1D batch>1 forward matches CPU (N=32, batch=4)", async (t) => {
    if (!device) return t.skip("WebGPU unavailable in this runtime");
    const N = 32;
    const batch = 4;
    const input = randomComplexInterleaved(N * batch);

    const gpuOut = await runFftOnce(
      device,
      { shape: [N], direction: "forward", normalize: "none", inPlace: false },
      { batch },
      input
    );

    const cpuOut = new Float32Array(input.length);
    for (let b = 0; b < batch; b++) {
      const slice = input.subarray(2 * b * N, 2 * (b + 1) * N);
      const out = fftNdRefInterleaved(slice, [N], "forward", "none");
      cpuOut.set(out, 2 * b * N);
    }
    assertAllCloseComplex(gpuOut, cpuOut);
  });

  test("1D inPlace and out-of-place match (N=16)", async (t) => {
    if (!device) return t.skip("WebGPU unavailable in this runtime");
    const N = 16;
    const input = randomComplexInterleaved(N);

    const outOfPlace = await runFftOnce(
      device,
      { shape: [N], direction: "forward", normalize: "none", inPlace: false },
      {},
      input
    );
    const inPlace = await runFftOnce(
      device,
      { shape: [N], direction: "forward", normalize: "none", inPlace: true },
      {},
      input
    );
    assertAllCloseComplex(inPlace, outOfPlace);
  });

  for (const [Nx, Ny] of [
    [8, 8],
    [16, 16],
  ]) {
    test(`2D forward matches CPU (${Nx}x${Ny})`, async (t) => {
      if (!device) return t.skip("WebGPU unavailable in this runtime");
      const nTotal = Nx * Ny;
      const input = randomComplexInterleaved(nTotal);

      const gpuOut = await runFftOnce(
        device,
        { shape: [Nx, Ny], direction: "forward", normalize: "none", inPlace: false },
        {},
        input
      );
      const cpuOut = fftNdRefInterleaved(input, [Nx, Ny], "forward", "none");
      assertAllCloseComplex(gpuOut, cpuOut, { atol: 3e-4, rtol: 3e-4 });
    });
  }

  test("3D forward matches CPU (8x8x8)", async (t) => {
    if (!device) return t.skip("WebGPU unavailable in this runtime");
    const shape = [8, 8, 8];
    const nTotal = shape[0] * shape[1] * shape[2];
    const input = randomComplexInterleaved(nTotal);

    const gpuOut = await runFftOnce(
      device,
      { shape, direction: "forward", normalize: "none", inPlace: false },
      {},
      input
    );
    const cpuOut = fftNdRefInterleaved(input, shape, "forward", "none");
    assertAllCloseComplex(gpuOut, cpuOut, { atol: 5e-4, rtol: 5e-4 });
  });
});
