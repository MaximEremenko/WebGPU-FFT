// Copyright (c) 2026 Maksim Eremenko

import { assertOneOf } from "./common.js";

import { C2CPlan } from "./plans/c2c.js";
import { R2CPlan } from "./plans/r2c.js";
import { C2RPlan } from "./plans/c2r.js";
import { DctPlan } from "./plans/dct_fft.js";
import { Conv2dPlan } from "./plans/conv2d.js";
import { FftConvPlan } from "./plans/fftconv.js";

export function createPlan(device, opts) {
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
