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

export function normalizeScaleFactor({ normalize, direction, nTotal }) {
  if (normalize === "none") return 1.0;
  if (normalize === "unitary") return 1.0 / Math.sqrt(nTotal);
  if (normalize === "backward") return direction === "inverse" ? 1.0 / nTotal : 1.0;
  throw new Error(`Unknown normalize mode: ${normalize}`);
}

/**
 * Reference radix-2 FFT (DIT with bit reversal), interleaved complex<f32>.
 * Returns a new Float32Array of length 2*N.
 */
export function fft1dRefInterleaved(input, N, direction) {
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

export function fftNdRefInterleaved(input, shape, direction, normalize = "none") {
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

export function randomComplexInterleaved(lengthComplex, rng = Math.random) {
  const out = new Float32Array(2 * lengthComplex);
  for (let i = 0; i < lengthComplex; i++) {
    // centered-ish small range to avoid overflow and to keep errors readable
    out[2 * i] = (rng() * 2 - 1) * 0.5;
    out[2 * i + 1] = (rng() * 2 - 1) * 0.5;
  }
  return out;
}

export function dft1dRefInterleaved(input, N, direction) {
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

export function fftNdRefAnySizeInterleaved(input, shape, direction, normalize = "none") {
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

export function r2cRefPackedInterleaved(inputReal, N, direction = "forward", normalize = "none") {
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

export function c2rRefFromPackedInterleaved(inputPacked, N, normalize = "none") {
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

export function dct2Ref(input, N, direction = "forward") {
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

export function dct3Ref(input, N, direction = "forward") {
  // DCT-III forward is inverse of DCT-II up to scale; we provide both directions.
  return dct2Ref(input, N, direction === "forward" ? "inverse" : "forward");
}

export function dct4Ref(input, N) {
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

export function dst1Ref(input, N) {
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

export function dst2Ref(input, N, direction = "forward") {
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

export function dst3Ref(input, N, direction = "forward") {
  // DST-III forward is inverse of DST-II up to scale; we provide both directions.
  return dst2Ref(input, N, direction === "forward" ? "inverse" : "forward");
}

export function dst4Ref(input, N) {
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

export function dct1Ref(input, N) {
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

export function conv2dRef({
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

export function fftConvRef({
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
