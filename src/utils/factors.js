export const SUPPORTED_RADICES = [2, 3, 4, 5, 7, 8, 11, 13];

export function factorizeSupportedRadices(n) {
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

export function isPrime(n) {
  if (!Number.isInteger(n) || n < 2) return false;
  if (n % 2 === 0) return n === 2;
  for (let d = 3; d * d <= n; d += 2) {
    if (n % d === 0) return false;
  }
  return true;
}

export function gcd(a, b) {
  let x = Math.abs(a | 0);
  let y = Math.abs(b | 0);
  while (y !== 0) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x;
}

export function modPow(base, exp, mod) {
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

export function primeFactors(n) {
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

export function primitiveRootPrime(p) {
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

export function nextSmoothAtLeast(minN) {
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

export function nextPow2(n) {
  if (!Number.isInteger(n) || n <= 0) throw new Error(`nextPow2: n must be positive int, got ${n}`);
  let x = 1;
  while (x < n) x <<= 1;
  return x;
}

