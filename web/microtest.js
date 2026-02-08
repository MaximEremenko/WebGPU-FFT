export class SkipError extends Error {
  constructor(message) {
    super(message);
    this.name = "SkipError";
  }
}

export function assert(cond, message = "assertion failed") {
  if (!cond) throw new Error(message);
}

export function assertCloseArray(actual, expected, tolAbs = 1e-4, tolRel = 1e-4, message = "arrays not close") {
  if (actual.length !== expected.length) {
    throw new Error(`${message}: length mismatch actual=${actual.length} expected=${expected.length}`);
  }
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const tol = tolAbs + tolRel * Math.abs(e);
    const diff = Math.abs(a - e);
    if (diff > tol) {
      throw new Error(`${message}: i=${i} actual=${a} expected=${e} diff=${diff} tol=${tol}`);
    }
  }
}

export function createMicroTestRunner({ onEvent } = {}) {
  const tests = [];

  function test(name, fn) {
    tests.push({ name, fn });
  }

  async function runAll(ctx = {}) {
    const t0 = performance.now();
    const results = [];
    onEvent?.({ type: "suite_start", count: tests.length });

    for (const { name, fn } of tests) {
      const start = performance.now();
      onEvent?.({ type: "test_start", name });
      try {
        await fn(ctx);
        const dt = performance.now() - start;
        results.push({ name, status: "pass", ms: dt });
        onEvent?.({ type: "test_pass", name, ms: dt });
      } catch (e) {
        const dt = performance.now() - start;
        if (e && e.name === "SkipError") {
          results.push({ name, status: "skip", ms: dt, message: e.message });
          onEvent?.({ type: "test_skip", name, ms: dt, message: e.message });
        } else {
          results.push({ name, status: "fail", ms: dt, error: e });
          onEvent?.({ type: "test_fail", name, ms: dt, error: e });
        }
      }
    }

    const dt = performance.now() - t0;
    const pass = results.filter((r) => r.status === "pass").length;
    const fail = results.filter((r) => r.status === "fail").length;
    const skip = results.filter((r) => r.status === "skip").length;
    const ok = fail === 0;

    const summary = { ok, pass, fail, skip, total: results.length, ms: dt, results };
    onEvent?.({ type: "suite_end", summary });
    return summary;
  }

  return { test, runAll, tests };
}

// Convenience singleton (useful for very small scripts):
const _default = createMicroTestRunner();
export const test = _default.test;
export const runAll = _default.runAll;
export function reset() {
  _default.tests.length = 0;
}
