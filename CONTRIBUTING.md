# Contributing

## Development Setup

1. Use a Node.js version that supports ESM and the built-in test runner.
2. Install dependencies:

```bash
npm install
```

3. Run the test suite:

```bash
npm test
```

4. Run benchmarks when changes affect scheduling or performance paths:

```bash
npm run bench
```

## Browser Validation

Use the built-in server and harness (no third-party runtime dependencies):

```bash
web\serve.cmd
web\run_browser_tests.cmd tests
web\run_browser_tests.cmd bench
```

If headless WebGPU is blocked on your machine, use headed mode:

```bash
web\run_browser_tests.cmd tests 8011 headed
```

## Code Change Expectations

- Keep pure JS + WGSL (ESM).
- Preserve backward compatibility for existing `createPlan` users.
- Respect device limits and alignment constraints.
- Add test coverage for behavior changes.
- Update docs in the same change set:
  - `README.md`
  - `docs/API.md` and/or `docs/PERFORMANCE.md`
  - `PORT_STATUS.md` and `FEATURE_GAP.md` when capability status changes.

## Documentation Policy

- Root docs are user-facing and should stay concise.
- Deep references belong under `docs/`.
- Internal process/checklist docs belong under `docs/internal/`.
- Avoid creating new one-off completed-phase checklist files in the repo root.

