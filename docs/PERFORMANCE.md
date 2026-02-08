# Performance And Validation

Benchmarking, browser validation, and runtime performance guidance for `webgpufft`.

## Performance Paths

- 2D transpose for better coalescing can be auto-selected for larger shapes.
- subgroup-accelerated small power-of-two single-axis FFTs are used when `subgroups` is available.

## Benchmark

```bash
npm run bench
```

## Browser validation (recommended path)

Use the built-in server (no third-party dependencies):

```bash
web\serve.cmd
```

Then open:

```text
http://localhost:8000/web/
```

You can also use:

```powershell
.\web\serve.ps1
```

`file://` may work on some machines, but many Chrome setups block WebGPU there.

To run browser validation tests automatically (headless Chrome + local harness):

```bash
web\run_browser_tests.cmd
```

Modes:
- `web\run_browser_tests.cmd tests` (default)
- `web\run_browser_tests.cmd bench`
- `web\run_browser_tests.cmd all`
- `web\run_browser_tests.cmd tests 8011 headed` (force headed mode if headless WebGPU is blocked)

Click:

- “Init WebGPU” (prints enabled features and key limits)
- “Run Tests”
- “Run Benchmarks”

If `navigator.gpu` is missing, check Chrome WebGPU flags (`chrome://flags`, search “WebGPU”), update GPU drivers, and reload.
