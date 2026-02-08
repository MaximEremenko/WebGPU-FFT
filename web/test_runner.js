import { createMicroTestRunner, assert, assertCloseArray, SkipError } from "./microtest.js";
import { registerCompleteTests, compareGoldenVectors } from "../test/complete.suite.js";

function el(id) {
  const x = document.getElementById(id);
  if (!x) throw new Error(`Missing element #${id}`);
  return x;
}

function appendLog(line) {
  const log = el("log");
  log.textContent += line + "\n";
  log.scrollTop = log.scrollHeight;
}

function setSummary(html) {
  el("summary").innerHTML = html;
}

function fmtMs(ms) {
  return `${ms.toFixed(2)} ms`;
}

function formatLimits(limits) {
  const m = limits?.maxComputeWorkgroupsPerDimension;
  return {
    maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: limits?.maxStorageBuffersPerShaderStage,
    maxComputeWorkgroupStorageSize: limits?.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: limits?.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits?.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: limits?.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: limits?.maxComputeWorkgroupSizeZ,
    maxComputeWorkgroupsPerDimension: m ? [m[0], m[1], m[2]] : undefined,
  };
}

function buildRequiredLimitsFromAdapter(adapter) {
  const lim = adapter?.limits;
  if (!lim) return null;
  const keys = [
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupSizeY",
    "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension",
  ];
  const out = {};
  for (const k of keys) {
    const v = lim[k];
    if (Number.isFinite(v) && v > 0) out[k] = Math.floor(v);
  }
  if (Number.isFinite(out.maxBufferSize) && Number.isFinite(out.maxStorageBufferBindingSize)) {
    out.maxStorageBufferBindingSize = Math.min(out.maxStorageBufferBindingSize, out.maxBufferSize);
  }
  return Object.keys(out).length ? out : null;
}

async function requestDeviceWithOptionalFeatures() {
  const gpu = globalThis.navigator?.gpu ?? globalThis.gpu;
  if (!gpu?.requestAdapter) throw new Error("WebGPU unavailable: navigator.gpu.requestAdapter missing");

  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Failed to acquire WebGPU adapter");

  const want = [];
  for (const f of ["shader-f16", "subgroups"]) {
    try {
      if (adapter.features?.has?.(f)) want.push(f);
    } catch {
      // ignore
    }
  }
  const requiredLimits = buildRequiredLimitsFromAdapter(adapter);
  const desc = want.length ? { requiredFeatures: want } : {};
  if (requiredLimits) desc.requiredLimits = requiredLimits;
  let device;
  try {
    device = await adapter.requestDevice(desc);
  } catch {
    device = await adapter.requestDevice(want.length ? { requiredFeatures: want } : {});
  }
  return { adapter, device, requiredFeatures: want };
}

async function getAdapterInfo(adapter) {
  if (adapter?.requestAdapterInfo) {
    try {
      return await adapter.requestAdapterInfo();
    } catch {
      return null;
    }
  }
  return null;
}

function renderGpuInfo({ adapterInfo, device, requiredFeatures }) {
  const info = el("gpuInfo");
  info.textContent = "";
  const kv = [];

  if (adapterInfo) {
    kv.push(["adapter.vendor", adapterInfo.vendor ?? "(unknown)"]);
    kv.push(["adapter.architecture", adapterInfo.architecture ?? "(unknown)"]);
    kv.push(["adapter.device", adapterInfo.device ?? "(unknown)"]);
    kv.push(["adapter.description", adapterInfo.description ?? "(unknown)"]);
  } else {
    kv.push(["adapter.info", "(adapter.requestAdapterInfo unavailable)"]);
  }

  const enabled = [];
  try {
    for (const f of device.features) enabled.push(f);
  } catch {
    // ignore
  }
  kv.push(["device.features.enabled", enabled.join(", ") || "(none)"]);
  kv.push(["device.features.requested", requiredFeatures.join(", ") || "(none)"]);

  const lim = formatLimits(device.limits);
  for (const [k, v] of Object.entries(lim)) {
    kv.push([`limits.${k}`, Array.isArray(v) ? JSON.stringify(v) : String(v)]);
  }

  for (const [k, v] of kv) {
    const a = document.createElement("div");
    a.textContent = k;
    const b = document.createElement("div");
    b.textContent = v;
    info.appendChild(a);
    info.appendChild(b);
  }
}

function downloadJson(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function initTestUI() {
  const btnInit = el("btnInit");
  const btnTests = el("btnTests");
  const chkExport = el("chkExport");
  const fileGolden = el("fileGolden");
  const btnCompare = el("btnCompareGolden");

  let state = {
    adapter: null,
    device: null,
    adapterInfo: null,
    requiredFeatures: [],
    golden: null,
  };

  async function ensureDevice() {
    if (state.device) return state.device;
    appendLog("Requesting WebGPU adapter/device...");
    const { adapter, device, requiredFeatures } = await requestDeviceWithOptionalFeatures();
    const adapterInfo = await getAdapterInfo(adapter);
    state = { ...state, adapter, device, adapterInfo, requiredFeatures };
    renderGpuInfo({ adapterInfo, device, requiredFeatures });
    appendLog("WebGPU initialized.");
    appendLog(`Enabled features: ${[...device.features].join(", ") || "(none)"}`);
    appendLog(`Limits: ${JSON.stringify(formatLimits(device.limits))}`);
    btnTests.disabled = false;
    const benchBtn = document.getElementById("btnBench");
    if (benchBtn) benchBtn.disabled = false;
    return device;
  }

  btnInit.addEventListener("click", async () => {
    try {
      await ensureDevice();
    } catch (e) {
      appendLog(String(e?.stack ?? e));
      setSummary(`<span class="fail">FAIL</span> â€“ init error`);
    }
  });

  btnTests.addEventListener("click", async () => {
    const artifacts = [];
    const exportEnabled = !!chkExport.checked;
    el("log").textContent = "";
    setSummary("Status: running tests...");

    try {
      const device = await ensureDevice();
      const runner = createMicroTestRunner({
        onEvent: (ev) => {
          if (ev.type === "test_pass") {
            appendLog(`PASS ${ev.name} (${fmtMs(ev.ms)})`);
            console.log("PASS", ev.name, ev.ms);
          }
          if (ev.type === "test_skip") {
            appendLog(`SKIP ${ev.name}: ${ev.message}`);
            console.warn("SKIP", ev.name, ev.message);
          }
          if (ev.type === "test_fail") {
            appendLog(`FAIL ${ev.name}: ${String(ev.error?.stack ?? ev.error)}`);
            console.error("FAIL", ev.name, ev.error);
          }
        },
      });

      registerCompleteTests({
        test: runner.test,
        getDevice: async () => device,
        assert,
        assertCloseArray,
        SkipError,
        log: appendLog,
        exportArtifact: exportEnabled ? (a) => artifacts.push(a) : null,
      });

      const summary = await runner.runAll();
      const status = summary.ok ? `<span class="pass">PASS</span>` : `<span class="fail">FAIL</span>`;
      const extra = summary.skip ? ` (<span class="skip">${summary.skip} skipped</span>)` : "";
      setSummary(`${status} â€“ ${summary.pass}/${summary.total} tests, ${fmtMs(summary.ms)}${extra}`);

      if (exportEnabled) {
        const payload = {
          schema: "webgpufft-golden",
          createdAt: new Date().toISOString(),
          requiredFeatures: state.requiredFeatures,
          enabledFeatures: [...device.features],
          limits: formatLimits(device.limits),
          adapterInfo: state.adapterInfo ?? null,
          artifacts,
        };
        const ts = payload.createdAt.replace(/[:.]/g, "-");
        downloadJson(`webgpufft_golden_${ts}.json`, payload);
        appendLog(`Exported ${artifacts.length} golden vectors.`);
      }
    } catch (e) {
      appendLog(String(e?.stack ?? e));
      setSummary(`<span class="fail">FAIL</span> â€“ runner error`);
    }
  });

  fileGolden.addEventListener("change", async () => {
    const f = fileGolden.files?.[0] ?? null;
    state.golden = null;
    btnCompare.disabled = true;
    if (!f) return;
    try {
      const text = await f.text();
      const obj = JSON.parse(text);
      state.golden = obj;
      btnCompare.disabled = false;
      appendLog(`Loaded golden JSON: ${f.name} (artifacts=${obj?.artifacts?.length ?? 0})`);
    } catch (e) {
      appendLog(`Failed to load JSON: ${String(e)}`);
    }
  });

  btnCompare.addEventListener("click", async () => {
    el("log").textContent = "";
    setSummary("Status: comparing imported golden...");
    try {
      const device = await ensureDevice();
      if (!state.golden) throw new Error("No golden JSON loaded");
      const runner = createMicroTestRunner({
        onEvent: (ev) => {
          if (ev.type === "test_pass") {
            appendLog(`PASS ${ev.name} (${fmtMs(ev.ms)})`);
            console.log("PASS", ev.name, ev.ms);
          }
          if (ev.type === "test_skip") {
            appendLog(`SKIP ${ev.name}: ${ev.message}`);
            console.warn("SKIP", ev.name, ev.message);
          }
          if (ev.type === "test_fail") {
            appendLog(`FAIL ${ev.name}: ${String(ev.error?.stack ?? ev.error)}`);
            console.error("FAIL", ev.name, ev.error);
          }
        },
      });

      compareGoldenVectors({
        test: runner.test,
        getDevice: async () => device,
        assert,
        assertCloseArray,
        SkipError,
        golden: state.golden,
        log: appendLog,
      });

      const summary = await runner.runAll();
      const status = summary.ok ? `<span class="pass">PASS</span>` : `<span class="fail">FAIL</span>`;
      const extra = summary.skip ? ` (<span class="skip">${summary.skip} skipped</span>)` : "";
      setSummary(`${status} â€“ ${summary.pass}/${summary.total} compares, ${fmtMs(summary.ms)}${extra}`);
    } catch (e) {
      appendLog(String(e?.stack ?? e));
      setSummary(`<span class="fail">FAIL</span> â€“ compare error`);
    }
  });
}

