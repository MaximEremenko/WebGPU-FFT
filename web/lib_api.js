import * as lib from "../src/index.js";

const api = {
  ...lib,
};

globalThis.webgpufft = api;
globalThis.webgpufftReady = Promise.resolve(api);

try {
  globalThis.dispatchEvent(new Event("webgpufft-ready"));
} catch (_) {
  // no-op
}

export default api;
