function assertDevice(device) {
  if (!device) throw new Error("Expected a WebGPU device");
}

/**
 * Uploads interleaved complex<f32> data `[re, im, re, im, ...]` into a GPUBuffer.
 * The returned buffer is usable as STORAGE and COPY_SRC for readback.
 */
export function uploadComplex(device, data) {
  assertDevice(device);
  if (!(data instanceof Float32Array)) {
    throw new Error("uploadComplex expects a Float32Array");
  }
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/**
 * Downloads interleaved complex<f32> data from a GPUBuffer into a Float32Array.
 * Returns a Promise because GPUBuffer.mapAsync is asynchronous.
 */
export async function downloadComplex(device, buffer, lengthComplex, offsetBytes = 0) {
  assertDevice(device);
  if (!buffer) throw new Error("downloadComplex expects a GPUBuffer");
  if (!Number.isInteger(lengthComplex) || lengthComplex <= 0) {
    throw new Error(`lengthComplex must be a positive integer; got ${lengthComplex}`);
  }
  if (!Number.isInteger(offsetBytes) || offsetBytes < 0 || offsetBytes % 8 !== 0) {
    throw new Error(`offsetBytes must be a non-negative multiple of 8; got ${offsetBytes}`);
  }

  const byteLength = lengthComplex * 8;
  const readback = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, offsetBytes, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const copy = readback.getMappedRange();
  const out = new Float32Array(copy.slice(0));
  readback.unmap();
  readback.destroy();
  return out;
}

