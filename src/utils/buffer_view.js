/**
 * BufferView describes a logical byte range potentially split across segments.
 * This is the public abstraction requested for buffer splitting.
 *
 * NOTE:
 * - Plans may operate on a contiguous internal workspace. When input/output are multi-segment, the library
 *   will pack/unpack once per exec (Tier A: WGSL segmented copy for small segment counts; Tier B: multiple
 *   GPU copyBufferToBuffer ops for larger segment counts). If the logical transform domain exceeds
 *   device.limits.maxStorageBufferBindingSize, plan creation rejects it (no out-of-core in current implementation).
 */
export class BufferView {
  /**
   * @param {object} opts
   * @param {{buffer: GPUBuffer, offsetBytes: number, sizeBytes: number}[]} opts.segments
   * @param {number} [opts.logicalByteOffset=0]
   * @param {number} opts.lengthBytes
   */
  constructor({ segments, logicalByteOffset = 0, lengthBytes }) {
    if (!Array.isArray(segments) || segments.length === 0) throw new Error("BufferView.segments must be a non-empty array");
    if (!Number.isInteger(logicalByteOffset) || logicalByteOffset < 0) throw new Error("BufferView.logicalByteOffset must be a non-negative integer");
    if (!Number.isInteger(lengthBytes) || lengthBytes <= 0) throw new Error("BufferView.lengthBytes must be a positive integer");
    for (const s of segments) {
      if (!s?.buffer) throw new Error("BufferView segment missing buffer");
      if (!Number.isInteger(s.offsetBytes) || s.offsetBytes < 0) throw new Error("BufferView segment offsetBytes must be non-negative integer");
      if (!Number.isInteger(s.sizeBytes) || s.sizeBytes <= 0) throw new Error("BufferView segment sizeBytes must be positive integer");
      if (s.offsetBytes + s.sizeBytes > s.buffer.size) {
        throw new Error(`BufferView segment out of bounds: offsetBytes+sizeBytes=${s.offsetBytes + s.sizeBytes} > buffer.size=${s.buffer.size}`);
      }
    }
    this.segments = segments;
    this.logicalByteOffset = logicalByteOffset;
    this.lengthBytes = lengthBytes;
  }

  static fromBuffer(buffer, offsetBytes = 0, lengthBytes = buffer.size - offsetBytes) {
    return new BufferView({
      segments: [{ buffer, offsetBytes, sizeBytes: lengthBytes }],
      logicalByteOffset: 0,
      lengthBytes,
    });
  }
}

