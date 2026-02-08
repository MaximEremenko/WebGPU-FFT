export function fnv1a32Bytes(u8) {
  let h = 0x811c9dc5;
  for (let i = 0; i < u8.length; i++) {
    h ^= u8[i];
    h = Math.imul(h, 0x01000193);
  }
  // unsigned
  return h >>> 0;
}

export function hashFloat32Array(a) {
  if (!(a instanceof Float32Array)) throw new Error("hashFloat32Array expects Float32Array");
  return fnv1a32Bytes(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
}

