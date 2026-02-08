// Copyright (c) 2026 Maksim Eremenko

import { BufferView } from "../utils/buffer_view.js";

export function viewFromArena(arena, offsetBytes, lengthBytes) {
  if (arena instanceof BufferView) {
    return new BufferView({
      segments: arena.segments,
      logicalByteOffset: arena.logicalByteOffset + offsetBytes,
      lengthBytes,
    });
  }
  return BufferView.fromBuffer(arena, offsetBytes, lengthBytes);
}

export function createInternalArena(device, sizeBytes) {
  if (sizeBytes <= 0) return null;
  return device.createBuffer({
    size: sizeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

