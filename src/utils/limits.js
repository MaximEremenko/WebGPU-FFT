export function formatDeviceLimits(limits) {
  const m = limits?.maxComputeWorkgroupsPerDimension;
  return JSON.stringify(
    {
      maxBufferSize: limits?.maxBufferSize,
      maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: limits?.maxStorageBuffersPerShaderStage,
      minStorageBufferOffsetAlignment: limits?.minStorageBufferOffsetAlignment,
      maxComputeInvocationsPerWorkgroup: limits?.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: limits?.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: limits?.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupSizeZ: limits?.maxComputeWorkgroupSizeZ,
      maxComputeWorkgroupStorageSize: limits?.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension: m ? [m[0], m[1], m[2]] : undefined,
    },
    null,
    2
  );
}

export function pickWorkgroupSizeX(limits, preferred = 256) {
  const maxX = limits?.maxComputeWorkgroupSizeX ?? preferred;
  const maxInv = limits?.maxComputeInvocationsPerWorkgroup ?? preferred;
  return Math.max(1, Math.min(preferred, maxX, maxInv));
}

