/** Antialiased resampling weights along one axis, as torch's antialiased
 *  `interpolate` and PIL build them: downscaling widens the kernel by the
 *  scale, so every source pixel counts and fine detail does not alias. */

/** For each output index, the first source index and the normalized
 *  weights of the source pixels it blends, `taps` slots per output. */
export interface Axis {
  start: Int32Array;
  count: Int32Array;
  weights: Float32Array;
  taps: number;
}

/** The weights of `kernel`, which is 0 past `radius` source pixels at
 *  scale 1. */
export function resampleAxis(
  outLen: number,
  srcLen: number,
  kernel: (x: number) => number,
  radius: number,
): Axis {
  const scale = srcLen / outLen;
  const support = radius * Math.max(scale, 1);
  const inv = 1 / Math.max(scale, 1);
  const taps = Math.ceil(support) * 2 + 1;
  const start = new Int32Array(outLen);
  const count = new Int32Array(outLen);
  const weights = new Float32Array(outLen * taps);
  for (let i = 0; i < outLen; i++) {
    const center = scale * (i + 0.5);
    const lo = Math.max(Math.trunc(center - support + 0.5), 0);
    const n = Math.min(Math.trunc(center + support + 0.5), srcLen) - lo;
    let total = 0;
    for (let j = 0; j < n; j++) {
      const w = kernel((j + lo - center + 0.5) * inv);
      weights[i * taps + j] = w;
      total += w;
    }
    if (total !== 0) for (let j = 0; j < n; j++) weights[i * taps + j]! /= total;
    start[i] = lo;
    count[i] = n;
  }
  return { start, count, weights, taps };
}
