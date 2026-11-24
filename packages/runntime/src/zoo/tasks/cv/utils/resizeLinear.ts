/** Antialiased bilinear resize of float planes on the CPU, like torch's
 *  `interpolate(mode='bilinear', antialias=True)`. */

import { resampleAxis } from './resampleAxis.ts';

/** The triangle kernel of linear interpolation. */
const triangle = (x: number) => Math.max(0, 1 - Math.abs(x));

/** `chw` [C, srcH, srcW] resized to [C, height, width], width first, with
 *  no rounding in between. */
export function resizeLinear(
  chw: Float32Array,
  srcW: number,
  srcH: number,
  width: number,
  height: number,
): Float32Array {
  const channels = chw.length / (srcW * srcH);
  const xs = resampleAxis(width, srcW, triangle, 1);
  const ys = resampleAxis(height, srcH, triangle, 1);
  const tmp = new Float32Array(channels * srcH * width);
  for (let r = 0; r < channels * srcH; r++) {
    for (let x = 0; x < width; x++) {
      const w0 = x * xs.taps;
      const src = r * srcW + xs.start[x]!;
      let acc = 0;
      for (let j = 0; j < xs.count[x]!; j++) acc += chw[src + j]! * xs.weights[w0 + j]!;
      tmp[r * width + x] = acc;
    }
  }
  const out = new Float32Array(channels * height * width);
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < height; y++) {
      const w0 = y * ys.taps;
      const dst = (c * height + y) * width;
      for (let j = 0; j < ys.count[y]!; j++) {
        const w = ys.weights[w0 + j]!;
        const src = (c * srcH + ys.start[y]! + j) * width;
        for (let x = 0; x < width; x++) out[dst + x]! += tmp[src + x]! * w;
      }
    }
  }
  return out;
}
