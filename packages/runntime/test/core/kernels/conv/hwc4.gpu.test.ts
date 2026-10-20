import { describe, it, onTestFinished } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import {
  add,
  avgPool2d,
  cat,
  maxPool2d,
  pad2d,
  resizeBilinear2d,
  toChw,
  toHwc4,
  upsample2d,
} from '../../../../src/core/graph/ops.ts';
import type { Value } from '../../../../src/core/graph/value.ts';
import { Conv2d } from '../../../../src/core/nn/layers.ts';
import { memoryStateDict } from '../../../../src/core/weights/safetensors.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root } = await gpu();

interface Map3 {
  data: Float32Array;
  c: number;
  h: number;
  w: number;
}

/** A random CHW map, already rounded to the f16 the hwc4 kernels store. */
function randomMap(c: number, h: number, w: number, seed: number): Map3 {
  return { data: roundF16(randomF32(c * h * w, seed)), c, h, w };
}

/** Uploads a CHW map and moves it into hwc4 storage. */
function hwc4(m: Map3): Value {
  return toHwc4(
    upload(root, m.data, { elems: m.data.length, dtype: 'f32', dims: [m.c, m.h, m.w] }),
  );
}

async function readChw(v: Value): Promise<Float32Array> {
  return toArray(toChw(v, 'f32'));
}

interface ConvCase {
  cIn: number;
  cOut: number;
  h: number;
  w: number;
  k?: number | readonly [number, number];
  stride?: number;
  padding?: number;
  act?: 'relu' | 'silu';
  bias?: boolean;
  winograd?: boolean;
}

function kernel(c: ConvCase): [number, number] {
  return typeof c.k === 'object' ? [c.k[0], c.k[1]] : [c.k ?? 1, c.k ?? 1];
}

/** torch's conv2d (groups 1) as a plain loop in f64, then the activation. */
function convRef(x: Map3, weight: Float32Array, bias: Float32Array | undefined, c: ConvCase) {
  const [kH, kW] = kernel(c);
  const { stride = 1, padding = 0 } = c;
  const hOut = Math.floor((x.h + 2 * padding - kH) / stride) + 1;
  const wOut = Math.floor((x.w + 2 * padding - kW) / stride) + 1;
  const out = new Float32Array(c.cOut * hOut * wOut);
  for (let o = 0; o < c.cOut; o++) {
    for (let oy = 0; oy < hOut; oy++) {
      for (let ox = 0; ox < wOut; ox++) {
        let s = bias?.[o] ?? 0;
        for (let i = 0; i < c.cIn; i++) {
          for (let ky = 0; ky < kH; ky++) {
            const iy = oy * stride + ky - padding;
            if (iy < 0 || iy >= x.h) continue;
            for (let kx = 0; kx < kW; kx++) {
              const ix = ox * stride + kx - padding;
              if (ix < 0 || ix >= x.w) continue;
              s +=
                x.data[(i * x.h + iy) * x.w + ix]! * weight[((o * c.cIn + i) * kH + ky) * kW + kx]!;
            }
          }
        }
        if (c.act === 'relu') s = Math.max(s, 0);
        if (c.act === 'silu') s = s / (1 + Math.exp(-s));
        out[(o * hOut + oy) * wOut + ox] = s;
      }
    }
  }
  return { data: out, c: c.cOut, h: hOut, w: wOut };
}

/** An nn.Conv2d with random f16 weights scaled to keep outputs near 1, and
 *  its CPU reference over `x`. */
async function makeConv(c: ConvCase, x: Map3, seed = 7) {
  const [kH, kW] = kernel(c);
  const scale = 1 / Math.sqrt(c.cIn * kH * kW);
  const weight = roundF16(randomF32(c.cOut * c.cIn * kH * kW, seed).map((v) => v * scale));
  const bias = c.bias ? roundF16(randomF32(c.cOut, seed + 1)) : undefined;
  const conv = new Conv2d(c.cIn, c.cOut, {
    kernelSize: [kH, kW],
    stride: c.stride,
    padding: c.padding,
    bias: c.bias ?? false,
    activation: c.act,
    winograd: c.winograd,
  });
  await conv.loadStateDict(
    memoryStateDict({
      weight: { data: weight, shape: [c.cOut, c.cIn, kH, kW] },
      ...(bias ? { bias } : {}),
    }),
  );
  onTestFinished(() => conv.dispose());
  return { conv, ref: convRef(x, weight, bias, c) };
}

/** f16 storage at every step: products and short runs in f16, sums in f32. */
const DIRECT_TOL = { atol: 1e-2, rtol: 1e-2 };
/** The Winograd transforms scale values by up to 100 before the f16 planes. */
const WINOGRAD_TOL = { atol: 3e-2, rtol: 2e-2 };

describe('conv2d hwc4 routes', () => {
  it.for([
    // 1×1: stride 1, hw % 4 == 0 and (hw / 4)·CO4 >= 2048. Unrolled K under 8192
    // threads; the f16 run is the longest of 8, 4, 2, 1 dividing CI4.
    { name: '1x1, 8 pixels a thread, K unrolled by 4', cIn: 32, cOut: 32, h: 32, w: 32 },
    { name: '1x1, 4 pixels a thread (hw % 8 != 0), K by 2', cIn: 24, cOut: 40, h: 30, w: 30 },
    { name: '1x1, 8192 threads: K not unrolled, odd CI4', cIn: 20, cOut: 64, h: 64, w: 64 },
    { name: '1x1, padded channels', cIn: 6, cOut: 10, h: 56, w: 56, bias: true, act: 'silu' },
    // 1×1s (split K): hw % 4 != 0
    { name: '1x1s, 7×7', cIn: 64, cOut: 48, h: 7, w: 7, bias: true },
    // direct: not tiled. CO4 <= 8 orders threads block-first; columns per
    // thread are the most of 4, 2, 1 that keep 16384 threads.
    {
      name: 'direct, narrow, 1 column, stride 2',
      cIn: 8,
      cOut: 16,
      h: 32,
      w: 32,
      k: 3,
      stride: 2,
      padding: 1,
    },
    {
      name: 'direct, narrow, 4 columns',
      cIn: 8,
      cOut: 16,
      h: 128,
      w: 128,
      k: 3,
      padding: 1,
      act: 'relu',
      bias: true,
    },
    { name: 'direct, wide, 2 columns', cIn: 4, cOut: 40, h: 64, w: 64, k: 3, padding: 1 },
    {
      name: 'direct, wide, 1 column, stride 2',
      cIn: 32,
      cOut: 64,
      h: 28,
      w: 28,
      k: 3,
      stride: 2,
      padding: 1,
    },
    { name: 'direct, 1×3 kernel', cIn: 12, cOut: 8, h: 9, w: 11, k: [1, 3] },
    // tiled: >= 24 workgroups of 64 pixels, and CO >= 64, or CO >= 32 on >= 6400 pixels
    {
      name: 'tiled, 16 blocks a workgroup',
      cIn: 16,
      cOut: 64,
      h: 48,
      w: 48,
      k: 3,
      padding: 1,
      bias: true,
    },
    { name: 'tiled, 8 blocks a workgroup', cIn: 8, cOut: 32, h: 80, w: 80, k: 3, padding: 1 },
    // winograd: 3×3, stride 1, padding 1, opted in on the module
    {
      name: 'winograd, whole tiles',
      cIn: 16,
      cOut: 16,
      h: 16,
      w: 16,
      k: 3,
      padding: 1,
      winograd: true,
    },
    {
      name: 'winograd, partial tiles, padded channels',
      cIn: 6,
      cOut: 10,
      h: 14,
      w: 13,
      k: 3,
      padding: 1,
      winograd: true,
    },
    {
      name: 'winograd, a map smaller than a tile',
      cIn: 8,
      cOut: 8,
      h: 3,
      w: 2,
      k: 3,
      padding: 1,
      winograd: true,
    },
    {
      name: 'winograd, bias and relu',
      cIn: 32,
      cOut: 48,
      h: 64,
      w: 64,
      k: 3,
      padding: 1,
      act: 'relu',
      bias: true,
      winograd: true,
    },
  ] as const)('$name: $cIn>$cOut @ $h×$w', async (c, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = randomMap(c.cIn, c.h, c.w, 3);
    const { conv, ref } = await makeConv(c, x);
    const got = await readChw(conv.forward(hwc4(x)));
    expectClose(got, ref.data, c.winograd ? WINOGRAD_TOL : DIRECT_TOL);
  });

  // The residual add rides the conv's epilogue: add(x, conv(x)).
  it.for([
    { name: '1x1', cIn: 32, cOut: 32, h: 32, w: 32, bias: true },
    { name: 'direct', cIn: 8, cOut: 8, h: 32, w: 32, k: 3, padding: 1 },
    {
      name: 'winograd',
      cIn: 12,
      cOut: 12,
      h: 18,
      w: 18,
      k: 3,
      padding: 1,
      act: 'relu',
      winograd: true,
    },
  ] as const)('$name with a fused residual add', async (c, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = randomMap(c.cIn, c.h, c.w, 5);
    const { conv, ref } = await makeConv(c, x);
    const xh = hwc4(x);
    const got = await readChw(add(xh, conv.forward(xh)));
    expectClose(
      got,
      ref.data.map((v, i) => v + x.data[i]!),
      c.winograd ? WINOGRAD_TOL : DIRECT_TOL,
    );
  });

  // A conv feeding a concat writes straight into the concat's buffer, at a
  // channel-block offset.
  it.for([
    // 1x1: (hw / 4)·CO4 = 2048
    { name: '1x1', cIn: 8, cOut: 32, h: 32, w: 32 },
    { name: 'winograd', cIn: 8, cOut: 8, h: 10, w: 10, k: 3, padding: 1, winograd: true },
  ] as const)('$name into the second half of a concat', async (c, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = randomMap(c.cIn, c.h, c.w, 9);
    const { conv, ref } = await makeConv(c, x);
    const xh = hwc4(x);
    const got = await readChw(cat([xh, conv.forward(xh)], 0));
    const want = new Float32Array(x.data.length + ref.data.length);
    want.set(x.data);
    want.set(ref.data, x.data.length);
    expectClose(got, want, c.winograd ? WINOGRAD_TOL : DIRECT_TOL);
  });
});

/** Value at (c, y, x), zero outside the map. */
const at = (m: Map3, c: number, y: number, x: number) =>
  y < 0 || y >= m.h || x < 0 || x >= m.w ? 0 : m.data[(c * m.h + y) * m.w + x]!;

function mapOf(c: number, h: number, w: number, f: (c: number, y: number, x: number) => number) {
  const data = new Float32Array(c * h * w);
  for (let k = 0; k < c; k++)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[(k * h + y) * w + x] = f(k, y, x);
  return data;
}

/** torch's bilinear source coordinate, clamped to the map. */
function bilinearRef(m: Map3, outH: number, outW: number, alignCorners: boolean) {
  const src = (o: number, n: number, out: number) =>
    alignCorners
      ? out > 1
        ? (o * (n - 1)) / (out - 1)
        : 0
      : Math.max(((o + 0.5) * n) / out - 0.5, 0);
  return mapOf(m.c, outH, outW, (c, oy, ox) => {
    const sy = src(oy, m.h, outH);
    const sx = src(ox, m.w, outW);
    const y0 = Math.min(Math.floor(sy), m.h - 1);
    const x0 = Math.min(Math.floor(sx), m.w - 1);
    const y1 = Math.min(y0 + 1, m.h - 1);
    const x1 = Math.min(x0 + 1, m.w - 1);
    const fy = sy - y0;
    const fx = sx - x0;
    const top = at(m, c, y0, x0) * (1 - fx) + at(m, c, y0, x1) * fx;
    const bot = at(m, c, y1, x0) * (1 - fx) + at(m, c, y1, x1) * fx;
    return top * (1 - fy) + bot * fy;
  });
}

// Every move kernel takes the channel block as its fastest thread index; 6
// channels leave a padded block.
describe('hwc4 move kernels', () => {
  const m = randomMap(6, 9, 7, 11);
  it.for([
    {
      name: 'resizeBilinear2d up, align corners',
      run: (x: Value) => resizeBilinear2d(x, { outH: 17, outW: 13 }),
      want: () => bilinearRef(m, 17, 13, true),
    },
    {
      name: 'resizeBilinear2d up, half pixel',
      run: (x: Value) => resizeBilinear2d(x, { outH: 18, outW: 14, mode: 'halfPixel' }),
      want: () => bilinearRef(m, 18, 14, false),
    },
    {
      name: 'resizeBilinear2d down, half pixel',
      run: (x: Value) => resizeBilinear2d(x, { outH: 4, outW: 3, mode: 'halfPixel' }),
      want: () => bilinearRef(m, 4, 3, false),
    },
    {
      name: 'upsample2d ×2',
      run: (x: Value) => upsample2d(x, { scale: 2 }),
      want: () => mapOf(6, 18, 14, (c, y, x) => at(m, c, y >> 1, x >> 1)),
    },
    {
      name: 'avgPool2d 2×2, stride 2',
      run: (x: Value) => avgPool2d(x, { kernelSize: 2 }),
      want: () =>
        mapOf(6, 4, 3, (c, y, x) => {
          let s = 0;
          for (let dy = 0; dy < 2; dy++)
            for (let dx = 0; dx < 2; dx++) s += at(m, c, 2 * y + dy, 2 * x + dx);
          return s / 4;
        }),
    },
    {
      name: 'maxPool2d 3×3, stride 2, padding 1',
      run: (x: Value) => maxPool2d(x, { kernelSize: 3, stride: 2, padding: 1 }),
      want: () =>
        mapOf(6, 5, 4, (c, y, x) => {
          let best = -Infinity;
          for (let dy = 0; dy < 3; dy++) {
            for (let dx = 0; dx < 3; dx++) {
              const iy = 2 * y + dy - 1;
              const ix = 2 * x + dx - 1;
              if (iy >= 0 && iy < m.h && ix >= 0 && ix < m.w)
                best = Math.max(best, at(m, c, iy, ix));
            }
          }
          return best;
        }),
    },
    {
      name: 'pad2d 1 row, 3 columns',
      run: (x: Value) => pad2d(x, { padH: 1, padW: 3 }),
      want: () => mapOf(6, 11, 13, (c, y, x) => at(m, c, y - 1, x - 3)),
    },
  ])('$name', async ({ run, want }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    expectClose(await readChw(run(hwc4(m))), want(), { atol: 2e-3, rtol: 2e-3 });
  });
});
