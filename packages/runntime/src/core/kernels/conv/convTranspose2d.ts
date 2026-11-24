import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type FloatBuffer,
  type KernelHandle,
  flatIndex,
  makeHandle,
  type F16Buffer,
  type F32Buffer,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import type { Elem } from '../elem.ts';

/** Runtime-variable dims: input spatial size only. */
const Dims = d.struct({ h: d.u32, w: d.u32 });

export interface ConvTranspose2dCfg {
  cIn: number;
  cOut: number;
  k: number;
  hasBias: number; // 0 | 1
}

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    w: { storage: d.arrayOf(d.u32), access: 'readonly' },
    bias: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });

const layouts = new Map<string, ReturnType<typeof makeLayout>>();
function layoutFor(elem: Elem): ReturnType<typeof makeLayout> {
  let l = layouts.get(elem.key);
  if (!l) {
    l = makeLayout(elem);
    layouts.set(elem.key, l);
  }
  return l;
}

function makeConvTranspose2dKernel(cfg: ConvTranspose2dCfg, elem: Elem) {
  const { cIn, cOut, k, hasBias } = cfg;
  const paired = cIn % 2 === 0;
  const layout = layoutFor(elem);
  const storeScalar = elem.scalar;
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = layout.$.dims;
    const hOut = dims.h * k;
    const wOut = dims.w * k;
    const idx = flatIndex(gid);
    if (idx >= cOut * hOut * wOut) {
      return;
    }
    const wo = idx % wOut;
    const ho = std.intdiv(idx, wOut) % hOut;
    const co = std.intdiv(idx, wOut * hOut);
    const hi = std.intdiv(ho, k);
    const wi = std.intdiv(wo, k);
    const di = ho % k;
    const dj = wo % k;
    const rowBase = (co * k * k + di * k + dj) * cIn;
    const xPix = hi * dims.w + wi;
    const plane = dims.h * dims.w;
    let acc = d.f32(0);
    if (paired) {
      const wordBase = std.intdiv(rowBase, 2); // rowBase is even when cIn is
      for (const c2 of std.range(cIn / 2)) {
        const pair = std.unpack2x16float(layout.$.w[wordBase + c2]!);
        const xb = 2 * c2 * plane + xPix;
        acc += pair.x * d.f32(layout.$.x[xb]!) + pair.y * d.f32(layout.$.x[xb + plane]!);
      }
    } else {
      for (const ci of std.range(cIn)) {
        const flat = rowBase + ci;
        const pair = std.unpack2x16float(layout.$.w[std.intdiv(flat, 2)]!);
        const wv = std.select(pair.x, pair.y, flat % 2 === 1);
        acc += wv * d.f32(layout.$.x[ci * plane + xPix]!);
      }
    }
    if (hasBias === 1) {
      acc += d.f32(layout.$.bias[co]!);
    }
    layout.$.out[idx] = storeScalar(acc);
  });
}

export function createConvTranspose2dPipeline(root: TgpuRoot, cfg: ConvTranspose2dCfg, elem: Elem) {
  return root.createComputePipeline({ compute: makeConvTranspose2dKernel(cfg, elem) });
}

export function convTranspose2dHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createConvTranspose2dPipeline>,
  cfg: ConvTranspose2dCfg,
  args: { h: number; w: number },
  buffers: { x: FloatBuffer; w: F16Buffer; bias: FloatBuffer; out: FloatBuffer },
  elem: Elem,
): KernelHandle {
  const { h, w } = args;
  const dims = cachedUniform(root, Dims, { h, w });
  const bindGroup = cachedBindGroup(root, layoutFor(elem), { ...buffers, dims });
  const total = cfg.cOut * h * cfg.k * w * cfg.k;
  return makeHandle(pipeline, 'convTranspose2d', bindGroup, Math.ceil(total / WORKGROUP_SIZE));
}

// ---------------------------------------------------------------- hwc4

/** Runtime-variable dims of the hwc4 route: input spatial size only. */
const Hwc4Dims = d.struct({ h: d.u32, w: d.u32 });

export interface ConvTranspose2dHwc4Cfg {
  cIn: number; // a multiple of 4
  cOut: number;
  k: number;
  hasBias: number; // 0 | 1
}

const hwc4Layout = tgpu.bindGroupLayout({
  x: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  w: { storage: d.arrayOf(d.u32), access: 'readonly' },
  bias: { storage: d.arrayOf(d.f32), access: 'readonly' },
  out: { storage: d.arrayOf(d.vec4h), access: 'mutable' },
  dims: { uniform: Hwc4Dims },
});

/** One thread per output pixel and block of 4 output channels. It reads the
 *  input pixel's channel blocks once and the 4 weight rows of its tap, two
 *  f16 per u32, and sums in f32: f16 sums lose bits past a few hundred
 *  channels. Lanes past C_out stay 0, as hwc4 padding has to. */
function makeConvTranspose2dHwc4Kernel(cfg: ConvTranspose2dHwc4Cfg) {
  const { cIn, cOut, k, hasBias } = cfg;
  const ci4 = cIn / 4;
  const co4 = Math.ceil(cOut / 4);
  const rowWords = cIn / 2;
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = hwc4Layout.$.dims;
    const wOut = dims.w * k;
    const idx = flatIndex(gid);
    if (idx >= co4 * dims.h * k * wOut) {
      return;
    }
    const oB = idx % co4;
    const ox = std.intdiv(idx, co4) % wOut;
    const oy = std.intdiv(idx, co4 * wOut);
    const tap = (oy % k) * k + (ox % k);
    const xBase = (std.intdiv(oy, k) * dims.w + std.intdiv(ox, k)) * ci4;
    const acc = d.arrayOf(d.f32, 4)();
    for (const l of std.range(4)) {
      const co = oB * 4 + l;
      if (co < cOut) {
        const rowBase = (co * k * k + tap) * rowWords;
        let sum = d.f32(0);
        for (const iB of std.range(ci4)) {
          const xv = d.vec4f(hwc4Layout.$.x[xBase + iB]!);
          const lo = std.unpack2x16float(hwc4Layout.$.w[rowBase + 2 * iB]!);
          const hi = std.unpack2x16float(hwc4Layout.$.w[rowBase + 2 * iB + 1]!);
          sum += std.dot(xv, d.vec4f(lo.x, lo.y, hi.x, hi.y));
        }
        if (hasBias === 1) {
          sum += hwc4Layout.$.bias[co]!;
        }
        acc[l] = sum;
      }
    }
    hwc4Layout.$.out[idx] = d.vec4h(d.vec4f(acc[0]!, acc[1]!, acc[2]!, acc[3]!));
  });
}

export function createConvTranspose2dHwc4Pipeline(root: TgpuRoot, cfg: ConvTranspose2dHwc4Cfg) {
  return root.createComputePipeline({ compute: makeConvTranspose2dHwc4Kernel(cfg) });
}

export function convTranspose2dHwc4Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createConvTranspose2dHwc4Pipeline>,
  cfg: ConvTranspose2dHwc4Cfg,
  args: { h: number; w: number },
  buffers: { x: F16Buffer; w: F16Buffer; bias: F32Buffer; out: F16Buffer },
): KernelHandle {
  const { h, w } = args;
  const dims = cachedUniform(root, Hwc4Dims, { h, w });
  const bindGroup = cachedBindGroup(root, hwc4Layout, { ...buffers, dims });
  const total = Math.ceil(cfg.cOut / 4) * h * cfg.k * w * cfg.k;
  return makeHandle(
    pipeline,
    'convTranspose2d_hwc4',
    bindGroup,
    Math.ceil(total / WORKGROUP_SIZE),
    `convTranspose2d_hwc4 ${cfg.cIn}→${cfg.cOut}@${h}x${w}`,
  );
}
