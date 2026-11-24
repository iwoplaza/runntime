/** Depth Pro 2 geometry: the DINOv3 ViT under each size, the blocks the
 *  pyramid taps, and the field-of-view estimator's. */

/** A DINOv3 ViT/16 with 4 register tokens, as timm builds it. */
export interface Dinov3Config {
  readonly hidden: number;
  readonly heads: number;
  readonly blocks: number;
}

export interface DepthPro2Config {
  readonly vit: Dinov3Config;
  /** Zero-based blocks whose patch tokens feed the 4 pyramid levels. */
  readonly taps: readonly [number, number, number, number];
  /** Blocks that attend across frames: one frame keeps its input class and
   *  register tokens through them. */
  readonly crossFrame: readonly number[];
  /** Channels of the pyramid levels, 1/4 resolution first. */
  readonly dimsReassemble: readonly [number, number, number, number];
  readonly dimDecoder: number;
}

export interface FovConfig {
  readonly vit: Dinov3Config;
  /** Blocks whose tokens the head reads, side by side. */
  readonly taps: readonly number[];
  /** Blocks that train across frames: one frame keeps its input class and
   *  register tokens through them. */
  readonly crossFrame: readonly number[];
  readonly dim: number;
  readonly heads: number;
  readonly depth: number;
}

/** Pixels per patch side. */
export const PATCH = 16;
/** The class token and 4 register tokens ahead of the patches. */
export const PREFIX_TOKENS = 5;
/** Image sides the model reads are padded to a multiple of this. */
export const INPUT_MULTIPLE = 32;
export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

const VIT_S = { hidden: 384, heads: 6, blocks: 12 } as const satisfies Dinov3Config;
const VIT_B = { hidden: 768, heads: 12, blocks: 12 } as const satisfies Dinov3Config;

export const DEPTH_PRO_2_S = Object.freeze({
  vit: VIT_S,
  taps: [2, 5, 8, 11],
  crossFrame: [5, 7, 9, 11],
  dimsReassemble: [48, 96, 192, 384],
  dimDecoder: 64,
} as const) satisfies DepthPro2Config;

export const DEPTH_PRO_2_B = Object.freeze({
  vit: VIT_B,
  taps: [2, 5, 8, 11],
  crossFrame: [5, 7, 9, 11],
  dimsReassemble: [96, 192, 384, 768],
  dimDecoder: 128,
} as const) satisfies DepthPro2Config;

export const DEPTH_PRO_2_FOV = Object.freeze({
  vit: VIT_S,
  taps: [2, 5, 8, 11],
  crossFrame: [5, 7, 9, 11],
  dim: 248,
  heads: 4,
  depth: 2,
} as const) satisfies FovConfig;
