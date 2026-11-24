/** Depth Pro 2's field-of-view estimator: a DINOv3 ViT-S/16 trunk, and a
 *  head where one query token attends over the trunk's pooled patch,
 *  class and register tokens. It gives the horizontal and vertical field of
 *  view of a photo, from which the focal length follows.
 *
 *  Field names follow the checkpoint: `trunk` and `head`. The geometry on
 *  the CPU (resize, crop, angles) is in fovInput and fovDegrees. */

import {
  add,
  cat,
  derive,
  matmul,
  nn,
  slice,
  softmax,
  transpose,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { requireTensor } from '../errors.ts';
import { INPUT_MULTIPLE, PREFIX_TOKENS, type FovConfig } from './config.ts';
import { Dinov3, type Rope } from './vit.ts';

/** The short side the estimator reads at most, in pixels. */
const MAX_SHORT_SIDE = 512;
/** Frames wider than this (long over short side) are center-cropped to
 *  it. */
const MAX_ASPECT = 1.7778;
const MIN_FOV_DEG = 1;
const MAX_FOV_DEG = 179;
/** Patches averaged per side into one head token. */
const POOL = 2;
/** The head blocks' LayerNorm eps, timm's default. */
const HEAD_BLOCK_EPS = 1e-6;
/** The tap norms' and the final norm's eps, torch's default. */
const HEAD_EPS = 1e-5;

/** Python's round(): halves go to the even neighbor. */
function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** What the estimator reads of a `height` × `width` frame: the size it is
 *  resized to, then the centered crop of that it keeps. Never upsamples
 *  past the frame. */
export function fovInput(height: number, width: number) {
  /** Every side holds at least one INPUT_MULTIPLE. */
  const atLeastOne = (v: number) => Math.max(INPUT_MULTIPLE, v);
  const minSide = Math.min(height, width);
  const short = atLeastOne(
    Math.min(MAX_SHORT_SIDE, Math.floor(minSide / INPUT_MULTIPLE) * INPUT_MULTIPLE),
  );
  const scale = short / minSide;
  const resize = {
    height: atLeastOne(roundHalfEven((height * scale) / INPUT_MULTIPLE) * INPUT_MULTIPLE),
    width: atLeastOne(roundHalfEven((width * scale) / INPUT_MULTIPLE) * INPUT_MULTIPLE),
  };
  const lo = Math.min(resize.height, resize.width);
  const hi = Math.max(resize.height, resize.width);
  if (hi <= MAX_ASPECT * lo) return { resize, crop: resize };
  const keep = atLeastOne(Math.floor((MAX_ASPECT * lo) / INPUT_MULTIPLE) * INPUT_MULTIPLE);
  const crop =
    resize.width > resize.height
      ? { height: resize.height, width: keep }
      : { height: keep, width: resize.width };
  return { resize, crop };
}

/** The head's [hfov, vfov] logits to angles in degrees of the whole
 *  `height` × `width` frame, undoing fovInput's crop. */
export function fovDegrees(logits: Float32Array, height: number, width: number): [number, number] {
  const deg = (v: number) => MIN_FOV_DEG + (MAX_FOV_DEG - MIN_FOV_DEG) / (1 + Math.exp(-v));
  const angles: [number, number] = [deg(logits[0]!), deg(logits[1]!)];
  const { resize, crop } = fovInput(height, width);
  if (crop.height === resize.height && crop.width === resize.width) return angles;
  // The crop's sides in source pixels, at the same focal length.
  const cropW = (crop.width * width) / resize.width;
  const cropH = (crop.height * height) / resize.height;
  const rad = Math.PI / 180;
  const fx = cropW / 2 / Math.tan((angles[0] * rad) / 2);
  const fy = cropH / 2 / Math.tan((angles[1] * rad) / 2);
  return [(2 * Math.atan(width / 2 / fx)) / rad, (2 * Math.atan(height / 2 / fy)) / rad];
}

/** The focal length in pixels from a frame's field of view, read along its
 *  shorter side. */
export function focalLengthPx([hfov, vfov]: [number, number], height: number, width: number) {
  const rad = Math.PI / 180;
  return width < height
    ? (0.5 * width) / Math.tan((hfov * rad) / 2)
    : (0.5 * height) / Math.tan((vfov * rad) / 2);
}

/** Pooling weights [N, rows · cols]: each head token averages its POOL ×
 *  POOL square of patches, row-major. `rows` and `cols` are even, as
 *  fovInput's sizes are. */
export function poolWeights(rows: number, cols: number): Float32Array {
  const pooledCols = cols / POOL;
  const n = pooledCols * (rows / POOL);
  const out = new Float32Array(n * rows * cols);
  for (let k = 0; k < rows * cols; k++) {
    const token =
      Math.floor((k % cols) / POOL) + pooledCols * Math.floor(Math.floor(k / cols) / POOL);
    out[token * rows * cols + k] = 1 / (POOL * POOL);
  }
  return out;
}

/** The head's position code [N, dim] of a pooled grid: sin and cos of the
 *  row, then of the column, at 2^0 … 2^7 cycles over the longer side. */
export function sincos2d(rows: number, cols: number, dim: number): Float32Array {
  const freqs = dim / 4;
  const longer = Math.max(rows, cols);
  const out = new Float32Array(rows * cols * dim);
  for (let k = 0; k < rows * cols; k++) {
    const coords = [
      (Math.floor(k / cols) + 0.5 - rows / 2) / longer,
      ((k % cols) + 0.5 - cols / 2) / longer,
    ];
    for (const [axis, c] of coords.entries()) {
      for (let f = 0; f < freqs; f++) {
        const angle = c * Math.PI * 2 ** ((7 * f) / (freqs - 1));
        out[k * dim + axis * 2 * freqs + f] = Math.sin(angle);
        out[k * dim + axis * 2 * freqs + freqs + f] = Math.cos(angle);
      }
    }
  }
  return out;
}

class HeadAttention extends nn.Module {
  /** q rows and bias scaled by 1/√headDim. */
  readonly qkv: nn.Linear;
  readonly proj: nn.Linear;
  constructor(dim: number) {
    super();
    this.qkv = new nn.Linear(dim, 3 * dim);
    this.proj = new nn.Linear(dim, dim);
  }
}

class HeadMlp extends nn.Module {
  readonly fc1: nn.Linear;
  readonly fc2: nn.Linear;
  constructor(dim: number) {
    super();
    this.fc1 = new nn.Linear(dim, 4 * dim, { activation: 'gelu' });
    this.fc2 = new nn.Linear(4 * dim, dim);
  }
}

/** A timm ViT block. Its head width (62) is not one sdpa runs, and the
 *  head reads a few hundred tokens, so attention runs on matmuls. */
class HeadBlock extends nn.Module {
  readonly norm1: nn.LayerNorm;
  readonly attn: HeadAttention;
  readonly norm2: nn.LayerNorm;
  readonly mlp: HeadMlp;
  constructor(
    private readonly dim: number,
    private readonly heads: number,
  ) {
    super();
    this.norm1 = new nn.LayerNorm(dim, HEAD_BLOCK_EPS, { bias: true });
    this.attn = new HeadAttention(dim);
    this.norm2 = new nn.LayerNorm(dim, HEAD_BLOCK_EPS, { bias: true });
    this.mlp = new HeadMlp(dim);
  }
  forward(x: Value): Value {
    const { dim, heads } = this;
    const d = dim / heads;
    const qkv = this.attn.qkv.forward(this.norm1.forward(x));
    const col = (start: number) => slice(qkv, 1, start, start + d);
    const outs = Array.from({ length: heads }, (_, h) =>
      matmul(softmax(matmul(col(h * d), transpose(col(dim + h * d)))), col(2 * dim + h * d)),
    );
    x = this.attn.proj.forward(cat(outs, 1), { addend: x });
    return this.mlp.fc2.forward(this.mlp.fc1.forward(this.norm2.forward(x)), { addend: x });
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const scale = 1 / Math.sqrt(this.dim / this.heads);
    const scaleQ = (name: string, rowElems: number) => {
      const key = `${prefix}.attn.qkv.${name}`;
      const t = requireTensor(sd, key, 'Depth Pro 2 field-of-view head');
      sd.tensors.set(
        key,
        derive(t.shape, [t], ([data]) =>
          data.map((v, i) => (i < this.dim * rowElems ? v * scale : v)),
        ),
      );
    };
    scaleQ('weight', this.dim);
    scaleQ('bias', 1);
  }
}

class FovHead extends nn.Module<[readonly Value[], FovInputs], Value> {
  readonly tap_norms: nn.ModuleList<nn.LayerNorm>;
  readonly proj: nn.Linear;
  readonly query: nn.Parameter;
  readonly prefix_embed: nn.Parameter;
  readonly blocks: nn.ModuleList<HeadBlock>;
  readonly norm: nn.LayerNorm;
  readonly out: nn.Linear;

  constructor(hidden: number, { dim, heads, depth, taps }: FovConfig) {
    super();
    this.tap_norms = new nn.ModuleList(
      taps.map(() => new nn.LayerNorm(hidden, HEAD_EPS, { bias: true })),
    );
    this.proj = new nn.Linear(taps.length * hidden, dim);
    this.query = new nn.Parameter({ elems: dim, dtype: 'f32', dims: [1, dim] });
    this.prefix_embed = new nn.Parameter({
      elems: PREFIX_TOKENS * dim,
      dtype: 'f32',
      dims: [PREFIX_TOKENS, dim],
    });
    this.blocks = new nn.ModuleList(Array.from({ length: depth }, () => new HeadBlock(dim, heads)));
    this.norm = new nn.LayerNorm(dim, HEAD_EPS, { bias: true });
    this.out = new nn.Linear(dim, 2);
  }

  /** The trunk's tapped tokens to the [hfov, vfov] logits [1, 2]. */
  override forward(taps: readonly Value[], inp: FovInputs): Value {
    const normed = taps.map((t, i) => this.tap_norms.items[i]!.forward(t));
    const x = this.proj.forward(cat(normed, 1));
    const rows = x.shape.dims![0]!;
    const prefix = add(slice(x, 0, 0, PREFIX_TOKENS), this.prefix_embed.value);
    const pooled = add(matmul(inp.pool, slice(x, 0, PREFIX_TOKENS, rows)), inp.position);
    let h = cat([this.query.value, prefix, pooled], 0);
    for (const block of this.blocks.items) h = block.forward(h);
    return this.out.forward(this.norm.forward(slice(h, 0, 0, 1)));
  }
}

/** One frame's estimator inputs, at fovInput's crop size. */
export interface FovInputs {
  /** Patches [P, 3 · 16²] from patchify. */
  readonly patches: Value;
  readonly rope: Rope;
  /** [N, P] from poolWeights. */
  readonly pool: Value;
  /** [N, dim] from sincos2d. */
  readonly position: Value;
}

export class FovModel extends nn.Module<[FovInputs], Value> {
  readonly trunk: Dinov3;
  readonly head: FovHead;
  constructor(readonly cfg: FovConfig) {
    super();
    this.trunk = new Dinov3(cfg.vit);
    this.head = new FovHead(cfg.vit.hidden, cfg);
  }
  /** The [hfov, vfov] logits [1, 2] of one frame. */
  override forward(inp: FovInputs): Value {
    const taps = this.trunk.encode(inp.patches, inp.rope, this.cfg.taps, this.cfg.crossFrame);
    return this.head.forward(taps, inp);
  }
}
