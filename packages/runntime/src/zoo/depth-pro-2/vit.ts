/** The DINOv3 ViT/16 under Depth Pro 2 and its field-of-view estimator, as
 *  timm's `vit_*_patch16_dinov3` runs it on one frame: a class token and 4
 *  register tokens ahead of the patches, a 2D rope on the patches' q and k,
 *  and LayerScale on both residual branches.
 *
 *  Field names follow the checkpoint. At load the patch conv becomes a
 *  Linear over CPU-cut patches, q is scaled by 1/√headDim and its and k's
 *  rows are interleaved for the rope kernel, and each LayerScale folds into
 *  the Linear before it. */

import {
  cat,
  derive,
  nn,
  permuteRows,
  reshapeWeight,
  rope,
  sdpa,
  slice,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { requireTensor } from '../errors.ts';
import { ropeOrder } from '../shared/rope.ts';
import { PATCH, PREFIX_TOKENS, type Dinov3Config } from './config.ts';

/** timm's DINOv3 rope temperature. */
const ROPE_TEMPERATURE = 100;
/** The blocks' LayerNorm eps. */
const NORM_EPS = 1e-5;

/** cos and sin [PREFIX_TOKENS + rows · cols, headDim] for a grid of patches
 *  in row-major order, in the kernel's interleaved-pair layout. The prefix
 *  rows turn by 0. Pair p turns with the patch's row for p < headDim / 4,
 *  with its column after, both centered and scaled by the longer side to
 *  [-1, 1], at periods 100^(i / (headDim / 4)). */
export function dinov3RopeTables(rows: number, cols: number, headDim: number) {
  const freqs = headDim / 4;
  const n = PREFIX_TOKENS + rows * cols;
  const cos = new Float32Array(n * headDim).fill(1, 0, PREFIX_TOKENS * headDim);
  const sin = new Float32Array(n * headDim);
  const longer = Math.max(rows, cols);
  const coord = (i: number) => (2 * (i + 0.5)) / longer - 1;
  for (let k = 0; k < rows * cols; k++) {
    const y = coord(Math.floor(k / cols));
    const x = coord(k % cols);
    const at = (PREFIX_TOKENS + k) * headDim;
    for (let p = 0; p < 2 * freqs; p++) {
      const pos = p < freqs ? y : x;
      const angle = (2 * Math.PI * pos) / ROPE_TEMPERATURE ** ((p % freqs) / freqs);
      cos[at + 2 * p] = cos[at + 2 * p + 1] = Math.cos(angle);
      sin[at + 2 * p] = sin[at + 2 * p + 1] = Math.sin(angle);
    }
  }
  return { cos, sin };
}

/** The image cut into patches [rows · cols, 3 · PATCH²], row-major, each in
 *  (channel, row, column) order like the patch conv's weight. `chw` is
 *  [3, rows · PATCH, cols · PATCH]. */
export function patchify(chw: Float32Array, rows: number, cols: number): Float32Array {
  const width = cols * PATCH;
  const plane = rows * PATCH * width;
  const out = new Float32Array(chw.length);
  let o = 0;
  for (let py = 0; py < rows; py++) {
    for (let px = 0; px < cols; px++) {
      for (let c = 0; c < 3; c++) {
        for (let y = py * PATCH; y < (py + 1) * PATCH; y++) {
          const row = c * plane + y * width + px * PATCH;
          out.set(chw.subarray(row, row + PATCH), o);
          o += PATCH;
        }
      }
    }
  }
  return out;
}

/** One frame's rope rows, from dinov3RopeTables. */
export interface Rope {
  readonly cos: Value;
  readonly sin: Value;
}

class PatchEmbed extends nn.Module {
  /** The checkpoint's PATCH×PATCH stride-PATCH conv, as a Linear over
   *  patchify's rows. */
  readonly proj: nn.Linear;
  constructor(private readonly hidden: number) {
    super();
    this.proj = new nn.Linear(3 * PATCH * PATCH, hidden);
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = `${prefix}.proj.weight`;
    const weight = requireTensor(sd, name, 'Depth Pro 2 patch embed');
    sd.tensors.set(name, reshapeWeight(weight, [this.hidden, 3 * PATCH * PATCH]));
  }
}

class Attention extends nn.Module {
  /** q rows scaled by 1/√headDim, q and k rows interleaved per head. */
  readonly qkv: nn.Linear;
  readonly proj: nn.Linear;
  constructor(hidden: number) {
    super();
    this.qkv = new nn.Linear(hidden, 3 * hidden, { bias: false });
    this.proj = new nn.Linear(hidden, hidden);
  }
}

class Mlp extends nn.Module {
  readonly fc1: nn.Linear;
  readonly fc2: nn.Linear;
  constructor(hidden: number) {
    super();
    this.fc1 = new nn.Linear(hidden, 4 * hidden, { activation: 'gelu' });
    this.fc2 = new nn.Linear(4 * hidden, hidden);
  }
}

class Block extends nn.Module<[Value, Rope], Value> {
  readonly norm1: nn.LayerNorm;
  readonly attn: Attention;
  readonly norm2: nn.LayerNorm;
  readonly mlp: Mlp;
  constructor(private readonly cfg: Dinov3Config) {
    super();
    this.norm1 = new nn.LayerNorm(cfg.hidden, NORM_EPS, { bias: true });
    this.attn = new Attention(cfg.hidden);
    this.norm2 = new nn.LayerNorm(cfg.hidden, NORM_EPS, { bias: true });
    this.mlp = new Mlp(cfg.hidden);
  }

  override forward(x: Value, { cos, sin }: Rope): Value {
    const { hidden: w, heads } = this.cfg;
    const headDim = w / heads;
    const qkv = this.attn.qkv.forward(this.norm1.forward(x));
    const qk = rope(qkv, cos, sin, { headDim, width: 2 * w });
    const a = sdpa(slice(qk, 1, 0, w), slice(qk, 1, w, 2 * w), slice(qkv, 1, 2 * w, 3 * w), {
      qHeads: heads,
      kvHeads: heads,
      headDim,
      windowLeft: Infinity,
      windowRight: Infinity,
    });
    x = this.attn.proj.forward(a, { addend: x });
    const h = this.mlp.fc1.forward(this.norm2.forward(x));
    return this.mlp.fc2.forward(h, { addend: x });
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const { hidden: w, heads } = this.cfg;
    const headDim = w / heads;
    const at = (name: string) => requireTensor(sd, `${prefix}.${name}`, 'Depth Pro 2 block');

    // Rows of q and k interleaved per head; v keeps its order.
    const order = [...ropeOrder(2 * w, headDim), ...Array.from({ length: w }, (_, i) => 2 * w + i)];
    const qScale = 1 / Math.sqrt(headDim);
    const qkv = permuteRows(at('attn.qkv.weight'), order);
    sd.tensors.set(
      `${prefix}.attn.qkv.weight`,
      derive([3 * w, w], [qkv], ([data]) => data.map((v, i) => (i < w * w ? v * qScale : v))),
    );

    // y + γ ⊙ (W·x + b) is y + (γ ⊙ W)·x + γ ⊙ b.
    for (const [gamma, linear] of [
      ['gamma_1', 'attn.proj'],
      ['gamma_2', 'mlp.fc2'],
    ] as const) {
      const g = at(gamma);
      const weight = at(`${linear}.weight`);
      const rowElems = weight.shape[1]!;
      sd.tensors.set(
        `${prefix}.${linear}.weight`,
        derive(weight.shape, [weight, g], ([data, gd]) =>
          data.map((v, i) => v * gd[Math.floor(i / rowElems)]!),
        ),
      );
      sd.tensors.set(
        `${prefix}.${linear}.bias`,
        derive([w], [at(`${linear}.bias`), g], ([b, gd]) => b.map((v, i) => v * gd[i]!)),
      );
      sd.tensors.delete(`${prefix}.${gamma}`);
    }
  }
}

/** A DINOv3 ViT/16 trunk: patch embed, class and register tokens, blocks. */
export class Dinov3 extends nn.Module {
  readonly patch_embed: PatchEmbed;
  readonly cls_token: nn.Parameter;
  readonly reg_token: nn.Parameter;
  readonly blocks: nn.ModuleList<Block>;

  constructor(readonly vit: Dinov3Config) {
    super();
    const { hidden } = vit;
    this.patch_embed = new PatchEmbed(hidden);
    this.cls_token = new nn.Parameter({ elems: hidden, dtype: 'f32', dims: [1, hidden] });
    this.reg_token = new nn.Parameter({
      elems: (PREFIX_TOKENS - 1) * hidden,
      dtype: 'f32',
      dims: [PREFIX_TOKENS - 1, hidden],
    });
    this.blocks = new nn.ModuleList(Array.from({ length: vit.blocks }, () => new Block(vit)));
  }

  /** Runs the blocks over one frame's patches [P, 3 · PATCH²] and returns
   *  the tokens [PREFIX_TOKENS + P, hidden] after each block in `taps`.
   *  Through the blocks in `crossFrame`, the class and register tokens
   *  keep their input values. */
  encode(
    patches: Value,
    rope: Rope,
    taps: readonly number[],
    crossFrame: readonly number[],
  ): Value[] {
    const embedded = this.patch_embed.proj.forward(patches);
    let x = cat([this.cls_token.value, this.reg_token.value, embedded], 0);
    const rows = x.shape.dims![0]!;
    const out: Value[] = [];
    for (const [i, block] of this.blocks.items.entries()) {
      const y = block.forward(x, rope);
      x = crossFrame.includes(i)
        ? cat([slice(x, 0, 0, PREFIX_TOKENS), slice(y, 0, PREFIX_TOKENS, rows)], 0)
        : y;
      if (taps.includes(i)) out.push(x);
    }
    return out;
  }
}
