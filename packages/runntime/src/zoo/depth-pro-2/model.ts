/** Depth Pro 2 on one frame: the DINOv3 encoder, its 4 tapped blocks
 *  reassembled into a pyramid at 1/4 to 1/32 of the input, a DPT decoder up
 *  to 1/2, and a head to inverse depth at the input's size.
 *
 *      encoder.blocks          DINOv3 ViT/16, taps after 4 blocks   (1/16)
 *      encoder.reassemble.0    1×1 conv, 4×4 deconv                 (1/4)
 *      encoder.reassemble.1    1×1 conv, 2×2 deconv                 (1/8)
 *      encoder.reassemble.2    1×1 conv                             (1/16)
 *      encoder.reassemble.3    1×1 conv, 3×3 stride-2 conv          (1/32)
 *      decoder                 fuse coarse to fine, each 2× up      (1/2)
 *      head                    conv, 2×2 deconv, conv, 1×1 conv     (1/1)
 *
 *  With one frame, the cross-frame blocks leave the class and register
 *  tokens as they were. The 1×1 convs
 *  run as Linears on the token rows, before the pyramid enters hwc4. Field
 *  names follow the checkpoint, the numeric ones its Sequentials' indices,
 *  where a ReLU's index holds no weights. */

import {
  add,
  clamp,
  nn,
  reshape,
  reshapeWeight,
  slice,
  toChw,
  toHwc4,
  transpose,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { requireTensor } from '../errors.ts';
import { PREFIX_TOKENS, type DepthPro2Config } from './config.ts';
import { Dinov3, type Rope } from './vit.ts';

const relu = (x: Value): Value => clamp(x, 0, Infinity);

/** One pyramid level: patch tokens to a feature map at its resolution. */
class Reassemble extends nn.Module<[Value, number, number], Value> {
  /** The checkpoint's 1×1 conv. */
  readonly 0: nn.Linear;
  readonly 1?: nn.ConvTranspose2d | nn.Conv2d;
  constructor(project: nn.Linear, resample?: nn.ConvTranspose2d | nn.Conv2d) {
    super();
    this[0] = project;
    this[1] = resample;
  }
  /** Patch tokens [rows · cols, hidden] to [dim, ·, ·] in hwc4. */
  override forward(tokens: Value, rows: number, cols: number): Value {
    const projected = this[0].forward(tokens);
    const x = toHwc4(reshape(transpose(projected), [this[0].outFeatures, rows, cols]));
    return this[1] ? this[1].forward(x) : x;
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = `${prefix}.0.weight`;
    const weight = requireTensor(sd, name, 'Depth Pro 2 reassemble');
    sd.tensors.set(name, reshapeWeight(weight, [this[0].outFeatures, this[0].inFeatures]));
  }
}

class MultiViewEncoder extends Dinov3 {
  readonly reassemble: nn.ModuleList<Reassemble>;

  constructor(private readonly cfg: DepthPro2Config) {
    super(cfg.vit);
    const { hidden } = cfg.vit;
    const [d0, d1, d2, d3] = cfg.dimsReassemble;
    const project = (dim: number) => new nn.Linear(hidden, dim);
    this.reassemble = new nn.ModuleList([
      new Reassemble(project(d0), new nn.ConvTranspose2d(d0, d0, { kernelSize: 4, stride: 4 })),
      new Reassemble(project(d1), new nn.ConvTranspose2d(d1, d1, { kernelSize: 2, stride: 2 })),
      new Reassemble(project(d2)),
      new Reassemble(project(d3), new nn.Conv2d(d3, d3, { kernelSize: 3, stride: 2, padding: 1 })),
    ]);
  }

  /** The pyramid of one frame, 1/4 resolution first, in hwc4. */
  pyramid(patches: Value, rope: Rope, rows: number, cols: number): Value[] {
    const end = PREFIX_TOKENS + rows * cols;
    return this.encode(patches, rope, this.cfg.taps, this.cfg.crossFrame).map((tokens, i) =>
      this.reassemble.items[i]!.forward(slice(tokens, 0, PREFIX_TOKENS, end), rows, cols),
    );
  }
}

/** ReLU, conv, ReLU, conv. The inner ReLU rides the first conv. */
class Residual extends nn.Module {
  readonly 1: nn.Conv2d;
  readonly 3: nn.Conv2d;
  constructor(dim: number) {
    super();
    this[1] = new nn.Conv2d(dim, dim, { kernelSize: 3, padding: 1, activation: 'relu' });
    this[3] = new nn.Conv2d(dim, dim, { kernelSize: 3, padding: 1 });
  }
  forward(x: Value): Value {
    return this[3].forward(this[1].forward(relu(x)));
  }
}

class ResidualBlock extends nn.Module {
  readonly residual: Residual;
  constructor(dim: number) {
    super();
    this.residual = new Residual(dim);
  }
  forward(x: Value): Value {
    return add(x, this.residual.forward(x));
  }
}

class FeatureFusionBlock extends nn.Module<[Value, Value | undefined], Value> {
  readonly resnet1?: ResidualBlock;
  readonly resnet2: ResidualBlock;
  readonly deconv: nn.ConvTranspose2d;
  readonly out_conv: nn.Conv2d;
  constructor(dim: number, withSkip: boolean) {
    super();
    this.resnet1 = withSkip ? new ResidualBlock(dim) : undefined;
    this.resnet2 = new ResidualBlock(dim);
    this.deconv = new nn.ConvTranspose2d(dim, dim, { kernelSize: 2, stride: 2, bias: false });
    this.out_conv = new nn.Conv2d(dim, dim, { kernelSize: 1 });
  }
  override forward(x: Value, skip: Value | undefined): Value {
    if (skip && this.resnet1) x = add(x, this.resnet1.forward(skip));
    return this.out_conv.forward(this.deconv.forward(this.resnet2.forward(x)));
  }
}

class DPTDecoder extends nn.Module<[readonly Value[]], Value> {
  readonly convs: nn.ModuleList<nn.Conv2d>;
  readonly fusions: nn.ModuleList<FeatureFusionBlock>;
  constructor(dims: readonly number[], dim: number) {
    super();
    this.convs = new nn.ModuleList(
      dims.map((c) => new nn.Conv2d(c, dim, { kernelSize: 3, padding: 1, bias: false })),
    );
    this.fusions = new nn.ModuleList(
      dims.map((_, i) => new FeatureFusionBlock(dim, i < dims.length - 1)),
    );
  }
  /** The pyramid, 1/4 resolution first, to features at 1/2. */
  override forward(levels: readonly Value[]): Value {
    const convs = this.convs.items;
    const fusions = this.fusions.items;
    const last = levels.length - 1;
    let x = fusions[last]!.forward(convs[last]!.forward(levels[last]!), undefined);
    for (let i = last - 1; i >= 0; i--) {
      x = fusions[i]!.forward(x, convs[i]!.forward(levels[i]!));
    }
    return x;
  }
}

/** Decoder features at 1/2 to inverse depth at the input's size. Both
 *  ReLUs ride their convs. */
class InverseDepthHead extends nn.Module {
  readonly 0: nn.Conv2d;
  readonly 1: nn.ConvTranspose2d;
  readonly 2: nn.Conv2d;
  readonly 4: nn.Conv2d;
  constructor(dim: number) {
    super();
    this[0] = new nn.Conv2d(dim, dim / 2, { kernelSize: 3, padding: 1 });
    this[1] = new nn.ConvTranspose2d(dim / 2, dim / 2, { kernelSize: 2, stride: 2 });
    this[2] = new nn.Conv2d(dim / 2, 32, { kernelSize: 3, padding: 1, activation: 'relu' });
    this[4] = new nn.Conv2d(32, 1, { kernelSize: 1, activation: 'relu' });
  }
  forward(x: Value): Value {
    return this[4].forward(this[2].forward(this[1].forward(this[0].forward(x))));
  }
}

export class DepthPro2Model extends nn.Module<[Value, Rope, number, number], Value> {
  readonly encoder: MultiViewEncoder;
  readonly decoder: DPTDecoder;
  readonly head: InverseDepthHead;

  constructor(readonly cfg: DepthPro2Config) {
    super();
    this.encoder = new MultiViewEncoder(cfg);
    this.decoder = new DPTDecoder(cfg.dimsReassemble, cfg.dimDecoder);
    this.head = new InverseDepthHead(cfg.dimDecoder);
  }

  /** Inverse depth [1, 16 · rows, 16 · cols] f32 of one frame's patches
   *  [rows · cols, 3 · 16²], normalized by the focal length: f / (width ·
   *  depth), with f and width in pixels. `rows` and `cols` are even. */
  override forward(patches: Value, rope: Rope, rows: number, cols: number): Value {
    const pyramid = this.encoder.pyramid(patches, rope, rows, cols);
    return toChw(this.head.forward(this.decoder.forward(pyramid)), 'f32');
  }
}
