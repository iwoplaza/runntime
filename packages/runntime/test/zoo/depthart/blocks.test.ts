import { describe, expect, it } from 'vitest';
import { foldBnIntoConv } from '../../../src/core/weights/foldBn.ts';
import { memoryStateDict } from '../../../src/core/weights/safetensors.ts';
import { ConvBlock } from '../../../src/zoo/depthart/blocks.ts';

describe('ConvBlock', () => {
  // fc2 is a 1×1 conv, 2 in → 2 out, followed by BatchNorm, then layer_scale.
  const EPS = 1e-5;
  const entries = () => ({
    'b.layer_scale': { data: new Float32Array([0.5, -2]), shape: [2, 1, 1] },
    'b.mlp.fc2.c.weight': { data: new Float32Array([1, 2, 3, 4]), shape: [2, 2, 1, 1] },
    'b.mlp.fc2.bn.weight': new Float32Array([2, 3]),
    'b.mlp.fc2.bn.bias': new Float32Array([1, -1]),
    'b.mlp.fc2.bn.running_mean': new Float32Array([0.5, 0.25]),
    'b.mlp.fc2.bn.running_var': new Float32Array([4 - EPS, 1 - EPS]),
  });

  it('scales the BatchNorm affine by layer_scale and drops it', async () => {
    const sd = memoryStateDict(entries());
    new ConvBlock(2).transformStateDict(sd, 'b');
    expect(sd.tensors.has('b.layer_scale')).toBe(false);
    expect(await sd.tensors.get('b.mlp.fc2.bn.weight')!.f32()).toEqual(new Float32Array([1, -6]));
    expect(await sd.tensors.get('b.mlp.fc2.bn.bias')!.f32()).toEqual(new Float32Array([0.5, 2]));
  });

  it('folds into fc2 as layer_scale · BatchNorm(conv)', async () => {
    const sd = memoryStateDict(entries());
    new ConvBlock(2).transformStateDict(sd, 'b');
    foldBnIntoConv(sd, 'b.mlp.fc2.c', 'b.mlp.fc2.bn', EPS);
    // Per out channel o: s = gamma / sqrt(var + eps) · layer_scale,
    // weight' = weight · s, bias' = (beta − mean · gamma / sqrt(var + eps)) · layer_scale.
    // o = 0: s = 2/2 · 0.5 = 0.5, bias' = (1 − 0.5·1) · 0.5 = 0.25.
    // o = 1: s = 3/1 · −2 = −6, bias' = (−1 − 0.25·3) · −2 = 3.5.
    expect(await sd.tensors.get('b.mlp.fc2.c.weight')!.f32()).toEqual(
      new Float32Array([0.5, 1, -18, -24]),
    );
    expect(await sd.tensors.get('b.mlp.fc2.c.bias')!.f32()).toEqual(new Float32Array([0.25, 3.5]));
  });
});
