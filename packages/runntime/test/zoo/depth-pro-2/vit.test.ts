import { describe, expect, it } from 'vitest';
import { dinov3RopeTables, patchify } from '../../../src/zoo/depth-pro-2/vit.ts';

const close = (want: number) => expect.closeTo(want, 6);

describe('dinov3RopeTables', () => {
  const headDim = 64;
  const { cos, sin } = dinov3RopeTables(2, 3, headDim);
  /** The entries of pair `p` of row `k`, both slots. */
  const pair = (table: Float32Array, k: number, p: number) => [
    table[k * headDim + 2 * p]!,
    table[k * headDim + 2 * p + 1]!,
  ];

  it('turns the class and register tokens by 0', () => {
    for (let k = 0; k < 5; k++) {
      expect(Array.from(cos.subarray(k * headDim, (k + 1) * headDim))).toEqual(
        new Array<number>(headDim).fill(1),
      );
      expect(Array.from(sin.subarray(k * headDim, (k + 1) * headDim))).toEqual(
        new Array<number>(headDim).fill(0),
      );
    }
  });

  // timm's vit_small_patch16_dinov3 rope with normalize_coords 'max',
  // get_embed(shape=[2, 3]): [patch, pair, cos, sin].
  it.each([
    [0, 0, -0.5000003, 0.8660253],
    [0, 7, 0.8480064, -0.529986],
    [0, 16, -0.5000003, 0.8660253],
    [0, 31, 0.9984403, -0.0558294],
    [4, 0, 1, 0],
    [4, 31, 1, 0],
    [5, 0, 1, 0],
    [5, 16, -0.5000003, -0.8660253],
    [5, 31, 0.9984403, 0.0558294],
  ])('patch %i, pair %i matches timm', (k, p, c, s) => {
    expect(pair(cos, 5 + k, p)).toEqual([close(c), close(c)]);
    expect(pair(sin, 5 + k, p)).toEqual([close(s), close(s)]);
  });
});

describe('patchify', () => {
  it('cuts row-major patches, each (channel, row, column)', () => {
    // [3, 32, 16]: two patches, one above the other.
    const chw = Float32Array.from({ length: 3 * 32 * 16 }, (_, i) => i);
    const out = patchify(chw, 2, 1);
    const at = (c: number, y: number, x: number) => c * 32 * 16 + y * 16 + x;
    expect(out.length).toBe(chw.length);
    // Patch 1, channel 2, row 3, column 5 is pixel (16 + 3, 5) of channel 2.
    expect(out[768 + 2 * 256 + 3 * 16 + 5]).toBe(at(2, 19, 5));
    expect(out[0]).toBe(at(0, 0, 0));
    expect(out[767]).toBe(at(2, 15, 15));
  });
});
