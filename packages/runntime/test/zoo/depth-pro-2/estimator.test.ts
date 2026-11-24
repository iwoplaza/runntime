import { describe, expect, it } from 'vitest';
import { padFrame } from '../../../src/zoo/depth-pro-2/estimator.ts';

describe('padFrame', () => {
  it('repeats the last column and row up to a multiple of 32', () => {
    const width = 33;
    const height = 2;
    const chw = Float32Array.from({ length: 3 * height * width }, (_, i) => i);
    const out = padFrame({ chw, width, height });
    expect([out.width, out.height]).toEqual([64, 32]);
    const at = (c: number, y: number, x: number) => out.chw[(c * 32 + y) * 64 + x];
    const src = (c: number, y: number, x: number) => chw[(c * height + y) * width + x];
    expect(at(1, 0, 10)).toBe(src(1, 0, 10));
    expect(at(1, 0, 63)).toBe(src(1, 0, 32));
    expect(at(2, 31, 0)).toBe(src(2, 1, 0));
    expect(at(2, 31, 63)).toBe(src(2, 1, 32));
  });

  it('leaves a frame already a multiple of 32 alone', () => {
    const frame = { chw: new Float32Array(3 * 32 * 64), width: 64, height: 32 };
    expect(padFrame(frame)).toEqual(frame);
  });
});
