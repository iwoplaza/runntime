import { describe, expect, it } from 'vitest';
import { resizeLinear } from '../../../../../src/zoo/tasks/cv/utils/resizeLinear.ts';

/** [2, 5, 7] of i^1.5, a ramp with curvature so every weight counts. */
const input = Float32Array.from({ length: 2 * 5 * 7 }, (_, i) => i ** 1.5);
const close = (want: number) => expect.closeTo(want, 3);

describe('resizeLinear', () => {
  // torch F.interpolate(x, size, mode='bilinear', antialias=True).
  it('matches torch when it scales down', () => {
    const got = resizeLinear(input, 7, 5, 4, 3);
    expect(Array.from(got)).toEqual(
      [
        9.1865, 13.5375, 19.9746, 26.2568, 57.5621, 66.5887, 77.8506, 87.8482, 129.8725, 141.8255,
        156.3996, 169.0885, 239.851, 254.4779, 272.1339, 287.3665, 349.695, 366.2391, 386.1209,
        403.203, 471.4541, 489.729, 511.6269, 530.3896,
      ].map(close),
    );
  });

  it('matches torch when it scales up', () => {
    const got = resizeLinear(input, 7, 5, 9, 8);
    expect(Array.from(got.subarray(0, 12))).toEqual(
      [
        0, 0.6667, 1.8126, 3.3546, 5.1962, 7.3769, 9.7669, 12.3525, 14.6969, 8.1026, 9.6755,
        11.7693,
      ].map(close),
    );
  });
});
