import { describe, expect, it } from 'vitest';
import {
  focalLengthPx,
  fovDegrees,
  fovInput,
  poolWeights,
  sincos2d,
} from '../../../src/zoo/depth-pro-2/fov.ts';

const close = (want: number, digits = 5) => expect.closeTo(want, digits);

describe('fovInput', () => {
  // depth_pro_2 FOVEstimator.input_size and the shape resize() returns.
  it.each([
    [480, 640, [480, 640], [480, 640]],
    [427, 640, [416, 608], [416, 608]],
    [300, 1000, [288, 960], [288, 512]],
    [1000, 300, [960, 288], [512, 288]],
    [64, 64, [64, 64], [64, 64]],
    [20, 50, [32, 64], [32, 32]],
  ])('%i×%i resizes to %j and keeps %j', (height, width, resize, crop) => {
    const got = fovInput(height, width);
    expect([got.resize.height, got.resize.width]).toEqual(resize);
    expect([got.crop.height, got.crop.width]).toEqual(crop);
  });
});

describe('fovDegrees', () => {
  const logit = (deg: number) => -Math.log(178 / (deg - 1) - 1);

  it('maps the logits through a sigmoid onto 1° … 179°', () => {
    const got = fovDegrees(Float32Array.of(logit(60), logit(40)), 480, 640);
    expect(got).toEqual([close(60), close(40)]);
  });

  it('turns the angles of a cropped frame into the whole frame’s', () => {
    // depth_pro_2 crop_angles_to_full((60, 40), (300, 533.33), (300, 1000)).
    const got = fovDegrees(Float32Array.of(logit(60), logit(40)), 300, 1000);
    expect(got).toEqual([close(94.53896, 3), close(40)]);
  });
});

describe('focalLengthPx', () => {
  // depth_pro_2 median_focal_length_px of one frame.
  it.each([
    [480, 640, 659.39459],
    [640, 480, 415.69217],
  ])('reads %i×%i along its shorter side', (height, width, want) => {
    expect(focalLengthPx([60, 40], height, width)).toEqual(close(want, 3));
  });
});

describe('sincos2d', () => {
  // depth_pro_2 sincos_2d(2, 3, 248): [token, column, value].
  it.each([
    [0, 0, -0.5],
    [0, 61, 0.8660264],
    [0, 62, 0.8660254],
    [0, 124, -0.8660254],
    [0, 186, 0.5],
    [0, 247, -0.5000032],
    [5, 0, 0.5],
    [5, 61, -0.8660264],
    [5, 124, 0.8660254],
  ])('token %i, column %i matches depth_pro_2', (k, j, want) => {
    expect(sincos2d(2, 3, 248)[k * 248 + j]).toEqual(close(want));
  });
});

describe('poolWeights', () => {
  it('averages each 2×2 square of patches into one token', () => {
    // A 2×4 grid pools to 1×2 tokens.
    const w = poolWeights(2, 4);
    expect(Array.from(w)).toEqual([
      ...[0.25, 0.25, 0, 0, 0.25, 0.25, 0, 0],
      ...[0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25],
    ]);
  });
});
