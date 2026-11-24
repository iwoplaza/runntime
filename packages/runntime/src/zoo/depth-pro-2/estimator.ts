/** The GPU side of Depth Pro 2: the weights in, functions from one
 *  normalized frame to its inverse depth and its field of view out. The
 *  frame's size varies, so each call runs eagerly and trims the pool. */

import {
  defaultRoot,
  gpuExecutor,
  tensor,
  toArray,
  warmUp,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import { combinedProgress, type ModelLoadOptions } from '../load.ts';
import { resizeLinear } from '../tasks/cv/utils/resizeLinear.ts';
import {
  DEPTH_PRO_2_B,
  DEPTH_PRO_2_FOV,
  DEPTH_PRO_2_S,
  INPUT_MULTIPLE,
  PATCH,
  type DepthPro2Config,
  type Dinov3Config,
} from './config.ts';
import { fovDegrees, fovInput, FovModel, poolWeights, sincos2d } from './fov.ts';
import { DepthPro2Model } from './model.ts';
import { dinov3RopeTables, patchify } from './vit.ts';

export type DepthPro2Size = 's' | 'b';

export const DEPTH_PRO_2_CONFIGS: Record<DepthPro2Size, DepthPro2Config> = {
  s: DEPTH_PRO_2_S,
  b: DEPTH_PRO_2_B,
};

/** A normalized frame: ImageNet-normalized RGB [3, height, width]. */
export interface Frame {
  readonly chw: Float32Array;
  readonly width: number;
  readonly height: number;
}

export interface DepthPro2Estimator {
  readonly size: DepthPro2Size;
  /** Inverse depth [height · width] of the frame, normalized by the focal
   *  length: f / (width · depth). */
  inverseDepth(frame: Frame): Promise<Float32Array>;
  /** The frame's [horizontal, vertical] field of view in degrees. */
  fieldOfView(frame: Frame): Promise<[number, number]>;
  dispose(): void;
}

/** The frame padded to a multiple of INPUT_MULTIPLE on the right and the
 *  bottom, repeating its last column and row. */
export function padFrame({ chw, width, height }: Frame): Frame {
  const w = Math.ceil(width / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const h = Math.ceil(height / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  if (w === width && h === height) return { chw, width, height };
  const out = new Float32Array(3 * h * w);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < h; y++) {
      const src = (c * height + Math.min(y, height - 1)) * width;
      const dst = (c * h + y) * w;
      out.set(chw.subarray(src, src + width), dst);
      out.fill(chw[src + width - 1]!, dst + width, dst + w);
    }
  }
  return { chw: out, width: w, height: h };
}

/** The centered `height` × `width` window of a frame. */
function cropFrame(frame: Frame, height: number, width: number): Frame {
  if (height === frame.height && width === frame.width) return frame;
  const top = (frame.height - height) >> 1;
  const left = (frame.width - width) >> 1;
  const out = new Float32Array(3 * height * width);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < height; y++) {
      const src = (c * frame.height + top + y) * frame.width + left;
      out.set(frame.chw.subarray(src, src + width), (c * height + y) * width);
    }
  }
  return { chw: out, width, height };
}

function sizeFromStateDict(sd: LazyStateDict): DepthPro2Size {
  const hidden = sd.tensors.get('encoder.cls_token')?.shape.at(-1);
  const found = (Object.keys(DEPTH_PRO_2_CONFIGS) as DepthPro2Size[]).find(
    (s) => DEPTH_PRO_2_CONFIGS[s].vit.hidden === hidden,
  );
  if (!found) {
    throw checkpointMismatch(
      'Depth Pro 2 weights',
      `not a Depth Pro 2 S or B checkpoint (class token of ${hidden ?? 'no'} channels, expected 384 or 768)`,
    );
  }
  return found;
}

/** Loads the depth weights and, when given, the field-of-view estimator's,
 *  and warms both up on a blank frame. */
export async function createDepthPro2(
  sd: LazyStateDict,
  fovSd: LazyStateDict | undefined,
  opts: ModelLoadOptions = {},
): Promise<DepthPro2Estimator> {
  const root = defaultRoot();
  const size = sizeFromStateDict(sd);
  const model = new DepthPro2Model(DEPTH_PRO_2_CONFIGS[size]);
  const fov = fovSd && new FovModel(DEPTH_PRO_2_FOV);
  const ex = gpuExecutor(root);
  const dispose = () => {
    model.dispose();
    fov?.dispose();
    ex.dispose();
  };

  /** Runs `forward` on uploads of `inputs` and reads its output back. */
  const run = async (
    inputs: readonly (readonly [Float32Array, number])[],
    forward: (values: Value[]) => Value,
  ): Promise<Float32Array> => {
    const values = inputs.map(([data, rows]) =>
      tensor(root, data, { elems: data.length, dtype: 'f32', dims: [rows, data.length / rows] }),
    );
    try {
      return await toArray(forward(values), ex);
    } finally {
      for (const v of values) v.buffer.destroy();
      ex.trim();
    }
  };

  const ropeInputs = (rows: number, cols: number, { hidden, heads }: Dinov3Config) => {
    const headDim = hidden / heads;
    const { cos, sin } = dinov3RopeTables(rows, cols, headDim);
    const n = cos.length / headDim;
    return [
      [cos, n],
      [sin, n],
    ] as const;
  };

  const inverseDepth = async (frame: Frame): Promise<Float32Array> => {
    const padded = padFrame(frame);
    const rows = padded.height / PATCH;
    const cols = padded.width / PATCH;
    const out = await run(
      [[patchify(padded.chw, rows, cols), rows * cols], ...ropeInputs(rows, cols, model.cfg.vit)],
      ([patches, cos, sin]) => model.forward(patches!, { cos: cos!, sin: sin! }, rows, cols),
    );
    if (padded === frame) return out;
    const cropped = new Float32Array(frame.height * frame.width);
    for (let y = 0; y < frame.height; y++) {
      const src = y * padded.width;
      cropped.set(out.subarray(src, src + frame.width), y * frame.width);
    }
    return cropped;
  };

  const fovLogits = async (frame: Frame): Promise<Float32Array> => {
    if (!fov) throw new Error('fieldOfView: no field-of-view weights loaded');
    const { resize, crop } = fovInput(frame.height, frame.width);
    const resized = {
      chw: resizeLinear(frame.chw, frame.width, frame.height, resize.width, resize.height),
      ...resize,
    };
    const input = cropFrame(resized, crop.height, crop.width);
    const rows = input.height / PATCH;
    const cols = input.width / PATCH;
    const pooled = (rows / 2) * (cols / 2);
    return run(
      [
        [patchify(input.chw, rows, cols), rows * cols],
        ...ropeInputs(rows, cols, fov.cfg.vit),
        [poolWeights(rows, cols), pooled],
        [sincos2d(rows / 2, cols / 2, DEPTH_PRO_2_FOV.dim), pooled],
      ],
      ([patches, cos, sin, pool, position]) =>
        fov.forward({
          patches: patches!,
          rope: { cos: cos!, sin: sin! },
          pool: pool!,
          position: position!,
        }),
    );
  };

  try {
    const track = combinedProgress(opts.onProgress);
    await model.loadStateDict(sd, { onProgress: track('model') });
    if (fov && fovSd) await fov.loadStateDict(fovSd, { onProgress: track('fov') });
    if (opts.warmUp !== false) {
      const blank = { chw: new Float32Array(3 * 64 * 64), width: 64, height: 64 };
      await warmUp(root.device, () => inverseDepth(blank));
      if (fov) await warmUp(root.device, () => fovLogits(blank));
    }
  } catch (err) {
    dispose();
    throw err;
  }

  return {
    size,
    inverseDepth,
    async fieldOfView(frame) {
      return fovDegrees(await fovLogits(frame), frame.height, frame.width);
    },
    dispose,
  };
}
