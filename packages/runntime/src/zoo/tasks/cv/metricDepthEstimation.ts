/** Metric depth estimation task: a photo in, the distance of every pixel in
 *  meters out. Runs Depth Pro 2 on the initRunntime() device, with its
 *  field-of-view estimator for photos of unknown focal length. */

import { RunntimeError, createResourceScope, supportsF16 } from '../../../core/index.ts';
import { openWeights, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import { IMAGENET_MEAN, IMAGENET_STD } from '../../depth-pro-2/config.ts';
import { createDepthPro2, type DepthPro2Size, type Frame } from '../../depth-pro-2/estimator.ts';
import { focalLengthPx } from '../../depth-pro-2/fov.ts';
import { models } from '../../models.ts';
import { checkImageBuffer, type ImageBuffer } from './image.ts';
import { createImagePreprocessor } from './utils/imagePreprocessor.ts';
import { resizeLinear } from './utils/resizeLinear.ts';

/** The longest side the model reads by default, in pixels. */
const DEFAULT_MAX_SIDE = 1024;

/** What to load. Every field is optional; a missing one comes from
 *  models.metricDepthEstimation.DEPTH_PRO_2.DEFAULT. */
export interface MetricDepthEstimatorModel {
  /** The weights: a safetensors export of a Depth Pro 2 S or B checkpoint. */
  readonly modelPath?: ModelPath;
  /** The field-of-view estimator's weights, which give the focal length of
   *  a photo that comes without one. */
  readonly fovPath?: ModelPath;
  /** Photos with a longer side are scaled down to it, aspect ratio kept,
   *  before the model reads them. Default 1024. Time and memory grow with
   *  the square of the pixel count. */
  readonly maxSide?: number;
}

export interface EstimateMetricDepthOptions {
  /** The camera's focal length in pixels of this image, horizontal and
   *  vertical. Default: estimated from the image. */
  readonly focalLengthPx?: number;
}

/** Depth in meters of the model input, one value per pixel, row by row. */
export interface MetricDepthMap {
  /** The image's size, or the size it was scaled down to by `maxSide`. */
  readonly width: number;
  readonly height: number;
  /** Distance along the camera's viewing direction (the z coordinate of
   *  each point), not along the ray. Sky can be Infinity or kilometers
   *  away. */
  readonly data: Float32Array;
  /** The focal length the depth is computed with, in pixels of the image
   *  passed in: the given one, or the estimate. */
  readonly focalLengthPx: number;
}

export interface MetricDepthEstimator {
  /** Model size, read from the weights: `s` or `b`. */
  readonly size: DepthPro2Size;
  /** Image in, its depth map in meters out.
   *  @throws RunntimeError `INVALID_ARGUMENT` for a malformed image or a
   *  focal length that is not a positive number. */
  estimateDepth(image: ImageBuffer, options?: EstimateMetricDepthOptions): Promise<MetricDepthMap>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** `image` as a normalized frame, scaled down to fit `maxSide`. */
function toFrame(image: ImageBuffer, maxSide: number): Frame {
  const { width, height } = image;
  const preprocessor = createImagePreprocessor(
    {
      resizeMode: 'stretch',
      normalizeOpts: {
        alpha: IMAGENET_STD.map((s) => 1 / (255 * s)),
        beta: IMAGENET_MEAN.map((m, c) => -m / IMAGENET_STD[c]!),
      },
    },
    { width, height },
  );
  const chw = preprocessor.process(image);
  const scale = maxSide / Math.max(width, height);
  if (scale >= 1) return { chw, width, height };
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  return { chw: resizeLinear(chw, width, height, w, h), width: w, height: h };
}

/** Loads both checkpoints, warms the GPU path once, and returns the
 *  runner. */
export async function createMetricDepthEstimator(
  config: MetricDepthEstimatorModel = {},
  opts: LoadOptions = {},
): Promise<MetricDepthEstimator> {
  if (!supportsF16()) {
    throw new RunntimeError(
      'UNSUPPORTED_DEVICE',
      "createMetricDepthEstimator: this device has no shader-f16, which the conv kernels need. Request it with tgpu.init({ device: { optionalFeatures: ['shader-f16'] } })",
    );
  }
  const defaults = models.metricDepthEstimation.DEPTH_PRO_2.DEFAULT;
  const {
    modelPath = defaults.modelPath,
    fovPath = defaults.fovPath,
    maxSide = DEFAULT_MAX_SIDE,
  } = config;
  if (!Number.isInteger(maxSide) || maxSide < 1) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `maxSide: expected a positive integer, got ${maxSide}`,
    );
  }
  const scope = createResourceScope();
  try {
    // Cache keys: the whole path, so same-named files of two sizes stay
    // apart. Bump the version to drop bytes cached under the old one.
    const cacheId = (path: ModelPath) =>
      typeof path === 'string' ? `depth-pro-2-v1/${path.replace(/^https?:\/\//, '')}` : undefined;
    const [sd, fovSd] = await Promise.all([
      openWeights(modelPath, { ...opts, cacheId: opts.cacheId ?? cacheId(modelPath) }),
      openWeights(fovPath, { ...opts, cacheId: cacheId(fovPath) }),
    ]);
    throwIfAborted(opts.signal);
    const estimator = scope.track(await createDepthPro2(sd, fovSd, opts));

    let disposed = false;
    // Calls run one after another, each at its own image size.
    let queue: Promise<unknown> = Promise.resolve();
    return {
      size: estimator.size,
      async estimateDepth(image, { focalLengthPx: given } = {}) {
        if (disposed) {
          throw new RunntimeError('RESOURCE_DISPOSED', 'metric depth estimator is disposed');
        }
        checkImageBuffer(image);
        if (given !== undefined && !(given > 0 && Number.isFinite(given))) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `focalLengthPx: expected a positive number, got ${given}`,
          );
        }
        const run = queue.then(async () => {
          const frame = toFrame(image, maxSide);
          const scale = frame.width / image.width;
          const inverse = await estimator.inverseDepth(frame);
          const f =
            given !== undefined
              ? given * scale
              : focalLengthPx(await estimator.fieldOfView(frame), frame.height, frame.width);
          // inverse = f / (width · depth), with f and width in pixels of the frame.
          const data = inverse.map((v) => f / (frame.width * v));
          return { width: frame.width, height: frame.height, data, focalLengthPx: f / scale };
        });
        queue = run.catch(() => undefined);
        return run.catch(rethrowRunError);
      },
      dispose() {
        disposed = true;
        scope.dispose();
      },
    };
  } catch (err) {
    scope.dispose();
    throw asLoadError(err, opts.signal);
  }
}
