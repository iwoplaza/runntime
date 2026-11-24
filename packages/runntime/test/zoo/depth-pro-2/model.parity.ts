/** Depth Pro 2 against Apple's PyTorch model on the same COCO photos.
 *  Level A compares the inverse depth and the field of view for the
 *  reference's own model input, level B the focal length and the depth in
 *  meters createMetricDepthEstimator gives for the photo. */
import { createMetricDepthEstimator, type MetricDepthEstimator } from '../../../src/zoo/index.ts';
import { openWeights, type ModelPath } from '../../../src/zoo/load.ts';
import {
  createDepthPro2,
  type DepthPro2Estimator,
} from '../../../src/zoo/depth-pro-2/estimator.ts';
import { gpu } from '../../gpu.ts';
import {
  readImage,
  readJson,
  readTensors,
  reversedWeights,
  runParity,
  summaryCell,
  type Measured,
} from '../../parity/harness.ts';
import { summarize } from '../../parity/metrics.ts';
import { DEPTH_PRO_2_B_PARITY, DEPTH_PRO_2_S_PARITY } from './parity.manifest.ts';

await gpu();

/** Mean of |got − want| / want over the pixels where want is positive and
 *  finite. */
function absRel(got: ArrayLike<number>, want: ArrayLike<number>): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < want.length; i++) {
    if (!(want[i]! > 0 && want[i]! < Infinity)) continue;
    sum += Math.abs(got[i]! - want[i]!) / want[i]!;
    count++;
  }
  return sum / count;
}

/** One set of weights loaded twice: the bare model for level A, the public
 *  runner for level B. */
interface Loaded {
  model: DepthPro2Estimator;
  runner: MetricDepthEstimator;
  dispose(): void;
}

async function load(modelPath: ModelPath, fovPath: string): Promise<Loaded> {
  const model = await createDepthPro2(await openWeights(modelPath), await openWeights(fovPath));
  const runner = await createMetricDepthEstimator({ modelPath, fovPath }).catch((err: unknown) => {
    model.dispose();
    throw err;
  });
  return {
    model,
    runner,
    dispose() {
      model.dispose();
      runner.dispose();
    },
  };
}

/** One size's references, read once. */
async function references(family: string) {
  const inputs = await readJson<{ images: string[] }>(`${family}/inputs.json`);
  const reference = await readTensors(`${family}/reference.safetensors`);
  const images = await Promise.all(inputs.images.map(readImage));
  const tensor = (name: string) => Float32Array.from(reference.get(name)!);
  return { images, tensor, wantFov: tensor('fov'), wantFocal: tensor('focal_length_px') };
}

async function measure(
  { model, runner }: Loaded,
  { images, tensor, wantFov, wantFocal }: Awaited<ReturnType<typeof references>>,
): Promise<Measured> {
  const inverseDepth: number[] = [];
  const fovError: number[] = [];
  const focalError: number[] = [];
  const metricInverse: number[] = [];
  for (const [i, image] of images.entries()) {
    const { width, height } = image;
    const frame = { chw: tensor(`pixels.${i}`), width, height };
    const want = tensor(`inverse_depth.${i}`);
    inverseDepth.push(absRel(await model.inverseDepth(frame), want));
    const [hfov, vfov] = await model.fieldOfView(frame);
    fovError.push(Math.max(Math.abs(hfov - wantFov[2 * i]!), Math.abs(vfov - wantFov[2 * i + 1]!)));

    const got = await runner.estimateDepth(image);
    const f = wantFocal[i]!;
    focalError.push(Math.abs(got.focalLengthPx - f) / f);
    // 1 / depth in 1/m: bounded where the sky goes to Infinity.
    metricInverse.push(
      absRel(
        got.data.map((v) => 1 / v),
        want.map((v) => (width * v) / f),
      ),
    );
  }
  const [a, fov, focal, b] = [inverseDepth, fovError, focalError, metricInverse].map(summarize);
  return {
    cells: {
      'inverse depth AbsRel (mean / p99 / max)': summaryCell(a!),
      'field of view error, degrees (mean / p99 / max)': summaryCell(fov!),
      'focal length relative error (mean / p99 / max)': summaryCell(focal!),
      '1 / depth AbsRel (mean / p99 / max)': summaryCell(b!),
    },
    // The field of view and the focal length are gated through 1 / depth:
    // a broken depth file leaves the estimator's own weights intact.
    metrics: { inverseDepthAbsRel: a!.max, metricInverseDepthAbsRel: b!.max },
  };
}

for (const manifest of [DEPTH_PRO_2_S_PARITY, DEPTH_PRO_2_B_PARITY]) {
  const refs = await references(manifest.family);
  await runParity({
    manifest,
    load: (config, modelPath) => load(modelPath, config.fovPath),
    measure: (loaded) => measure(loaded, refs),
    broken: {
      label: 'one norm weight reversed',
      weights: (modelPath) => reversedWeights(modelPath, 'encoder.blocks.0.norm1.weight'),
    },
  });
}
