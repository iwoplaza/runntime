/** Parity manifests of Depth Pro 2, one per size: Apple's PyTorch model in
 *  f32 as the reference, photos from COCO as inputs. */
import { models } from '../../../src/zoo/models.ts';
import type { ParityManifest } from '../../parity/manifest.ts';

/** Where every threshold comes from: the reference run again under bf16
 *  autocast, the precision the paper evaluates in, at its largest
 *  difference over the inputs (`bf16` in meta.json). runntime runs the
 *  convolutions in f16, so it has to stay within that. */
const bf16 = (value: number) => ({
  value,
  source: `the reference under bf16 autocast, ${value} in meta.json`,
});

const reference = {
  repo: 'apple-aiml-research/ml-depth-pro-2',
  revision: 'aa389cbb0b74341664ed663a0966c859905cd6c0',
  dtype: 'f32',
} as const;

const inputs = {
  source:
    'COCO val2017 (Lin et al., 2014) from images.cocodataset.org: the first 25 images under CC BY 2.0, by image id',
  license: 'CC BY 2.0',
  count: 25,
} as const;

export const DEPTH_PRO_2_S_PARITY = {
  family: 'depthpro2-s',
  reference,
  inputs,
  settings: { model: 'depthpro2-s' },
  variants: [
    {
      name: 'S',
      config: models.metricDepthEstimation.DEPTH_PRO_2.S,
      thresholds: {
        // Level A: inverse depth normalized by the focal length.
        inverseDepthAbsRel: bf16(0.0295),
        // Level B: 1 / depth in 1/m, with the estimated focal length.
        metricInverseDepthAbsRel: bf16(0.0301),
      },
    },
  ],
} as const satisfies ParityManifest;

export const DEPTH_PRO_2_B_PARITY = {
  family: 'depthpro2-b',
  reference,
  inputs,
  settings: { model: 'depthpro2-b' },
  variants: [
    {
      name: 'B',
      config: models.metricDepthEstimation.DEPTH_PRO_2.B,
      thresholds: {
        inverseDepthAbsRel: bf16(0.0124),
        metricInverseDepthAbsRel: bf16(0.0159),
      },
    },
  ],
} as const satisfies ParityManifest;
