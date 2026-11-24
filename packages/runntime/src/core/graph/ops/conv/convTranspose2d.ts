import { hwc4Meta, pending, Value } from '../../value.ts';
import { FLOAT_DTYPES, want3d, wantHwc4 } from '../shared.ts';

/** Transposed 2D convolution, restricted to the no-overlap case kernelSize ===
 *  stride, padding 0 (torch nn.ConvTranspose2d(cIn, cOut, k, k)): x=[C_in,H,W]
 *  → [C_out, H·k, W·k]. With k === s every output pixel gets exactly one
 *  contribution: out[co, k·i+di, k·j+dj] = Σ_ci W[ci,co,di,dj]·x[ci,i,j] + b.
 *  Weight is f16 [C_out·k·k, C_in] — the exporter pre-permutes torch's
 *  [C_in, C_out, kH, kW] so row co·k²+di·k+dj holds the (di,dj) tap of output
 *  channel co (contiguous C_in per row → sequential f16-pair decode).
 *  Generalizing to k ≠ s later = a gather path behind this same signature.
 *
 *  An hwc4-stored x gives an hwc4 output, with the sums in f32 and an f32
 *  bias. That route needs C_in to be a multiple of 4.
 *  attrs: [k, hasBias] */
export function convTranspose2d(
  x: Value,
  weight: Value,
  bias: Value | undefined,
  opts: { kernelSize: number; stride: number },
): Value {
  const hwc4 = x.shape.layout === 'hwc4';
  const [cIn, h, w] = hwc4
    ? wantHwc4('convTranspose2d', x)
    : want3d('convTranspose2d', x, FLOAT_DTYPES);
  if (hwc4 && cIn % 4 !== 0) {
    throw new Error(`convTranspose2d: an hwc4 input needs C_in a multiple of 4, got ${cIn}`);
  }
  const { kernelSize: k, stride } = opts;
  if (!Number.isInteger(k) || k < 1 || !Number.isInteger(stride)) {
    throw new Error(
      `convTranspose2d: kernelSize and stride must be positive integers, got k=${k} stride=${stride}`,
    );
  }
  if (stride !== k) {
    throw new Error(
      `convTranspose2d: only kernelSize === stride (no output overlap) is implemented, got k=${k} stride=${stride}`,
    );
  }
  const wd = weight.shape.dims;
  if (weight.shape.dtype !== 'f16' || !wd || wd.length !== 2) {
    throw new Error('convTranspose2d: weight must be 2D f16 [C_out·k·k, C_in]');
  }
  const [rows, wCin] = wd as [number, number];
  if (wCin !== cIn) {
    throw new Error(`convTranspose2d: weight C_in ${wCin} != input C_in ${cIn}`);
  }
  if (rows % (k * k) !== 0) {
    throw new Error(`convTranspose2d: weight rows ${rows} not a multiple of k²=${k * k}`);
  }
  const cOut = rows / (k * k);
  const biasDtype = hwc4 ? 'f32' : x.shape.dtype;
  if (bias !== undefined && (bias.shape.dtype !== biasDtype || bias.shape.elems !== cOut)) {
    throw new Error(`convTranspose2d: bias must be ${biasDtype} with ${cOut} elems`);
  }
  const inputs: Value[] = bias !== undefined ? [x, weight, bias] : [x, weight];
  if (hwc4) {
    return pending(hwc4Meta(cOut, h * k, w * k), 'convTranspose2dHwc4', inputs, undefined, [
      k,
      bias !== undefined ? 1 : 0,
    ]);
  }
  return pending(
    { elems: cOut * h * k * w * k, dtype: x.shape.dtype, dims: [cOut, h * k, w * k] },
    'convTranspose2d',
    inputs,
    undefined,
    [k, bias !== undefined ? 1 : 0],
  );
}
