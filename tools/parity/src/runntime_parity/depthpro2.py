"""Depth Pro 2 reference of one size, S or B: the inverse depth, the field
of view and the focal length of Apple's model in f32 on the CPU, one COCO
photo at a time at its own size.

    uv run python -m runntime_parity.depthpro2 s
"""

import copy
import sys
from contextlib import contextmanager

import numpy as np
import torch
from depth_pro_2 import create_model_and_transforms
from depth_pro_2.network.fov import median_focal_length_px

from .coco import ATTRIBUTION, coco_images
from .common import save

REPO = "apple-aiml-research/ml-depth-pro-2"
REVISION = "aa389cbb0b74341664ed663a0966c859905cd6c0"


@contextmanager
def float_is_double():
    """Makes `Tensor.float()` give f64, so the model's casts to f32 before
    its heads keep an f64 run in f64."""
    original = torch.Tensor.float
    torch.Tensor.float = torch.Tensor.double
    try:
        yield
    finally:
        torch.Tensor.float = original


def abs_rel(got: np.ndarray, want: np.ndarray) -> float:
    """Mean of |got - want| / want over the pixels where want is positive
    and finite."""
    mask = (want > 0) & np.isfinite(want)
    return float(np.mean(np.abs(got[mask] - want[mask]) / want[mask]))


def predict(model, pixels: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse depth (H, W) normalized by the focal length, [hfov, vfov] in
    degrees and 1 / depth in 1/m (H, W) of one photo."""
    height, width = pixels.shape[-2:]
    inverse_depth = model.predict_inverse_depth(pixels[None])[0].double()
    fov = model.fov(pixels[None])
    focal = median_focal_length_px(fov.double(), height, width)
    meters_inverse = width * inverse_depth / focal
    return inverse_depth.numpy(), fov[0].double().numpy(), meters_inverse.numpy()


def main() -> None:
    size = sys.argv[1]
    model_name = f"depthpro2-{size}"
    paths, images, sources = coco_images()
    model, transform = create_model_and_transforms(model_name, device="cpu", precision=torch.float32)
    double = copy.deepcopy(model).double()
    bf16, _ = create_model_and_transforms(model_name, device="cpu", precision=torch.bfloat16)
    tensors: dict[str, np.ndarray] = {}
    fovs, focals = [], []
    noise = {"inverseDepthAbsRel": 0.0, "fovError": 0.0, "metricInverseDepthAbsRel": 0.0}
    bf16_spread = dict.fromkeys(noise, 0.0)

    with torch.inference_mode():
        for i, image in enumerate(images):
            pixels = transform(np.asarray(image.convert("RGB")))
            inverse_depth, fov, meters_inverse = predict(model, pixels)
            tensors[f"pixels.{i}"] = pixels.numpy()
            tensors[f"inverse_depth.{i}"] = inverse_depth.astype(np.float32)
            fovs.append(fov)
            focals.append(float(median_focal_length_px(torch.from_numpy(fov[None]), *pixels.shape[-2:])))

            # The same photo in f64 gives the reference noise, and under bf16
            # autocast the spread of the precision the paper evaluates in.
            with float_is_double():
                runs = [predict(double, pixels.double())]
            runs.append(predict(bf16, pixels))
            for spread, (other_inverse, other_fov, other_meters) in zip((noise, bf16_spread), runs, strict=True):
                for name, value in (
                    ("inverseDepthAbsRel", abs_rel(other_inverse, inverse_depth)),
                    ("fovError", float(np.abs(fov - other_fov).max())),
                    ("metricInverseDepthAbsRel", abs_rel(other_meters, meters_inverse)),
                ):
                    spread[name] = max(spread[name], value)

    tensors["fov"] = np.stack(fovs).astype(np.float32)
    tensors["focal_length_px"] = np.array(focals, dtype=np.float32)
    save(
        model_name,
        {"images": paths},
        tensors,
        {
            "repo": REPO,
            "revision": REVISION,
            "dtype": "f32",
            "inputs": ATTRIBUTION | {"count": len(images), "images": sources},
            "noise": noise,
            "bf16": bf16_spread,
            "settings": {"model": model_name},
        },
        ["torch", "timm", "pillow", "depth_pro_2"],
    )


if __name__ == "__main__":
    main()
