#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import torch

from skimage.transform import resize
from skimage.morphology import disk
from scipy.ndimage import (
    binary_opening,
    binary_closing,
    binary_erosion,
    binary_dilation,
    label as ndi_label,
)

from MultiRetina_Curvelet_UNet import (
    matlab_uint8,
    normalize01,
    local_green_enhancement,
    curvelet_greycontrast_step,
    curvelet_edge_step,
    get_model_classes,
)


def ensure_dir(path):
    path = Path(path)
    if str(path) not in ["", "."]:
        path.mkdir(parents=True, exist_ok=True)


def read_rgb_general(path, size):
    img = np.squeeze(iio.imread(path))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim == 4:
        img = img[0]

    if img.ndim == 3 and img.shape[0] in [3, 4] and img.shape[-1] not in [3, 4]:
        img = np.moveaxis(img, 0, -1)

    if img.ndim != 3:
        raise ValueError(f"Cannot read RGB image {path}, shape={img.shape}")

    if img.shape[-1] > 3:
        img = img[:, :, :3]

    original_shape = img.shape[:2]

    img = resize(
        img,
        (size, size, 3),
        preserve_range=True,
        anti_aliasing=True,
    )

    return matlab_uint8(img), original_shape


def read_binary_mask_general(path, size):
    m = np.squeeze(iio.imread(path))

    if m.ndim == 3:
        if m.shape[-1] >= 3:
            m = m[..., :3].max(axis=-1)
        else:
            m = m.max(axis=0)

    if m.ndim != 2:
        raise ValueError(f"Cannot read ROI mask {path}, shape={m.shape}")

    m_bin = m > 0

    out = resize(
        m_bin.astype(np.uint8),
        (size, size),
        preserve_range=True,
        anti_aliasing=False,
        order=0,
    )

    return out > 0


def make_fov_from_rgb(rgb, erode_radius=8):
    green = rgb[:, :, 1].astype(np.float32)

    fov = green > 10

    fov = binary_opening(
        fov,
        structure=np.ones((5, 5), dtype=bool),
    )

    fov = binary_closing(
        fov,
        structure=np.ones((25, 25), dtype=bool),
    )

    lab, num = ndi_label(fov)

    if num > 0:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        fov = lab == np.argmax(counts)

    if erode_radius > 0:
        fov_inner = binary_erosion(fov, structure=disk(erode_radius))
    else:
        fov_inner = fov.copy()

    return fov.astype(bool), fov_inner.astype(bool)


def make_fov(rgb, roi_path, size, erode_radius=8):
    if roi_path is not None and str(roi_path).strip() != "" and Path(roi_path).exists():
        fov = read_binary_mask_general(roi_path, size=size)

        lab, num = ndi_label(fov)

        if num > 0:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            fov = lab == np.argmax(counts)

        if erode_radius > 0:
            fov_inner = binary_erosion(fov, structure=disk(erode_radius))
        else:
            fov_inner = fov.copy()

        return fov.astype(bool), fov_inner.astype(bool)

    return make_fov_from_rgb(rgb, erode_radius=erode_radius)


def build_features_from_image(image_path, size=512, roi_path=None):
    rgb, original_shape = read_rgb_general(image_path, size=size)

    fov, fov_inner = make_fov(
        rgb,
        roi_path=roi_path,
        size=size,
        erode_radius=8,
    )

    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)

    enhanced_green = local_green_enhancement(green)

    if np.any(fov):
        mean_inside = np.mean(enhanced_green[fov])
    else:
        mean_inside = np.mean(enhanced_green)

    binary_for_curvelet = np.ones((size, size), dtype=np.float64)
    binary_for_curvelet[matlab_uint8(enhanced_green) > mean_inside] = 0.0
    binary_for_curvelet[~fov] = 0.0

    curvelet_first = curvelet_greycontrast_step(enhanced_green)
    curvelet_first[~fov] = 0.0

    x2 = matlab_uint8(curvelet_first)

    curvelet_second = curvelet_edge_step(x2.astype(np.float64))

    curvelet_positive = np.where(curvelet_second > 0, curvelet_second, 0.0)
    curvelet_positive[~fov_inner] = 0.0

    if np.max(curvelet_positive) > 0:
        curvelet_positive = curvelet_positive / np.max(curvelet_positive) * 256.0
    else:
        curvelet_positive = np.zeros_like(curvelet_positive)

    vals = curvelet_positive[fov_inner]
    vals = vals[vals > 0]

    if vals.size > 0:
        candidate_thr = np.percentile(vals, 70)
    else:
        candidate_thr = 0.0

    candidate = curvelet_positive > candidate_thr
    candidate[~fov_inner] = False

    candidate_context = binary_dilation(candidate, structure=disk(1))
    candidate_context[~fov_inner] = False

    features = np.stack(
        [
            normalize01(red, fov),
            normalize01(green, fov),
            normalize01(blue, fov),
            normalize01(enhanced_green, fov),
            normalize01(curvelet_first, fov_inner),
            normalize01(curvelet_positive, fov_inner),
            candidate_context.astype(np.float32),
            fov_inner.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    return features, fov, original_shape


def load_model(model_path, device, base_channels=32):
    ckpt = torch.load(model_path, map_location=device)

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        raise KeyError(f"Cannot find model weights. Checkpoint keys: {list(ckpt.keys())}")

    in_channels = ckpt.get("in_channels", 8)
    base_channels = ckpt.get("base_channels", base_channels)

    ModelClass = get_model_classes()

    model = ModelClass(
        in_channels=in_channels,
        base=base_channels,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    print("Loaded model")
    print(f"  model_path: {model_path}")
    print(f"  in_channels: {in_channels}")
    print(f"  base_channels: {base_channels}")

    if "epoch" in ckpt:
        print(f"  epoch: {ckpt['epoch']}")

    if "best_dice" in ckpt:
        print(f"  best_dice: {ckpt['best_dice']}")

    return model


def segment_one_image(args):
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device: {device}")

    features, fov, original_shape = build_features_from_image(
        args.input_image,
        size=args.size,
        roi_path=args.roi,
    )

    model = load_model(
        args.model_path,
        device=device,
        base_channels=args.base_channels,
    )
    expected_channels = next(model.parameters()).shape[1]
    actual_channels = features.shape[-1]
    
    if actual_channels != expected_channels:
        raise ValueError(
            f"Input-channel mismatch: the generated image features have {actual_channels} channels, "
            f"but the loaded model expects {expected_channels} channels. "
            "Please use the same feature-generation pipeline that was used during training."
        )

    x = torch.from_numpy(features.transpose(2, 0, 1)[None]).to(device)

    with torch.no_grad():
        logits = model(x)

        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    pred = prob > args.threshold
    pred[~fov] = False

    output_mask = matlab_uint8(pred.astype(np.uint8) * 255)
    output_prob = matlab_uint8(prob * 255)

    output_mask_resized = resize(
        output_mask,
        original_shape,
        preserve_range=True,
        anti_aliasing=False,
        order=0,
    ).astype(np.uint8)

    output_prob_resized = resize(
        output_prob,
        original_shape,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)

    output_mask_path = Path(args.output_mask)
    ensure_dir(output_mask_path.parent)

    iio.imwrite(output_mask_path, output_mask_resized)

    if args.output_prob is not None:
        output_prob_path = Path(args.output_prob)
        ensure_dir(output_prob_path.parent)
        iio.imwrite(output_prob_path, output_prob_resized)

    if args.output_overlay is not None:
        rgb_original = np.squeeze(iio.imread(args.input_image))

        if rgb_original.ndim == 2:
            rgb_original = np.stack([rgb_original, rgb_original, rgb_original], axis=-1)

        if rgb_original.ndim == 3 and rgb_original.shape[-1] > 3:
            rgb_original = rgb_original[:, :, :3]

        rgb_original = matlab_uint8(rgb_original)

        overlay = rgb_original.copy()
        vessel_pixels = output_mask_resized > 0

        overlay[vessel_pixels, 0] = 255
        overlay[vessel_pixels, 1] = 0
        overlay[vessel_pixels, 2] = 0

        output_overlay_path = Path(args.output_overlay)
        ensure_dir(output_overlay_path.parent)
        iio.imwrite(output_overlay_path, overlay)

    print("Done.")
    print(f"Saved mask: {output_mask_path}")

    if args.output_prob is not None:
        print(f"Saved probability map: {args.output_prob}")

    if args.output_overlay is not None:
        print(f"Saved overlay: {args.output_overlay}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Segment retinal vessels from one input retinal image."
    )

    parser.add_argument(
        "input_image",
        help="Path to input retinal image, for example input.png",
    )

    parser.add_argument(
        "output_mask",
        help="Path to output binary vessel mask, for example output_mask.png",
    )

    parser.add_argument(
    "--model_path",
    default="models/ablation_rgb_fov_scale_directional_last.pt",
    help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--output_prob",
        default=None,
        help="Optional path to save probability map.",
    )

    parser.add_argument(
        "--output_overlay",
        default=None,
        help="Optional path to save vessel overlay on original image.",
    )

    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI/FOV mask path.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Network input size. Use 512 if this was used during training.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary mask.",
    )

    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, or cpu.",
    )

    return parser


def main():
    args = build_parser().parse_args()
    segment_one_image(args)


if __name__ == "__main__":
    main()
