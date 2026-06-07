#!/usr/bin/env python3
"""
vessel_seg.py

Single-image inference script for curvelet-guided retinal blood vessel segmentation.

This script is designed for the checkpoint:
    models/ablation_rgb_fov_scale_directional_last.pt

It generates the same full 14-channel feature tensor used during training:
    0  red
    1  green
    2  blue
    3  gaussian_enhanced_green
    4  scale_energy_s1
    5  scale_energy_s2
    6  scale_energy_s3
    7  scale_energy_s4
    8  scale_energy_s5
    9  directional_energy_g1
    10 directional_energy_g2
    11 directional_energy_g3
    12 directional_energy_g4
    13 fov

Then it selects the exact channels stored inside the checkpoint.
"""

import argparse
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import torch
from skimage.transform import resize

from MultiRetina_Curvelet_UNet import (
    ensure_dir,
    matlab_uint8,
    normalize01,
    read_binary_mask_general,
    make_fov,
    curvelet_energy_feature_maps,
    enhance_green_gaussian_bg,
    get_model_classes,
)


FULL_CHANNEL_NAMES = [
    "red",
    "green",
    "blue",
    "gaussian_enhanced_green",
    "scale_energy_s1",
    "scale_energy_s2",
    "scale_energy_s3",
    "scale_energy_s4",
    "scale_energy_s5",
    "directional_energy_g1",
    "directional_energy_g2",
    "directional_energy_g3",
    "directional_energy_g4",
    "fov",
]


def read_rgb_with_original_shape(path, size):
    """Read an image as RGB, resize for the network, and keep original H,W."""
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

    img_resized = resize(
        img,
        (size, size, 3),
        preserve_range=True,
        anti_aliasing=True,
    )

    return matlab_uint8(img_resized), original_shape, matlab_uint8(img)


def build_full_feature_tensor(image_path, size=512, roi_path=None):
    """
    Build the full 14-channel feature tensor used by the multi-retina training code.
    Returns:
        features_full: [H, W, 14]
        fov:           [H, W] boolean mask in resized/network space
        original_shape: original image shape before resizing
        rgb_original:  original RGB image for overlay output
    """
    rgb, original_shape, rgb_original = read_rgb_with_original_shape(image_path, size=size)

    # Same FOV strategy as training: ROI if provided, otherwise estimated from RGB.
    fov = make_fov(rgb, roi_path=roi_path, size=size)

    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)

    # Same enhancement used for scale/directional energy maps in training.
    enhanced_green = enhance_green_gaussian_bg(green, sigma=15)

    scale_energy_maps, directional_energy_maps = curvelet_energy_feature_maps(
        enhanced_green,
        fov=fov,
        n_dir_groups=4,
        skip_coarse=True,
    )

    # The training setup expects 5 scale maps and 4 directional maps.
    if len(scale_energy_maps) != 5:
        raise RuntimeError(
            f"Expected 5 scale-energy maps, but got {len(scale_energy_maps)}. "
            "Check curvelops/FDCT settings in MultiRetina_Curvelet_UNet.py."
        )

    if len(directional_energy_maps) != 4:
        raise RuntimeError(
            f"Expected 4 directional-energy maps, but got {len(directional_energy_maps)}."
        )

    feature_list = [
        normalize01(red, fov),
        normalize01(green, fov),
        normalize01(blue, fov),
        normalize01(enhanced_green, fov),
    ]

    feature_list.extend([m.astype(np.float32) for m in scale_energy_maps])
    feature_list.extend([m.astype(np.float32) for m in directional_energy_maps])
    feature_list.append(fov.astype(np.float32))

    features_full = np.stack(feature_list, axis=-1).astype(np.float32)

    if features_full.shape[-1] != 14:
        raise RuntimeError(f"Expected 14 full channels, but got {features_full.shape[-1]}")

    return features_full, fov, original_shape, rgb_original


def load_checkpoint_and_model(model_path, device, base_channels=32):
    ckpt = torch.load(model_path, map_location=device)

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        raise KeyError(f"Cannot find model weights. Checkpoint keys: {list(ckpt.keys())}")

    in_channels = int(ckpt.get("in_channels", 13))
    base_channels = int(ckpt.get("base_channels", base_channels))

    ModelClass = get_model_classes()
    model = ModelClass(in_channels=in_channels, base=base_channels).to(device)
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

    selected_channels = ckpt.get("selected_channels", None)
    selected_channel_names = ckpt.get("selected_channel_names", None)

    if selected_channels is None:
        # Fallback for rgb_fov_scale_directional model: RGB + scale + directional + FOV, no Gaussian.
        selected_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        selected_channel_names = [FULL_CHANNEL_NAMES[i] for i in selected_channels]
        print("WARNING: checkpoint does not contain selected_channels.")
        print("Using default RGB + scale-energy + directional-energy + FOV channels.")
    else:
        selected_channels = [int(i) for i in selected_channels]
        if selected_channel_names is None:
            selected_channel_names = [FULL_CHANNEL_NAMES[i] for i in selected_channels]
        else:
            selected_channel_names = [str(x) for x in selected_channel_names]

    print(f"  selected_channels: {selected_channels}")
    print(f"  selected_channel_names: {selected_channel_names}")

    if len(selected_channels) != in_channels:
        raise ValueError(
            f"Checkpoint expects {in_channels} input channels, but selected_channels has "
            f"{len(selected_channels)} entries: {selected_channels}"
        )

    return ckpt, model, selected_channels


def save_resized_outputs(prob, pred, fov, original_shape, rgb_original, args):
    pred = pred.astype(bool)
    prob = prob.astype(np.float32)

    pred[~fov] = False
    prob_masked = prob.copy()
    prob_masked[~fov] = 0.0

    output_mask = matlab_uint8(pred.astype(np.uint8) * 255)
    output_prob = matlab_uint8(prob_masked * 255)

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
        if rgb_original.ndim == 2:
            rgb_original = np.stack([rgb_original, rgb_original, rgb_original], axis=-1)
        if rgb_original.shape[-1] > 3:
            rgb_original = rgb_original[:, :, :3]

        overlay = matlab_uint8(rgb_original).copy()
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


def segment_one_image(args):
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device: {device}")

    ckpt, model, selected_channels = load_checkpoint_and_model(
        args.model_path,
        device=device,
        base_channels=args.base_channels,
    )

    features_full, fov, original_shape, rgb_original = build_full_feature_tensor(
        args.input_image,
        size=args.size,
        roi_path=args.roi,
    )

    if max(selected_channels) >= features_full.shape[-1]:
        raise ValueError(
            f"Selected channel index {max(selected_channels)} is outside the full feature tensor "
            f"with {features_full.shape[-1]} channels."
        )

    features = features_full[:, :, selected_channels]

    expected_channels = int(ckpt.get("in_channels", features.shape[-1]))
    actual_channels = features.shape[-1]

    if actual_channels != expected_channels:
        raise ValueError(
            f"Input-channel mismatch: selected features have {actual_channels} channels, "
            f"but the loaded model expects {expected_channels} channels."
        )

    x = torch.from_numpy(features.transpose(2, 0, 1)[None]).to(device)

    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    pred = prob > args.threshold
    save_resized_outputs(prob, pred, fov, original_shape, rgb_original, args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Segment retinal vessels from one colour fundus image using a trained curvelet-guided U-Net."
    )

    parser.add_argument(
        "input_image",
        help="Path to input retinal image, for example examples/test_01_test.tif",
    )

    parser.add_argument(
        "output_mask",
        help="Path to output binary vessel mask, for example outputs/test_01_vessel_mask.png",
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
