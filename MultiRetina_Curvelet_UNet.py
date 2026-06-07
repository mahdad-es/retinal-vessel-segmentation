#!/usr/bin/env python3
import argparse, csv, random, re
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F

import numpy as np
import imageio.v3 as iio
from scipy.ndimage import (
    binary_opening, binary_closing, binary_dilation,
    label as ndi_label, gaussian_filter
)
from skimage.transform import resize
from skimage.morphology import disk, skeletonize

def ensure_dir(path):
    path = Path(path)
    if str(path) not in ["", "."]:
        path.mkdir(parents=True, exist_ok=True)


def matlab_uint8(x):
    x = np.asarray(x)
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


def normalize01(x, mask=None, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)

    if mask is not None and np.any(mask):
        vals = x[mask]
    else:
        vals = x.reshape(-1)

    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)

    if abs(hi - lo) < eps:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1).astype(np.float32)


def save_png(path, img):
    ensure_dir(Path(path).parent)
    img = np.asarray(img)

    if img.dtype == bool:
        out = img.astype(np.uint8) * 255
    else:
        out = matlab_uint8(normalize01(img) * 255)

    iio.imwrite(path, out)


def make_fdct_operator(shape):
    import curvelops as cl

    return cl.FDCT2D(
        dims=shape,
        nbscales=6,
        nbangles_coarse=32,
        allcurvelets=False,
        dtype="complex128",
    )


# =============================================================================
# Classical curvelet helper functions
# These are included here so this repository does NOT depend on
# CurveletGuidedDRIVE_UNet.py.
# =============================================================================

def local_green_enhancement(green):
    """
    Local green-channel enhancement used by the earlier curvelet pipeline.
    This is kept for compatibility with vessel_seg.py.
    """
    from scipy.ndimage import uniform_filter, maximum_filter

    green = green.astype(np.float64)
    m, n = green.shape

    padded = np.zeros((m + 10, n + 10), dtype=np.float64)
    padded[4:m + 4, 4:n + 4] = green

    mean9 = uniform_filter(padded, size=9, mode="constant", cval=0.0)
    max9 = maximum_filter(padded, size=9, mode="constant", cval=0.0)

    enhanced = (
        5.0 * padded[4:m + 4, 4:n + 4]
        - 4.0 * mean9[4:m + 4, 4:n + 4]
        + (80.0 * mean9[4:m + 4, 4:n + 4])
        / (max9[4:m + 4, 4:n + 4] + 1.0)
    )

    return enhanced


def greycontrast8(D, m, st):
    """
    Coefficient contrast operation adapted from the MATLAB/curvelet pipeline.
    The real part is processed and returned as complex128 for FDCT compatibility.
    """
    D = np.asarray(D).copy()
    realD = np.real(D).copy()

    M = np.max(realD)
    p = 0.2
    spo = 0.3
    no = 0.1 * M

    if abs(no) < 1e-12:
        return D

    out = realD.copy()

    mask1 = realD < 0.5 * no
    out[mask1] = 2.0 * realD[mask1]

    mask2 = (0.5 * no < realD) & (realD < 3.0 * no)
    if np.any(mask2):
        s = (((realD[mask2] - no) / no) * ((m / no) ** p)) + (
            (2.0 * no - realD[mask2]) / no
        )
        out[mask2] = np.abs(realD[mask2]) * s

    mask3 = (3.0 * no < realD) & (realD < 4.0 * no)
    if np.any(mask3):
        denom = realD[mask3]
        safe = np.abs(denom) > 1e-12
        tmp = out[mask3]
        tmp[safe] = ((m / denom[safe]) ** p) * np.abs(denom[safe])
        out[mask3] = tmp

    mask4 = realD > 4.0 * no
    if np.any(mask4):
        denom = realD[mask4]
        safe = np.abs(denom) > 1e-12
        tmp = out[mask4]
        tmp[safe] = ((m / denom[safe]) ** spo) * denom[safe]
        out[mask4] = tmp

    return out.astype(np.complex128)


def mim_eslah_zarayeb(D, sa=2):
    """
    Placeholder for the MATLAB coefficient-correction step.
    In the currently used MATLAB version this step returns the coefficients.
    """
    return D


def edgenhance(D):
    """
    Placeholder for the MATLAB edge-enhancement step.
    In the currently used MATLAB version this step suppresses low-scale coefficients.
    """
    return np.zeros_like(D)


def inverse_fdct(fdct, coeff_struct, shape):
    coeff_vector = fdct.vect(coeff_struct)
    rec = fdct.H @ coeff_vector
    return np.real(np.asarray(rec).reshape(shape))


def curvelet_greycontrast_step(img):
    """
    First curvelet enhancement step used by the single-image inference script.
    """
    img = img.astype(np.float64)
    shape = img.shape

    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img_n = (img - p1) / (p99 - p1)
        img_n = np.clip(img_n, 0.0, 1.0) * 255.0
    else:
        img_n = img.copy()

    st = np.std(img_n)
    M = np.percentile(img_n, 99)
    m = 0.1 * M

    fdct = make_fdct_operator(shape)
    coeff = fdct @ img_n.astype(np.complex128)
    coeff_struct = fdct.struct(np.asarray(coeff).ravel())

    weights = [0.0, 0.05, 0.6, 1.0, 0.7, 0.3]

    new_struct = []
    for s, scale in enumerate(coeff_struct):
        w = weights[s] if s < len(weights) else 0.3
        new_scale = []
        for wedge in scale:
            new_scale.append(w * greycontrast8(wedge, m, st))
        new_struct.append(new_scale)

    return inverse_fdct(fdct, new_struct, shape)


def curvelet_edge_step(img):
    """
    Second curvelet enhancement step used by the single-image inference script.
    """
    img = img.astype(np.float64)
    shape = img.shape

    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img_n = (img - p1) / (p99 - p1)
        img_n = np.clip(img_n, 0.0, 1.0) * 255.0
    else:
        img_n = img.copy()

    fdct = make_fdct_operator(shape)
    coeff = fdct @ img_n.astype(np.complex128)
    coeff_struct = fdct.struct(np.asarray(coeff).ravel())

    weights = [0.0, 0.15, 0.8, 1.0, 0.8, 0.4]

    new_struct = []
    for s, scale in enumerate(coeff_struct):
        w = weights[s] if s < len(weights) else 0.4
        new_scale = []
        for wedge in scale:
            if s in [0, 1]:
                processed = edgenhance(wedge)
            else:
                processed = mim_eslah_zarayeb(wedge, 1.5)
            new_scale.append(w * processed)
        new_struct.append(new_scale)

    return inverse_fdct(fdct, new_struct, shape)


# =============================================================================
# PyTorch model and loss
# Included here so no external DRIVE-specific script is required.
# =============================================================================

def get_model_classes():
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self, in_channels=8, base=32):
            super().__init__()

            self.enc1 = DoubleConv(in_channels, base)
            self.enc2 = DoubleConv(base, base * 2)
            self.enc3 = DoubleConv(base * 2, base * 4)
            self.enc4 = DoubleConv(base * 4, base * 8)

            self.pool = nn.MaxPool2d(2)

            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(base * 8, base * 4)

            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(base * 4, base * 2)

            self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(base * 2, base)

            self.out = nn.Conv2d(base, 1, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            d3 = self.up3(e4)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))

            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))

            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))

            return self.out(d1)

    return SmallUNet


def dice_loss_from_logits(logits, target, eps=1e-6):
    import torch

    prob = torch.sigmoid(logits)
    prob = prob.reshape(prob.size(0), -1)
    target = target.reshape(target.size(0), -1)

    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)

    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()

def safe_id(s):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s)).strip("_")
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true or false")

def read_manifest(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "sample_id": safe_id(r["sample_id"]),
                "dataset": r["dataset"].upper(),
                "image_path": r["image_path"],
                "mask_path": r["mask_path"],
                "roi_path": r.get("roi_path", ""),
                "split": r["split"].lower(),
            })
    return rows
def scnp_binary_logits(logits, y, kernel_size=3, big=9999.0):
    """
    Binary SCNP for one-channel vessel segmentation.

    logits: [B, 1, H, W], raw model output
    y:      [B, 1, H, W], binary target {0,1}

    For vessel pixels y=1:
        use local minimum among vessel-neighbor logits.
    For background pixels y=0:
        use local maximum among background-neighbor logits.
    """
    pad = kernel_size // 2
    y = y.float()

    # foreground: min-pooling over foreground pixels
    fg_min = -F.max_pool2d(
        -(logits * y + big * (1.0 - y)),
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )

    # background: max-pooling over background pixels
    bg_max = F.max_pool2d(
        logits * (1.0 - y) - big * y,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )

    scnp_logits = fg_min * y + bg_max * (1.0 - y)
    return scnp_logits
def feature_file(features_dir, sample_id):
    return Path(features_dir) / f"{safe_id(sample_id)}_features.npz"

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
    img = resize(img, (size, size, 3), preserve_range=True, anti_aliasing=True)
    return matlab_uint8(img)

def read_binary_mask_general(path, size, invert_if_huge=True):
    m = np.squeeze(iio.imread(path))
    if m.ndim == 3:
        if m.shape[-1] == 4:
            # RGBA mask: ignore alpha channel
            rgb = m[..., :3]
    
            # Use RGB channels only
            m_rgb = rgb.max(axis=-1)
    
            # If RGB is empty, fallback to alpha
            if np.max(m_rgb) == 0:
                m = m[..., 3]
            else:
                m = m_rgb
    
        elif m.shape[-1] == 3:
            # RGB mask
            m = m.max(axis=-1)
    
        else:
            # multi-frame mask
            m = m.max(axis=0)
    if m.ndim != 2:
        raise ValueError(f"Cannot read mask {path}, shape={m.shape}")
    m_bin = m > 0
    
    # If foreground is too large, try inversion.
    # But choose the one with a plausible vessel ratio.
    if invert_if_huge:
        r1 = m_bin.mean()
        r2 = (~m_bin).mean()
    
        if r1 > 0.50 and 0.01 <= r2 <= 0.30:
            m_bin = ~m_bin
    out = resize(m_bin.astype(np.uint8), (size, size), preserve_range=True, anti_aliasing=False, order=0)
    return (out > 0).astype(np.uint8)
#######################
def curvelet_energy_feature_maps(img, fov, n_dir_groups=4, skip_coarse=True):
    """
    Create curvelet coefficient-domain feature maps.

    Outputs:
        scale_maps: one energy map per curvelet scale
        dir_maps: directional-group energy maps aggregated across scales

    Each coefficient wedge is converted to magnitude-squared energy,
    resized to the image size, and accumulated.
    """
    img = img.astype(np.float64)
    h, w = img.shape

    # Robust normalization before curvelet transform
    p1, p99 = np.percentile(img[fov], [1, 99]) if np.any(fov) else np.percentile(img, [1, 99])
    if p99 > p1:
        img_n = (img - p1) / (p99 - p1)
        img_n = np.clip(img_n, 0.0, 1.0) * 255.0
    else:
        img_n = img.copy()

    # Avoid a hard zero boundary outside FOV before FDCT
    fill_value = np.median(img_n[fov]) if np.any(fov) else np.median(img_n)
    img_n = img_n.copy()
    img_n[~fov] = fill_value

    fdct = make_fdct_operator(img_n.shape)
    coeff = fdct @ img_n.astype(np.complex128)
    coeff_struct = fdct.struct(np.asarray(coeff).ravel())

    scale_maps = []
    dir_maps = [np.zeros((h, w), dtype=np.float64) for _ in range(n_dir_groups)]

    for s, scale in enumerate(coeff_struct):
        if skip_coarse and s == 0:
            continue

        scale_energy = np.zeros((h, w), dtype=np.float64)
        n_wedges = max(len(scale), 1)

        for d, wedge in enumerate(scale):
            arr = np.asarray(wedge)

            # Magnitude-squared curvelet coefficient energy
            energy = np.abs(arr) ** 2

            # Make sure resize receives at least 2-D input
            if energy.ndim == 0:
                energy = energy.reshape(1, 1)
            elif energy.ndim == 1:
                energy = energy.reshape(-1, 1)

            energy_resized = resize(
                energy,
                (h, w),
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.float64)

            scale_energy += energy_resized

            # Group wedge directions into n_dir_groups directional groups.
            # This is a direction-index grouping, not a hard anatomical angle label.
            g = int(np.floor(d * n_dir_groups / n_wedges))
            g = min(g, n_dir_groups - 1)
            dir_maps[g] += energy_resized

        scale_energy[~fov] = 0.0
        scale_maps.append(normalize01(scale_energy, fov))

    dir_maps_out = []
    for dm in dir_maps:
        dm[~fov] = 0.0
        dir_maps_out.append(normalize01(dm, fov))

    return scale_maps, dir_maps_out

####################
def enhance_green_gaussian_bg(green, sigma=15):
    """
    Gaussian background-subtraction enhancement.
    A smooth illumination/background image is estimated by Gaussian filtering
    and subtracted from the green channel. The result is rescaled to [0, 255].
    """
    green = green.astype(np.float64)

    background = gaussian_filter(green, sigma=sigma)
    enhanced = green - background

    e_min, e_max = np.min(enhanced), np.max(enhanced)
    if e_max > e_min:
        enhanced = (enhanced - e_min) / (e_max - e_min)
    else:
        enhanced = np.zeros_like(enhanced)

    return enhanced.astype(np.float64) * 255.0


##############
def make_fov_from_rgb(rgb):
    green = rgb[:, :, 1].astype(np.float32)

    fov = green > 10
    fov = binary_opening(fov, structure=np.ones((5, 5), dtype=bool))
    fov = binary_closing(fov, structure=np.ones((25, 25), dtype=bool))

    lab, num = ndi_label(fov)
    if num > 0:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        fov = lab == np.argmax(counts)

    return fov.astype(bool)

def make_fov(rgb, roi_path, size):
    roi_path = str(roi_path).strip()

    if roi_path and Path(roi_path).exists():
        fov = read_binary_mask_general(
            roi_path,
            size=size,
            invert_if_huge=False
        ).astype(bool)

        lab, num = ndi_label(fov)
        if num > 0:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            fov = lab == np.argmax(counts)

        return fov.astype(bool)

    return make_fov_from_rgb(rgb)

def extract_one(
    row,
    features_dir,
    debug_dir=None,
    size=512,
    use_rgb=True,
    use_gaussian=False,
    use_scale_energy=False,
    use_directional_energy=False,
    use_fov=True,
):
    sample_id = safe_id(row["sample_id"])
    dataset = row["dataset"].upper()

    rgb = read_rgb_general(row["image_path"], size=size)
    target = read_binary_mask_general(row["mask_path"], size=size, invert_if_huge=True)

    # Use only the full field-of-view mask.
    fov = make_fov(rgb, roi_path=row.get("roi_path", ""), size=size)

    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)

    # Use only Gaussian background-subtraction enhancement.
    enhanced_green = enhance_green_gaussian_bg(green, sigma=15)


    # Direct curvelet coefficient-domain features
    if use_scale_energy or use_directional_energy:
        scale_energy_maps, directional_energy_maps = curvelet_energy_feature_maps(
            enhanced_green,
            fov=fov,
            n_dir_groups=4,
            skip_coarse=True
        )
    else:
        scale_energy_maps, directional_energy_maps = [], []
    
    target = target.astype(np.uint8)
    target[~fov] = 0
    
    feature_list = []
    channel_names = []
    
    if use_rgb:
        feature_list.extend([
            normalize01(red, fov),
            normalize01(green, fov),
            normalize01(blue, fov),
        ])
        channel_names.extend(["red", "green", "blue"])
    
    if use_gaussian:
        feature_list.append(normalize01(enhanced_green, fov))
        channel_names.append("gaussian_enhanced_green")
    
    if use_scale_energy:
        for i, emap in enumerate(scale_energy_maps):
            feature_list.append(emap.astype(np.float32))
            channel_names.append(f"scale_energy_s{i+1}")
    
    if use_directional_energy:
        for j, dmap in enumerate(directional_energy_maps):
            feature_list.append(dmap.astype(np.float32))
            channel_names.append(f"directional_energy_g{j+1}")
    
    if use_fov:
        feature_list.append(fov.astype(np.float32))
        channel_names.append("fov")
    
    if len(feature_list) == 0:
        raise ValueError("No input channels selected. At least one feature group must be enabled.")
    
    features = np.stack(feature_list, axis=-1).astype(np.float32)
    
    print(f"[FEATURES] {sample_id}: {features.shape[-1]} channels -> {channel_names}")

    out_path = feature_file(features_dir, sample_id)
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        features=features,
        target=target,
        fov=fov.astype(np.uint8),
        sample_id=sample_id,
        dataset=dataset,
        split=row["split"],
        image_path=row["image_path"],
        mask_path=row["mask_path"],
        roi_path=row.get("roi_path", ""),
        green_enhance="gaussian_background_subtraction",
        fov_strategy="full_fov",
        curvelet_representation="coefficient_energy_maps",
        use_rgb=use_rgb,
        use_gaussian=use_gaussian,
        use_scale_energy=use_scale_energy,
        use_directional_energy=use_directional_energy,
        use_fov=use_fov,
        n_scale_energy_maps=len(scale_energy_maps),
        n_directional_energy_maps=len(directional_energy_maps),
        n_input_channels=features.shape[-1],
        channel_names=np.array(channel_names)
    )

    if debug_dir is not None:
        ds_debug = Path(debug_dir) / dataset
        ensure_dir(ds_debug)
    
        # ---------------------------------------------------------
        # Save only the feature channels actually used by the model
        # ---------------------------------------------------------
        for c, name in enumerate(channel_names):
            save_png(
                ds_debug / f"{sample_id}_ch{c+1:02d}_{name}.png",
                features[:, :, c]
            )
    
        # ---------------------------------------------------------
        # Save combined scale-energy map if scale maps were computed
        # Useful for paper visualization
        # ---------------------------------------------------------
        if len(scale_energy_maps) > 0:
            combined_scale_energy = np.mean(
                np.stack(scale_energy_maps, axis=-1),
                axis=-1
            )
            save_png(
                ds_debug / f"{sample_id}_combined_scale_energy.png",
                combined_scale_energy
            )
    
            # Also save each scale map separately with clear names
            for i, emap in enumerate(scale_energy_maps):
                save_png(
                    ds_debug / f"{sample_id}_scale_energy_s{i+1}.png",
                    emap
                )
    
        # ---------------------------------------------------------
        # Save combined directional-energy map if directional maps exist
        # Useful for paper visualization
        # ---------------------------------------------------------
        if len(directional_energy_maps) > 0:
            combined_directional_energy = np.mean(
                np.stack(directional_energy_maps, axis=-1),
                axis=-1
            )
            save_png(
                ds_debug / f"{sample_id}_combined_directional_energy.png",
                combined_directional_energy
            )
    
            # Also save each directional group separately
            for j, dmap in enumerate(directional_energy_maps):
                save_png(
                    ds_debug / f"{sample_id}_directional_energy_g{j+1}.png",
                    dmap
                )
    
        # ---------------------------------------------------------
        # Save FOV and ground-truth mask for checking
        # ---------------------------------------------------------
        save_png(
            ds_debug / f"{sample_id}_fov.png",
            fov.astype(np.float32)
        )
    
        save_png(
            ds_debug / f"{sample_id}_gt.png",
            target.astype(np.float32) * 255
        )
    
        # ---------------------------------------------------------
        # Save a text file describing the channel order
        # ---------------------------------------------------------
        with open(ds_debug / f"{sample_id}_channel_order.txt", "w") as f:
            f.write(f"Sample ID: {sample_id}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Number of input channels: {features.shape[-1]}\n\n")
            f.write("Input channel order:\n")
            for c, name in enumerate(channel_names):
                f.write(f"Channel {c+1:02d}: {name}\n")

    ratio = float(target[fov].mean()) if np.any(fov) else 0.0
    print(f"[EXTRACT] {dataset} {sample_id} ratio={ratio:.4f} -> {out_path}")
    return out_path

def mode_extract(args):
    rows = read_manifest(args.manifest)
    if args.extract_split != "all":
        rows = [r for r in rows if r["split"] == args.extract_split]
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    debug_dir = args.debug_dir if args.save_debug else None
    print(f"Extracting {len(rows)} samples")
    for i, row in enumerate(rows, 1):
        print(f"Sample {i}/{len(rows)}")
        extract_one(
            row,
            args.features_dir,
            debug_dir=debug_dir,
            size=args.size,
            use_rgb=args.use_rgb,
            use_gaussian=args.use_gaussian,
            use_scale_energy=args.use_scale_energy,
            use_directional_energy=args.use_directional_energy,
            use_fov=args.use_fov,
        )

def rows_to_files(rows, features_dir):
    files = []
    for r in rows:
        p = feature_file(features_dir, r["sample_id"])
        if p.exists():
            files.append(p)
        else:
            print(f"WARNING missing feature file: {p}")
    return files

def compute_pos_weight(files):
    pos, total = 0, 0
    for p in files:
        d = np.load(p, allow_pickle=True)
        y = d["target"].astype(np.uint8)
        fov = d["fov"].astype(bool)
        pos += int(y[fov].sum())
        total += int(fov.sum())
    pos = max(pos, 1)
    neg = max(total - pos, 1)
    return float(min(max(neg / pos, 1.0), 30.0))
##################
def safe_div(a, b, eps=1e-6):
    return float(a) / float(b + eps)


def binary_skeleton(mask):
    """
    Compute 2-D skeleton of a binary vessel mask.
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    return skeletonize(mask).astype(bool)


def skeleton_dice_score(pred, target, fov):
    """
    Dice score between skeletonized prediction and skeletonized ground truth.
    This measures centerline/geometric agreement.
    """
    pred = pred.astype(bool) & fov.astype(bool)
    target = target.astype(bool) & fov.astype(bool)

    sk_pred = binary_skeleton(pred)
    sk_target = binary_skeleton(target)

    inter = np.logical_and(sk_pred, sk_target).sum()
    denom = sk_pred.sum() + sk_target.sum()

    return safe_div(2.0 * inter, denom)


def cldice_score(pred, target, fov):
    """
    Centerline Dice, commonly used to evaluate tubular structures.

    tprec = fraction of predicted skeleton lying inside the target mask
    tsens = fraction of target skeleton recovered by the prediction mask
    clDice = harmonic mean of tprec and tsens
    """
    pred = pred.astype(bool) & fov.astype(bool)
    target = target.astype(bool) & fov.astype(bool)

    sk_pred = binary_skeleton(pred)
    sk_target = binary_skeleton(target)

    tprec = safe_div(np.logical_and(sk_pred, target).sum(), sk_pred.sum())
    tsens = safe_div(np.logical_and(sk_target, pred).sum(), sk_target.sum())

    return safe_div(2.0 * tprec * tsens, tprec + tsens)


def remove_small_components(mask, min_size=20):
    """
    Remove tiny connected components from a binary mask.
    """
    mask = mask.astype(bool)
    lab, num = ndi_label(mask)

    if num == 0:
        return mask

    counts = np.bincount(lab.ravel())
    keep = np.zeros(num + 1, dtype=bool)

    for i in range(1, num + 1):
        if counts[i] >= min_size:
            keep[i] = True

    return keep[lab]


def component_count_error(pred, target, fov, min_size=20):
    """
    Difference in number of connected vessel components after removing tiny components.
    Lower is better.
    """
    pred = pred.astype(bool) & fov.astype(bool)
    target = target.astype(bool) & fov.astype(bool)

    pred = remove_small_components(pred, min_size=min_size)
    target = remove_small_components(target, min_size=min_size)

    _, n_pred = ndi_label(pred)
    _, n_target = ndi_label(target)

    return abs(int(n_pred) - int(n_target))
###############
def mode_train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class PatchDataset(Dataset):
        def __init__(
            self,
            files,
            selected_channels,
            patch_size=128,
            patches_per_image=40,
            augment=True,
            balanced_by_dataset=False,
            samples_per_epoch=None,
        ):
            self.files = list(files)
            if len(self.files) == 0:
                raise RuntimeError("No training feature files found")
    
            self.selected_channels = selected_channels
            self.patch_size = patch_size
            self.patches_per_image = patches_per_image
            self.augment = augment
            self.balanced_by_dataset = balanced_by_dataset
    
            # Read dataset name from each npz file
            self.by_dataset = defaultdict(list)
            for p in self.files:
                d = np.load(p, allow_pickle=True)
                ds = str(d["dataset"])
                self.by_dataset[ds].append(p)
    
            self.dataset_names = sorted(self.by_dataset.keys())
    
            print("Training files by dataset:")
            for ds in self.dataset_names:
                print(f"  {ds}: {len(self.by_dataset[ds])} files")
    
            if samples_per_epoch is None:
                self.samples_per_epoch = len(self.files) * self.patches_per_image
            else:
                self.samples_per_epoch = samples_per_epoch
    
        def __len__(self):
            return self.samples_per_epoch
    
        def _choose_file(self, index):
            if self.balanced_by_dataset:
                # Choose dataset uniformly
                ds = self.dataset_names[index % len(self.dataset_names)]
    
                # Then choose one image randomly from that dataset
                return random.choice(self.by_dataset[ds])
    
            # Original behavior: image-balanced, not dataset-balanced
            file_index = (index // self.patches_per_image) % len(self.files)
            return self.files[file_index]
    
        def __getitem__(self, index):
            p = self._choose_file(index)
    
            d = np.load(p, allow_pickle=True)
            x = d["features"].astype(np.float32)[:, :, self.selected_channels]
            y = d["target"].astype(np.float32)
            fov = d["fov"].astype(bool)
    
            h, w, _ = x.shape
            ps = self.patch_size
    
            for _ in range(30):
                r = random.randint(0, h - ps)
                c = random.randint(0, w - ps)
    
                yp = y[r:r+ps, c:c+ps]
                fp = fov[r:r+ps, c:c+ps]
    
                if fp.mean() > 0.5 and yp.sum() > 10:
                    break
            else:
                r = random.randint(0, h - ps)
                c = random.randint(0, w - ps)
    
            x = x[r:r+ps, c:c+ps, :]
            y = y[r:r+ps, c:c+ps]
    
            if self.augment:
                if random.random() < 0.5:
                    x = np.flip(x, axis=1).copy()
                    y = np.flip(y, axis=1).copy()
    
                if random.random() < 0.5:
                    x = np.flip(x, axis=0).copy()
                    y = np.flip(y, axis=0).copy()
    
                k = random.randint(0, 3)
                if k:
                    x = np.rot90(x, k, axes=(0, 1)).copy()
                    y = np.rot90(y, k, axes=(0, 1)).copy()
    
            import torch
            return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(y[None, :, :])

    
    class FullDataset(Dataset):
        def __init__(self, files, selected_channels):
            self.files = list(files)
            if len(self.files) == 0:
                raise RuntimeError("No validation feature files found")
            self.selected_channels = selected_channels

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            import torch
            p = self.files[index]
            d = np.load(p, allow_pickle=True)
            x = d["features"].astype(np.float32)[:, :, self.selected_channels]
            y = d["target"].astype(np.float32)
            fov = d["fov"].astype(np.float32)
            return (
                torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy(y[None, :, :]),
                torch.from_numpy(fov[None, :, :]),
                str(d["sample_id"]),
                str(d["dataset"]),
            )

    
    @torch.no_grad()
    def evaluate(model, loader, device, pred_dir=None):
        model.eval()
        overall = []
        by_dataset = defaultdict(list)
    
        if pred_dir is not None:
            ensure_dir(pred_dir)
    
        for x, y, fov, sample_id, dataset in loader:
            x, y, fov = x.to(device), y.to(device), fov.to(device)
    
            prob = torch.sigmoid(model(x))
            pred = (prob > args.threshold).float() * fov
    
            for b in range(x.size(0)):
                p = pred[b, 0]
                t = y[b, 0]
                m = fov[b, 0] > 0.5
    
                p_m = p[m]
                t_m = t[m]
    
                tp = ((p_m == 1) & (t_m == 1)).sum().item()
                fp = ((p_m == 1) & (t_m == 0)).sum().item()
                fn = ((p_m == 0) & (t_m == 1)).sum().item()
                tn = ((p_m == 0) & (t_m == 0)).sum().item()
    
                # Convert to numpy for geometric/topological metrics
                p_np = p.detach().cpu().numpy().astype(bool)
                t_np = t.detach().cpu().numpy().astype(bool)
                fov_np = m.detach().cpu().numpy().astype(bool)
    
                met = {
                    "dice": safe_div(2 * tp, 2 * tp + fp + fn),
                    "iou": safe_div(tp, tp + fp + fn),
                    "sensitivity": safe_div(tp, tp + fn),
                    "specificity": safe_div(tn, tn + fp),
                
                    # Geometric/topological metrics
                    "skeleton_dice": skeleton_dice_score(p_np, t_np, fov_np),
                    "cldice": cldice_score(p_np, t_np, fov_np),
                
                    # Component error after removing very small noisy components
                    "component_error": component_count_error(
                        p_np,
                        t_np,
                        fov_np,
                        min_size=20
                    ),
                }
    
                ds = str(dataset[b])
                sid = str(sample_id[b])
    
                overall.append(met)
                by_dataset[ds].append(met)
    
                if pred_dir is not None:
                    ds_dir = Path(pred_dir) / ds
                    ensure_dir(ds_dir)
    
                    prob_np = prob[b, 0].detach().cpu().numpy()
                    pred_np = pred[b, 0].detach().cpu().numpy()
    
                    iio.imwrite(ds_dir / f"{sid}_prob.png", matlab_uint8(prob_np * 255))
                    iio.imwrite(ds_dir / f"{sid}_mask.png", matlab_uint8(pred_np.astype(np.uint8) * 255))
    
        metric_keys = [
            "dice",
            "iou",
            "sensitivity",
            "specificity",
            "skeleton_dice",
            "cldice",
            "component_error",
        ]
    
        def mean(items, key):
            return float(np.mean([m[key] for m in items])) if items else 0.0
    
        out = {
            "overall": {},
            "by_dataset": {},
            "macro": {},
        }
    
        # Overall image-wise average.
        # This is dominated by datasets with many validation images, especially FIVES.
        for k in metric_keys:
            out["overall"][k] = mean(overall, k)
    
        # Dataset-wise average
        for ds, items in sorted(by_dataset.items()):
            out["by_dataset"][ds] = {k: mean(items, k) for k in metric_keys}
            out["by_dataset"][ds]["n"] = len(items)
    
        # Macro average across datasets.
        # Each dataset contributes equally, regardless of number of validation images.
        for k in metric_keys:
            vals = [out["by_dataset"][ds][k] for ds in out["by_dataset"]]
            out["macro"][k] = float(np.mean(vals)) if vals else 0.0
    
        return out

    rows = read_manifest(args.manifest)
    train_files = rows_to_files([r for r in rows if r["split"] == "train"], args.features_dir)
    val_files = rows_to_files([r for r in rows if r["split"] == "val"], args.features_dir)
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

    first = np.load(train_files[0], allow_pickle=True)
    total_channels = first["features"].shape[-1]
    
    selected_channels, selected_channel_names = get_selected_channel_indices(args, total_channels)
    in_channels = len(selected_channels)
    
    print(f"Total channels in npz: {total_channels}")
    print(f"Selected channel indices: {selected_channels}")
    print(f"Selected channel names: {selected_channel_names}")
    print(f"Training input channels: {in_channels}")
    
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    SmallUNet = get_model_classes()
    model = SmallUNet(in_channels=in_channels, base=args.base_channels).to(device)
    train_loader = DataLoader(
        PatchDataset(
            train_files,
            selected_channels,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            augment=True,
            balanced_by_dataset=args.balanced_by_dataset,
            samples_per_epoch=args.samples_per_epoch if args.samples_per_epoch > 0 else None,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        FullDataset(val_files, selected_channels),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    pos_weight = compute_pos_weight(train_files)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_model_path = Path(args.model_path)
    last_model_path = best_model_path.with_name(best_model_path.stem + "_last" + best_model_path.suffix)
    
    ensure_dir(best_model_path.parent)

    print(f"Device: {device}")
    print(f"Input channels: {in_channels}")
    print(f"Positive-class BCE weight: {pos_weight:.3f}")

    best_dice = -1.0
    start_epoch = 1
    
    if args.resume and last_model_path.exists():
        print(f"Resuming from LAST checkpoint: {last_model_path}")
        ckpt = torch.load(last_model_path, map_location=device)
    
        model.load_state_dict(ckpt["model_state"])
    
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
    
        best_dice = ckpt.get("best_dice", -1.0)
        start_epoch = ckpt.get("epoch", 0) + 1
    
        print(f"Resume start epoch: {start_epoch}")
        print(f"Previous best dice: {best_dice:.4f}")
    
    elif args.resume:
        print(f"WARNING: --resume was used, but last checkpoint was not found: {last_model_path}")
        print("Starting training from scratch.")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ---------------------------------------------------------
            # Forward pass
            # ---------------------------------------------------------
            logits = model(x)

            # ---------------------------------------------------------
            # Normal loss on original logits
            # ---------------------------------------------------------
            normal_bce_loss = bce(logits, y)
            normal_dice_loss = dice_loss_from_logits(logits, y)

            normal_loss = 0.5 * normal_bce_loss + 0.5 * normal_dice_loss

            # ---------------------------------------------------------
            # SCNP loss
            # SCNP is used only during training.
            # It uses ground-truth y, so never use this in validation/test.
            # ---------------------------------------------------------
            if args.use_scnp:
                scnp_logits = scnp_binary_logits(
                    logits,
                    y,
                    kernel_size=args.scnp_kernel
                )

                scnp_bce_loss = bce(scnp_logits, y)
                scnp_dice_loss = dice_loss_from_logits(scnp_logits, y)

                scnp_loss = 0.5 * scnp_bce_loss + 0.5 * scnp_dice_loss

                # Combined loss:
                # scnp_weight=0.5 means 50% normal loss + 50% SCNP loss
                loss = (
                    (1.0 - args.scnp_weight) * normal_loss
                    + args.scnp_weight * scnp_loss
                )

            else:
                scnp_loss = torch.tensor(0.0, device=device)
                loss = normal_loss

            # ---------------------------------------------------------
            # Backpropagation
            # ---------------------------------------------------------
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        metrics = evaluate(model, val_loader, device, pred_dir=None)
        o = metrics["overall"]
        ma = metrics["macro"]
        
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={np.mean(losses):.5f} | "
              f"val_dice={o['dice']:.4f} | val_iou={o['iou']:.4f} | "
              f"val_sens={o['sensitivity']:.4f} | val_spec={o['specificity']:.4f} | "
              f"macro_dice={ma['dice']:.4f} | macro_iou={ma['iou']:.4f} | "
              f"macro_cldice={ma['cldice']:.4f} | macro_skel_dice={ma['skeleton_dice']:.4f}")
        for ds, m in metrics["by_dataset"].items():
            print(
                f"  {ds} n={m['n']} "
                f"dice={m['dice']:.4f} "
                f"iou={m['iou']:.4f} "
                f"sens={m['sensitivity']:.4f} "
                f"spec={m['specificity']:.4f} "
                f"cldice={m['cldice']:.4f} "
                f"skel_dice={m['skeleton_dice']:.4f} "
                f"comp_err={m['component_error']:.2f}"
            )
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "in_channels": in_channels,
                "base_channels": args.base_channels,
                "best_dice": best_dice,
                "epoch": epoch,
                "selected_channels": selected_channels,
                "selected_channel_names": selected_channel_names,
                "use_rgb": args.use_rgb,
                "use_gaussian": args.use_gaussian,
                "use_scale_energy": args.use_scale_energy,
                "use_directional_energy": args.use_directional_energy,
                "use_fov": args.use_fov,                
                                
            },
            last_model_path,
        )
        print(f"Saved last checkpoint: {last_model_path}")

        score_for_best = ma["dice"]
        
        if score_for_best > best_dice:
            best_dice = score_for_best
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "in_channels": in_channels,
                    "base_channels": args.base_channels,
                    "best_dice": best_dice,
                    "epoch": epoch,
                    "selected_channels": selected_channels,
                    "selected_channel_names": selected_channel_names,
                    "use_rgb": args.use_rgb,
                    "use_gaussian": args.use_gaussian,
                    "use_scale_energy": args.use_scale_energy,
                    "use_directional_energy": args.use_directional_energy,
                    "use_fov": args.use_fov,                    
                    
                },
                best_model_path,
            )
            print(f"Saved best model: {best_model_path}")

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_metrics = evaluate(model, val_loader, device, pred_dir=args.pred_dir)
    print("Final validation metrics:", final_metrics)
#######################
def get_selected_channel_indices(args, total_channels):
    """
    Select channels from the full 14-channel feature tensor.

    Expected full channel order:
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
    """
    selected = []
    names = []

    if args.use_rgb:
        selected.extend([0, 1, 2])
        names.extend(["red", "green", "blue"])

    if args.use_gaussian:
        selected.append(3)
        names.append("gaussian_enhanced_green")

    if args.use_scale_energy:
        selected.extend([4, 5, 6, 7, 8])
        names.extend([
            "scale_energy_s1",
            "scale_energy_s2",
            "scale_energy_s3",
            "scale_energy_s4",
            "scale_energy_s5",
        ])

    if args.use_directional_energy:
        selected.extend([9, 10, 11, 12])
        names.extend([
            "directional_energy_g1",
            "directional_energy_g2",
            "directional_energy_g3",
            "directional_energy_g4",
        ])

    if args.use_fov:
        selected.append(13)
        names.append("fov")

    if len(selected) == 0:
        raise ValueError("No channels selected. Enable at least one feature group.")

    if max(selected) >= total_channels:
        raise ValueError(
            f"Selected channel index {max(selected)}, but feature tensor has only {total_channels} channels."
        )

    return selected, names
############

def mode_predict(args):
    import torch

    rows = [r for r in read_manifest(args.manifest) if r["split"] == args.predict_split]
    files = rows_to_files(rows, args.features_dir)

    if len(files) == 0:
        raise RuntimeError(f"No feature files found for split: {args.predict_split}")

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(args.model_path, map_location=device)

    # ---------------------------------------------------------
    # Use exactly the same input channels used during training
    # ---------------------------------------------------------
    if "selected_channels" in ckpt:
        selected_channels = ckpt["selected_channels"]
        print("Using selected channels from checkpoint:", selected_channels)

        if "selected_channel_names" in ckpt:
            print("Selected channel names:", ckpt["selected_channel_names"])
    else:
        first = np.load(files[0], allow_pickle=True)
        total_channels = first["features"].shape[-1]
        selected_channels, selected_channel_names = get_selected_channel_indices(args, total_channels)

        print("WARNING: checkpoint does not contain selected_channels.")
        print("Using selected channels from args:", selected_channels)
        print("Selected channel names:", selected_channel_names)

    # ---------------------------------------------------------
    # Build and load model
    # ---------------------------------------------------------
    SmallUNet = get_model_classes()

    model = SmallUNet(
        in_channels=ckpt["in_channels"],
        base=ckpt.get("base_channels", args.base_channels)
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ensure_dir(args.pred_dir)

    overall = []
    by_dataset = defaultdict(list)

    # Optional CSV file for saving per-image metrics
    metrics_csv_path = Path(args.pred_dir) / f"{args.predict_split}_metrics.csv"

    with open(metrics_csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)

        writer.writerow([
            "dataset",
            "sample_id",
            "dice",
            "iou",
            "sensitivity",
            "specificity",
            "cldice",
            "skeleton_dice",
            "component_error"
        ])

        with torch.no_grad():
            for p in files:
                d = np.load(p, allow_pickle=True)

                x = d["features"].astype(np.float32)[:, :, selected_channels]
                y = d["target"].astype(np.uint8)
                fov = d["fov"].astype(bool)

                sid = str(d["sample_id"])
                ds = str(d["dataset"])

                xt = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(device)

                # ---------------------------------------------------------
                # IMPORTANT:
                # SCNP is NOT used during prediction/test.
                # SCNP needs ground truth and must only be used in training.
                # ---------------------------------------------------------
                logits = model(xt)
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

                pred = prob > args.threshold
                pred[~fov] = False

                # ---------------------------------------------------------
                # Save probability map and binary mask
                # ---------------------------------------------------------
                ds_dir = Path(args.pred_dir) / ds
                ensure_dir(ds_dir)

                iio.imwrite(
                    ds_dir / f"{sid}_prob.png",
                    matlab_uint8(prob * 255)
                )

                iio.imwrite(
                    ds_dir / f"{sid}_mask.png",
                    matlab_uint8(pred.astype(np.uint8) * 255)
                )

                # ---------------------------------------------------------
                # Compute pixel metrics inside FOV only
                # ---------------------------------------------------------
                p_m = pred[fov].astype(bool)
                t_m = y[fov].astype(bool)

                tp = np.logical_and(p_m == 1, t_m == 1).sum()
                fp = np.logical_and(p_m == 1, t_m == 0).sum()
                fn = np.logical_and(p_m == 0, t_m == 1).sum()
                tn = np.logical_and(p_m == 0, t_m == 0).sum()

                met = {
                    "dice": safe_div(2 * tp, 2 * tp + fp + fn),
                    "iou": safe_div(tp, tp + fp + fn),
                    "sensitivity": safe_div(tp, tp + fn),
                    "specificity": safe_div(tn, tn + fp),

                    # Topology / vessel-continuity metrics
                    "skeleton_dice": skeleton_dice_score(pred, y, fov),
                    "cldice": cldice_score(pred, y, fov),
                    "component_error": component_count_error(
                        pred,
                        y,
                        fov,
                        min_size=20
                    ),
                }

                overall.append(met)
                by_dataset[ds].append(met)

                writer.writerow([
                    ds,
                    sid,
                    met["dice"],
                    met["iou"],
                    met["sensitivity"],
                    met["specificity"],
                    met["cldice"],
                    met["skeleton_dice"],
                    met["component_error"],
                ])

                print(
                    f"Saved {ds}/{sid} | "
                    f"dice={met['dice']:.4f} | "
                    f"iou={met['iou']:.4f} | "
                    f"sens={met['sensitivity']:.4f} | "
                    f"spec={met['specificity']:.4f} | "
                    f"cldice={met['cldice']:.4f} | "
                    f"skel_dice={met['skeleton_dice']:.4f} | "
                    f"comp_err={met['component_error']:.2f}"
                )

    metric_keys = [
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "skeleton_dice",
        "cldice",
        "component_error",
    ]

    def mean(items, key):
        return float(np.mean([m[key] for m in items])) if items else 0.0

    print("\n==============================")
    print(f"Final {args.predict_split.upper()} metrics")
    print("==============================")

    # ---------------------------------------------------------
    # Overall image-wise average
    # ---------------------------------------------------------
    overall_metrics = {k: mean(overall, k) for k in metric_keys}

    print(
        f"Overall | "
        f"dice={overall_metrics['dice']:.4f} | "
        f"iou={overall_metrics['iou']:.4f} | "
        f"sens={overall_metrics['sensitivity']:.4f} | "
        f"spec={overall_metrics['specificity']:.4f} | "
        f"cldice={overall_metrics['cldice']:.4f} | "
        f"skel_dice={overall_metrics['skeleton_dice']:.4f} | "
        f"comp_err={overall_metrics['component_error']:.2f}"
    )

    # ---------------------------------------------------------
    # Dataset-wise average
    # ---------------------------------------------------------
    dataset_metrics = {}

    for ds, items in sorted(by_dataset.items()):
        dataset_metrics[ds] = {k: mean(items, k) for k in metric_keys}
        dataset_metrics[ds]["n"] = len(items)

        m = dataset_metrics[ds]

        print(
            f"{ds} n={m['n']} | "
            f"dice={m['dice']:.4f} | "
            f"iou={m['iou']:.4f} | "
            f"sens={m['sensitivity']:.4f} | "
            f"spec={m['specificity']:.4f} | "
            f"cldice={m['cldice']:.4f} | "
            f"skel_dice={m['skeleton_dice']:.4f} | "
            f"comp_err={m['component_error']:.2f}"
        )

    # ---------------------------------------------------------
    # Macro average across datasets
    # Each dataset contributes equally
    # ---------------------------------------------------------
    macro_metrics = {}

    for k in metric_keys:
        vals = [dataset_metrics[ds][k] for ds in dataset_metrics]
        macro_metrics[k] = float(np.mean(vals)) if vals else 0.0

    print(
        f"Macro | "
        f"dice={macro_metrics['dice']:.4f} | "
        f"iou={macro_metrics['iou']:.4f} | "
        f"sens={macro_metrics['sensitivity']:.4f} | "
        f"spec={macro_metrics['specificity']:.4f} | "
        f"cldice={macro_metrics['cldice']:.4f} | "
        f"skel_dice={macro_metrics['skeleton_dice']:.4f} | "
        f"comp_err={macro_metrics['component_error']:.2f}"
    )

    print(f"\nSaved per-image metrics to: {metrics_csv_path}")


def mode_check(args):
    rows = read_manifest(args.manifest)
    stats = defaultdict(list)

    for r in rows:
        p = feature_file(args.features_dir, r["sample_id"])

        if not p.exists():
            continue

        d = np.load(p, allow_pickle=True)

        y = d["target"].astype(np.uint8)
        fov = d["fov"].astype(bool)

        ratio = float(y[fov].mean()) if fov.sum() > 0 else 0.0
        stats[str(d["dataset"])].append(ratio)

    for ds, vals in sorted(stats.items()):
        print(
            ds,
            "n=", len(vals),
            "min=", min(vals),
            "mean=", sum(vals) / len(vals),
            "max=", max(vals)
        )


def build_parser():
    p = argparse.ArgumentParser(
        description="Manifest-based multi-retina curvelet-guided U-Net."
    )

    # ---------------------------------------------------------
    # General mode and paths
    # ---------------------------------------------------------
    p.add_argument(
        "--mode",
        choices=["extract", "train", "predict", "check"],
        required=True
    )

    p.add_argument("--manifest", required=True)

    p.add_argument(
        "--features_dir",
        default="features"
    )

    p.add_argument(
        "--debug_dir",
        default="debug"
    )

    p.add_argument("--save_debug", action="store_true")

    p.add_argument(
        "--extract_split",
        choices=["all", "train", "val", "test"],
        default="all"
    )

    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--size", type=int, default=512)

    p.add_argument(
        "--model_path",
        default="models/ablation_rgb_fov_scale_directional_last.pt"
    )

    p.add_argument(
        "--pred_dir",
        default="predictions"
    )

    p.add_argument(
        "--predict_split",
        choices=["train", "val", "test"],
        default="test"
    )

    # ---------------------------------------------------------
    # Training settings
    # ---------------------------------------------------------
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--patches_per_image", type=int, default=40)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cuda', or 'cpu'."
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary prediction."
    )

    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the *_last.pt checkpoint if it exists."
    )

    # ---------------------------------------------------------
    # SCNP settings
    # Used only in training, never in prediction/test.
    # ---------------------------------------------------------
    p.add_argument(
        "--use_scnp",
        type=str2bool,
        default=True,
        help="Use SCNP loss during training."
    )

    p.add_argument(
        "--scnp_kernel",
        type=int,
        default=3,
        help="SCNP neighborhood kernel size. Recommended first value: 3."
    )

    p.add_argument(
        "--scnp_weight",
        type=float,
        default=0.5,
        help="Weight of SCNP loss. 0.5 means half normal loss and half SCNP loss."
    )

    # ---------------------------------------------------------
    # Input channel selection
    # ---------------------------------------------------------
    p.add_argument("--use_rgb", type=str2bool, default=True)
    p.add_argument("--use_gaussian", type=str2bool, default=False)
    p.add_argument("--use_scale_energy", type=str2bool, default=False)
    p.add_argument("--use_directional_energy", type=str2bool, default=False)
    p.add_argument("--use_fov", type=str2bool, default=True)

    # ---------------------------------------------------------
    # Dataset balancing
    # ---------------------------------------------------------
    p.add_argument(
        "--balanced_by_dataset",
        type=str2bool,
        default=False,
        help="If true, sample training patches equally from each dataset."
    )

    p.add_argument(
        "--samples_per_epoch",
        type=int,
        default=0,
        help="Number of training patches per epoch. If 0, use len(train_files) * patches_per_image."
    )

    return p

def main():
    args = build_parser().parse_args()
    if args.mode == "extract":
        mode_extract(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "predict":
        mode_predict(args)
    elif args.mode == "check":
        mode_check(args)

if __name__ == "__main__":
    main()
