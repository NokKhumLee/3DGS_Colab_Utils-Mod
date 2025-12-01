#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

# Try to import fused_ssim for faster SSIM computation
_FUSED_SSIM_AVAILABLE = False
try:
    from fused_ssim import fused_ssim as _fused_ssim_func
    _FUSED_SSIM_AVAILABLE = True
except ImportError:
    _FUSED_SSIM_AVAILABLE = False

# Global configuration for SSIM implementation
_SSIM_USE_FUSED = None
_SSIM_PADDING = "same"
_SSIM_IMPLEMENTATION_NOTIFIED = False

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _notify_ssim_implementation(use_fused, padding):
    """Print notification about which SSIM implementation is being used."""
    global _SSIM_IMPLEMENTATION_NOTIFIED
    if _SSIM_IMPLEMENTATION_NOTIFIED:
        return
    
    if use_fused and _FUSED_SSIM_AVAILABLE:
        print(f"[SSIM] Using fused_ssim implementation (5-8x faster) with padding='{padding}'")
    elif use_fused and not _FUSED_SSIM_AVAILABLE:
        print(f"[SSIM] WARNING: fused_ssim requested but not available. Falling back to PyTorch SSIM.")
        print(f"[SSIM] Install with: pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation")
    else:
        print(f"[SSIM] Using PyTorch SSIM implementation (original)")
    
    _SSIM_IMPLEMENTATION_NOTIFIED = True

def configure_ssim(use_fused=None, padding="same"):
    """
    Configure SSIM implementation.
    
    Args:
        use_fused: If True, use fused_ssim when available. If False, use PyTorch SSIM.
                   If None, auto-detect (use fused_ssim if available).
        padding: Padding mode for fused_ssim. Options: "same" (default, matches original) or "valid"
    """
    global _SSIM_USE_FUSED, _SSIM_PADDING, _SSIM_IMPLEMENTATION_NOTIFIED
    
    if use_fused is None:
        # Auto-detect: use fused if available
        _SSIM_USE_FUSED = _FUSED_SSIM_AVAILABLE
    else:
        _SSIM_USE_FUSED = use_fused and _FUSED_SSIM_AVAILABLE
    
    _SSIM_PADDING = padding
    _SSIM_IMPLEMENTATION_NOTIFIED = False  # Reset to notify again
    _notify_ssim_implementation(_SSIM_USE_FUSED, _SSIM_PADDING)

def ssim(img1, img2, window_size=11, size_average=True, use_fused=None, padding=None):
    """
    Compute SSIM between two images.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        window_size: Window size (only used for PyTorch SSIM, fused_ssim uses fixed 11x11)
        size_average: If True, return mean SSIM. If False, return per-image SSIM.
        use_fused: Override global config. If None, uses global configuration.
        padding: Override global padding config. If None, uses global configuration.
    
    Returns:
        SSIM similarity value (0-1 range, higher is better)
    """
    global _SSIM_USE_FUSED, _SSIM_PADDING
    
    # Determine which implementation to use
    if use_fused is None:
        use_fused_impl = _SSIM_USE_FUSED if _SSIM_USE_FUSED is not None else _FUSED_SSIM_AVAILABLE
    else:
        use_fused_impl = use_fused and _FUSED_SSIM_AVAILABLE
    
    padding_mode = padding if padding is not None else (_SSIM_PADDING if _SSIM_PADDING else "same")
    
    # Notify on first call
    if _SSIM_USE_FUSED is None:
        configure_ssim(use_fused=use_fused_impl, padding=padding_mode)
        use_fused_impl = _SSIM_USE_FUSED
    
    # Use fused_ssim if available and requested
    if use_fused_impl:
        # fused_ssim expects images in [0, 1] range and returns mean by default
        ssim_value = _fused_ssim_func(img1, img2, padding=padding_mode, train=True)
        
        if not size_average:
            # If size_average=False, we need per-image SSIM
            # fused_ssim always returns mean, so we need to compute per-image
            # This is a limitation - for per-image, fall back to PyTorch
            channel = img1.size(-3)
            window = create_window(window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            return _ssim(img1, img2, window, window_size, channel, size_average)
        
        return ssim_value
    
    # Fall back to original PyTorch implementation
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

