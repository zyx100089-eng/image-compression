"""Image quality metrics: PSNR and SSIM, both from scratch.

PSNR = 10 log10(255^2 / MSE) — pure pixel error, easy to compute,
widely used, but blind to structure.

SSIM (Wang et al. 2004) compares *local* luminance, contrast and
structure over sliding windows:

    SSIM(x, y) = (2 mu_x mu_y + c1)(2 sigma_xy + c2)
                / ((mu_x^2 + mu_y^2 + c1)(sigma_x^2 + sigma_y^2 + c2))

with c1 = (k1 L)^2, c2 = (k2 L)^2, L = 255, k1 = 0.01, k2 = 0.03.
Implemented with a Gaussian window via separable convolutions.
"""

from __future__ import annotations

import numpy as np


def psnr(original: np.ndarray, distorted: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB.  Images in 0..255."""
    mse = float(np.mean((original - distorted) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - size // 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _conv2d_separable(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable 2D convolution (kernel outer product applied in two
    passes), 'same' size with edge-reflection padding.

    Convolving along rows then along columns of a padded image and
    cropping back to the original shape is exactly the 2D convolution
    with the outer-product kernel - but costs O(N*k) instead of
    O(N*k^2)."""
    pad = len(kernel) // 2
    padded = np.pad(img, pad, mode="edge")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"),
                              1, padded)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"),
                              0, tmp)
    return out[pad:-pad, pad:-pad]


def _conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Direct 2D convolution with 'same' padding (edge reflection)."""
    pad = len(kernel) // 2
    padded = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img)
    k = kernel
    for i in range(-pad, pad + 1):
        for j in range(-pad, pad + 1):
            out += k[i + pad, j + pad] * padded[pad + i:pad + i + img.shape[0],
                                                pad + j:pad + j + img.shape[1]]
    return out


def ssim(original: np.ndarray, distorted: np.ndarray,
         window_size: int = 11, sigma: float = 1.5) -> float:
    """Mean structural similarity (Wang et al.).  Images in 0..255.

    For 3-channel (RGB) images, the per-channel SSIM is averaged —
    this matches how colour SSIM is usually reported."""
    if original.ndim == 3 and original.shape[2] == 3:
        vals = [ssim(original[..., k], distorted[..., k],
                     window_size, sigma) for k in range(3)]
        return float(np.mean(vals))
    L = 255.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    kernel = _gaussian_kernel(window_size, sigma)
    k2d = kernel[:, None] * kernel[None, :]

    mu_x = _conv2d(original, k2d)
    mu_y = _conv2d(distorted, k2d)

    mu_x2, mu_y2 = mu_x * mu_x, mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _conv2d(original * original, k2d) - mu_x2
    sigma_y2 = _conv2d(distorted * distorted, k2d) - mu_y2
    sigma_xy = _conv2d(original * distorted, k2d) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return float(ssim_map.mean())
