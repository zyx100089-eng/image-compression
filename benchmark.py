"""Rate-distortion benchmark: my codec vs PIL's JPEG.

For a set of test images, encode with both codecs at several quality
levels and report (bits-per-pixel, PSNR, SSIM).  A codec dominates on
the rate-distortion plane when it achieves a given quality with fewer
bits (or better quality at the same size).

The comparison is deliberately "apples to apples": same image, same
quality parameter semantics (JPEG's quality scaling), same metrics.
Note: PIL's JPEG uses chroma subsampling + optimized Huffman and 8x8
DCT with the industry-tuned standard tables, so it is a strong
baseline; my codec is from scratch with the same table, no
optimisations beyond the essentials.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from codec import decode_image, encode_image
from metrics import psnr, ssim


def test_images() -> list[tuple[str, np.ndarray]]:
    """A few deterministic synthetic images: smooth, edge-heavy, noisy."""
    rng = np.random.default_rng(1)
    h = w = 256
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    imgs = []

    smooth = 128 + 60 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    imgs.append(("smooth", smooth))

    edgey = 128 + 30 * np.tanh((x - 0.5) * 20) * np.tanh((y - 0.5) * 20)
    edgey += 100 * ((x > 0.5) ^ (y > 0.5)).astype(float) * 0.5
    imgs.append(("edges", edgey))

    noisy = 128 + 60 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y) \
        + rng.normal(0, 12, size=(h, w))
    imgs.append(("noisy", noisy))

    # colour: smooth gradients in all three channels (YCbCr 4:2:0 path)
    colour = np.zeros((h, w, 3))
    colour[..., 0] = 180 + 50 * np.sin(6 * np.pi * x) * np.cos(4 * np.pi * y)
    colour[..., 1] = 100 + 60 * np.sin(4 * np.pi * x + 1.3) * np.cos(3 * np.pi * y)
    colour[..., 2] = 200 + 40 * np.tanh((x - 0.5) * 10) * np.tanh((y - 0.5) * 10)
    imgs.append(("colour", np.clip(colour, 0, 255)))
    return imgs


def my_encode(img: np.ndarray, quality: float) -> tuple[bytes, np.ndarray]:
    data = encode_image(img, quality)
    rec, _ = decode_image(data)
    return data, rec


def jpeg_encode(img: np.ndarray, quality: float) -> tuple[bytes, np.ndarray]:
    arr = np.clip(img, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    if arr.ndim == 2:
        Image.fromarray(arr, "L").save(buf, "JPEG", quality=int(quality))
    else:
        Image.fromarray(arr).save(buf, "JPEG", quality=int(quality))
    data = buf.getvalue()
    jimg = np.asarray(Image.open(io.BytesIO(data))).astype(float)
    return data, jimg


def report_row(codec: str, img: np.ndarray, quality: float,
               encoder) -> None:
    data, rec = encoder(img, quality)
    bpp = len(data) * 8 / img.size
    print(f"  {codec:8s} q={quality:5.1f}  {len(data):6d} B  "
          f"{bpp:5.3f} bpp  PSNR {psnr(img, rec):5.1f} dB  "
          f"SSIM {ssim(img, rec):.3f}")


def run() -> None:
    print("Rate-distortion benchmark (256x256 grayscale)")
    for name, img in test_images():
        print(f"\n--- {name} ---")
        print("  my codec:")
        for q in (20, 35, 50, 75, 95):
            report_row("mine", img, q, my_encode)
        print("  PIL JPEG:")
        for q in (20, 35, 50, 75, 95):
            report_row("JPEG", img, q, jpeg_encode)


if __name__ == "__main__":
    run()
