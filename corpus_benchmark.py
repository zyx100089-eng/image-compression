"""Real-photo corpus benchmark: my codec vs PIL's JPEG.

The synthetic images in benchmark.py are deterministic and fast, but
they are not photographs. This script runs the same rate-distortion
comparison on real photographs (scikit-image's bundled dataset:
astronaut, coffee, rocket, cat, hubble deep field), downscaled to
256x256 so the pure-Python codec stays fast.

Usage:
    pip install scikit-image
    python3 corpus_benchmark.py

Results are printed and saved to results/corpus_benchmark.csv.
"""

from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from codec import decode_image, encode_image
from metrics import psnr, ssim

QUALITIES = (20, 50, 75, 95)


def corpus_images() -> list[tuple[str, np.ndarray]]:
    from skimage import data
    imgs = []
    for name in ("astronaut", "coffee", "rocket", "cat", "hubble_deep_field"):
        arr = getattr(data, name)()
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.asarray(arr, dtype=np.float64)
        # downscale to 256x256 (PIL LANCZOS) to keep the pure-Python
        # codec's runtime acceptable
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        pil = pil.resize((256, 256), Image.LANCZOS)
        imgs.append((name, np.asarray(pil).astype(np.float64)))
    return imgs


def my_encode(img: np.ndarray, quality: float) -> tuple[bytes, np.ndarray]:
    data = encode_image(img, quality)
    rec, _ = decode_image(data)
    return data, rec


def jpeg_encode(img: np.ndarray, quality: float) -> tuple[bytes, np.ndarray]:
    arr = np.clip(img, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=int(quality))
    data = buf.getvalue()
    jimg = np.asarray(Image.open(io.BytesIO(data))).astype(float)
    return data, jimg


def run() -> None:
    rows = []
    for name, img in corpus_images():
        print(f"\n--- {name} (256x256 colour) ---")
        for q in QUALITIES:
            t0 = time.time()
            my_data, my_rec = my_encode(img, q)
            t_mine = time.time() - t0
            jp_data, jp_rec = jpeg_encode(img, q)
            my_bpp = len(my_data) * 8 / img.size
            jp_bpp = len(jp_data) * 8 / img.size
            my_p = psnr(img, my_rec)
            jp_p = psnr(img, jp_rec)
            my_s = ssim(img, my_rec)
            jp_s = ssim(img, jp_rec)
            print(f"  q={q:3d}  mine {my_bpp:5.3f} bpp {my_p:5.1f} dB "
                  f"SSIM {my_s:.3f} ({t_mine:.1f}s) | "
                  f"JPEG {jp_bpp:5.3f} bpp {jp_p:5.1f} dB SSIM {jp_s:.3f}")
            rows.append({
                "image": name, "quality": q,
                "mine_bpp": round(my_bpp, 4), "mine_psnr": round(my_p, 2),
                "mine_ssim": round(my_s, 3),
                "jpeg_bpp": round(jp_bpp, 4), "jpeg_psnr": round(jp_p, 2),
                "jpeg_ssim": round(jp_s, 3),
            })

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "corpus_benchmark.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {path}")

    # summary: mean PSNR at matched bpp (nearest quality)
    print("\nsummary (mean PSNR over corpus, per quality):")
    for q in QUALITIES:
        mine = [r["mine_psnr"] for r in rows if r["quality"] == q]
        jpeg = [r["jpeg_psnr"] for r in rows if r["quality"] == q]
        print(f"  q={q:3d}  mine {np.mean(mine):5.1f} dB   "
              f"JPEG {np.mean(jpeg):5.1f} dB")


if __name__ == "__main__":
    run()
