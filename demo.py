"""The personal-statement story for the image compression project.

Walkthrough:
1. Why DCT (and not DFT): basis functions, orthogonality, energy
   compaction - shown numerically.
2. The pipeline: DCT -> quantise -> zig-zag -> Huffman, with file sizes
   and PSNR at each quality level.
3. The magic: most coefficients quantise to zero; count them.
4. Fair comparison against PIL's real JPEG.
5. Why it works: entropy + the rate-distortion trade-off.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from codec import decode_image, encode_image
from dct import dct2
from metrics import psnr, ssim
from quant import quantize, zigzag

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def make_test_image(h: int = 256, w: int = 256) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img = 128 + 60 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    img += 30 * np.tanh((x - 0.5) * 20) * np.tanh((y - 0.5) * 20)
    return np.clip(img, 0, 255)


def save_figure(img: np.ndarray, path: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "L").save(path)
    print(f"  -> {path}")


def demo_dct() -> None:
    print("=" * 70)
    print("1. WHY DCT?")
    print("   A block is a vector; we project it onto cosine basis")
    print("   vectors.  For smooth content, the low-frequency axes")
    print("   carry almost all the energy.")
    print("=" * 70)
    block = np.full((8, 8), 128.0) + 20 * np.sin(np.arange(8) * 0.7)
    F = dct2(block)
    energy = np.cumsum(np.sort((F ** 2).ravel())[::-1])
    top4 = 100 * energy[3] / energy[-1]
    print(f"  8x8 smooth block: the top-4 of 64 coefficients hold "
          f"{top4:.1f}% of the energy")
    print("  (a flat block is a single nonzero coefficient: the DC)")


def demo_pipeline() -> None:
    print("=" * 70)
    print("2. THE PIPELINE")
    print("   image -> 8x8 blocks -> DCT -> quantise -> zig-zag ->")
    print("   run-length + Huffman -> bits")
    print("=" * 70)
    img = make_test_image()
    for q in (20, 50, 75):
        data = encode_image(img, quality=q)
        rec, _ = decode_image(data)
        print(f"  q={q:3d}  {len(data):6d} B  {len(data)*8/img.size:.3f} bpp  "
              f"PSNR {psnr(img, rec):5.1f} dB  SSIM {ssim(img, rec):.3f}")


def demo_zeros() -> None:
    print("=" * 70)
    print("3. THE MAGIC: MOST COEFFICIENTS ARE ZERO")
    print("   After quantisation, high-frequency coefficients round to")
    print("   zero.  Zig-zag ordering turns them into long zero runs;")
    print("   RLE + Huffman then encode those runs in a few bits.")
    print("=" * 70)
    img = make_test_image()
    import codec
    from quant import JPEG_LUMA_Q
    padded, _, _ = codec._pad_to_blocks(img)
    table = codec.quant_table(50)
    total, zeros = 0, 0
    for b in codec._to_blocks(padded):
        zz = zigzag(quantize(dct2(b), table))
        total += 64
        zeros += int((zz == 0).sum())
    print(f"  q=50: {zeros}/{total} coefficients quantise to zero "
          f"({100 * zeros / total:.0f}%)")
    print("  (only the nonzero ones and their run-lengths are coded)")


def demo_vs_jpeg() -> None:
    print("=" * 70)
    print("4. FAIR COMPARISON vs REAL JPEG (PIL)")
    print("   Same image, same quality semantics, same metrics.")
    print("=" * 70)
    import io
    img = make_test_image()
    for q in (20, 50, 75, 95):
        data = encode_image(img, quality=q)
        rec, _ = decode_image(data)
        p_mine = psnr(img, rec)
        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8), "L").save(buf, "JPEG", quality=q)
        jrec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L")).astype(float)
        p_jpeg = psnr(img, jrec)
        print(f"  q={q:3d}  mine {len(data):6d} B {p_mine:5.1f} dB   "
              f"JPEG {len(buf.getvalue()):6d} B {p_jpeg:5.1f} dB")


def demo_entropy() -> None:
    print("=" * 70)
    print("5. WHY IT WORKS: ENTROPY")
    print("   Huffman is within 1 bit of the Shannon lower bound, so")
    print("   the file size tracks the information content of the")
    print("   quantised coefficients - not the pixel count.")
    print("=" * 70)
    img = make_test_image()
    data0 = encode_image(img, quality=20)
    data1 = encode_image(img, quality=95)
    print(f"  q=20: {len(data0)} B   q=95: {len(data1)} B")
    print(f"  (raw grayscale would be {img.size} B - both are far below)")
    print(f"  ratio: {img.size / len(data0):.0f}x smaller at q=20")


def demo_colour() -> None:
    print("=" * 70)
    print("6. COLOUR: YCbCr + 4:2:0 CHROMA SUBSAMPLING")
    print("   RGB -> YCbCr (BT.601), then Cb/Cr are averaged over 2x2")
    print("   blocks: the eye is far less sensitive to colour detail")
    print("   than to luminance, so chroma can spend ~4x fewer bits.")
    print("=" * 70)
    h = w = 128
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img = np.zeros((h, w, 3))
    img[..., 0] = 180 + 50 * np.sin(6 * np.pi * x) * np.cos(4 * np.pi * y)
    img[..., 1] = 100 + 60 * np.sin(4 * np.pi * x + 1.3) * np.cos(3 * np.pi * y)
    img[..., 2] = 200 + 40 * np.tanh((x - 0.5) * 10) * np.tanh((y - 0.5) * 10)
    img = np.clip(img, 0, 255)
    import io
    for q in (20, 50, 75, 95):
        data = encode_image(img, quality=q)
        rec, _ = decode_image(data)
        p_mine = psnr(img, rec)
        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(buf, "JPEG", quality=q)
        jrec = np.asarray(Image.open(io.BytesIO(buf.getvalue()))).astype(float)
        p_jpeg = psnr(img, jrec)
        print(f"  q={q:3d}  mine {len(data):6d} B {p_mine:5.1f} dB   "
              f"JPEG {len(buf.getvalue()):6d} B {p_jpeg:5.1f} dB")
    print("  raw RGB would be", img.size, "B")


def main() -> None:
    demo_dct()
    demo_pipeline()
    demo_zeros()
    demo_vs_jpeg()
    demo_entropy()
    demo_colour()
    # save a comparison image
    img = make_test_image()
    rec, _ = decode_image(encode_image(img, quality=50))
    save_figure(img, os.path.join(OUT_DIR, "original.png"))
    save_figure(rec, os.path.join(OUT_DIR, "q50.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
