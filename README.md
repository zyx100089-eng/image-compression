# JPEG-Style Image Codec

A from-scratch JPEG-style image codec: 8x8 DCT-II, quantisation, zig-zag ordering, run-length encoding, canonical Huffman coding, and YCbCr 4:2:0 colour, packaged into a bit-level file format. The codec is benchmarked against PIL's JPEG at matched file sizes.

## Background

JPEG is the dominant lossy image format, and its pipeline is well documented but rarely implemented. This project implements the full pipeline in Python without relying on image codecs: the DCT is computed as a matrix product, the Huffman tables are built from the image's own frequency distribution, and the bitstream is written by hand. The goal is to understand each stage — transform, quantisation, entropy coding — and to measure how the result compares against a production implementation.

## Pipeline

The codec (`codec.py`) processes an image in the following stages:

1. Split into 8x8 blocks.
2. Apply the 2D DCT-II (matrix form) to each block.
3. Quantise coefficients using JPEG luma/chroma tables with scaling.
4. Zig-zag order the quantised coefficients.
5. Encode runs as (run, size) symbols with raw amplitude bits, with EOB/ZRL handling.
6. Huffman-encode the symbol stream (canonical codes) into a bitstream (MCJP format).

Colour images are converted RGB → YCbCr (BT.601) with 4:2:0 chroma subsampling. Quality metrics (PSNR, SSIM with Gaussian windows, per-channel for colour) are implemented from scratch in `metrics.py`.

The Huffman coding is near-optimal variable-length coding: the measured average length is within 1 bit of the Shannon lower bound, verified in the test suite alongside prefix-freeness and DCT orthogonality.

## Files

| File | Purpose |
|---|---|
| `dct.py` | 8x8 2D DCT-II (matrix form) and IDCT |
| `quant.py` | Quantisation tables (JPEG luma + chroma, scaling), zig-zag order |
| `huffman.py` | Huffman: frequency table → canonical codes, decode tree, entropy/average-length helpers |
| `codec.py` | Full pipeline: image → blocks → DCT → quant → zig-zag → RLE → Huffman → bitstream (MCJP format); grayscale (H,W) or colour (H,W,3) |
| `metrics.py` | PSNR and SSIM from scratch |
| `benchmark.py` | Rate-distortion comparison vs PIL's JPEG at matching quality |
| `verify.py` | DCT orthogonality, Huffman prefix-freeness + entropy bound, codec bit-exactness, PSNR monotonicity, colour round trips, benchmark sanity |
| `test_codec.py` | Unit test suite (29 tests) |
| `demo.py` | Demonstration of the full pipeline; saves images to `out/` |

## Results (256x256 test images)

Grayscale:

| quality | mine (B) | mine PSNR | JPEG (B) | JPEG PSNR |
|--------:|---------:|----------:|---------:|----------:|
|      20 |     2344 | 42.5 dB   | 2340     | 39.5 dB   |
|      50 |     3079 | 45.4 dB   | 3608     | 44.6 dB   |
|      75 |     3903 | 48.4 dB   | 4875     | 48.4 dB   |
|      95 |     4363 | 50.3 dB   | 10637    | 51.9 dB   |

Colour (YCbCr 4:2:0):

| quality | mine (B) | mine PSNR | JPEG (B) | JPEG PSNR |
|--------:|---------:|----------:|---------:|----------:|
|      20 |     1306 | 38.0 dB   | 1267     | 33.3 dB   |
|      50 |     1692 | 40.8 dB   | 1663     | 38.6 dB   |
|      75 |     2088 | 43.1 dB   | 2105     | 42.0 dB   |
|      95 |     2389 | 44.5 dB   | 3846     | 46.8 dB   |

At typical quality levels the codec matches or exceeds PIL's JPEG in PSNR at equal file size; PIL wins at q=95, where its optimised entropy coder is stronger. On the demo image, 94% of coefficients quantise to zero at q=50 (82% on the noisy benchmark image).

**Fair-comparison caveat:** the codec stores per-image adaptive Huffman tables, while PIL's JPEG uses libjpeg's fixed standard tables (no `optimize=True`). The adaptive tables are part of the codec's design, but a fully optimised JPEG would close part of the gap at high quality.

## Running

```
python3 -m pytest test_codec.py -q   # unit tests
python3 verify.py   # full verification
python3 benchmark.py  # rate-distortion vs JPEG
python3 demo.py       # demonstration
```

## What I Learned

- DCT-II is preferred over the DFT for images because the even-reflection extension avoids boundary discontinuities, which concentrates energy in far fewer coefficients.
- Quantisation is where most of the compression happens; the entropy coder is near-optimal but accounts for a smaller share of the savings.
- Zig-zag ordering matters: it groups large runs of zeros before the entropy stage.
- Comparing fairly against a production codec requires controlling for configuration differences (here, fixed vs adaptive Huffman tables) — otherwise the comparison measures the wrong thing.

## References

- Wallace, *The JPEG Still Picture Compression Standard* (IEEE, 1992)
- Wang, Bovik, Sheikh, Simoncelli, *Image Quality Assessment: From Error Visibility to Structural Similarity* (IEEE TIP, 2004)
