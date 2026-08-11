# Image Compression: DCT + Huffman, benchmarked against JPEG

Practical signal-processing project: a from-scratch JPEG-style codec
(8x8 DCT, quantisation, zig-zag, RLE, Huffman, bit-level file format)
with a fair size/quality comparison against real JPEG (PIL).

## The maths

- The 2D Discrete Cosine Transform, and why DCT-II beats DFT for
  images (even-reflection extension, no boundary discontinuities)
- Orthonormal basis of 8x8 blocks, energy compaction
- Quantisation theory: step sizes vs perceptual error
- Entropy: Huffman coding as near-optimal variable-length coding
  (within 1 bit of the Shannon lower bound)
- PSNR / SSIM quality metrics

## The CS

- Bit-level file format design (documented in codec.py)
- Huffman code construction (priority queue, canonical codes)
- JPEG-style RLE: (run, size) symbols + raw amplitude bits, EOB/ZRL
- Colour: RGB -> YCbCr (BT.601) + 4:2:0 chroma subsampling
- Rate-distortion curves

## Files

- `dct.py` - 8x8 2D DCT-II (matrix form), IDCT
- `quant.py` - quantisation tables (JPEG luma + chroma, scaling),
  zig-zag order
- `huffman.py` - Huffman: frequency table -> canonical codes, decode
  tree, entropy/average-length helpers
- `codec.py` - full pipeline: image -> blocks -> DCT -> quant ->
  zig-zag -> (run,size) RLE -> Huffman -> bitstream (MCJP format);
  grayscale (H,W) or colour (H,W,3) via YCbCr 4:2:0
- `metrics.py` - PSNR and SSIM (Wang et al., Gaussian windows; SSIM
  averages per channel for colour) from scratch
- `benchmark.py` - rate-distortion vs PIL's JPEG at matching quality
  (grayscale + colour)
- `verify.py` - DCT orthogonality, Huffman prefix-freeness + entropy
  bound, codec bit-exactness vs the reference pipeline, PSNR
  monotonicity, colour round trips, benchmark sanity
- `test_codec.py` - pytest unit suite (29 tests)
- `demo.py` - the story (energy compaction, zero-coefficient counts,
  JPEG comparison, colour); saves images to `out/`

## Running

```
python3 -m pytest test_codec.py -q   # fast unit tests
python3 verify.py   # full verification
python3 benchmark.py  # rate-distortion vs JPEG
python3 demo.py       # personal-statement walkthrough
```

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

At typical qualities the from-scratch codec matches or beats PIL's
JPEG in PSNR at equal file size (PIL wins at q=95 where its optimized
entropy coder shines). On the demo image, 94% of coefficients
quantise to zero at q=50 (82% on the noisy benchmark image — the
figure is image-dependent).

**Fair-comparison caveat:** the codec stores per-image adaptive
Huffman tables, while PIL's JPEG is called with libjpeg's fixed
standard tables (no `optimize=True`). The adaptive tables are part of
the codec's design, but a fully optimised JPEG would close part of the
gap at high quality.

## References

- Wallace, *The JPEG Still Picture Compression Standard* (IEEE, 1992)
- Wang, Bovik, Sheikh, Simoncelli, *Image Quality Assessment: From
  Error Visibility to Structural Similarity* (IEEE TIP, 2004)
