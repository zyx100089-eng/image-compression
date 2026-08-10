# Image Compression from Scratch: a JPEG-style Codec

*A report for admissions — written as if for a short undergraduate project write-up.*

---

## Summary

I implemented a complete JPEG-style image codec from first principles:
the 8×8 Discrete Cosine Transform, quantisation, zig-zag ordering,
run-length coding, and Huffman entropy coding — all in my own documented
bit-level file format, with no image-compression library involved. I
then benchmarked it honestly against real JPEG (PIL/libjpeg) on
rate-distortion.

The project's central claims, each verified by its test suite:

- The codec is **bit-exact**: decoding reproduces the reference
  DCT→quantise→IDCT pipeline to machine precision, across sizes and
  qualities.
- On a smooth 8×8 block, the **top 4 of 64 DCT coefficients hold
  ~100% of the energy**; at quality 50, **94% of coefficients
  quantise to zero** — the two facts that make the whole pipeline
  work.
- At typical qualities (20–75) the from-scratch codec **matches or
  beats PIL's real JPEG in PSNR at equal file size**, on both
  grayscale and colour images.
- Colour is handled via YCbCr with 4:2:0 chroma subsampling — the
  eye's lower sensitivity to colour detail is exploited to spend
  ~4× fewer bits on chroma.

---

## 1. Problem statement

Compression is the art of *deciding what to throw away*. A photograph
is millions of numbers; storing them raw costs 8 bits per pixel per
channel. JPEG achieves ~10–30× reduction with barely visible loss, and
it does so with a pipeline that decomposes cleanly into mathematics
(the transform), psychophysics (what error the eye tolerates), and
computer science (how to encode the survivors efficiently). The goal
of this project was to build that pipeline myself — every bit of the
file format, every line of the entropy coder — and then measure whether
a from-scratch implementation can compete with thirty years of
industrial tuning.

## 2. Method

### 2.1 The transform: DCT-II, not DFT

Each 8×8 block is treated as a vector and projected onto a set of
cosine basis vectors. The 2D DCT-II is separable:

```
F = C f Cᵀ,   C[u][x] = a_u cos(π(2x+1)u/16),   a₀ = 1/√8, a_u = 1/2
```

C is orthonormal (verified: C Cᵀ = I to 1e-12), so the inverse is
exactly f = Cᵀ F C and the round trip is lossless to 1e-10.

Why DCT-II rather than the more obvious DFT? The DFT implicitly treats
the block as one period of an infinite signal, so a bright edge on one
side of a block meets a dark edge on the other — creating artificial
high frequencies. DCT-II's cosines have half-sample offsets, which
corresponds to reflecting the block at its boundary instead of
wrapping it; there is no discontinuity, so the energy compacts better.
This is exactly why JPEG chose it.

### 2.2 Quantisation: the only lossy step

Each coefficient is divided by a step size and rounded:
`q[i][j] = round(F[i][j] / Q[i][j])`. The table Q uses small steps for
low frequencies and large steps for high ones, because the eye
tolerates more error at high spatial frequencies. This rounding is the
**only** loss in the entire codec — the decoder is bit-exact
afterwards. The error per coefficient is bounded by half its step size
(verified).

### 2.3 Zig-zag and run-length coding

Quantisation makes most high-frequency coefficients zero. Zig-zag
ordering walks the 8×8 block along diagonals, grouping similar
frequencies so the zeros become long runs. Each nonzero AC coefficient
is then coded as a `(run, size)` symbol — how many zeros precede it and
how many bits its magnitude needs — followed by the raw amplitude
bits. An end-of-block marker terminates the tail of zeros with a
single symbol. DC coefficients are coded as differences from the
previous block (adjacent blocks have similar averages), so they
cluster near zero.

### 2.4 Huffman entropy coding

The symbol frequencies are measured and a Huffman code built with a
priority queue: repeatedly merge the two rarest subtrees. The result
is converted to canonical codes (stored as just the code lengths),
which are prefix-free and decodable left-to-right with no ambiguity.
Huffman's average code length is within 1 bit of the Shannon entropy
(verified on real streams, e.g. 3.000 vs 2.999 bits/symbol). The
tables are stored per image, which beats JPEG's fixed standard tables.

### 2.5 Colour: YCbCr + 4:2:0

For RGB images, the colour is converted to YCbCr (BT.601), separating
luminance from colour. The two chroma planes are then averaged over
2×2 blocks (4:2:0 subsampling) — the eye's low sensitivity to colour
detail means the chroma can be represented at quarter resolution with
no visible effect. Each plane is coded with its own Huffman tables and
a coarser quantisation table for chroma.

## 3. The bit-level format

The file format is documented in the code: magic `MCJP`, dimensions,
quality, channel count, then per channel the two Huffman tables
(symbol, code-length pairs), the DC symbol stream and its raw
amplitude bits, then the AC stream. Big-endian fixed-width fields,
deliberately simple — the point was to understand every byte, and to
make the decoder auditable.

## 4. Results

### 4.1 Grayscale vs PIL's JPEG

256×256 synthetic test image (smooth gradients + sharp edges):

| q | mine (B) | mine PSNR | JPEG (B) | JPEG PSNR |
|---|---------:|----------:|---------:|----------:|
| 20 | 2344 | 42.5 dB | 2340 | 39.5 dB |
| 50 | 3079 | 45.4 dB | 3608 | 44.6 dB |
| 75 | 3903 | 48.4 dB | 4875 | 48.4 dB |
| 95 | 4363 | 50.3 dB | 10637 | 51.9 dB |

At q ≤ 75 the from-scratch codec matches or beats libjpeg in PSNR at
equal or smaller file size; at q = 95 libjpeg's optimized entropy coder
wins the PSNR race (51.9 vs 50.3 dB) but needs 2.4× the bytes.

### 4.2 Colour vs PIL's JPEG

| q | mine (B) | mine PSNR | JPEG (B) | JPEG PSNR |
|---|---------:|----------:|---------:|----------:|
| 20 | 1306 | 38.0 dB | 1267 | 33.3 dB |
| 50 | 1692 | 40.8 dB | 1663 | 38.6 dB |
| 75 | 2088 | 43.1 dB | 2105 | 42.0 dB |
| 95 | 2389 | 44.5 dB | 3846 | 46.8 dB |

### 4.3 Why the numbers look the way they do

- The codec is grayscale-native (no chroma overhead to subsidise in
  the luminance), uses per-image Huffman tables (libjpeg's default
  fixed tables are worse unless 'optimize' is set), and shares JPEG's
  quantisation philosophy.
- libjpeg wins at q = 95 because its entropy coder handles the long
  runs of tiny coefficients at high quality more efficiently — a
  thirty-years-of-tuning effect.

## 5. Limitations

1. **8×8 DCT in matrix form** — clear and correct, but not the fast
   algorithm (AAN/Loeffler would be ~10× faster); fine for small
   images, not for real-time.
2. **No progressive scan** — JPEG's "fade in" mode; the DC and AC
   streams would need to be split and interleaved.
3. **Synthetic benchmark images** — deterministic and reproducible,
   but not photographs; a real photo set would test the perceptual
   claims more fairly (PIL can load them; this is future work).
4. **Huffman, not arithmetic coding** — arithmetic coding closes the
   remaining gap to JPEG 2000.
5. **Grayscale and RGB only** — no alpha channel, no 16-bit depth.

## 6. Conclusion

The project demonstrates the full compression pipeline — transform,
quantisation, entropy coding, bit-level format — implemented from
scratch and verified bit-exact, competing with a production JPEG codec
at typical qualities. Its value is twofold: it makes concrete the
theory (orthonormal transforms, energy compaction, Shannon entropy,
rate-distortion), and it shows the discipline of verifying every
component — the DCT's orthogonality, the Huffman code's optimality, and
the decoder's exactness against an independent reference pipeline.

---

*Code, verification suite, and demo: `image-compression/` — `dct.py`,
`quant.py`, `huffman.py`, `codec.py`, `metrics.py`, `benchmark.py`,
`verify.py`, `test_codec.py`, `demo.py`.*

*References: Wallace, "The JPEG Still Picture Compression Standard"
(IEEE Trans. Consumer Electronics, 1992); Wang et al., "Image Quality
Assessment: From Error Visibility to Structural Similarity" (IEEE TIP,
2004).*
