# Image Compression Project — Complete Interview Guide

*Everything you need to explain, defend, and extend this project in an
interview. Read it top to bottom once; then use the quick-reference and
Q&A sections to revise. `demo.py` is your 2-minute pitch.*

---

## 1. The pitch (60 seconds)

> I wrote a complete JPEG-style image codec from scratch — the 8×8
> Discrete Cosine Transform, quantisation, zig-zag ordering, run-length
> coding, and Huffman entropy coding, all in my own bit-level file
> format — and benchmarked it honestly against real JPEG.
>
> The interesting parts are the maths and the engineering. The maths:
> the DCT is an orthonormal transform that concentrates image energy
> into a few low-frequency coefficients — on a smooth 8×8 block the top
> 4 of 64 coefficients hold essentially all the energy. Quantisation
> then rounds 94% of coefficients to zero, and the entropy coder
> (Huffman, within 1 bit of the Shannon bound) stores the rest in
> almost nothing.
>
> The engineering: a documented bit-level format, a decoder that's
> *bit-exact* against the reference pipeline, and a quality metric
> suite (PSNR and SSIM implemented from scratch). The benchmark shows
> my codec matching or beating PIL's JPEG at equal file size for
> typical qualities — 45.4 dB vs 44.6 dB at the same size at
> quality 50.
>
> What I learned: lossy compression is really about *deciding what to
> throw away* (quantisation) and then *paying for only what remains*
> (entropy coding). The 8×8 block size and the DCT aren't magic — they
> are choices tuned for how the eye perceives error.

---

## 2. The maths (know these cold)

### 2.1 The DCT-II and why it beats the DFT

The 2D DCT-II of an 8×8 block is separable:

```
F = C f Cᵀ,    C[u][x] = a_u · cos(π(2x+1)u / 16)
a_0 = 1/√8,    a_u = 1/2 for u > 0
```

The inverse is `f = Cᵀ F C` because C is orthonormal (verified:
`C Cᵀ = I` to 1e-12, round-trip exact to 1e-10).

**Why DCT and not DFT?** The DFT treats the block as one period of an
infinite periodic signal — it implicitly *wraps around*, so a bright
edge on the right boundary of a block meets a dark left edge and
creates fake high frequencies. The DCT-II basis uses cosines with
half-pixel offsets, which corresponds to an *even reflection* of the
signal at the boundary. No discontinuity, so no fake high frequencies.
For natural images (smooth, mostly low-frequency), DCT-II is also the
best energy-compacting transform of the practical ones — which is
exactly why JPEG chose it.

**Energy compaction** is the whole game: after the transform, the
energy is concentrated in the top-left (low-frequency) corner, so
quantisation can discard the rest.

### 2.2 Quantisation — where the loss actually happens

Each coefficient is divided by a step size and rounded:

```
q[i][j] = round(F[i][j] / Q[i][j])
```

The table Q has small steps for low frequencies and large steps for
high ones, because the eye tolerates more error at high spatial
frequencies. The rounding is the ONLY loss in the whole codec — the
decoder is bit-exact after that.

The quantisation error per coefficient is bounded by half its step
size: `|dequant(q) - F| ≤ Q/2` (verified in the test suite).

### 2.3 Entropy and Huffman optimality

The entropy of a symbol distribution is the Shannon lower bound:

```
H = -Σ p_s log₂ p_s  bits/symbol
```

Huffman's algorithm builds a prefix-free code with average length
within 1 bit of H (sometimes exactly H). Verified: on real AC symbol
streams, `avg ≈ entropy` (e.g. 3.000 vs 2.999 bits/symbol).

Kraft's inequality is why prefix-free codes are decodable: no code is
a prefix of another, so a bitstream can be decoded greedily left to
right with no ambiguity.

### 2.4 PSNR and SSIM

- **PSNR** = `10 log₁₀(255² / MSE)` — pure pixel error. Infinite for
  identical images; every 6 dB is a 4× error reduction.
- **SSIM** (Wang et al. 2004) compares local *luminance, contrast and
  structure* with Gaussian windows: identical images score 1.0, a
  constant +5 offset scores ~0.999 (structure unchanged), random noise
  ~0.4-0.6, and a structural change (half the image zeroed) ~0.45.

---

## 3. Architecture (files and responsibilities)

| File | Responsibility |
|---|---|
| `dct.py` | 8×8 DCT-II and IDCT in matrix form, orthogonality self-test |
| `quant.py` | JPEG standard luminance table, quality scaling, zig-zag order |
| `huffman.py` | Huffman: frequency table → canonical codes, decode tree, entropy/average-length helpers |
| `codec.py` | The codec: image → blocks → DCT → quant → zig-zag → RLE → Huffman → bytes, and back. Documents the MCJP file format |
| `metrics.py` | PSNR and SSIM (Gaussian windows) from scratch |
| `benchmark.py` | Rate-distortion comparison vs PIL's JPEG at matched quality |
| `verify.py` | The proof suite |
| `demo.py` | The personal-statement story |

---

## 4. The codec pipeline (the heart)

### Encode

1. **Pad** the image to multiples of 8 (edge-replicate; the decoder
   crops back).
2. **Blockify**: H×W → (H/8)·(W/8) blocks of 8×8.
3. **DCT-II** each block (matrix form — `F = C f Cᵀ`).
4. **Quantise**: `q = round(F / Q)` — the lossy step.
5. **Zig-zag**: flatten 8×8 → 64 values in diagonal order, grouping
   similar frequencies so the zeros at the end become long runs.
6. **DC (differential)**: the first value is a block's average
   brightness. Adjacent blocks are similar, so encode the *difference*
   from the previous block's DC — differences are small, so their
   Huffman symbols are short. Symbol = category (bit-length of the
   diff), amplitude bits stored raw.
7. **AC (run,size RLE)**: for the remaining 63 values, skip zero runs.
   Each nonzero is encoded as symbol `(run, size)` — run = zeros
   before it (0..15), size = bits of |value| — followed by raw
   amplitude bits. A `(0,0)` symbol marks end-of-block, `(16,0)`
   (ZRL) handles runs longer than 15. Symbols stay in a small range,
   which keeps the Huffman alphabet tiny.
8. **Huffman** both symbol streams (adaptive: the frequency tables are
   computed from the actual data and stored in the header — smarter
   than JPEG's fixed standard tables).
9. **Pack** into bytes with a documented header: magic `MCJP`, size,
   quality, the two Huffman tables (symbol, length) pairs, symbol
   counts and byte counts, then the DC stream, DC amplitudes, AC
   stream, AC amplitudes.

### Decode

Mirror everything: parse header → rebuild canonical Huffman trees →
decode symbols → read amplitude bits → un-RLE → un-zig-zag →
dequantise → IDCT → crop. It is **bit-exact**: verified against a
reference numpy pipeline (DCT → quant → dequant → IDCT) with max
difference 0.0 for every tested size and quality.

### Why the format is defensible

- Big-endian, fixed-width integer fields — easy to parse, hard to
  get wrong.
- The amplitude bits are stored raw (not Huffman-coded) — exactly as
  in real JPEG. Huffman-coding them would save little, because their
  distribution is nearly uniform.
- Storing the Huffman tables adaptively beats JPEG's fixed standard
  tables for our images.

---

## 5. The metrics

- **PSNR**: `10 log₁₀(255²/MSE)`. Simple, standard, blind to
  structure.
- **SSIM**: sliding Gaussian window (σ = 1.5, 11×11); compare mean,
  variance and covariance of the two patches:

```
SSIM = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / ((μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂))
```

Implemented with a Gaussian window via direct 2D convolution
(O(N·k²) per image; fine at 256×256).

---

## 6. Benchmark results (quote these)

256×256 grayscale test image (smooth gradients + sharp edges),
quality 20-95, my codec vs PIL's JPEG (PIL = libjpeg):

| q | mine (B) | mine PSNR | JPEG (B) | JPEG PSNR |
|---|---------:|----------:|---------:|----------:|
| 20 | 2344 | 42.5 dB | 2340 | 39.5 dB |
| 50 | 3079 | 45.4 dB | 3608 | 44.6 dB |
| 75 | 3903 | 48.4 dB | 4875 | 48.4 dB |
| 95 | 4363 | 50.3 dB | 10637 | 51.9 dB |

- At q ≤ 75 my from-scratch codec matches or **beats** libjpeg in PSNR
  at equal file size.
- At q = 95 libjpeg wins (10.6 kB vs 4.4 kB but 51.9 vs 50.3 dB).
- 94% of coefficients quantise to zero at q = 50 (61468/65536).

Why is a from-scratch codec competitive?
- Grayscale only: no chroma-subsampling overhead.
- Adaptive Huffman tables per image (JPEG uses fixed standard tables
  unless you enable "optimize").
- Same quantisation table and DCT design as JPEG.

Where JPEG wins: its entropy coder at very high quality (long runs of
small coefficients) and its 30 years of tuning.

---

## 7. Complexity notes

- DCT of one block: two 8×8 matrix multiplies — O(64) each with
  numpy's BLAS; O(n) total over n pixels.
- Huffman build: O(S log S) with a heap, S = distinct symbols (≤ 251).
- Huffman encode/decode: O(total bits) — linear in output size.
- Zig-zag: O(64) per block via a precomputed index map.
- Overall: the codec is **linear in the number of pixels**, with a
  small constant — a 256×256 image encodes in milliseconds.
- SSIM: O(W·H·k²) with window size k = 11 (direct 2D convolution;
  a separable version would be O(W·H·k) but is not implemented).

---

## 8. Design decisions (be ready to defend each)

1. **8×8 blocks, not the whole image or 16×16.** Small enough that the
   "smooth" assumption holds locally; big enough that the transform
   and the per-block header overhead amortise. JPEG's choice; there is
   research on adaptive block sizes, but 8×8 is the sweet spot.

2. **DCT-II over DFT.** Even reflection at block boundaries → no fake
   high frequencies → better energy compaction (see §2.1).

3. **Quantisation as the only lossy step.** Everything else is
   reversible. This makes the codec's behaviour easy to reason about
   and the decoder easy to verify (bit-exact).

4. **Differential DC.** Adjacent blocks have similar averages;
   differences concentrate near zero → short Huffman codes.

5. **RLE + size-split symbols, not raw coefficients.** The symbol
   alphabet stays small (≤ 251), so Huffman works well; amplitude bits
   are raw because they're near-uniform (following JPEG's proven
   design).

6. **Adaptive Huffman tables in the header.** Costs a few hundred
   bytes, buys better coding than fixed tables on our images.

7. **Edge-replicate padding.** Avoids injecting artificial edges at
   image borders.

8. **My own documented format instead of emitting real JPEG.**
   Writing the bit-level format was the point — it forces you to
   understand every byte; PIL's JPEG was used only as the benchmark
   baseline.

---

## 9. Bugs I found and fixed (great interview material)

1. **Amplitude bits lost their leading pad bits.** `bit_length()` on a
   packed negative amplitude drops the sign-pad bits (e.g. -100 in
   size 7 needs 7 bits, `bit_length` gives 5). Every negative
   coefficient silently corrupted. Fixed by packing exactly `size`
   bits per amplitude.

2. **AC symbol/amplitude misalignment.** EOB and ZRL symbols carry no
   amplitude, so `zip(symbols, amplitudes)` silently misaligned the
   whole stream. Fixed by aligning sizes with the symbols that
   actually have amplitudes.

3. **Every block decoded block 0's AC symbols.** `_unrle_ac` was
   called with the full symbol list, so each block re-read from the
   start (stopping at the first EOB). Fixed with a shared iterator.

4. **Symbol overflow on noisy images.** The first AC scheme packed
   `run*64 + (value+31)`, which breaks when quantised values exceed 31
   — white noise produces huge coefficients, and the decode silently
   corrupted (PSNR *dropped* as quality rose — a wonderful canary).
   Switched to JPEG's `(run,size)` + raw-amplitude scheme.

5. **DC amplitude count mismatch.** Decode recomputed sizes with
   `_category(size)` instead of reading them directly from the decoded
   symbols — off-by-N bits that shifted the AC stream.

6. **Rectangular images broke the block reshape.** The first
   `_from_blocks` assumed a square block grid; 23×37 images crashed.
   Fixed to carry the padded H and W.

The meta-lesson (worth saying): the failure mode that caught bugs 1-4
was **monotonicity** — PSNR must increase with quality, and file size
must too. A codec whose quality goes *down* as you turn quality *up*
is provably broken somewhere in the entropy layer.

---

## 10. Measured numbers (cite these)

- DCT round trip exact to < 1e-10; C orthonormal to 1e-12.
- Huffman: avg 3.000 bits/symbol vs entropy 2.999 (within 1 bit).
- 94% of coefficients zero after quantisation at q = 50.
- Decode bit-exact vs reference pipeline (max diff 0.0) across sizes
  16×16, 23×37, 32×48 and qualities 20/50/95.
- PSNR strictly increasing in quality: 19.7 → 22.8 → 27.9 dB.
- Benchmark table in §6.

---

## 11. What I'd do next

1. **Chroma subsampling + RGB**: encode YCbCr, subsample chroma 4:2:0
   (the eye is less sensitive to colour detail) — the standard way to
   double compression on colour photos.
2. **Optimised Huffman / arithmetic coding**: CABAC-style arithmetic
   coding closes most of the remaining gap to JPEG 2000.
3. **Adaptive quantisation**: vary the table per 8×8 block based on
   local contrast (perceptual coding).
4. **Progressive scan**: encode DC then AC layers so images "fade in"
   — the mechanism behind progressive JPEG.
5. **Real-image benchmark**: pull a few photos (PIL can load them) and
   extend the benchmark to colour.

---

## 12. Rapid-fire Q&A

**Q: Why is quantisation lossy but necessary?**
A: Without it every coefficient needs full precision and you save
nothing. Quantisation rounds small (perceptually unimportant)
coefficients to zero — most of them — leaving a few big ones that
encode cheaply. The loss is chosen to match how the eye perceives
error.

**Q: Why DCT over DFT?**
A: DFT wraps the block around, creating boundary discontinuities that
generate fake high frequencies. DCT-II's even reflection avoids them,
so the same image compresses better.

**Q: Why 8×8?**
A: Small enough that blocks look smooth (energy concentrates in a few
coefficients), big enough that transform and header overhead
amortise. It's JPEG's empirically-chosen sweet spot.

**Q: What does zig-zag actually buy?**
A: It orders coefficients by frequency, so the many zeros are grouped
into long runs. RLE then encodes "16 zeros" with a single symbol, and
end-of-block marks the tail of zeros with one symbol.

**Q: Why is Huffman near-optimal?**
A: Huffman's algorithm provably builds a minimum-average-length
prefix-free code for a given frequency table; its average length is
within 1 bit of the Shannon entropy. It's optimal for symbol-by-symbol
coding; only block/arithmetic coding can beat it.

**Q: How do you know the decoder is right?**
A: I verified decode output is bit-identical to a reference pipeline
(DCT → quant → dequant → IDCT) across sizes and qualities, and that
PSNR is monotonic in quality. Any entropy-layer bug breaks one of
those almost immediately.

**Q: Why is my codec competitive with libjpeg?**
A: Grayscale (no chroma overhead), adaptive per-image Huffman tables
(JPEG's default fixed tables are worse), the same DCT and quantisation
design, and a lean header.

**Q: What does PSNR not capture?**
A: Structure. Two images with the same MSE can look very different —
one may be blurry (high-frequency error) and one may have sharp
blocking artefacts. That's why SSIM exists.

**Q: What's the rate-distortion trade-off?**
A: Rate (bits per pixel) vs distortion (PSNR/SSIM). You can always
spend more bits for less distortion; a codec is good when it achieves
high quality per bit. The benchmark plots exactly this.

**Q: Worst case for your codec?**
A: White noise — no energy compaction, 94% zeros becomes maybe 20%,
and file size approaches the raw size. Natural images are the opposite.

**Q: What would you change for colour images?**
A: Convert to YCbCr and subsample chroma 4:2:0 — the eye is much less
sensitive to colour detail than luminance, so you can spend ~4× fewer
bits on chroma with no visible loss.

---

## 13. Whiteboard script (5 minutes)

1. Draw an 8×8 block of pixels. Draw a horizontal cosine of frequency
   u=1 and one of u=7. Explain: "the DCT projects the block onto each
   of 64 such patterns; smooth blocks have big projections only for
   low-frequency patterns."
2. Write the separable formula `F = C f Cᵀ` and the basis
   `a_u cos(π(2x+1)u/16)`; say orthonormal, inverse is transpose.
3. Draw the zig-zag diagonal over an 8×8 grid; shade the top-left
   corner "big coefficients here, zeros in the tail".
4. Quantisation: `q = round(F/Q)`; show a tiny 2×2 example rounding to
   mostly zeros.
5. Draw a Huffman binary tree from two symbol frequencies; show how a
   rare symbol gets a long code, a common one a short code.
6. The one-line summary of the pipeline, left to right.

---

## 14. The one-sentence summary

> **"I built a JPEG-style image codec entirely from scratch — DCT,
> quantisation, zig-zag, run-length and Huffman coding in my own
> documented bitstream format — verified the decoder to be bit-exact,
> and showed it matches or beats PIL's real JPEG in quality-per-byte
> at typical settings."**

---

*Files: `dct.py` (transform) · `quant.py` (loss) · `huffman.py`
(entropy) · `codec.py` (pipeline + format) · `metrics.py` (quality) ·
`benchmark.py` (vs JPEG) · `verify.py` (proofs) · `demo.py` (the
story). Run `python3 demo.py` to re-walk the whole story.*
