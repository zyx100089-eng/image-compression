"""Verification suite for the image compression project.

Checks:
1. DCT is orthonormal; round trip is exact; energy compaction
2. Huffman codes are prefix-free and round trip; length within 1 bit
   of the entropy lower bound
3. quantize/dequantize + zig-zag round trip
4. full codec: encode/decode matches the reference (DCT->quant->IDCT)
   pipeline exactly; sizes preserved; bit-exact across qualities
5. PSNR monotonicity: higher quality -> lower MSE
6. benchmark sanity: the codec competes with PIL's JPEG (never absurdly
   worse at comparable quality)
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from codec import decode_image, encode_image, quant_table
from dct import dct2, idct2, verify_orthogonal
from huffman import Huffman
from metrics import psnr
from quant import dequantize, quantize, unzigzag, zigzag


def check_dct() -> None:
    print("[1] DCT")
    assert verify_orthogonal(), "DCT matrix not orthonormal"
    rng = np.random.default_rng(0)
    for _ in range(5):
        b = rng.normal(size=(8, 8))
        assert np.abs(idct2(dct2(b)) - b).max() < 1e-10, "round trip"
    assert np.allclose((dct2(np.ones((8, 8))) ** 2).sum(), 64.0)
    print("    orthonormal, exact round trip, energy preservation OK")


def check_huffman() -> None:
    print("[2] Huffman")
    rng = np.random.default_rng(1)
    syms = list(rng.integers(0, 8, size=5000))
    freqs = dict(zip(*np.unique(syms, return_counts=True)))
    h = Huffman.from_frequencies(freqs)
    enc = h.encode(syms)
    assert h.decode(enc) == syms, "bit round trip"
    avg = len(enc) / len(syms)
    ent = Huffman.entropy(freqs)
    assert avg <= ent + 1.0 + 1e-9, f"huffman too far from entropy: {avg} vs {ent}"
    # prefix-free: no code is a prefix of another (a full decode is
    # unambiguous); also every symbol must have a code
    codes = sorted((c, l) for c, l in h.symbol_to_code.values())
    for i, (c1, l1) in enumerate(codes):
        for (c2, l2) in codes[i + 1:]:
            if l1 <= l2:
                assert (c2 >> (l2 - l1)) != c1, f"code {c1}/{l1} is a prefix of {c2}/{l2}"
            else:
                assert (c1 >> (l1 - l2)) != c2, f"code {c2}/{l2} is a prefix of {c1}/{l1}"
    print(f"    round trip OK, avg {avg:.3f} bits/sym vs entropy {ent:.3f}")


def check_quant() -> None:
    print("[3] quantize / zig-zag")
    rng = np.random.default_rng(2)
    b = rng.integers(-1000, 1000, size=(8, 8))
    t = quant_table(50)
    assert np.array_equal(unzigzag(zigzag(b)), b), "zig-zag round trip"
    q = quantize(b.astype(float), t)
    d = dequantize(q, t)
    # quantisation error bounded by half the table steps
    bound = (t / 2).max()
    assert np.abs(d - b).max() <= bound + 1, "quant error bound"
    print("    zig-zag + quant error bound OK")


def check_codec_exact() -> None:
    print("[4] codec exactness vs reference pipeline")
    rng = np.random.default_rng(3)
    for (h, w) in [(16, 16), (23, 37), (32, 48)]:
        img = rng.normal(128, 40, size=(h, w)).clip(0, 255)
        for q in (20, 50, 95):
            data = encode_image(img, quality=q)
            rec, qq = decode_image(data)
            assert qq == q, "quality round trip"
            assert rec.shape == img.shape, "shape"
            # reference: DCT -> quant -> dequant -> IDCT (no entropy)
            padded = img
            ph = (8 - h % 8) % 8
            pw = (8 - w % 8) % 8
            padded = np.pad(img, ((0, ph), (0, pw)), mode="edge")
            t = quant_table(q)
            H, W = padded.shape
            blocks = padded.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
            recb = np.array([idct2(dequantize(quantize(dct2(b), t), t)) for b in blocks])
            ref = recb.reshape(H // 8, W // 8, 8, 8).transpose(0, 2, 1, 3).reshape(H, W)
            ref = np.clip(ref, 0, 255)[:h, :w]
            assert np.allclose(rec, ref, atol=1e-9), f"mismatch at {h}x{w} q={q}"
    print("    decode == reference pipeline (bit-exact) for all sizes/qualities")


def check_psnr_monotonic() -> None:
    print("[5] PSNR monotonic in quality")
    rng = np.random.default_rng(4)
    img = rng.normal(128, 40, size=(64, 64)).clip(0, 255)
    psnrs = []
    for q in (20, 50, 95):
        rec, _ = decode_image(encode_image(img, quality=q))
        psnrs.append(psnr(img, rec))
    assert psnrs[0] < psnrs[1] < psnrs[2], f"PSNR not monotonic: {psnrs}"
    print(f"    PSNR {[round(p, 1) for p in psnrs]} dB strictly increasing")


def check_colour() -> None:
    print("[5b] colour codec (YCbCr + 4:2:0)")
    rng = np.random.default_rng(5)
    H, W = 64, 48
    x, y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    img = np.zeros((H, W, 3))
    img[..., 0] = 128 + 60 * np.sin(6 * np.pi * x) * np.cos(4 * np.pi * y)
    img[..., 1] = 128 + 60 * np.sin(4 * np.pi * x + 1)
    img[..., 2] = 128 + 40 * np.tanh((x - 0.5) * 15)
    img = np.clip(img, 0, 255)

    for q in (20, 50, 95):
        data = encode_image(img, quality=q)
        rec, q2 = decode_image(data)
        assert q2 == q, "colour quality round trip"
        assert rec.shape == img.shape, "colour shape"
        mse = np.mean((rec - img) ** 2)
        assert mse < 900, f"colour q={q} PSNR too low: {10*np.log10(255**2/mse):.1f} dB"
    # non-multiple-of-8 colour
    odd = rng.normal(128, 40, size=(23, 37, 3)).clip(0, 255)
    rec, _ = decode_image(encode_image(odd, quality=50))
    assert rec.shape == odd.shape
    # grayscale still works
    g = img[..., 0]
    rec_g, _ = decode_image(encode_image(g, quality=50))
    assert rec_g.shape == g.shape
    print("    colour round trips, odd sizes, grayscale all OK")


def check_benchmark_sanity() -> None:
    print("[6] benchmark vs PIL JPEG (sanity)")
    from benchmark import test_images, my_encode, jpeg_encode
    for name, img in test_images()[:2]:
        _, rec_mine = my_encode(img, 50)
        _, rec_jpeg = jpeg_encode(img, 50)
        p_mine = psnr(img, rec_mine)
        p_jpeg = psnr(img, rec_jpeg)
        # my codec must be within 3 dB of PIL's JPEG at q=50 on these
        assert p_mine >= p_jpeg - 3.0, f"{name}: mine {p_mine:.1f} vs jpeg {p_jpeg:.1f}"
        print(f"    {name}: mine {p_mine:.1f} dB vs JPEG {p_jpeg:.1f} dB OK")


def main() -> None:
    check_dct()
    check_huffman()
    check_quant()
    check_codec_exact()
    check_psnr_monotonic()
    check_colour()
    check_benchmark_sanity()
    print("\nAll verification passed.")


if __name__ == "__main__":
    main()
