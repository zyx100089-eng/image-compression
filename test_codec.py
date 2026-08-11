"""pytest suite for the image compression project.

Run with:  python3 -m pytest test_codec.py -q

Fast unit tests (no external data).  The slow end-to-end checks
(bit-exactness across sizes/qualities, benchmark sanity) live in
verify.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from codec import decode_image, encode_image
from dct import dct2, idct2
from huffman import Huffman
from metrics import psnr, ssim
from quant import dequantize, quantize, unzigzag, zigzag


# ----------------------------------------------------------------------
# DCT
# ----------------------------------------------------------------------

class TestDCT:
    def test_orthonormal(self):
        from dct import _C
        assert np.allclose(_C @ _C.T, np.eye(8), atol=1e-12)

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(3):
            b = rng.normal(size=(8, 8))
            assert np.abs(idct2(dct2(b)) - b).max() < 1e-10

    def test_energy_compaction(self):
        F = dct2(np.ones((8, 8)))
        assert abs(F[0, 0] - 8.0) < 1e-9          # DC = 8 * mean
        assert np.abs(F[1:, :]).max() < 1e-9       # AC ~ 0

    def test_parseval(self):
        rng = np.random.default_rng(1)
        b = rng.normal(size=(8, 8))
        assert np.allclose((b ** 2).sum(), (dct2(b) ** 2).sum(), atol=1e-8)


# ----------------------------------------------------------------------
# Huffman
# ----------------------------------------------------------------------

class TestHuffman:
    def test_round_trip(self):
        rng = np.random.default_rng(2)
        syms = list(rng.integers(0, 8, size=1000))
        freqs = dict(zip(*np.unique(syms, return_counts=True)))
        h = Huffman.from_frequencies(freqs)
        assert h.decode(h.encode(syms)) == syms

    def test_within_one_bit_of_entropy(self):
        rng = np.random.default_rng(3)
        syms = list(rng.integers(0, 8, size=5000))
        freqs = dict(zip(*np.unique(syms, return_counts=True)))
        h = Huffman.from_frequencies(freqs)
        avg = len(h.encode(syms)) / len(syms)
        assert avg <= Huffman.entropy(freqs) + 1.0 + 1e-9

    def test_prefix_free(self):
        freqs = {i: i + 1 for i in range(8)}
        h = Huffman.from_frequencies(freqs)
        codes = list(h.symbol_to_code.values())
        for i, (c1, l1) in enumerate(codes):
            for (c2, l2) in codes[i + 1:]:
                if l1 <= l2:
                    assert (c2 >> (l2 - l1)) != c1
                else:
                    assert (c1 >> (l1 - l2)) != c2

    def test_single_symbol(self):
        h = Huffman.from_frequencies({5: 100})
        assert h.code_lengths[5] >= 1
        assert h.decode(h.encode([5, 5, 5])) == [5, 5, 5]

    def test_bytes_round_trip(self):
        rng = np.random.default_rng(4)
        syms = list(rng.integers(0, 5, size=300))
        freqs = dict(zip(*np.unique(syms, return_counts=True)))
        h = Huffman.from_frequencies(freqs)
        data, _ = h.encode_to_bytes(syms)
        assert h.decode_from_bytes(data, len(syms)) == syms


# ----------------------------------------------------------------------
# quantisation
# ----------------------------------------------------------------------

class TestQuant:
    def test_zigzag_round_trip(self):
        rng = np.random.default_rng(5)
        b = rng.integers(-1000, 1000, size=(8, 8))
        assert np.array_equal(unzigzag(zigzag(b)), b)

    def test_zigzag_covers_all(self):
        assert sorted(zigzag(np.arange(64).reshape(8, 8)).tolist()) == list(range(64))

    def test_quant_error_bound(self):
        rng = np.random.default_rng(6)
        b = rng.normal(size=(8, 8)) * 500
        from codec import quant_table
        t = quant_table(50)
        q = quantize(b, t)
        d = dequantize(q, t)
        assert np.abs(d - b).max() <= (t / 2).max() + 1

    def test_table_steps_positive(self):
        from codec import quant_table
        for q in (10, 50, 90):
            assert (quant_table(q) >= 1).all()


# ----------------------------------------------------------------------
# codec
# ----------------------------------------------------------------------

class TestCodec:
    @pytest.fixture
    def gray(self):
        rng = np.random.default_rng(7)
        return np.clip(rng.normal(128, 40, size=(32, 40)), 0, 255)

    @pytest.fixture
    def colour(self):
        rng = np.random.default_rng(8)
        H, W = 32, 40
        x, y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        img = np.zeros((H, W, 3))
        img[..., 0] = 128 + 60 * np.sin(6 * np.pi * x)
        img[..., 1] = 128 + 60 * np.cos(4 * np.pi * y)
        img[..., 2] = 128 + 40 * np.tanh((x - 0.5) * 10)
        return np.clip(img, 0, 255)

    def test_gray_round_trip_shape(self, gray):
        data = encode_image(gray, quality=50)
        rec, q = decode_image(data)
        assert rec.shape == gray.shape
        assert q == 50

    def test_colour_round_trip_shape(self, colour):
        data = encode_image(colour, quality=50)
        rec, q = decode_image(data)
        assert rec.shape == colour.shape
        assert q == 50

    def test_quality_int_consistency(self):
        # fractional quality must behave like its integer floor
        rng = np.random.default_rng(9)
        g = rng.random((16, 16)) * 255
        a = decode_image(encode_image(g, quality=50.7))[0]
        b = decode_image(encode_image(g, quality=50))[0]
        assert np.array_equal(a, b)

    def test_nan_rejected(self):
        g = np.zeros((16, 16))
        g[3, 3] = np.nan
        with pytest.raises(ValueError):
            encode_image(g, quality=50)

    def test_odd_size_round_trip(self):
        rng = np.random.default_rng(10)
        for shape in [(23, 37), (23, 37, 3)]:
            img = rng.normal(128, 30, size=shape).clip(0, 255)
            rec, _ = decode_image(encode_image(img, quality=50))
            assert rec.shape == shape

    def test_quality_bounds(self):
        g = np.zeros((8, 8))
        with pytest.raises(ValueError):
            encode_image(g, quality=0)
        with pytest.raises(ValueError):
            encode_image(g, quality=101)

    def test_truncated_file_raises(self):
        g = np.zeros((16, 16))
        data = encode_image(g, quality=50)
        with pytest.raises(Exception):
            decode_image(data[:20])

    def test_corrupt_magic_raises(self):
        with pytest.raises(ValueError):
            decode_image(b"XXXX" + b"\x00" * 32)

    def test_malformed_ac_stream_raises(self):
        # A (run,size) symbol that overruns the 63 AC coefficients must
        # raise, not silently truncate.  Build a stream whose first
        # symbol claims a run that exceeds the block.
        from codec import _unrle_ac
        # sym = run*16 + size; run=60, size=1 -> idx lands at 60, then
        # idx += 1 -> 61, next symbol run=5 -> 66 >= 63 -> raise
        syms = iter([60 * 16 + 1, 5 * 16 + 1, 0])
        bits = iter([1, 1])
        with pytest.raises(ValueError):
            _unrle_ac(syms, bits)


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

class TestMetrics:
    def test_psnr_identical(self):
        a = np.full((32, 32), 100.0)
        assert psnr(a, a) == float("inf")

    def test_psnr_sane(self):
        rng = np.random.default_rng(11)
        a = rng.random((32, 32)) * 255
        b = np.clip(a + 10, 0, 255)
        p = psnr(a, b)
        assert 20 < p < 60

    def test_ssim_identical(self):
        a = np.full((32, 32), 100.0)
        assert ssim(a, a) == 1.0

    def test_ssim_structure_sensitive(self):
        a = np.full((64, 64), 100.0)
        b = a.copy()
        b[32:, :] = 0
        assert ssim(a, b) < 0.9

    def test_ssim_colour(self):
        a = np.full((32, 32, 3), 100.0)
        b = a + 5
        assert abs(ssim(a, b) - 1.0) < 0.01

    def test_ssim_vs_psnr_differ(self):
        # a blur (structure change) should hurt SSIM more than uniform noise
        rng = np.random.default_rng(12)
        a = rng.random((64, 64)) * 255
        noise = np.clip(a + rng.normal(0, 15, size=a.shape), 0, 255)
        blurred = np.ones((64, 64))
        blurred[16:48, 16:48] = 128
        assert ssim(a, noise) > ssim(a, blurred)

    def test_ssim_matches_scipy_reference(self):
        # Independent cross-check against scipy's gaussian_filter path
        # (a completely different implementation than the manual conv).
        from scipy.ndimage import gaussian_filter

        def ssim_ref(x, y, window_size=11, sigma=1.5):
            L = 255.0
            c1 = (0.01 * L) ** 2
            c2 = (0.03 * L) ** 2
            trunc = window_size / (2 * sigma)
            mu_x = gaussian_filter(x, sigma, truncate=trunc)
            mu_y = gaussian_filter(y, sigma, truncate=trunc)
            mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
            sx2 = gaussian_filter(x * x, sigma, truncate=trunc) - mu_x2
            sy2 = gaussian_filter(y * y, sigma, truncate=trunc) - mu_y2
            sxy = gaussian_filter(x * y, sigma, truncate=trunc) - mu_xy
            num = (2 * mu_xy + c1) * (2 * sxy + c2)
            den = (mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2)
            return float(np.mean(num / den))

        rng = np.random.default_rng(42)
        for _ in range(3):
            a = rng.integers(0, 256, (64, 64)).astype(float)
            b = a + rng.normal(0, 15, a.shape)
            assert abs(ssim(a, b) - ssim_ref(a, b)) < 0.001
