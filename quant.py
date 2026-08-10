"""Quantisation and zig-zag ordering.

Quantisation is where lossy compression actually happens: each DCT
coefficient is divided by a step size and rounded.  Large steps -> small
coefficients -> few Huffman symbols, at the cost of reconstruction
error.  The eye tolerates more error at high spatial frequencies, so
standard tables (JPEG luminance table) use small steps for low
frequencies and large steps for high ones.

zig-zag orders the 8x8 coefficient matrix along diagonals, which groups
similar frequencies together: after quantisation, most high-frequency
coefficients are zero, so the zig-zag produces long runs of zeros that
are cheap to encode (RLE in codec.py).
"""

from __future__ import annotations

import numpy as np

# Standard JPEG luminance quantisation table (quality ~50-75).
JPEG_LUMA_Q = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float64)

# Standard JPEG chrominance table (coarser: colour error is less
# visible than luminance error).
JPEG_CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def scaled_table(base: np.ndarray, quality: float) -> np.ndarray:
    """JPEG-style quality scaling.  quality in (0, 100]; lower = smaller
    steps = better quality, larger files.

    scale = 50/quality for quality >= 50 else (100-quality)/50 (JPEG
    semantics, so quality 50 keeps the base table).
    """
    if quality <= 0 or quality > 100:
        raise ValueError("quality must be in (0, 100]")
    if quality >= 50:
        s = 50.0 / quality
    else:
        s = (100.0 - quality) / 50.0
    return np.maximum(1.0, np.round(base * s))


def quantize(coeffs: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Divide coefficients by the table and round to integers."""
    return np.rint(coeffs / table).astype(np.int64)


def dequantize(q: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Multiply quantised coefficients back by the table."""
    return (q * table).astype(np.float64)


# zig-zag index map for an 8x8 matrix
_ZIGZAG: list[tuple[int, int]] = []
_r, _c = 0, 0
_up = True
for _ in range(64):
    _ZIGZAG.append((_r, _c))
    if _up:
        if _c == 7:
            _r += 1
            _up = False
        elif _r == 0:
            _c += 1
            _up = False
        else:
            _r -= 1
            _c += 1
    else:
        if _r == 7:
            _c += 1
            _up = True
        elif _c == 0:
            _r += 1
            _up = True
        else:
            _r += 1
            _c -= 1
_zigzag_lookup = {pos: i for i, pos in enumerate(_ZIGZAG)}


def zigzag(qblock: np.ndarray) -> np.ndarray:
    """Flatten an 8x8 quantised block into 64 values in zig-zag order."""
    return np.array([qblock[r, c] for (r, c) in _ZIGZAG], dtype=np.int64)


def unzigzag(values: np.ndarray) -> np.ndarray:
    """Reconstruct an 8x8 block from a zig-zag ordered 64-vector."""
    out = np.zeros((8, 8), dtype=np.int64)
    for i, (r, c) in enumerate(_ZIGZAG):
        out[r, c] = values[i]
    return out
