"""8x8 2D Discrete Cosine Transform (DCT-II) and its inverse.

Why DCT and not DFT for images?  The DFT implicitly repeats the image
periodically, so a sharp edge at the boundary creates artificial high
frequencies.  DCT-II's basis vectors are cosines with half-sample
offsets; the implied extension is *even* reflection, which avoids the
boundary discontinuity.  For smooth images DCT-II is also the best
energy-compacting orthogonal transform of the common ones, which is
why JPEG chose it.

Math: for an 8x8 block f, the 2D DCT is separable:

    F = C f C^T,   C[u][x] = a_u cos(pi(2x+1)u / 16)

with a_0 = 1/sqrt(8), a_u = 1/2 for u > 0.  Because C is orthogonal,
the inverse is f = C^T F C.  The implementation uses matrix
multiplication (numpy), which is clear and correct; speed is fine for
8x8 blocks since numpy's BLAS handles the 8x8 matmuls.
"""

from __future__ import annotations

import numpy as np

_N = 8


def _dct_matrix(n: int = _N) -> np.ndarray:
    """The n x n DCT-II basis matrix C (rows = frequencies)."""
    x = np.arange(n)
    u = np.arange(n)
    C = np.cos(np.pi * (2 * x[None, :] + 1) * u[:, None] / (2 * n))
    C[0, :] /= np.sqrt(n)
    C[1:, :] *= np.sqrt(2.0 / n)
    return C


_C = _dct_matrix()


def dct2(block: np.ndarray) -> np.ndarray:
    """Forward 2D DCT of an 8x8 block -> 8x8 coefficients."""
    return _C @ block @ _C.T


def idct2(coeffs: np.ndarray) -> np.ndarray:
    """Inverse 2D DCT of an 8x8 coefficient block -> 8x8 spatial block."""
    return _C.T @ coeffs @ _C


def dct2_fast(block: np.ndarray) -> np.ndarray:
    """Separable row-then-column DCT (equivalent, used as a cross-check)."""
    rows = block @ _C.T
    return _C @ rows


def verify_orthogonal() -> bool:
    """C is orthogonal up to scaling conventions: C @ C^T should be I."""
    return np.allclose(_C @ _C.T, np.eye(_N), atol=1e-12)
