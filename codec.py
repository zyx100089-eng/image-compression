"""Full JPEG-style codec: image -> bitstream and back.

Pipeline (encode):
    image -> pad to 8x8 multiples -> 8x8 blocks
        -> DCT-II -> quantize -> zig-zag
        -> per-block: DC (differential), AC (JPEG-style (run,size) RLE)
        -> Huffman on the symbol stream + raw amplitude bits
        -> header + packed bytes

Colour: RGB -> YCbCr (BT.601), chroma subsampled 4:2:0 (2x2 averaged,
replicated on decode - the eye is far less sensitive to colour detail
than to luminance), each plane coded with its own Huffman tables.  The
chroma planes use a coarser quantisation table (JPEG chrominance).

File format (my own, documented):
    magic   'MCJP'  (4 bytes)
    width   u32, height u32        (original image size)
    quality u8                     (quantisation level used)
    channels u8                    (1 = grayscale, 3 = RGB/YCbCr)
    then per channel:
      dc table:  n_syms u32, then n_syms x (sym i32, len u8)
      ac table:  n_syms u32, then n_syms x (sym i32, len u8)
      dc stream: n_syms u32, n_bytes u32, packed bytes
      ac stream: n_syms u32, n_bytes u32, packed bytes + raw amplitudes
    (amplitude bits are raw, not Huffman-coded; their count is implied
     by the sizes in the AC symbols)

Symbols (JPEG-style, amplitude-limited so symbols stay small):
    DC block: symbol = category of the difference (number of bits), and
              the amplitude bits are stored raw after the Huffman symbol.
    AC block: symbols (run,size): run = zeros before a nonzero, size =
              bits of |value|; amplitude bits follow raw.  (0,0) = EOB,
              (16,0) = ZRL (16 zeros).  run in 0..16 and size in 0..10
              keep the Huffman alphabet small (<= 256 symbols).

The decoder is exact: quantised coefficients come back bit-identical,
so the only loss is the quantisation step itself (the same lossy
behaviour as real JPEG).
"""

from __future__ import annotations

import struct

import numpy as np

from dct import dct2, idct2
from huffman import Huffman
from quant import (JPEG_CHROMA_Q, JPEG_LUMA_Q, dequantize, quantize,
                   scaled_table, unzigzag, zigzag)


def _pad_to_blocks(img: np.ndarray, n: int = 8) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    ph = (n - h % n) % n
    pw = (n - w % n) % n
    padded = np.pad(img, ((0, ph), (0, pw)) + ((0, 0),) * (img.ndim - 2),
                    mode="edge")
    return padded, ph, pw


def _to_blocks(img: np.ndarray) -> np.ndarray:
    """img (H, W) -> array of 8x8 blocks, shape (nb, 8, 8)."""
    H, W = img.shape
    return img.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)


def _category(value: int) -> int:
    """Number of bits to represent |value| in two's-complement style
    (JPEG 'size'): 0 -> 0, 1 -> 1, 2-3 -> 2, 4-7 -> 3, ..."""
    if value == 0:
        return 0
    return int(abs(value)).bit_length()


def _amplitude_bits(value: int, size: int) -> int:
    """Raw bits for a value of given size: negative values are stored
    as (1 << size) - 1 - (-value), positive as value."""
    if value >= 0:
        return value
    return (1 << size) - 1 + value


def _value_from_bits(bits: int, size: int) -> int:
    if size == 0:
        return 0
    if bits & (1 << (size - 1)):  # leading 1 -> positive
        return bits
    return bits - (1 << size) + 1


# ----------------------------------------------------------------------
# RGB <-> YCbCr (BT.601) colour conversion + 4:2:0 chroma subsampling
# ----------------------------------------------------------------------

def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB float (0..255) -> (3, H, W) YCbCr float."""
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128.0 + (-0.168736 * R - 0.331264 * G + 0.5 * B)
    Cr = 128.0 + (0.5 * R - 0.418688 * G - 0.081312 * B)
    return np.stack([Y, Cb, Cr])


def ycbcr_to_rgb(planes: np.ndarray) -> np.ndarray:
    """(3, H, W) YCbCr float -> (H, W, 3) RGB float, clipped to 0..255."""
    Y, Cb, Cr = planes
    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255)


def subsample_chroma(planes: np.ndarray) -> np.ndarray:
    """4:2:0 chroma subsampling: Cb/Cr averaged over 2x2 blocks.  The
    eye is much less sensitive to colour detail than to luminance."""
    H, W = planes.shape[1], planes.shape[2]
    out = planes.copy()
    for k in (1, 2):  # Cb, Cr
        p = planes[k][: H - H % 2, : W - W % 2]
        avg = (p[0::2, 0::2] + p[1::2, 0::2]
               + p[0::2, 1::2] + p[1::2, 1::2]) / 4.0
        out[k, :H - H % 2, :W - W % 2] = np.repeat(np.repeat(avg, 2, 0), 2, 1)
    return out


def _plane_from_blocks(blocks: np.ndarray, H: int, W: int) -> np.ndarray:
    n_h = H // 8
    n_w = W // 8
    img = blocks.reshape(n_h, n_w, 8, 8).transpose(0, 2, 1, 3)
    return img.reshape(H, W)


def _encode_plane(plane: np.ndarray, table: np.ndarray) -> tuple[list, list, list, list]:
    """DCT -> quant -> zig-zag -> RLE for one plane.  The plane is
    padded to 8x8 multiples (edge-replicate) before blocking.  Returns
    (dc_syms, dc_amp, ac_syms, ac_amp)."""
    padded, _, _ = _pad_to_blocks(plane)
    dc_syms: list[int] = []
    dc_amp: list[int] = []
    ac_syms: list[int] = []
    ac_amp: list[int] = []
    prev_dc = 0
    for block in _to_blocks(padded):
        coeffs = dct2(block)
        q = quantize(coeffs, table)
        zz = zigzag(q)
        dc, ac = zz[0], zz[1:]
        diff = int(dc) - prev_dc
        prev_dc = int(dc)
        sz = _category(diff)
        dc_syms.append(sz)
        dc_amp.append(_amplitude_bits(diff, sz))
        _rle_ac(ac, ac_syms, ac_amp)
    return dc_syms, dc_amp, ac_syms, ac_amp


def _decode_plane(dc_syms, ac_syms, dc_bits, ac_bits, table,
                  n_blocks: int, H: int, W: int) -> np.ndarray:
    blocks = np.zeros((n_blocks, 8, 8), dtype=np.float64)
    prev_dc = 0
    dc_iter = iter(dc_syms)
    dc_bit_iter = iter(dc_bits)
    ac_bit_iter = iter(ac_bits)
    ac_sym_iter = iter(ac_syms)
    for bi in range(n_blocks):
        sz = next(dc_iter)
        bits = 0
        for _ in range(sz):
            bits = (bits << 1) | next(dc_bit_iter)
        diff = _value_from_bits(bits, sz)
        dc = diff + prev_dc
        prev_dc = dc
        ac_vals = _unrle_ac(ac_sym_iter, ac_bit_iter)
        zz = np.concatenate([[dc], ac_vals])
        q = unzigzag(zz)
        blocks[bi] = idct2(dequantize(q, table))
    return _plane_from_blocks(blocks, H, W)


def encode_image(img: np.ndarray, quality: float) -> bytes:
    """Encode an image to bytes.  Grayscale (H, W) or RGB (H, W, 3),
    float values in 0..255.  Colour goes through YCbCr with 4:2:0
    chroma subsampling (JPEG's choice: the eye is less sensitive to
    colour detail).

    quality is clamped to an integer in [1, 100] (JPEG semantics) so
    that the quantisation table used matches the value stored in the
    header - a fractional quality would silently desynchronise the
    encoder and decoder tables.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3):
        raise ValueError("encode_image expects (H, W) or (H, W, 3)")
    if not np.isfinite(img).all():
        raise ValueError("encode_image requires finite pixel values")
    quality = int(quality)
    if quality < 1 or quality > 100:
        raise ValueError("quality must be in (0, 100]")

    n_channels = 1 if img.ndim == 2 else 3
    if n_channels == 1:
        planes = [img]
    else:
        planes = list(rgb_to_ycbcr(img))
        sub = subsample_chroma(np.stack(planes))
        planes[1] = sub[1]
        planes[2] = sub[2]

    # tables: luma for channel 0 (or the gray plane), chroma otherwise
    luma_table = scaled_table(JPEG_LUMA_Q, quality)
    chroma_table = scaled_table(JPEG_CHROMA_Q, quality)

    header = b"MCJP" + struct.pack(">IIBB", img.shape[1], img.shape[0],
                                   int(quality), n_channels)
    body = b""
    for ci, plane in enumerate(planes):
        table = luma_table if ci == 0 else chroma_table
        dc_syms, dc_amp, ac_syms, ac_amp = _encode_plane(plane, table)
        dc_huff = Huffman.from_frequencies(_freq(dc_syms))
        ac_huff = Huffman.from_frequencies(_freq(ac_syms))
        # per-channel layout: tables immediately followed by the
        # channel's stream (decoder reads them interleaved)
        body += _table_bytes(dc_huff)
        body += _table_bytes(ac_huff)
        dc_bytes, _ = dc_huff.encode_to_bytes(dc_syms)
        ac_bytes, _ = ac_huff.encode_to_bytes(ac_syms)
        ac_sizes = _ac_sizes(ac_syms)
        body += (struct.pack(">IIII", len(dc_syms), len(dc_bytes),
                             len(ac_syms), len(ac_bytes))
                 + dc_bytes + dc_amp_bytes(dc_amp, dc_syms)
                 + ac_bytes + ac_amp_bytes(ac_amp, ac_sizes))
    return header + body


def _ac_sizes(ac_syms: list[int]) -> list[int]:
    """Sizes of the amplitude bits, aligned with `ac_amp`: one entry
    per symbol with size > 0 (EOB/ZRL carry no amplitude)."""
    return [divmod(s, 16)[1] for s in ac_syms if divmod(s, 16)[1] > 0]


def dc_amp_bytes(amps: list[int], sizes: list[int]) -> bytes:
    """Pack DC amplitude bits back-to-back into bytes (exactly `size`
    bits per amplitude; negative amplitudes need their pad bits)."""
    return _pack_bits(amps, sizes)


def ac_amp_bytes(amps: list[int], sizes: list[int]) -> bytes:
    """Pack AC amplitude bits.  `sizes` is aligned with the symbols that
    carry amplitudes (EOB/ZRL contribute nothing)."""
    return _pack_bits(amps, sizes)


def _pack_bits(amps: list[int], sizes: list[int]) -> bytes:
    bits: list[int] = []
    for a, size in zip(amps, sizes):
        for shift in range(size - 1, -1, -1):
            bits.append((a >> shift) & 1)
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        out[i >> 3] |= b << (7 - (i & 7))
    return bytes(out)


def _unpack_bits(data: bytes, total: int) -> list[int]:
    bits = [(byte >> shift) & 1
            for byte in data
            for shift in range(7, -1, -1)]
    return bits[:total]


def decode_image(data: bytes) -> tuple[np.ndarray, float]:
    """Decode MCJP bytes back to a float image (0..255).  Grayscale
    (H, W) or RGB (H, W, 3), matching the encoded shape.

    Returns (image, quality).  The image is cropped back to the
    original size recorded in the header.
    """
    if data[:4] != b"MCJP":
        raise ValueError("not an MCJP file")
    w, h, quality, n_channels = struct.unpack(">IIBB", data[4:14])
    pos = 14

    H = ((h + 7) // 8) * 8
    W = ((w + 7) // 8) * 8
    luma_table = scaled_table(JPEG_LUMA_Q, quality)
    chroma_table = scaled_table(JPEG_CHROMA_Q, quality)

    planes = []
    for ci in range(n_channels):
        table = luma_table if ci == 0 else chroma_table
        dc_huff, pos = _read_table(data, pos)
        ac_huff, pos = _read_table(data, pos)
        n_dc, n_dc_bytes, n_ac, n_ac_bytes = struct.unpack(">IIII", data[pos:pos + 16])
        pos += 16
        dc_syms = dc_huff.decode_from_bytes(data[pos:pos + n_dc_bytes], n_dc)
        pos += n_dc_bytes
        n_dc_amp = int(sum(dc_syms))
        n_dc_amp_bytes = (n_dc_amp + 7) // 8
        dc_bits = _unpack_bits(data[pos:pos + n_dc_amp_bytes], n_dc_amp)
        pos += n_dc_amp_bytes

        ac_syms = ac_huff.decode_from_bytes(data[pos:pos + n_ac_bytes], n_ac)
        pos += n_ac_bytes
        n_ac_amp = _count_ac_amplitude_bits(ac_syms)
        n_ac_amp_bytes = (n_ac_amp + 7) // 8
        ac_bits = _unpack_bits(data[pos:pos + n_ac_amp_bytes], n_ac_amp)
        pos += n_ac_amp_bytes

        n_blocks = len(dc_syms)
        plane = _decode_plane(dc_syms, ac_syms, dc_bits, ac_bits, table,
                              n_blocks, H, W)
        planes.append(plane)

    if n_channels == 1:
        return np.clip(planes[0], 0, 255)[:h, :w], quality
    rgb = ycbcr_to_rgb(np.stack(planes))
    return rgb[:h, :w], quality


def _rle_ac(ac: np.ndarray, syms: list[int], amps: list[int]) -> None:
    """Append (run,size) symbols + raw amplitude bits for a block's AC."""
    run = 0
    for v in ac:
        v = int(v)
        if v == 0:
            run += 1
            continue
        while run > 15:
            syms.append(16 * 16 + 0)  # ZRL: 16 zeros, no amplitude
            run -= 16
        size = _category(v)
        syms.append(run * 16 + size)
        amps.append(_amplitude_bits(v, size))
        run = 0
    syms.append(0)  # EOB


def _unrle_ac(sym_iter, bit_iter) -> np.ndarray:
    """Decode one block's (run,size) symbols + amplitude bits to 63
    values.  Reads from the shared symbol iterator until EOB.

    A malformed stream whose run+size overruns the 63 AC coefficients
    is rejected with ValueError rather than silently truncated — a
    corrupt file should fail loudly, not produce a wrong block."""
    vals = np.zeros(63, dtype=np.int64)
    idx = 0
    for sym in sym_iter:
        if sym == 0:  # EOB
            break
        run, size = divmod(sym, 16)
        if size == 0:  # ZRL: 16 zeros
            idx += 16
            if idx > 63:
                raise ValueError("malformed AC stream: ZRL overruns block")
            continue
        bits = 0
        for _ in range(size):
            bits = (bits << 1) | next(bit_iter)
        idx += run
        if idx >= 63:
            raise ValueError("malformed AC stream: run overruns block")
        vals[idx] = _value_from_bits(bits, size)
        idx += 1
    return vals


def _count_ac_amplitude_bits(syms: list[int]) -> int:
    return sum(divmod(s, 16)[1] for s in syms if s != 0)


def _freq(syms: list[int]) -> dict[int, int]:
    from collections import Counter
    return dict(Counter(syms))


def quant_table(quality: float) -> np.ndarray:
    from quant import JPEG_LUMA_Q, scaled_table
    return scaled_table(JPEG_LUMA_Q, quality)


def _table_bytes(huff: Huffman) -> bytes:
    syms = sorted(huff.code_lengths.items())
    out = struct.pack(">I", len(syms))
    for sym, length in syms:
        out += struct.pack(">iB", sym, length)
    return out


def _read_table(data: bytes, pos: int) -> tuple[Huffman, int]:
    n, = struct.unpack(">I", data[pos:pos + 4])
    pos += 4
    lengths = {}
    for _ in range(n):
        sym, length = struct.unpack(">iB", data[pos:pos + 5])
        lengths[sym] = length
        pos += 5
    return Huffman(lengths), pos
