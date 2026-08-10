"""Huffman coding from scratch.

Huffman's algorithm builds a prefix-free code that is optimal for the
given symbol frequencies (a binary tree where each merge combines the
two rarest subtrees).  We then convert the tree into *canonical* codes:
the code lengths are kept from the tree, but the bit patterns are
assigned canonically (shorter codes first, lexicographic within a
length).  Canonical codes are what make decoding and header storage
simple and compact.

Decoding walks a binary tree built from the canonical lengths.
"""

from __future__ import annotations

import heapq
from collections import Counter


class Huffman:
    """Encode/decode a symbol stream with a Huffman code.

    Usage:
        h = Huffman.from_frequencies(Counter(symbols))
        bits = h.encode(symbols)      # list of ints (0/1)
        out = h.decode(bits)
    """

    def __init__(self, code_lengths: dict[int, int]):
        """Build canonical codes from {symbol: code length}."""
        self.code_lengths = dict(code_lengths)
        self.symbol_to_code: dict[int, tuple[int, int]] = {}
        self._tree: dict = {}
        self._build_canonical()
        self._build_tree()

    @classmethod
    def from_frequencies(cls, freqs: dict[int, int]) -> "Huffman":
        """Run Huffman's algorithm on a frequency table."""
        heap: list[tuple[int, int, object]] = []
        counter = 0  # tie-breaker so the heap never compares leaves
        for sym, f in freqs.items():
            if f <= 0:
                continue
            counter += 1
            heap.append((f, counter, ("leaf", sym)))
        heapq.heapify(heap)
        while len(heap) > 1:
            f1, c1, n1 = heapq.heappop(heap)
            f2, c2, n2 = heapq.heappop(heap)
            counter += 1
            heapq.heappush(heap, (f1 + f2, counter, ("node", n1, n2)))
        # collect lengths by walking the tree
        lengths: dict[int, int] = {}
        root = heap[0][2] if heap else ("leaf", None)

        def walk(node, depth):
            kind = node[0]
            if kind == "leaf":
                lengths[node[1]] = max(depth, 1)  # a code must be >= 1 bit
            else:
                walk(node[1], depth + 1)
                walk(node[2], depth + 1)

        walk(root, 0)
        return cls(lengths)

    def _build_canonical(self) -> None:
        """Kraft-compliant canonical assignment:
        sort symbols by (length, symbol); shorter codes first; within a
        length, incrementing counters."""
        syms = sorted(self.code_lengths.items(), key=lambda kv: (kv[1], kv[0]))
        code = 0
        prev_len = 0
        for sym, length in syms:
            code <<= length - prev_len
            self.symbol_to_code[sym] = (code, length)
            code += 1
            prev_len = length

    def _build_tree(self) -> None:
        """Decoding tree from canonical lengths: at depth d, the symbols
        of that length occupy consecutive canonical codes."""
        by_len: dict[int, list[tuple[int, int]]] = {}
        for sym, (code, length) in self.symbol_to_code.items():
            by_len.setdefault(length, []).append((code, sym))
        for v in by_len.values():
            v.sort()

        self._tree = {}  # path of bits -> symbol
        for length, pairs in by_len.items():
            for code, sym in pairs:
                node = self._tree
                for shift in range(length - 1, -1, -1):
                    bit = (code >> shift) & 1
                    node = node.setdefault(bit, {})
                node["$"] = sym

    # ------------------------------------------------------------------

    def encode(self, symbols) -> list[int]:
        return [b for s in symbols for b in self._bits(s)]

    def encode_to_bytes(self, symbols) -> tuple[bytes, int]:
        """Pack bits into bytes; returns (bytes, padding bits in last
        byte).  The padding is all zeros and is ignored by decode."""
        bits = self.encode(symbols)
        if not bits:
            return b"", 0
        nbytes = (len(bits) + 7) // 8
        buf = bytearray(nbytes)
        for i, b in enumerate(bits):
            buf[i >> 3] |= b << (7 - (i & 7))
        return bytes(buf), (8 - len(bits) % 8) % 8

    def _bits(self, symbol: int) -> list[int]:
        code, length = self.symbol_to_code[symbol]
        return [(code >> shift) & 1 for shift in range(length - 1, -1, -1)]

    def decode(self, bits) -> list[int]:
        out = []
        node = self._tree
        for b in bits:
            node = node[b]
            if "$" in node:
                out.append(node["$"])
                node = self._tree
        if node is not self._tree:
            raise ValueError("incomplete code at end of stream")
        return out

    def decode_from_bytes(self, data: bytes, n_symbols: int) -> list[int]:
        bits = [(byte >> shift) & 1
                for byte in data
                for shift in range(7, -1, -1)]
        out = []
        node = self._tree
        for b in bits:
            node = node[b]
            if "$" in node:
                out.append(node["$"])
                node = self._tree
                if len(out) == n_symbols:
                    return out
        raise ValueError("stream ended before all symbols decoded")

    # ------------------------------------------------------------------

    @staticmethod
    def average_length(freqs: dict[int, int]) -> float:
        """Mean code length weighted by frequency (bits per symbol)."""
        total = sum(freqs.values())
        if total == 0:
            return 0.0
        # Huffman length is within 1 bit of the entropy lower bound
        lengths = Huffman.from_frequencies(freqs).code_lengths
        return sum(freqs[s] * lengths[s] for s in freqs) / total

    @staticmethod
    def entropy(freqs: dict[int, int]) -> float:
        """Shannon entropy in bits: the theoretical lower bound."""
        import math
        total = sum(freqs.values())
        return -sum((f / total) * math.log2(f / total)
                    for f in freqs.values() if f > 0)
