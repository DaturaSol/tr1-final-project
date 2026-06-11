# ./src/layer_manager/link/detection.py
"""Error-detection schemes for the data-link layer.

Each class satisfies :class:`layer_manager.protocol.ErrorDetector`. None of
these may rely on an external library for the calculation itself (e.g. ``zlib``
for CRC); the algorithm must be implemented here.
"""

from layer_manager.protocol import ErrorDetector
from layer_manager.types import Bits


class ParityDetector(ErrorDetector):
    """Appends one even-parity bit per block."""

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with an even-parity bit appended."""
        raise NotImplementedError

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Verify the parity bit and strip it from ``data``."""
        raise NotImplementedError


class ChecksumDetector(ErrorDetector):
    """Appends the one's-complement checksum of fixed-size blocks."""

    def __init__(self, block_bits: int) -> None:
        """Store the block size used to split the data.

        Args:
            block_bits: Width, in bits, of each block summed for the checksum.
        """
        self.block_bits = block_bits

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its block checksum appended."""
        raise NotImplementedError

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Recompute the checksum to validate and strip it from ``data``."""
        raise NotImplementedError


class CRC32Detector(ErrorDetector):
    """Appends a CRC-32 (IEEE 802) remainder."""

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its 32-bit CRC remainder appended."""
        raise NotImplementedError

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Divide by the CRC-32 polynomial to validate and strip the code."""
        raise NotImplementedError
