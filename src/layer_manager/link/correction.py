# ./src/layer_manager/link/correction.py
"""Error-correction schemes for the data-link layer.

Satisfies :class:`layer_manager.protocol.ErrorCorrector`.
"""

from layer_manager.protocol import ErrorCorrector
from layer_manager.types import Bits


class HammingCorrector(ErrorCorrector):
    """Hamming code: interleaves parity bits to fix single-bit errors."""

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with Hamming parity bits interleaved."""
        raise NotImplementedError

    def decode(self, data: Bits) -> tuple[Bits, int]:
        """Locate and flip a corrupted bit, then strip the parity bits."""
        raise NotImplementedError
