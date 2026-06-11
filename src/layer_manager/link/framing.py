# ./src/layer_manager/link/framing.py
"""Framing schemes for the data-link layer.

Each class satisfies :class:`layer_manager.protocol.Framer`: it concatenates
payloads into a delimited bit stream and recovers them on the other side.
See the "Enquadramento" reference material for the exact field layouts.
"""

from layer_manager.protocol import Framer
from layer_manager.types import Bits


class CharCountFramer(Framer):
    """Prefixes each frame with a header counting its characters/bytes."""

    def frame(self, payloads: list[Bits]) -> Bits:
        """Prepend a length header to each payload and concatenate."""
        raise NotImplementedError

    def deframe(self, stream: Bits) -> list[Bits]:
        """Read each length header to slice the stream back into payloads."""
        raise NotImplementedError


class ByteStuffingFramer(Framer):
    """Delimits frames with FLAG bytes, escaping FLAGs inside the payload."""

    def frame(self, payloads: list[Bits]) -> Bits:
        """Escape payload FLAG/ESC bytes and wrap each frame in FLAG bytes."""
        raise NotImplementedError

    def deframe(self, stream: Bits) -> list[Bits]:
        """Split on FLAG bytes and undo the byte stuffing."""
        raise NotImplementedError


class BitStuffingFramer(Framer):
    """Delimits frames with a FLAG bit pattern, stuffing bits to avoid it."""

    def frame(self, payloads: list[Bits]) -> Bits:
        """Insert a 0 after each run of five 1s and wrap with FLAG patterns."""
        raise NotImplementedError

    def deframe(self, stream: Bits) -> list[Bits]:
        """Split on FLAG patterns and remove the stuffed bits."""
        raise NotImplementedError
