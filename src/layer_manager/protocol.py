# ./src/layer_manager/protocol.py
"""Structural interfaces (``typing.Protocol``) for the simulator's layers.

Each network layer is assembled from interchangeable *strategies* that are
picked at runtime from the GUI configuration (which framing scheme, which
error-control scheme, which modulation). These protocols describe the shape
every strategy must have, so the simulator can hold a strategy by its
interface without caring about the concrete class behind it.

They are structural types: a class satisfies a protocol simply by providing
matching methods, with no explicit inheritance required. Concrete
implementations live in the physical- and link-layer modules of this package.

Pipeline composition (transmitter side, top of the stack first)::

    text --> bits --> [ErrorCorrector/ErrorDetector] --> [Framer]
         --> [Modulator] --> signal --> [Channel] --> ...

The receiver applies the inverse of each step in reverse order.
"""

from typing import Protocol, runtime_checkable

from layer_manager.types import Bits, Signal


@runtime_checkable
class Framer(Protocol):
    """Delimits payloads into frames so the receiver can find boundaries.

    Implementations: character count, byte/flag stuffing, bit stuffing.
    """

    def frame(self, payloads: list[Bits]) -> Bits:
        """Delimit and concatenate payloads into one transmittable stream.

        Args:
            payloads: The per-frame payloads, already split to the maximum
                frame size and protected by the error-control layer.

        Returns:
            A single bit stream containing every framed payload.
        """
        ...

    def deframe(self, stream: Bits) -> list[Bits]:
        """Recover the original payloads from a received framed stream.

        Args:
            stream: The bit stream produced by :meth:`frame`, possibly
                altered by channel noise.

        Returns:
            The extracted payloads, one entry per recovered frame.
        """
        ...


@runtime_checkable
class ErrorDetector(Protocol):
    """Appends an error-detecting code and verifies it on receipt.

    Implementations: even parity bit, checksum, CRC-32 (IEEE 802).
    """

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its error-detecting code appended."""
        ...

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Validate a received block and strip its error-detecting code.

        Args:
            data: The received bits, code included.

        Returns:
            A tuple ``(payload, ok)`` where ``payload`` is ``data`` without
            the code and ``ok`` is ``True`` when no error was detected.
        """
        ...


@runtime_checkable
class ErrorCorrector(Protocol):
    """Adds redundancy that lets the receiver repair bit errors.

    Implementation: Hamming code.
    """

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with correction (parity) bits interleaved."""
        ...

    def decode(self, data: Bits) -> tuple[Bits, int]:
        """Repair and strip the redundancy from a received block.

        Args:
            data: The received bits, redundancy included.

        Returns:
            A tuple ``(payload, corrected)`` where ``payload`` is the
            recovered data and ``corrected`` is the number of bit errors
            that were fixed.
        """
        ...


@runtime_checkable
class Modulator(Protocol):
    """Maps bits to a signal and back.

    Covers both baseband encoders (NRZ-Polar, Manchester, Bipolar) and
    carrier modulators (ASK, FSK, QPSK, 16-QAM): they share this interface.
    """

    def modulate(self, bits: Bits) -> Signal:
        """Convert a bit sequence into signal samples (V / W)."""
        ...

    def demodulate(self, signal: Signal) -> Bits:
        """Recover the bit sequence from (possibly noisy) signal samples."""
        ...


@runtime_checkable
class Channel(Protocol):
    """The communication medium between transmitter and receiver."""

    def transmit(self, signal: Signal) -> Signal:
        """Return ``signal`` after the medium acts on it.

        For this project the medium adds zero-mean Gaussian noise
        ``n(x, sigma)`` to the electrical (V / W) sample values.
        """
        ...
