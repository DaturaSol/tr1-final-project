# ./src/layer_manager/link/detection.py
"""Error-detection schemes for the data-link layer.

Each class satisfies :class:`layer_manager.protocol.ErrorDetector`. None of
these may rely on an external library for the calculation itself (e.g. ``zlib``
for CRC); the algorithm must be implemented here.
"""

from layer_manager.protocol import ErrorDetector
from layer_manager.types import Bits


class DetectorBase(ErrorDetector):
    """Shared base for the concrete error detectors.

    Mirrors framing's ``FramerBase``: it declares the
    :class:`~layer_manager.protocol.ErrorDetector` interface for subclasses to
    override and holds the bit/integer conversion helpers shared by the
    checksum and CRC schemes.
    """

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its error-detecting code appended."""
        raise NotImplementedError

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Validate a received block and strip its error-detecting code."""
        raise NotImplementedError

    @staticmethod
    def _int_to_bits(value: int, width: int) -> Bits:
        """Write ``value`` as ``width`` bits, MSB first."""
        return [bool((value >> (width - 1 - i)) & 1) for i in range(width)]

    @staticmethod
    def _bits_to_int(bits: Bits) -> int:
        """Read a bit list (MSB first) as an unsigned integer."""
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value


class ParityDetector(DetectorBase):
    """Appends one even-parity bit per block.

    Even parity: the appended bit is chosen so the **total** number of ``1``
    bits in the block (payload + parity bit) is even. On receipt, a block whose
    ``1`` count is odd must have been corrupted.

    This is the cheapest detector and also the weakest: it only catches an
    **odd** number of bit flips. Two flips (or any even number) leave the
    parity even and slip through undetected. Note it appends a single bit, so
    the result is no longer byte-aligned.
    """

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with an even-parity bit appended.

        Args:
            data: The payload bits to protect.

        Returns:
            ``data`` followed by one parity bit: ``1`` when ``data`` has an odd
            number of ones (to even it out), ``0`` otherwise.
        """
        parity = sum(data) % 2
        bits: Bits = [*data, bool(parity)]
        return bits

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Verify the parity bit and strip it from ``data``.

        Args:
            data: A received block, parity bit included (as produced by
                :meth:`encode`).

        Returns:
            ``(payload, ok)`` where ``payload`` is ``data`` without its trailing
            parity bit and ``ok`` is ``True`` when the block's total ``1`` count
            is even (no odd-sized error detected).
        """
        ok = sum(data) % 2 == 0
        return data[:-1], ok


class ChecksumDetector(DetectorBase):
    """Appends the one's-complement checksum of fixed-size blocks.

    The Internet-checksum scheme shown in class: split the data into
    ``block_bits``-wide blocks, add them with **end-around carry** (one's-
    complement addition), and append the one's complement of that sum. On
    receipt, summing every block *including* the checksum gives all ones when
    the data is intact. Catches more than parity (it sees the magnitude of a
    change), but reordered blocks or offsetting errors can still cancel out.
    """

    def __init__(self, block_bits: int) -> None:
        """Store the block size used to split the data.

        Args:
            block_bits: Width, in bits, of each block summed for the checksum.
        """
        self.block_bits = block_bits

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its block checksum appended.

        Args:
            data: The payload bits to protect.

        Returns:
            ``data`` followed by the ``block_bits``-wide one's-complement
            checksum of its blocks.
        """
        total = self._ones_complement_sum(self._blocks(data))
        mask = (1 << self.block_bits) - 1  # 2^block_bits - 1
        checksum = ~total & mask  # one's complement, kept to block_bits
        return [*data, *self._int_to_bits(checksum, self.block_bits)]

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Recompute the checksum to validate and strip it from ``data``.

        Args:
            data: A received block, checksum included (as produced by
                :meth:`encode`).

        Returns:
            ``(payload, ok)`` where ``payload`` is ``data`` without its trailing
            ``block_bits`` checksum and ``ok`` is ``True`` when the blocks plus
            the checksum sum to all ones (no error detected).
        """
        payload = data[: -self.block_bits]
        received = self._bits_to_int(data[-self.block_bits :])
        total = self._ones_complement_sum([*self._blocks(payload), received])
        ok = total == (1 << self.block_bits) - 1  # data + complement == 1...1
        return payload, ok

    def _blocks(self, bits: Bits) -> list[int]:
        """Split ``bits`` into ``block_bits``-wide ints (last zero-padded)."""
        pad = (-len(bits)) % self.block_bits
        padded = [*bits, *([False] * pad)]  # zero padding for the last block
        width = self.block_bits
        return [
            self._bits_to_int(padded[i : i + width])
            for i in range(0, len(padded), width)
        ]

    def _ones_complement_sum(self, values: list[int]) -> int:
        """Add ``values`` with end-around carry, folded to ``block_bits``."""
        mask = (1 << self.block_bits) - 1
        total = sum(values)
        # ones complement addition:
        # fold every overflow bit back into the low bits
        while total > mask:
            total = (total & mask) + (total >> self.block_bits)
        return total


class CRC32Detector(DetectorBase):
    """Appends a CRC-32 (IEEE 802 / ISO-HDLC) remainder.

    Treats the bit stream as a polynomial over GF(2) and appends the remainder
    of dividing it by the CRC-32 generator; a received block is intact when its
    recomputed remainder matches. This is the strongest detector here -- it
    catches every burst error up to 32 bits, every odd number of bit flips, and
    the overwhelming majority of longer bursts.

    Uses the canonical CRC-32/ISO-HDLC parameters (as in Ethernet and zip):
    reflected generator ``0xEDB88320``, register preset to ``0xFFFFFFFF``, and a
    final XOR with ``0xFFFFFFFF``. With those, the standard check value of
    ``"123456789"`` is ``0xCBF43926``. The division is hand-rolled (no ``zlib``)
    and processed byte-by-byte to match the reflected convention.
    """

    # <https://media.neliti.com/media/publications/501671-analysis-and-design-of-crc-32-ieee-8023-d727547b.pdf>
    # _POLY = 11101101101110001000001100100000,
    # LSB first (reflected form of 0x04C11DB7)
    _POLY = 0xEDB88320  # reflected CRC-32 generator (IEEE 802 / ISO-HDLC)
    _INIT = 0xFFFFFFFF  # register preset (all ones)
    _XOROUT = 0xFFFFFFFF  # final inversion applied to the register
    _WIDTH = 32  # remainder width, in bits

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its 32-bit CRC remainder appended.

        Args:
            data: The payload bits to protect.

        Returns:
            ``data`` followed by the 32-bit CRC remainder, MSB first.
        """
        crc = self._crc(data)
        return list(data) + self._int_to_bits(crc, self._WIDTH)

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Recompute the CRC to validate and strip it from ``data``.

        Args:
            data: A received block, CRC included (as produced by
                :meth:`encode`).

        Returns:
            ``(payload, ok)`` where ``payload`` is ``data`` without its trailing
            32-bit CRC and ``ok`` is ``True`` when the recomputed CRC matches
            the received one. A block shorter than the CRC is rejected as
            ``(data, False)``.
        """
        if len(data) < self._WIDTH:
            return list(data), False
        payload = list(data[: -self._WIDTH])
        received = data[-self._WIDTH :]
        expected = self._int_to_bits(self._crc(payload), self._WIDTH)
        return payload, received == expected

    def _crc(self, data: Bits) -> int:
        """Compute the CRC-32 register over ``data`` (MSB-first bits).

        Packs the bits into bytes (MSB first), folds each through
        :meth:`_update_byte`, then applies the final XOR. A trailing partial
        byte is zero-padded on the right.

        Args:
            data: The bits to run the CRC over.

        Returns:
            The 32-bit CRC value.
        """
        crc = self._INIT
        bit_count = 0
        byte = 0
        for bit in data:
            byte = (byte << 1) | bit
            bit_count += 1
            # Upon a full byte, fold it into the CRC register
            # and reset for the next one.
            if bit_count == 8:
                crc = self._update_byte(crc, byte)
                byte = 0
                bit_count = 0
        # Fold a trailing partial byte, if any, padded with zeros on the right.
        if bit_count:
            byte <<= 8 - bit_count  # left-align a trailing partial byte
            crc = self._update_byte(crc, byte)
        # Apply inversion and bound to 32 bits.
        return crc ^ self._XOROUT

    def _update_byte(self, crc: int, byte: int) -> int:
        """Fold one byte into the running CRC register.

        The reflected generator and right-shifts already encode the bit
        reflection, so the byte is fed at its natural value (no reversal).

        Args:
            crc: The current 32-bit register.
            byte: The next message byte (0-255).

        Returns:
            The updated 32-bit register.
        """
        # Folds byte into the crc.
        crc ^= byte
        for _ in range(8):
            # if remainder is 1.
            if crc & 1:
                crc = (crc >> 1) ^ self._POLY  # Subtract the divisor poly.
            else:
                crc >>= 1
        # Mask back to 32 bits in case of Python's unbounded int.
        return crc & 0xFFFFFFFF
