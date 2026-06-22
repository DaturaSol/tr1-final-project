# ./src/layer_manager/link/framing.py
"""Framing schemes for the data-link layer.

Each class satisfies :class:`layer_manager.protocol.Framer`: it concatenates
payloads into a delimited bit stream and recovers them on the other side.
See the "Enquadramento" reference material for the exact field layouts.
"""

from itertools import chain

import numpy as np
import numpy.typing as npt

from layer_manager.protocol import Framer
from layer_manager.types import Bits


class FramerBase(Framer):
    """Shared base for the concrete framers.

    Mirrors the physical layer's ``CarrierModulator``/``BasebandModulator``:
    it declares the :class:`~layer_manager.protocol.Framer` interface for
    subclasses to override and holds the bit/integer conversion helpers they
    reuse.
    """

    def frame(self, payloads: list[Bits]) -> Bits:
        """Delimit and concatenate ``payloads`` into one bit stream."""
        raise NotImplementedError

    def deframe(self, stream: Bits) -> list[Bits]:
        """Recover the original payloads from a framed ``stream``."""
        raise NotImplementedError

    @staticmethod
    def _encode(count: list[int]) -> npt.NDArray[np.bool_]:
        """Return ``count`` as a fixed-width bool array, MSB first."""
        return np.unpackbits(np.array(count, dtype=np.uint8)).astype(dtype=bool)

    @staticmethod
    def _decode(bits: Bits) -> npt.NDArray[np.uint8]:
        """Return the integer represented by the fixed-width bool array."""
        return np.packbits(np.array(bits, dtype=np.uint8))


class CharCountFramer(FramerBase):
    """Prefixes each frame with a header counting its bytes.

    Each frame is ``[count header][payload]``: a fixed-width header states how
    many bytes the payload holds, so the receiver reads a count, takes that
    many bytes, and repeats. Payloads are byte-aligned ``Bits`` (length a
    multiple of 8, MSB first), so the count is in bytes.

    Design choice: the count covers the payload bytes only (the header is
    separate) and is ``HEADER_BITS`` wide. Known weakness to note: a single
    corrupted count desynchronises every following frame.
    """

    HEADER_BITS = 8  # one byte -> counts up to 255 payload bytes per frame

    def frame(self, payloads: list[Bits]) -> Bits:
        """Prepend a length header to each payload and concatenate.

        Steps:
          1. For each ``payload``, compute its length in bytes
             (``len(payload) // 8``).
          2. Encode that count as ``HEADER_BITS`` bools, MSB first.
          3. Concatenate ``header + payload`` for every frame into one stream.
        """
        new_payload: list[Bits] = [
            self._encode([len(frame) // 8]).tolist() + frame
            for frame in payloads
        ]
        bits: Bits = [bits for sublist in new_payload for bits in sublist]
        return bits

    def deframe(self, stream: Bits) -> list[Bits]:
        """Read each length header to slice the stream back into payloads.

        Inverse of :meth:`frame`. Steps:
          1. Walk the stream with a cursor starting at 0.
          2. Decode ``HEADER_BITS`` bits (MSB first) into ``count`` bytes.
          3. Take the next ``count * 8`` bits as the payload.
          4. Advance past header + payload; repeat until the stream is consumed.
        """
        cursor = 0

        def take(n: int) -> Bits:
            nonlocal cursor
            chunk = stream[cursor : cursor + n]
            cursor += n
            return chunk

        payloads: list[Bits] = []
        while cursor < len(stream):
            count = int(self._decode(take(self.HEADER_BITS))[0])
            payloads.append(take(count * 8))
        return payloads


class ByteStuffingFramer(FramerBase):
    """Delimits frames with FLAG bytes, escaping FLAG/ESC bytes in the payload.

    Each frame is ``FLAG | stuffed(payload) | FLAG``. Inside the payload any
    byte equal to FLAG or ESC is prefixed with an ESC byte, so a bare FLAG in
    the stream is always a real delimiter. Payloads are byte-aligned, so
    stuffing operates on whole bytes (convert with np.packbits/np.unpackbits).

    Design choice: simple insertion (the escaped byte is left unchanged, not
    XOR'd like PPP); FLAG/ESC are the HDLC-style 0x7E / 0x7D.
    """

    FLAG = 0x7E  # frame delimiter byte (0b0111_1110)
    ESC = 0x7D  # escape byte (0b0111_1101)

    def frame(self, payloads: list[Bits]) -> Bits:
        """Escape payload FLAG/ESC bytes and wrap each frame in FLAG bytes.

        Each frame is: opening FLAG, the payload bytes (with FLAG/ESC bytes
        escaped by a preceding ESC), then a closing FLAG.
        """
        stuffed: list[int] = []
        for payload in payloads:
            data = self._decode(payload)
            needs_esc = (data == self.FLAG) | (
                data == self.ESC
            )  # data.shape = (n,)

            pairs = np.empty((len(data), 2), dtype=data.dtype)  # (n, 2)
            pairs[:, 0] = self.ESC
            pairs[:, 1] = data
            # Ravel is faster than flatten and we don't need
            # a copy since we'll slice it right away;
            flat = pairs.ravel()  # (2n,)

            keep = np.ones(len(flat), dtype=bool)  # (2n,)
            keep[0::2] = (
                needs_esc  # keep the ESC only if the byte needs escaping
            )
            body = flat[keep]  # (n + number of bytes that need escaping,)

            stuffed.append(self.FLAG)
            stuffed.extend(body.tolist())
            stuffed.append(self.FLAG)

        bits: Bits = self._encode(stuffed).tolist()
        return bits

    def deframe(self, stream: Bits) -> list[Bits]:
        """Split on FLAG bytes and undo the byte stuffing.

        Inverse of :meth:`frame`. A bare FLAG opens or closes a frame; inside a
        frame an ESC marks the next byte as literal data (the ESC is dropped).
        """
        payloads: list[Bits] = []
        current: list[int] = []
        in_frame = False
        escape_next = False

        for byte in self._decode(stream).tolist():
            if not in_frame:
                if byte == self.FLAG:  # opening FLAG
                    in_frame = True
                continue

            if escape_next:
                current.append(byte)
                escape_next = False
            elif byte == self.ESC:
                escape_next = True
            elif byte == self.FLAG:  # closing FLAG
                payloads.append(self._encode(current).tolist())
                current = []
                in_frame = False
            else:
                current.append(byte)

        return payloads


class BitStuffingFramer(FramerBase):
    """Delimits frames with a FLAG bit pattern, stuffing bits to avoid it."""

    FLAG = 0x7E  # frame delimiter byte (0b0111_1110)

    def frame(self, payloads: list[Bits]) -> Bits:
        """Stuff a 0 after every run of five 1s; wrap each frame in FLAGs."""
        flag: Bits = self._encode([self.FLAG]).tolist()
        stream: Bits = []
        for payload in payloads:
            stream += flag  # opening FLAG
            ones = 0
            for bit in payload:
                stream.append(bit)
                if bit:
                    ones += 1
                    if ones == 5:
                        stream.append(False)  # insert stuffing 0
                        ones = 0
                else:
                    ones = 0
            stream += flag  # closing FLAG
        return stream

    def deframe(self, stream: Bits) -> list[Bits]:
        """Split on FLAG patterns and remove stuffed bits (inverse of frame)."""
        bits = np.array(stream, dtype=bool)
        flag_bits = np.asarray(self._encode([self.FLAG]), dtype=bool)
        flag_len = len(flag_bits)

        if bits.size < flag_len:  # empty / too-short stream: no frames
            return []

        # Locate FLAG patterns via a sliding-window equality check.
        windows = np.lib.stride_tricks.sliding_window_view(
            bits, flag_len
        )  #  (N - flag_len + 1, flag_len)
        matches = (windows == flag_bits).all(axis=1)  # (n_windows,) bool
        flag_indices = np.where(matches)[0]  # (n_flags,) indices of FLAG starts

        if len(flag_indices) % 2 != 0:
            raise ValueError(
                "Invalid frame structure: unbalanced FLAG patterns"
            )

        payloads: list[Bits] = []
        for i in range(0, len(flag_indices), 2):
            start = flag_indices[i] + flag_len
            end = flag_indices[i + 1]
            payloads.append(self._unstuff(bits[start:end]))
        return payloads

    @staticmethod
    def _unstuff(payload: npt.NDArray[np.bool_]) -> Bits:
        """Drop the 0 that follows each run of five consecutive 1s."""
        out: list[bool] = []
        ones = 0
        it = iter(payload.tolist())
        for bit in it:
            out.append(bit)
            if bit:
                ones += 1
                if ones == 5:
                    next(it, None)  # skip the stuffed 0
                    ones = 0
            else:
                ones = 0
        return out


def chunk(bits: Bits, max_frame_size: int) -> list[Bits]:
    """Split a flat bit stream into frames of at most max_frame_size bytes.

    Args:
        bits: The application bit stream (length a multiple of 8).
        max_frame_size: Maximum payload per frame, in bytes.

    Returns:
        The per-frame payloads, in order; the last may be shorter. Empty
        input gives an empty list (no frames).
    """
    step = max_frame_size * 8  # bytes -> bits per full frame
    return [bits[i : i + step] for i in range(0, len(bits), step)]


def join(frames: list[Bits]) -> Bits:
    """Concatenate recovered frame payloads back into one bit stream.

    Inverse of :func:`chunk` (after framing/EDC have been stripped).
    """
    return list(chain.from_iterable(frames))
