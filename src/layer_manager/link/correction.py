# ./src/layer_manager/link/correction.py
"""Error-correction schemes for the data-link layer.

Satisfies :class:`layer_manager.protocol.ErrorCorrector`.
"""

from layer_manager.protocol import ErrorCorrector
from layer_manager.types import Bits


class HammingCorrector(ErrorCorrector):
    """Hamming single-error correction (SEC).

    Interleaves parity bits at the power-of-two positions (1, 2, 4, 8, ...) of
    a 1-indexed codeword; the data fills the remaining positions. Each parity
    bit ``2^i`` covers exactly the positions whose index has bit ``i`` set, so
    on receipt the recomputed parities -- read as a binary number -- spell out
    the **position of the flipped bit** (the *syndrome*); 0 means no error.

    The whole payload is treated as one block: ``r`` parity bits are added with
    ``2^r >= m + r + 1`` for ``m`` data bits, enough to address every position
    plus the no-error case. That corrects **one** bit error per block; a second
    error in the same block is mislocated and "corrected" wrongly (this is SEC,
    not SECDED), so pair it with a detector (e.g. CRC) to catch that case.
    """

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with Hamming parity bits interleaved.

        Picks the smallest ``r`` with ``2^r >= m + r + 1``, lays the data into
        the non-power-of-two positions of an ``m + r`` bit codeword, then sets
        each power-of-two parity bit to even parity over the positions it
        covers.

        Args:
            data: The payload bits to protect.

        Returns:
            The Hamming codeword: ``data`` with ``r`` parity bits interleaved.
        """
        m = len(data)
        # Find r such that the codeword length m + r fits: 2^r >= m + r + 1.
        r = 0
        while (1 << r) < m + r + 1:
            r += 1
        n = m + r

        # Place data bits into non-power-of-two positions (1-indexed).
        code = [False] * (n + 1)  # index 0 unused
        di = 0
        for pos in range(1, n + 1):
            if not self._is_power_of_two(pos):
                code[pos] = data[di]
                di += 1

        # Each parity bit at 2^i is even parity over the positions (other than
        # itself) whose index has bit i set, which are all data positions.
        for i in range(r):
            p = 1 << i
            parity = False
            for pos in range(1, n + 1):
                # pos & p is True if pos has bit p,
                # pos != p. Excludes itselt.
                if pos & p and pos != p:
                    parity ^= code[pos]
            code[p] = parity

        return code[1:]  # Ignores idx 0.

    def decode(self, data: Bits) -> tuple[Bits, bool]:
        """Locate and flip a corrupted bit, then strip the parity bits.

        Computes the syndrome (XOR of the indices of every set bit): 0 for an
        intact codeword, otherwise the position of the single flipped bit,
        which is corrected in place.

        Args:
            data: A received Hamming codeword (as produced by :meth:`encode`).

        Returns:
            ``(payload, corrected)`` where ``payload`` is the data bits with the
            parity bits removed and ``corrected`` is ``True`` when a single-bit
            error was located and repaired, ``False`` otherwise.
        """
        n = len(data)
        code = [False, *data]  # shift to 1-indexed; index 0 unused

        # Syndrome = XOR of the indices of all set bits. Each bit i of the
        # syndrome is the parity of group 2^i, so an intact codeword gives 0.
        syndrome = 0  # Will point to the flipped bit.
        for pos in range(1, n + 1):
            if code[pos]:
                syndrome ^= pos

        corrected = False
        if 0 < syndrome <= n:  # a locatable single-bit error
            code[syndrome] = not code[syndrome]  # flip it back
            corrected = True

        # Strip parity bits (power-of-two positions) to recover the payload.
        payload = [
            code[pos]
            for pos in range(1, n + 1)
            if not self._is_power_of_two(pos)
        ]
        return payload, corrected

    @staticmethod
    def _is_power_of_two(pos: int) -> bool:
        """Return whether ``pos`` is a power of two (a parity position)."""
        return pos > 0 and (pos & (pos - 1)) == 0
