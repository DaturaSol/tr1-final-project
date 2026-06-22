# ./src/layer_manager/phy/baseband.py
"""Baseband (digital) line codes for the physical layer.

Each class satisfies :class:`layer_manager.protocol.Modulator`, mapping a bit
sequence to electrical samples (V) and back. See the "Modulação" slides for
the reference waveforms.

Every line code holds each bit for ``samples_per_symbol`` samples, so a message
of ``n`` bits becomes a signal of ``n * samples_per_symbol`` samples. The
transmitter and receiver stay aligned simply by sharing ``samples_per_symbol``
through the common configuration: the channel only adds amplitude noise, so the
sample index of each bit is preserved end to end and no clock recovery is
needed.
"""

import numpy as np
import numpy.typing as npt

from layer_manager.protocol import Modulator
from layer_manager.types import Bits, Signal


class BasebandModulator(Modulator):
    """Shared parameters for the baseband line codes below."""

    def __init__(self, amplitude: float, samples_per_symbol: int = 2) -> None:
        """Store the voltage level and per-bit sample count.

        Args:
            amplitude: Voltage level in volts; the sign convention is set by
                each concrete line code.
            samples_per_symbol: Discrete samples emitted per bit (>= 2).
        """
        self.amplitude = amplitude
        self.samples_per_symbol = samples_per_symbol

    def modulate(self, bits: Bits) -> Signal:
        """Return the line-coded samples for ``bits``."""
        # NOTE: bits.shape is (n,).
        raise NotImplementedError

    def demodulate(self, signal: Signal) -> Bits:
        """Recover the bit sequence from ``signal``."""
        # NOTE: signal.shape is (n * sps,).
        raise NotImplementedError


class NRZPolar(BasebandModulator):
    """Non-Return-to-Zero Polar: 1 -> +V, 0 -> -V, held for the whole bit."""

    def modulate(self, bits: Bits) -> Signal:
        """Map each bit to a constant level held for the whole symbol.

        Args:
            bits: The bit sequence to encode.

        Returns:
            ``+V`` samples for 1s and ``-V`` samples for 0s.
        """
        # +V where the bit is True, -V where it is False.
        levels = np.where(bits, self.amplitude, -self.amplitude)  # (n,)
        # Hold each level for the whole symbol duration.
        # NOTE: np.repeat repeats each element straight after itself.
        # Explicite notation to satisfy mypy.
        signal: Signal = np.repeat(
            levels, self.samples_per_symbol
        )  # (n * sps,)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Recover bits by thresholding one sample per symbol at 0 V.

        Args:
            signal: The received samples.

        Returns:
            A 1 wherever the level is positive, a 0 otherwise.
        """
        # Level is constant across a symbol; one sample per bit suffices.
        # NOTE: The `::` notation takes every N-th element,
        # starting with the first (index 0).
        symbols = signal[:: self.samples_per_symbol]  # (n,)
        bits: Bits = [bool(sample > 0) for sample in symbols]  # (n,)
        return bits


class Manchester(BasebandModulator):
    """Manchester: each bit is a mid-bit transition (data XOR clock).

    Polar convention (matches :class:`NRZPolar`): the signal swings between +V
    and -V, so its average voltage is zero. The clock is low for the first half
    of every bit and high for the second half; XOR-ing the held bit with that
    clock yields a rising edge for a 1 and a falling edge for a 0.
    """

    def _clock(self, symbol_count: int) -> npt.NDArray[np.bool_]:
        """Build the half-bit clock: low first half, high second half.

        Args:
            symbol_count: Number of bits/symbols the clock must cover.

        Returns:
            A boolean signal ``symbol_count * samples_per_symbol`` long.
        """
        half = self.samples_per_symbol // 2
        # One symbol of clock, e.g. [F, F, T, T] for samples_per_symbol == 4.
        one_symbol = [False] * half + [True] * (
            self.samples_per_symbol - half
        )  # (sps,)
        # NOTE: np.tile repeats the whole array, different than np.repeat.
        clock: npt.NDArray[np.bool_] = np.tile(
            np.array(one_symbol), symbol_count
        )  # (symbol_count * sps,)
        return clock

    def modulate(self, bits: Bits) -> Signal:
        """Encode each bit as a mid-bit transition (XOR with the clock).

        Args:
            bits: The bit sequence to encode.

        Returns:
            A polar (+V/-V) signal with one transition per bit.
        """
        # Hold each bit for the whole symbol, then XOR with the half-bit clock.
        bits_held = np.repeat(bits, self.samples_per_symbol)  # (n * sps,)
        high = np.bitwise_xor(bits_held, self._clock(len(bits)))  # (n * sps,)
        # High half -> +V, low half -> -V.
        signal: Signal = np.where(
            high, self.amplitude, -self.amplitude
        )  # (n * sps,)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Undo the clock XOR and read back one sample per symbol.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        symbol_count = len(signal) // self.samples_per_symbol
        # Which samples sit in the high (+V) half of their bit.
        high = signal > 0  # (n * sps,)
        # NOTE: XOR is its own inverse, so this rebuilds the held bits.
        bits_held = np.bitwise_xor(
            high, self._clock(symbol_count)
        )  # (n * sps,)
        # Collapse each symbol back to a single bit.
        symbols = bits_held[:: self.samples_per_symbol]  # (n,)
        bits: Bits = [bool(bit) for bit in symbols]  # (n,)
        return bits


class Bipolar(BasebandModulator):
    """Bipolar (AMI): 0 -> 0 V, 1 -> marks alternating between +V and -V.

    Only the 1s ("marks") carry a pulse, and consecutive marks flip sign: the
    first 1 is +V, the next 1 is -V, and so on, while 0s stay at 0 V and never
    affect the alternation. This keeps the average voltage near zero and lets a
    receiver flag "bipolar violations" (two same-sign marks in a row).
    """

    def modulate(self, bits: Bits) -> Signal:
        """Encode 0 as 0 V and each 1 as a sign-alternating pulse.

        Args:
            bits: The bit sequence to encode.

        Returns:
            A three-level (+V/0/-V) signal.
        """
        bit_array = np.array(bits)  # (n,)
        # NOTE: np.cumsum (cumulative sum) also works for bolean arrays,
        # treating True as 1 and False as 0.
        mark_index = np.cumsum(bit_array)  # (n,)
        # Odd  -> +1; Even -> -1.
        sign = np.where(mark_index % 2 == 1, 1.0, -1.0)  # (n,)
        # 0 bits stay at 0 V; 1 bits take the alternating signed amplitude.
        levels = np.where(bit_array, sign * self.amplitude, 0.0)  # (n,)
        # Hold each level for the whole symbol duration.
        signal: Signal = np.repeat(
            levels, self.samples_per_symbol
        )  # (n * sps,)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Recover bits from pulse magnitude; the sign carries no data.

        Args:
            signal: The received samples.

        Returns:
            A 1 wherever a pulse is present (either polarity), a 0 otherwise.
        """
        # Level is constant across a symbol; one sample per bit suffices.
        symbols = signal[:: self.samples_per_symbol]  # (n,)
        # A mark sits near +/-V; a 0 near 0 V. Threshold the magnitude halfway.
        bits: Bits = [
            bool(abs(sample) > self.amplitude / 2) for sample in symbols
        ]  # (n,)
        return bits
