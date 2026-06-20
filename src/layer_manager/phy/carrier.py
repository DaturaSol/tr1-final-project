# ./src/layer_manager/phy/carrier.py
"""Carrier modulations for the physical layer.

Each class satisfies :class:`layer_manager.protocol.Modulator`, mapping bits
onto a sinusoidal carrier and back. See the "Modulação por portadora" and
"BPSK, QPSK, 8PSK" slides for the reference constellations and waveforms.

All four schemes build the carrier from :meth:`CarrierModulator._angle`, whose
phase restarts at every symbol boundary. Demodulation is then a matched filter:
correlate each received symbol against the same reference wave(s) and decide.
For that correlation to separate cleanly, each symbol should span a whole number
of carrier cycles, i.e. ``carrier_frequency * samples_per_symbol / sample_rate``
should be an integer. Schemes that pack several bits per symbol (QPSK: 2,
16-QAM: 4) need ``len(bits)`` to be a multiple of that count.
"""

import numpy as np
import numpy.typing as npt

from layer_manager.protocol import Modulator
from layer_manager.types import Bits, Signal


class CarrierModulator(Modulator):
    """Shared carrier parameters for the concrete modulators below."""

    def __init__(
        self,
        amplitude: float,
        samples_per_symbol: int,
        carrier_frequency: float,
        sample_rate: float,
    ) -> None:
        """Store the carrier parameters.

        Args:
            amplitude: Carrier amplitude in volts.
            samples_per_symbol: Discrete samples emitted per symbol.
            carrier_frequency: Carrier frequency in hertz.
            sample_rate: Sampling rate in hertz.
        """
        self.amplitude = amplitude
        self.samples_per_symbol = samples_per_symbol
        self.carrier_frequency = carrier_frequency
        self.sample_rate = sample_rate

    def _angle(self, frequency: float, symbol_count: int) -> Signal:
        """Carrier angle 2*pi*f*t, with the phase reset every symbol.

        Args:
            frequency: Tone frequency in hertz.
            symbol_count: Number of symbols the angle must cover.

        Returns:
            Angle samples, ``symbol_count * samples_per_symbol`` long.
        """
        # Local time within one symbol, repeated for every symbol.
        symbol_time = np.arange(self.samples_per_symbol) / self.sample_rate
        time = np.tile(symbol_time, symbol_count)
        angle: Signal = 2 * np.pi * frequency * time
        return angle

    def _symbols(self, signal: Signal) -> Signal:
        """Reshape a flat signal into one row of samples per symbol."""
        blocks: Signal = signal.reshape(-1, self.samples_per_symbol)
        return blocks

    def modulate(self, bits: Bits) -> Signal:
        """Modulate ``bits`` onto the carrier."""
        raise NotImplementedError

    def demodulate(self, signal: Signal) -> Bits:
        """Recover bits from the modulated carrier ``signal``."""
        raise NotImplementedError


class ASK(CarrierModulator):
    """Amplitude Shift Keying: bit value selects the carrier amplitude."""

    def modulate(self, bits: Bits) -> Signal:
        """Scale the carrier amplitude by each bit (full for 1, zero for 0).

        Args:
            bits: The bit sequence to encode.

        Returns:
            The on/off-keyed carrier: ``A sin(2*pi*f*t)`` during a 1, 0 during
            a 0.
        """
        carrier = self.amplitude * np.sin(
            self._angle(self.carrier_frequency, len(bits))
        )
        # Keep the carrier during a 1, blank it during a 0 (on/off keying).
        keep = np.repeat(bits, self.samples_per_symbol)
        signal: Signal = np.where(keep, carrier, 0.0)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Match each symbol against the carrier: strong response means 1.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        reference = np.sin(self._angle(self.carrier_frequency, 1))
        correlation = self._symbols(signal) @ reference
        # A "1" correlates to ~ A * sps / 2, a "0" to ~0; threshold halfway.
        threshold = self.amplitude * self.samples_per_symbol / 4
        return [bool(value > threshold) for value in correlation]


class FSK(CarrierModulator):
    """Frequency Shift Keying: bit value selects the carrier frequency.

    A 0 is sent at ``carrier_frequency`` and a 1 at twice that frequency.
    """

    def modulate(self, bits: Bits) -> Signal:
        """Send each bit as a tone: low frequency for 0, double for 1.

        Args:
            bits: The bit sequence to encode.

        Returns:
            The frequency-keyed carrier.
        """
        tone0 = np.sin(self._angle(self.carrier_frequency, len(bits)))
        tone1 = np.sin(self._angle(2 * self.carrier_frequency, len(bits)))
        held = np.repeat(bits, self.samples_per_symbol)
        signal: Signal = self.amplitude * np.where(held, tone1, tone0)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Correlate each symbol with both tones; the stronger one wins.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        blocks = self._symbols(signal)
        ref0 = np.sin(self._angle(self.carrier_frequency, 1))
        ref1 = np.sin(self._angle(2 * self.carrier_frequency, 1))
        corr0 = np.abs(blocks @ ref0)
        corr1 = np.abs(blocks @ ref1)
        return [bool(value) for value in corr1 > corr0]


class QPSK(CarrierModulator):
    """Quadrature PSK: each symbol carries two bits as a carrier phase.

    The first bit picks the in-phase (cosine) sign, the second the quadrature
    (sine) sign; a False bit maps to +1 and a True bit to -1.
    """

    def modulate(self, bits: Bits) -> Signal:
        """Encode bit pairs as the sign of the cosine and sine components.

        Args:
            bits: The bit sequence to encode (length a multiple of 2).

        Returns:
            The phase-keyed carrier.
        """
        pairs = np.array(bits).reshape(-1, 2)
        in_phase = np.where(pairs[:, 0], -1.0, 1.0)
        quadrature = np.where(pairs[:, 1], -1.0, 1.0)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))
        sine = np.sin(self._angle(self.carrier_frequency, 1))
        scale = self.amplitude / np.sqrt(2)
        # Combine I and Q into one symbol each, then flatten to a signal.
        symbols = in_phase[:, None] * cosine - quadrature[:, None] * sine
        signal: Signal = scale * symbols.reshape(-1)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Recover each bit pair from the sign of the I and Q correlations.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        blocks = self._symbols(signal)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))
        sine = np.sin(self._angle(self.carrier_frequency, 1))
        first = (blocks @ cosine) < 0
        second = (blocks @ sine) > 0
        pairs = np.stack([first, second], axis=1)
        return [bool(value) for value in pairs.reshape(-1)]


class QAM16(CarrierModulator):
    """16-QAM: each symbol carries four bits as an amplitude/phase point.

    The first two bits set the in-phase level, the last two the quadrature
    level, each on the Gray-coded ladder ``{-3, -1, +1, +3}``: the sign bit
    chooses +/-, the magnitude bit chooses the inner (1) or outer (3) ring.
    """

    def modulate(self, bits: Bits) -> Signal:
        """Map each group of four bits to one amplitude/phase point.

        Args:
            bits: The bit sequence to encode (length a multiple of 4).

        Returns:
            The quadrature-amplitude-modulated carrier.
        """
        groups = np.array(bits).reshape(-1, 4)
        in_level = self._levels(groups[:, 0], groups[:, 1])
        quad_level = self._levels(groups[:, 2], groups[:, 3])
        cosine = np.cos(self._angle(self.carrier_frequency, 1))
        sine = np.sin(self._angle(self.carrier_frequency, 1))
        # Scale so the outer level (3) reaches the full amplitude.
        unit = self.amplitude / 3
        symbols = in_level[:, None] * cosine - quad_level[:, None] * sine
        signal: Signal = unit * symbols.reshape(-1)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Recover the I and Q levels by correlation, then decode each pair.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        blocks = self._symbols(signal)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))
        sine = np.sin(self._angle(self.carrier_frequency, 1))
        # Correlation returns each level scaled by unit * sps / 2.
        reference = self.amplitude / 3 * self.samples_per_symbol / 2
        in_level = (blocks @ cosine) / reference
        quad_level = (blocks @ sine) / -reference
        bits = np.concatenate(
            [self._bits(in_level), self._bits(quad_level)], axis=1
        )
        return [bool(value) for value in bits.reshape(-1)]

    @staticmethod
    def _levels(sign_bit: Signal, magnitude_bit: Signal) -> Signal:
        """Map two bits to a Gray-coded level in ``{-3, -1, +1, +3}``."""
        sign = np.where(sign_bit, 1.0, -1.0)
        magnitude = np.where(magnitude_bit, 1.0, 3.0)
        levels: Signal = sign * magnitude
        return levels

    @staticmethod
    def _bits(level: Signal) -> npt.NDArray[np.bool_]:
        """Invert :meth:`_levels`: recover the sign and magnitude bits."""
        sign_bit = level > 0
        magnitude_bit = np.abs(level) < 2
        pairs: npt.NDArray[np.bool_] = np.stack(
            [sign_bit, magnitude_bit], axis=1
        )
        return pairs
