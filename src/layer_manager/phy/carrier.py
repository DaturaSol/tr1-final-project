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
        symbol_time = (
            np.arange(self.samples_per_symbol) / self.sample_rate
        )  # (sps,)
        time = np.tile(symbol_time, symbol_count)  # (symbol_count * sps,)
        angle: Signal = 2 * np.pi * frequency * time  # (symbol_count * sps,)
        return angle  # (symbol_count * sps,)

    def _symbols(self, signal: Signal) -> Signal:
        """Reshape a flat signal into one row of samples per symbol."""
        blocks: Signal = signal.reshape(
            -1, self.samples_per_symbol
        )  # (signal.shape[0] // sps, sps,)
        return blocks

    def modulate(self, bits: Bits) -> Signal:
        """Modulate ``bits`` onto the carrier."""
        # NOTE: bits.shape is (n,).
        raise NotImplementedError

    def demodulate(self, signal: Signal) -> Bits:
        """Recover bits from the modulated carrier ``signal``."""
        # NOTE: signal.shape is (n * sps,).
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
        )  # (n * sps,)
        keep = np.repeat(bits, self.samples_per_symbol)  # (n * sps,)
        # Keep the carrier during a 1, blank it during a 0 (on/off keying).
        signal: Signal = np.where(keep, carrier, 0.0)  # (n * sps,)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Match each symbol against the carrier: strong response means 1.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        reference = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
        # Correlate each symbol with the reference.
        correlation = (
            self._symbols(signal) @ reference
        )  # (n, sps) @ (sps,) -> (n,)
        # A "1" correlates to ~ A * sps / 2 (average energy of a sine wave),
        # a "0" to ~0; threshold halfway.
        # NOTE: A*sum{n=0}^{sps-1}sin^2(2*pi*n*f/sr)~ A*sps/2,
        # if f*sps/sr is an integer.
        # threshold = self.amplitude * self.samples_per_symbol / 4
        # True threshold, but slower to compute.
        threshold = self.amplitude * reference @ reference / 2  # scalar
        bits: Bits = [bool(value > threshold) for value in correlation]  # (n,)
        return bits


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
        # Default waveform.
        tone0 = np.sin(
            self._angle(self.carrier_frequency, len(bits))
        )  # (n * sps,)
        # Double frequency waveform.
        tone1 = np.sin(
            self._angle(2 * self.carrier_frequency, len(bits))
        )  # (n * sps,)
        held = np.repeat(bits, self.samples_per_symbol)  # (n * sps,)
        signal: Signal = self.amplitude * np.where(
            held, tone1, tone0
        )  # (n * sps,)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Correlate each symbol with both tones; the stronger one wins.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        blocks = self._symbols(signal)  # (n, sps,)
        # Default waveform.
        ref0 = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
        # Double frequency waveform.
        ref1 = np.sin(self._angle(2 * self.carrier_frequency, 1))  # (sps,)
        # Take np.abs, we only want to compare the strength of the correlation,
        # not its sign.
        corr0 = np.abs(blocks @ ref0)  # (n, sps) @ (sps,) -> (n,)
        corr1 = np.abs(blocks @ ref1)  # (n, sps) @ (sps,) -> (n,)
        bits: Bits = [bool(value) for value in corr1 > corr0]  # (n,)
        return bits


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
        # I * cos(w) - Q * sin(w) = sqrt(I^2+Q^2) * sin(w + atan2(I, Q)).
        # QPSK(1), I,Q in [-1, 1] -> atan2(I, Q) = (45, 135, 225, 315) degrees.
        # NOTE: (n,) is even, so we dont get errors when reshaping.
        pairs = np.array(bits).reshape(-1, 2)  # (n // 2, 2)
        in_phase = np.where(pairs[:, 0], -1.0, 1.0)  # (n // 2,)
        quadrature = np.where(pairs[:, 1], -1.0, 1.0)  # (n // 2,)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))  # (sps,)
        sine = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
        # Divide by sqrt(2) to keep the total power A^2 constant across
        # all four constellation points.
        scale = self.amplitude / np.sqrt(2)
        # Combine I and Q into one symbol each, then flatten to a signal.
        symbols = (
            in_phase[:, np.newaxis] * cosine - quadrature[:, np.newaxis] * sine
        )  # (n // 2, sps)
        signal: Signal = scale * symbols.reshape(-1)  # (n // 2 * sps,)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Recover each bit pair by jointly de-mixing the I and Q correlations.

        The matched filter assumes the cosine and sine references are orthogonal
        over one symbol, which only holds for a whole number of carrier cycles
        per symbol. To stay correct for any carrier, this solves the 2x2 system
        coupling the in-phase (I) and quadrature (Q) components, instead of
        trusting the raw correlation signs.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        blocks = self._symbols(signal)  # (n // 2, sps)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))  # (sps,)
        sine = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
        # Idea: the received symbol is a linear combination
        # of the two references:
        # [cc, cs] = [I, Q] @ [[a, -b], [b, -d]]
        # -> [I, Q] = [cc, cs] @ [[-d, b], [-b, a]] / det
        a = cosine @ cosine  # scalar
        b = sine @ cosine  # scalar
        d = sine @ sine  # scalar
        # Each symbol is scale * (I * cosine - Q * sine),
        cc = blocks @ cosine  # (n // 2,)
        cs = blocks @ sine  # (n // 2,)
        det = b * b - a * d  # det([[a, -b], [b, -d]]); < 0 for independent refs
        in_phase = (-d * cc + b * cs) / det  # (n // 2,)
        quadrature = (-b * cc + a * cs) / det  # (n // 2,)
        # I, Q < 0 encode a set bit (1), I, Q >= 0 encode a clear bit (0).
        first = in_phase < 0
        second = quadrature < 0
        pairs = np.stack([first, second], axis=1)  # (n // 2, 2)
        bits: Bits = [bool(value) for value in pairs.reshape(-1)]  # (n,)
        return bits


class QAM16(CarrierModulator):
    """16-QAM (CF-13): each symbol carries four bits as one constellation point.

    The quadbit is laid out as
    ``[I sign, Q sign, I magnitude, Q magnitude]`` (sign bits first, then
    magnitudes). Each axis rides the Gray-coded ladder ``{-3, -1, +1, +3}``:
    the sign bit picks +/-, the magnitude bit picks the inner (1) or outer (3)
    ring. The constellation is normalised so the outer corner reaches the
    configured amplitude ``A``, which yields the three envelope levels
    ``A/3``, ``A*sqrt(5)/3`` and ``A`` (~0.33, 0.75, 1.00 for ``A = 1``).
    """

    def modulate(self, bits: Bits) -> Signal:
        """Map each group of four bits to one amplitude/phase point.

        Args:
            bits: The bit sequence to encode (length a multiple of 4).

        Returns:
            The quadrature-amplitude-modulated carrier.
        """
        # [I sign, Q sign, I magnitude, Q magnitude].
        groups = np.array(bits).reshape(-1, 4)  # (n // 4, 4)
        in_level = self._levels(groups[:, 0], groups[:, 2])  # (n // 4,)
        quad_level = self._levels(groups[:, 1], groups[:, 3])  # (n // 4,)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))  # (sps,)
        sine = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
        # Normalise so the outer corner (3, 3) reaches amplitude A:
        # unit * sqrt(3**2 + 3**2) = unit * 3*sqrt(2) = A.
        unit = self.amplitude / (3 * np.sqrt(2))
        # symbol = unit*(I*cos - Q*sin); phase = atan2(Q, I).
        symbols = (
            in_level[:, np.newaxis] * cosine - quad_level[:, np.newaxis] * sine
        )  # (n // 4, sps)
        signal: Signal = unit * symbols.reshape(-1)  # (n // 4 * sps,)
        return signal

    def demodulate(self, signal: Signal) -> Bits:
        """Recover the I/Q levels by de-mixing the correlations, then decode.

        Like the QPSK receiver, this solves the 2x2 system coupling the cosine
        and sine references, so it stays correct even when the carrier does not
        complete a whole number of cycles per symbol. It then thresholds each
        recovered level on the {-3, -1, +1, +3} ladder.

        Args:
            signal: The received samples.

        Returns:
            The recovered bit sequence.
        """
        blocks = self._symbols(signal)  # (n // 4, sps)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))  # (sps,)
        sine = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
        # Idea: the received symbol is a linear combination
        # of the two references:
        # [cc, cs] = unit * [I, Q] @ [[a, -b], [b, -d]]
        # -> [I, Q] = [cc, cs] @ [[-d, b], [-b, a]] / (det * unit)
        a = cosine @ cosine  # scalar
        b = sine @ cosine  # scalar
        d = sine @ sine  # scalar
        cc = blocks @ cosine  # (n // 4,)
        cs = blocks @ sine  # (n // 4,)
        det = b * b - a * d  # det([[a, -b], [b, -d]]); < 0 for independent refs
        unit = self.amplitude / (3 * np.sqrt(2))
        in_level = (-d * cc + b * cs) / (det * unit)  # (n // 4,)
        quad_level = (-b * cc + a * cs) / (det * unit)  # (n // 4,)
        # _bits returns [sign, magnitude]; quadbit order
        # [I sign, Q sign, I magnitude, Q magnitude].
        i_bits = self._bits(in_level)  # (n // 4, 2)
        q_bits = self._bits(quad_level)  # (n // 4, 2)
        quadbits = np.stack(
            [i_bits[:, 0], q_bits[:, 0], i_bits[:, 1], q_bits[:, 1]], axis=1
        )  # (n // 4, 4)
        bits: Bits = [bool(value) for value in quadbits.reshape(-1)]  # (n,)
        return bits

    @staticmethod
    def _levels(sign_bit: Signal, magnitude_bit: Signal) -> Signal:
        """Map a sign bit and a magnitude bit to a level in {-3,-1,+1,+3}.

        Per CF-13: sign bit ``1 -> +``, ``0 -> -``; magnitude bit
        ``1 -> outer (3)``, ``0 -> inner (1)``. Gray-coded along the ladder.
        """
        sign = np.where(sign_bit, 1.0, -1.0)
        magnitude = np.where(magnitude_bit, 3.0, 1.0)
        levels: Signal = sign * magnitude
        return levels

    @staticmethod
    def _bits(level: Signal) -> npt.NDArray[np.bool_]:
        """Invert :meth:`_levels`: recover the ``[sign, magnitude]`` bits."""
        sign_bit = level > 0  # positive level -> 1
        magnitude_bit = np.abs(level) > 2  # outer (3) -> 1, inner (1) -> 0
        pairs: npt.NDArray[np.bool_] = np.stack(
            [sign_bit, magnitude_bit], axis=1
        )
        return pairs
