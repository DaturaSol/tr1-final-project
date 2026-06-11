# ./src/layer_manager/phy/carrier.py
"""Carrier modulations for the physical layer.

Each class satisfies :class:`layer_manager.protocol.Modulator`, mapping bits
onto a sinusoidal carrier and back. See the "Modulação por portadora" and
"BPSK, QPSK, 8PSK" slides for the reference constellations and waveforms.
"""

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

    def modulate(self, bits: Bits) -> Signal:
        """Modulate ``bits`` onto the carrier."""
        raise NotImplementedError

    def demodulate(self, signal: Signal) -> Bits:
        """Recover bits from the modulated carrier ``signal``."""
        raise NotImplementedError


class ASK(CarrierModulator):
    """Amplitude Shift Keying: bit value selects the carrier amplitude."""


class FSK(CarrierModulator):
    """Frequency Shift Keying: bit value selects the carrier frequency."""


class QPSK(CarrierModulator):
    """Quadrature PSK: each symbol carries two bits as a carrier phase."""


class QAM16(CarrierModulator):
    """16-QAM: each symbol carries four bits as an amplitude/phase point."""
