# ./src/layer_manager/factory.py
"""Turn a :class:`SimulationConfig` into concrete layer strategies.

This is the single place that maps each configuration enum to its concrete
implementation, so the GUI and the socket nodes never duplicate the wiring.
Builders that can be switched off (detection, correction, carrier) return
``None`` when their config value is ``NONE``.
"""

from layer_manager.config import (
    CarrierModulation,
    CorrectionType,
    DetectionType,
    DigitalModulation,
    FramingType,
    SimulationConfig,
)
from layer_manager.link.correction import HammingCorrector
from layer_manager.link.detection import (
    ChecksumDetector,
    CRC32Detector,
    ParityDetector,
)
from layer_manager.link.framing import (
    BitStuffingFramer,
    ByteStuffingFramer,
    CharCountFramer,
)
from layer_manager.phy.baseband import (
    BasebandModulator,
    Bipolar,
    Manchester,
    NRZPolar,
)
from layer_manager.phy.carrier import ASK, FSK, QAM16, QPSK, CarrierModulator
from layer_manager.phy.channel import GaussianChannel
from layer_manager.protocol import (
    Channel,
    ErrorCorrector,
    ErrorDetector,
    Framer,
    Modulator,
)


def build_framer(config: SimulationConfig) -> Framer:
    """Build the framing strategy named by ``config.framing``."""
    match config.framing:
        case FramingType.CHAR_COUNT:
            return CharCountFramer()
        case FramingType.BYTE_STUFFING:
            return ByteStuffingFramer()
        case FramingType.BIT_STUFFING:
            return BitStuffingFramer()


def build_detector(config: SimulationConfig) -> ErrorDetector | None:
    """Build the error detector, or ``None`` when detection is disabled."""
    match config.detection:
        case DetectionType.NONE:
            return None
        case DetectionType.PARITY:
            return ParityDetector()
        case DetectionType.CHECKSUM:
            return ChecksumDetector(config.checksum_block_bits)
        case DetectionType.CRC32:
            return CRC32Detector()


def build_corrector(config: SimulationConfig) -> ErrorCorrector | None:
    """Build the error corrector, or ``None`` when correction is disabled."""
    match config.correction:
        case CorrectionType.NONE:
            return None
        case CorrectionType.HAMMING:
            return HammingCorrector()


def build_digital_modulator(config: SimulationConfig) -> Modulator:
    """Build the baseband modulator named by ``config.digital_modulation``."""
    match config.digital_modulation:
        case DigitalModulation.NRZ_POLAR:
            cls: type[BasebandModulator] = NRZPolar
        case DigitalModulation.MANCHESTER:
            cls = Manchester
        case DigitalModulation.BIPOLAR:
            cls = Bipolar
    return cls(config.amplitude_v, config.samples_per_symbol)


def build_carrier_modulator(config: SimulationConfig) -> Modulator | None:
    """Build the carrier modulator, or ``None`` when it is disabled."""
    match config.carrier_modulation:
        case CarrierModulation.NONE:
            return None
        case CarrierModulation.ASK:
            cls: type[CarrierModulator] = ASK
        case CarrierModulation.FSK:
            cls = FSK
        case CarrierModulation.QPSK:
            cls = QPSK
        case CarrierModulation.QAM16:
            cls = QAM16
    return cls(
        config.amplitude_v,
        config.samples_per_symbol,
        config.carrier_frequency,
        config.sample_rate,
    )


def build_channel(config: SimulationConfig) -> Channel:
    """Build the noisy channel from the configured noise parameters."""
    return GaussianChannel(config.noise_mean, config.noise_std)
