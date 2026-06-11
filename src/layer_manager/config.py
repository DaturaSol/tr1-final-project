# ./src/layer_manager/config.py
"""Configuration models for the simulator's general settings.

These mirror the "Configuração geral" box of the assignment diagram: the
single source of truth the GUI fills in and that both the transmitter and the
receiver read back. Every selectable strategy is named by an enum so the
:mod:`layer_manager.factory` can turn a choice into a concrete object.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class FramingType(StrEnum):
    """Data-link framing scheme."""

    CHAR_COUNT = "char_count"
    BYTE_STUFFING = "byte_stuffing"
    BIT_STUFFING = "bit_stuffing"


class DetectionType(StrEnum):
    """Error-detection scheme (may be combined with a corrector)."""

    NONE = "none"
    PARITY = "parity"
    CHECKSUM = "checksum"
    CRC32 = "crc32"


class CorrectionType(StrEnum):
    """Error-correction scheme."""

    NONE = "none"
    HAMMING = "hamming"


class DigitalModulation(StrEnum):
    """Baseband (digital) line code."""

    NRZ_POLAR = "nrz_polar"
    MANCHESTER = "manchester"
    BIPOLAR = "bipolar"


class CarrierModulation(StrEnum):
    """Carrier modulation; ``NONE`` transmits the baseband signal as-is."""

    NONE = "none"
    ASK = "ask"
    FSK = "fsk"
    QPSK = "qpsk"
    QAM16 = "qam16"


# pydantic's mypy plugin synthesizes Any-typed helpers on BaseModel, which
# `disallow_any_explicit` flags here; the models themselves use no explicit Any.
class SimulationConfig(BaseModel):  # type: ignore[explicit-any]
    """All parameters of a single transmit/receive simulation run.

    Field descriptions double as GUI labels and carry validation bounds, so
    the settings panel can be generated from this model.
    """

    # --- link layer ---
    max_frame_size: int = Field(
        default=8,
        ge=1,
        description="Maximum payload per frame, in bytes.",
    )
    framing: FramingType = Field(
        default=FramingType.BYTE_STUFFING,
        description="Framing scheme.",
    )
    detection: DetectionType = Field(
        default=DetectionType.CRC32,
        description="Error-detection scheme.",
    )
    correction: CorrectionType = Field(
        default=CorrectionType.NONE,
        description="Error-correction scheme.",
    )
    checksum_block_bits: int = Field(
        default=8,
        ge=1,
        description="Block size for the checksum/EDC field, in bits.",
    )

    # --- physical layer ---
    digital_modulation: DigitalModulation = Field(
        default=DigitalModulation.NRZ_POLAR,
        description="Baseband line code.",
    )
    carrier_modulation: CarrierModulation = Field(
        default=CarrierModulation.NONE,
        description="Carrier modulation.",
    )
    amplitude_v: float = Field(
        default=1.0,
        gt=0,
        description="Signal amplitude in volts (V).",
    )
    samples_per_symbol: int = Field(
        default=100,
        ge=2,
        description="Discrete samples generated per symbol/bit.",
    )
    carrier_frequency: float = Field(
        default=1000.0,
        gt=0,
        description="Carrier frequency in hertz (Hz).",
    )
    sample_rate: float = Field(
        default=100_000.0,
        gt=0,
        description="Sampling rate in hertz (Hz).",
    )

    # --- channel ---
    noise_mean: float = Field(
        default=0.0,
        description="Mean (x) of the Gaussian noise added by the channel.",
    )
    noise_std: float = Field(
        default=0.0,
        ge=0,
        description="Standard deviation (sigma) of the Gaussian noise.",
    )
