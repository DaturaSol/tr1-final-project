"""End-to-end pipeline tests (also exercise the factory's link builders)."""

import numpy as np
import pytest

from layer_manager import factory, pipeline
from layer_manager.config import (
    CarrierModulation,
    CorrectionType,
    DetectionType,
    FramingType,
    SimulationConfig,
)

F, D, C, CM = FramingType, DetectionType, CorrectionType, CarrierModulation

# Whole carrier cycles per symbol (20 * 100 / 1000 == 2) -> clean carriers.
_BASE: dict[str, object] = {
    "samples_per_symbol": 100,
    "carrier_frequency": 20.0,
    "sample_rate": 1000.0,
}

# Combos whose protected frame stays byte-aligned for char/byte stuffing;
# the bit-granular codes (parity, Hamming) ride bit stuffing, which is happy
# with any bit length. ASK/FSK pack one bit per symbol, so any framed length
# works. Together these touch every link-layer builder in the factory.
_COMBOS: list[dict[str, object]] = [
    {"framing": F.CHAR_COUNT, "detection": D.NONE},
    {"framing": F.CHAR_COUNT, "detection": D.CRC32},
    {"framing": F.CHAR_COUNT, "detection": D.CHECKSUM},
    {"framing": F.BYTE_STUFFING, "detection": D.NONE},
    {"framing": F.BYTE_STUFFING, "detection": D.CRC32},
    {"framing": F.BYTE_STUFFING, "detection": D.CHECKSUM},
    {"framing": F.BIT_STUFFING, "detection": D.NONE},
    {"framing": F.BIT_STUFFING, "detection": D.CRC32},
    {"framing": F.BIT_STUFFING, "detection": D.PARITY},
    {"framing": F.BIT_STUFFING, "detection": D.CRC32, "correction": C.HAMMING},
    {"framing": F.BIT_STUFFING, "detection": D.NONE, "correction": C.HAMMING},
    {"framing": F.BYTE_STUFFING, "detection": D.CRC32,
     "carrier_modulation": CM.ASK},
    {"framing": F.BYTE_STUFFING, "detection": D.CRC32,
     "carrier_modulation": CM.FSK},
]


@pytest.mark.parametrize("overrides", _COMBOS)
@pytest.mark.parametrize("text", ["Hi!", "Hello, the whole link layer!"])
def test_pipeline_clean_round_trip(
    overrides: dict[str, object], text: str
) -> None:
    """Every layer combo recovers the message exactly over a clean channel."""
    config = SimulationConfig(**_BASE, **overrides)
    tx = pipeline.transmit(config, text)
    rx = pipeline.receive(config, tx.signal)
    assert rx.text == text
    assert rx.ok
    assert rx.corrected == 0


def test_pipeline_multiple_frames() -> None:
    """A message longer than max_frame_size splits into several frames."""
    config = SimulationConfig(**_BASE, max_frame_size=4, detection=D.CRC32)
    tx = pipeline.transmit(config, "0123456789")  # 10 bytes -> 3 frames
    assert len(tx.frames) == 3
    rx = pipeline.receive(config, tx.signal)
    assert rx.text == "0123456789"
    assert len(rx.frames) == 3


def test_pipeline_noise_corrupts_message() -> None:
    """Heavy channel noise corrupts the recovered text (errors get through)."""
    config = SimulationConfig(**_BASE, detection=D.CRC32, noise_std=2.0)
    tx = pipeline.transmit(config, "Reliable link?")
    np.random.seed(0)
    noisy = factory.build_channel(config).transmit(tx.signal)
    rx = pipeline.receive(config, noisy)
    assert rx.text != "Reliable link?"


def test_pipeline_empty_message() -> None:
    """An empty message round-trips to empty with no frames, reported ok."""
    config = SimulationConfig(**_BASE, detection=D.CRC32)
    tx = pipeline.transmit(config, "")
    assert tx.frames == []
    rx = pipeline.receive(config, tx.signal)
    assert rx.text == ""
    assert rx.ok  # genuinely-empty input is not a failure


def test_pipeline_lost_signal_is_not_ok() -> None:
    """A non-empty signal that yields no frames is reported as not ok."""
    config = SimulationConfig(**_BASE, detection=D.CRC32)
    tx = pipeline.transmit(config, "Hello world")
    rx = pipeline.receive(config, np.zeros_like(tx.signal))
    assert not rx.ok  # not vacuously True via all([])


def test_pipeline_rejects_byte_framer_with_bit_aligned_code() -> None:
    """Parity (+1 bit) through a byte framer is refused, not mangled."""
    config = SimulationConfig(
        **_BASE, framing=F.BYTE_STUFFING, detection=D.PARITY
    )
    with pytest.raises(ValueError, match="byte-aligned"):
        pipeline.transmit(config, "Hi")


def test_pipeline_undecodable_stream_is_graceful() -> None:
    """An unparseable stream reports '<undecodable>' instead of raising."""
    config = SimulationConfig(**_BASE, framing=F.BIT_STUFFING)
    # A lone FLAG pattern -> bit-stuffing deframe sees an odd FLAG count.
    bits = [False, True, True, True, True, True, True, False, True, True]
    signal = factory.build_digital_modulator(config).modulate(bits)
    rx = pipeline.receive(config, signal)  # must not raise
    assert rx.text == "<undecodable>"
    assert rx.ok is False


def test_pipeline_receive_survives_heavy_noise() -> None:
    """Heavy noise never makes the receiver raise (it reports a failure)."""
    config = SimulationConfig(
        **_BASE, framing=F.BIT_STUFFING, detection=D.CRC32
    )
    tx = pipeline.transmit(config, "stuffing under heavy noise 11111111")
    for seed in range(25):
        rng = np.random.RandomState(seed)
        noisy = tx.signal + rng.normal(0.0, 3.0, tx.signal.shape)
        rx = pipeline.receive(config, noisy)  # must not raise, any seed
        assert rx.ok is False


def test_pipeline_rejects_indivisible_carrier() -> None:
    """A framed length the carrier can't pack raises a clear error."""
    config = SimulationConfig(
        **_BASE,
        framing=F.BIT_STUFFING,
        correction=C.HAMMING,
        carrier_modulation=CM.QAM16,
    )
    with pytest.raises(ValueError, match="bits per symbol"):
        pipeline.transmit(config, "Hi")
