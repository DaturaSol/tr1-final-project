# ./tests/test_carrier.py
"""Tests for the carrier modulations."""

import numpy as np
import pytest

from layer_manager.phy.carrier import ASK, FSK, QAM16, QPSK, CarrierModulator
from layer_manager.types import Bits

# Preserved bit pattern; length 8 divides 1, 2 and 4 bits-per-symbol cleanly.
SAMPLE_BITS: Bits = [b == 1 for b in (1, 0, 1, 0, 1, 0, 1, 1)]

# carrier_frequency * samples_per_symbol / sample_rate == 2 -> whole cycles per
# symbol, so the matched-filter correlations separate cleanly.
PARAMS = {
    "amplitude": 1.0,
    "samples_per_symbol": 100,
    "carrier_frequency": 20.0,
    "sample_rate": 1000.0,
}

MODULATORS = [ASK, FSK, QPSK, QAM16]


@pytest.mark.parametrize("modulator_cls", MODULATORS)
def test_round_trip(modulator_cls: type[CarrierModulator]) -> None:
    """Each carrier scheme recovers the original bits with no noise."""
    modulator = modulator_cls(**PARAMS)
    signal = modulator.modulate(SAMPLE_BITS)
    assert modulator.demodulate(signal) == SAMPLE_BITS


@pytest.mark.parametrize("modulator_cls", MODULATORS)
def test_round_trip_survives_light_noise(
    modulator_cls: type[CarrierModulator],
) -> None:
    """Demodulation still recovers the bits under mild Gaussian noise."""
    np.random.seed(0)
    modulator = modulator_cls(**PARAMS)
    signal = modulator.modulate(SAMPLE_BITS)
    noisy = signal + np.random.normal(0.0, 0.05, signal.shape)
    assert modulator.demodulate(noisy) == SAMPLE_BITS


@pytest.mark.parametrize(
    ("modulator_cls", "bits_per_symbol"),
    [(ASK, 1), (FSK, 1), (QPSK, 2), (QAM16, 4)],
)
def test_signal_length(
    modulator_cls: type[CarrierModulator], bits_per_symbol: int
) -> None:
    """A signal holds samples_per_symbol samples per packed symbol."""
    signal = modulator_cls(**PARAMS).modulate(SAMPLE_BITS)
    symbols = len(SAMPLE_BITS) // bits_per_symbol
    assert signal.size == symbols * PARAMS["samples_per_symbol"]


def test_ask_blanks_the_zero_bit() -> None:
    """ASK carries energy for a 1 and is silent for a 0."""
    signal = ASK(**PARAMS).modulate([True, False])
    one, zero = signal.reshape(2, PARAMS["samples_per_symbol"])
    assert np.sum(one**2) > 0.0
    assert np.sum(zero**2) == 0.0
