# ./tests/test_baseband.py
"""Tests for the baseband line codes."""

import numpy as np
import pytest

from layer_manager.phy.baseband import (
    BasebandModulator,
    Bipolar,
    Manchester,
    NRZPolar,
)
from layer_manager.types import Bits

# Preserved from the original module demo.
SAMPLE_BITS: Bits = [b == 1 for b in (1, 0, 1, 0, 1, 0, 1, 1)]
AMPLITUDE = 2.0

MODULATORS = [NRZPolar, Manchester, Bipolar]


@pytest.mark.parametrize("modulator_cls", MODULATORS)
@pytest.mark.parametrize("samples_per_symbol", [2, 4, 8])
def test_round_trip(
    modulator_cls: type[BasebandModulator], samples_per_symbol: int
) -> None:
    """Each line code recovers the original bits after demodulation."""
    modulator = modulator_cls(AMPLITUDE, samples_per_symbol=samples_per_symbol)
    signal = modulator.modulate(SAMPLE_BITS)
    assert signal.size == len(SAMPLE_BITS) * samples_per_symbol
    assert modulator.demodulate(signal) == SAMPLE_BITS


@pytest.mark.parametrize("samples_per_symbol", [2, 4, 8])
def test_nrz_uses_only_two_levels(samples_per_symbol: int) -> None:
    """NRZ-Polar emits only +V and -V."""
    signal = NRZPolar(AMPLITUDE, samples_per_symbol).modulate(SAMPLE_BITS)
    assert set(np.unique(signal).tolist()) <= {AMPLITUDE, -AMPLITUDE}


@pytest.mark.parametrize("samples_per_symbol", [2, 4, 8])
def test_manchester_zero_dc(samples_per_symbol: int) -> None:
    """Manchester holds a zero average voltage (equal half-bits)."""
    signal = Manchester(AMPLITUDE, samples_per_symbol).modulate(SAMPLE_BITS)
    assert np.isclose(signal.mean(), 0.0)


def test_bipolar_known_levels() -> None:
    """Bipolar encodes the canonical pattern with sign-alternating marks."""
    signal = Bipolar(AMPLITUDE, samples_per_symbol=2).modulate(SAMPLE_BITS)
    # One sample per symbol: marks alternate +V/-V, zeros stay at 0 V.
    expected = [
        AMPLITUDE, 0.0, -AMPLITUDE, 0.0, AMPLITUDE, 0.0, -AMPLITUDE, AMPLITUDE,
    ]
    assert signal[::2].tolist() == expected


def test_bipolar_marks_alternate_sign() -> None:
    """Consecutive Bipolar marks always have opposite signs."""
    signal = Bipolar(AMPLITUDE, samples_per_symbol=2).modulate(SAMPLE_BITS)
    levels = signal[::2]
    marks = levels[levels != 0]
    # Adjacent marks multiply to a negative number iff they differ in sign.
    assert bool(np.all(marks[:-1] * marks[1:] < 0))
