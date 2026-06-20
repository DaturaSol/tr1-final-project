# ./tests/test_app_simulation.py
"""Tests for the app's modulation pipeline helper."""

import pytest

from app.simulation import ALL_SCHEMES, build_modulator, run_modulation
from layer_manager.config import SimulationConfig
from layer_manager.protocol import Modulator

# Whole carrier cycles per symbol (20 * 100 / 1000 == 2) -> clean demodulation.
CONFIG = SimulationConfig(
    samples_per_symbol=100,
    carrier_frequency=20.0,
    sample_rate=1000.0,
    noise_std=0.0,
)


@pytest.mark.parametrize("scheme", list(ALL_SCHEMES))
def test_build_modulator_returns_a_modulator(scheme: str) -> None:
    """Every scheme name resolves to a usable modulator."""
    assert isinstance(build_modulator(scheme, CONFIG), Modulator)


@pytest.mark.parametrize("scheme", list(ALL_SCHEMES))
def test_noiseless_round_trip_recovers_text(scheme: str) -> None:
    """With no noise, each scheme recovers the original message exactly."""
    result = run_modulation(scheme, CONFIG, "Hi!")
    assert result.matches
    assert result.recovered_text == "Hi!"
