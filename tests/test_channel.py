# ./tests/test_channel.py
"""Tests for the Gaussian noise channel."""

import numpy as np

from layer_manager.phy.channel import GaussianChannel


def test_zero_noise_is_passthrough() -> None:
    """With std == 0 the signal crosses the channel unchanged."""
    signal = np.array([2.0, -2.0, 0.0, 1.5])
    channel = GaussianChannel(mean=0.0, std=0.0)
    assert np.array_equal(channel.transmit(signal), signal)


def test_shape_is_preserved() -> None:
    """The noisy signal has the same shape as the input."""
    signal = np.zeros(64)
    noisy = GaussianChannel(mean=0.0, std=1.0).transmit(signal)
    assert noisy.shape == signal.shape


def test_added_noise_matches_distribution() -> None:
    """Over many samples, signal - input approaches N(mean, std)."""
    np.random.seed(0)
    signal = np.full(100_000, 5.0)
    mean, std = 1.0, 2.0
    noise = GaussianChannel(mean=mean, std=std).transmit(signal) - signal
    # The signal must still be present (noise centred on `mean`, not on 0).
    assert np.isclose(noise.mean(), mean, atol=0.05)
    assert np.isclose(noise.std(), std, atol=0.05)
