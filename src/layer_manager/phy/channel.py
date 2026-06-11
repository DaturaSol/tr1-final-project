# ./src/layer_manager/phy/channel.py
"""The communication medium between transmitter and receiver.

Satisfies :class:`layer_manager.protocol.Channel`.
"""

import numpy as np

from layer_manager.protocol import Channel
from layer_manager.types import Signal


class GaussianChannel(Channel):
    """Adds Gaussian noise n(x, sigma) to each V/W sample of the signal."""

    def __init__(self, mean: float, std: float) -> None:
        """Store the noise distribution parameters.

        Args:
            mean: Mean (x) of the Gaussian noise, in the signal's units.
            std: Standard deviation (sigma) of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def transmit(self, signal: Signal) -> Signal:
        """Return ``signal`` with a Gaussian noise sample added per element.

        Args:
            signal: The transmitted samples, in V/W.

        Returns:
            ``signal + n`` where each ``n`` is drawn from ``N(mean, std)``.
            With ``std == 0`` the signal passes through unchanged.
        """
        noise = np.random.normal(
            loc=self.mean, scale=self.std, size=signal.shape
        )
        noisy_signal: Signal = signal + noise
        return noisy_signal
