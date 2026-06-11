# ./src/layer_manager/types.py
"""Centralized type aliases shared across the simulator's layers."""

import numpy as np
import numpy.typing as npt

type Bits = list[bool]
"""A sequence of bits, each element either ``False`` or ``True`` (MSB first)."""

type Signal = npt.NDArray[np.float64]
"""Discrete signal samples in electrical units (V / W)."""
