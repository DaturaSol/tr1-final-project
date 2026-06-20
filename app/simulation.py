# ./app/simulation.py
"""Streamlit-free helpers that drive the physical-layer demo.

Kept apart from the UI so the modulation pipeline can be unit-tested and reused
by the live view. Everything here is pure: given a scheme name, a
:class:`SimulationConfig` and some text, it returns the signals to plot.
"""

from dataclasses import dataclass

import numpy as np

from layer_manager import factory
from layer_manager.config import (
    CarrierModulation,
    DigitalModulation,
    SimulationConfig,
)
from layer_manager.protocol import Modulator
from layer_manager.types import Bits, Signal
from layer_manager.utils import bits_to_text, text_to_bits

# Display name -> enum, split by layer. Merged into one list for the UI.
DIGITAL_SCHEMES: dict[str, DigitalModulation] = {
    "NRZ-Polar": DigitalModulation.NRZ_POLAR,
    "Manchester": DigitalModulation.MANCHESTER,
    "Bipolar": DigitalModulation.BIPOLAR,
}
CARRIER_SCHEMES: dict[str, CarrierModulation] = {
    "ASK": CarrierModulation.ASK,
    "FSK": CarrierModulation.FSK,
    "QPSK": CarrierModulation.QPSK,
    "16-QAM": CarrierModulation.QAM16,
}
ALL_SCHEMES: dict[str, DigitalModulation | CarrierModulation] = {
    **DIGITAL_SCHEMES,
    **CARRIER_SCHEMES,
}


@dataclass
class ModulationResult:
    """Everything the UI needs to render one modulation run."""

    bits: Bits
    clean: Signal
    noisy: Signal
    reference: Signal
    recovered_text: str
    matches: bool


def build_modulator(scheme: str, config: SimulationConfig) -> Modulator:
    """Return the modulator for a scheme name, wired from ``config``.

    Args:
        scheme: A key of :data:`ALL_SCHEMES`.
        config: The simulation parameters to build the modulator from.

    Returns:
        The configured baseband or carrier modulator.
    """
    if scheme in DIGITAL_SCHEMES:
        updated = config.model_copy(
            update={"digital_modulation": DIGITAL_SCHEMES[scheme]}
        )
        return factory.build_digital_modulator(updated)
    updated = config.model_copy(
        update={"carrier_modulation": CARRIER_SCHEMES[scheme]}
    )
    carrier = factory.build_carrier_modulator(updated)
    if carrier is None:  # pragma: no cover - scheme is always a real carrier
        raise ValueError(f"Unknown modulation scheme: {scheme}")
    return carrier


def run_modulation(
    scheme: str, config: SimulationConfig, text: str
) -> ModulationResult:
    """Encode ``text`` through ``scheme``, add channel noise, decode it back.

    Args:
        scheme: A key of :data:`ALL_SCHEMES`.
        config: The simulation parameters.
        text: The message to transmit.

    Returns:
        The bits, the clean and noisy signals, and the recovered text.
    """
    modulator = build_modulator(scheme, config)
    bits = text_to_bits(text)
    clean = modulator.modulate(bits)
    noisy = factory.build_channel(config).transmit(clean)
    recovered_bits = modulator.demodulate(noisy)
    return ModulationResult(
        bits=bits,
        clean=clean,
        noisy=noisy,
        reference=_unipolar(bits, config.amplitude_v, clean.size),
        recovered_text=_safe_text(recovered_bits),
        matches=recovered_bits == bits,
    )


def _unipolar(bits: Bits, amplitude: float, length: int) -> Signal:
    """A 0/amplitude square wave of ``bits``, for overlay as a reference.

    The bit timeline is stretched to ``length`` samples so it lines up with the
    modulated signal regardless of how many bits each symbol packs.
    """
    if not bits:
        return np.zeros(0)
    # Map each output sample back to the bit it belongs to.
    bit_index = np.arange(length) * len(bits) // length
    levels: Signal = np.where(np.array(bits)[bit_index], amplitude, 0.0)
    return levels


def _safe_text(bits: Bits) -> str:
    """Decode bits to text, tolerating the garbage a noisy channel yields."""
    try:
        return bits_to_text(bits)
    except (UnicodeDecodeError, ValueError):
        return "<undecodable>"
