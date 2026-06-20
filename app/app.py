# ./app/app.py
"""Streamlit front end: a live view of the physical-layer modulation.

Run from the project root with::

    uv run streamlit run app/app.py

Scope (for now): the physical layer only. Pick a message and a modulation
scheme, watch the waveform scroll like an oscilloscope, and see whether the
receiver recovers the text after the noisy channel. The data-link layer and the
end-to-end run over the Docker nodes come later.
"""

import sys
from pathlib import Path

# Make the project root importable when launched via `streamlit run app/app.py`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.simulation import (  # noqa: E402
    ALL_SCHEMES,
    ModulationResult,
    run_modulation,
)
from layer_manager.config import SimulationConfig  # noqa: E402

_REFRESH_SECONDS = 0.12


def sidebar_config() -> tuple[str, str, SimulationConfig]:
    """Render the sidebar controls and return (scheme, message, config)."""
    st.sidebar.header("Configuration")
    message = st.sidebar.text_input("Message", value="Hi!")
    scheme = st.sidebar.selectbox("Modulation", list(ALL_SCHEMES))
    amplitude = st.sidebar.slider("Amplitude (V)", 0.5, 5.0, 1.0, 0.5)
    samples = st.sidebar.slider("Samples per symbol", 20, 200, 100, 20)
    frequency = st.sidebar.slider("Carrier frequency (Hz)", 10.0, 100.0, 20.0)
    sample_rate = st.sidebar.slider("Sample rate (Hz)", 200.0, 4000.0, 1000.0)
    noise_std = st.sidebar.slider("Noise sigma", 0.0, 1.0, 0.0, 0.05)
    config = SimulationConfig(
        amplitude_v=amplitude,
        samples_per_symbol=samples,
        carrier_frequency=frequency,
        sample_rate=sample_rate,
        noise_std=noise_std,
    )
    cycles = frequency * samples / sample_rate
    st.sidebar.caption(
        f"Carrier cycles per symbol: {cycles:.2f} (whole = cleaner)"
    )
    return scheme, message, config


def _plot_window(
    result: ModulationResult, offset: int, width: int, noise_on: bool
) -> None:
    """Draw the bit pattern and waveform(s) as stacked, aligned panels."""
    end = offset + width
    st.caption("Bits (unipolar V)")
    st.line_chart(result.reference[offset:end], height=140)
    st.caption("Modulated (V)")
    st.line_chart(result.clean[offset:end], height=140)
    if noise_on:
        st.caption("Noisy (V)")
        st.line_chart(result.noisy[offset:end], height=140)
    st.caption(f"samples {offset}-{end} of {result.clean.size}")


def render_oscilloscope(
    result: ModulationResult, config: SimulationConfig
) -> None:
    """Render the scrolling waveform with play/scrub controls."""
    samples = config.samples_per_symbol
    width = st.slider("Window (symbols)", 1, 20, 6) * samples
    playing = st.toggle("Play (scroll)")
    noise_on = config.noise_std > 0
    max_offset = max(0, result.clean.size - width)
    if "offset" not in st.session_state:
        st.session_state.offset = 0

    if not playing and max_offset > 0:
        st.session_state.offset = st.slider(
            "Position (sample)",
            0,
            max_offset,
            min(st.session_state.offset, max_offset),
        )

    @st.fragment(run_every=_REFRESH_SECONDS if playing else None)
    def _draw() -> None:
        if playing and max_offset > 0:
            st.session_state.offset = (
                st.session_state.offset + samples
            ) % (max_offset + 1)
        offset = min(st.session_state.offset, max_offset)
        _plot_window(result, offset, width, noise_on)

    _draw()


def main() -> None:
    """Compose the page."""
    st.set_page_config(page_title="TR1 Simulator", layout="wide")
    st.title("TR1 - Physical-layer modulation (live)")
    scheme, message, config = sidebar_config()
    result = run_modulation(scheme, config, message)

    left, right = st.columns(2)
    left.metric("Sent", message or "(empty)")
    right.metric("Recovered", result.recovered_text or "(empty)")
    if result.matches:
        st.success("Receiver recovered the message exactly.")
    else:
        st.error("Recovered bits differ from the original (adjust noise).")

    render_oscilloscope(result, config)

    with st.expander("Bit stream"):
        bit_string = "".join("1" if bit else "0" for bit in result.bits)
        st.code(bit_string or "(empty)")


main()
