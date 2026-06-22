# ./app/app.py
"""Streamlit front end: the full link + physical layer simulator.

Run from the project root with::

    uv run streamlit run app/app.py

Drives the shared :mod:`layer_manager.pipeline` in-process: text is taken
through the link layer (chunk -> error control -> framing) and the physical
layer (baseband or carrier modulation), pushed through the noisy channel, and
recovered. The waveform is drawn by a self-contained HTML canvas
(``scope.html``) that animates entirely in the browser -- no per-frame server
round-trips -- so the live plot stays smooth; the link-layer stages and the
per-frame receiver verdict are shown with ordinary Streamlit panels.
"""

import json
import sys
from pathlib import Path

# Make the project root importable when launched via `streamlit run app/app.py`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.simulation import RunResult, run_pipeline  # noqa: E402
from layer_manager.config import (  # noqa: E402
    CarrierModulation,
    CorrectionType,
    DetectionType,
    DigitalModulation,
    FramingType,
    SimulationConfig,
)
from layer_manager.factory import build_framer  # noqa: E402
from layer_manager.types import Signal  # noqa: E402

# Human labels -> config enums, for the sidebar selectboxes.
_FRAMING = {
    "Character count": FramingType.CHAR_COUNT,
    "Byte stuffing": FramingType.BYTE_STUFFING,
    "Bit stuffing": FramingType.BIT_STUFFING,
}
_DETECTION = {
    "None": DetectionType.NONE,
    "Parity": DetectionType.PARITY,
    "Checksum": DetectionType.CHECKSUM,
    "CRC-32": DetectionType.CRC32,
}
_CORRECTION = {
    "None": CorrectionType.NONE,
    "Hamming": CorrectionType.HAMMING,
}
_DIGITAL = {
    "NRZ-Polar": DigitalModulation.NRZ_POLAR,
    "Manchester": DigitalModulation.MANCHESTER,
    "Bipolar": DigitalModulation.BIPOLAR,
}
_CARRIER = {
    "None (baseband)": CarrierModulation.NONE,
    "ASK": CarrierModulation.ASK,
    "FSK": CarrierModulation.FSK,
    "QPSK": CarrierModulation.QPSK,
    "16-QAM": CarrierModulation.QAM16,
}

_SCOPE_HTML = (Path(__file__).parent / "scope.html").read_text()
_TRACE_HEIGHT = 96  # canvas pixels per stacked stage panel
_MAX_BITS_SHOWN = 512  # truncate long bit strings in the expanders
_MAX_TRACE_POINTS = 8000  # decimate any trace past this to bound the payload
_MAX_FRAMES_SHOWN = 64  # cap per-frame rows/lines so the DOM can't blow up


def sidebar_config() -> tuple[str, SimulationConfig]:
    """Render every configuration control and return ``(message, config)``."""
    st.sidebar.header("Configuration")
    message = st.sidebar.text_input("Message", value="Hi! TR1")

    st.sidebar.subheader("Link layer")
    max_frame = st.sidebar.slider("Max frame size (bytes)", 1, 16, 8)
    framing = st.sidebar.selectbox("Framing", list(_FRAMING))
    detection = st.sidebar.selectbox("Detection", list(_DETECTION), index=3)
    correction = st.sidebar.selectbox("Correction", list(_CORRECTION))
    block_bits = st.sidebar.slider("Checksum block (bits)", 4, 32, 8, 4)

    st.sidebar.subheader("Physical layer")
    digital = st.sidebar.selectbox("Baseband line code", list(_DIGITAL))
    carrier = st.sidebar.selectbox("Carrier", list(_CARRIER))
    amplitude = st.sidebar.slider("Amplitude (V)", 0.5, 5.0, 1.0, 0.5)
    samples = st.sidebar.slider("Samples per symbol", 20, 200, 100, 20)
    frequency = st.sidebar.slider("Carrier frequency (Hz)", 10.0, 100.0, 20.0)
    sample_rate = st.sidebar.slider("Sample rate (Hz)", 200.0, 4000.0, 1000.0)

    st.sidebar.subheader("Channel")
    noise_std = st.sidebar.slider("Noise sigma", 0.0, 1.0, 0.0, 0.05)

    config = SimulationConfig(
        max_frame_size=max_frame,
        framing=_FRAMING[framing],
        detection=_DETECTION[detection],
        correction=_CORRECTION[correction],
        checksum_block_bits=block_bits,
        digital_modulation=_DIGITAL[digital],
        carrier_modulation=_CARRIER[carrier],
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
    return message, config


@st.cache_data(show_spinner=False)
def cached_run(message: str, config_json: str) -> RunResult:
    """Run (and memoize) the whole pipeline, keyed on primitive arguments.

    Args:
        message: The text to transmit.
        config_json: ``SimulationConfig`` serialised with ``model_dump_json``.

    Returns:
        The full :class:`app.simulation.RunResult`.
    """
    config = SimulationConfig.model_validate_json(config_json)
    return run_pipeline(config, message)


def _bits_str(bits: list[bool]) -> str:
    """Render bits as a 0/1 string, truncated if very long."""
    text = "".join("1" if bit else "0" for bit in bits)
    if len(text) > _MAX_BITS_SHOWN:
        return f"{text[:_MAX_BITS_SHOWN]}... ({len(bits)} bits)"
    return text or "(empty)"


def _frames_str(frames: list[list[bool]]) -> str:
    """Render each frame's bits on its own line (one frame per row)."""
    if not frames:
        return "(no frames)"
    lines = [
        f"frame {index}: {_bits_str(frame)}"
        for index, frame in enumerate(frames[:_MAX_FRAMES_SHOWN])
    ]
    if len(frames) > _MAX_FRAMES_SHOWN:
        lines.append(f"... (+{len(frames) - _MAX_FRAMES_SHOWN} more frames)")
    return "\n".join(lines)


def _bits_trace(
    label: str,
    color: str,
    bits: list[bool],
    regions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """A digital (square-wave) canvas trace built from a bit sequence.

    ``regions`` optionally shade structural bits (framing/EDC/Hamming).
    Decimated past ``_MAX_TRACE_POINTS`` bits so a long message (or a small
    frame size, which multiplies framing overhead) can't bloat the iframe; when
    thinned, the per-bit overlays (``regions`` and the byte grid) are dropped
    because their bit indices no longer line up with the thinned data. They are
    also dropped for an empty trace (an undecodable receive yields no frames):
    a band over zero-length data divides by zero in the canvas and spins the
    draw loop forever.
    """
    step = max(1, len(bits) // _MAX_TRACE_POINTS)
    keep_overlays = step == 1 and len(bits) > 0
    return {
        "label": label,
        "color": color,
        "digital": True,
        "data": [1 if bit else 0 for bit in bits[::step]],
        "regions": (regions or []) if keep_overlays else [],
        "bytegrid": keep_overlays,
    }


def _signal_trace(label: str, color: str, signal: Signal) -> dict[str, object]:
    """An analog (line) canvas trace, decimated to keep the payload small.

    A signal is ``framed_bits * samples_per_symbol`` samples -- for a long
    message that is hundreds of thousands of floats, which bloats the iframe
    JSON until the browser kills the tab. The canvas draws one point per pixel,
    so decimating to ``_MAX_TRACE_POINTS`` preserves the full-view shape while
    bounding the payload. Analog traces carry no bit overlays, so it is safe.
    """
    step = max(1, signal.size // _MAX_TRACE_POINTS)
    return {
        "label": label,
        "color": color,
        "digital": False,
        "data": signal[::step].round(3).tolist(),
    }


def _concat(frames: list[list[bool]]) -> list[bool]:
    """Flatten per-frame bits into one continuous stream."""
    return [bit for frame in frames for bit in frame]


def _protected_regions(
    run: RunResult, config: SimulationConfig
) -> list[dict[str, object]]:
    """Bands over the error-control bits of the concatenated protected frames.

    Marks the Hamming parity bits (power-of-two positions) when correction is
    on; otherwise the trailing detector code (CRC / checksum / parity).
    """
    regions: list[dict[str, object]] = []
    offset = 0
    for raw, protected in zip(run.tx.frames, run.tx.protected, strict=True):
        if config.correction is CorrectionType.HAMMING:
            power = 1
            labelled = False
            while power <= len(protected):
                regions.append({
                    "from": offset + power - 1,
                    "to": offset + power,
                    "color": "#f78c6c",
                    "label": "" if labelled else "Hamming parity",
                })
                labelled = True
                power *= 2
        elif config.detection is not DetectionType.NONE:
            regions.append({
                "from": offset + len(raw),
                "to": offset + len(protected),
                "color": "#c792ea",
                "label": config.detection.value,
            })
        offset += len(protected)
    return regions


def _framed_regions(
    run: RunResult, config: SimulationConfig
) -> list[dict[str, object]]:
    """Bands over the framing overhead of the framed stream.

    Marks the character-count header, or the opening/closing FLAG of byte/bit
    stuffing, per frame. Per-frame framing concatenates, so framing each frame
    alone gives exact offsets into the full stream.
    """
    framer = build_framer(config)
    regions: list[dict[str, object]] = []
    offset = 0
    for protected in run.tx.protected:
        length = len(framer.frame([protected]))
        if config.framing is FramingType.CHAR_COUNT:
            regions.append({
                "from": offset, "to": offset + 8,
                "color": "#48d1cc", "label": "count",
            })
        else:
            regions.append({
                "from": offset, "to": offset + 8,
                "color": "#48d1cc", "label": "FLAG",
            })
            regions.append({
                "from": offset + length - 8, "to": offset + length,
                "color": "#48d1cc", "label": "FLAG",
            })
        offset += length
    return regions


def render_scope(run: RunResult, config: SimulationConfig) -> None:
    """Graph every pipeline stage as a waveform, in order.

    The digital bit stages (message bits, before framing, with framing, then
    the receive inverse) are drawn as square waves; the analog signal stages
    (modulated, received) as lines. The bit stages carry shaded bands marking
    the framing / EDC / Hamming structure. All panels share one fractional
    viewport in the HTML canvas, so they stay aligned and scroll together.
    """
    tx, rx = run.tx, run.rx
    if tx.signal.size == 0:
        st.info("Type a message in the sidebar to see the waveforms.")
        return

    protected_regions = _protected_regions(run, config)
    framed_regions = _framed_regions(run, config)
    traces: list[dict[str, object]] = [
        _bits_trace("TX · message bits", "#9aa0a6", tx.bits),
        _bits_trace(
            "TX · before framing (error-controlled)",
            "#aeb4be",
            _concat(tx.protected),
            protected_regions,
        ),
        _bits_trace(
            "TX · with framing (on the wire)",
            "#5c9dff",
            tx.framed,
            framed_regions,
        ),
        _signal_trace("TX · modulated", "#5c9dff", tx.signal),
        _signal_trace("RX · received (noisy)", "#ffa24d", run.noisy),
        _bits_trace(
            "RX · with framing (demodulated)",
            "#ffd166",
            rx.framed,
            framed_regions,
        ),
        _bits_trace(
            "RX · after deframing",
            "#aeb4be",
            _concat([report.received for report in rx.frames]),
            protected_regions,
        ),
        _bits_trace("RX · recovered bits", "#7ee787", rx.bits),
    ]
    height = _TRACE_HEIGHT * len(traces)
    html = (
        _SCOPE_HTML.replace("__PAYLOAD__", json.dumps({"traces": traces}))
        .replace("__HEIGHT__", str(height))
        .replace("__WIN0__", "12")
    )
    st.iframe(html, height=height + 70)


def render_link_layer(run: RunResult, config: SimulationConfig) -> None:
    """Show every transmit and receive pipeline step, stage by stage."""
    tx, rx = run.tx, run.rx
    has_detector = config.detection is not DetectionType.NONE
    has_corrector = config.correction is not CorrectionType.NONE
    active = [
        name
        for name, on in (
            ("detection", has_detector),
            ("correction", has_corrector),
        )
        if on
    ]
    control = ", ".join(active) if active else "none"

    transmit_col, receive_col = st.columns(2)
    with transmit_col:
        st.subheader("Transmit")
        st.caption(
            f"text → {len(tx.bits)} bits → chunk → {len(tx.frames)} frame(s) "
            f"→ error control ({control}) → framing → {len(tx.framed)} bits "
            f"→ {tx.signal.size} signal samples"
        )
        pairs = list(zip(tx.frames, tx.protected, strict=True))
        tx_rows: list[dict[str, object]] = [
            {
                "frame": index,
                "payload bytes": len(raw) // 8,
                "protected bits": len(protected),
            }
            for index, (raw, protected) in enumerate(pairs[:_MAX_FRAMES_SHOWN])
        ]
        if tx_rows:
            st.table(tx_rows)
            if len(pairs) > _MAX_FRAMES_SHOWN:
                st.caption(f"{_MAX_FRAMES_SHOWN} of {len(pairs)} frames shown")
        with st.expander("Every transmit step (exact bits)"):
            st.caption("1 · message")
            st.code(tx.text or "(empty)")
            st.caption("2 · bits — application codec (text → bits)")
            st.code(_bits_str(tx.bits))
            st.caption("3 · frames — chunked, before error control")
            st.code(_frames_str(tx.frames))
            st.caption(f"4 · protected — after error control ({control})")
            st.code(_frames_str(tx.protected))
            st.caption("5 · framed — concatenated, on the wire")
            st.code(_bits_str(tx.framed))

    with receive_col:
        st.subheader("Receive")
        st.caption(
            f"{rx.signal.size} samples → demod → {len(rx.framed)} bits "
            f"→ deframe → {len(rx.frames)} frame(s) → error control → text"
        )
        rx_rows: list[dict[str, object]] = [
            {
                "frame": index,
                "detected": ("✓" if report.detected_ok else "✗")
                if has_detector
                else "—",
                "corrected": ("✓" if report.corrected else "·")
                if has_corrector
                else "—",
            }
            for index, report in enumerate(rx.frames[:_MAX_FRAMES_SHOWN])
        ]
        if rx_rows:
            st.table(rx_rows)
            if len(rx.frames) > _MAX_FRAMES_SHOWN:
                st.caption(
                    f"{_MAX_FRAMES_SHOWN} of {len(rx.frames)} frames shown"
                )
        with st.expander("Every receive step (exact bits)"):
            st.caption("1 · framed — demodulated off the wire")
            st.code(_bits_str(rx.framed))
            st.caption("2 · received frames — deframed, still protected")
            st.code(_frames_str([report.received for report in rx.frames]))
            st.caption(f"3 · recovered — after error control ({control})")
            st.code(_frames_str([report.payload for report in rx.frames]))
            st.caption("4 · bits — joined payloads")
            st.code(_bits_str(rx.bits))
            st.caption("5 · message — bits → text")
            st.code(rx.text or "(empty)")


def main() -> None:
    """Compose the page."""
    st.set_page_config(page_title="TR1 Simulator", layout="wide")
    st.title("TR1 — Link + Physical layer simulator")
    message, config = sidebar_config()

    try:
        run = cached_run(message, config.model_dump_json())
    except ValueError as exc:
        st.error(f"This layer combination can't run: {exc}")
        st.info(
            "Tip: pair parity/Hamming with bit-stuffing framing (the byte "
            "framers need byte-aligned frames), and give QPSK/16-QAM a framed "
            "length divisible by 2/4 — or use a baseband code or ASK/FSK."
        )
        return

    sent, frames, recovered = st.columns(3)
    sent.metric("Sent", message or "(empty)")
    frames.metric("Frames", len(run.tx.frames))
    recovered.metric("Recovered", run.rx.text or "(empty)")

    if run.rx.text == message:
        st.success("Receiver recovered the message exactly.")
    elif run.rx.text == "<undecodable>":
        st.error(
            "The receiver couldn't parse the frame structure — the channel "
            "corrupted it beyond recovery (e.g. a lost or spurious FLAG). "
            "Lower the noise, or add Hamming correction."
        )
    elif config.detection is DetectionType.NONE:
        st.warning(
            "No error detection is enabled and the recovered text differs — "
            "the channel likely corrupted it. Add a detector (e.g. CRC-32) or "
            "lower the noise."
        )
    elif run.rx.ok:
        st.warning(
            "The detector passed but the text still differs — an error slipped "
            "past it under heavy noise. Try CRC-32, or add Hamming correction."
        )
    else:
        st.error(
            "The link detected errors. Lower the noise, or add Hamming "
            "correction to repair single-bit flips."
        )

    render_scope(run, config)
    render_link_layer(run, config)


main()
