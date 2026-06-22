# ./src/layer_manager/pipeline.py
"""End-to-end simulator orchestration: text <-> signal.

This is the assignment's *Simulador* -- the routine that calls every layer in
order. It lives in ``layer_manager`` (rather than in ``app/`` or a single node)
because the Streamlit GUI and the three socket nodes all need the *same*
pipeline, and ``src/layer_manager`` is the only code shared across the node
containers.

It owns no algorithms: it only sequences the strategies built by
:mod:`layer_manager.factory`. The transmit side stops at the **clean** signal;
the noisy channel is applied separately (by the ``ch`` node, or by the GUI), so
these functions serve both the in-process GUI and the over-the-wire nodes.

Transmit (receive inverts it, step by step)::

    text -> bits -> chunk(max_frame_size)
         -> per frame: [detector.encode] then [corrector.encode]
         -> framer.frame -> modulate (baseband, or carrier if one is selected)
         -> Signal

Error control is layered detector-inside-corrector: the transmitter adds the
integrity code first and the FEC around it, so the receiver corrects bit errors
*before* the detector has the final say on whether the frame is clean. Baseband
and carrier modulation are *alternatives* -- the carrier (ASK/FSK/QPSK/16-QAM)
maps the bits straight to a passband signal and replaces the baseband line code
on the wire when selected; the baseband signal is still computed for display.
"""

from dataclasses import dataclass

from layer_manager import factory
from layer_manager.config import (
    CarrierModulation,
    FramingType,
    SimulationConfig,
)
from layer_manager.link.framing import chunk, join
from layer_manager.types import Bits, Signal
from layer_manager.utils import bits_to_text, text_to_bits


@dataclass
class TxResult:
    """Every stage the transmitter produced, for display and transmission."""

    text: str
    bits: Bits  # application codec output
    frames: list[Bits]  # payloads after chunking to max_frame_size
    protected: list[Bits]  # frames after detector/corrector encoding
    framed: Bits  # one bit stream after framing
    baseband: Signal  # baseband line code (always computed, for display)
    signal: Signal  # what goes on the wire (carrier if selected, else baseband)


@dataclass
class FrameReport:
    """The receiver's verdict for one recovered frame."""

    received: Bits  # the frame as deframed, before error control is undone
    payload: Bits  # the recovered payload, codes stripped
    detected_ok: bool  # detector found no error (True when no detector is used)
    corrected: bool  # the corrector repaired a bit error (False when none used)


@dataclass
class RxResult:
    """Every stage the receiver recovered, for display."""

    signal: Signal  # the received (possibly noisy) signal
    framed: Bits  # the demodulated bit stream
    frames: list[FrameReport]  # one report per recovered frame
    bits: Bits  # the joined payload bits
    text: str  # the decoded message (``"<undecodable>"`` on garbage)
    corrected: int  # number of frames the corrector repaired
    ok: bool  # True when every frame passed its detector


def transmit(config: SimulationConfig, text: str) -> TxResult:
    """Run the full transmit stack, returning every intermediate stage.

    Args:
        config: The simulation parameters (shared with the receiver).
        text: The message to send.

    Returns:
        A :class:`TxResult` whose ``signal`` is the clean (noiseless) waveform.
    """
    bits = text_to_bits(text)
    frames = chunk(bits, config.max_frame_size)

    detector = factory.build_detector(config)
    corrector = factory.build_corrector(config)
    protected: list[Bits] = []
    for frame in frames:
        payload = frame
        if detector is not None:
            payload = detector.encode(payload)  # inner: integrity code
        if corrector is not None:
            payload = corrector.encode(payload)  # outer: FEC over data + code
        protected.append(payload)

    _require_byte_aligned(config, protected)
    framed = factory.build_framer(config).frame(protected)
    baseband = factory.build_digital_modulator(config).modulate(framed)
    carrier = factory.build_carrier_modulator(config)
    if carrier is None:
        signal = baseband
    else:
        _require_divisible(config, framed)
        signal = carrier.modulate(framed)
    return TxResult(text, bits, frames, protected, framed, baseband, signal)


def receive(config: SimulationConfig, signal: Signal) -> RxResult:
    """Run the full receive stack, inverting :func:`transmit`.

    Args:
        config: The simulation parameters (must match the transmitter's).
        signal: The received signal, typically after the noisy channel.

    Returns:
        An :class:`RxResult` with the decoded text and a per-frame verdict.
    """
    carrier = factory.build_carrier_modulator(config)
    if carrier is not None:
        framed = carrier.demodulate(signal)
    else:
        framed = factory.build_digital_modulator(config).demodulate(signal)

    detector = factory.build_detector(config)
    corrector = factory.build_corrector(config)
    try:
        protected = factory.build_framer(config).deframe(framed)
        reports: list[FrameReport] = []
        for received in protected:
            payload = received
            corrected = False
            if corrector is not None:
                payload, corrected = corrector.decode(payload)  # undo FEC
            ok = True
            if detector is not None:
                payload, ok = detector.check(payload)  # verify integrity
            reports.append(
                FrameReport(
                    received=received,
                    payload=payload,
                    detected_ok=ok,
                    corrected=corrected,
                )
            )
    except ValueError:
        # A corrupted stream can be unparseable (e.g. bit stuffing finding an
        # odd number of FLAGs). That is a noisy-channel outcome, not a crash:
        # report an undecodable receive rather than letting it propagate.
        return RxResult(
            signal=signal,
            framed=framed,
            frames=[],
            bits=[],
            text="<undecodable>",
            corrected=0,
            ok=False,
        )

    bits = join([report.payload for report in reports])
    ok = all(report.detected_ok for report in reports)
    if not reports and signal.size > 0:
        ok = False  # a non-empty signal that yields no frames lost everything
    return RxResult(
        signal=signal,
        framed=framed,
        frames=reports,
        bits=bits,
        text=_safe_text(bits),
        corrected=sum(report.corrected for report in reports),
        ok=ok,
    )


def _safe_text(bits: Bits) -> str:
    """Decode bits to text, tolerating the garbage a noisy channel yields."""
    try:
        decoded: str = bits_to_text(bits)
    except (UnicodeDecodeError, ValueError):
        return "<undecodable>"
    return decoded


# Bits packed per carrier symbol; absent schemes (ASK/FSK/none) carry one.
_BITS_PER_SYMBOL = {CarrierModulation.QPSK: 2, CarrierModulation.QAM16: 4}


def _require_byte_aligned(config: SimulationConfig, frames: list[Bits]) -> None:
    """Reject non-byte-aligned frames before a byte framer mangles them.

    Character-count and byte-stuffing framing pack whole bytes, so a code that
    leaves an odd bit count (parity, Hamming) must ride bit stuffing instead.
    Without this guard the byte framers silently pad/truncate the payload and
    the round trip returns wrong-but-plausible data.

    Raises:
        ValueError: If a byte framer is paired with a non-byte-aligned frame.
    """
    if config.framing is FramingType.BIT_STUFFING:
        return
    bad = next((len(frame) for frame in frames if len(frame) % 8), None)
    if bad is not None:
        raise ValueError(
            "Character-count and byte-stuffing framing need byte-aligned "
            f"frames, but error control produced {bad} bits. Pair parity or "
            "Hamming with bit stuffing, or use a byte-sized detector (CRC-32)."
        )


def _require_divisible(config: SimulationConfig, framed: Bits) -> None:
    """Reject a framed length the chosen carrier cannot pack into symbols.

    Raises:
        ValueError: If QPSK/16-QAM gets a stream whose length is not a
            multiple of its bits-per-symbol (a raw reshape error otherwise).
    """
    per_symbol = _BITS_PER_SYMBOL.get(config.carrier_modulation, 1)
    if len(framed) % per_symbol:
        raise ValueError(
            f"{config.carrier_modulation.value} packs {per_symbol} bits per "
            f"symbol, but the framed stream is {len(framed)} bits. Use a "
            "baseband line code or ASK/FSK, or adjust the message length."
        )
