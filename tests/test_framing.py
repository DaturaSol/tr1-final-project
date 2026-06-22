"""Tests for the data-link framing schemes and the chunk/join helpers."""

import pytest

from layer_manager.link.framing import (
    BitStuffingFramer,
    ByteStuffingFramer,
    CharCountFramer,
    chunk,
    join,
)
from layer_manager.utils import text_to_bits

# --- chunk / join (the split/merge bookends) ---


@pytest.mark.parametrize(
    "text", ["", "A", "Hi!", "Hello friend.", "x" * 40]
)
def test_join_chunk_round_trip(text: str) -> None:
    """join(chunk(bits)) returns the original stream for any frame size."""
    bits = text_to_bits(text)
    assert join(chunk(bits, max_frame_size=4)) == bits


@pytest.mark.parametrize("max_frame_size", [1, 2, 3, 8])
def test_chunk_sizes(max_frame_size: int) -> None:
    """Every chunk but the last is exactly max_frame_size bytes."""
    bits = text_to_bits("abcdefgh")  # 8 bytes -> 64 bits
    frames = chunk(bits, max_frame_size)
    for frame in frames[:-1]:
        assert len(frame) == max_frame_size * 8
    assert 0 < len(frames[-1]) <= max_frame_size * 8
    assert join(frames) == bits


def test_chunk_join_empty() -> None:
    """Empty input gives no frames, and joins back to empty."""
    assert chunk([], 4) == []
    assert join([]) == []


# --- CharCountFramer (the round-trip contract: deframe(frame(x)) == x) ---


def test_charcount_empty_list() -> None:
    """No payloads frames to an empty stream and back."""
    framer = CharCountFramer()
    assert framer.deframe(framer.frame([])) == []


def test_charcount_single_one_byte_frame() -> None:
    """A single one-byte payload survives the round trip."""
    framer = CharCountFramer()
    payloads = [text_to_bits("A")]  # 1 byte
    assert framer.deframe(framer.frame(payloads)) == payloads


def test_charcount_two_frames() -> None:
    """Two frames are sliced back apart at the right boundary."""
    framer = CharCountFramer()
    payloads = [text_to_bits("Hi"), text_to_bits("!")]  # 2 bytes, 1 byte
    assert framer.deframe(framer.frame(payloads)) == payloads


def test_charcount_multibyte_frame() -> None:
    """A frame whose byte count needs more than one digit round-trips."""
    framer = CharCountFramer()
    payloads = [text_to_bits("x" * 12)]  # count = 12
    assert framer.deframe(framer.frame(payloads)) == payloads


@pytest.mark.parametrize(
    "text", ["A", "Hi!", "Hello friend.", "x" * 40]
)
def test_charcount_via_chunk_round_trip(text: str) -> None:
    """The realistic path: chunk -> frame -> deframe -> recovers chunks."""
    framer = CharCountFramer()
    payloads = chunk(text_to_bits(text), max_frame_size=4)
    assert framer.deframe(framer.frame(payloads)) == payloads


# --- ByteStuffingFramer ('~' is 0x7E = FLAG, '}' is 0x7D = ESC) ---


def test_bytestuff_empty_list() -> None:
    """No payloads frames to an empty stream and back."""
    framer = ByteStuffingFramer()
    assert framer.deframe(framer.frame([])) == []


def test_bytestuff_single_frame() -> None:
    """A single ordinary payload survives the round trip."""
    framer = ByteStuffingFramer()
    payloads = [text_to_bits("Hi")]
    assert framer.deframe(framer.frame(payloads)) == payloads


def test_bytestuff_two_frames() -> None:
    """Two frames come back as two separate payloads, in order."""
    framer = ByteStuffingFramer()
    payloads = [text_to_bits("Hi"), text_to_bits("!")]
    assert framer.deframe(framer.frame(payloads)) == payloads


def test_bytestuff_flag_in_payload() -> None:
    """A payload byte equal to FLAG must be escaped and recovered."""
    framer = ByteStuffingFramer()
    payloads = [text_to_bits("a~b")]  # '~' == 0x7E == FLAG
    assert framer.deframe(framer.frame(payloads)) == payloads


def test_bytestuff_esc_in_payload() -> None:
    """A payload byte equal to ESC must be escaped and recovered."""
    framer = ByteStuffingFramer()
    payloads = [text_to_bits("a}b")]  # '}' == 0x7D == ESC
    assert framer.deframe(framer.frame(payloads)) == payloads


@pytest.mark.parametrize(
    "text", ["A", "Hi!", "~~~", "a~}b", "Hello friend.", "x" * 40]
)
def test_bytestuff_via_chunk_round_trip(text: str) -> None:
    """The realistic path: chunk -> frame -> deframe -> recovers chunks."""
    framer = ByteStuffingFramer()
    payloads = chunk(text_to_bits(text), max_frame_size=4)
    assert framer.deframe(framer.frame(payloads)) == payloads


# --- BitStuffingFramer ('~' = 0x7E, whose bits are the flag pattern) ---


def test_bitstuff_empty_list() -> None:
    """No payloads frames to an empty stream and back."""
    framer = BitStuffingFramer()
    assert framer.deframe(framer.frame([])) == []


def test_bitstuff_single_frame() -> None:
    """A single ordinary payload survives the round trip."""
    framer = BitStuffingFramer()
    payloads = [text_to_bits("Hi")]
    assert framer.deframe(framer.frame(payloads)) == payloads


def test_bitstuff_two_frames() -> None:
    """Two (differently sized) frames come back as two payloads, in order."""
    framer = BitStuffingFramer()
    payloads = [text_to_bits("Hi"), text_to_bits("!")]
    assert framer.deframe(framer.frame(payloads)) == payloads


def test_bitstuff_long_run_of_ones() -> None:
    """Runs of >5 ones are stuffed and recovered exactly."""
    framer = BitStuffingFramer()
    payloads = [[True] * 8, [True] * 16]  # 0xFF bytes: long runs of 1s
    assert framer.deframe(framer.frame(payloads)) == payloads


def test_bitstuff_flag_pattern_in_payload() -> None:
    """A payload whose bits equal the flag pattern must not break framing."""
    framer = BitStuffingFramer()
    payloads = [text_to_bits("~")]  # 0x7E bits == 01111110 == the flag
    assert framer.deframe(framer.frame(payloads)) == payloads


@pytest.mark.parametrize(
    "text", ["A", "Hi!", "~~~", "Hello friend.", "x" * 40]
)
def test_bitstuff_via_chunk_round_trip(text: str) -> None:
    """The realistic path: chunk -> frame -> deframe -> recovers chunks."""
    framer = BitStuffingFramer()
    payloads = chunk(text_to_bits(text), max_frame_size=4)
    assert framer.deframe(framer.frame(payloads)) == payloads
