"""Tests for the data-link error-detection schemes."""

import pytest

from layer_manager.link.detection import (
    ChecksumDetector,
    CRC32Detector,
    ParityDetector,
)
from layer_manager.types import Bits
from layer_manager.utils import text_to_bits


def _flip(bits: Bits, *indices: int) -> Bits:
    """Return a copy of ``bits`` with the bits at ``indices`` inverted."""
    out = list(bits)
    for i in indices:
        out[i] = not out[i]
    return out


# --- ParityDetector ---


def test_parity_appends_one_bit() -> None:
    """Encoding adds exactly one parity bit."""
    det = ParityDetector()
    data = text_to_bits("Hi")
    assert len(det.encode(data)) == len(data) + 1


@pytest.mark.parametrize("text", ["", "A", "Hi!", "Hello friend.", "~"])
def test_parity_round_trip(text: str) -> None:
    """A clean block validates and strips back to the original payload."""
    det = ParityDetector()
    data = text_to_bits(text)
    payload, ok = det.check(det.encode(data))
    assert ok
    assert payload == data


@pytest.mark.parametrize("text", ["A", "Hi!", "Hello friend."])
def test_parity_encoded_count_is_even(text: str) -> None:
    """The encoded block always carries an even number of ones."""
    det = ParityDetector()
    encoded = det.encode(text_to_bits(text))
    assert sum(encoded) % 2 == 0


def test_parity_single_flip_detected() -> None:
    """A single corrupted bit makes the parity odd and is detected."""
    det = ParityDetector()
    encoded = det.encode(text_to_bits("Hi!"))
    _, ok = det.check(_flip(encoded, 3))
    assert not ok


def test_parity_double_flip_undetected() -> None:
    """Two bit flips keep the parity even, so they slip past (known limit)."""
    det = ParityDetector()
    encoded = det.encode(text_to_bits("Hi!"))
    _, ok = det.check(_flip(encoded, 3, 5))
    assert ok  # NOT detected -- documents parity's weakness


# --- ChecksumDetector ---


@pytest.mark.parametrize("block_bits", [8, 16])
def test_checksum_appends_block(block_bits: int) -> None:
    """Encoding adds exactly one block_bits-wide checksum field."""
    det = ChecksumDetector(block_bits)
    data = text_to_bits("Hello")
    assert len(det.encode(data)) == len(data) + block_bits


@pytest.mark.parametrize("block_bits", [8, 16])
@pytest.mark.parametrize("text", ["", "A", "Hi!", "Hello friend.", "~"])
def test_checksum_round_trip(block_bits: int, text: str) -> None:
    """A clean block validates and strips back to the original payload."""
    det = ChecksumDetector(block_bits)
    data = text_to_bits(text)
    payload, ok = det.check(det.encode(data))
    assert ok
    assert payload == data


@pytest.mark.parametrize("block_bits", [8, 16])
def test_checksum_single_flip_detected(block_bits: int) -> None:
    """Any single corrupted payload bit changes the sum and is detected."""
    det = ChecksumDetector(block_bits)
    encoded = det.encode(text_to_bits("Hello friend."))
    for i in range(len(encoded) - block_bits):  # every payload bit
        _, ok = det.check(_flip(encoded, i))
        assert not ok, f"flip at {i} slipped through"


def test_checksum_flip_in_checksum_detected() -> None:
    """Corrupting the checksum field itself is detected too."""
    det = ChecksumDetector(8)
    encoded = det.encode(text_to_bits("Hello"))
    _, ok = det.check(_flip(encoded, len(encoded) - 1))  # last bit = checksum
    assert not ok


def test_checksum_end_around_carry() -> None:
    """All-ones blocks exercise the carry fold and still round-trip."""
    det = ChecksumDetector(8)
    data = [True] * 24  # three 0xFF bytes -> sums overflow and fold
    payload, ok = det.check(det.encode(data))
    assert ok
    assert payload == data


# --- CRC32Detector ---


def _bits_to_int(bits: Bits) -> int:
    """Read a bit list (MSB first) as an unsigned integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


@pytest.mark.parametrize(
    "text", ["", "A", "Hi!", "Hello friend.", "~", "x" * 40]
)
def test_crc_round_trip(text: str) -> None:
    """A clean block validates and strips back to the original payload."""
    det = CRC32Detector()
    data = text_to_bits(text)
    payload, ok = det.check(det.encode(data))
    assert ok
    assert payload == data


def test_crc_appends_32_bits() -> None:
    """Encoding adds exactly a 32-bit remainder."""
    det = CRC32Detector()
    data = text_to_bits("Hello")
    assert len(det.encode(data)) == len(data) + 32


def test_crc_standard_check_vector() -> None:
    """Canonical CRC-32/ISO-HDLC: CRC of '123456789' is 0xCBF43926."""
    det = CRC32Detector()
    encoded = det.encode(text_to_bits("123456789"))
    assert _bits_to_int(encoded[-32:]) == 0xCBF43926


def test_crc_single_flip_detected() -> None:
    """Every single-bit flip (payload or CRC field) is detected."""
    det = CRC32Detector()
    encoded = det.encode(text_to_bits("Hello friend."))
    for i in range(len(encoded)):
        _, ok = det.check(_flip(encoded, i))
        assert not ok, f"flip at {i} slipped through"


def test_crc_burst_error_detected() -> None:
    """A short burst error is detected (CRC-32 catches bursts <= 32 bits)."""
    det = CRC32Detector()
    encoded = det.encode(text_to_bits("Hello friend."))
    _, ok = det.check(_flip(encoded, 5, 6, 7, 8))
    assert not ok
