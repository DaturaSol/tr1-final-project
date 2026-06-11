# ./tests/test_utils.py
"""Tests for the application-layer text/bit codec."""

import pytest

from layer_manager.utils import bits_to_text, text_to_bits

# Preserved from the original module demo.
SAMPLE_TEXT = "Hello friend."


def test_text_to_bits_known_vector() -> None:
    """'A' (0x41) encodes to its 8-bit big-endian representation."""
    assert text_to_bits("A") == [
        False, True, False, False, False, False, False, True,
    ]


def test_length_is_eight_bits_per_byte() -> None:
    """Each UTF-8 byte becomes exactly eight bits."""
    assert len(text_to_bits(SAMPLE_TEXT)) == len(SAMPLE_TEXT.encode()) * 8


@pytest.mark.parametrize("text", ["", "A", "café", "1234567890", SAMPLE_TEXT])
def test_round_trip(text: str) -> None:
    """Decoding the encoded text returns the original string."""
    assert bits_to_text(text_to_bits(text)) == text
