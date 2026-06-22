"""Tests for the data-link Hamming error corrector."""

import pytest

from layer_manager.link.correction import HammingCorrector
from layer_manager.types import Bits
from layer_manager.utils import text_to_bits


def _flip(bits: Bits, i: int) -> Bits:
    """Return a copy of ``bits`` with bit ``i`` inverted."""
    out = list(bits)
    out[i] = not out[i]
    return out


@pytest.mark.parametrize("text", ["", "A", "Hi!", "Hello friend.", "~"])
def test_hamming_round_trip(text: str) -> None:
    """A clean codeword decodes to the original payload, zero corrections."""
    ham = HammingCorrector()
    data = text_to_bits(text)
    payload, corrected = ham.decode(ham.encode(data))
    assert payload == data
    assert corrected is False


def test_hamming_grows_codeword() -> None:
    """Encoding interleaves parity bits, so the codeword is longer."""
    ham = HammingCorrector()
    data = text_to_bits("A")  # 8 data bits -> +4 parity bits
    assert len(ham.encode(data)) > len(data)


def test_hamming_corrects_every_single_flip() -> None:
    """Any single-bit flip (data or parity) is located, fixed, counted once."""
    ham = HammingCorrector()
    data = text_to_bits("Hi")
    code = ham.encode(data)
    for i in range(len(code)):
        payload, corrected = ham.decode(_flip(code, i))
        assert payload == data, f"flip at {i} was not corrected"
        assert corrected is True, f"flip at {i} should report a correction"


def test_hamming_corrected_is_bool_flag() -> None:
    """``corrected`` is a True/False flag, not a count or the error position."""
    ham = HammingCorrector()
    code = ham.encode(text_to_bits("Hi!"))
    _, corrected = ham.decode(_flip(code, len(code) - 1))
    assert corrected is True
