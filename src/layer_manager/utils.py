# ./src/layer_manager/utils.py
"""Application-layer codec: convert the input text to bits and back.

These are the "Codificador em bits" (transmitter) and "Conversor bits em
texto" (receiver) boxes of the assignment diagram. They sit at the top of the
stack: the GUI hands :func:`text_to_bits` the user's message, and
:func:`bits_to_text` turns the recovered bits back into readable text.
"""

import numpy as np

from layer_manager.types import Bits


def text_to_bits(text: str) -> Bits:
    """Convert text into a flat bit sequence, 8 bits per byte (MSB first).

    Args:
        text: The string to encode, using UTF-8.

    Returns:
        The flattened list of bits, ``len(text.encode()) * 8`` long.
    """
    return (
        np
        .unpackbits(np.frombuffer(text.encode(), dtype=np.uint8))
        .astype(dtype=bool)
        .tolist()
    )


def bits_to_text(bits: Bits) -> str:
    """Convert a bit sequence back into text, reading 8 bits per byte.

    Args:
        bits: A bit sequence whose length is a multiple of 8 (MSB first),
            as produced by :func:`text_to_bits`.

    Returns:
        The decoded UTF-8 string.
    """
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes().decode()
