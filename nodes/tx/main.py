# ./nodes/tx/main.py
"""Transmitter node: turn text into a clean signal and send it downstream.

Reads the message from ``argv[1]`` or ``$TR1_MESSAGE``, runs the transmit
pipeline (text -> bits -> link layer -> modulation), and ships the resulting
signal plus the :class:`SimulationConfig` the receiver needs to the ``ch`` node
over a socket. The host defaults to the ``ch_node`` container name and can be
overridden with ``$CH_HOST`` (e.g. ``127.0.0.1`` for a local run).
"""

import os
import sys

from layer_manager.config import SimulationConfig
from layer_manager.pipeline import transmit
from layer_manager.wire import connect, load_config, send_message


def main() -> None:
    """Build the signal for the message and send it to the channel node."""
    cfg = load_config()
    host = os.environ.get("CH_HOST", "ch_node")
    port = cfg.ports["ch"]

    message = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("TR1_MESSAGE", "Hello from TX over the wire!")
    )
    sim = SimulationConfig()
    noise = os.environ.get("TR1_NOISE_STD")
    if noise is not None:  # let the channel actually perturb the signal
        sim = sim.model_copy(update={"noise_std": float(noise)})

    result = transmit(sim, message)
    print(f"[tx] {message!r} -> {result.signal.size} samples", flush=True)

    with connect(host, port) as sock:
        send_message(
            sock,
            {
                "config": sim.model_dump(mode="json"),
                "signal": result.signal.tolist(),
            },
        )
    print(f"[tx] sent to {host}:{port}", flush=True)


if __name__ == "__main__":
    main()
