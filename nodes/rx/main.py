# ./nodes/rx/main.py
"""Receiver node: recover the original text from a noisy signal off the wire.

Listens for the noisy signal forwarded by ``ch``, runs the receive pipeline
(demodulate -> deframe -> error control -> bits -> text), and reports the
decoded message together with each frame's detector verdict and how many bits
the corrector fixed.
"""

import socket

import numpy as np

from layer_manager.config import SimulationConfig
from layer_manager.pipeline import receive
from layer_manager.wire import load_config, recv_message


def main() -> None:
    """Accept signals forever, decoding and reporting each one."""
    cfg = load_config()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((cfg.address, cfg.ports["rx"]))
    server.listen()
    print(f"[rx] listening on {cfg.address}:{cfg.ports['rx']}", flush=True)

    with server:
        while True:
            try:
                conn, _ = server.accept()
                with conn:
                    message = recv_message(conn, cfg.buffer_size)
                if message is None:
                    continue
                config = SimulationConfig.model_validate(message["config"])
                signal = np.asarray(message["signal"], dtype=np.float64)
                result = receive(config, signal)
                status = "ok" if result.ok else "ERRORS DETECTED"
                print(
                    f"[rx] recovered {result.text!r} "
                    f"[{status}, {result.corrected} frame(s) corrected]",
                    flush=True,
                )
            except Exception as error:  # keep the receiver alive on any failure
                print(f"[rx] dropped a message: {error!r}", flush=True)


if __name__ == "__main__":
    main()
