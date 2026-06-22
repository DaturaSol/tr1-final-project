# ./nodes/ch/main.py
"""Channel node -- the communication medium.

Listens for a clean signal from ``tx``, adds Gaussian noise ``n(mean, sigma)``
to its V/W samples (the one thing a medium does), and forwards the noisy signal
to ``rx``. The noise parameters come from the :class:`SimulationConfig` carried
in the message, so the whole run stays driven by one config. The downstream
host defaults to the ``rx_node`` container name (override with ``$RX_HOST``).
"""

import os
import socket

import numpy as np

from layer_manager.config import SimulationConfig
from layer_manager.factory import build_channel
from layer_manager.wire import connect, load_config, recv_message, send_message


def main() -> None:
    """Relay each received signal to the receiver after adding noise."""
    cfg = load_config()
    rx_host = os.environ.get("RX_HOST", "rx_node")
    rx_port = cfg.ports["rx"]

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((cfg.address, cfg.ports["ch"]))
    server.listen()
    print(f"[ch] listening on {cfg.address}:{cfg.ports['ch']}", flush=True)

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
                noisy = build_channel(config).transmit(signal)
                print(
                    f"[ch] {signal.size} samples, sigma={config.noise_std}",
                    flush=True,
                )
                with connect(rx_host, rx_port) as out:
                    send_message(
                        out,
                        {"config": message["config"], "signal": noisy.tolist()},
                    )
                print(f"[ch] forwarded to {rx_host}:{rx_port}", flush=True)
            except Exception as error:  # keep the channel alive on any failure
                print(f"[ch] dropped a message: {error!r}", flush=True)


if __name__ == "__main__":
    main()
