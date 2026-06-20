<!-- ./app/README.md -->
# Simulator GUI

Streamlit front end for the TR1 simulator. It runs the pipeline **in-process**
(importing `layer_manager` directly) so it can plot every stage live.

## Run

From the project root:

```bash
uv run streamlit run app/app.py
```

It serves on <http://localhost:8501> (the port is published by the `dev`
container in `docker/docker-compose.yml`).

## Current scope

Physical layer only, for now:

- Type a **message** and choose a **modulation** (NRZ-Polar, Manchester,
  Bipolar, ASK, FSK, QPSK, 16-QAM).
- Tune amplitude, samples per symbol, carrier frequency, sample rate, and the
  channel **noise sigma**.
- Watch the waveform scroll like an **oscilloscope** (Play to animate, or drag
  the position slider to scrub).
- See whether the receiver recovers the message after the noisy channel.

> Tip: keep *carrier cycles per symbol* a whole number (shown in the sidebar)
> so the carrier demodulators separate cleanly.

## Layout

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI: sidebar controls, the live oscilloscope, page layout. |
| `simulation.py` | **Streamlit-free** logic: build a modulator from the config, run text → bits → modulate → channel → demodulate. Unit-tested in `tests/test_app_simulation.py`. |
| `pages/` | Reserved for future multipage views (link layer, end-to-end over the nodes). |

The UI / logic split keeps `simulation.py` testable and reusable, and means the
heavy lifting lives in pure functions rather than in the Streamlit script.

## Planned next

- Stack the full pipeline per frame (bits → baseband → carrier → noisy →
  recovered) once the data-link layer lands.
- An "end-to-end over the nodes" panel showing the result after the real
  `tx → ch → rx` socket round-trip.
