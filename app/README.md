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

## What it shows

The **full link + physical pipeline** end to end, driven by the shared
`layer_manager.pipeline`:

- Configure every layer in the sidebar — **link** (max frame size, framing,
  detection, correction, checksum block), **physical** (baseband line code,
  carrier, amplitude, samples/symbol, frequency, sample rate), and the channel
  **noise sigma**.
- Watch the framed-bit reference, the modulated signal, and the received (noisy)
  signal scroll like an **oscilloscope**, full-width and stacked on a single
  shared time axis (gridlines on the symbol boundaries, a per-panel **volts
  axis**). The animation is a self-contained HTML canvas (`scope.html`) that
  runs **entirely in the browser** — no per-frame server round-trips — with
  **Play/Pause**, **Speed**, and **Window** controls.
- Inspect the **transmit stages** (chunking into frames, per-frame protected
  sizes, the framed bit stream) and the **receive verdict** (a per-frame table
  of detector ✓/✗ and bits corrected), plus whether the message round-tripped.

> Tips: keep *carrier cycles per symbol* a whole number (shown in the sidebar)
> so the carrier demodulators separate cleanly; pair parity/Hamming with
> bit-stuffing (char-count and byte-stuffing need byte-aligned frames).

## Layout

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI: full config sidebar, page layout, the canvas oscilloscope, and the transmit/receive panels. |
| `scope.html` | Self-contained HTML/canvas/JS oscilloscope. Streamlit injects the signals as JSON; all animation runs client-side. |
| `simulation.py` | **Streamlit-free** logic: `run_pipeline` wraps `layer_manager.pipeline` (transmit → channel → receive) and prepares the canvas traces. Unit-tested in `tests/test_app_simulation.py`. |
| `pages/` | Reserved for future multipage views. |

The UI / logic split keeps `simulation.py` testable and reusable, and means the
heavy lifting lives in pure functions rather than in the Streamlit script. The
actual layer orchestration lives in `layer_manager.pipeline` so the GUI and the
socket nodes share one implementation.
