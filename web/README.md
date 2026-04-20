# gpu_ppm web visualizer

Interactive WebGPU visualization of pulsar pulse profiles from the OS1 physics
pipeline in `gpu/os1_gpu.cu`. A 3D sphere shows the hotspot configuration; a
light curve shows the observed flux vs. phase; dragging horizontally rotates the
pulsar and slides a dashed phase marker on the curve.

This is Phase 1 of the plan: scaffold only — WebGPU device acquisition and
lensing-table upload. No compute or render yet.

## Prerequisites

- Lensing tables generated (from the repo root):
  ```
  python3 c_lensing.py
  ```
- A browser with WebGPU enabled:
  - Chrome / Edge ≥ 113
  - Firefox Nightly with `dom.webgpu.enabled` in `about:config`

## Build artifact (not in git)

Pack the lensing tables into a single float32 blob:

```
python3 web/pack_lensing.py              # default: --setting std
python3 web/pack_lensing.py --setting high
```

Outputs `web/public/lensing.bin` and `web/public/lensing.json`.

## Run

Any static file server from the repo root works. The app loads
`web/public/lensing.*` via `fetch`, so file://-served pages will not work.

```
python3 -m http.server --directory web 8000
```

Then open http://localhost:8000/ and check the status panel: it should report
adapter info and the loaded table dimensions (`N_u=256  N_cos_psi=512  N_cos_alpha=512`).

## Layout

```
web/
  index.html          entry
  pack_lensing.py     build helper (→ public/lensing.{bin,json})
  public/             generated artifacts (gitignored)
  src/
    main.js           WebGPU device + table loader
    style.css
```
