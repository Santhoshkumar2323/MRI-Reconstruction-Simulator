# MRI Reconstruction Simulator

An interactive tool for exploring how MRI images are formed from raw k-space data, and what happens to the reconstruction when that data is filtered, split across multiple receiver coils, or undersampled to accelerate acquisition.

Built with Streamlit. Operates on real MRI slice data (`.npy` matrices) sourced from the TCIA public imaging archive.

---

## Architecture

![Architecture Diagram](architecture/how_it_works.svg)

The simulator's central idea: an MRI scanner does not acquire an image directly — it acquires **k-space**, the spatial-frequency representation of the object, and the image is recovered via an inverse 2D FFT. Every experiment in this tool manipulates k-space in a different, physically motivated way and shows the resulting effect on the reconstructed image.

---

## Experiments

### 1. K-Space Filters
Circular low-pass and high-pass masks applied around the center of k-space (`kspace_filters.py`).

- **Low-pass** retains only low spatial frequencies near the center → smoother, blurrier image, but contrast and coarse anatomy are preserved.
- **High-pass** removes the center, keeping only high spatial frequencies → an edge-map-like result.

This demonstrates that image *contrast* lives near the center of k-space, while fine *detail* lives in its periphery.

### 2. Multi-Coil Merge (RSS vs. Optimal Combine)
Simulates a phased-array receiver coil setup (`multi_coil.py`):

- Coil elements are placed evenly around a ring surrounding the anatomy.
- Each coil's **sensitivity map** is derived from a magnetic-dipole near-field expression (Biot-Savart–style falloff), not an arbitrary Gaussian approximation.
- A synthetic object phase can be applied, since real MRI images carry phase information from field inhomogeneity and other sources.
- Two combination methods are compared:
  - **Root-Sum-of-Squares (RSS)** — magnitude-only combination; simple, but discards phase entirely.
  - **Optimal (Roemer) Combine** — sensitivity-weighted complex combination that recovers true object phase. This is the same underlying principle used in SENSE/GRAPPA parallel-imaging reconstruction.

### 3. Undersampling (Aliasing)
Simulates accelerated k-space acquisition by keeping only every Nth line, plus a small fully-sampled center strip (an ACS/auto-calibration region), via `undersample.py`.

Reconstructing this directly (without an unfolding algorithm) produces classic **aliasing artifacts** — overlapping, ghosted copies of the anatomy. This illustrates why real accelerated scans require dedicated reconstruction techniques (SENSE, GRAPPA, etc.) rather than a plain inverse FFT.

### Shared: Thermal Noise Model
A single sidebar control (`noise.py`) injects complex Gaussian noise directly into k-space, split evenly across real and imaginary components. This mirrors how real thermal (Johnson–Nyquist) receiver noise behaves physically, rather than adding noise to the final image.

---

## Module Reference

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI — loads a scan, routes to the selected experiment, renders k-space and image panels |
| `fetch_samples.py` | One-time data preparation script; downloads real DICOM series from TCIA and converts slices to `.npy` |
| `noise.py` | Adds complex Gaussian noise to k-space |
| `kspace_filters.py` | Low-pass / high-pass circular masking of k-space |
| `multi_coil.py` | Coil geometry, physics-based sensitivity maps, coil simulation, RSS and Roemer combine |
| `undersample.py` | Line-skipping undersampling mask with preserved center (ACS) region |

---

## Data Flow Summary

```
image_data  →  fft2()  →  k-space  →  + noise
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
            k-space filters        multi-coil split      undersampling mask
            (low/high pass)      (sensitivity maps)      (line skip + ACS)
                    │                     │                     │
                    ▼                     ▼                     ▼
                ifft2()          RSS / Roemer combine        ifft2()
                    │                     │                     │
                    └─────────────────────┴─────────────────────┘
                                          ▼
                          rendered in Streamlit (matplotlib)
```