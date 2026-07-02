# xas-helpers

These are a lightweight collection of Python CLI utilities for preprocessing XAS data. The current toolkit includes:

- Interpolation-free rebinning and averaging (`bin_average.py`).
- Batch processing (`batch_average.py`).
- Energy alignment (`shift_es.py`).

These scripts are designed to be light on dependencies for users that don't want to work with a full installation of `xraylarch`.

# Quickstart

1. Clone the repo.
2. (Optional) Create a Python venv: `uv venv`.
2. Install dependencies (into the new venv): `uv pip install -r requirements.txt`.

All of the utilities can be run by passing CLI arguments. For ease of reuse, YAML input files are also supported.

# Thanks

- [RSXAP](https://lise.lbl.gov/RSXAP/): Inspiration for the modular CLI + input file interface.
- [xraylarch](https://github.com/xraypy/xraylarch): Original implementation of the `E0` identification algorithm using adaptive Gaussian smoothing.