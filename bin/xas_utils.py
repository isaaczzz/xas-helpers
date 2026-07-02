#!/usr/bin/env python3
"""
xas_utils.py — shared utilities for XAS processing scripts.

Provides constants, YAML helpers, file utilities, spectrum loaders,
and E0 estimation used by bin_average.py, shift_es.py, and batch_average.py.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -------------------------
# Constants
# -------------------------
HC_EV_ANG = 12398.419843320025       # eV*Å
K_CONV_EV_PER_A2 = 3.8099819442818976  # E-E0 = K_CONV * k^2


# -------------------------
# YAML helpers
# -------------------------
safe_load = None   # type: ignore
safe_dump = None   # type: ignore


def _ensure_yaml() -> None:
    """Lazy-load PyYAML safe functions; exit with a helpful message if missing."""
    global safe_load, safe_dump
    if safe_load is not None and safe_dump is not None:
        return
    try:
        from yaml import safe_load as sl, safe_dump as sd
        safe_load = sl
        safe_dump = sd
    except ImportError:
        print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)


# -------------------------
# Dict utilities
# -------------------------
def deep_update(base: dict, override: dict) -> dict:
    """Return deep-merged dict(base <- override) without mutating either input."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path | None) -> dict:
    """Load a JSON or YAML config file; return empty dict if path is None."""
    if config_path is None:
        return {}
    suffix = config_path.suffix.lower()
    txt = config_path.read_text()

    if suffix == ".json":
        return json.loads(txt)
    elif suffix in (".yaml", ".yml"):
        _ensure_yaml()
        return safe_load(txt) or {}
    else:
        raise ValueError(
            f"Unsupported config extension: {suffix}. Use .json/.yaml/.yml"
        )


# -------------------------
# File utilities
# -------------------------
def _ensure_parent(path: str | Path | None) -> None:
    """Create parent directory of path if it has a directory component."""
    if not path or ("/" not in str(path) and "\\" not in str(path)):
        return
    p = Path(str(path))
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)


def _validate_files_list(files: Any, where: str) -> list[str]:
    """Validate and normalize a files list from YAML."""
    if not isinstance(files, list) or len(files) == 0:
        raise ValueError(f"{where}: 'files' must be a non-empty list")
    out: list[str] = []
    for i, x in enumerate(files):
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f"{where}: files[{i}] must be a non-empty string")
        out.append(x.strip())
    return out


def _apply_prefix_suffix(tokens: list[str], prefix: str, suffix: str,
                          where: str) -> list[str]:
    """Apply file prefix/suffix transformation with light sanity warnings."""
    out = []
    for t in tokens:
        if suffix and t.endswith(suffix):
            print(
                f"Warning: {where}: token '{t}' already ends with suffix '{suffix}'. "
                "Result may duplicate suffix.",
                file=sys.stderr,
            )
        out.append(f"{prefix}{t}{suffix}")
    return out


def _expand_files(tokens: list[str], dir_path: Path | None) -> list[Path]:
    """Resolve a list of file tokens (paths or globs) relative to dir_path."""
    resolved: list[Path] = []
    for t in tokens:
        p = Path(t)
        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            resolved.append(p)
            continue

        base = dir_path if dir_path else Path(".")
        if "*" in t or "?" in t:
            matches = sorted(base.glob(t))
            if not matches:
                raise FileNotFoundError(
                    f"No files matched pattern '{t}' under {base}"
                )
            resolved.extend(matches)
        else:
            full = base / t
            if not full.exists():
                raise FileNotFoundError(f"File not found: {full}")
            resolved.append(full)

    return resolved


# -------------------------
# Spectrum loading
# -------------------------
def parse_spring8_file(filepath: Path, mode: str = "transmission",
                       angle_col: str = "Angle(o)", time_col: str = "time/s",
                       i0_col: str = "2", i1_col: str = "3",
                       fluo_cols: list[str] | None = None) -> dict:
    """Parse an SPring-8/Aichi-style raw scan file.

    Returns a dict with keys:
      path, d_spacing, theta_deg, energy_eV, time_s, i0, signal, mode, mu, df
    """
    text = filepath.read_text(errors="ignore").splitlines()

    d_spacing = np.nan
    for line in text[:120]:
        m = re.search(r"D=\s*([0-9.]+)\s*A", line)
        if m:
            d_spacing = float(m.group(1))
            break
    if np.isnan(d_spacing):
        raise ValueError(f"Could not find D=...A in header: {filepath}")

    header_idx = None
    col_names = None
    for i, line in enumerate(text):
        if "Angle(o)" in line and "time/s" in line:
            header_idx = i
            col_names = line.split()
            break
    if header_idx is None:
        raise ValueError(
            f"Could not find data header with Angle(o), time/s in {filepath}"
        )

    data_lines = []
    for line in text[header_idx + 1:]:
        toks = line.split()
        if len(toks) < len(col_names):
            continue
        ok = True
        for t in toks[:len(col_names)]:
            try:
                float(t)
            except Exception:
                ok = False
                break
        if ok:
            data_lines.append(toks[:len(col_names)])

    if not data_lines:
        raise ValueError(f"No numeric rows found in {filepath}")

    arr = np.array(data_lines, dtype=float)
    df = pd.DataFrame(arr, columns=col_names)

    for req in [angle_col, time_col, i0_col]:
        if req not in df.columns:
            raise ValueError(
                f"{req} not in columns for {filepath}: {list(df.columns)}"
            )

    theta_deg = df[angle_col].to_numpy()
    time_s = df[time_col].to_numpy()
    i0 = df[i0_col].to_numpy()

    eps = 1e-12
    if mode == "transmission":
        if i1_col not in df.columns:
            raise ValueError(f"{i1_col} not found in {filepath}")
        i1 = df[i1_col].to_numpy()
        signal = i1
        mu = np.log((i0 + eps) / (i1 + eps))
    else:
        if not fluo_cols:
            reserved = {angle_col, "Angle(c)", time_col, i0_col}
            fluo_cols = [c for c in df.columns if c not in reserved]
        for c in fluo_cols:
            if c not in df.columns:
                raise ValueError(f"Fluo column {c} missing in {filepath}")
        fluo = df[fluo_cols].sum(axis=1).to_numpy()
        signal = fluo
        mu = fluo / (i0 + eps)

    theta_rad = np.deg2rad(theta_deg)
    energy = HC_EV_ANG / (2.0 * d_spacing * np.sin(theta_rad))

    order = np.argsort(energy)
    return {
        "path": filepath,
        "d_spacing": d_spacing,
        "theta_deg": theta_deg[order],
        "energy_eV": energy[order],
        "time_s": time_s[order],
        "i0": i0[order],
        "signal": signal[order],
        "mode": mode,
        "mu": mu[order],
        "df": df.iloc[order].reset_index(drop=True),
    }


def parse_simple_spectrum(filepath: Path) -> dict:
    """Load a simple TSV/CSV spectrum such as bin_average output.

    Recognized energy columns: E_eV, Energy(eV), etc.
    Recognized mu columns: mu_raw, mu, absorbance.
    """
    df = None
    for sep in ("\t", ","):
        try:
            candidate = pd.read_csv(filepath, sep=sep, comment="#")
            if len(candidate.columns) >= 2:
                df = candidate
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Could not parse {filepath} as TSV or CSV")

    cols_lower = {c.strip().lower(): c for c in df.columns}

    energy_col = None
    for cand in ["e_ev", "energy(ev)", "e (ev)", "energy_ev"]:
        if cand in cols_lower:
            energy_col = cols_lower[cand]
            break

    mu_col = None
    for cand in ["mu_raw", "mu", "absorbance"]:
        if cand in cols_lower:
            mu_col = cols_lower[cand]
            break

    if energy_col is None or mu_col is None:
        raise ValueError(
            f"Could not find suitable energy/mu columns in {filepath}. "
            f"Columns: {list(df.columns)}"
        )

    order = np.argsort(df[energy_col].to_numpy())
    return {
        "path": filepath,
        "energy_eV": df[energy_col].iloc[order].to_numpy(),
        "mu": df[mu_col].iloc[order].to_numpy(),
        "df": df.iloc[order].reset_index(drop=True),
    }


def load_spectrum(filepath: Path) -> dict:
    """Try SPring-8 parser first, fall back to simple TSV."""
    try:
        return parse_spring8_file(filepath)
    except Exception:
        return parse_simple_spectrum(filepath)


# -------------------------
# E0 estimation
# -------------------------
def gaussian_smooth(y: np.ndarray, x: np.ndarray, sigma_x: float) -> np.ndarray:
    """Gaussian-kernel weighted average along x axis."""
    n = len(x)
    if n < 3 or sigma_x <= 0:
        return np.asarray(y, dtype=float)

    h = int(4 * sigma_x / np.median(np.diff(x)) + 0.5)
    h = max(h, 1)

    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(i - h, 0)
        hi = min(i + h + 1, n)
        dx = x[lo:hi] - x[i]
        w = np.exp(-0.5 * (dx / sigma_x) ** 2)
        out[i] = np.dot(w, y[lo:hi]) / w.sum()

    return out


def estimate_e0_coarse(energy: np.ndarray, mu: np.ndarray,
                       search_min: float | None = None,
                       search_max: float | None = None) -> float:
    """Locate E0 as peak of Gaussian-smoothed dmu/dE, excluding scan edges."""
    n = len(energy)

    if n < 50:
        lo, hi = int(n * 0.1), int(n * 0.9)
        dmu = np.gradient(mu[lo:hi], np.asarray(energy[lo:hi]))
        return float(energy[lo + int(np.argmax(dmu))])

    e_min, e_max = float(energy[0]), float(energy[-1])
    span = e_max - e_min
    sigma_x = max(span * 0.005, 0.3)

    mu_s = gaussian_smooth(mu, energy, sigma_x=sigma_x)
    dmu = np.gradient(np.asarray(mu_s), np.asarray(energy))

    if search_min is None:
        search_min = e_min + 0.15 * span
    if search_max is None:
        search_max = e_max - 0.15 * span

    mask = (energy >= search_min) & (energy <= search_max)
    if not np.any(mask):
        lo, hi = int(n * 0.1), int(n * 0.9)
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True

    pos_idx = int(np.where(mask)[0][int(np.argmax(dmu[mask]))])
    return float(energy[pos_idx])


def refine_e0_local(energy: np.ndarray, mu: np.ndarray, E0_coarse: float,
                    fit_degree: int = 4,
                    max_shift: float = 3.0,
                    min_R2: float = 0.95) -> tuple[float, bool]:
    """Refine E0 via local polynomial fit and 2nd-derivative zeros."""
    w_lo, w_hi = E0_coarse - 7.0, E0_coarse + 8.0
    mask = (energy >= w_lo) & (energy <= w_hi)
    if np.sum(mask) < fit_degree + 3:
        return float(E0_coarse), False

    x, y = energy[mask], mu[mask]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            coeffs = np.polyfit(x, y, deg=fit_degree)
    except Exception:
        return float(E0_coarse), False

    y_fit = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0 or (1.0 - ss_res / ss_tot) < min_R2:
        return float(E0_coarse), False

    p1 = np.polyder(coeffs, 1)
    p2 = np.polyder(p1, 1)
    roots = [float(np.real(r)) for r in np.roots(p2)
             if not np.iscomplex(r) and w_lo <= np.real(r) <= w_hi]
    if not roots:
        return float(E0_coarse), False

    best_r = max(roots, key=lambda r: float(np.polyval(p1, r)))
    if abs(best_r - E0_coarse) > max_shift:
        return float(E0_coarse), False

    return float(best_r), True


def estimate_e0(energy: np.ndarray, mu: np.ndarray,
                search_min: float | None = None,
                search_max: float | None = None) -> float:
    """Return the best E0 estimate: Gaussian coarse search + local polynomial refinement."""
    E0_coarse = estimate_e0_coarse(energy, mu, search_min=search_min, search_max=search_max)
    E0_refined, ok = refine_e0_local(energy, mu, E0_coarse)
    return E0_refined if ok else E0_coarse


# -------------------------
# Energy alignment
# -------------------------
def align_scan_energy(scan: Any, e0_target: float) -> None:
    """Shift scan.energy_eV so scan.e0 matches e0_target (mutates in place)."""
    delta = e0_target - scan.e0
    scan.energy_eV = scan.energy_eV + delta
    scan.e0 = e0_target
