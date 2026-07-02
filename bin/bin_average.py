#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xas_bin_average.py

XAS scan QC, E0 alignment, variable-grid binning, averaging, uncertainty estimation.
Supports SPring-8/Aichi/SAGA-style text files.

Features
--------
- JSON/YAML config support
- default grid options when none supplied
- optional plotting

"""

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# -------------------------
# Constants / conversions
# -------------------------
HC_EV_ANG = 12398.419843320025  # eV*Å
K_CONV_EV_PER_A2 = 3.8099819442818976  # E-E0 = K_CONV * k^2


@dataclass
class Scan:
    path: Path
    d_spacing: float
    theta_deg: np.ndarray
    energy_eV: np.ndarray
    time_s: np.ndarray
    i0: np.ndarray
    signal: np.ndarray
    mode: str
    mu: np.ndarray
    e0: float = np.nan
    flags: list = field(default_factory=list)


DEFAULTS = {
    "mode": "transmission",
    "i0_col": "2",
    "i1_col": "3",
    "fluo_cols": [],

    "align": "mean",        # none | mean | ref | value
    "ref_index": 0,
    "e0_value": None,

    "qc": {
        "shutter_threshold_frac": 0.05,
        "min_range_frac": 0.95,
        "median_i0_z": 4.0,
    },

    "grid": {
        # relative to E0
        "pre_start": -200.0,
        "pre_end": -30.0,
        "de_pre": 2.0,
        "xanes_end": 50.0,
        "de_xanes": 0.2,
        "kmax": 14.0,
        "dk": 0.05,
    },

    "smooth_uncertainty": True,

    "plot": {
        "enabled": False,
        "show": True,
        "save_prefix": None,   # if set, save PNGs
    },

    "output": "averaged_xas.dat"
}


# -------------------------
# Config helpers
# -------------------------
def deep_update(base: dict, upd: dict):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: Path):
    if config_path is None:
        return {}
    suffix = config_path.suffix.lower()
    txt = config_path.read_text()

    if suffix == ".json":
        return json.loads(txt)
    elif suffix in [".yaml", ".yml"]:
        if not HAS_YAML:
            raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
        return yaml.safe_load(txt)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}. Use .json/.yaml/.yml")


def build_settings(args):
    settings = json.loads(json.dumps(DEFAULTS))  # deep copy via JSON-serializable dict
    cfg = load_config(Path(args.config)) if args.config else {}
    if cfg:
        deep_update(settings, cfg)

    # CLI overrides config
    if args.mode:
        settings["mode"] = args.mode
    if args.i0_col:
        settings["i0_col"] = args.i0_col
    if args.i1_col:
        settings["i1_col"] = args.i1_col
    if args.fluo_cols is not None:
        settings["fluo_cols"] = [x.strip() for x in args.fluo_cols.split(",") if x.strip()]

    if args.align:
        settings["align"] = args.align
    if args.ref_index is not None:
        settings["ref_index"] = args.ref_index
    if args.e0_value is not None:
        settings["e0_value"] = args.e0_value

    # grid overrides
    g = settings["grid"]
    for key in ["pre_start", "pre_end", "de_pre", "xanes_end", "de_xanes", "kmax", "dk"]:
        v = getattr(args, key)
        if v is not None:
            g[key] = v

    if args.output:
        settings["output"] = args.output

    if args.plot:
        settings["plot"]["enabled"] = True
    if args.no_show:
        settings["plot"]["show"] = False
    if args.plot_prefix:
        settings["plot"]["save_prefix"] = args.plot_prefix

    return settings


# -------------------------
# Parsing
# -------------------------
def parse_spring8_file(filepath: Path, mode: str, angle_col="Angle(o)", time_col="time/s",
                       i0_col="2", i1_col="3", fluo_cols=None) -> Scan:
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
        raise ValueError(f"Could not find data header with Angle(o), time/s in {filepath}")

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
            raise ValueError(f"{req} not in columns for {filepath}: {list(df.columns)}")

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
    return Scan(
        path=filepath,
        d_spacing=d_spacing,
        theta_deg=theta_deg[order],
        energy_eV=energy[order],
        time_s=time_s[order],
        i0=i0[order],
        signal=signal[order],
        mode=mode,
        mu=mu[order],
    )


# -------------------------
# QC + E0
# -------------------------
def estimate_e0(energy, mu, window=31, poly=3, exclude=(0.05, 0.05)):
    """Estimate E0 as energy at maximum d(mu)/dE, excluding scan edges."""
    n = len(mu)
    if n < 10:
        dmu = np.gradient(mu, energy)
        return energy[np.argmax(dmu)]

    lo_frac, hi_frac = exclude
    lo = int(n * lo_frac)
    hi = int(n * (1.0 - hi_frac))

    w = min(window, n - 2)
    if w < 7:
        w = 7
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - (n % 2)
    if w < 5:
        dmu = np.gradient(mu, energy)
        idx = np.argmax(dmu[lo:hi]) + lo
        return energy[idx]

    mu_s = savgol_filter(mu, int(w), poly, mode="interp")
    dmu = np.gradient(mu_s, energy)

    idx = np.argmax(dmu[lo:hi]) + lo
    return energy[idx]


def qc_scans(scans, shutter_threshold_frac, min_range_frac, median_i0_z):
    ranges = np.array([s.energy_eV.max() - s.energy_eV.min() for s in scans])
    med_range = np.median(ranges)

    med_i0s = np.array([np.median(s.i0) for s in scans])
    med_i0 = np.median(med_i0s)
    mad = np.median(np.abs(med_i0s - med_i0)) + 1e-12

    med_signals = np.array([np.median(s.signal) for s in scans])
    global_med_signal = np.median(med_signals)

    for i, s in enumerate(scans):
        if ranges[i] < min_range_frac * med_range:
            s.flags.append("TRUNCATED_RANGE")
        if med_signals[i] < shutter_threshold_frac * global_med_signal:
            s.flags.append("LOW_SIGNAL_SHUTTER_OR_GAIN")
        z = np.abs(med_i0s[i] - med_i0) / (1.4826 * mad)
        if z > median_i0_z:
            s.flags.append("I0_GAIN_OUTLIER")


# -------------------------
# Grid construction
# -------------------------
def _arange_edges_inclusive(start: float, end: float, step: float, eps: float = 1e-9) -> np.ndarray:
    """Create monotonic edges from start to end with fixed step, forcing inclusion of end."""
    if step <= 0:
        raise ValueError("step must be > 0")
    if end < start:
        raise ValueError("end must be >= start")

    n = int(np.floor((end - start) / step + 1e-12)) + 1
    edges = start + step * np.arange(n, dtype=float)

    if edges[-1] < end - eps:
        edges = np.append(edges, end)
    else:
        edges[-1] = end
    return edges


def make_piecewise_edges(e0: float, g: dict) -> np.ndarray:
    """Build pre-edge/XANES/EXAFS edge grid anchored at e0."""
    e_pre0 = e0 + g["pre_start"]
    e_pre1 = e0 + g["pre_end"]
    e_x1 = e0 + g["xanes_end"]
    e_ex1 = e0 + K_CONV_EV_PER_A2 * g["kmax"] ** 2

    edges_pre = _arange_edges_inclusive(e_pre0, e_pre1, g["de_pre"])
    edges_x = _arange_edges_inclusive(e_pre1, e_x1, g["de_xanes"])

    k0 = np.sqrt(max(e_x1 - e0, 0.0) / K_CONV_EV_PER_A2)
    ks = _arange_edges_inclusive(float(k0), float(g["kmax"]), float(g["dk"]))
    e_k = e0 + K_CONV_EV_PER_A2 * ks**2
    e_k[-1] = e_ex1

    edges = np.unique(np.concatenate([edges_pre, edges_x, e_k]))
    edges.sort()
    return edges


# -------------------------
# Binning
# -------------------------
def bin_scan_to_edges(energy, mu, edges):
    """
    Bin raw (E, μ) data onto a shared edge grid.

    Returns
    -------
    b_e : mean E per bin (NaN if empty)
    b_mu : mean μ per bin (NaN if empty)
    b_sigma : standard error per bin (NaN if <2 points)
    n_pts : number of points per bin
    """
    energy = np.asarray(energy, dtype=float)
    mu = np.asarray(mu, dtype=float)
    edges = np.asarray(edges, dtype=float)

    nb = len(edges) - 1
    idx = np.digitize(energy, edges) - 1

    valid = (idx >= 0) & (idx < nb) & np.isfinite(energy) & np.isfinite(mu)
    if not np.any(valid):
        return (
            np.full(nb, np.nan),
            np.full(nb, np.nan),
            np.full(nb, np.nan),
            np.zeros(nb, dtype=int),
        )

    idxv = idx[valid]
    ev = energy[valid]
    muv = mu[valid]

    n_pts = np.bincount(idxv, minlength=nb).astype(int)
    sum_e = np.bincount(idxv, weights=ev, minlength=nb)
    sum_mu = np.bincount(idxv, weights=muv, minlength=nb)
    sumsq_mu = np.bincount(idxv, weights=muv**2, minlength=nb)

    b_e = np.full(nb, np.nan)
    b_mu = np.full(nb, np.nan)
    b_sigma = np.full(nb, np.nan)

    nonempty = n_pts > 0
    b_e[nonempty] = sum_e[nonempty] / n_pts[nonempty]
    b_mu[nonempty] = sum_mu[nonempty] / n_pts[nonempty]

    good_var = n_pts >= 2
    if np.any(good_var):
        n = n_pts[good_var].astype(float)
        ss = sumsq_mu[good_var]
        s = sum_mu[good_var]
        var = (ss - (s**2) / n) / (n - 1.0)
        var = np.maximum(var, 0.0)
        b_sigma[good_var] = np.sqrt(var / n)

    return b_e, b_mu, b_sigma, n_pts


def smooth_sigma(sig, win=11):
    """Nearest-fill invalid entries, then median-filter for robust uncertainty smoothing."""
    x = np.asarray(sig, dtype=float).copy()
    good = np.isfinite(x)
    if np.sum(good) < 2:
        return x

    miss = ~good
    if np.any(miss):
        xi = np.arange(len(x))
        good_idx = xi[good]
        miss_idx = xi[miss]

        pos = np.searchsorted(good_idx, miss_idx)
        left_i = np.clip(pos - 1, 0, len(good_idx) - 1)
        right_i = np.clip(pos, 0, len(good_idx) - 1)

        left_idx = good_idx[left_i]
        right_idx = good_idx[right_i]

        left_dist = np.abs(miss_idx - left_idx)
        right_dist = np.abs(miss_idx - right_idx)

        nearest_idx = left_idx.copy()
        choose_right = right_dist < left_dist
        nearest_idx[choose_right] = right_idx[choose_right]

        x[miss] = x[nearest_idx]

    if win % 2 == 0:
        win += 1
    win = min(win, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if win < 3:
        return x

    return median_filter(x, size=win, mode="nearest")


def average_binned(scans, edges, smooth_unc=True):
    """Rebin each scan, then combine bins with uncertainty-aware weighting."""
    all_e = []
    all_mu = []
    all_sigma = []

    for s in scans:
        be, bm, bs, _ = bin_scan_to_edges(s.energy_eV, s.mu, edges)
        all_e.append(be)
        all_mu.append(bm)
        all_sigma.append(bs)

    all_e = np.asarray(all_e)
    all_mu = np.asarray(all_mu)
    all_sigma = np.asarray(all_sigma)

    valid_mu = np.isfinite(all_mu)
    n_eff = valid_mu.sum(axis=0)

    known_sigma = np.isfinite(all_sigma) & valid_mu
    n_has_sig = known_sigma.sum(axis=0)
    bins_has = n_has_sig > 0

    nb = all_mu.shape[1]

    # largest known sigma in each bin
    sig_masked = np.where(known_sigma, all_sigma, np.nan)
    max_sig = np.nanmax(sig_masked, axis=0)

    # weights
    w = np.zeros_like(all_mu, dtype=float)

    # known sigma => inverse variance
    w += np.where(known_sigma, 1.0 / (all_sigma**2), 0.0)

    # missing sigma where some sigma exists => conservative fallback
    inv_missing = np.zeros(nb, dtype=float)
    inv_missing[bins_has] = 1.0 / (max_sig[bins_has] ** 2)
    missing_sigma_scans = valid_mu & ~known_sigma
    w += missing_sigma_scans * inv_missing[None, :]

    # no sigma in bin => equal weights among valid scans
    w += valid_mu * (~bins_has)[None, :]

    w_sum = w.sum(axis=0)

    mean_mu = np.sum(w * all_mu, axis=0) / w_sum
    mean_e = np.sum(w * all_e, axis=0) / w_sum

    resid = all_mu - mean_mu[None, :]
    chi2 = np.sum(w * resid**2, axis=0)

    sigma_scatter = np.full(nb, np.nan)
    mask_nv = n_eff > 1
    sigma_scatter[mask_nv] = np.sqrt(
        chi2[mask_nv] / ((n_eff[mask_nv] - 1.0) * w_sum[mask_nv])
    )

    sigma_prop = np.zeros(nb, dtype=float)
    sigma_prop[bins_has] = 1.0 / np.sqrt(w_sum[bins_has])

    sigma_mean = np.sqrt(sigma_prop**2 + sigma_scatter**2)

    keep = n_eff >= 2
    mean_e = mean_e[keep]
    mean_mu = mean_mu[keep]
    sigma_mean = sigma_mean[keep]
    n_eff_keep = n_eff[keep]
    all_mu_keep = all_mu[:, keep]

    if smooth_unc:
        sigma_mean = smooth_sigma(sigma_mean, win=11)

    return mean_e, mean_mu, sigma_mean, n_eff_keep, all_mu_keep


# -------------------------
# Diagnostics + plotting
# -------------------------
def report_raw_deltaE(scans):
    print("\n=== Raw delta-E diagnostic ===")
    for s in scans:
        dE = np.abs(np.diff(s.energy_eV))
        i = np.argmax(dE)
        e_mid = 0.5 * (s.energy_eV[i] + s.energy_eV[i + 1])
        print(f"{s.path.name}: max dE={dE[i]:.6f} eV at E~{e_mid:.3f} eV")


def plot_qc(scans, used_scans, save_prefix=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in scans:
        c = "tab:blue" if s in used_scans else "tab:red"
        ax.plot(s.energy_eV, s.mu, color=c, alpha=0.25, lw=0.8)
        ax.axvline(s.e0, color=c, alpha=0.08)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Raw absorbance (mu)")
    ax.set_title("QC overlay: scans and E0 markers (blue=used, red=excluded)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_qc_overlay.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    e0s = np.array([s.e0 for s in scans])
    ax2.hist(e0s, bins=max(8, min(30, len(scans) // 2 + 3)), color="gray", edgecolor="black")
    ax2.set_xlabel("E0 (eV)")
    ax2.set_ylabel("Count")
    ax2.set_title("E0 distribution")
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    if save_prefix:
        fig2.savefig(f"{save_prefix}_e0_hist.png", dpi=150)

    return [fig, fig2]


def plot_average(E, MU, SIG, all_mu=None, save_prefix=None, regions=None, coarsened_intervals=None):
    """Plot averaged XAS with optional rebinned scan overlay and coarsening indicators."""
    fig, ax = plt.subplots(figsize=(10, 5))

    region_colors = {
        "Pre-edge": "#FFDDC1",
        "XANES": "#C8E6FF",
        "EXAFS": "#D4FFD7",
    }
    if isinstance(regions, dict):
        for label, (lo, hi) in regions.items():
            ax.axvspan(lo, hi, facecolor=region_colors.get(label, "#EEEEEE"),
                       alpha=0.35, zorder=0, label=label)

    n_scans = None
    if all_mu is not None and all_mu.ndim == 2:
        n_scans = all_mu.shape[0]
        for i in range(n_scans):
            valid = np.isfinite(all_mu[i])
            if np.sum(valid) > 2:
                ax.plot(E[valid], all_mu[i, valid], color="tab:gray", alpha=0.35, lw=0.8, zorder=1)

    label = f"Averaged μ ({n_scans} scans)" if n_scans else "Averaged μ"
    ax.scatter(E, MU, color="black", s=8, edgecolor="none", alpha=0.9, zorder=2, label=label)
    ax.plot(E, MU, color="tab:blue", lw=1.2, alpha=0.7, zorder=2)
    ax.fill_between(E, MU - SIG, MU + SIG, color="tab:blue", alpha=0.25, label="±σ(mean)")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Raw absorbance (mu)")
    ax.set_title(f"Binned/averaged XAS spectrum ({len(E)} points)")
    ax.grid(alpha=0.2)
    ax.legend()

    if isinstance(coarsened_intervals, list) and coarsened_intervals:
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        y_base = ymin - 0.03 * y_range
        bar_height = 0.015 * y_range

        for (lo, hi) in coarsened_intervals:
            rect = plt.Rectangle((lo, y_base), hi - lo, bar_height, color="tab:red", alpha=0.9, zorder=10)
            ax.add_patch(rect)

        ax.annotate("Auto-coarsened region(s)",
                    xy=(coarsened_intervals[0][0], y_base + bar_height),
                    fontsize=7, color="tab:red", ha="left", va="bottom")
        ax.set_ylim(bottom=ymin - 0.05 * y_range)

    fig.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_average.png", dpi=150)
    return fig


def align_scan_energy(scan, e0_target):
    """Shift energy axis so scan.e0 matches e0_target."""
    delta = e0_target - scan.e0
    scan.energy_eV = scan.energy_eV + delta
    scan.e0 = e0_target


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="XAS bin/average tool with config and plotting.")
    ap.add_argument("files", nargs="+", help="Input scan files (base names, paths, or globs)")
    ap.add_argument("-d", "--dir", type=str, default=None,
                    help="Base directory for relative files/globs")
    ap.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config")

    # optional CLI overrides
    ap.add_argument("--mode", choices=["transmission", "fluorescence"], default=None)
    ap.add_argument("--i0-col", default=None)
    ap.add_argument("--i1-col", default=None)
    ap.add_argument("--fluo-cols", default=None)

    ap.add_argument("--align", choices=["none", "mean", "ref", "value"], default=None)
    ap.add_argument("--ref-index", type=int, default=None)
    ap.add_argument("--e0-value", type=float, default=None)

    # grid overrides
    ap.add_argument("--pre-start", type=float, default=None)
    ap.add_argument("--pre-end", type=float, default=None)
    ap.add_argument("--de-pre", type=float, default=None)
    ap.add_argument("--xanes-end", type=float, default=None)
    ap.add_argument("--de-xanes", type=float, default=None)
    ap.add_argument("--kmax", type=float, default=None)
    ap.add_argument("--dk", type=float, default=None)

    # region-specific coarsening toggles
    ap.add_argument("--no-coarse-pre-edge", action="store_true", help="Disable pre-edge auto-coarsening")
    ap.add_argument("--no-coarse-xanes", action="store_true", help="Disable XANES auto-coarsening")
    ap.add_argument("--no-coarse-exafs", action="store_true", help="Disable EXAFS auto-coarsening")

    ap.add_argument("--rebin-scans", action="store_true",
                    help="Rebin each scan individually before averaging")
    ap.add_argument("--shift-minima", action="store_true",
                    help="Shift each scan μ so low-end baseline minimum is zero")
    ap.add_argument("--output", default=None)

    # plotting
    ap.add_argument("--plot", action="store_true", help="Enable plotting")
    ap.add_argument("--no-show", action="store_true", help="Do not show interactive windows")
    ap.add_argument("--plot-prefix", default=None, help="Prefix to save plot PNGs")

    args = ap.parse_args()
    settings = build_settings(args)

    # Expand inputs
    base_dir = Path(args.dir) if args.dir else None
    expanded_files = []
    for token in args.files:
        tpath = Path(token)
        if any(c in token for c in ("*", "?")) and not tpath.is_absolute():
            root = base_dir or Path(".")
            matches = sorted(root.glob(token))
            if not matches:
                raise FileNotFoundError(
                    f"No files matched pattern '{token}' in '{root}'. "
                    "Check --dir and working directory."
                )
            expanded_files.extend(matches)
        else:
            candidate = (base_dir / token) if (base_dir is not None and not tpath.is_absolute()) else tpath
            expanded_files.append(candidate)

    files = expanded_files
    scans = []
    for f in files:
        s = parse_spring8_file(
            f,
            mode=settings["mode"],
            i0_col=settings["i0_col"],
            i1_col=settings["i1_col"],
            fluo_cols=settings["fluo_cols"]
        )
        s.e0 = estimate_e0(s.energy_eV, s.mu)
        scans.append(s)

    # QC
    qc = settings["qc"]
    qc_scans(scans, qc["shutter_threshold_frac"], qc["min_range_frac"], qc["median_i0_z"])

    print("\n=== Scan QC summary ===")
    for i, s in enumerate(scans):
        flag = ",".join(s.flags) if s.flags else "OK"
        print(f"[{i}] {s.path.name}  E0={s.e0:.3f} eV  FLAGS={flag}")

    report_raw_deltaE(scans)

    # filter
    use = [s for s in scans if ("TRUNCATED_RANGE" not in s.flags and "LOW_SIGNAL_SHUTTER_OR_GAIN" not in s.flags)]
    if len(use) < 2:
        raise RuntimeError("Not enough usable scans after QC filtering.")

    # alignment target
    align_mode = settings["align"]
    if align_mode in ("none", "mean"):
        e0_align = float(np.mean([s.e0 for s in use]))
    elif align_mode == "ref":
        e0_align = scans[settings["ref_index"]].e0
    else:
        if settings["e0_value"] is None:
            raise ValueError("align='value' but e0_value is not set.")
        e0_align = settings["e0_value"]

    if align_mode != "none":
        for s in use:
            align_scan_energy(s, e0_align)

    # e0 used for grid
    e0_grid = settings["e0_value"] if settings["e0_value"] is not None else e0_align

    # optional baseline offset correction
    if args.shift_minima:
        for s in use:
            n = len(s.mu)
            k = max(1, int(n * 0.05))
            min_mu = np.min(np.sort(s.mu)[:k])
            s.mu -= min_mu

    # Effective raw dE step (robust low percentile)
    eps_de = 1e-6
    all_raw_de = []
    for s in use:
        dE = np.diff(s.energy_eV)
        valid = dE[dE > eps_de]
        if len(valid) > 0:
            all_raw_de.extend(np.sort(valid).tolist())
    eff_de = float(np.percentile(all_raw_de, 5.0)) if all_raw_de else None

    g = settings["grid"]
    too_fine_params = []
    for label, val in [("de_pre", g["de_pre"]), ("de_xanes", g["de_xanes"])]:
        if eff_de is not None and val < eff_de * 0.95:
            too_fine_params.append(f"{label}={val}")

    # initial edges
    edges = make_piecewise_edges(e0_grid, g)

    # region bounds
    pre_lo = e0_grid + g["pre_start"]
    pre_hi = e0_grid + g["pre_end"]
    xanes_lo = pre_hi
    xanes_hi = e0_grid + g["xanes_end"]

    coarse_pre = not args.no_coarse_pre_edge
    coarse_xanes = not args.no_coarse_xanes
    coarse_exafs = not args.no_coarse_exafs

    coarsened_regions = []
    coarsened_intervals = []

    if eff_de is not None and eff_de > 0:
        edges_before = list(edges)

        def region_of(e, tol=1e-9):
            if e < pre_hi - tol:
                return "pre-edge"
            elif e < xanes_hi - tol:
                return "xanes"
            else:
                return "exafs"

        edges_out = [edges[0]]
        i = 1
        while i < len(edges):
            lo = float(edges_out[-1])
            hi = float(edges[i])

            region = region_of(lo)
            coarse_ok = ((region == "pre-edge" and coarse_pre) or
                         (region == "xanes" and coarse_xanes) or
                         (region == "exafs" and coarse_exafs))

            if coarse_ok and (hi - lo) < eff_de:
                i += 1
            else:
                edges_out.append(edges[i])
                i += 1

        edges = np.array(edges_out)

        if len(edges_before) > len(edges):
            before_set = set(np.round(edges_before, 9))
            after_set = set(np.round(edges, 9))
            removed_values = sorted(before_set - after_set)

            if removed_values:
                for i in range(len(edges_before) - 1):
                    lo_b = float(edges_before[i])
                    hi_b = float(edges_before[i + 1])
                    if any(lo_b <= rv <= hi_b for rv in removed_values):
                        coarsened_intervals.append((lo_b, hi_b))

                # merge overlapping intervals
                if coarsened_intervals:
                    merged = [coarsened_intervals[0]]
                    for lo, hi in coarsened_intervals[1:]:
                        mlo, mhi = merged[-1]
                        if lo <= mhi + 1e-6:
                            merged[-1] = (mlo, max(mhi, hi))
                        else:
                            merged.append((lo, hi))
                    coarsened_intervals = merged

                # summarize regional coverage
                e_ex1 = e0_grid + K_CONV_EV_PER_A2 * g["kmax"]**2
                pre_len = max(pre_hi - pre_lo, 1e-6)
                xanes_len = max(xanes_hi - xanes_lo, 1e-6)
                exafs_lo = xanes_hi
                exafs_len = max(e_ex1 - exafs_lo, 1e-6)

                pre_cov = xanes_cov = exafs_cov = 0.0
                for lo, hi in coarsened_intervals:
                    ov0, ov1 = max(lo, pre_lo), min(hi, pre_hi)
                    if ov0 < ov1:
                        pre_cov += (ov1 - ov0) / pre_len

                    ov0, ov1 = max(lo, xanes_lo), min(hi, xanes_hi)
                    if ov0 < ov1:
                        xanes_cov += (ov1 - ov0) / xanes_len

                    ov0, ov1 = max(lo, exafs_lo), min(hi, e_ex1)
                    if ov0 < ov1:
                        exafs_cov += (ov1 - ov0) / exafs_len

                if pre_cov > 0.25:
                    coarsened_regions.append("pre-edge")
                if xanes_cov > 0.10:
                    coarsened_regions.append("XANES")
                if exafs_cov > 0.10:
                    coarsened_regions.append("EXAFS (low-k oversampling control)")

    bin_widths = np.diff(edges)
    too_fine_mask = (bin_widths < (eff_de * 0.95)) if (eff_de is not None and eff_de > 0) else np.array([], dtype=bool)
    any_too_fine = np.any(too_fine_mask) if too_fine_mask.size else False

    if bool(coarsened_regions) or any_too_fine:
        print("\nNOTE: Automatic energy-grid coarsening applied:")
        if too_fine_params:
            print(f"  - Specified bin sizes finer than data resolution: {', '.join(too_fine_params)}")
        else:
            print("  - Some requested bins were finer than the effective raw dE step.")

        for r in coarsened_regions:
            print(f"  - Coarsened in: {r}")
        print(f"  - Grid was coarsened so no bin is smaller than ~{eff_de:.4f} eV.")

    if any_too_fine:
        regions_warn = []
        i = int(np.nonzero(too_fine_mask)[0][0])
        while i < len(bin_widths):
            if not too_fine_mask[i]:
                i += 1
                continue
            j = i + 1
            while j < len(bin_widths) and too_fine_mask[j]:
                j += 1
            regions_warn.append((float(edges[i]), float(edges[j])))
            i = j

        print("\nWARNING: Some bins are finer than the effective raw dE step.")
        for e0, e1 in regions_warn:
            print(f"  Over-resolved region from {e0:.3f} to {e1:.3f} eV (bin < {eff_de:.4f} eV).")

    # averaging modes
    if args.rebin_scans:
        E, MU, SIG, N, all_mu = average_binned(use, edges, smooth_unc=settings["smooth_uncertainty"])
        n_label = "n_contrib_scans"
    else:
        pool_e = np.concatenate([s.energy_eV for s in use])
        pool_mu = np.concatenate([s.mu for s in use])
        E, MU, SIG, n_pts = bin_scan_to_edges(pool_e, pool_mu, edges)

        # keep bins with >=2 pooled points so sigma is defined robustly
        keep = n_pts >= 2
        E, MU, SIG, n_pts = E[keep], MU[keep], SIG[keep], n_pts[keep]

        if settings["smooth_uncertainty"]:
            SIG = smooth_sigma(SIG, win=11)

        N = n_pts
        all_mu = None
        n_label = "n_contrib_points"

        print(f"\nPooled {len(pool_e)} raw points from {len(use)} scans into {len(E)} bins.")

    out = settings["output"]
    out_df = pd.DataFrame({
        "E_eV": E,
        "mu_raw": MU,
        "sigma_mu": SIG,
        n_label: N
    })
    out_df.to_csv(out, sep="\t", index=False, float_format="%.8g")

    print(f"\nSaved: {out}")
    print(f"Used scans: {len(use)} / {len(scans)}")
    print(f"Output points: {len(out_df)}")

    if eff_de is not None and eff_de > 0:
        print(f"Effective raw dE step across scans: {eff_de:.5f} eV")
    else:
        print("Effective raw dE step across scans: could not be determined (no usable steps).")

    if settings["plot"]["enabled"]:
        e_ex1 = e0_grid + K_CONV_EV_PER_A2 * g["kmax"]**2
        regions = {
            "Pre-edge": (e0_grid + g["pre_start"], e0_grid + g["pre_end"]),
            "XANES": (e0_grid + g["pre_end"], e0_grid + g["xanes_end"]),
            "EXAFS": (e0_grid + g["xanes_end"], e_ex1),
        }

        figs = []
        figs.extend(plot_qc(scans, use, save_prefix=settings["plot"]["save_prefix"]))
        figs.append(plot_average(
            E, MU, SIG, all_mu=all_mu,
            save_prefix=settings["plot"]["save_prefix"],
            regions=regions,
            coarsened_intervals=coarsened_intervals
        ))

        if settings["plot"]["show"]:
            plt.show()
        else:
            for fig in figs:
                plt.close(fig)


if __name__ == "__main__":
    main()