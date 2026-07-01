#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xas_bin_average.py

XAS scan QC, E0 alignment, variable-grid binning, averaging, uncertainty estimation.
Supports SPring-8/Aichi/SAGA-style text files.

New in this version:
- JSON/YAML config support
- default grid options when none supplied
- optional plotting
"""

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass
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
    flags: list = None


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

    if suffix in [".json"]:
        return json.loads(txt)
    elif suffix in [".yaml", ".yml"]:
        if not HAS_YAML:
            raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
        return yaml.safe_load(txt)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}. Use .json/.yaml/.yml")


def build_settings(args):
    settings = json.loads(json.dumps(DEFAULTS))  # deep copy
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

    # grid overrides (only if provided)
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
        flags=[]
    )


# -------------------------
# QC + E0
# -------------------------
def estimate_e0(energy, mu, window=31, poly=3, exclude=(0.05, 0.05)):
    """Estimate E0 as the energy of maximum d(mu)/d(E), excluding edge regions.

    Parameters
    ----------
    energy : 1D array of energy values (monotonic).
    mu : corresponding absorbance.
    window : Savitzky–Golay window size (will be adjusted if needed).
    poly : polynomial order for SG filter.
    exclude : (low_frac, high_frac) fractions to exclude from search range
              to avoid spurious maxima at scan edges; default 5% each side.

    Returns
    -------
    Estimated E0 (energy with maximum derivative in the interior region).
    """
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

    # Constrained search to avoid edge artifacts.
    slice_ = dmu[lo:hi]
    idx = np.argmax(slice_) + lo
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
def make_piecewise_edges(e0, g):
    e_pre0 = e0 + g["pre_start"]
    e_pre1 = e0 + g["pre_end"]
    e_x1 = e0 + g["xanes_end"]
    e_ex1 = e0 + K_CONV_EV_PER_A2 * g["kmax"]**2

    edges_pre = np.arange(e_pre0, e_pre1 + g["de_pre"], g["de_pre"])
    if edges_pre[-1] < e_pre1:
        edges_pre = np.append(edges_pre, e_pre1)

    edges_x = np.arange(e_pre1, e_x1 + g["de_xanes"], g["de_xanes"])
    if edges_x[-1] < e_x1:
        edges_x = np.append(edges_x, e_x1)

    k0 = np.sqrt(max(e_x1 - e0, 0.0) / K_CONV_EV_PER_A2)
    ks = np.arange(k0, g["kmax"] + g["dk"], g["dk"])
    e_k = e0 + K_CONV_EV_PER_A2 * ks**2
    if e_k[-1] < e_ex1:
        e_k = np.append(e_k, e_ex1)

    edges = np.unique(np.concatenate([edges_pre, edges_x, e_k]))
    edges.sort()
    return edges


def coarsen_edges_to_min_step(edges: np.ndarray, min_step: float) -> np.ndarray:
    """Merge adjacent bin edges that are closer than min_step.

    Keeps the first edge and removes intermediate edges until we have a gap >= min_step.
    This prevents over-resolved bins that produce zig-zag artifacts.
    """
    if len(edges) <= 2:
        return edges

    out = [edges[0]]
    i = 1
    while i < len(edges):
        # Extend from last kept edge until we exceed min_step.
        lo = float(out[-1])
        hi = float(edges[i])
        if (hi - lo) >= min_step:
            out.append(edges[i])
            i += 1
        else:
            # Skip this edge; it's too close to last kept.
            i += 1

    return np.array(out)


# -------------------------
# Binning
# -------------------------
def bin_scan_to_edges(energy, mu, edges):
    idx = np.digitize(energy, edges) - 1
    nb = len(edges) - 1
    b_mu = np.full(nb, np.nan)
    b_e = np.full(nb, np.nan)

    for i in range(nb):
        m = idx == i
        if np.any(m):
            b_mu[i] = np.mean(mu[m])
            b_e[i] = np.mean(energy[m])
    return b_e, b_mu


def smooth_sigma(sig, win=11):
    x = sig.copy()
    good = np.isfinite(x)
    if np.sum(good) < 3:
        return x
    xi = np.arange(len(x))
    x[~good] = np.interp(xi[~good], xi[good], x[good])
    if win % 2 == 0:
        win += 1
    win = min(win, len(x) - (1 - len(x) % 2))
    if win < 3:
        return x
    return median_filter(x, size=win, mode="nearest")


def average_binned(scans, edges, smooth_unc=True):
    all_e = []
    all_mu = []
    for s in scans:
        be, bm = bin_scan_to_edges(s.energy_eV, s.mu, edges)
        all_e.append(be)
        all_mu.append(bm)

    all_e = np.array(all_e)
    all_mu = np.array(all_mu)

    mean_e = np.nanmean(all_e, axis=0)
    mean_mu = np.nanmean(all_mu, axis=0)
    n_eff = np.sum(np.isfinite(all_mu), axis=0)

    std = np.nanstd(all_mu, axis=0, ddof=1)
    sigma_mean = std / np.sqrt(np.maximum(n_eff, 1))
    if smooth_unc:
        sigma_mean = smooth_sigma(sigma_mean, win=11)

    keep = n_eff >= 2
    return mean_e[keep], mean_mu[keep], sigma_mean[keep], n_eff[keep], all_mu[:, keep]


# -------------------------
# Diagnostics + plotting
# -------------------------
def report_raw_deltaE(scans):
    print("\n=== Raw delta-E diagnostic ===")
    for s in scans:
        dE = np.abs(np.diff(s.energy_eV))
        i = np.argmax(dE)
        e_mid = 0.5 * (s.energy_eV[i] + s.energy_eV[i+1])
        print(f"{s.path.name}: max dE={dE[i]:.6f} eV at E~{e_mid:.3f} eV")


def plot_qc(scans, used_scans, save_prefix=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in scans:
        c = "tab:blue" if s in used_scans else "tab:red"
        ax.plot(s.energy_eV, s.mu, color=c, alpha=0.25, lw=0.8)
        ax.axvline(s.e0, color=c, alpha=0.08)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Raw absorbance (mu)")
    ax.set_title("QC overlay: scans and E0 markers\n(blue=used, red=excluded)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_qc_overlay.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    e0s = np.array([s.e0 for s in scans])
    ax2.hist(e0s, bins=max(8, min(30, len(scans)//2 + 3)), color="gray", edgecolor="black")
    ax2.set_xlabel("E0 (eV)")
    ax2.set_ylabel("Count")
    ax2.set_title("E0 distribution")
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    if save_prefix:
        fig2.savefig(f"{save_prefix}_e0_hist.png", dpi=150)

    return [fig, fig2]


def plot_average(E, MU, SIG, save_prefix=None, regions=None):
    """Plot averaged XAS spectrum with optional region shading.

    Parameters
    ----------
    E : 1D array of energy values.
    MU: corresponding mu (absorbance).
    SIG: uncertainty.
    save_prefix: prefix for saving PNGs.
    regions: optional dict mapping label -> (lo, hi) energy bounds to shade.
             Example: {"Pre-edge": (-200+E0, -30+E0),
                        "XANES":    (-30+E0, 50+E0),
                        "EXAFS":    (50+E0, E_max)}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Shade regions faintly if provided.
    region_colors = {
        "Pre-edge": "#FFDDC1",
        "XANES":    "#C8E6FF",
        "EXAFS":    "#D4FFD7",
    }

    if isinstance(regions, dict):
        for label, (lo, hi) in regions.items():
            color = region_colors.get(label, "#EEEEEE")
            ax.axvspan(lo, hi, facecolor=color, alpha=0.35, zorder=0,
                       label=label)

    # Plot points as markers to reveal the final grid spacing.
    ax.scatter(
        E, MU,
        color="black",
        s=8,
        edgecolor="none",
        alpha=0.9,
        label="Averaged μ (points)"
    )

    # Optional: light connecting line to emphasize continuity without hiding grid.
    ax.plot(E, MU, color="black", lw=0.3, alpha=0.6)

    # Uncertainty band.
    ax.fill_between(
        E, MU - SIG, MU + SIG,
        color="tab:blue",
        alpha=0.25,
        label="±σ(mean)"
    )

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Raw absorbance (mu)")
    ax.set_title("Binned/averaged XAS spectrum")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_average.png", dpi=150)
    return fig


def align_scan_energy(scan, e0_target):
    scan.energy_eV = scan.energy_eV + (e0_target - scan.e0)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="XAS bin/average tool with config and plotting.")
    ap.add_argument("files", nargs="+", help="Input scan files (base names or paths)")
    ap.add_argument(
        "-d", "--dir",
        type=str,
        default=None,
        help="Base directory to prepend to each file argument "
             "(e.g. -d scans --file scan1.dat scan2.dat)"
    )
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

    ap.add_argument("--output", default=None)

    # plotting
    ap.add_argument("--plot", action="store_true", help="Enable plotting")
    ap.add_argument("--no-show", action="store_true", help="Do not show interactive windows")
    ap.add_argument("--plot-prefix", default=None, help="Prefix to save plot PNGs")

    args = ap.parse_args()
    settings = build_settings(args)

    base_dir = Path(args.dir) if args.dir else None
    expanded_files: list[Path] = []

    for token in args.files:
        # Decide candidate root(s) for this token.
        # If it already includes a directory component, prefer that as-is.
        tpath = Path(token)

        # Heuristic: if it looks like a glob pattern (contains * or ?), expand it.
        if any(c in token for c in ("*", "?")) and not tpath.is_absolute():
            # If --dir given, expand relative to that; otherwise current dir.
            root = base_dir or Path(".")
            candidates = list(root.glob(token))
            if not candidates:
                raise FileNotFoundError(
                    f"No files matched pattern '{token}' "
                    f"in '{root}'. Check --dir and your working directory."
                )
            expanded_files.extend(candidates)
        else:
            # Otherwise treat as literal; optionally prepend base_dir.
            if base_dir is not None and not tpath.is_absolute():
                candidate = base_dir / token
            else:
                candidate = Path(token)
            expanded_files.append(candidate)

    files = [f for f in expanded_files]
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

    qc = settings["qc"]
    qc_scans(scans, qc["shutter_threshold_frac"], qc["min_range_frac"], qc["median_i0_z"])

    print("\n=== Scan QC summary ===")
    for i, s in enumerate(scans):
        flag = ",".join(s.flags) if s.flags else "OK"
        print(f"[{i}] {s.path.name}  E0={s.e0:.3f} eV  FLAGS={flag}")

    report_raw_deltaE(scans)

    # Exclude obvious bad scans (basic rule; can be expanded)
    use = [s for s in scans if ("TRUNCATED_RANGE" not in s.flags and "LOW_SIGNAL_SHUTTER_OR_GAIN" not in s.flags)]
    if len(use) < 2:
        raise RuntimeError("Not enough usable scans after QC filtering.")

    # alignment target
    align_mode = settings["align"]
    if align_mode == "none":
        e0_target = np.mean([s.e0 for s in use])
    elif align_mode == "mean":
        e0_target = np.mean([s.e0 for s in use])
    elif align_mode == "ref":
        e0_target = scans[settings["ref_index"]].e0
    else:
        if settings["e0_value"] is None:
            raise ValueError("align='value' but e0_value is not set.")
        e0_target = settings["e0_value"]

    if align_mode != "none":
        for s in use:
            align_scan_energy(s, e0_target)

    # Estimate effective raw energy step across usable scans.
    # Use a low percentile (e.g., 5%) to be robust against tiny outlier steps.
    eps_de = 1e-6
    all_raw_de: list[float] = []
    for s in use:
        dE = np.diff(s.energy_eV)
        valid = dE[dE > eps_de]
        if len(valid) > 0:
            all_raw_de.extend(np.sort(valid).tolist())

    # Use a robust effective step instead of absolute minimum.
    eff_de = float(np.percentile(all_raw_de, 5.0)) if all_raw_de else None

    g = settings["grid"]
    too_fine_params: list[str] = []
    for label, val in [
        ("de_pre", g["de_pre"]),
        ("de_xanes", g["de_xanes"]),
    ]:
        if eff_de is not None and val < eff_de * 0.95:
            too_fine_params.append(f"{label}={val}")

    # Build initial bin edges.
    edges = make_piecewise_edges(e0_target, g)

    # Coarsen bins that are finer than what the raw data can support.
    # This both prevents over-resolution warnings and avoids zig-zag artifacts.
    coarsened = False
    if eff_de is not None and eff_de > 0:
        edges_before = len(edges)
        edges = coarsen_edges_to_min_step(edges, eff_de)
        coarsened = (len(edges) < edges_before)

    # After coarsening, check whether any bins are still finer than allowed.
    bin_widths = np.diff(edges)
    too_fine_mask: np.ndarray | bool = False
    if eff_de is not None and eff_de > 0:
        too_fine_mask = bin_widths < (eff_de * 0.95)

    any_too_fine = isinstance(too_fine_mask, np.ndarray) and np.any(too_fine_mask)

    # Notify user about auto-coarsening if it occurred or some bins are still too fine.
    if coarsened or any_too_fine:
        print("\nNOTE: Automatic energy-grid coarsening applied:")
        if too_fine_params:
            print(f"  - Specified bin sizes finer than data resolution: {', '.join(too_fine_params)}")
        else:
            print("  - Some requested bins were finer than the effective raw dE step.")
        print(f"  - Grid was coarsened so no bin is smaller than ~{eff_de:.4f} eV.")

    if any_too_fine:
        regions: list[tuple[float, float]] = []
        i = int(np.nonzero(too_fine_mask)[0][0])
        while i < len(bin_widths):
            if not too_fine_mask[i]:
                i += 1
                continue
            j = i + 1
            while j < len(bin_widths) and too_fine_mask[j]:
                j += 1
            regions.append((float(edges[i]), float(edges[j])))
            i = j

        print("\nWARNING: Some bins are finer than the effective raw dE step.")
        for (e0, e1) in regions:
            print(f"  Over-resolved region from {e0:.3f} to {e1:.3f} eV "
                  f"(bin < {eff_de:.4f} eV).")

    E, MU, SIG, N, _ = average_binned(
        use, edges, smooth_unc=settings["smooth_uncertainty"]
    )

    out = settings["output"]
    out_df = pd.DataFrame({
        "E_eV": E,
        "mu_raw": MU,
        "sigma_mu": SIG,
        "n_scans": N
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
        g = settings["grid"]
        e_ex1 = e0_target + K_CONV_EV_PER_A2 * g["kmax"]**2

        regions = {
            "Pre-edge": (e0_target + g["pre_start"],
                         e0_target + g["pre_end"]),
            "XANES":    (e0_target + g["pre_end"],
                         e0_target + g["xanes_end"]),
            "EXAFS":    (e0_target + g["xanes_end"],
                         e_ex1),
        }

        figs = []
        figs.extend(plot_qc(scans, use, save_prefix=settings["plot"]["save_prefix"]))
        figs.append(plot_average(
            E, MU, SIG,
            save_prefix=settings["plot"]["save_prefix"],
            regions=regions
        ))
        if settings["plot"]["show"]:
            plt.show()
        else:
            for fig in figs:
                plt.close(fig)


if __name__ == "__main__":
    main()