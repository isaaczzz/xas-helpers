#!/usr/bin/env python3
"""
shift_es.py — energy calibration via E0 alignment or explicit offset.

Three modes:

  align     Each spectrum is shifted individually so its detected E0 moves to ref-e0.

  offset    A fixed energy offset is applied to all spectra without E0 detection.

  reference One or more reference spectra are used to compute a single shift:
              shift = ref-e0 - mean(detected E0 of ref-spectra)
            That same shift is then applied rigidly to all remaining spectra
            listed under 'files'. The reference spectra themselves are also
            written out with the shift applied.

YAML format:

  defaults:
    dir: /data/session
    file-prefix: "scan"
    file-suffix: ".dat"
    mode: align
    ref-e0: 8980.3

  jobs:
    - name: align_all
      mode: align
      ref-e0: 8980.3
      files: ["sample_A.dat", "sample_B.dat"]

    - name: fixed_shift
      mode: offset
      offset: -2.5
      files: ["sample_C.dat"]

    - name: ref_based
      mode: reference
      ref-e0: 8980.3
      ref-spectra: ["cufoil.dat"]        # one or more reference spectra
      files: ["sample_A.dat", "sample_B.dat"]   # receive the same shift

Usage:
    python bin/shift_es.py jobs.yaml
    python bin/shift_es.py batch1.yaml --ref-e0 8980.3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Lazy-loaded YAML callables
safe_load = None   # type: ignore


def _ensure_yaml() -> None:
    global safe_load
    if safe_load is not None:
        return
    try:
        from yaml import safe_load as sl
        safe_load = sl
    except ImportError:
        print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)


# -------------------------
# Spectrum loading
# -------------------------

HC_EV_ANG = 12398.4193


def parse_spring8_file(filepath: Path, mode: str = "transmission",
                       angle_col="Angle(o)", time_col="time/s",
                       i0_col="2", i1_col="3", fluo_cols=None) -> dict:
    """Parse an SPring-8/Aichi-style raw scan file."""
    import re

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
    time_s    = df[time_col].to_numpy()
    i0        = df[i0_col].to_numpy()
    eps = 1e-12

    if mode == "transmission":
        if i1_col not in df.columns:
            raise ValueError(f"{i1_col} not found in {filepath}")
        i1 = df[i1_col].to_numpy()
        mu = np.log((i0 + eps) / (i1 + eps))
    else:
        if not fluo_cols:
            reserved = {angle_col, "Angle(c)", time_col, i0_col}
            fluo_cols = [c for c in df.columns if c not in reserved]
        fluo = df[fluo_cols].sum(axis=1).to_numpy()
        mu = fluo / (i0 + eps)

    energy = HC_EV_ANG / (2.0 * d_spacing * np.sin(np.deg2rad(theta_deg)))
    order = np.argsort(energy)
    return {
        "path": filepath,
        "energy_eV": energy[order],
        "mu": mu[order],
        "df": df.iloc[order].reset_index(drop=True),
    }


def parse_simple_spectrum(filepath: Path) -> dict:
    """
    Load a simple TSV/CSV spectrum such as bin_average output.
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
        "mu":        df[mu_col].iloc[order].to_numpy(),
        "df":        df.iloc[order].reset_index(drop=True),
    }


def load_spectrum(filepath: Path) -> dict:
    """Try SPring-8 parser first, fall back to simple TSV."""
    try:
        return parse_spring8_file(filepath)
    except Exception:
        return parse_simple_spectrum(filepath)


# -------------------------
# Gaussian smoothing
# -------------------------

def gaussian_smooth(y: np.ndarray, x: np.ndarray, sigma_x: float) -> np.ndarray:
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


# -------------------------
# E0 estimation
# -------------------------

def estimate_e0_coarse(energy: np.ndarray, mu: np.ndarray,
                       search_min: float | None = None,
                       search_max: float | None = None) -> float:
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
    """Return the best E0 estimate for a single spectrum."""
    E0_coarse = estimate_e0_coarse(energy, mu, search_min=search_min, search_max=search_max)
    E0_refined, ok = refine_e0_local(energy, mu, E0_coarse)
    return E0_refined if ok else E0_coarse


# -------------------------
# Shifting utility
# -------------------------

def apply_shift(spectrum: dict, shift_eV: float) -> dict:
    out = dict(spectrum)
    out["energy_eV"] = spectrum["energy_eV"] + shift_eV
    return out


# -------------------------
# File helpers
# -------------------------

_JOB_META = {
    "name", "files", "ref-spectra", "dir", "file-prefix", "file-suffix",
    "mode", "ref-e0", "offset",
}


def _validate_files_list(files: Any, where: str) -> list[str]:
    if not isinstance(files, list) or len(files) == 0:
        raise ValueError(f"{where}: 'files' must be a non-empty list")
    out: list[str] = []
    for i, x in enumerate(files):
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f"{where}: files[{i}] must be a non-empty string")
        out.append(x.strip())
    return out


def _apply_prefix_suffix(tokens: list[str], prefix: str, suffix: str, where: str) -> list[str]:
    out = []
    for t in tokens:
        if suffix and t.endswith(suffix):
            print(f"Warning: {where}: token '{t}' already ends with suffix '{suffix}'.",
                  file=sys.stderr)
        out.append(f"{prefix}{t}{suffix}")
    return out


def _expand_files(tokens: list[str], dir_path: Path | None) -> list[Path]:
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
                raise FileNotFoundError(f"No files matched pattern '{t}' under {base}")
            resolved.extend(matches)
        else:
            full = base / t
            if not full.exists():
                raise FileNotFoundError(f"File not found: {full}")
            resolved.append(full)

    return resolved


# -------------------------
# YAML batch loading
# -------------------------

def load_batch(filepath: str, cli_overrides: dict) -> list[dict]:
    """Load one YAML batch file and return a list of normalized job dicts."""
    _ensure_yaml()
    p = Path(filepath)

    with p.open("r", encoding="utf-8") as f:
        data = safe_load(f)

    if data is None:
        raise ValueError(f"{filepath}: empty file")

    if isinstance(data, list):
        # Bare mode: plain list of filenames -> single align job.
        files = _validate_files_list(data, f"{filepath} (bare mode)")
        return [{"name": p.stem, "files_tokens": files, "ref_spectra_tokens": [],
                 "dir": None, "mode": "align",
                 "ref_e0": None, "offset": None}]

    if not isinstance(data, dict):
        raise ValueError(f"{filepath}: expected a mapping or list at top level")

    defaults_raw: dict[str, Any] = data.get("defaults", {}) or {}
    jobs_raw = data.get("jobs")

    # CLI overrides < YAML defaults: YAML defaults win.
    defaults: dict[str, Any] = {**cli_overrides}
    for k, v in defaults_raw.items():
        if v is not None:
            defaults[k] = v

    if not isinstance(defaults, dict):
        raise ValueError(f"{filepath}: 'defaults' must be a mapping")
    if jobs_raw is None:
        raise ValueError(f"{filepath}: no 'jobs' key found")
    if not isinstance(jobs_raw, list) or len(jobs_raw) == 0:
        raise ValueError(f"{filepath}: 'jobs' must be a non-empty list")

    resolved: list[dict] = []

    for i, job in enumerate(jobs_raw):
        where = f"{filepath}: jobs[{i}]"
        if not isinstance(job, dict):
            raise ValueError(f"{where}: each job must be a mapping")

        name = job.get("name", defaults.get("name", f"job_{i+1}"))
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}: 'name' must be a non-empty string if provided")
        name = name.strip()

        prefix = str(job.get("file-prefix", defaults.get("file-prefix", "")) or "")
        suffix = str(job.get("file-suffix", defaults.get("file-suffix", "")) or "")

        # 'files': spectra to shift (required).
        files_raw = _validate_files_list(job.get("files"), where)
        files_tokens = _apply_prefix_suffix(files_raw, prefix, suffix, where)

        # 'ref-spectra': reference spectra for reference mode (optional).
        ref_spectra_raw = job.get("ref-spectra", defaults.get("ref-spectra"))
        if ref_spectra_raw is not None:
            ref_spectra_raw = _validate_files_list(ref_spectra_raw, f"{where} ref-spectra")
            ref_spectra_tokens = _apply_prefix_suffix(ref_spectra_raw, prefix, suffix,
                                                       f"{where} ref-spectra")
        else:
            ref_spectra_tokens = []

        scan_dir: Path | None = None
        for src in (job, defaults):
            if "dir" in src and src["dir"] is not None:
                scan_dir = Path(str(src["dir"]))
                break

        mode = str(job.get("mode", defaults.get("mode", "")) or "").strip().lower()
        ref_e0  = job.get("ref-e0",  defaults.get("ref-e0"))
        offset_val = job.get("offset", defaults.get("offset"))

        # Infer mode if not specified.
        if not mode:
            if offset_val is not None:
                mode = "offset"
            elif ref_spectra_tokens:
                mode = "reference"
            elif ref_e0 is not None:
                mode = "align"
            else:
                raise ValueError(f"{where}: cannot infer mode — specify mode, ref-e0, or offset")

        # Validate required parameters per mode.
        if mode == "align":
            if ref_e0 is None:
                raise ValueError(f"{where}: mode=align requires ref-e0")
        elif mode == "offset":
            if offset_val is None:
                raise ValueError(f"{where}: mode=offset requires offset")
        elif mode == "reference":
            if ref_e0 is None:
                raise ValueError(f"{where}: mode=reference requires ref-e0")
            if not ref_spectra_tokens:
                raise ValueError(f"{where}: mode=reference requires ref-spectra")
        else:
            raise ValueError(f"{where}: unknown mode '{mode}'. Use align, offset, or reference.")

        search_min = job.get("search-min", defaults.get("search-min"))
        search_max = job.get("search-max", defaults.get("search-max"))

        resolved.append({
            "name": name,
            "files_tokens": files_tokens,
            "ref_spectra_tokens": ref_spectra_tokens,
            "dir": scan_dir,
            "mode": mode,
            "ref_e0":  float(ref_e0) if ref_e0 is not None else None,
            "offset":  float(offset_val) if offset_val is not None else None,
            "search_min": float(search_min) if search_min is not None else None,
            "search_max": float(search_max) if search_max is not None else None,
        })

    return resolved


# -------------------------
# Output helpers
# -------------------------

def make_output_path(input_path: Path, output_dir: Path | None,
                     suffix: str = "_shifted") -> Path:
    """Insert suffix before the extension: scan001.dat -> scan001_shifted.dat."""
    out_name = f"{input_path.stem}{suffix}{input_path.suffix or ''}"
    return (output_dir if output_dir else Path(".")) / out_name


def write_spectrum(spectrum: dict, out_path: Path,
                   e0_original: float | None = None,
                   e0_shifted: float | None = None,
                   shift_eV: float | None = None,
                   extra_header: list[str] | None = None) -> None:
    """Write shifted spectrum to file, overwriting the energy column in place."""
    df = spectrum["df"].copy()

    energy_col = "E_eV"
    for c in ["E_eV", "Energy(eV)", "energy_eV", "E(eV)", "E (eV)"]:
        if c in df.columns:
            energy_col = c
            break

    df[energy_col] = spectrum["energy_eV"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines: list[str] = []
    if e0_original is not None and e0_shifted is not None and shift_eV is not None:
        header_lines.append(
            f"# E0_original = {e0_original:.4f} eV  "
            f"E0_shifted = {e0_shifted:.4f} eV  "
            f"shift = {shift_eV:+.4f} eV"
        )
    for line in (extra_header or []):
        header_lines.append(line if line.startswith("#") else f"# {line}")

    with out_path.open("w", encoding="utf-8") as fh:
        for line in header_lines:
            fh.write(line + "\n")
        fh.write(df.to_csv(sep="\t", index=False, float_format="%.8g"))


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="shift_es.py — energy calibration via E0 alignment or explicit offset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Modes:
  align      Shift each spectrum individually so its E0 aligns to ref-e0.
  offset     Apply a fixed energy offset to all spectra.
  reference  Compute shift from ref-spectra and apply same shift to all files.

Examples:
  python bin/shift_es.py jobs.yaml
  python bin/shift_es.py batch.yaml --ref-e0 8980.3
""",
    )

    ap.add_argument("batch_files", nargs="+", help="YAML files specifying jobs")
    ap.add_argument("-d", "--dir", default=None,
                    help="Default scan root directory.")
    ap.add_argument("--ref-e0", type=float, default=None,
                    help="Target E0 value (eV) for align/reference modes.")
    ap.add_argument("--offset", type=float, default=None,
                    help="Fixed energy offset (eV) for offset mode.")
    ap.add_argument("--output-dir", default=None,
                    help="Directory for shifted output files (default: cwd).")
    ap.add_argument("--suffix", default="_shifted",
                    help="Suffix inserted before extension in output filenames.")
    ap.add_argument("--report", default=None,
                    help="Path to write a full shift report.")

    args = ap.parse_args()

    cli_overrides: dict[str, Any] = {}
    if args.dir is not None:
        cli_overrides["dir"] = args.dir
    if args.ref_e0 is not None:
        cli_overrides["ref-e0"] = float(args.ref_e0)
    if args.offset is not None:
        cli_overrides["offset"] = float(args.offset)

    output_dir = Path(args.output_dir) if args.output_dir else None
    out_suffix = args.suffix or "_shifted"

    all_jobs: list[tuple[str, dict]] = []
    for fp in args.batch_files:
        p = Path(fp)
        if not p.exists():
            print(f"Error: {fp} not found.", file=sys.stderr)
            sys.exit(1)
        try:
            all_jobs.extend((p.name, j) for j in load_batch(fp, cli_overrides))
        except Exception as e:
            print(f"Error loading {fp}: {e}", file=sys.stderr)
            sys.exit(1)

    if not all_jobs:
        print("No jobs found.", file=sys.stderr)
        sys.exit(1)

    report_lines: list[str] = []
    n_total = n_ok = n_fail = 0

    for batch_file, job in all_jobs:
        label = f"{batch_file}: {job['name']}"
        mode_job   = job["mode"]
        ref_e0     = job["ref_e0"]
        offset_val = job["offset"]
        scan_dir   = job["dir"]
        search_min = job["search_min"]
        search_max = job["search_max"]

        print(f"\n[{label}]  mode={mode_job}")

        # Resolve and load target files.
        try:
            files = _expand_files(job["files_tokens"], scan_dir)
        except Exception as e:
            print(f"  FAILED to resolve files: {e}", file=sys.stderr)
            n_fail += 1
            report_lines.append(f"{label}: FAILED (file resolution: {e})")
            continue

        scans: list[dict] = []
        load_ok = True
        for f in files:
            try:
                scans.append(load_spectrum(f))
            except Exception as e:
                print(f"  FAILED to load {f.name}: {e}", file=sys.stderr)
                load_ok = False
        if not load_ok or not scans:
            n_fail += 1
            report_lines.append(f"{label}: FAILED (load error)")
            continue

        job_ok = True
        try:
            if mode_job == "offset":
                shift_eV = float(offset_val)
                for s in scans:
                    out_path = make_output_path(s["path"], output_dir, out_suffix)
                    write_spectrum(apply_shift(s, shift_eV), out_path,
                                   shift_eV=shift_eV,
                                   extra_header=["mode: offset"])
                    n_total += 1
                    print(f"  {s['path'].name}: shift {shift_eV:+.4f} eV -> {out_path.name}")
                    report_lines.append(
                        f"  {s['path'].name}: mode=offset  shift={shift_eV:+.4f} eV"
                        f"  -> {out_path}"
                    )
                n_ok += len(scans)

            elif mode_job == "align":
                # Each spectrum shifted independently to ref-e0.
                for s in scans:
                    E0_orig = estimate_e0(s["energy_eV"], s["mu"],
                                          search_min=search_min, search_max=search_max)
                    shift_eV = ref_e0 - E0_orig
                    out_path = make_output_path(s["path"], output_dir, out_suffix)
                    write_spectrum(apply_shift(s, shift_eV), out_path,
                                   e0_original=E0_orig, e0_shifted=ref_e0, shift_eV=shift_eV,
                                   extra_header=[f"mode: align  ref_e0={ref_e0:.4f} eV"])
                    n_total += 1
                    print(f"  {s['path'].name}: E0 {E0_orig:.4f} -> {ref_e0:.4f} eV"
                          f"  (shift {shift_eV:+.4f} eV) -> {out_path.name}")
                    report_lines.append(
                        f"  {s['path'].name}: E0_original={E0_orig:.4f} eV"
                        f"  E0_shifted={ref_e0:.4f} eV  shift={shift_eV:+.4f} eV"
                        f"  -> {out_path}"
                    )
                n_ok += len(scans)

            elif mode_job == "reference":
                # Load reference spectra.
                try:
                    ref_files = _expand_files(job["ref_spectra_tokens"], scan_dir)
                except Exception as e:
                    raise RuntimeError(f"failed to resolve ref-spectra: {e}") from e

                ref_scans: list[dict] = []
                for f in ref_files:
                    try:
                        ref_scans.append(load_spectrum(f))
                    except Exception as e:
                        raise RuntimeError(f"failed to load ref-spectrum {f.name}: {e}") from e

                # Compute shift from mean E0 of reference spectra.
                ref_E0s = [
                    estimate_e0(r["energy_eV"], r["mu"],
                                search_min=search_min, search_max=search_max)
                    for r in ref_scans
                ]
                mean_ref_E0 = float(np.mean(ref_E0s))
                shift_eV = ref_e0 - mean_ref_E0

                if len(ref_scans) == 1:
                    print(f"  Reference ({ref_scans[0]['path'].name}):"
                          f" E0 = {ref_E0s[0]:.4f} eV")
                else:
                    for r, e0 in zip(ref_scans, ref_E0s):
                        print(f"  Reference ({r['path'].name}): E0 = {e0:.4f} eV")
                    print(f"  Mean reference E0 = {mean_ref_E0:.4f} eV")
                print(f"  Shift = {ref_e0:.4f} - {mean_ref_E0:.4f} = {shift_eV:+.4f} eV")

                ref_header = [
                    f"mode: reference  ref_e0={ref_e0:.4f} eV"
                    f"  mean_ref_E0={mean_ref_E0:.4f} eV  shift={shift_eV:+.4f} eV"
                ]

                # Write shifted reference spectra.
                for r, E0_orig in zip(ref_scans, ref_E0s):
                    out_path = make_output_path(r["path"], output_dir, out_suffix)
                    write_spectrum(apply_shift(r, shift_eV), out_path,
                                   e0_original=E0_orig, e0_shifted=E0_orig + shift_eV,
                                   shift_eV=shift_eV,
                                   extra_header=ref_header + ["role: reference"])
                    print(f"  {r['path'].name} [ref]: E0 {E0_orig:.4f} -> {E0_orig + shift_eV:.4f} eV"
                          f" -> {out_path.name}")
                    report_lines.append(
                        f"  {r['path'].name} [ref]: E0_original={E0_orig:.4f} eV"
                        f"  E0_shifted={E0_orig + shift_eV:.4f} eV  shift={shift_eV:+.4f} eV"
                        f"  -> {out_path}"
                    )

                # Write shifted target spectra with same shift.
                for s in scans:
                    E0_orig = estimate_e0(s["energy_eV"], s["mu"],
                                          search_min=search_min, search_max=search_max)
                    out_path = make_output_path(s["path"], output_dir, out_suffix)
                    write_spectrum(apply_shift(s, shift_eV), out_path,
                                   e0_original=E0_orig, e0_shifted=E0_orig + shift_eV,
                                   shift_eV=shift_eV,
                                   extra_header=ref_header)
                    n_total += 1
                    print(f"  {s['path'].name}: E0 {E0_orig:.4f} -> {E0_orig + shift_eV:.4f} eV"
                          f"  (shift {shift_eV:+.4f} eV) -> {out_path.name}")
                    report_lines.append(
                        f"  {s['path'].name}: E0_original={E0_orig:.4f} eV"
                        f"  E0_shifted={E0_orig + shift_eV:.4f} eV  shift={shift_eV:+.4f} eV"
                        f"  -> {out_path}"
                    )
                n_ok += len(scans)

        except Exception as e:
            job_ok = False
            print(f"  FAILED: {e}", file=sys.stderr)
            report_lines.append(f"{label}: FAILED ({e})")

        if job_ok:
            report_lines.insert(len(report_lines) - len(scans), f"{label}: OK  mode={mode_job}")
        else:
            n_fail += 1

    print(f"\n=== shift_es.py summary ===")
    print(f"Total target spectra processed: {n_total}  succeeded: {n_ok}  failed: {n_fail}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"Full report written to: {args.report}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
