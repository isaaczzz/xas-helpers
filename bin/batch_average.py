#!/usr/bin/env python3
"""Batch wrapper for bin_average.py.

Runs multiple bin_average jobs from one or more YAML specification files.
Each job is executed as a separate subprocess call to bin_average.py.

Usage:
    python bin/batch_average.py batch1.yaml [batch2.yaml ...]

YAML format (with defaults):
    defaults:
      dir: /data/session1
      config: base_config.yaml
      file-prefix: "scan"
      file-suffix: ".dat"
      grid:                          # deep-merged into config.yaml
        kmax: 18.0

    jobs:
      - name: sample_A
        files: ["_001", "_002"]      # becomes scan_001.dat, scan_002.dat
        dir: /data/session1/sampleA  # overrides defaults.dir
        grid:
          kmax: 14.0

      - name: sample_B               # inherits defaults
        files: ["_003", "_004"]

YAML bare mode (plain list = single implicit job):
    - scan001.dat
    - scan002.dat
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import xas_utils
from xas_utils import (
    _ensure_yaml,
    deep_update,
    _validate_files_list, _apply_prefix_suffix,
)


def merge_config(config_path: str | None, overrides: dict) -> tuple[Path | None, bool]:
    """Deep-merge overrides into config contents.

    Returns
    -------
    (path, is_temp)
      path: merged config path (or original path if no overrides)
      is_temp: True if wrapper created a temporary merged file
    """
    if not overrides:
        return (Path(config_path) if config_path else None), False

    _ensure_yaml()
    base: dict[str, Any] = {}

    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with p.open("r", encoding="utf-8") as f:
            base = xas_utils.safe_load(f) or {}
        if not isinstance(base, dict):
            raise ValueError(f"Config file must contain a mapping at top level: {config_path}")

    merged = deep_update(base, overrides)

    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False, encoding="utf-8")
    try:
        xas_utils.safe_dump(merged, tmp, default_flow_style=False, sort_keys=False)
    finally:
        tmp.close()

    return Path(tmp.name), True


SCRIPT_DIR = Path(__file__).resolve().parent
BIN_AVERAGE = SCRIPT_DIR / "bin_average.py"

# Keys treated as job metadata (not config overrides)
_JOB_META = {"name", "files", "dir", "config", "file-prefix", "file-suffix", "output", "output-dir"}


def load_batch(filepath: str) -> list[dict]:
    """Load one YAML batch spec and return normalized job dicts."""
    _ensure_yaml()
    p = Path(filepath)

    with p.open("r", encoding="utf-8") as f:
        data = xas_utils.safe_load(f)

    if data is None:
        raise ValueError(f"{filepath}: empty file")

    # Bare mode: plain list of strings -> one implicit job (no prefix/suffix)
    if isinstance(data, list):
        files = _validate_files_list(data, f"{filepath} (bare mode)")
        return [{"name": p.stem, "files": files, "config-overrides": {}}]

    if not isinstance(data, dict):
        raise ValueError(f"{filepath}: expected a mapping or list at top level")

    defaults = data.get("defaults", {}) or {}
    jobs = data.get("jobs", None)

    if not isinstance(defaults, dict):
        raise ValueError(f"{filepath}: 'defaults' must be a mapping")
    if jobs is None:
        raise ValueError(f"{filepath}: no 'jobs' key found")
    if not isinstance(jobs, list) or len(jobs) == 0:
        raise ValueError(f"{filepath}: 'jobs' must be a non-empty list")

    resolved: list[dict] = []

    for i, job in enumerate(jobs):
        where = f"{filepath}: jobs[{i}]"
        if not isinstance(job, dict):
            raise ValueError(f"{where}: each job must be a mapping")

        name = job.get("name", f"job_{i+1}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}: 'name' must be a non-empty string if provided")
        name = name.strip()

        prefix = job.get("file-prefix", defaults.get("file-prefix", "")) or ""
        suffix = job.get("file-suffix", defaults.get("file-suffix", "")) or ""
        if not isinstance(prefix, str) or not isinstance(suffix, str):
            raise ValueError(f"{where}: 'file-prefix'/'file-suffix' must be strings")

        files_raw = _validate_files_list(job.get("files"), where)
        files = _apply_prefix_suffix(files_raw, prefix, suffix, where)

        entry: dict[str, Any] = {"name": name, "files": files}

        # Inherit dir/config unless overridden locally
        if "dir" in job:
            entry["dir"] = job["dir"]
        elif "dir" in defaults:
            entry["dir"] = defaults["dir"]

        if "config" in job:
            entry["config"] = job["config"]
        elif "config" in defaults:
            entry["config"] = defaults["config"]

        # output-dir: can be set in defaults or overridden per job; default to CWD
        if "output-dir" in job:
            entry["output-dir"] = job["output-dir"]
        elif "output-dir" in defaults:
            entry["output-dir"] = defaults["output-dir"]

        # Output is job-level only
        if "output" in job:
            entry["output"] = job["output"]

        # Collect config overrides from defaults, then job (job wins)
        cfg_overrides: dict[str, Any] = {}
        for k, v in defaults.items():
            if k not in _JOB_META and v is not None:
                cfg_overrides[k] = v
        for k, v in job.items():
            if k not in _JOB_META and v is not None:
                cfg_overrides[k] = v

        entry["config-overrides"] = cfg_overrides
        resolved.append(entry)

    return resolved


def build_cli_args(job: dict) -> tuple[list[str], Path | None, bool]:
    """Build CLI args for one job. Returns (args, cfg_path, cfg_is_temp)."""
    args: list[str] = []

    if "dir" in job:
        args.extend(["-d", str(job["dir"])])

    cfg_overrides = job.get("config-overrides", {}) or {}
    merged_cfg, cfg_is_temp = merge_config(
        job.get("config"),
        {k: v for k, v in cfg_overrides.items() if k != "output"},
    )
    if merged_cfg:
        args.extend(["--config", str(merged_cfg)])

    out_dir = Path(job.get("output-dir") or ".")
    base_output = job.get("output", f"{job['name']}.dat")
    output = out_dir / base_output
    args.extend(["--output", str(output)])

    # Always provide non-interactive plotting args;
    # include output-dir so figures land with the .dat file.
    plot_prefix = out_dir / job["name"]
    args.extend(["--no-show", "--plot-prefix", str(plot_prefix)])

    args.extend([str(x) for x in job["files"]])

    return args, merged_cfg, cfg_is_temp


def run_job(job: dict) -> tuple[bool, str]:
    """Run bin_average.py for one job. Returns (success, message)."""
    cli, cfg_path, cfg_is_temp = build_cli_args(job)
    cmd = [sys.executable or "python", str(BIN_AVERAGE), *cli]

    name = job["name"]
    print(f"\n{'='*60}")
    print(f"Job: {name}")
    if cfg_path:
        print(f"Config: {cfg_path}{' (temp merged)' if cfg_is_temp else ''}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return False, f"subprocess error: {e}"
    finally:
        if cfg_is_temp and cfg_path is not None:
            try:
                cfg_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: failed to remove temp config {cfg_path}: {e}", file=sys.stderr)

    if result.stdout.strip():
        lines = result.stdout.strip().splitlines()
        for line in lines[-5:]:
            print(f"  {line}")

    if result.returncode == 0:
        return True, "OK"

    err = (result.stderr or "").strip()
    msg = err.splitlines()[0] if err else f"exit code {result.returncode}"
    return False, msg


def main():
    ap = argparse.ArgumentParser(
        description="Batch wrapper for bin_average.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python bin/batch_average.py batch.yaml
  python bin/batch_average.py session1.yaml session2.yaml
""",
    )
    ap.add_argument("batch_files", nargs="+", help="YAML files specifying jobs")
    args = ap.parse_args()

    if not BIN_AVERAGE.exists():
        print(f"Error: {BIN_AVERAGE} not found.", file=sys.stderr)
        sys.exit(1)

    all_jobs: list[tuple[str, dict]] = []  # (batch filename, job)
    for fp in args.batch_files:
        p = Path(fp)
        if not p.exists():
            print(f"Error: {fp} not found.", file=sys.stderr)
            sys.exit(1)
        try:
            jobs = load_batch(fp)
        except Exception as e:
            print(f"Error loading {fp}: {e}", file=sys.stderr)
            sys.exit(1)
        all_jobs.extend((p.name, j) for j in jobs)

    if not all_jobs:
        print("No jobs found.", file=sys.stderr)
        sys.exit(1)

    results: list[tuple[str, bool, str]] = []
    for batch_file, job in all_jobs:
        label = f"{batch_file}: {job['name']}"
        ok, msg = run_job(job)
        results.append((label, ok, msg))

    n_total = len(results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = n_total - n_ok

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for label, ok, msg in results:
        status = "OK" if ok else f"FAILED ({msg})"
        print(f"  {label:<40} {status}")

    print(f"\n{n_ok}/{n_total} jobs succeeded.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()