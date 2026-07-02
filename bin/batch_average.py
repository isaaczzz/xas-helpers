#!/usr/bin/env python3
"""Batch wrapper for bin_average.py.

Runs multiple bin_average jobs from a YAML specification file. Each job is
executed as a separate subprocess call to bin_average.py.

Usage:
    python bin/batch_average.py batch1.yaml [batch2.yaml ...]

YAML format (with defaults):
    defaults:
      dir: /data/session1
      config: base_config.yaml
      file-prefix: "scan"
      file-suffix: ".dat"

    jobs:
      - name: sample_A
        files: ["_001", "_002"]       # becomes scan_001.dat, scan_002.dat
        dir: /data/session1/sampleA   # overrides defaults.dir

      - name: sample_B                # inherits all defaults
        files: ["_003", "_004"]

YAML bare mode (plain list = single implicit job):
    - scan001.dat
    - scan002.dat
"""

import argparse
import subprocess
import sys
from pathlib import Path

_yaml = None


def _ensure_yaml():
    global _yaml
    if _yaml is not None:
        return
    try:
        import yaml as _yaml
    except ImportError:
        print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_AVERAGE = SCRIPT_DIR / "bin_average.py"


def load_batch(filepath: str) -> list[dict]:
    """Load a YAML batch spec and return a list of job dicts."""
    _ensure_yaml()
    with open(filepath, "r") as f:
        data = _yaml.safe_load(f)

    if data is None:
        raise ValueError(f"{filepath}: empty file")

    # Bare mode: plain list of strings → single implicit job (no prefix/suffix)
    if isinstance(data, list):
        return [{"name": Path(filepath).stem, "files": data}]

    if not isinstance(data, dict):
        raise ValueError(f"{filepath}: expected a mapping or list at top level")

    defaults = data.get("defaults", {}) or {}
    jobs = data.get("jobs", [])
    if not jobs:
        raise ValueError(f"{filepath}: no 'jobs' key found")

    resolved = []
    for i, job in enumerate(jobs):
        name = job.get("name", f"job_{i+1}")
        prefix = job.get("file-prefix", defaults.get("file-prefix", "")) or ""
        suffix = job.get("file-suffix", defaults.get("file-suffix", "")) or ""

        files = [f"{prefix}{f}{suffix}" for f in job["files"]]

        entry = {"name": name, "files": files}

        # Inherit dir and config from defaults unless overridden locally
        if "dir" in job:
            entry["dir"] = job["dir"]
        elif "dir" in defaults:
            entry["dir"] = defaults["dir"]

        if "config" in job:
            entry["config"] = job["config"]
        elif "config" in defaults:
            entry["config"] = defaults["config"]

    # Optional overrides
        if "output" in job:
            entry["output"] = job["output"]

        # Boolean flags: pass through if truthy
        for flag, cli_name in [
            ("shift-minima", "--shift-minima"),
            ("rebin-scans", "--rebin-scans"),
        ]:
            if job.get(flag):
                entry.setdefault("flags", []).append(cli_name)

        # Grid overrides (passed as-is to CLI)
        grid_keys = ["pre-start", "pre-end", "de-pre", "xanes-end", "de-xanes", "kmax", "dk"]
        for gk in grid_keys:
            if gk in job:
                entry.setdefault("grid", {})[gk] = job[gk]

        # Other CLI overrides
        for key in ("mode", "align"):
            if key in job:
                entry[key] = job[key]

        resolved.append(entry)

    return resolved


def build_cli_args(job: dict) -> list[str]:
    """Build CLI argument list for one job."""
    args = []

    # Dir
    if "dir" in job:
        args.extend(["-d", str(job["dir"])])

    # Config
    if "config" in job:
        args.extend(["--config", str(job["config"])])

    # Mode / alignment overrides
    for key, flag in [("mode", "--mode"), ("align", "--align")]:
        if key in job:
            args.extend([flag, str(job[key])])

    # Grid overrides
    grid = job.get("grid", {})
    for key, value in grid.items():
        args.extend([f"--{key}", str(value)])

    # Boolean flags
    if "flags" in job:
        args.extend(job["flags"])

    # Output (default to <name>.dat)
    output = job.get("output", f"{job['name']}.dat")
    args.extend(["--output", output])

    # Plotting: always pass --no-show and set plot-prefix to job name.
    # Whether plotting actually runs is controlled by the config file's
    # plot.enabled block; this just ensures a save_prefix is available.
    args.extend(["--no-show", "--plot-prefix", job["name"]])

    # Positional files
    args.extend(job["files"])

    return args


def run_job(job: dict) -> tuple[bool, str]:
    """Run bin_average.py for one job. Returns (success, message)."""
    cli = build_cli_args(job)
    cmd = [sys.executable or "python", str(BIN_AVERAGE), *cli]

    name = job["name"]
    print(f"\n{'='*60}")
    print(f"Job: {name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return False, f"subprocess error: {e}"

    if result.stdout.strip():
        # Print last few lines of stdout for progress
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

    all_jobs: list[tuple[str, dict]] = []  # (batch_file, job)
    for fp in args.batch_files:
        if not Path(fp).exists():
            print(f"Error: {fp} not found.", file=sys.stderr)
            sys.exit(1)
        try:
            jobs = load_batch(fp)
        except Exception as e:
            print(f"Error loading {fp}: {e}", file=sys.stderr)
            sys.exit(1)
        all_jobs.extend((Path(fp).name, j) for j in jobs)

    if not all_jobs:
        print("No jobs found.", file=sys.stderr)
        sys.exit(1)

    results: list[tuple[str, bool, str]] = []  # (label, ok, msg)
    for batch_file, job in all_jobs:
        label = f"{batch_file}: {job['name']}"
        ok, msg = run_job(job)
        results.append((label, ok, msg))

    # Summary
    n_total = len(results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = n_total - n_ok

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for label, ok, msg in results:
        status = "OK" if ok else f"FAILED ({msg})"
        print(f"  {label:<40} {status}")

    print(f"\n{ n_ok}/{n_total} jobs succeeded.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
