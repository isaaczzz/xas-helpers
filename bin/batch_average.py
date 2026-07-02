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
      grid:                          # overrides merged into config.yaml
        kmax: 18.0

    jobs:
      - name: sample_A
        files: ["_001", "_002"]       # becomes scan_001.dat, scan_002.dat
        dir: /data/session1/sampleA   # overrides defaults.dir
        grid:                         # job-level override (deep-merged)
          kmax: 14.0

      - name: sample_B                # inherits all defaults
        files: ["_003", "_004"]

YAML bare mode (plain list = single implicit job):
    - scan001.dat
    - scan002.dat
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _ensure_yaml():
    """Lazy-load yaml; exits with message if missing."""
    global safe_load, dump
    if safe_load is not None or dump is not None:
        return
    try:
        from yaml import safe_load as sl, dump as dp
        safe_load = sl
        dump = dp
    except ImportError:
        print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)


# Forward declarations for lazy loading
safe_load = None  # type: ignore
dump = None       # type: ignore


def deep_update(base: dict, override: dict) -> dict:
    """Recursively update base with values from override."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def merge_config(config_path: str | None, overrides: dict) -> Path | None:
    """Deep-merge overrides into config file contents; return temp YAML path or None."""
    if not overrides:
        return Path(config_path) if config_path else None

    _ensure_yaml()
    base = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            base = safe_load(f) or {}

    merged = deep_update(base, overrides)
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
    dump(merged, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return Path(tmp.name)


SCRIPT_DIR = Path(__file__).resolve().parent
BIN_AVERAGE = SCRIPT_DIR / "bin_average.py"

# Keys recognized as job metadata (not config overrides)
_JOB_META = {"name", "files", "dir", "config", "file-prefix", "file-suffix", "output"}


def load_batch(filepath: str) -> list[dict]:
    """Load a YAML batch spec and return a list of job dicts."""
    _ensure_yaml()
    with open(filepath, "r") as f:
        data = safe_load(f)

    if data is None:
        raise ValueError(f"{filepath}: empty file")

    # Bare mode: plain list of strings -> single implicit job (no prefix/suffix)
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

        entry: dict = {"name": name, "files": files}

        # Inherit dir and config from defaults unless overridden locally
        if "dir" in job:
            entry["dir"] = job["dir"]
        elif "dir" in defaults:
            entry["dir"] = defaults["dir"]

        if "config" in job:
            entry["config"] = job["config"]
        elif "config" in defaults:
            entry["config"] = defaults["config"]

        # Output (job-level only; not inherited from defaults)
        if "output" in job:
            entry["output"] = job["output"]

        # Collect config overrides from both defaults and job.
        # Anything that is not a known job metadata key is treated as a
        # setting to deep-merge into the config file (grid, qc, mode, align, etc.).
        cfg_overrides: dict = {}
        for k in list(defaults.keys()):
            if k not in _JOB_META and defaults[k] is not None:
                cfg_overrides[k] = defaults[k]
        for k in job:
            if k not in _JOB_META and job[k] is not None:
                cfg_overrides[k] = job[k]

        entry["config-overrides"] = cfg_overrides
        resolved.append(entry)

    return resolved


def build_cli_args(job: dict) -> tuple[list[str], Path | None]:
    """Build CLI argument list for one job. Returns (args, tmp_config_path)."""
    args = []

    # Dir
    if "dir" in job:
        args.extend(["-d", str(job["dir"])])

    # Merge config overrides into a temp config file
    cfg_overrides = job.get("config-overrides", {})
    merged_cfg = merge_config(
        job.get("config"),
        {k: v for k, v in cfg_overrides.items() if k != "output"},
    )
    if merged_cfg:
        args.extend(["--config", str(merged_cfg)])

    # Output (default to <name>.dat)
    output = job.get("output", f"{job['name']}.dat")
    args.extend(["--output", output])

    # Plotting: always pass --no-show and set plot-prefix to job name.
    # Whether plotting actually runs is controlled by the config file's
    # plot.enabled block; this just ensures a save_prefix is available.
    args.extend(["--no-show", "--plot-prefix", job["name"]])

    # Positional files
    args.extend(job["files"])

    return args, merged_cfg


def run_job(job: dict) -> tuple[bool, str]:
    """Run bin_average.py for one job. Returns (success, message)."""
    cli, tmp_cfg = build_cli_args(job)
    cmd = [sys.executable or "python", str(BIN_AVERAGE), *cli]

    name = job["name"]
    print(f"\n{'='*60}")
    print(f"Job: {name}")
    if tmp_cfg:
        print(f"Config: {tmp_cfg}")
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
