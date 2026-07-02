# Rebinning and averaging scans

## Synopsis

Rebinning a spectrum onto a coarser energy grid makes XAS data faster to process by simply reducing the number of data points. Normally, the energy grid of the raw spectrum is oversampled, anyway, especially in the EXAFS region where the features are intrinsically very broad. In addition, if we need to average a set of scans together, we need to resample all of the scans onto a universal energy grid, anyway. The typical workflow looks like this:

1. Interpolate each scan onto the same coarsened energy grid.
2. Sum the interpolated scans.

Statistically, rebinning isn't straightforward, because using interpolation to resample the data correlates neighboring data points. In addition, interpolation is sensitive to noise and, as a result, can introduce systematic error into the rebinned data. The difficult-to-predict side effects of interpolation get dicey in the high-$k$ EXAFS region where the signal-to-noise ratio is intrinsically poor and fake oscillations can easily dominate the true signal. Yikes. Further, interpolation algorithms require the raw dataset to be monotonically increasing and will fail if there are duplicate points at the same energy value.

The approach in `bin_average` avoids interpolation entirely by taking advantage of the desired coarsening of the energy grid that happens during rebinning:

1. Aggregate all datapoints from all scans into a single vector, sorted by increasing energy.
2. Group datapoints into energy bins according to the desired grid.
3. 


## 1) Parsing raw scans into μ(E) curves

For each input file (currently SPring-8/Aichi/SAGA-style text files):

- Read measured quantities: detector angle(s), time, I0, and signal channels.
- Convert angles and the crystal’s d-spacing to absolute energy E using Bragg’s law:
  - E = hc / (2d sin θ)
  - This is how diffractometers define energy; small differences in θ or d produce small energy shifts between scans.
- Build a per-scan μ(E):
  - Transmission mode: μ ≈ ln(I0 / I1).
  - Fluorescence mode: μ ∝ (fluorescent signal) / I0.

At this point each scan is just an (E, μ(E)) array with metadata.

## 2) Quality control and per-scan E0 estimation

Before combining scans, we estimate the absorption edge position E0 for each one, then use simple statistical checks to flag suspicious files.

- Estimating E0:
  - Smooth μ(E) using a Savitzky–Golay filter (local polynomial fit).
  - Compute dμ/dE; define E0 as the energy where this derivative is maximal.
  - Exclude the outer ~5% of each scan to avoid spurious maxima at edges.
- Flagging suspicious scans:
  - TRUNCATED_RANGE: energy span unusually short compared with other scans.
  - LOW_SIGNAL_SHUTTER_OR_GAIN: overall signal unusually low (e.g., shutter closed, bad gain).
  - I0_GAIN_OUTLIER: median I0 is far from the group’s typical value.

Scans flagged as clearly bad can be excluded automatically so they don’t distort later averages. This step prevents obviously defective files from silently corrupting results.

## 3) Energy alignment (correcting small drifts between scans)

Even when using the same instrument, each scan’s energy scale may be slightly shifted due to mechanical or thermal changes. If we average them “as is,” these shifts smear the edge and distort structure.

The script:

- Chooses a target E0 based on the alignment mode:
  - mean: use the average of all (included) scans’ E0 estimates.
  - ref: use one selected scan as reference.
  - value: use an explicitly specified energy value.
- For each included scan, shifts its entire energy axis so that:
  - Its estimated E0 aligns exactly with this target E0.

This is a rigid shift (no stretching). The result is a set of scans whose edges line up before averaging, reducing smearing and systematic misalignment between files.

## 4) Building a physically motivated energy grid

Instead of using whatever raw energies appear in the data, we define one global energy grid for all scans. The grid is specified as a sorted list of **bin edges** (not centers). Each interval between two consecutive edges defines one bin. The final output will have one data point per bin.

The edge list is constructed in three regions around E0, then concatenated and deduplicated:

### Pre-edge region (coarse bins)

Starting from `pre_start` (default −200 eV relative to E0), edges are placed at uniform intervals of `de_pre` (default 2.0 eV) up to `pre_end` (default −30 eV):

```
edges = [E0 + pre_start, E0 + pre_start + de_pre, ..., E0 + pre_end]
```

This produces relatively wide bins that capture the baseline and any weak pre-peak structure without overfitting noise.

### XANES region (fine uniform bins)

From `pre_end` to `xanes_end` (default +50 eV), edges are placed at finer intervals of `de_xanes` (default 0.2 eV):

```
edges = [E0 + pre_end, E0 + pre_end + de_xanes, ..., E0 + xanes_end]
```

This resolves the main features near the edge where chemistry-sensitive structure lives. With a default step of 0.2 eV, each bin is narrow enough to capture sharp XANES features while still averaging over multiple raw data points.

### EXAFS region (k-space-derived bins)

Above `xanes_end`, bins are no longer uniformly spaced in energy. Instead, we define a quasi-uniform grid in k-space and convert it back to energy:

```
E = E0 + K_CONV × k²    (where K_CONV ≈ 3.81 eV·Å⁻²)
```

Starting from `k0` (the k-value corresponding to `xanes_end`), we step by `dk` (default 0.05 Å⁻¹) up to `kmax` (default 14.0 Å⁻¹). Each k-step is converted to an energy edge via the relation above.

Because E ∝ k², equal steps in k produce increasingly wider bins in energy at higher k. This matches how EXAFS oscillations behave and ensures the output density of points reflects the physics rather than being artificially uniform.

### Merging into a single edge list

The three sets of edges are concatenated, duplicates removed, and sorted. The result is a monotonically increasing array `edges[0], edges[1], ..., edges[M]` defining `M-1` bins. This same edge list is used for every scan.

Why this matters:

- Adequate resolution where it’s important (XANES), coarser bins elsewhere.
- A physically sensible density of points for EXAFS analysis.
- One standard grid across all scans, making comparisons straightforward.

## 5) Coarsening overly fine bins (done before binning)

Before assigning data to bins, we check whether the proposed grid claims more resolution than the raw data can support. A grid that is much finer than the actual measurement step produces artificial “wiggles” — noise re-packaged at higher resolution.

### Estimating the effective minimum energy step

For each usable scan, we collect all consecutive energy differences ΔE between adjacent raw data points. From this pooled set of steps, we take the **5th percentile** as an effective minimum step `eff_de`. The low percentile (rather than the absolute minimum) makes this robust against outlier steps from, e.g., slow-scan regions or instrument quirks.

If any configured bin spacing (`de_pre` or `de_xanes`) is finer than 95% of `eff_de`, a warning is printed.

### Merging narrow bins

We then do a single pass through the proposed edge list. Starting from the first edge, we keep each subsequent edge only if it would create a bin at least `eff_de` wide (measured from the last kept edge). Edges that are too close to the previous one are simply skipped — effectively merging their intervals into the neighboring bin:

```
kept_edges = [edges[0]]
for i in 1 .. M-1:
    if edges[i] - kept_edges[-1] >= eff_de:
        keep edges[i]
    else:
        skip edges[i]  (merge into current bin)
```

This coarsening is applied per-region, so the pre-edge, XANES, and EXAFS regions can each be independently controlled. The user can also override it with `--no-coarse-pre-edge`, `--no-coarse-xanes`, or `--no-coarse-exafs` flags to force finer bins in a specific region (with a warning).

The result is a final edge list where no bin is unrealistically narrow compared to the actual data resolution. Regions that were coarsened are recorded so they can be annotated on plots.

## 6) Assigning raw data points to bins

With the final edge list in hand, each scan’s raw (E, μ(E)) data is mapped onto it:

### Point-to-bin assignment

For each raw data point, we determine which bin it falls into using `np.digitize`:

```
bin_index = digitize(raw_energy, edges) - 1
```

This assigns each point to exactly one bin (or −1 if below the first edge, or out-of-range if above the last). Points outside all bins are silently discarded.

### Computing per-bin values for a single scan

For each of the `M-1` bins:

1. Collect all raw points whose `bin_index` equals that bin’s index.
2. If at least one point falls in the bin, compute:
   - **binned μ** = mean of μ for all points in this bin.
   - **binned E** = mean energy of those same points (this becomes the data point’s representative energy).
3. If no points fall in the bin (e.g., a scan doesn’t cover that region), the bin is left as NaN.

After this step, each scan is represented as three arrays of length `M-1`: one for binned E, one for binned μ, and one for per-bin uncertainty σ, with NaN where data was absent. This process smooths out tiny fluctuations within each bin (since multiple raw points are averaged) while preserving the overall shape.

### Why per-bin averaging matters

Raw scans typically have hundreds or thousands of closely spaced points. Many bins will contain dozens of raw measurements. Averaging them into a single value:

- Reduces high-frequency noise naturally (like any mean).
- Ensures that each output point represents a real chunk of measured data, not an interpolated guess.
- Means the final grid resolution is determined by bin width and raw sampling density together — you can’t get more information out than what was measured.

## 7) Averaging across scans and estimating uncertainty

Once every scan has been binned onto the same grid, we have a matrix of shape `(num_scans × num_bins)` where each cell is one scan’s μ value for one bin (or NaN if that scan didn’t cover it). We now average column-wise:

### Computing per-bin measurement precision

Before averaging across scans, we evaluate how well-constrained each bin is **within each individual scan**. For each scan *i* in each bin *j*:

- If the bin contains ≥2 raw points: compute σ[i,j] = std(μ_points) / √n (standard error of the mean).
- If the bin contains only 1 point: set σ[i,j] = inf (no internal scatter estimate — handled by fallback weighting below).
- If the bin is empty: σ[i,j] = NaN.

### Inverse-variance weighted averaging with fallbacks

Scans are **not** averaged equally. Instead, each scan’s contribution to a bin is weighted by how precisely it measured that region — scans with tightly clustered points in a bin get more weight than noisy ones.

For each bin `j`, we check which contributing scans have a finite σ estimate:

- **All scans have σ**: standard inverse-variance weighting (w = 1/σ²), followed by weighted mean and error calculation.
- **No scans have σ** (all single-point bins): equal-weight fallback — all scans are averaged with identical weight, and σ_mean is computed as std/√N. This prevents data loss in fine-binned regions like XANES where every scan may only contribute one point per bin.
- **Some scans have σ, some don’t**: a warning is printed once, and the missing σ values are set to the largest known σ in that bin (conservative: gives them the smallest weight among weighted scans). Then inverse-variance weighting proceeds normally.

For bins using inverse-variance weighting:
1. Compute weights: **w[i] = 1 / σ²[i,j]**.
2. Weighted mean absorbance: **μ_avg[j] = Σ(w·μ) / Σw**.
3. Weighted representative energy: **E_avg[j] = Σ(w·b_e) / Σw** (ensures the reported E is consistent with the weighted μ).
4. Weighted standard error: **σ_mean[j] = 1 / √Σw** — how well-constrained this point is from all contributing scans combined.

This means a noisy scan in a particular region naturally contributes less to the final average, without requiring manual exclusion of entire files. Fine-binned regions (like XANES) still produce smooth output even when individual bins contain only one raw point per scan.

### Smoothing the uncertainty

The raw σ_mean can have wild local spikes (e.g., one scan has an outlier in a single bin). To avoid this, we apply a **median filter with window size 11** to the σ_mean array. This preserves overall shape while removing isolated spikes, and handles NaN values by interpolating over gaps before filtering.

### Filtering out unreliable bins

Only bins where at least two scans contribute (`N_eff ≥ 2`) are kept in the final output. Bins covered by only a single scan (or none) are dropped entirely — there’s no point reporting an “average” from one measurement, and no meaningful uncertainty to compute either.

### What you get as output

The final output is four arrays:

| Array | Meaning |
|-------|---------|
| E_avg | Representative energy for each bin (averaged across scans) |
| μ_avg | Averaged absorbance — the main spectrum |
| σ_mean | Uncertainty estimate (standard error of the mean) |
| N_eff | Number of contributing scans per bin |

Why this matters:

- You don’t just get a smooth curve; you also get per-point uncertainties reflecting how consistent the data are across scans.
- The inverse-variance weighting means noisy scans contribute less than clean ones, without requiring manual exclusion.
- Regions where many scans agree will have small error bars; regions with disagreement or fewer contributing scans will show larger uncertainty instead of looking artificially clean.
- The N_eff column lets you see at a glance which energy ranges had full coverage and which were only partially sampled.

## Short summary (what problems this solves)

This rebinning approach — seven stages from raw file to averaged spectrum — is designed to:

- Correct small energy drifts between scans, so features line up instead of smearing (§3).
- Automatically detect and exclude clearly bad scans before they distort results (§2).
- Provide a single, physically motivated energy grid across all scans for direct comparison (§4).
- Avoid false precision by refusing to create bins finer than what your data can justify (§5).
- Map raw points into bins honestly, so output values reflect actual measurements not interpolation (§6).
- Weight each scan's contribution by its measurement precision (inverse-variance), so noisy scans contribute less without manual exclusion (§7).
- Produce realistic error bars showing where the average is well-constrained vs uncertain, and drop bins that only have a single contributing scan (§7).
