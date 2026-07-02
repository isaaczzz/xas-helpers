# E0 detection algorithm

## What E0 is and why it is hard to find automatically

The absorption edge energy E0 is the energy at which a core electron is promoted into the continuum, producing the characteristic step-like rise in the absorption coefficient μ(E). By convention, E0 is defined as the **first inflection point** of that rising step — the energy where d²μ/dE² = 0 and dμ/dE is increasing towards its first local maximum from the low-energy side.

This specific definition has practical consequences:

- E0 sets the zero of the photoelectron wave vector k = 0 in EXAFS analysis. A systematic error of a few eV in E0 shifts the entire k-axis, distorting extracted bond lengths.
- E0 anchors the energy grid used during scan alignment and rebinning. If scans from the same sample have different apparent E0 values (due to small thermal or mechanical drifts in the monochromator), they must be aligned to a common reference before averaging or the edge will appear artificially broadened.
- E0 is the reference value against which energy calibration shifts are defined when using a reference standard.

Automated detection is complicated by several factors that are generic to measured XAS data:

**Noise.** Raw spectra contain shot noise, detector electronics noise, and systematic artefacts (e.g., glitches from diffraction contamination in the beam). These produce structure in the derivative dμ/dE that is indistinguishable from real features without prior smoothing.

**Pre-edge features.** Many transition-metal edges have genuine pre-edge peaks (1s→3d or 1s→4p transitions) that appear as weak bumps on the low-energy side of the main edge. Their derivatives peak below the main edge energy and can be mistaken for E0 without an appropriate threshold.

**Multiple inflection points on the main edge.** The rising edge of many materials is not a single clean step. For example, the Cu K-edge has a shoulder at the conventional E0 (~8979 eV) followed by a steeper rise peaking several eV higher. The global maximum of dμ/dE falls at the steeper feature, not at the first inflection — systematically overestimating E0 by 3–6 eV compared to the values reported by standard packages such as Athena and xraylarch.

**Wide scan ranges.** A single scan may span 1000–1600 eV from the pre-edge through the deep EXAFS. A smoothing width proportional to this total span (a common heuristic) produces a Gaussian with σ ≈ 5–8 eV on typical data. At that width, distinct features separated by only a few eV — precisely the scale of the pre-edge/main-edge structure described above — are merged into a single smooth hump, and the derivative maximum lands at the center-of-mass of the hump rather than at the first inflection.

**Finite step size.** The measurement grid is typically 0.2–0.5 eV in the XANES region. Without refinement, any E0 estimate is limited to about one step width of precision.

The algorithm described here is designed to address all of these simultaneously. It operates in two stages: a coarse estimate that locates the correct feature at grid-point precision, followed by a local polynomial refinement that sharpens that estimate below the grid spacing.

---

## Stage 1: Coarse E0 estimate

### Step 1a — Restrict to a central search window

Before any derivative analysis, the spectrum is cropped to a search window that excludes the outer 15% of the full energy range on each side:

```
search_min = E_min + 0.15 × (E_max − E_min)
search_max = E_max − 0.15 × (E_max − E_min)
```

Scan endpoints are typically the noisiest part of the spectrum: the scan may not yet have settled at the low end, and the far EXAFS at the high end often has poor signal-to-noise. The 15% exclusion eliminates these boundary effects without touching the edge or the nearby EXAFS that is scientifically relevant. For a 1600 eV scan (typical for a K-edge scan extending to k ≈ 13 Å⁻¹), this exclusion removes the outermost 240 eV on each side, leaving a 1120 eV central window that reliably contains the edge.

If the computed window is empty (which can happen only if the scan is extremely short), the fallback is the central 80% of data points by index.

### Step 1b — Calibrate the smoothing width

The Gaussian smoothing width σ is set to five times the **5th percentile of energy step sizes inside the search window**:

```
σ = max(p₅(ΔE_window) × 5,  0.3 eV)
```

where ΔE_window is the array of consecutive energy differences among points inside the search window, and the floor of 0.3 eV prevents degenerate behaviour on extremely coarse grids.

**Why the 5th percentile, not the median or the minimum?**

XAS scans almost always have variable step sizes. In the XANES region the step is typically 0.2–0.5 eV (set by the measurement protocol), but in the EXAFS region it grows rapidly because EXAFS grids are defined in k-space, where equal Δk steps translate to energy steps that grow as 2k·K_CONV·Δk. For a search window that extends well into the EXAFS, the median step size is dominated by these coarser EXAFS steps and would produce a σ far too large for resolving features near the edge. The 5th percentile robustly isolates the fine XANES resolution — the lower 5% of steps by size corresponds to the XANES region — even when the majority of the search window contains EXAFS data with larger steps. A simple minimum would be excessively sensitive to any duplicate or near-duplicate energy point.

**Why multiply by 5?**

The factor of 5 makes σ approximately equal to five raw step widths. In the XANES region this typically gives σ ≈ 1–2.5 eV. At this width:

- The Gaussian kernel spans ±4σ ≈ ±5–10 eV of data, averaging over 40–80 measurement points and suppressing point-to-point noise effectively.
- Features separated by ~5 eV (the typical scale of pre-edge/main-edge structure) remain resolved as distinct peaks in the derivative rather than being merged into one.

The ×5 rule is scale-invariant: it produces the same effective smoothing in units of data steps regardless of energy range or scan design.

**Why not scale σ with the total scan span?**

A formula such as `σ = 0.005 × (E_max − E_min)` gives σ ≈ 8 eV for a 1600 eV scan. This was the original approach and was found to cause systematic overestimation of E0 by 3–6 eV on the Cu K-edge test data. At 8 eV smoothing, the pre-edge shoulder at ~8979 eV and the main-edge peak at ~8984 eV are merged by the Gaussian, and the derivative maximum lands between them, closer to the steeper feature. The step-based formula gives σ ≈ 1.6 eV for the same data, preserving the two features as distinct peaks and placing the derivative maximum correctly at the first inflection.

### Step 1c — Gaussian smoothing

The smoothed spectrum μ̃(E) is computed as a Gaussian-kernel weighted average of the raw μ(E):

```
μ̃(Eᵢ) = Σⱼ wᵢⱼ · μ(Eⱼ)  /  Σⱼ wᵢⱼ

wᵢⱼ = exp(−½ · ((Eⱼ − Eᵢ) / σ)²)
```

The sum over j runs over a half-window of h = ⌊4σ / median(ΔE) + 0.5⌋ grid points on each side of i. The 4σ truncation retains all but exp(−8) ≈ 3×10⁻⁴ of the Gaussian weight, making the approximation error negligible. Points near the ends of the array use a narrower (asymmetric) window; the kernel normalisation by Σwᵢⱼ ensures the result is still a proper weighted average.

**Why Gaussian?**

The Gaussian kernel has a uniquely smooth Fourier transform — a Gaussian in the frequency domain — and produces no ringing or Gibbs-phenomenon artefacts. It is also the maximum-entropy noise-suppressing kernel for Gaussian noise: given a constraint on the total spectral energy removed, the Gaussian kernel removes the maximum Shannon entropy of noise. More practically, it has a single parameter (σ) with a clear physical interpretation (the standard deviation of the smoothing scale), whereas Savitzky–Golay filters require choosing both window width and polynomial degree, with less transparent interactions between them on non-uniform grids.

### Step 1d — Derivative and first-inflection selection

The derivative dμ̃/dE is estimated from the smoothed spectrum using central differences. Within the search window the derivative array is **normalised** to the range [0, 1]:

```
d̃(i) = [ dμ̃/dE(i)  −  min(dμ̃/dE) ]  /  [ max(dμ̃/dE)  −  min(dμ̃/dE) ]
```

E0 is then defined as the energy of the **first local maximum** of d̃ that exceeds a threshold of 0.60:

1. Scan the search window from the low-energy end.
2. At each index i, test: d̃(i) > 0.60 AND d̃(i) ≥ d̃(i−1) AND d̃(i) ≥ d̃(i+1).
3. Return the energy at the first index that satisfies all three conditions.
4. If no index qualifies at threshold 0.60, halve the threshold (to 0.30, then to 0.15) and repeat from step 1.
5. If no candidate is found at any threshold, return the energy at the global maximum of dμ̃/dE as a fallback.

**Why the first local maximum, not the global?**

The global maximum of dμ̃/dE falls at the steepest inflection point, which is the most prominent feature in the derivative but is not necessarily the first. On edges with a pre-edge shoulder (the conventional E0 location) followed by a steeper main rise, the global maximum is several eV above the first inflection — it corresponds to the steepest part of the edge jump rather than its onset. The first-inflection definition matches the convention used by Athena and xraylarch, and it is what makes E0 physically meaningful: it locates the binding energy of the core state being excited, not the peak of the density of unoccupied states above it.

**Why the 60% threshold?**

Normalisation maps the derivative to [0, 1] relative to the range within the search window. Genuine edge features rise to a large fraction of this range; noise bumps and slowly varying pre-edge backgrounds stay much lower. The 60% threshold selects only features that are clearly dominant within the window, ruling out:

- Small oscillatory artefacts in the pre-edge (typically <20% of the normalised derivative).
- Residual EXAFS oscillations entering from the high-energy end of the window (also typically low compared with the edge peak).

The adaptive fallback (0.30, then 0.15) ensures the algorithm degrades gracefully on spectra with very weak or diffuse edges where even the genuine feature may not reach 60%.

---

## Stage 2: Local polynomial refinement

The coarse estimate is accurate to approximately one energy step (typically 0.2–0.5 eV). The refinement uses a local polynomial model to locate the true inflection point analytically, achieving sub-step precision.

### Step 2a — Extract a local window

A window of [E0_coarse − 7, E0_coarse + 8] eV is extracted from the **raw (unsmoothed)** μ(E) data. The asymmetry (7 eV on the low-energy side, 8 eV on the high-energy side) gives slightly more room on the side where the inflection point's curvature is visible: the function transitions from concave-up (pre-edge baseline) through the inflection to concave-down (post-edge plateau) as energy increases. Having a little more room on the post-edge side ensures the polynomial sees enough of the concave-down region to identify the inflection reliably.

Critically, the **raw** spectrum is used here rather than the Gaussian-smoothed one. The polynomial fit performs its own implicit smoothing by fitting a smooth global curve; using a pre-smoothed input would smooth twice and reduce the information available for locating the inflection precisely.

If the window contains fewer than (fit_degree + 3) = 7 data points — which can happen if the coarse estimate is near a scan boundary — refinement is skipped and the coarse estimate is returned unchanged.

### Step 2b — Fit a degree-4 polynomial

The window data (x = E − E0_coarse, y = μ, shifted to reduce numerical condition number) is fitted by ordinary least squares with a degree-4 polynomial:

```
p(E) = a₄E⁴ + a₃E³ + a₂E² + a₁E + a₀
```

Degree 4 is the minimum even degree that can represent the qualitative shape of the rising edge with full generality. The edge has an inflection (change of sign of curvature) followed by the beginning of a concave-down plateau; representing this requires at least one inflection point and one local extremum in the first derivative. Degree 3 can represent one inflection but not an independent local extremum of the derivative. Degree 4 adds this freedom. Higher degrees are unnecessary and risk overfitting the noise.

The fit is computed with suppressed rank-deficiency warnings, since degree-4 fitting on a 15 eV window is occasionally mildly ill-conditioned.

### Step 2c — Assess fit quality via R²

Before using the polynomial, its goodness of fit is tested:

```
R² = 1 − SS_res / SS_tot

SS_res = Σ (μᵢ − p(Eᵢ))²
SS_tot = Σ (μᵢ − μ̄)²
```

If R² < 0.95, the raw spectrum in the fitting window is too noisy or irregular for the polynomial to represent it faithfully. This can happen at edges with sharp pre-edge peaks, unusual noise levels, or glitches. When the guard triggers, the refinement is abandoned and the coarse estimate is returned.

The 0.95 threshold corresponds to the polynomial explaining at least 95% of the variance in the window. In practice this is generous enough to pass on clean XAS data and restrictive enough to catch genuinely poor fits.

### Step 2d — Find inflection points via the second derivative

All inflection points of p(E) lie at zeros of its second derivative p''(E). Since p is degree 4, p'' is quadratic:

```
p''(E) = 12a₄E² + 6a₃E + 2a₂ = 0
```

The roots of this quadratic are found analytically via `numpy.roots`. Complex roots (which arise when the discriminant is negative, i.e., p'' has no real zeros) are discarded; only real roots that fall within the fitting window [E0_coarse − 7, E0_coarse + 8] are retained.

Among the remaining candidates, the root with the **largest value of p'(E)** — the largest first derivative — is selected. This identifies the steepest inflection point, which on a rising edge is the one closest to the true E0. The analytic computation is what gives the refinement its sub-step precision: the root is a real-valued number computed from polynomial coefficients, with no constraint to lie on the measurement grid.

### Step 2e — Shift guard

A final sanity check ensures the refined value is plausible:

```
|E0_refined − E0_coarse| ≤ 3.0 eV
```

If this is violated, the polynomial's best inflection is responding to something other than the coarse estimate's neighbourhood — perhaps a nearby pre-edge feature or a fitting artefact near the window boundary. The coarse estimate is returned unchanged.

The 3 eV limit was chosen to be wider than any expected precision improvement (the coarse estimate is typically accurate to 0.2–0.5 eV on good data) while being narrower than the typical separation between physically distinct features (pre-edge peaks, white lines) that could plausibly confuse the fit.

---

## Parameter summary

| Parameter | Value | Where used |
|-----------|-------|------------|
| Outer exclusion | 15% of span on each side | Search window boundaries |
| σ formula | max(p₅(ΔE_window) × 5, 0.3 eV) | Gaussian smoothing width |
| Gaussian half-window | ⌊4σ / median(ΔE) + 0.5⌋ | Number of neighbours in kernel |
| Normalised threshold | 0.60 (with fallback to 0.30, 0.15) | First-local-max selection |
| Refinement window | [E0_coarse − 7, E0_coarse + 8] eV | Local polynomial domain |
| Polynomial degree | 4 | Local fit order |
| R² guard | 0.95 | Minimum acceptable fit quality |
| Shift guard | 3.0 eV | Maximum allowed refinement displacement |

---

## Fallback behaviour

The algorithm is designed to degrade gracefully rather than produce a confident wrong answer:

- Fewer than 50 spectrum points: coarse-only estimate from unsmoothed derivative in the central 80%.
- No local maximum above any threshold in the search window: global maximum of the smoothed derivative.
- Fewer than 7 points in the refinement window: coarse estimate returned unchanged.
- R² < 0.95 in the refinement fit: coarse estimate returned unchanged.
- Shift guard violated: coarse estimate returned unchanged.

In all fallback cases the coarse estimate from Stage 1 is the output. Because Stage 1 is itself robust (step-calibrated smoothing, normalised threshold), the coarse estimate is already close to the correct value in all normal cases.
