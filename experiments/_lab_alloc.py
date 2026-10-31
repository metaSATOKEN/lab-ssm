# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
LAB-D shared Lyapunov-aware bit-allocation helpers (Realisations A and B).

Source-of-truth: lab-ssm paper.md SS 3.5 (Two Integer Realisations of the R_g Surface)
and the canonical reference implementation in lab-ssm/experiments/phase_c16_matched_mean_sweep.py
(lines 305 to 365). This module lifts the four profile builders (uniform / full_slope /
half_slope / half_slope_rg) into a shared utility so every LAB-D S-substrate script
(phase_d2_ssm_*, phase_d3_*, phase_d5_policy_*, phase_d7_3_jamba_kse,
phase_d10a_jamba_e_int5, phase_d18a/b, phase_d22a_jamba_hero_ci) can import a single
canonical version. The shared module prevents the Realisation-A-label vs Realisation-B-
numbers disconnect that was discovered on Day 15 and audited cross-paper through the
Pattern X-S-T-W-U integrity protocol.

Realisations (lab-ssm paper.md SS 3.5):

  Realisation A (recommended, `half_slope_rg`)
    b_g = b_0 + ceil( (1/2) log_2 R_g + log_2 r_g - mean(log_2 r_g) )_+
    with R_g = Gamma_g / (1 - rho_g), r_g = k * RMS(h_g), shift normalisation,
    b_min = 2 floor, 16-bit ceiling. This is the closed-form rounded integer analogue
    of the continuous bit-budget optimum b_g ~ (1/2) log_2 R_g + log_2 r_g + const
    derived in lab-ssm SS 3.4.

  Realisation B (slope-1 simplification, `full_slope`)
    b_g = b_0 + ceil( log_2 R_g )_+
    Drops the log_2 r_g term and doubles the slope on log_2 R_g. Used in lab-ssm
    SS 4 to SS 7 scale-side experiments. Retained here for ablation and for
    reproducing the lab-ssm SS 5 transfer chain results.

  Realisation half_slope (intermediate)
    b_g = b_0 + ceil( (1/2) log_2 R_g )_+
    Slope-(1/2) without the r_g term. lab-ssm SS 6.2 reports this as approximately
    a wash against `full_slope` at matched mean bit width; the dominant structural
    factor is the log_2 r_g term in Realisation A. Retained here for completeness
    of the four-profile sweep.

Shape conventions (caller responsibility):
  - Mamba-1 path (Jamba via slow_forward; lab-ssm phase_c8d):
      rho is per-channel, gamma is per-state. Caller computes
      log_R = log_2(gamma.unsqueeze(0) / (1 - rho).clamp(min=EPS))
      and log_rg = log_2(RANGE_K * rms_h.clamp(min=EPS)) with matching shape.
  - Mamba-2 SSD path (Zamba2 via torch_forward; lab-ssm phase_c5b):
      rho is per-head, gamma is per-(head, state). Caller computes
      log_R = log_2(gamma / (1 - rho).unsqueeze(-1))
      and log_rg correspondingly.
  These helpers operate on already-broadcast log_R / log_rg tensors so the
  caller controls its own shape conventions and this module remains shape-
  agnostic.

Constants (lab-ssm paper.md SS 3.4 / SS 4.6):
  RANGE_K = 7.0  ; k = 7 saturates the clipping-versus-quantisation-noise
                   trade-off at b_0 >= 5 (calibrated on Mamba-2.8B-hf in C-4)
  B_MIN   = 2    ; smallest groups held at INT2 to keep the standard sigma_q
                   propto 2^(-b) noise analysis applicable (lab-ssm SS 3.5)
  B_MAX   = 16   ; deployment-side upper bound
  EPS     = 1e-6 ; clamp floor for numerical stability across log_2 / division
"""

from __future__ import annotations

import torch

# ----- Module-level constants (lab-ssm paper.md SS 3.4 / SS 4.6) -----
RANGE_K = 7.0
EPS = 1e-6
B_MIN = 2
B_MAX = 16


# =============================================================================
# Global mean (for Realisation A shift normalisation)
# =============================================================================

def compute_global_mean_log_rg(
    captured: dict,
    rms_key: str = "rms_h",
    range_k: float = RANGE_K,
    eps: float = EPS,
) -> float:
    """Compute the global mean of log_2(k * RMS(h_g)) for Realisation A shift
    normalisation. The mean is over all per-group RMS values across the captured
    SSM modules.

    The Realisation A formula b_g - b_0 = ceil( (1/2) log_2 R_g + log_2 r_g
    - mean(log_2 r_g) )_+ requires the global mean to be subtracted from each
    per-group log_2 r_g term; this function computes that mean once per profile
    construction.

    Args:
        captured: dict mapping module-name to per-module calibrated quantities.
            Each entry must contain RMS(h_g) under the key `rms_key`.
        rms_key: dict key holding RMS(h_g). Default 'rms_h' matches the lab-ssm
            phase_c5b / phase_c8d / phase_c16 / phase_d2_ssm_* conventions.
        range_k: per-group range multiplier (lab-ssm SS 3.4); default k = 7.
        eps: clamp floor for numerical stability inside log_2.

    Returns:
        Python float; the global mean over all groups, suitable as the
        `mean_log_rg` argument to `extras_half_slope_rg`.
    """
    rms_all = []
    for d in captured.values():
        rms_all.append(d[rms_key].flatten())
    rms_cat = torch.cat(rms_all)
    return float(torch.log2((range_k * rms_cat).clamp(min=eps)).mean())


# =============================================================================
# Profile builders (one per Realisation)
# =============================================================================

def extras_uniform(reference_shape_tensor: torch.Tensor) -> torch.Tensor:
    """Control profile: b_g - b_0 = 0 for all groups (uniform allocation).

    Args:
        reference_shape_tensor: any per-module tensor of the target shape
            (typically log_R or rho); the returned zero-tensor matches its shape
            and device.

    Returns:
        Zero-tensor matching the reference tensor's shape, dtype, and device.
    """
    return torch.zeros_like(reference_shape_tensor)


def extras_full_slope(log_R: torch.Tensor) -> torch.Tensor:
    """Realisation B (`full_slope`) extras: b_g - b_0 = ceil( log_2 R_g )_+

    This is the slope-1 simplification of lab-ssm SS 3.5 (eq. between line 270
    and 280 of paper.md), used in the scale-side experiments of SS 4 to SS 7.
    Retained here for ablation; the recommended LAB-D state-cache memory recipe
    is Realisation A (`extras_half_slope_rg`).

    Args:
        log_R: log_2 R_g = log_2(Gamma_g / (1 - rho_g)). Caller is responsible
            for broadcasting (per-channel rho with per-state gamma for Mamba-1;
            per-head rho with per-(head, state) gamma for Mamba-2 SSD).

    Returns:
        Per-group extras tensor (ceil, clamped to >= 0).
    """
    return torch.ceil(log_R).clamp(min=0)


def extras_half_slope(log_R: torch.Tensor) -> torch.Tensor:
    """Intermediate profile (`half_slope`) extras: b_g - b_0 = ceil( (1/2) log_2 R_g )_+

    The strict slope-(1/2) form without the per-group log_2 r_g correction.
    lab-ssm SS 6.2 reports this as approximately a wash against `full_slope` at
    matched mean bit width on the Mamba-1 five-size head-to-head; the dominant
    structural factor relative to Realisation A is the log_2 r_g term that this
    profile drops. Retained here for the four-profile sweep.

    Args:
        log_R: log_2 R_g, caller-broadcast.

    Returns:
        Per-group extras tensor (ceil, clamped to >= 0).
    """
    return torch.ceil(0.5 * log_R).clamp(min=0)


def extras_half_slope_rg(
    log_R: torch.Tensor,
    log_rg: torch.Tensor,
    mean_log_rg: float,
) -> torch.Tensor:
    """Realisation A (`half_slope_rg`) extras:
    b_g - b_0 = ceil( (1/2) log_2 R_g + log_2 r_g - mean(log_2 r_g) )_+

    This is the recommended state-cache memory recipe of lab-ssm (paper.md
    SS 3.5, SS 6.2). It is the closed-form rounded integer analogue of the
    continuous bit-budget optimum b_g ~ (1/2) log_2 R_g + log_2 r_g + const
    derived in SS 3.4 from Lagrange optimisation of the per-group noise variance
    sum_g R_g * r_g^2 * 4^(-b_g) subject to a fixed total bit budget.

    The matched-mean head-to-head in lab-ssm SS 6.2 establishes that this
    Realisation wins on every (model, b_0) cell of the five-size state-spaces
    Mamba-1 family, with the b_0 = 4 advantage being structurally large
    (Realisation A: +5.9% to +19.3% ΔPPL vs Realisation B: +22.0% to +59.0% on
    the same five sizes at INT4 effective budget). The factorisation experiment
    in SS 6.3 attributes the dominant structural advantage to the log_2 r_g term;
    R_g is the structural backbone of the recipe.

    Args:
        log_R: log_2 R_g = log_2(Gamma_g / (1 - rho_g)). Caller-broadcast.
        log_rg: log_2(k * RMS(h_g)), same shape as log_R (caller-broadcast so
            that the per-(channel, state) or per-(head, state) elements align).
        mean_log_rg: scalar global mean of log_2(k * RMS(h_g)), computed once
            via `compute_global_mean_log_rg` over the full captured dict.

    Returns:
        Per-group extras tensor (ceil, clamped to >= 0).
    """
    raw = 0.5 * log_R + log_rg - mean_log_rg
    return torch.ceil(raw).clamp(min=0)


# =============================================================================
# Shared post-processing (shift normalisation + integer clamp)
# =============================================================================

def build_b_per_unit_shift_norm(
    extras_dict: dict,
    b_0: float,
    b_min: int = B_MIN,
    b_max: int = B_MAX,
) -> dict:
    """Add base b_0, apply additive shift to enforce mean bit width = b_0,
    clamp to [b_min, b_max], and round to integer. This is the shared post-
    processing step for all four profiles (uniform / full_slope / half_slope /
    half_slope_rg), per lab-ssm paper.md SS 3.5 (eq. b_g = round(b_0 + extras_g
    - mean(extras))|_[b_min, 16]).

    The shift normalisation ensures that any two profiles using the same b_0
    deploy at the same mean bit width, isolating the per-group allocation
    pattern from the absolute memory budget. This is what makes the matched-
    mean head-to-head of lab-ssm SS 6.2 a fair comparison: every profile's
    mean(b_g) is forced to b_0 to within rounding.

    Args:
        extras_dict: per-module raw extras tensors (the ceil(...) output of any
            of the four profile builders).
        b_0: target mean bit width.
        b_min: minimum bit width per group; default 2 (lab-ssm SS 3.5 justifies
            INT2 floor for keeping standard noise analysis applicable).
        b_max: maximum bit width per group; default 16 (FP16 ceiling).

    Returns:
        Per-module dict of integer bit-width tensors (long dtype), with mean
        approximately equal to b_0 (subject to ceil and clamp).
    """
    raw_b = {n: b_0 + e for n, e in extras_dict.items()}
    sum_b = sum(float(b.sum()) for b in raw_b.values())
    n_total = sum(b.numel() for b in raw_b.values())
    mean_b = sum_b / n_total
    shift = b_0 - mean_b
    out = {}
    for n, b in raw_b.items():
        b_shift = b + shift
        b_int = torch.round(b_shift).long().clamp(min=b_min, max=b_max)
        out[n] = b_int
    return out


# =============================================================================
# Per-layer quantiser range r_g = k * RMS(h_g)
# =============================================================================

def build_r_per_unit(
    captured: dict,
    rms_key: str = "rms_h",
    range_k: float = RANGE_K,
) -> dict:
    """Build the per-unit quantiser range r_g = k * RMS(h_g). This range is
    used by the mid-tread quantiser (lab-ssm SS 3.5) regardless of Realisation
    A or B: only the bit-width assignment b_g differs between Realisations,
    while the per-group range r_g is shared.

    Args:
        captured: dict mapping module-name to per-module calibrated quantities.
        rms_key: dict key holding RMS(h_g). Default 'rms_h'.
        range_k: per-group range multiplier; default k = 7.

    Returns:
        Per-module dict of r_g tensors (same shape as the corresponding rms_h).
    """
    return {n: range_k * d[rms_key] for n, d in captured.items()}


# =============================================================================
# Profile dispatch (string name to builder, for ablation sweeps)
# =============================================================================

PROFILE_NAMES = ("uniform", "full_slope", "half_slope", "half_slope_rg")


def profile_stats(b_dict: dict) -> tuple[float, float]:
    """Return (mean, std) of the per-group bit-width tensor concatenation.

    Convenience for the per-condition diagnostic prints in the calling scripts
    (e.g. the CELL V6 sanity output of phase_d2_ssm_*).

    Args:
        b_dict: per-module dict of integer bit-width tensors.

    Returns:
        Tuple of (mean, std) as Python floats.
    """
    flat = torch.cat([b.flatten() for b in b_dict.values()])
    return float(flat.float().mean()), float(flat.float().std())
