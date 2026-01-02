"""
Wavefunction Overlap Integrals for Yukawa Couplings

Computes the triple overlap integrals that determine Yukawa normalization prefactors:

    f_sector = ∫_CY3 ψ_i(z) ψ_j(z) ψ_H(z) √g d⁶z

where ψ_i are D-brane wavefunctions on T²×T²×T² ⊂ CY3.

Physics:
- D-branes wrap holomorphic cycles with wrapping numbers n_i = (n₁, n₂, n₃)
- Wavefunctions are Gaussian-like profiles localized at wrapping modes
- Width determined by brane tension and modular parameter τ
- Different sectors (leptons, up, down) wrap different cycles → different overlaps

Implementation:
- Factorizes into 3 T² contributions (product structure)
- Each T² gives 2D overlap integral
- Uses τ-dependent width: ℓ ~ √(Im[τ]) in string units
"""

import numpy as np
from scipy import integrate
from typing import Tuple, Optional

# ============================================================================
# D-BRANE WAVEFUNCTION ON T²
# ============================================================================

def dbrane_wavefunction_T2(z: complex, n: Tuple[int, int], tau: complex,
                           width_factor: float = 1.0) -> float:
    """
    D-brane wavefunction on a single T² = ℂ/(ℤ⊕τℤ).

    Physics:
    - Localized Gaussian profile centered at wrapping mode n = (n₁, n₂)
    - Width set by: ℓ = width_factor × √(Im[τ]) (string units)
    - Normalized: ∫_T² |ψ|² √g d²z = 1

    Args:
        z: Complex coordinate on T² (in fundamental domain)
        n: Wrapping numbers (n₁, n₂)
        tau: Modular parameter τ = τ₁ + iτ₂
        width_factor: Multiplies √(Im[τ]) to set wavefunction width

    Returns:
        Wavefunction value |ψ(z)|² (real, for overlap calculation)
    """
    tau_2 = tau.imag

    # Wavefunction width in string units
    width = width_factor * np.sqrt(tau_2)

    # Center position on T² from wrapping numbers
    # z_center = n₁ + n₂×τ (wrapping mode position)
    z_center = n[0] + n[1] * tau

    # Distance on T² (with periodicity)
    # Must account for fundamental domain [0,1] × [0,τ₂]
    dz = z - z_center

    # Project to fundamental domain (shortest distance on torus)
    # Real part: mod 1
    # Imaginary part: mod τ₂
    dz_real = dz.real - np.round(dz.real)
    dz_imag = dz.imag - np.round(dz.imag / tau_2) * tau_2
    dz_periodic = dz_real + 1j * dz_imag

    # Gaussian profile with τ-dependent width
    # |ψ|² = N × exp(-|z-z_center|²/width²)
    # where N is normalization constant
    distance_sq = abs(dz_periodic)**2
    psi_squared = np.exp(-distance_sq / width**2)

    # Normalization: N = 1/(π × width² × sqrt(g))
    # where sqrt(g) = τ₂ is volume form on T²
    normalization = 1.0 / (np.pi * width**2 * tau_2)

    return normalization * psi_squared


def triple_overlap_T2(n_i: Tuple[int, int], n_j: Tuple[int, int],
                      n_H: Tuple[int, int], tau: complex,
                      width_factor: float = 1.0,
                      num_points: int = 50) -> float:
    """
    Triple overlap integral on a single T²:

        I = ∫_T² ψ_i(z) ψ_j(z) ψ_H(z) √g d²z

    where √g = Im[τ] is the volume form.

    Args:
        n_i: Wrapping numbers for first fermion
        n_j: Wrapping numbers for second fermion
        n_H: Wrapping numbers for Higgs
        tau: Modular parameter
        width_factor: Controls wavefunction width
        num_points: Grid resolution for numerical integration

    Returns:
        Overlap integral value (dimensionless)
    """
    tau_2 = tau.imag

    # Integration domain: fundamental domain [0,1] × [0,τ₂]
    x_grid = np.linspace(0, 1, num_points)
    y_grid = np.linspace(0, tau_2, num_points)

    # Compute integrand on grid
    integrand = np.zeros((num_points, num_points))

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            z = x + 1j * y

            # Triple product of wavefunctions (all real since |ψ|²)
            psi_i = dbrane_wavefunction_T2(z, n_i, tau, width_factor)
            psi_j = dbrane_wavefunction_T2(z, n_j, tau, width_factor)
            psi_H = dbrane_wavefunction_T2(z, n_H, tau, width_factor)

            integrand[i, j] = psi_i * psi_j * psi_H

    # Numerical integration using trapezoidal rule
    # d²z = dx dy, volume form √g = τ₂ already in wavefunctions
    dx = 1.0 / (num_points - 1)
    dy = tau_2 / (num_points - 1)

    # Use np.trapezoid (numpy >= 2.0) or np.trapz (older versions)
    try:
        overlap = np.trapezoid(np.trapezoid(integrand, dx=dy, axis=1), dx=dx)
    except AttributeError:
        overlap = np.trapz(np.trapz(integrand, dx=dy, axis=1), dx=dx)

    return overlap
# ============================================================================
# FULL CY3 = T²×T²×T² OVERLAP INTEGRALS
# ============================================================================

def yukawa_overlap_integral(wrapping_i: np.ndarray, wrapping_j: np.ndarray,
                            wrapping_H: np.ndarray, tau_values: np.ndarray,
                            width_factor: float = 1.0,
                            num_points: int = 30) -> float:
    """
    Full Yukawa overlap integral on CY3 = T²×T²×T².

    Factorizes into product of 3 T² integrals:

        ∫_CY3 ψ_i ψ_j ψ_H = ∏_{k=1}^3 ∫_T²_k ψ_i^(k) ψ_j^(k) ψ_H^(k)

    Args:
        wrapping_i: Wrapping numbers for fermion i, shape (3,2) for 3 T²
        wrapping_j: Wrapping numbers for fermion j, shape (3,2)
        wrapping_H: Wrapping numbers for Higgs, shape (3,2)
        tau_values: Modular parameters for each T², shape (3,)
        width_factor: Controls wavefunction width (same for all)
        num_points: Grid resolution per T² (total points = num_points^6)

    Returns:
        Total overlap integral (dimensionless)
    """
    overlap_total = 1.0

    # Product over 3 tori
    for k in range(3):
        n_i = tuple(wrapping_i[k])
        n_j = tuple(wrapping_j[k])
        n_H = tuple(wrapping_H[k])
        tau_k = tau_values[k]

        overlap_k = triple_overlap_T2(n_i, n_j, n_H, tau_k,
                                      width_factor, num_points)
        overlap_total *= overlap_k

    return overlap_total


# ============================================================================
# SECTOR-SPECIFIC OVERLAP CALCULATIONS
# ============================================================================

def compute_lepton_overlap(tau_values: np.ndarray,
                          wrapping_leptons: np.ndarray,
                          wrapping_higgs: np.ndarray,
                          width_factor: float = 1.0) -> float:
    """
    Compute overlap for lepton Yukawa coupling.

    Args:
        tau_values: τ values for 3 generations (use generation 1)
        wrapping_leptons: Lepton wrapping numbers (3,2)
        wrapping_higgs: Higgs wrapping numbers (3,2)
        width_factor: Wavefunction width parameter

    Returns:
        f_lep = ∫ ψ_e ψ_e ψ_H
    """
    # For lepton Yukawa: e† e H (same fermion twice)
    overlap = yukawa_overlap_integral(
        wrapping_leptons, wrapping_leptons, wrapping_higgs,
        tau_values, width_factor
    )

    return overlap


def compute_up_overlap(tau_values: np.ndarray,
                      wrapping_up: np.ndarray,
                      wrapping_higgs: np.ndarray,
                      width_factor: float = 1.0) -> float:
    """
    Compute overlap for up-type quark Yukawa coupling.

    Args:
        tau_values: τ values for 3 generations (use generation 1)
        wrapping_up: Up-quark wrapping numbers (3,2)
        wrapping_higgs: Higgs wrapping numbers (3,2)
        width_factor: Wavefunction width parameter

    Returns:
        f_up = ∫ ψ_u ψ_u ψ_H
    """
    overlap = yukawa_overlap_integral(
        wrapping_up, wrapping_up, wrapping_higgs,
        tau_values, width_factor
    )

    return overlap


def compute_down_overlap(tau_values: np.ndarray,
                        wrapping_down: np.ndarray,
                        wrapping_higgs: np.ndarray,
                        width_factor: float = 1.0) -> float:
    """
    Compute overlap for down-type quark Yukawa coupling.

    Args:
        tau_values: τ values for 3 generations (use generation 1)
        wrapping_down: Down-quark wrapping numbers (3,2)
        wrapping_higgs: Higgs wrapping numbers (3,2)
        width_factor: Wavefunction width parameter

    Returns:
        f_down = ∫ ψ_d ψ_d ψ_H
    """
    overlap = yukawa_overlap_integral(
        wrapping_down, wrapping_down, wrapping_higgs,
        tau_values, width_factor
    )

    return overlap


# ============================================================================
# OPTIMIZATION: FIT WIDTH TO MATCH OBSERVED OVERLAPS
# ============================================================================

def optimize_width_factor(tau_values_lep: np.ndarray,
                         tau_values_up: np.ndarray,
                         tau_values_down: np.ndarray,
                         wrapping_lep: np.ndarray,
                         wrapping_up: np.ndarray,
                         wrapping_down: np.ndarray,
                         wrapping_higgs: np.ndarray,
                         target_overlaps: Tuple[float, float, float],
                         width_bounds: Tuple[float, float] = (0.3, 3.0)) -> float:
    """
    Optimize wavefunction width factor to match observed overlap prefactors.

    Currently, we have fitted values:
        f_lep = 0.053
        f_up = 0.197
        f_down = 0.178

    Goal: Find width_factor that gives these values from geometry.

    Args:
        tau_values_*: Modular parameters for each sector (3,)
        wrapping_*: Wrapping numbers for fermions (3,2)
        wrapping_higgs: Higgs wrapping numbers (3,2)
        target_overlaps: (f_lep, f_up, f_down) target values
        width_bounds: Search range for width_factor

    Returns:
        Optimal width_factor
    """
    from scipy.optimize import minimize_scalar

    target_lep, target_up, target_down = target_overlaps

    def objective(width: float) -> float:
        """Sum of squared relative errors."""
        f_lep = compute_lepton_overlap(tau_values_lep, wrapping_lep,
                                       wrapping_higgs, width)
        f_up = compute_up_overlap(tau_values_up, wrapping_up,
                                  wrapping_higgs, width)
        f_down = compute_down_overlap(tau_values_down, wrapping_down,
                                      wrapping_higgs, width)

        # Relative errors
        err_lep = abs(f_lep - target_lep) / target_lep
        err_up = abs(f_up - target_up) / target_up
        err_down = abs(f_down - target_down) / target_down

        return err_lep**2 + err_up**2 + err_down**2

    result = minimize_scalar(objective, bounds=width_bounds, method='bounded')

    return result.x


# ============================================================================
# ANALYTIC APPROXIMATIONS (FOR INSIGHT)
# ============================================================================

def overlap_gaussian_approximation(n_i: Tuple[int, int], n_j: Tuple[int, int],
                                   n_H: Tuple[int, int], tau: complex,
                                   width_factor: float = 1.0) -> float:
    """
    Analytic approximation for triple Gaussian overlap on T².

    For normalized Gaussians ψ_k(z) = N exp(-|z-z_k|²/2ℓ²):

        ∫ ψ_i ψ_j ψ_H d²z ≈ N³ × (2πℓ²/3)^(1/2) × exp(-Σ|z_k-z_l|²/6ℓ²)

    where N = 1/(√(2π)ℓ√τ₂) is normalization.

    Args:
        n_i, n_j, n_H: Wrapping numbers
        tau: Modular parameter
        width_factor: Width multiplier

    Returns:
        Approximate overlap value (dimensionless)
    """
    tau_2 = tau.imag
    width = width_factor * np.sqrt(tau_2)

    # Positions on T²
    z_i = n_i[0] + n_i[1] * tau
    z_j = n_j[0] + n_j[1] * tau
    z_H = n_H[0] + n_H[1] * tau

    # Distances (shortest path on torus)
    def torus_distance(z1, z2, tau):
        dz = z1 - z2
        # Project to fundamental domain
        dz_real = dz.real - np.round(dz.real)
        dz_imag = dz.imag - np.round(dz.imag / tau_2) * tau_2
        return abs(dz_real + 1j * dz_imag)

    d_ij = torus_distance(z_i, z_j, tau)
    d_jH = torus_distance(z_j, z_H, tau)
    d_Hi = torus_distance(z_H, z_i, tau)

    # Triple Gaussian overlap with proper normalization
    # Each ψ has norm: N = 1/(√(2π) ℓ √τ₂)
    # Product of 3: N³ = 1/[(2π)^(3/2) ℓ³ τ₂^(3/2)]
    normalization = 1.0 / ((2 * np.pi)**(3/2) * width**3 * tau_2**(3/2))

    # Overlap integral for 3 Gaussians:
    # Result ~ (2πσ²/3)^(1/2) × exp(-sum of distances²/6σ²)
    # where σ = ℓ/√2 is the Gaussian width
    sigma = width / np.sqrt(2)
    overlap_factor = np.sqrt(2 * np.pi * sigma**2 / 3.0)
    exponential = np.exp(-(d_ij**2 + d_jH**2 + d_Hi**2) / (6 * sigma**2))

    overlap = normalization * overlap_factor * exponential * tau_2  # × volume form

    return overlap
# ============================================================================
# MAIN INTERFACE
# ============================================================================

def compute_all_overlaps(tau_lep: np.ndarray, tau_up: np.ndarray,
                        tau_down: np.ndarray,
                        wrapping_lep: np.ndarray, wrapping_up: np.ndarray,
                        wrapping_down: np.ndarray, wrapping_higgs: np.ndarray,
                        width_factor: float = 1.0,
                        sector_width_ratios: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                        use_approximation: bool = False) -> Tuple[float, float, float]:
    """
    Compute all three sector overlap integrals.

    Args:
        tau_*: Modular parameters for each sector (generation 1)
        wrapping_*: Wrapping numbers for fermions and Higgs
        width_factor: Base wavefunction width parameter
        sector_width_ratios: (r_lep, r_up, r_down) width multipliers per sector
        use_approximation: Use analytic approximation (faster but less accurate)

    Returns:
        (f_lep, f_up, f_down) overlap prefactors
    """
    r_lep, r_up, r_down = sector_width_ratios

    if use_approximation:
        # Analytic approximation (product over 3 T²)
        f_lep = np.prod([overlap_gaussian_approximation(
            tuple(wrapping_lep[k]), tuple(wrapping_lep[k]),
            tuple(wrapping_higgs[k]), tau_lep[k], width_factor * r_lep
        ) for k in range(3)])

        f_up = np.prod([overlap_gaussian_approximation(
            tuple(wrapping_up[k]), tuple(wrapping_up[k]),
            tuple(wrapping_higgs[k]), tau_up[k], width_factor * r_up
        ) for k in range(3)])

        f_down = np.prod([overlap_gaussian_approximation(
            tuple(wrapping_down[k]), tuple(wrapping_down[k]),
            tuple(wrapping_higgs[k]), tau_down[k], width_factor * r_down
        ) for k in range(3)])
    else:
        # Numerical integration (accurate but slower)
        f_lep = compute_lepton_overlap(tau_lep, wrapping_lep,
                                       wrapping_higgs, width_factor * r_lep)
        f_up = compute_up_overlap(tau_up, wrapping_up,
                                  wrapping_higgs, width_factor * r_up)
        f_down = compute_down_overlap(tau_down, wrapping_down,
                                      wrapping_higgs, width_factor * r_down)

    return f_lep, f_up, f_down
if __name__ == "__main__":
    """
    Test overlap calculations with current wrapping numbers.
    """
    print("=" * 80)
    print("WAVEFUNCTION OVERLAP INTEGRAL CALCULATION")
    print("=" * 80)
    print()

    # Import wrapping numbers from main code
    import sys
    sys.path.append('.')

    # Use values from unified_predictions_complete.py
    tau_0 = 2.7j

    # Simplified wrapping numbers (will import actual values)
    # For now, use example values
    wrapping_lep = np.array([[1, 0], [0, 1], [1, 1]])
    wrapping_up = np.array([[2, 0], [0, 1], [1, 0]])
    wrapping_down = np.array([[1, 1], [1, 0], [0, 1]])
    wrapping_higgs = np.array([[1, 0], [1, 0], [1, 0]])

    tau_lep = np.array([tau_0, tau_0, tau_0])
    tau_up = np.array([tau_0, tau_0, tau_0])
    tau_down = np.array([tau_0, tau_0, tau_0])

    print("Computing overlaps with width_factor = 1.0...")
    print()

    # Analytic approximation (fast)
    f_lep_approx, f_up_approx, f_down_approx = compute_all_overlaps(
        tau_lep, tau_up, tau_down,
        wrapping_lep, wrapping_up, wrapping_down, wrapping_higgs,
        width_factor=1.0, use_approximation=True
    )

    print("ANALYTIC APPROXIMATION:")
    print(f"  f_lep = {f_lep_approx:.6e}")
    print(f"  f_up = {f_up_approx:.6e}")
    print(f"  f_down = {f_down_approx:.6e}")
    print()

    print("Target values (from Kähler potential fit):")
    print(f"  f_lep = 5.300e-02")
    print(f"  f_up = 1.970e-01")
    print(f"  f_down = 1.780e-01")
    print()

    # Note: Full numerical integration would take ~1 minute
    print("Note: Full numerical integration available but slow (30^6 grid points)")
    print("      Use width optimization to find best fit to target values")
