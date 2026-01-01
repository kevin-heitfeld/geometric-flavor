"""
Test generation-dependent τ_i to improve mass hierarchies

Goal: Reduce errors on mass ratios from ~130-200% to <50%
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.generation_dependent_tau import (
    mass_ratio_with_generation_tau,
    optimize_delta_scheme,
    print_scheme_comparison
)

def dedekind_eta(tau, n_terms=50):
    """Dedekind eta η(τ) = q^(1/24) ∏(1-q^n)"""
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24)
    for n in range(1, n_terms):
        eta *= (1 - q**n)
    return eta

print("="*80)
print("TESTING GENERATION-DEPENDENT MODULAR PARAMETERS")
print("="*80)
print()

# Base parameters
tau_0 = 2.7j
g_s = np.exp(-np.log(np.imag(tau_0)))

print(f"Base parameter: τ₀ = {tau_0}")
print(f"String coupling: g_s = {g_s:.4f}")
print()

# k-patterns and localization from phenomenology
k_mass = np.array([8, 6, 4])
A_leptons = np.array([0.00, -0.80, -1.00])
A_up = np.array([0.00, -1.00, -1.60])
A_down = np.array([0.00, -0.20, -0.80])

# Observed mass ratios
r_lep_obs = np.array([1.0, 206.8, 3477])
r_up_obs = np.array([1.0, 577, 78636])
r_down_obs = np.array([1.0, 18.3, 890])

print("CURRENT STATUS (single τ = 2.7i):")
print("-"*80)

# Current predictions (from unified_predictions_complete.py output)
r_lep_current = np.array([1.0, 479.4, 9050.2])
r_up_current = np.array([1.0, 1411.6, 231084.7])
r_down_current = np.array([1.0, 18.8, 3073.4])

err_lep = np.mean(np.abs(r_lep_current[1:] - r_lep_obs[1:]) / r_lep_obs[1:] * 100)
err_up = np.mean(np.abs(r_up_current[1:] - r_up_obs[1:]) / r_up_obs[1:] * 100)
err_down = np.mean(np.abs(r_down_current[1:] - r_down_obs[1:]) / r_down_obs[1:] * 100)

print(f"Leptons:  m_μ/m_e = {r_lep_current[1]:.1f}, m_τ/m_e = {r_lep_current[2]:.1f}")
print(f"          (obs: {r_lep_obs[1]:.1f}, {r_lep_obs[2]:.1f}) → avg error {err_lep:.1f}%")
print(f"Up:       m_c/m_u = {r_up_current[1]:.1f}, m_t/m_u = {r_up_current[2]:.1f}")
print(f"          (obs: {r_up_obs[1]:.1f}, {r_up_obs[2]:.1f}) → avg error {err_up:.1f}%")
print(f"Down:     m_s/m_d = {r_down_current[1]:.1f}, m_b/m_d = {r_down_current[2]:.1f}")
print(f"          (obs: {r_down_obs[1]:.1f}, {r_down_obs[2]:.1f}) → avg error {err_down:.1f}%")
print()

# Test generation-dependent τ
print("="*80)
print("TESTING GENERATION-DEPENDENT τ_i")
print("="*80)
print()

sectors = [
    ("LEPTONS", k_mass, A_leptons, r_lep_obs),
    ("UP QUARKS", k_mass, A_up, r_up_obs),
    ("DOWN QUARKS", k_mass, A_down, r_down_obs),
]

for sector_name, k_pat, A_pat, obs_ratios in sectors:
    print(f"{sector_name}:")
    print("-"*60)

    best_scheme, best_error, results = optimize_delta_scheme(
        k_pat, tau_0, g_s, dedekind_eta, A_pat, obs_ratios
    )

    print_scheme_comparison(results, obs_ratios)

    print(f"✓ Best scheme: {best_scheme} (avg error: {best_error:.1f}%)")
    print()

print("="*80)
print("SUMMARY")
print("="*80)
print()

# Run with best scheme for each sector
print("Implementing best schemes:")
print()

for sector_name, k_pat, A_pat, obs_ratios in sectors:
    best_scheme, best_error, results = optimize_delta_scheme(
        k_pat, tau_0, g_s, dedekind_eta, A_pat, obs_ratios
    )

    ratios = results[best_scheme]['ratios']

    print(f"{sector_name}:")
    print(f"  Scheme: {best_scheme}")
    print(f"  m₂/m₁: {ratios[1]:.1f} (obs: {obs_ratios[1]:.1f})")
    print(f"  m₃/m₁: {ratios[2]:.1f} (obs: {obs_ratios[2]:.1f})")
    print(f"  Error: {best_error:.1f}% (was {err_lep if 'LEPTON' in sector_name else (err_up if 'UP' in sector_name else err_down):.1f}%)")

    improvement = (err_lep if 'LEPTON' in sector_name else (err_up if 'UP' in sector_name else err_down)) - best_error
    print(f"  Improvement: {improvement:.1f} percentage points")
    print()

print("Next step: Integrate best schemes into unified_predictions_complete.py")
