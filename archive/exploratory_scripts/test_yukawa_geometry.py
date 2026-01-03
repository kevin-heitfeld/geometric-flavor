"""
Test: Compare geometric Yukawa normalizations with fitted values
"""

import numpy as np
from yukawa_from_geometry import compute_yukawa_normalizations, compare_with_fitted_values

# Observed fermion masses
m_e_obs = 0.511e-3  # GeV
m_mu_obs = 105.7e-3
m_tau_obs = 1.777

m_u_obs = 2.16e-3   # GeV
m_c_obs = 1.27
m_t_obs = 173.0

m_d_obs = 4.67e-3   # GeV
m_s_obs = 95e-3
m_b_obs = 4.18

# Higgs VEV
v_higgs = 246.0  # GeV

# These are the fitted Yukawa eigenvalues (dimensionless) from the full code
# They come from modular forms with weights and localization
# Approximate values for testing:
m_lep = np.array([2.16e-8, 4.47e-6, 7.50e-5])  # e, μ, τ ratios
m_up_quarks = np.array([7.89e-9, 4.65e-6, 6.34e-4])  # u, c, t ratios
m_down_quarks = np.array([1.55e-8, 3.16e-6, 1.39e-5])  # d, s, b ratios

# Compute fitted Y₀ values (current method)
Y_0_lep_fitted = m_e_obs / (v_higgs * m_lep[0])
Y_0_up_fitted = m_u_obs / (v_higgs * m_up_quarks[0])
Y_0_down_fitted = m_d_obs / (v_higgs * m_down_quarks[0])

print("="*70)
print("FITTED YUKAWA NORMALIZATIONS (Current Method)")
print("="*70)
print(f"  Y₀_lep  = {Y_0_lep_fitted:.6e} (from m_e / (v × y_e))")
print(f"  Y₀_up   = {Y_0_up_fitted:.6e} (from m_u / (v × y_u))")
print(f"  Y₀_down = {Y_0_down_fitted:.6e} (from m_d / (v × y_d))")
print()
print(f"  Ratios:")
print(f"    Y₀_up / Y₀_lep   = {Y_0_up_fitted/Y_0_lep_fitted:.4f}")
print(f"    Y₀_down / Y₀_lep = {Y_0_down_fitted/Y_0_lep_fitted:.4f}")
print()

# Compute geometric predictions
print("="*70)
print("GEOMETRIC YUKAWA NORMALIZATIONS (New Method)")
print("="*70)
Y_0_up_geom, Y_0_down_geom, Y_0_lep_geom = compute_yukawa_normalizations(verbose=True)

print()
compare_with_fitted_values(
    Y_0_up_geom, Y_0_down_geom, Y_0_lep_geom,
    Y_0_up_fitted, Y_0_down_fitted, Y_0_lep_fitted
)
