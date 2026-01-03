"""
Optimize PMNS with Inverse Seesaw - Simplified Parameterization
================================================================

Strategy that worked for CKM:
1. Use differential evolution for global search
2. Refine with local optimizers
3. Keep it simple - diagonal masses fixed, optimize off-diagonals

Inverse seesaw: M_ν = M_D^T M_R^(-1) μ M_R^(-1) M_D
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


def dedekind_eta(tau, n_terms=50):
    """Dedekind eta function."""
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24)
    for n in range(1, n_terms + 1):
        eta *= (1 - q**n)
    return eta


def inverse_seesaw(M_D, M_R, mu):
    """Effective neutrino mass."""
    M_R_inv = np.linalg.inv(M_R)
    M_nu = M_D.T @ M_R_inv @ mu @ M_R_inv.T @ M_D
    return M_nu


def pmns_from_mass(M_nu):
    """Extract PMNS and masses."""
    eigenvalues, eigenvectors = np.linalg.eig(M_nu)
    idx = np.argsort(np.abs(eigenvalues))
    nu_masses = np.abs(eigenvalues[idx])
    U_PMNS = eigenvectors[:, idx]

    # Phase convention
    for i in range(3):
        if np.real(U_PMNS[i, i]) < 0:
            U_PMNS[:, i] *= -1

    return U_PMNS, nu_masses


# Fixed scales (from previous optimization)
M_R_scale = 0.408  # GeV (from previous run)
mu_scale = 2.48e-5  # GeV

# Mass ratios (diagonal, fixed)
d_R = np.array([1.0, 1.0, 1.0])  # Democratic M_R
d_mu = np.array([1.0, 1.0, 1.0])  # Democratic μ
d_D = np.array([1.0, 1.0, 1.0])  # Democratic M_D

# Observed values
sin2_12_obs = 0.307
sin2_23_obs = 0.546
sin2_13_obs = 0.0218
Delta_m21_sq_obs = 7.5e-5  # eV²
Delta_m31_sq_obs = 2.5e-3  # eV²

print("="*80)
print("PMNS OPTIMIZATION VIA INVERSE SEESAW - SIMPLIFIED")
print("="*80)
print()
print("Target PMNS angles:")
print(f"  sin²θ₁₂ = {sin2_12_obs:.4f}")
print(f"  sin²θ₂₃ = {sin2_23_obs:.4f}")
print(f"  sin²θ₁₃ = {sin2_13_obs:.4f}")
print(f"Target mass splittings:")
print(f"  Δm²₂₁ = {Delta_m21_sq_obs:.2e} eV²")
print(f"  Δm²₃₁ = {Delta_m31_sq_obs:.2e} eV²")
print()


def build_matrices(params):
    """
    Build M_D, M_R, μ from parameters.

    Parameters (15 total):
    - params[0:2]: log10(M_R_scale), log10(mu_scale)
    - params[2:5]: M_D off-diagonals (real, 3)
    - params[5:8]: M_R off-diagonals (real, 3)
    - params[8:11]: μ off-diagonals (real, 3)
    - params[11:15]: μ diagonal adjustments (4: 3 diagonal + 1 overall scale)
    """
    M_R_sc = 10**params[0]
    mu_sc = 10**params[1]

    # M_D: democratic diagonal + off-diagonals
    base_yukawa = 1e-6  # Typical neutrino Yukawa
    M_D = np.diag([base_yukawa, base_yukawa, base_yukawa]).astype(complex)
    M_D[0, 1] = params[2] * base_yukawa
    M_D[1, 0] = M_D[0, 1]
    M_D[1, 2] = params[3] * base_yukawa
    M_D[2, 1] = M_D[1, 2]
    M_D[0, 2] = params[4] * base_yukawa
    M_D[2, 0] = M_D[0, 2]
    M_D *= 246  # Higgs VEV

    # M_R: TeV scale, diagonal + small off-diagonals
    M_R = np.diag([M_R_sc, M_R_sc, M_R_sc]).astype(complex)
    M_R[0, 1] = params[5] * M_R_sc * 0.1
    M_R[1, 0] = M_R[0, 1]
    M_R[1, 2] = params[6] * M_R_sc * 0.1
    M_R[2, 1] = M_R[1, 2]
    M_R[0, 2] = params[7] * M_R_sc * 0.1
    M_R[2, 0] = M_R[0, 2]

    # μ: keV-MeV scale, this is the key for small neutrino masses
    mu_diag = mu_sc * np.array([params[11], params[12], params[13]]) * params[14]
    mu = np.diag(mu_diag).astype(complex)
    mu[0, 1] = params[8] * mu_sc
    mu[1, 0] = mu[0, 1]
    mu[1, 2] = params[9] * mu_sc
    mu[2, 1] = mu[1, 2]
    mu[0, 2] = params[10] * mu_sc
    mu[2, 0] = mu[0, 2]

    return M_D, M_R, mu


def objective(params):
    """Minimax objective for angles and mass splittings."""
    try:
        M_D, M_R, mu = build_matrices(params)
        M_nu = inverse_seesaw(M_D, M_R, mu)
        U_PMNS, nu_masses = pmns_from_mass(M_nu)

        # Extract angles
        sin2_12 = np.abs(U_PMNS[0, 1])**2
        sin2_23 = np.abs(U_PMNS[1, 2])**2
        sin2_13 = np.abs(U_PMNS[0, 2])**2

        # Mass splittings (GeV² to eV²)
        Delta_m21_sq = (nu_masses[1]**2 - nu_masses[0]**2) * 1e18
        Delta_m31_sq = (nu_masses[2]**2 - nu_masses[0]**2) * 1e18

        # Relative errors
        err_12 = abs(sin2_12 - sin2_12_obs) / sin2_12_obs
        err_23 = abs(sin2_23 - sin2_23_obs) / sin2_23_obs
        err_13 = abs(sin2_13 - sin2_13_obs) / sin2_13_obs
        err_m21 = abs(Delta_m21_sq - Delta_m21_sq_obs) / Delta_m21_sq_obs
        err_m31 = abs(Delta_m31_sq - Delta_m31_sq_obs) / Delta_m31_sq_obs

        return max(err_12, err_23, err_13, err_m21, err_m31)

    except:
        return 1e10


# Initial guess from previous run
x0 = [
    np.log10(0.408),    # log10(M_R)
    np.log10(2.48e-5),  # log10(μ)
    0.5, 0.5, 0.1,      # M_D off-diagonals
    0.1, 0.1, 0.01,     # M_R off-diagonals
    0.5, 0.5, 0.1,      # μ off-diagonals
    1.0, 1.0, 1.0, 1.0  # μ diagonal factors
]

# Bounds for differential evolution
bounds = [
    (-2, 4),            # M_R: 0.01 GeV to 10 TeV
    (-6, -2),           # μ: 1 μeV to 10 MeV
    (-2, 2), (-2, 2), (-2, 2),  # M_D off-diag
    (-1, 1), (-1, 1), (-1, 1),  # M_R off-diag
    (-2, 2), (-2, 2), (-2, 2),  # μ off-diag
    (0.1, 10), (0.1, 10), (0.1, 10), (0.1, 10)  # μ diagonal
]

print(f"Optimizing {len(x0)} parameters...")
print()

# Phase 1: Global search
print("Phase 1: Differential evolution (global search)...")
result_de = differential_evolution(objective, bounds, maxiter=5000,
                                   seed=42, atol=1e-10, tol=1e-10,
                                   workers=1, updating='deferred')
print(f"DE complete: max error = {result_de.fun*100:.2f}%")
print()

# Phase 2: Local refinement
print("Phase 2: L-BFGS-B refinement...")
result = minimize(objective, result_de.x, method='L-BFGS-B', bounds=bounds,
                 options={'maxiter': 10000, 'ftol': 1e-14})
print(f"L-BFGS-B complete: max error = {result.fun*100:.2f}%")
print()

# Phase 3: Nelder-Mead
print("Phase 3: Nelder-Mead final polish...")
result2 = minimize(objective, result.x, method='Nelder-Mead',
                  options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
print(f"Nelder-Mead complete: max error = {result2.fun*100:.2f}%")
print()

if result2.fun < result.fun:
    result = result2
    print("Using Nelder-Mead result")
else:
    print("Using L-BFGS-B result")
print()

# Extract and display results
params_opt = result.x
M_D_opt, M_R_opt, mu_opt = build_matrices(params_opt)
M_nu_opt = inverse_seesaw(M_D_opt, M_R_opt, mu_opt)
U_PMNS_opt, nu_masses_opt = pmns_from_mass(M_nu_opt)

sin2_12_opt = np.abs(U_PMNS_opt[0, 1])**2
sin2_23_opt = np.abs(U_PMNS_opt[1, 2])**2
sin2_13_opt = np.abs(U_PMNS_opt[0, 2])**2

Delta_m21_sq_opt = (nu_masses_opt[1]**2 - nu_masses_opt[0]**2) * 1e18
Delta_m31_sq_opt = (nu_masses_opt[2]**2 - nu_masses_opt[0]**2) * 1e18

print("OPTIMIZED SCALES:")
print(f"  M_R = {10**params_opt[0]:.3e} GeV")
print(f"  μ   = {10**params_opt[1]:.3e} GeV")
print()

print("OPTIMIZED OFF-DIAGONALS:")
print(f"  M_D: [{params_opt[2]:7.4f}, {params_opt[3]:7.4f}, {params_opt[4]:7.4f}]")
print(f"  M_R: [{params_opt[5]:7.4f}, {params_opt[6]:7.4f}, {params_opt[7]:7.4f}]")
print(f"  μ:   [{params_opt[8]:7.4f}, {params_opt[9]:7.4f}, {params_opt[10]:7.4f}]")
print()

print("PREDICTED PMNS ANGLES:")
print(f"  sin²θ₁₂ = {sin2_12_opt:.4f} (obs: {sin2_12_obs:.4f})")
print(f"  sin²θ₂₃ = {sin2_23_opt:.4f} (obs: {sin2_23_obs:.4f})")
print(f"  sin²θ₁₃ = {sin2_13_opt:.4f} (obs: {sin2_13_obs:.4f})")
print()

print("PREDICTED MASS SPLITTINGS:")
print(f"  Δm²₂₁ = {Delta_m21_sq_opt:.3e} eV² (obs: {Delta_m21_sq_obs:.2e})")
print(f"  Δm²₃₁ = {Delta_m31_sq_opt:.3e} eV² (obs: {Delta_m31_sq_obs:.2e})")
print()

err_12 = abs(sin2_12_opt - sin2_12_obs) / sin2_12_obs * 100
err_23 = abs(sin2_23_opt - sin2_23_obs) / sin2_23_obs * 100
err_13 = abs(sin2_13_opt - sin2_13_obs) / sin2_13_obs * 100
err_m21 = abs(Delta_m21_sq_opt - Delta_m21_sq_obs) / Delta_m21_sq_obs * 100
err_m31 = abs(Delta_m31_sq_opt - Delta_m31_sq_obs) / Delta_m31_sq_obs * 100

print("ERRORS:")
print(f"  sin²θ₁₂: {err_12:.2f}%")
print(f"  sin²θ₂₃: {err_23:.2f}%")
print(f"  sin²θ₁₃: {err_13:.2f}%")
print(f"  Δm²₂₁:   {err_m21:.2f}%")
print(f"  Δm²₃₁:   {err_m31:.2f}%")
print(f"  Maximum: {max(err_12, err_23, err_13, err_m21, err_m31):.2f}%")
print()

print("="*80)
print("Parameters for unified_predictions_complete.py:")
print("="*80)
print(f"M_R_scale = {10**params_opt[0]:.6e}  # GeV")
print(f"mu_scale = {10**params_opt[1]:.6e}  # GeV")
print(f"M_D_offdiag = np.array([{params_opt[2]:.6f}, {params_opt[3]:.6f}, {params_opt[4]:.6f}])")
print(f"M_R_offdiag = np.array([{params_opt[5]:.6f}, {params_opt[6]:.6f}, {params_opt[7]:.6f}])")
print(f"mu_offdiag = np.array([{params_opt[8]:.6f}, {params_opt[9]:.6f}, {params_opt[10]:.6f}])")
print(f"mu_diag_factors = np.array([{params_opt[11]:.6f}, {params_opt[12]:.6f}, {params_opt[13]:.6f}, {params_opt[14]:.6f}])")
