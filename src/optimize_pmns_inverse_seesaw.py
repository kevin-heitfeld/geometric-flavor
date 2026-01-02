"""
Optimize PMNS Angles via Inverse Seesaw
========================================

Optimize the structure of M_D, M_R, and μ matrices to match
observed PMNS mixing angles and neutrino mass splittings.
"""

import numpy as np
from scipy.optimize import minimize


def dedekind_eta(tau, n_terms=50):
    """Dedekind eta function η(τ) = q^(1/24) Π(1-q^n) where q = exp(2πiτ)."""
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24)
    for n in range(1, n_terms + 1):
        eta *= (1 - q**n)
    return eta


def build_dirac_matrix(k_pattern, tau, eta_func, eps_params, v_EW=246):
    """
    Build M_D with parameterized off-diagonals.

    M_D = diag(d_i) + complex off-diagonals

    Parameters
    ----------
    eps_params : array
        [d1, d2, d3, |ε12|, φ12, |ε23|, φ23, |ε13|, φ13]
        9 parameters: 3 diagonal + 6 off-diagonal (mag+phase)
    """
    eta = eta_func(tau, n_terms=50)
    k = np.array(k_pattern)

    # Base scale from electron Yukawa
    m_e = 0.511e-3  # GeV
    k_e = 5
    y_base = m_e / (v_EW * np.abs(eta)**(k_e/2))

    # Diagonal elements with modular hierarchy
    d = np.array([
        eps_params[0] * y_base * np.abs(eta)**(k[0]/2),
        eps_params[1] * y_base * np.abs(eta)**(k[1]/2),
        eps_params[2] * y_base * np.abs(eta)**(k[2]/2)
    ])

    M_D = np.diag(d).astype(complex)

    # Off-diagonals with magnitude and phase
    M_D[0, 1] = eps_params[3] * np.exp(1j * eps_params[4]) * y_base * v_EW
    M_D[1, 0] = M_D[0, 1]
    M_D[1, 2] = eps_params[5] * np.exp(1j * eps_params[6]) * y_base * v_EW
    M_D[2, 1] = M_D[1, 2]
    M_D[0, 2] = eps_params[7] * np.exp(1j * eps_params[8]) * y_base * v_EW
    M_D[2, 0] = M_D[0, 2]

    return M_D


def build_majorana_matrix(M_scale, eps_params):
    """
    Build M_R at TeV scale with hierarchy.

    Parameters
    ----------
    eps_params : array
        [r1, r2, r3] - relative hierarchies
    """
    M_R = np.diag([
        M_scale * eps_params[0],
        M_scale * eps_params[1],
        M_scale * eps_params[2]
    ])
    return M_R.astype(complex)


def build_lnv_matrix(mu_scale, eps_params):
    """
    Build μ matrix (small LNV).

    Parameters
    ----------
    eps_params : array
        [u1, u2, u3, |μ12|, φ12, |μ23|, φ23, |μ13|, φ13]
        9 parameters
    """
    mu = np.diag([
        mu_scale * eps_params[0],
        mu_scale * eps_params[1],
        mu_scale * eps_params[2]
    ]).astype(complex)

    # Off-diagonals
    mu[0, 1] = mu_scale * eps_params[3] * np.exp(1j * eps_params[4])
    mu[1, 0] = mu[0, 1]
    mu[1, 2] = mu_scale * eps_params[5] * np.exp(1j * eps_params[6])
    mu[2, 1] = mu[1, 2]
    mu[0, 2] = mu_scale * eps_params[7] * np.exp(1j * eps_params[8])
    mu[2, 0] = mu[0, 2]

    return mu


def inverse_seesaw(M_D, M_R, mu):
    """Effective neutrino mass via inverse seesaw."""
    M_R_inv = np.linalg.inv(M_R)
    M_nu = M_D.T @ M_R_inv @ mu @ M_R_inv.T @ M_D
    return M_nu


def pmns_from_mass_matrix(M_nu):
    """Extract PMNS and masses from M_ν."""
    eigenvalues, eigenvectors = np.linalg.eig(M_nu)
    idx = np.argsort(np.abs(eigenvalues))
    nu_masses = np.abs(eigenvalues[idx])
    U_PMNS = eigenvectors[:, idx]

    # Phase convention: maximize real part of diagonal
    for i in range(3):
        if np.real(U_PMNS[i, i]) < 0:
            U_PMNS[:, i] *= -1

    return U_PMNS, nu_masses


# Fixed parameters
k_pattern = np.array([5, 3, 1])  # Neutrino modular weights
tau = 2.7j  # Fixed modular parameter
v_EW = 246  # GeV

# Observed values
sin2_12_obs = 0.307
sin2_23_obs = 0.546
sin2_13_obs = 0.0218
Delta_m21_sq_obs = 7.5e-5  # eV²
Delta_m31_sq_obs = 2.5e-3  # eV²

print("="*80)
print("OPTIMIZING PMNS VIA INVERSE SEESAW")
print("="*80)
print()
print(f"Target PMNS angles:")
print(f"  sin²θ₁₂ = {sin2_12_obs:.4f}")
print(f"  sin²θ₂₃ = {sin2_23_obs:.4f}")
print(f"  sin²θ₁₃ = {sin2_13_obs:.4f}")
print(f"Target mass splittings:")
print(f"  Δm²₂₁ = {Delta_m21_sq_obs:.2e} eV²")
print(f"  Δm²₃₁ = {Delta_m31_sq_obs:.2e} eV²")
print()


def objective(params):
    """
    Minimize PMNS angle and mass splitting errors.

    Parameters:
    - params[0]: log10(M_R_scale) in GeV
    - params[1]: log10(mu_scale) in GeV
    - params[2:11]: M_D structure (9)
    - params[11:14]: M_R hierarchy (3)
    - params[14:23]: μ structure (9)

    Total: 23 parameters
    """
    try:
        M_R_scale = 10**params[0]  # GeV
        mu_scale = 10**params[1]   # GeV

        # Build matrices
        M_D = build_dirac_matrix(k_pattern, tau, dedekind_eta, params[2:11], v_EW)
        M_R = build_majorana_matrix(M_R_scale, params[11:14])
        mu = build_lnv_matrix(mu_scale, params[14:23])

        # Inverse seesaw
        M_nu = inverse_seesaw(M_D, M_R, mu)
        U_PMNS, nu_masses = pmns_from_mass_matrix(M_nu)

        # Extract angles
        sin2_12 = np.abs(U_PMNS[0, 1])**2
        sin2_23 = np.abs(U_PMNS[1, 2])**2
        sin2_13 = np.abs(U_PMNS[0, 2])**2

        # Mass splittings (in eV²)
        Delta_m21_sq = (nu_masses[1]**2 - nu_masses[0]**2) * 1e18  # GeV² to eV²
        Delta_m31_sq = (nu_masses[2]**2 - nu_masses[0]**2) * 1e18

        # Relative errors
        err_12 = abs(sin2_12 - sin2_12_obs) / sin2_12_obs
        err_23 = abs(sin2_23 - sin2_23_obs) / sin2_23_obs
        err_13 = abs(sin2_13 - sin2_13_obs) / sin2_13_obs
        err_m21 = abs(Delta_m21_sq - Delta_m21_sq_obs) / Delta_m21_sq_obs
        err_m31 = abs(Delta_m31_sq - Delta_m31_sq_obs) / Delta_m31_sq_obs

        # Minimax: minimize maximum error
        return max(err_12, err_23, err_13, err_m21, err_m31)

    except:
        return 1e10  # Penalty for invalid parameters


# Initial guess
x0 = [
    3.0,  # log10(M_R) = 3 → M_R = 1 TeV
    -3.0, # log10(μ) = -3 → μ = 1 MeV
    # M_D structure (9): diagonal + off-diagonal
    1.0, 1.0, 1.0,  # diagonal ratios
    0.1, 0.0,       # |ε12|, φ12
    0.1, 0.0,       # |ε23|, φ23
    0.01, 0.0,      # |ε13|, φ13
    # M_R hierarchy (3)
    1.0, 1.0, 1.0,
    # μ structure (9)
    1.0, 1.0, 1.0,  # diagonal ratios
    0.1, 0.0,       # |μ12|, φ12
    0.1, 0.0,       # |μ23|, φ23
    0.01, 0.0       # |μ13|, φ13
]

print(f"Optimizing {len(x0)} parameters...")
print(f"Initial guess: M_R ~ {10**x0[0]:.0e} GeV, μ ~ {10**x0[1]:.0e} GeV")
print()

result = minimize(objective, x0, method='Nelder-Mead',
                 options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-8})

print(f"Optimization complete!")
print(f"Success: {result.success}")
print(f"Iterations: {result.nit}")
print(f"Maximum relative error: {result.fun*100:.1f}%")
print()

# Extract optimized values
params_opt = result.x
M_R_scale_opt = 10**params_opt[0]
mu_scale_opt = 10**params_opt[1]

M_D_opt = build_dirac_matrix(k_pattern, tau, dedekind_eta, params_opt[2:11], v_EW)
M_R_opt = build_majorana_matrix(M_R_scale_opt, params_opt[11:14])
mu_opt = build_lnv_matrix(mu_scale_opt, params_opt[14:23])

M_nu_opt = inverse_seesaw(M_D_opt, M_R_opt, mu_opt)
U_PMNS_opt, nu_masses_opt = pmns_from_mass_matrix(M_nu_opt)

sin2_12_opt = np.abs(U_PMNS_opt[0, 1])**2
sin2_23_opt = np.abs(U_PMNS_opt[1, 2])**2
sin2_13_opt = np.abs(U_PMNS_opt[0, 2])**2

Delta_m21_sq_opt = (nu_masses_opt[1]**2 - nu_masses_opt[0]**2) * 1e18
Delta_m31_sq_opt = (nu_masses_opt[2]**2 - nu_masses_opt[0]**2) * 1e18

print("OPTIMIZED SCALES:")
print(f"  M_R = {M_R_scale_opt:.2e} GeV")
print(f"  μ   = {mu_scale_opt:.2e} GeV")
print()

print("PREDICTED PMNS ANGLES:")
print(f"  sin²θ₁₂ = {sin2_12_opt:.4f} (obs: {sin2_12_obs:.4f})")
print(f"  sin²θ₂₃ = {sin2_23_opt:.4f} (obs: {sin2_23_obs:.4f})")
print(f"  sin²θ₁₃ = {sin2_13_opt:.4f} (obs: {sin2_13_obs:.4f})")
print()

print("PREDICTED MASS SPLITTINGS:")
print(f"  Δm²₂₁ = {Delta_m21_sq_opt:.2e} eV² (obs: {Delta_m21_sq_obs:.2e})")
print(f"  Δm²₃₁ = {Delta_m31_sq_opt:.2e} eV² (obs: {Delta_m31_sq_obs:.2e})")
print()

err_12 = abs(sin2_12_opt - sin2_12_obs) / sin2_12_obs * 100
err_23 = abs(sin2_23_opt - sin2_23_obs) / sin2_23_obs * 100
err_13 = abs(sin2_13_opt - sin2_13_obs) / sin2_13_obs * 100
err_m21 = abs(Delta_m21_sq_opt - Delta_m21_sq_obs) / Delta_m21_sq_obs * 100
err_m31 = abs(Delta_m31_sq_opt - Delta_m31_sq_obs) / Delta_m31_sq_obs * 100

print("ERRORS:")
print(f"  sin²θ₁₂: {err_12:.1f}%")
print(f"  sin²θ₂₃: {err_23:.1f}%")
print(f"  sin²θ₁₃: {err_13:.1f}%")
print(f"  Δm²₂₁:   {err_m21:.1f}%")
print(f"  Δm²₃₁:   {err_m31:.1f}%")
print(f"  Maximum: {max(err_12, err_23, err_13, err_m21, err_m31):.1f}%")
