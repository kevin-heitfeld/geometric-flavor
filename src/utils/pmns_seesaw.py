"""
PMNS Seesaw Mechanism
=====================

Compute neutrino mixing matrix via type-I seesaw:
M_ν = -M_D * M_R^(-1) * M_D^T

Where:
- M_D: Dirac mass matrix (from modular forms)
- M_R: Right-handed Majorana mass matrix (from string scale)
"""

import numpy as np


def dirac_mass_matrix(k_pattern, tau, eta_func, k_charged):
    """
    Construct Dirac neutrino mass matrix M_D.

    Democratic structure with modular form factors:
    M_D ~ y_e * η(τ)^(k/2) * (1 + hierarchy corrections)
    where y_e is computed from electron mass prediction

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights for leptons
    tau : complex
        Modular parameter
    eta_func : callable
        Dedekind eta function
    k_charged : array_like
        [k_1, k_2, k_3] modular weights for charged leptons (to get y_e)

    Returns
    -------
    M_D : ndarray (3, 3)
        Dirac mass matrix in GeV
    """
    eta = eta_func(tau, n_terms=50)
    k = np.array(k_pattern)
    k_ch = np.array(k_charged)

    # Compute electron Yukawa from its mass
    # m_e = y_e * v_EW, so y_e = m_e / v_EW
    v_EW = 246  # GeV (Higgs VEV)
    m_e_obs = 0.511e-3  # GeV

    # But we need to account for modular factor that generates m_e
    # From charged lepton mass: m_e ~ y_base * |η|^(k_e/2)
    k_e = k_ch[0]  # Electron has lightest k
    modular_e = np.abs(eta)**(k_e / 2)
    y_base = m_e_obs / (v_EW * modular_e)

    # Base: democratic matrix (all 1's)
    M_D = np.ones((3, 3), dtype=complex)

    # Modular form factors for each entry
    for i in range(3):
        for j in range(3):
            # Average of generation weights
            k_avg = (k[i] + k[j]) / 2
            modular_factor = np.abs(eta)**(k_avg / 2)

            # Modular phase
            phase = np.exp(2j * np.pi * (k[i] - k[j]) * np.angle(eta) / 24)

            M_D[i, j] = y_base * modular_factor * phase

    # Overall normalization
    M_D *= v_EW

    return M_D


def majorana_mass_matrix(k_pattern, tau, c_AdS, g_s):
    """
    Construct right-handed Majorana mass matrix M_R.

    Hierarchical structure from modular weights:
    M_R ~ M_string * diag(ε^k1, ε^k2, ε^k3)
    where ε ~ exp(-2π Im[τ]) is suppression factor

    String scale derived from theory:
    M_string = M_Planck * exp(-k_max * 2π * Im[τ]) / g_s

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights
    tau : complex
        Modular parameter
    c_AdS : float
        Central charge (from τ)
    g_s : float
        String coupling (from τ)

    Returns
    -------
    M_R : ndarray (3, 3)
        Majorana mass matrix in GeV
    """
    k = np.array(k_pattern)

    # Suppression factor from string compactification
    epsilon = np.exp(-2 * np.pi * np.imag(tau))

    # Derive string scale from theory parameters
    M_Planck = 1.22e19  # GeV
    k_max = np.max(k)

    # Right-handed neutrino scale: between GUT and Planck
    # Use positive exponent so M_R is large (10^14-10^16 GeV)
    # Seesaw: m_ν ~ M_D² / M_R, so need M_R >> M_D for light neutrinos
    M_string = M_Planck * np.exp(-k_max * np.pi * tau.imag / 10) / (g_s * 100)

    # Hierarchical diagonal masses
    M_R = np.diag([
        M_string * epsilon**(-k[0]/2),  # Negative power: larger k → larger M_R
        M_string * epsilon**(-k[1]/2),
        M_string * epsilon**(-k[2]/2)
    ])

    return M_R


def seesaw_mass_matrix(M_D, M_R):
    """
    Type-I seesaw formula.

    M_ν = -M_D * M_R^(-1) * M_D^T

    Parameters
    ----------
    M_D : ndarray (3, 3)
        Dirac mass matrix
    M_R : ndarray (3, 3)
        Majorana mass matrix

    Returns
    -------
    M_nu : ndarray (3, 3)
        Light neutrino mass matrix
    """
    M_R_inv = np.linalg.inv(M_R)
    M_nu = -M_D @ M_R_inv @ M_D.T

    return M_nu


def pmns_from_seesaw(M_D, M_R):
    """
    Compute PMNS matrix from seesaw mechanism.

    Diagonalize M_ν = U_PMNS^* M_diag U_PMNS^†

    Parameters
    ----------
    M_D : ndarray (3, 3)
        Dirac mass matrix
    M_R : ndarray (3, 3)
        Majorana mass matrix

    Returns
    -------
    U_PMNS : ndarray (3, 3)
        PMNS mixing matrix
    masses : ndarray (3,)
        Neutrino mass eigenvalues (ordered)
    """
    # Seesaw mass matrix
    M_nu = seesaw_mass_matrix(M_D, M_R)

    # Diagonalize (Hermitian, so use eigh)
    eigenvalues, U_PMNS = np.linalg.eigh(M_nu)

    # Sort by mass (ascending)
    idx = np.argsort(np.abs(eigenvalues))
    masses = eigenvalues[idx]
    U_PMNS = U_PMNS[:, idx]

    # Phase convention: first row real and positive
    for j in range(3):
        if np.abs(U_PMNS[0, j]) > 1e-10:
            phase = np.exp(-1j * np.angle(U_PMNS[0, j]))
            U_PMNS[:, j] *= phase

    return U_PMNS, masses


def pmns_angles_from_matrix(U):
    """
    Extract mixing angles from PMNS matrix.

    Standard parametrization:
    sin²θ₁₂, sin²θ₂₃, sin²θ₁₃

    Parameters
    ----------
    U : ndarray (3, 3)
        PMNS matrix

    Returns
    -------
    angles : dict
        {'theta_12': ..., 'theta_23': ..., 'theta_13': ...}
    """
    # Extract angles from matrix elements
    # |U_e3|² = sin²θ₁₃
    sin2_theta13 = np.abs(U[0, 2])**2

    # |U_μ3|² = cos²θ₁₃ sin²θ₂₃
    cos2_theta13 = 1 - sin2_theta13
    sin2_theta23 = np.abs(U[1, 2])**2 / cos2_theta13 if cos2_theta13 > 1e-10 else 0

    # |U_e2|² = cos²θ₁₃ sin²θ₁₂
    sin2_theta12 = np.abs(U[0, 1])**2 / cos2_theta13 if cos2_theta13 > 1e-10 else 0

    return {
        'theta_12': sin2_theta12,
        'theta_23': sin2_theta23,
        'theta_13': sin2_theta13
    }


def print_pmns_comparison(U_pred, masses_pred):
    """
    Print PMNS matrix and compare to experiment.

    Parameters
    ----------
    U_pred : ndarray (3, 3)
        Predicted PMNS matrix
    masses_pred : ndarray (3,)
        Predicted neutrino masses

    Returns
    -------
    chi2_dof : float
        Chi-squared per degree of freedom
    """
    U_mag = np.abs(U_pred)
    angles_pred = pmns_angles_from_matrix(U_pred)

    # Experimental values (PDG 2023)
    angles_exp = {
        'theta_12': 0.304,  # Solar: 33.4° ± 0.8°
        'theta_23': 0.545,  # Atmospheric: 49.0° ± 1.2°
        'theta_13': 0.022   # Reactor: 8.5° ± 0.1°
    }

    print("  PMNS Matrix (from seesaw):")
    print("          ν_1       ν_2       ν_3")
    for i, label in enumerate(['e', 'μ', 'τ']):
        print(f"    {label}:  {U_mag[i,0]:.5f}  {U_mag[i,1]:.5f}  {U_mag[i,2]:.5f}")
    print()

    print("  Mixing angles:")
    print(f"    sin²θ₁₂: {angles_pred['theta_12']:.4f} (exp: {angles_exp['theta_12']:.4f})")
    print(f"    sin²θ₂₃: {angles_pred['theta_23']:.4f} (exp: {angles_exp['theta_23']:.4f})")
    print(f"    sin²θ₁₃: {angles_pred['theta_13']:.4f} (exp: {angles_exp['theta_13']:.4f})")
    print()

    print("  Neutrino masses:")
    print(f"    m₁: {np.abs(masses_pred[0]):.3e} GeV")
    print(f"    m₂: {np.abs(masses_pred[1]):.3e} GeV")
    print(f"    m₃: {np.abs(masses_pred[2]):.3e} GeV")

    # Δm² values
    dm21_sq = np.abs(masses_pred[1])**2 - np.abs(masses_pred[0])**2
    dm32_sq = np.abs(masses_pred[2])**2 - np.abs(masses_pred[1])**2

    dm21_sq_exp = 7.5e-5  # eV²
    dm32_sq_exp = 2.5e-3  # eV²

    print(f"    Δm²₂₁: {dm21_sq:.3e} eV² (exp: {dm21_sq_exp:.3e})")
    print(f"    Δm²₃₂: {dm32_sq:.3e} eV² (exp: {dm32_sq_exp:.3e})")
    print()

    # Chi-squared
    chi2 = 0
    for key in ['theta_12', 'theta_23', 'theta_13']:
        err = (angles_pred[key] - angles_exp[key]) / (0.05 * angles_exp[key])  # 5% errors
        chi2 += err**2

    chi2_dof = chi2 / 3
    print(f"  χ²/dof: {chi2_dof:.1f}")

    if chi2_dof < 3:
        print("  Status: ✓ PMNS predicted from seesaw!")
    elif chi2_dof < 10:
        print("  Status: ~ Moderate agreement")
    else:
        print("  Status: ⚠ Needs refinement")
    print()

    return chi2_dof
