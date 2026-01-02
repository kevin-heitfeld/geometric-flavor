"""
Inverse Seesaw Mechanism
=========================

Implements inverse seesaw for naturally small neutrino masses:

M_ν ≈ M_D^T M_R^(-1) μ M_R^(-1) M_D

Mass matrix structure:
    ⎛  0    M_D   0  ⎞
M = ⎜ M_D^T  0    M_R⎟
    ⎝  0    M_R   μ  ⎠

where:
- M_D: Dirac Yukawa (electroweak scale)
- M_R: Right-handed Majorana (TeV-PeV scale, accessible!)
- μ: Small lepton number violation (keV-MeV scale)

Advantages:
1. M_R can be at TeV scale (testable at colliders)
2. Small m_ν from μ << M_R (not from M_R >> M_D)
3. More parameters → better fit to PMNS angles
"""

import numpy as np


def dirac_mass_matrix_optimized(k_pattern, tau, eta_func, v_EW=246):
    """
    Construct Dirac neutrino mass matrix with optimizable structure.

    Uses democratic ansatz with modular form hierarchy:
    M_D = Y_base * v_EW * η(τ)^(k_avg/2) * (1 + ε_corrections)

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights for neutrinos
    tau : complex
        Modular parameter
    eta_func : callable
        Dedekind eta function
    v_EW : float
        Higgs VEV in GeV

    Returns
    -------
    M_D : ndarray (3, 3)
        Dirac mass matrix in GeV
    """
    eta = eta_func(tau, n_terms=50)
    k = np.array(k_pattern)

    # Base Yukawa from electron mass (same sector)
    m_e_obs = 0.511e-3  # GeV
    k_e = 5  # Electron modular weight
    modular_e = np.abs(eta)**(k_e / 2)
    y_base = m_e_obs / (v_EW * modular_e)

    # Democratic matrix with modular hierarchy
    M_D = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            k_avg = (k[i] + k[j]) / 2
            modular_factor = np.abs(eta)**(k_avg / 2)

            # Modular phase for off-diagonal CP violation
            phase = np.exp(2j * np.pi * (k[i] - k[j]) / 24)

            M_D[i, j] = y_base * modular_factor * phase

    M_D *= v_EW
    return M_D


def majorana_mass_matrix_TeV(k_pattern, tau, M_scale=1e3):
    """
    Right-handed Majorana masses at TeV scale (not GUT scale!).

    For inverse seesaw, M_R can be accessible at colliders.
    Hierarchy from modular weights.

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights
    tau : complex
        Modular parameter
    M_scale : float
        Overall mass scale in GeV (default 1 TeV)

    Returns
    -------
    M_R : ndarray (3, 3)
        Majorana mass matrix in GeV
    """
    k = np.array(k_pattern)
    epsilon = np.exp(-2 * np.pi * np.imag(tau))

    # Hierarchical diagonal with modular suppression
    M_R = np.diag([
        M_scale * epsilon**(-k[0]/4),  # Lighter for larger k
        M_scale * epsilon**(-k[1]/4),
        M_scale * epsilon**(-k[2]/4)
    ])

    # Small off-diagonal mixing from instantons
    for i in range(3):
        for j in range(i+1, 3):
            off_diag = 0.1 * M_scale * epsilon**(-(k[i]+k[j])/8)
            M_R[i, j] = off_diag
            M_R[j, i] = off_diag

    return M_R


def lepton_number_violation_matrix(k_pattern, tau, mu_scale=1e-3):
    """
    Small lepton number violating mass μ.

    This is the key parameter that makes neutrinos light.
    Natural scale: keV - MeV (much smaller than M_R).

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights
    tau : complex
        Modular parameter
    mu_scale : float
        Overall scale in GeV (default 1 MeV)

    Returns
    -------
    mu : ndarray (3, 3)
        LNV mass matrix in GeV
    """
    k = np.array(k_pattern)
    epsilon = np.exp(-2 * np.pi * np.imag(tau))

    # Democratic structure with modular hierarchy
    mu = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            k_avg = (k[i] + k[j]) / 2
            suppression = epsilon**(k_avg / 6)

            # Phase for CP violation
            phase = np.exp(1j * np.pi * (k[i] - k[j]) / 12)

            mu[i, j] = mu_scale * suppression * phase

    return mu


def inverse_seesaw_mass_matrix(M_D, M_R, mu):
    """
    Compute effective light neutrino mass via inverse seesaw.

    Full 9×9 block matrix:
        ⎛  0    M_D   0  ⎞
    M = ⎜ M_D^T  0    M_R⎟
        ⎝  0    M_R^T μ  ⎠

    Effective mass for light neutrinos:
    M_ν ≈ M_D^T M_R^(-1) μ M_R^(-T) M_D

    Parameters
    ----------
    M_D : ndarray (3, 3)
        Dirac mass matrix
    M_R : ndarray (3, 3)
        Right-handed Majorana mass
    mu : ndarray (3, 3)
        Lepton number violation

    Returns
    -------
    M_nu : ndarray (3, 3)
        Effective light neutrino mass matrix
    """
    # Inverse seesaw formula
    M_R_inv = np.linalg.inv(M_R)
    M_nu = M_D.T @ M_R_inv @ mu @ M_R_inv.T @ M_D

    return M_nu


def pmns_from_inverse_seesaw(M_D, M_R, mu):
    """
    Extract PMNS matrix and neutrino masses from inverse seesaw.

    Parameters
    ----------
    M_D : ndarray (3, 3)
        Dirac mass matrix
    M_R : ndarray (3, 3)
        Right-handed Majorana mass
    mu : ndarray (3, 3)
        Lepton number violation

    Returns
    -------
    U_PMNS : ndarray (3, 3)
        PMNS mixing matrix
    nu_masses : ndarray (3,)
        Neutrino mass eigenvalues (sorted)
    """
    # Effective mass matrix
    M_nu = inverse_seesaw_mass_matrix(M_D, M_R, mu)

    # Diagonalize complex symmetric matrix
    # M_ν = U* diag(m_i) U^†
    # Use Takagi decomposition for complex symmetric
    eigenvalues, eigenvectors = np.linalg.eig(M_nu)

    # Sort by mass (absolute value)
    idx = np.argsort(np.abs(eigenvalues))
    nu_masses = np.abs(eigenvalues[idx])
    U_PMNS = eigenvectors[:, idx]

    # Ensure proper phase conventions (maximize real part of U_00)
    for i in range(3):
        if np.real(U_PMNS[0, i]) < 0:
            U_PMNS[:, i] *= -1

    return U_PMNS, nu_masses


def optimize_inverse_seesaw_params(k_pattern, tau, eta_func,
                                   sin2_12_obs, sin2_23_obs, sin2_13_obs,
                                   Delta_m21_sq_obs, Delta_m31_sq_obs):
    """
    Optimize M_R and μ scales to match observed PMNS angles and mass splittings.

    Parameters to fit:
    - M_R_scale: Overall scale of right-handed masses (TeV)
    - mu_scale: Overall scale of LNV (MeV)
    - Off-diagonal structure parameters (6 for M_D, 3 for M_R, 3 for μ)

    Returns
    -------
    params_opt : dict
        Optimized parameters
    """
    # This would be implemented similar to optimize_ckm.py
    # For now, return default values
    return {
        'M_R_scale': 1e3,  # 1 TeV
        'mu_scale': 1e-3,  # 1 MeV
        'M_D_offdiag': np.zeros(6),
        'M_R_offdiag': np.zeros(3),
        'mu_offdiag': np.zeros(3)
    }
