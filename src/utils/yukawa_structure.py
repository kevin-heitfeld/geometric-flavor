"""
Full Yukawa Matrix Structure from Modular Forms
================================================

Implements complete Yukawa matrix including off-diagonal elements:
Y = diag(d_i) + ε × J

Where:
- d_i: Diagonal couplings from modular forms η(τ)^(k_i/2)
- ε: Democratic coupling (flavor-universal)
- J: All-ones matrix (democratic structure)

This structure is motivated by:
1. Higgs couples democratically to all generations (SU(2)_L)
2. Hierarchy from generation-dependent modular weights
3. Mixing from interference between diagonal and democratic terms
"""

import numpy as np


def diagonal_yukawas_from_modular(k_pattern, tau, eta_func):
    """
    Compute diagonal Yukawa couplings from modular forms.

    Y_ii ~ η(τ)^(k_i/2)

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights
    tau : complex
        Modular parameter
    eta_func : callable
        Dedekind eta function

    Returns
    -------
    d : ndarray
        Diagonal Yukawa couplings (3,)
    """
    eta = eta_func(tau, n_terms=50)
    k = np.array(k_pattern)

    # Diagonal couplings from modular forms
    d = np.array([np.abs(eta)**(k_i/2) for k_i in k])

    return d


def democratic_coupling_from_tau(tau, g_s):
    """
    Compute democratic (off-diagonal) coupling ε.

    The democratic term represents universal Higgs-fermion coupling.
    Its magnitude comes from string coupling and modular parameter.

    ε ~ g_s × Im[τ]^(-1/2)

    Parameters
    ----------
    tau : complex
        Modular parameter
    g_s : float
        String coupling

    Returns
    -------
    epsilon : float
        Democratic coupling strength
    """
    # Democratic coupling from string physics
    # Scales with g_s (string perturbation parameter)
    # And inverse sqrt of volume (~ Im[τ])
    Im_tau = np.imag(tau)

    epsilon = g_s * Im_tau**(-0.5)

    # Normalization: should be smaller than diagonal for hierarchy
    # Typical: ε/d_max ~ 0.1-0.3 (10-30% mixing)
    epsilon *= 0.15  # Calibration factor

    return epsilon


def yukawa_matrix_full(k_pattern, tau, g_s, eta_func, sector='up'):
    """
    Construct full 3×3 Yukawa matrix with off-diagonal structure.

    Y = diag(d_1, d_2, d_3) + ε × J

    where J = [[1,1,1], [1,1,1], [1,1,1]] is democratic matrix.

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights
    tau : complex
        Modular parameter
    g_s : float
        String coupling
    eta_func : callable
        Dedekind eta function
    sector : str
        'up' or 'down' (different phase structure)

    Returns
    -------
    Y : ndarray (3, 3)
        Full Yukawa matrix with off-diagonals
    """
    # Diagonal couplings from modular forms
    d = diagonal_yukawas_from_modular(k_pattern, tau, eta_func)

    # Democratic coupling
    epsilon = democratic_coupling_from_tau(tau, g_s)

    # Sector-dependent phase
    if sector == 'down':
        # Down quarks have different modular phase
        eta = eta_func(tau, n_terms=50)
        phase_shift = np.exp(1j * np.angle(eta) / 13)
        epsilon *= phase_shift

    # Construct full matrix
    Y = np.diag(d) + epsilon * np.ones((3, 3))

    return Y


def diagonalize_yukawa(Y):
    """
    Diagonalize Yukawa matrix to get mass eigenvalues and mixing.

    Y = U^† · diag(y_1, y_2, y_3) · V

    For mass matrix M = Y^† Y:
    M = V^† · diag(m_1^2, m_2^2, m_3^2) · V

    Parameters
    ----------
    Y : ndarray (3, 3)
        Yukawa matrix

    Returns
    -------
    masses : ndarray (3,)
        Mass eigenvalues (sorted)
    V : ndarray (3, 3)
        Mixing matrix
    """
    # Mass-squared matrix
    M = Y.conj().T @ Y

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Masses (positive roots)
    masses = np.sqrt(np.abs(eigenvalues))

    # Sort by mass (ascending)
    idx = np.argsort(masses)
    masses = masses[idx]
    V = eigenvectors[:, idx]

    return masses, V


def ckm_from_yukawa_matrices(Y_up, Y_down):
    """
    Compute CKM matrix from up and down Yukawa matrices.

    V_CKM = V_up^† · V_down

    where V_up, V_down diagonalize the respective mass matrices.

    Parameters
    ----------
    Y_up, Y_down : ndarray (3, 3)
        Up and down Yukawa matrices

    Returns
    -------
    V_CKM : ndarray (3, 3)
        CKM mixing matrix
    """
    # Diagonalize both sectors
    _, V_up = diagonalize_yukawa(Y_up)
    _, V_down = diagonalize_yukawa(Y_down)

    # CKM from relative rotation
    V_CKM = V_up.conj().T @ V_down

    # Make it unitary (numerical cleanup)
    U, _, Vh = np.linalg.svd(V_CKM)
    V_CKM = U @ Vh

    return V_CKM


def mass_ratios_from_full_yukawa(k_pattern, tau, g_s, eta_func):
    """
    Compute mass ratios including off-diagonal structure.

    This should give much better predictions than diagonal-only.

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights
    tau : complex
        Modular parameter
    g_s : float
        String coupling
    eta_func : callable
        Dedekind eta function

    Returns
    -------
    ratios : tuple
        (m2/m1, m3/m1) from full Yukawa diagonalization
    """
    # Construct full Yukawa (use up sector as representative)
    Y = yukawa_matrix_full(k_pattern, tau, g_s, eta_func, sector='up')

    # Diagonalize to get physical masses
    masses, _ = diagonalize_yukawa(Y)

    return masses[1] / masses[0], masses[2] / masses[0]


def ckm_from_full_yukawas(k_pattern, tau, g_s, eta_func):
    """
    Compute CKM matrix from full Yukawa matrices (up and down).

    Parameters
    ----------
    k_pattern : array_like
        [k_1, k_2, k_3] modular weights
    tau : complex
        Modular parameter
    g_s : float
        String coupling
    eta_func : callable
        Dedekind eta function

    Returns
    -------
    V_CKM : ndarray (3, 3)
        CKM matrix from full Yukawa structure
    masses_up : ndarray (3,)
        Up quark masses
    masses_down : ndarray (3,)
        Down quark masses
    """
    # Full Yukawa matrices for both sectors
    Y_up = yukawa_matrix_full(k_pattern, tau, g_s, eta_func, sector='up')
    Y_down = yukawa_matrix_full(k_pattern, tau, g_s, eta_func, sector='down')

    # Get masses and CKM
    masses_up, _ = diagonalize_yukawa(Y_up)
    masses_down, _ = diagonalize_yukawa(Y_down)
    V_CKM = ckm_from_yukawa_matrices(Y_up, Y_down)

    return V_CKM, masses_up, masses_down
