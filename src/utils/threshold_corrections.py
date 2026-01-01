"""
Threshold Corrections for Precision Mass Predictions
====================================================

Implements stringy threshold corrections:
1. KK tower contributions from compactified dimensions
2. GUT-scale matching corrections (E6 → SU(5))
3. D-brane threshold corrections
4. Wrapped brane corrections
"""

import numpy as np


def kk_tower_correction(k_i, tau, g_s, n_modes=10):
    """
    Kaluza-Klein tower corrections to Yukawa couplings.

    When extra dimensions are compactified on radius R ~ Im[τ],
    KK modes contribute threshold corrections at each mass level.

    Y_eff = Y_tree * (1 + Σ_n A_n e^(-2πnR))

    Parameters
    ----------
    k_i : int
        Modular weight
    tau : complex
        Modular parameter (Im[τ] ~ compactification radius)
    g_s : float
        String coupling
    n_modes : int
        Number of KK modes to include

    Returns
    -------
    delta_KK : float
        Relative correction from KK tower
    """
    R = np.imag(tau)

    # Each KK mode contributes with exponential suppression
    correction = 0.0
    for n in range(1, n_modes + 1):
        # Amplitude ~ (k/n)^2 from mode structure
        amplitude = (k_i / n)**2
        # Exponential suppression from mass M_n ~ n/R
        suppression = np.exp(-2 * np.pi * n * R)
        # String coupling dependence
        correction += amplitude * suppression * g_s**2

    # Normalization factor
    correction *= 1.0 / (16 * np.pi**2)

    return correction


def gut_threshold_correction(k_i, k_j, tau, M_GUT=2e16, M_string=9e16):
    """
    GUT-scale threshold corrections from E6 → SU(5) breaking.

    At M_GUT, the unified E6 Yukawa couplings split into separate
    up and down quark sectors. Matching conditions generate corrections:

    δY/Y ~ log(M_string/M_GUT) * (k_i - k_j)^2 / (16π^2)

    Parameters
    ----------
    k_i, k_j : int
        Modular weights for generations i, j
    tau : complex
        Modular parameter
    M_GUT : float
        GUT scale in GeV
    M_string : float
        String scale in GeV

    Returns
    -------
    delta_GUT : float
        Relative threshold correction
    """
    # Logarithmic running from string to GUT scale
    t = np.log(M_string / M_GUT)

    # Weight difference controls correction size
    delta_k = k_i - k_j

    # Modular form derivative correction
    # (weight splitting induces RG flow)
    correction = (delta_k**2 / (16 * np.pi**2)) * t

    # Additional modular factor from SU(5) breaking
    # (E6 → SU(5) × U(1) at τ-dependent VEV)
    modular_factor = 1.0 + 0.1 * np.abs(np.imag(tau) - 3.0)

    return correction * modular_factor


def dbrane_threshold(k_i, tau, g_s, n_wrapping=1):
    """
    D-brane threshold corrections.

    When D-branes wrap cycles in the internal space,
    they contribute to the effective Yukawa couplings
    through open string exchange.

    δY ~ exp(-n × Area) where Area ~ 2π × Im[τ] × k

    Parameters
    ----------
    k_i : int
        Modular weight (related to brane position)
    tau : complex
        Modular parameter
    g_s : float
        String coupling
    n_wrapping : int
        Wrapping number of D-brane

    Returns
    -------
    delta_brane : float
        D-brane threshold correction
    """
    # Cycle area ~ modular parameter × weight
    area = 2 * np.pi * np.imag(tau) * k_i / 8  # Normalized by k_max=8

    # D-brane action S ~ Area / g_s
    action = n_wrapping * area / g_s

    # Threshold contribution ~ exp(-S)
    correction = np.exp(-action)

    # Geometric factor from brane intersection
    # (matter fields live at brane intersections)
    geometric_factor = (k_i / 8)**2  # Normalized

    return correction * geometric_factor


def wavefunction_renormalization(k_i, k_j, tau, g_s):
    """
    Wavefunction renormalization from string loops.

    Field redefinition from canonical to physical basis
    introduces correction:

    Z_i = 1 + δZ_i
    Y_phys = Y_bare / sqrt(Z_i Z_j)

    Parameters
    ----------
    k_i, k_j : int
        Modular weights for generations i, j
    tau : complex
        Modular parameter
    g_s : float
        String coupling

    Returns
    -------
    Z_factor : float
        Wavefunction renormalization factor sqrt(Z_i Z_j)
    """
    # 1-loop wavefunction anomalous dimension
    gamma_i = k_i / (16 * np.pi**2)
    gamma_j = k_j / (16 * np.pi**2)

    # String loop correction to wavefunction
    delta_Z_i = g_s**2 * gamma_i * np.log(np.abs(np.imag(tau)))
    delta_Z_j = g_s**2 * gamma_j * np.log(np.abs(np.imag(tau)))

    # Physical normalization
    Z_i = 1.0 + delta_Z_i
    Z_j = 1.0 + delta_Z_j

    return np.sqrt(Z_i * Z_j)


def total_threshold_correction(k_i, k_j, tau, g_s):
    """
    Combine all threshold corrections for Yukawa coupling Y_ij.

    Y_eff = Y_tree × (1 + δ_KK + δ_GUT + δ_brane) / Z

    Parameters
    ----------
    k_i, k_j : int
        Modular weights for generations i, j
    tau : complex
        Modular parameter
    g_s : float
        String coupling

    Returns
    -------
    correction_factor : float
        Total multiplicative correction: Y_eff / Y_tree
    """
    # Individual corrections
    delta_kk = kk_tower_correction((k_i + k_j) // 2, tau, g_s, n_modes=5)
    delta_gut = gut_threshold_correction(k_i, k_j, tau)
    delta_brane = dbrane_threshold((k_i + k_j) // 2, tau, g_s)

    # Wavefunction renormalization
    Z = wavefunction_renormalization(k_i, k_j, tau, g_s)

    # Combined multiplicative correction
    correction = (1.0 + delta_kk + delta_gut + delta_brane) / Z

    return correction


def yukawa_with_thresholds(k_i, k_j, tau, g_s, eta_func):
    """
    Full Yukawa coupling including all threshold corrections.

    Starting from tree-level modular form prediction,
    add all string-scale threshold corrections.

    Parameters
    ----------
    k_i, k_j : int
        Modular weights
    tau : complex
        Modular parameter
    g_s : float
        String coupling
    eta_func : callable
        Dedekind eta function

    Returns
    -------
    Y_full : complex
        Yukawa coupling with all corrections
    """
    # Tree-level from modular forms
    eta = eta_func(tau, n_terms=50)
    k_avg = (k_i + k_j) / 2
    Y_tree = np.abs(eta)**(k_avg / 2)

    # Modular phase
    phase = np.exp(2j * np.pi * (k_i - k_j) * np.angle(eta) / 24)

    # Apply threshold corrections
    threshold_factor = total_threshold_correction(k_i, k_j, tau, g_s)

    Y_full = Y_tree * phase * threshold_factor

    return Y_full


def mass_ratios_with_thresholds(k_pattern, tau, g_s, eta_func):
    """
    Compute mass ratios including all threshold corrections.

    This gives much more precise predictions than tree-level alone.

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
        (m2/m1, m3/m1) with threshold corrections
    """
    k = np.array(k_pattern)

    # Diagonal Yukawas with thresholds
    Y1 = yukawa_with_thresholds(k[0], k[0], tau, g_s, eta_func)
    Y2 = yukawa_with_thresholds(k[1], k[1], tau, g_s, eta_func)
    Y3 = yukawa_with_thresholds(k[2], k[2], tau, g_s, eta_func)

    # Mass eigenvalues ~ |Y_ii|
    m1 = np.abs(Y1)
    m2 = np.abs(Y2)
    m3 = np.abs(Y3)

    return m2 / m1, m3 / m1
