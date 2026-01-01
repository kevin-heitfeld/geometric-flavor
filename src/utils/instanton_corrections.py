"""
Non-Perturbative Instanton Corrections
=======================================

Implements:
- Worldsheet instantons (membrane wrapping cycles)
- D-brane instantons
- CP-violating phases from instanton effects
"""

import numpy as np


def worldsheet_instanton_action(k_i, tau, n_wrapping=1):
    """
    Worldsheet instanton action S_inst for wrapped membrane.

    S_inst = (2π/g_s) * k_i * n_wrapping * Im[τ]

    Parameters
    ----------
    k_i : int
        Modular weight (wrapping number)
    tau : complex
        Modular parameter
    n_wrapping : int
        Number of times membrane wraps the cycle

    Returns
    -------
    S_inst : float
        Instanton action
    """
    g_s = np.exp(-np.log(np.imag(tau)))
    return (2 * np.pi / g_s) * k_i * n_wrapping * np.imag(tau)


def instanton_contribution(k_i, tau, n_max=3):
    """
    Sum over multi-instanton contributions.

    δ_inst = Σ A_n * exp(-n * S_inst)

    Parameters
    ----------
    k_i : int
        Modular weight
    tau : complex
        Modular parameter
    n_max : int
        Maximum instanton number to sum

    Returns
    -------
    delta_inst : complex
        Total instanton correction (complex for CP violation)
    """
    S_0 = worldsheet_instanton_action(k_i, tau, n_wrapping=1)

    delta = 0.0j
    for n in range(1, n_max + 1):
        # Instanton amplitude with phase from fermion zero modes
        # Phase depends on k_i to give different CKM phases
        phase = np.exp(2j * np.pi * k_i * n / 12)

        # Amplitude ~ 1/n for multi-instantons
        amplitude = 1.0 / n

        delta += amplitude * phase * np.exp(-n * S_0)

    return delta


def cp_phase_from_instantons(k_up, k_down, tau):
    """
    CP-violating phase for CKM from instanton interference.

    δ_CP ~ arg(I_up * I_down*)

    Parameters
    ----------
    k_up : int
        Up-type quark modular weight
    k_down : int
        Down-type quark modular weight
    tau : complex
        Modular parameter

    Returns
    -------
    delta_CP : float
        CP-violating phase (radians)
    """
    I_up = instanton_contribution(k_up, tau, n_max=3)
    I_down = instanton_contribution(k_down, tau, n_max=3)

    # CP phase from interference
    interference = I_up * np.conj(I_down)

    return np.angle(interference)


def yukawa_with_instantons(k_i, k_j, tau, eta_func):
    """
    Yukawa coupling with instanton corrections.

    Y_ij = Y_ij^tree * (1 + δ_loop + δ_inst)

    Parameters
    ----------
    k_i, k_j : int
        Modular weights for generations i, j
    tau : complex
        Modular parameter
    eta_func : callable
        Dedekind eta function

    Returns
    -------
    Y_ij : complex
        Yukawa coupling with instantons
    """
    # Tree-level
    eta = eta_func(tau, n_terms=50)
    Y_tree = eta**((k_i + k_j) / 2)

    # Instanton corrections (different for each generation)
    delta_i = instanton_contribution(k_i, tau, n_max=3)
    delta_j = instanton_contribution(k_j, tau, n_max=3)

    # Combined correction
    delta_inst = (delta_i + delta_j) / 2

    return Y_tree * (1.0 + delta_inst)


def ckm_phase_corrections(k_pattern_up, k_pattern_down, tau):
    """
    CP phases for all CKM elements from instantons.

    Returns 3x3 matrix of phases.

    Parameters
    ----------
    k_pattern_up : array_like
        [k_u, k_c, k_t] modular weights
    k_pattern_down : array_like
        [k_d, k_s, k_b] modular weights
    tau : complex
        Modular parameter

    Returns
    -------
    phases : ndarray (3, 3)
        CP-violating phases for V_ij
    """
    phases = np.zeros((3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            phases[i, j] = cp_phase_from_instantons(
                k_pattern_up[i],
                k_pattern_down[j],
                tau
            )

    return phases


# Typical instanton scales
INSTANTON_SCALE = {
    'worldsheet': 2 * np.pi,  # 2π from action
    'D-brane': 4 * np.pi**2,  # 4π² from D-brane action
    'membrane': 6 * np.pi**2  # 6π² from M5-brane
}
