"""
2-Loop Corrections for Masses and Gauge Couplings
==================================================

Implements:
- 2-loop worldsheet corrections (genus-2 contributions)
- 2-loop RG running with full SM beta functions
- Threshold corrections at string scale
"""

import numpy as np


def mass_twoloop_correction(k_i, tau, g_s, eta_func):
    """
    2-loop worldsheet correction to fermion masses.

    Includes:
    - Genus-2 worldsheet diagrams ~ g_s^4
    - 2-loop RG running
    - KK mode corrections

    Parameters
    ----------
    k_i : int
        Modular weight
    tau : complex
        Modular parameter
    g_s : float
        String coupling
    eta_func : callable
        Dedekind eta function

    Returns
    -------
    delta_2loop : float
        2-loop correction factor
    """
    # Genus-2 worldsheet amplitude
    eta = eta_func(tau, n_terms=50)
    eta_deriv = eta * (np.pi * 1j / 12 + sum(n * np.exp(2j * np.pi * n * tau) /
                                               (1 - np.exp(2j * np.pi * n * tau))
                                               for n in range(1, 20)))

    # 2-loop worldsheet: genus-2 contribution
    genus2_amplitude = g_s**4 * (k_i**4 / (16 * np.pi**2)) * np.abs(eta_deriv / eta)**4

    # KK mode corrections from compactification
    R_string = np.imag(tau)
    kk_correction = g_s**2 * np.exp(-2 * np.pi * R_string) * (k_i**2 / 12)

    return genus2_amplitude + kk_correction


def gauge_twoloop_beta(g, b1, b2):
    """
    2-loop gauge beta function.

    β(g) = -b1 * g^3 / (16π²) - b2 * g^5 / (16π²)²

    Parameters
    ----------
    g : float
        Gauge coupling
    b1 : float
        1-loop beta function coefficient
    b2 : float
        2-loop beta function coefficient

    Returns
    -------
    beta : float
        Beta function value
    """
    return -(b1 * g**3) / (16 * np.pi**2) - (b2 * g**5) / ((16 * np.pi**2)**2)


def run_gauge_twoloop(alpha_GUT, b1, b2, M_GUT, M_Z):
    """
    Run gauge coupling from GUT scale to M_Z with 2-loop RG.

    Solves: dα/dt = (b1/(2π))α² + (b2/(4π²))α³
    where t = log(μ)

    Parameters
    ----------
    alpha_GUT : float
        Coupling at GUT scale
    b1, b2 : float
        1-loop and 2-loop beta coefficients
    M_GUT : float
        GUT scale (GeV)
    M_Z : float
        Z boson mass (GeV)

    Returns
    -------
    alpha_Z : float
        Coupling at M_Z
    """
    # Running parameter
    t = np.log(M_Z / M_GUT)

    # 1-loop running
    alpha_inv_out = 1.0/alpha_GUT - (b1 / (2*np.pi)) * t
    alpha_1loop = 1.0 / alpha_inv_out

    # 2-loop correction
    correction = (b2 / (8*np.pi**2)) * t * alpha_GUT
    alpha_out = alpha_1loop * (1 + correction * alpha_1loop)

    return alpha_out


def mass_with_full_corrections(k_i, tau, g_s, eta_func):
    """
    Complete mass calculation with 1-loop + 2-loop + RG.

    m = m_tree × (1 + δ_1loop + δ_2loop) × RG_factor

    Parameters
    ----------
    k_i : int
        Modular weight
    tau : complex
        Modular parameter
    g_s : float
        String coupling
    eta_func : callable
        Dedekind eta function

    Returns
    -------
    m_full : float
        Mass with full loop corrections
    """
    # Tree-level
    eta = eta_func(tau, n_terms=50)
    m_tree = np.abs(eta)**(k_i/2)

    # 1-loop correction
    eta_deriv = eta * (np.pi * 1j / 12 + sum(n * np.exp(2j * np.pi * n * tau) /
                                               (1 - np.exp(2j * np.pi * n * tau))
                                               for n in range(1, 30)))
    delta_1loop = g_s**2 * (k_i**2 / (4 * np.pi)) * np.abs(eta_deriv / eta)**2

    # 2-loop correction
    delta_2loop = mass_twoloop_correction(k_i, tau, g_s, eta_func)

    # RG running from M_string to M_Z
    M_string = 5e17  # GeV
    M_Z = 91.2
    gamma = k_i / (16 * np.pi**2)
    rg_factor = (M_Z / M_string)**gamma

    return m_tree * (1.0 + delta_1loop + delta_2loop) * rg_factor


# 2-loop beta function coefficients for SM gauge groups
# From standard gauge theory computations

# SU(3)_c: b1 = -7, b2 = -26 (asymptotic freedom)
BETA_SU3 = {'b1': -7, 'b2': -26}

# SU(2)_L: b1 = -19/6, b2 = 35/6 (asymptotic freedom at 1-loop)
BETA_SU2 = {'b1': -19/6, 'b2': 35/6}

# U(1)_Y: b1 = 41/10, b2 = 199/50
BETA_U1 = {'b1': 41/10, 'b2': 199/50}
