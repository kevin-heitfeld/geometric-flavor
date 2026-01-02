"""
Lightweight wrapper for computing observables with custom k_mass.
Extracts just the necessary functions without running the full script.
"""

import numpy as np

def dedekind_eta(tau, n_terms=50):
    """Dedekind eta η(τ) = q^(1/24) ∏(1-q^n)"""
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24)
    for n in range(1, n_terms):
        eta *= (1 - q**n)
    return eta

def mass_with_localization(k_i, tau, A_i, g_s, eta_func):
    """Mass with wavefunction localization"""
    eta_val = eta_func(tau)
    Im_tau = tau.imag if hasattr(tau, 'imag') else abs(tau)

    # Base mass: m ~ |η(τ)|^k / (Im τ)^2
    mass_factor = np.abs(eta_val)**k_i / (Im_tau**2)

    # Localization suppression: exp(-A_i × d_i/ℓ_s)
    localization = np.exp(-A_i) if A_i != 0 else 1.0

    return mass_factor * localization

def compute_observables_with_kmass(tau_value, g_s_value, k_mass_override, verbose=False):
    """
    Compute all observables with custom k_mass pattern.

    Parameters:
    -----------
    tau_value : complex
        Kähler modulus value
    g_s_value : float
        String coupling
    k_mass_override : array-like
        Custom k_mass pattern [k_lep, k_up, k_down]
    verbose : bool
        Print detailed output

    Returns:
    --------
    dict with keys:
        'quark_masses': [m_u, m_c, m_t, m_d, m_s, m_b] in GeV
        'lepton_masses': [m_e, m_mu, m_tau] in GeV
        'mass_ratios_up': [m_c/m_u, m_t/m_u]
        'mass_ratios_down': [m_s/m_d, m_b/m_d]
        'mass_ratios_lep': [m_mu/m_e, m_tau/m_e]
        'chi_squared': overall fit quality
    """

    # Use cached generation factors and localization
    g_lep_local = np.array([1.00, 1.10599770, 1.00816488])
    g_up_local = np.array([1.00, 1.12996338, 1.01908896])
    g_down_local = np.array([1.00, 0.96185547, 1.00057316])
    A_leptons_local = np.array([0.00, -0.72084622, -0.92315966])
    A_up_local = np.array([0.00, -0.87974875, -1.48332060])
    A_down_local = np.array([0.00, -0.33329575, -0.88288836])

    # Sector constants
    c_lep_local = 13/14
    c_up_local = 19/20
    c_down_local = 7/9

    # Compute sector-dependent tau values
    tau_lep_local = tau_value * c_lep_local * g_lep_local
    tau_up_local = tau_value * c_up_local * g_up_local
    tau_down_local = tau_value * c_down_local * g_down_local

    # Compute masses with override k_mass
    k_mass_arr = np.array(k_mass_override)
    m_lep_local = np.array([mass_with_localization(k_mass_arr[i], tau_lep_local[i],
                                                    A_leptons_local[i], g_s_value, dedekind_eta)
                           for i in range(3)])
    m_up_local = np.array([mass_with_localization(k_mass_arr[i], tau_up_local[i],
                                                   A_up_local[i], g_s_value, dedekind_eta)
                          for i in range(3)])
    m_down_local = np.array([mass_with_localization(k_mass_arr[i], tau_down_local[i],
                                                     A_down_local[i], g_s_value, dedekind_eta)
                            for i in range(3)])

    # Compute mass ratios
    ratio_lep = m_lep_local / m_lep_local[0]
    ratio_up = m_up_local / m_up_local[0]
    ratio_down = m_down_local / m_down_local[0]

    # Observed ratios
    m_e_obs, m_mu_obs, m_tau_obs = 0.511e-3, 0.1057, 1.777  # GeV
    m_u_obs, m_c_obs, m_t_obs = 0.00216, 1.27, 173.0  # GeV
    m_d_obs, m_s_obs, m_b_obs = 0.00467, 0.095, 4.18  # GeV

    ratio_lep_obs = np.array([1, m_mu_obs/m_e_obs, m_tau_obs/m_e_obs])
    ratio_up_obs = np.array([1, m_c_obs/m_u_obs, m_t_obs/m_u_obs])
    ratio_down_obs = np.array([1, m_s_obs/m_d_obs, m_b_obs/m_d_obs])

    # Compute chi-squared (only ratios, ignore absolute scale)
    chi2_lep = np.sum(((ratio_lep[1:] - ratio_lep_obs[1:]) / ratio_lep_obs[1:])**2)
    chi2_up = np.sum(((ratio_up[1:] - ratio_up_obs[1:]) / ratio_up_obs[1:])**2)
    chi2_down = np.sum(((ratio_down[1:] - ratio_down_obs[1:]) / ratio_down_obs[1:])**2)
    chi2_total = chi2_lep + chi2_up + chi2_down

    if verbose:
        print(f"\nk_mass = {k_mass_override}")
        print(f"  Lepton ratios: {ratio_lep[1:]} (obs: {ratio_lep_obs[1:]})")
        print(f"  Up ratios: {ratio_up[1:]} (obs: {ratio_up_obs[1:]})")
        print(f"  Down ratios: {ratio_down[1:]} (obs: {ratio_down_obs[1:]})")
        print(f"  χ² = {chi2_total:.3f}")

    return {
        'quark_masses': np.concatenate([m_up_local, m_down_local]),
        'lepton_masses': m_lep_local,
        'mass_ratios_up': ratio_up[1:],
        'mass_ratios_down': ratio_down[1:],
        'mass_ratios_lep': ratio_lep[1:],
        'chi_squared': chi2_total,
        'k_mass': k_mass_arr
    }
