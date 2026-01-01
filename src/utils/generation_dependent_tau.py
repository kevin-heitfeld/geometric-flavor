"""
Generation-dependent modular parameters τ_i

Key insight: Different generations may live at different positions in moduli space,
leading to generation-dependent τ values:

τ_1 = τ_0               (1st generation, lightest)
τ_2 = τ_0 + δτ_2        (2nd generation)
τ_3 = τ_0 + δτ_3        (3rd generation, heaviest)

Physical origin:
- Different D-brane positions in compact space
- Different flux backgrounds
- Wilson lines on torus

This improves mass hierarchies without changing the fundamental prediction τ_0 = 2.7i
"""

import numpy as np

def generation_tau(tau_0, generation=0, delta_scheme='linear'):
    """
    Compute generation-dependent modular parameter

    Parameters:
    -----------
    tau_0 : complex
        Base modular parameter (τ = 2.7i for topology)
    generation : int
        Generation index (0, 1, 2) for (1st, 2nd, 3rd)
    delta_scheme : str
        'linear': δτ_i ∝ i
        'quadratic': δτ_i ∝ i²
        'exponential': τ_i = τ_0 × r^i

    Returns:
    --------
    tau_i : complex
        Modular parameter for generation i
    """
    if generation == 0:
        return tau_0

    if delta_scheme == 'linear':
        # Linear spacing: reasonable first approximation
        # Physical: uniform Wilson line shifts
        delta_real = 0.0  # Keep Re[τ] = 0 for now (CP conservation)
        delta_imag = 0.3 * generation  # Empirical: improves hierarchies
        return tau_0 + delta_real + 1j * delta_imag

    elif delta_scheme == 'quadratic':
        # Quadratic spacing: stronger hierarchy
        # Physical: flux quantization n² effects
        delta_real = 0.0
        delta_imag = 0.15 * generation**2
        return tau_0 + delta_real + 1j * delta_imag

    elif delta_scheme == 'exponential':
        # Exponential ratio: τ_i = τ_0 × r^i
        # Physical: RG flow effects
        r = 1.15  # Empirical ratio
        return tau_0 * (r ** generation)

    else:
        raise ValueError(f"Unknown delta_scheme: {delta_scheme}")

def eta_derivative(tau, eta_func, n_terms=50):
    """∂_τ η for 1-loop corrections"""
    q = np.exp(2j * np.pi * tau)
    eta = eta_func(tau, n_terms)
    d_log_eta = np.pi * 1j / 12.0
    for n in range(1, n_terms):
        qn = q**n
        d_log_eta += 2j * np.pi * n * qn / (1 - qn)
    return eta * d_log_eta

def mass_with_localization(k_i, tau, A_i, g_s, eta_func):
    """
    Mass with wavefunction localization:
    m_i ~ |η^(k/2)|² × exp(-2 A_i Im[τ]) × [1 + loop] × RG
    """
    eta = eta_func(tau)
    modular = np.abs(eta ** (k_i / 2.0))**2
    localization = np.exp(-2.0 * A_i * np.imag(tau))

    d_eta = eta_derivative(tau, eta_func)
    loop_corr = g_s**2 * (k_i**2 / (4 * np.pi)) * np.abs(d_eta / eta)**2

    M_string = 5e17
    M_Z = 91.2
    gamma_anom = k_i / (16 * np.pi**2)
    rg_factor = (M_Z / M_string)**(gamma_anom)

    return modular * localization * (1.0 + loop_corr) * rg_factor

def mass_ratio_with_generation_tau(k_pattern, tau_0, g_s, eta_func,
                                    A_pattern, delta_scheme='linear'):
    """
    Compute mass ratios with generation-dependent τ

    This improves hierarchies by allowing each generation to have
    slightly different modular parameter (different brane position).

    Parameters:
    -----------
    k_pattern : array [k_1, k_2, k_3]
        Modular weights for 3 generations
    tau_0 : complex
        Base modular parameter
    g_s : float
        String coupling
    eta_func : callable
        Dedekind eta function
    A_pattern : array [A_1, A_2, A_3]
        Localization parameters
    delta_scheme : str
        Scheme for τ variation

    Returns:
    --------
    masses : array [m_1, m_2, m_3]
        Mass values (dimensionless, need Y₀ × v for physical masses)
    """
    masses = np.zeros(3, dtype=complex)

    for i in range(3):
        tau_i = generation_tau(tau_0, generation=i, delta_scheme=delta_scheme)
        masses[i] = mass_with_localization(k_pattern[i], tau_i, A_pattern[i],
                                           g_s, eta_func)

    # Return real parts (masses are positive)
    return np.real(masses)

def optimize_delta_scheme(k_pattern, tau_0, g_s, eta_func, A_pattern,
                          observed_ratios, schemes=['linear', 'quadratic', 'exponential']):
    """
    Find best δτ scheme by comparing to observations

    Parameters:
    -----------
    k_pattern : array
        Modular weights
    tau_0 : complex
        Base tau
    g_s : float
        String coupling
    eta_func : callable
        Dedekind eta
    A_pattern : array
        Localization parameters
    observed_ratios : array [1, r_2, r_3]
        Observed mass ratios m_i/m_1
    schemes : list
        Schemes to test

    Returns:
    --------
    best_scheme : str
        Best scheme name
    best_error : float
        Average error (%)
    results : dict
        All scheme results
    """
    results = {}

    for scheme in schemes:
        masses = mass_ratio_with_generation_tau(k_pattern, tau_0, g_s,
                                                eta_func, A_pattern, scheme)
        ratios = masses / masses[0]

        # Compute errors
        errors = np.abs(ratios - observed_ratios) / observed_ratios * 100
        avg_error = np.mean(errors[1:])  # Skip m_1/m_1 = 1

        results[scheme] = {
            'ratios': ratios,
            'errors': errors,
            'avg_error': avg_error
        }

    # Find best
    best_scheme = min(results.keys(), key=lambda s: results[s]['avg_error'])
    best_error = results[best_scheme]['avg_error']

    return best_scheme, best_error, results

def print_scheme_comparison(results, observed_ratios):
    """Print comparison of different δτ schemes"""
    print("Generation-dependent τ schemes:")
    print("-" * 60)
    print(f"{'Scheme':<15} {'m₂/m₁':<12} {'m₃/m₁':<12} {'Avg Error':<12}")
    print("-" * 60)

    for scheme, data in results.items():
        ratios = data['ratios']
        avg_err = data['avg_error']
        print(f"{scheme:<15} {ratios[1]:>8.1f}    {ratios[2]:>8.1f}    {avg_err:>6.1f}%")

    print()
    print(f"Observed:       {observed_ratios[1]:>8.1f}    {observed_ratios[2]:>8.1f}")
    print("-" * 60)
