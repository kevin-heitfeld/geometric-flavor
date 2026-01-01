"""
Optimize generation-dependent τ_i to minimize prediction errors

This fits τ_1, τ_2, τ_3 to observations, then we try to derive them from geometry.
"""

import numpy as np
from scipy.optimize import minimize
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def dedekind_eta(tau, n_terms=50):
    """Dedekind eta η(τ) = q^(1/24) ∏(1-q^n)"""
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24)
    for n in range(1, n_terms):
        eta *= (1 - q**n)
    return eta

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
    """Mass with wavefunction localization"""
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

def compute_mass_ratios(tau_values, k_pattern, A_pattern, g_s):
    """
    Compute mass ratios given τ values for each generation

    Parameters:
    -----------
    tau_values : array [Im[τ_1], Im[τ_2], Im[τ_3]]
        Imaginary parts of modular parameters (keep Re[τ]=0)
    k_pattern, A_pattern, g_s : arrays
        Theory parameters

    Returns:
    --------
    ratios : array [m_1/m_1, m_2/m_1, m_3/m_1]
    """
    masses = np.zeros(3)

    for i in range(3):
        tau_i = 1j * tau_values[i]
        masses[i] = np.real(mass_with_localization(k_pattern[i], tau_i,
                                                    A_pattern[i], g_s, dedekind_eta))

    return masses / masses[0]

def objective_function(tau_values, k_pattern, A_pattern, g_s, observed_ratios):
    """
    Objective to minimize: mean squared log error

    Using log scale because mass hierarchies span orders of magnitude
    """
    try:
        predicted = compute_mass_ratios(tau_values, k_pattern, A_pattern, g_s)

        # Log scale errors (skip first which is always 1)
        log_errors = np.log10(predicted[1:]) - np.log10(observed_ratios[1:])

        return np.mean(log_errors**2)
    except:
        return 1e10  # Penalty for invalid parameters

def fit_generation_tau(k_pattern, A_pattern, g_s, observed_ratios, tau_0=2.7):
    """
    Fit τ_i values to minimize errors

    Returns:
    --------
    tau_opt : array [τ_1, τ_2, τ_3]
        Optimal modular parameters
    error : float
        Final average error (%)
    """
    # Initial guess: all equal to τ_0 (should reproduce current results)
    tau_init = np.array([tau_0, tau_0, tau_0])

    print(f"  Initial guess: τ = [{tau_init[0]:.2f}i, {tau_init[1]:.2f}i, {tau_init[2]:.2f}i]")

    # Check initial error
    initial_ratios = compute_mass_ratios(tau_init, k_pattern, A_pattern, g_s)
    initial_errors = np.abs(initial_ratios - observed_ratios) / observed_ratios * 100
    initial_avg = np.mean(initial_errors[1:])
    print(f"  Initial error: {initial_avg:.1f}%")
    print(f"  Initial ratios: [{initial_ratios[0]:.1f}, {initial_ratios[1]:.1f}, {initial_ratios[2]:.1f}]")
    print(f"  Observed: [{observed_ratios[0]:.1f}, {observed_ratios[1]:.1f}, {observed_ratios[2]:.1f}]")

    # Optimize
    print(f"  Optimizing...")
    result = minimize(
        objective_function,
        tau_init,
        args=(k_pattern, A_pattern, g_s, observed_ratios),
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-4}
    )

    tau_opt = result.x

    # Compute final error
    final_ratios = compute_mass_ratios(tau_opt, k_pattern, A_pattern, g_s)
    final_errors = np.abs(final_ratios - observed_ratios) / observed_ratios * 100
    final_avg = np.mean(final_errors[1:])

    print(f"  Optimal: τ = [{tau_opt[0]:.2f}i, {tau_opt[1]:.2f}i, {tau_opt[2]:.2f}i]")
    print(f"  Final error: {final_avg:.1f}%")
    print(f"  Final ratios: [{final_ratios[0]:.1f}, {final_ratios[1]:.1f}, {final_ratios[2]:.1f}]")
    print(f"  Improvement: {initial_avg - final_avg:.1f} percentage points")

    return tau_opt, final_avg, final_ratios

# ============================================================================
# MAIN FITTING
# ============================================================================

print("="*80)
print("FITTING GENERATION-DEPENDENT τ_i TO OBSERVATIONS")
print("="*80)
print()

tau_0 = 2.7
g_s = np.exp(-np.log(tau_0))

k_mass = np.array([8, 6, 4])
A_leptons = np.array([0.00, -0.80, -1.00])
A_up = np.array([0.00, -1.00, -1.60])
A_down = np.array([0.00, -0.20, -0.80])

r_lep_obs = np.array([1.0, 206.8, 3477])
r_up_obs = np.array([1.0, 577, 78636])
r_down_obs = np.array([1.0, 18.3, 890])

print(f"Base theory: τ₀ = {tau_0}i, g_s = {g_s:.4f}")
print(f"k-pattern: {k_mass}")
print()

sectors = [
    ("LEPTONS", A_leptons, r_lep_obs),
    ("UP QUARKS", A_up, r_up_obs),
    ("DOWN QUARKS", A_down, r_down_obs),
]

results = {}

for sector_name, A_pattern, obs_ratios in sectors:
    print(f"{sector_name}:")
    print("-"*80)

    tau_opt, error, ratios = fit_generation_tau(k_mass, A_pattern, g_s, obs_ratios, tau_0)

    results[sector_name] = {
        'tau': tau_opt,
        'error': error,
        'ratios': ratios
    }

    print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("SUMMARY OF FITTED τ_i VALUES")
print("="*80)
print()

for sector_name, data in results.items():
    tau = data['tau']
    error = data['error']

    print(f"{sector_name}:")
    print(f"  τ_1 = {tau[0]:.3f}i  (1st generation)")
    print(f"  τ_2 = {tau[1]:.3f}i  (2nd generation)")
    print(f"  τ_3 = {tau[2]:.3f}i  (3rd generation)")
    print(f"  Final error: {error:.1f}%")
    print()

print("Next: Look for patterns in fitted τ_i to derive from geometry")
