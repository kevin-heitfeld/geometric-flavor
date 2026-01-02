"""
Optimize localization parameters A_i for generation-dependent τ model

Model: τ_i^sector = τ₀ × c_sector × g_i
where c_sector = 13/14 (lep), 19/20 (up), 7/9 (down)
      g_i = [1.0, 1.0612, 1.0139]

Find A_i that minimize mass ratio errors
"""

import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.loop_corrections import mass_with_full_corrections

def dedekind_eta(tau, n_terms=50):
    """Dedekind eta η(τ) = q^(1/24) ∏(1-q^n)"""
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24)
    for n in range(1, n_terms):
        eta *= (1 - q**n)
    return eta

def eta_derivative(tau, n_terms=50):
    """∂_τ η for 1-loop corrections"""
    q = np.exp(2j * np.pi * tau)
    eta = dedekind_eta(tau, n_terms)
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

    d_eta = eta_derivative(tau)
    loop_corr = g_s**2 * (k_i**2 / (4 * np.pi)) * np.abs(d_eta / eta)**2

    M_string = 5e17
    M_Z = 91.2
    gamma_anom = k_i / (16 * np.pi**2)
    rg_factor = (M_Z / M_string)**(gamma_anom)

    return modular * localization * (1.0 + loop_corr) * rg_factor

# Parameters
tau_0 = 2.7j
phi_dilaton = -np.log(np.imag(tau_0))  # CORRECT formula
g_s = np.exp(phi_dilaton)

print(f"DEBUG: tau_0 = {tau_0}")
print(f"DEBUG: phi_dilaton = {phi_dilaton}")
print(f"DEBUG: g_s = {g_s}")
print()

# Sector constants from geometry
c_lep = 13/14
c_up = 19/20
c_down = 7/9

# k-pattern for masses
k_mass = np.array([8, 6, 4])

# Observed mass ratios
r_lep_obs = np.array([1.0, 206.8, 3477])
r_up_obs = np.array([1.0, 577, 78636])
r_down_obs = np.array([1.0, 20.3, 890])  # Fixed: m_s/m_d = 95/4.67 = 20.3, not 18.3

print("="*80)
print("OPTIMIZING τ AND A_i PARAMETERS (SECTOR-DEPENDENT GENERATION FACTORS)")
print("="*80)
print()

# Optimization function
def objective(params):
    """Minimize MAXIMUM relative error (minimax optimization)

    Parameters:
    - params[0:2]: g_lep[1:3] (generation factors for leptons)
    - params[2:4]: g_up[1:3]
    - params[4:6]: g_down[1:3]
    - params[6:8]: A_lep[1:3]
    - params[8:10]: A_up[1:3]
    - params[10:12]: A_down[1:3]
    """
    # Extract generation factors (first generation always 1.0)
    g_lep = np.array([1.0, params[0], params[1]])
    g_up = np.array([1.0, params[2], params[3]])
    g_down = np.array([1.0, params[4], params[5]])

    # Construct τ values
    tau_lep = tau_0 * c_lep * g_lep
    tau_up = tau_0 * c_up * g_up
    tau_down = tau_0 * c_down * g_down

    # Extract localization parameters
    A_lep = np.array([0.0, params[6], params[7]])
    A_up = np.array([0.0, params[8], params[9]])
    A_down = np.array([0.0, params[10], params[11]])

    # Compute masses
    m_lep = np.array([mass_with_localization(k_mass[i], tau_lep[i], A_lep[i], g_s, dedekind_eta)
                      for i in range(3)])
    m_up = np.array([mass_with_localization(k_mass[i], tau_up[i], A_up[i], g_s, dedekind_eta)
                     for i in range(3)])
    m_down = np.array([mass_with_localization(k_mass[i], tau_down[i], A_down[i], g_s, dedekind_eta)
                       for i in range(3)])

    # Normalize to lightest
    r_lep = m_lep / m_lep[0]
    r_up = m_up / m_up[0]
    r_down = m_down / m_down[0]

    # Relative errors
    err_lep = np.abs(r_lep - r_lep_obs) / r_lep_obs
    err_up = np.abs(r_up - r_up_obs) / r_up_obs
    err_down = np.abs(r_down - r_down_obs) / r_down_obs

    # Return MAXIMUM error (minimax optimization)
    max_error = np.max(np.concatenate([err_lep, err_up, err_down]))

    return max_error

# Initial guess: generation factors + localization
# g_lep[1:3], g_up[1:3], g_down[1:3], A_lep[1:3], A_up[1:3], A_down[1:3]
x0 = [1.06, 1.01,  # g_lep
      1.06, 1.01,  # g_up
      1.06, 1.01,  # g_down
      -0.75, -0.89,  # A_lep
      -0.91, -1.49,  # A_up
      -0.31, -0.91]  # A_down

print("Optimizing generation factors g_i and localization A_i...")
print(f"Initial guess:")
print(f"  g_lep  = [1.00, {x0[0]:.2f}, {x0[1]:.2f}]")
print(f"  g_up   = [1.00, {x0[2]:.2f}, {x0[3]:.2f}]")
print(f"  g_down = [1.00, {x0[4]:.2f}, {x0[5]:.2f}]")
print(f"  A_lep  = [0.00, {x0[6]:.2f}, {x0[7]:.2f}]")
print(f"  A_up   = [0.00, {x0[8]:.2f}, {x0[9]:.2f}]")
print(f"  A_down = [0.00, {x0[10]:.2f}, {x0[11]:.2f}]")
print()

result = minimize(objective, x0, method='Nelder-Mead',
                 options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})

print("Optimization complete!")
print(f"Success: {result.success}")
print(f"Iterations: {result.nit}")
print()

# Extract optimized parameters
g_lep_opt = np.array([1.0, result.x[0], result.x[1]])
g_up_opt = np.array([1.0, result.x[2], result.x[3]])
g_down_opt = np.array([1.0, result.x[4], result.x[5]])
A_lep_opt = np.array([0.0, result.x[6], result.x[7]])
A_up_opt = np.array([0.0, result.x[8], result.x[9]])
A_down_opt = np.array([0.0, result.x[10], result.x[11]])

# Construct optimized τ values
tau_lep_opt = tau_0 * c_lep * g_lep_opt
tau_up_opt = tau_0 * c_up * g_up_opt
tau_down_opt = tau_0 * c_down * g_down_opt

print("OPTIMIZED GENERATION FACTORS:")
print(f"  g_leptons = {g_lep_opt}")
print(f"  g_up      = {g_up_opt}")
print(f"  g_down    = {g_down_opt}")
print()

print("OPTIMIZED τ VALUES:")
print(f"  τ_leptons = {tau_lep_opt}")
print(f"  τ_up      = {tau_up_opt}")
print(f"  τ_down    = {tau_down_opt}")
print()

print("OPTIMIZED LOCALIZATION PARAMETERS:")
print(f"  A_leptons = {A_lep_opt}")
print(f"  A_up      = {A_up_opt}")
print(f"  A_down    = {A_down_opt}")
print()

# Compute final predictions
m_lep_opt = np.array([mass_with_localization(k_mass[i], tau_lep_opt[i], A_lep_opt[i], g_s, dedekind_eta)
                      for i in range(3)])
m_up_opt = np.array([mass_with_localization(k_mass[i], tau_up_opt[i], A_up_opt[i], g_s, dedekind_eta)
                     for i in range(3)])
m_down_opt = np.array([mass_with_localization(k_mass[i], tau_down_opt[i], A_down_opt[i], g_s, dedekind_eta)
                       for i in range(3)])

r_lep_opt = m_lep_opt / m_lep_opt[0]
r_up_opt = m_up_opt / m_up_opt[0]
r_down_opt = m_down_opt / m_down_opt[0]

print("PREDICTED MASS RATIOS:")
print(f"  Leptons:  {r_lep_opt}")
print(f"  Observed: {r_lep_obs}")
err_lep_opt = np.abs(r_lep_opt - r_lep_obs) / r_lep_obs * 100
print(f"  Errors:   {err_lep_opt[1]:.1f}%, {err_lep_opt[2]:.1f}%")
print()

print(f"  Up-type:  {r_up_opt}")
print(f"  Observed: {r_up_obs}")
err_up_opt = np.abs(r_up_opt - r_up_obs) / r_up_obs * 100
print(f"  Errors:   {err_up_opt[1]:.1f}%, {err_up_opt[2]:.1f}%")
print()

print(f"  Down-type: {r_down_opt}")
print(f"  Observed: {r_down_obs}")
err_down_opt = np.abs(r_down_opt - r_down_obs) / r_down_obs * 100
print(f"  Errors:   {err_down_opt[1]:.1f}%, {err_down_opt[2]:.1f}%")
print()

avg_error = np.mean(np.concatenate([err_lep_opt[1:], err_up_opt[1:], err_down_opt[1:]]))
max_error = np.max(np.concatenate([err_lep_opt[1:], err_up_opt[1:], err_down_opt[1:]]))
print(f"AVERAGE ERROR: {avg_error:.1f}%")
print(f"MAXIMUM ERROR: {max_error:.1f}%")
print()

print("="*80)
print("TO USE THESE VALUES, UPDATE unified_predictions_complete.py:")
print("="*80)
print(f"# Generation factors (sector-dependent)")
print(f"g_lep = np.array([1.00, {g_lep_opt[1]:.8f}, {g_lep_opt[2]:.8f}])")
print(f"g_up = np.array([1.00, {g_up_opt[1]:.8f}, {g_up_opt[2]:.8f}])")
print(f"g_down = np.array([1.00, {g_down_opt[1]:.8f}, {g_down_opt[2]:.8f}])")
print()
print(f"# Localization parameters")
print(f"A_leptons = np.array([0.00, {A_lep_opt[1]:.8f}, {A_lep_opt[2]:.8f}])")
print(f"A_up = np.array([0.00, {A_up_opt[1]:.8f}, {A_up_opt[2]:.8f}])")
print(f"A_down = np.array([0.00, {A_down_opt[1]:.8f}, {A_down_opt[2]:.8f}])")
print()
