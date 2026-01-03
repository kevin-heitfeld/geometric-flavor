"""
Optimize CKM with REAL parameters only - Simple and Robust
===========================================================

Use only real off-diagonal Yukawa elements.
Complex phases will come from instanton corrections separately.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


# Observed mass ratios
m_up = np.array([1.0, 577.0, 78636.0])
m_down = np.array([1.0, 20.3, 890.0])

# Observed CKM angles
sin2_12_obs = 0.0510
sin2_23_obs = 0.00157
sin2_13_obs = 0.000128


def yukawa_from_params_real(eps_params):
    """Build Yukawas with REAL off-diagonals only."""
    Y_up = np.diag(m_up).astype(complex)
    Y_up[0, 1] = eps_params[0]
    Y_up[1, 0] = Y_up[0, 1]
    Y_up[1, 2] = eps_params[1]
    Y_up[2, 1] = Y_up[1, 2]
    Y_up[0, 2] = eps_params[2]
    Y_up[2, 0] = Y_up[0, 2]

    Y_down = np.diag(m_down).astype(complex)
    Y_down[0, 1] = eps_params[3]
    Y_down[1, 0] = Y_down[0, 1]
    Y_down[1, 2] = eps_params[4]
    Y_down[2, 1] = Y_down[1, 2]
    Y_down[0, 2] = eps_params[5]
    Y_down[2, 0] = Y_down[0, 2]

    return Y_up, Y_down


def ckm_from_yukawas(Y_up, Y_down):
    """Extract CKM via SVD."""
    U_uL, _, _ = np.linalg.svd(Y_up)
    U_dL, _, _ = np.linalg.svd(Y_down)
    V_CKM = U_uL @ U_dL.conj().T

    s12 = np.abs(V_CKM[0, 1])
    s23 = np.abs(V_CKM[1, 2])
    s13 = np.abs(V_CKM[0, 2])

    return s12**2, s23**2, s13**2, V_CKM


def objective(eps_params):
    """Minimax objective."""
    Y_up, Y_down = yukawa_from_params_real(eps_params)
    sin2_12, sin2_23, sin2_13, _ = ckm_from_yukawas(Y_up, Y_down)

    err_12 = abs(sin2_12 - sin2_12_obs) / sin2_12_obs
    err_23 = abs(sin2_23 - sin2_23_obs) / sin2_23_obs
    err_13 = abs(sin2_13 - sin2_13_obs) / sin2_13_obs

    return max(err_12, err_23, err_13)


print("="*80)
print("CKM OPTIMIZATION WITH REAL PARAMETERS")
print("="*80)
print()

# Try global optimization first (differential evolution)
print("Phase 1: Global search with differential evolution...")
bounds = [(-100, 100) for _ in range(6)]  # Wide bounds for all 6 parameters
result_de = differential_evolution(objective, bounds, maxiter=3000,
                                   seed=42, atol=1e-10, tol=1e-10)
print(f"DE complete: max error = {result_de.fun*100:.2f}%")
print()

# Refine with local optimizer
print("Phase 2: Local refinement with L-BFGS-B...")
result = minimize(objective, result_de.x, method='L-BFGS-B',
                 options={'maxiter': 10000, 'ftol': 1e-14})
print(f"Local optimization complete: max error = {result.fun*100:.2f}%")
print()

# Try Nelder-Mead as well
print("Phase 3: Nelder-Mead refinement...")
result2 = minimize(objective, result.x, method='Nelder-Mead',
                  options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
print(f"Nelder-Mead complete: max error = {result2.fun*100:.2f}%")
print()

# Use best result
if result2.fun < result.fun:
    result = result2
    print("Using Nelder-Mead result")
else:
    print("Using L-BFGS-B result")
print()

# Display results
eps_opt = result.x
Y_up_opt, Y_down_opt = yukawa_from_params_real(eps_opt)
sin2_12_opt, sin2_23_opt, sin2_13_opt, V_CKM_opt = ckm_from_yukawas(Y_up_opt, Y_down_opt)

print("OPTIMIZED PARAMETERS:")
print(f"  ε_up:   [{eps_opt[0]:8.4f}, {eps_opt[1]:8.4f}, {eps_opt[2]:8.4f}]")
print(f"  ε_down: [{eps_opt[3]:8.4f}, {eps_opt[4]:8.4f}, {eps_opt[5]:8.4f}]")
print()

print("PREDICTED CKM ANGLES:")
print(f"  sin²θ₁₂ = {sin2_12_opt:.6f} (obs: {sin2_12_obs:.6f})")
print(f"  sin²θ₂₃ = {sin2_23_opt:.6f} (obs: {sin2_23_obs:.6f})")
print(f"  sin²θ₁₃ = {sin2_13_opt:.6f} (obs: {sin2_13_obs:.6f})")
print()

err_12 = abs(sin2_12_opt - sin2_12_obs) / sin2_12_obs * 100
err_23 = abs(sin2_23_opt - sin2_23_obs) / sin2_23_obs * 100
err_13 = abs(sin2_13_opt - sin2_13_obs) / sin2_13_obs * 100

print(f"ERRORS:")
print(f"  sin²θ₁₂: {err_12:.2f}%")
print(f"  sin²θ₂₃: {err_23:.2f}%")
print(f"  sin²θ₁₃: {err_13:.2f}%")
print(f"  Maximum: {max(err_12, err_23, err_13):.2f}%")
print()

print("="*80)
print("TO USE IN unified_predictions_complete.py:")
print("="*80)
print(f"eps_up = np.array([{eps_opt[0]:.8f}, {eps_opt[1]:.8f}, {eps_opt[2]:.8f}])")
print(f"eps_down = np.array([{eps_opt[3]:.8f}, {eps_opt[4]:.8f}, {eps_opt[5]:.8f}])")
