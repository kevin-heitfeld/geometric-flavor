"""
Optimize CP Violation Parameters
==================================

Optimize both magnitudes and phases of Yukawa off-diagonals
to simultaneously match CKM angles AND CP violation.

Strategy:
- Start from optimized real parameters as initial guess
- Allow both magnitudes and phases to vary
- Use minimax objective over all 5 observables (3 angles + 2 CP)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


# Mass ratios
m_up = np.array([1.0, 577.0, 78636.0])
m_down = np.array([1.0, 20.3, 890.0])

# Optimized real Yukawa parameters (starting point)
eps_up_real = np.array([35.47327650, -18.09899180, -12.61395681])
eps_down_real = np.array([3.01805261, -35.66037969, 1.86984141])

# Observed values
sin2_12_obs = 0.0510
sin2_23_obs = 0.00157
sin2_13_obs = 0.000128
delta_CP_obs = 1.22  # radians (~70°)
J_CP_obs = 3.0e-5


def yukawa_with_complex_offdiag(params):
    """
    Build Yukawa matrices with complex off-diagonals.

    Y_ij = diag(m_i) + ε_ij (complex)

    Parameters
    ----------
    params : array [12]
        [Re(ε₁₂_up), Im(ε₁₂_up), Re(ε₂₃_up), Im(ε₂₃_up), Re(ε₁₃_up), Im(ε₁₃_up),
         Re(ε₁₂_down), Im(ε₁₂_down), Re(ε₂₃_down), Im(ε₂₃_down), Re(ε₁₃_down), Im(ε₁₃_down)]
    """
    eps_up = np.array([params[0] + 1j*params[1],
                       params[2] + 1j*params[3],
                       params[4] + 1j*params[5]], dtype=complex)

    eps_down = np.array([params[6] + 1j*params[7],
                         params[8] + 1j*params[9],
                         params[10] + 1j*params[11]], dtype=complex)

    Y_up = np.diag(m_up).astype(complex)
    Y_up[0, 1] = eps_up[0]
    Y_up[1, 0] = eps_up[0]
    Y_up[1, 2] = eps_up[1]
    Y_up[2, 1] = eps_up[1]
    Y_up[0, 2] = eps_up[2]
    Y_up[2, 0] = eps_up[2]

    Y_down = np.diag(m_down).astype(complex)
    Y_down[0, 1] = eps_down[0]
    Y_down[1, 0] = eps_down[0]
    Y_down[1, 2] = eps_down[1]
    Y_down[2, 1] = eps_down[1]
    Y_down[0, 2] = eps_down[2]
    Y_down[2, 0] = eps_down[2]

    return Y_up, Y_down
def ckm_from_yukawas(Y_up, Y_down):
    """Extract CKM via SVD."""
    U_uL, _, _ = np.linalg.svd(Y_up)
    U_dL, _, _ = np.linalg.svd(Y_down)
    V_CKM = U_uL @ U_dL.conj().T
    return V_CKM


def extract_cp_observables(V_CKM):
    """
    Extract δ_CP and Jarlskog invariant from CKM matrix.

    Standard parametrization:
    V_CKM = R23 * U13(δ) * R12

    Where U13(δ) contains the CP phase:
    U13 = [[c13, 0, s13*e^(-iδ)],
           [0, 1, 0],
           [-s13*e^(iδ), 0, c13]]
    """
    # Extract mixing angles
    s12 = np.abs(V_CKM[0, 1])
    s23 = np.abs(V_CKM[1, 2])
    s13 = np.abs(V_CKM[0, 2])

    c12 = np.sqrt(1 - s12**2)
    c23 = np.sqrt(1 - s23**2)
    c13 = np.sqrt(1 - s13**2)

    # δ_CP from phase of V_ub (element [0,2])
    # V_ub = s13 * e^(-iδ)
    delta_CP = -np.angle(V_CKM[0, 2])

    # Jarlskog invariant: J = Im[V_us V_cb V_ub* V_cs*]
    J_CP = np.imag(V_CKM[0, 1] * V_CKM[1, 2] *
                   np.conj(V_CKM[0, 2]) * np.conj(V_CKM[1, 1]))

    # Alternative: J = s12 * s23 * s13 * c12 * c23 * c13^2 * sin(δ)
    J_CP_alternative = s12 * s23 * s13 * c12 * c23 * c13**2 * np.sin(delta_CP)

    return delta_CP, J_CP


def objective(params):
    """
    Minimize maximum error over all observables (angles + CP).

    Parameters
    ----------
    params : array [12]
        Complex Yukawa parameters [Re, Im] × 6
    """
    try:
        Y_up, Y_down = yukawa_with_complex_offdiag(params)
        V_CKM = ckm_from_yukawas(Y_up, Y_down)
        delta_CP, J_CP = extract_cp_observables(V_CKM)

        # Extract CKM angles
        s12 = np.abs(V_CKM[0, 1])
        s23 = np.abs(V_CKM[1, 2])
        s13 = np.abs(V_CKM[0, 2])

        # Angle errors
        err_angle_12 = abs(s12**2 - sin2_12_obs) / sin2_12_obs
        err_angle_23 = abs(s23**2 - sin2_23_obs) / sin2_23_obs
        err_angle_13 = abs(s13**2 - sin2_13_obs) / sin2_13_obs

        # CP errors
        err_delta = abs(delta_CP - delta_CP_obs) / delta_CP_obs
        err_J = abs(J_CP - J_CP_obs) / J_CP_obs

        # Minimax: minimize maximum error
        return max(err_angle_12, err_angle_23, err_angle_13, err_delta, err_J)

    except:
        return 1e10
print("="*80)
print("OPTIMIZING CP VIOLATION PARAMETERS")
print("="*80)
print()
print(f"Target observables:")
print(f"  sin²θ₁₂ = {sin2_12_obs:.6f}")
print(f"  sin²θ₂₃ = {sin2_23_obs:.6f}")
print(f"  sin²θ₁₃ = {sin2_13_obs:.6f}")
print(f"  δ_CP = {delta_CP_obs:.2f} rad = {np.degrees(delta_CP_obs):.1f}°")
print(f"  J_CP = {J_CP_obs:.2e}")
print()

# Initial guess: start from real parameters with zero imaginary parts
x0 = np.zeros(12)
x0[0::2] = np.concatenate([eps_up_real, eps_down_real])  # Real parts

# Bounds: reasonable range for Yukawa off-diagonals
bounds = [(-100, 100) for _ in range(12)]

print("Phase 1: Differential evolution (global search)...")
result_de = differential_evolution(objective, bounds, maxiter=5000,
                                   seed=42, atol=1e-10, tol=1e-10)
print(f"DE complete: max error = {result_de.fun*100:.2f}%")
print()

print("Phase 2: L-BFGS-B refinement...")
result = minimize(objective, result_de.x, method='L-BFGS-B', bounds=bounds,
                 options={'maxiter': 10000, 'ftol': 1e-14})
print(f"L-BFGS-B complete: max error = {result.fun*100:.2f}%")
print()

print("Phase 3: Nelder-Mead final polish...")
result2 = minimize(objective, result.x, method='Nelder-Mead',
                  options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
print(f"Nelder-Mead complete: max error = {result2.fun*100:.2f}%")
print()

if result2.fun < result.fun:
    result = result2
    print("Using Nelder-Mead result")
else:
    print("Using L-BFGS-B result")
print()

# Extract results
params_opt = result.x
Y_up_opt, Y_down_opt = yukawa_with_complex_offdiag(params_opt)
V_CKM_opt = ckm_from_yukawas(Y_up_opt, Y_down_opt)
delta_CP_opt, J_CP_opt = extract_cp_observables(V_CKM_opt)

# Extract complex parameters
eps_up_opt = np.array([params_opt[0] + 1j*params_opt[1],
                       params_opt[2] + 1j*params_opt[3],
                       params_opt[4] + 1j*params_opt[5]])

eps_down_opt = np.array([params_opt[6] + 1j*params_opt[7],
                         params_opt[8] + 1j*params_opt[9],
                         params_opt[10] + 1j*params_opt[11]])

print("OPTIMIZED COMPLEX YUKAWA PARAMETERS:")
print(f"  Up-type:")
for i, label in enumerate(['ε₁₂', 'ε₂₃', 'ε₁₃']):
    mag = np.abs(eps_up_opt[i])
    phase = np.angle(eps_up_opt[i])
    print(f"    {label} = {mag:8.3f} * e^(i*{phase:6.3f}) = {eps_up_opt[i].real:8.3f} + i*{eps_up_opt[i].imag:8.3f}")
print(f"  Down-type:")
for i, label in enumerate(['ε₁₂', 'ε₂₃', 'ε₁₃']):
    mag = np.abs(eps_down_opt[i])
    phase = np.angle(eps_down_opt[i])
    print(f"    {label} = {mag:8.3f} * e^(i*{phase:6.3f}) = {eps_down_opt[i].real:8.3f} + i*{eps_down_opt[i].imag:8.3f}")
print()

# CKM angles
s12 = np.abs(V_CKM_opt[0, 1])
s23 = np.abs(V_CKM_opt[1, 2])
s13 = np.abs(V_CKM_opt[0, 2])

err_12 = abs(s12**2 - sin2_12_obs) / sin2_12_obs * 100
err_23 = abs(s23**2 - sin2_23_obs) / sin2_23_obs * 100
err_13 = abs(s13**2 - sin2_13_obs) / sin2_13_obs * 100

# CP observables
err_delta = abs(delta_CP_opt - delta_CP_obs) / delta_CP_obs * 100
err_J = abs(J_CP_opt - J_CP_obs) / J_CP_obs * 100

print("PREDICTIONS:")
print(f"  sin²θ₁₂: {s12**2:.6f} (obs: {sin2_12_obs:.6f}) - error: {err_12:.2f}%")
print(f"  sin²θ₂₃: {s23**2:.6f} (obs: {sin2_23_obs:.6f}) - error: {err_23:.2f}%")
print(f"  sin²θ₁₃: {s13**2:.6f} (obs: {sin2_13_obs:.6f}) - error: {err_13:.2f}%")
print(f"  δ_CP: {delta_CP_opt:.3f} rad = {np.degrees(delta_CP_opt):5.1f}° (obs: {delta_CP_obs:.2f} rad = {np.degrees(delta_CP_obs):.1f}°) - error: {err_delta:.2f}%")
print(f"  J_CP: {J_CP_opt:.3e} (obs: {J_CP_obs:.2e}) - error: {err_J:.2f}%")
print()
print(f"Maximum error: {max(err_12, err_23, err_13, err_delta, err_J):.2f}%")
print()

print("="*80)
print("Parameters for unified_predictions_complete.py:")
print("="*80)
print(f"eps_up = np.array([{eps_up_opt[0]}, {eps_up_opt[1]}, {eps_up_opt[2]}])")
print(f"eps_down = np.array([{eps_down_opt[0]}, {eps_down_opt[1]}, {eps_down_opt[2]}])")
