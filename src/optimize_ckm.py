"""
Optimize CKM Matrix Off-Diagonal Structure
===========================================

Given perfect mass ratios, optimize the off-diagonal Yukawa structure
to reproduce observed CKM angles.
"""

import numpy as np
from scipy.optimize import minimize


def yukawa_from_params(m_up, m_down, eps_params):
    """
    Construct Yukawa matrices with complex off-diagonals.

    Y_ij = diag(m_i) + |ε_ij| * exp(i*φ_ij)

    Parameters
    ----------
    m_up : array
        Up-type mass ratios [1, m_c/m_u, m_t/m_u]
    m_down : array
        Down-type mass ratios [1, m_s/m_d, m_b/m_d]
    eps_params : array
        [|ε₁₂_up|, |ε₂₃_up|, |ε₁₃_up|, φ₁₂_up, φ₂₃_up, φ₁₃_up,
         |ε₁₂_down|, |ε₂₃_down|, |ε₁₃_down|, φ₁₂_down, φ₂₃_down, φ₁₃_down]
        First 6: magnitudes, Next 6: phases (radians)

    Returns
    -------
    Y_up, Y_down : arrays
        3×3 Yukawa matrices
    """
    # Extract magnitudes and phases
    eps_mag_up = eps_params[0:3]
    phi_up = eps_params[3:6]
    eps_mag_down = eps_params[6:9]
    phi_down = eps_params[9:12]

    # Up-type Yukawa with complex phases
    Y_up = np.diag(m_up).astype(complex)
    Y_up[0, 1] = eps_mag_up[0] * np.exp(1j * phi_up[0])
    Y_up[1, 0] = Y_up[0, 1]
    Y_up[1, 2] = eps_mag_up[1] * np.exp(1j * phi_up[1])
    Y_up[2, 1] = Y_up[1, 2]
    Y_up[0, 2] = eps_mag_up[2] * np.exp(1j * phi_up[2])
    Y_up[2, 0] = Y_up[0, 2]

    # Down-type Yukawa with complex phases
    Y_down = np.diag(m_down).astype(complex)
    Y_down[0, 1] = eps_mag_down[0] * np.exp(1j * phi_down[0])
    Y_down[1, 0] = Y_down[0, 1]
    Y_down[1, 2] = eps_mag_down[1] * np.exp(1j * phi_down[1])
    Y_down[2, 1] = Y_down[1, 2]
    Y_down[0, 2] = eps_mag_down[2] * np.exp(1j * phi_down[2])
    Y_down[2, 0] = Y_down[0, 2]

    return Y_up, Y_down
def ckm_from_yukawas(Y_up, Y_down):
    """Extract CKM matrix from Yukawa diagonalization."""
    # Diagonalize Yukawas (use SVD for numerical stability)
    U_uL, _, _ = np.linalg.svd(Y_up)
    U_dL, _, _ = np.linalg.svd(Y_down)

    # CKM matrix
    V_CKM = U_uL @ U_dL.conj().T

    # Extract standard parametrization angles
    s12 = np.abs(V_CKM[0, 1])
    s23 = np.abs(V_CKM[1, 2])
    s13 = np.abs(V_CKM[0, 2])

    return s12**2, s23**2, s13**2, V_CKM
# Observed mass ratios (perfect from optimization)
m_up = np.array([1.0, 577.0, 78636.0])
m_down = np.array([1.0, 20.3, 890.0])  # Fixed: m_s/m_d = 95/4.67 = 20.3

print("Mass ratios being used:")
print(f"  m_up = {m_up}")
print(f"  m_down = {m_down}")
print()

# Observed CKM angles
sin2_12_obs = 0.0510
sin2_23_obs = 0.00157
sin2_13_obs = 0.000128

print("="*80)
print("OPTIMIZING CKM OFF-DIAGONAL STRUCTURE")
print("="*80)
print()

print(f"Target CKM angles:")
print(f"  sin²θ₁₂ = {sin2_12_obs:.6f}")
print(f"  sin²θ₂₃ = {sin2_23_obs:.6f}")
print(f"  sin²θ₁₃ = {sin2_13_obs:.6f}")
print()


def objective(eps_params):
    """Minimize errors in CKM angles."""
    Y_up, Y_down = yukawa_from_params(m_up, m_down, eps_params)
    sin2_12, sin2_23, sin2_13, _ = ckm_from_yukawas(Y_up, Y_down)

    err_12 = abs(sin2_12 - sin2_12_obs) / sin2_12_obs
    err_23 = abs(sin2_23 - sin2_23_obs) / sin2_23_obs
    err_13 = abs(sin2_13 - sin2_13_obs) / sin2_13_obs

    # Return maximum error (minimax)
    return max(err_12, err_23, err_13)
# Initial guess: magnitudes from previous REAL optimization + zero phases initially
# Previous best: [38.60, -20.98, -2.15, -5.53, 37.73, 2.50] (real)
# Allow signs in magnitudes, phases can compensate
x0 = [
    38.60, 20.98, 2.15,      # |ε_up|
    0.0, 0.0, 0.0,           # φ_up (start with zero, let optimizer find them)
    5.53, 37.73, 2.50,       # |ε_down|
    0.0, 0.0, 0.0            # φ_down
]

print("Optimizing off-diagonal parameters with complex phases...")
print(f"Initial guess (12 parameters: 6 magnitudes + 6 phases):")
print(f"  |ε_up|  = [{x0[0]:.2f}, {x0[1]:.2f}, {x0[2]:.2f}]")
print(f"  φ_up   = [{x0[3]:.2f}, {x0[4]:.2f}, {x0[5]:.2f}]")
print(f"  |ε_down| = [{x0[6]:.2f}, {x0[7]:.2f}, {x0[8]:.2f}]")
print(f"  φ_down  = [{x0[9]:.2f}, {x0[10]:.2f}, {x0[11]:.2f}]")
print()

result = minimize(objective, x0, method='Nelder-Mead',
                 options={'maxiter': 30000, 'xatol': 1e-12, 'fatol': 1e-12})
print("Optimization complete!")
print(f"Success: {result.success}")
print(f"Iterations: {result.nit}")
print(f"Function value: {result.fun:.6f}")
print()

# Try different optimizer for comparison
print("Trying L-BFGS-B for refinement...")
result2 = minimize(objective, result.x, method='L-BFGS-B',
                  options={'maxiter': 10000, 'ftol': 1e-12})
print(f"L-BFGS-B complete!")
print(f"Success: {result2.success}")
print(f"Function value: {result2.fun:.6f}")
print()

# Use best of both
if result2.fun < result.fun:
    print("L-BFGS-B achieved better result, using it")
    result = result2
else:
    print("Nelder-Mead achieved better result, using it")

print("Optimization complete!")
print(f"Success: {result.success}")
print(f"Iterations: {result.nit}")
print()

# Extract optimized parameters
eps_opt = result.x

# Parse into magnitudes and phases
eps_mag_up = eps_opt[0:3]
phi_up = eps_opt[3:6]
eps_mag_down = eps_opt[6:9]
phi_down = eps_opt[9:12]

Y_up_opt, Y_down_opt = yukawa_from_params(m_up, m_down, eps_opt)

print("DEBUG: Testing with optimized parameters")
print(f"Y_up_opt[0,1] = {Y_up_opt[0,1]:.4f}")
print(f"Y_down_opt[0,1] = {Y_down_opt[0,1]:.4f}")
print()

sin2_12_opt, sin2_23_opt, sin2_13_opt, V_CKM_opt = ckm_from_yukawas(Y_up_opt, Y_down_opt)

print(f"V_CKM_opt:")
print(V_CKM_opt)
print()

print("OPTIMIZED OFF-DIAGONAL PARAMETERS:")
print(f"  Up-type:")
print(f"    |ε₁₂| = {eps_mag_up[0]:.6f},  φ₁₂ = {phi_up[0]:.6f} rad = {np.degrees(phi_up[0]):.2f}°")
print(f"    |ε₂₃| = {eps_mag_up[1]:.6f},  φ₂₃ = {phi_up[1]:.6f} rad = {np.degrees(phi_up[1]):.2f}°")
print(f"    |ε₁₃| = {eps_mag_up[2]:.6f},  φ₁₃ = {phi_up[2]:.6f} rad = {np.degrees(phi_up[2]):.2f}°")
print(f"  Down-type:")
print(f"    |ε₁₂| = {eps_mag_down[0]:.6f},  φ₁₂ = {phi_down[0]:.6f} rad = {np.degrees(phi_down[0]):.2f}°")
print(f"    |ε₂₃| = {eps_mag_down[1]:.6f},  φ₂₃ = {phi_down[1]:.6f} rad = {np.degrees(phi_down[1]):.2f}°")
print(f"    |ε₁₃| = {eps_mag_down[2]:.6f},  φ₁₃ = {phi_down[2]:.6f} rad = {np.degrees(phi_down[2]):.2f}°")
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
print(f"  sin²θ₁₂: {err_12:.1f}%")
print(f"  sin²θ₂₃: {err_23:.1f}%")
print(f"  sin²θ₁₃: {err_13:.1f}%")
print(f"  Maximum: {max(err_12, err_23, err_13):.1f}%")
print()

print("="*80)
print("TO USE THESE VALUES, UPDATE unified_predictions_complete.py:")
print("="*80)
print(f"# Off-diagonal Yukawa structure with complex phases (optimized)")
print(f"eps_mag_up = np.array([{eps_mag_up[0]:.8f}, {eps_mag_up[1]:.8f}, {eps_mag_up[2]:.8f}])")
print(f"phi_up = np.array([{phi_up[0]:.8f}, {phi_up[1]:.8f}, {phi_up[2]:.8f}])")
print(f"eps_mag_down = np.array([{eps_mag_down[0]:.8f}, {eps_mag_down[1]:.8f}, {eps_mag_down[2]:.8f}])")
print(f"phi_down = np.array([{phi_down[0]:.8f}, {phi_down[1]:.8f}, {phi_down[2]:.8f}])")
print()
print("# Build complex Yukawas:")
print("eps_up = eps_mag_up * np.exp(1j * phi_up)")
print("eps_down = eps_mag_down * np.exp(1j * phi_down)")
print()
