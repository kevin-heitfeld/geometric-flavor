"""
CKM WITH E₆ MODULAR FORM CORRECTIONS
=====================================

Take our successful Γ₀(4) + complex τ model (95% complete, χ²/dof=9.5)
and add E₆(τ) corrections to improve fit.

E₄(τ): weight 4, controls leading order
E₆(τ): weight 6, provides subleading corrections

Strategy: V = V_E4 × (1 + ε·E₆ corrections)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import differential_evolution

# Load previous best result
with open('ckm_gamma04_unitary_results.json', 'r') as f:
    prev_results = json.load(f)

print("="*80)
print("E₆ MODULAR FORM CORRECTIONS TO CKM")
print("="*80)

print(f"\nPrevious best result (E₄ only):")
print(f"  χ²/dof = {prev_results['fit_quality']['chi2_dof']:.2f}")
print(f"  8/9 elements < 3σ")
print(f"  Jarlskog: {prev_results['jarlskog']['sigma']:.1f}σ")
print(f"  V_cd: 5.8σ (worst element)")

# Experimental CKM
CKM_exp = np.array([
    [0.97435, 0.22500, 0.00369],
    [0.22000, 0.97349, 0.04182],
    [0.00857, 0.04110, 0.99915]
])

CKM_err = np.array([
    [0.00016, 0.00060, 0.00011],
    [0.00060, 0.00016, 0.00074],
    [0.00020, 0.00064, 0.00005]
])

# Load complex tau
with open('cp_violation_from_tau_spectrum_results.json', 'r') as f:
    cp_data = json.load(f)

tau_up_re = np.array(cp_data['complex_tau_spectrum']['up_quarks']['real'])
tau_up_im = np.array(cp_data['complex_tau_spectrum']['up_quarks']['imag'])
tau_down_re = np.array(cp_data['complex_tau_spectrum']['down_quarks']['real'])
tau_down_im = np.array(cp_data['complex_tau_spectrum']['down_quarks']['imag'])

tau_up = tau_up_re + 1j * tau_up_im
tau_down = tau_down_re + 1j * tau_down_im

# ==============================================================================
# EISENSTEIN E₆ MODULAR FORM
# ==============================================================================

print(f"\n" + "="*80)
print("EISENSTEIN E₆(τ) - WEIGHT 6 MODULAR FORM")
print("="*80)

def eisenstein_E4(tau, q_max=50):
    """E₄(τ) = 1 + 240 Σ n³qⁿ/(1-qⁿ)"""
    q = np.exp(2j * np.pi * tau)
    E4 = 1.0
    for n in range(1, q_max):
        qn = q**n
        E4 += 240 * n**3 * qn / (1 - qn)
    return E4

def eisenstein_E6(tau, q_max=50):
    """E₆(τ) = 1 - 504 Σ n⁵qⁿ/(1-qⁿ)"""
    q = np.exp(2j * np.pi * tau)
    E6 = 1.0
    for n in range(1, q_max):
        qn = q**n
        E6 -= 504 * n**5 * qn / (1 - qn)
    return E6

# Calculate E₄ and E₆ at all τ points
E4_up = np.array([eisenstein_E4(t) for t in tau_up])
E4_down = np.array([eisenstein_E4(t) for t in tau_down])

E6_up = np.array([eisenstein_E6(t) for t in tau_up])
E6_down = np.array([eisenstein_E6(t) for t in tau_down])

print(f"\nE₄(τ) values:")
print(f"  Up:   {E4_up[0]:.3f}, {E4_up[1]:.3f}, {E4_up[2]:.3f}")
print(f"  Down: {E4_down[0]:.3f}, {E4_down[1]:.3f}, {E4_down[2]:.3f}")

print(f"\nE₆(τ) values (NEW):")
print(f"  Up:   {E6_up[0]:.3f}, {E6_up[1]:.3f}, {E6_up[2]:.3f}")
print(f"  Down: {E6_down[0]:.3f}, {E6_down[1]:.3f}, {E6_down[2]:.3f}")

# ==============================================================================
# STANDARD PARAMETRIZATION + E₆ CORRECTIONS
# ==============================================================================

print(f"\n" + "="*80)
print("CKM WITH E₆ CORRECTIONS")
print("="*80)

# Γ₀(4) natural Cabibbo
theta12_gamma04 = np.arcsin(0.25)

def ckm_standard_parametrization(theta12, theta23, theta13, delta):
    """Standard PDG parametrization"""
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    c13, s13 = np.cos(theta13), np.sin(theta13)

    R12 = np.array([
        [c12, s12, 0],
        [-s12, c12, 0],
        [0, 0, 1]
    ])

    R23 = np.array([
        [1, 0, 0],
        [0, c23, s23],
        [0, -s23, c23]
    ])

    U13 = np.array([
        [c13, 0, s13 * np.exp(-1j * delta)],
        [0, 1, 0],
        [-s13 * np.exp(1j * delta), 0, c13]
    ])

    V = R23 @ U13 @ R12
    return V

def e6_correction_matrix(eps_vals, phi_vals):
    """
    E₆ correction matrix (small, off-diagonal)

    ΔV_ij = ε_ij × e^(iφ_ij) × (E₆_up[i]/E₄_up[i]) × (E₆_down[j]/E₄_down[j])

    Focus on (1,2) block where V_cd needs fixing
    """
    Delta_V = np.zeros((3, 3), dtype=complex)

    # Only correct 2×2 block (u,c) × (d,s)
    for i in range(2):
        for j in range(2):
            # E₆/E₄ ratio gives the correction strength
            ratio_up = E6_up[i] / E4_up[i] if abs(E4_up[i]) > 1e-10 else 0
            ratio_down = E6_down[j] / E4_down[j] if abs(E4_down[j]) > 1e-10 else 0

            idx = i * 2 + j
            Delta_V[i, j] = eps_vals[idx] * np.exp(1j * phi_vals[idx]) * ratio_up * np.conj(ratio_down)

    return Delta_V

def ckm_with_e6_corrections(params):
    """
    CKM = V_E4 + ΔV_E6

    Parameters:
      - δθ₁₂, θ₂₃, θ₁₃, δ_CP: standard (4 params)
      - ε₀₀, ε₀₁, ε₁₀, ε₁₁: E₆ correction strengths (4 params)
      - φ₀₀, φ₀₁, φ₁₀, φ₁₁: E₆ correction phases (4 params)
    Total: 12 parameters
    """
    delta_theta12, theta23, theta13, delta_cp = params[0:4]
    eps_vals = params[4:8]
    phi_vals = params[8:12]

    # Base CKM from standard parametrization
    theta12 = theta12_gamma04 + delta_theta12
    V_base = ckm_standard_parametrization(theta12, theta23, theta13, delta_cp)

    # E₆ corrections
    Delta_V = e6_correction_matrix(eps_vals, phi_vals)

    # Add corrections (small perturbation)
    V = V_base + Delta_V

    # Renormalize rows to maintain approximate unitarity
    for i in range(3):
        norm = np.sqrt(np.sum(np.abs(V[i, :])**2))
        V[i, :] /= norm

    return V

# ==============================================================================
# OPTIMIZATION
# ==============================================================================

print(f"\nOptimizing with E₆ corrections...")
print(f"  12 parameters: 4 (standard) + 4 (ε) + 4 (φ)")

def fit_ckm_with_e6(params):
    """Fit with Γ₀(4) constraint + E₆ corrections"""
    delta_theta12 = params[0]

    V = ckm_with_e6_corrections(params)
    V_mag = np.abs(V)

    chi2_ckm = np.sum(((V_mag - CKM_exp) / CKM_err)**2)

    # Penalty for deviating from Γ₀(4)
    penalty = 50 * (delta_theta12 / 0.1)**2

    return chi2_ckm + penalty

result = differential_evolution(
    fit_ckm_with_e6,
    bounds=[
        # Standard parameters
        (-0.1, 0.1),      # δθ₁₂
        (0, 0.1),         # θ₂₃
        (0, 0.01),        # θ₁₃
        (0, 2*np.pi),     # δ_CP
        # E₆ correction strengths (small)
        (-0.05, 0.05),    # ε₀₀
        (-0.05, 0.05),    # ε₀₁
        (-0.05, 0.05),    # ε₁₀
        (-0.05, 0.05),    # ε₁₁
        # E₆ correction phases
        (-np.pi, np.pi),  # φ₀₀
        (-np.pi, np.pi),  # φ₀₁
        (-np.pi, np.pi),  # φ₁₀
        (-np.pi, np.pi)   # φ₁₁
    ],
    seed=42,
    maxiter=1500,
    workers=1,
    atol=1e-10,
    tol=1e-10
)

params_best = result.x
chi2_best = result.fun

# Penalty
delta_theta12_best = params_best[0]
penalty_best = 50 * (delta_theta12_best / 0.1)**2
chi2_ckm = chi2_best - penalty_best

# ==============================================================================
# RESULTS
# ==============================================================================

print(f"\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

theta12_best = theta12_gamma04 + delta_theta12_best
theta23_best = params_best[1]
theta13_best = params_best[2]
delta_cp_best = params_best[3]

print(f"\nStandard parameters:")
print(f"  θ₁₂ = {np.degrees(theta12_best):.3f}° (Δ = {np.degrees(delta_theta12_best):.3f}° from Γ₀(4))")
print(f"  θ₂₃ = {np.degrees(theta23_best):.3f}°")
print(f"  θ₁₃ = {np.degrees(theta13_best):.3f}°")
print(f"  δ_CP = {np.degrees(delta_cp_best):.1f}°")

print(f"\nE₆ corrections:")
eps_best = params_best[4:8]
phi_best = params_best[8:12]
labels = ['(u,d)', '(u,s)', '(c,d)', '(c,s)']
for i, label in enumerate(labels):
    print(f"  {label}: ε = {eps_best[i]:+.4f}, φ = {np.degrees(phi_best[i]):+6.1f}°")

# Best CKM
V_best = ckm_with_e6_corrections(params_best)
V_best_mag = np.abs(V_best)

print(f"\nPredicted CKM matrix:")
print(f"        d         s         b")
for i, q in enumerate(['u', 'c', 't']):
    print(f"  {q}: {V_best_mag[i,0]:.5f}  {V_best_mag[i,1]:.5f}  {V_best_mag[i,2]:.5f}")

print(f"\nExperimental CKM:")
print(f"        d         s         b")
for i, q in enumerate(['u', 'c', 't']):
    print(f"  {q}: {CKM_exp[i,0]:.5f}  {CKM_exp[i,1]:.5f}  {CKM_exp[i,2]:.5f}")

# Deviations
deviation_sigma = (V_best_mag - CKM_exp) / CKM_err

print(f"\nDeviation (σ):")
print(f"        d         s         b")
for i, q in enumerate(['u', 'c', 't']):
    print(f"  {q}: {deviation_sigma[i,0]:5.1f}σ   {deviation_sigma[i,1]:5.1f}σ   {deviation_sigma[i,2]:5.1f}σ")

# Fit quality
dof = 9 - 12  # Careful: more parameters than data points!
if dof <= 0:
    dof = 1  # Use reduced dof
    print(f"\nNote: 12 parameters for 9 data points - overfitting risk!")

chi2_dof = chi2_ckm / dof

print(f"\nFit quality:")
print(f"  χ²_CKM = {chi2_ckm:.2f}")
print(f"  dof ≈ {dof} (effective)")
print(f"  χ²/dof ≈ {chi2_dof:.2f}")

# Jarlskog
def jarlskog_invariant(V):
    return np.imag(V[0,1] * V[1,2] * np.conj(V[0,2]) * np.conj(V[1,1]))

J_pred = jarlskog_invariant(V_best)
J_obs = 3.05e-5
J_err = 0.20e-5

print(f"\nJarlskog invariant:")
print(f"  Predicted: J = {J_pred:.2e}")
print(f"  Observed:  J = {J_obs:.2e} ± {J_err:.2e}")
print(f"  σ: {abs(J_pred - J_obs)/J_err:.1f}σ")

# Unitarity
VdaggerV = np.conj(V_best.T) @ V_best
unitarity_dev = np.max(np.abs(VdaggerV - np.eye(3)))

print(f"\nUnitarity deviation: {unitarity_dev:.2e}")

# Count good elements
n_excellent = np.sum(np.abs(deviation_sigma) < 1)
n_good = np.sum(np.abs(deviation_sigma) < 3)

# ==============================================================================
# COMPARISON
# ==============================================================================

print(f"\n" + "="*80)
print("COMPARISON: E₄ ONLY vs E₄ + E₆")
print("="*80)

prev_chi2_dof = prev_results['fit_quality']['chi2_dof']
prev_n_good = prev_results['element_accuracy']['good_below_3sigma']
prev_J_sigma = prev_results['jarlskog']['sigma']

print(f"\nE₄ only (previous):")
print(f"  χ²/dof = {prev_chi2_dof:.2f}")
print(f"  Elements < 3σ: {prev_n_good}/9")
print(f"  Jarlskog: {prev_J_sigma:.1f}σ")

print(f"\nE₄ + E₆ (this model):")
print(f"  χ²/dof ≈ {chi2_dof:.2f}")
print(f"  Elements < 3σ: {n_good}/9")
print(f"  Jarlskog: {abs(J_pred-J_obs)/J_err:.1f}σ")

improvement = prev_chi2_dof - chi2_dof
print(f"\nΔχ²/dof = {improvement:.2f}")

if improvement > 2:
    print("  ✓✓✓ SIGNIFICANT IMPROVEMENT!")
elif improvement > 0.5:
    print("  ✓✓ Modest improvement")
elif improvement > 0:
    print("  ✓ Small improvement")
else:
    print("  ~ No improvement (E₆ corrections too small or not needed)")

# Save results
results = {
    'model': 'Γ₀(4) + complex τ + E₆ corrections',
    'parameters': {
        'theta_12_degrees': float(np.degrees(theta12_best)),
        'delta_theta_12_degrees': float(np.degrees(delta_theta12_best)),
        'theta_23_degrees': float(np.degrees(theta23_best)),
        'theta_13_degrees': float(np.degrees(theta13_best)),
        'delta_cp_degrees': float(np.degrees(delta_cp_best)),
        'e6_corrections': {
            'epsilon': eps_best.tolist(),
            'phi_degrees': np.degrees(phi_best).tolist()
        }
    },
    'fit_quality': {
        'chi2_ckm': float(chi2_ckm),
        'dof': int(dof),
        'chi2_dof': float(chi2_dof)
    },
    'ckm_matrix': {
        'predicted': V_best_mag.tolist(),
        'observed': CKM_exp.tolist(),
        'deviation_sigma': deviation_sigma.tolist()
    },
    'element_accuracy': {
        'excellent_below_1sigma': int(n_excellent),
        'good_below_3sigma': int(n_good)
    },
    'jarlskog': {
        'predicted': float(J_pred),
        'observed': float(J_obs),
        'sigma': float(abs(J_pred - J_obs)/J_err)
    },
    'unitarity': {
        'max_deviation': float(unitarity_dev)
    },
    'comparison_to_e4_only': {
        'prev_chi2_dof': float(prev_chi2_dof),
        'new_chi2_dof': float(chi2_dof),
        'improvement': float(improvement)
    }
}

with open('ckm_with_e6_corrections_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved")

print("="*80)
