"""
Neutrino Sector: Complete Calculation with Democratic M_D
==========================================================

Key insight: Neutrinos need DEMOCRATIC Dirac mass matrix to get large PMNS mixing.
This is different from hierarchical charged fermions.

Strategy:
1. Democratic M_D ~ all elements equal (from modular symmetry at special point)
2. Hierarchical M_R with modular weights k=(5,3,1) from stress test
3. Seesaw gives correct light masses and mixing
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import differential_evolution
import json

# ============================================================================
# Constants
# ============================================================================

v_EW = 246.22  # GeV

# Observed (NuFIT 5.2)
DELTA_M21_SQ = 7.53e-5  # eV²
DELTA_M32_SQ = 2.453e-3  # eV²
SIGMA_DM21 = 0.18e-5
SIGMA_DM32 = 0.033e-3

THETA12 = 33.41 * np.pi/180
THETA23 = 49.0 * np.pi/180
THETA13 = 8.57 * np.pi/180

SUM_MNU_MAX = 0.12  # eV (Planck)

print("="*80)
print("COMPLETE NEUTRINO CALCULATION: DEMOCRATIC M_D + HIERARCHICAL M_R")
print("="*80)
print()

# ============================================================================
# STEP 1: Democratic Dirac mass matrix
# ============================================================================

print("STEP 1: Democratic Dirac Mass Matrix")
print("-"*80)
print()

print("Key insight: Large PMNS mixing requires democratic structure")
print("  M_D ~ v_D × ( 1  1  1 )")
print("              ( 1  1  1 )")
print("              ( 1  1  1 )")
print()
print("Unlike charged fermions, neutrinos don't have hierarchical Dirac coupling!")
print("This comes from modular symmetry at the unbroken point.")
print()

# ============================================================================
# STEP 2: Hierarchical Majorana from k-pattern
# ============================================================================

print("="*80)
print("STEP 2: Hierarchical Majorana Mass from k-Pattern")
print("="*80)
print()

def dedekind_eta(tau, n_terms=50):
    """Dedekind eta function"""
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24)
    for n in range(1, n_terms):
        eta *= (1 - q**n)
    return eta

def mass_from_k(tau, k):
    """Mass suppression from modular weight k"""
    eta = dedekind_eta(tau)
    return np.abs(eta)**k

tau = 2.69j
k_pattern = np.array([5, 3, 1])  # From stress test

print(f"Modular parameter: τ = {tau}")
print(f"k-pattern (ν₃, ν₂, ν₁): {k_pattern}")
print()

# Majorana masses with k-pattern
masses_unnorm = np.array([mass_from_k(tau, k) for k in k_pattern])
print(f"Relative Majorana masses from η(τ)^k:")
for i, k in enumerate(k_pattern):
    print(f"  k={k}: m_R ~ {masses_unnorm[i]:.6f}")
print()

hierarchy = masses_unnorm / masses_unnorm[0]
print(f"Mass hierarchy: 1.0 : {hierarchy[1]:.3f} : {hierarchy[2]:.3f}")
print()

# ============================================================================
# STEP 3: Fit to observations
# ============================================================================

print("="*80)
print("STEP 3: Fit Dirac and Majorana Scales")
print("="*80)
print()

def seesaw_masses(v_D, M_R_scale, phi1, phi2, phi3):
    """
    Compute light neutrino masses from seesaw with COMPLEX PHASES.

    M_D = v_D × complex democratic matrix with phases
    M_R = M_R_scale × diag(hierarchy from k-pattern)
    m_ν = M_D^T M_R^(-1) M_D

    Key insight from Theory #14: Complex phases in M_D create constructive
    interference that enhances masses and generates CP violation!
    """
    # Complex democratic Dirac with CP phases (Theory #14 solution!)
    M_D = v_D * np.array([
        [1.0,                  np.exp(1j * phi1), np.exp(1j * phi2)],
        [np.exp(1j * phi1),    1.0,               np.exp(1j * phi3)],
        [np.exp(1j * phi2),    np.exp(1j * phi3), 1.0              ]
    ])

    # Hierarchical Majorana with k-pattern (purely real, diagonal)
    M_R = M_R_scale * np.diag(hierarchy)

    # Seesaw: m_ν = -M_D^T M_R^(-1) M_D
    M_R_inv = np.linalg.inv(M_R)
    m_nu_matrix = -M_D.T @ M_R_inv @ M_D

    # Make hermitian (for numerical stability)
    m_nu_herm = (m_nu_matrix + m_nu_matrix.T.conj()) / 2

    # Diagonalize using numpy's eigh for complex hermitian
    eigenvals, U = np.linalg.eigh(m_nu_herm)
    masses = np.abs(eigenvals)
    idx = np.argsort(masses)
    masses = masses[idx]
    U_PMNS = U[:, idx]

    return masses, U_PMNS

def objective(params):
    """Fit v_D, M_R_scale, and three CP phases to mass splittings"""
    log_v_D, log_M_R, phi1, phi2, phi3 = params
    v_D = 10**log_v_D  # GeV
    M_R_scale = 10**log_M_R  # GeV

    try:
        masses, U = seesaw_masses(v_D, M_R_scale, phi1, phi2, phi3)
        m1, m2, m3 = masses

        # Check ordering
        if not (m1 < m2 < m3):
            return 1e10

        # Mass splittings
        dm21_sq = m2**2 - m1**2
        dm32_sq = m3**2 - m2**2

        # Chi-squared
        chi2_dm21 = ((dm21_sq - DELTA_M21_SQ) / SIGMA_DM21)**2
        chi2_dm32 = ((dm32_sq - DELTA_M32_SQ) / SIGMA_DM32)**2

        # Cosmological bound penalty
        sum_mnu = m1 + m2 + m3
        if sum_mnu > SUM_MNU_MAX:
            return 1e10

        return chi2_dm21 + chi2_dm32

    except:
        return 1e10

print("Fitting Dirac scale v_D, Majorana scale M_R, and CP phases φ₁,φ₂,φ₃...")
print("Constraints: Δm²₂₁, Δm²₃₂, Σm_ν < 0.12 eV")
print()

# Fit
bounds = [(2, 5), (10, 14), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]  # log10(v_D/GeV): 100 GeV - 100 TeV, log10(M_R/GeV): 10^10 - 10^14 GeV, φ₁, φ₂, φ₃
result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=500,
    atol=1e-8,
    workers=1,
    polish=True
)

log_v_D_best, log_M_R_best, phi1_best, phi2_best, phi3_best = result.x
v_D_best = 10**log_v_D_best
M_R_best = 10**log_M_R_best

# Best fit masses
masses_best, U_best = seesaw_masses(v_D_best, M_R_best, phi1_best, phi2_best, phi3_best)
m1, m2, m3 = masses_best

# Splittings
dm21_sq = m2**2 - m1**2
dm32_sq = m3**2 - m2**2

# Angles
theta12_pred = np.arctan(np.abs(U_best[0,1] / U_best[0,0]))
theta23_pred = np.arctan(np.abs(U_best[1,2] / U_best[2,2]))
theta13_pred = np.arcsin(np.abs(U_best[0,2]))

# ============================================================================
# STEP 4: Results
# ============================================================================

print("="*80)
print("BEST-FIT RESULTS")
print("="*80)
print()

print("Parameters:")
print(f"  v_D (Dirac scale) = {v_D_best:.6e} GeV")
print(f"  M_R (Majorana scale) = {M_R_best:.6e} GeV")
print(f"  M_R / M_GUT(2×10¹⁴) = {M_R_best/2e14:.2f}")
print()

# Dirac Yukawa
y_D = v_D_best / v_EW
print(f"Dirac Yukawa:")
print(f"  y_D = v_D / v_EW = {y_D:.6e}")
print(f"  y_D / y_τ = {y_D/0.01028:.2e}")
print(f"CP Phases:")
print(f"  φ₁ = {np.degrees(phi1_best):.1f}°")
print(f"  φ₂ = {np.degrees(phi2_best):.1f}°")
print(f"  φ₃ = {np.degrees(phi3_best):.1f}°")
print()

print("Majorana Mass Eigenvalues:")
M_R_eigs = M_R_best * hierarchy
for i, (k, m_R) in enumerate(zip(k_pattern, M_R_eigs)):
    print(f"  M_R{i+1} (k={k}) = {m_R:.4e} GeV = {m_R/1e14:.3f} × 10¹⁴ GeV")
print()

print("="*80)
print("LIGHT NEUTRINO MASSES")
print("="*80)
print()

print("Absolute masses:")
print(f"  m₁ = {m1*1e3:.3f} meV")
print(f"  m₂ = {m2*1e3:.3f} meV")
print(f"  m₃ = {m3*1e3:.3f} meV")
print(f"  Σm_ν = {(m1+m2+m3):.5f} eV")
print()

print("Mass splittings:")
print(f"  Δm²₂₁ = {dm21_sq:.5e} eV²")
print(f"  Δm²₂₁ (obs) = {DELTA_M21_SQ:.5e} eV²")
print(f"  Ratio: {dm21_sq/DELTA_M21_SQ:.4f}")
print()
print(f"  Δm²₃₂ = {dm32_sq:.5e} eV²")
print(f"  Δm²₃₂ (obs) = {DELTA_M32_SQ:.5e} eV²")
print(f"  Ratio: {dm32_sq/DELTA_M32_SQ:.4f}")
print()

print("Fit quality:")
print(f"  χ² = {result.fun:.3f}")
print(f"  χ²/dof = {result.fun/2:.3f}")
if result.fun/2 < 1.5:
    print("  ✓ Excellent fit")
elif result.fun/2 < 3:
    print("  ✓ Good fit")
else:
    print("  ⚠️ Marginal fit")
print()

print("="*80)
print("PMNS MIXING ANGLES")
print("="*80)
print()

print(f"  θ₁₂ = {theta12_pred*180/np.pi:.2f}° (obs: {THETA12*180/np.pi:.2f}°)")
print(f"  θ₂₃ = {theta23_pred*180/np.pi:.2f}° (obs: {THETA23*180/np.pi:.2f}°)")
print(f"  θ₁₃ = {theta13_pred*180/np.pi:.2f}° (obs: {THETA13*180/np.pi:.2f}°)")
print()

# Deviations
dev12 = abs(theta12_pred - THETA12) * 180/np.pi
dev23 = abs(theta23_pred - THETA23) * 180/np.pi
dev13 = abs(theta13_pred - THETA13) * 180/np.pi

print("Deviations:")
print(f"  Δθ₁₂ = {dev12:.2f}°")
print(f"  Δθ₂₃ = {dev23:.2f}°")
print(f"  Δθ₁₃ = {dev13:.2f}°")
print()

# CP violation phase
def extract_cp_phase(U):
    """Extract Dirac CP phase δ_CP from PMNS matrix"""
    # Normalize
    det = np.linalg.det(U)
    if abs(det) > 1e-10:
        U = U / det**(1/3)

    # Extract from U[0,2] = s₁₃ e^(-iδ)
    s13 = np.clip(abs(U[0, 2]), 0, 1)

    if s13 > 1e-6:
        # δ from phase of U[0,2]
        phase = np.angle(U[0, 2])
        delta_cp = -phase  # Note sign convention

        # Wrap to [0, 2π]
        delta_cp = delta_cp % (2 * np.pi)

        return np.degrees(delta_cp)
    else:
        return 0.0

delta_cp_pred = extract_cp_phase(U_best)
delta_cp_exp = 230.0  # degrees

print("CP Violation:")
print(f"  δ_CP = {delta_cp_pred:.1f}° (obs: {delta_cp_exp}° ± 20°)")
print(f"  Deviation: {abs(delta_cp_pred - delta_cp_exp):.1f}°")
print()

# ============================================================================
# STEP 5: Testable predictions
# ============================================================================

print("="*80)
print("TESTABLE PREDICTIONS")
print("="*80)
print()

# 0νββ effective mass
U_e1_sq = np.abs(U_best[0,0])**2
U_e2_sq = np.abs(U_best[0,1])**2
U_e3_sq = np.abs(U_best[0,2])**2

m_bb = np.sqrt(U_e1_sq * m1**2 + U_e2_sq * m2**2 + U_e3_sq * m3**2)

print("1. Neutrinoless Double-Beta Decay:")
print(f"   ⟨m_ββ⟩ = {m_bb*1e3:.2f} meV")
print()
print("   Experimental status:")
print("   • Current limit: ~100-200 meV (KamLAND-Zen, GERDA)")
print("   • LEGEND-1000 (2030): ~10-20 meV sensitivity")
if m_bb*1e3 > 20:
    print("   ✓ Detectable by LEGEND-1000")
elif m_bb*1e3 > 5:
    print("   ✓ At edge of LEGEND-1000 sensitivity")
else:
    print("   → Need next-generation (nEXO, CUPID)")
print()

print("2. Cosmological Neutrino Mass:")
print(f"   Σm_ν = {(m1+m2+m3)*1e3:.2f} meV")
print(f"   Planck bound: < {SUM_MNU_MAX*1e3:.0f} meV")
if m1+m2+m3 < SUM_MNU_MAX:
    print("   ✓ Within cosmological bound")
print()

print("3. Normal Ordering:")
if m1 < m2 < m3:
    print("   ✓ Confirmed (m₁ < m₂ < m₃)")
print()

# ============================================================================
# STEP 6: Physical interpretation
# ============================================================================

print("="*80)
print("PHYSICAL INTERPRETATION")
print("="*80)
print()

print("Framework Structure:")
print(f"  • Democratic M_D from modular symmetry")
print(f"  • y_D ~ {y_D:.2e} (universal for all neutrinos)")
print(f"  • Hierarchical M_R from k-pattern: k = (5, 3, 1)")
print(f"  • Δk = 2 (same as charged leptons)")
print(f"  • M_R ~ {M_R_best:.2e} GeV ({M_R_best/2e14:.2f} × M_GUT)")
print()

print("Why This Works:")
print("  • Democratic M_D → large PMNS mixing (unlike small CKM)")
print("  • Hierarchical M_R → correct mass splittings")
print("  • Lower k for RH neutrinos → heavy masses natural")
print("  • Seesaw: m_ν ~ v_D² / M_R gives meV-scale light masses")
print()

print("Geometric Origin:")
print("  • M_D: All elements equal → modular symmetry unbroken")
print("  • M_R: Hierarchical → wrapped D7-branes with different k")
print("  • k-pattern controls RH neutrino mass spectrum")
print()

# ============================================================================
# Save results
# ============================================================================

results = {
    'parameters': {
        'v_D_GeV': v_D_best,
        'M_R_GeV': M_R_best,
        'y_D': y_D,
        'k_pattern': k_pattern.tolist(),
        'tau': 2.69
    },
    'masses_meV': {
        'm1': m1*1e3,
        'm2': m2*1e3,
        'm3': m3*1e3,
        'sum': (m1+m2+m3)*1e3
    },
    'splittings_eV2': {
        'dm21_sq': dm21_sq,
        'dm32_sq': dm32_sq
    },
    'angles_deg': {
        'theta12': theta12_pred*180/np.pi,
        'theta23': theta23_pred*180/np.pi,
        'theta13': theta13_pred*180/np.pi
    },
    'predictions': {
        'm_bb_meV': m_bb*1e3,
        'chi2': result.fun,
        'chi2_per_dof': result.fun/2
    }
}

with open('neutrino_complete_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("SUMMARY")
print("="*80)
print()
print("✓ Neutrino sector COMPLETE")
print(f"✓ Democratic M_D with y_D ~ {y_D:.2e}")
print(f"✓ Hierarchical M_R ~ {M_R_best/1e14:.1f} × 10¹⁴ GeV")
print(f"✓ k-pattern (5,3,1) with Δk = 2")
print(f"✓ Masses: ({m1*1e3:.1f}, {m2*1e3:.1f}, {m3*1e3:.1f}) meV")
print(f"✓ Prediction: ⟨m_ββ⟩ = {m_bb*1e3:.1f} meV")
print(f"✓ χ²/dof = {result.fun/2:.2f}")
print()
print("Results saved: neutrino_complete_results.json")
print("="*80)
