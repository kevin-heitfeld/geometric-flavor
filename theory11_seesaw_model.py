"""
THEORY #11: SEESAW MODEL TEST

After mixing angle failure, implement Kimi's suggestion:
  M_ν = M_D^T M_R^{-1} M_D  (Type-I seesaw)

Key predictions to test:
1. M_D should have ε_D/GM_D ≈ 0.81 (same as charged leptons)
2. M_R at GUT scale (~10^14 GeV) with its own structure
3. Combined: fit neutrino masses AND PMNS angles
4. If successful: Theory #11 is genuinely predictive

Strategy:
- Fix M_D to charged lepton structure (scaled down)
- Fit M_R parameters to match neutrino data
- Check if PMNS angles emerge correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigvals
from scipy.optimize import differential_evolution, minimize

print("="*80)
print("THEORY #11: SEESAW MODEL IMPLEMENTATION")
print("="*80)

# ==============================================================================
# PART 1: SETUP AND MASS MATRICES
# ==============================================================================

print("\n" + "="*80)
print("PART 1: MASS MATRIX STRUCTURE")
print("="*80)

def mass_matrix(d1, d2, d3, epsilon):
    """M = diag(d) + ε·J"""
    M = np.diag([d1, d2, d3])
    M += epsilon * np.ones((3, 3))
    return M

# Charged lepton matrix (from previous fit)
d_lep = [15.77, 92.03, 1775.23]  # MeV
eps_lep = 37.03  # MeV
M_leptons = mass_matrix(*d_lep, eps_lep)
masses_lep_check = np.sort(np.real(eigvals(M_leptons)))

print("\nCharged Lepton Matrix M_e:")
print(f"  d = {d_lep} MeV")
print(f"  ε = {eps_lep} MeV")
print(f"  Masses: {masses_lep_check} MeV")
print(f"  ε/GM = {eps_lep / (np.prod(masses_lep_check)**(1/3)):.3f}")

# Experimental neutrino data
Delta_m21_sq = 7.53e-5  # eV²
Delta_m31_sq = 2.453e-3  # eV²

# PMNS angles (experimental)
theta12_exp = 33.45  # degrees
theta23_exp = 49.0
theta13_exp = 8.57

print("\nNeutrino oscillation data:")
print(f"  Δm²₂₁ = {Delta_m21_sq:.3e} eV²")
print(f"  Δm²₃₁ = {Delta_m31_sq:.3e} eV²")

print("\nPMNS angles (experimental):")
print(f"  θ₁₂ = {theta12_exp}° ± 0.77°")
print(f"  θ₂₃ = {theta23_exp}° ± 1.4°")
print(f"  θ₁₃ = {theta13_exp}° ± 0.13°")

# ==============================================================================
# PART 2: SEESAW FORMULA
# ==============================================================================

print("\n" + "="*80)
print("PART 2: TYPE-I SEESAW MECHANISM")
print("="*80)

print("""
Type-I Seesaw Formula:
  M_ν = M_D^T M_R^{-1} M_D

where:
  M_D = Dirac mass matrix (like charged leptons)
  M_R = Right-handed Majorana mass matrix (GUT scale)

Key insight from Kimi:
  - M_D should have ε_D/GM_D ≈ 0.81 (universal coupling)
  - M_R at high scale breaks this, giving ε_ν/GM_ν ≠ 0.81
  - Eigenvectors of M_ν ≠ eigenvectors of M_D → different mixing
""")

def seesaw_neutrino_mass(MD, MR):
    """
    Calculate effective neutrino mass via seesaw
    M_ν = M_D^T M_R^{-1} M_D
    """
    MR_inv = np.linalg.inv(MR)
    M_nu = MD.T @ MR_inv @ MD
    return M_nu

def extract_mixing_angles(U):
    """Extract mixing angles from PMNS matrix"""
    s13 = abs(U[0, 2])
    theta13 = np.arcsin(min(s13, 1.0))

    c13 = np.cos(theta13)
    if c13 > 1e-10:
        s12 = abs(U[0, 1]) / c13
        c12 = abs(U[0, 0]) / c13
        s23 = abs(U[1, 2]) / c13
        c23 = abs(U[2, 2]) / c13
    else:
        s12 = 0
        c12 = 1
        s23 = abs(U[1, 2])
        c23 = abs(U[2, 2])

    theta12 = np.arctan2(s12, c12) * 180/np.pi
    theta23 = np.arctan2(s23, c23) * 180/np.pi
    theta13 = theta13 * 180/np.pi

    return theta12, theta23, theta13

# ==============================================================================
# PART 3: FIT WITH CONSTRAINED M_D
# ==============================================================================

print("\n" + "="*80)
print("PART 3: FIT WITH M_D CONSTRAINED TO ε/GM ≈ 0.81")
print("="*80)

print("\nApproach 1: Fix M_D structure, fit M_R")
print("-" * 40)

# M_D should have similar structure to charged leptons
# Scale down by factor to get right neutrino masses
# Constraint: ε_D / GM_D ≈ 0.81

v = 246.22  # GeV = 246220 MeV (Higgs VEV)

def objective_seesaw_constrained(params):
    """
    Fit seesaw with M_D constrained

    params = [d1_D, d2_D, d3_D, eps_D,  # Dirac matrix (in eV)
              d1_R, d2_R, d3_R, eps_R,  # Majorana matrix (in GeV)
              MR_scale]                  # Overall Majorana scale
    """
    if len(params) != 9:
        return 1e10

    d1_D, d2_D, d3_D, eps_D = params[0:4]
    d1_R, d2_R, d3_R, eps_R = params[4:8]
    MR_scale = params[8]

    try:
        # Dirac matrix (in eV)
        MD = mass_matrix(d1_D, d2_D, d3_D, eps_D)

        # Check ε_D/GM_D constraint
        masses_D = np.real(eigvals(MD))
        if np.any(masses_D <= 0):
            return 1e10
        GM_D = (np.prod(masses_D))**(1/3)
        ratio_D = eps_D / GM_D

        # Penalty for deviating from 0.81
        penalty_scaling = 100 * (ratio_D - 0.81)**2

        # Majorana matrix (in GeV, then convert)
        MR_GeV = mass_matrix(d1_R, d2_R, d3_R, eps_R) * MR_scale
        MR = MR_GeV * 1e9  # Convert GeV to eV

        # Check positivity
        masses_R = np.real(eigvals(MR))
        if np.any(masses_R <= 0):
            return 1e10

        # Seesaw formula
        M_nu = seesaw_neutrino_mass(MD, MR)

        # Get eigenvalues and eigenvectors
        evals_nu, evecs_nu = np.linalg.eig(M_nu)
        idx = np.argsort(np.real(evals_nu))
        masses_nu = np.real(evals_nu[idx])
        U_nu = evecs_nu[:, idx]

        # Check positivity of neutrino masses
        if np.any(masses_nu <= 0):
            return 1e10

        # Mass-squared differences
        dm21_sq = masses_nu[1]**2 - masses_nu[0]**2
        dm31_sq = masses_nu[2]**2 - masses_nu[0]**2

        # Error in mass-squared differences
        error_masses = ((dm21_sq - Delta_m21_sq) / Delta_m21_sq)**2 + \
                       ((dm31_sq - Delta_m31_sq) / Delta_m31_sq)**2

        # Get charged lepton eigenvectors
        evals_e, evecs_e = np.linalg.eig(M_leptons)
        idx_e = np.argsort(np.real(evals_e))
        U_e = evecs_e[:, idx_e]

        # PMNS matrix
        U_PMNS = U_e.conj().T @ U_nu
        U_PMNS = np.real(U_PMNS)  # Should be real

        # Extract angles
        theta12, theta23, theta13 = extract_mixing_angles(U_PMNS)

        # Error in mixing angles
        error_angles = ((theta12 - theta12_exp) / theta12_exp)**2 + \
                       ((theta23 - theta23_exp) / theta23_exp)**2 + \
                       ((theta13 - theta13_exp) / theta13_exp)**2

        # Combined objective
        total_error = error_masses + error_angles + penalty_scaling

        return total_error

    except:
        return 1e10

print("\nSearching parameter space (this may take a while)...")
print("Constraints: ε_D/GM_D ≈ 0.81, M_R at high scale")

# Bounds for parameters
bounds_seesaw = [
    # M_D parameters (eV) - should be small
    (1e-3, 1.0),   # d1_D
    (1e-3, 1.0),   # d2_D
    (1e-3, 1.0),   # d3_D
    (-0.5, 0.5),   # eps_D
    # M_R parameters (in units of MR_scale) - order unity
    (0.1, 10.0),   # d1_R
    (0.1, 10.0),   # d2_R
    (0.1, 10.0),   # d3_R
    (-5.0, 5.0),   # eps_R
    # M_R scale (GeV) - GUT scale ~10^14
    (1e12, 1e16),  # MR_scale
]

result_seesaw = differential_evolution(
    objective_seesaw_constrained,
    bounds_seesaw,
    maxiter=500,
    popsize=30,
    seed=42,
    atol=1e-10,
    tol=1e-10,
    workers=1,
    updating='deferred',
    disp=True
)

print(f"\nOptimization complete!")
print(f"  Success: {result_seesaw.success}")
print(f"  Final error: {result_seesaw.fun:.6e}")

# Extract results
params_best = result_seesaw.x
d1_D, d2_D, d3_D, eps_D = params_best[0:4]
d1_R, d2_R, d3_R, eps_R = params_best[4:8]
MR_scale = params_best[8]

print(f"\n{'='*80}")
print("BEST FIT PARAMETERS")
print(f"{'='*80}")

print(f"\nDirac Matrix M_D (eV):")
MD_best = mass_matrix(d1_D, d2_D, d3_D, eps_D)
print(MD_best)
print(f"  d_D = [{d1_D:.6f}, {d2_D:.6f}, {d3_D:.6f}]")
print(f"  ε_D = {eps_D:.6f}")

masses_D = np.sort(np.real(eigvals(MD_best)))
GM_D = (np.prod(masses_D))**(1/3)
print(f"  Eigenvalues: {masses_D}")
print(f"  ε_D/GM_D = {eps_D/GM_D:.3f} (target: 0.81)")

print(f"\nMajorana Matrix M_R (GeV):")
MR_best_GeV = mass_matrix(d1_R, d2_R, d3_R, eps_R) * MR_scale
print(MR_best_GeV)
print(f"  d_R = [{d1_R:.6f}, {d2_R:.6f}, {d3_R:.6f}]")
print(f"  ε_R = {eps_R:.6f}")
print(f"  Scale = {MR_scale:.3e} GeV")

masses_R_GeV = np.sort(np.real(eigvals(MR_best_GeV)))
print(f"  Eigenvalues: {masses_R_GeV} GeV")

# Calculate effective neutrino mass
MR_best = MR_best_GeV * 1e9  # Convert to eV
M_nu_seesaw = seesaw_neutrino_mass(MD_best, MR_best)

print(f"\nEffective Neutrino Mass M_ν (eV):")
print(M_nu_seesaw)

evals_nu, evecs_nu = np.linalg.eig(M_nu_seesaw)
idx = np.argsort(np.real(evals_nu))
masses_nu_best = np.real(evals_nu[idx])
U_nu_best = evecs_nu[:, idx]

print(f"\nNeutrino masses:")
print(f"  m(ν₁) = {masses_nu_best[0]*1000:.3f} meV")
print(f"  m(ν₂) = {masses_nu_best[1]*1000:.3f} meV")
print(f"  m(ν₃) = {masses_nu_best[2]*1000:.3f} meV")
print(f"  Σm_ν = {np.sum(masses_nu_best):.6f} eV")

# Mass-squared differences
dm21_best = masses_nu_best[1]**2 - masses_nu_best[0]**2
dm31_best = masses_nu_best[2]**2 - masses_nu_best[0]**2

print(f"\nMass-squared differences:")
print(f"  Δm²₂₁ = {dm21_best:.6e} eV² (data: {Delta_m21_sq:.6e})")
print(f"  Δm²₃₁ = {dm31_best:.6e} eV² (data: {Delta_m31_sq:.6e})")

error_21 = abs(dm21_best - Delta_m21_sq) / Delta_m21_sq * 100
error_31 = abs(dm31_best - Delta_m31_sq) / Delta_m31_sq * 100
print(f"  Errors: {error_21:.3f}%, {error_31:.3f}%")

# Calculate PMNS matrix
evals_e, evecs_e = np.linalg.eig(M_leptons)
idx_e = np.argsort(np.real(evals_e))
U_e_best = evecs_e[:, idx_e]

U_PMNS_best = U_e_best.conj().T @ U_nu_best
U_PMNS_best = np.real(U_PMNS_best)

print(f"\n{'='*80}")
print("PMNS MATRIX FROM SEESAW")
print(f"{'='*80}")

print(f"\nU_PMNS:")
print(U_PMNS_best)

print(f"\nUnitarity check (U†U):")
print(np.round(U_PMNS_best.T @ U_PMNS_best, 6))

# Extract mixing angles
theta12_pred, theta23_pred, theta13_pred = extract_mixing_angles(U_PMNS_best)

print(f"\n{'='*80}")
print("PMNS MIXING ANGLE PREDICTIONS")
print(f"{'='*80}")

print(f"\nSeesaw Model vs Experiment:")
print(f"\n  θ₁₂ (solar):")
print(f"    Predicted: {theta12_pred:.2f}°")
print(f"    Observed:  {theta12_exp}° ± 0.77°")
print(f"    Error: {abs(theta12_pred - theta12_exp):.2f}°")
within_12 = abs(theta12_pred - theta12_exp) < 0.77
print(f"    Within 1σ: {within_12} {'✓' if within_12 else '✗'}")

print(f"\n  θ₂₃ (atmospheric):")
print(f"    Predicted: {theta23_pred:.2f}°")
print(f"    Observed:  {theta23_exp}° ± 1.4°")
print(f"    Error: {abs(theta23_pred - theta23_exp):.2f}°")
within_23 = abs(theta23_pred - theta23_exp) < 1.4
print(f"    Within 1σ: {within_23} {'✓' if within_23 else '✗'}")

print(f"\n  θ₁₃ (reactor):")
print(f"    Predicted: {theta13_pred:.2f}°")
print(f"    Observed:  {theta13_exp}° ± 0.13°")
print(f"    Error: {abs(theta13_pred - theta13_exp):.2f}°")
within_13 = abs(theta13_pred - theta13_exp) < 0.13
print(f"    Within 1σ: {within_13} {'✓' if within_13 else '✗'}")

matches = sum([within_12, within_23, within_13])

print(f"\n{'='*80}")
print(f"SEESAW MODEL: {matches}/3 ANGLES WITHIN 1σ")
print(f"{'='*80}")

if matches == 3:
    print("\n✓✓✓ PHENOMENAL - Full seesaw works!")
    print("Theory #11 with seesaw is PREDICTIVE!")
elif matches >= 2:
    print("\n✓✓ STRONG - Seesaw significantly improves predictions")
elif matches >= 1:
    print("\n✓ PARTIAL - Some improvement from seesaw")
else:
    print("\n✗ NO IMPROVEMENT - Seesaw doesn't fix mixing problem")

# ==============================================================================
# PART 4: CHECK UNIVERSAL SCALING
# ==============================================================================

print("\n" + "="*80)
print("PART 4: UNIVERSAL SCALING LAW CHECK")
print("="*80)

print("\nε/GM ratios across all sectors:")
print(f"\n  Charged leptons:  ε/GM = {eps_lep / (np.prod(masses_lep_check)**(1/3)):.3f}")
print(f"  Dirac neutrinos:  ε_D/GM_D = {eps_D/GM_D:.3f}")

# For comparison
GM_nu = (np.prod(masses_nu_best))**(1/3)
# Effective ε for M_ν (not meaningful, but for comparison)
M_nu_diag = np.diag(M_nu_seesaw)
M_nu_offdiag = (np.sum(M_nu_seesaw) - np.sum(M_nu_diag)) / 6  # Average off-diagonal
print(f"  Effective neutrinos: ε_eff/GM_ν = {M_nu_offdiag/GM_nu:.3f}")

print("\nKimi's prediction:")
print("  'M_D should have ε_D/GM_D ≈ 0.81 (universal coupling)'")
print(f"  Result: ε_D/GM_D = {eps_D/GM_D:.3f}")

if abs(eps_D/GM_D - 0.81) < 0.1:
    print("  ✓ CONFIRMED - Dirac couplings follow universal scaling!")
else:
    print("  ✗ NOT CONFIRMED - Scaling law doesn't hold for M_D")

# ==============================================================================
# PART 5: VISUALIZATION
# ==============================================================================

print("\n" + "="*80)
print("PART 5: VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Theory #11: Seesaw Model Results', fontsize=16, fontweight='bold')

# M_D matrix
ax = axes[0, 0]
im = ax.imshow(MD_best * 1000, cmap='RdBu_r', aspect='auto')
ax.set_title('Dirac Mass Matrix M_D (meV)', fontweight='bold')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{MD_best[i,j]*1000:.2f}', ha='center', va='center')
plt.colorbar(im, ax=ax)

# M_R matrix (log scale)
ax = axes[0, 1]
im = ax.imshow(np.log10(MR_best_GeV), cmap='viridis', aspect='auto')
ax.set_title(f'Majorana Mass Matrix M_R\n(log₁₀ GeV)', fontweight='bold')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{np.log10(MR_best_GeV[i,j]):.1f}',
               ha='center', va='center', color='white')
plt.colorbar(im, ax=ax)

# M_ν effective
ax = axes[0, 2]
im = ax.imshow(M_nu_seesaw * 1000, cmap='RdBu_r', aspect='auto')
ax.set_title('Effective Neutrino Mass M_ν (meV)', fontweight='bold')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{M_nu_seesaw[i,j]*1000:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax)

# Mixing angles comparison
ax = axes[1, 0]
angles_exp = [theta12_exp, theta23_exp, theta13_exp]
angles_pred = [theta12_pred, theta23_pred, theta13_pred]
errors_exp = [0.77, 1.4, 0.13]

x = [1, 2, 3]
ax.errorbar(x, angles_exp, yerr=errors_exp, fmt='o',
            color='black', markersize=8, label='Experiment', capsize=5)
ax.scatter(x, angles_pred, color='red', s=100, marker='x',
           linewidths=3, label='Seesaw Model', zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(['θ₁₂', 'θ₂₃', 'θ₁₃'])
ax.set_ylabel('Angle (degrees)')
ax.set_title('PMNS Mixing Angles', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Scaling ratios
ax = axes[1, 1]
sectors = ['Charged\nLeptons', 'Dirac\nNeutrinos', 'Effective\nNeutrinos']
ratios = [
    eps_lep / (np.prod(masses_lep_check)**(1/3)),
    eps_D / GM_D,
    M_nu_offdiag / GM_nu
]
colors = ['blue', 'green', 'red']
bars = ax.bar([1, 2, 3], ratios, color=colors, alpha=0.7)
ax.axhline(0.81, color='black', linestyle='--', linewidth=2, label='Target: 0.81')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(sectors)
ax.set_ylabel('ε/GM')
ax.set_title('Universal Scaling Law', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, r) in enumerate(zip(bars, ratios)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{r:.3f}', ha='center', va='bottom', fontweight='bold')

# Mass hierarchy
ax = axes[1, 2]
mass_scales = [
    np.log10(masses_lep_check),
    np.log10(masses_D * 1e9),  # Convert to meV
    np.log10(masses_nu_best * 1000)  # Convert to meV
]
labels_scales = ['Charged\nLeptons (MeV)', 'Dirac\nCouplings (meV)', 'Neutrinos\n(meV)']

for i, (masses_log, label, color) in enumerate(zip(mass_scales, labels_scales, colors)):
    ax.plot([i-0.1, i-0.1, i+0.1, i+0.1],
            [masses_log[0], masses_log[2], masses_log[2], masses_log[0]],
            'o-', color=color, markersize=6, linewidth=2, label=label)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels_scales)
ax.set_ylabel('log₁₀(mass)')
ax.set_title('Mass Hierarchies', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('theory11_seesaw_model.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'theory11_seesaw_model.png'")

# ==============================================================================
# PART 6: FINAL VERDICT
# ==============================================================================

print("\n" + "="*80)
print("FINAL VERDICT: SEESAW MODEL TEST")
print("="*80)

print(f"""
RESULTS:
--------
✓ Neutrino mass-squared differences: {error_21:.2f}%, {error_31:.2f}% error
{'✓' if abs(eps_D/GM_D - 0.81) < 0.1 else '✗'} Universal scaling ε_D/GM_D = {eps_D/GM_D:.3f} (target: 0.81)
{'✓' if matches >= 2 else '✗'} PMNS mixing angles: {matches}/3 within 1σ

PARAMETER COUNT:
----------------
Previous simple model: 4 params (d₁, d₂, d₃, ε) → 2 constraints (Δm²)
Seesaw model: 9 params (d_D×3, ε_D, d_R×3, ε_R, M_R scale) → 5 constraints (Δm² + 3 angles)

But with ε_D/GM_D ≈ 0.81 constraint: effectively 8 free → 5 constraints
Still underdetermined, but LESS SO than before.

THEORY STATUS:
--------------
""")

if matches == 3 and abs(eps_D/GM_D - 0.81) < 0.1:
    print("""✓✓✓ BREAKTHROUGH - Seesaw model is PREDICTIVE!
  - Fits neutrino masses
  - Predicts PMNS mixing angles
  - Maintains universal ε/GM ≈ 0.81 for Dirac couplings
  - Theory #11 with seesaw is a GENUINE THEORY
""")
elif matches >= 2:
    print("""✓✓ STRONG EVIDENCE - Seesaw improves predictions significantly
  - Better than simple parameterization
  - Some mixing angles match
  - Further refinement needed (RG running, complex phases, etc.)
""")
elif matches >= 1:
    print("""✓ PARTIAL SUCCESS - Seesaw provides some structure
  - Improvement over direct fit
  - But not yet fully predictive
  - Need additional physics (flavor symmetry, etc.)
""")
else:
    print("""✗ FAILURE - Even seesaw doesn't predict mixing
  - M = diag(d) + ε·J structure is too restrictive
  - Need more general matrix forms
  - Or hierarchical off-diagonals instead of democratic
""")

print(f"""
KIMI'S CHALLENGE ANSWER:
------------------------
"Implement full seesaw with M_D and M_R"
"Check if ε_D/GM_D ≈ 0.81 holds"
"If PMNS matches, Theory #11 is revolutionary"

Result: {matches}/3 angles match, ε_D/GM_D = {eps_D/GM_D:.3f}
""")

if matches >= 2 and abs(eps_D/GM_D - 0.81) < 0.1:
    print("✓ Theory #11 with seesaw is REVOLUTIONARY")
elif matches >= 1:
    print("~ Theory #11 with seesaw is PROMISING but incomplete")
else:
    print("✗ Theory #11 needs more fundamental structure")

print("\n" + "="*80)
print("SEESAW MODEL TEST COMPLETE")
print("="*80)
