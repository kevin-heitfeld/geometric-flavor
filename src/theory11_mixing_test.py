"""
THEORY #11: MIXING ANGLE TEST

Critical test from Kimi's analysis:
- Calculate PMNS mixing matrix from M_e and M_nu
- Calculate CKM mixing matrix from M_u and M_d
- Compare to experimental values

This is THE decisive test:
- If mixing angles match → genuine theory
- If they don't match → just parameterization

Experimental values (PDG 2024):
  PMNS:
    θ₁₂ = 33.45° ± 0.77°  (solar)
    θ₂₃ = 49.0° ± 1.4°    (atmospheric)
    θ₁₃ = 8.57° ± 0.13°   (reactor)
    
  CKM:
    θ₁₂ = 13.04° ± 0.05°  (Cabibbo angle)
    θ₂₃ = 2.38° ± 0.06°   
    θ₁₃ = 0.201° ± 0.011°
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigvals

print("="*80)
print("THEORY #11: MIXING ANGLE PREDICTIONS")
print("="*80)

# ==============================================================================
# PART 1: RECONSTRUCT MASS MATRICES
# ==============================================================================

print("\n" + "="*80)
print("PART 1: RECONSTRUCT MASS MATRICES FROM PREVIOUS FITS")
print("="*80)

def mass_matrix(d1, d2, d3, epsilon):
    """M = diag(d) + ε·J"""
    M = np.diag([d1, d2, d3])
    M += epsilon * np.ones((3, 3))
    return M

# From theory11_quarks_test.py
print("\nCharged Leptons (e, μ, τ):")
d_lep = [15.77, 92.03, 1775.23]  # MeV
eps_lep = 37.03  # MeV
M_leptons = mass_matrix(*d_lep, eps_lep)
masses_lep = np.sort(np.real(eigvals(M_leptons)))
print(f"  Parameters: d = {d_lep} MeV, ε = {eps_lep} MeV")
print(f"  Masses: {masses_lep}")

print("\nUp Quarks (u, c, t):")
d_up = [0.92, 3.40, 172119.00]  # MeV
eps_up = 636.28  # MeV
M_up = mass_matrix(*d_up, eps_up)
masses_up = np.sort(np.real(eigvals(M_up)))
print(f"  Parameters: d = {d_up} MeV, ε = {eps_up} MeV")
print(f"  Masses: {masses_up}")

print("\nDown Quarks (d, s, b):")
d_down = [5.04, 93.77, 4180.37]  # MeV
eps_down = -0.37  # MeV
M_down = mass_matrix(*d_down, eps_down)
masses_down = np.sort(np.real(eigvals(M_down)))
print(f"  Parameters: d = {d_down} MeV, ε = {eps_down} MeV")
print(f"  Masses: {masses_down}")

print("\nNeutrinos (ν₁, ν₂, ν₃) - Normal Hierarchy:")
d_nu = [0.051727, 0.011870, 0.013678]  # eV
eps_nu = -0.001340  # eV
M_neutrinos = mass_matrix(*d_nu, eps_nu)
masses_nu = np.sort(np.real(eigvals(M_neutrinos)))
print(f"  Parameters: d = {d_nu} eV, ε = {eps_nu} eV")
print(f"  Masses: {masses_nu * 1000} meV")

# ==============================================================================
# PART 2: CALCULATE MIXING MATRICES
# ==============================================================================

print("\n" + "="*80)
print("PART 2: CALCULATE MIXING MATRICES")
print("="*80)

def calculate_mixing_matrix(M1, M2, name1="M1", name2="M2"):
    """
    Calculate mixing matrix U = U1† U2
    
    M1, M2: Mass matrices
    Returns: mixing matrix and angles
    """
    # Diagonalize both matrices
    evals1, U1 = np.linalg.eig(M1)
    evals2, U2 = np.linalg.eig(M2)
    
    # Sort by eigenvalues
    idx1 = np.argsort(np.real(evals1))
    idx2 = np.argsort(np.real(evals2))
    U1 = U1[:, idx1]
    U2 = U2[:, idx2]
    
    # Mixing matrix
    U_mix = U1.conj().T @ U2
    
    # Make it real by phase choice (convention)
    # Choose phases so diagonal elements are positive
    for i in range(3):
        if np.real(U_mix[i, i]) < 0:
            U_mix[:, i] *= -1
    
    return np.real(U_mix), U1, U2

def extract_mixing_angles(U, convention="standard"):
    """
    Extract mixing angles from 3×3 unitary matrix
    
    Standard parametrization:
    U = R₂₃(θ₂₃) · R₁₃(θ₁₃) · R₁₂(θ₁₂)
    
    where R_ij is rotation in i-j plane
    """
    # From PDG parametrization
    # |U| = | c12 c13         s12 c13        s13     |
    #       |-s12 c23-c12 s23 s13  c12 c23-s12 s23 s13  s23 c13 |
    #       | s12 s23-c12 c23 s13 -c12 s23-s12 c23 s13  c23 c13 |
    
    # Extract angles
    s13 = abs(U[0, 2])
    theta13 = np.arcsin(min(s13, 1.0))
    
    c13 = np.cos(theta13)
    if c13 > 1e-10:
        s12 = abs(U[0, 1]) / c13
        c12 = abs(U[0, 0]) / c13
        s23 = abs(U[1, 2]) / c13
        c23 = abs(U[2, 2]) / c13
    else:
        # θ₁₃ ≈ 90°, special case
        s12 = 0
        c12 = 1
        s23 = abs(U[1, 2])
        c23 = abs(U[2, 2])
    
    theta12 = np.arctan2(s12, c12)
    theta23 = np.arctan2(s23, c23)
    
    # Convert to degrees
    angles_deg = [theta12 * 180/np.pi, theta23 * 180/np.pi, theta13 * 180/np.pi]
    
    return angles_deg

# ==============================================================================
# PART 3: PMNS MATRIX (Lepton Mixing)
# ==============================================================================

print("\n" + "="*80)
print("PART 3: PMNS MATRIX (LEPTON SECTOR)")
print("="*80)

print("\nCalculating U_PMNS = U_e† U_ν...")

U_PMNS, U_e, U_nu = calculate_mixing_matrix(M_leptons, M_neutrinos, "M_e", "M_ν")

print("\nPMNS Matrix:")
print(U_PMNS)

print("\nChecking unitarity: U†U =")
print(np.round(U_PMNS.T @ U_PMNS, 6))

# Extract mixing angles
theta12_pmns, theta23_pmns, theta13_pmns = extract_mixing_angles(U_PMNS)

print("\n" + "="*80)
print("PMNS MIXING ANGLE PREDICTIONS")
print("="*80)

print(f"\nTheory #11 Predictions vs Experiment:")
print(f"\n  θ₁₂ (solar):")
print(f"    Predicted: {theta12_pmns:.2f}°")
print(f"    Observed:  33.45° ± 0.77°")
print(f"    Error: {abs(theta12_pmns - 33.45):.2f}°")
print(f"    Within 1σ: {abs(theta12_pmns - 33.45) < 0.77}")

print(f"\n  θ₂₃ (atmospheric):")
print(f"    Predicted: {theta23_pmns:.2f}°")
print(f"    Observed:  49.0° ± 1.4°")
print(f"    Error: {abs(theta23_pmns - 49.0):.2f}°")
print(f"    Within 1σ: {abs(theta23_pmns - 49.0) < 1.4}")

print(f"\n  θ₁₃ (reactor):")
print(f"    Predicted: {theta13_pmns:.2f}°")
print(f"    Observed:  8.57° ± 0.13°")
print(f"    Error: {abs(theta13_pmns - 8.57):.2f}°")
print(f"    Within 1σ: {abs(theta13_pmns - 8.57) < 0.13}")

# Overall assessment
errors_pmns = [
    abs(theta12_pmns - 33.45),
    abs(theta23_pmns - 49.0),
    abs(theta13_pmns - 8.57)
]
within_1sigma = [
    errors_pmns[0] < 0.77,
    errors_pmns[1] < 1.4,
    errors_pmns[2] < 0.13
]

print(f"\n{'='*80}")
print(f"PMNS VERDICT: {sum(within_1sigma)}/3 angles within 1σ")
if all(within_1sigma):
    print("✓✓✓ PHENOMENAL SUCCESS - Theory #11 predicts PMNS!")
elif sum(within_1sigma) >= 2:
    print("✓✓ STRONG SUCCESS - 2/3 angles match")
elif sum(within_1sigma) >= 1:
    print("✓ PARTIAL SUCCESS - Some predictive power")
else:
    print("✗ FAILURE - Just parameterization")
print("="*80)

# ==============================================================================
# PART 4: CKM MATRIX (Quark Mixing)
# ==============================================================================

print("\n" + "="*80)
print("PART 4: CKM MATRIX (QUARK SECTOR)")
print("="*80)

print("\nCalculating V_CKM = U_u† U_d...")

V_CKM, U_u, U_d = calculate_mixing_matrix(M_up, M_down, "M_u", "M_d")

print("\nCKM Matrix:")
print(V_CKM)

print("\nChecking unitarity: V†V =")
print(np.round(V_CKM.T @ V_CKM, 6))

# Extract mixing angles
theta12_ckm, theta23_ckm, theta13_ckm = extract_mixing_angles(V_CKM)

print("\n" + "="*80)
print("CKM MIXING ANGLE PREDICTIONS")
print("="*80)

print(f"\nTheory #11 Predictions vs Experiment:")
print(f"\n  θ₁₂ (Cabibbo):")
print(f"    Predicted: {theta12_ckm:.2f}°")
print(f"    Observed:  13.04° ± 0.05°")
print(f"    Error: {abs(theta12_ckm - 13.04):.2f}°")
print(f"    Within 1σ: {abs(theta12_ckm - 13.04) < 0.05}")

print(f"\n  θ₂₃:")
print(f"    Predicted: {theta23_ckm:.2f}°")
print(f"    Observed:  2.38° ± 0.06°")
print(f"    Error: {abs(theta23_ckm - 2.38):.2f}°")
print(f"    Within 1σ: {abs(theta23_ckm - 2.38) < 0.06}")

print(f"\n  θ₁₃:")
print(f"    Predicted: {theta13_ckm:.2f}°")
print(f"    Observed:  0.201° ± 0.011°")
print(f"    Error: {abs(theta13_ckm - 0.201):.2f}°")
print(f"    Within 1σ: {abs(theta13_ckm - 0.201) < 0.011}")

# Overall assessment
errors_ckm = [
    abs(theta12_ckm - 13.04),
    abs(theta23_ckm - 2.38),
    abs(theta13_ckm - 0.201)
]
within_1sigma_ckm = [
    errors_ckm[0] < 0.05,
    errors_ckm[1] < 0.06,
    errors_ckm[2] < 0.011
]

print(f"\n{'='*80}")
print(f"CKM VERDICT: {sum(within_1sigma_ckm)}/3 angles within 1σ")
if all(within_1sigma_ckm):
    print("✓✓✓ PHENOMENAL SUCCESS - Theory #11 predicts CKM!")
elif sum(within_1sigma_ckm) >= 2:
    print("✓✓ STRONG SUCCESS - 2/3 angles match")
elif sum(within_1sigma_ckm) >= 1:
    print("✓ PARTIAL SUCCESS - Some predictive power")
else:
    print("✗ FAILURE - Just parameterization")
print("="*80)

# ==============================================================================
# PART 5: COMPARISON
# ==============================================================================

print("\n" + "="*80)
print("PART 5: QUARK VS LEPTON MIXING")
print("="*80)

print("\nKey Observation: PMNS vs CKM")
print(f"\n  Largest angle:")
print(f"    PMNS: θ₂₃ ~ {theta23_pmns:.0f}° (large atmospheric mixing)")
print(f"    CKM:  θ₁₂ ~ {theta12_ckm:.0f}° (Cabibbo angle)")
print(f"    Ratio: {theta23_pmns/theta12_ckm:.1f}× larger")

print(f"\n  Smallest angle:")
print(f"    PMNS: θ₁₃ ~ {theta13_pmns:.1f}° (reactor)")
print(f"    CKM:  θ₁₃ ~ {theta13_ckm:.1f}° (tiny)")
print(f"    Ratio: {theta13_pmns/theta13_ckm:.0f}× larger")

print("\n  Hierarchy:")
print(f"    PMNS: θ₂₃ > θ₁₂ > θ₁₃  (large mixing)")
print(f"    CKM:  θ₁₂ > θ₂₃ > θ₁₃  (small mixing)")

print("\nPhysical interpretation:")
print("  - Quarks: Small mixing → heavy mass differences")
print("  - Leptons: Large mixing → special neutrino physics (seesaw?)")

# ==============================================================================
# PART 6: VISUALIZATION
# ==============================================================================

print("\n" + "="*80)
print("PART 6: VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Theory #11: Mixing Angle Predictions', fontsize=16, fontweight='bold')

# PMNS angles
ax = axes[0, 0]
angles_pmns_exp = [33.45, 49.0, 8.57]
angles_pmns_pred = [theta12_pmns, theta23_pmns, theta13_pmns]
errors_pmns_exp = [0.77, 1.4, 0.13]

x = [1, 2, 3]
ax.errorbar(x, angles_pmns_exp, yerr=errors_pmns_exp, fmt='o', 
            color='black', markersize=8, label='Experiment', capsize=5)
ax.scatter(x, angles_pmns_pred, color='red', s=100, marker='x', 
           linewidths=3, label='Theory #11', zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(['θ₁₂', 'θ₂₃', 'θ₁₃'])
ax.set_ylabel('Angle (degrees)')
ax.set_title('PMNS Mixing Angles', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# CKM angles
ax = axes[0, 1]
angles_ckm_exp = [13.04, 2.38, 0.201]
angles_ckm_pred = [theta12_ckm, theta23_ckm, theta13_ckm]
errors_ckm_exp = [0.05, 0.06, 0.011]

ax.errorbar(x, angles_ckm_exp, yerr=errors_ckm_exp, fmt='o', 
            color='black', markersize=8, label='Experiment', capsize=5)
ax.scatter(x, angles_ckm_pred, color='blue', s=100, marker='x', 
           linewidths=3, label='Theory #11', zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(['θ₁₂', 'θ₂₃', 'θ₁₃'])
ax.set_ylabel('Angle (degrees)')
ax.set_title('CKM Mixing Angles', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Comparison
ax = axes[0, 2]
width = 0.35
x_pos = np.arange(3)
ax.bar(x_pos - width/2, angles_pmns_pred, width, label='PMNS', alpha=0.7, color='red')
ax.bar(x_pos + width/2, angles_ckm_pred, width, label='CKM', alpha=0.7, color='blue')
ax.set_xticks(x_pos)
ax.set_xticklabels(['θ₁₂', 'θ₂₃', 'θ₁₃'])
ax.set_ylabel('Angle (degrees)')
ax.set_title('PMNS vs CKM', fontweight='bold')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

# PMNS matrix visualization
ax = axes[1, 0]
im = ax.imshow(np.abs(U_PMNS)**2, cmap='viridis', aspect='auto', vmin=0, vmax=1)
ax.set_title('|U_PMNS|² (Probability)', fontweight='bold')
ax.set_xlabel('ν mass eigenstate')
ax.set_ylabel('ν flavor eigenstate')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['ν₁', 'ν₂', 'ν₃'])
ax.set_yticklabels(['νₑ', 'νμ', 'ντ'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{np.abs(U_PMNS[i,j])**2:.3f}', 
               ha='center', va='center', color='white', fontweight='bold')
plt.colorbar(im, ax=ax)

# CKM matrix visualization
ax = axes[1, 1]
im = ax.imshow(np.abs(V_CKM)**2, cmap='viridis', aspect='auto', vmin=0, vmax=1)
ax.set_title('|V_CKM|² (Probability)', fontweight='bold')
ax.set_xlabel('d-type mass eigenstate')
ax.set_ylabel('u-type mass eigenstate')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['d', 's', 'b'])
ax.set_yticklabels(['u', 'c', 't'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{np.abs(V_CKM[i,j])**2:.3f}', 
               ha='center', va='center', color='white', fontweight='bold')
plt.colorbar(im, ax=ax)

# Error comparison
ax = axes[1, 2]
labels = ['θ₁₂', 'θ₂₃', 'θ₁₃']
x_pos = np.arange(len(labels))
width = 0.35

errors_pmns_sigma = [e/s for e, s in zip(errors_pmns, [0.77, 1.4, 0.13])]
errors_ckm_sigma = [e/s for e, s in zip(errors_ckm, [0.05, 0.06, 0.011])]

ax.bar(x_pos - width/2, errors_pmns_sigma, width, label='PMNS', alpha=0.7, color='red')
ax.bar(x_pos + width/2, errors_ckm_sigma, width, label='CKM', alpha=0.7, color='blue')
ax.axhline(1, color='black', linestyle='--', linewidth=1, label='1σ')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Error (σ)')
ax.set_title('Prediction Errors', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('theory11_mixing_angles.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'theory11_mixing_angles.png'")

# ==============================================================================
# PART 7: FINAL VERDICT
# ==============================================================================

print("\n" + "="*80)
print("FINAL VERDICT: THEORY #11 PREDICTIVE POWER")
print("="*80)

total_within_sigma = sum(within_1sigma) + sum(within_1sigma_ckm)

print(f"\nMixing angle predictions: {total_within_sigma}/6 within 1σ")
print(f"  PMNS: {sum(within_1sigma)}/3")
print(f"  CKM:  {sum(within_1sigma_ckm)}/3")

print("\nAssessment:")
if total_within_sigma >= 5:
    print("  ★★★★★ EXTRAORDINARY - Theory #11 is PREDICTIVE!")
    print("  This is NOT just parameterization - it derives mixing from mass structure!")
elif total_within_sigma >= 4:
    print("  ★★★★ EXCELLENT - Strong predictive power")
    print("  Theory #11 captures essential physics of flavor mixing")
elif total_within_sigma >= 3:
    print("  ★★★ GOOD - Significant predictive content")
    print("  More than coincidence, less than complete theory")
elif total_within_sigma >= 2:
    print("  ★★ FAIR - Some predictive power")
    print("  Qualitative agreement, quantitative refinement needed")
else:
    print("  ★ POOR - Minimal predictive power")
    print("  Matrices parameterize masses but don't predict mixing")

print("\nKimi's Challenge Answer:")
print("  'If mixing angles match, Theory #11 is genuine theory.'")
print("  'If they don't match, it's just parameterization.'")
print(f"\n  Result: {total_within_sigma}/6 angles within 1σ")

if total_within_sigma >= 4:
    print("\n  ✓ Theory #11 PASSES the mixing angle test!")
    print("  ✓ This is a GENUINE THEORY of flavor, not just a fit!")
else:
    print("\n  ✗ Theory #11 needs refinement for mixing")
    print("  → Consider full seesaw implementation")
    print("  → May need complex phases or RG running")

print("\n" + "="*80)
print("MIXING ANGLE TEST COMPLETE")
print("="*80)
