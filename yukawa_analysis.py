"""
YUKAWA MATRIX STRUCTURE ANALYSIS

Goal: Extract patterns from SM Yukawa couplings
      Look for symmetries, texture, hierarchical structure

Data: Experimental fermion masses → Yukawa couplings
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==============================================================================
# PART 1: EXPERIMENTAL DATA
# ==============================================================================

print("="*80)
print("YUKAWA MATRIX STRUCTURE ANALYSIS")
print("="*80)

# Higgs VEV
v = 246.22  # GeV (electroweak scale)

# Charged Lepton Masses (MeV)
m_e = 0.5109989461  # electron
m_mu = 105.6583745   # muon
m_tau = 1776.86      # tau

# Quark Masses (MeV) - running masses at μ = 2 GeV
m_u = 2.16   # up
m_d = 4.67   # down
m_s = 93.4   # strange
m_c = 1270   # charm
m_b = 4180   # bottom
m_t = 172760 # top (pole mass)

# Neutrino mass-squared differences (eV²) - from oscillations
# We don't know absolute scale, only differences
delta_m21_sq = 7.53e-5  # solar
delta_m32_sq = 2.453e-3  # atmospheric

print("\n" + "="*80)
print("PART 1: EXPERIMENTAL MASSES")
print("="*80)

print("\nCharged Leptons (MeV):")
print(f"  e:  {m_e:.6f}")
print(f"  μ:  {m_mu:.6f}")
print(f"  τ:  {m_tau:.2f}")

print("\nQuarks (MeV):")
print(f"  u:  {m_u:.2f}")
print(f"  d:  {m_d:.2f}")
print(f"  c:  {m_c:.0f}")
print(f"  s:  {m_s:.1f}")
print(f"  t:  {m_t:.0f}")
print(f"  b:  {m_b:.0f}")

# ==============================================================================
# PART 2: YUKAWA COUPLINGS
# ==============================================================================

print("\n" + "="*80)
print("PART 2: YUKAWA COUPLINGS")
print("="*80)

# Convert masses to Yukawa couplings: y_f = sqrt(2) * m_f / v
def mass_to_yukawa(m_MeV):
    """Convert mass in MeV to Yukawa coupling (dimensionless)"""
    m_GeV = m_MeV / 1000.0
    return np.sqrt(2) * m_GeV / v

# Charged Leptons
y_e = mass_to_yukawa(m_e)
y_mu = mass_to_yukawa(m_mu)
y_tau = mass_to_yukawa(m_tau)

# Quarks
y_u = mass_to_yukawa(m_u)
y_d = mass_to_yukawa(m_d)
y_c = mass_to_yukawa(m_c)
y_s = mass_to_yukawa(m_s)
y_t = mass_to_yukawa(m_t)
y_b = mass_to_yukawa(m_b)

print("\nCharged Lepton Yukawa Couplings:")
print(f"  y_e  = {y_e:.6e}")
print(f"  y_μ  = {y_mu:.6e}")
print(f"  y_τ  = {y_tau:.6e}")

print("\nUp-type Quark Yukawa Couplings:")
print(f"  y_u  = {y_u:.6e}")
print(f"  y_c  = {y_c:.6e}")
print(f"  y_t  = {y_t:.6e}")

print("\nDown-type Quark Yukawa Couplings:")
print(f"  y_d  = {y_d:.6e}")
print(f"  y_s  = {y_s:.6e}")
print(f"  y_b  = {y_b:.6e}")

# ==============================================================================
# PART 3: YUKAWA MATRICES (Diagonal Basis)
# ==============================================================================

print("\n" + "="*80)
print("PART 3: YUKAWA MATRICES (DIAGONAL BASIS)")
print("="*80)

# In mass eigenstate basis, Yukawa matrices are diagonal
Y_e = np.diag([y_e, y_mu, y_tau])
Y_u = np.diag([y_u, y_c, y_t])
Y_d = np.diag([y_d, y_s, y_b])

print("\nCharged Lepton Yukawa Matrix Y_e:")
print(Y_e)

print("\nUp Quark Yukawa Matrix Y_u:")
print(Y_u)

print("\nDown Quark Yukawa Matrix Y_d:")
print(Y_d)

# ==============================================================================
# PART 4: HIERARCHIES AND RATIOS
# ==============================================================================

print("\n" + "="*80)
print("PART 4: HIERARCHIES AND RATIOS")
print("="*80)

def analyze_hierarchy(name, y1, y2, y3):
    """Analyze hierarchy pattern in 3 Yukawa couplings"""
    print(f"\n{name}:")
    print(f"  Absolute values:")
    print(f"    y1 = {y1:.6e}")
    print(f"    y2 = {y2:.6e}")
    print(f"    y3 = {y3:.6e}")

    print(f"  Ratios:")
    r21 = y2 / y1
    r32 = y3 / y2
    r31 = y3 / y1
    print(f"    y2/y1 = {r21:.2f}")
    print(f"    y3/y2 = {r32:.2f}")
    print(f"    y3/y1 = {r31:.2f}")

    print(f"  Log ratios:")
    print(f"    log10(y2/y1) = {np.log10(r21):.3f}")
    print(f"    log10(y3/y2) = {np.log10(r32):.3f}")
    print(f"    log10(y3/y1) = {np.log10(r31):.3f}")

    # Check for power law: y3/y2 ≈ (y2/y1)^α
    alpha = np.log(r32) / np.log(r21)
    print(f"  Power law test: (y2/y1)^α = y3/y2")
    print(f"    α = {alpha:.3f}")
    print(f"    (y2/y1)^{alpha:.3f} = {r21**alpha:.2f} vs y3/y2 = {r32:.2f}")

    # Span
    span = y3 / y1
    print(f"  Total span: {span:.1f}× ({np.log10(span):.1f} orders of magnitude)")

    return {'r21': r21, 'r32': r32, 'r31': r31, 'alpha': alpha}

# Analyze each sector
results = {}
results['leptons'] = analyze_hierarchy("Charged Leptons", y_e, y_mu, y_tau)
results['up_quarks'] = analyze_hierarchy("Up Quarks", y_u, y_c, y_t)
results['down_quarks'] = analyze_hierarchy("Down Quarks", y_d, y_s, y_b)

# ==============================================================================
# PART 5: CROSS-SECTOR COMPARISONS
# ==============================================================================

print("\n" + "="*80)
print("PART 5: CROSS-SECTOR COMPARISONS")
print("="*80)

print("\nCompare Gen1 across sectors:")
print(f"  y_e / y_u = {y_e/y_u:.3f}")
print(f"  y_d / y_u = {y_d/y_u:.3f}")
print(f"  y_e / y_d = {y_e/y_d:.3f}")

print("\nCompare Gen2 across sectors:")
print(f"  y_μ / y_c = {y_mu/y_c:.3f}")
print(f"  y_s / y_c = {y_s/y_c:.3f}")
print(f"  y_μ / y_s = {y_mu/y_s:.3f}")

print("\nCompare Gen3 across sectors:")
print(f"  y_τ / y_t = {y_tau/y_t:.3f}")
print(f"  y_b / y_t = {y_b/y_t:.3f}")
print(f"  y_τ / y_b = {y_tau/y_b:.3f}")

print("\nCross-generation comparisons:")
print(f"  y_s / y_μ = {y_s/y_mu:.3f}  (Gen2 quark / Gen2 lepton)")
print(f"  y_b / y_τ = {y_b/y_tau:.3f}  (Gen3 quark / Gen3 lepton)")

# ==============================================================================
# PART 6: TEXTURE ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("PART 6: TEXTURE ANALYSIS")
print("="*80)

print("\nCurrent texture (diagonal):")
print("  All off-diagonal elements = 0")
print("  This is mass eigenstate basis by construction")

print("\nQuestions:")
print("  - Is there a basis where structure is simpler?")
print("  - Are there texture zeros in flavor basis?")
print("  - Is hierarchical pattern universal?")

# CKM Matrix (quark mixing)
# Approximate values
V_ud = 0.974
V_us = 0.225
V_ub = 0.004
V_cd = 0.225
V_cs = 0.973
V_cb = 0.041
V_td = 0.009
V_ts = 0.040
V_tb = 0.999

V_CKM = np.array([
    [V_ud, V_us, V_ub],
    [V_cd, V_cs, V_cb],
    [V_td, V_ts, V_tb]
])

print("\nCKM Matrix (quark mixing):")
print(V_CKM)
print("\nCKM pattern: Nearly diagonal with small off-diagonal")
print("  Diagonal: ~1")
print("  First off-diagonal: ~0.2")
print("  Second off-diagonal: ~0.04")
print("  Third off-diagonal: ~0.004")
print("\nWolfenstein parameterization: powers of λ ≈ 0.22")

# PMNS Matrix (lepton mixing) - much different!
# Approximate values (tri-bimaximal mixing close)
s12 = 0.55  # sin(θ12) - solar
s23 = 0.71  # sin(θ23) - atmospheric (≈ 1/√2)
s13 = 0.15  # sin(θ13) - reactor

print("\nPMNS Matrix (lepton mixing) - VERY different from CKM!")
print(f"  sin²(θ12) ≈ {s12**2:.3f}  (solar, ~1/3)")
print(f"  sin²(θ23) ≈ {s23**2:.3f}  (atmospheric, ~1/2)")
print(f"  sin²(θ13) ≈ {s13**2:.3f}  (reactor, small)")
print("\nNOTE: Lepton mixing is LARGE, quark mixing is SMALL")

# ==============================================================================
# PART 7: SEARCH FOR PATTERNS
# ==============================================================================

print("\n" + "="*80)
print("PART 7: PATTERN SEARCH")
print("="*80)

# Pattern 1: Geometric sequence
print("\nPattern 1: Geometric sequence y_i = y_0 × r^i")
for name, y1, y2, y3 in [
    ("Leptons", y_e, y_mu, y_tau),
    ("Up quarks", y_u, y_c, y_t),
    ("Down quarks", y_d, y_s, y_b)
]:
    r12 = y2/y1
    r23 = y3/y2
    print(f"  {name}:")
    print(f"    r_12 = {r12:.2f}, r_23 = {r23:.2f}")
    print(f"    Ratio: r_23/r_12 = {r23/r12:.3f}")
    if abs(r23/r12 - 1.0) < 0.5:
        print(f"    → Nearly geometric!")
    else:
        print(f"    → Not geometric")

# Pattern 2: Exponential y_i = A * exp(B*i)
print("\nPattern 2: Exponential y_i = A × exp(B×i)")
for name, y1, y2, y3 in [
    ("Leptons", y_e, y_mu, y_tau),
    ("Up quarks", y_u, y_c, y_t),
    ("Down quarks", y_d, y_s, y_b)
]:
    # Fit: log(y_i) = log(A) + B*i
    # From y1, y2, y3 extract B
    B_12 = np.log(y2/y1)
    B_23 = np.log(y3/y2)
    print(f"  {name}:")
    print(f"    B_12 = {B_12:.3f}, B_23 = {B_23:.3f}")
    print(f"    Ratio: B_23/B_12 = {B_23/B_12:.3f}")
    if abs(B_23/B_12 - 1.0) < 0.3:
        print(f"    → Nearly exponential!")
        B_avg = (B_12 + B_23) / 2
        print(f"    Average B = {B_avg:.3f}")
    else:
        print(f"    → Not exponential")

# Pattern 3: Powers of small parameter ε
print("\nPattern 3: Powers of small parameter y_i ~ ε^(Q_i)")
print("  Assume y_3 = 1 (largest), others suppressed by powers")
for name, y1, y2, y3 in [
    ("Leptons", y_e, y_mu, y_tau),
    ("Up quarks", y_u, y_c, y_t),
    ("Down quarks", y_d, y_s, y_b)
]:
    # If y1 = ε^Q1, y2 = ε^Q2, y3 = ε^0 = 1
    # Then: Q1 = log(y1/y3)/log(ε), Q2 = log(y2/y3)/log(ε)
    # Try ε = 0.22 (Cabibbo angle)
    epsilon = 0.225
    Q1 = np.log(y1/y3) / np.log(epsilon)
    Q2 = np.log(y2/y3) / np.log(epsilon)
    print(f"  {name} (with ε = {epsilon}):")
    print(f"    Q_1 = {Q1:.2f}")
    print(f"    Q_2 = {Q2:.2f}")
    print(f"    Q_3 = 0")
    # Check if charges are near integers
    Q1_round = round(Q1)
    Q2_round = round(Q2)
    if abs(Q1 - Q1_round) < 0.3 and abs(Q2 - Q2_round) < 0.3:
        print(f"    → Charges near integers: ({Q1_round}, {Q2_round}, 0)")
    else:
        print(f"    → Charges not near integers")

# Pattern 4: Koide formula (leptons only)
print("\nPattern 4: Koide Formula (leptons)")
print("  K = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²")
K = (m_e + m_mu + m_tau) / (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
print(f"  K = {K:.8f}")
print(f"  Predicted: K = 2/3 = {2/3:.8f}")
print(f"  Difference: {abs(K - 2/3):.8f}")
print(f"  Relative error: {abs(K - 2/3)/(2/3) * 100:.4f}%")
if abs(K - 2/3) < 0.0001:
    print("  → EXTREMELY CLOSE TO 2/3!")
    print("  → This is NOT coincidence - suggests hidden structure")

# ==============================================================================
# PART 8: HIERARCHICAL STRUCTURE
# ==============================================================================

print("\n" + "="*80)
print("PART 8: HIERARCHICAL STRUCTURE SUMMARY")
print("="*80)

# Collect all Yukawas
all_yukawas = [
    ('e', y_e), ('μ', y_mu), ('τ', y_tau),
    ('u', y_u), ('c', y_c), ('t', y_t),
    ('d', y_d), ('s', y_s), ('b', y_b)
]
all_yukawas_sorted = sorted(all_yukawas, key=lambda x: x[1])

print("\nAll fermions by Yukawa coupling:")
print(f"{'Fermion':<10} {'Yukawa':<15} {'log10(y)':<15}")
print("-"*40)
for name, y in all_yukawas_sorted:
    print(f"{name:<10} {y:<15.6e} {np.log10(y):<15.3f}")

print(f"\nTotal span: {all_yukawas_sorted[-1][1] / all_yukawas_sorted[0][1]:.2e}")
print(f"  = {np.log10(all_yukawas_sorted[-1][1] / all_yukawas_sorted[0][1]):.1f} orders of magnitude")

# ==============================================================================
# PART 9: KEY OBSERVATIONS
# ==============================================================================

print("\n" + "="*80)
print("PART 9: KEY OBSERVATIONS")
print("="*80)

print("""
1. DIAGONAL STRUCTURE
   - Yukawa matrices are diagonal in mass eigenstate basis
   - No tree-level flavor changing neutral currents (FCNC)
   - Off-diagonal elements arise from mixing (CKM/PMNS)

2. HUGE HIERARCHIES
   - Leptons: ~10^5 range (y_τ/y_e ≈ 3×10^5)
   - Up quarks: ~10^5 range (y_t/y_u ≈ 8×10^4)
   - Down quarks: ~10^3 range (y_b/y_d ≈ 9×10^2)

3. PATTERN WITHIN GENERATIONS
   - Gen 1: y_e < y_d < y_u (all tiny, ~10^-5 to 10^-4)
   - Gen 2: y_μ < y_s < y_c (intermediate, ~10^-3 to 10^-2)
   - Gen 3: y_τ < y_b < y_t (large, ~10^-2 to 1)

4. PATTERN ACROSS GENERATIONS
   - Each generation roughly exponentially heavier than previous
   - But not precisely geometric/exponential
   - Power law exponents differ between sectors

5. KOIDE FORMULA (LEPTONS)
   - K = (Σm_i) / (Σ√m_i)² = 2/3 to 10^-5 precision!
   - No accepted theoretical explanation
   - Suggests hidden mathematical structure

6. MIXING PATTERNS
   - Quarks: Small mixing (CKM nearly diagonal)
   - Leptons: Large mixing (PMNS not hierarchical)
   - This asymmetry is mysterious!

7. NO OBVIOUS SIMPLE PATTERN
   - Not pure geometric sequence
   - Not pure exponential
   - Not simple powers of λ_Cabibbo
   - But clearly not random either

8. GENERATION STRUCTURE
   - Why 3 generations?
   - Why this specific hierarchy?
   - What symmetry is broken?
""")

# ==============================================================================
# PART 10: VISUALIZATIONS
# ==============================================================================

print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Yukawa couplings on log scale
ax1 = plt.subplot(3, 3, 1)
sectors = ['e', 'μ', 'τ', 'u', 'c', 't', 'd', 's', 'b']
yukawas = [y_e, y_mu, y_tau, y_u, y_c, y_t, y_d, y_s, y_b]
colors = ['blue']*3 + ['red']*3 + ['green']*3
ax1.bar(sectors, yukawas, color=colors, alpha=0.7)
ax1.set_yscale('log')
ax1.set_ylabel('Yukawa coupling', fontsize=12)
ax1.set_title('All Yukawa Couplings', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(['Leptons', 'Up quarks', 'Down quarks'], loc='upper left')

# Plot 2: Ratios between generations
ax2 = plt.subplot(3, 3, 2)
ratios_data = {
    'Leptons': [results['leptons']['r21'], results['leptons']['r32']],
    'Up quarks': [results['up_quarks']['r21'], results['up_quarks']['r32']],
    'Down quarks': [results['down_quarks']['r21'], results['down_quarks']['r32']]
}
x = np.arange(2)
width = 0.25
for i, (name, ratios) in enumerate(ratios_data.items()):
    ax2.bar(x + i*width, ratios, width, label=name, alpha=0.7)
ax2.set_ylabel('Ratio', fontsize=12)
ax2.set_title('Generation Ratios', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(['y₂/y₁', 'y₃/y₂'])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_yscale('log')

# Plot 3: Power law test
ax3 = plt.subplot(3, 3, 3)
alphas = [results['leptons']['alpha'], results['up_quarks']['alpha'],
          results['down_quarks']['alpha']]
sectors_names = ['Leptons', 'Up quarks', 'Down quarks']
ax3.bar(sectors_names, alphas, color=['blue', 'red', 'green'], alpha=0.7)
ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='α=1 (geometric)')
ax3.set_ylabel('Exponent α', fontsize=12)
ax3.set_title('Power Law: (y₂/y₁)^α = y₃/y₂', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Yukawa matrices heatmap - Leptons
ax4 = plt.subplot(3, 3, 4)
Y_e_log = np.log10(np.abs(Y_e) + 1e-10)
im1 = ax4.imshow(Y_e_log, cmap='viridis', aspect='auto')
ax4.set_title('Lepton Yukawa (log₁₀)', fontsize=12, fontweight='bold')
ax4.set_xticks([0, 1, 2])
ax4.set_yticks([0, 1, 2])
ax4.set_xticklabels(['e', 'μ', 'τ'])
ax4.set_yticklabels(['e', 'μ', 'τ'])
plt.colorbar(im1, ax=ax4)

# Plot 5: Yukawa matrices heatmap - Up quarks
ax5 = plt.subplot(3, 3, 5)
Y_u_log = np.log10(np.abs(Y_u) + 1e-10)
im2 = ax5.imshow(Y_u_log, cmap='viridis', aspect='auto')
ax5.set_title('Up Quark Yukawa (log₁₀)', fontsize=12, fontweight='bold')
ax5.set_xticks([0, 1, 2])
ax5.set_yticks([0, 1, 2])
ax5.set_xticklabels(['u', 'c', 't'])
ax5.set_yticklabels(['u', 'c', 't'])
plt.colorbar(im2, ax=ax5)

# Plot 6: Yukawa matrices heatmap - Down quarks
ax6 = plt.subplot(3, 3, 6)
Y_d_log = np.log10(np.abs(Y_d) + 1e-10)
im3 = ax6.imshow(Y_d_log, cmap='viridis', aspect='auto')
ax6.set_title('Down Quark Yukawa (log₁₀)', fontsize=12, fontweight='bold')
ax6.set_xticks([0, 1, 2])
ax6.set_yticks([0, 1, 2])
ax6.set_xticklabels(['d', 's', 'b'])
ax6.set_yticklabels(['d', 's', 'b'])
plt.colorbar(im3, ax=ax6)

# Plot 7: All fermions ordered
ax7 = plt.subplot(3, 3, 7)
names_sorted = [x[0] for x in all_yukawas_sorted]
yukawas_sorted = [x[1] for x in all_yukawas_sorted]
colors_sorted = ['blue' if n in ['e','μ','τ'] else 'red' if n in ['u','c','t'] else 'green'
                 for n in names_sorted]
ax7.bar(range(len(names_sorted)), yukawas_sorted, color=colors_sorted, alpha=0.7)
ax7.set_yscale('log')
ax7.set_xticks(range(len(names_sorted)))
ax7.set_xticklabels(names_sorted)
ax7.set_ylabel('Yukawa coupling', fontsize=12)
ax7.set_title('All Fermions (ordered)', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: CKM matrix
ax8 = plt.subplot(3, 3, 8)
im4 = ax8.imshow(np.abs(V_CKM), cmap='Reds', aspect='auto', vmin=0, vmax=1)
ax8.set_title('CKM Matrix (quark mixing)', fontsize=12, fontweight='bold')
ax8.set_xticks([0, 1, 2])
ax8.set_yticks([0, 1, 2])
ax8.set_xticklabels(['d', 's', 'b'])
ax8.set_yticklabels(['u', 'c', 't'])
for i in range(3):
    for j in range(3):
        ax8.text(j, i, f'{V_CKM[i,j]:.3f}', ha='center', va='center', color='black')
plt.colorbar(im4, ax=ax8)

# Plot 9: Koide formula visualization
ax9 = plt.subplot(3, 3, 9)
masses = [m_e, m_mu, m_tau]
sqrt_masses = [np.sqrt(m) for m in masses]
ax9.bar(['e', 'μ', 'τ'], masses, alpha=0.5, label='Masses')
ax9_twin = ax9.twinx()
ax9_twin.plot(['e', 'μ', 'τ'], sqrt_masses, 'ro-', linewidth=2, markersize=10, label='√masses')
ax9.set_ylabel('Mass (MeV)', fontsize=12)
ax9_twin.set_ylabel('√Mass (MeV^{1/2})', fontsize=12)
ax9.set_title(f'Koide Formula: K = {K:.6f} ≈ 2/3', fontsize=12, fontweight='bold')
ax9.legend(loc='upper left')
ax9_twin.legend(loc='upper right')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('yukawa_structure_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: yukawa_structure_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
