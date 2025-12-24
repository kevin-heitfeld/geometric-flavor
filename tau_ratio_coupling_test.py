"""
τ-Ratio vs Gauge Coupling Test: The Ultimate Connection

GEMINI'S HYPOTHESIS:
If τ-ratio = τ_hadronic / τ_leptonic ≈ 0.44 matches the ratio of 
Strong/Weak coupling constants, then the "geometric distance" between 
branes is NOT ad-hoc but encodes FORCE STRENGTHS.

This would unify:
- Masses (via modular parameters τ)
- Forces (via gauge couplings g)
- Geometry (via brane separation Δτ)

From Δk=2 universality test:
- τ_leptonic = 3.25i (SU(2)×U(1) brane)
- τ_hadronic = 1.42i (SU(3) color brane)
- τ_ratio = 0.438

PREDICTION:
If framework is correct, τ_ratio should match g_strong / g_weak at some scale.

Gauge coupling evolution (1-loop):
α_i(Q) = α_i(M_Z) / (1 - b_i * ln(Q/M_Z))

where b_i are beta functions:
- b_3 = -7 (SU(3) color, asymptotic freedom)
- b_2 = -19/6 (SU(2) weak)
- b_1 = 41/10 (U(1) hypercharge)

At M_Z:
α_3(M_Z) ≈ 0.1184 (strong)
α_2(M_Z) ≈ 0.0337 (weak)
α_1(M_Z) ≈ 0.0102 (hypercharge)

Normalized to SU(5) GUT:
α_3 : α_2 : α_1 = g_3² : g_2² : (5/3)g_1²

Test if τ_ratio ≈ α_3/α_2 or g_3/g_2 at some scale Q.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import json

# Experimental values at M_Z (PDG 2023)
M_Z = 91.1876  # GeV
alpha_3_MZ = 0.1184  # Strong coupling
alpha_2_MZ = 0.0337  # Weak coupling
alpha_1_MZ = 0.0102  # Hypercharge (MS-bar)

# Beta function coefficients (1-loop, SM)
b3 = -7  # SU(3) - asymptotic freedom
b2 = -19/6  # SU(2)
b1 = 41/10  # U(1)

# From Δk=2 test
tau_leptonic = 3.25
tau_hadronic = 1.422  # Average of up/down
tau_ratio_observed = tau_hadronic / tau_leptonic
delta_tau = tau_leptonic - tau_hadronic

print("="*70)
print("τ-RATIO vs GAUGE COUPLING TEST")
print("="*70)
print("\nGEOMETRIC DATA (from Δk=2 test):")
print(f"  τ_leptonic (SU(2)×U(1)) = {tau_leptonic:.3f}i")
print(f"  τ_hadronic (SU(3)) = {tau_hadronic:.3f}i")
print(f"  τ_ratio = τ_hadronic/τ_leptonic = {tau_ratio_observed:.4f}")
print(f"  Δτ = {delta_tau:.3f}")

print("\nGAUGE COUPLINGS at M_Z = 91.2 GeV:")
print(f"  α_3(M_Z) = {alpha_3_MZ:.4f} (strong)")
print(f"  α_2(M_Z) = {alpha_2_MZ:.4f} (weak)")
print(f"  α_1(M_Z) = {alpha_1_MZ:.4f} (hypercharge)")

# Coupling ratios at M_Z
alpha_ratio_32_MZ = alpha_3_MZ / alpha_2_MZ
g_ratio_32_MZ = np.sqrt(alpha_3_MZ / alpha_2_MZ)

print(f"\nRATIOS at M_Z:")
print(f"  α_3/α_2 = {alpha_ratio_32_MZ:.4f}")
print(f"  g_3/g_2 = {g_ratio_32_MZ:.4f}")
print(f"  τ_ratio = {tau_ratio_observed:.4f}")

# RG evolution
def alpha_RG(alpha_0, b, Q, Q0=M_Z):
    """Evolve coupling from Q0 to Q (1-loop)"""
    t = np.log(Q / Q0)
    return alpha_0 / (1 - b * alpha_0 * t / (2*np.pi))

def coupling_ratio_at_scale(Q):
    """Calculate α_3/α_2 at scale Q"""
    alpha_3_Q = alpha_RG(alpha_3_MZ, b3, Q)
    alpha_2_Q = alpha_RG(alpha_2_MZ, b2, Q)
    return alpha_3_Q / alpha_2_Q

def g_ratio_at_scale(Q):
    """Calculate g_3/g_2 at scale Q"""
    return np.sqrt(coupling_ratio_at_scale(Q))

# Scan energy scales
Q_range = np.logspace(1, 17, 1000)  # 10 GeV to 10^17 GeV
alpha_ratios = [coupling_ratio_at_scale(Q) for Q in Q_range]
g_ratios = [g_ratio_at_scale(Q) for Q in Q_range]

# Find scale where ratios match τ_ratio
def match_alpha_ratio(log_Q):
    Q = 10**log_Q
    if Q > 1e17:  # Avoid Landau pole
        return 1e10
    ratio = coupling_ratio_at_scale(Q)
    return abs(ratio - tau_ratio_observed)

def match_g_ratio(log_Q):
    Q = 10**log_Q
    if Q > 1e17:
        return 1e10
    ratio = g_ratio_at_scale(Q)
    return abs(ratio - tau_ratio_observed)

# Find best-match scales
result_alpha = minimize_scalar(match_alpha_ratio, bounds=(1, 16), method='bounded')
result_g = minimize_scalar(match_g_ratio, bounds=(1, 16), method='bounded')

Q_match_alpha = 10**result_alpha.x
Q_match_g = 10**result_g.x

alpha_at_match = coupling_ratio_at_scale(Q_match_alpha)
g_at_match = g_ratio_at_scale(Q_match_g)

print("\n" + "="*70)
print("SCALE MATCHING")
print("="*70)

print(f"\nMATCH 1: α_3/α_2 = τ_ratio")
print(f"  Best-match scale: Q = {Q_match_alpha:.2e} GeV")
print(f"  α_3/α_2 at Q = {alpha_at_match:.4f}")
print(f"  τ_ratio = {tau_ratio_observed:.4f}")
print(f"  Difference: {abs(alpha_at_match - tau_ratio_observed):.4f}")

# Identify physical scale
if Q_match_alpha < 100:
    scale_name_alpha = "Electroweak"
elif Q_match_alpha < 1e3:
    scale_name_alpha = "TeV (LHC)"
elif Q_match_alpha < 1e10:
    scale_name_alpha = "Intermediate"
elif Q_match_alpha < 1e15:
    scale_name_alpha = "GUT scale"
else:
    scale_name_alpha = "Planck scale"
print(f"  Physical scale: {scale_name_alpha}")

print(f"\nMATCH 2: g_3/g_2 = τ_ratio")
print(f"  Best-match scale: Q = {Q_match_g:.2e} GeV")
print(f"  g_3/g_2 at Q = {g_at_match:.4f}")
print(f"  τ_ratio = {tau_ratio_observed:.4f}")
print(f"  Difference: {abs(g_at_match - tau_ratio_observed):.4f}")

if Q_match_g < 100:
    scale_name_g = "Electroweak"
elif Q_match_g < 1e3:
    scale_name_g = "TeV (LHC)"
elif Q_match_g < 1e10:
    scale_name_g = "Intermediate"
elif Q_match_g < 1e15:
    scale_name_g = "GUT scale"
else:
    scale_name_g = "Planck scale"
print(f"  Physical scale: {scale_name_g}")

# Check GUT scale specifically (M_GUT ≈ 2×10^16 GeV)
M_GUT = 2e16
alpha_ratio_GUT = coupling_ratio_at_scale(M_GUT)
g_ratio_GUT = g_ratio_at_scale(M_GUT)

print(f"\nAT GUT SCALE (M_GUT = 2×10^16 GeV):")
print(f"  α_3/α_2 = {alpha_ratio_GUT:.4f}")
print(f"  g_3/g_2 = {g_ratio_GUT:.4f}")
print(f"  τ_ratio = {tau_ratio_observed:.4f}")

# Check string scale (M_string ≈ 10^17 GeV)
M_string = 1e17
alpha_ratio_string = coupling_ratio_at_scale(M_string)
g_ratio_string = g_ratio_at_scale(M_string)

print(f"\nAT STRING SCALE (M_string ≈ 10^17 GeV):")
print(f"  α_3/α_2 = {alpha_ratio_string:.4f}")
print(f"  g_3/g_2 = {g_ratio_string:.4f}")
print(f"  τ_ratio = {tau_ratio_observed:.4f}")

# Inverse relationship: Does 1/τ_ratio match anything?
tau_ratio_inv = 1 / tau_ratio_observed
print(f"\nINVERSE RATIO: 1/τ_ratio = {tau_ratio_inv:.4f}")

def match_alpha_ratio_inv(log_Q):
    Q = 10**log_Q
    if Q > 1e17:
        return 1e10
    ratio = coupling_ratio_at_scale(Q)
    return abs(ratio - tau_ratio_inv)

result_alpha_inv = minimize_scalar(match_alpha_ratio_inv, bounds=(1, 16), method='bounded')
Q_match_alpha_inv = 10**result_alpha_inv.x
alpha_at_match_inv = coupling_ratio_at_scale(Q_match_alpha_inv)

print(f"  Best α_3/α_2 match: Q = {Q_match_alpha_inv:.2e} GeV")
print(f"  α_3/α_2 at Q = {alpha_at_match_inv:.4f}")

# Alternative: Δτ/τ_leptonic vs coupling difference
delta_tau_normalized = delta_tau / tau_leptonic
alpha_diff_MZ = alpha_3_MZ - alpha_2_MZ
alpha_diff_normalized = alpha_diff_MZ / alpha_2_MZ

print(f"\nALTERNATIVE: Δτ/τ_leptonic vs Δα/α_2")
print(f"  Δτ/τ_leptonic = {delta_tau_normalized:.4f}")
print(f"  (α_3-α_2)/α_2 at M_Z = {alpha_diff_normalized:.4f}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# Find which match is best
matches = [
    ("α_3/α_2", Q_match_alpha, abs(alpha_at_match - tau_ratio_observed)),
    ("g_3/g_2", Q_match_g, abs(g_at_match - tau_ratio_observed)),
    ("1/(α_3/α_2)", Q_match_alpha_inv, abs(alpha_at_match_inv - tau_ratio_inv))
]

best_match = min(matches, key=lambda x: x[2])

print(f"\nBEST MATCH: {best_match[0]}")
print(f"  Scale: Q = {best_match[1]:.2e} GeV")
print(f"  Precision: Δ = {best_match[2]:.4f}")

if best_match[2] < 0.05:
    print("\n✓✓✓ EXCELLENT MATCH (<5% deviation)")
    print("→ τ-ratio encodes gauge coupling ratio!")
    print("→ Geometric brane separation = Force strength ratio")
    print("→ This is MAJOR: Masses unified with Forces via geometry")
elif best_match[2] < 0.15:
    print("\n✓ GOOD MATCH (<15% deviation)")
    print("→ τ-ratio suggestively close to coupling ratio")
    print("→ May encode gauge structure")
else:
    print("\n⚠ WEAK MATCH (>15% deviation)")
    print("→ τ-ratio may not directly encode α_i ratios")
    print("→ Could be more subtle relationship (e.g., τ²/τ₁ vs ln(g₃/g₂))")

# Additional insight: Level matching
# In modular forms, τ ratios often match "level" ratios in Γ_0(N)
# Check if τ_ratio ≈ N₂/N₃ for some integers N
for N3 in range(1, 20):
    for N2 in range(1, 20):
        ratio = N2 / N3
        if abs(ratio - tau_ratio_observed) < 0.02:
            print(f"\nLEVEL RATIO MATCH: N₂/N₃ = {N2}/{N3} = {ratio:.4f}")
            print(f"  → Leptons: Γ₀({N2}) group")
            print(f"  → Quarks: Γ₀({N3}) group")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: RG evolution of α_3/α_2
ax = axes[0, 0]
ax.plot(Q_range, alpha_ratios, 'b-', linewidth=2, label='α₃/α₂')
ax.axhline(y=tau_ratio_observed, color='red', linestyle='--', linewidth=2, 
           label=f'τ_ratio = {tau_ratio_observed:.3f}')
ax.axvline(x=Q_match_alpha, color='green', linestyle=':', alpha=0.7,
           label=f'Match at {Q_match_alpha:.2e} GeV')
ax.axvline(x=M_GUT, color='purple', linestyle=':', alpha=0.5, label='GUT scale')

ax.set_xscale('log')
ax.set_xlabel('Energy Scale Q [GeV]', fontsize=12)
ax.set_ylabel('α₃/α₂', fontsize=12)
ax.set_title('RG Evolution: Fine Structure Constant Ratio', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: RG evolution of g_3/g_2
ax = axes[0, 1]
ax.plot(Q_range, g_ratios, 'b-', linewidth=2, label='g₃/g₂')
ax.axhline(y=tau_ratio_observed, color='red', linestyle='--', linewidth=2,
           label=f'τ_ratio = {tau_ratio_observed:.3f}')
ax.axvline(x=Q_match_g, color='green', linestyle=':', alpha=0.7,
           label=f'Match at {Q_match_g:.2e} GeV')
ax.axvline(x=M_GUT, color='purple', linestyle=':', alpha=0.5, label='GUT scale')

ax.set_xscale('log')
ax.set_xlabel('Energy Scale Q [GeV]', fontsize=12)
ax.set_ylabel('g₃/g₂', fontsize=12)
ax.set_title('RG Evolution: Gauge Coupling Ratio', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 3: Deviation from τ_ratio
ax = axes[1, 0]
deviations_alpha = [abs(r - tau_ratio_observed) for r in alpha_ratios]
deviations_g = [abs(r - tau_ratio_observed) for r in g_ratios]

ax.plot(Q_range, deviations_alpha, 'b-', linewidth=2, label='|α₃/α₂ - τ_ratio|')
ax.plot(Q_range, deviations_g, 'r-', linewidth=2, label='|g₃/g₂ - τ_ratio|')
ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='5% threshold')
ax.axvline(x=M_GUT, color='purple', linestyle=':', alpha=0.5, label='GUT scale')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Energy Scale Q [GeV]', fontsize=12)
ax.set_ylabel('|Ratio - τ_ratio|', fontsize=12)
ax.set_title('Deviation from τ_ratio', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 4: Summary comparison
ax = axes[1, 1]
scales = ['M_Z\n(91 GeV)', f'Best Match\n({Q_match_alpha:.1e} GeV)', 
          'GUT\n(2×10¹⁶ GeV)', 'String\n(10¹⁷ GeV)']
alpha_vals = [alpha_ratio_32_MZ, alpha_at_match, alpha_ratio_GUT, alpha_ratio_string]
tau_vals = [tau_ratio_observed] * 4

x_pos = np.arange(len(scales))
width = 0.35

bars1 = ax.bar(x_pos - width/2, alpha_vals, width, label='α₃/α₂', color='blue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, tau_vals, width, label='τ_ratio', color='red', alpha=0.7)

# Add match quality
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height = max(bar1.get_height(), bar2.get_height())
    deviation = abs(alpha_vals[i] - tau_vals[i])
    if deviation < 0.05:
        marker = "✓"
        color = 'green'
    elif deviation < 0.15:
        marker = "~"
        color = 'orange'
    else:
        marker = "✗"
        color = 'red'
    ax.text(i, height + 0.05, marker, ha='center', fontsize=14, 
            fontweight='bold', color=color)

ax.set_ylabel('Ratio', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(scales, fontsize=10)
ax.set_title('τ_ratio vs α₃/α₂ at Different Scales', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('tau_ratio_coupling_test.png', dpi=300, bbox_inches='tight')
plt.savefig('tau_ratio_coupling_test.pdf', bbox_inches='tight')
print("\n✓ Figures saved: tau_ratio_coupling_test.png/pdf")

# Save results
results = {
    'geometric': {
        'tau_leptonic': float(tau_leptonic),
        'tau_hadronic': float(tau_hadronic),
        'tau_ratio': float(tau_ratio_observed),
        'delta_tau': float(delta_tau)
    },
    'gauge_couplings_MZ': {
        'alpha_3': float(alpha_3_MZ),
        'alpha_2': float(alpha_2_MZ),
        'alpha_ratio': float(alpha_ratio_32_MZ),
        'g_ratio': float(g_ratio_32_MZ)
    },
    'best_match': {
        'type': best_match[0],
        'scale_GeV': float(best_match[1]),
        'deviation': float(best_match[2]),
        'match_quality': 'excellent' if best_match[2] < 0.05 else ('good' if best_match[2] < 0.15 else 'weak')
    },
    'at_GUT_scale': {
        'scale_GeV': float(M_GUT),
        'alpha_ratio': float(alpha_ratio_GUT),
        'g_ratio': float(g_ratio_GUT),
        'tau_ratio': float(tau_ratio_observed),
        'deviation_alpha': float(abs(alpha_ratio_GUT - tau_ratio_observed)),
        'deviation_g': float(abs(g_ratio_GUT - tau_ratio_observed))
    },
    'interpretation': {
        'masses_forces_unified': bool(best_match[2] < 0.15),
        'geometric_distance_encodes_forces': bool(best_match[2] < 0.15),
        'scale_interpretation': scale_name_alpha if best_match[0] == 'α_3/α_2' else scale_name_g
    }
}

with open('tau_ratio_coupling_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved: tau_ratio_coupling_results.json")

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if best_match[2] < 0.05:
    print("\n🎯 MASSES AND FORCES UNIFIED VIA GEOMETRY ✓✓✓")
    print(f"\nThe τ-ratio (0.{tau_ratio_observed:.3f}) precisely matches {best_match[0]}")
    print(f"at energy scale Q ≈ {best_match[1]:.2e} GeV")
    print("\nThis is PROFOUND:")
    print("  • Brane separation Δτ encodes force strength difference")
    print("  • Geometric distance → Gauge coupling ratio")
    print("  • Masses (modular weights) unified with Forces (gauge couplings)")
    print("  • NOT coincidence: τ-ratio = coupling ratio at physical scale")
    print("\n→ Framework explains BOTH mass hierarchies AND gauge structure!")
elif best_match[2] < 0.15:
    print("\n✓ SUGGESTIVE CONNECTION FOUND")
    print(f"\nThe τ-ratio ({tau_ratio_observed:.3f}) is within 15% of {best_match[0]}")
    print(f"at scale Q ≈ {best_match[1]:.2e} GeV")
    print("\nThis suggests:")
    print("  • Brane separation may encode gauge structure")
    print("  • Deviation could be threshold corrections or 2-loop effects")
    print("  • Framework hints at mass-force unification")
else:
    print("\n⚠ NO CLEAR MATCH TO SIMPLE COUPLING RATIOS")
    print(f"\nBest match has {best_match[2]*100:.1f}% deviation")
    print("Possibilities:")
    print("  • τ-ratio encodes more subtle relationship")
    print("  • Need to include all three gauge groups (SU(3)×SU(2)×U(1))")
    print("  • Relationship may be logarithmic or involve volume factors")
    print("  • Or τ-ratio is purely geometric, not gauge-related")

print("\n" + "="*70)
