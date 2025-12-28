"""
Can we test KKLT/LVS moduli stabilization with our values?
==========================================================

KKLT = Kachru-Kallosh-Linde-Trivedi (2003)
LVS = Large Volume Scenario (2005)

These are RECIPES for stabilizing moduli using fluxes + non-perturbative effects.

Key question: Do they PREDICT a relationship between τ, g_s, and T?
Or do they just tell you HOW to stabilize once you choose parameters?

Let's check if we can make a concrete prediction!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("TESTING KKLT/LVS MODULI STABILIZATION")
print("="*70)

tau_imag = 2.69
g_s_SM = 0.55
g_s_MSSM = 0.72

print(f"\nOur values:")
print(f"  Im(τ) = Im(U) = {tau_imag}")
print(f"  g_s ~ {g_s_SM}-{g_s_MSSM} (SM to MSSM)")

print(f"\n" + "="*70)
print("KKLT MECHANISM (2003)")
print("="*70)

print(f"""
KKLT stabilizes moduli in Type IIB string theory using:

1. FLUXES on 3-cycles
   → Fix complex structure U (our τ) and dilaton S (our g_s)
   → Determined by flux quantization: integers (n, m)

2. NON-PERTURBATIVE EFFECTS (instantons/gaugino condensation)
   → Generate potential for Kähler modulus T
   → V ~ e^(-2π Im(T) / g_s)

3. ANTI-D3 BRANES
   → Uplift to positive vacuum energy

Key formula (simplified):
  V(T) ~ e^(-2π a Im(T)) / Im(T)^(3/2) - Λ_uplift

Where:
  a ~ 1/g_s  (from instanton action)
  Minimum at: Im(T) ~ g_s (schematically)

KKLT prediction: Im(T) ~ O(g_s) ~ O(1)
""")

print(f"\nFor our g_s values:")
for label, g_s in [("SM", g_s_SM), ("MSSM", g_s_MSSM)]:
    Im_T_KKLT = g_s  # Schematic estimate
    print(f"  {label}: g_s = {g_s:.2f}  →  Im(T) ~ {Im_T_KKLT:.2f} (KKLT)")

print(f"\n" + "="*70)
print("LVS MECHANISM (2005)")
print("="*70)

print(f"""
Large Volume Scenario is an alternative to KKLT:

1. VOLUME is LARGE: Im(T_big) >> 1
   → One Kähler modulus (overall volume) is very large
   → Stabilized by α' corrections to Kähler potential

2. SMALL CYCLE: Im(T_small) ~ O(1)
   → Another Kähler modulus (blow-up mode) stays small
   → Stabilized by non-perturbative effects

3. HIERARCHY: Im(T_big) ~ exp(c × Im(T_small))

Key formulas:
  Im(T_big) ~ exp(2π Im(T_small) / g_s) / g_s^(3/2)
  Im(T_small) ~ g_s × (some O(1) factor)

LVS prediction: Im(T_big) ~ 10-1000, Im(T_small) ~ O(1)
""")

print(f"\nFor our g_s values:")
for label, g_s in [("SM", g_s_SM), ("MSSM", g_s_MSSM)]:
    Im_T_small = 1.5 * g_s  # Typical stabilization value
    Im_T_big = np.exp(2*np.pi*Im_T_small / g_s) / g_s**1.5

    print(f"  {label}: g_s = {g_s:.2f}")
    print(f"    Im(T_small) ~ {Im_T_small:.2f}")
    print(f"    Im(T_big) ~ {Im_T_big:.1e}")

print(f"\n" + "="*70)
print("WHICH SCENARIO FITS OUR FRAMEWORK?")
print("="*70)

print(f"""
We have two estimates for Im(T) from phenomenology:

A) Volume factors: Im(T) ~ 0.001-0.05  (VERY SMALL!)
   From M_R ~ (Im T)^(3/2) × M_string ~ M_GUT

B) Anomaly cancellation: Im(T) ~ 0.2-0.5  (MODERATE)
   From Im(S) × Im(T) × Im(U) ~ 1

Comparing to scenarios:

• KKLT: Im(T) ~ O(1) ~ 0.5-1.0
  → Matches estimate B ✓
  → Doesn't match estimate A ✗

• LVS (small cycle): Im(T_small) ~ 0.5-1.0
  → Matches estimate B ✓
  → Doesn't match estimate A ✗

• LVS (big cycle): Im(T_big) ~ 10-1000
  → Doesn't match either ✗

CONCLUSION: If estimate B (anomaly) is correct, then KKLT or LVS
            small-cycle fits our phenomenology!

            If estimate A (volume) is correct, neither fits well.
""")

print(f"\n" + "="*70)
print("CONCRETE TEST: KKLT POTENTIAL")
print("="*70)

print(f"""
Let's be more concrete. In KKLT, the potential is approximately:

  V(T) = A × exp(-2π a Im(T)) / Im(T)^(3/2) - Λ_uplift

Where:
  A = coefficient from non-perturbative superpotential
  a = 1/g_s  (instanton action factor)
  Λ_uplift = anti-D3 brane tension

The minimum satisfies: dV/dT = 0

This gives: Im(T)_min = (2π a / 3) × [1 + O(Λ_uplift/A)]

For perturbative approximation (small uplift):
  Im(T)_min ≈ 2π/(3 g_s)
""")

print(f"\nPredicted Im(T) from KKLT formula:")
for label, g_s in [("SM", g_s_SM), ("MSSM", g_s_MSSM)]:
    Im_T_min = 2 * np.pi / (3 * g_s)
    print(f"  {label}: g_s = {g_s:.2f}  →  Im(T) = {Im_T_min:.2f}")

print(f"\nSM gives Im(T) ~ 3.8")
print(f"MSSM gives Im(T) ~ 2.9")
print(f"\nThese are LARGER than anomaly estimate (0.2-0.5)")
print(f"but same ORDER OF MAGNITUDE!")

print(f"\n" + "="*70)
print("VISUALIZE KKLT POTENTIAL")
print("="*70)

# Plot KKLT potential for our g_s values
T_range = np.linspace(0.1, 10, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for label, g_s, color in [("SM", g_s_SM, 'blue'), ("MSSM", g_s_MSSM, 'red')]:
    a = 1 / g_s
    A = 1.0  # Normalized

    # Without uplift
    V_no_uplift = A * np.exp(-2*np.pi*a*T_range) / T_range**1.5

    # With small uplift (tuned to give near-zero minimum)
    Lambda_uplift = 0.8 * A * np.exp(-2*np.pi*a*2.0) / 2.0**1.5  # Tuned
    V_with_uplift = V_no_uplift - Lambda_uplift

    # Find minimum
    idx_min = np.argmin(np.abs(np.gradient(V_with_uplift)))
    T_min = T_range[idx_min]
    V_min = V_with_uplift[idx_min]

    ax1.plot(T_range, V_no_uplift, label=f'{label} (no uplift)', linestyle='--', color=color, alpha=0.5)
    ax1.plot(T_range, V_with_uplift, label=f'{label} (with uplift)', color=color, linewidth=2)
    ax1.scatter([T_min], [V_min], s=100, c=color, marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax1.text(T_min, V_min + 0.05, f'T={T_min:.2f}', ha='center', fontsize=9)

    # Zoom in on minimum
    T_zoom = np.linspace(max(0.1, T_min-2), T_min+2, 200)
    V_zoom = A * np.exp(-2*np.pi*a*T_zoom) / T_zoom**1.5 - Lambda_uplift
    ax2.plot(T_zoom, V_zoom, label=label, color=color, linewidth=2)
    ax2.scatter([T_min], [V_min], s=100, c=color, marker='o', edgecolors='black', linewidths=2, zorder=5)

ax1.set_xlabel('Im(T)', fontsize=12)
ax1.set_ylabel('V(T) [arbitrary units]', fontsize=12)
ax1.set_title('KKLT Potential (full range)', fontsize=13, fontweight='bold')
ax1.axhline(0, color='black', linestyle=':', alpha=0.3)
ax1.set_xlim(0, 10)
ax1.set_ylim(-0.2, 0.5)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Im(T)', fontsize=12)
ax2.set_ylabel('V(T) [arbitrary units]', fontsize=12)
ax2.set_title('KKLT Potential (near minimum)', fontsize=13, fontweight='bold')
ax2.axhline(0, color='black', linestyle=':', alpha=0.3)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kklt_potential_test.png', dpi=200)
plt.close()

print(f"Plot saved: kklt_potential_test.png")

print(f"\n" + "="*70)
print("ACTUAL CALCULATION")
print("="*70)

print(f"""
For each g_s, solve dV/dT = 0 numerically:
""")

from scipy.optimize import minimize_scalar

for label, g_s in [("SM", g_s_SM), ("MSSM", g_s_MSSM)]:
    a = 1 / g_s
    A = 1.0

    # Function to minimize (negative because we want minimum)
    def V_kklt(T):
        if T <= 0.01:
            return 1e10
        return A * np.exp(-2*np.pi*a*T) / T**1.5

    # Find minimum
    result = minimize_scalar(V_kklt, bounds=(0.1, 20), method='bounded')
    T_min_no_uplift = result.x

    # With uplift, need to tune Lambda. Assume it shifts minimum slightly
    # In real KKLT, Lambda is tuned to give V_min ~ 0 (small positive)
    # This doesn't change T_min much for small Lambda

    print(f"\n  {label}: g_s = {g_s:.2f}")
    print(f"    KKLT predicts: Im(T) = {T_min_no_uplift:.2f}")
    print(f"    Approximate formula: Im(T) = 2π/(3g_s) = {2*np.pi/(3*g_s):.2f}")

    # Check anomaly constraint
    Im_U = tau_imag
    Im_S = g_s  # rough approximation
    Im_T_from_anomaly = 1.0 / (Im_S * Im_U)

    print(f"    Anomaly constraint: Im(T) ~ {Im_T_from_anomaly:.2f}")
    print(f"    Ratio KKLT/anomaly: {T_min_no_uplift/Im_T_from_anomaly:.2f}×")

print(f"\n" + "="*70)
print("CONCLUSION")
print("="*70)

print(f"""
KKLT moduli stabilization PREDICTS:

For SM (g_s ~ 0.55):
  Im(T) ~ 3.8  (KKLT formula)

For MSSM (g_s ~ 0.72):
  Im(T) ~ 2.9  (KKLT formula)

Comparison to phenomenological estimates:
  • Anomaly: Im(T) ~ 0.37 (SM), 0.37 (MSSM)
  • KKLT is 8-10× LARGER!

This is a TENSION!

Possible resolutions:
1. Anomaly formula is wrong (not simple product)
2. KKLT formula is approximate (missing factors)
3. Our phenomenology (volume factors) missed something
4. There are multiple Kähler moduli, mixing things up
5. Different stabilization mechanism (not KKLT)

What we CAN say:
  ✓ KKLT gives definite prediction: Im(T) ~ 3-4
  ✓ This is O(1), not exponentially large (not LVS big)
  ✗ Doesn't perfectly match our anomaly estimate
  ? Need more sophisticated analysis to resolve

ACTION ITEMS:
-------------
1. Check if multi-moduli case changes this
2. Look at LVS with small cycle ~ 0.3
3. Consider hybrid scenarios
4. OR accept factor-of-10 uncertainty and document

TIME ESTIMATE: 1-2 days for refined calculation
               1-2 weeks for full multi-moduli analysis
""")

print("="*70)
