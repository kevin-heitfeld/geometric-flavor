"""
Test multiplicative connection: What if k_eff = k_bare × f(g_s)?
================================================================

If the effective coupling strength in instantons depends on BOTH
geometry (τ) and string coupling (g_s), maybe:

  S_inst = k_eff/Im(τ) × d²

Where:  k_eff = k_bare × g_s^n  (for some power n)

Or:  k_eff = k_bare × (1 + β × g_s)

This would mean the k = -86 we extracted from data is actually:
  k_observed = k_bare × f(g_s_true)

If we can determine k_bare from theory, we can solve for g_s!
"""

import numpy as np

print("="*70)
print("MULTIPLICATIVE CONNECTION: k_eff = k_bare × f(g_s)")
print("="*70)

# Our fitted value
k_obs = -86
tau_imag = 2.69

# From gauge unification
gauge_k_values = [1, 2, 3, 5]
g_s_values = [0.7195, 1.0176, 1.2463, 1.6089]

print(f"\nWe measured: k_eff = {k_obs} from Yukawa fits")
print(f"Question: Is this k_eff = k_bare × f(g_s)?")

print(f"\n" + "="*70)
print("SCENARIO 1: k_eff = k_bare × g_s")
print("="*70)

print(f"\nIf the string coupling multiplicatively modifies k:")
for k_gut, g_s in zip(gauge_k_values, g_s_values):
    k_bare = k_obs / g_s
    print(f"  k_GUT={k_gut}, g_s={g_s:.3f}  →  k_bare = {k_bare:.1f}")

print(f"\nFor consistency, k_bare should be the SAME for all g_s values.")
print(f"But we get different k_bare for each g_s → rules out linear relation.")

print(f"\n" + "="*70)
print("SCENARIO 2: k_eff = k_bare / g_s  (inverse)")
print("="*70)

print(f"\nMaybe strong coupling SUPPRESSES the instanton effect:")
for k_gut, g_s in zip(gauge_k_values, g_s_values):
    k_bare = k_obs * g_s
    print(f"  k_GUT={k_gut}, g_s={g_s:.3f}  →  k_bare = {k_bare:.1f}")

print(f"\nAgain, not consistent unless one specific g_s is correct.")

print(f"\n" + "="*70)
print("SCENARIO 3: Scan for power law k_eff = k_bare × g_s^n")
print("="*70)

print(f"\nFor each power n, compute k_bare for each g_s:")
print(f"Then check consistency (k_bare should be constant):")

for n in [-2, -1, -0.5, 0.5, 1, 2]:
    print(f"\n  Power n = {n:+.1f}:")
    k_bare_values = []
    for k_gut, g_s in zip(gauge_k_values, g_s_values):
        k_bare = k_obs / (g_s ** n)
        k_bare_values.append(k_bare)
        print(f"    k_GUT={k_gut}, g_s={g_s:.3f}  →  k_bare = {k_bare:.1f}")

    # Check consistency
    k_bare_mean = np.mean(k_bare_values)
    k_bare_std = np.std(k_bare_values)
    consistency = k_bare_std / abs(k_bare_mean) * 100 if k_bare_mean != 0 else 999
    print(f"    Variation: {consistency:.1f}%  {'✓ CONSISTENT!' if consistency < 10 else ''}")

print(f"\n" + "="*70)
print("SCENARIO 4: Look at Im(τ) × g_s product")
print("="*70)

print(f"\nIf there's a mirror/duality symmetry, maybe:")
print(f"  Im(τ) × g_s = constant")
print(f"Or: Im(τ) / g_s = constant")

for k_gut, g_s in zip(gauge_k_values, g_s_values):
    product = tau_imag * g_s
    ratio = tau_imag / g_s
    print(f"\n  k_GUT={k_gut}, g_s={g_s:.3f}:")
    print(f"    Im(τ) × g_s = {product:.3f}")
    print(f"    Im(τ) / g_s = {ratio:.3f}")

print(f"\nNo obvious constant relationship.")

print(f"\n" + "="*70)
print("SCENARIO 5: Both are independent, but correlation from anomaly")
print("="*70)

print(f"""
In heterotic string theory with E8×E8, the anomaly cancellation requires:

  Tr(R²) - Tr(F²) = 0

Where R is curvature and F is gauge field strength.

This leads to a constraint on the dilaton and moduli. In 4D effective theory:

  Im(S) × Im(T) × Im(U) ~ O(1)  (schematic)

Where:
  S = dilaton (our g_s)
  T = Kähler modulus
  U = complex structure (our τ)

If we identify U ~ τ = 2.69i, then:
  Im(S) × Im(T) ~ 1/2.69 ~ 0.37

For different g_s values:
""")

for k_gut, g_s in zip(gauge_k_values, g_s_values):
    # Im(S) for dilaton is not just g_s, it's related to the 4D gauge coupling
    # But schematically, let's use g_s as proxy
    Im_T_needed = 0.37 / g_s
    print(f"  k_GUT={k_gut}, g_s={g_s:.3f}  →  Im(T) ~ {Im_T_needed:.3f}")

print(f"\nThis would constrain the Kähler modulus T based on g_s!")

print(f"\n" + "="*70)
print("KEY INSIGHT")
print("="*70)

print(f"""
Our k = -86 was fitted to data assuming ONLY geometric instantons.

But if there are BOTH geometric and worldsheet contributions, the
true relationship might be:

  Observable Yukawas = f(τ, g_s, k_bare, geometry)

The fact that our fits work so well with just (τ, k_eff) suggests:

  1. Either worldsheet instantons don't contribute to Yukawas, OR
  2. The g_s dependence is already "baked into" our effective k

To distinguish:
  • If (1): Then τ and g_s are INDEPENDENT moduli
  • If (2): Then we need k_bare(theory) to extract g_s from k_obs

For (2), the Kac-Moody level from gauge theory would give k_bare.
In heterotic E8×E8:
  k_bare = level of E8 gauge group = 1 (for standard embedding)

Then:  k_obs = -86 = k_bare × f(g_s)
       → f(g_s) = -86

This seems too large unless there's a winding number or
flux quantization involved.

ALTERNATIVE: Our |k| = 86 might be:
  k_eff = (2π/g_s) × n_wrapping
  86 = (2π/g_s) × n

For k_GUT=2, g_s=1.02:
  n = 86 × 1.02 / (2π) = 13.9 ≈ 14 wrappings

This would be a TOPOLOGICAL connection: the number of times
the worldsheet wraps the cycle depends on both geometric
(τ determines which cycle) and coupling (g_s gives the cost).
""")

print("="*70)
