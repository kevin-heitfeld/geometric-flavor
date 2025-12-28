"""
Testing if τ = 2.69i and g_s are related through instanton action
==================================================================

Hypothesis: The instanton action that generates Yukawas has BOTH geometric
and worldsheet contributions:

  S_inst = S_geo + S_ws

Where:
  S_geo = (k/Im(τ)) × d²  (from geometry, distances on CY)
  S_ws = contribution from string coupling g_s

In heterotic string theory, worldsheet instantons (fundamental strings wrapping
cycles) have action:

  S_ws = 2π/g_s × (volume of cycle)

Or in our notation where cycles have "distances" d:
  S_ws = (2π/g_s) × (d²/characteristic_scale)

If both contributions are present:
  S_total = (k/Im(τ)) × d² + (C/g_s) × d²
          = d² × [k/Im(τ) + C/g_s]

Where C is some numerical factor.

For this to work, we need:
  k/Im(τ) + C/g_s = constant (for each fermion family)

This would RELATE τ and g_s!

Let's test if our observed values satisfy this:
"""

import numpy as np

print("="*70)
print("CONNECTING τ = 2.69i WITH g_s FROM GAUGE UNIFICATION")
print("="*70)

# Our values
tau_imag = 2.69
k_value = -86  # From our analysis

# From gauge unification (for different k_GUT levels)
gauge_k_values = [1, 2, 3, 5]
g_s_values = [0.7195, 1.0176, 1.2463, 1.6089]

print(f"\nOur framework:")
print(f"  τ = {tau_imag}i")
print(f"  k = {k_value} (from instanton fits)")
print(f"  Geometric term: k/Im(τ) = {k_value/tau_imag:.2f}")

print(f"\nFrom gauge unification:")
for k_gut, g_s in zip(gauge_k_values, g_s_values):
    print(f"  k_GUT={k_gut}: g_s = {g_s:.4f}  →  2π/g_s = {2*np.pi/g_s:.2f}")

print(f"\n" + "="*70)
print("HYPOTHESIS 1: Additive contributions")
print("="*70)

print(f"\nIf S_total = d² × [k/Im(τ) + α(2π/g_s)], what α gives consistency?")

# For electron: y_e ~ 10^-6 requires S_total ~ 14
# With our d²_e estimate and k/Im(τ):
d_squared_e = 0.44  # Approximate from our fits
S_geo_e = abs(k_value) / tau_imag * d_squared_e

print(f"\nFor electron (y_e ~ 10^-6):")
print(f"  d²_e ~ {d_squared_e:.2f}")
print(f"  S_geo = |k|/Im(τ) × d²_e = {S_geo_e:.2f}")
print(f"  Need S_total ~ 13.8 to get y_e ~ 10^-6")
print(f"  Missing: ΔS = {13.8 - S_geo_e:.2f}")

for k_gut, g_s in zip(gauge_k_values, g_s_values):
    ws_term = 2 * np.pi / g_s
    alpha_needed = (13.8 - S_geo_e) / (ws_term * d_squared_e)

    print(f"\n  k_GUT={k_gut}, g_s={g_s:.3f}:")
    print(f"    2π/g_s = {ws_term:.2f}")
    print(f"    Need α = {alpha_needed:.3f} to match y_e")

    # Check other Yukawas with this α
    d_squared_mu = 0.20  # Approximate
    d_squared_tau = 0.14  # Approximate

    S_mu = S_geo_e * (d_squared_mu/d_squared_e) + alpha_needed * ws_term * d_squared_mu
    S_tau = S_geo_e * (d_squared_tau/d_squared_e) + alpha_needed * ws_term * d_squared_tau

    y_mu_pred = np.exp(-S_mu)
    y_tau_pred = np.exp(-S_tau)

    print(f"    → Predicts y_μ ~ {y_mu_pred:.2e} (need ~2×10^-3) {'✓' if 0.5e-3 < y_mu_pred < 5e-3 else '✗'}")
    print(f"    → Predicts y_τ ~ {y_tau_pred:.2e} (need ~1×10^-2) {'✓' if 0.5e-2 < y_tau_pred < 2e-2 else '✗'}")

print(f"\n" + "="*70)
print("HYPOTHESIS 2: Multiplicative/coupled contributions")
print("="*70)

print(f"\nMaybe the effective Im(τ) is modified by string coupling?")
print(f"  Im(τ)_eff = Im(τ) × f(g_s)")
print(f"\nOr the Kähler modulus T is related to both:")
print(f"  Im(T) = f(Im(τ), g_s)")

# In Type IIB with O3/O7 planes, there's a relation:
# The dilaton S and complex structure U can mix in the effective 4D theory

print(f"\nIn Type IIB string theory:")
print(f"  Dilaton: S = φ + i a = ln(g_s) + i a")
print(f"  Complex structure: U (our τ)")
print(f"  Kähler modulus: T")

for k_gut, g_s in zip(gauge_k_values, g_s_values):
    phi = np.log(g_s)

    print(f"\n  k_GUT={k_gut}:")
    print(f"    g_s = {g_s:.3f}")
    print(f"    Re(S) = φ = {phi:+.3f}")
    print(f"    Our τ = {tau_imag}i")

    # Test if they satisfy some relation
    # E.g., in heterotic E8×E8, there's: S = T = U at special points
    # Or: Im(S) × Im(T) × Im(U) = constant (from anomaly cancellation)

    # Our τ is purely imaginary, so Re(τ) = 0
    # If there's a mirror symmetry: Im(τ) ↔ some function of g_s?

    ratio = tau_imag / g_s
    product = tau_imag * g_s

    print(f"    Im(τ)/g_s = {ratio:.3f}")
    print(f"    Im(τ) × g_s = {product:.3f}")

print(f"\n" + "="*70)
print("HYPOTHESIS 3: Tadpole/Anomaly constraints")
print("="*70)

print(f"""
In Type IIB with D7-branes (for flavor physics), tadpole cancellation requires:

  N_D7 + N_O7 = 8  (for O7-planes)

Where the worldvolume instantons on D7s generate Yukawas.

The instanton action on a D7-brane wrapping a 4-cycle is:
  S_D7 = (2π/g_s) × Vol_4cycle / α'^2

The volume can be written in terms of Kähler moduli:
  Vol_4cycle = function(T, U)

If our τ corresponds to U (complex structure), and T is related to g_s
through gauge coupling unification, then:

  S_inst = (2π/g_s) × f(τ)

This would directly connect τ and g_s!

For our case:
  τ = 2.69i  (complex structure)
  g_s ~ 0.7-1.6  (from gauge unification)

The volume factor f(τ) in our E6 geometry would need to be calculated
from the specific CY3 metric.

But qualitatively: Im(τ) = 2.69 is the "radius" in complex structure moduli
space, and this affects the 4-cycle volumes that D7-branes wrap.

If the Yukawa suppressions match observations ONLY for a specific g_s value,
that would prove τ and g_s are linked!
""")

print(f"\n" + "="*70)
print("TESTING WHICH g_s GIVES BEST YUKAWA MATCH")
print("="*70)

# Observed Yukawas (rough values at M_GUT)
y_e_obs = 2.8e-6
y_mu_obs = 5.9e-4
y_tau_obs = 1.0e-2

y_u_obs = 1.3e-5
y_c_obs = 7.3e-3
y_t_obs = 0.99

# Our geometric distances squared (from previous fits)
# Charged leptons
d2_e = 0.44
d2_mu = 0.20
d2_tau = 0.14

# Up quarks (assuming similar structure)
d2_u = 0.42
d2_c = 0.22
d2_t = 0.12

k_abs = abs(k_value)
tau_im = tau_imag

print(f"\nTrying combined action: S = (k/Im(τ)) × d² + α × (2π/g_s) × d²")
print(f"Finding best α for each k_GUT:")

for k_gut, g_s in zip(gauge_k_values, g_s_values):
    ws_factor = 2 * np.pi / g_s

    # Find α that minimizes error across all 6 fermions
    alphas = np.linspace(0, 2, 100)
    best_alpha = 0
    best_error = 1e10

    for alpha in alphas:
        # Predict Yukawas
        S_e = (k_abs/tau_im + alpha * ws_factor) * d2_e
        S_mu = (k_abs/tau_im + alpha * ws_factor) * d2_mu
        S_tau = (k_abs/tau_im + alpha * ws_factor) * d2_tau

        S_u = (k_abs/tau_im + alpha * ws_factor) * d2_u
        S_c = (k_abs/tau_im + alpha * ws_factor) * d2_c
        S_t = (k_abs/tau_im + alpha * ws_factor) * d2_t

        y_e_pred = np.exp(-S_e)
        y_mu_pred = np.exp(-S_mu)
        y_tau_pred = np.exp(-S_tau)
        y_u_pred = np.exp(-S_u)
        y_c_pred = np.exp(-S_c)
        y_t_pred = np.exp(-S_t)

        # Log error (since Yukawas span many orders of magnitude)
        error = (
            (np.log10(y_e_pred) - np.log10(y_e_obs))**2 +
            (np.log10(y_mu_pred) - np.log10(y_mu_obs))**2 +
            (np.log10(y_tau_pred) - np.log10(y_tau_obs))**2 +
            (np.log10(y_u_pred) - np.log10(y_u_obs))**2 +
            (np.log10(y_c_pred) - np.log10(y_c_obs))**2 +
            (np.log10(y_t_pred) - np.log10(y_t_obs))**2
        )

        if error < best_error:
            best_error = error
            best_alpha = alpha

    # Compute predictions with best α
    S_e = (k_abs/tau_im + best_alpha * ws_factor) * d2_e
    S_mu = (k_abs/tau_im + best_alpha * ws_factor) * d2_mu
    S_tau = (k_abs/tau_im + best_alpha * ws_factor) * d2_tau

    y_e_pred = np.exp(-S_e)
    y_mu_pred = np.exp(-S_mu)
    y_tau_pred = np.exp(-S_tau)

    print(f"\n  k_GUT={k_gut}, g_s={g_s:.3f}, α_best={best_alpha:.3f}:")
    print(f"    Error metric: {best_error:.3f}")
    print(f"    y_e: {y_e_pred:.2e} vs {y_e_obs:.2e}  (ratio {y_e_pred/y_e_obs:.2f})")
    print(f"    y_μ: {y_mu_pred:.2e} vs {y_mu_obs:.2e}  (ratio {y_mu_pred/y_mu_obs:.2f})")
    print(f"    y_τ: {y_tau_pred:.2e} vs {y_tau_obs:.2e}  (ratio {y_tau_pred/y_tau_obs:.2f})")

print(f"\n" + "="*70)
print("CONCLUSION")
print("="*70)

print(f"""
If one value of k_GUT (and thus g_s) gives significantly better Yukawa
predictions than others, that would suggest:

  τ = 2.69i and g_s are LINKED through the instanton physics!

The link would be: both contribute to the total instanton action,
and the combination that matches ALL observed Yukawas uniquely
determines both moduli.

This would be the "consistency overdetermination" approach working again:
  • 6 Yukawas + 3 gauge couplings = 9 observables
  • 2 moduli (τ, φ) + geometric distances = fit
  • Overdetermined system → unique solution!
""")

print("="*70)
