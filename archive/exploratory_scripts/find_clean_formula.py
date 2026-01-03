"""
The pattern from test_without_structure_constants.py:
  δ_1 = +3.055
  δ_2 = -0.516
  δ_3 = -0.523

So β_i = -2k_i + δ_gen where:
  δ_1 ≈ +3.0
  δ_23 ≈ -0.5

This looks like it could be:
  β_i = -2k_i + f(gen)

Let me find the cleanest formula.
"""

import numpy as np

V_EW = 246.0
PI = np.pi
TAU = 2.69j

LEPTON_YUKAWAS = {'e': 0.000511 / V_EW, 'μ': 0.105658 / V_EW, 'τ': 1.77686 / V_EW}
LEPTON_K = {'τ': 8, 'μ': 6, 'e': 4}
GENERATION = {'e': 1, 'μ': 2, 'τ': 3}

def dedekind_eta(tau):
    q = np.exp(2j * PI * tau)
    eta_asymp = np.exp(PI * 1j * tau / 12)
    correction = 1.0
    for n in range(1, 20):
        correction *= (1 - q**n)
    return eta_asymp * correction

eta = dedekind_eta(TAU)
eta_abs = abs(eta)

print("="*80)
print("FINDING CLEAN FORMULA FOR GENERATION CORRECTIONS")
print("="*80)
print()

# Fitted values from previous script
delta_fitted = {1: 3.055, 2: -0.516, 3: -0.523}

print("Fitted δ_gen values:")
for gen in [1, 2, 3]:
    print(f"  δ_{gen} = {delta_fitted[gen]:+.3f}")
print()

# Test simple formulas
print("-"*80)
print()
print("Testing formulas for δ_gen:")
print()

test_formulas = {
    'δ = 0 for all': lambda g: 0,
    'δ = 4 - g': lambda g: 4 - g,
    'δ = 3 - g': lambda g: 3 - g,
    'δ = 5 - 2g': lambda g: 5 - 2*g,
    'δ = 4 - 1.5g': lambda g: 4 - 1.5*g,
    'δ = 3.5 - 1.5g': lambda g: 3.5 - 1.5*g,
    'δ = (4-g)/2': lambda g: (4-g)/2,
    'δ = (5-g)/sqrt(g)': lambda g: (5-g)/np.sqrt(g),
}

print(f"{'Formula':<25} {'δ_1':<10} {'δ_2':<10} {'δ_3':<10} {'χ²':<12}")
print("-"*67)

best_formula = None
best_chi2 = float('inf')

for name, formula in test_formulas.items():
    chi2 = sum((formula(g) - delta_fitted[g])**2 for g in [1, 2, 3])

    if chi2 < best_chi2:
        best_chi2 = chi2
        best_formula = name

    marker = "✓✓✓" if chi2 < 0.01 else "✓✓" if chi2 < 0.1 else "✓" if chi2 < 1 else ""

    d1, d2, d3 = formula(1), formula(2), formula(3)
    print(f"{name:<25} {d1:<+10.3f} {d2:<+10.3f} {d3:<+10.3f} {chi2:<12.6f} {marker}")

print()

# Manual search for best linear fit
print("-"*80)
print()
print("Manual search: δ_gen = a - b×gen")
print()

best_a, best_b = None, None
best_chi2_manual = float('inf')

for a in np.arange(0, 6, 0.1):
    for b in np.arange(0, 3, 0.1):
        chi2 = sum((a - b*g - delta_fitted[g])**2 for g in [1, 2, 3])
        if chi2 < best_chi2_manual:
            best_chi2_manual = chi2
            best_a = a
            best_b = b

print(f"Best fit: δ_gen = {best_a:.2f} - {best_b:.2f}×gen")
print(f"χ² = {best_chi2_manual:.6f}")
print()

for g in [1, 2, 3]:
    d_pred = best_a - best_b * g
    d_fit = delta_fitted[g]
    print(f"  Generation {g}: δ_pred = {d_pred:+.3f}, δ_fit = {d_fit:+.3f}, error = {abs(d_pred - d_fit):.3f}")

print()

# Test if correction is related to k
print("-"*80)
print()
print("Alternative: δ as function of k")
print()

k_values = {1: 4, 2: 6, 3: 8}

print("Testing: δ_gen = a + b×k_gen")
print()

best_ak, best_bk = None, None
best_chi2_k = float('inf')

for a in np.arange(-10, 10, 0.1):
    for b in np.arange(-2, 2, 0.05):
        chi2 = sum((a + b*k_values[g] - delta_fitted[g])**2 for g in [1, 2, 3])
        if chi2 < best_chi2_k:
            best_chi2_k = chi2
            best_ak = a
            best_bk = b

print(f"Best fit: δ = {best_ak:.2f} + {best_bk:.2f}×k")
print(f"χ² = {best_chi2_k:.6f}")
print()

for g in [1, 2, 3]:
    k = k_values[g]
    d_pred = best_ak + best_bk * k
    d_fit = delta_fitted[g]
    print(f"  Generation {g} (k={k}): δ_pred = {d_pred:+.3f}, δ_fit = {d_fit:+.3f}, error = {abs(d_pred - d_fit):.3f}")

print()

# Compare the two approaches
print("="*80)
print("COMPARISON")
print("="*80)
print()

print(f"Option 1: δ_gen = {best_a:.2f} - {best_b:.2f}×gen  (χ² = {best_chi2_manual:.6f})")
print(f"Option 2: δ_gen = {best_ak:.2f} + {best_bk:.2f}×k  (χ² = {best_chi2_k:.6f})")
print()

if best_chi2_manual < best_chi2_k:
    print("→ Generation number formula is better")
    print()
    print(f"FINAL: β_i = -2k_i + {best_a:.2f} - {best_b:.2f}×gen_i")
    final_formula = lambda k, g: -2*k + best_a - best_b*g
else:
    print("→ k-weight formula is better")
    print()
    print(f"FINAL: β_i = -2k_i + {best_ak:.2f} + {best_bk:.2f}×k_i")
    final_formula = lambda k, g: -2*k + best_ak + best_bk*k

print()

# Simplify if possible
print("-"*80)
print()
print("Simplification check:")
print()

if best_chi2_k < best_chi2_manual and abs(best_bk + 2) < 0.01:
    print(f"β = -2k + {best_ak:.2f} + {best_bk:.2f}×k")
    print(f"β = {best_bk - 2:.2f}×k + {best_ak:.2f}")
    print()
    effective_coeff = best_bk - 2
    print(f"→ Can write as: β_i = {effective_coeff:.2f}×k_i + {best_ak:.2f}")

print()

# Final test with clean formula
print("="*80)
print("FINAL FORMULA VALIDATION")
print("="*80)
print()

# Use best formula
N = 6.383499e-08

print(f"Y_i = N × |η(τ)|^(β_i)")
print()

if best_chi2_manual < best_chi2_k:
    print(f"β_i = -2k_i + {best_a:.2f} - {best_b:.2f}×gen_i")
    beta_func = lambda k, g: -2*k + best_a - best_b*g
else:
    print(f"β_i = (-2 + {best_bk:.3f})×k_i + {best_ak:.2f}")
    print(f"β_i ≈ {best_bk-2:.3f}×k_i + {best_ak:.2f}")
    beta_func = lambda k, g: (best_bk - 2)*k + best_ak

print()
print(f"N = {N:.6e}")
print(f"|η(τ)| = {eta_abs:.6f}")
print()

print(f"{'Particle':<10} {'Gen':<5} {'k':<6} {'β':<10} {'Y_pred':<15} {'Y_obs':<15} {'Error %'}")
print("-"*80)

chi2_final = 0
for p in ['e', 'μ', 'τ']:
    gen = GENERATION[p]
    k = LEPTON_K[p]
    beta = beta_func(k, gen)

    y_pred = N * (eta_abs ** beta)
    y_obs = LEPTON_YUKAWAS[p]
    error = abs(y_pred - y_obs) / y_obs * 100

    chi2_final += ((y_pred - y_obs) / y_obs) ** 2

    print(f"{p:<10} {gen:<5} {k:<6} {beta:<10.3f} {y_pred:<15.6e} {y_obs:<15.6e} {error:.2f}%")

print()
print(f"χ²/dof = {chi2_final/3:.6f}")
print()

print("="*80)
