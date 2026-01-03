"""
Test: What if electron has effective k ≠ 4?

The diagnostic showed μ and τ are consistent, but electron is off by 12x.
This suggests k_e might be effectively different.

Let's find what k_e would make all three consistent.
"""

import numpy as np
from scipy.special import gamma as gamma_func

V_EW = 246.0
PI = np.pi
TAU = 2.69j

LEPTON_YUKAWAS = {'e': 0.000511 / V_EW, 'μ': 0.105658 / V_EW, 'τ': 1.77686 / V_EW}
LEPTON_K = {'τ': 8, 'μ': 6, 'e': 4}  # Nominal values

def dedekind_eta(tau):
    q = np.exp(2j * PI * tau)
    eta_asymp = np.exp(PI * 1j * tau / 12)
    correction = 1.0
    for n in range(1, 20):
        correction *= (1 - q**n)
    return eta_asymp * correction

def conformal_C(delta_i, delta_H=1.0):
    delta_sum = 2 * delta_i
    delta_diff = delta_sum - delta_H
    if delta_diff <= 0:
        return 1.0
    try:
        C = gamma_func(delta_diff) / (gamma_func(delta_i)**2 * gamma_func(delta_H))**0.5
        return abs(C)
    except:
        return 1.0

eta = dedekind_eta(TAU)
eta_abs = abs(eta)

print("="*80)
print("FINDING EFFECTIVE k_e")
print("="*80)
print()

# Use μ and τ to determine N (they're consistent)
print("Step 1: Determine N from μ and τ (which are consistent)")
print()

N_from_mu_tau = []
for p in ['μ', 'τ']:
    k = LEPTON_K[p]
    delta = k / 6.0
    y_obs = LEPTON_YUKAWAS[p]
    C = conformal_C(delta)
    eta_factor = eta_abs ** (-2 * k)
    N = y_obs / (C * eta_factor)
    N_from_mu_tau.append(N)
    print(f"  {p}: N = {N:.6e}")

N_best = np.mean(N_from_mu_tau)
print(f"\n  Average N = {N_best:.6e}")
print()

# Now find what k_e would fit
print("-"*80)
print()
print("Step 2: What k_e makes electron consistent with this N?")
print()

y_e = LEPTON_YUKAWAS['e']

print(f"  Target: Y_e = {y_e:.6e}")
print(f"  We have: Y_e = N × C_ee × |η|^(-2k_e)")
print(f"  So: |η|^(-2k_e) = Y_e / (N × C_ee)")
print()

# Try different k_e values
print("  Testing different k_e values:")
print()
print(f"  {'k_e':<8} {'Δ_e':<10} {'C_ee':<10} {'Y_pred':<15} {'Y_obs':<15} {'Ratio':<10} {'Error %'}")
print("  " + "-"*80)

best_k = None
best_error = float('inf')

for k_test in np.arange(3.0, 6.0, 0.1):
    delta_e = k_test / 6.0
    C_ee = conformal_C(delta_e)
    eta_factor = eta_abs ** (-2 * k_test)
    y_pred = N_best * C_ee * eta_factor
    ratio = y_pred / y_e
    error = abs(1 - ratio) * 100

    if error < best_error:
        best_error = error
        best_k = k_test

    if k_test in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0] or error < 5:
        print(f"  {k_test:<8.1f} {delta_e:<10.3f} {C_ee:<10.4f} {y_pred:<15.6e} {y_e:<15.6e} {ratio:<10.4f} {error:.2f}%")

print()
print(f"  ✓ Best fit: k_e = {best_k:.2f} (error = {best_error:.2f}%)")
print()

# Calculate what this means
k_nominal = 4.0
k_shift = best_k - k_nominal

print("-"*80)
print()
print("Step 3: Physical interpretation")
print()
print(f"  Nominal k_e = {k_nominal}")
print(f"  Effective k_e = {best_k:.2f}")
print(f"  Shift: Δk = {k_shift:+.2f}")
print()

if abs(k_shift) < 0.5:
    print("  → Small correction (~10% shift)")
    print("  → Could be from threshold corrections or RG running")
elif abs(k_shift) > 1.0:
    print("  → Large correction - suggests electron is special")
    print("  → Possible physics:")
    print("    • Boundary effects (lightest fermion)")
    print("    • Different modular representation")
    print("    • Selection rules")

print()
print("-"*80)
print()
print("Step 4: Test full consistency with corrected k_e")
print()

LEPTON_K_CORRECTED = {'τ': 8, 'μ': 6, 'e': best_k}

print(f"Using k-weights: τ={LEPTON_K_CORRECTED['τ']}, μ={LEPTON_K_CORRECTED['μ']}, e={LEPTON_K_CORRECTED['e']:.2f}")
print()
print(f"{'Particle':<10} {'k':<8} {'Δ':<10} {'C_iiH':<10} {'Y_pred':<15} {'Y_obs':<15} {'Ratio':<10} {'Error %'}")
print("-"*95)

chi2 = 0
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K_CORRECTED[p]
    delta = k / 6.0
    y_obs = LEPTON_YUKAWAS[p]
    C = conformal_C(delta)
    eta_factor = eta_abs ** (-2 * k)
    y_pred = N_best * C * eta_factor
    ratio = y_pred / y_obs
    error = abs(1 - ratio) * 100

    chi2 += ((y_pred - y_obs) / y_obs) ** 2

    print(f"{p:<10} {k:<8.2f} {delta:<10.3f} {C:<10.4f} {y_pred:<15.6e} {y_obs:<15.6e} {ratio:<10.4f} {error:.2f}%")

print()
print(f"χ²/dof = {chi2/3:.4f}")
print()

# Test hierarchies with corrected k_e
print("-"*80)
print()
print("Step 5: Hierarchy predictions with corrected k_e")
print()

for p1, p2 in [('τ', 'μ'), ('μ', 'e'), ('τ', 'e')]:
    k1 = LEPTON_K_CORRECTED[p1]
    k2 = LEPTON_K_CORRECTED[p2]

    ratio_pred = eta_abs ** (-2 * (k1 - k2))
    ratio_obs = LEPTON_YUKAWAS[p1] / LEPTON_YUKAWAS[p2]
    agreement = ratio_pred / ratio_obs
    error = abs(1 - agreement) * 100

    status = "✓✓✓" if error < 1 else "✓✓" if error < 5 else "✓"

    print(f"  Y_{p1}/Y_{p2}:")
    print(f"    Predicted: {ratio_pred:.4f}")
    print(f"    Observed:  {ratio_obs:.4f}")
    print(f"    Agreement: {agreement:.4f} ({error:.2f}% error) {status}")
    print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()

if best_error < 5 and chi2/3 < 0.1:
    print("✓✓✓ PERFECT FIT with k_e correction!")
    print()
    print(f"Final k-weights: τ=8, μ=6, e={best_k:.2f}")
    print()
    print("Physical interpretation:")
    print(f"  • Electron has effective k_e = {best_k:.2f} instead of 4.0")
    print(f"  • This {best_k-4:.2f}-unit shift could come from:")
    print("    - Threshold corrections")
    print("    - Mixing with other modular representations")
    print("    - Special boundary conditions (lightest fermion)")
    print()
    print("ALL THREE LEPTONS NOW MATCH!")
else:
    print("→ Additional refinement needed beyond k_e correction")

print()
