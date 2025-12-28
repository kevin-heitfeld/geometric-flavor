"""
Deep Diagnostic: Why don't absolute values match?

The τ/μ ratio is 99.5% accurate, which means:
    Y_τ/Y_μ = |η|^(-2(k_τ - k_μ)) = |η|^(-4) ✓

But individually:
    Y_τ predicted = 0.0006, observed = 0.0072 (off by 12x)
    Y_μ predicted = 0.00003, observed = 0.0004 (off by 12x)

Same factor! This means we're missing a CONSTANT factor.

Let's find what that constant should be.
"""

import numpy as np
from scipy.special import gamma as gamma_func

V_EW = 246.0
PI = np.pi
TAU = 2.69j

LEPTON_YUKAWAS = {'e': 0.000511 / V_EW, 'μ': 0.105658 / V_EW, 'τ': 1.77686 / V_EW}
LEPTON_K = {'τ': 8, 'μ': 6, 'e': 4}

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
print("DEEP DIAGNOSTIC: FINDING THE MISSING CONSTANT")
print("="*80)
print()

# The ratios work, so the formula Y ~ |η|^(-2k) is correct
# But we need the right normalization

print("Step 1: Check if all three leptons have the SAME missing factor")
print()

for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    y_obs = LEPTON_YUKAWAS[p]

    # Predicted (just the eta part)
    eta_factor = eta_abs ** (-2 * k)

    # What constant N would make it match?
    N_needed = y_obs / eta_factor

    print(f"  {p}: k={k}, Y_obs={y_obs:.6e}")
    print(f"     |η|^(-2k) = {eta_factor:.6e}")
    print(f"     N needed = {N_needed:.6e}")
    print()

print("-"*80)
print()

# They should all give the same N if formula is right
print("Step 2: Check consistency - do all three give same N?")
print()

N_values = []
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    y_obs = LEPTON_YUKAWAS[p]
    eta_factor = eta_abs ** (-2 * k)
    N = y_obs / eta_factor
    N_values.append(N)
    print(f"  N from {p}: {N:.6e}")

N_avg = np.mean(N_values)
N_std = np.std(N_values)
N_variation = N_std / N_avg

print()
print(f"  Average N: {N_avg:.6e}")
print(f"  Std dev: {N_std:.6e}")
print(f"  Variation: {N_variation*100:.1f}%")
print()

if N_variation < 0.1:
    print("  ✓✓✓ All three consistent! Formula is CORRECT!")
elif N_variation < 0.5:
    print("  ✓✓ Reasonably consistent (factor <2 variation)")
else:
    print("  ✗ Large variation - formula needs refinement")

print()
print("-"*80)
print()

# Let's also check if CFT structure constants help
print("Step 3: Include CFT structure constants C_iiH")
print()

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

LEPTON_DELTA = {p: k/6.0 for p, k in LEPTON_K.items()}

print("  Testing formula: Y_i = N × C_iiH × |η|^(-2k)")
print()

N_with_C = []
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    delta = LEPTON_DELTA[p]
    y_obs = LEPTON_YUKAWAS[p]

    C = conformal_C(delta)
    eta_factor = eta_abs ** (-2 * k)

    N = y_obs / (C * eta_factor)
    N_with_C.append(N)

    print(f"  {p}: C_iiH={C:.4f}, N needed = {N:.6e}")

N_avg_C = np.mean(N_with_C)
N_std_C = np.std(N_with_C)
N_variation_C = N_std_C / N_avg_C

print()
print(f"  Average N: {N_avg_C:.6e}")
print(f"  Variation: {N_variation_C*100:.1f}%")
print()

if N_variation_C < N_variation:
    print(f"  ✓ Including C_iiH IMPROVES consistency!")
    print(f"    Variation: {N_variation*100:.1f}% → {N_variation_C*100:.1f}%")
else:
    print(f"  → C_iiH doesn't help (variation unchanged)")

print()
print("-"*80)
print()

# Now calculate with best formula
print("Step 4: Final calculation with optimal formula")
print()

# Use average N including C factors
N_best = N_avg_C

print(f"Using: Y_i = {N_best:.6e} × C_iiH × |η|^(-2k)")
print()
print(f"{'Particle':<10} {'k':<6} {'C_iiH':<10} {'Y_pred':<15} {'Y_obs':<15} {'Ratio':<10} {'Error %'}")
print("-"*85)

for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    delta = LEPTON_DELTA[p]
    y_obs = LEPTON_YUKAWAS[p]

    C = conformal_C(delta)
    eta_factor = eta_abs ** (-2 * k)

    y_pred = N_best * C * eta_factor
    ratio = y_pred / y_obs
    error = abs(1 - ratio) * 100

    print(f"{p:<10} {k:<6} {C:<10.4f} {y_pred:<15.6e} {y_obs:<15.6e} {ratio:<10.4f} {error:.2f}%")

print()
print("="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print()

if N_variation_C < 0.1:
    print("✓✓✓ FORMULA VALIDATED: Y_i = N × C_iiH × |η(τ)|^(-2k_i)")
    print()
    print(f"    Normalization constant: N = {N_best:.6e}")
    print()
    print("    This constant includes:")
    print("      • String coupling g_s")
    print("      • Geometric factors from compactification")
    print("      • Instanton/non-perturbative effects")
    print()
    print("    ALL THREE LEPTONS should now match within ~10%!")
else:
    print("→ Formula needs additional refinement")

print()
