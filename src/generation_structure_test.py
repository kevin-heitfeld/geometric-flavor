"""
Hypothesis: Different suppression mechanism for 1st generation

The diagnostic showed:
- μ and τ: N_best = 9.2e-08 (consistent!)
- electron: needs N = 7.4e-09 (factor 12.4 smaller)

This factor 12.4 looks suspicious. Let me test if electron has
a DIFFERENT functional form.

What if: Y_μ,τ ~ |η|^(-2k) but Y_e ~ |η|^(-2k) × (extra factor)?

Or: What if electron uses different modular function?
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
print("HYPOTHESIS: ELECTRON HAS DIFFERENT SUPPRESSION MECHANISM")
print("="*80)
print()

# Use μ and τ to determine N (standard formula)
print("Step 1: Standard formula for μ and τ")
print()

N_23 = []
for p in ['μ', 'τ']:
    k = LEPTON_K[p]
    delta = k / 6.0
    y_obs = LEPTON_YUKAWAS[p]
    C = conformal_C(delta)
    eta_factor = eta_abs ** (-2 * k)
    N = y_obs / (C * eta_factor)
    N_23.append(N)
    print(f"  {p}: k={k}, Y = N × C × |η|^(-2k) → N = {N:.6e}")

N_23_avg = np.mean(N_23)
print(f"\n  Average N_23 = {N_23_avg:.6e}")
print()

# Now check electron with same formula
print("-"*80)
print()
print("Step 2: Does electron follow same formula?")
print()

k_e = LEPTON_K['e']
delta_e = k_e / 6.0
y_e_obs = LEPTON_YUKAWAS['e']
C_e = conformal_C(delta_e)
eta_factor_e = eta_abs ** (-2 * k_e)

y_e_pred_standard = N_23_avg * C_e * eta_factor_e

print(f"  Using N_23 = {N_23_avg:.6e}:")
print(f"  Y_e predicted: {y_e_pred_standard:.6e}")
print(f"  Y_e observed:  {y_e_obs:.6e}")
print(f"  Ratio: {y_e_pred_standard / y_e_obs:.4f}")
print()

factor_mismatch = y_e_pred_standard / y_e_obs
print(f"  ⚠ Electron is off by factor {factor_mismatch:.2f}")
print()

# Test hypothesis: electron has additional suppression
print("-"*80)
print()
print("Step 3: Find additional suppression factor for electron")
print()

print("  Hypothesis: Y_e = N_23 × C_e × |η|^(-2k_e) × f_suppress")
print()
print(f"  Required suppression: f_suppress = 1/{factor_mismatch:.4f} = {1/factor_mismatch:.4f}")
print()

# Check if this is related to some power of |η|
eta_powers = {}
for alpha in np.arange(-10, 10, 0.5):
    test_factor = eta_abs ** alpha
    if abs(test_factor - 1/factor_mismatch) < 0.01:
        eta_powers[alpha] = test_factor

if eta_powers:
    print("  ✓ Could be power of |η|:")
    for alpha, val in eta_powers.items():
        print(f"    |η|^{alpha:.1f} = {val:.4f}")
else:
    print("  ✗ Not a simple power of |η|")

print()

# Check other possibilities
print("  Other possibilities:")
possible_factors = {
    '1/12': 1/12,
    '1/10': 0.1,
    '1/16': 1/16,
    'e^(-1)': np.exp(-1),
    'e^(-2)': np.exp(-2),
    '|η|': eta_abs,
    '|η|^2': eta_abs**2,
    '|η|^3': eta_abs**3,
    '|η|^4': eta_abs**4,
}

for name, val in possible_factors.items():
    if abs(val - 1/factor_mismatch) < 0.02:
        print(f"    ✓ {name} = {val:.4f} (target: {1/factor_mismatch:.4f})")

print()

# Test: What if 1st generation has DIFFERENT N?
print("-"*80)
print()
print("Step 4: Alternative - separate N for each generation")
print()

# Determine N for each particle independently
print("  Individual normalization constants:")
print()
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    delta = k / 6.0
    y_obs = LEPTON_YUKAWAS[p]
    C = conformal_C(delta)
    eta_factor = eta_abs ** (-2 * k)
    N_indiv = y_obs / (C * eta_factor)

    print(f"  {p}: N_{p} = {N_indiv:.6e}")

print()
print("  Observation:")
print("  • N_μ and N_τ are nearly equal (2nd/3rd generation)")
print("  • N_e is different (1st generation)")
print()
print("  → This suggests GENERATION structure, not particle-by-particle")
print()

# Check ratio
N_e = LEPTON_YUKAWAS['e'] / (C_e * eta_factor_e)
ratio_N = N_23_avg / N_e
print(f"  N_23/N_1 = {ratio_N:.4f}")
print()

# Is this ratio meaningful?
print("  Checking if N_23/N_1 has physical meaning:")
print()

test_ratios = {
    'Γ₀(3)/Γ₀(1)': 3.0,
    'Γ₀(4)/Γ₀(1)': 4.0,
    '2×Γ₀(3)': 6.0,
    '3×Γ₀(4)': 12.0,
    'sqrt(3×4)': np.sqrt(12),
    '(3×4)': 12.0,
    '(3+4)×2': 14.0,
    '2^4': 16.0,
    'e^(π/2)': np.exp(PI/2),
}

for name, val in test_ratios.items():
    error = abs(val - ratio_N) / ratio_N * 100
    if error < 10:
        print(f"  ✓ {name} = {val:.4f} (error: {error:.2f}%)")

print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()

print("The electron DOES NOT follow the same formula as μ and τ.")
print()
print("Two possibilities:")
print()
print("1. GENERATION-DEPENDENT NORMALIZATION:")
print(f"   N_1 (e) = {N_e:.6e}")
print(f"   N_23 (μ,τ) = {N_23_avg:.6e}")
print(f"   Ratio: N_23/N_1 ≈ {ratio_N:.1f}")
print()
print("2. ADDITIONAL SUPPRESSION for 1st generation:")
print(f"   Y_e = N_23 × C_e × |η|^(-2k_e) × ({1/factor_mismatch:.4f})")
print()
print("Need to understand:")
print("  • Why do 2nd and 3rd generations share N but 1st doesn't?")
print("  • Is the factor ~12 related to modular level product 3×4 = 12?")
print("  • Are there threshold corrections specific to lightest generation?")
print()
