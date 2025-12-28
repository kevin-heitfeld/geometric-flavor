"""
BREAKTHROUGH: Non-integer effective exponents!

From generation_structure_test.py:
- Electron needs additional |η|^4.5 suppression
- This means effective exponent is -2k + 4.5 for electron

Let me test if there's a pattern:
Y_i = N × C_i × |η|^(α_i) where α_i are the TRUE exponents
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
print("FINDING TRUE EXPONENTS α_i in Y_i = N × C_i × |η|^(α_i)")
print("="*80)
print()

# For each lepton, find what exponent α gives the observed Yukawa
print("Step 1: Fit α_i for each lepton independently")
print()

# We have: Y_obs = N × C_i × |η|^(α_i)
# So: α_i = log(Y_obs / (N × C_i)) / log(|η|)

# Need to choose N - let's use μ and τ average with α = -2k
N_guess_list = []
for p in ['μ', 'τ']:
    k = LEPTON_K[p]
    delta = k / 6.0
    y_obs = LEPTON_YUKAWAS[p]
    C = conformal_C(delta)
    alpha_nom = -2 * k
    N_guess = y_obs / (C * (eta_abs ** alpha_nom))
    N_guess_list.append(N_guess)

N_best = np.mean(N_guess_list)

print(f"Using N = {N_best:.6e} (from μ and τ with nominal α = -2k)")
print()

print(f"{'Particle':<10} {'k':<6} {'α_nom':<10} {'Δ':<10} {'C':<10} {'α_fit':<10} {'Δα':<10} {'Y_pred':<15} {'Y_obs':<15} {'Error %'}")
print("-"*115)

alphas = {}
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    delta = k / 6.0
    y_obs = LEPTON_YUKAWAS[p]
    C = conformal_C(delta)

    alpha_nominal = -2 * k

    # Fit: find α such that N × C × |η|^α = Y_obs
    # α = log(Y_obs / (N × C)) / log(|η|)
    alpha_fit = np.log(y_obs / (N_best * C)) / np.log(eta_abs)
    alphas[p] = alpha_fit

    delta_alpha = alpha_fit - alpha_nominal

    y_pred = N_best * C * (eta_abs ** alpha_fit)
    error = abs(y_pred - y_obs) / y_obs * 100

    print(f"{p:<10} {k:<6} {alpha_nominal:<10.2f} {delta:<10.3f} {C:<10.4f} {alpha_fit:<10.2f} {delta_alpha:<+10.2f} {y_pred:<15.6e} {y_obs:<15.6e} {error:.2f}%")

print()

# Check pattern in Δα
print("-"*80)
print()
print("Step 2: Pattern in α deviations")
print()

print(f"{'Particle':<10} {'k':<6} {'α_nom = -2k':<15} {'α_fit':<15} {'Δα = α_fit + 2k':<20}")
print("-"*70)

delta_alphas = {}
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    alpha_nom = -2 * k
    alpha_fit = alphas[p]
    delta_alpha = alpha_fit - alpha_nom
    delta_alphas[p] = delta_alpha

    print(f"{p:<10} {k:<6} {alpha_nom:<15.2f} {alpha_fit:<15.2f} {delta_alpha:<+20.2f}")

print()

# Check if Δα follows a pattern
print("  Analysis of Δα:")
print()

# Check ratios
d_e = delta_alphas['e']
d_mu = delta_alphas['μ']
d_tau = delta_alphas['τ']

print(f"  Δα_e = {d_e:+.2f}")
print(f"  Δα_μ = {d_mu:+.2f}")
print(f"  Δα_τ = {d_tau:+.2f}")
print()

if abs(d_mu) < 0.5 and abs(d_tau) < 0.5:
    print("  ✓ μ and τ have Δα ≈ 0 (follow α = -2k rule)")
    print(f"  ✗ Electron has Δα = {d_e:+.2f} (DOES NOT follow rule)")
    print()
    print("  → First generation is special!")
else:
    print("  → All generations deviate from α = -2k")

print()

# Test hierarchies with fitted exponents
print("-"*80)
print()
print("Step 3: Hierarchy predictions with fitted α_i")
print()

for p1, p2 in [('τ', 'μ'), ('μ', 'e'), ('τ', 'e')]:
    alpha1 = alphas[p1]
    alpha2 = alphas[p2]

    ratio_pred = (eta_abs ** alpha1) / (eta_abs ** alpha2)
    ratio_obs = LEPTON_YUKAWAS[p1] / LEPTON_YUKAWAS[p2]
    agreement = ratio_pred / ratio_obs
    error = abs(1 - agreement) * 100

    status = "✓✓✓" if error < 0.1 else "✓✓" if error < 1 else "✓"

    print(f"  Y_{p1}/Y_{p2}:")
    print(f"    Predicted: {ratio_pred:.4f}")
    print(f"    Observed:  {ratio_obs:.4f}")
    print(f"    Agreement: {agreement:.4f} ({error:.4f}% error) {status}")
    print()

# Now try to find formula for α_i
print("="*80)
print("FINDING FORMULA FOR α_i")
print("="*80)
print()

print("From the data:")
print(f"  α_e = {alphas['e']:.2f} (k=4, Δ=0.667)")
print(f"  α_μ = {alphas['μ']:.2f} (k=6, Δ=1.000)")
print(f"  α_τ = {alphas['τ']:.2f} (k=8, Δ=1.333)")
print()

# Test different formulas
print("Testing formulas:")
print()

test_formulas = {
    'α = -2k': lambda k, d: -2*k,
    'α = -2k + 4': lambda k, d: -2*k + 4,
    'α = -2k + 5': lambda k, d: -2*k + 5,
    'α = -3/2 k': lambda k, d: -1.5*k,
    'α = -k - 2': lambda k, d: -k - 2,
    'α = -k - Δ': lambda k, d: -k - d,
    'α = -2Δ - 2': lambda k, d: -2*d - 2,
    'α = -3Δ': lambda k, d: -3*d,
}

print(f"{'Formula':<20} {'α_e pred':<12} {'α_μ pred':<12} {'α_τ pred':<12} {'χ² (×10)':<12}")
print("-"*70)

best_formula = None
best_chi2 = float('inf')

for name, formula in test_formulas.items():
    chi2 = 0
    preds = {}
    for p in ['e', 'μ', 'τ']:
        k = LEPTON_K[p]
        delta = k / 6.0
        alpha_pred = formula(k, delta)
        alpha_obs = alphas[p]
        chi2 += (alpha_pred - alpha_obs)**2
        preds[p] = alpha_pred

    if chi2 < best_chi2:
        best_chi2 = chi2
        best_formula = name

    marker = "✓✓✓" if chi2 < 0.1 else "✓✓" if chi2 < 1 else "✓" if chi2 < 10 else ""

    print(f"{name:<20} {preds['e']:<12.2f} {preds['μ']:<12.2f} {preds['τ']:<12.2f} {chi2*10:<12.4f} {marker}")

print()
print(f"Best formula: {best_formula}")
print()

# Manual tuning
print("-"*80)
print()
print("Manual parameter search:")
print()

best_a = None
best_b = None
best_chi2_manual = float('inf')

for a in np.arange(-3, 0, 0.1):
    for b in np.arange(-10, 10, 0.5):
        chi2 = 0
        for p in ['e', 'μ', 'τ']:
            k = LEPTON_K[p]
            alpha_pred = a * k + b
            alpha_obs = alphas[p]
            chi2 += (alpha_pred - alpha_obs)**2

        if chi2 < best_chi2_manual:
            best_chi2_manual = chi2
            best_a = a
            best_b = b

print(f"Best linear fit: α = {best_a:.3f} × k + {best_b:.3f}")
print(f"χ² = {best_chi2_manual:.6f}")
print()

print("Testing:")
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    alpha_pred = best_a * k + best_b
    alpha_obs = alphas[p]
    error = abs(alpha_pred - alpha_obs)
    print(f"  {p}: α_pred = {alpha_pred:.3f}, α_obs = {alpha_obs:.3f}, error = {error:.3f}")

print()
print("="*80)
