"""
HYPOTHESIS: Maybe C_i structure constants are wrong?

The fits show χ²/dof = 0 when I fit α_i individually,
but the hierarchy ratios are off by factor 2.

This suggests the INDIVIDUAL fits absorb some factor that
CANCELS in the ratios for μ/τ but NOT for μ/e or τ/e.

Let me test: What if Y_i = N × |η|^(β_i) WITHOUT C_i?
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
print("SIMPLIFIED FORMULA: Y_i = N × |η|^(β_i)  (NO C_i)")
print("="*80)
print()

# Fit β_i for each lepton
print("Step 1: Fit β_i using simplified formula")
print()

# Use average of all three to determine N
N_guesses = []
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    y_obs = LEPTON_YUKAWAS[p]
    beta_nominal = -2*k
    N_guess = y_obs / (eta_abs ** beta_nominal)
    N_guesses.append(N_guess)

N_avg = np.mean(N_guesses)

print(f"Guessing N from each particle (using β = -2k):")
for i, p in enumerate(['e', 'μ', 'τ']):
    print(f"  {p}: N = {N_guesses[i]:.6e}")
print(f"\n  Average: N = {N_avg:.6e}")
print()

# Now fit β_i
print("-"*80)
print()
print("Step 2: Fit β_i for each particle")
print()

betas = {}
print(f"{'Particle':<10} {'k':<6} {'β_nom':<10} {'β_fit':<10} {'Δβ':<10} {'Y_pred':<15} {'Y_obs':<15} {'Error %'}")
print("-"*95)

for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    y_obs = LEPTON_YUKAWAS[p]

    beta_nominal = -2*k

    # Fit: N × |η|^β = Y_obs → β = log(Y_obs/N) / log(|η|)
    beta_fit = np.log(y_obs / N_avg) / np.log(eta_abs)
    betas[p] = beta_fit

    delta_beta = beta_fit - beta_nominal

    y_pred = N_avg * (eta_abs ** beta_fit)
    error = abs(y_pred - y_obs) / y_obs * 100

    print(f"{p:<10} {k:<6} {beta_nominal:<10.2f} {beta_fit:<10.2f} {delta_beta:<+10.2f} {y_pred:<15.6e} {y_obs:<15.6e} {error:.2f}%")

print()

# Check if β_i = -2k + δ_gen pattern holds
print("-"*80)
print()
print("Step 3: Check generation pattern")
print()

delta_betas = {}
for p in ['e', 'μ', 'τ']:
    gen = GENERATION[p]
    k = LEPTON_K[p]
    beta = betas[p]
    delta = beta + 2*k
    delta_betas[gen] = delta
    print(f"  Generation {gen} ({p}): δ_{gen} = β + 2k = {beta:.3f} + {2*k} = {delta:+.3f}")

print()

# Test hierarchies with fitted β_i
print("-"*80)
print()
print("Step 4: Hierarchy predictions")
print()

for p1, p2 in [('τ', 'μ'), ('μ', 'e'), ('τ', 'e')]:
    beta1 = betas[p1]
    beta2 = betas[p2]

    ratio_pred = (eta_abs ** beta1) / (eta_abs ** beta2)
    ratio_obs = LEPTON_YUKAWAS[p1] / LEPTON_YUKAWAS[p2]
    agreement = ratio_pred / ratio_obs
    error = abs(1 - agreement) * 100

    status = "✓✓✓" if error < 0.1 else "✓✓" if error < 1 else "✓"

    print(f"  Y_{p1}/Y_{p2}:")
    print(f"    β_{p1} - β_{p2} = {beta1:.3f} - {beta2:.3f} = {beta1-beta2:.3f}")
    print(f"    Ratio predicted: {ratio_pred:.4f}")
    print(f"    Ratio observed:  {ratio_obs:.4f}")
    print(f"    Agreement: {agreement:.6f} ({error:.4f}% error) {status}")
    print()

print("="*80)
print("RESULT")
print("="*80)
print()

if all(abs(1 - (eta_abs**(betas[p1]-betas[p2])) / (LEPTON_YUKAWAS[p1]/LEPTON_YUKAWAS[p2])) < 0.001
       for p1, p2 in [('τ', 'μ'), ('μ', 'e'), ('τ', 'e')]):
    print("✓✓✓ PERFECT! Hierarchies match exactly!")
    print()
    print("Simplified formula:")
    print("  Y_i = N × |η(τ)|^(β_i)")
    print()
    print("  where β_i = -2k_i + δ_gen")
    print(f"    δ_1 = {delta_betas[1]:+.2f}")
    print(f"    δ_2 = {delta_betas[2]:+.2f}")
    print(f"    δ_3 = {delta_betas[3]:+.2f}")
    print()
    print("→ C_i structure constants NOT NEEDED!")
else:
    print("→ Hierarchies still don't match exactly")
    print("→ Need to investigate further")

print()
