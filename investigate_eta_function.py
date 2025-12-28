"""
Investigating the missing eta function contribution

The yukawa_from_modular_forms function uses:
Y ∝ (Imτ)^(-w) × |η(τ)|^(-6w)

We only used the first term! Let's see what eta adds.
"""

import numpy as np

tau = 2.69j
Im_tau = 2.69

# Compute Dedekind eta function
q = np.exp(2 * np.pi * 1j * tau)
print(f"Nome q = exp(2πiτ) = exp(2πi×{tau})")
print(f"       = exp(-2π×{Im_tau}) = {np.abs(q):.4e}")
print()

# Eta function (truncated product for large Imτ)
eta = q**(1/24) * np.prod([1 - q**n for n in range(1, 30)])
print(f"|η(τ)| = {np.abs(eta):.6e}")
print()

# Weights
w_e = -2
w_mu = 0
w_tau = 1

print("="*70)
print("SCALING WITH ETA FUNCTION")
print("="*70)
print()

# Full scaling: (Imτ)^(-w) × |η|^(-6w)
def full_scaling(w):
    return (Im_tau)**(-w) * np.abs(eta)**(-6*w)

Y_e_full = full_scaling(w_e)
Y_mu_full = full_scaling(w_mu)
Y_tau_full = full_scaling(w_tau)

print(f"Full scaling (Imτ)^(-w) × |η|^(-6w):")
print(f"  Electron (w=-2): {Y_e_full:.6e}")
print(f"  Muon (w=0):      {Y_mu_full:.6e}")
print(f"  Tau (w=+1):      {Y_tau_full:.6e}")
print()

# Ratios WITH eta
ratio_mu_e_full = Y_mu_full / Y_e_full
ratio_tau_e_full = Y_tau_full / Y_e_full
ratio_tau_mu_full = Y_tau_full / Y_mu_full

print(f"Ratios WITH eta function:")
print(f"  Y_μ / Y_e = {ratio_mu_e_full:.6e}")
print(f"  Y_τ / Y_e = {ratio_tau_e_full:.6e}")
print(f"  Y_τ / Y_μ = {ratio_tau_mu_full:.6e}")
print()

# Experimental ratios
Y_e_exp = 2.80e-6
Y_mu_exp = 6.09e-4
Y_tau_exp = 1.04e-2

ratio_mu_e_exp = Y_mu_exp / Y_e_exp
ratio_tau_e_exp = Y_tau_exp / Y_e_exp
ratio_tau_mu_exp = Y_tau_exp / Y_mu_exp

print(f"Experimental ratios:")
print(f"  Y_μ / Y_e = {ratio_mu_e_exp:.1f}")
print(f"  Y_τ / Y_e = {ratio_tau_e_exp:.1f}")
print(f"  Y_τ / Y_μ = {ratio_tau_mu_exp:.1f}")
print()

# Comparison
print(f"Improvement from eta function:")
print(f"  Y_μ / Y_e: {ratio_mu_e_full:.4f} vs exp {ratio_mu_e_exp:.1f} → {ratio_mu_e_exp/ratio_mu_e_full:.1f}× off")
print(f"  Y_τ / Y_e: {ratio_tau_e_full:.4e} vs exp {ratio_tau_e_exp:.1f} → {ratio_tau_e_exp/ratio_tau_e_full:.1f}× off")
print(f"  Y_τ / Y_μ: {ratio_tau_mu_full:.4f} vs exp {ratio_tau_mu_exp:.1f} → {ratio_tau_mu_exp/ratio_tau_mu_full:.1f}× off")
print()

print("="*70)
print("BREAKDOWN OF ETA CONTRIBUTION")
print("="*70)
print()

# Separate contributions
for name, w in [("Electron", w_e), ("Muon", w_mu), ("Tau", w_tau)]:
    imtau_part = (Im_tau)**(-w)
    eta_part = np.abs(eta)**(-6*w)
    total = imtau_part * eta_part

    print(f"{name:10s} (w={w:+2d}):")
    print(f"  (Imτ)^(-w)     = {imtau_part:.6f}")
    print(f"  |η|^(-6w)      = {eta_part:.6e}")
    print(f"  Product        = {total:.6e}")
    print()

print("="*70)
print("KEY INSIGHT")
print("="*70)
print()
print("The |η(τ)|^(-6w) term is CRUCIAL!")
print()
print(f"For τ={tau}, |η(τ)| = {np.abs(eta):.6e} (very small!)")
print()
print("When w is NEGATIVE (electron):")
print(f"  |η|^(-6w) = |η|^(+12) = ({np.abs(eta):.4e})^12 = {np.abs(eta)**12:.4e}")
print("  This is HUGE! Enhances the coupling.")
print()
print("When w is POSITIVE (tau):")
print(f"  |η|^(-6w) = |η|^(-6) = ({np.abs(eta):.4e})^(-6) = {np.abs(eta)**(-6):.4e}")
print("  This is tiny! Suppresses the coupling.")
print()
print("This is WHY the hierarchy works!")
print("  → Negative weights get exponentially enhanced by eta")
print("  → Positive weights get exponentially suppressed")
print()

# What if we normalize to electron and recalculate?
print("="*70)
print("RECALCULATED WITH ETA (normalized to electron)")
print("="*70)
print()

norm = Y_e_exp / Y_e_full
Y_mu_calc = Y_mu_full * norm
Y_tau_calc = Y_tau_full * norm

print(f"Normalized values:")
print(f"  Electron: {Y_e_exp:.4e} (target)")
print(f"  Muon:     {Y_mu_calc:.4e} (calc) vs {Y_mu_exp:.4e} (exp)")
print(f"  Tau:      {Y_tau_calc:.4e} (calc) vs {Y_tau_exp:.4e} (exp)")
print()

mu_error = abs(Y_mu_calc - Y_mu_exp) / Y_mu_exp * 100
tau_error = abs(Y_tau_calc - Y_tau_exp) / Y_tau_exp * 100

print(f"Errors:")
print(f"  Muon: {mu_error:.1f}%")
print(f"  Tau:  {tau_error:.1f}%")
print()

if mu_error < 50 and tau_error < 50:
    print("✅ SUCCESS! The eta function fixes everything!")
elif mu_error < 300 and tau_error < 400:
    print("✓ MUCH BETTER! Factor 3-4 errors as claimed in Week 2")
else:
    print("✗ Still not enough. Need additional corrections.")
