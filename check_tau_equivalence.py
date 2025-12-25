"""
Check if the different τ values in our documents are SL(2,ℤ) equivalent

SL(2,ℤ) transformation: τ → (aτ + b)/(cτ + d) with ad - bc = 1

If two τ values are equivalent, they give the SAME modular forms and physics.
"""

import numpy as np
from itertools import product

# The four τ values found in our documents
tau_values = {
    'manuscript_baseline': 1.2 + 0.8j,
    'orbifold_fixed': 0.5 + 1.6j,
    'theory14_fit': 2.69j,
    'manuscript_imaginary_axis': 5.0j
}

def sl2z_transform(tau, a, b, c, d):
    """Apply SL(2,ℤ) transformation"""
    return (a * tau + b) / (c * tau + d)

def are_equivalent(tau1, tau2, max_cd=10, tol=1e-6):
    """
    Check if tau1 and tau2 are SL(2,ℤ) equivalent

    Search over integer matrices with |c|, |d| <= max_cd
    """
    for c in range(-max_cd, max_cd+1):
        for d in range(-max_cd, max_cd+1):
            if c == 0 and d == 0:
                continue

            # Determine a, b from ad - bc = 1
            # Try all combinations
            for a in range(-max_cd, max_cd+1):
                # b = (ad - 1) / c if c != 0
                if c != 0:
                    if (a * d - 1) % c == 0:
                        b = (a * d - 1) // c
                        if abs(b) <= max_cd:
                            # Check ad - bc = 1
                            if a * d - b * c != 1:
                                continue
                            # Apply transformation
                            tau_transformed = sl2z_transform(tau1, a, b, c, d)
                            if abs(tau_transformed - tau2) < tol:
                                return True, (a, b, c, d)
                else:  # c = 0
                    # Need ad = 1, so a = d = ±1
                    if a * d == 1 and a * d - b * c == 1:
                        # b is free
                        for b_try in range(-max_cd, max_cd+1):
                            tau_transformed = sl2z_transform(tau1, a, b_try, 0, d)
                            if abs(tau_transformed - tau2) < tol:
                                return True, (a, b_try, 0, d)

    return False, None

def modular_j(tau):
    """
    Compute j-invariant (SL(2,ℤ) invariant quantity)

    If two τ values have the same j, they are equivalent.
    j(τ) = 1728 * E₄(τ)³ / (E₄(τ)³ - E₆(τ)²)

    For quick check, we use approximation valid at moderate Im(τ)
    """
    q = np.exp(2j * np.pi * tau)

    # E₄ q-expansion (first few terms)
    E4 = 1 + 240 * np.sum([n**3 * q**n / (1 - q**n) for n in range(1, 20)])

    # E₆ q-expansion
    E6 = 1 - 504 * np.sum([n**5 * q**n / (1 - q**n) for n in range(1, 20)])

    # j-invariant
    j = 1728 * E4**3 / (E4**3 - E6**2)

    return j

print("="*70)
print("TAU PARAMETER EQUIVALENCE CHECK")
print("="*70)

print("\n1. TAU VALUES FROM DOCUMENTS:")
print("-" * 70)
for name, tau in tau_values.items():
    print(f"  {name:30s}: τ = {tau:.3f}")
    print(f"    {'':30s}  Re(τ) = {tau.real:.3f}, Im(τ) = {tau.imag:.3f}")

print("\n2. J-INVARIANTS (SL(2,ℤ) invariant):")
print("-" * 70)
print("If two τ values have the same j-invariant, they are equivalent.")
print()

j_values = {}
for name, tau in tau_values.items():
    try:
        j = modular_j(tau)
        j_values[name] = j
        print(f"  {name:30s}: j(τ) = {j:.6e}")
    except:
        print(f"  {name:30s}: j(τ) = [computation failed]")

print("\n3. PAIRWISE EQUIVALENCE CHECK:")
print("-" * 70)
print("Searching for SL(2,ℤ) transformations relating τ values...")
print("(checking matrices with |a|,|b|,|c|,|d| ≤ 10)")
print()

names = list(tau_values.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        name1, name2 = names[i], names[j]
        tau1, tau2 = tau_values[name1], tau_values[name2]

        print(f"  Checking: {name1} ↔ {name2}")
        print(f"    τ₁ = {tau1:.3f}, τ₂ = {tau2:.3f}")

        # Check j-invariants first
        if name1 in j_values and name2 in j_values:
            j_diff = abs(j_values[name1] - j_values[name2])
            print(f"    |j(τ₁) - j(τ₂)| = {j_diff:.3e}")
            if j_diff < 1e-3:
                print(f"    ✓ j-invariants match → LIKELY EQUIVALENT")
            else:
                print(f"    ✗ j-invariants differ → NOT EQUIVALENT")

        # Try to find explicit transformation
        equiv, matrix = are_equivalent(tau1, tau2, max_cd=10)
        if equiv:
            a, b, c, d = matrix
            print(f"    ✓✓ FOUND TRANSFORMATION: τ₂ = ({a}τ₁ + {b})/({c}τ₁ + {d})")
            # Verify
            tau_check = sl2z_transform(tau1, a, b, c, d)
            print(f"    Verification: ({a}τ₁ + {b})/({c}τ₁ + {d}) = {tau_check:.6f}")
            print(f"    Error: |τ_check - τ₂| = {abs(tau_check - tau2):.3e}")
        else:
            print(f"    ✗ No SL(2,ℤ) transformation found (up to max_cd=10)")

        print()

print("\n4. FUNDAMENTAL DOMAIN CHECK:")
print("-" * 70)
print("Standard fundamental domain: -0.5 ≤ Re(τ) ≤ 0.5, |τ| ≥ 1")
print()

for name, tau in tau_values.items():
    in_domain = (-0.5 <= tau.real <= 0.5) and (abs(tau) >= 1.0)
    status = "✓ In domain" if in_domain else "✗ Outside domain"
    print(f"  {name:30s}: {status}")
    print(f"    Re(τ) = {tau.real:.3f}, |τ| = {abs(tau):.3f}")

print("\n5. IMPLICATIONS FOR OUR THEORY:")
print("-" * 70)

# Check if manuscript_baseline and theory14_fit are close
tau_manuscript = tau_values['manuscript_baseline']
tau_theory14 = tau_values['theory14_fit']
tau_orbifold = tau_values['orbifold_fixed']

print(f"\nKey question: Are τ_manuscript = {tau_manuscript:.3f} and τ_theory14 = {tau_theory14:.3f} equivalent?")

if 'manuscript_baseline' in j_values and 'theory14_fit' in j_values:
    j_diff = abs(j_values['manuscript_baseline'] - j_values['theory14_fit'])
    if j_diff < 1e-3:
        print("  → YES: j-invariants match, these are SL(2,ℤ) equivalent!")
        print("  → The inconsistency is just a choice of coordinates")
        print("  → We should pick one canonical representative")
    else:
        print("  → NO: j-invariants differ significantly")
        print("  → These are DIFFERENT points in moduli space")
        print("  → This is a REAL INCONSISTENCY that must be resolved")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

# Summary
if 'manuscript_baseline' in j_values and 'theory14_fit' in j_values:
    j_man = j_values['manuscript_baseline']
    j_th14 = j_values['theory14_fit']

    if abs(j_man - j_th14) < 1e-3:
        print("\n✓ GOOD NEWS: τ = 1.2+0.8i and τ = 2.69i are SL(2,ℤ) equivalent!")
        print("  → Same physics, just different coordinate choice")
        print("  → Pick one canonical form and stick with it")
        print("  → Update all documents to use same representative")
    else:
        print("\n✗ BAD NEWS: τ = 1.2+0.8i and τ = 2.69i are NOT equivalent!")
        print("  → Different points in moduli space")
        print("  → Give DIFFERENT modular forms")
        print("  → Give DIFFERENT flavor predictions")
        print("  → We have a SERIOUS CONSISTENCY PROBLEM")
        print("\n  MUST determine which τ actually fits the data!")

print("\n" + "="*70)
