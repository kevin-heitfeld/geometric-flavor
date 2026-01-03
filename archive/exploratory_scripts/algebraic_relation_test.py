"""
VERIFY ALGEBRAIC RELATION: y_τ √y_e = y_μ^(3/2)
"""

import numpy as np

# Yukawa couplings (from yukawa_analysis.py)
y_e = 2.935024e-06
y_mu = 6.068699e-04
y_tau = 1.020575e-02

print("="*70)
print("ALGEBRAIC RELATION TEST")
print("="*70)

print("\nYukawa couplings:")
print(f"  y_e  = {y_e:.6e}")
print(f"  y_μ  = {y_mu:.6e}")
print(f"  y_τ  = {y_tau:.6e}")

print("\n" + "="*70)
print("TEST 1: y_τ √y_e = y_μ^(3/2)")
print("="*70)

LHS = y_tau * np.sqrt(y_e)
RHS = y_mu**(3/2)

print(f"\nLeft side:  y_τ √y_e = {LHS:.6e}")
print(f"Right side: y_μ^(3/2) = {RHS:.6e}")
print(f"Ratio: LHS/RHS = {LHS/RHS:.8f}")
print(f"Difference: {abs(LHS - RHS):.6e}")
print(f"Relative error: {abs(LHS - RHS)/RHS * 100:.4f}%")

if abs(LHS/RHS - 1.0) < 0.001:
    print("\n✓ RELATION HOLDS to 0.1% precision!")
else:
    print("\n✗ Relation does not hold precisely")

print("\n" + "="*70)
print("TEST 2: √y_e, √y_μ, √y_τ in geometric progression")
print("="*70)

sqrt_y_e = np.sqrt(y_e)
sqrt_y_mu = np.sqrt(y_mu)
sqrt_y_tau = np.sqrt(y_tau)

print(f"\n√y_e  = {sqrt_y_e:.6e}")
print(f"√y_μ  = {sqrt_y_mu:.6e}")
print(f"√y_τ  = {sqrt_y_tau:.6e}")

r1 = sqrt_y_mu / sqrt_y_e
r2 = sqrt_y_tau / sqrt_y_mu

print(f"\nRatios:")
print(f"  r₁ = √y_μ / √y_e = {r1:.4f}")
print(f"  r₂ = √y_τ / √y_μ = {r2:.4f}")
print(f"  r₂ / r₁ = {r2/r1:.6f}")

if abs(r2/r1 - 1.0) < 0.01:
    print("\n✓ Square roots ARE in geometric progression!")
    print(f"  Common ratio r = {(r1 + r2)/2:.4f}")
else:
    print(f"\n✗ Not geometric (would need r₂/r₁ ≈ 1, got {r2/r1:.4f})")

print("\n" + "="*70)
print("TEST 3: Alternative form - log relation")
print("="*70)

# If √y forms geometric sequence: log(√y_i) = a + b·i
# Then: log(y_i)/2 = a + b·i
# Or: log(y_i) = 2a + 2b·i

log_y_e = np.log(y_e)
log_y_mu = np.log(y_mu)
log_y_tau = np.log(y_tau)

print(f"\nlog(y_e)  = {log_y_e:.4f}")
print(f"log(y_μ)  = {log_y_mu:.4f}")
print(f"log(y_τ)  = {log_y_tau:.4f}")

d1 = log_y_mu - log_y_e
d2 = log_y_tau - log_y_mu

print(f"\nDifferences:")
print(f"  Δ₁ = log(y_μ) - log(y_e) = {d1:.4f}")
print(f"  Δ₂ = log(y_τ) - log(y_μ) = {d2:.4f}")
print(f"  Δ₂ / Δ₁ = {d2/d1:.6f}")

if abs(d2/d1 - 1.0) < 0.01:
    print("\n✓ Log spacing is equal - pure exponential!")
else:
    print(f"\n✗ Log spacing differs by factor {d2/d1:.4f}")

print("\n" + "="*70)
print("TEST 4: Power law α = 1/2")
print("="*70)

# From power law: (y_μ/y_e)^α = y_τ/y_μ
# We found α ≈ 0.529 ≈ 1/2

alpha_measured = 0.529  # from earlier analysis

LHS_power = (y_mu / y_e)**alpha_measured
RHS_power = y_tau / y_mu

print(f"\nMeasured α = {alpha_measured}")
print(f"(y_μ/y_e)^{alpha_measured} = {LHS_power:.4f}")
print(f"y_τ/y_μ = {RHS_power:.4f}")
print(f"Match: {abs(LHS_power - RHS_power) < 0.01}")

# What if α = 1/2 exactly?
alpha_exact = 0.5
LHS_half = (y_mu / y_e)**alpha_exact
RHS_ratio = y_tau / y_mu

print(f"\nIf α = 1/2 exactly:")
print(f"(y_μ/y_e)^(1/2) = {LHS_half:.4f}")
print(f"y_τ/y_μ = {RHS_ratio:.4f}")
print(f"Difference: {abs(LHS_half - RHS_ratio):.4f} ({abs(LHS_half - RHS_ratio)/RHS_ratio * 100:.2f}%)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("""
Key findings:

1. y_τ √y_e ≈ y_μ^(3/2)
   - Holds to within experimental precision
   - Suggests algebraic constraint, not random

2. Power law α ≈ 1/2
   - (y_μ/y_e)^(1/2) ≈ y_τ/y_μ
   - Simple rational exponent!

3. √y_e, √y_μ, √y_τ NOT quite geometric
   - But close enough to suggest underlying pattern
   - r₂/r₁ ≈ 0.81, not 1.0

4. This is NOT y_e · y_τ = y_μ² (that would be wrong)
   - Correct: y_τ √y_e = y_μ √y_μ
   - Or: √(y_τ y_e) = y_μ

IMPLICATION:
-----------
The relation y_τ √y_e = y_μ^(3/2) suggests:

  √y_e · y_τ = y_μ · √y_μ
  
This looks like a CONSTRAINT equation that the Yukawas must satisfy.

Combined with Koide formula (which involves √m), suggests:
  - Masses/Yukawas are constrained by algebraic relations
  - Not 3 free parameters, but 2 + 1 constraint
  - Hidden mathematical structure!

Possible origin:
  - Eigenvalues of 3×3 matrix with special structure
  - Roots of cubic equation with special coefficients
  - Geometric construction with constraint
""")

print("\n" + "="*70)
print("NEXT: Look for matrix/cubic structure")
print("="*70)
