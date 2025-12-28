"""
INTEGER SENSITIVITY ANALYSIS: Neutrino k-pattern
=================================================

TEST: Does rounding neutrino weights to EXACT integers maintain fit quality?

HYPOTHESIS FROM STRESS TEST:
- Free fit:        k ≈ (7.3, 3.3, 1.7)
- Constrained fit: k ≈ (5.0, 3.0, 1.0)  [Δk=2 enforced]

CRITICAL QUESTION:
Can we lock neutrino sector to EXACT INTEGER k-pattern with Δk=2?

CANDIDATES TO TEST:
1. k = (5, 3, 1) [from constrained fit]
2. k = (6, 4, 2) [alternative Δk=2 pattern]
3. k = (7, 5, 3) [higher weight alternative]
4. k = (8, 6, 4) [same as charged leptons - should fail]

CROSS-SECTOR CONSISTENCY TEST:
- Charged leptons: k=(8,6,4), τ≈3.25i
- If neutrinos work with SAME τ, this is HUGE (unified modular parameter)
- If different τ needed, sectors partially decouple
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# ==============================================================================
# EXPERIMENTAL DATA (NuFIT 5.2)
# ==============================================================================

Delta_m21_sq = 7.53e-5  # eV²
Delta_m21_sq_err = 0.18e-5

Delta_m31_sq = 2.453e-3  # eV² (normal hierarchy)
Delta_m31_sq_err = 0.033e-3

print("="*80)
print("INTEGER SENSITIVITY ANALYSIS: Neutrino k-pattern")
print("="*80)
print("\nTEST: Can we lock neutrino weights to EXACT integers with Δk=2?")
print("\nCharged sector: k=(8,6,4), τ≈3.25i")
print("Question: Do neutrinos share the SAME τ? (Cross-sector unification)")
print()

# ==============================================================================
# MODULAR FORM MASS FORMULA
# ==============================================================================

def eta_function_weight_k(tau, k):
    """Dedekind eta function |η(τ)|^k for mass calculation."""
    q = np.exp(2j * np.pi * tau)
    prefactor = np.abs(q)**(k/24)
    product = 1.0
    for n in range(1, 51):
        product *= np.abs(1 - q**n)**k
    return prefactor * product

def mass_from_k(k_values, tau, m_scale=1.0):
    """
    Compute masses from k-pattern.
    Returns mass-squared differences.
    """
    masses = np.array([m_scale * eta_function_weight_k(tau, k) for k in k_values])
    masses = np.sort(masses)

    Delta_m21_sq = masses[1]**2 - masses[0]**2
    Delta_m31_sq = masses[2]**2 - masses[0]**2

    return Delta_m21_sq, Delta_m31_sq

def chi_squared(tau_im, log_m_scale, k_pattern):
    """
    χ² for fixed integer k-pattern.
    Only fit τ and m_scale.
    """
    tau = 1j * tau_im
    m_scale = 10**log_m_scale

    try:
        dm21, dm31 = mass_from_k(k_pattern, tau, m_scale)
    except:
        return 1e10

    chi2_solar = ((dm21 - Delta_m21_sq) / Delta_m21_sq_err)**2
    chi2_atm = ((dm31 - Delta_m31_sq) / Delta_m31_sq_err)**2

    return chi2_solar + chi2_atm

# ==============================================================================
# TEST 1: Integer k-patterns with FREE τ
# ==============================================================================

print("="*80)
print("TEST 1: INTEGER k-PATTERNS (τ and m_scale free)")
print("="*80)
print("\nFit τ and m_scale for each integer k-pattern.")
print("Check which pattern(s) give acceptable χ².\n")

# Candidates
k_patterns = {
    '(5,3,1)': [5, 3, 1],
    '(6,4,2)': [6, 4, 2],
    '(7,5,3)': [7, 5, 3],
    '(8,6,4)': [8, 6, 4],  # Same as charged - should fail or need different τ
}

results = {}

for name, k_pattern in k_patterns.items():
    # Initial guess: tau~3-4i, m_scale~0.05-0.1 eV
    initial = [3.5, -1.0]  # tau_im, log10(m_scale)

    # Wrapper to match minimize signature
    def chi2_wrapper(params):
        return chi_squared(params[0], params[1], k_pattern)

    result = minimize(
        chi2_wrapper,
        initial,
        bounds=[(1, 10), (-2, 0)],
        method='L-BFGS-B'
    )

    tau_fit = result.x[0]
    m_scale_fit = 10**result.x[1]
    chi2_min = result.fun

    # Compute p-value (2 observables, 2 fitted params → dof=0, but use dof=1 conservatively)
    p_value = 1 - chi2.cdf(chi2_min, 1)

    results[name] = {
        'k': k_pattern,
        'tau': tau_fit,
        'm_scale': m_scale_fit,
        'chi2': chi2_min,
        'p_value': p_value
    }

    print(f"k = {name}:")
    print(f"  Best fit: τ = {tau_fit:.3f}i, m_scale = {m_scale_fit:.4f} eV")
    print(f"  χ² = {chi2_min:.3f}, p-value = {p_value:.3f}")

    if chi2_min < 4:
        print(f"  ✓ ACCEPTABLE FIT (χ² < 4)")
    else:
        print(f"  ❌ POOR FIT (χ² > 4)")
    print()

# ==============================================================================
# TEST 2: Lock τ to charged sector value (τ ≈ 3.25i)
# ==============================================================================

print("="*80)
print("TEST 2: CROSS-SECTOR CONSISTENCY (τ = 3.25i FIXED)")
print("="*80)
print("\nCANONICAL TEST: Use τ from charged leptons.")
print("If neutrinos fit with SAME τ, sectors are unified.\n")

tau_charged = 3.25j  # From charged lepton fits

print("Charged sector: τ = 3.25i (from e/μ/τ masses)")
print("Testing neutrino sector with SAME τ...\n")

results_fixed_tau = {}

for name, k_pattern in k_patterns.items():
    # Only fit m_scale, τ is fixed
    initial_m = [-1.0]  # log10(m_scale)

    def chi2_fixed_tau(log_m):
        return chi_squared(tau_charged.imag, log_m[0], k_pattern)

    result = minimize(
        chi2_fixed_tau,
        initial_m,
        bounds=[(-2, 0)],
        method='L-BFGS-B'
    )

    m_scale_fit = 10**result.x[0]
    chi2_min = result.fun
    p_value = 1 - chi2.cdf(chi2_min, 1)

    results_fixed_tau[name] = {
        'k': k_pattern,
        'tau': tau_charged.imag,
        'm_scale': m_scale_fit,
        'chi2': chi2_min,
        'p_value': p_value
    }

    print(f"k = {name} with τ = 3.25i:")
    print(f"  Best m_scale = {m_scale_fit:.4f} eV")
    print(f"  χ² = {chi2_min:.3f}, p-value = {p_value:.3f}")

    if chi2_min < 4:
        print(f"  ✓ CROSS-SECTOR UNIFICATION WORKS!")
        print(f"    Same τ for charged and neutral sectors")
    else:
        print(f"  ❌ Requires different τ (sectors decouple)")
    print()

# ==============================================================================
# TEST 3: k-offset hypothesis (k_ν = k_charged - offset)
# ==============================================================================

print("="*80)
print("TEST 3: k-OFFSET HYPOTHESIS")
print("="*80)
print("\nHYPOTHESIS: k_ν = k_charged - Δ for some offset Δ")
print("Charged: k=(8,6,4) → Neutrino: k=(8-Δ, 6-Δ, 4-Δ)")
print()

k_charged = [8, 6, 4]

# Test various offsets
offsets_to_test = [0, 1, 2, 3, 4, 5, 6]

print("Testing k_ν = k_charged - Δ with τ=3.25i:\n")

offset_results = []

for Delta in offsets_to_test:
    k_offset = [k - Delta for k in k_charged]

    # Skip if any k becomes negative
    if any(k < 0 for k in k_offset):
        print(f"Δ={Delta}: k={k_offset} → SKIP (negative weight)")
        continue

    # Fit with fixed tau
    initial_m = [-1.0]

    def chi2_offset(log_m):
        return chi_squared(tau_charged.imag, log_m[0], k_offset)

    result = minimize(
        chi2_offset,
        initial_m,
        bounds=[(-2, 0)],
        method='L-BFGS-B'
    )

    m_scale = 10**result.x[0]
    chi2_min = result.fun

    offset_results.append({
        'Delta': Delta,
        'k': k_offset,
        'chi2': chi2_min,
        'm_scale': m_scale
    })

    print(f"Δ={Delta}: k={k_offset}, χ²={chi2_min:.3f}, m={m_scale:.4f} eV", end='')

    if chi2_min < 1:
        print(" ✓✓ EXCELLENT")
    elif chi2_min < 4:
        print(" ✓ GOOD")
    else:
        print(" ❌ POOR")

# Find best offset
best_offset = min(offset_results, key=lambda x: x['chi2'])

print(f"\n⚠️  BEST OFFSET: Δ = {best_offset['Delta']}")
print(f"   k_ν = {best_offset['k']}")
print(f"   χ² = {best_offset['chi2']:.3f}")
print(f"   m_scale = {best_offset['m_scale']:.4f} eV")

if best_offset['Delta'] == 3:
    print("\n✓ PATTERN CONFIRMED: k_ν = k_charged - 3")
    print("  Charged:  k = (8, 6, 4)")
    print("  Neutrino: k = (5, 3, 1)")
    print("\n  This is a UNIVERSAL TRANSFORMATION:")
    print("  Neutral sector = Charged sector - 3 units")
    print("  Possible interpretation: Majorana vs Dirac, or double cover of modular group")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: χ² comparison for integer patterns (free τ)
ax = axes[0, 0]
names = list(results.keys())
chi2_vals = [results[n]['chi2'] for n in names]
colors = ['green' if c < 4 else 'red' for c in chi2_vals]

bars = ax.bar(names, chi2_vals, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(4, color='red', linestyle='--', linewidth=2, label='χ²=4 threshold')
ax.set_ylabel('χ²', fontsize=11)
ax.set_title('A. Integer k-patterns (τ free)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bar, val in zip(bars, chi2_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.2,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# Panel B: χ² with τ=3.25i fixed
ax = axes[0, 1]
chi2_fixed = [results_fixed_tau[n]['chi2'] for n in names]
colors_fixed = ['green' if c < 4 else 'red' for c in chi2_fixed]

bars = ax.bar(names, chi2_fixed, color=colors_fixed, alpha=0.7, edgecolor='black')
ax.axhline(4, color='red', linestyle='--', linewidth=2, label='χ²=4 threshold')
ax.set_ylabel('χ²', fontsize=11)
ax.set_title('B. Integer k-patterns (τ=3.25i fixed)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bar, val in zip(bars, chi2_fixed):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.2,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# Panel C: k-offset scan
ax = axes[1, 0]
deltas = [r['Delta'] for r in offset_results]
chi2_offset = [r['chi2'] for r in offset_results]

ax.plot(deltas, chi2_offset, 'bo-', linewidth=2, markersize=8)
ax.axhline(4, color='red', linestyle='--', linewidth=2, label='χ²=4 threshold')
ax.axhline(1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='χ²=1 (excellent)')
ax.axvline(best_offset['Delta'], color='orange', linestyle=':', linewidth=2, label=f"Best: Δ={best_offset['Delta']}")

ax.set_xlabel('Offset Δ (k_ν = k_charged - Δ)', fontsize=11)
ax.set_ylabel('χ²', fontsize=11)
ax.set_title('C. k-offset Hypothesis (τ=3.25i)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel D: τ comparison across patterns
ax = axes[1, 1]
tau_vals = [results[n]['tau'] for n in names]
x_pos = np.arange(len(names))

bars = ax.bar(x_pos, tau_vals, alpha=0.7, color='blue', edgecolor='black')
ax.axhline(3.25, color='red', linestyle='--', linewidth=2, label='τ=3.25i (charged)')
ax.set_ylabel('Im(τ)', fontsize=11)
ax.set_xlabel('k-pattern', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.set_title('D. Fitted τ for each k-pattern', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bar, val in zip(bars, tau_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('neutrino_integer_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('neutrino_integer_sensitivity.pdf', dpi=300, bbox_inches='tight')

print("\n" + "="*80)
print("FINAL VERDICT: INTEGER SENSITIVITY")
print("="*80)

# Find best overall pattern
best_free = min(results.items(), key=lambda x: x[1]['chi2'])
best_fixed = min(results_fixed_tau.items(), key=lambda x: x[1]['chi2'])

print(f"\n✓ BEST INTEGER PATTERN (τ free):")
print(f"  k = {best_free[0]}")
print(f"  τ = {best_free[1]['tau']:.3f}i")
print(f"  m_scale = {best_free[1]['m_scale']:.4f} eV")
print(f"  χ² = {best_free[1]['chi2']:.3f}")

print(f"\n✓ BEST INTEGER PATTERN (τ=3.25i fixed):")
print(f"  k = {best_fixed[0]}")
print(f"  m_scale = {best_fixed[1]['m_scale']:.4f} eV")
print(f"  χ² = {best_fixed[1]['chi2']:.3f}")

# Cross-sector unification test
if best_fixed[1]['chi2'] < 4:
    print("\n✓✓✓ CROSS-SECTOR UNIFICATION ACHIEVED!")
    print("    Charged and neutrino sectors share SAME τ")
    print("    This is MAJOR: unified modular parameter across fermions")
else:
    print("\n⚠️  Cross-sector unification MARGINAL")
    print("   Neutrinos prefer different τ than charged leptons")

# k-offset pattern
if best_offset['Delta'] in [2, 3, 4]:
    print(f"\n✓ k-OFFSET PATTERN CONFIRMED: Δ = {best_offset['Delta']}")
    print(f"  k_charged = (8, 6, 4)")
    print(f"  k_neutrino = {best_offset['k']}")
    print(f"  Transformation: k_ν = k_charged - {best_offset['Delta']}")

    if best_offset['Delta'] == 3:
        print("\n  INTERPRETATION:")
        print("  • Δ=3 suggests HALF-INTEGER SPIN difference (3/2 → 1/2)?")
        print("  • Or double cover: Mp(2,ℤ) vs SL(2,ℤ)")
        print("  • Majorana vs Dirac: different multiplier systems")

print("\n" + "="*80)
print("Figures saved: neutrino_integer_sensitivity.png/pdf")
print("="*80)
