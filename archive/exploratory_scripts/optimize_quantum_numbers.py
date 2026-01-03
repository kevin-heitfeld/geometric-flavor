"""
Find optimal quantum number assignments for each sector

The key insight: Different sectors may have different (q₃, q₄) assignments
while still using w = -2q₃ + q₄ formula
"""

import numpy as np
from itertools import product

tau = 2.69j
Im_tau = 2.69

# Dedekind eta
q = np.exp(2 * np.pi * 1j * tau)
eta = q**(1/24) * np.prod([1 - q**n for n in range(1, 30)])

def yukawa_scaling(w):
    return (Im_tau)**(-w) * np.abs(eta)**(-6*w)

def modular_weight(q3, q4):
    return -2*q3 + q4

v_higgs = 246.0

# Experimental values
sectors_exp = {
    'Up Quarks': {
        'up': np.sqrt(2) * 2.16e-3 / v_higgs,
        'charm': np.sqrt(2) * 1.27 / v_higgs,
        'top': np.sqrt(2) * 172.57 / v_higgs
    },
    'Down Quarks': {
        'down': np.sqrt(2) * 4.67e-3 / v_higgs,
        'strange': np.sqrt(2) * 93.4e-3 / v_higgs,
        'bottom': np.sqrt(2) * 4.18 / v_higgs
    },
    'Charged Leptons': {
        'electron': 2.80e-6,
        'muon': 6.09e-4,
        'tau': 1.04e-2
    }
}

# Generate all possible quantum number combinations
# q₃ ∈ {0, 1, 2}, q₄ ∈ {0, 1, 2, 3}
possible_q3 = range(3)
possible_q4 = range(4)

def find_best_assignment(sector_name, exp_values):
    """
    Search for best quantum number assignment.

    Require: three distinct (q₃, q₄) pairs giving distinct weights
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZING: {sector_name}")
    print(f"{'='*70}\n")

    particles = list(exp_values.keys())
    n = len(particles)

    best_assignment = None
    best_error = float('inf')

    # Try all combinations of 3 distinct quantum number pairs
    all_pairs = list(product(possible_q3, possible_q4))

    tested = 0
    for combo in product(all_pairs, repeat=n):
        # Check all distinct
        if len(set(combo)) != n:
            continue

        tested += 1

        # Compute weights
        weights = {particles[i]: modular_weight(*combo[i]) for i in range(n)}

        # Check weights are distinct (otherwise can't distinguish generations)
        if len(set(weights.values())) != n:
            continue

        # Compute Yukawas
        yukawas_calc = {p: yukawa_scaling(w) for p, w in weights.items()}

        # Normalize to lightest
        norm = exp_values[particles[0]] / yukawas_calc[particles[0]]
        yukawas_calc_norm = {p: y * norm for p, y in yukawas_calc.items()}

        # Compute error
        errors = []
        for p in particles:
            error = abs(yukawas_calc_norm[p] / exp_values[p] - 1.0) * 100
            errors.append(error)

        avg_error = np.mean(errors)

        if avg_error < best_error:
            best_error = avg_error
            best_assignment = {
                'quantum_numbers': {particles[i]: combo[i] for i in range(n)},
                'weights': weights,
                'yukawas': yukawas_calc_norm,
                'errors': {particles[i]: errors[i] for i in range(n)}
            }

    print(f"Tested {tested} distinct assignments\n")

    if best_assignment:
        print(f"BEST ASSIGNMENT (avg error {best_error:.1f}%):")
        print("-" * 70)

        for p in particles:
            q3, q4 = best_assignment['quantum_numbers'][p]
            w = best_assignment['weights'][p]
            y_calc = best_assignment['yukawas'][p]
            y_exp = exp_values[p]
            error = best_assignment['errors'][p]

            status = "✓" if error < 50 else ("~" if error < 200 else "✗")
            print(f"{p:10s}: (q₃={q3}, q₄={q4}) → w={w:+2d} | "
                  f"Y={y_calc:.3e} vs {y_exp:.3e} | {error:6.1f}% {status}")

        return best_assignment
    else:
        print("No valid assignment found!")
        return None

# Find best for each sector
results = {}
for sector_name, exp_values in sectors_exp.items():
    results[sector_name] = find_best_assignment(sector_name, exp_values)

# Compare
print(f"\n{'='*70}")
print("COMPARISON OF OPTIMAL ASSIGNMENTS")
print(f"{'='*70}\n")

for sector_name, result in results.items():
    if result:
        print(f"{sector_name}:")
        for p, (q3, q4) in result['quantum_numbers'].items():
            w = result['weights'][p]
            print(f"  {p:10s}: (q₃={q3}, q₄={q4}) → w={w:+2d}")
        print()

print("="*70)
print("KEY FINDINGS")
print("="*70)
print()
print("If optimal assignments are DIFFERENT across sectors:")
print("  → Each sector has sector-specific localization on T⁶")
print("  → Different wrapping numbers or brane positions")
print("  → Formula w=-2q₃+q₄ is UNIVERSAL but (q₃,q₄) are sector-dependent")
print()
print("If optimal assignments are THE SAME:")
print("  → Universal quantum numbers")
print("  → Errors come from missing physics (RG, thresholds, Kähler)")
print()
