"""
Investigate sector-dependent differences

Up quarks work well, but down quarks fail badly.
What's different?
"""

import numpy as np

tau = 2.69j
Im_tau = 2.69

# Dedekind eta
q = np.exp(2 * np.pi * 1j * tau)
eta = q**(1/24) * np.prod([1 - q**n for n in range(1, 30)])

def yukawa_scaling(w):
    return (Im_tau)**(-w) * np.abs(eta)**(-6*w)

print("="*70)
print("SECTOR COMPARISON")
print("="*70)
print()

# Same quantum numbers for all
quantum_numbers = {
    'lightest': (1, 0, -2),  # w=-2
    'middle': (0, 0, 0),     # w=0
    'heaviest': (0, 1, 1)    # w=+1
}

# Experimental Yukawa couplings
v_higgs = 246.0

sectors = {
    'Charged Leptons': {
        'lightest': 2.80e-6,
        'middle': 6.09e-4,
        'heaviest': 1.04e-2
    },
    'Up Quarks': {
        'lightest': np.sqrt(2) * 2.16e-3 / v_higgs,   # 1.24e-5
        'middle': np.sqrt(2) * 1.27 / v_higgs,         # 7.30e-3
        'heaviest': np.sqrt(2) * 172.57 / v_higgs      # 0.992
    },
    'Down Quarks': {
        'lightest': np.sqrt(2) * 4.67e-3 / v_higgs,   # 2.69e-5
        'middle': np.sqrt(2) * 93.4e-3 / v_higgs,      # 5.37e-4
        'heaviest': np.sqrt(2) * 4.18 / v_higgs        # 2.40e-2
    }
}

for sector_name, exp_values in sectors.items():
    print(f"\n{sector_name}:")
    print("-" * 70)

    # Calculate with same quantum numbers
    calc_values = {}
    for gen, (q3, q4, w) in quantum_numbers.items():
        calc_values[gen] = yukawa_scaling(w)

    # Normalize to lightest
    norm = exp_values['lightest'] / calc_values['lightest']
    calc_values_norm = {g: v * norm for g, v in calc_values.items()}

    # Compare
    for gen in ['lightest', 'middle', 'heaviest']:
        y_calc = calc_values_norm[gen]
        y_exp = exp_values[gen]
        ratio = y_calc / y_exp
        error = abs(ratio - 1.0) * 100

        status = "✓" if error < 50 else ("~" if error < 200 else "✗")
        print(f"  {gen:10s}: {y_calc:.3e} vs {y_exp:.3e} → {ratio:6.2f}× ({error:6.1f}%) {status}")

    # Ratios
    print(f"\n  Ratio analysis:")
    ratio_mid_light_calc = calc_values_norm['middle'] / calc_values_norm['lightest']
    ratio_mid_light_exp = exp_values['middle'] / exp_values['lightest']
    print(f"    middle/light: calc {ratio_mid_light_calc:.1f}, exp {ratio_mid_light_exp:.1f} → {ratio_mid_light_exp/ratio_mid_light_calc:.2f}× off")

    ratio_heavy_light_calc = calc_values_norm['heaviest'] / calc_values_norm['lightest']
    ratio_heavy_light_exp = exp_values['heaviest'] / exp_values['lightest']
    print(f"    heavy/light:  calc {ratio_heavy_light_calc:.1f}, exp {ratio_heavy_light_exp:.1f} → {ratio_heavy_light_exp/ratio_heavy_light_calc:.2f}× off")

print()
print("="*70)
print("HYPOTHESIS: Different sectors need different τ or quantum numbers")
print("="*70)
print()

# Try optimizing tau for down quarks
print("Testing different τ values for down quarks:")
print()

exp_down = sectors['Down Quarks']
best_tau = None
best_error = float('inf')

for Im_tau_test in [1.5, 2.0, 2.5, 2.69, 3.0, 3.5, 4.0, 5.0]:
    tau_test = 1j * Im_tau_test
    q_test = np.exp(2 * np.pi * 1j * tau_test)
    eta_test = q_test**(1/24) * np.prod([1 - q_test**n for n in range(1, 30)])

    def yukawa_test(w):
        return (Im_tau_test)**(-w) * np.abs(eta_test)**(-6*w)

    calc_test = {g: yukawa_test(w) for g, (_, _, w) in quantum_numbers.items()}
    norm_test = exp_down['lightest'] / calc_test['lightest']
    calc_test_norm = {g: v * norm_test for g, v in calc_test.items()}

    errors = []
    for gen in ['middle', 'heaviest']:
        error = abs(calc_test_norm[gen] / exp_down[gen] - 1.0) * 100
        errors.append(error)

    avg_error = np.mean(errors)

    status = "←" if avg_error < best_error else ""
    print(f"  Imτ = {Im_tau_test:.2f}: middle {errors[0]:6.1f}%, heavy {errors[1]:6.1f}% → avg {avg_error:6.1f}% {status}")

    if avg_error < best_error:
        best_error = avg_error
        best_tau = Im_tau_test

print()
print(f"Best τ for down quarks: Imτ = {best_tau:.2f} (avg error {best_error:.1f}%)")
print(f"Standard τ: Imτ = 2.69 (avg error ~2435%)")
print()
print("→ Down quarks may require DIFFERENT modular parameter!")
print("→ OR different quantum number assignment")
