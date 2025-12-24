"""
Multi-Brane Δk=2 Test: Does Δk=2 emerge for quarks with free τ?

CRITICAL QUESTION: Is Δk=2 UNIVERSAL or only leptonic?

After discovering:
- Leptons: τ=3.25i, Δk=2 ✓✓✓
- Quarks with τ=3.25i: FAIL (χ²>40,000)

This script tests: If we let quarks find their OWN τ, does Δk=2 still emerge?

If YES: Universal Δk=2 (multi-brane with geometric gauge split)
If NO: Framework incomplete (Δk=2 only accidental for leptons)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
import json

# Quark masses - constituent (best from sensitivity test)
quark_masses_up = np.array([0.330, 1.500, 172.5])  # u,c,t [GeV]
quark_masses_down = np.array([0.330, 0.500, 4.75])  # d,s,b [GeV]
sigma_up = 0.1 * quark_masses_up
sigma_down = 0.1 * quark_masses_down

# Known leptonic results
tau_leptons = 3.25
k_charged = np.array([8, 6, 4])  # Δk=2
k_neutral = np.array([5, 3, 1])  # Δk=2

print("="*70)
print("ΔKEY=2 UNIVERSALITY TEST")
print("="*70)
print("\nLeptonic sector (KNOWN):")
print(f"  τ = {tau_leptons}i")
print(f"  Charged: k={k_charged}, Δk=2 ✓")
print(f"  Neutral: k={k_neutral}, Δk=2 ✓")
print("\nQuark sector (TESTING):")
print(f"  With τ=3.25i: FAILS (χ²>40,000)")
print(f"  Question: Does Δk=2 emerge with FREE τ?")

# Mass formula
def mass_formula(k_values, tau, m0):
    """m_i = m_0 * |η(iτ)|^{2k_i}"""
    def eta(tau_val):
        q_val = np.exp(2j * np.pi * tau_val)
        result = q_val**(1/24)
        for n in range(1, 50):
            result *= (1 - q_val**n)
        return result

    masses = np.zeros(len(k_values))
    for i, k in enumerate(k_values):
        eta_val = eta(1j * tau)
        masses[i] = m0 * np.abs(eta_val)**(2*k)

    return masses

def fit_free_delta_k(masses_obs, sigma_obs, sector_name):
    """
    Test 1: Fit with COMPLETELY FREE parameters (τ, k₁, k₂, k₃, m₀)
    Test 2: Fit with Δk=2 CONSTRAINED (τ, k₁, m₀) where k₂=k₁-2, k₃=k₁-4

    Compare: Does Δk=2 emerge naturally or need to be imposed?
    """
    print(f"\n{'='*70}")
    print(f"{sector_name.upper()}: FREE Δk FIT")
    print(f"{'='*70}")
    print(f"Observed masses: {masses_obs}")

    # Test 1: Completely free
    print("\nTEST 1: Completely free (5 parameters)")
    print("Fitting: τ, k₁, k₂, k₃, m₀")

    def chi2_free(params):
        tau, k1, k2, k3, log_m0 = params
        m0 = np.exp(log_m0)
        k_vals = np.array([k1, k2, k3])

        if np.any(k_vals <= 0) or tau <= 0:
            return 1e10

        masses_theory = mass_formula(k_vals, tau, m0)
        return np.sum(((masses_obs - masses_theory) / sigma_obs)**2)

    tau_init = 3.0
    k_init = [8, 6, 4]
    m0_init = np.exp(np.mean(np.log(masses_obs)))

    result_free = minimize(
        chi2_free,
        x0=[tau_init, k_init[0], k_init[1], k_init[2], np.log(m0_init)],
        method='Nelder-Mead',
        options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-10}
    )

    tau_free = result_free.x[0]
    k_free = np.array([result_free.x[1], result_free.x[2], result_free.x[3]])
    m0_free = np.exp(result_free.x[4])
    chi2_free_val = result_free.fun

    delta_k_12_free = k_free[0] - k_free[1]
    delta_k_23_free = k_free[1] - k_free[2]
    delta_k_avg_free = (delta_k_12_free + delta_k_23_free) / 2

    print(f"  τ = {tau_free:.3f}i")
    print(f"  k = ({k_free[0]:.2f}, {k_free[1]:.2f}, {k_free[2]:.2f})")
    print(f"  Δk = {delta_k_avg_free:.2f}")
    print(f"  χ² = {chi2_free_val:.2f}")

    # Test 2: Δk=2 constrained
    print("\nTEST 2: Δk=2 constrained (3 parameters)")
    print("Fitting: τ, k₁, m₀ with k₂=k₁-2, k₃=k₁-4")

    def chi2_delta_k_2(params):
        tau, k1, log_m0 = params
        m0 = np.exp(log_m0)
        k_vals = np.array([k1, k1-2, k1-4])

        if np.any(k_vals <= 0) or tau <= 0:
            return 1e10

        masses_theory = mass_formula(k_vals, tau, m0)
        return np.sum(((masses_obs - masses_theory) / sigma_obs)**2)

    # Try multiple starting points for k₁
    best_result = None
    best_chi2 = np.inf

    for k1_start in [8, 7, 6, 5]:
        result = minimize(
            chi2_delta_k_2,
            x0=[3.0, k1_start, np.log(m0_init)],
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.fun < best_chi2:
            best_chi2 = result.fun
            best_result = result

    tau_const = best_result.x[0]
    k1_const = best_result.x[1]
    m0_const = np.exp(best_result.x[2])
    k_const = np.array([k1_const, k1_const-2, k1_const-4])
    chi2_const = best_result.fun

    # Calculate Δχ² cost
    delta_chi2 = chi2_const - chi2_free_val

    # p-value for constrained fit (3 data, 3 params → 0 dof, use 1 minimum)
    dof_const = max(len(masses_obs) - 3, 1)
    p_const = 1 - chi2.cdf(chi2_const, dof_const)

    print(f"  τ = {tau_const:.3f}i")
    print(f"  k = ({k_const[0]:.2f}, {k_const[1]:.2f}, {k_const[2]:.2f})")
    print(f"  χ² = {chi2_const:.2f}")
    print(f"  Δχ² = {delta_chi2:.2f} (cost of imposing Δk=2)")
    print(f"  p = {p_const:.3f}")

    # Verdict
    if abs(delta_k_avg_free - 2.0) < 0.3:
        print("\n  ✓✓✓ Δk≈2 EMERGES naturally!")
    elif delta_chi2 < 3.84:  # 95% confidence, 2 dof constraint
        print("\n  ✓ Δk=2 acceptable (Δχ²<3.84)")
    else:
        print(f"\n  ✗ Δk=2 strongly disfavored (Δχ²={delta_chi2:.1f} >> 3.84)")

    masses_fit = mass_formula(k_const, tau_const, m0_const)

    return {
        'tau_free': tau_free,
        'k_free': k_free,
        'chi2_free': chi2_free_val,
        'delta_k_free': delta_k_avg_free,
        'tau_const': tau_const,
        'k_const': k_const,
        'chi2_const': chi2_const,
        'delta_chi2': delta_chi2,
        'p_value': p_const,
        'masses_fit': masses_fit,
        'delta_k_2_emerges': abs(delta_k_avg_free - 2.0) < 0.3,
        'delta_k_2_acceptable': delta_chi2 < 3.84
    }# Test up-type quarks
print("\n" + "="*70)
print("UP-TYPE QUARKS (u, c, t)")
print("="*70)
up_result = fit_free_delta_k(quark_masses_up, sigma_up, "Up-type quarks")

# Test down-type quarks
print("\n" + "="*70)
print("DOWN-TYPE QUARKS (d, s, b)")
print("="*70)
down_result = fit_free_delta_k(quark_masses_down, sigma_down, "Down-type quarks")

# Summary
print("\n" + "="*70)
print("UNIVERSALITY VERDICT")
print("="*70)

print(f"\nUp-type quarks:")
print(f"  Free fit: τ={up_result['tau_free']:.3f}i, Δk={up_result['delta_k_free']:.2f}, χ²={up_result['chi2_free']:.2f}")
print(f"  Δk=2 fit: τ={up_result['tau_const']:.3f}i, k={np.round(up_result['k_const'])}, χ²={up_result['chi2_const']:.2f}")
print(f"  Δχ² cost: {up_result['delta_chi2']:.2f}")
if up_result['delta_k_2_emerges']:
    print(f"  ✓ Δk≈2 emerges naturally")
elif up_result['delta_k_2_acceptable']:
    print(f"  ✓ Δk=2 acceptable (Δχ²<3.84)")
else:
    print(f"  ✗ Δk=2 strongly disfavored")

print(f"\nDown-type quarks:")
print(f"  Free fit: τ={down_result['tau_free']:.3f}i, Δk={down_result['delta_k_free']:.2f}, χ²={down_result['chi2_free']:.2f}")
print(f"  Δk=2 fit: τ={down_result['tau_const']:.3f}i, k={np.round(down_result['k_const'])}, χ²={down_result['chi2_const']:.2f}")
print(f"  Δχ² cost: {down_result['delta_chi2']:.2f}")
if down_result['delta_k_2_emerges']:
    print(f"  ✓ Δk≈2 emerges naturally")
elif down_result['delta_k_2_acceptable']:
    print(f"  ✓ Δk=2 acceptable (Δχ²<3.84)")
else:
    print(f"  ✗ Δk=2 strongly disfavored")

# Check τ consistency (use constrained fit τ)
tau_avg_quarks = (up_result['tau_const'] + down_result['tau_const']) / 2
tau_diff = abs(up_result['tau_const'] - down_result['tau_const'])

print(f"\nHadronic brane (Δk=2 fit):")
print(f"  τ_up = {up_result['tau_const']:.3f}i")
print(f"  τ_down = {down_result['tau_const']:.3f}i")
print(f"  Δτ = {tau_diff:.3f}")
if tau_diff < 1.0:
    print(f"  ✓ Up and down share τ (single hadronic brane)")
    print(f"  <τ_hadronic> = {tau_avg_quarks:.3f}i")
else:
    print(f"  ⚠ Up and down have different τ")

# Compare with leptons
print(f"\nComparison with leptons:")
print(f"  τ_leptonic = {tau_leptons:.3f}i")
print(f"  τ_hadronic = {tau_avg_quarks:.3f}i")
print(f"  Δτ = {abs(tau_avg_quarks - tau_leptons):.3f}")
tau_ratio = tau_avg_quarks / tau_leptons
print(f"  τ_ratio = {tau_ratio:.3f}")

# Final verdict
both_acceptable = up_result['delta_k_2_acceptable'] and down_result['delta_k_2_acceptable']
both_emerges = up_result['delta_k_2_emerges'] and down_result['delta_k_2_emerges']
both_good_fit = up_result['chi2_const'] < 10 and down_result['chi2_const'] < 10

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if both_emerges:
    print("\n🎯 ΔKEY=2 IS UNIVERSAL ✓✓✓")
    print("\nΔk=2 EMERGES NATURALLY for both sectors:")
    print(f"  • Leptons (τ={tau_leptons:.2f}i): Δk=2 ✓")
    print(f"  • Quarks (τ={tau_avg_quarks:.2f}i): Δk≈2 ✓")
    print("\nThis is FUNDAMENTAL PATTERN!")
elif both_acceptable:
    print("\n✓ ΔKEY=2 IS UNIVERSAL (with modest cost)")
    print("\nΔk=2 acceptable (Δχ²<3.84) for both sectors:")
    print(f"  • Leptons (τ={tau_leptons:.2f}i): Δk=2 ✓")
    print(f"  • Quarks (τ={tau_avg_quarks:.2f}i): Δk=2 acceptable ✓")
    print(f"  • Up-type: Δχ²={up_result['delta_chi2']:.2f}")
    print(f"  • Down-type: Δχ²={down_result['delta_chi2']:.2f}")

    if both_good_fit:
        print(f"\nFit quality: GOOD (χ²<10)")
        print("→ MULTI-BRANE SCENARIO SUPPORTED")
        print(f"→ Geometric gauge group split:")
        print(f"   • SU(3) color at τ≈{tau_avg_quarks:.2f}i")
        print(f"   • SU(2)×U(1) weak at τ={tau_leptons:.2f}i")
        print(f"→ Universal Δk=2 on both branes")
    else:
        print(f"\nFit quality: MARGINAL (χ²>10)")
        print(f"  Up-type: χ²={up_result['chi2_const']:.1f}")
        print(f"  Down-type: χ²={down_result['chi2_const']:.1f}")
        print("→ Δk=2 universal, but quantitative fit needs work")
else:
    print("\n⚠ ΔKEY=2 NOT UNIVERSAL")
    if not up_result['delta_k_2_acceptable']:
        print(f"  Up-type: Δχ²={up_result['delta_chi2']:.1f} >> 3.84 (rejected)")
    if not down_result['delta_k_2_acceptable']:
        print(f"  Down-type: Δχ²={down_result['delta_chi2']:.1f} >> 3.84 (rejected)")
    print("\n→ Framework incomplete")
    print("→ Δk=2 is leptonic feature, not fundamental")# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Up-type fit
ax = axes[0, 0]
x_pos = np.arange(3)
width = 0.35

ax.bar(x_pos - width/2, quark_masses_up, width, label='Observed',
       color='gray', alpha=0.6, edgecolor='black')
ax.bar(x_pos + width/2, up_result['masses_fit'], width, label='Fitted',
       color='red', alpha=0.7, edgecolor='black')

ax.set_yscale('log')
ax.set_ylabel('Mass [GeV]', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(['u', 'c', 't'])
ax.set_title(f"Up-type: τ={up_result['tau']:.2f}i, k={up_result['k_rounded']}, Δk={up_result['delta_k_avg']:.2f}",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Down-type fit
ax = axes[0, 1]
ax.bar(x_pos - width/2, quark_masses_down, width, label='Observed',
       color='gray', alpha=0.6, edgecolor='black')
ax.bar(x_pos + width/2, down_result['masses_fit'], width, label='Fitted',
       color='darkred', alpha=0.7, edgecolor='black')

ax.set_yscale('log')
ax.set_ylabel('Mass [GeV]', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(['d', 's', 'b'])
ax.set_title(f"Down-type: τ={down_result['tau']:.2f}i, k={down_result['k_rounded']}, Δk={down_result['delta_k_avg']:.2f}",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 3: τ comparison
ax = axes[1, 0]
sectors = ['Charged\nleptons', 'Neutral\nleptons', 'Up-type\nquarks', 'Down-type\nquarks']
tau_values = [tau_leptons, tau_leptons, up_result['tau'], down_result['tau']]
delta_k_values = [2, 2, up_result['delta_k_avg'], down_result['delta_k_avg']]
colors = ['blue', 'lightblue', 'red', 'darkred']

bars = ax.bar(sectors, tau_values, color=colors, alpha=0.7, edgecolor='black')

for bar, dk in zip(bars, delta_k_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
            f"Δk={dk:.1f}", ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('τ (imaginary part)', fontsize=12)
ax.set_title('Modular Parameters Across Sectors', fontsize=13, fontweight='bold')
ax.axhline(y=tau_leptons, color='blue', linestyle='--', alpha=0.5, label='Leptonic')
ax.axhline(y=tau_avg_quarks, color='red', linestyle='--', alpha=0.5, label='Hadronic')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Δk universality
ax = axes[1, 1]
sectors_dk = ['Charged\nleptons', 'Neutral\nleptons', 'Up\nquarks', 'Down\nquarks']
delta_k_plot = [2, 2, up_result['delta_k_avg'], down_result['delta_k_avg']]
colors_dk = ['blue', 'lightblue', 'red', 'darkred']

bars = ax.bar(sectors_dk, delta_k_plot, color=colors_dk, alpha=0.7, edgecolor='black')

# Add universality indicators
for i, (bar, dk) in enumerate(zip(bars, delta_k_plot)):
    height = bar.get_height()
    if abs(dk - 2.0) < 0.3:
        marker = "✓"
        color = 'green'
    else:
        marker = "✗"
        color = 'red'
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
            marker, ha='center', fontsize=16, fontweight='bold', color=color)

ax.axhline(y=2.0, color='black', linestyle='--', linewidth=2, label='Δk=2')
ax.set_ylabel('Δk (average)', fontsize=12)
ax.set_title('Δk=2 Universality Test', fontsize=13, fontweight='bold')
ax.set_ylim([0, max(delta_k_plot) * 1.3])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('delta_k_universality_test.png', dpi=300, bbox_inches='tight')
plt.savefig('delta_k_universality_test.pdf', bbox_inches='tight')
print("\n✓ Figures saved: delta_k_universality_test.png/pdf")

# Save results
results = {
    'leptonic': {
        'tau': float(tau_leptons),
        'k_charged': k_charged.tolist(),
        'k_neutral': k_neutral.tolist(),
        'delta_k': 2
    },
    'hadronic': {
        'tau_up': float(up_result['tau']),
        'tau_down': float(down_result['tau']),
        'tau_average': float(tau_avg_quarks),
        'k_up': [int(x) for x in np.round(up_result['k_const'])],
        'k_down': [int(x) for x in np.round(down_result['k_const'])],
        'chi2_up': float(up_result['chi2_const']),
        'chi2_down': float(down_result['chi2_const']),
        'delta_chi2_up': float(up_result['delta_chi2']),
        'delta_chi2_down': float(down_result['delta_chi2']),
        'p_value_up': float(up_result['p_value']),
        'p_value_down': float(down_result['p_value'])
    },
    'universality': {
        'delta_k_2_emerges': bool(both_emerges),
        'delta_k_2_acceptable': bool(both_acceptable),
        'fit_quality_good': bool(both_good_fit),
        'tau_ratio': float(tau_ratio),
        'tau_difference': float(abs(tau_avg_quarks - tau_leptons)),
        'multi_brane_supported': bool(both_acceptable and (tau_diff < 1.0))
    }
}

with open('delta_k_universality_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved: delta_k_universality_results.json")

print("\n" + "="*70)
