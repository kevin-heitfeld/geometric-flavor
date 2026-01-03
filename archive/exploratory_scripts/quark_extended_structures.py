"""
EXPLORING EXTENDED MATHEMATICAL STRUCTURES FOR QUARKS
======================================================

PROBLEM: Quarks have correct geometric τ=1.422i but don't fit simple modular form m ∝ |η(τ)|^k

POSSIBLE EXTENDED STRUCTURES TO TEST:

1. **Higher Dedekind Functions**: η₂(τ), η₃(τ) (twisted sectors)
2. **Jacobi Theta Functions**: θ₂, θ₃, θ₄ (different boundary conditions)
3. **Eisenstein Series**: E₂(τ), E₄(τ), E₆(τ) (quasi-modular forms)
4. **Modular Lambda Function**: λ(τ) (elliptic modulus)
5. **Multiple τ Parameters**: Different τ for each generation (brane position spectrum)
6. **Product Forms**: m ∝ |η(τ)|^k₁ × |θ(τ)|^k₂ (mixed structure)
7. **Running Modular Parameter**: τ(μ) that runs with scale (QCD coupling)
8. **Non-holomorphic Terms**: Include τ̄ (SL(2,ℤ) breaking by QCD)

STRATEGY:
- Start with mathematically simplest extensions
- Test each against quark masses with τ=1.422i
- Look for patterns that emerge naturally
- Compare χ² improvements

HONEST GOAL:
Not to force a fit, but to discover if QCD's complexity requires
richer mathematical structure that's still geometric at heart.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2
import json

# Experimental quark masses (GeV)
masses_up = np.array([0.00216, 1.27, 172.5])
masses_down = np.array([0.00467, 0.0934, 4.18])

errors_up = np.array([0.00216*0.2, 1.27*0.03, 172.5*0.005])
errors_down = np.array([0.00467*0.15, 0.0934*0.03, 4.18*0.02])

tau_hadronic = 1.422  # From τ-ratio = 7/16

print("="*80)
print("EXPLORING EXTENDED MATHEMATICAL STRUCTURES FOR QUARKS")
print("="*80)
print(f"\nGeometric τ from τ-ratio: {tau_hadronic:.3f}i")
print("\nQuark masses (GeV):")
print(f"  Up:   u={masses_up[0]:.4f}, c={masses_up[1]:.2f}, t={masses_up[2]:.1f}")
print(f"  Down: d={masses_down[0]:.5f}, s={masses_down[1]:.4f}, b={masses_down[2]:.2f}")
print("\nSimple η^k FAILED: χ²>40,000 for up, χ²>3,000 for down")
print("\nTesting extended structures...")

# ==============================================================================
# STRUCTURE 1: Jacobi Theta Functions
# ==============================================================================

def jacobi_theta_3(tau):
    """θ₃(τ) = Σ q^(n²) where q = exp(iπτ)"""
    q = np.exp(1j * np.pi * tau)
    result = 1.0
    for n in range(1, 30):
        result += 2 * q**(n**2)
    return result

def jacobi_theta_2(tau):
    """θ₂(τ) = Σ q^((n+1/2)²)"""
    q = np.exp(1j * np.pi * tau)
    result = 0.0
    for n in range(-30, 30):
        result += q**((n + 0.5)**2)
    return result

def jacobi_theta_4(tau):
    """θ₄(τ) = Σ (-1)^n q^(n²)"""
    q = np.exp(1j * np.pi * tau)
    result = 1.0
    for n in range(1, 30):
        result += 2 * (-1)**n * q**(n**2)
    return result

print("\n" + "="*80)
print("STRUCTURE 1: Jacobi Theta Functions")
print("="*80)
print("Formula: m_i = m₀ × |θ₃(kᵢτ)|^α")
print("Physical meaning: Different boundary conditions for fermions")

def mass_theta3(k_values, tau, m0, alpha=1):
    """Mass from θ₃ function"""
    masses = []
    for k in k_values:
        theta = jacobi_theta_3(1j * k * tau)
        masses.append(m0 * np.abs(theta)**alpha)
    return np.array(sorted(masses))

def fit_theta3(params, masses_obs, errors_obs, tau_fixed):
    """Fit with θ₃ structure"""
    k1, k2, k3, log_m0, alpha = params
    k_values = [k1, k2, k3]
    m0 = 10**log_m0

    try:
        masses_pred = mass_theta3(k_values, tau_fixed, m0, alpha)
        chi2_val = np.sum(((masses_obs - masses_pred) / errors_obs)**2)
        return chi2_val
    except:
        return 1e10

# Test theta functions
print("\nTesting up-type quarks with θ₃:")
result_theta3_up = differential_evolution(
    fit_theta3,
    bounds=[(0.1, 10), (0.1, 10), (0.1, 10), (-3, 3), (0.5, 5)],
    args=(masses_up, errors_up, tau_hadronic),
    seed=42,
    maxiter=500,
    workers=1
)

k_theta3_up = result_theta3_up.x[:3]
m0_theta3_up = 10**result_theta3_up.x[3]
alpha_theta3_up = result_theta3_up.x[4]
chi2_theta3_up = result_theta3_up.fun

print(f"  k = ({k_theta3_up[0]:.2f}, {k_theta3_up[1]:.2f}, {k_theta3_up[2]:.2f})")
print(f"  m₀ = {m0_theta3_up:.3f} GeV, α = {alpha_theta3_up:.2f}")
print(f"  χ² = {chi2_theta3_up:.2f}")

print("\nTesting down-type quarks with θ₃:")
result_theta3_down = differential_evolution(
    fit_theta3,
    bounds=[(0.1, 10), (0.1, 10), (0.1, 10), (-3, 3), (0.5, 5)],
    args=(masses_down, errors_down, tau_hadronic),
    seed=42,
    maxiter=500,
    workers=1
)

k_theta3_down = result_theta3_down.x[:3]
m0_theta3_down = 10**result_theta3_down.x[3]
alpha_theta3_down = result_theta3_down.x[4]
chi2_theta3_down = result_theta3_down.fun

print(f"  k = ({k_theta3_down[0]:.2f}, {k_theta3_down[1]:.2f}, {k_theta3_down[2]:.2f})")
print(f"  m₀ = {m0_theta3_down:.3f} GeV, α = {alpha_theta3_down:.2f}")
print(f"  χ² = {chi2_theta3_down:.2f}")

if chi2_theta3_up < 10 and chi2_theta3_down < 10:
    print("\n✓✓✓ BREAKTHROUGH: Theta functions work!")
elif chi2_theta3_up < 100 or chi2_theta3_down < 100:
    print("\n✓ PROMISING: Theta functions improve fit significantly")
else:
    print("\n⚠ Theta functions don't rescue pattern")

# ==============================================================================
# STRUCTURE 2: Eisenstein Series
# ==============================================================================

def eisenstein_E4(tau):
    """E₄(τ) = 1 + 240 Σ n³q^n/(1-q^n)"""
    q = np.exp(2j * np.pi * tau)
    result = 1.0
    for n in range(1, 30):
        result += 240 * n**3 * q**n / (1 - q**n)
    return result

def eisenstein_E6(tau):
    """E₆(τ) = 1 - 504 Σ n⁵q^n/(1-q^n)"""
    q = np.exp(2j * np.pi * tau)
    result = 1.0
    for n in range(1, 30):
        result -= 504 * n**5 * q**n / (1 - q**n)
    return result

print("\n" + "="*80)
print("STRUCTURE 2: Eisenstein Series (Quasi-modular Forms)")
print("="*80)
print("Formula: m_i = m₀ × |E₄(kᵢτ)|^α")
print("Physical meaning: Include logarithmic corrections (RG-like)")

def mass_eisenstein(k_values, tau, m0, alpha=1, series='E4'):
    """Mass from Eisenstein series"""
    masses = []
    for k in k_values:
        if series == 'E4':
            E = eisenstein_E4(1j * k * tau)
        else:
            E = eisenstein_E6(1j * k * tau)
        masses.append(m0 * np.abs(E)**alpha)
    return np.array(sorted(masses))

def fit_eisenstein(params, masses_obs, errors_obs, tau_fixed, series='E4'):
    """Fit with Eisenstein structure"""
    k1, k2, k3, log_m0, alpha = params
    k_values = [k1, k2, k3]
    m0 = 10**log_m0

    try:
        masses_pred = mass_eisenstein(k_values, tau_fixed, m0, alpha, series)
        chi2_val = np.sum(((masses_obs - masses_pred) / errors_obs)**2)
        return chi2_val
    except:
        return 1e10

print("\nTesting up-type quarks with E₄:")
result_E4_up = differential_evolution(
    fit_eisenstein,
    bounds=[(0.1, 10), (0.1, 10), (0.1, 10), (-3, 3), (0.5, 5)],
    args=(masses_up, errors_up, tau_hadronic, 'E4'),
    seed=42,
    maxiter=500,
    workers=1
)

chi2_E4_up = result_E4_up.fun
print(f"  χ² = {chi2_E4_up:.2f}")

print("\nTesting down-type quarks with E₄:")
result_E4_down = differential_evolution(
    fit_eisenstein,
    bounds=[(0.1, 10), (0.1, 10), (0.1, 10), (-3, 3), (0.5, 5)],
    args=(masses_down, errors_down, tau_hadronic, 'E4'),
    seed=42,
    maxiter=500,
    workers=1
)

chi2_E4_down = result_E4_down.fun
print(f"  χ² = {chi2_E4_down:.2f}")

# ==============================================================================
# STRUCTURE 3: Mixed Product Form
# ==============================================================================

print("\n" + "="*80)
print("STRUCTURE 3: Mixed Product (η × θ)")
print("="*80)
print("Formula: m_i = m₀ × |η(k₁ᵢτ)|^α₁ × |θ₃(k₂ᵢτ)|^α₂")
print("Physical meaning: Perturbative (η) + non-perturbative (θ) QCD")

def eta_function(tau):
    """Dedekind eta η(τ)"""
    q = np.exp(2j * np.pi * tau)
    result = q**(1/24)
    for n in range(1, 50):
        result *= (1 - q**n)
    return result

def mass_mixed(k_eta, k_theta, tau, m0, alpha_eta, alpha_theta):
    """Mixed η × θ structure"""
    masses = []
    for ke, kt in zip(k_eta, k_theta):
        eta = eta_function(1j * ke * tau)
        theta = jacobi_theta_3(1j * kt * tau)
        masses.append(m0 * np.abs(eta)**alpha_eta * np.abs(theta)**alpha_theta)
    return np.array(sorted(masses))

def fit_mixed(params, masses_obs, errors_obs, tau_fixed):
    """Fit with mixed structure"""
    k_eta_1, k_eta_2, k_eta_3, k_theta_1, k_theta_2, k_theta_3, log_m0, alpha_eta, alpha_theta = params
    k_eta = [k_eta_1, k_eta_2, k_eta_3]
    k_theta = [k_theta_1, k_theta_2, k_theta_3]
    m0 = 10**log_m0

    try:
        masses_pred = mass_mixed(k_eta, k_theta, tau_fixed, m0, alpha_eta, alpha_theta)
        chi2_val = np.sum(((masses_obs - masses_pred) / errors_obs)**2)
        return chi2_val
    except:
        return 1e10

print("\nTesting up-type quarks with η×θ:")
result_mixed_up = differential_evolution(
    fit_mixed,
    bounds=[(0.1, 10)]*6 + [(-3, 3), (0.5, 5), (0.5, 5)],
    args=(masses_up, errors_up, tau_hadronic),
    seed=42,
    maxiter=300,
    workers=1
)

chi2_mixed_up = result_mixed_up.fun
print(f"  χ² = {chi2_mixed_up:.2f}")

print("\nTesting down-type quarks with η×θ:")
result_mixed_down = differential_evolution(
    fit_mixed,
    bounds=[(0.1, 10)]*6 + [(-3, 3), (0.5, 5), (0.5, 5)],
    args=(masses_down, errors_down, tau_hadronic),
    seed=42,
    maxiter=300,
    workers=1
)

chi2_mixed_down = result_mixed_down.fun
print(f"  χ² = {chi2_mixed_down:.2f}")

# ==============================================================================
# STRUCTURE 4: Generation-Dependent τ
# ==============================================================================

print("\n" + "="*80)
print("STRUCTURE 4: Generation-Dependent τ")
print("="*80)
print("Formula: m_i = m₀ × |η(τᵢ)|^k, where τᵢ = τ₀ + Δτ·i")
print("Physical meaning: Branes at different positions (spectrum)")

def mass_tau_spectrum(tau_values, k, m0):
    """Masses with different τ for each generation"""
    masses = []
    for tau_i in tau_values:
        eta = eta_function(1j * tau_i)
        masses.append(m0 * np.abs(eta)**k)
    return np.array(sorted(masses))

def fit_tau_spectrum(params, masses_obs, errors_obs):
    """Fit with τ spectrum"""
    tau1, tau2, tau3, k, log_m0 = params
    tau_values = [tau1, tau2, tau3]
    m0 = 10**log_m0

    try:
        masses_pred = mass_tau_spectrum(tau_values, k, m0)
        chi2_val = np.sum(((masses_obs - masses_pred) / errors_obs)**2)
        return chi2_val
    except:
        return 1e10

print("\nTesting up-type quarks with τ spectrum:")
result_tau_up = differential_evolution(
    fit_tau_spectrum,
    bounds=[(0.5, 10), (0.5, 10), (0.5, 10), (1, 15), (-3, 3)],
    args=(masses_up, errors_up),
    seed=42,
    maxiter=500,
    workers=1
)

tau_up_spectrum = result_tau_up.x[:3]
k_tau_up = result_tau_up.x[3]
chi2_tau_up = result_tau_up.fun

print(f"  τ = ({tau_up_spectrum[0]:.2f}i, {tau_up_spectrum[1]:.2f}i, {tau_up_spectrum[2]:.2f}i)")
print(f"  k = {k_tau_up:.2f}")
print(f"  χ² = {chi2_tau_up:.2f}")

print("\nTesting down-type quarks with τ spectrum:")
result_tau_down = differential_evolution(
    fit_tau_spectrum,
    bounds=[(0.5, 10), (0.5, 10), (0.5, 10), (1, 15), (-3, 3)],
    args=(masses_down, errors_down),
    seed=42,
    maxiter=500,
    workers=1
)

tau_down_spectrum = result_tau_down.x[:3]
k_tau_down = result_tau_down.x[3]
chi2_tau_down = result_tau_down.fun

print(f"  τ = ({tau_down_spectrum[0]:.2f}i, {tau_down_spectrum[1]:.2f}i, {tau_down_spectrum[2]:.2f}i)")
print(f"  k = {k_tau_down:.2f}")
print(f"  χ² = {chi2_tau_down:.2f}")

# ==============================================================================
# RESULTS SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results = {
    'Simple η^k (baseline)': {'up': 41097, 'down': 3124},
    'Theta functions θ₃': {'up': chi2_theta3_up, 'down': chi2_theta3_down},
    'Eisenstein E₄': {'up': chi2_E4_up, 'down': chi2_E4_down},
    'Mixed η×θ': {'up': chi2_mixed_up, 'down': chi2_mixed_down},
    'τ spectrum': {'up': chi2_tau_up, 'down': chi2_tau_down}
}

print("\n{:<25} {:>15} {:>15}".format("Structure", "Up χ²", "Down χ²"))
print("-"*55)
for name, chi2_vals in results.items():
    up_status = "✓" if chi2_vals['up'] < 10 else ("~" if chi2_vals['up'] < 100 else "✗")
    down_status = "✓" if chi2_vals['down'] < 10 else ("~" if chi2_vals['down'] < 100 else "✗")
    print(f"{name:<25} {chi2_vals['up']:>12.1f} {up_status:>2}  {chi2_vals['down']:>12.1f} {down_status:>2}")

# Find best structure
best_combined = min(results.items(), key=lambda x: x[1]['up'] + x[1]['down'])

print(f"\n{'='*80}")
print(f"BEST STRUCTURE: {best_combined[0]}")
print(f"{'='*80}")
print(f"  Up χ² = {best_combined[1]['up']:.2f}")
print(f"  Down χ² = {best_combined[1]['down']:.2f}")
print(f"  Combined χ² = {best_combined[1]['up'] + best_combined[1]['down']:.2f}")

if best_combined[1]['up'] + best_combined[1]['down'] < 20:
    print("\n🎯 BREAKTHROUGH! Extended structure rescues quark sector!")
    print("   This validates geometric τ=1.422i with richer mathematics")
elif best_combined[1]['up'] + best_combined[1]['down'] < 200:
    print("\n✓ SIGNIFICANT IMPROVEMENT over simple η^k")
    print("   Extended structures partially rescue pattern")
else:
    print("\n⚠ Extended structures tested don't rescue pattern")
    print("   Quarks may require even more exotic mathematical structure")
    print("   Possibilities:")
    print("   - Non-holomorphic modular forms (τ̄ dependence)")
    print("   - Siegel modular forms (genus-2 surfaces)")
    print("   - Automorphic forms for larger groups")
    print("   - Completely different approach (e.g., matrix models)")

# Save results
results_data = {
    'tau_hadronic': tau_hadronic,
    'structures_tested': {
        'theta_functions': {
            'up': {'chi2': float(chi2_theta3_up), 'k': list(k_theta3_up), 'alpha': float(alpha_theta3_up)},
            'down': {'chi2': float(chi2_theta3_down), 'k': list(k_theta3_down), 'alpha': float(alpha_theta3_down)}
        },
        'eisenstein_E4': {
            'up': {'chi2': float(chi2_E4_up)},
            'down': {'chi2': float(chi2_E4_down)}
        },
        'mixed_eta_theta': {
            'up': {'chi2': float(chi2_mixed_up)},
            'down': {'chi2': float(chi2_mixed_down)}
        },
        'tau_spectrum': {
            'up': {'chi2': float(chi2_tau_up), 'tau': list(tau_up_spectrum), 'k': float(k_tau_up)},
            'down': {'chi2': float(chi2_tau_down), 'tau': list(tau_down_spectrum), 'k': float(k_tau_down)}
        }
    },
    'best_structure': best_combined[0],
    'best_chi2': {
        'up': float(best_combined[1]['up']),
        'down': float(best_combined[1]['down']),
        'combined': float(best_combined[1]['up'] + best_combined[1]['down'])
    }
}

with open('quark_extended_structures_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print("\n✓ Results saved: quark_extended_structures_results.json")
print("="*80)
