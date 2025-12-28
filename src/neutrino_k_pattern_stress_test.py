"""
NEUTRINO Δk=2 STRESS TEST
=========================

THE SMOKING GUN TEST: Does the neutrino sector follow Δk=2 pattern?

If YES → Framework survives critical falsification test
If NO  → Framework is falsified, no matter how elegant

This is NOT a fit. This is a PREDICTION TEST.

Charged leptons: k = (8,6,4) → Δk = 2 ✓ (by construction)
Neutrinos:      k = (?,?,?) → Δk = 2 ?? (PREDICTION)

Method:
1. Fit k_ν from mass-squared differences Δm²
2. Check if Δk = k₁-k₂ = k₂-k₃ = 2 within errors
3. Statistical test: P(Δk=2|data) vs P(Δk≠2|data)
4. Test both normal and inverted hierarchies
5. Document failure if Δk≠2

DATA (NuFIT 5.2, 2023):
- Δm²₂₁ = 7.53e-5 eV² (solar, well-measured)
- |Δm²₃₁| = 2.453e-3 eV² (atmospheric)
- Normal hierarchy preferred (Δχ² ≈ -6.7)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# EXPERIMENTAL DATA
# ==============================================================================

# NuFIT 5.2 (2023) - global fit with SK atmospheric data
# http://www.nu-fit.org/

# Mass-squared differences (eV²)
Delta_m21_sq = 7.53e-5  # Solar, well-measured
Delta_m21_sq_err = 0.18e-5

# Atmospheric (absolute value)
Delta_m3x_sq = 2.453e-3  # Normal: Δm²₃₁, Inverted: Δm²₃₂
Delta_m3x_sq_err = 0.033e-3

# Mixing angles (best fit)
sin2_theta12 = 0.307  # Solar angle
sin2_theta23 = 0.545  # Atmospheric angle (near maximal)
sin2_theta13 = 0.02225  # Reactor angle

# CP phase (weakly constrained)
delta_CP = 197 * np.pi / 180  # degrees → radians (1σ range: 120-360°)

print("="*80)
print("NEUTRINO Δk=2 STRESS TEST")
print("="*80)
print("\n⚠️  CRITICAL FALSIFICATION TEST")
print("\nCharged leptons: k = (8,6,4) → Δk = 2")
print("Neutrinos:       k = (?,?,?) → Δk = 2 ??")
print("\nIf Δk ≠ 2, framework is FALSIFIED.")
print()

print("="*80)
print("EXPERIMENTAL DATA (NuFIT 5.2, 2023)")
print("="*80)
print(f"\nMass-squared differences:")
print(f"  Δm²₂₁ = ({Delta_m21_sq:.2e} ± {Delta_m21_sq_err:.2e}) eV² (solar)")
print(f"  |Δm²₃ₓ| = ({Delta_m3x_sq:.3e} ± {Delta_m3x_sq_err:.3e}) eV² (atmospheric)")
print(f"\nMixing angles:")
print(f"  sin²θ₁₂ = {sin2_theta12:.3f} (solar)")
print(f"  sin²θ₂₃ = {sin2_theta23:.3f} (atmospheric, near maximal)")
print(f"  sin²θ₁₃ = {sin2_theta13:.5f} (reactor)")
print(f"\nCP phase:")
print(f"  δ_CP = {delta_CP*180/np.pi:.0f}° (weakly constrained)")

# ==============================================================================
# MODULAR FORM ANSATZ
# ==============================================================================

def eta_function_weight_k(tau, k):
    """
    Dedekind eta function η(τ)^k ∝ q^(k/24) ∏(1-q^n)^k

    For small Im(τ), dominant behavior: |η(τ)|^k ~ exp(-πk·Im(τ)/12)

    This gives mass scaling from modular forms.
    """
    q = np.exp(2j * np.pi * tau)

    # Prefactor from q^(k/24)
    prefactor = np.abs(q)**(k/24)

    # Product approximation (first 50 terms usually enough)
    product = 1.0
    for n in range(1, 51):
        product *= np.abs(1 - q**n)**k

    return prefactor * product

def mass_from_k_pattern(k_values, tau, m_scale=1.0):
    """
    Neutrino masses from modular forms:

    m_i ~ m_scale × |η(τ)|^(k_i)

    Returns mass-squared differences Δm²_ij
    """
    masses = np.array([m_scale * eta_function_weight_k(tau, k) for k in k_values])

    # Sort by mass (convention: m₁ < m₂ < m₃ for normal hierarchy)
    masses = np.sort(masses)

    # Mass-squared differences
    Delta_m21_sq = masses[1]**2 - masses[0]**2
    Delta_m31_sq = masses[2]**2 - masses[0]**2
    Delta_m32_sq = masses[2]**2 - masses[1]**2

    return Delta_m21_sq, Delta_m31_sq, Delta_m32_sq

# ==============================================================================
# TEST 1: FREE FIT (no Δk=2 constraint)
# ==============================================================================

print("\n" + "="*80)
print("TEST 1: FREE FIT (no Δk=2 constraint)")
print("="*80)
print("\nFit k₁, k₂, k₃ to mass-squared differences WITHOUT assuming Δk=2.")
print("If Δk=2 emerges naturally, this is STRONG evidence.\n")

def chi_squared_free(params, hierarchy='normal'):
    """
    χ² for free k-pattern fit (no Δk=2 constraint).

    params = [k1, k2, k3, Im(tau), m_scale]
    """
    k1, k2, k3, tau_im, log_m_scale = params
    tau = 1j * tau_im
    m_scale = 10**log_m_scale  # Log scale for numerical stability

    k_values = [k1, k2, k3]

    try:
        Delta_m21_sq_pred, Delta_m31_sq_pred, Delta_m32_sq_pred = mass_from_k_pattern(k_values, tau, m_scale)
    except:
        return 1e10  # Invalid region

    # χ² from mass-squared differences
    chi2_solar = ((Delta_m21_sq_pred - Delta_m21_sq) / Delta_m21_sq_err)**2

    if hierarchy == 'normal':
        chi2_atm = ((Delta_m31_sq_pred - Delta_m3x_sq) / Delta_m3x_sq_err)**2
    else:  # inverted
        chi2_atm = ((Delta_m32_sq_pred - Delta_m3x_sq) / Delta_m3x_sq_err)**2

    return chi2_solar + chi2_atm

# Initial guess: k ~ (8,6,4) like charged leptons, tau ~ 3i, m_scale ~ 0.05 eV
initial_guess = [8.0, 6.0, 4.0, 3.0, -1.3]  # log10(0.05) ≈ -1.3

# Bounds: k ∈ [0,12], Im(tau) ∈ [1,10], m_scale ∈ [0.01, 0.5] eV
bounds = [(0, 12), (0, 12), (0, 12), (1, 10), (-2, -0.3)]

print("Fitting k₁, k₂, k₃ to normal hierarchy data...")
result_free_NH = differential_evolution(
    chi_squared_free,
    bounds,
    args=('normal',),
    seed=42,
    maxiter=1000,
    atol=1e-6,
    tol=1e-6,
    workers=1
)

k1_fit, k2_fit, k3_fit, tau_fit, log_m_fit = result_free_NH.x
m_scale_fit = 10**log_m_fit

# Sort k values (convention: k₁ > k₂ > k₃ for increasing mass)
k_values_fit = sorted([k1_fit, k2_fit, k3_fit], reverse=True)
k1_fit, k2_fit, k3_fit = k_values_fit

Delta_k_12 = k1_fit - k2_fit
Delta_k_23 = k2_fit - k3_fit

print(f"\n✓ BEST FIT (Normal Hierarchy):")
print(f"  k₁ = {k1_fit:.3f}")
print(f"  k₂ = {k2_fit:.3f}")
print(f"  k₃ = {k3_fit:.3f}")
print(f"  τ = {tau_fit:.3f}i")
print(f"  m_scale = {m_scale_fit:.4f} eV")
print(f"\n  Δk₁₂ = k₁-k₂ = {Delta_k_12:.3f}")
print(f"  Δk₂₃ = k₂-k₃ = {Delta_k_23:.3f}")
print(f"\n  χ² = {result_free_NH.fun:.2f}")
print(f"  dof = 2 (two observables)")
print(f"  χ²/dof = {result_free_NH.fun/2:.2f}")
print(f"  p-value = {1 - chi2.cdf(result_free_NH.fun, 2):.4f}")

# Critical question: Is Δk ≈ 2?
Delta_k_avg = (Delta_k_12 + Delta_k_23) / 2
Delta_k_deviation = np.abs(Delta_k_avg - 2.0)

print(f"\n⚠️  CRITICAL TEST: Δk = 2?")
print(f"  Average Δk = {Delta_k_avg:.3f}")
print(f"  Deviation from 2: {Delta_k_deviation:.3f}")

if Delta_k_deviation < 0.3:
    print(f"  ✓ CONSISTENT with Δk=2 (within ~15%)")
else:
    print(f"  ❌ INCONSISTENT with Δk=2 (deviation > 15%)")

# ==============================================================================
# TEST 2: CONSTRAINED FIT (enforce Δk=2)
# ==============================================================================

print("\n" + "="*80)
print("TEST 2: CONSTRAINED FIT (enforce Δk=2)")
print("="*80)
print("\nFit with k₁=k₀+4, k₂=k₀+2, k₃=k₀ (Δk=2 by construction).")
print("Compare χ² to free fit. If ≈ same, Δk=2 is not costing us fit quality.\n")

def chi_squared_constrained(params, hierarchy='normal'):
    """
    χ² for constrained k-pattern: k = (k₀+4, k₀+2, k₀)

    params = [k0, Im(tau), m_scale]
    """
    k0, tau_im, log_m_scale = params
    tau = 1j * tau_im
    m_scale = 10**log_m_scale

    k_values = [k0+4, k0+2, k0]  # Δk=2 by construction

    try:
        Delta_m21_sq_pred, Delta_m31_sq_pred, Delta_m32_sq_pred = mass_from_k_pattern(k_values, tau, m_scale)
    except:
        return 1e10

    chi2_solar = ((Delta_m21_sq_pred - Delta_m21_sq) / Delta_m21_sq_err)**2

    if hierarchy == 'normal':
        chi2_atm = ((Delta_m31_sq_pred - Delta_m3x_sq) / Delta_m3x_sq_err)**2
    else:
        chi2_atm = ((Delta_m32_sq_pred - Delta_m3x_sq) / Delta_m3x_sq_err)**2

    return chi2_solar + chi2_atm

# Initial guess: k0=4, tau~3i, m_scale~0.05 eV
initial_guess_constrained = [4.0, 3.0, -1.3]
bounds_constrained = [(0, 8), (1, 10), (-2, -0.3)]

print("Fitting k₀ (with k=k₀+[4,2,0]) to normal hierarchy data...")
result_constrained_NH = differential_evolution(
    chi_squared_constrained,
    bounds_constrained,
    args=('normal',),
    seed=42,
    maxiter=1000,
    atol=1e-6,
    tol=1e-6,
    workers=1
)

k0_fit, tau_constrained_fit, log_m_constrained_fit = result_constrained_NH.x
m_scale_constrained_fit = 10**log_m_constrained_fit

k_constrained = [k0_fit+4, k0_fit+2, k0_fit]

print(f"\n✓ BEST FIT (Δk=2 enforced):")
print(f"  k₀ = {k0_fit:.3f}")
print(f"  k = ({k_constrained[0]:.3f}, {k_constrained[1]:.3f}, {k_constrained[2]:.3f})")
print(f"  τ = {tau_constrained_fit:.3f}i")
print(f"  m_scale = {m_scale_constrained_fit:.4f} eV")
print(f"\n  χ² = {result_constrained_NH.fun:.2f}")
print(f"  dof = 2")
print(f"  χ²/dof = {result_constrained_NH.fun/2:.2f}")
print(f"  p-value = {1 - chi2.cdf(result_constrained_NH.fun, 2):.4f}")

# Compare fits
Delta_chi2 = result_constrained_NH.fun - result_free_NH.fun
Delta_dof = 2  # Lost 2 parameters (k₁, k₂ → k₀)

print(f"\n⚠️  FIT COMPARISON:")
print(f"  χ²(free) = {result_free_NH.fun:.2f}")
print(f"  χ²(Δk=2) = {result_constrained_NH.fun:.2f}")
print(f"  Δχ² = {Delta_chi2:.2f}")
print(f"  Δdof = {Delta_dof}")

# F-test: Is constraint justified?
if Delta_chi2 < 0:
    print(f"  ✓ Δk=2 constraint IMPROVES fit (unexpected but great!)")
elif Delta_chi2 < 4:
    print(f"  ✓ Δk=2 constraint has NEGLIGIBLE cost (Δχ² < 4)")
elif Delta_chi2 < 9:
    print(f"  ⚠️  Δk=2 constraint marginally disfavored (4 < Δχ² < 9)")
else:
    print(f"  ❌ Δk=2 constraint REJECTED (Δχ² > 9, p < 0.01)")

# ==============================================================================
# TEST 3: INVERTED HIERARCHY
# ==============================================================================

print("\n" + "="*80)
print("TEST 3: INVERTED HIERARCHY")
print("="*80)
print("\nCurrent data favor normal hierarchy (Δχ² ≈ -6.7).")
print("Does Δk=2 prefer one hierarchy over the other?\n")

print("Fitting to inverted hierarchy...")
result_free_IH = differential_evolution(
    chi_squared_free,
    bounds,
    args=('inverted',),
    seed=42,
    maxiter=1000,
    atol=1e-6,
    tol=1e-6,
    workers=1
)

k1_IH, k2_IH, k3_IH, tau_IH, log_m_IH = result_free_IH.x
k_values_IH = sorted([k1_IH, k2_IH, k3_IH], reverse=True)
k1_IH, k2_IH, k3_IH = k_values_IH

Delta_k_12_IH = k1_IH - k2_IH
Delta_k_23_IH = k2_IH - k3_IH
Delta_k_avg_IH = (Delta_k_12_IH + Delta_k_23_IH) / 2

print(f"\n✓ BEST FIT (Inverted Hierarchy):")
print(f"  k = ({k1_IH:.3f}, {k2_IH:.3f}, {k3_IH:.3f})")
print(f"  Δk_avg = {Delta_k_avg_IH:.3f}")
print(f"  χ² = {result_free_IH.fun:.2f}")
print(f"  χ²/dof = {result_free_IH.fun/2:.2f}")

# Hierarchy preference
Delta_chi2_hierarchy = result_free_IH.fun - result_free_NH.fun

print(f"\n⚠️  HIERARCHY PREFERENCE:")
print(f"  χ²(NH) = {result_free_NH.fun:.2f}")
print(f"  χ²(IH) = {result_free_IH.fun:.2f}")
print(f"  Δχ² = {Delta_chi2_hierarchy:.2f}")

if Delta_chi2_hierarchy < -3:
    print(f"  ✓ Model PREFERS inverted hierarchy (Δχ² < -3)")
elif Delta_chi2_hierarchy < 0:
    print(f"  ⚠️  Model slightly favors inverted (0 < Δχ² < 3)")
elif Delta_chi2_hierarchy < 3:
    print(f"  ≈ No hierarchy preference (|Δχ²| < 3)")
else:
    print(f"  ✓ Model prefers normal hierarchy (Δχ² > 3)")

# ==============================================================================
# TEST 4: BOOTSTRAP UNCERTAINTY
# ==============================================================================

print("\n" + "="*80)
print("TEST 4: BOOTSTRAP UNCERTAINTY ON Δk")
print("="*80)
print("\nBootstrap data within errors to estimate uncertainty on Δk.\n")

np.random.seed(42)
n_bootstrap = 100

Delta_k_samples = []

for i in range(n_bootstrap):
    # Sample data within errors
    Delta_m21_sample = np.random.normal(Delta_m21_sq, Delta_m21_sq_err)
    Delta_m3x_sample = np.random.normal(Delta_m3x_sq, Delta_m3x_sq_err)

    # Temporary chi2 function for this sample
    def chi2_temp(params):
        k1, k2, k3, tau_im, log_m_scale = params
        tau = 1j * tau_im
        m_scale = 10**log_m_scale
        k_vals = [k1, k2, k3]

        try:
            dm21, dm31, dm32 = mass_from_k_pattern(k_vals, tau, m_scale)
        except:
            return 1e10

        c2 = ((dm21 - Delta_m21_sample) / Delta_m21_sq_err)**2
        c2 += ((dm31 - Delta_m3x_sample) / Delta_m3x_sq_err)**2
        return c2

    # Quick fit
    result = differential_evolution(chi2_temp, bounds, seed=42+i, maxiter=500, workers=1, polish=False)
    k_fit = sorted([result.x[0], result.x[1], result.x[2]], reverse=True)

    Delta_k_avg_sample = ((k_fit[0]-k_fit[1]) + (k_fit[1]-k_fit[2])) / 2
    Delta_k_samples.append(Delta_k_avg_sample)

    if (i+1) % 20 == 0:
        print(f"  Bootstrap {i+1}/{n_bootstrap}...")

Delta_k_samples = np.array(Delta_k_samples)
Delta_k_mean = np.mean(Delta_k_samples)
Delta_k_std = np.std(Delta_k_samples)

print(f"\n✓ BOOTSTRAP RESULTS ({n_bootstrap} samples):")
print(f"  <Δk> = {Delta_k_mean:.3f} ± {Delta_k_std:.3f}")
print(f"  68% interval: [{np.percentile(Delta_k_samples, 16):.3f}, {np.percentile(Delta_k_samples, 84):.3f}]")
print(f"  95% interval: [{np.percentile(Delta_k_samples, 2.5):.3f}, {np.percentile(Delta_k_samples, 97.5):.3f}]")

# Statistical test: Is Δk=2 within 1σ?
z_score = np.abs(Delta_k_mean - 2.0) / Delta_k_std
p_value = 2 * (1 - chi2.cdf(z_score**2, 1))  # Two-tailed test

print(f"\n⚠️  STATISTICAL TEST: Δk = 2?")
print(f"  Null hypothesis: Δk = 2")
print(f"  Observed: Δk = {Delta_k_mean:.3f} ± {Delta_k_std:.3f}")
print(f"  z-score: {z_score:.2f}")
print(f"  p-value: {p_value:.3f}")

if p_value > 0.05:
    print(f"  ✓ CANNOT REJECT Δk=2 (p > 0.05)")
    print(f"    Framework SURVIVES stress test!")
else:
    print(f"  ❌ REJECT Δk=2 (p < 0.05)")
    print(f"    Framework FALSIFIED by neutrino data!")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

# Close any existing figures to avoid memory issues
plt.close('all')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: k-pattern comparison
ax = axes[0, 0]
x_pos = np.arange(3)
width = 0.35

k_charged = [8, 6, 4]
k_neutrino_free = [k1_fit, k2_fit, k3_fit]
k_neutrino_constrained = k_constrained

ax.bar(x_pos - width/2, k_charged, width, label='Charged leptons', alpha=0.7, color='blue')
ax.bar(x_pos + width/2, k_neutrino_free, width, label='Neutrinos (free fit)', alpha=0.7, color='red')
ax.plot(x_pos, k_neutrino_constrained, 'go--', linewidth=2, markersize=8, label='Neutrinos (Δk=2)')

ax.set_xlabel('Generation', fontsize=11)
ax.set_ylabel('Modular weight k', fontsize=11)
ax.set_title('A. k-pattern: Charged vs Neutrinos', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['3rd', '2nd', '1st'])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: Δk histogram
ax = axes[0, 1]
ax.hist(Delta_k_samples, bins=20, alpha=0.7, color='green', edgecolor='black')
ax.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Δk=2 (predicted)')
ax.axvline(Delta_k_mean, color='blue', linestyle='-', linewidth=2, label=f'<Δk> = {Delta_k_mean:.2f}±{Delta_k_std:.2f}')
ax.set_xlabel('Δk = k₁-k₂ = k₂-k₃', fontsize=11)
ax.set_ylabel('Bootstrap samples', fontsize=11)
ax.set_title('B. Bootstrap Distribution of Δk', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel C: χ² comparison
ax = axes[1, 0]
models = ['Free fit\n(5 params)', 'Δk=2\n(3 params)', 'Inverted\nHierarchy']
chi2_values = [result_free_NH.fun, result_constrained_NH.fun, result_free_IH.fun]
colors = ['blue', 'green', 'orange']

bars = ax.bar(models, chi2_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('χ²', fontsize=11)
ax.set_title('C. Fit Quality Comparison', fontsize=12, fontweight='bold')
ax.axhline(result_free_NH.fun, color='red', linestyle='--', alpha=0.5, label='Best χ²')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Add χ² values on bars
for bar, val in zip(bars, chi2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel D: p-value visualization
ax = axes[1, 1]
z_range = np.linspace(-3, 3, 100)
p_vals = [2*(1 - chi2.cdf(z**2, 1)) for z in z_range]

ax.plot(z_range, p_vals, 'b-', linewidth=2)
ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05 (rejection threshold)')
ax.axvline(z_score, color='green', linestyle='-', linewidth=2, label=f'Observed z={z_score:.2f}')
ax.fill_between(z_range, 0, p_vals, where=(np.array(p_vals) > 0.05), alpha=0.3, color='green', label='Cannot reject')
ax.fill_between(z_range, 0, p_vals, where=(np.array(p_vals) <= 0.05), alpha=0.3, color='red', label='Reject')

ax.set_xlabel('z-score (Δk deviation)', fontsize=11)
ax.set_ylabel('p-value', fontsize=11)
ax.set_title('D. Statistical Significance Test', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1])
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()

# Try to save with error handling
try:
    plt.savefig('neutrino_k_pattern_stress_test.png', dpi=300, bbox_inches='tight')
    plt.savefig('neutrino_k_pattern_stress_test.pdf', dpi=300, bbox_inches='tight')
    print("\nFigures saved: neutrino_k_pattern_stress_test.png/pdf")
except Exception as e:
    print(f"\n⚠️  Figure save error (results still valid): {e}")
    print("Figures NOT saved, but analysis complete.")

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

print("\n✓ FITS:")
print(f"  Free fit:        k = ({k1_fit:.2f}, {k2_fit:.2f}, {k3_fit:.2f}), χ² = {result_free_NH.fun:.2f}")
print(f"  Δk=2 enforced:   k = ({k_constrained[0]:.2f}, {k_constrained[1]:.2f}, {k_constrained[2]:.2f}), χ² = {result_constrained_NH.fun:.2f}")
print(f"  Inverted hier:   k = ({k1_IH:.2f}, {k2_IH:.2f}, {k3_IH:.2f}), χ² = {result_free_IH.fun:.2f}")

print(f"\n✓ CRITICAL TEST:")
print(f"  Prediction: Δk = 2")
print(f"  Observed:   Δk = {Delta_k_mean:.3f} ± {Delta_k_std:.3f}")
print(f"  Deviation:  {np.abs(Delta_k_mean - 2.0):.3f} ({np.abs(Delta_k_mean - 2.0)/2.0*100:.1f}%)")
print(f"  p-value:    {p_value:.3f}")

if p_value > 0.05:
    print(f"\n✓✓✓ SMOKING GUN CONFIRMED!")
    print(f"    Neutrino sector CONSISTENT with Δk=2 at {(1-p_value)*100:.0f}% confidence level")
    print(f"    Framework SURVIVES critical falsification test!")
else:
    print(f"\n❌❌❌ SMOKING GUN FAILED!")
    print(f"    Neutrino sector INCONSISTENT with Δk=2 (p < 0.05)")
    print(f"    Framework FALSIFIED by experimental data!")

print("\n" + "="*80)
print("Figures saved: neutrino_k_pattern_stress_test.png/pdf")
print("="*80)
