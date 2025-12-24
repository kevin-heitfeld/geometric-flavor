"""
QUARK MASS SCALE SENSITIVITY ANALYSIS
======================================

PROBLEM: Quarks failed to fit with τ=3.25i (need τ≈6i instead)

HYPOTHESIS: We used MS-bar masses at μ=2 GeV, which might be wrong scale.
Quarks run significantly with energy due to QCD.

TEST: Try different mass definitions/scales:
1. MS-bar at various μ (1 GeV, m_q, m_Z, m_t)
2. Pole masses (physical masses, no running)
3. Yukawa couplings at m_Z (Higgs couplings)
4. Constituent quark masses (non-perturbative)

If ANY scale works with τ=3.25i, unification is saved!
If ALL scales fail, quarks genuinely decouple.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==============================================================================
# QUARK MASSES AT DIFFERENT SCALES
# ==============================================================================

# Reference: PDG 2023, Workman et al.

print("="*80)
print("QUARK MASS SCALE SENSITIVITY ANALYSIS")
print("="*80)
print("\nHYPOTHESIS: Different mass scale might fix τ=3.25i fit")
print("\nTesting multiple mass definitions...\n")

# 1. MS-bar masses at μ=2 GeV (what we used)
masses_MSbar_2GeV = {
    'name': 'MS-bar at μ=2 GeV',
    'up': {'u': 0.00216, 'c': 1.27, 't': 172.5},  # GeV
    'down': {'d': 0.00467, 's': 0.0934, 'b': 4.18},
    'errors': {
        'u': 0.00216*0.2, 'c': 1.27*0.03, 't': 172.5*0.005,
        'd': 0.00467*0.15, 's': 0.0934*0.03, 'b': 4.18*0.02
    }
}

# 2. MS-bar masses at μ=m_Z (electroweak scale)
# Running effects: light quarks heavier, heavy quarks similar
masses_MSbar_mZ = {
    'name': 'MS-bar at μ=m_Z (91 GeV)',
    'up': {'u': 0.00132, 'c': 0.619, 't': 171.7},  # GeV
    'down': {'d': 0.00287, 's': 0.0567, 'b': 2.86},
    'errors': {
        'u': 0.00132*0.2, 'c': 0.619*0.03, 't': 171.7*0.005,
        'd': 0.00287*0.15, 's': 0.0567*0.03, 'b': 2.86*0.02
    }
}

# 3. Pole masses (physical, no running)
# These are what you'd measure in isolation
masses_pole = {
    'name': 'Pole masses',
    'up': {'u': 0.0022, 'c': 1.67, 't': 172.5},  # GeV (u,d poorly defined)
    'down': {'d': 0.0047, 's': 0.500, 'b': 4.78},
    'errors': {
        'u': 0.0022*0.3, 'c': 1.67*0.07, 't': 172.5*0.005,
        'd': 0.0047*0.3, 's': 0.500*0.15, 'b': 4.78*0.02
    }
}

# 4. Yukawa couplings × v/√2 at m_Z
# These are Higgs couplings (most "fundamental")
v_Higgs = 246  # GeV (Higgs VEV)
yukawa_mZ = {
    'name': 'Yukawa at m_Z',
    'up': {'u': 5.9e-6 * v_Higgs/np.sqrt(2), 'c': 3.6e-3 * v_Higgs/np.sqrt(2), 't': 0.988 * v_Higgs/np.sqrt(2)},
    'down': {'d': 1.35e-5 * v_Higgs/np.sqrt(2), 's': 2.74e-4 * v_Higgs/np.sqrt(2), 'b': 1.65e-2 * v_Higgs/np.sqrt(2)},
    'errors': {
        'u': 5.9e-6 * v_Higgs/np.sqrt(2) * 0.2,
        'c': 3.6e-3 * v_Higgs/np.sqrt(2) * 0.03,
        't': 0.988 * v_Higgs/np.sqrt(2) * 0.005,
        'd': 1.35e-5 * v_Higgs/np.sqrt(2) * 0.15,
        's': 2.74e-4 * v_Higgs/np.sqrt(2) * 0.03,
        'b': 1.65e-2 * v_Higgs/np.sqrt(2) * 0.02
    }
}

# 5. Constituent quark masses (non-perturbative, ~300 MeV from QCD)
# These include gluon dressing effects
masses_constituent = {
    'name': 'Constituent masses',
    'up': {'u': 0.330, 'c': 1.55, 't': 172.5},  # GeV
    'down': {'d': 0.330, 's': 0.50, 'b': 4.95},
    'errors': {
        'u': 0.330*0.2, 'c': 1.55*0.1, 't': 172.5*0.005,
        'd': 0.330*0.2, 's': 0.50*0.1, 'b': 4.95*0.02
    }
}

mass_datasets = [
    masses_MSbar_2GeV,
    masses_MSbar_mZ,
    masses_pole,
    yukawa_mZ,
    masses_constituent
]

# ==============================================================================
# MODULAR FORM MACHINERY
# ==============================================================================

def eta_function_weight_k(tau, k):
    """Dedekind eta function |η(τ)|^k."""
    q = np.exp(2j * np.pi * tau)
    prefactor = np.abs(q)**(k/24)
    product = 1.0
    for n in range(1, 51):
        product *= np.abs(1 - q**n)**k
    return prefactor * product

def masses_from_k(k_values, tau, m_scale=1.0):
    """Compute masses from k-pattern."""
    masses = np.array([m_scale * eta_function_weight_k(tau, k) for k in k_values])
    return np.sort(masses)

def chi_squared_fit(log_m_scale, k_pattern, tau, masses_obs, errors_obs):
    """χ² for mass fit with fixed k and τ."""
    m_scale = 10**log_m_scale
    masses_pred = masses_from_k(k_pattern, tau, m_scale)
    
    chi2 = 0
    for m_obs, m_pred, err_obs in zip(masses_obs, masses_pred, errors_obs):
        chi2 += ((m_pred - m_obs) / err_obs)**2
    
    return chi2

# ==============================================================================
# TEST ALL MASS SCALES WITH τ=3.25i
# ==============================================================================

tau_leptons = 3.25j
k_patterns_test = {
    'up': [(8,6,4), (7,5,3), (6,4,2), (5,3,1)],
    'down': [(8,6,4), (7,5,3), (6,4,2), (5,3,1)]
}

results_by_scale = {}

for dataset in mass_datasets:
    name = dataset['name']
    print("="*80)
    print(f"TESTING: {name}")
    print("="*80)
    
    # Extract masses
    masses_up_arr = np.array([dataset['up']['u'], dataset['up']['c'], dataset['up']['t']])
    masses_down_arr = np.array([dataset['down']['d'], dataset['down']['s'], dataset['down']['b']])
    errors_up_arr = np.array([dataset['errors']['u'], dataset['errors']['c'], dataset['errors']['t']])
    errors_down_arr = np.array([dataset['errors']['d'], dataset['errors']['s'], dataset['errors']['b']])
    
    print(f"\nUp-type:   u={masses_up_arr[0]:.4f}, c={masses_up_arr[1]:.3f}, t={masses_up_arr[2]:.1f} GeV")
    print(f"Down-type: d={masses_down_arr[0]:.5f}, s={masses_down_arr[1]:.4f}, b={masses_down_arr[2]:.2f} GeV")
    
    results_up = {}
    results_down = {}
    
    # Test up-type
    print(f"\nUp-type quarks (τ=3.25i):")
    for k_pattern in k_patterns_test['up']:
        result = minimize(
            chi_squared_fit,
            [0],  # log10(m_scale)
            args=(k_pattern, tau_leptons, masses_up_arr, errors_up_arr),
            bounds=[(-3, 3)],
            method='L-BFGS-B'
        )
        
        chi2_min = result.fun
        m_scale = 10**result.x[0]
        p_value = 1 - np.exp(-chi2_min/2)  # rough p-value for dof=2
        
        results_up[str(k_pattern)] = {
            'chi2': chi2_min,
            'm_scale': m_scale,
            'p_value': p_value
        }
        
        status = "✓" if chi2_min < 10 else "❌"
        print(f"  k={k_pattern}: χ²={chi2_min:.2f}, m={m_scale:.3f} GeV {status}")
    
    best_up = min(results_up.items(), key=lambda x: x[1]['chi2'])
    
    # Test down-type
    print(f"\nDown-type quarks (τ=3.25i):")
    for k_pattern in k_patterns_test['down']:
        result = minimize(
            chi_squared_fit,
            [0],
            args=(k_pattern, tau_leptons, masses_down_arr, errors_down_arr),
            bounds=[(-3, 3)],
            method='L-BFGS-B'
        )
        
        chi2_min = result.fun
        m_scale = 10**result.x[0]
        p_value = 1 - np.exp(-chi2_min/2)
        
        results_down[str(k_pattern)] = {
            'chi2': chi2_min,
            'm_scale': m_scale,
            'p_value': p_value
        }
        
        status = "✓" if chi2_min < 10 else "❌"
        print(f"  k={k_pattern}: χ²={chi2_min:.2f}, m={m_scale:.3f} GeV {status}")
    
    best_down = min(results_down.items(), key=lambda x: x[1]['chi2'])
    
    # Summary
    chi2_total = best_up[1]['chi2'] + best_down[1]['chi2']
    
    results_by_scale[name] = {
        'up': best_up,
        'down': best_down,
        'chi2_total': chi2_total
    }
    
    print(f"\n✓ BEST FIT:")
    print(f"  Up:   k={best_up[0]}, χ²={best_up[1]['chi2']:.2f}")
    print(f"  Down: k={best_down[0]}, χ²={best_down[1]['chi2']:.2f}")
    print(f"  Total: χ²={chi2_total:.2f}")
    
    if chi2_total < 20:
        print(f"  ✓✓ ACCEPTABLE! This scale works with τ=3.25i!")
    elif chi2_total < 100:
        print(f"  ⚠️  MARGINAL (χ² < 100)")
    else:
        print(f"  ❌ FAILS (χ² > 100)")
    
    print()

# ==============================================================================
# SUMMARY: Best mass scale
# ==============================================================================

print("="*80)
print("SUMMARY: Which mass scale works best?")
print("="*80)

best_scale = min(results_by_scale.items(), key=lambda x: x[1]['chi2_total'])

print("\nRanking by total χ²:\n")
for i, (name, result) in enumerate(sorted(results_by_scale.items(), key=lambda x: x[1]['chi2_total']), 1):
    chi2 = result['chi2_total']
    status = "✓✓ WORKS" if chi2 < 20 else ("⚠️  MARGINAL" if chi2 < 100 else "❌ FAILS")
    print(f"{i}. {name:30s} χ²={chi2:8.2f}  {status}")

print(f"\n⚠️  BEST MASS DEFINITION: {best_scale[0]}")
print(f"   Total χ² = {best_scale[1]['chi2_total']:.2f}")
print(f"   Up:   k={best_scale[1]['up'][0]}, χ²={best_scale[1]['up'][1]['chi2']:.2f}")
print(f"   Down: k={best_scale[1]['down'][0]}, χ²={best_scale[1]['down'][1]['chi2']:.2f}")

if best_scale[1]['chi2_total'] < 20:
    print("\n✓✓✓ UNIFICATION SAVED!")
    print("    Quarks FIT with τ=3.25i using correct mass scale")
    print("    Framework works across all fermions!")
elif best_scale[1]['chi2_total'] < 100:
    print("\n⚠️  PARTIAL SUCCESS")
    print("   Some mass scales work better but still not perfect")
    print("   Framework structure correct, needs refinement")
else:
    print("\n❌ UNIFICATION FAILS")
    print("   NO mass scale rescues τ=3.25i fit")
    print("   Quarks genuinely require different τ (sectors decouple)")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: χ² by mass scale (up-type)
ax = axes[0, 0]
scales = list(results_by_scale.keys())
chi2_up = [results_by_scale[s]['up'][1]['chi2'] for s in scales]
colors = ['green' if c < 10 else ('orange' if c < 100 else 'red') for c in chi2_up]

y_pos = np.arange(len(scales))
ax.barh(y_pos, chi2_up, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(scales, fontsize=9)
ax.set_xlabel('χ²', fontsize=11)
ax.set_title('A. Up-type Quarks (τ=3.25i)', fontsize=12, fontweight='bold')
ax.axvline(10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='χ²=10')
ax.axvline(100, color='red', linestyle=':', linewidth=2, alpha=0.5, label='χ²=100')
ax.legend()
ax.grid(alpha=0.3, axis='x')
ax.set_xscale('log')

# Panel B: χ² by mass scale (down-type)
ax = axes[0, 1]
chi2_down = [results_by_scale[s]['down'][1]['chi2'] for s in scales]
colors = ['green' if c < 10 else ('orange' if c < 100 else 'red') for c in chi2_down]

ax.barh(y_pos, chi2_down, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(scales, fontsize=9)
ax.set_xlabel('χ²', fontsize=11)
ax.set_title('B. Down-type Quarks (τ=3.25i)', fontsize=12, fontweight='bold')
ax.axvline(10, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.axvline(100, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax.grid(alpha=0.3, axis='x')
ax.set_xscale('log')

# Panel C: Total χ² comparison
ax = axes[1, 0]
chi2_total = [results_by_scale[s]['chi2_total'] for s in scales]
colors = ['green' if c < 20 else ('orange' if c < 100 else 'red') for c in chi2_total]

ax.bar(range(len(scales)), chi2_total, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(scales)))
ax.set_xticklabels([s.split()[0] for s in scales], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Total χ² (up + down)', fontsize=11)
ax.set_title('C. Combined Fit Quality', fontsize=12, fontweight='bold')
ax.axhline(20, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (χ²<20)')
ax.axhline(100, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Marginal')
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_yscale('log')

for i, val in enumerate(chi2_total):
    ax.text(i, val * 1.2, f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel D: Mass hierarchy comparison
ax = axes[1, 1]

# Compare mass ratios across scales
scales_short = ['MS(2GeV)', 'MS(mZ)', 'Pole', 'Yukawa', 'Constit.']
mt_mc_ratio = []
mc_mu_ratio = []

for dataset in mass_datasets:
    mu, mc, mt = dataset['up']['u'], dataset['up']['c'], dataset['up']['t']
    mt_mc_ratio.append(mt/mc)
    mc_mu_ratio.append(mc/mu)

x = np.arange(len(scales_short))
width = 0.35

ax.bar(x - width/2, mt_mc_ratio, width, label='mt/mc', alpha=0.7, color='blue')
ax.bar(x + width/2, mc_mu_ratio, width, label='mc/mu', alpha=0.7, color='orange')

ax.set_ylabel('Mass ratio', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(scales_short, rotation=45, ha='right', fontsize=9)
ax.set_title('D. Mass Hierarchy Sensitivity', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('quark_mass_scale_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('quark_mass_scale_sensitivity.pdf', dpi=300, bbox_inches='tight')

print("\n" + "="*80)
print("Figures saved: quark_mass_scale_sensitivity.png/pdf")
print("="*80)
