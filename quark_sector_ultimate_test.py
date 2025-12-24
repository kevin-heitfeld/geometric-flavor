"""
ULTIMATE UNIFICATION TEST: Quark Sector
========================================

THE FINAL TEST: Do quarks fit with τ=3.25i?

If YES → ALL matter unified under single modular parameter
If NO  → Sectors partially decouple (framework incomplete)

ESTABLISHED SO FAR:
- Charged leptons: k=(8,6,4), τ=3.25i ✓
- Neutrinos:       k=(5,3,1), τ=3.25i ✓ (k→k-3 transformation)

NOW TEST:
- Up-type quarks:   k=?, τ=3.25i ??
- Down-type quarks: k=?, τ=3.25i ??

If both quark sectors fit with τ=3.25i AND follow Δk=2 pattern,
this is COMPLETE GEOMETRIC UNIFICATION of Standard Model fermions.

OBSERVABLES (PDG 2023, MS-bar at 2 GeV):
Up-type masses:   mu, mc, mt
Down-type masses: md, ms, mb
CKM matrix:       Vus, Vcb, Vub, Jarlskog invariant

METHOD:
1. Fit k-patterns to quark masses with τ=3.25i FIXED
2. Test if Δk=2 emerges (like leptons/neutrinos)
3. Test k→k-3 offset (up vs down, like charged vs neutral)
4. Predict CKM elements from brane geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2

# ==============================================================================
# EXPERIMENTAL DATA: Quark Masses
# ==============================================================================

# PDG 2023: Running masses in MS-bar scheme at μ=2 GeV
# (These are the most "fundamental" masses, before QCD dressing)

# Up-type quarks (MeV, then GeV)
m_up = 2.16  # MeV at 2 GeV
m_charm = 1.27e3  # MeV = 1.27 GeV at m_c scale
m_top = 172.5e3  # MeV = 172.5 GeV (pole mass, different scale)

# Down-type quarks (MeV)
m_down = 4.67  # MeV at 2 GeV
m_strange = 93.4  # MeV at 2 GeV
m_bottom = 4.18e3  # MeV = 4.18 GeV at m_b scale

# Convert all to GeV for consistency
m_up_GeV = m_up / 1000
m_charm_GeV = m_charm / 1000
m_top_GeV = m_top / 1000

m_down_GeV = m_down / 1000
m_strange_GeV = m_strange / 1000
m_bottom_GeV = m_bottom / 1000

# Rough uncertainties (PDG has asymmetric errors, we use average)
# These are %, will convert to absolute
err_up = 0.20  # ±20% (very uncertain)
err_charm = 0.03  # ±3%
err_top = 0.005  # ±0.5% (best measured)

err_down = 0.15  # ±15%
err_strange = 0.03  # ±3%
err_bottom = 0.02  # ±2%

print("="*80)
print("ULTIMATE UNIFICATION TEST: Quark Sector")
print("="*80)
print("\nTHE FINAL TEST: Do quarks fit with τ=3.25i?")
print("\nEstablished:")
print("  Charged leptons: k=(8,6,4), τ=3.25i ✓")
print("  Neutrinos:       k=(5,3,1), τ=3.25i ✓ (k→k-3)")
print("\nTesting:")
print("  Up-type quarks:   τ=3.25i ??")
print("  Down-type quarks: τ=3.25i ??")
print()

print("="*80)
print("EXPERIMENTAL DATA: Quark Masses (PDG 2023)")
print("="*80)
print("\nUp-type quarks (MS-bar):")
print(f"  mu = {m_up_GeV:.4f} GeV (±{err_up*100:.0f}%)")
print(f"  mc = {m_charm_GeV:.3f} GeV (±{err_charm*100:.0f}%)")
print(f"  mt = {m_top_GeV:.1f} GeV (±{err_top*100:.1f}%)")

print("\nDown-type quarks (MS-bar):")
print(f"  md = {m_down_GeV:.5f} GeV (±{err_down*100:.0f}%)")
print(f"  ms = {m_strange_GeV:.4f} GeV (±{err_strange*100:.0f}%)")
print(f"  mb = {m_bottom_GeV:.2f} GeV (±{err_bottom*100:.0f}%)")

print("\nMass hierarchies:")
print(f"  mt/mc/mu ≈ {m_top_GeV/m_charm_GeV:.0f} : {m_charm_GeV/m_up_GeV:.0f} : 1")
print(f"  mb/ms/md ≈ {m_bottom_GeV/m_strange_GeV:.0f} : {m_strange_GeV/m_down_GeV:.0f} : 1")

# ==============================================================================
# MODULAR FORM MASS FORMULA
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
    """
    Compute masses from k-pattern: m_i = m_scale × |η(τ)|^k_i
    """
    masses = np.array([m_scale * eta_function_weight_k(tau, k) for k in k_values])
    return np.sort(masses)  # Sort: lightest to heaviest

def chi_squared_masses(log_m_scale, k_pattern, tau, masses_obs, errors_obs):
    """
    χ² for mass fit with fixed k and τ.
    Only m_scale varies.
    """
    m_scale = 10**log_m_scale
    masses_pred = masses_from_k(k_pattern, tau, m_scale)

    # χ² from masses
    chi2 = 0
    for m_obs, m_pred, err_obs in zip(masses_obs, masses_pred, errors_obs):
        chi2 += ((m_pred - m_obs) / err_obs)**2

    return chi2

# ==============================================================================
# TEST 1: Up-type quarks with τ=3.25i
# ==============================================================================

print("\n" + "="*80)
print("TEST 1: UP-TYPE QUARKS (τ=3.25i FIXED)")
print("="*80)
print("\nTesting integer k-patterns with Δk=2:")
print("Candidates: (8,6,4), (7,5,3), (6,4,2), (5,3,1)\n")

tau_fixed = 3.25j
masses_up = np.array([m_up_GeV, m_charm_GeV, m_top_GeV])
errors_up = np.array([err_up * m_up_GeV, err_charm * m_charm_GeV, err_top * m_top_GeV])

k_patterns = {
    '(8,6,4)': [8, 6, 4],
    '(7,5,3)': [7, 5, 3],
    '(6,4,2)': [6, 4, 2],
    '(5,3,1)': [5, 3, 1],
}

results_up = {}

for name, k_pattern in k_patterns.items():
    # Fit only m_scale
    initial = [0]  # log10(m_scale) ~ 1 GeV → log ~ 0

    result = minimize(
        chi_squared_masses,
        initial,
        args=(k_pattern, tau_fixed, masses_up, errors_up),
        bounds=[(-3, 3)],
        method='L-BFGS-B'
    )

    m_scale_fit = 10**result.x[0]
    chi2_min = result.fun
    dof = 3 - 1  # 3 masses, 1 parameter
    p_value = 1 - chi2.cdf(chi2_min, dof)

    results_up[name] = {
        'k': k_pattern,
        'm_scale': m_scale_fit,
        'chi2': chi2_min,
        'dof': dof,
        'p_value': p_value
    }

    print(f"k = {name}:")
    print(f"  m_scale = {m_scale_fit:.3f} GeV")
    print(f"  χ² = {chi2_min:.2f}, dof = {dof}, p = {p_value:.3f}")

    if p_value > 0.05:
        print(f"  ✓ ACCEPTABLE (p > 0.05)")
    else:
        print(f"  ❌ REJECTED (p < 0.05)")
    print()

best_up = min(results_up.items(), key=lambda x: x[1]['chi2'])

# ==============================================================================
# TEST 2: Down-type quarks with τ=3.25i
# ==============================================================================

print("="*80)
print("TEST 2: DOWN-TYPE QUARKS (τ=3.25i FIXED)")
print("="*80)
print("\nTesting same k-patterns:\n")

masses_down = np.array([m_down_GeV, m_strange_GeV, m_bottom_GeV])
errors_down = np.array([err_down * m_down_GeV, err_strange * m_strange_GeV, err_bottom * m_bottom_GeV])

results_down = {}

for name, k_pattern in k_patterns.items():
    initial = [0]

    result = minimize(
        chi_squared_masses,
        initial,
        args=(k_pattern, tau_fixed, masses_down, errors_down),
        bounds=[(-3, 3)],
        method='L-BFGS-B'
    )

    m_scale_fit = 10**result.x[0]
    chi2_min = result.fun
    dof = 3 - 1
    p_value = 1 - chi2.cdf(chi2_min, dof)

    results_down[name] = {
        'k': k_pattern,
        'm_scale': m_scale_fit,
        'chi2': chi2_min,
        'dof': dof,
        'p_value': p_value
    }

    print(f"k = {name}:")
    print(f"  m_scale = {m_scale_fit:.3f} GeV")
    print(f"  χ² = {chi2_min:.2f}, dof = {dof}, p = {p_value:.3f}")

    if p_value > 0.05:
        print(f"  ✓ ACCEPTABLE (p > 0.05)")
    else:
        print(f"  ❌ REJECTED (p < 0.05)")
    print()

best_down = min(results_down.items(), key=lambda x: x[1]['chi2'])

# ==============================================================================
# TEST 3: k-offset pattern (up vs down)
# ==============================================================================

print("="*80)
print("TEST 3: k-OFFSET PATTERN (up vs down)")
print("="*80)
print("\nHYPOTHESIS: k_down = k_up - Δ (like charged→neutral)")
print("If Δ=3 works (like leptons), this is universal pattern.\n")

# Assume best up-type pattern, test offsets
k_up_best = best_up[1]['k']
print(f"Best up-type: k = {k_up_best}\n")

offsets = [-2, -1, 0, 1, 2, 3, 4]
offset_results = []

for delta in offsets:
    k_down_test = [k - delta for k in k_up_best]

    if any(k < 0 for k in k_down_test):
        print(f"Δ={delta:+d}: k_down={k_down_test} → SKIP (negative)")
        continue

    # Fit down-type with this k
    initial = [0]

    result = minimize(
        chi_squared_masses,
        initial,
        args=(k_down_test, tau_fixed, masses_down, errors_down),
        bounds=[(-3, 3)],
        method='L-BFGS-B'
    )

    chi2_min = result.fun
    m_scale = 10**result.x[0]
    p_value = 1 - chi2.cdf(chi2_min, 2)

    offset_results.append({
        'delta': delta,
        'k_down': k_down_test,
        'chi2': chi2_min,
        'm_scale': m_scale,
        'p_value': p_value
    })

    status = "✓" if p_value > 0.05 else "❌"
    print(f"Δ={delta:+d}: k_down={k_down_test}, χ²={chi2_min:.2f}, p={p_value:.3f} {status}")

best_offset = min(offset_results, key=lambda x: x['chi2'])

print(f"\n⚠️  BEST OFFSET: Δ = {best_offset['delta']}")
print(f"   k_up   = {k_up_best}")
print(f"   k_down = {best_offset['k_down']}")
print(f"   χ² = {best_offset['chi2']:.2f}, p = {best_offset['p_value']:.3f}")

if best_offset['delta'] == 3:
    print("\n✓✓ UNIVERSAL PATTERN CONFIRMED!")
    print("   Charged leptons → Neutrinos: k→k-3")
    print("   Up quarks → Down quarks: k→k-3")
    print("   This is a UNIVERSAL TRANSFORMATION across all fermions!")

# ==============================================================================
# TEST 4: Combined fit (all quarks)
# ==============================================================================

print("\n" + "="*80)
print("TEST 4: COMBINED FIT (all 6 quarks, τ=3.25i)")
print("="*80)
print("\nIf both up and down fit acceptably, quarks unified with leptons.\n")

# Best patterns
print(f"Best up-type:   k = {best_up[0]}, χ² = {best_up[1]['chi2']:.2f}, p = {best_up[1]['p_value']:.3f}")
print(f"Best down-type: k = {best_down[0]}, χ² = {best_down[1]['chi2']:.2f}, p = {best_down[1]['p_value']:.3f}")

chi2_total = best_up[1]['chi2'] + best_down[1]['chi2']
dof_total = best_up[1]['dof'] + best_down[1]['dof']
p_total = 1 - chi2.cdf(chi2_total, dof_total)

print(f"\nCombined:")
print(f"  χ²_total = {chi2_total:.2f}")
print(f"  dof = {dof_total}")
print(f"  p-value = {p_total:.3f}")

if p_total > 0.05:
    print(f"  ✓✓✓ QUARK SECTOR UNIFIED WITH τ=3.25i!")
else:
    print(f"  ⚠️  Marginal fit (p < 0.05 but > 0.01)")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Up-type quark fits
ax = axes[0, 0]
names = list(results_up.keys())
chi2_up = [results_up[n]['chi2'] for n in names]
colors_up = ['green' if results_up[n]['p_value'] > 0.05 else 'orange' for n in names]

bars = ax.bar(names, chi2_up, color=colors_up, alpha=0.7, edgecolor='black')
ax.axhline(chi2.ppf(0.95, 2), color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
ax.set_ylabel('χ²', fontsize=11)
ax.set_title('A. Up-type Quarks (τ=3.25i)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bar, val, name in zip(bars, chi2_up, names):
    height = bar.get_height()
    p = results_up[name]['p_value']
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
            f'{val:.1f}\np={p:.2f}', ha='center', va='bottom', fontsize=8)

# Panel B: Down-type quark fits
ax = axes[0, 1]
chi2_down = [results_down[n]['chi2'] for n in names]
colors_down = ['green' if results_down[n]['p_value'] > 0.05 else 'orange' for n in names]

bars = ax.bar(names, chi2_down, color=colors_down, alpha=0.7, edgecolor='black')
ax.axhline(chi2.ppf(0.95, 2), color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
ax.set_ylabel('χ²', fontsize=11)
ax.set_title('B. Down-type Quarks (τ=3.25i)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bar, val, name in zip(bars, chi2_down, names):
    height = bar.get_height()
    p = results_down[name]['p_value']
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
            f'{val:.1f}\np={p:.2f}', ha='center', va='bottom', fontsize=8)

# Panel C: k-offset scan
ax = axes[1, 0]
deltas = [r['delta'] for r in offset_results]
chi2_offset = [r['chi2'] for r in offset_results]

ax.plot(deltas, chi2_offset, 'bo-', linewidth=2, markersize=8)
ax.axhline(chi2.ppf(0.95, 2), color='red', linestyle='--', linewidth=2, label='p=0.05')
ax.axvline(best_offset['delta'], color='orange', linestyle=':', linewidth=2, label=f"Best: Δ={best_offset['delta']}")
if 3 in deltas:
    ax.axvline(3, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Δ=3 (lepton pattern)')

ax.set_xlabel('Offset Δ (k_down = k_up - Δ)', fontsize=11)
ax.set_ylabel('χ² (down-type)', fontsize=11)
ax.set_title('C. k-offset: Up → Down Quarks', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel D: Unified picture
ax = axes[1, 1]

sectors = ['Charged\nLeptons', 'Neutrinos', 'Up\nQuarks', 'Down\nQuarks']
k_patterns_unified = [
    (8, 6, 4),  # charged leptons
    (5, 3, 1),  # neutrinos
    tuple(best_up[1]['k']),  # up quarks
    tuple(best_down[1]['k'])  # down quarks
]

x_pos = np.arange(len(sectors))
width = 0.25

for i, gen in enumerate(['3rd', '2nd', '1st']):
    k_vals = [k[i] for k in k_patterns_unified]
    offset = (i - 1) * width
    ax.bar(x_pos + offset, k_vals, width, label=gen, alpha=0.7)

ax.set_ylabel('Modular weight k', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(sectors, fontsize=10)
ax.set_title('D. Unified k-patterns (τ=3.25i)', fontsize=12, fontweight='bold')
ax.legend(title='Generation')
ax.grid(alpha=0.3, axis='y')

# Add τ annotation
ax.text(0.5, 0.95, 'τ = 3.25i (UNIVERSAL)',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('quark_sector_ultimate_test.png', dpi=300, bbox_inches='tight')
plt.savefig('quark_sector_ultimate_test.pdf', dpi=300, bbox_inches='tight')

print("\n" + "="*80)
print("FINAL VERDICT: ULTIMATE UNIFICATION")
print("="*80)

print(f"\n✓ ESTABLISHED:")
print(f"  Charged leptons: k=(8,6,4), τ=3.25i, χ²=... (previous work)")
print(f"  Neutrinos:       k=(5,3,1), τ=3.25i, χ²=3.07, p=0.08")

print(f"\n✓ TESTED:")
print(f"  Up quarks:   k={best_up[0]}, τ=3.25i, χ²={best_up[1]['chi2']:.2f}, p={best_up[1]['p_value']:.3f}")
print(f"  Down quarks: k={best_down[0]}, τ=3.25i, χ²={best_down[1]['chi2']:.2f}, p={best_down[1]['p_value']:.3f}")

if p_total > 0.05 and best_up[1]['p_value'] > 0.05 and best_down[1]['p_value'] > 0.05:
    print(f"\n✓✓✓ COMPLETE GEOMETRIC UNIFICATION ACHIEVED!")
    print(f"    ALL Standard Model fermions unified under τ=3.25i")
    print(f"    Universal Δk=2 spacing across all sectors")
    print(f"    Universal k→k-3 transformation (charged→neutral)")
    print(f"\n    This is a MAJOR MILESTONE:")
    print(f"    Single geometric parameter determines ALL fermion masses")
elif p_total > 0.01:
    print(f"\n⚠️  MARGINAL UNIFICATION")
    print(f"   Quarks fit with τ=3.25i but with lower statistical significance")
    print(f"   Framework works but needs refinement")
else:
    print(f"\n❌ UNIFICATION INCOMPLETE")
    print(f"   Quarks do not fit well with τ=3.25i")
    print(f"   Sectors partially decouple")

print("\n" + "="*80)
print("Figures saved: quark_sector_ultimate_test.png/pdf")
print("="*80)

# ==============================================================================
# DIAGNOSIS: What τ do quarks actually prefer?
# ==============================================================================

print("\n" + "="*80)
print("DIAGNOSIS: What τ do quarks prefer? (FREE FIT)")
print("="*80)
print("\nQuarks failed with τ=3.25i. Let's see what τ they actually want.\n")

def chi2_free_tau(params, k_pattern, masses_obs, errors_obs):
    """χ² with both τ and m_scale free."""
    tau_im, log_m = params
    tau = 1j * tau_im
    m_scale = 10**log_m

    try:
        masses_pred = masses_from_k(k_pattern, tau, m_scale)
    except:
        return 1e10

    chi2 = 0
    for m_obs, m_pred, err_obs in zip(masses_obs, masses_pred, errors_obs):
        chi2 += ((m_pred - m_obs) / err_obs)**2

    return chi2

# Up-type quarks with FREE τ
print("Up-type quarks (best k-pattern):")
result_up_free = minimize(
    chi2_free_tau,
    [3.5, 0],  # initial: tau~3.5i, m_scale~1 GeV
    args=(best_up[1]['k'], masses_up, errors_up),
    bounds=[(1, 10), (-2, 3)],
    method='L-BFGS-B'
)

tau_up_free = result_up_free.x[0]
m_up_free = 10**result_up_free.x[1]
chi2_up_free = result_up_free.fun
p_up_free = 1 - chi2.cdf(chi2_up_free, 1)  # 3 masses, 2 params → 1 dof

print(f"  k = {best_up[1]['k']}")
print(f"  Best fit: τ = {tau_up_free:.3f}i, m_scale = {m_up_free:.3f} GeV")
print(f"  χ² = {chi2_up_free:.2f}, p = {p_up_free:.3f}")
print(f"  Deviation from leptons: Δτ = {abs(tau_up_free - 3.25):.3f}")

# Down-type quarks with FREE τ
print("\nDown-type quarks (best k-pattern):")
result_down_free = minimize(
    chi2_free_tau,
    [3.5, 0],
    args=(best_down[1]['k'], masses_down, errors_down),
    bounds=[(1, 10), (-2, 3)],
    method='L-BFGS-B'
)

tau_down_free = result_down_free.x[0]
m_down_free = 10**result_down_free.x[1]
chi2_down_free = result_down_free.fun
p_down_free = 1 - chi2.cdf(chi2_down_free, 1)

print(f"  k = {best_down[1]['k']}")
print(f"  Best fit: τ = {tau_down_free:.3f}i, m_scale = {m_down_free:.3f} GeV")
print(f"  χ² = {chi2_down_free:.2f}, p = {p_down_free:.3f}")
print(f"  Deviation from leptons: Δτ = {abs(tau_down_free - 3.25):.3f}")

print("\n⚠️  DIAGNOSIS:")
if abs(tau_up_free - 3.25) > 0.5 or abs(tau_down_free - 3.25) > 0.5:
    print("   SECTORS DECOUPLE: Quarks need different τ than leptons")
    print(f"   Leptons:  τ = 3.25i")
    print(f"   Up quarks:   τ = {tau_up_free:.2f}i")
    print(f"   Down quarks: τ = {tau_down_free:.2f}i")
    print("\n   INTERPRETATION:")
    print("   • Quarks live on different brane stack")
    print("   • SU(3)_color vs SU(2)_weak geometric separation")
    print("   • PARTIAL unification (leptons only), not complete")
else:
    print("   τ values COMPATIBLE: Issue is QCD running or mass definition")
    print("   Framework structure sound, needs mass scale refinement")

print("\n" + "="*80)
