"""
Dark Energy Parameter Scan: Finding the Right Scale

The first exploration showed our saxion quintessence achieves w ≈ -1, but
the energy density is ~166 orders of magnitude too small!

This script systematically scans the string theory parameters to find
the combination that produces the observed dark energy density:

ρ_DE = (2.3 meV)⁴ ≈ 2.8 × 10⁻⁴⁷ GeV⁴

Key parameters to tune:
1. String scale M_string (normally ~10¹⁶ GeV for GUT scale)
2. String coupling g_s (typically ~0.1)
3. Loop suppression factor c (related to instanton action)
4. Field VEV ⟨Re ρ⟩ (determines where we sit in potential)

Physical constraints:
- Must not affect inflation (M_string sets inflaton mass)
- Must not destabilize flavor hierarchy (τ modulus stabilization)
- Should naturally explain why dark energy scale is so tiny
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# Physical constants
M_Pl = 2.4e18  # Reduced Planck mass in GeV
H0 = 67.4e-33  # Hubble constant in GeV
rho_DE_today = (2.3e-12)**4  # Dark energy density in GeV⁴ = 2.8e-47
Omega_DE = 0.7  # Dark energy fraction

print("=" * 80)
print("DARK ENERGY PARAMETER SCAN")
print("=" * 80)
print()
print(f"Target: ρ_DE = {rho_DE_today:.3e} GeV⁴")
print(f"Target: m_φ ~ H₀ = {H0:.3e} GeV")
print(f"Target: Ω_DE ≈ {Omega_DE}")
print()

# ============================================================================
# SECTION 1: The Cosmological Constant Problem
# ============================================================================
print("SECTION 1: Understanding the Scale Problem")
print("-" * 80)

print("""
The cosmological constant problem is that ρ_DE is absurdly small:

ρ_DE / M_Pl⁴ ≈ (2.3 meV / 2.4×10¹⁸ GeV)⁴ ≈ 10⁻¹²³

This is the worst fine-tuning problem in physics!

Our saxion potential has the form:
V(φ) = V₀ exp(-c φ) / φⁿ

where:
- V₀ ~ M_string⁴ g_s² (string loop corrections)
- c = 2π/g_s (instanton action)
- φ = Re ρ (saxion field)

For quintessence, we sit at large φ where V is exponentially suppressed.
The question is: what φ gives V ~ (2.3 meV)⁴?
""")

# ============================================================================
# SECTION 2: Parameter Space Scan
# ============================================================================
print("SECTION 2: Scanning String Theory Parameters")
print("-" * 80)

def V_saxion(phi, M_string, g_s, n=2):
    """
    Saxion potential from string loops:
    V(φ) = M_string⁴ g_s² exp(-2π φ / g_s) / φⁿ
    """
    c = 2 * np.pi / g_s
    V0 = M_string**4 * g_s**2
    return V0 * np.exp(-c * phi) / phi**n

def find_phi_for_target_V(M_string, g_s, V_target=rho_DE_today, n=2):
    """
    Find field value φ where V(φ) = V_target
    """
    # We need to solve: V₀ exp(-c φ) / φⁿ = V_target
    # This is transcendental, so use numerical solver

    def residual(phi):
        return np.abs(V_saxion(phi, M_string, g_s, n) - V_target)

    # Start search at reasonable range
    result = minimize_scalar(residual, bounds=(1e-10, 1e10), method='bounded')

    if result.success:
        return result.x
    else:
        return None

# Scan over string scales and couplings
M_string_values = np.logspace(10, 18, 50)  # 10¹⁰ to 10¹⁸ GeV
g_s_values = np.array([0.01, 0.03, 0.1, 0.3, 1.0])

print("Scanning parameter space...")
print(f"  M_string: {M_string_values[0]:.2e} to {M_string_values[-1]:.2e} GeV")
print(f"  g_s: {g_s_values}")
print()

results = []

for g_s in g_s_values:
    phi_values = []
    m_eff_values = []

    for M_string in M_string_values:
        phi_opt = find_phi_for_target_V(M_string, g_s)

        if phi_opt is not None:
            # Compute effective mass at this field value
            h = 1e-6 * phi_opt
            V = V_saxion(phi_opt, M_string, g_s)
            V_plus = V_saxion(phi_opt + h, M_string, g_s)
            V_minus = V_saxion(phi_opt - h, M_string, g_s)

            m_eff_sq = (V_plus - 2*V + V_minus) / h**2
            m_eff = np.sqrt(np.abs(m_eff_sq)) if m_eff_sq > 0 else 0

            phi_values.append(phi_opt)
            m_eff_values.append(m_eff)
        else:
            phi_values.append(np.nan)
            m_eff_values.append(np.nan)

    results.append({
        'g_s': g_s,
        'phi': np.array(phi_values),
        'm_eff': np.array(m_eff_values)
    })

print("✓ Parameter scan complete")
print()

# ============================================================================
# SECTION 3: Results and Physical Viability
# ============================================================================
print("SECTION 3: Viable Parameter Combinations")
print("-" * 80)

# Find combinations where m_eff ~ H₀
viable_solutions = []

for i, res in enumerate(results):
    g_s = res['g_s']
    phi = res['phi']
    m_eff = res['m_eff']

    # Criterion: 0.1 H₀ < m_eff < 10 H₀ (within order of magnitude)
    mask = (m_eff > 0.1 * H0) & (m_eff < 10 * H0)

    if np.any(mask):
        indices = np.where(mask)[0]
        for idx in indices:
            viable_solutions.append({
                'M_string': M_string_values[idx],
                'g_s': g_s,
                'phi': phi[idx],
                'm_eff': m_eff[idx],
                'm_over_H0': m_eff[idx] / H0
            })

            print(f"✓ Viable solution found:")
            print(f"    M_string = {M_string_values[idx]:.3e} GeV")
            print(f"    g_s = {g_s:.3f}")
            print(f"    ⟨Re ρ⟩ = {phi[idx]:.3e}")
            print(f"    m_eff = {m_eff[idx]:.3e} GeV = {m_eff[idx]/H0:.2f} H₀")
            print()

if len(viable_solutions) == 0:
    print("⚠ No viable solutions found in this parameter range.")
    print("  → Need to extend search or modify potential form")
    print()

# ============================================================================
# SECTION 4: Visualizations
# ============================================================================
print("SECTION 4: Visualizing Parameter Space")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Field value vs string scale
ax1 = axes[0, 0]
for res in results:
    mask = ~np.isnan(res['phi'])
    ax1.loglog(M_string_values[mask], res['phi'][mask],
              label=f"g_s = {res['g_s']:.2f}", linewidth=2)

ax1.set_xlabel('String Scale M_string (GeV)', fontsize=12)
ax1.set_ylabel('⟨Re ρ⟩ (field VEV)', fontsize=12)
ax1.set_title('Field Value for ρ_DE = (2.3 meV)⁴', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Effective mass vs string scale
ax2 = axes[0, 1]
for res in results:
    mask = ~np.isnan(res['m_eff']) & (res['m_eff'] > 0)
    if np.any(mask):
        ax2.loglog(M_string_values[mask], res['m_eff'][mask] / H0,
                  label=f"g_s = {res['g_s']:.2f}", linewidth=2)

ax2.axhline(1, color='r', linestyle='--', linewidth=2, label='m = H₀ (target)')
ax2.axhspan(0.1, 10, alpha=0.2, color='green', label='Viable range')
ax2.set_xlabel('String Scale M_string (GeV)', fontsize=12)
ax2.set_ylabel('m_eff / H₀', fontsize=12)
ax2.set_title('Effective Mass (units of H₀)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Potential shape for viable solution (if exists)
ax3 = axes[1, 0]
if len(viable_solutions) > 0:
    # Pick first viable solution
    sol = viable_solutions[0]
    phi_range = np.logspace(np.log10(sol['phi']/10), np.log10(sol['phi']*10), 1000)
    V_range = np.array([V_saxion(p, sol['M_string'], sol['g_s']) for p in phi_range])

    ax3.loglog(phi_range, V_range / M_Pl**4, 'b-', linewidth=2)
    ax3.axvline(sol['phi'], color='r', linestyle='--', linewidth=2,
               label=f"⟨Re ρ⟩ = {sol['phi']:.2e}")
    ax3.axhline(rho_DE_today / M_Pl**4, color='g', linestyle='--', linewidth=2,
               label='ρ_DE (observed)')

    ax3.set_xlabel('Field Value Re ρ', fontsize=12)
    ax3.set_ylabel('V(Re ρ) / M_Pl⁴', fontsize=12)
    ax3.set_title(f'Viable Potential (M_string={sol["M_string"]:.2e} GeV, g_s={sol["g_s"]:.2f})',
                 fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No viable solutions\nin scanned range',
            ha='center', va='center', fontsize=14, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

# Plot 4: Phase diagram (viable vs non-viable regions)
ax4 = axes[1, 1]
for res in results:
    m_ratio = res['m_eff'] / H0
    viable_mask = (m_ratio > 0.1) & (m_ratio < 10)

    # Plot viable points
    if np.any(viable_mask):
        ax4.scatter(M_string_values[viable_mask],
                   np.full(np.sum(viable_mask), res['g_s']),
                   c='green', s=50, alpha=0.7, marker='o')

    # Plot non-viable points
    non_viable_mask = ~viable_mask & ~np.isnan(m_ratio)
    if np.any(non_viable_mask):
        ax4.scatter(M_string_values[non_viable_mask],
                   np.full(np.sum(non_viable_mask), res['g_s']),
                   c='red', s=20, alpha=0.3, marker='x')

ax4.set_xscale('log')
ax4.set_xlabel('String Scale M_string (GeV)', fontsize=12)
ax4.set_ylabel('String Coupling g_s', fontsize=12)
ax4.set_title('Phase Diagram: Viable (green) vs Non-viable (red)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Viable (m ~ H₀)'),
    Patch(facecolor='red', alpha=0.3, label='Non-viable')
]
ax4.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('dark_energy_parameter_scan.png', dpi=300, bbox_inches='tight')
print("→ Saved: dark_energy_parameter_scan.png")
print()

# ============================================================================
# SECTION 5: Physical Interpretation
# ============================================================================
print("=" * 80)
print("SECTION 5: Physical Interpretation")
print("=" * 80)
print()

if len(viable_solutions) > 0:
    print("GOOD NEWS: Viable quintessence solutions exist!")
    print()

    # Analyze best solution
    best_sol = min(viable_solutions, key=lambda s: abs(np.log10(s['m_over_H0'])))

    print("Best-fit solution:")
    print(f"  M_string = {best_sol['M_string']:.3e} GeV")
    print(f"  g_s = {best_sol['g_s']:.3f}")
    print(f"  ⟨Re ρ⟩ = {best_sol['phi']:.3e}")
    print(f"  m_eff = {best_sol['m_eff']:.3e} GeV = {best_sol['m_over_H0']:.2f} H₀")
    print()

    print("Physical implications:")
    print()

    # Check if consistent with GUT scale
    if 1e15 < best_sol['M_string'] < 1e17:
        print("  ✓ String scale consistent with GUT scale (~10¹⁶ GeV)")
        print("    → Unification with flavor physics preserved")
    elif best_sol['M_string'] < 1e15:
        print("  ⚠ String scale below GUT scale")
        print(f"    → M_string ~ {best_sol['M_string']:.2e} GeV might affect flavor hierarchy")
        print("    → Need to check if τ modulus stabilization still works")
    else:
        print("  ✓ String scale above GUT scale")
        print("    → Extra decoupling of heavy states")
    print()

    # Check field value
    print(f"  Saxion VEV: ⟨Re ρ⟩ ~ {best_sol['phi']:.2e}")
    if best_sol['phi'] > 1:
        print("    → Large field value (weakly coupled regime)")
        print("    → String loop expansion valid")
    else:
        print("    ⚠ Small field value (strongly coupled?)")
        print("    → May need to check string loop convergence")
    print()

    # Estimate slow-roll parameters
    phi = best_sol['phi']
    M_s = best_sol['M_string']
    g_s = best_sol['g_s']

    V = V_saxion(phi, M_s, g_s)
    h = 1e-6 * phi
    dV = (V_saxion(phi + h, M_s, g_s) - V_saxion(phi - h, M_s, g_s)) / (2*h)

    epsilon_V = (M_Pl**2 / 2) * (dV / V)**2
    eta_V = M_Pl**2 * (V_saxion(phi + h, M_s, g_s) - 2*V + V_saxion(phi - h, M_s, g_s)) / (h**2 * V)

    print(f"  Slow-roll parameters:")
    print(f"    ε_V = {epsilon_V:.3e}")
    print(f"    η_V = {eta_V:.3e}")

    if epsilon_V < 1 and abs(eta_V) < 1:
        print("    ✓ Slow-roll conditions satisfied (ε, |η| ≪ 1)")
        print("    → Field slowly rolling, w ≈ -1")
    else:
        print("    ⚠ Slow-roll violated")
        print("    → Field in kinetic regime, need full dynamics")
    print()

else:
    print("BAD NEWS: No viable solutions in scanned parameter range")
    print()
    print("Possible resolutions:")
    print("  1. Extend parameter scan (different M_string, g_s ranges)")
    print("  2. Modify potential form (different powers n, multiple exponentials)")
    print("  3. Use different modulus (σ or combined ρ-σ dynamics)")
    print("  4. Add stabilization terms from higher-order corrections")
    print()
    print("The issue is likely that pure exponential runaway is too steep.")
    print("Real string compactifications have additional features (KK modes,")
    print("worldsheet instantons) that can flatten the potential.")
    print()

# ============================================================================
# SECTION 6: Connection to Observed Universe
# ============================================================================
print("=" * 80)
print("SECTION 6: Testable Predictions")
print("=" * 80)
print()

print("""
If our saxion quintessence is correct, we predict:

1. **Equation of state today**: w₀ ≈ -1 + ε_V
   → Current: w₀ = -1.03 ± 0.03 (Planck + SNe)
   → Distinguishable if ε_V > 0.01

2. **Time evolution**: wₐ = dw/da ≈ -√(2ε_V) (thawing quintessence)
   → Measurable by DESI, Euclid, Roman

3. **Fifth force**: Saxion couples to matter through ρ modulus
   → Constraints from Eöt-Wash (10⁻¹¹ M_Pl), atom interferometry
   → Related to axion coupling (same modulus!)

4. **Cosmic coincidence**: Why is ρ_DE ~ ρ_matter today?
   → Tracker solution: field tracks dominant component
   → Initial conditions washed out (attractor)

5. **H₀ tension**: Early dark energy at z ~ 1000?
   → Requires Ω_DE(z_rec) ~ 0.05
   → Testable with CMB-S4, Simons Observatory

Next step: Full cosmological evolution with tracking behavior.
""")

print("=" * 80)
print("SCAN COMPLETE")
print("=" * 80)
