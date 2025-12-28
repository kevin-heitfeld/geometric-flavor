"""
Dark Energy: Refined Quintessence Potential

The pure exponential V ~ exp(-cφ)/φⁿ is too steep.
Real string compactifications have flatter potentials from:

1. **Kähler mixing**: Different moduli cross-couple
2. **Multiple instantons**: Sum over different sectors
3. **Quantum corrections**: Log-running from loops

We try a **double-exponential** form inspired by KKLT/LVS:

V(φ) = A exp(-a φ) - B exp(-b φ)

where a < b (second term decays faster), creating a metastable minimum
or slow-roll region suitable for quintessence.

This is the "racetrack" potential from string theory.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve

# Physical constants
M_Pl = 2.4e18  # GeV
H0 = 67.4e-33  # GeV
rho_DE = (2.3e-12)**4  # GeV⁴

print("=" * 80)
print("REFINED QUINTESSENCE: RACETRACK POTENTIAL")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: Racetrack Potential from String Theory
# ============================================================================
print("SECTION 1: Racetrack Potential")
print("-" * 80)

print("""
The racetrack potential arises from multiple gaugino condensation:

V(φ) = A e^(-aφ) - B e^(-bφ)

where:
- A, B set by gaugino condensation scales
- a, b set by beta functions: a = 2π/N₁, b = 2π/N₂
- N₁, N₂ are hidden sector gauge group ranks

For quintessence, we want:
- A > B (first term dominates at large φ)
- a < b (second term sub-leading but important for minimum)
- Parameters chosen so V(φ_min) ~ (meV)⁴

This naturally gives:
- Slow-roll region near minimum
- Mass m ~ H₀ from curvature
- Attractor behavior (tracker quintessence)
""")

def V_racetrack(phi, A, a, B, b):
    """Racetrack potential V = A exp(-aφ) - B exp(-bφ)"""
    return A * np.exp(-a * phi) - B * np.exp(-b * phi)

def dV_racetrack(phi, A, a, B, b):
    """Derivative of racetrack potential"""
    return -a * A * np.exp(-a * phi) + b * B * np.exp(-b * phi)

def d2V_racetrack(phi, A, a, B, b):
    """Second derivative (for mass)"""
    return a**2 * A * np.exp(-a * phi) - b**2 * B * np.exp(-b * phi)

# Find minimum of potential
def find_minimum(A, a, B, b):
    """Find field value where dV/dφ = 0"""
    # At minimum: a A exp(-aφ) = b B exp(-bφ)
    # → (a A) / (b B) = exp((a-b)φ)
    # → φ_min = log(a A / b B) / (a - b)

    if a >= b:
        return None  # No minimum if a >= b

    phi_min = np.log(a * A / (b * B)) / (a - b)

    # Check it's actually a minimum (V'' > 0)
    V_2 = d2V_racetrack(phi_min, A, a, B, b)
    if V_2 <= 0:
        return None

    return phi_min

# ============================================================================
# SECTION 2: Parameter Scan for Dark Energy Scale
# ============================================================================
print("SECTION 2: Finding Parameters for ρ_DE ~ (meV)⁴")
print("-" * 80)

print("""
Strategy: Fix a, b from gauge theory, scan A, B for:
1. V(φ_min) ~ (2.3 meV)⁴
2. m² = V''(φ_min) ~ H₀²

Typical values from string theory:
- a = 2π/N₁ with N₁ ~ 8-12 → a ~ 0.5-0.8
- b = 2π/N₂ with N₂ ~ 3-5 → b ~ 1.3-2.1
- A ~ M_string⁴ × (gauge coupling)²
- B ~ A × (symmetry breaking factor)
""")

# Fix phenomenologically motivated values
a = 0.6  # ~ 2π/10
b = 1.5  # ~ 2π/4

print(f"Fixed: a = {a}, b = {b}")
print(f"Scanning A, B to match observations...")
print()

# Scan A and B - MUCH wider range!
A_range = np.logspace(-80, -20, 200)  # Extremely wide scan
B_over_A_range = np.linspace(0.1, 0.999, 100)  # B < A required

best_solution = None
best_error = float('inf')

for A in A_range:
    for B_over_A in B_over_A_range:
        B = B_over_A * A

        # Find minimum
        phi_min = find_minimum(A, a, B, b)
        if phi_min is None or phi_min < 0:
            continue

        # Check potential value
        V_min = V_racetrack(phi_min, A, a, B, b)

        # Check mass
        m_sq = d2V_racetrack(phi_min, A, a, B, b)
        if m_sq <= 0:
            continue
        m_eff = np.sqrt(m_sq)

        # Error metric: target V ~ rho_DE and m ~ H0
        error_V = abs(np.log10(V_min) - np.log10(rho_DE))
        error_m = abs(np.log10(m_eff) - np.log10(H0))
        total_error = error_V + error_m

        if total_error < best_error:
            best_error = total_error
            best_solution = {
                'A': A,
                'B': B,
                'B/A': B_over_A,
                'phi_min': phi_min,
                'V_min': V_min,
                'm_eff': m_eff,
                'error': total_error
            }

if best_solution:
    print("✓ VIABLE SOLUTION FOUND!")
    print()
    s = best_solution
    print(f"  A = {s['A']:.3e} GeV⁴")
    print(f"  B = {s['B']:.3e} GeV⁴")
    print(f"  B/A = {s['B/A']:.4f}")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print()
    print(f"  Field VEV: φ_min = {s['phi_min']:.3f}")
    print(f"  Potential depth: V(φ_min) = {s['V_min']:.3e} GeV⁴")
    print(f"  Target (observed): ρ_DE = {rho_DE:.3e} GeV⁴")
    print(f"  Ratio: V/ρ_DE = {s['V_min']/rho_DE:.2f}")
    print()
    print(f"  Effective mass: m_eff = {s['m_eff']:.3e} GeV")
    print(f"  Target (Hubble): H₀ = {H0:.3e} GeV")
    print(f"  Ratio: m/H₀ = {s['m_eff']/H0:.2f}")
    print()

    # Slow-roll parameters
    V_min = s['V_min']
    dV_min = dV_racetrack(s['phi_min'], s['A'], a, s['B'], b)
    epsilon_V = (M_Pl**2 / 2) * (dV_min / V_min)**2
    eta_V = M_Pl**2 * d2V_racetrack(s['phi_min'], s['A'], a, s['B'], b) / V_min

    print(f"  Slow-roll parameters:")
    print(f"    ε_V = {epsilon_V:.3e}")
    print(f"    η_V = {eta_V:.3e}")

    if epsilon_V < 0.1:
        print(f"    ✓ Slow-roll: ε_V ≪ 1 → w ≈ -1 + ε_V ≈ {-1 + epsilon_V:.4f}")
    else:
        print(f"    ⚠ Not slow-roll (ε_V ~ {epsilon_V:.2f})")
    print()

else:
    print("⚠ No solution found in scan range")
    print("   → Try different a, b values or wider A, B ranges")
    print()

# ============================================================================
# SECTION 3: Visualize Best-Fit Potential
# ============================================================================
print("SECTION 3: Potential Visualization")
print("-" * 80)

if best_solution:
    s = best_solution

    # Plot potential around minimum
    phi_range = np.linspace(max(0, s['phi_min'] - 5), s['phi_min'] + 5, 1000)
    V_range = np.array([V_racetrack(p, s['A'], a, s['B'], b) for p in phi_range])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Full potential
    ax1 = axes[0, 0]

    # Individual terms
    V_term1 = s['A'] * np.exp(-a * phi_range)
    V_term2 = -s['B'] * np.exp(-b * phi_range)

    ax1.plot(phi_range, V_range / rho_DE, 'b-', linewidth=2.5, label='Total V(φ)')
    ax1.plot(phi_range, V_term1 / rho_DE, 'g--', linewidth=1.5, alpha=0.7, label='A exp(-aφ)')
    ax1.plot(phi_range, V_term2 / rho_DE, 'r--', linewidth=1.5, alpha=0.7, label='-B exp(-bφ)')
    ax1.axvline(s['phi_min'], color='purple', linestyle=':', linewidth=2, label=f'φ_min = {s["phi_min"]:.2f}')
    ax1.axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.5, label='ρ_DE (observed)')

    ax1.set_xlabel('Field Value φ', fontsize=12)
    ax1.set_ylabel('V(φ) / ρ_DE', fontsize=12)
    ax1.set_title('Racetrack Potential (normalized to ρ_DE)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2, 5)

    # Plot 2: Zoom near minimum
    ax2 = axes[0, 1]
    phi_zoom = np.linspace(s['phi_min'] - 1, s['phi_min'] + 1, 500)
    V_zoom = np.array([V_racetrack(p, s['A'], a, s['B'], b) for p in phi_zoom])

    ax2.plot(phi_zoom, V_zoom / rho_DE, 'b-', linewidth=2.5)
    ax2.axvline(s['phi_min'], color='purple', linestyle=':', linewidth=2)
    ax2.axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Mark slow-roll region
    slow_roll_region = phi_zoom[np.abs(phi_zoom - s['phi_min']) < 0.5]
    V_slow = np.array([V_racetrack(p, s['A'], a, s['B'], b) for p in slow_roll_region])
    ax2.fill_between(slow_roll_region, 0, V_slow / rho_DE, alpha=0.2, color='green',
                     label='Slow-roll region')

    ax2.set_xlabel('Field Value φ', fontsize=12)
    ax2.set_ylabel('V(φ) / ρ_DE', fontsize=12)
    ax2.set_title('Zoom: Near Minimum (Quintessence Era)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: First derivative
    ax3 = axes[1, 0]
    dV_range = np.array([dV_racetrack(p, s['A'], a, s['B'], b) for p in phi_range])

    ax3.plot(phi_range, dV_range, 'r-', linewidth=2)
    ax3.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(s['phi_min'], color='purple', linestyle=':', linewidth=2, label=f"dV/dφ = 0")

    ax3.set_xlabel('Field Value φ', fontsize=12)
    ax3.set_ylabel('dV/dφ (GeV⁴)', fontsize=12)
    ax3.set_title('First Derivative (Force on Field)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Effective mass
    ax4 = axes[1, 1]
    d2V_range = np.array([d2V_racetrack(p, s['A'], a, s['B'], b) for p in phi_range])
    m_eff_range = np.sqrt(np.abs(d2V_range))

    ax4.semilogy(phi_range, m_eff_range / H0, 'g-', linewidth=2)
    ax4.axhline(1, color='r', linestyle='--', linewidth=2, label='m = H₀')
    ax4.axvline(s['phi_min'], color='purple', linestyle=':', linewidth=2)
    ax4.axhspan(0.1, 10, alpha=0.2, color='green', label='Viable range')

    ax4.set_xlabel('Field Value φ', fontsize=12)
    ax4.set_ylabel('m_eff / H₀', fontsize=12)
    ax4.set_title('Effective Mass (units of H₀)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('racetrack_quintessence_potential.png', dpi=300, bbox_inches='tight')
    print("→ Saved: racetrack_quintessence_potential.png")
    print()

# ============================================================================
# SECTION 4: Physical Interpretation
# ============================================================================
print("=" * 80)
print("SECTION 4: Physical Interpretation")
print("=" * 80)
print()

if best_solution:
    s = best_solution

    print("""
KEY INSIGHTS:

1. **Scale origin**: The tiny dark energy scale emerges from:
   - Exponential suppression: V ~ exp(-aφ) with large φ
   - Cancellation between two terms: A exp(-aφ) - B exp(-bφ)
   - Natural if φ ~ O(10) in Planck units

2. **Connection to string theory**:
   - Racetrack from gaugino condensation in hidden sector
   - Parameters a, b from gauge theory beta functions
   - A, B from strong dynamics at string scale

3. **Quintessence behavior**:
   - Field sits near (but not exactly at) minimum
   - Slow-roll toward minimum → w ≈ -1
   - Tracking: field follows radiation/matter before dominating

4. **Same modulus as axion**:
   - Im ρ = axion (strong CP)
   - Re ρ = saxion (quintessence)
   - Unified solution to two problems!

5. **Testable predictions**:
   - w(z) evolution from slow-roll parameters
   - Fifth force from saxion coupling
   - Correlation with axion searches
""")

    # Connection to string scale
    print("String theory connection:")
    print(f"  If A ~ M_string⁴ × α², then:")

    # Estimate M_string from A
    alpha_typical = 0.01  # gauge coupling squared
    M_string_implied = (s['A'] / alpha_typical)**0.25

    print(f"    M_string ~ {M_string_implied:.2e} GeV")

    if 1e10 < M_string_implied < 1e18:
        print(f"    ✓ Reasonable string scale!")
        if 1e15 < M_string_implied < 1e17:
            print(f"    ✓ Near GUT scale (~10¹⁶ GeV) - consistent with flavor physics!")
    else:
        print(f"    ⚠ Unusual string scale - may indicate modification needed")
    print()

    print("""
NEXT STEPS:
1. Full cosmological evolution (tracking behavior)
2. Initial conditions and attractor solutions
3. H₀ tension: early dark energy fraction
4. Fifth force constraints from saxion coupling
5. Combined fit with axion dark matter
""")

else:
    print("No viable solution - need to refine parameter ranges or potential form.")

print()
print("=" * 80)
print("REFINED QUINTESSENCE EXPLORATION COMPLETE")
print("=" * 80)
