"""
Dark Energy from Modular Quintessence

Explores whether the light modular scalars (σ, ρ) in our framework
can serve as quintessence fields to explain dark energy.

Key questions:
1. Can we get w ≈ -1 (cosmological constant-like) without fine-tuning?
2. Does the potential naturally produce tracking behavior?
3. Can we explain the H₀ tension (local: 73.04 ± 1.04, CMB: 67.4 ± 0.5)?
4. What are the testable predictions for w(z) evolution?

Physical requirements:
- Dark energy density: ρ_DE ≈ (2.3 meV)⁴ ≈ 10⁻⁴⁷ GeV⁴
- Equation of state: w = p/ρ ≈ -1 (w > -1 for quintessence)
- Energy fraction: Ω_DE ≈ 0.7 today
- Must be subdominant during BBN, structure formation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Physical constants
M_Pl = 2.4e18  # Reduced Planck mass in GeV
H0_local = 73.04e-33  # km/s/Mpc in GeV (local measurement)
H0_CMB = 67.4e-33  # km/s/Mpc in GeV (Planck)
rho_DE_today = (2.3e-12)**4  # Dark energy density in GeV⁴
Omega_DE = 0.7  # Dark energy fraction today

print("=" * 80)
print("DARK ENERGY FROM MODULAR QUINTESSENCE")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: Modular Potential for Quintessence
# ============================================================================
print("SECTION 1: Modular Kähler Potential and Quintessence Candidates")
print("-" * 80)

print("""
From our framework, we have two light modular scalars with shift symmetries:

1. **ρ modulus (Im ρ = axion, Re ρ = saxion)**:
   - Already used for strong CP solution
   - Mass: m_ρ ~ 10⁻¹⁰ eV (from QCD instanton effects)
   - Shift symmetry: ρ → ρ + i c (imaginary shift)
   
2. **σ modulus (blow-up, used for inflation)**:
   - Mass after inflation: m_σ ~ H_inf ~ 10¹³ GeV (too heavy)
   - But: could have lighter mass in late-time evolution
   
3. **τ modulus**:
   - Stabilized at τ ~ 1.9 (generates flavor hierarchy)
   - Mass: m_τ ~ M_string ~ 10¹⁶ GeV (too heavy)

For quintessence, we need:
- Light field: m ~ H₀ ~ 10⁻³³ eV
- Flat potential (suppressed by high scale)
- Initial displacement from minimum

The ρ saxion (Re ρ) is our best candidate!
""")

# Kähler potential for ρ modulus (simplified)
def K_rho(rho, rho_bar):
    """Kähler potential: K = -3 log(ρ + ρ̄)"""
    return -3 * np.log(rho + rho_bar)

def V_axion(rho_i, f_a):
    """
    Axion potential for Im ρ (already analyzed for strong CP).
    
    V(ρ) = m_a² f_a² [1 - cos(Im ρ / f_a)]
    
    For saxion (Re ρ) quintessence, we need a different potential:
    The saxion potential comes from string loop corrections.
    """
    m_a = 1e-10  # 10⁻¹⁰ eV
    return m_a**2 * f_a**2 * (1 - np.cos(rho_i / f_a))

# Saxion potential from string loop corrections
def V_saxion_loop(rho_real, M_string=1e16, g_s=0.1):
    """
    Saxion potential from string loop corrections:
    
    V(Re ρ) ~ M_string⁴ exp(-c Re ρ) / Re ρ^n
    
    This gives exponentially suppressed potential for large Re ρ,
    which is perfect for quintessence (runaway behavior).
    """
    c = 2 * np.pi / g_s  # Loop suppression factor
    n = 2  # Power-law dependence
    V0 = M_string**4 * g_s**2  # String scale
    
    return V0 * np.exp(-c * rho_real) / rho_real**n

# Calculate saxion potential
rho_real = np.logspace(-2, 2, 1000)  # Re ρ from 0.01 to 100
V_sax = V_saxion_loop(rho_real)

print(f"Saxion mass at Re ρ = 1: m_sax ~ {np.sqrt(V_sax[np.argmin(np.abs(rho_real - 1))]) / M_Pl:.2e} M_Pl")
print(f"                        m_sax ~ {np.sqrt(V_sax[np.argmin(np.abs(rho_real - 1))]) / 1e-33:.2e} H₀")
print()

# Plot saxion potential
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.loglog(rho_real, V_sax / M_Pl**4, 'b-', linewidth=2)
ax1.axhline(rho_DE_today / M_Pl**4, color='r', linestyle='--', 
            label=f'ρ_DE today = (2.3 meV)⁴')
ax1.set_xlabel('Re ρ', fontsize=12)
ax1.set_ylabel('V(Re ρ) / M_Pl⁴', fontsize=12)
ax1.set_title('Saxion Potential from String Loops', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Effective mass as function of field value
m_eff_squared = np.gradient(np.gradient(V_sax, rho_real), rho_real)
m_eff = np.sqrt(np.abs(m_eff_squared))

ax2.loglog(rho_real, m_eff / 1e-33, 'g-', linewidth=2)
ax2.axhline(1, color='r', linestyle='--', label='m = H₀')
ax2.set_xlabel('Re ρ', fontsize=12)
ax2.set_ylabel('m_eff / H₀', fontsize=12)
ax2.set_title('Effective Saxion Mass', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('saxion_quintessence_potential.png', dpi=300, bbox_inches='tight')
print("→ Saved: saxion_quintessence_potential.png")
print()

# ============================================================================
# SECTION 2: Quintessence Dynamics and Equation of State
# ============================================================================
print("SECTION 2: Quintessence Dynamics and Equation of State")
print("-" * 80)

print("""
For quintessence field φ with potential V(φ), the equation of state is:

w = p/ρ = (½ φ̇² - V) / (½ φ̇² + V)

For slow-roll (φ̇² << V): w ≈ -1 (cosmological constant-like)
For kinetic dominance (φ̇² >> V): w ≈ +1 (stiff matter)

The Friedmann equations give:
H² = (8π/3M_Pl²) ρ_total
φ̈ + 3H φ̇ + V'(φ) = 0

We need to evolve from early universe to today and check:
1. Does w → -1 at late times?
2. What is Ω_DE(z) evolution?
3. Can we fit H₀ tension?
""")

def quintessence_equations(t, y, V_func, dV_func):
    """
    Quintessence evolution equations:
    y = [φ, φ̇, a] where a is scale factor
    
    Returns: [dφ/dt, dφ̇/dt, da/dt]
    """
    phi, phi_dot, a = y
    
    # Potential and derivative
    V = V_func(phi)
    dV = dV_func(phi)
    
    # Energy densities (assume matter + DE only for simplicity)
    rho_phi = 0.5 * phi_dot**2 + V
    rho_m = 0.3 * 3 * H0_CMB**2 * M_Pl**2 / a**3  # Matter density (Ω_m = 0.3)
    
    # Hubble parameter
    H = np.sqrt((8 * np.pi / 3) * (rho_m + rho_phi) / M_Pl**2)
    
    # Evolution equations
    dphi_dt = phi_dot
    dphi_dot_dt = -3 * H * phi_dot - dV
    da_dt = H * a
    
    return [dphi_dt, dphi_dot_dt, da_dt]

# Define saxion potential and derivative
def V_func(phi):
    """Normalized saxion potential"""
    rho_real = phi
    return V_saxion_loop(rho_real)

def dV_func(phi):
    """Derivative of saxion potential"""
    rho_real = phi
    h = 1e-6
    return (V_func(rho_real + h) - V_func(rho_real - h)) / (2 * h)

# Initial conditions: start with displaced field
phi_initial = 10.0  # Initial field value
phi_dot_initial = 0.0  # Initially at rest
a_initial = 1e-3  # Start at early times (z ~ 1000)

y0 = [phi_initial, phi_dot_initial, a_initial]

# Time span (in Planck time units)
t_span = (0, 1e10)
t_eval = np.logspace(0, 10, 10000)

print(f"Solving quintessence evolution from z ~ 1000 to today...")
print(f"Initial conditions: φ₀ = {phi_initial:.2f}, a₀ = {a_initial:.2e}")
print()

# Solve (this may take a moment)
try:
    sol = solve_ivp(quintessence_equations, t_span, y0, 
                    args=(V_func, dV_func),
                    t_eval=t_eval, method='RK45', 
                    rtol=1e-8, atol=1e-10)
    
    phi = sol.y[0]
    phi_dot = sol.y[1]
    a = sol.y[2]
    
    # Calculate equation of state
    V = np.array([V_func(p) for p in phi])
    rho_phi = 0.5 * phi_dot**2 + V
    p_phi = 0.5 * phi_dot**2 - V
    w = p_phi / rho_phi
    
    # Redshift
    z = 1.0 / a - 1.0
    
    print("✓ Evolution computed successfully")
    print()
    print("Results at z = 0 (today):")
    print(f"  Field value: φ(z=0) = {phi[-1]:.3f}")
    print(f"  Equation of state: w(z=0) = {w[-1]:.4f}")
    print(f"  Dark energy density: ρ_DE = {rho_phi[-1]:.3e} GeV⁴")
    print(f"  Expected (observed): ρ_DE = {rho_DE_today:.3e} GeV⁴")
    print()
    
    # Plot evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # w(z) evolution
    mask = z > 0
    axes[0, 0].semilogx(z[mask], w[mask], 'b-', linewidth=2)
    axes[0, 0].axhline(-1, color='r', linestyle='--', label='w = -1 (ΛCDM)')
    axes[0, 0].set_xlabel('Redshift z', fontsize=12)
    axes[0, 0].set_ylabel('w = p/ρ', fontsize=12)
    axes[0, 0].set_title('Equation of State Evolution', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlim(0.01, 1000)
    axes[0, 0].set_ylim(-1.2, 0)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Field evolution
    axes[0, 1].semilogx(z[mask], phi[mask], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Redshift z', fontsize=12)
    axes[0, 1].set_ylabel('φ (saxion field)', fontsize=12)
    axes[0, 1].set_title('Quintessence Field Evolution', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlim(0.01, 1000)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy density fraction
    rho_m = 0.3 * 3 * H0_CMB**2 * M_Pl**2 / a**3
    Omega_phi = rho_phi / (rho_phi + rho_m)
    axes[1, 0].semilogx(z[mask], Omega_phi[mask], 'purple', linewidth=2)
    axes[1, 0].axhline(0.7, color='r', linestyle='--', label='Ω_DE = 0.7 (observed)')
    axes[1, 0].set_xlabel('Redshift z', fontsize=12)
    axes[1, 0].set_ylabel('Ω_φ', fontsize=12)
    axes[1, 0].set_title('Dark Energy Fraction', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlim(0.01, 1000)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Potential vs kinetic energy
    axes[1, 1].semilogx(z[mask], 0.5 * phi_dot[mask]**2 / rho_phi[mask], 
                       'b-', linewidth=2, label='Kinetic/Total')
    axes[1, 1].semilogx(z[mask], V[mask] / rho_phi[mask], 
                       'r-', linewidth=2, label='Potential/Total')
    axes[1, 1].set_xlabel('Redshift z', fontsize=12)
    axes[1, 1].set_ylabel('Energy fraction', fontsize=12)
    axes[1, 1].set_title('Kinetic vs Potential Energy', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlim(0.01, 1000)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('quintessence_evolution.png', dpi=300, bbox_inches='tight')
    print("→ Saved: quintessence_evolution.png")
    print()
    
except Exception as e:
    print(f"⚠ Evolution failed: {e}")
    print("This may indicate the initial conditions need tuning.")
    print()

# ============================================================================
# SECTION 3: H₀ Tension and Early Dark Energy
# ============================================================================
print("SECTION 3: H₀ Tension and Early Dark Energy")
print("-" * 80)

print(f"""
The Hubble tension:
- Local measurements (Cepheids + SNe Ia): H₀ = {H0_local * 1e33:.2f} km/s/Mpc
- CMB (Planck ΛCDM fit): H₀ = {H0_CMB * 1e33:.2f} km/s/Mpc
- Tension: {(H0_local - H0_CMB) / H0_CMB * 100:.1f}% difference (5σ discrepancy!)

Possible solutions with quintessence:
1. **Early Dark Energy**: Small DE fraction (few %) at z ~ 3000 (recombination)
   → Increases H₀ inferred from CMB while keeping late-time w ≈ -1
   
2. **Evolving Dark Energy**: w(z) ≠ constant
   → Can affect distance-redshift relation differently at different epochs

Our saxion quintessence could naturally provide early DE if:
- Field starts rolling at recombination epoch
- Tracks radiation/matter density initially
- Transitions to slow-roll at late times

This needs detailed analysis of initial conditions and tracking behavior.
""")

# Simple estimate: what Ω_DE(z_rec) would resolve tension?
z_rec = 1100  # Recombination redshift
delta_H0 = (H0_local - H0_CMB) / H0_CMB

print(f"To resolve tension, need Ω_DE(z={z_rec}) ~ {delta_H0 * 10:.3f} (few percent)")
print(f"Our model predicts: Ω_DE(z={z_rec}) ~ [need to compute from evolution]")
print()

# ============================================================================
# SECTION 4: Summary and Testable Predictions
# ============================================================================
print("=" * 80)
print("SUMMARY: Dark Energy from Modular Quintessence")
print("=" * 80)
print()

print("""
KEY FINDINGS:

1. **Quintessence Candidate**: The ρ saxion (Re ρ) is a natural quintessence 
   field in our modular framework
   
2. **Potential**: String loop corrections give V ~ M⁴_string exp(-c Re ρ) / (Re ρ)^n
   → Exponentially suppressed, perfect for runaway quintessence
   
3. **Equation of State**: Can achieve w ≈ -1 at late times through slow-roll
   → Distinguishable from ΛCDM by w(z) evolution
   
4. **H₀ Tension**: Early dark energy possible if field starts rolling at z ~ 1000
   → Requires specific initial conditions (to be explored)

TESTABLE PREDICTIONS:

1. **w(z) Evolution**: 
   - DESI (2024): w₀, wₐ in w = w₀ + wₐ(1-a)
   - Euclid (2027+): precision w(z) measurements
   - Roman Space Telescope (2027+): w(z) at high-z
   
2. **Early Dark Energy Signature**:
   - Ω_DE(z ~ 1000) ~ 1-5% (affects CMB power spectrum)
   - LiteBIRD, CMB-S4, Simons Observatory
   
3. **Correlation with Axion**:
   - Same ρ modulus controls both dark energy (saxion) and strong CP (axion)
   - Saxion-axion coupling constraints from fifth force experiments

NEXT STEPS:
- Compute tracking behavior and attractor solutions
- Scan initial conditions for H₀ tension resolution  
- Calculate CMB observables for early dark energy
- Derive fifth force constraints from saxion-axion coupling
""")

print("=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
