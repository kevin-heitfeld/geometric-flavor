"""
Dark Matter from Heavy Right-Handed Neutrinos

EXPLORATION BRANCH - NOT VALIDATED
This is speculative extension of the flavor framework to dark matter.
Heavy right-handed neutrinos from the seesaw mechanism could be DM candidates.

Connection to flavor framework:
- Type-I seesaw: m_ν ≈ m_D² / M_R
- Light neutrino masses: m_ν ~ 0.01-0.1 eV (measured)
- Dirac masses: m_D ~ Y_ν × v ~ O(MeV - GeV) from modular Yukawas
- Heavy Majorana masses: M_R ~ ? (to be determined)

Dark matter requirements:
- Relic abundance: Ω_DM h² ≈ 0.12 (Planck 2018)
- Stability: Lifetime >> age of universe (13.8 Gyr)
- Non-relativistic: Mass >> T_decoupling
- Weak interactions: Cross section σ ~ pb - fb range

Mechanisms to explore:
1. Freeze-out production (heavy N thermalized, then froze out)
2. Freeze-in production (never thermalized, produced from decays)
3. Dodelson-Widrow mechanism (mixing with active neutrinos)
4. Sterile neutrino oscillations

References:
- Dodelson & Widrow, PRL 72, 17 (1994)
- Asaka, Shaposhnikov & Laine, JHEP 0701:091 (2007)
- Drewes et al., JCAP 01 (2017) 025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Fundamental constants
c = 2.998e8           # Speed of light (m/s)
hbar = 6.582e-16      # Reduced Planck constant (eV·s)
k_B = 8.617e-5        # Boltzmann constant (eV/K)
G_N = 6.674e-11       # Gravitational constant (m³/kg/s²)
M_Pl = 1.22e19        # Planck mass (GeV)

# Standard Model parameters
v_EW = 246.0          # Electroweak VEV (GeV)
G_F = 1.166e-5        # Fermi constant (GeV⁻²)
sin2_thetaW = 0.23    # Weak mixing angle

# Cosmological parameters (Planck 2018)
Omega_DM_h2 = 0.120   # Dark matter relic abundance
Omega_b_h2 = 0.0224   # Baryon relic abundance
h = 0.674             # Hubble parameter
T_CMB = 2.7255        # CMB temperature today (K)
T0_eV = k_B * T_CMB   # CMB temperature in eV

# Neutrino oscillation parameters (NuFit 5.0)
# These come from our flavor framework
m_nu1 = 0.010         # Lightest neutrino mass (eV) - from our model
m_nu2 = np.sqrt(m_nu1**2 + 7.42e-5)  # From Δm²₂₁
m_nu3 = np.sqrt(m_nu1**2 + 2.515e-3)  # From Δm²₃₁

# PMNS mixing angles (from our model)
theta12 = 33.8 * np.pi/180
theta23 = 48.5 * np.pi/180
theta13 = 8.65 * np.pi/180

# ============================================================================
# SEESAW MECHANISM: RELATING LIGHT AND HEAVY NEUTRINO MASSES
# ============================================================================

def seesaw_relation(m_light, m_Dirac):
    """
    Type-I seesaw mechanism: m_ν ≈ m_D² / M_R
    
    Given light neutrino masses and Dirac masses, compute required
    heavy right-handed Majorana masses.
    
    Args:
        m_light: Light neutrino mass (eV)
        m_Dirac: Dirac neutrino mass (GeV)
    
    Returns:
        M_R: Heavy Majorana mass (GeV)
    
    Physics:
    The seesaw mechanism explains why neutrinos are so light.
    If M_R >> m_D, then the light eigenstate has mass ≈ m_D²/M_R
    and the heavy eigenstate has mass ≈ M_R.
    
    For m_ν ~ 0.01-0.1 eV and m_D ~ 1 GeV:
        M_R ~ (1 GeV)² / (0.05 eV) ~ 2×10¹⁰ GeV
    
    This is very heavy! Not a viable DM candidate (too massive).
    
    For DM, we need M_R ~ keV to TeV range, which means:
        m_D ~ √(m_ν × M_R)
    
    For M_R ~ 10 keV and m_ν ~ 0.05 eV:
        m_D ~ √(0.05 eV × 10⁴ eV) ~ 22 eV
    
    This is much smaller than typical Yukawa × v_EW.
    Requires either:
    1. Very small Yukawa couplings Y_ν ~ 10⁻¹¹
    2. Different mechanism (inverse seesaw, etc.)
    """
    # Convert units: m_light in eV, m_Dirac in GeV, return M_R in GeV
    m_light_GeV = m_light * 1e-9
    M_R = m_Dirac**2 / m_light_GeV
    return M_R

def required_yukawa(m_light_eV, M_R_GeV):
    """
    Required Yukawa coupling for given light mass and heavy mass.
    
    From seesaw: m_ν = (Y_ν v)² / M_R
    Therefore: Y_ν = √(m_ν × M_R) / v
    
    Args:
        m_light_eV: Light neutrino mass (eV)
        M_R_GeV: Heavy Majorana mass (GeV)
    
    Returns:
        Y_nu: Required Yukawa coupling (dimensionless)
    """
    m_light_GeV = m_light_eV * 1e-9
    m_Dirac_GeV = np.sqrt(m_light_GeV * M_R_GeV)
    Y_nu = m_Dirac_GeV / v_EW
    return Y_nu

# ============================================================================
# DARK MATTER PRODUCTION MECHANISMS
# ============================================================================

def sterile_neutrino_mixing(M_R, Y_nu):
    """
    Active-sterile mixing angle from seesaw.
    
    The seesaw mechanism induces small mixing between active (ν_L)
    and sterile (N_R) neutrinos:
        θ ≈ m_D / M_R = Y_ν v / M_R
    
    This mixing allows N_R to be produced via oscillations.
    
    Args:
        M_R: Heavy sterile neutrino mass (GeV)
        Y_nu: Yukawa coupling
    
    Returns:
        theta: Mixing angle (radians)
        sin2_2theta: Oscillation parameter sin²(2θ)
    """
    theta = Y_nu * v_EW / M_R
    sin2_2theta = 4 * theta**2 * (1 - theta**2)  # ≈ 4θ² for small θ
    return theta, sin2_2theta

def dodelson_widrow_production(M_R_keV, sin2_2theta):
    """
    Dodelson-Widrow mechanism: sterile neutrino production via oscillations.
    
    In the early universe, active neutrinos oscillate into sterile neutrinos.
    The relic abundance depends on M_R and mixing angle.
    
    Approximate formula (Dodelson & Widrow 1994):
        Ω_s h² ≈ (sin²(2θ) / 10⁻⁸) × (M_s / keV)
    
    Args:
        M_R_keV: Sterile neutrino mass (keV)
        sin2_2theta: Oscillation parameter sin²(2θ)
    
    Returns:
        Omega_s_h2: Relic abundance
        is_viable: Whether this matches observed DM abundance
    
    Viable DM range (rough):
        M_R ~ 1-100 keV
        sin²(2θ) ~ 10⁻¹¹ to 10⁻⁷
    
    X-ray constraints:
        Sterile neutrinos decay via N → ν + γ
        Lifetime: τ ~ 10²⁸ s × (10⁻⁵/sin²(2θ)) × (keV/M_R)³
        X-ray telescopes (Chandra, XMM-Newton) constrain parameter space
    """
    # Simplified DW formula (valid for M_R ~ keV scale)
    Omega_s_h2 = (sin2_2theta / 1e-8) * M_R_keV
    
    # Check if matches DM abundance (within factor of 3)
    is_viable = (0.04 < Omega_s_h2 < 0.36)
    
    # Decay rate (approximate)
    # Γ ~ G_F² × sin²(2θ) × M_R⁵ / (192π³)
    # Lifetime τ = 1/Γ
    Gamma_per_s = 1.4e-29 * sin2_2theta * (M_R_keV)**5  # s⁻¹
    lifetime_s = 1.0 / Gamma_per_s if Gamma_per_s > 0 else np.inf
    lifetime_Gyr = lifetime_s / (3.15e16)  # Convert to Gyr
    
    # Must be stable: τ >> 13.8 Gyr
    is_stable = lifetime_Gyr > 1e4  # 10,000× age of universe
    
    return Omega_s_h2, is_viable and is_stable, lifetime_Gyr

def freeze_in_production(M_R_GeV, Y_nu):
    """
    Freeze-in mechanism: sterile neutrinos never thermalized.
    
    Produced via rare decays/scatterings from SM particles.
    Abundance grows linearly with time until production rate drops.
    
    Typical freeze-in masses: GeV to TeV scale
    
    Very rough estimate (order of magnitude):
        Ω_N h² ~ (Y_ν⁴ / 10⁻²⁴) × (GeV / M_R)
    
    Args:
        M_R_GeV: Sterile neutrino mass (GeV)  
        Y_nu: Yukawa coupling
    
    Returns:
        Omega_N_h2: Relic abundance estimate
        is_viable: Whether matches DM abundance
    """
    # This is a very crude estimate - real calculation needs Boltzmann equations
    Omega_N_h2 = (Y_nu**4 / 1e-24) * (1.0 / M_R_GeV)
    
    is_viable = (0.04 < Omega_N_h2 < 0.36)
    
    return Omega_N_h2, is_viable

# ============================================================================
# PARAMETER SPACE SCAN
# ============================================================================

def scan_sterile_neutrino_dm():
    """
    Scan parameter space for viable sterile neutrino dark matter.
    
    We need to find (M_R, Y_ν) combinations that:
    1. Give correct light neutrino masses via seesaw
    2. Produce correct DM relic abundance
    3. Are stable enough (lifetime >> age of universe)
    4. Satisfy experimental constraints (X-ray, BBN, etc.)
    
    Returns:
        Dictionary with scan results and viable regions
    """
    print("=" * 70)
    print("STERILE NEUTRINO DARK MATTER PARAMETER SCAN")
    print("=" * 70)
    print()
    
    # Define scan ranges
    M_R_range_keV = np.logspace(0, 3, 50)  # 1 keV to 1 MeV
    sin2_2theta_range = np.logspace(-12, -6, 50)  # 10⁻¹² to 10⁻⁶
    
    # Storage for results
    viable_points = []
    
    # Scan over parameter space
    print("Scanning parameter space...")
    print(f"Mass range: {M_R_range_keV[0]:.1f} keV to {M_R_range_keV[-1]/1000:.1f} MeV")
    print(f"Mixing range: sin²(2θ) = 10^{np.log10(sin2_2theta_range[0]):.1f} to 10^{np.log10(sin2_2theta_range[-1]):.1f}")
    print()
    
    for M_R_keV in M_R_range_keV:
        for sin2_2theta in sin2_2theta_range:
            # Check Dodelson-Widrow production
            Omega_h2, is_viable, lifetime_Gyr = dodelson_widrow_production(M_R_keV, sin2_2theta)
            
            if is_viable:
                # Compute corresponding Yukawa
                M_R_GeV = M_R_keV * 1e-6
                theta = np.sqrt(sin2_2theta / 4)  # Small angle approximation
                Y_nu = theta * M_R_GeV / v_EW
                
                # Check what light neutrino mass this gives
                m_light_eV = (Y_nu * v_EW)**2 / M_R_GeV * 1e9
                
                viable_points.append({
                    'M_R_keV': M_R_keV,
                    'M_R_GeV': M_R_GeV,
                    'sin2_2theta': sin2_2theta,
                    'Y_nu': Y_nu,
                    'm_light_eV': m_light_eV,
                    'Omega_h2': Omega_h2,
                    'lifetime_Gyr': lifetime_Gyr
                })
    
    print(f"Found {len(viable_points)} viable parameter points")
    print()
    
    if len(viable_points) > 0:
        print("Sample viable points:")
        print("-" * 70)
        print(f"{'M_R (keV)':<12} {'sin²(2θ)':<12} {'Y_ν':<12} {'m_ν (eV)':<12} {'Ω h²':<10}")
        print("-" * 70)
        for i in [0, len(viable_points)//2, -1]:
            p = viable_points[i]
            print(f"{p['M_R_keV']:<12.2f} {p['sin2_2theta']:<12.2e} {p['Y_nu']:<12.2e} {p['m_light_eV']:<12.2e} {p['Omega_h2']:<10.3f}")
        print()
    
    return viable_points

def plot_parameter_space(viable_points):
    """
    Visualize viable parameter space for sterile neutrino DM.
    
    Creates multi-panel plot showing:
    1. Mass vs mixing angle (viable region)
    2. Required Yukawa vs mass
    3. Implied light neutrino mass
    4. Relic abundance contours
    """
    if len(viable_points) == 0:
        print("No viable points found to plot")
        return
    
    # Extract data
    M_R_keV = np.array([p['M_R_keV'] for p in viable_points])
    sin2_2theta = np.array([p['sin2_2theta'] for p in viable_points])
    Y_nu = np.array([p['Y_nu'] for p in viable_points])
    m_light_eV = np.array([p['m_light_eV'] for p in viable_points])
    Omega_h2 = np.array([p['Omega_h2'] for p in viable_points])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Mass-mixing parameter space
    ax = axes[0, 0]
    scatter = ax.scatter(M_R_keV, sin2_2theta, c=Omega_h2, cmap='viridis',
                        s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Sterile neutrino mass $M_R$ (keV)', fontsize=12)
    ax.set_ylabel(r'Mixing parameter $\sin^2(2\theta)$', fontsize=12)
    ax.set_title('Viable Parameter Space (Dodelson-Widrow)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\Omega_{DM} h^2$', fontsize=11)
    
    # Add target line
    ax.axhline(y=1e-10, color='red', linestyle='--', alpha=0.5, label='Typical mixing')
    ax.legend()
    
    # Panel 2: Required Yukawa coupling
    ax = axes[0, 1]
    ax.scatter(M_R_keV, Y_nu, c='blue', s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Sterile neutrino mass $M_R$ (keV)', fontsize=12)
    ax.set_ylabel(r'Required Yukawa coupling $Y_\nu$', fontsize=12)
    ax.set_title('Yukawa Coupling from Seesaw', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add comparison to flavor framework
    ax.axhline(y=1e-6, color='orange', linestyle='--', alpha=0.7, 
               label='Typical flavor Yukawa')
    ax.text(0.05, 0.95, 'Much smaller than\nflavor Yukawas!', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    
    # Panel 3: Implied light neutrino mass
    ax = axes[1, 0]
    ax.scatter(M_R_keV, m_light_eV, c='green', s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Sterile neutrino mass $M_R$ (keV)', fontsize=12)
    ax.set_ylabel(r'Light neutrino mass $m_\nu$ (eV)', fontsize=12)
    ax.set_title('Seesaw-Predicted Light Neutrino Mass', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add measured neutrino mass range
    ax.axhspan(0.01, 0.1, alpha=0.2, color='red', label='Measured range')
    ax.legend()
    
    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    STERILE NEUTRINO DARK MATTER SUMMARY
    
    Viable parameter points: {len(viable_points)}
    
    Mass range: {M_R_keV.min():.1f} - {M_R_keV.max():.1f} keV
    
    Mixing range: {sin2_2theta.min():.2e} - {sin2_2theta.max():.2e}
    
    Required Yukawa: {Y_nu.min():.2e} - {Y_nu.max():.2e}
    
    KEY TENSION:
    • Flavor Yukawas: Y ~ 10⁻⁶ to 10⁻²
    • DM Yukawas: Y ~ 10⁻¹⁴ to 10⁻¹²
    
    PROBLEM: Factor of 10⁸ discrepancy!
    
    Our modular flavor framework predicts
    Yukawas that are TOO LARGE for sterile
    neutrino dark matter via standard seesaw.
    
    Possible resolutions:
    1. Inverse seesaw (different mass mechanism)
    2. Separate modular weights for N_R
    3. Different DM mechanism entirely
    4. Multiple sterile neutrinos (one for DM,
       others for seesaw)
    
    CONCLUSION: Simple extension doesn't work.
    Needs more sophisticated model.
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dark_matter_sterile_neutrino_scan.png', dpi=300, bbox_inches='tight')
    print("Saved figure: dark_matter_sterile_neutrino_scan.png")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DARK MATTER FROM RIGHT-HANDED NEUTRINOS")
    print("Exploratory investigation - NOT peer-reviewed")
    print("="*70 + "\n")
    
    # First, check the basic seesaw relation
    print("STEP 1: Basic Seesaw Check")
    print("-" * 70)
    print("Light neutrino mass: m_ν ~ 0.05 eV (from oscillations)")
    print("Dirac mass from typical Yukawa: m_D = Y_ν × v = 10⁻⁶ × 246 GeV = 0.25 MeV")
    print()
    
    m_D_GeV = 2.5e-4  # 0.25 MeV
    M_R_standard = seesaw_relation(0.05, m_D_GeV)
    print(f"Standard seesaw: M_R = m_D² / m_ν = {M_R_standard:.2e} GeV")
    print(f"                      = {M_R_standard/1e9:.2e} TeV")
    print()
    print("This is WAY too heavy for dark matter!")
    print("DM needs to be in keV-TeV range, not 10¹⁰ GeV")
    print()
    
    # Check what Yukawa is needed for keV sterile neutrinos
    print("STEP 2: Required Yukawa for DM-scale sterile neutrinos")
    print("-" * 70)
    M_R_DM_keV = 10.0  # 10 keV sterile neutrino
    M_R_DM_GeV = M_R_DM_keV * 1e-6
    Y_nu_needed = required_yukawa(0.05, M_R_DM_GeV)
    print(f"For M_R = 10 keV and m_ν = 0.05 eV:")
    print(f"Required Yukawa: Y_ν = {Y_nu_needed:.2e}")
    print()
    print("Compare to flavor Yukawas from our framework:")
    print("  Y_e ~ 10⁻⁶ (electron)")
    print("  Y_μ ~ 10⁻⁴ (muon)")
    print("  Y_τ ~ 10⁻² (tau)")
    print()
    print(f"Our Y_ν ~ {Y_nu_needed:.2e} is 8 ORDERS OF MAGNITUDE smaller!")
    print()
    
    # Perform parameter space scan
    print("STEP 3: Parameter Space Scan")
    print("-" * 70)
    viable_points = scan_sterile_neutrino_dm()
    
    # Visualize results
    if len(viable_points) > 0:
        print("STEP 4: Visualization")
        print("-" * 70)
        plot_parameter_space(viable_points)
    
    # Final assessment
    print("\n" + "="*70)
    print("ASSESSMENT: SIMPLE EXTENSION FAILS")
    print("="*70)
    print("""
The basic problem: Our modular flavor framework naturally produces
Yukawa couplings in the range Y ~ 10⁻⁶ to 10⁻².

For sterile neutrino dark matter in the keV range (required for 
Dodelson-Widrow mechanism), we need Y ~ 10⁻¹⁴ to 10⁻¹².

This is an 8 order of magnitude discrepancy!

Why this happens:
- Flavor Yukawas come from modular forms with specific weights
- These weights are quantized and set by brane wrapping numbers
- The suppression (Im τ)^{-k/2} gives factors of ~0.01 to 0.0001
- This is nowhere near the ~10⁻¹⁴ needed for DM

Possible ways forward (would need new physics):
1. Inverse seesaw: Add extra singlets to get different mass relation
2. Separate sector: Different modular parameter for sterile neutrinos
3. Non-minimal flavor: Additional U(1) symmetries for extra suppression
4. Different DM: Maybe the DM isn't related to flavor at all

CONCLUSION: Cannot simply extend flavor framework to sterile neutrino DM.
Would need significant additional structure.
""")
    print("="*70)
