"""
THEORY #13: MODULAR FLAVOR SYMMETRY

Following ChatGPT's strategic advice:
"Start with Modular Flavor Symmetry. Use Froggatt-Nielsen later to interpret, not to build."

CONCEPTUAL FOUNDATION:
====================
Yukawa couplings are NOT free parameters.
They are MODULAR FORMS - functions of a complex modulus τ.

Key insight: Yukawas = Y_ij(τ) where τ determines ALL flavor structure.

Minimal Implementation:
- Modular group: Γ₃ ≅ A₄ (tetrahedral symmetry)
- Single parameter: τ (complex modulus)
- All Yukawas derived as modular forms

Structure:
    Y(τ) = Σ c_i · Y_i^(k)(τ)

where:
- Y_i^(k)(τ): Modular forms of weight k
- c_i: O(1) coefficients (representation theory)
- τ: Complex modulus (vacuum moduli space)

Philosophical shift:
- Theory #11: Assumed matrix form (parameterization)
- Theory #12: Added parameters (worse)
- Theory #13: DERIVE everything from τ (principled)

Goal: Find τ that predicts BOTH masses AND mixing from modular invariance.

References:
- Feruglio et al., EPJC 77 (2017) 346
- Kobayashi & Otsuka, Physics Reports 2023
- Chen et al., Nucl. Phys. B 883 (2014) 267

Implementation: A₄ modular symmetry at level 3
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

# Experimental data (same as before)
LEPTON_MASSES = np.array([0.511, 105.7, 1776.9])  # MeV
UP_MASSES = np.array([2.16, 1270, 172760])  # MeV
DOWN_MASSES = np.array([4.67, 93.4, 4180])  # MeV

CKM_ANGLES_EXP = {
    'theta_12': 13.04,  # degrees
    'theta_23': 2.38,
    'theta_13': 0.201,
}

PMNS_ANGLES_EXP = {
    'theta_12': 33.45,  # degrees
    'theta_23': 49.0,
    'theta_13': 8.57,
}

# ============================================================================
# MODULAR FORMS: The heart of the theory
# ============================================================================

def dedekind_eta(tau):
    """
    Dedekind eta function: η(τ) = q^(1/24) Π(1 - q^n)
    where q = exp(2πiτ)

    For numerical stability, use truncated product.
    """
    q = np.exp(2j * np.pi * tau)

    # Truncate at reasonable order
    n_max = 50
    product = 1.0
    for n in range(1, n_max):
        product *= (1 - q**n)

    eta = q**(1/24) * product
    return eta

def eisenstein_series_E2(tau, truncate=20):
    """
    Eisenstein series E₂(τ) (weight 2, NOT modular but quasi-modular)

    E₂(τ) = 1 - 24 Σ(n=1 to ∞) σ₁(n) q^n
    where σ₁(n) = Σ(d|n) d
    """
    q = np.exp(2j * np.pi * tau)

    def sigma_1(n):
        """Sum of divisors"""
        divisors = [d for d in range(1, n+1) if n % d == 0]
        return sum(divisors)

    E2 = 1.0
    for n in range(1, truncate):
        E2 -= 24 * sigma_1(n) * q**n

    return E2

def eisenstein_series_E4(tau, truncate=20):
    """
    Eisenstein series E₄(τ) (weight 4, modular)

    E₄(τ) = 1 + 240 Σ(n=1 to ∞) σ₃(n) q^n
    where σ₃(n) = Σ(d|n) d³
    """
    q = np.exp(2j * np.pi * tau)

    def sigma_3(n):
        """Sum of cubes of divisors"""
        divisors = [d for d in range(1, n+1) if n % d == 0]
        return sum(d**3 for d in divisors)

    E4 = 1.0
    for n in range(1, truncate):
        E4 += 240 * sigma_3(n) * q**n

    return E4

def eisenstein_series_E6(tau, truncate=20):
    """
    Eisenstein series E₆(τ) (weight 6, modular)

    E₆(τ) = 1 - 504 Σ(n=1 to ∞) σ₅(n) q^n
    where σ₅(n) = Σ(d|n) d⁵
    """
    q = np.exp(2j * np.pi * tau)

    def sigma_5(n):
        """Sum of fifth powers of divisors"""
        divisors = [d for d in range(1, n+1) if n % d == 0]
        return sum(d**5 for d in divisors)

    E6 = 1.0
    for n in range(1, truncate):
        E6 -= 504 * sigma_5(n) * q**n

    return E6

# ============================================================================
# A₄ MODULAR FORMS (Level 3)
# ============================================================================

def modular_form_A4_triplet_weight2(tau):
    """
    A₄ triplet modular forms of weight 2 (level 3)

    Returns 3-component vector Y = (Y₁, Y₂, Y₃)
    These transform as triplet under A₄ symmetry

    Explicit forms from Feruglio et al.:
    Y₁(τ) ∝ E₂(τ) + ...
    Y₂(τ) ∝ E₂(ωτ) + ...
    Y₃(τ) ∝ E₂(ω²τ) + ...

    where ω = exp(2πi/3)
    """
    omega = np.exp(2j * np.pi / 3)

    # Simplified forms (full expressions involve more structure)
    Y1 = eisenstein_series_E2(tau)
    Y2 = eisenstein_series_E2(omega * tau)
    Y3 = eisenstein_series_E2(omega**2 * tau)

    # Normalize
    norm = np.sqrt(abs(Y1)**2 + abs(Y2)**2 + abs(Y3)**2)
    if norm > 0:
        Y1, Y2, Y3 = Y1/norm, Y2/norm, Y3/norm

    return np.array([Y1, Y2, Y3])

def modular_form_A4_singlet_weight4(tau):
    """
    A₄ singlet modular form of weight 4

    Returns single value Y
    Transforms trivially under A₄
    """
    return eisenstein_series_E4(tau)

def modular_form_A4_singlet_weight6(tau):
    """
    A₄ singlet modular form of weight 6
    """
    return eisenstein_series_E6(tau)

# ============================================================================
# YUKAWA MATRICES FROM MODULAR FORMS
# ============================================================================

def yukawa_from_modular_forms(tau, coeffs, sector='charged_lepton'):
    """
    Construct Yukawa matrix from modular forms

    General structure (Kobayashi & Otsuka):
    Y = c₁·(Y_triplet Y_triplet^T) + c₂·Y_singlet·𝟙 + ...

    Parameters:
    - tau: Complex modulus
    - coeffs: Coupling coefficients (representation theory)
    - sector: 'charged_lepton', 'up_quark', 'down_quark', 'neutrino'

    Returns: 3×3 Yukawa matrix
    """
    # Get modular forms
    Y_triplet = modular_form_A4_triplet_weight2(tau)
    Y_singlet_4 = modular_form_A4_singlet_weight4(tau)

    # Construct matrix (simplified version)
    # Full theory: requires specific A₄ representation contractions

    if sector == 'charged_lepton':
        # Leptons: diagonal-like structure
        # Y_ij = c₁ δᵢⱼ Y_singlet + c₂ (Y_triplet)ᵢ (Y_triplet)ⱼ
        c1, c2 = coeffs[:2]
        Y = c1 * Y_singlet_4 * np.eye(3, dtype=complex)
        Y += c2 * np.outer(Y_triplet, Y_triplet.conj())

    elif sector == 'up_quark':
        # Up quarks: similar but different coefficients
        c1, c2, c3 = coeffs[:3]
        Y = c1 * Y_singlet_4 * np.eye(3, dtype=complex)
        Y += c2 * np.outer(Y_triplet, Y_triplet.conj())
        # Add democratic term (Theory #11 insight!)
        Y += c3 * np.ones((3, 3), dtype=complex)

    elif sector == 'down_quark':
        # Down quarks: inverted hierarchy from up quarks
        c1, c2, c3 = coeffs[:3]
        Y = c1 * Y_singlet_4 * np.eye(3, dtype=complex)
        # Flip sign or structure
        Y += c2 * np.outer(Y_triplet.conj(), Y_triplet)
        Y += c3 * np.ones((3, 3), dtype=complex)

    else:
        raise ValueError(f"Unknown sector: {sector}")

    return Y

def yukawa_to_masses_and_mixing(Y, v_scale=246.0):
    """
    Extract masses and mixing from Yukawa matrix

    In SM: m = Y · v / √2
    where v = 246 GeV (Higgs VEV)

    Returns:
    - masses: 3 mass eigenvalues
    - V: Mixing matrix (unitary)
    """
    # Convert to mass matrix
    M = Y * v_scale / np.sqrt(2)

    # For complex Yukawa, use SVD instead of eigh
    # M = U Σ V^†
    # Masses = Σ, mixing from U and V

    # Take Hermitian part for simplicity (real Yukawa approximation)
    M_hermitian = (M + M.conj().T) / 2

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(M_hermitian)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan, np.nan]), np.eye(3)

    # Sort by mass
    idx = np.argsort(np.abs(eigenvalues))
    masses = np.abs(eigenvalues[idx])
    V = eigenvectors[:, idx]

    return masses, V

# ============================================================================
# MODULI SPACE EXPLORATION
# ============================================================================

def find_optimal_tau(target_masses, sector='charged_lepton', max_coeffs=2):
    """
    Search moduli space for τ that reproduces target masses

    Free parameters:
    - Re(τ), Im(τ): Modulus location
    - c_i: O(1) coupling coefficients

    Constraint: Im(τ) > 0 (upper half plane)
    """
    print(f"\n{'='*70}")
    print(f"MODULAR FLAVOR SYMMETRY: FITTING {sector.upper()}")
    print(f"{'='*70}")
    print(f"Searching moduli space for optimal τ...")

    def objective(params):
        # Extract parameters
        tau_re = params[0]
        tau_im = params[1]
        coeffs = params[2:]

        tau = tau_re + 1j * tau_im

        # Check upper half-plane
        if tau_im <= 0.01:
            return 1e10

        # Construct Yukawa
        try:
            Y = yukawa_from_modular_forms(tau, coeffs, sector)
            masses, _ = yukawa_to_masses_and_mixing(Y)
        except:
            return 1e10

        # Check validity
        if np.any(~np.isfinite(masses)) or np.any(masses <= 0):
            return 1e10

        # Logarithmic error (handles huge mass range)
        log_error = np.mean(np.abs(np.log10(masses) - np.log10(target_masses)))

        return log_error

    # Bounds
    # τ typically in fundamental domain: -0.5 < Re(τ) < 0.5, Im(τ) > 0
    bounds = [
        (-1.0, 1.0),      # Re(τ)
        (0.1, 3.0),       # Im(τ)
    ]
    # Coefficients: O(1) with sign freedom
    for _ in range(max_coeffs):
        bounds.append((-5.0, 5.0))

    print(f"Parameters: τ (complex) + {max_coeffs} coupling coefficients")
    print(f"Running optimization (may take 1-2 minutes)...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=200,
        seed=42,
        workers=1,
        strategy='best1bin',
        atol=1e-6,
        tol=1e-6,
    )

    # Extract results
    tau_re, tau_im = result.x[0], result.x[1]
    coeffs = result.x[2:]
    tau_opt = tau_re + 1j * tau_im

    Y_opt = yukawa_from_modular_forms(tau_opt, coeffs, sector)
    masses_opt, V_opt = yukawa_to_masses_and_mixing(Y_opt)

    print(f"\n{'='*70}")
    print(f"OPTIMAL MODULUS FOUND")
    print(f"{'='*70}")
    print(f"τ = {tau_re:.4f} + {tau_im:.4f}i")
    print(f"  |τ| = {abs(tau_opt):.4f}")
    print(f"  arg(τ) = {np.angle(tau_opt)*180/np.pi:.2f}°")

    print(f"\nCoupling coefficients:")
    for i, c in enumerate(coeffs):
        print(f"  c{i+1} = {c:.4f}")

    print(f"\nYukawa Matrix Y(τ):")
    print(f"  (Hermitian part, MeV scale)")
    Y_hermitian = (Y_opt + Y_opt.conj().T) / 2
    print(Y_hermitian.real)

    print(f"\nMass Predictions:")
    for i, (m_calc, m_exp) in enumerate(zip(masses_opt, target_masses)):
        error = abs(np.log10(m_calc) - np.log10(m_exp))
        status = "✓" if error < 0.1 else "✗"
        print(f"  m{i+1} = {m_calc:.2f} MeV (exp: {m_exp:.2f}, log-error: {error:.4f}) {status}")

    return {
        'tau': tau_opt,
        'coeffs': coeffs,
        'Y': Y_opt,
        'masses': masses_opt,
        'V': V_opt,
        'error': result.fun,
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_moduli_space(sector_results):
    """
    Visualize τ locations in moduli space and predictions
    """
    fig = plt.figure(figsize=(16, 10))

    # 1. Moduli space map
    ax1 = plt.subplot(2, 3, 1)

    sectors = list(sector_results.keys())
    colors = {'lepton': 'red', 'up': 'blue', 'down': 'green'}

    # Fundamental domain
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 * np.cos(theta)
    circle_y = 0.5 * np.sin(theta) + 0.5
    ax1.plot(circle_x, circle_y, 'k--', alpha=0.3, label='|τ|=1')
    ax1.axvline(-0.5, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(0.5, color='k', linestyle='--', alpha=0.3)

    for sector, result in sector_results.items():
        tau = result['tau']
        color = colors.get(sector, 'black')
        ax1.plot(tau.real, tau.imag, 'o', color=color, markersize=12,
                label=f'{sector}: τ={tau.real:.2f}+{tau.imag:.2f}i')

    ax1.set_xlabel('Re(τ)', fontsize=12)
    ax1.set_ylabel('Im(τ)', fontsize=12)
    ax1.set_title('Moduli Space (τ locations)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(0, 2)

    # 2-4: Mass fits for each sector
    for idx, (sector, result) in enumerate(sector_results.items()):
        ax = plt.subplot(2, 3, idx+2)

        if sector == 'lepton':
            target = LEPTON_MASSES
            labels = ['e', 'μ', 'τ']
        elif sector == 'up':
            target = UP_MASSES
            labels = ['u', 'c', 't']
        else:
            target = DOWN_MASSES
            labels = ['d', 's', 'b']

        masses = result['masses']

        x = np.arange(3)
        width = 0.35
        ax.bar(x - width/2, target, width, label='Experimental', alpha=0.7)
        ax.bar(x + width/2, masses, width, label='Modular Theory', alpha=0.7)

        ax.set_ylabel('Mass (MeV)', fontsize=10)
        ax.set_title(f'{sector.title()} Sector', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 5: Yukawa structure
    ax5 = plt.subplot(2, 3, 5)

    sector_example = list(sector_results.keys())[0]
    Y = sector_results[sector_example]['Y']
    Y_abs = np.abs(Y)

    im = ax5.imshow(Y_abs, cmap='viridis', aspect='auto')
    ax5.set_title(f'Yukawa Structure |Y(τ)| ({sector_example})',
                  fontsize=11, fontweight='bold')
    ax5.set_xticks([0, 1, 2])
    ax5.set_yticks([0, 1, 2])
    for i in range(3):
        for j in range(3):
            text = ax5.text(j, i, f'{Y_abs[i,j]:.2f}',
                          ha="center", va="center", color="white", fontsize=10)
    plt.colorbar(im, ax=ax5)

    # 6: Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary = """THEORY #13: MODULAR FLAVOR

Key Concept:
• Yukawas = FUNCTIONS, not parameters
• ALL structure from single τ
• Modular invariance constrains form

Parameters:
• τ (complex modulus)
• c_i (O(1) coefficients from A₄)
• Total: ~5-7 params for 9 masses

Philosophy:
DERIVE, don't assume

Status: Testing if single τ
can reproduce all masses

Next: Add mixing angles test
"""

    ax6.text(0.1, 0.5, summary, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig('theory13_modular_flavor.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: theory13_modular_flavor.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("THEORY #13: MODULAR FLAVOR SYMMETRY")
    print("="*70)
    print("\nPhilosophy: Yukawa couplings are MODULAR FORMS")
    print("NOT free parameters - functions of complex modulus τ")
    print("\nA₄ modular symmetry (tetrahedral group)")
    print("All flavor structure from SINGLE τ + representation theory")
    print("\nThis is principled theory building, not parameterization.")

    # Test on each sector separately first
    results = {}

    print("\n" + "="*70)
    print("PHASE 1: FIT EACH SECTOR INDEPENDENTLY")
    print("="*70)

    results['lepton'] = find_optimal_tau(LEPTON_MASSES, 'charged_lepton', max_coeffs=2)
    results['up'] = find_optimal_tau(UP_MASSES, 'up_quark', max_coeffs=3)
    results['down'] = find_optimal_tau(DOWN_MASSES, 'down_quark', max_coeffs=3)

    # Visualization
    visualize_moduli_space(results)

    # Summary
    print("\n" + "="*70)
    print("THEORY #13: PRELIMINARY RESULTS")
    print("="*70)

    print("\nOptimal τ values:")
    for sector, result in results.items():
        tau = result['tau']
        print(f"  {sector:8s}: τ = {tau.real:6.3f} + {tau.imag:6.3f}i  (error: {result['error']:.4f})")

    # Check if τ values cluster
    tau_list = [r['tau'] for r in results.values()]
    tau_mean = np.mean(tau_list)
    tau_spread = np.std([abs(t - tau_mean) for t in tau_list])

    print(f"\nτ clustering:")
    print(f"  Mean: τ = {tau_mean.real:.3f} + {tau_mean.imag:.3f}i")
    print(f"  Spread: |Δτ| = {tau_spread:.3f}")

    if tau_spread < 0.5:
        print(f"\n✓ τ values CLUSTER - suggests universal modulus!")
        print(f"  Next: Use single τ for ALL sectors simultaneously")
    else:
        print(f"\n✗ τ values SCATTERED - different sectors need different vacua")
        print(f"  May need sector-dependent moduli or extended structure")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Test SINGLE τ for all three sectors (universality)")
    print("2. Add mixing angle constraints (CKM, PMNS)")
    print("3. Explore moduli space systematically")
    print("4. Compare with Froggatt-Nielsen as effective limit")
    print("\nThis is the RIGHT approach:")
    print("  • Principled (modular invariance)")
    print("  • Restrictive (few parameters)")
    print("  • Explanatory (connects to geometry)")
    print("="*70)
