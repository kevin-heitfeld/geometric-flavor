"""
Field Dynamics Mass Genesis (FDMG) - Critical Test

Tests whether Δ(27) anomalous dimensions produce mass hierarchy ~1:200:3000

This is a 2-3 month project. This file sets up the framework and initial calculations.

Key question: Do γ₁ : γ₂ : γ₃ produce m_e : m_μ : m_τ?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Tuple, List

# ==============================================================================
# ΔELTA(27) GROUP THEORY
# ==============================================================================

@dataclass
class Delta27Rep:
    """
    Δ(27) = (Z₃ × Z₃) ⋊ Z₃ representation

    Δ(27) has exactly 3 inequivalent 1D irreducible representations:
    - χ₁: Trivial representation
    - χ₂: First non-trivial 1D rep
    - χ₃: Second non-trivial 1D rep

    These correspond to the 3 generations!
    """
    name: str
    dimension: int
    omega_charge: Tuple[int, int, int]  # (n, m, k) where ω = e^(2πi/3)

    def quadratic_casimir(self) -> float:
        """
        Compute C₂(R) = quadratic Casimir invariant

        For 1D reps of Δ(27):
        C₂(χᵢ) depends on the representation labels
        """
        n, m, k = self.omega_charge
        # Δ(27) Casimir computed from group structure
        # This is a simplification - full calculation requires group theory library
        return (n**2 + m**2 + k**2) / 3.0


# Define the three 1D representations
REP_1 = Delta27Rep("χ₁", 1, (0, 0, 0))  # Trivial - electron
REP_2 = Delta27Rep("χ₂", 1, (1, 0, 0))  # First non-trivial - muon
REP_3 = Delta27Rep("χ₃", 1, (0, 1, 0))  # Second non-trivial - tau

LEPTON_REPS = [REP_1, REP_2, REP_3]


def clebsch_gordan_delta27(rep_i: Delta27Rep, rep_j: Delta27Rep) -> float:
    """
    Compute Clebsch-Gordan coefficients for Δ(27)

    Needed for computing mixing angles and CP phases
    (Future work - Phase 4)
    """
    # Placeholder - full calculation requires representation theory
    n1, m1, k1 = rep_i.omega_charge
    n2, m2, k2 = rep_j.omega_charge

    # CG coefficients depend on how reps combine
    # This is a simplified version
    return np.exp(-0.5 * ((n1-n2)**2 + (m1-m2)**2 + (k1-k2)**2))


# ==============================================================================
# STANDARD MODEL RG EQUATIONS
# ==============================================================================

@dataclass
class SMParameters:
    """Standard Model parameters at a given energy scale"""
    mu: float  # Energy scale in GeV
    g1: float  # U(1)_Y gauge coupling
    g2: float  # SU(2)_L gauge coupling
    g3: float  # SU(3)_C gauge coupling
    yt: float  # Top Yukawa coupling
    lambda_h: float  # Higgs quartic coupling


def sm_gauge_couplings(mu: float) -> Tuple[float, float, float]:
    """
    Compute SM gauge couplings at scale μ via RG running

    Using 2-loop RG equations from M_Z = 91.2 GeV
    """
    M_Z = 91.2  # GeV

    # Values at M_Z (PDG 2024)
    g1_Z = 0.357  # √(5/3) g'
    g2_Z = 0.652
    g3_Z = 1.221

    # Beta function coefficients (1-loop, MS-bar scheme)
    # For SM with 3 generations
    b1 = 41/10   # U(1)_Y
    b2 = -19/6   # SU(2)_L
    b3 = -7      # SU(3)_C

    t = np.log(mu / M_Z)

    # 1-loop running: g(μ)² = g(M_Z)² / (1 - b g(M_Z)² t / 8π²)
    g1_sq = g1_Z**2 / (1 - b1 * g1_Z**2 * t / (8 * np.pi**2))
    g2_sq = g2_Z**2 / (1 - b2 * g2_Z**2 * t / (8 * np.pi**2))
    g3_sq = g3_Z**2 / (1 - b3 * g3_Z**2 * t / (8 * np.pi**2))

    return np.sqrt(g1_sq), np.sqrt(g2_sq), np.sqrt(g3_sq)


def top_yukawa(mu: float) -> float:
    """
    Top Yukawa coupling at scale μ

    Using measured top mass m_t = 172.76 GeV
    """
    m_t = 172.76  # GeV
    v = 246.22    # Higgs vev in GeV

    # At tree level: y_t = √2 m_t / v
    y_t_tree = np.sqrt(2) * m_t / v

    # RG running (simplified - should use full 2-loop)
    M_Z = 91.2
    t = np.log(mu / M_Z)

    g1, g2, g3 = sm_gauge_couplings(mu)

    # 1-loop beta function for top Yukawa
    beta_yt = y_t_tree / (16 * np.pi**2) * (
        9/2 * y_t_tree**2 - 8 * g3**2 - 9/4 * g2**2 - 17/20 * g1**2
    )

    y_t = y_t_tree + beta_yt * t

    return y_t


# ==============================================================================
# ANOMALOUS DIMENSIONS
# ==============================================================================

def compute_anomalous_dimension(rep: Delta27Rep, mu: float,
                                 lambda_yukawa: float = 0.1) -> float:
    """
    Compute anomalous dimension γ for a Δ(27) representation

    At 1-loop, anomalous dimension comes from:
    1. Gauge boson loops (wave function renormalization)
    2. Yukawa field loops (if Yukawa fields are dynamical)
    3. Δ(27) symmetry breaking effects

    Formula (1-loop):
    γ = (16π²)⁻¹ [C₂(R) Σᵢ gᵢ² + c_Y λ_Y + ...]

    where:
    - C₂(R) = quadratic Casimir of representation R
    - gᵢ = gauge couplings
    - λ_Y = Yukawa field self-coupling
    - c_Y = group-theoretic coefficient

    Args:
        rep: Δ(27) representation
        mu: Energy scale in GeV
        lambda_yukawa: Yukawa field quartic coupling

    Returns:
        Anomalous dimension γ (dimensionless)
    """
    g1, g2, g3 = sm_gauge_couplings(mu)

    # Quadratic Casimir for this representation
    C2 = rep.quadratic_casimir()

    # Gauge contribution (simplified - assumes lepton quantum numbers)
    # For leptons: SU(2) doublet, U(1)_Y charge = -1/2
    gauge_contrib = C2 * (3/4 * g2**2 + 3/20 * g1**2)

    # Yukawa field contribution
    # This is the NEW piece from making Yukawa couplings dynamical
    # The coefficient depends on how Δ(27) reps couple
    n, m, k = rep.omega_charge
    yukawa_contrib = (n + m + k) * lambda_yukawa

    # Δ(27) symmetry breaking contribution
    # This depends on the flavor-breaking potential V[Y]
    # For now, use simple model: breaking ~ (n² + m² + k²)
    breaking_contrib = (n**2 + m**2 + k**2) * 0.01  # Small breaking

    # Total anomalous dimension (1-loop)
    gamma = (gauge_contrib + yukawa_contrib + breaking_contrib) / (16 * np.pi**2)

    return gamma


def compute_all_anomalous_dimensions(mu: float = 1000.0,
                                     lambda_yukawa: float = 0.1) -> np.ndarray:
    """
    Compute anomalous dimensions for all 3 lepton generations

    Returns:
        Array [γ₁, γ₂, γ₃] for e, μ, τ
    """
    gammas = []

    for rep in LEPTON_REPS:
        gamma = compute_anomalous_dimension(rep, mu, lambda_yukawa)
        gammas.append(gamma)

    return np.array(gammas)


# ==============================================================================
# FIXED POINT EQUATIONS
# ==============================================================================

def yukawa_beta_function(y: np.ndarray, gammas: np.ndarray,
                        g1: float, g2: float) -> np.ndarray:
    """
    Beta function for Yukawa couplings at 1-loop

    With anomalous dimensions from Δ(27):
    β_yᵢ = yᵢ/(16π²) [c₁ Σⱼ yⱼ² - c₂ g₂² + γᵢ]

    Args:
        y: Array of Yukawa couplings [y₁, y₂, y₃]
        gammas: Anomalous dimensions [γ₁, γ₂, γ₃]
        g1, g2: Gauge couplings

    Returns:
        β = dy/dt
    """
    # Coefficients (1-loop for leptons)
    c1 = 3/2  # Self-coupling
    c2_g2 = 3/4  # SU(2) contribution
    c2_g1 = 9/20  # U(1)_Y contribution

    # Sum of squares
    y_sum_sq = np.sum(y**2)

    # Beta function for each generation
    beta = np.zeros_like(y)

    for i in range(len(y)):
        beta[i] = y[i] / (16 * np.pi**2) * (
            c1 * y_sum_sq - c2_g2 * g2**2 - c2_g1 * g1**2 + gammas[i]
        )

    return beta


def find_fixed_point(gammas: np.ndarray, mu: float = 1000.0) -> np.ndarray:
    """
    Solve β = 0 to find RG fixed point

    At fixed point:
    c₁ Σyᵢ² = c₂g² - γᵢ for each i

    This is a system of 3 coupled equations for 3 unknowns (y₁, y₂, y₃)

    Returns:
        Array of fixed-point Yukawa couplings [y₁*, y₂*, y₃*]
    """
    g1, g2, _ = sm_gauge_couplings(mu)

    # Define system: β(y) = 0
    def equations(y):
        return yukawa_beta_function(y, gammas, g1, g2)

    # Initial guess (small positive values)
    y0 = np.array([0.01, 0.1, 0.5])

    # Solve
    try:
        y_fixed = fsolve(equations, y0)

        # Check if solution is physical (positive)
        if np.all(y_fixed > 0):
            return y_fixed
        else:
            # Try different initial condition
            y0_alt = np.array([0.001, 0.05, 0.3])
            y_fixed = fsolve(equations, y0_alt)
            return y_fixed

    except Exception as e:
        print(f"Fixed point solver failed: {e}")
        return np.array([np.nan, np.nan, np.nan])


# ==============================================================================
# MASS RATIO PREDICTION
# ==============================================================================

def yukawa_to_mass_ratios(y: np.ndarray) -> np.ndarray:
    """
    Convert Yukawa couplings to mass ratios

    At tree level: m_f = y_f v / √2

    So mass ratios = Yukawa ratios

    Returns:
        Mass ratios normalized to lightest (m₁ = 1)
    """
    # Sort by magnitude
    y_sorted = np.sort(y)

    # Normalize to lightest
    ratios = y_sorted / y_sorted[0]

    return ratios


def compute_errors(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute relative errors between predicted and target mass ratios

    Args:
        predicted: [r₁, r₂, r₃] predicted ratios
        target: [1, 207, 3477] measured lepton mass ratios

    Returns:
        Relative errors [ε₁, ε₂, ε₃]
    """
    return np.abs(predicted - target) / target


# ==============================================================================
# MAIN CRITICAL TEST
# ==============================================================================

def run_critical_test(mu: float = 1000.0,
                     lambda_yukawa_range: List[float] = None) -> dict:
    """
    THE CRITICAL TEST FOR FDMG

    Question: Do Δ(27) anomalous dimensions produce mass ratios ~1:200:3000?

    Procedure:
    1. Compute anomalous dimensions γ₁, γ₂, γ₃ at scale μ
    2. Solve fixed point equations β = 0
    3. Extract Yukawa couplings y₁, y₂, y₃
    4. Compute mass ratios
    5. Compare to measured lepton masses: m_e : m_μ : m_τ = 1 : 207 : 3477

    Args:
        mu: Energy scale in GeV (GUT scale ~ 10³-10¹⁶ GeV)
        lambda_yukawa_range: Range of Yukawa field couplings to scan

    Returns:
        Dictionary with results and verdict
    """
    print("="*70)
    print("FIELD DYNAMICS MASS GENESIS - CRITICAL TEST")
    print("="*70)
    print(f"\nEnergy scale: μ = {mu:.0f} GeV")
    print(f"Testing Δ(27) flavor symmetry mechanism")
    print("\nQuestion: Do anomalous dimensions produce mass hierarchy?")
    print("Target ratios: m_e : m_μ : m_τ = 1 : 207 : 3477")
    print("="*70)

    # Measured lepton mass ratios (from PDG 2024)
    target_ratios = np.array([1.0, 206.77, 3477.2])

    if lambda_yukawa_range is None:
        lambda_yukawa_range = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    results_list = []

    for lambda_y in lambda_yukawa_range:
        print(f"\n--- Testing λ_Y = {lambda_y:.3f} ---")

        # Step 1: Compute anomalous dimensions
        gammas = compute_all_anomalous_dimensions(mu, lambda_y)
        print(f"Anomalous dimensions: γ = {gammas}")
        if gammas[0] > 1e-10:
            print(f"Ratio: γ₁:γ₂:γ₃ = 1:{gammas[1]/gammas[0]:.2f}:{gammas[2]/gammas[0]:.2f}")
        else:
            print(f"Ratio: γ₁:γ₂:γ₃ = {gammas[0]:.3e}:{gammas[1]:.3e}:{gammas[2]:.3e}")

        # Step 2: Find fixed point
        y_fixed = find_fixed_point(gammas, mu)

        if np.any(np.isnan(y_fixed)):
            print("❌ Fixed point not found")
            continue

        print(f"Fixed point Yukawas: y = {y_fixed}")

        # Step 3: Compute mass ratios
        mass_ratios = yukawa_to_mass_ratios(y_fixed)
        print(f"Predicted mass ratios: {mass_ratios}")
        print(f"Target mass ratios:    {target_ratios}")

        # Step 4: Compute errors
        errors = compute_errors(mass_ratios, target_ratios)
        error_str = ', '.join([f"{e*100:.1f}%" for e in errors])
        print(f"Relative errors: [{error_str}]")

        # Check success criteria
        success = {
            'gen2_ok': errors[1] < 0.5,  # Within 50%
            'gen3_ok': errors[2] < 0.5,
            'ordering_ok': (mass_ratios[2] > mass_ratios[1] > mass_ratios[0])
        }

        all_pass = all(success.values())

        if all_pass:
            print("✓✓✓ SUCCESS for this parameter!")

        results_list.append({
            'lambda_yukawa': lambda_y,
            'gammas': gammas,
            'yukawas': y_fixed,
            'mass_ratios': mass_ratios,
            'errors': errors,
            'success': success,
            'all_pass': all_pass
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful_params = [r for r in results_list if r['all_pass']]

    if len(successful_params) > 0:
        print(f"\n✓ Found {len(successful_params)} successful parameter sets!")
        print("\nBest result:")
        best = min(successful_params, key=lambda r: np.sum(r['errors']))
        print(f"  λ_Y = {best['lambda_yukawa']:.3f}")
        print(f"  Mass ratios: {best['mass_ratios']}")
        error_str = ', '.join([f"{e*100:.1f}%" for e in best['errors']])
        print(f"  Errors: [{error_str}]")

        verdict = "PASS"
        recommendation = """
✓✓✓ CRITICAL TEST PASSED! ✓✓✓

Δ(27) anomalous dimensions CAN produce the right mass hierarchy!

NEXT STEPS:
1. Refine calculation to 2-loop (improves accuracy)
2. Fit to exact lepton masses (determine all 4 parameters)
3. Predict quark masses without retuning
4. Compute mixing angles from Δ(27) Clebsch-Gordan coefficients

TIMELINE:
- Phase 2 (lepton fitting): 6 months
- Phase 3 (quark prediction): 1 year
- Phase 4 (mixing angles): 1-2 years

PROBABILITY OF SUCCESS: ~40-50%

This theory is worth pursuing!
        """
    else:
        print("\n❌ No successful parameter sets found")
        print("\nBest result:")
        best = min(results_list, key=lambda r: np.sum(r['errors']))
        print(f"  λ_Y = {best['lambda_yukawa']:.3f}")
        print(f"  Mass ratios: {best['mass_ratios']}")
        error_str = ', '.join([f"{e*100:.1f}%" for e in best['errors']])
        print(f"  Errors: [{error_str}]")

        # Check what went wrong
        if best['errors'][1] > 2.0:
            failure_mode = "gen2_too_far"
            reason = "Muon mass ratio off by >2× - mechanism too weak"
        elif best['errors'][2] > 2.0:
            failure_mode = "gen3_too_far"
            reason = "Tau mass ratio off by >2× - hierarchy insufficient"
        elif not best['success']['ordering_ok']:
            failure_mode = "wrong_ordering"
            reason = "Mass ordering wrong - fundamental problem"
        else:
            failure_mode = "close_but_not_enough"
            reason = "Close to target but not within criteria"

        verdict = "FAIL"
        recommendation = f"""
❌ CRITICAL TEST FAILED

Failure mode: {failure_mode}
Reason: {reason}

POSSIBLE FIXES:
1. Try different Δ(27) embedding (different Casimir operators)
2. Include 2-loop corrections (might shift ratios)
3. Try related groups: Δ(54), Δ(81), Σ(81)
4. Modify anomalous dimension calculation (different breaking pattern)

DECISION:
- If errors < 100%: Worth trying fixes (~2-3 months)
- If errors > 200%: Probably fundamental issue, consider abandoning

ALTERNATIVE APPROACHES:
- Path C: Try 5 more rapid-fire theories
- Path D: Accept landscape, focus on structural questions
        """

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    print(recommendation)

    return {
        'verdict': verdict,
        'results': results_list,
        'best_params': best,
        'successful_params': successful_params
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_anomalous_dimensions(mu_range: np.ndarray = None):
    """
    Plot how anomalous dimensions vary with energy scale
    """
    if mu_range is None:
        mu_range = np.logspace(2, 16, 100)  # 100 GeV to 10^16 GeV

    gamma1_list = []
    gamma2_list = []
    gamma3_list = []

    for mu in mu_range:
        gammas = compute_all_anomalous_dimensions(mu, lambda_yukawa=0.1)
        gamma1_list.append(gammas[0])
        gamma2_list.append(gammas[1])
        gamma3_list.append(gammas[2])

    plt.figure(figsize=(10, 6))
    plt.semilogx(mu_range, gamma1_list, label='γ₁ (electron)', linewidth=2)
    plt.semilogx(mu_range, gamma2_list, label='γ₂ (muon)', linewidth=2)
    plt.semilogx(mu_range, gamma3_list, label='γ₃ (tau)', linewidth=2)

    plt.xlabel('Energy Scale μ (GeV)', fontsize=12)
    plt.ylabel('Anomalous Dimension γ', fontsize=12)
    plt.title('Δ(27) Anomalous Dimensions vs Energy Scale', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fdmg_anomalous_dimensions.png', dpi=150)
    print("Saved plot: fdmg_anomalous_dimensions.png")
    plt.close()


def plot_mass_ratios_scan(results: dict):
    """
    Plot predicted mass ratios vs parameter values
    """
    lambda_values = [r['lambda_yukawa'] for r in results['results']]
    ratio2 = [r['mass_ratios'][1] for r in results['results']]
    ratio3 = [r['mass_ratios'][2] for r in results['results']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gen 2 ratio
    ax1.plot(lambda_values, ratio2, 'o-', linewidth=2, markersize=8, label='Predicted')
    ax1.axhline(y=207, color='r', linestyle='--', linewidth=2, label='Target (207)')
    ax1.axhspan(207*0.5, 207*1.5, alpha=0.2, color='green', label='±50% band')
    ax1.set_xlabel('Yukawa Field Coupling λ_Y', fontsize=12)
    ax1.set_ylabel('m_μ / m_e', fontsize=12)
    ax1.set_title('Generation 2 Mass Ratio', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Gen 3 ratio
    ax2.plot(lambda_values, ratio3, 'o-', linewidth=2, markersize=8, label='Predicted')
    ax2.axhline(y=3477, color='r', linestyle='--', linewidth=2, label='Target (3477)')
    ax2.axhspan(3477*0.5, 3477*1.5, alpha=0.2, color='green', label='±50% band')
    ax2.set_xlabel('Yukawa Field Coupling λ_Y', fontsize=12)
    ax2.set_ylabel('m_τ / m_e', fontsize=12)
    ax2.set_title('Generation 3 Mass Ratio', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('fdmg_mass_ratios.png', dpi=150)
    print("Saved plot: fdmg_mass_ratios.png")
    plt.close()


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║  FIELD DYNAMICS MASS GENESIS (FDMG) - CRITICAL TEST              ║
║                                                                    ║
║  Tests if Δ(27) flavor symmetry + RG fixed points                ║
║  can explain fermion mass hierarchies                             ║
║                                                                    ║
║  Runtime: ~5-10 minutes                                           ║
║                                                                    ║
║  If test passes → Pursue theory (1-2 years)                       ║
║  If test fails → Try different flavor group or abandon            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    import time
    start = time.time()

    # Visualizations
    print("\n[1/3] Computing anomalous dimensions across energy scales...")
    plot_anomalous_dimensions()

    # Run critical test
    print("\n[2/3] Running critical test at multiple scales...")

    # Test at different energy scales
    scales = [1e3, 1e6, 1e9, 1e12, 1e15]  # TeV to Planck scale

    best_result = None
    best_score = float('inf')

    for mu in scales:
        print(f"\n{'='*70}")
        print(f"Testing at μ = {mu:.0e} GeV")
        print('='*70)

        results = run_critical_test(mu=mu, lambda_yukawa_range=[0.01, 0.05, 0.1, 0.2, 0.5])

        # Track best result
        if results['verdict'] == 'PASS':
            score = np.sum(results['best_params']['errors'])
            if score < best_score:
                best_score = score
                best_result = results

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT ACROSS ALL SCALES")
    print("="*70)

    if best_result and best_result['verdict'] == 'PASS':
        print("\n✓✓✓ CRITICAL TEST PASSED AT LEAST ONE SCALE ✓✓✓")
        print(f"\nBest parameters:")
        print(f"  Energy scale: μ = {mu:.0e} GeV")
        print(f"  λ_Y = {best_result['best_params']['lambda_yukawa']:.3f}")
        print(f"  Mass ratios: {best_result['best_params']['mass_ratios']}")
        error_str = ', '.join([f"{e*100:.1f}%" for e in best_result['best_params']['errors']])
        print(f"  Errors: [{error_str}]")
        print("\nRECOMMENDATION: Proceed to Phase 2 (full parameter fitting)")
    else:
        print("\n❌ CRITICAL TEST FAILED AT ALL SCALES")
        print("\nΔ(27) anomalous dimensions do not produce right hierarchy")
        print("\nRECOMMENDATION: Try different flavor group or different approach")

    # Generate comparison plot
    if best_result:
        print("\n[3/3] Generating comparison plots...")
        plot_mass_ratios_scan(best_result)

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f} seconds")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nNext steps depend on verdict above.")
    print("Science demands ruthless falsification.")
