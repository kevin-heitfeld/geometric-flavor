"""
Gauge Coupling Unification - TRULY FINAL VERSION
==================================================

Using CORRECT convention: α⁻¹(μ) = α⁻¹(μ₀) - b/(2π) ln(μ/μ₀) [MINUS]

With this:
- b > 0 → α⁻¹ decreases → coupling STRENGTHENS (not asymptotic free) ✓
- b < 0 → α⁻¹ increases → coupling WEAKENS (asymptotic freedom) ✓
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constants
M_Z = 91.1876  # GeV
alpha_em_inv_MZ = 127.95
sin2_theta_W = 0.23121
alpha_s_MZ = 0.1179

def get_initial():
    """Initial gauge couplings at M_Z in GUT normalization."""
    alpha_em = 1.0 / alpha_em_inv_MZ
    cos2_theta_W = 1 - sin2_theta_W

    # CORRECTED GUT normalization
    alpha_1 = (5.0/3.0) * alpha_em / cos2_theta_W  # Was sin2, should be cos2!
    alpha_2 = alpha_em / sin2_theta_W               # Was cos2, should be sin2!
    alpha_3 = alpha_s_MZ
    
    return 1/alpha_1, 1/alpha_2, 1/alpha_3

def run_coupling(alpha_inv_0, b, M_0, M_final):
    """RG running: α⁻¹(M) = α⁻¹(M₀) - b/(2π) ln(M/M₀) [PDG convention]"""
    t = np.log(M_final / M_0)
    return alpha_inv_0 - (b / (2.0 * np.pi)) * t

def find_unification(a1_inv_0, a2_inv_0, a3_inv_0, b1, b2, b3):
    """Find scale where couplings are closest."""

    def spread_at_scale(log_M):
        M = 10**log_M
        a1 = run_coupling(a1_inv_0, b1, M_Z, M)
        a2 = run_coupling(a2_inv_0, b2, M_Z, M)
        a3 = run_coupling(a3_inv_0, b3, M_Z, M)

        # Penalize if any go negative
        if min(a1, a2, a3) < 0.1:
            return 1e10

        # Compute spread of α values (not α⁻¹)
        alpha1, alpha2, alpha3 = 1/a1, 1/a2, 1/a3
        mean = (alpha1 + alpha2 + alpha3) / 3
        return np.std([alpha1, alpha2, alpha3]) / mean

    # Search in range 10^14 to 10^17 GeV
    result = minimize_scalar(spread_at_scale, bounds=(14, 17), method='bounded')
    M_GUT = 10**result.x

    # Get couplings at unification
    a1_GUT = run_coupling(a1_inv_0, b1, M_Z, M_GUT)
    a2_GUT = run_coupling(a2_inv_0, b2, M_Z, M_GUT)
    a3_GUT = run_coupling(a3_inv_0, b3, M_Z, M_GUT)

    alpha1 = 1/a1_GUT
    alpha2 = 1/a2_GUT
    alpha3 = 1/a3_GUT
    alpha_GUT = (alpha1 + alpha2 + alpha3) / 3
    spread = np.std([alpha1, alpha2, alpha3]) / alpha_GUT * 100

    return M_GUT, alpha_GUT, spread, (a1_GUT, a2_GUT, a3_GUT)

def plot_unification(model, a1_inv_0, a2_inv_0, a3_inv_0, b1, b2, b3, M_GUT):
    """Plot gauge coupling running."""
    log_scales = np.linspace(np.log10(M_Z), 17, 1000)
    scales = 10**log_scales

    a1_inv = [run_coupling(a1_inv_0, b1, M_Z, M) for M in scales]
    a2_inv = [run_coupling(a2_inv_0, b2, M_Z, M) for M in scales]
    a3_inv = [run_coupling(a3_inv_0, b3, M_Z, M) for M in scales]

    plt.figure(figsize=(12, 8))
    plt.plot(scales, a1_inv, 'b-', linewidth=2.5, label=r'$\alpha_1^{-1}$ (U(1)$_Y$)')
    plt.plot(scales, a2_inv, 'g-', linewidth=2.5, label=r'$\alpha_2^{-1}$ (SU(2)$_L$)')
    plt.plot(scales, a3_inv, 'r-', linewidth=2.5, label=r'$\alpha_3^{-1}$ (SU(3)$_c$)')

    plt.axvline(M_GUT, color='purple', linestyle='--', alpha=0.7, linewidth=2,
                label=f'M_GUT = {M_GUT:.2e} GeV')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

    # Mark unification point
    a1_at_GUT = run_coupling(a1_inv_0, b1, M_Z, M_GUT)
    a2_at_GUT = run_coupling(a2_inv_0, b2, M_Z, M_GUT)
    a3_at_GUT = run_coupling(a3_inv_0, b3, M_Z, M_GUT)

    plt.scatter([M_GUT]*3, [a1_at_GUT, a2_at_GUT, a3_at_GUT],
                s=150, c=['blue', 'green', 'red'], marker='o',
                edgecolors='black', linewidths=2, zorder=5)

    plt.xscale('log')
    plt.xlabel(r'Energy Scale $\mu$ [GeV]', fontsize=14, fontweight='bold')
    plt.ylabel(r'$\alpha_i^{-1}$', fontsize=14, fontweight='bold')
    plt.title(f'Gauge Coupling Unification: {model}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(M_Z, 10**17)

    # Set y-limits to show relevant range
    all_vals = a1_inv + a2_inv + a3_inv
    valid_vals = [v for v in all_vals if v > -10 and v < 100]
    if valid_vals:
        plt.ylim(min(valid_vals)-5, max(valid_vals)+5)

    plt.tight_layout()
    plt.savefig(f'unification_{model.lower().replace(" ", "_")}.png', dpi=200)
    plt.close()

def main():
    print("="*70)
    print("GAUGE COUPLING UNIFICATION - PHASE 1 TEST")
    print("="*70)

    a1_inv_0, a2_inv_0, a3_inv_0 = get_initial()

    print(f"\nInitial couplings at M_Z = {M_Z:.2f} GeV (GUT normalized):")
    print(f"  α_1^-1 = {a1_inv_0:.3f}  →  α_1 = {1/a1_inv_0:.6f}")
    print(f"  α_2^-1 = {a2_inv_0:.3f}  →  α_2 = {1/a2_inv_0:.6f}")
    print(f"  α_3^-1 = {a3_inv_0:.3f}  →  α_3 = {1/a3_inv_0:.6f}")

    # Standard Model
    print("\n" + "="*70)
    print("STANDARD MODEL")
    print("="*70)
    b1_sm, b2_sm, b3_sm = 41.0/10.0, -19.0/6.0, -7.0  # Standard values
    print(f"Beta coefficients: b₁={b1_sm:+.2f}, b₂={b2_sm:+.2f}, b₃={b3_sm:+.2f}")

    M_GUT_sm, alpha_GUT_sm, spread_sm, (a1_sm, a2_sm, a3_sm) = \
        find_unification(a1_inv_0, a2_inv_0, a3_inv_0, b1_sm, b2_sm, b3_sm)

    print(f"\nUnification at M_GUT = {M_GUT_sm:.3e} GeV (10^{np.log10(M_GUT_sm):.2f})")
    print(f"  α_1 = {1/a1_sm:.6f}  (α_1^-1 = {a1_sm:.2f})")
    print(f"  α_2 = {1/a2_sm:.6f}  (α_2^-1 = {a2_sm:.2f})")
    print(f"  α_3 = {1/a3_sm:.6f}  (α_3^-1 = {a3_sm:.2f})")
    print(f"  α_GUT = {alpha_GUT_sm:.6f}  ±{spread_sm:.1f}%")

    if alpha_GUT_sm > 0:
        g_s_sm = np.sqrt(4 * np.pi * alpha_GUT_sm)
        print(f"  g_s (k=1) = {g_s_sm:.4f}")

    plot_unification("Standard Model", a1_inv_0, a2_inv_0, a3_inv_0,
                     b1_sm, b2_sm, b3_sm, M_GUT_sm)
    print("  Plot: unification_standard_model.png")

    # MSSM
    print("\n" + "="*70)
    print("MINIMAL SUPERSYMMETRIC SM (MSSM)")
    print("="*70)
    b1_mssm, b2_mssm, b3_mssm = 33.0/5.0, 1.0, -3.0  # Standard MSSM values
    print(f"Beta coefficients: b₁={b1_mssm:+.2f}, b₂={b2_mssm:+.2f}, b₃={b3_mssm:+.2f}")

    M_GUT_mssm, alpha_GUT_mssm, spread_mssm, (a1_mssm, a2_mssm, a3_mssm) = \
        find_unification(a1_inv_0, a2_inv_0, a3_inv_0, b1_mssm, b2_mssm, b3_mssm)

    print(f"\nUnification at M_GUT = {M_GUT_mssm:.3e} GeV (10^{np.log10(M_GUT_mssm):.2f})")
    print(f"  α_1 = {1/a1_mssm:.6f}  (α_1^-1 = {a1_mssm:.2f})")
    print(f"  α_2 = {1/a2_mssm:.6f}  (α_2^-1 = {a2_mssm:.2f})")
    print(f"  α_3 = {1/a3_mssm:.6f}  (α_3^-1 = {a3_mssm:.2f})")
    print(f"  α_GUT = {alpha_GUT_mssm:.6f}  ±{spread_mssm:.1f}%")

    if alpha_GUT_mssm > 0:
        g_s_mssm = np.sqrt(4 * np.pi * alpha_GUT_mssm)
        print(f"  g_s (k=1) = {g_s_mssm:.4f}")

    plot_unification("MSSM", a1_inv_0, a2_inv_0, a3_inv_0,
                     b1_mssm, b2_mssm, b3_mssm, M_GUT_mssm)
    print("  Plot: unification_mssm.png")

    # Summary
    print("\n" + "="*70)
    print("PHASE 1 ASSESSMENT")
    print("="*70)

    print(f"\nStandard Model:")
    print(f"  Unification: {'POOR' if spread_sm > 50 else 'MODERATE' if spread_sm > 10 else 'GOOD'}")
    print(f"  Spread: ±{spread_sm:.1f}%")
    print(f"  g_s ≈ {g_s_sm:.2f} (if k=1)")

    print(f"\nMSSM:")
    print(f"  Unification: {'POOR' if spread_mssm > 50 else 'MODERATE' if spread_mssm > 10 else 'EXCELLENT' if spread_mssm < 3 else 'GOOD'}")
    print(f"  Spread: ±{spread_mssm:.1f}%")
    print(f"  g_s ≈ {g_s_mssm:.2f} (if k=1)")

    if spread_mssm < 5:
        print("\n✓ Phase 1 SUCCESS: MSSM shows excellent unification!")
        print(f"  → Gauge couplings constrain α_GUT = {alpha_GUT_mssm:.4f}")
        print(f"  → This suggests g_s ≈ {g_s_mssm:.2f} (assuming k=1)")
        print(f"\nNext: Phase 2 - Check if g_s ~ {g_s_mssm:.1f} is consistent with:")
        print("  • τ = 2.69i framework")
        print("  • Instanton calculations")
        print("  • M_GUT vs M_string")
    elif spread_mssm < 15:
        print("\n≈ Phase 1 MODERATE: Unification exists but needs refinement")
        print("  Possible improvements:")
        print("  • 2-loop RG corrections")
        print("  • Threshold corrections at M_SUSY")
        print("  • GUT-scale threshold effects")
    else:
        print("\n✗ Phase 1 INCONCLUSIVE: Poor unification quality")
        print("  May need to abandon this approach")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
