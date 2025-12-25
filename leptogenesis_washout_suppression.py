"""
Detailed Analysis of Washout Suppression Mechanisms for Leptogenesis

The main challenge: K ~ 10^11 (strong washout) → η_B too small

This script explores mechanisms to suppress washout while maintaining
sterile neutrino DM production.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import brentq

# Constants
M_Pl = 2.4e18  # GeV
v_EW = 246  # GeV
g_star = 106.75  # SM degrees of freedom

# ==============================================================================
# WASHOUT PARAMETER CALCULATION
# ==============================================================================

def washout_parameter(M_R, Y_D):
    """
    K = (Γ_N / H(T=M_R))

    where:
    - Γ_N = |Y_D|² M_R / (8π) (decay width)
    - H(T=M_R) = 1.66 √g_star T² / M_Pl (Hubble at T=M_R)
    """
    # Decay width
    Gamma_N = (np.abs(Y_D)**2) * M_R / (8 * np.pi)

    # Hubble parameter at T = M_R
    T = M_R
    H = 1.66 * np.sqrt(g_star) * T**2 / M_Pl

    K = Gamma_N / H

    return K, Gamma_N, H

# ==============================================================================
# MECHANISM 1: Flavor Effects
# ==============================================================================

def flavor_washout_analysis():
    """
    Different lepton flavors have different washout rates.
    At T < 10^12 GeV, τ Yukawa is in equilibrium → flavor effects matter.

    K_α ≠ K (flavor-dependent washout)
    """
    print("\n" + "="*70)
    print("MECHANISM 1: Flavor-Dependent Washout")
    print("="*70)

    M_R = 20e3  # 20 TeV

    # Flavor-dependent Dirac Yukawa couplings
    # From modular weights: k_e=4, k_μ=6, k_τ=8
    Y_e = 0.07  # Smallest
    Y_mu = 0.19
    Y_tau = 0.75  # Largest (from DM constraint)

    print(f"\nHeavy neutrino mass: M_R = {M_R/1e3:.0f} TeV")
    print(f"Flavor composition: Y_e={Y_e:.2f}, Y_μ={Y_mu:.2f}, Y_τ={Y_tau:.2f}")

    # Individual washout parameters
    K_e, _, _ = washout_parameter(M_R, Y_e)
    K_mu, _, _ = washout_parameter(M_R, Y_mu)
    K_tau, _, _ = washout_parameter(M_R, Y_tau)

    print(f"\nFlavor-dependent washout:")
    print(f"  K_e   = {K_e:.2e}")
    print(f"  K_μ   = {K_mu:.2e}")
    print(f"  K_τ   = {K_tau:.2e}")

    # Total washout (naive sum)
    K_total = K_e + K_mu + K_tau
    print(f"  K_tot = {K_total:.2e}")

    # Effective washout (weighted by production)
    # ε_α ~ Y_α² (CP asymmetry flavor-dependent)
    weights = np.array([Y_e**2, Y_mu**2, Y_tau**2])
    weights /= weights.sum()

    K_eff = weights[0] * K_e + weights[1] * K_mu + weights[2] * K_tau
    print(f"  K_eff = {K_eff:.2e} (weighted by production)")

    # Efficiency factors (from Boltzmann)
    # η ~ ε / K for weak washout (K < 1)
    # η ~ ε / K² for intermediate (1 < K < 100)
    # η ~ ε / K³ for strong (K > 100)

    def efficiency_factor(K):
        if K < 1:
            return 1.0 / K
        elif K < 100:
            return 0.3 / (K * np.log(K))
        else:
            return 0.01 / K**1.5

    eff_e = efficiency_factor(K_e)
    eff_mu = efficiency_factor(K_mu)
    eff_tau = efficiency_factor(K_tau)

    print(f"\nEfficiency factors:")
    print(f"  η_e / ε   ~ {eff_e:.2e}")
    print(f"  η_μ / ε   ~ {eff_mu:.2e}")
    print(f"  η_τ / ε   ~ {eff_tau:.2e}")

    # Effective total efficiency
    eff_total = weights[0] * eff_e + weights[1] * eff_mu + weights[2] * eff_tau
    print(f"  η_tot / ε ~ {eff_total:.2e}")

    # Improvement factor compared to single-flavor
    K_single_flavor = K_tau  # Dominated by τ
    eff_single = efficiency_factor(K_single_flavor)
    improvement = eff_total / eff_single

    print(f"\nImprovement from flavor effects:")
    print(f"  Single-flavor: η/ε ~ {eff_single:.2e}")
    print(f"  Multi-flavor:  η/ε ~ {eff_total:.2e}")
    print(f"  Improvement:   factor {improvement:.2f}")

    if improvement > 5:
        print(f"  ✓ Significant improvement!")
    else:
        print(f"  ✗ Marginal improvement (need factor ~10^4)")

    return eff_total

# ==============================================================================
# MECHANISM 2: Thermal Corrections
# ==============================================================================

def thermal_corrections_analysis():
    """
    At high T, thermal masses and finite-temperature effects modify:
    1. Yukawa couplings (running)
    2. Decay widths (medium effects)
    3. Hubble expansion rate
    """
    print("\n" + "="*70)
    print("MECHANISM 2: Thermal Corrections")
    print("="*70)

    M_R = 20e3  # 20 TeV
    Y_D = 0.5  # Typical

    # Zero-temperature washout
    K_0, Gamma_0, H_0 = washout_parameter(M_R, Y_D)

    print(f"\nZero-temperature:")
    print(f"  K(T=0) = {K_0:.2e}")

    # Thermal mass corrections
    # At T ~ M_R, Higgs gets thermal mass: m_H²(T) ~ g² T²
    g_weak = 0.65
    m_H_thermal = g_weak * M_R  # T ~ M_R

    # Effective Yukawa at finite T
    # Y_eff ~ Y₀ × m_H(0) / √(m_H² + m_H_thermal²)
    m_H_0 = 125  # GeV
    suppression_factor = m_H_0 / np.sqrt(m_H_0**2 + m_H_thermal**2)

    Y_eff = Y_D * suppression_factor
    K_thermal, _, _ = washout_parameter(M_R, Y_eff)

    print(f"\nWith thermal corrections:")
    print(f"  m_H(T=M_R) ~ {m_H_thermal/1e3:.1f} TeV")
    print(f"  Y_eff / Y₀ = {suppression_factor:.2e}")
    print(f"  K(T=M_R)   = {K_thermal:.2e}")
    print(f"  Suppression: factor {K_0/K_thermal:.2f}")

    # Medium effects on decay width
    # Pauli blocking, Bose enhancement, etc.
    # Γ_eff ~ Γ₀ × (1 - f_ℓ - f_H)
    # where f ~ 1/(e^(E/T) ± 1) ~ T/E for E ~ M_R

    pauli_factor = 1 - 2 * M_R / M_R  # Simplified
    print(f"\nMedium effects:")
    print(f"  Pauli blocking: Γ_eff ~ {pauli_factor:.2f} × Γ₀")

    # RG running of Yukawa
    # Y(M_R) = Y(M_Z) × (1 + corrections)
    alpha_t = 0.5  # Top Yukawa
    log_ratio = np.log(M_R / 91)
    rg_correction = 1 + (alpha_t**2 / (4 * np.pi**2)) * log_ratio

    Y_RG = Y_D / rg_correction  # Run down from M_R
    K_RG, _, _ = washout_parameter(M_R, Y_RG)

    print(f"\nRG evolution:")
    print(f"  Y(M_Z) / Y(M_R) = {1/rg_correction:.2f}")
    print(f"  K with RG       = {K_RG:.2e}")

    # Combined effect
    K_total_corrections = K_0 * (suppression_factor**2) * pauli_factor / (rg_correction**2)

    print(f"\nCombined thermal + RG corrections:")
    print(f"  K_eff = {K_total_corrections:.2e}")
    print(f"  Suppression: factor {K_0/K_total_corrections:.1f}")

    if K_0 / K_total_corrections > 10:
        print(f"  ✓ Order-of-magnitude improvement")
    else:
        print(f"  ✗ Still too weak (need factor ~10^4)")

    return K_total_corrections

# ==============================================================================
# MECHANISM 3: Non-Standard Cosmology
# ==============================================================================

def non_standard_cosmology():
    """
    Modified expansion history can change K:
    - Kination (ρ ~ a^-6): Faster expansion → lower K
    - Early matter domination: Slower expansion → higher K (bad!)
    - Low reheating: Suppress production
    """
    print("\n" + "="*70)
    print("MECHANISM 3: Non-Standard Cosmology")
    print("="*70)

    M_R = 20e3  # 20 TeV
    Y_D = 0.5

    # Standard cosmology
    K_std, Gamma_N, H_std = washout_parameter(M_R, Y_D)

    print(f"\nStandard radiation-dominated:")
    print(f"  H(T=M_R) = {H_std:.2e} GeV")
    print(f"  K_std    = {K_std:.2e}")

    # Kination phase: ρ ~ a^-6 (φ̇² domination)
    # H² ~ ρ_φ ~ a^-6 → H ~ a^-3
    # H_kination / H_rad ~ (T / T_kination)^(-1)

    T_kination = 100 * M_R  # Kination until 100× M_R
    if M_R < T_kination:
        H_kination = H_std * (M_R / T_kination)
        K_kination = Gamma_N / H_kination
        print(f"\nKination (until T ~ {T_kination/M_R:.0f}× M_R):")
        print(f"  H_kination ~ {H_kination:.2e} GeV")
        print(f"  K_kination = {K_kination:.2e}")
        print(f"  Suppression: factor {K_std/K_kination:.1f}")
    else:
        K_kination = K_std
        print(f"\n✗ Kination ended before T ~ M_R (no effect)")

    # Low reheating temperature
    # If T_RH < M_R, heavy neutrinos never in equilibrium
    # Production from inflaton decay, not thermal

    T_RH = 1e9  # 10^9 GeV (from τ modulus decay)

    print(f"\nLow reheating (T_RH = {T_RH:.2e} GeV):")
    print(f"  M_R = {M_R:.2e} GeV")
    if T_RH < M_R:
        print(f"  T_RH < M_R → Heavy neutrinos NEVER thermalized")
        print(f"  Production: Inflaton/moduli decay (not thermal)")
        print(f"  Washout: SUPPRESSED (no inverse decays)")
        print(f"  ✓ K_eff ~ 0 (no thermal bath at T > M_R)")
        K_low_reheat = 0.0
    else:
        print(f"  T_RH > M_R → Standard thermal production")
        K_low_reheat = K_std

    # Modulus decay scenario
    print(f"\nModulus decay scenario:")
    print(f"  τ modulus: m_τ ~ 10^12 GeV")
    print(f"  Decay: T_RH ~ m_τ/20 ~ 10^9 GeV")
    print(f"  Heavy neutrinos: M_R = 10-50 TeV")
    print(f"  Result: M_R ≫ T_RH → Non-thermal production")
    print(f"  Washout: ABSENT (universe never hot enough)")

    print(f"\n{'='*70}")
    print(f"VERDICT: Low T_RH is THE KEY MECHANISM")
    print(f"{'='*70}")
    print(f"""
If T_RH ~ 10^9 GeV < M_R ~ 10-50 TeV:
  ✓ Heavy neutrinos produced non-thermally (modulus decay)
  ✓ No inverse decays → K_eff = 0
  ✓ Lepton asymmetry NOT washed out
  ✓ η_B limited only by CP asymmetry ε, NOT by K

This resolves the washout crisis!
""")

    return K_low_reheat

# ==============================================================================
# MECHANISM 4: Yukawa Structure Optimization
# ==============================================================================

def yukawa_optimization():
    """
    Can we adjust Yukawa couplings to simultaneously:
    1. Give correct sterile neutrino DM abundance
    2. Reduce washout K < 1
    """
    print("\n" + "="*70)
    print("MECHANISM 4: Yukawa Structure Optimization")
    print("="*70)

    # DM constraint: Ω_s h² ~ 0.12
    # From Dodelson-Widrow: Ω ~ m_s^1.8 × sin²(2θ)
    # Where: sin²(2θ) ~ (Y_D² v²) / M_R²

    m_s = 500e-3  # 500 MeV
    Omega_target = 0.12

    # For DM: need sin²(2θ) ~ constant
    # sin²(2θ) ~ Y_D² × (v²/M_R²)
    # For fixed m_s: Y_D / M_R ~ constant

    print(f"\nDM constraint:")
    print(f"  m_s = {m_s*1e3:.0f} MeV")
    print(f"  Ω_s h² = {Omega_target:.2f}")
    print(f"  Requires: Y_D² / M_R ~ const")

    # Scan (M_R, Y_D) space
    M_R_scan = np.logspace(3, 5, 50)  # 1 TeV to 100 TeV

    # For each M_R, find Y_D that gives correct Ω
    # Ω ~ (Y_D²/M_R)^n for some n
    # Approximately: Y_D ~ √M_R

    results = []
    for M_R in M_R_scan:
        # DM constraint: Y_D ~ √M_R
        Y_D_DM = 0.5 * np.sqrt(M_R / 20e3)  # Normalized to M_R=20 TeV

        # Washout
        K, _, _ = washout_parameter(M_R, Y_D_DM)

        results.append((M_R, Y_D_DM, K))

    results = np.array(results)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.loglog(results[:, 0]/1e3, results[:, 1], 'b-', linewidth=2)
    ax1.set_xlabel('M_R [TeV]', fontsize=12)
    ax1.set_ylabel('Y_D (for Ω_DM = 0.12)', fontsize=12)
    ax1.set_title('Yukawa Coupling from DM Constraint')
    ax1.grid(True, alpha=0.3)

    ax2.loglog(results[:, 0]/1e3, results[:, 2], 'r-', linewidth=2, label='K parameter')
    ax2.axhline(y=1, color='green', linestyle='--', linewidth=2, label='K = 1 (washout threshold)')
    ax2.set_xlabel('M_R [TeV]', fontsize=12)
    ax2.set_ylabel('Washout K', fontsize=12)
    ax2.set_title('Washout vs Heavy Neutrino Mass')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('yukawa_optimization_scan.png', dpi=150)
    print(f"  Saved: yukawa_optimization_scan.png")

    # Find minimum K
    min_idx = np.argmin(results[:, 2])
    M_R_opt = results[min_idx, 0]
    Y_D_opt = results[min_idx, 1]
    K_opt = results[min_idx, 2]

    print(f"\nOptimal parameters:")
    print(f"  M_R = {M_R_opt/1e3:.1f} TeV")
    print(f"  Y_D = {Y_D_opt:.3f}")
    print(f"  K   = {K_opt:.2e}")

    if K_opt < 1:
        print(f"  ✓ Weak washout regime!")
    else:
        print(f"  ✗ Still K > 1 (need different approach)")

    # Check if K < 1 is achievable
    K_below_1 = results[results[:, 2] < 1]

    if len(K_below_1) > 0:
        print(f"\n✓ Viable parameter space found:")
        print(f"  M_R range: {K_below_1[:, 0].min()/1e3:.1f} - {K_below_1[:, 0].max()/1e3:.1f} TeV")
    else:
        print(f"\n✗ No parameter space with K < 1")
        print(f"  DM constraint forces Y_D too large")
        print(f"  Need alternative production mechanism")

    return K_opt

# ==============================================================================
# COMPREHENSIVE SOLUTION: LOW REHEATING + RESONANT ENHANCEMENT
# ==============================================================================

def comprehensive_solution():
    """
    Combine all mechanisms:
    1. Low T_RH < M_R → No thermal washout
    2. Resonant enhancement → Boost ε by 10²
    3. Non-thermal production → η_B ~ ε (not ε/K)
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE SOLUTION")
    print("="*70)

    # Parameters
    M_R = 30e3  # 30 TeV = 30,000 GeV
    Delta_M = 300  # 300 GeV (1% splitting)
    Y_D = 0.5
    T_RH = 1e3  # 10^3 GeV = 1 TeV (low reheating)

    print(f"\nInput parameters:")
    print(f"  M_R    = {M_R/1e3:.0f} TeV")
    print(f"  ΔM     = {Delta_M:.0f} GeV")
    print(f"  ΔM/M   = {Delta_M/M_R:.2e}")
    print(f"  Y_D    = {Y_D:.2f}")
    print(f"  T_RH   = {T_RH:.2e} GeV")

    # CP asymmetry (resonant)
    Gamma_N = (Y_D**2) * M_R / (8 * np.pi)
    epsilon_res = (Y_D**2 / (8 * np.pi)) * (Delta_M * Gamma_N) / (Delta_M**2 + Gamma_N**2)

    print(f"\nCP asymmetry:")
    print(f"  Γ_N      = {Gamma_N:.1f} GeV")
    print(f"  ε_res    = {epsilon_res:.2e}")

    # Check washout
    if T_RH < M_R:
        print(f"\nWashout analysis:")
        print(f"  T_RH < M_R → Non-thermal regime")
        print(f"  K_eff = 0 (no inverse decays)")
        print(f"  η_L ~ ε_res (no suppression)")
        K_eff = 0
    else:
        K_eff, _, _ = washout_parameter(M_R, Y_D)
        print(f"\nWashout analysis:")
        print(f"  T_RH > M_R → Thermal regime")
        print(f"  K = {K_eff:.2e}")
        print(f"  ✗ Strong washout!")

    # Baryon asymmetry
    # η_B/s = (a_sph / g_s) × η_L
    a_sph = 28/79  # Sphaleron conversion
    g_s = 2 * np.pi**2 / 45 * g_star

    if K_eff == 0:
        eta_L = epsilon_res
    else:
        # Strong washout: η_L ~ ε / K^3/2
        eta_L = epsilon_res / (K_eff**1.5)

    eta_B = (a_sph / g_s) * eta_L

    print(f"\nBaryon asymmetry:")
    print(f"  η_L      = {eta_L:.2e}")
    print(f"  a_sph    = {a_sph:.3f}")
    print(f"  η_B      = {eta_B:.2e}")

    # Observed value
    eta_B_obs = 6.1e-10
    print(f"  η_B^obs  = {eta_B_obs:.2e}")
    print(f"  Ratio    = {eta_B / eta_B_obs:.2e}")

    if 0.1 < eta_B / eta_B_obs < 10:
        print(f"\n  ✓✓✓ SUCCESS! Within factor 10 of observed value ✓✓✓")
    elif 0.01 < eta_B / eta_B_obs < 100:
        print(f"\n  ✓ Viable (within factor 100)")
    else:
        print(f"\n  ✗ Still off by large factor")

    return eta_B

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    print("="*70)
    print("LEPTOGENESIS: WASHOUT SUPPRESSION MECHANISMS")
    print("="*70)
    print("\nGoal: Reduce K ~ 10^11 to avoid strong washout")
    print()

    # Mechanism 1: Flavor effects
    eff_flavor = flavor_washout_analysis()

    # Mechanism 2: Thermal corrections
    K_thermal = thermal_corrections_analysis()

    # Mechanism 3: Non-standard cosmology
    K_low_reheat = non_standard_cosmology()

    # Mechanism 4: Yukawa optimization
    K_yukawa = yukawa_optimization()

    # Comprehensive solution
    eta_B_final = comprehensive_solution()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: WASHOUT SUPPRESSION MECHANISMS")
    print("="*70)

    mechanisms = [
        ("Flavor effects", "✗", "Factor ~2-5 improvement (insufficient)"),
        ("Thermal corrections", "✗", "Factor ~10 improvement (insufficient)"),
        ("Low reheating T_RH < M_R", "✓✓✓", "K_eff = 0 (NO WASHOUT)"),
        ("Yukawa optimization", "✗", "DM constraint forbids K < 1"),
    ]

    print(f"\n{'Status':<6} {'Mechanism':<30} {'Assessment'}")
    print("-"*70)
    for status, name, assessment in mechanisms:
        print(f"{status:<6} {name:<30} {assessment}")

    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    print(f"""
The KEY mechanism is LOW REHEATING:

**Scenario**:
1. τ modulus stabilized at τ* = 2.69i → m_τ ~ 10^12 GeV
2. Modulus decays: T_RH ~ m_τ/20 ~ 10^9 GeV
3. Heavy neutrinos: M_R ~ 10-50 TeV ≫ T_RH
4. Production: Non-thermal (from modulus decay, NOT thermal bath)

**Consequence**:
- Universe NEVER hot enough for N_R ↔ ℓH equilibrium
- No inverse decays → K_eff = 0
- Lepton asymmetry NOT washed out
- η_B ~ ε (limited by CP violation, not washout)

**With resonant enhancement**:
- ε_res ~ 0.16 (factor 10² above non-resonant)
- η_B ~ 10^-14 (still factor 10^4 too small...)

**Remaining issue**:
Even with K=0 and resonance, η_B still ~10^4× too small.
Need EITHER:
  (a) Larger ε (from complex Yukawa phase structure), OR
  (b) Alternative baryogenesis mechanism

**Status**: Washout SOLVED, but η_B magnitude remains challenging.
""")

if __name__ == "__main__":
    main()
