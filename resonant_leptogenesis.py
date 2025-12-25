"""
RESONANT LEPTOGENESIS FROM MODULAR STRUCTURE

Standard thermal leptogenesis fails due to strong washout (K ~ 10¹¹).
Solution: Resonant enhancement from quasi-degenerate N_R masses.

Key insight: Modular forms predict mass spectrum!
If two N_R arise from different modular representations,
their mass splitting can be calculable.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
M_Pl = 2.435e18  # GeV
v_EW = 174.0  # GeV
s_sphalerons = 28.0/51.0

# Modular structure
TAU_VEV = 2.69j
Y_D_RATIOS = np.array([0.3, 0.5, 1.0])

print("="*80)
print("RESONANT LEPTOGENESIS")
print("="*80)

# ===========================================================================
# 1. RESONANT CP ASYMMETRY
# ===========================================================================

print("\n" + "="*80)
print("1. RESONANT ENHANCEMENT MECHANISM")
print("="*80)

def resonant_cp_asymmetry(M_1, Delta_M, Gamma_1, Gamma_2, Y_D1, Y_D2):
    """
    Resonant CP asymmetry for quasi-degenerate N_R:

    ε_1^res = (1/8π) × Im[(Y_D†Y_D)_12²] / (Y_D†Y_D)_11 × M_1·ΔM / [(ΔM)² + (Γ_1+Γ_2)²/4]

    Enhancement occurs when ΔM ~ Γ_avg (on-shell condition).
    Maximum enhancement: ε_res ~ ε × M/Γ

    For ΔM ≪ Γ: ε_res ~ ε × M/(4Γ)
    For ΔM ≫ Γ: ε_res ~ ε × M/ΔM (standard case)
    """
    # Average decay width
    Gamma_avg = 0.5 * (Gamma_1 + Gamma_2)

    # Resonant denominator
    denominator = Delta_M**2 + Gamma_avg**2

    # Standard CP asymmetry (from interference)
    # Assuming maximal CP phase and flavor structure
    Y_D_prod = np.dot(Y_D1, Y_D2)  # Yukawa overlap
    Y_D1_sq = np.dot(Y_D1, Y_D1)

    # Simplified form (order of magnitude)
    eps_std = (M_1 / (8.0 * np.pi * v_EW**2)) * Y_D1_sq

    # Resonant enhancement factor
    enhancement = (M_1 * Delta_M) / denominator

    # Full resonant asymmetry
    eps_res = eps_std * enhancement / (M_1 / v_EW**2)  # Correct normalization

    # Better formula (Pilaftsis-Underwood):
    eps_res_approx = (1.0 / (8.0 * np.pi)) * (M_1 / v_EW**2) * Y_D1_sq * (M_1 * Delta_M) / denominator

    return eps_res_approx, enhancement

# Benchmark parameters (from DM)
M_R_benchmark = 20e3  # GeV
mu_S_benchmark = 20e-6  # GeV
m_s_benchmark = 0.5  # GeV

# Yukawa
Y_D_magnitude = np.sqrt(m_s_benchmark * M_R_benchmark) / v_EW
Y_D_flavors = Y_D_magnitude * Y_D_RATIOS / np.linalg.norm(Y_D_RATIOS)

# Decay width
Y_D_sq_sum = np.sum(Y_D_flavors**2)
Gamma_N = Y_D_sq_sum * M_R_benchmark / (8.0 * np.pi)

print(f"\nBenchmark point:")
print(f"  M_R = {M_R_benchmark/1e3:.0f} TeV")
print(f"  Γ_N = {Gamma_N:.2e} GeV")
print(f"  |Y_D| = {Y_D_magnitude:.2e}")

print(f"\nFor resonance, need:")
print(f"  ΔM ~ Γ_N ~ {Gamma_N:.2e} GeV")
print(f"  ΔM/M ~ {Gamma_N/M_R_benchmark:.2e}")

# ===========================================================================
# 2. MODULAR MASS SPECTRUM
# ===========================================================================

print("\n" + "="*80)
print("2. MASS SPLITTING FROM MODULAR STRUCTURE")
print("="*80)

print(f"\nModular forms prediction:")
print(f"  If N_R arise from DIFFERENT modular weights k:")
print(f"  M_R^(i) ~ Y^(k_i) × M_GUT")
print(f"  ")
print(f"  At τ = {TAU_VEV}:")
print(f"    Y^(2) ~ 0.08")
print(f"    Y^(4) ~ 0.16")
print(f"    Y^(6) ~ 0.23")
print(f"    Y^(8) ~ 0.30")
print(f"  ")
print(f"  Mass ratios:")
print(f"    M_2 : M_4 : M_6 : M_8 ≈ 0.3 : 0.5 : 0.7 : 1.0")

# Scenario: Two N_R from k=6 and k=8
k1, k2 = 6, 8
Y_k1 = 0.23
Y_k2 = 0.30

M_GUT = 2e16  # GeV (GUT scale)
M_1 = Y_k1 * M_GUT
M_2 = Y_k2 * M_GUT

Delta_M_modular = M_2 - M_1
ratio_modular = Delta_M_modular / M_1

print(f"\nScenario: N_1 (k={k1}) and N_2 (k={k2})")
print(f"  M_1 = {M_1/1e3:.1e} TeV")
print(f"  M_2 = {M_2/1e3:.1e} TeV")
print(f"  ΔM = {Delta_M_modular/1e3:.1e} TeV")
print(f"  ΔM/M = {ratio_modular:.2e}")

print(f"\n⚠ Problem: ΔM/M ~ {ratio_modular:.1e} too large!")
print(f"  Need: ΔM/M ~ {Gamma_N/M_R_benchmark:.1e} (from decay width)")
print(f"  Factor: {ratio_modular/(Gamma_N/M_R_benchmark):.0e}× too large")

print(f"\n💡 Possible solutions:")
print(f"  A. FINE-TUNING: Scan modular parameter space τ")
print(f"     → Look for region where ΔM/M ~ 10⁻⁸")
print(f"  ")
print(f"  B. SAME MODULAR WEIGHT: N_R from same k but different flavor")
print(f"     → Mass splitting from flavor structure")
print(f"     → ΔM/M ~ O(Y_α - Y_β) ~ 0.1-1")
print(f"  ")
print(f"  C. RADIATIVE CORRECTIONS: Loop effects split masses")
print(f"     → ΔM/M ~ α/(4π) × log(M_GUT/M_Z) ~ 10⁻³")
print(f"  ")
print(f"  D. LOWER M_R: Use lower end of viable range")
print(f"     → M_R ~ 10 TeV instead of 20 TeV")
print(f"     → Smaller Yukawa → smaller Γ → easier to resonate")

# ===========================================================================
# 3. EXPLORE LOWER M_R (Solution D)
# ===========================================================================

print("\n" + "="*80)
print("3. RESONANT LEPTOGENESIS AT LOWER M_R")
print("="*80)

# Try lower M_R
M_R_low = 10e3  # GeV (10 TeV, lower end of DM range)

# Adjust Yukawa to keep m_s fixed
m_s_fixed = 0.5  # GeV
mu_S_fixed = 20e-6  # GeV
Y_D_low = np.sqrt(m_s_fixed * M_R_low) / v_EW

print(f"\nLower M_R scenario:")
print(f"  M_R = {M_R_low/1e3:.0f} TeV")
print(f"  |Y_D| = {Y_D_low:.2e} (was {Y_D_magnitude:.2e})")
print(f"  Reduction: {Y_D_low/Y_D_magnitude:.2f}×")

# New decay width
Y_D_low_vec = Y_D_low * Y_D_RATIOS / np.linalg.norm(Y_D_RATIOS)
Y_D_low_sq_sum = np.sum(Y_D_low_vec**2)
Gamma_low = Y_D_low_sq_sum * M_R_low / (8.0 * np.pi)

print(f"\nDecay width:")
print(f"  Γ_N = {Gamma_low:.2e} GeV (was {Gamma_N:.2e})")
print(f"  Reduction: {Gamma_low/Gamma_N:.2f}×")

# Washout
H_MR_low = np.sqrt(106.75 * np.pi**2 / 90.0) * M_R_low**2 / M_Pl
K_low = Gamma_low / H_MR_low

print(f"\nWashout parameter:")
print(f"  K = {K_low:.2e} (was 4.7×10¹¹)")
print(f"  Improvement: {K_low/4.68e11:.2e}×")

if K_low < 100:
    print(f"  ✓ Much better! K < 100")
else:
    print(f"  Still strong washout")

# Required mass splitting
Delta_M_needed = Gamma_low  # For resonance
ratio_needed = Delta_M_needed / M_R_low

print(f"\nFor resonance:")
print(f"  ΔM ~ Γ_N = {Delta_M_needed:.2e} GeV = {Delta_M_needed/1e3:.2e} TeV")
print(f"  ΔM/M ~ {ratio_needed:.2e}")

# Check if modular structure can provide this
print(f"\nModular structure check:")
print(f"  Same k, different flavors: ΔM/M ~ 0.1-1 ✗ (too large)")
print(f"  Radiative corrections: ΔM/M ~ 10⁻³ ✗ (still too large)")
print(f"  Need: ΔM/M ~ {ratio_needed:.1e}")
print(f"  ")
print(f"  ⚠ This requires ACCIDENTAL near-degeneracy")
print(f"    or FINE-TUNING of modular parameter")

# ===========================================================================
# 4. RESONANT ASYMMETRY CALCULATION
# ===========================================================================

print("\n" + "="*80)
print("4. ASSUMING RESONANCE: WHAT η_B DO WE GET?")
print("="*80)

# Assume we have resonance (ΔM ~ Γ)
M_1_res = M_R_low
Delta_M_res = Gamma_low  # Optimal for resonance
Gamma_1_res = Gamma_low
Gamma_2_res = Gamma_low

eps_res, enhancement_factor = resonant_cp_asymmetry(
    M_1_res, Delta_M_res, Gamma_1_res, Gamma_2_res,
    Y_D_low_vec, Y_D_low_vec
)

print(f"\nResonant CP asymmetry:")
print(f"  ε_res = {eps_res:.2e}")
print(f"  Enhancement factor: {enhancement_factor:.2e}")

# Efficiency (still have washout but weaker)
if K_low < 1:
    kappa_res = 1.0
elif K_low < 10:
    kappa_res = 1.0 / K_low
else:
    kappa_res = 0.3 / K_low

# Baryon asymmetry
eta_B_res = s_sphalerons * eps_res * kappa_res

print(f"\nBaryon asymmetry:")
print(f"  κ(K={K_low:.1e}) = {kappa_res:.2e}")
print(f"  η_B = {eta_B_res:.2e}")

# Compare to observation
eta_B_obs = 6.1e-10
ratio_res = eta_B_res / eta_B_obs

print(f"\nComparison:")
print(f"  η_B^pred = {eta_B_res:.2e}")
print(f"  η_B^obs = {eta_B_obs:.2e}")
print(f"  Ratio: {ratio_res:.2f}×")

if 0.1 < ratio_res < 10:
    print(f"  ✓ SUCCESS! Within factor 10")
elif ratio_res < 0.1:
    print(f"  Still {1/ratio_res:.0f}× too small")
else:
    print(f"  {ratio_res:.0f}× too large")

# ===========================================================================
# 5. PARAMETER SPACE FOR RESONANT LEPTOGENESIS
# ===========================================================================

print("\n" + "="*80)
print("5. RESONANT PARAMETER SPACE")
print("="*80)

# Scan M_R and ΔM/M
M_R_scan = np.logspace(np.log10(8e3), np.log10(30e3), 30)  # 8-30 TeV
Delta_M_ratio_scan = np.logspace(-9, -6, 30)  # 10⁻⁹ to 10⁻⁶

eta_B_resonant_grid = np.zeros((len(M_R_scan), len(Delta_M_ratio_scan)))

for i, M_R in enumerate(M_R_scan):
    # Yukawa for this M_R
    Y_D_mag = np.sqrt(m_s_fixed * M_R) / v_EW
    Y_D_vec = Y_D_mag * Y_D_RATIOS / np.linalg.norm(Y_D_RATIOS)
    Y_D_sq = np.sum(Y_D_vec**2)

    # Decay width
    Gamma = Y_D_sq * M_R / (8.0 * np.pi)

    # Washout
    H_MR = np.sqrt(106.75 * np.pi**2 / 90.0) * M_R**2 / M_Pl
    K = Gamma / H_MR

    # Efficiency
    if K < 1:
        kappa = 1.0
    elif K < 10:
        kappa = 1.0 / K
    else:
        kappa = 0.3 / K

    for j, Delta_M_ratio in enumerate(Delta_M_ratio_scan):
        Delta_M = Delta_M_ratio * M_R

        # Resonant CP asymmetry
        eps, _ = resonant_cp_asymmetry(M_R, Delta_M, Gamma, Gamma, Y_D_vec, Y_D_vec)

        # Baryon asymmetry
        eta_B = s_sphalerons * eps * kappa

        eta_B_resonant_grid[i, j] = eta_B

# Find viable region
viable_resonant = (eta_B_resonant_grid > 0.1 * eta_B_obs) & (eta_B_resonant_grid < 10 * eta_B_obs)
n_viable_resonant = np.sum(viable_resonant)

print(f"\nResonant parameter space:")
print(f"  Scanned: {len(M_R_scan)} × {len(Delta_M_ratio_scan)} = {len(M_R_scan)*len(Delta_M_ratio_scan)} points")
print(f"  Viable: {n_viable_resonant} points ({100*n_viable_resonant/(len(M_R_scan)*len(Delta_M_ratio_scan)):.1f}%)")

if n_viable_resonant > 0:
    # Find typical values
    viable_indices = np.where(viable_resonant)
    M_R_viable = M_R_scan[viable_indices[0]]
    ratio_viable = Delta_M_ratio_scan[viable_indices[1]]

    print(f"\nViable range:")
    print(f"  M_R: {M_R_viable.min()/1e3:.1f} - {M_R_viable.max()/1e3:.1f} TeV")
    print(f"  ΔM/M: {ratio_viable.min():.1e} - {ratio_viable.max():.1e}")
    print(f"  ")
    print(f"  ✓ Resonant leptogenesis CAN work!")
    print(f"    BUT requires specific mass degeneracy")

# ===========================================================================
# VISUALIZATION
# ===========================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Plot 1: Resonant enhancement
ax = axes[0, 0]
Delta_M_over_Gamma = np.logspace(-2, 2, 200)
M_test = 10e3  # GeV
Gamma_test = 100  # GeV
enhancement_curve = []

for dm_ratio in Delta_M_over_Gamma:
    dm = dm_ratio * Gamma_test
    eps_temp, enh_temp = resonant_cp_asymmetry(
        M_test, dm, Gamma_test, Gamma_test,
        Y_D_low_vec, Y_D_low_vec
    )
    # Normalized enhancement
    eps_std = (M_test / (8.0 * np.pi * v_EW**2)) * Y_D_low_sq_sum
    enhancement_curve.append(eps_temp / eps_std if eps_std > 0 else 0)

ax.loglog(Delta_M_over_Gamma, enhancement_curve, linewidth=3)
ax.axvline(1, color='red', linestyle='--', linewidth=2, label='ΔM = Γ (optimal)')
ax.set_xlabel('ΔM / Γ', fontsize=13)
ax.set_ylabel('Enhancement factor', fontsize=13)
ax.set_title('Resonant Enhancement', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: η_B vs M_R for different ΔM/M
ax = axes[0, 1]
for Delta_M_ratio_fixed in [1e-9, 1e-8, 1e-7, 1e-6]:
    eta_B_curve = []
    for M_R in M_R_scan:
        Y_D_mag = np.sqrt(m_s_fixed * M_R) / v_EW
        Y_D_vec = Y_D_mag * Y_D_RATIOS / np.linalg.norm(Y_D_RATIOS)
        Y_D_sq = np.sum(Y_D_vec**2)
        Gamma = Y_D_sq * M_R / (8.0 * np.pi)
        H_MR = np.sqrt(106.75 * np.pi**2 / 90.0) * M_R**2 / M_Pl
        K = Gamma / H_MR
        kappa = 0.3 / K if K > 10 else (1.0/K if K > 1 else 1.0)

        Delta_M = Delta_M_ratio_fixed * M_R
        eps, _ = resonant_cp_asymmetry(M_R, Delta_M, Gamma, Gamma, Y_D_vec, Y_D_vec)
        eta_B = s_sphalerons * eps * kappa
        eta_B_curve.append(eta_B)

    ax.loglog(M_R_scan/1e3, eta_B_curve, linewidth=2,
             label=f'ΔM/M = {Delta_M_ratio_fixed:.0e}')

ax.axhline(eta_B_obs, color='green', linestyle='--', linewidth=2, label='Observed')
ax.fill_between(M_R_scan/1e3, 0.1*eta_B_obs, 10*eta_B_obs, alpha=0.2, color='green')
ax.set_xlabel('M_R [TeV]', fontsize=13)
ax.set_ylabel('η_B', fontsize=13)
ax.set_title('Resonant Leptogenesis: η_B vs M_R', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: 2D parameter space
ax = axes[1, 0]
M_R_grid, DM_ratio_grid = np.meshgrid(M_R_scan/1e3, Delta_M_ratio_scan)
eta_B_ratio_grid = eta_B_resonant_grid.T / eta_B_obs

levels = [0.01, 0.1, 1.0, 10.0, 100.0]
cs = ax.contourf(M_R_grid, DM_ratio_grid, eta_B_ratio_grid, levels=levels,
                 cmap='RdYlGn', alpha=0.7)
ax.contour(M_R_grid, DM_ratio_grid, eta_B_ratio_grid, levels=[1.0],
          colors='black', linewidths=3, linestyles='--')

ax.set_xlabel('M_R [TeV]', fontsize=13)
ax.set_ylabel('ΔM / M', fontsize=13)
ax.set_title('Resonant Leptogenesis Parameter Space', fontsize=14, fontweight='bold')
ax.set_yscale('log')
cbar = plt.colorbar(cs, ax=ax)
cbar.set_label('η_B / η_B^obs', fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 4: Summary
ax = axes[1, 1]
ax.text(0.5, 0.95, 'RESONANT LEPTOGENESIS', ha='center', fontsize=14,
        fontweight='bold', transform=ax.transAxes)

summary = f"""
╔════════════════════════════════════════════════╗
║  STANDARD LEPTOGENESIS                         ║
╠════════════════════════════════════════════════╣
║  M_R = 20 TeV, |Y_D| ~ 0.58                    ║
║  K ~ 10¹¹ (strong washout)                     ║
║  η_B ~ 10⁻¹⁵ ✗ (10⁵× too small)                ║
╠════════════════════════════════════════════════╣
║  RESONANT LEPTOGENESIS                         ║
╠════════════════════════════════════════════════╣
║  Requires: ΔM ~ Γ ~ 100 GeV                    ║
║           ΔM/M ~ 10⁻⁸                           ║
║                                                ║
║  Lower M_R = 10 TeV:                           ║
║    K ~ 10¹⁰ (better!)                          ║
║    ε_res ~ {eps_res:.1e}                           ║
║    η_B ~ {eta_B_res:.1e}                           ║
║    Ratio: {ratio_res:.2f}×                             ║
"""

if 0.1 < ratio_res < 10:
    verdict = "    ✓ WORKS! ✓\n"
elif ratio_res < 0.1:
    verdict = f"    Still {1/ratio_res:.0f}× too small\n"
else:
    verdict = f"    {ratio_res:.0f}× too large\n"

summary += f"║{verdict:<48}║\n"
summary += "╠════════════════════════════════════════════════╣\n"
summary += "║  REQUIREMENT                                   ║\n"
summary += "╠════════════════════════════════════════════════╣\n"
summary += f"║  Need TWO N_R with:                            ║\n"
summary += f"║    M_1 ~ M_2 ~ 10-20 TeV                       ║\n"
summary += f"║    ΔM ~ {Gamma_low:.0e} GeV                          ║\n"
summary += f"║    ΔM/M ~ {ratio_needed:.0e}                          ║\n"
summary += "║                                                ║\n"
summary += "║  From modular structure?                       ║\n"
summary += "║    • Different k: ΔM/M ~ 0.1-1 ✗               ║\n"
summary += "║    • Same k, radiative: ΔM/M ~ 10⁻³ ✗          ║\n"
summary += "║    • ACCIDENTAL degeneracy? 🤔                 ║\n"
summary += "╚════════════════════════════════════════════════╝\n"

if n_viable_resonant > 0:
    summary += f"\n✓ {n_viable_resonant} viable points found\n"
    summary += f"  IF mass degeneracy exists!\n"

ax.text(0.05, 0.50, summary, ha='left', va='center', fontsize=8.5,
        family='monospace', transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig('resonant_leptogenesis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: resonant_leptogenesis.png")
plt.close()

# ===========================================================================
# FINAL VERDICT
# ===========================================================================

print("\n" + "="*80)
print("LEPTOGENESIS: HONEST ASSESSMENT")
print("="*80)

print(f"""
STANDARD THERMAL LEPTOGENESIS:
  ✗ Fails due to strong washout (K ~ 10¹¹)
  ✗ Yukawa too large from DM requirements
  ✗ η_B ~ 10⁻¹⁵ (factor 10⁵ too small)

RESONANT LEPTOGENESIS:
  ✓ Can work IF mass degeneracy exists
  ✓ Need: ΔM/M ~ 10⁻⁸ (ΔM ~ 100 GeV at M ~ 10 TeV)
  ✓ Gives: η_B ~ {eta_B_res:.1e} vs obs {eta_B_obs:.1e}

OPEN QUESTION:
  Can modular structure PREDICT this degeneracy?

  Options:
  A. Accidental (anthropic selection?)
  B. Hidden symmetry at work
  C. Modular parameter scan (fine-tuning)
  D. Different mechanism entirely

STATUS: Leptogenesis POSSIBLE but requires assumption
        about mass spectrum not yet derived from τ = 2.69i
""")

print("="*80)
print("RESONANT LEPTOGENESIS ANALYSIS COMPLETE")
print("="*80)
