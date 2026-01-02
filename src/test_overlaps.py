"""
Test overlap integral calculations with actual wrapping numbers from the theory.

Compares computed overlaps from D-brane geometry with fitted values.
"""

import numpy as np
import sys
sys.path.append('src')

from overlap_integrals import (
    compute_all_overlaps,
    optimize_width_factor,
    overlap_gaussian_approximation
)

# ============================================================================
# WRAPPING NUMBERS FROM THEORY
# ============================================================================

# These are the wrapping numbers used in unified_predictions_complete.py
# For T²×T²×T² compactification

# Leptons: (e, μ, τ) - use first generation
wrapping_leptons = np.array([
    [1, 0],   # T²₁: (1,0) wrapping
    [0, 1],   # T²₂: (0,1) wrapping
    [1, 1]    # T²₃: (1,1) wrapping
])

# Up-type quarks: (u, c, t) - use first generation
wrapping_up = np.array([
    [2, 0],   # T²₁: (2,0) wrapping (different from leptons!)
    [0, 1],   # T²₂: (0,1) wrapping
    [1, 0]    # T²₃: (1,0) wrapping (different from leptons!)
])

# Down-type quarks: (d, s, b) - use first generation
wrapping_down = np.array([
    [1, 1],   # T²₁: (1,1) wrapping
    [1, 0],   # T²₂: (1,0) wrapping (different from others!)
    [0, 1]    # T²₃: (0,1) wrapping
])

# Higgs boson
wrapping_higgs = np.array([
    [1, 0],   # T²₁: (1,0) wrapping
    [1, 0],   # T²₂: (1,0) wrapping
    [1, 0]    # T²₃: (1,0) wrapping
])

# Modular parameters (use first generation values)
tau_0 = 2.7j

# Generation-dependent τ values (from optimization)
# For now, use τ₀ for all (will refine later)
tau_lep = np.array([tau_0, tau_0, tau_0])
tau_up = np.array([tau_0, tau_0, tau_0])
tau_down = np.array([tau_0, tau_0, tau_0])

# Target overlap values (from Kähler potential fit)
target_f_lep = 0.053
target_f_up = 0.197
target_f_down = 0.178

print("=" * 80)
print("YUKAWA OVERLAP INTEGRAL CALCULATION")
print("=" * 80)
print()

print("WRAPPING NUMBERS:")
print(f"  Leptons: {wrapping_leptons.tolist()}")
print(f"  Up quarks: {wrapping_up.tolist()}")
print(f"  Down quarks: {wrapping_down.tolist()}")
print(f"  Higgs: {wrapping_higgs.tolist()}")
print()

print("MODULAR PARAMETERS:")
print(f"  τ₀ = {tau_0}")
print(f"  Im[τ₀] = {tau_0.imag}")
print()

# ============================================================================
# TEST 1: ANALYTIC APPROXIMATION WITH DEFAULT WIDTH
# ============================================================================

print("=" * 80)
print("TEST 1: ANALYTIC APPROXIMATION (width_factor = 1.0)")
print("=" * 80)
print()

width_default = 1.0
f_lep_1, f_up_1, f_down_1 = compute_all_overlaps(
    tau_lep, tau_up, tau_down,
    wrapping_leptons, wrapping_up, wrapping_down, wrapping_higgs,
    width_factor=width_default, use_approximation=True
)

print(f"Computed overlaps:")
print(f"  f_lep  = {f_lep_1:.6e}")
print(f"  f_up   = {f_up_1:.6e}")
print(f"  f_down = {f_down_1:.6e}")
print()

print(f"Target overlaps (from Kähler fit):")
print(f"  f_lep  = {target_f_lep:.6e}")
print(f"  f_up   = {target_f_up:.6e}")
print(f"  f_down = {target_f_down:.6e}")
print()

print(f"Ratios (computed/target):")
print(f"  Lepton:  {f_lep_1/target_f_lep:.3f}×")
print(f"  Up:      {f_up_1/target_f_up:.3f}×")
print(f"  Down:    {f_down_1/target_f_down:.3f}×")
print()

# ============================================================================
# TEST 2: OPTIMIZE WIDTH TO MATCH TARGETS
# ============================================================================

print("=" * 80)
print("TEST 2: OPTIMIZE WIDTH FACTOR")
print("=" * 80)
print()

print("Optimizing width_factor to match target overlaps...")
print("(This may take ~30 seconds)")
print()

try:
    optimal_width = optimize_width_factor(
        tau_lep, tau_up, tau_down,
        wrapping_leptons, wrapping_up, wrapping_down, wrapping_higgs,
        (target_f_lep, target_f_up, target_f_down),
        width_bounds=(0.3, 3.0)
    )

    print(f"Optimal width_factor = {optimal_width:.4f}")
    print()

    # Compute overlaps with optimal width
    f_lep_opt, f_up_opt, f_down_opt = compute_all_overlaps(
        tau_lep, tau_up, tau_down,
        wrapping_leptons, wrapping_up, wrapping_down, wrapping_higgs,
        width_factor=optimal_width, use_approximation=True
    )

    print(f"Optimized overlaps:")
    print(f"  f_lep  = {f_lep_opt:.6e}  (target: {target_f_lep:.6e}, error: {abs(f_lep_opt-target_f_lep)/target_f_lep*100:.1f}%)")
    print(f"  f_up   = {f_up_opt:.6e}  (target: {target_f_up:.6e}, error: {abs(f_up_opt-target_f_up)/target_f_up*100:.1f}%)")
    print(f"  f_down = {f_down_opt:.6e}  (target: {target_f_down:.6e}, error: {abs(f_down_opt-target_f_down)/target_f_down*100:.1f}%)")
    print()

    # Check if we can match all three simultaneously
    max_error = max(
        abs(f_lep_opt - target_f_lep) / target_f_lep,
        abs(f_up_opt - target_f_up) / target_f_up,
        abs(f_down_opt - target_f_down) / target_f_down
    ) * 100

    print(f"Maximum error: {max_error:.1f}%")
    print()

    if max_error < 10:
        print("✓ SUCCESS: Width optimization matches all three sectors within 10%!")
        print(f"  → Physical width: ℓ = {optimal_width:.3f} × √Im[τ] = {optimal_width:.3f} × {np.sqrt(tau_0.imag):.3f} = {optimal_width*np.sqrt(tau_0.imag):.3f} ℓ_s")
    else:
        print("⚠ PARTIAL: Cannot match all three sectors simultaneously with single width")
        print("  → May need sector-dependent widths (from different brane tensions)")

except Exception as e:
    print(f"Error during optimization: {e}")
    print("Skipping optimization test")

print()

# ============================================================================
# TEST 3: SCAN WIDTH RANGE
# ============================================================================

print("=" * 80)
print("TEST 3: WIDTH SCAN")
print("=" * 80)
print()

print("Scanning width_factor from 0.3 to 3.0...")
print()

widths = np.linspace(0.3, 3.0, 20)
overlaps_lep = []
overlaps_up = []
overlaps_down = []

for w in widths:
    f_l, f_u, f_d = compute_all_overlaps(
        tau_lep, tau_up, tau_down,
        wrapping_leptons, wrapping_up, wrapping_down, wrapping_higgs,
        width_factor=w, use_approximation=True
    )
    overlaps_lep.append(f_l)
    overlaps_up.append(f_u)
    overlaps_down.append(f_d)

# Find best match for each sector
idx_lep = np.argmin([abs(f - target_f_lep) for f in overlaps_lep])
idx_up = np.argmin([abs(f - target_f_up) for f in overlaps_up])
idx_down = np.argmin([abs(f - target_f_down) for f in overlaps_down])

print(f"Best width for each sector:")
print(f"  Leptons: width = {widths[idx_lep]:.3f}  →  f_lep = {overlaps_lep[idx_lep]:.6e}  (error: {abs(overlaps_lep[idx_lep]-target_f_lep)/target_f_lep*100:.1f}%)")
print(f"  Up:      width = {widths[idx_up]:.3f}  →  f_up = {overlaps_up[idx_up]:.6e}  (error: {abs(overlaps_up[idx_up]-target_f_up)/target_f_up*100:.1f}%)")
print(f"  Down:    width = {widths[idx_down]:.3f}  →  f_down = {overlaps_down[idx_down]:.6e}  (error: {abs(overlaps_down[idx_down]-target_f_down)/target_f_down*100:.1f}%)")
print()

if idx_lep == idx_up == idx_down:
    print("✓ All three sectors prefer the same width!")
else:
    print("⚠ Different sectors prefer different widths")
    print(f"  Spread: {widths[min(idx_lep, idx_up, idx_down)]:.3f} - {widths[max(idx_lep, idx_up, idx_down)]:.3f}")

print()

# ============================================================================
# ANALYSIS: WHY DIFFERENT OVERLAPS?
# ============================================================================

print("=" * 80)
print("PHYSICAL INTERPRETATION")
print("=" * 80)
print()

print("Wrapping number differences (relative to Higgs):")
print()

for k in range(3):
    print(f"T²_{k+1}:")
    delta_lep = wrapping_leptons[k] - wrapping_higgs[k]
    delta_up = wrapping_up[k] - wrapping_higgs[k]
    delta_down = wrapping_down[k] - wrapping_higgs[k]

    print(f"  Δn_lep  = {delta_lep}  →  distance² = {abs(delta_lep[0] + delta_lep[1]*tau_0)**2:.3f}")
    print(f"  Δn_up   = {delta_up}  →  distance² = {abs(delta_up[0] + delta_up[1]*tau_0)**2:.3f}")
    print(f"  Δn_down = {delta_down}  →  distance² = {abs(delta_down[0] + delta_down[1]*tau_0)**2:.3f}")
    print()

print("Overlap hierarchy:")
print(f"  f_up > f_down > f_lep")
print(f"  {target_f_up:.3f} > {target_f_down:.3f} > {target_f_lep:.3f}")
print()
print("Physical origin:")
print("  • Up quarks have largest overlap → closest to Higgs")
print("  • Leptons have smallest overlap → furthest from Higgs")
print("  • This determines Yukawa coupling strengths!")
print()
print("Connection to mass hierarchy:")
print("  m_t >> m_b >> m_τ")
print("  Top quark: largest overlap × largest geometric factor")
print("  Electron: smallest overlap × smallest geometric factor")

print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. ✓ Implemented analytic Gaussian approximation (fast)")
print("2. ⚠ Full numerical integration (slow but exact)")
print("3. → Optimize sector-dependent widths (physical: different tensions)")
print("4. → Include modular form corrections (η(τ) dependence)")
print("5. → Test with generation-dependent τ values")
print()
