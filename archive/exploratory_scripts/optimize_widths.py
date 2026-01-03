"""
Optimize sector-dependent width factors to match observed overlap prefactors.

Physics: Different D-brane sectors have different tensions → different wavefunction widths.
"""

import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append('src')

from overlap_integrals import compute_all_overlaps

# Wrapping numbers
wrapping_leptons = np.array([[1, 0], [0, 1], [1, 1]])
wrapping_up = np.array([[2, 0], [0, 1], [1, 0]])
wrapping_down = np.array([[1, 1], [1, 0], [0, 1]])
wrapping_higgs = np.array([[1, 0], [1, 0], [1, 0]])

tau_0 = 2.7j
tau_lep = np.array([tau_0, tau_0, tau_0])
tau_up = np.array([tau_0, tau_0, tau_0])
tau_down = np.array([tau_0, tau_0, tau_0])

# Target overlaps
target_f_lep = 0.053
target_f_up = 0.197
target_f_down = 0.178

print("=" * 80)
print("OPTIMIZE SECTOR-DEPENDENT WIDTHS")
print("=" * 80)
print()

def objective(params):
    """Minimize squared relative errors."""
    width_base, r_lep, r_up, r_down = params

    f_lep, f_up, f_down = compute_all_overlaps(
        tau_lep, tau_up, tau_down,
        wrapping_leptons, wrapping_up, wrapping_down, wrapping_higgs,
        width_factor=width_base,
        sector_width_ratios=(r_lep, r_up, r_down),
        use_approximation=True
    )

    err_lep = (f_lep - target_f_lep) / target_f_lep
    err_up = (f_up - target_f_up) / target_f_up
    err_down = (f_down - target_f_down) / target_f_down

    return err_lep**2 + err_up**2 + err_down**2

# Initial guess: base width + sector ratios
x0 = [0.3, 1.0, 1.5, 1.3]  # width, r_lep, r_up, r_down

# Bounds
bounds = [
    (0.1, 2.0),   # width_base
    (0.5, 3.0),   # r_lep
    (0.5, 3.0),   # r_up
    (0.5, 3.0)    # r_down
]

print("Optimizing 4 parameters:")
print("  width_base: base wavefunction width")
print("  r_lep, r_up, r_down: sector-dependent multipliers")
print()
print("Target overlaps:")
print(f"  f_lep  = {target_f_lep:.6e}")
print(f"  f_up   = {target_f_up:.6e}")
print(f"  f_down = {target_f_down:.6e}")
print()

result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                 options={'maxiter': 100})

width_opt, r_lep_opt, r_up_opt, r_down_opt = result.x

print("=" * 80)
print("OPTIMIZATION RESULTS")
print("=" * 80)
print()
print(f"Optimal parameters:")
print(f"  width_base = {width_opt:.4f}")
print(f"  r_lep = {r_lep_opt:.4f}  →  ℓ_lep = {width_opt * r_lep_opt:.4f} × √Im[τ]")
print(f"  r_up = {r_up_opt:.4f}  →  ℓ_up = {width_opt * r_up_opt:.4f} × √Im[τ]")
print(f"  r_down = {r_down_opt:.4f}  →  ℓ_down = {width_opt * r_down_opt:.4f} × √Im[τ]")
print()

# Compute final overlaps
f_lep_opt, f_up_opt, f_down_opt = compute_all_overlaps(
    tau_lep, tau_up, tau_down,
    wrapping_leptons, wrapping_up, wrapping_down, wrapping_higgs,
    width_factor=width_opt,
    sector_width_ratios=(r_lep_opt, r_up_opt, r_down_opt),
    use_approximation=True
)

print(f"Optimized overlaps:")
print(f"  f_lep  = {f_lep_opt:.6e}  (target: {target_f_lep:.6e}, error: {abs(f_lep_opt-target_f_lep)/target_f_lep*100:.1f}%)")
print(f"  f_up   = {f_up_opt:.6e}  (target: {target_f_up:.6e}, error: {abs(f_up_opt-target_f_up)/target_f_up*100:.1f}%)")
print(f"  f_down = {f_down_opt:.6e}  (target: {target_f_down:.6e}, error: {abs(f_down_opt-target_f_down)/target_f_down*100:.1f}%)")
print()

max_error = max(
    abs(f_lep_opt - target_f_lep) / target_f_lep,
    abs(f_up_opt - target_f_up) / target_f_up,
    abs(f_down_opt - target_f_down) / target_f_down
) * 100

print(f"Maximum error: {max_error:.1f}%")
print()

if max_error < 5:
    print("✓ SUCCESS: Matched all three sectors within 5%!")
elif max_error < 20:
    print("✓ GOOD: Matched within 20% - reasonable agreement")
else:
    print("⚠ NEEDS WORK: Errors > 20% - may need better model")

print()
print("=" * 80)
print("PHYSICAL INTERPRETATION")
print("=" * 80)
print()

tau_2 = tau_0.imag
print(f"Physical widths (in string units ℓ_s):")
print(f"  ℓ_lep  = {width_opt * r_lep_opt * np.sqrt(tau_2):.4f} ℓ_s")
print(f"  ℓ_up   = {width_opt * r_up_opt * np.sqrt(tau_2):.4f} ℓ_s")
print(f"  ℓ_down = {width_opt * r_down_opt * np.sqrt(tau_2):.4f} ℓ_s")
print()

print("Width ratios:")
print(f"  ℓ_up / ℓ_lep = {r_up_opt / r_lep_opt:.3f}")
print(f"  ℓ_down / ℓ_lep = {r_down_opt / r_lep_opt:.3f}")
print()

print("Connection to brane tension:")
print("  ℓ ~ 1/√T_brane")
print("  Wider wavefunctions → lower tension → weaker localization")
print("  Up quarks have largest ℓ → lowest tension → strongest overlap with Higgs!")
print()

print("Overlap ratio explanation:")
print(f"  f_up / f_lep = {f_up_opt / f_lep_opt:.3f} (observed: {target_f_up / target_f_lep:.3f})")
print(f"  f_down / f_lep = {f_down_opt / f_lep_opt:.3f} (observed: {target_f_down / target_f_lep:.3f})")
print("  Larger widths → larger overlaps → stronger Yukawa couplings → heavier fermions")
