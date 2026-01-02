"""
Test B-field enhanced geometric CKM optimization.

Compares:
1. Without B-field (current approach)
2. With B-field flux (should improve CP phase)
"""

import numpy as np
import sys
sys.path.append('src')

# Import from main code
exec(open('src/unified_predictions_complete.py').read())

print("="*80)
print("GEOMETRIC CKM: B-FIELD COMPARISON")
print("="*80)
print()

# Use actual wrapping numbers from theory
tau_0 = 2.7j

# Test with generation 1 wrapping numbers
wrapping_up_test = [
    [(2, 0), (0, 1), (1, 0)],  # u quark
    [(2, 0), (0, 1), (1, 0)],  # c quark (same for simplicity)
    [(2, 0), (0, 1), (1, 0)]   # t quark
]

wrapping_down_test = [
    [(1, 1), (1, 0), (0, 1)],  # d quark
    [(1, 1), (1, 0), (0, 1)],  # s quark
    [(1, 1), (1, 0), (0, 1)]   # b quark
]

tau_values_test = [tau_0, tau_0, tau_0]

print("TEST 1: WITHOUT B-FIELD (baseline)")
print("-"*80)
result_no_B = optimize_geometric_ckm(wrapping_up_test, wrapping_down_test,
                                     tau_values_test, use_bfield=False, verbose=True)

# Compute final predictions without B-field
if len(result_no_B) == 5:
    sigma, alpha_12, alpha_23, alpha_13, lambda_inst = result_no_B
    B_field = None

    sin2_12, sin2_23, sin2_13, delta_cp, J_cp = compute_ckm_from_geometry(
        wrapping_up_test, wrapping_down_test, tau_values_test,
        sigma, alpha_12, alpha_23, alpha_13, lambda_inst, B_field
    )

    print("Results WITHOUT B-field:")
    print(f"  sin²θ₁₂ = {sin2_12:.6f} (obs: 0.051000)")
    print(f"  sin²θ₂₃ = {sin2_23:.6f} (obs: 0.001570)")
    print(f"  sin²θ₁₃ = {sin2_13:.6f} (obs: 0.000128)")
    print(f"  δ_CP = {delta_cp:.3f} rad (obs: 1.220 rad)")
    print(f"  J_CP = {J_cp:.3e} (obs: 3.0e-05)")
    print()

    err_12 = abs(sin2_12 - 0.051) / 0.051 * 100
    err_23 = abs(sin2_23 - 0.00157) / 0.00157 * 100
    err_13 = abs(sin2_13 - 0.000128) / 0.000128 * 100
    err_dcp = abs(delta_cp - 1.22) / 1.22 * 100
    err_jcp = abs(J_cp - 3.0e-5) / 3.0e-5 * 100

    print(f"Errors:")
    print(f"  Angles: {err_12:.1f}%, {err_23:.1f}%, {err_13:.1f}%")
    print(f"  CP phase: {err_dcp:.1f}%")
    print(f"  J_CP: {err_jcp:.1f}%")
    print()

print("="*80)
print("TEST 2: WITH B-FIELD (enhanced)")
print("-"*80)
result_with_B = optimize_geometric_ckm(wrapping_up_test, wrapping_down_test,
                                       tau_values_test, use_bfield=True, verbose=True)

# Compute final predictions with B-field
if len(result_with_B) == 6:
    sigma, alpha_12, alpha_23, alpha_13, lambda_inst, B_field = result_with_B

    sin2_12, sin2_23, sin2_13, delta_cp, J_cp = compute_ckm_from_geometry(
        wrapping_up_test, wrapping_down_test, tau_values_test,
        sigma, alpha_12, alpha_23, alpha_13, lambda_inst, B_field
    )

    print("Results WITH B-field:")
    print(f"  sin²θ₁₂ = {sin2_12:.6f} (obs: 0.051000)")
    print(f"  sin²θ₂₃ = {sin2_23:.6f} (obs: 0.001570)")
    print(f"  sin²θ₁₃ = {sin2_13:.6f} (obs: 0.000128)")
    print(f"  δ_CP = {delta_cp:.3f} rad (obs: 1.220 rad)")
    print(f"  J_CP = {J_cp:.3e} (obs: 3.0e-05)")
    print()

    err_12 = abs(sin2_12 - 0.051) / 0.051 * 100
    err_23 = abs(sin2_23 - 0.00157) / 0.00157 * 100
    err_13 = abs(sin2_13 - 0.000128) / 0.000128 * 100
    err_dcp = abs(delta_cp - 1.22) / 1.22 * 100
    err_jcp = abs(J_cp - 3.0e-5) / 3.0e-5 * 100

    print(f"Errors:")
    print(f"  Angles: {err_12:.1f}%, {err_23:.1f}%, {err_13:.1f}%")
    print(f"  CP phase: {err_dcp:.1f}%")
    print(f"  J_CP: {err_jcp:.1f}%")
    print()

    print(f"B-field values:")
    print(f"  B = [{B_field[0]:.4f}, {B_field[1]:.4f}, {B_field[2]:.4f}]")
    print(f"  |B| = {np.sqrt(sum(b**2 for b in B_field)):.4f}")
    print()

print("="*80)
print("COMPARISON SUMMARY")
print("="*80)
print()
print("Key question: Does B-field significantly improve CP observables?")
print()
print("Expected:")
print("  - Angles: similar with/without B (already fit well)")
print("  - δ_CP: MUCH better with B-field (was 100% error)")
print("  - J_CP: MUCH better with B-field (was 95% error)")
print()
print("Physics: B-field flux makes τ_eff = τ + i×B truly complex")
print("         → Breaks CP symmetry at fundamental level")
print("         → Natural source of CP violation in string theory")
