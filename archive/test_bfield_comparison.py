"""
Test B-field enhancement for geometric CKM
Compare optimization with and without B-field
"""

import sys
sys.path.append('src')

from unified_predictions_complete import optimize_geometric_ckm, compute_ckm_from_geometry
import numpy as np

# Test wrapping numbers
tau_0 = 2.7j
wrapping_up = [
    [(2, 0), (0, 1), (1, 0)],
    [(2, 0), (0, 1), (1, 0)],
    [(2, 0), (0, 1), (1, 0)]
]
wrapping_down = [
    [(1, 1), (1, 0), (0, 1)],
    [(1, 1), (1, 0), (0, 1)],
    [(1, 1), (1, 0), (0, 1)]
]
tau_values = [tau_0] * 3

# Observed CKM values
obs_sin2_12 = 0.051
obs_sin2_23 = 0.00157
obs_sin2_13 = 0.000128
obs_delta = 1.22  # rad
obs_J = 3.0e-5

print("="*80)
print("TEST 1: OPTIMIZATION WITHOUT B-FIELD (5 parameters)")
print("="*80)
print("Downweights CP observables (δ_CP, J_CP) by factor 0.3")
print("This allows good fit to mixing angles even with bad CP phase\n")

result_no_B = optimize_geometric_ckm(
    wrapping_up, wrapping_down, tau_values,
    use_bfield=False,
    verbose=False
)

sigma_opt, alpha_12, alpha_23, alpha_13, lambda_inst = result_no_B[:5]

print(f"\nOptimal parameters (no B-field):")
print(f"  σ_overlap = {sigma_opt:.6f}")
print(f"  α_12 = {alpha_12:.6f}")
print(f"  α_23 = {alpha_23:.6f}")
print(f"  α_13 = {alpha_13:.6f}")
print(f"  λ_inst = {lambda_inst:.6f}")

# Compute CKM with these parameters (no B-field)
sin2_12, sin2_23, sin2_13, delta, J = compute_ckm_from_geometry(
    wrapping_up, wrapping_down, tau_values,
    sigma_overlap=sigma_opt,
    alpha_12=alpha_12,
    alpha_23=alpha_23,
    alpha_13=alpha_13,
    instanton_strength=lambda_inst,
    B_field=None
)

print(f"\nPredicted CKM (no B-field):")
print(f"  sin²θ₁₂ = {sin2_12:.6f}  (obs: {obs_sin2_12:.6f})  Error: {100*abs(sin2_12-obs_sin2_12)/obs_sin2_12:.1f}%")
print(f"  sin²θ₂₃ = {sin2_23:.6f}  (obs: {obs_sin2_23:.6f})  Error: {100*abs(sin2_23-obs_sin2_23)/obs_sin2_23:.1f}%")
print(f"  sin²θ₁₃ = {sin2_13:.6f}  (obs: {obs_sin2_13:.6f})  Error: {100*abs(sin2_13-obs_sin2_13)/obs_sin2_13:.1f}%")
print(f"  δ_CP    = {delta:.3f} rad  (obs: {obs_delta:.2f} rad)  Error: {100*abs(delta-obs_delta)/obs_delta:.1f}%")
print(f"  J_CP    = {J:.3e}       (obs: {obs_J:.2e})       Error: {100*abs(J-obs_J)/obs_J:.1f}%")

print("\n" + "="*80)
print("TEST 2: OPTIMIZATION WITH B-FIELD (8 parameters)")
print("="*80)
print("Equal weight on all 5 CKM observables")
print("B-field flux through 2-cycles: τ_eff = τ + i×B")
print("Should improve CP phase significantly!\n")

result_with_B = optimize_geometric_ckm(
    wrapping_up, wrapping_down, tau_values,
    use_bfield=True,
    verbose=False
)

sigma_opt_B, alpha_12_B, alpha_23_B, alpha_13_B, lambda_inst_B, B_opt = result_with_B[:6]
B1, B2, B3 = B_opt

print(f"\nOptimal parameters (with B-field):")
print(f"  σ_overlap = {sigma_opt_B:.6f}")
print(f"  α_12 = {alpha_12_B:.6f}")
print(f"  α_23 = {alpha_23_B:.6f}")
print(f"  α_13 = {alpha_13_B:.6f}")
print(f"  λ_inst = {lambda_inst_B:.6f}")
print(f"  B₁ = {B1:.6f}")
print(f"  B₂ = {B2:.6f}")
print(f"  B₃ = {B3:.6f}")

# Compute CKM with these parameters (with B-field)
sin2_12_B, sin2_23_B, sin2_13_B, delta_B, J_B = compute_ckm_from_geometry(
    wrapping_up, wrapping_down, tau_values,
    sigma_overlap=sigma_opt_B,
    alpha_12=alpha_12_B,
    alpha_23=alpha_23_B,
    alpha_13=alpha_13_B,
    instanton_strength=lambda_inst_B,
    B_field=B_opt
)

print(f"\nPredicted CKM (with B-field):")
print(f"  sin²θ₁₂ = {sin2_12_B:.6f}  (obs: {obs_sin2_12:.6f})  Error: {100*abs(sin2_12_B-obs_sin2_12)/obs_sin2_12:.1f}%")
print(f"  sin²θ₂₃ = {sin2_23_B:.6f}  (obs: {obs_sin2_23:.6f})  Error: {100*abs(sin2_23_B-obs_sin2_23)/obs_sin2_23:.1f}%")
print(f"  sin²θ₁₃ = {sin2_13_B:.6f}  (obs: {obs_sin2_13:.6f})  Error: {100*abs(sin2_13_B-obs_sin2_13)/obs_sin2_13:.1f}%")
print(f"  δ_CP    = {delta_B:.3f} rad  (obs: {obs_delta:.2f} rad)  Error: {100*abs(delta_B-obs_delta)/obs_delta:.1f}%")
print(f"  J_CP    = {J_B:.3e}       (obs: {obs_J:.2e})       Error: {100*abs(J_B-obs_J)/obs_J:.1f}%")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print("\nMixing angles:")
print(f"  sin²θ₁₂:  {100*abs(sin2_12-obs_sin2_12)/obs_sin2_12:.1f}% (no B) → {100*abs(sin2_12_B-obs_sin2_12)/obs_sin2_12:.1f}% (with B)")
print(f"  sin²θ₂₃:  {100*abs(sin2_23-obs_sin2_23)/obs_sin2_23:.1f}% (no B) → {100*abs(sin2_23_B-obs_sin2_23)/obs_sin2_23:.1f}% (with B)")
print(f"  sin²θ₁₃:  {100*abs(sin2_13-obs_sin2_13)/obs_sin2_13:.1f}% (no B) → {100*abs(sin2_13_B-obs_sin2_13)/obs_sin2_13:.1f}% (with B)")

print("\nCP violation (KEY TEST):")
print(f"  δ_CP:     {100*abs(delta-obs_delta)/obs_delta:.1f}% (no B) → {100*abs(delta_B-obs_delta)/obs_delta:.1f}% (with B)")
print(f"  J_CP:     {100*abs(J-obs_J)/obs_J:.1f}% (no B) → {100*abs(J_B-obs_J)/obs_J:.1f}% (with B)")

delta_improvement = abs(delta-obs_delta) / abs(delta_B-obs_delta)
J_improvement = abs(J-obs_J) / abs(J_B-obs_J)

print(f"\nImprovement factors:")
print(f"  δ_CP error reduced by: {delta_improvement:.1f}×")
print(f"  J_CP error reduced by: {J_improvement:.1f}×")

if delta_improvement > 2.0:
    print("\n✓ SUCCESS: B-field significantly improves CP phase!")
else:
    print("\n⚠ PARTIAL: B-field helps but needs further tuning")

print("\nB-field interpretation:")
print(f"  B₁ = {B1:.3f} → Im[τ₁] shift: {tau_0.imag:.2f} → {tau_0.imag + B1:.2f}")
print(f"  B₂ = {B2:.3f} → Im[τ₂] shift: {tau_0.imag:.2f} → {tau_0.imag + B2:.2f}")
print(f"  B₃ = {B3:.3f} → Im[τ₃] shift: {tau_0.imag:.2f} → {tau_0.imag + B3:.2f}")
print("  τ_eff breaks CP symmetry at fundamental level!")

print("\n" + "="*80)
