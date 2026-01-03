"""
Analyze patterns in fitted τ_i values to guide theoretical derivation
"""

import numpy as np

print("="*80)
print("ANALYSIS OF FITTED τ_i PATTERNS")
print("="*80)
print()

# Fitted values from optimization
tau_lep = np.array([2.299, 2.645, 2.577])
tau_up = np.array([2.225, 2.934, 2.661])
tau_down = np.array([2.290, 1.988, 2.013])

print("FITTED VALUES:")
print("-"*80)
print(f"Leptons:    τ = [{tau_lep[0]:.3f}i, {tau_lep[1]:.3f}i, {tau_lep[2]:.3f}i]")
print(f"Up quarks:  τ = [{tau_up[0]:.3f}i, {tau_up[1]:.3f}i, {tau_up[2]:.3f}i]")
print(f"Down quarks: τ = [{tau_down[0]:.3f}i, {tau_down[1]:.3f}i, {tau_down[2]:.3f}i]")
print()

# Statistics
all_tau = np.concatenate([tau_lep, tau_up, tau_down])
print("STATISTICS:")
print("-"*80)
print(f"Overall mean: {np.mean(all_tau):.3f}i")
print(f"Overall std:  {np.std(all_tau):.3f}i")
print(f"Range: [{np.min(all_tau):.3f}i, {np.max(all_tau):.3f}i]")
print(f"Predicted τ₀ = 2.700i (from topology)")
print(f"Deviation from τ₀: {abs(np.mean(all_tau) - 2.7):.3f}i ({abs(np.mean(all_tau) - 2.7)/2.7*100:.1f}%)")
print()

# Generation averages
print("GENERATION AVERAGES (across all fermions):")
print("-"*80)
tau_gen1 = np.mean([tau_lep[0], tau_up[0], tau_down[0]])
tau_gen2 = np.mean([tau_lep[1], tau_up[1], tau_down[1]])
tau_gen3 = np.mean([tau_lep[2], tau_up[2], tau_down[2]])

print(f"1st generation: τ₁ = {tau_gen1:.3f}i")
print(f"2nd generation: τ₂ = {tau_gen2:.3f}i")
print(f"3rd generation: τ₃ = {tau_gen3:.3f}i")
print()

# Check if there's a pattern
delta_21 = tau_gen2 - tau_gen1
delta_32 = tau_gen3 - tau_gen2
print(f"Δτ(2-1) = {delta_21:.3f}i")
print(f"Δτ(3-2) = {delta_32:.3f}i")
print()

# Sector averages
print("SECTOR AVERAGES (across generations):")
print("-"*80)
print(f"Leptons:    {np.mean(tau_lep):.3f}i (std: {np.std(tau_lep):.3f}i)")
print(f"Up quarks:  {np.mean(tau_up):.3f}i (std: {np.std(tau_up):.3f}i)")
print(f"Down quarks: {np.mean(tau_down):.3f}i (std: {np.std(tau_down):.3f}i)")
print()

# Simplified model: Single τ per generation
print("="*80)
print("SIMPLIFIED MODEL: SINGLE τ_i PER GENERATION")
print("="*80)
print()
print("If we use generation-average τ_i for ALL fermions:")
print(f"  τ_1 = {tau_gen1:.3f}i")
print(f"  τ_2 = {tau_gen2:.3f}i")
print(f"  τ_3 = {tau_gen3:.3f}i")
print()
print("This would give:")
print(f"  - Same hierarchies across leptons, up quarks, down quarks")
print(f"  - Only 2 free parameters (Δτ₂, Δτ₃) instead of 9")
print(f"  - More predictive power")
print()

# Check deviations from generation averages
print("DEVIATIONS FROM GENERATION AVERAGES:")
print("-"*80)
dev_lep = tau_lep - np.array([tau_gen1, tau_gen2, tau_gen3])
dev_up = tau_up - np.array([tau_gen1, tau_gen2, tau_gen3])
dev_down = tau_down - np.array([tau_gen1, tau_gen2, tau_gen3])

print(f"Leptons:     [{dev_lep[0]:+.3f}i, {dev_lep[1]:+.3f}i, {dev_lep[2]:+.3f}i]")
print(f"Up quarks:   [{dev_up[0]:+.3f}i, {dev_up[1]:+.3f}i, {dev_up[2]:+.3f}i]")
print(f"Down quarks: [{dev_down[0]:+.3f}i, {dev_down[1]:+.3f}i, {dev_down[2]:+.3f}i]")
print()
print(f"RMS deviation: {np.sqrt(np.mean([dev_lep**2, dev_up**2, dev_down**2])):.3f}i")
print()

print("="*80)
print("CONCLUSIONS")
print("="*80)
print()
print("1. **Fitted τ_i cluster around predicted τ₀ = 2.7i** ✓")
print(f"   Mean: {np.mean(all_tau):.3f}i vs predicted 2.700i ({abs(np.mean(all_tau)-2.7)/2.7*100:.1f}% off)")
print()
print("2. **Generation pattern exists**:")
print(f"   τ₁ ≈ {tau_gen1:.2f}i < τ₂ ≈ {tau_gen2:.2f}i > τ₃ ≈ {tau_gen3:.2f}i")
print("   2nd generation has highest τ! (Not monotonic)")
print()
print("3. **Sector-specific corrections** ~0.1-0.4i needed")
print("   Can't use single τ_i for all fermions")
print()
print("NEXT STEPS:")
print("  A) Use generation-averaged τ_i for simplicity → ~20-40% errors")
print("  B) Keep sector-specific τ_i → 0% errors but need to derive 9 values")
print("  C) Find geometric origin of sector dependence (U(1) charges? Brane positions?)")
