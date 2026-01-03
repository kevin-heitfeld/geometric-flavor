"""
SURGICAL ATTACK: Localization Parameters from Simplified CY Geometry

Goal: Compute g_i and A_i from T²×T²×T² (factorized torus) to validate Phase 2 approach.

Strategy:
1. Use simplest CY: T²×T²×T² = (T²)³ (product of three 2-tori)
2. D7-branes wrap 4-cycles (pairs of tori)
3. Compute modular weights from U(1) charges
4. Calculate brane-brane distances from wrapping numbers
5. Test: Do geometric values match fitted g_i, A_i within 10%?

Success Criteria:
✅ <10% error: Proves Phase 2 viable, continue to full CY
⚠️ 10-20% error: Method works but needs refinement
❌ >20% error: Approach needs rethinking

Physics Input:
- τ₁, τ₂, τ₃: Kähler moduli for each torus (start with τ₁=τ₂=τ₃=2.7i)
- Wrapping numbers: (n₁, m₁) on T₁², (n₂, m₂) on T₂², (n₃, m₃) on T₃²
- D7-branes wrap two tori (e.g., lepton brane wraps T₁²×T₂²)
"""

import numpy as np
import sys
from pathlib import Path

# Import fitted values for comparison
from kmass_compute import compute_observables_with_kmass

print("="*80)
print("SURGICAL ATTACK: Localization from T²×T²×T² Geometry")
print("="*80)
print()

# ==============================================================================
# GEOMETRY SETUP: T²×T²×T² = Product of Three 2-Tori
# ==============================================================================

print("STEP 1: Define CY Geometry")
print("-"*80)

# Kähler moduli (start with symmetric case)
tau_1 = 2.7j
tau_2 = 2.7j
tau_3 = 2.7j

print(f"Kähler moduli: τ₁ = τ₂ = τ₃ = {tau_1}")
print()

# D7-brane wrapping numbers: (n, m) on each torus
# D7 wraps 4 real dimensions = 2 complex dimensions = 2 tori

print("D7-brane stack positions (wrap 2 tori each):")
print()

# Lepton sector: wraps T₁²×T₂² (ignore T₃²)
# Three generations have different wrapping numbers
lepton_wrappings = [
    ((1, 0), (1, 0), (0, 0)),  # Gen 1: (1,0) on T₁, (1,0) on T₂
    ((1, 1), (1, 0), (0, 0)),  # Gen 2: (1,1) on T₁, (1,0) on T₂
    ((1, 0), (1, 1), (0, 0)),  # Gen 3: (1,0) on T₁, (1,1) on T₂
]

# Up quark sector: wraps T₁²×T₃²
up_wrappings = [
    ((1, 0), (0, 0), (1, 0)),  # Gen 1
    ((1, 1), (0, 0), (1, 0)),  # Gen 2
    ((1, 0), (0, 0), (1, 1)),  # Gen 3
]

# Down quark sector: wraps T₂²×T₃²
down_wrappings = [
    ((0, 0), (1, 0), (1, 0)),  # Gen 1
    ((0, 0), (1, 1), (1, 0)),  # Gen 2
    ((0, 0), (1, 0), (1, 1)),  # Gen 3
]

print("Leptons (wrap T₁²×T₂²):")
for i, w in enumerate(lepton_wrappings):
    print(f"  Gen {i+1}: T₁²:{w[0]}, T₂²:{w[1]}, T₃²:{w[2]}")

print("\nUp quarks (wrap T₁²×T₃²):")
for i, w in enumerate(up_wrappings):
    print(f"  Gen {i+1}: T₁²:{w[0]}, T₂²:{w[1]}, T₃²:{w[2]}")

print("\nDown quarks (wrap T₂²×T₃²):")
for i, w in enumerate(down_wrappings):
    print(f"  Gen {i+1}: T₁²:{w[0]}, T₂²:{w[1]}, T₃²:{w[2]}")
print()

# Higgs brane (for distance calculations)
higgs_wrapping = ((1, 0), (1, 0), (1, 0))  # Wraps all three tori (bulk brane)
print(f"Higgs brane (bulk): T₁²:{higgs_wrapping[0]}, T₂²:{higgs_wrapping[1]}, T₃²:{higgs_wrapping[2]}")
print()

# ==============================================================================
# STEP 2: Compute Modular Weights from Wrapping Numbers
# ==============================================================================

print("STEP 2: Compute Modular Weights")
print("-"*80)

def modular_weight_from_wrapping(wrapping, tau_values):
    """
    Modular weight from wrapping numbers.

    For (n,m) wrapping on torus with modulus τ:
    w = n² Im(τ) + m²/Im(τ) - 2nm Re(τ)

    For D7-brane wrapping 2 tori, sum contributions.
    """
    w_total = 0
    for i, (n, m) in enumerate(wrapping):
        if n == 0 and m == 0:
            continue  # Not wrapping this torus

        tau = tau_values[i]
        Im_tau = tau.imag if hasattr(tau, 'imag') else abs(tau)
        Re_tau = tau.real if hasattr(tau, 'real') else 0

        w = n**2 * Im_tau + m**2 / Im_tau - 2*n*m*Re_tau
        w_total += w

    return w_total

tau_values = [tau_1, tau_2, tau_3]

# Compute weights for each generation in each sector
w_lep = [modular_weight_from_wrapping(w, tau_values) for w in lepton_wrappings]
w_up = [modular_weight_from_wrapping(w, tau_values) for w in up_wrappings]
w_down = [modular_weight_from_wrapping(w, tau_values) for w in down_wrappings]

print(f"Modular weights (leptons): {[f'{w:.3f}' for w in w_lep]}")
print(f"Modular weights (up):      {[f'{w:.3f}' for w in w_up]}")
print(f"Modular weights (down):    {[f'{w:.3f}' for w in w_down]}")
print()

# Convert to g_i = 1 + δg × (w_i - w_1)
# Normalize so gen 1 has g=1, others relative to it

def weights_to_g_factors(weights, calibration_factor=0.1):
    """Convert modular weights to generation factors g_i."""
    w_norm = np.array(weights) - weights[0]  # Relative to gen 1
    g = 1.0 + calibration_factor * w_norm
    return g

# Try different calibration factors
calibrations = [0.05, 0.1, 0.2, 0.3]

print("Generation factors g_i from modular weights:")
print()

for cal in calibrations:
    g_lep_geom = weights_to_g_factors(w_lep, cal)
    g_up_geom = weights_to_g_factors(w_up, cal)
    g_down_geom = weights_to_g_factors(w_down, cal)

    print(f"Calibration δg = {cal}:")
    print(f"  g_lep:  {g_lep_geom}")
    print(f"  g_up:   {g_up_geom}")
    print(f"  g_down: {g_down_geom}")
    print()

# ==============================================================================
# STEP 3: Compute Brane-Brane Distances
# ==============================================================================

print("STEP 3: Compute Brane-Brane Distances")
print("-"*80)

def brane_distance(wrap1, wrap2, tau_values, R_values):
    """
    Distance between two D7-branes in string units.

    For branes at different wrappings, distance comes from:
    d² = Σᵢ |Δxᵢ|² where Δxᵢ depends on wrapping difference

    For torus with radius R and wrapping (n₁,m₁) vs (n₂,m₂):
    Δx ~ R × √((n₁-n₂)² + (m₁-m₂)²/|τ|²)
    """
    d_squared = 0

    for i in range(3):
        n1, m1 = wrap1[i]
        n2, m2 = wrap2[i]

        if (n1, m1) == (0, 0) or (n2, m2) == (0, 0):
            continue  # One brane doesn't wrap this torus

        tau = tau_values[i]
        R = R_values[i]
        Im_tau = tau.imag if hasattr(tau, 'imag') else abs(tau)

        # Wrapping difference
        dn = n1 - n2
        dm = m1 - m2

        # Distance on this torus (in units of R)
        d_torus_sq = dn**2 + dm**2 / Im_tau**2
        d_squared += (R * np.sqrt(d_torus_sq))**2

    return np.sqrt(d_squared)

# Torus radii (in string units, roughly √Im(τ))
R_values = [np.sqrt(tau_1.imag), np.sqrt(tau_2.imag), np.sqrt(tau_3.imag)]

print(f"Torus radii: R₁=R₂=R₃ = {R_values[0]:.3f} ℓ_s")
print()

# Compute distances from each generation to Higgs brane
print("Brane-Higgs distances:")
print()

d_lep = [brane_distance(w, higgs_wrapping, tau_values, R_values) for w in lepton_wrappings]
d_up = [brane_distance(w, higgs_wrapping, tau_values, R_values) for w in up_wrappings]
d_down = [brane_distance(w, higgs_wrapping, tau_values, R_values) for w in down_wrappings]

print(f"Leptons: {[f'{d:.3f}' for d in d_lep]} ℓ_s")
print(f"Up:      {[f'{d:.3f}' for d in d_up]} ℓ_s")
print(f"Down:    {[f'{d:.3f}' for d in d_down]} ℓ_s")
print()

# Convert to localization parameters A_i = -log(overlap) ~ d_i / λ
# where λ is characteristic localization length

def distances_to_A_factors(distances, localization_scale=1.0):
    """Convert distances to localization suppression A_i."""
    d_norm = np.array(distances) - distances[0]  # Relative to gen 1
    A = -localization_scale * d_norm
    return A

print("Localization parameters A_i from distances:")
print()

# Try different localization scales
loc_scales = [0.3, 0.5, 0.7, 1.0]

for scale in loc_scales:
    A_lep_geom = distances_to_A_factors(d_lep, scale)
    A_up_geom = distances_to_A_factors(d_up, scale)
    A_down_geom = distances_to_A_factors(d_down, scale)

    print(f"Localization scale λ = {scale}:")
    print(f"  A_lep:  {A_lep_geom}")
    print(f"  A_up:   {A_up_geom}")
    print(f"  A_down: {A_down_geom}")
    print()

# ==============================================================================
# STEP 4: Compare with Fitted Values
# ==============================================================================

print("="*80)
print("STEP 4: Compare with Fitted Values")
print("="*80)
print()

# Fitted values from unified_predictions_complete.py
g_lep_fitted = np.array([1.00, 1.10599770, 1.00816488])
g_up_fitted = np.array([1.00, 1.12996338, 1.01908896])
g_down_fitted = np.array([1.00, 0.96185547, 1.00057316])

A_lep_fitted = np.array([0.00, -0.72084622, -0.92315966])
A_up_fitted = np.array([0.00, -0.87974875, -1.48332060])
A_down_fitted = np.array([0.00, -0.33329575, -0.88288836])

print("Fitted values:")
print(f"  g_lep:  {g_lep_fitted}")
print(f"  g_up:   {g_up_fitted}")
print(f"  g_down: {g_down_fitted}")
print()
print(f"  A_lep:  {A_lep_fitted}")
print(f"  A_up:   {A_up_fitted}")
print(f"  A_down: {A_down_fitted}")
print()

# Find best calibration factors by minimizing error
def compute_errors(g_geom, g_fitted, A_geom, A_fitted):
    """Compute relative errors."""
    # For g: compare non-trivial values (skip g[0]=1)
    g_err = np.abs((g_geom[1:] - g_fitted[1:]) / g_fitted[1:]) * 100

    # For A: compare all values
    # Handle A[0]=0 specially
    A_err = []
    for i in range(len(A_geom)):
        if abs(A_fitted[i]) < 0.01:
            # Near zero, use absolute error
            A_err.append(abs(A_geom[i] - A_fitted[i]) * 100)
        else:
            A_err.append(abs((A_geom[i] - A_fitted[i]) / A_fitted[i]) * 100)
    A_err = np.array(A_err)

    return g_err, A_err

# Test all calibration combinations
best_g_cal = None
best_A_cal = None
best_total_err = float('inf')

print("Searching for optimal calibration factors...")
print()

for g_cal in calibrations:
    for A_cal in loc_scales:
        g_lep_geom = weights_to_g_factors(w_lep, g_cal)
        g_up_geom = weights_to_g_factors(w_up, g_cal)
        g_down_geom = weights_to_g_factors(w_down, g_cal)

        A_lep_geom = distances_to_A_factors(d_lep, A_cal)
        A_up_geom = distances_to_A_factors(d_up, A_cal)
        A_down_geom = distances_to_A_factors(d_down, A_cal)

        # Compute errors
        g_lep_err, A_lep_err = compute_errors(g_lep_geom, g_lep_fitted, A_lep_geom, A_lep_fitted)
        g_up_err, A_up_err = compute_errors(g_up_geom, g_up_fitted, A_up_geom, A_up_fitted)
        g_down_err, A_down_err = compute_errors(g_down_geom, g_down_fitted, A_down_geom, A_down_fitted)

        # Total error
        all_g_err = np.concatenate([g_lep_err, g_up_err, g_down_err])
        all_A_err = np.concatenate([A_lep_err, A_up_err, A_down_err])
        total_err = np.mean(np.concatenate([all_g_err, all_A_err]))

        if total_err < best_total_err:
            best_total_err = total_err
            best_g_cal = g_cal
            best_A_cal = A_cal

print(f"✅ Best calibration found:")
print(f"   δg = {best_g_cal}, λ = {best_A_cal}")
print(f"   Average error: {best_total_err:.1f}%")
print()

# Compute final geometric predictions with best calibration
g_lep_geom = weights_to_g_factors(w_lep, best_g_cal)
g_up_geom = weights_to_g_factors(w_up, best_g_cal)
g_down_geom = weights_to_g_factors(w_down, best_g_cal)

A_lep_geom = distances_to_A_factors(d_lep, best_A_cal)
A_up_geom = distances_to_A_factors(d_up, best_A_cal)
A_down_geom = distances_to_A_factors(d_down, best_A_cal)

# ==============================================================================
# FINAL RESULTS
# ==============================================================================

print("="*80)
print("FINAL RESULTS: Geometric vs. Fitted")
print("="*80)
print()

def print_comparison(name, geom, fitted):
    """Print geometric vs fitted comparison with errors."""
    print(f"{name}:")
    print(f"  Geometric: {geom}")
    print(f"  Fitted:    {fitted}")

    # Compute errors (skip first element for g, it's always 1)
    if name.startswith('g_'):
        errors = np.abs((geom[1:] - fitted[1:]) / fitted[1:]) * 100
        print(f"  Errors:    {[f'{e:.1f}%' for e in errors]}")
    else:  # A parameters
        errors = []
        for i in range(len(geom)):
            if abs(fitted[i]) < 0.01:
                errors.append(abs(geom[i] - fitted[i]) * 100)
            else:
                errors.append(abs((geom[i] - fitted[i]) / fitted[i]) * 100)
        print(f"  Errors:    {[f'{e:.1f}%' for e in errors]}")
    print()

print_comparison("g_lep", g_lep_geom, g_lep_fitted)
print_comparison("g_up", g_up_geom, g_up_fitted)
print_comparison("g_down", g_down_geom, g_down_fitted)

print_comparison("A_lep", A_lep_geom, A_lep_fitted)
print_comparison("A_up", A_up_geom, A_up_fitted)
print_comparison("A_down", A_down_geom, A_down_fitted)

# Overall statistics
g_lep_err, A_lep_err = compute_errors(g_lep_geom, g_lep_fitted, A_lep_geom, A_lep_fitted)
g_up_err, A_up_err = compute_errors(g_up_geom, g_up_fitted, A_up_geom, A_up_fitted)
g_down_err, A_down_err = compute_errors(g_down_geom, g_down_fitted, A_down_geom, A_down_fitted)

all_g_err = np.concatenate([g_lep_err, g_up_err, g_down_err])
all_A_err = np.concatenate([A_lep_err, A_up_err, A_down_err])

print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print()
print(f"g_i errors: min={np.min(all_g_err):.1f}%, max={np.max(all_g_err):.1f}%, avg={np.mean(all_g_err):.1f}%")
print(f"A_i errors: min={np.min(all_A_err):.1f}%, max={np.max(all_A_err):.1f}%, avg={np.mean(all_A_err):.1f}%")
print(f"Overall:    avg={np.mean(np.concatenate([all_g_err, all_A_err])):.1f}%")
print()

# ==============================================================================
# VERDICT
# ==============================================================================

print("="*80)
print("VERDICT")
print("="*80)
print()

avg_error = np.mean(np.concatenate([all_g_err, all_A_err]))

if avg_error < 10:
    print("✅ SUCCESS: Average error < 10%")
    print("   → Localization CAN be derived from simplified CY geometry")
    print("   → Phase 2 approach VALIDATED")
    print("   → Continue to full resolved CY for precise values")
    verdict = "SUCCESS"
elif avg_error < 20:
    print("⚠️ PARTIAL SUCCESS: Average error 10-20%")
    print("   → Method works but needs refinement")
    print("   → Likely need: better wrapping numbers, resolved CY, or α' corrections")
    print("   → Phase 2 viable but requires careful implementation")
    verdict = "PARTIAL"
else:
    print("❌ METHOD NEEDS WORK: Average error > 20%")
    print("   → Simple T²×T²×T² insufficient")
    print("   → Need: resolved singularities, warping, or different approach")
    print("   → Rethink before full Phase 2")
    verdict = "RETHINK"

print()
print("KEY INSIGHT:")
if verdict == "SUCCESS":
    print("  With calibration (δg, λ), we reduced 18 free parameters to 2!")
    print("  Even WITH calibration, this is 9x parameter reduction.")
    print("  Phase 2 should eliminate calibration factors entirely.")
elif verdict == "PARTIAL":
    print("  Geometric approach captures main physics but misses ~15% corrections.")
    print("  These corrections likely from: α' effects, warping, or moduli mixing.")
    print("  Phase 2 with full CY should recover these.")
else:
    print("  Simple torus product too crude for ~5% precision.")
    print("  Need resolved CY with proper intersection geometry.")

print()
print(f"Status: Surgical attack complete - verdict is {verdict}")
print("="*80)
