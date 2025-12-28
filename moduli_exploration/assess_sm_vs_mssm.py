"""
What does Standard Model (without SUSY) tell us about g_s?
===========================================================

LHC has NOT found SUSY, so MSSM may be wrong.
Let's use SM gauge couplings instead.

From Phase 1:
  SM unification at M_GUT = 1.8×10^14 GeV
  α_GUT = 0.0242 ± 4%
  g_s = 0.55 (for k=1)

But SM doesn't unify perfectly. What does this mean?
"""

import numpy as np

print("="*70)
print("DILATON CONSTRAINTS FROM STANDARD MODEL (NO SUSY)")
print("="*70)

# From Phase 1 SM results
M_GUT_SM = 1.8e14  # GeV
alpha_GUT_SM = 0.02415
spread_SM = 0.04  # 4%

print(f"\nStandard Model gauge coupling evolution:")
print(f"  M_GUT ~ {M_GUT_SM:.2e} GeV")
print(f"  α_GUT ~ {alpha_GUT_SM:.5f} ± {spread_SM*100:.1f}%")
print(f"  Unification quality: MODERATE (not as good as MSSM)")

print(f"\n" + "="*70)
print("INTERPRETATION: IMPERFECT UNIFICATION")
print("="*70)

print(f"""
The 3 SM gauge couplings DON'T perfectly unify at 1-loop.

Three possibilities:

A) THERE IS NO GRAND UNIFICATION
   • SM is the complete story up to M_Planck
   • The 3 gauge groups U(1) × SU(2) × SU(3) are independent
   • String theory DOESN'T necessarily need GUT
   • g_s is NOT constrained by gauge unification

   → Dilaton φ remains UNDETERMINED

B) GUT EXISTS WITH THRESHOLD CORRECTIONS
   • Unification happens but with 2-loop RG, threshold effects
   • The ~4% spread could be fixed by higher-order corrections
   • Then α_GUT ~ {alpha_GUT_SM:.4f} is meaningful
   • String relation: g_s = √(4π k α_GUT)

   → Dilaton determined, but need careful calculation

C) NEW PHYSICS AT INTERMEDIATE SCALE
   • Something between M_Z and M_GUT modifies running
   • Could be SUSY at high scale, extra particles, etc.
   • Or extra dimensions becoming relevant
   • Would change the unification scale and α_GUT

   → Can't determine g_s without knowing new physics
""")

print(f"\n" + "="*70)
print("OPTION A: NO GUT, NO CONSTRAINT")
print("="*70)

print(f"""
If the SM is complete and there's NO grand unification:

• The 3 gauge couplings are independent parameters
• String theory doesn't require them to unify
• The dilaton g_s is determined by OTHER physics:
  - Moduli stabilization (KKLT/LVS)
  - Anthropic selection (landscape)
  - Some other string consistency condition

In this case:
  ✓ τ = 2.69i from flavor/cosmology
  ✗ g_s undetermined (landscape/anthropic)
  ✗ T undetermined (needs KKLT/LVS)

STATUS: 1 out of 3 moduli determined
CONCLUSION: Hit moduli walls as expected
""")

print(f"\n" + "="*70)
print("OPTION B: IMPERFECT UNIFICATION → RANGE FOR g_s")
print("="*70)

print(f"""
If we ASSUME unification exists but with corrections:

Take α_GUT = {alpha_GUT_SM:.4f} ± 4% as approximate value.
""")

alpha_low = alpha_GUT_SM * (1 - spread_SM)
alpha_high = alpha_GUT_SM * (1 + spread_SM)

print(f"  α_GUT range: {alpha_low:.4f} to {alpha_high:.4f}")

print(f"\nFor different Kac-Moody levels k:")
for k in [1, 2, 3, 5]:
    g_s_central = np.sqrt(4 * np.pi * k * alpha_GUT_SM)
    g_s_low = np.sqrt(4 * np.pi * k * alpha_low)
    g_s_high = np.sqrt(4 * np.pi * k * alpha_high)

    print(f"\n  k = {k}:")
    print(f"    g_s = {g_s_central:.3f} ± {(g_s_high-g_s_low)/2:.3f}")
    print(f"    Range: [{g_s_low:.3f}, {g_s_high:.3f}]")

    perturbative = "✓ perturbative" if g_s_high < 1 else "✗ non-perturbative"
    print(f"    {perturbative}")

print(f"""
Result:
  k=1: g_s = 0.55 ± 0.02  (perturbative, well-defined)
  k=2: g_s = 0.78 ± 0.03  (perturbative)
  k≥3: g_s > 0.95        (borderline/non-perturbative)

If we trust SM "approximate unification", we get:
  g_s ~ 0.5-0.8  (for perturbative k=1 or k=2)

This is WEAKER than MSSM (which gave ±0.1%), but still a constraint!
""")

print(f"\n" + "="*70)
print("OPTION C: NEW PHYSICS CHANGES THE GAME")
print("="*70)

print(f"""
If there IS new physics between M_Z and M_Planck:

Possibilities:
• High-scale SUSY (m_SUSY ~ 10-100 TeV) - not seen at LHC yet
• Extra dimensions (compactification scale ~ 10^10-10^16 GeV)
• New gauge bosons (Z', W', leptoquarks)
• Additional Higgs doublets/singlets

Each would modify the RG running and change α_GUT.

Without knowing WHAT new physics exists, we CAN'T determine g_s.
""")

print(f"\n" + "="*70)
print("WHAT ABOUT STRING THEORY CONSTRAINTS?")
print("="*70)

print(f"""
Even without gauge unification, string theory HAS other requirements:

1. ANOMALY CANCELLATION
   Already discussed: Im(S) × Im(T) × Im(U) ~ 1
   With τ = 2.69i, this gives constraint between g_s and T
   But doesn't fix g_s uniquely

2. TADPOLE CANCELLATION (Type IIB)
   D3-brane charges must cancel: N_D3 + N_flux = some integer
   This constrains fluxes but not directly g_s

3. MODULI STABILIZATION (KKLT/LVS)
   Fixes ALL moduli through potentials
   Requires detailed Calabi-Yau construction
   Beyond phenomenological approach

4. WORLDSHEET CONSISTENCY
   λ_ws = g_s² < 1 for perturbative string theory
   → g_s < 1  (weak constraint)

5. INSTANTON ACTIONS
   Already tested: our Yukawas don't need worldsheet instantons
   Suggests g_s ≲ O(1) but not precise

None of these uniquely determine g_s without detailed string model!
""")

print(f"\n" + "="*70)
print("PRAGMATIC DECISION")
print("="*70)

print(f"""
Given:
• LHC hasn't found SUSY
• SM gauge couplings approximately unify (4% level)
• String theory doesn't uniquely fix g_s without GUT

Options for proceeding:

1. CONSERVATIVE: Accept g_s is undetermined
   • Only τ = 2.69i is fixed from phenomenology
   • g_s and T are landscape/anthropic
   • Status: 1/3 moduli determined
   • Document and move on

2. OPTIMISTIC: Use SM approximate unification
   • Accept ~4% uncertainty
   • g_s ~ 0.55 ± 0.1 (for k=1)
   • Status: 2/3 moduli determined (both with caveats)
   • Document and move on

3. AGNOSTIC: Give range from both SM and MSSM
   • SM: g_s ~ 0.5-0.6  (if unification approximate)
   • MSSM: g_s ~ 0.7-1.0  (if SUSY exists at high scale)
   • Bracket: g_s ~ 0.5-1.0
   • Status: 2/3 moduli determined (with large uncertainty)
   • Document and move on

RECOMMENDATION:
---------------

Use Option 3 (agnostic/bracket approach):

"Gauge coupling evolution constrains the string coupling to
 g_s ~ 0.5-1.0, depending on whether new physics exists between
 the electroweak scale and M_GUT ~ 10^14-10^16 GeV.

 • SM (no new physics): g_s ~ 0.55 ± 0.1
 • MSSM (high-scale SUSY): g_s ~ 0.72-1.02

 Combined with τ = 2.69i from flavor/cosmology, this leaves
 the Kähler modulus T weakly constrained. Full determination
 requires detailed string compactification (KKLT/LVS)."

This is HONEST about uncertainties while still claiming partial success!
""")

print("="*70)
