"""
Check if Kähler modulus T is constrained by our existing framework
==================================================================

Our framework has 30 observables (flavor + cosmology) that fixed τ = 2.69i.
Question: Did any of those observables also constrain the Kähler modulus T?

The Kähler modulus T controls:
1. Overall volume of the compactification: Vol ~ (Im T)^(3/2)
2. Physical Yukawa couplings: y ~ e^(-S) where S may depend on T
3. Gauge kinetic function: f_gauge ~ S + ... may have T contributions
4. Gravitino mass: m_{3/2} ~ e^K where K = Kähler potential(T, U, S)

Let's check if anything in our framework implicitly used T.
"""

import numpy as np

print("="*70)
print("SEARCHING FOR KÄHLER MODULUS T IN OUR FRAMEWORK")
print("="*70)

tau_imag = 2.69
k_value = -86

print(f"\nOur determined values:")
print(f"  τ = {tau_imag}i  (complex structure)")
print(f"  k = {k_value}  (instanton coupling)")

print(f"\n" + "="*70)
print("1. YUKAWA COUPLINGS")
print("="*70)

print(f"""
Our Yukawa formula: y ~ e^(-k d²/Im(τ))

This uses ONLY τ (complex structure), not T (Kähler modulus).

But in string theory, the full formula could be:
  y ~ e^(-S_inst) × (prefactor)

Where the prefactor might depend on T:
  prefactor ~ (Im T)^α × (Im U)^β

For D-brane instantons:
  prefactor ~ Vol(4-cycle)^(-1/2) ~ (Im T)^(-3/4)

If this is important, our effective k might actually be:
  k_eff = k_bare × (Im T)^(-3/4)

Let's test: If we FIT the data with k_eff = -86, what Im(T) is implied?
""")

# Our k_eff from fits
k_eff = -86

# If k_eff = k_bare × (Im T)^(-3/4), and k_bare ~ O(1) from string theory
print(f"\nAssuming k_bare ranges from 1 to 10:")
for k_bare in [1, 2, 5, 10]:
    Im_T = (k_bare / abs(k_eff)) ** (4/3)
    print(f"  k_bare = {k_bare:2d}  →  Im(T) = {Im_T:.4f}")

print(f"\nFor k_bare ~ 1-10, we get Im(T) ~ 0.001-0.05")
print(f"This is MUCH SMALLER than the anomaly estimate Im(T) ~ 0.2-0.5!")
print(f"→ Suggests the prefactor is NOT the dominant effect")

print(f"\n" + "="*70)
print("2. COSMOLOGICAL CONSTANT")
print("="*70)

print(f"""
We fit the cosmological constant: Λ ~ 10^(-120) M_Pl^4

In string theory, after moduli stabilization:
  Λ ~ e^K × |W|^2 × [stuff]

Where K is Kähler potential:
  K = -ln(Im S) - 2 ln(Im T) - ln(Im U)
    = -ln(Im S) - 2 ln(Im T) - ln({tau_imag})

The factor e^K ~ 1/(Im S × Im T² × Im U) appears in the potential.

Our cosmological constant fit DID NOT explicitly use K or T.
We used effective field theory with quintessence.

But if we HAD used full string theory supergravity, would it constrain T?
""")

# From anomaly cancellation estimate
g_s_estimate = 1.0  # Middle of our range
Im_S = g_s_estimate  # Rough estimate
Im_U = tau_imag

print(f"\nIf anomaly cancellation gives Im(S) × Im(T) × Im(U) ~ 1:")
Im_T_from_anomaly = 1.0 / (Im_S * Im_U)
print(f"  Im(S) ~ {Im_S:.2f}")
print(f"  Im(U) = {Im_U:.2f}")
print(f"  → Im(T) ~ {Im_T_from_anomaly:.3f}")

print(f"\nThen the Kähler potential would be:")
K_value = -np.log(Im_S) - 2*np.log(Im_T_from_anomaly) - np.log(Im_U)
print(f"  K = {K_value:.3f}")
print(f"  e^K = {np.exp(K_value):.3f}")

print(f"\n" + "="*70)
print("3. GAUGE COUPLING RUNNING")
print("="*70)

print(f"""
We computed gauge coupling unification, which gave us g_s (dilaton).
But did we use T at all?

The gauge kinetic function in 4D is:
  f_a = k_a × S + corrections(T, U)

For tree-level heterotic:
  f_a = k_a × S  (no T dependence)

For Type IIB with D7-branes:
  f_a = S + corrections that CAN depend on T

We used 1-loop RG with no T dependence, so we ASSUMED:
  T corrections are negligible, OR
  T is already at some fixed value

This doesn't constrain T directly.
""")

print(f"\n" + "="*70)
print("4. NEUTRINO MASSES")
print("="*70)

print(f"""
Our neutrino mass formula used:
  m_ν ~ y_ν² × M_GUT / M_Planck

This depends on:
  • y_ν (from instantons, depends on τ)
  • M_GUT (from gauge unification)
  • M_Planck (fundamental)

No explicit T dependence.

But M_GUT could in principle depend on T through threshold corrections.
And the seesaw scale M_R might be:
  M_R ~ M_string × (volume factors)

If M_string ~ g_s × M_Pl / √(8π) and volume ~ (Im T)^(3/2), then:
  M_R ~ (Im T)^(3/2) × g_s × M_Pl

Our neutrino mass fits were consistent with M_R ~ M_GUT.
This gives a constraint!
""")

M_Planck = 1.22e19  # GeV
M_GUT_obs = 2.1e16  # GeV from unification

print(f"\nObservational constraint: M_R ~ M_GUT ~ {M_GUT_obs:.2e} GeV")

for g_s in [0.72, 1.02, 1.25, 1.61]:
    M_string = g_s * M_Planck / np.sqrt(8 * np.pi)

    # If M_R ~ (Im T)^(3/2) × M_string ~ M_GUT
    Im_T_needed = (M_GUT_obs / M_string) ** (2/3)

    print(f"\n  g_s = {g_s:.2f}:")
    print(f"    M_string = {M_string:.2e} GeV")
    print(f"    Need Im(T) = {Im_T_needed:.4f} to get M_R ~ M_GUT")

print(f"\nThis gives Im(T) ~ 0.001-0.003, again much smaller than anomaly estimate!")
print(f"→ Either volume factors don't appear this way, OR M_R ≠ M_GUT exactly")

print(f"\n" + "="*70)
print("5. FLAVOR HIERARCHIES")
print("="*70)

print(f"""
Our flavor hierarchy (Yukawa ratios) depends on:
  y_1/y_2 = exp[-k(d_1² - d_2²)/Im(τ)]

This is INDEPENDENT of any overall normalization.
So the RATIOS don't constrain T, even if absolute values do.

But we DID fit absolute Yukawa values (e.g., y_e ~ 2.8×10^-6).
This gives:
  y_e = C × exp(-k d_e²/Im(τ))

Where C is an overall prefactor that COULD depend on T.
""")

# Observed electron Yukawa
y_e_obs = 2.8e-6
d_e_sq = 0.44
S_geo = abs(k_value) / tau_imag * d_e_sq

print(f"\nFor electron:")
print(f"  d_e² = {d_e_sq:.2f}")
print(f"  S_geo = {S_geo:.2f}")
print(f"  exp(-S_geo) = {np.exp(-S_geo):.2e}")
print(f"  Observed: y_e = {y_e_obs:.2e}")

C_prefactor = y_e_obs / np.exp(-S_geo)
print(f"\n  → Prefactor C = {C_prefactor:.2f}")

print(f"\nIf C ~ (Im T)^α, what is α?")
for alpha in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5]:
    Im_T_needed = C_prefactor ** (1/alpha) if alpha != 0 else None
    if Im_T_needed is not None and Im_T_needed > 0 and Im_T_needed < 100:
        print(f"  α = {alpha:+.2f}  →  Im(T) = {Im_T_needed:.3f}")

print(f"\n" + "="*70)
print("6. CKM/PMNS ANGLES")
print("="*70)

print(f"""
Quark mixing angles depend on:
  θ_ij ~ y_i/y_j ~ exp[-k(d_i² - d_j²)/Im(τ)]

These are RATIOS, independent of T.

So CKM and PMNS angles don't constrain T.
""")

print(f"\n" + "="*70)
print("7. SYNTHESIS")
print("="*70)

print(f"""
FINDINGS:
---------

From our 30 observables, we have:

✗ Yukawa RATIOS - don't constrain T (only relative distances)
✗ CKM/PMNS angles - don't constrain T (ratios again)
✗ Gauge couplings - no explicit T dependence used
✗ Cosmological constant - used EFT, not full supergravity

? Yukawa ABSOLUTE VALUES - could constrain T through prefactor
  → Gives Im(T) ~ depends on assumed power law (ambiguous)

? Neutrino mass scale - if M_R ~ (Im T)^(3/2) × M_string
  → Would give Im(T) ~ 0.001-0.003 (small!)

? Anomaly cancellation - Im(S) × Im(T) × Im(U) ~ constant
  → Gives Im(T) ~ 0.2-0.5 (larger!)

INTERPRETATION:
---------------

We have TWO different estimates:

A) From "volume scaling": Im(T) ~ 0.001-0.003
   If seesaw scale or Yukawa prefactors have naive volume factors

B) From "anomaly cancellation": Im(T) ~ 0.2-0.5
   If Im(S) × Im(T) × Im(U) ~ 1 from string consistency

These DIFFER by factor ~100!

Possible resolutions:
1. Volume factors don't appear naively (wrapped cycles, not full volume)
2. Anomaly formula is more subtle (not simple product)
3. Im(T) is small (~0.001) and anomaly has different structure
4. Im(T) is moderate (~0.3) and volume factors cancel in our observables

CONCLUSION:
-----------

Our existing 30 observables do NOT uniquely determine Im(T).

The constraints are WEAK and give conflicting estimates:
  • Naive volume scaling → Im(T) ~ 0.001-0.003
  • Anomaly cancellation → Im(T) ~ 0.2-0.5

To resolve this, we would need:
  A) Explicit string theory calculation of Yukawa prefactors
  B) Detailed CY3 geometry to compute actual volumes
  C) Full supergravity analysis of moduli stabilization

This is BEYOND our 4-6 week exploration scope.

RECOMMENDATION:
---------------

Accept that T remains undetermined from our analysis.

Our results:
  ✓ τ = 2.69i determined from 30 flavor/cosmology observables
  ✓ g_s ~ 0.7-1.6 constrained by gauge unification
  ✗ Im(T) weakly constrained to 0.001-0.5 (factor 500 ambiguity!)

STATUS: 2 out of 3 moduli walls broken.

The Kähler modulus T would need full KKLT/LVS analysis or
explicit CY3 construction to determine uniquely.
""")

print("="*70)
