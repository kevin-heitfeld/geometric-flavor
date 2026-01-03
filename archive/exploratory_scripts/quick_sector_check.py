"""
Quick check: Apply w = -2q₃ + q₄ to ALL fermion sectors
Week 2, Day 14 - Testing if formula and normalization issues are universal

Goal: See if the SAME pattern (electron perfect, heavier generations ~3-4× too large)
appears in quarks and neutrinos, or if it's lepton-specific.
"""

import numpy as np

# Experimental Yukawa couplings
# For leptons: directly computed in compute_yukawa_matrix_full.py
# Y = √2 × m / v where v = 246 GeV

v_higgs = 246.0  # GeV

# CHARGED LEPTONS - Yukawa couplings
leptons_exp = {
    'electron': 2.80e-6,     # From our Week 2 calculation
    'muon': 6.09e-4,         # Experiment
    'tau': 1.04e-2           # Experiment
}

# QUARKS - Convert masses to Yukawas
quarks_up_exp = {
    'up': np.sqrt(2) * 2.16e-3 / v_higgs,        # 1.24e-5
    'charm': np.sqrt(2) * 1.27 / v_higgs,         # 7.30e-3
    'top': np.sqrt(2) * 172.57 / v_higgs          # 0.992 (almost 1!)
}

quarks_down_exp = {
    'down': np.sqrt(2) * 4.67e-3 / v_higgs,      # 2.69e-5
    'strange': np.sqrt(2) * 93.4e-3 / v_higgs,    # 5.37e-4
    'bottom': np.sqrt(2) * 4.18 / v_higgs         # 2.40e-2
}

# NEUTRINOS - Yukawa couplings (assuming Dirac, no seesaw)
# Y_ν = √2 × m_ν / v ~ 10^-12 for m_ν ~ 0.05 eV
neutrinos_exp = {
    'nu1': 0.0,  # Effectively zero
    'nu2': np.sqrt(2) * np.sqrt(7.53e-5 * 1e-18) / v_higgs,
    'nu3': np.sqrt(2) * np.sqrt(2.5e-3 * 1e-18) / v_higgs
}# Modular weights from w = -2q₃ + q₄
# Assume same Z₃×Z₄ orbifold structure for all sectors

def modular_weight(q3, q4):
    """w = -2q₃ + q₄"""
    return -2*q3 + q4

# HYPOTHESIS: All sectors use same quantum number pattern
# but may differ in overall coupling strength

def test_quantum_number_assignment(sector_name, assignments, exp_values, tau=2.69j):
    """
    Test if quantum number assignment reproduces hierarchy.

    assignments: dict {particle: (q3, q4)}
    exp_values: dict {particle: experimental mass/yukawa}
    """
    print(f"\n{'='*70}")
    print(f"{sector_name.upper()}")
    print(f"{'='*70}")

    # Compute modular weights
    weights = {}
    for particle, (q3, q4) in assignments.items():
        w = modular_weight(q3, q4)
        weights[particle] = w
        print(f"{particle:12s}: (q₃={q3}, q₄={q4}) → w = {w:+.1f}")

    print()

    # Compute LO Yukawa scaling: Y ∝ (Imτ)^(-w) × |η(τ)|^(-6w)
    Im_tau = np.imag(tau)

    # Dedekind eta function
    q = np.exp(2 * np.pi * 1j * tau)
    eta = q**(1/24) * np.prod([1 - q**n for n in range(1, 30)])

    yukawas_calc = {}
    for particle, w in weights.items():
        yukawas_calc[particle] = (Im_tau)**(-w) * np.abs(eta)**(-6*w)

    # Normalize calculated values to lightest particle
    particles = list(exp_values.keys())
    lightest = particles[0]  # Assume ordered light→heavy

    norm = exp_values[lightest] / yukawas_calc[lightest]
    yukawas_calc_norm = {p: y * norm for p, y in yukawas_calc.items()}    # Compare
    print(f"{'Particle':<12} {'w':>6} {'Y_calc':>12} {'Y_exp':>12} {'Ratio':>8} {'Error':>8}")
    print("-" * 70)

    errors = []
    for particle in particles:
        y_calc = yukawas_calc_norm[particle]
        y_exp = exp_values[particle]
        w = weights[particle]

        ratio = y_calc / y_exp if y_exp > 0 else 0
        error_pct = abs(ratio - 1.0) * 100 if y_exp > 0 else 100
        errors.append(error_pct)

        status = "✓" if error_pct < 50 else "✗"
        print(f"{particle:<12} {w:+6.1f} {y_calc:12.3e} {y_exp:12.3e} {ratio:8.2f} {error_pct:7.1f}% {status}")

    avg_error = np.mean(errors)
    print("-" * 70)
    print(f"Average error: {avg_error:.1f}%")

    # Check if pattern matches leptons (lightest OK, heavier 3-4× too large)
    if len(particles) >= 2:
        lightest_err = errors[0]
        heavy_err_avg = np.mean(errors[1:])
        print(f"Lightest error: {lightest_err:.1f}%")
        print(f"Heavier avg error: {heavy_err_avg:.1f}%")

        if lightest_err < 50 and heavy_err_avg > 150:
            print("⚠️  SAME PATTERN as leptons: lightest OK, heavier ~3-4× too large")
            return "PATTERN_MATCHES"
        elif avg_error < 100:
            print("✓ All reasonably close - different from lepton pattern")
            return "BETTER"
        else:
            print("✗ Different pattern - may need sector-specific quantum numbers")
            return "DIFFERENT"

    return "INCONCLUSIVE"


# =============================================================================
# TEST 1: CHARGED LEPTONS (known result)
# =============================================================================

lepton_assignments = {
    'electron': (1, 0),  # w = -2
    'muon': (0, 0),      # w = 0
    'tau': (0, 1)        # w = +1
}

result_leptons = test_quantum_number_assignment(
    "Charged Leptons (Known)",
    lepton_assignments,
    leptons_exp
)


# =============================================================================
# TEST 2: UP-TYPE QUARKS
# =============================================================================

# Try same pattern as leptons
up_quark_assignments_v1 = {
    'up': (1, 0),        # w = -2 (like electron)
    'charm': (0, 0),     # w = 0 (like muon)
    'top': (0, 1)        # w = +1 (like tau)
}

result_up_v1 = test_quantum_number_assignment(
    "Up Quarks (v1: same as leptons)",
    up_quark_assignments_v1,
    quarks_up_exp
)


# =============================================================================
# TEST 3: DOWN-TYPE QUARKS
# =============================================================================

down_quark_assignments_v1 = {
    'down': (1, 0),      # w = -2
    'strange': (0, 0),   # w = 0
    'bottom': (0, 1)     # w = +1
}

result_down_v1 = test_quantum_number_assignment(
    "Down Quarks (v1: same as leptons)",
    down_quark_assignments_v1,
    quarks_down_exp
)


# =============================================================================
# TEST 4: NEUTRINOS (if we have mass ordering)
# =============================================================================

neutrino_assignments_v1 = {
    'nu1': (1, 0),       # w = -2 (lightest)
    'nu2': (0, 0),       # w = 0
    'nu3': (0, 1)        # w = +1 (heaviest, normal ordering)
}

result_nu_v1 = test_quantum_number_assignment(
    "Neutrinos (v1: normal ordering, same pattern)",
    neutrino_assignments_v1,
    neutrinos_exp
)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("CROSS-SECTOR SUMMARY")
print("="*70)
print(f"Leptons:      {result_leptons}")
print(f"Up quarks:    {result_up_v1}")
print(f"Down quarks:  {result_down_v1}")
print(f"Neutrinos:    {result_nu_v1}")
print()

# Count how many sectors show the same pattern
pattern_matches = sum([
    result_leptons == "PATTERN_MATCHES",
    result_up_v1 == "PATTERN_MATCHES",
    result_down_v1 == "PATTERN_MATCHES",
    result_nu_v1 == "PATTERN_MATCHES"
])

print(f"Sectors with SAME PATTERN (lightest OK, heavier 3-4× too large): {pattern_matches}/4")
print()

if pattern_matches >= 3:
    print("🔍 CONCLUSION: Systematic issue across ALL sectors")
    print("   → The formula w=-2q₃+q₄ gives RIGHT HIERARCHY everywhere")
    print("   → But NORMALIZATION is universally off by O(1) factor")
    print("   → Likely missing: RG running, threshold corrections, or Kähler metric")
    print()
    print("   NEXT STEPS:")
    print("   1. Accept formula as qualitatively correct")
    print("   2. Study systematic normalization factor")
    print("   3. Investigate RG evolution from string scale")
elif pattern_matches == 1:
    print("⚠️  CONCLUSION: Issue is LEPTON-SPECIFIC")
    print("   → Other sectors may work better")
    print("   → Need sector-dependent quantum number assignments")
    print()
    print("   NEXT STEPS:")
    print("   1. Optimize quantum numbers for each sector separately")
    print("   2. Look for selection rules or constraints")
else:
    print("❓ CONCLUSION: MIXED results")
    print("   → Need more careful analysis of each sector")
    print("   → May need different τ values or localization effects")
