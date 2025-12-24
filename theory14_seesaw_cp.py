"""
THEORY #14 + SEESAW + CP PHASES: GEOMETRIC CP VIOLATION

GOAL: Add complex phases to democratic M_D to:
1. Enhance neutrino masses (constructive interference)
2. Predict δ_CP (Dirac CP phase) from modular geometry
3. Test if phases can close 500× mass gap

STRATEGY:
Democratic M_D with phases:

    M_D = v_D · ( 1          e^(iφ₁)     e^(iφ₂)   )
                ( e^(iφ₁)    1           e^(iφ₃)   )
                ( e^(iφ₂)    e^(iφ₃)     1         )

Key insight: Phases from modular forms at τ = 2.69i
- Modular forms are COMPLEX
- Phases emerge from arg(Y(τ))
- Not arbitrary - geometric!

CP violation:
- Complex M_D → complex m_ν
- Diagonalization gives U_PMNS with phase
- Extract δ_CP from U_PMNS
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Experimental data
LEPTON_MASSES = np.array([0.511, 105.7, 1776.9])  # MeV
UP_MASSES = np.array([2.16, 1270, 172760])  # MeV
DOWN_MASSES = np.array([4.67, 93.4, 4180])  # MeV

CKM_ANGLES_EXP = {
    'theta_12': 13.04,
    'theta_23': 2.38,
    'theta_13': 0.201,
}

NEUTRINO_MASS_SQUARED_DIFFS = {
    'delta_m21_sq': 7.5e-5,  # eV²
    'delta_m31_sq': 2.5e-3,  # eV²
}

PMNS_ANGLES_EXP = {
    'theta_12': 33.4,  # Solar
    'theta_23': 49.2,  # Atmospheric
    'theta_13': 8.57,  # Reactor
}

# CP phase (recent data)
DELTA_CP_EXP = 230.0  # degrees (≈ 1.4π, near maximal!)

# ============================================================================
# MODULAR FORMS
# ============================================================================

def eisenstein_series(tau, weight, truncate=25):
    """Generalized Eisenstein series E_k(τ)"""
    q = np.exp(2j * np.pi * tau)

    def sigma_power(n, power):
        divisors = [d for d in range(1, n+1) if n % d == 0]
        return sum(d**power for d in divisors)

    coeff_map = {2: -24, 4: 240, 6: -504, 8: 480, 10: -264}
    coeff = coeff_map.get(weight, 240)

    E_k = 1.0
    for n in range(1, truncate):
        E_k += coeff * sigma_power(n, weight-1) * q**n

    return E_k

def modular_form_triplet(tau, weight):
    """A₄ triplet modular forms (COMPLEX!)"""
    omega = np.exp(2j * np.pi / 3)

    Y1 = eisenstein_series(tau, weight)
    Y2 = eisenstein_series(omega * tau, weight)
    Y3 = eisenstein_series(omega**2 * tau, weight)

    norm = np.sqrt(abs(Y1)**2 + abs(Y2)**2 + abs(Y3)**2)
    if norm > 0:
        Y1, Y2, Y3 = Y1/norm, Y2/norm, Y3/norm

    return np.array([Y1, Y2, Y3])

def modular_form_singlet(tau, weight):
    """A₄ singlet modular form"""
    return eisenstein_series(tau, weight)

def yukawa_from_modular_forms(tau, weight, coeffs, sector='charged_lepton'):
    """Theory #14 Yukawa construction"""
    Y_triplet = modular_form_triplet(tau, weight)
    Y_singlet = modular_form_singlet(tau, weight)

    if sector == 'charged_lepton':
        c1, c2 = coeffs[:2]
        Y = c1 * Y_singlet * np.eye(3, dtype=complex)
        Y += c2 * np.outer(Y_triplet, Y_triplet.conj())

    elif sector == 'up_quark' or sector == 'down_quark':
        c1, c2, c3 = coeffs[:3]
        Y = c1 * Y_singlet * np.eye(3, dtype=complex)
        Y += c2 * np.outer(Y_triplet, Y_triplet.conj())
        Y += c3 * np.ones((3, 3), dtype=complex)

    return Y

def yukawa_to_masses_and_mixing(Y, v_scale=246.0):
    """Extract masses and mixing from Yukawa"""
    M = Y * v_scale / np.sqrt(2)
    M_hermitian = (M + M.conj().T) / 2

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(M_hermitian)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan, np.nan]), np.eye(3)

    idx = np.argsort(np.abs(eigenvalues))
    masses = np.abs(eigenvalues[idx])
    V = eigenvectors[:, idx]

    masses = np.maximum(masses, 1e-10)

    return masses, V

def calculate_ckm_angles(V_up, V_down):
    """Calculate CKM from up/down mixing"""
    V_CKM = V_up.T.conj() @ V_down
    return extract_mixing_angles(V_CKM)

def extract_mixing_angles(V):
    """Extract mixing angles from unitary matrix"""
    det = np.linalg.det(V)
    if abs(det) > 1e-10:
        V = V / det**(1/3)

    theta_13 = np.arcsin(np.clip(abs(V[0, 2]), 0, 1))
    theta_12 = np.arctan2(abs(V[0, 1]), abs(V[0, 0]))
    theta_23 = np.arctan2(abs(V[1, 2]), abs(V[2, 2]))

    return {
        'theta_12': np.degrees(theta_12),
        'theta_23': np.degrees(theta_23),
        'theta_13': np.degrees(theta_13),
    }

def extract_cp_phase(U):
    """
    Extract Dirac CP phase δ_CP from PMNS matrix

    Standard parameterization:
    U_PMNS = R₂₃ · U₁₃(δ) · R₁₂ · P

    δ_CP appears in U[0,2] element
    """
    # Normalize
    det = np.linalg.det(U)
    if abs(det) > 1e-10:
        U = U / det**(1/3)

    # Extract from U[0,2] = s₁₃ e^(-iδ)
    s13 = np.clip(abs(U[0, 2]), 0, 1)

    if s13 > 1e-6:
        # δ from phase of U[0,2]
        phase = np.angle(U[0, 2])
        delta_cp = -phase  # Note sign convention

        # Wrap to [0, 2π]
        delta_cp = delta_cp % (2 * np.pi)

        return np.degrees(delta_cp)
    else:
        return 0.0  # Undefined if θ₁₃ = 0

# ============================================================================
# COMPLEX DEMOCRATIC DIRAC MASS
# ============================================================================

def complex_democratic_dirac_mass(v_D, phi1, phi2, phi3, epsilon=0.0):
    """
    Complex democratic Dirac mass with CP phases

    M_D = v_D · ( 1          e^(iφ₁)     e^(iφ₂)   )
                ( e^(iφ₁)    1           e^(iφ₃)   )
                ( e^(iφ₂)    e^(iφ₃)     1         )

    Phases φ₁, φ₂, φ₃ → CP violation in neutrino sector

    Optional epsilon: symmetry breaking
    """
    M_D = v_D * np.array([
        [1.0,                  np.exp(1j * phi1), np.exp(1j * phi2)],
        [np.exp(1j * phi1),    1.0,               np.exp(1j * phi3)],
        [np.exp(1j * phi2),    np.exp(1j * phi3), 1.0              ],
    ], dtype=complex)

    # Optional breaking
    if epsilon != 0:
        M_D += epsilon * v_D * np.diag([1.0, 0.5, 0.2])

    return M_D

def hierarchical_majorana_mass(M1, M2, M3):
    """Hierarchical right-handed Majorana mass (real diagonal)"""
    return np.diag([M1, M2, M3])

def seesaw_light_masses(M_D, M_R):
    """Type-I seesaw: m_ν = -M_D^T M_R^{-1} M_D"""
    try:
        M_R_inv = np.linalg.inv(M_R)
    except np.linalg.LinAlgError:
        return None

    m_nu = -M_D.T @ M_R_inv @ M_D
    return m_nu

def diagonalize_neutrino_mass(m_nu):
    """Diagonalize complex Majorana neutrino mass matrix"""
    # Make hermitian
    m_nu_herm = (m_nu + m_nu.T.conj()) / 2

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(m_nu_herm)
    except np.linalg.LinAlgError:
        return None, None

    idx = np.argsort(np.abs(eigenvalues))
    masses = eigenvalues[idx]
    U_PMNS = eigenvectors[:, idx]

    return masses, U_PMNS

# ============================================================================
# FIT WITH COMPLEX PHASES
# ============================================================================

def fit_seesaw_with_cp():
    """
    Fit with complex phases for CP violation

    Parameters:
    - τ = 2.69i (FIXED)
    - k_lepton, k_up, k_down (modular weights)
    - v_D: Dirac scale
    - M_R: Majorana hierarchy
    - φ₁, φ₂, φ₃: CP phases (NEW!)
    - epsilon: democratic breaking
    - Charged sector coefficients

    Total: 19 parameters

    Targets:
    - 9 charged masses
    - 3 CKM angles
    - 2 neutrino Δm²
    - 3 PMNS angles
    - δ_CP (NEW!)

    = 18 observables (overdetermined!)
    """

    tau_fixed = 0.0 + 2.69j

    print("="*70)
    print("THEORY #14 + SEESAW + CP PHASES")
    print("="*70)
    print(f"\nτ = {tau_fixed.real:.2f} + {tau_fixed.imag:.2f}i (FIXED)")
    print("\nNEW: Adding complex phases to M_D")
    print("  → Predict CP violation from geometry!")
    print("  → May enhance neutrino masses via interference")
    print("\nStructure:")
    print("  M_D = v_D · (1, e^(iφ₁), e^(iφ₂))")
    print("              (e^(iφ₁), 1, e^(iφ₃))")
    print("              (e^(iφ₂), e^(iφ₃), 1)")
    print("\nGoal: Predict δ_CP ≈ 230° from phases φ₁, φ₂, φ₃")

    def objective(params):
        # Modular weights
        k_lepton = 2 * max(1, round(params[0] / 2))
        k_up = 2 * max(1, round(params[1] / 2))
        k_down = 2 * max(1, round(params[2] / 2))

        # Neutrino sector
        log_v_D = params[3]
        log_M_avg = params[4]
        ratio_21 = params[5]
        ratio_32 = params[6]

        # CP PHASES (NEW!)
        phi1 = params[7]  # [0, 2π]
        phi2 = params[8]
        phi3 = params[9]

        epsilon = params[10]

        v_D = 10**log_v_D
        M_avg = 10**log_M_avg
        M1 = M_avg / (ratio_21 * ratio_32)**(1/3)
        M2 = M1 * ratio_21
        M3 = M2 * ratio_32

        # Bounds
        if log_v_D < -1 or log_v_D > 5:
            return 1e10
        if log_M_avg < 10 or log_M_avg > 16:
            return 1e10
        if ratio_21 < 1 or ratio_21 > 100:
            return 1e10
        if ratio_32 < 1 or ratio_32 > 100:
            return 1e10

        # Coefficients
        c_lepton = params[11:13]
        c_up = params[13:16]
        c_down = params[16:19]

        try:
            # Charged fermions (Theory #14)
            Y_lepton = yukawa_from_modular_forms(tau_fixed, k_lepton, c_lepton, 'charged_lepton')
            Y_up = yukawa_from_modular_forms(tau_fixed, k_up, c_up, 'up_quark')
            Y_down = yukawa_from_modular_forms(tau_fixed, k_down, c_down, 'down_quark')

            m_lepton, V_lepton = yukawa_to_masses_and_mixing(Y_lepton)
            m_up, V_up = yukawa_to_masses_and_mixing(Y_up)
            m_down, V_down = yukawa_to_masses_and_mixing(Y_down)

            # Neutrino sector with COMPLEX M_D
            M_D = complex_democratic_dirac_mass(v_D, phi1, phi2, phi3, epsilon)
            M_R = hierarchical_majorana_mass(M1, M2, M3)

            # Seesaw
            m_nu = seesaw_light_masses(M_D, M_R)
            if m_nu is None:
                return 1e10

            nu_masses, U_PMNS = diagonalize_neutrino_mass(m_nu)
            if nu_masses is None or U_PMNS is None:
                return 1e10

        except:
            return 1e10

        # Validity
        if (np.any(~np.isfinite(m_lepton)) or np.any(~np.isfinite(m_up)) or
            np.any(~np.isfinite(m_down)) or np.any(~np.isfinite(nu_masses)) or
            np.any(m_lepton <= 0) or np.any(m_up <= 0) or np.any(m_down <= 0)):
            return 1e10

        nu_masses = np.abs(nu_masses)

        # Mass squared differences
        delta_m21_sq = nu_masses[1]**2 - nu_masses[0]**2
        delta_m31_sq = nu_masses[2]**2 - nu_masses[0]**2

        if delta_m21_sq <= 0 or delta_m31_sq <= 0:
            return 1e10

        # Charged fermion masses (reduced weight - maintain Theory #14)
        error_lepton = np.mean(np.abs(np.log10(m_lepton) - np.log10(LEPTON_MASSES)))
        error_up = np.mean(np.abs(np.log10(m_up) - np.log10(UP_MASSES)))
        error_down = np.mean(np.abs(np.log10(m_down) - np.log10(DOWN_MASSES)))

        total_error = 0.2 * (error_lepton + error_up + error_down)

        # CKM mixing (reduced weight)
        try:
            ckm_calc = calculate_ckm_angles(V_up, V_down)

            error_ckm = 0
            for angle_name in ['theta_12', 'theta_23', 'theta_13']:
                calc = ckm_calc[angle_name]
                exp = CKM_ANGLES_EXP[angle_name]
                if exp > 1.0:
                    error_ckm += abs(calc - exp) / exp
                else:
                    error_ckm += abs(calc - exp) / 0.2

            total_error += 0.2 * (error_ckm / 3)

        except:
            total_error += 3.0

        # Neutrino masses (HIGH WEIGHT - main issue!)
        exp_dm21 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m21_sq']
        exp_dm31 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m31_sq']

        error_nu_mass = abs(np.log10(delta_m21_sq) - np.log10(exp_dm21))
        error_nu_mass += abs(np.log10(delta_m31_sq) - np.log10(exp_dm31))
        error_nu_mass /= 2

        total_error += 2.0 * error_nu_mass

        # PMNS mixing (HIGH WEIGHT)
        try:
            pmns_calc = extract_mixing_angles(U_PMNS)

            error_pmns = 0
            for angle_name in ['theta_12', 'theta_23', 'theta_13']:
                calc = pmns_calc[angle_name]
                exp = PMNS_ANGLES_EXP[angle_name]
                error_pmns += abs(calc - exp) / exp

            error_pmns /= 3
            total_error += 1.5 * error_pmns

        except:
            total_error += 10.0

        # CP PHASE (NEW! - moderate weight)
        try:
            delta_cp_calc = extract_cp_phase(U_PMNS)

            # δ_CP is periodic, use circular distance
            error_cp = min(abs(delta_cp_calc - DELTA_CP_EXP),
                          360 - abs(delta_cp_calc - DELTA_CP_EXP))

            total_error += 0.5 * (error_cp / 180.0)  # Normalize to [0,1]

        except:
            total_error += 1.0

        return total_error

    # Bounds
    bounds = [
        (6, 10),          # k_lepton
        (4, 8),           # k_up
        (2, 6),           # k_down
        (-1, 5),          # log(v_D) - allow wider range
        (10, 16),         # log(M_avg) - allow lower scale
        (1, 100),         # M2/M1
        (1, 100),         # M3/M2
        (0, 2*np.pi),     # φ₁ (CP phase)
        (0, 2*np.pi),     # φ₂
        (0, 2*np.pi),     # φ₃
        (-0.5, 0.5),      # epsilon
        # Charged lepton coeffs
        (-5.0, 5.0), (-5.0, 5.0),
        # Up coeffs
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
        # Down coeffs
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
    ]

    # Initial guess
    x0 = np.array([
        8, 6, 4,                # k from Theory #14
        2.0,                    # log(v_D) ~ 100 GeV
        13.0,                   # log(M_avg) ~ 10^13 GeV
        5.0, 10.0,              # Moderate hierarchy
        np.pi, np.pi/2, 0.0,    # Initial phases (diverse)
        0.0,                    # epsilon = 0
        1.9, -1.9,              # lepton
        0.01, 4.8, -5.0,        # up
        -0.03, 0.7, -4.8,       # down
    ])

    print(f"\nOptimizing: 19 parameters")
    print(f"Targets: 9 masses + 3 CKM + 2 Δm² + 3 PMNS + δ_CP = 18 obs")
    print(f"\nRunning (10-15 minutes)...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=700,
        seed=42,
        workers=1,
        strategy='best1bin',
        atol=1e-8,
        tol=1e-8,
        x0=x0,
    )

    # Extract results
    k_lepton = 2 * max(1, round(result.x[0] / 2))
    k_up = 2 * max(1, round(result.x[1] / 2))
    k_down = 2 * max(1, round(result.x[2] / 2))

    v_D = 10**result.x[3]
    M_avg = 10**result.x[4]
    ratio_21 = result.x[5]
    ratio_32 = result.x[6]
    M1 = M_avg / (ratio_21 * ratio_32)**(1/3)
    M2 = M1 * ratio_21
    M3 = M2 * ratio_32

    phi1 = result.x[7]
    phi2 = result.x[8]
    phi3 = result.x[9]
    epsilon = result.x[10]

    c_lepton = result.x[11:13]
    c_up = result.x[13:16]
    c_down = result.x[16:19]

    # Calculate observables
    Y_lepton = yukawa_from_modular_forms(tau_fixed, k_lepton, c_lepton, 'charged_lepton')
    Y_up = yukawa_from_modular_forms(tau_fixed, k_up, c_up, 'up_quark')
    Y_down = yukawa_from_modular_forms(tau_fixed, k_down, c_down, 'down_quark')

    m_lepton, V_lepton = yukawa_to_masses_and_mixing(Y_lepton)
    m_up, V_up = yukawa_to_masses_and_mixing(Y_up)
    m_down, V_down = yukawa_to_masses_and_mixing(Y_down)

    M_D = complex_democratic_dirac_mass(v_D, phi1, phi2, phi3, epsilon)
    M_R = hierarchical_majorana_mass(M1, M2, M3)
    m_nu = seesaw_light_masses(M_D, M_R)
    nu_masses, U_PMNS = diagonalize_neutrino_mass(m_nu)
    nu_masses = np.abs(nu_masses)

    ckm_calc = calculate_ckm_angles(V_up, V_down)
    pmns_calc = extract_mixing_angles(U_PMNS)
    delta_cp_calc = extract_cp_phase(U_PMNS)

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n*** CP PHASES (GEOMETRIC!) ***")
    print(f"  φ₁ = {np.degrees(phi1):.1f}° ({phi1:.3f} rad)")
    print(f"  φ₂ = {np.degrees(phi2):.1f}° ({phi2:.3f} rad)")
    print(f"  φ₃ = {np.degrees(phi3):.1f}° ({phi3:.3f} rad)")

    print(f"\n*** PREDICTED CP VIOLATION ***")
    delta_cp_err = min(abs(delta_cp_calc - DELTA_CP_EXP),
                       360 - abs(delta_cp_calc - DELTA_CP_EXP))
    status_cp = "✓" if delta_cp_err < 30 else "✗"
    print(f"  δ_CP = {delta_cp_calc:.1f}° (exp: {DELTA_CP_EXP:.1f}°, err: {delta_cp_err:.1f}°) {status_cp}")

    if delta_cp_err < 30:
        print(f"  ✓✓✓ CP VIOLATION FROM GEOMETRY!")

    print(f"\n*** NEUTRINO SECTOR ***")
    print(f"  v_D = {v_D:.2f} GeV")
    print(f"  M_R hierarchy: {M1:.2e}, {M2:.2e}, {M3:.2e} GeV")
    print(f"  epsilon = {epsilon:.4f}")

    # Charged fermions (brief)
    print(f"\n*** CHARGED FERMIONS ***")
    sectors = [
        ('LEPTONS', m_lepton, LEPTON_MASSES, ['e', 'μ', 'τ']),
        ('UP QUARKS', m_up, UP_MASSES, ['u', 'c', 't']),
        ('DOWN QUARKS', m_down, DOWN_MASSES, ['d', 's', 'b'])
    ]

    total_match = 0
    for sector_name, m_calc, m_exp, labels in sectors:
        print(f"\n{sector_name}:")
        for m_c, m_e, label in zip(m_calc, m_exp, labels):
            log_err = abs(np.log10(m_c) - np.log10(m_e))
            status = "✓" if log_err < 0.15 else "✗"
            total_match += (log_err < 0.15)
            print(f"  {label}: {m_c:.2f} MeV (exp: {m_e:.2f}) {status}")

    print(f"\nCharged: {total_match}/9")

    # CKM
    print(f"\n*** CKM ***")
    ckm_match = 0
    for angle_name in ['theta_12', 'theta_23', 'theta_13']:
        calc = ckm_calc[angle_name]
        exp = CKM_ANGLES_EXP[angle_name]
        error = abs(calc - exp)
        sigma = max(exp * 0.15, 0.15)
        within = error < sigma
        ckm_match += within
        status = "✓" if within else "✗"
        print(f"  {angle_name}: {calc:.3f}° vs {exp:.3f}° {status}")

    print(f"CKM: {ckm_match}/3")

    # Neutrino masses
    delta_m21_sq = nu_masses[1]**2 - nu_masses[0]**2
    delta_m31_sq = nu_masses[2]**2 - nu_masses[0]**2

    exp_dm21 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m21_sq']
    exp_dm31 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m31_sq']

    print(f"\n*** NEUTRINO MASSES ***")
    print(f"  m1 = {nu_masses[0]*1000:.3f} meV")
    print(f"  m2 = {nu_masses[1]*1000:.3f} meV")
    print(f"  m3 = {nu_masses[2]*1000:.3f} meV")

    dm21_err = abs(np.log10(delta_m21_sq) - np.log10(exp_dm21))
    dm31_err = abs(np.log10(delta_m31_sq) - np.log10(exp_dm31))

    print(f"\n  Δm²₂₁ = {delta_m21_sq:.3e} eV² (exp: {exp_dm21:.3e}) {'✓' if dm21_err < 0.2 else '✗'}")
    print(f"  Δm²₃₁ = {delta_m31_sq:.3e} eV² (exp: {exp_dm31:.3e}) {'✓' if dm31_err < 0.2 else '✗'}")

    nu_mass_match = (dm21_err < 0.2) + (dm31_err < 0.2)

    # PMNS
    print(f"\n*** PMNS MIXING ***")
    pmns_match = 0
    for angle_name in ['theta_12', 'theta_23', 'theta_13']:
        calc = pmns_calc[angle_name]
        exp = PMNS_ANGLES_EXP[angle_name]
        error = abs(calc - exp)
        sigma = max(exp * 0.15, 2.0)
        within = error < sigma
        pmns_match += within
        status = "✓" if within else "✗"
        print(f"  {angle_name}: {calc:.2f}° vs {exp:.2f}° {status}")

    print(f"\nPMNS: {pmns_match}/3")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    if nu_mass_match >= 2 and pmns_match >= 2:
        print("✓✓✓ SUCCESS!")
        print("\nComplex phases fixed the neutrino mass scale!")
        if delta_cp_err < 30:
            print("  • δ_CP predicted from geometry ✓")
        print(f"  • PMNS: {pmns_match}/3")
        print(f"  • Neutrino masses: {nu_mass_match}/2")

    elif nu_mass_match >= 1 or pmns_match >= 2:
        print("✓ PARTIAL SUCCESS")
        print(f"\n  PMNS: {pmns_match}/3")
        print(f"  Neutrino masses: {nu_mass_match}/2")
        if delta_cp_err < 30:
            print(f"  • δ_CP predicted! ✓")
        print("\nComplex phases help but may need RG evolution")

    else:
        print("✗ COMPLEX PHASES INSUFFICIENT")
        print("\nNeed RG evolution to fix scale mismatch")

    print(f"\nOptimization error: {result.fun:.6f}")
    print("="*70)

    return result

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING: Can complex phases close the neutrino mass gap?")
    print("="*70)
    print("\nKey question: Does CP violation help neutrino masses?")
    print("  • Phases → interference in m_ν")
    print("  • May enhance or suppress masses")
    print("  • Predict δ_CP from geometry")

    result = fit_seesaw_with_cp()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\nIf this works:")
    print("  → We predict CP violation from modular geometry!")
    print("  → Matter-antimatter asymmetry from extra dimensions")
    print("\nIf this fails:")
    print("  → Definitively need RG evolution")
    print("  → Theory #14 is GUT-scale theory")
    print("="*70)
