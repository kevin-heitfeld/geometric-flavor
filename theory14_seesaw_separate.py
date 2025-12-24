"""
THEORY #14 + SEESAW + CP: SEPARATE OPTIMIZATION

STRATEGY: Lock charged sector at Theory #14's solution,
         optimize neutrinos independently

MOTIVATION: Test if sectors can work simultaneously when decoupled
           Avoid optimization conflict that sacrificed charged sector

IMPLEMENTATION:
1. Fix charged sector parameters from Theory #14:
   - τ = 2.69i (universal modulus)
   - k = (8, 6, 4) (modular weights)
   - c_lepton, c_up, c_down (Yukawa coefficients)

2. Optimize neutrino sector only:
   - v_D, M_R hierarchy, φ₁,φ₂,φ₃, ε
   - 8 parameters for 6 observables (overdetermined!)

EXPECTED: Both sectors working → Complete phenomenology
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
    'theta_12': 33.4,
    'theta_23': 49.2,
    'theta_13': 8.57,
}

DELTA_CP_EXP = 230.0  # degrees

# ============================================================================
# THEORY #14 PARAMETERS (FIXED!)
# ============================================================================

TAU_FIXED = 0.0 + 2.69j

# From Theory #14 optimization
K_LEPTON_FIXED = 8
K_UP_FIXED = 6
K_DOWN_FIXED = 4

C_LEPTON_FIXED = np.array([1.9, -1.9])
C_UP_FIXED = np.array([0.01, 4.8, -5.0])
C_DOWN_FIXED = np.array([-0.03, 0.7, -4.8])

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
    """A₄ triplet modular forms"""
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
    """Extract Dirac CP phase δ_CP from PMNS matrix"""
    det = np.linalg.det(U)
    if abs(det) > 1e-10:
        U = U / det**(1/3)

    s13 = np.clip(abs(U[0, 2]), 0, 1)

    if s13 > 1e-6:
        phase = np.angle(U[0, 2])
        delta_cp = -phase
        delta_cp = delta_cp % (2 * np.pi)
        return np.degrees(delta_cp)
    else:
        return 0.0

# ============================================================================
# NEUTRINO SECTOR
# ============================================================================

def complex_democratic_dirac_mass(v_D, phi1, phi2, phi3, epsilon=0.0):
    """Complex democratic Dirac mass with CP phases"""
    M_D = v_D * np.array([
        [1.0,                  np.exp(1j * phi1), np.exp(1j * phi2)],
        [np.exp(1j * phi1),    1.0,               np.exp(1j * phi3)],
        [np.exp(1j * phi2),    np.exp(1j * phi3), 1.0              ],
    ], dtype=complex)

    if epsilon != 0:
        M_D += epsilon * v_D * np.diag([1.0, 0.5, 0.2])

    return M_D

def hierarchical_majorana_mass(M1, M2, M3):
    """Hierarchical right-handed Majorana mass"""
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
    """Diagonalize Majorana neutrino mass matrix"""
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
# SEPARATE OPTIMIZATION
# ============================================================================

def fit_separate_optimization():
    """
    Lock charged sector at Theory #14, optimize neutrinos only

    FIXED (from Theory #14):
    - τ = 2.69i
    - k = (8, 6, 4)
    - Charged sector coefficients

    OPTIMIZED (neutrino sector only):
    - v_D: Dirac scale
    - M_R: Majorana hierarchy (3 params)
    - φ₁, φ₂, φ₃: CP phases
    - ε: Democratic breaking

    Total: 8 free parameters
    Targets: 2 Δm² + 3 PMNS + δ_CP = 6 observables
    Overdetermined!
    """

    print("="*70)
    print("THEORY #14 + SEESAW: SEPARATE OPTIMIZATION")
    print("="*70)
    print("\nSTRATEGY: Lock charged sector, optimize neutrinos independently")
    print("\n*** FIXED FROM THEORY #14 ***")
    print(f"  τ = {TAU_FIXED.real:.2f} + {TAU_FIXED.imag:.2f}i")
    print(f"  k = ({K_LEPTON_FIXED}, {K_UP_FIXED}, {K_DOWN_FIXED})")
    print(f"  Charged sector coefficients locked")
    print("\n*** OPTIMIZING (NEUTRINOS ONLY) ***")
    print("  v_D, M_R hierarchy (3), φ₁, φ₂, φ₃, ε")
    print("  8 parameters → 6 observables (overdetermined!)")

    def objective(params):
        # Neutrino sector parameters ONLY
        log_v_D = params[0]
        log_M_avg = params[1]
        ratio_21 = params[2]
        ratio_32 = params[3]
        phi1 = params[4]
        phi2 = params[5]
        phi3 = params[6]
        epsilon = params[7]

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

        try:
            # Charged fermions (FIXED - just calculate for comparison)
            Y_lepton = yukawa_from_modular_forms(TAU_FIXED, K_LEPTON_FIXED, C_LEPTON_FIXED, 'charged_lepton')
            Y_up = yukawa_from_modular_forms(TAU_FIXED, K_UP_FIXED, C_UP_FIXED, 'up_quark')
            Y_down = yukawa_from_modular_forms(TAU_FIXED, K_DOWN_FIXED, C_DOWN_FIXED, 'down_quark')

            m_lepton, V_lepton = yukawa_to_masses_and_mixing(Y_lepton)
            m_up, V_up = yukawa_to_masses_and_mixing(Y_up)
            m_down, V_down = yukawa_to_masses_and_mixing(Y_down)

            # Neutrino sector (OPTIMIZED)
            M_D = complex_democratic_dirac_mass(v_D, phi1, phi2, phi3, epsilon)
            M_R = hierarchical_majorana_mass(M1, M2, M3)

            m_nu = seesaw_light_masses(M_D, M_R)
            if m_nu is None:
                return 1e10

            nu_masses, U_PMNS = diagonalize_neutrino_mass(m_nu)
            if nu_masses is None or U_PMNS is None:
                return 1e10

        except:
            return 1e10

        # Validity
        if (np.any(~np.isfinite(nu_masses)) or
            np.any(m_lepton <= 0) or np.any(m_up <= 0) or np.any(m_down <= 0)):
            return 1e10

        nu_masses = np.abs(nu_masses)

        # Mass squared differences
        delta_m21_sq = nu_masses[1]**2 - nu_masses[0]**2
        delta_m31_sq = nu_masses[2]**2 - nu_masses[0]**2

        if delta_m21_sq <= 0 or delta_m31_sq <= 0:
            return 1e10

        # ERROR FUNCTION: NEUTRINOS ONLY (charged sector fixed!)
        total_error = 0.0

        # Neutrino masses (HIGH WEIGHT)
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
            total_error += 2.0 * error_pmns

        except:
            total_error += 10.0

        # CP phase (MODERATE WEIGHT)
        try:
            delta_cp_calc = extract_cp_phase(U_PMNS)

            error_cp = min(abs(delta_cp_calc - DELTA_CP_EXP),
                          360 - abs(delta_cp_calc - DELTA_CP_EXP))

            total_error += 1.0 * (error_cp / 180.0)

        except:
            total_error += 1.0

        return total_error

    # Bounds (neutrino sector only)
    bounds = [
        (-1, 5),          # log(v_D)
        (10, 16),         # log(M_avg)
        (1, 100),         # M2/M1
        (1, 100),         # M3/M2
        (0, 2*np.pi),     # φ₁
        (0, 2*np.pi),     # φ₂
        (0, 2*np.pi),     # φ₃
        (-0.5, 0.5),      # ε
    ]

    # Initial guess (from successful CP run)
    x0 = np.array([
        np.log10(15000),      # v_D ~ 15 TeV
        np.log10(1e10),       # M_avg ~ 10^10 GeV
        5.0, 40.0,            # Moderate hierarchy
        1.317, 1.093, 3.271,  # Phases from CP run
        -0.49,                # ε
    ])

    print(f"\nOptimizing: 8 parameters (neutrinos only)")
    print(f"Targets: 2 Δm² + 3 PMNS + δ_CP = 6 observables")
    print(f"\nExpected: Charged sector preserved, neutrinos optimized")
    print(f"\nRunning (~5-10 minutes)...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=500,
        seed=42,
        workers=1,
        strategy='best1bin',
        atol=1e-8,
        tol=1e-8,
        x0=x0,
    )

    # Extract results
    v_D = 10**result.x[0]
    M_avg = 10**result.x[1]
    ratio_21 = result.x[2]
    ratio_32 = result.x[3]
    M1 = M_avg / (ratio_21 * ratio_32)**(1/3)
    M2 = M1 * ratio_21
    M3 = M2 * ratio_32

    phi1 = result.x[4]
    phi2 = result.x[5]
    phi3 = result.x[6]
    epsilon = result.x[7]

    # Calculate observables
    Y_lepton = yukawa_from_modular_forms(TAU_FIXED, K_LEPTON_FIXED, C_LEPTON_FIXED, 'charged_lepton')
    Y_up = yukawa_from_modular_forms(TAU_FIXED, K_UP_FIXED, C_UP_FIXED, 'up_quark')
    Y_down = yukawa_from_modular_forms(TAU_FIXED, K_DOWN_FIXED, C_DOWN_FIXED, 'down_quark')

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
    print("RESULTS: SEPARATE OPTIMIZATION")
    print("="*70)

    print(f"\n*** CHARGED SECTOR (THEORY #14 - FIXED) ***")

    # Charged fermion masses
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

    print(f"\nCharged: {total_match}/9 (Theory #14: 4/9)")

    # CKM
    print(f"\n*** CKM MIXING (THEORY #14 - FIXED) ***")
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

    print(f"CKM: {ckm_match}/3 (Theory #14: 3/3)")

    # Neutrino sector
    print(f"\n*** NEUTRINO SECTOR (OPTIMIZED) ***")

    print(f"\nCP Phases:")
    print(f"  φ₁ = {np.degrees(phi1):.1f}°")
    print(f"  φ₂ = {np.degrees(phi2):.1f}°")
    print(f"  φ₃ = {np.degrees(phi3):.1f}°")

    print(f"\nSeesaw parameters:")
    print(f"  v_D = {v_D:.2f} GeV")
    print(f"  M_R: {M1:.2e}, {M2:.2e}, {M3:.2e} GeV")
    print(f"  ε = {epsilon:.4f}")

    # Neutrino masses
    delta_m21_sq = nu_masses[1]**2 - nu_masses[0]**2
    delta_m31_sq = nu_masses[2]**2 - nu_masses[0]**2

    exp_dm21 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m21_sq']
    exp_dm31 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m31_sq']

    print(f"\nNeutrino masses:")
    print(f"  m1 = {nu_masses[0]*1000:.3f} meV")
    print(f"  m2 = {nu_masses[1]*1000:.3f} meV")
    print(f"  m3 = {nu_masses[2]*1000:.3f} meV")

    dm21_err = abs(np.log10(delta_m21_sq) - np.log10(exp_dm21))
    dm31_err = abs(np.log10(delta_m31_sq) - np.log10(exp_dm31))

    print(f"\n  Δm²₂₁ = {delta_m21_sq:.3e} eV² (exp: {exp_dm21:.3e}) {'✓' if dm21_err < 0.2 else '✗'}")
    print(f"  Δm²₃₁ = {delta_m31_sq:.3e} eV² (exp: {exp_dm31:.3e}) {'✓' if dm31_err < 0.2 else '✗'}")

    nu_mass_match = (dm21_err < 0.2) + (dm31_err < 0.2)

    # PMNS
    print(f"\nPMNS mixing:")
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

    # CP phase
    delta_cp_err = min(abs(delta_cp_calc - DELTA_CP_EXP),
                       360 - abs(delta_cp_calc - DELTA_CP_EXP))
    status_cp = "✓" if delta_cp_err < 30 else "✗"
    print(f"\nCP violation:")
    print(f"  δ_CP = {delta_cp_calc:.1f}° (exp: {DELTA_CP_EXP:.1f}°) {status_cp}")

    print(f"\nPMNS: {pmns_match}/3")
    print(f"Neutrino masses: {nu_mass_match}/2")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    if total_match >= 4 and ckm_match >= 2 and pmns_match >= 2 and nu_mass_match >= 1:
        print("✓✓✓ BOTH SECTORS WORKING!")
        print(f"\nCharged fermions: {total_match}/9 + {ckm_match}/3 CKM")
        print(f"Neutrinos: {pmns_match}/3 PMNS + {nu_mass_match}/2 masses")
        if delta_cp_err < 30:
            print(f"δ_CP: ✓")
        print("\nSeparate optimization successful!")
        print("→ Complete phenomenology from geometry")
        print("→ RG evolution would provide theoretical unification")

    elif pmns_match >= 2 and nu_mass_match >= 1:
        print("✓ NEUTRINOS MAINTAINED")
        print(f"\nNeutrinos: {pmns_match}/3 + {nu_mass_match}/2")
        print(f"Charged: {total_match}/9 + {ckm_match}/3 CKM")
        print("\nPartial success - neutrino optimization worked")
        print("Theory #14 structure maintained")

    else:
        print("✗ NEEDS MORE WORK")
        print("\nSeparate optimization had issues")

    print(f"\nOptimization error: {result.fun:.6f}")
    print("="*70)

    return result

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST: Can sectors work simultaneously when decoupled?")
    print("="*70)
    print("\nHypothesis: Optimization conflict caused trade-off")
    print("Solution: Lock charged sector, optimize neutrinos only")
    print("\nIf successful:")
    print("  → Complete phenomenology (both sectors working)")
    print("  → RG evolution for theoretical unification")
    print("\nIf fails:")
    print("  → Must do RG evolution (no alternative)")

    result = fit_separate_optimization()
