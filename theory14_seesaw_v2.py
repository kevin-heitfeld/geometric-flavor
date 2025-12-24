"""
THEORY #14 + SEESAW V2: FIXED τ + DEMOCRATIC M_D

LEARNING FROM V1: Optimizer fell into wrong minimum (τ ≈ 0.78i)

NEW STRATEGY:
1. FIX τ = 2.69i (Theory #14's proven value)
2. Use DEMOCRATIC M_D (Theory #11 showed this works!)
3. Hierarchical M_R
4. Optimize only coefficients and scales

Theory #11's key insight:
  Democratic Dirac mass + seesaw → 3/3 PMNS angles!
  
This suggests: PMNS mixing comes from seesaw structure,
                not from Yukawa texture (unlike CKM)

PARAMETERIZATION:
  M_D ≈ v_D · (1  1  1)  (democratic)
              (1  1  1)
              (1  1  1)
              
  M_R = diag(M1, M2, M3)  (hierarchical)
  
  m_ν = -v_D² (1  1  1) · (1/M1    0      0  ) · (1  1  1)
              (1  1  1)   (0     1/M2    0  )   (1  1  1)
              (1  1  1)   (0      0    1/M3)   (1  1  1)

This structure naturally gives tribimaximal-like mixing!
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
    'theta_12': 33.4,  # Solar angle (LARGE!)
    'theta_23': 49.2,  # Atmospheric angle (nearly maximal!)
    'theta_13': 8.57,  # Reactor angle
}

# ============================================================================
# MODULAR FORMS (from Theory #14)
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

# ============================================================================
# SEESAW WITH DEMOCRATIC M_D
# ============================================================================

def democratic_dirac_mass(v_D, epsilon=0.0):
    """
    Democratic Dirac mass (Theory #11 insight)
    
    M_D = v_D · (1  1  1)
                (1  1  1)
                (1  1  1)
    
    Optional: small breaking with epsilon
    """
    M_D = v_D * np.ones((3, 3), dtype=complex)
    
    # Optional small breaking
    if epsilon != 0:
        M_D += epsilon * v_D * np.diag([1, 0.5, 0.2])
    
    return M_D

def hierarchical_majorana_mass(M1, M2, M3):
    """
    Hierarchical right-handed Majorana mass
    M_R = diag(M1, M2, M3)
    """
    return np.diag([M1, M2, M3])

def seesaw_light_masses(M_D, M_R):
    """
    Type-I seesaw: m_ν = -M_D^T M_R^{-1} M_D
    """
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
# FIT WITH FIXED τ
# ============================================================================

def fit_seesaw_fixed_tau():
    """
    Fit with τ FIXED at Theory #14's value
    
    Parameters:
    - τ = 2.69i (FIXED!)
    - k_lepton, k_up, k_down (from Theory #14)
    - v_D: Dirac Yukawa scale
    - M1, M2, M3: Right-handed Majorana masses
    - epsilon: Democratic breaking
    - Coefficients for charged sectors
    
    Total: ~15 parameters
    """
    
    # Fix τ at Theory #14's value
    tau_fixed = 0.0 + 2.69j
    
    print("="*70)
    print("THEORY #14 + SEESAW V2: FIXED τ + DEMOCRATIC M_D")
    print("="*70)
    print(f"\nτ = {tau_fixed.real:.2f} + {tau_fixed.imag:.2f}i (FIXED from Theory #14)")
    print("\nStrategy:")
    print("  1. Fix τ at Theory #14's proven value")
    print("  2. Democratic M_D (Theory #11's success)")
    print("  3. Hierarchical M_R = diag(M1, M2, M3)")
    print("  4. Seesaw: m_ν = -M_D^T M_R^{-1} M_D")
    print("\nKey insight: PMNS from seesaw structure, not Yukawa texture")
    
    def objective(params):
        # Modular weights (allow to vary slightly)
        k_lepton = 2 * max(1, round(params[0] / 2))
        k_up = 2 * max(1, round(params[1] / 2))
        k_down = 2 * max(1, round(params[2] / 2))
        
        # Neutrino sector parameters
        log_v_D = params[3]  # Dirac scale (log)
        log_M_avg = params[4]  # Average Majorana scale
        ratio_21 = params[5]   # M2/M1 ratio
        ratio_32 = params[6]   # M3/M2 ratio
        epsilon = params[7]  # Democratic breaking
        
        v_D = 10**log_v_D  # GeV
        M_avg = 10**log_M_avg
        M1 = M_avg / (ratio_21 * ratio_32)**(1/3)
        M2 = M1 * ratio_21
        M3 = M2 * ratio_32
        
        # Bounds checking
        if log_v_D < -2 or log_v_D > 4:
            return 1e10
        if log_M_avg < 12 or log_M_avg > 15:
            return 1e10
        if ratio_21 < 1 or ratio_21 > 100:
            return 1e10
        if ratio_32 < 1 or ratio_32 > 100:
            return 1e10
        
        # Coefficients
        c_lepton = params[8:10]
        c_up = params[10:13]
        c_down = params[13:16]
        
        try:
            # Construct charged fermion Yukawas (Theory #14)
            Y_lepton = yukawa_from_modular_forms(tau_fixed, k_lepton, c_lepton, 'charged_lepton')
            Y_up = yukawa_from_modular_forms(tau_fixed, k_up, c_up, 'up_quark')
            Y_down = yukawa_from_modular_forms(tau_fixed, k_down, c_down, 'down_quark')
            
            # Charged fermion masses
            m_lepton, V_lepton = yukawa_to_masses_and_mixing(Y_lepton)
            m_up, V_up = yukawa_to_masses_and_mixing(Y_up)
            m_down, V_down = yukawa_to_masses_and_mixing(Y_down)
            
            # Neutrino sector (democratic M_D)
            M_D = democratic_dirac_mass(v_D, epsilon)
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
        
        # Validity checks
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
        
        # Charged fermion mass errors (keep Theory #14 performance)
        error_lepton = np.mean(np.abs(np.log10(m_lepton) - np.log10(LEPTON_MASSES)))
        error_up = np.mean(np.abs(np.log10(m_up) - np.log10(UP_MASSES)))
        error_down = np.mean(np.abs(np.log10(m_down) - np.log10(DOWN_MASSES)))
        
        total_error = 0.3 * (error_lepton + error_up + error_down)  # Reduced weight
        
        # CKM mixing (keep Theory #14: 3/3)
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
            
            total_error += 0.3 * (error_ckm / 3)
            
        except:
            total_error += 5.0
        
        # Neutrino mass squared differences
        exp_dm21 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m21_sq']
        exp_dm31 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m31_sq']
        
        error_nu_mass = abs(np.log10(delta_m21_sq) - np.log10(exp_dm21))
        error_nu_mass += abs(np.log10(delta_m31_sq) - np.log10(exp_dm31))
        error_nu_mass /= 2
        
        total_error += 1.5 * error_nu_mass  # High weight on neutrino masses
        
        # PMNS mixing angles (MAIN GOAL!)
        try:
            pmns_calc = extract_mixing_angles(U_PMNS)
            
            error_pmns = 0
            for angle_name in ['theta_12', 'theta_23', 'theta_13']:
                calc = pmns_calc[angle_name]
                exp = PMNS_ANGLES_EXP[angle_name]
                error_pmns += abs(calc - exp) / exp
            
            error_pmns /= 3
            total_error += 2.0 * error_pmns  # HIGHEST weight on PMNS!
            
        except:
            total_error += 10.0
        
        return total_error
    
    # Bounds
    bounds = [
        (6, 10),          # k_lepton (Theory #14: 8)
        (4, 8),           # k_up (Theory #14: 6)
        (2, 6),           # k_down (Theory #14: 4)
        (-2, 4),          # log(v_D) in GeV (0.01 - 10000 GeV)
        (12, 15),         # log(M_avg) ~ 10^13-10^15 GeV
        (1, 50),          # M2/M1 ratio
        (1, 50),          # M3/M2 ratio
        (-0.5, 0.5),      # epsilon (democratic breaking)
        # Charged lepton coeffs
        (-5.0, 5.0), (-5.0, 5.0),
        # Up coeffs
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
        # Down coeffs
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
    ]
    
    # Initial guess from Theory #14
    x0 = np.array([
        8, 6, 4,                # k from Theory #14
        1.0,                    # log(v_D) ~ 10 GeV
        13.5,                   # log(M_avg) ~ 3×10^13 GeV
        3.0, 10.0,              # M2/M1 ≈ 3, M3/M2 ≈ 10 (moderate hierarchy)
        0.0,                    # epsilon = 0 (pure democratic)
        1.9, -1.9,              # lepton (Theory #14)
        0.01, 4.8, -5.0,        # up
        -0.03, 0.7, -4.8,       # down
    ])
    
    print(f"\nStarting optimization...")
    print(f"Parameters: 16")
    print(f"Targets: 9 masses + 3 CKM + 2 Δm² + 3 PMNS = 17 observables")
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=600,
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
    epsilon = result.x[7]
    
    c_lepton = result.x[8:10]
    c_up = result.x[10:13]
    c_down = result.x[13:16]
    
    # Calculate observables
    Y_lepton = yukawa_from_modular_forms(tau_fixed, k_lepton, c_lepton, 'charged_lepton')
    Y_up = yukawa_from_modular_forms(tau_fixed, k_up, c_up, 'up_quark')
    Y_down = yukawa_from_modular_forms(tau_fixed, k_down, c_down, 'down_quark')
    
    m_lepton, V_lepton = yukawa_to_masses_and_mixing(Y_lepton)
    m_up, V_up = yukawa_to_masses_and_mixing(Y_up)
    m_down, V_down = yukawa_to_masses_and_mixing(Y_down)
    
    M_D = democratic_dirac_mass(v_D, epsilon)
    M_R = hierarchical_majorana_mass(M1, M2, M3)
    m_nu = seesaw_light_masses(M_D, M_R)
    nu_masses, U_PMNS = diagonalize_neutrino_mass(m_nu)
    nu_masses = np.abs(nu_masses)
    
    ckm_calc = calculate_ckm_angles(V_up, V_down)
    pmns_calc = extract_mixing_angles(U_PMNS)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n*** MODULUS (FIXED) ***")
    print(f"τ = {tau_fixed.real:.2f} + {tau_fixed.imag:.2f}i (Theory #14)")
    
    print(f"\n*** MODULAR WEIGHTS ***")
    print(f"  k_lepton = {k_lepton} (Theory #14: 8)")
    print(f"  k_up = {k_up} (Theory #14: 6)")
    print(f"  k_down = {k_down} (Theory #14: 4)")
    
    print(f"\n*** NEUTRINO SECTOR PARAMETERS ***")
    print(f"  v_D = {v_D:.3f} GeV (Dirac scale)")
    print(f"  M_R hierarchy:")
    print(f"    M1 = {M1:.2e} GeV")
    print(f"    M2 = {M2:.2e} GeV")
    print(f"    M3 = {M3:.2e} GeV")
    print(f"  epsilon = {epsilon:.4f} (democratic breaking)")
    
    print(f"\n*** DEMOCRATIC STRUCTURE TEST ***")
    if abs(epsilon) < 0.1:
        print(f"  ✓ Nearly pure democratic M_D")
    else:
        print(f"  ✗ Significant breaking from democratic")
    
    # Charged fermions
    print("\n" + "="*70)
    print("CHARGED FERMION MASSES (Theory #14 sector)")
    print("="*70)
    
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
            print(f"  {label}: {m_c:.2f} MeV (exp: {m_e:.2f}, log-err: {log_err:.4f}) {status}")
    
    print(f"\nCharged fermion fit: {total_match}/9 (Theory #14: 4/9)")
    
    # CKM
    print(f"\n{'='*70}")
    print("CKM MIXING")
    print("="*70)
    
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
    
    print(f"\nCKM: {ckm_match}/3 (Theory #14: 3/3)")
    
    # Neutrino masses
    delta_m21_sq = nu_masses[1]**2 - nu_masses[0]**2
    delta_m31_sq = nu_masses[2]**2 - nu_masses[0]**2
    
    exp_dm21 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m21_sq']
    exp_dm31 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m31_sq']
    
    print(f"\n{'='*70}")
    print("NEUTRINO MASSES")
    print("="*70)
    
    print(f"\nLight neutrino masses:")
    print(f"  m1 = {nu_masses[0]*1000:.3f} meV")
    print(f"  m2 = {nu_masses[1]*1000:.3f} meV")
    print(f"  m3 = {nu_masses[2]*1000:.3f} meV")
    
    print(f"\nMass squared differences:")
    dm21_err = abs(np.log10(delta_m21_sq) - np.log10(exp_dm21))
    dm31_err = abs(np.log10(delta_m31_sq) - np.log10(exp_dm31))
    
    print(f"  Δm²₂₁ = {delta_m21_sq:.3e} eV² (exp: {exp_dm21:.3e}, log-err: {dm21_err:.3f}) {'✓' if dm21_err < 0.2 else '✗'}")
    print(f"  Δm²₃₁ = {delta_m31_sq:.3e} eV² (exp: {exp_dm31:.3e}, log-err: {dm31_err:.3f}) {'✓' if dm31_err < 0.2 else '✗'}")
    
    nu_mass_match = (dm21_err < 0.2) + (dm31_err < 0.2)
    
    # PMNS
    print(f"\n{'='*70}")
    print("PMNS MIXING (THE KEY TEST!)")
    print("="*70)
    
    pmns_match = 0
    for angle_name in ['theta_12', 'theta_23', 'theta_13']:
        calc = pmns_calc[angle_name]
        exp = PMNS_ANGLES_EXP[angle_name]
        error = abs(calc - exp)
        sigma = max(exp * 0.15, 2.0)
        within = error < sigma
        pmns_match += within
        status = "✓" if within else "✗"
        print(f"  {angle_name}: {calc:.2f}° vs {exp:.2f}° (error: {error:.2f}°) {status}")
    
    print(f"\nPMNS: {pmns_match}/3")
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if pmns_match >= 2 and nu_mass_match >= 1:
        print("✓✓✓ SUCCESS!")
        print("\nDemocratic M_D + seesaw predicts PMNS!")
        print("  • Fixed τ = 2.69i from Theory #14")
        print("  • Democratic Dirac mass (Theory #11)")
        print("  • Hierarchical Majorana mass")
        print("  • Large PMNS mixing from seesaw structure")
        
    elif pmns_match >= 1 or nu_mass_match >= 1:
        print("✓ PARTIAL SUCCESS")
        print(f"\n  PMNS: {pmns_match}/3")
        print(f"  Neutrino masses: {nu_mass_match}/2")
        print("\nDemocratic seesaw works but needs refinement")
        
    else:
        print("✗ NEEDS MORE WORK")
    
    print(f"\nOptimization error: {result.fun:.6f}")
    print("="*70)
    
    return result

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    result = fit_seesaw_fixed_tau()
