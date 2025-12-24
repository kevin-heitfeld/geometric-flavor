"""
THEORY #14 + SEESAW: GEOMETRIC NEUTRINO MASSES AND PMNS MIXING

Building on Theory #14's success (4/9 masses + 3/3 CKM from τ ≈ 2.7i)

KEY INSIGHT FROM THEORY #11:
Democratic Dirac neutrino mass matrix M_D predicted PMNS perfectly (3/3)!
This suggests: Neutrino mixing comes from seesaw structure, not Yukawa texture

SEESAW MECHANISM:
  m_ν = -M_D^T M_R^{-1} M_D
  
where:
  • M_D: Dirac neutrino mass (ν_L - N_R coupling)
  • M_R: Right-handed Majorana mass (heavy, ~10^14 GeV)
  • m_ν: Light neutrino masses (~0.1 eV)

STRATEGY:
Use Theory #14's modular framework for BOTH sectors:

1. M_D from modular forms (like charged leptons)
   • Democratic-like structure (Theory #11 insight!)
   • Or modular forms at τ ≈ 2.7i
   
2. M_R from modular forms at different weight
   • Hierarchical structure
   • Controls light neutrino mass scale

PREDICTION:
If Theory #14 gives CKM from geometry,
Then seesaw + modular forms should give PMNS from geometry!

Parameters:
  • Same τ ≈ 2.7i (universal!)
  • k_D: Dirac weight
  • k_R: Majorana weight
  • Coefficients for M_D, M_R

Experimental targets (normal ordering):
  • Δm²_21 = 7.5 × 10^-5 eV²  (solar)
  • Δm²_31 = 2.5 × 10^-3 eV²  (atmospheric)
  • θ_12 = 33.4°  (solar angle, large!)
  • θ_23 = 49.2°  (atmospheric, nearly maximal!)
  • θ_13 = 8.57°  (reactor angle)
  • δ_CP = ?     (unknown CP phase)

Note: PMNS angles are LARGE (unlike CKM which is small)
This suggests different structure than quark sector
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Experimental data (from Theory #14)
LEPTON_MASSES = np.array([0.511, 105.7, 1776.9])  # MeV
UP_MASSES = np.array([2.16, 1270, 172760])  # MeV
DOWN_MASSES = np.array([4.67, 93.4, 4180])  # MeV

CKM_ANGLES_EXP = {
    'theta_12': 13.04,
    'theta_23': 2.38,
    'theta_13': 0.201,
}

# Neutrino oscillation data (normal ordering)
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
    """
    Theory #14 Yukawa construction
    """
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
        Y += c3 * np.ones((3, 3), dtype=complex)  # Democratic
    
    elif sector == 'dirac_neutrino':
        # For Dirac neutrinos, try democratic-like structure (Theory #11 success!)
        c1, c2 = coeffs[:2]
        # Democratic dominant
        Y = c1 * np.ones((3, 3), dtype=complex)
        # Small modular correction
        Y += c2 * Y_singlet * np.eye(3, dtype=complex)
        
    elif sector == 'majorana':
        # Right-handed Majorana mass - hierarchical
        c1, c2 = coeffs[:2]
        Y = c1 * Y_singlet * np.eye(3, dtype=complex)
        Y += c2 * np.outer(Y_triplet, Y_triplet.conj())
    
    return Y

# ============================================================================
# SEESAW MECHANISM
# ============================================================================

def seesaw_light_masses(M_D, M_R):
    """
    Type-I seesaw: m_ν = -M_D^T M_R^{-1} M_D
    
    Returns light neutrino mass matrix
    """
    try:
        M_R_inv = np.linalg.inv(M_R)
    except np.linalg.LinAlgError:
        return None
    
    m_nu = -M_D.T @ M_R_inv @ M_D
    
    return m_nu

def diagonalize_neutrino_mass(m_nu):
    """
    Diagonalize Majorana neutrino mass matrix
    Returns masses and PMNS matrix
    """
    # Make hermitian
    m_nu_herm = (m_nu + m_nu.T.conj()) / 2
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(m_nu_herm)
    except np.linalg.LinAlgError:
        return None, None
    
    # Sort by absolute value
    idx = np.argsort(np.abs(eigenvalues))
    masses = eigenvalues[idx]
    U_PMNS = eigenvectors[:, idx]
    
    return masses, U_PMNS

def extract_mixing_angles(V):
    """Extract mixing angles from unitary matrix"""
    # Standard parameterization
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

# ============================================================================
# THEORY #14 + SEESAW FIT
# ============================================================================

def fit_theory14_with_seesaw():
    """
    Fit Theory #14 framework + seesaw neutrinos
    
    Strategy:
    1. Keep Theory #14 parameters for charged fermions
    2. Add M_D (Dirac neutrino mass)
    3. Add M_R (Majorana mass)
    4. Predict light neutrino masses + PMNS from seesaw
    
    Parameters:
    - τ: universal (from Theory #14: ~2.7i)
    - k_lepton, k_up, k_down (from Theory #14)
    - k_D: Dirac neutrino weight
    - k_R: Majorana weight
    - v_R: Right-handed scale (~10^14 GeV)
    - Coefficients for all sectors
    
    Total: ~20 parameters for 12 masses + 6 mixing angles
    """
    
    print("="*70)
    print("THEORY #14 + SEESAW: GEOMETRIC NEUTRINO MASSES")
    print("="*70)
    print("\nBuilding on Theory #14's success:")
    print("  • 4/9 charged fermion masses ✓")
    print("  • 3/3 CKM angles ✓")
    print("  • τ ≈ 2.7i (geometric attractor)")
    print("\nAdding:")
    print("  • Type-I seesaw mechanism")
    print("  • M_D from modular forms (democratic-like)")
    print("  • M_R from modular forms (hierarchical)")
    print("\nGoal:")
    print("  • Predict 3 neutrino masses")
    print("  • Predict 3 PMNS angles (large, unlike CKM!)")
    print("  • All from same universal τ")
    
    def objective(params):
        # Universal τ (start from Theory #14)
        tau = params[0] + 1j * params[1]
        
        if params[1] <= 0.05:
            return 1e10
        
        # Modular weights (Theory #14 + new for neutrinos)
        k_lepton = 2 * max(1, round(params[2] / 2))
        k_up = 2 * max(1, round(params[3] / 2))
        k_down = 2 * max(1, round(params[4] / 2))
        k_D = 2 * max(1, round(params[5] / 2))  # Dirac
        k_R = 2 * max(1, round(params[6] / 2))  # Majorana
        
        # Right-handed scale (in GeV)
        log_v_R = params[7]  # Log scale to explore wide range
        v_R = 10**log_v_R  # Typical: 10^10 - 10^15 GeV
        
        if log_v_R < 10 or log_v_R > 16:
            return 1e10
        
        # Coefficients
        c_lepton = params[8:10]
        c_up = params[10:13]
        c_down = params[13:16]
        c_D = params[16:18]  # Dirac neutrino
        c_R = params[18:20]  # Majorana
        
        try:
            # Construct charged fermion Yukawas (Theory #14)
            Y_lepton = yukawa_from_modular_forms(tau, k_lepton, c_lepton, 'charged_lepton')
            Y_up = yukawa_from_modular_forms(tau, k_up, c_up, 'up_quark')
            Y_down = yukawa_from_modular_forms(tau, k_down, c_down, 'down_quark')
            
            # Construct neutrino sector
            Y_D = yukawa_from_modular_forms(tau, k_D, c_D, 'dirac_neutrino')
            Y_R = yukawa_from_modular_forms(tau, k_R, c_R, 'majorana')
            
            # Charged fermion masses and mixing
            m_lepton, V_lepton = yukawa_to_masses_and_mixing(Y_lepton)
            m_up, V_up = yukawa_to_masses_and_mixing(Y_up)
            m_down, V_down = yukawa_to_masses_and_mixing(Y_down)
            
            # Dirac and Majorana masses
            v_EW = 246.0  # GeV
            M_D = Y_D * v_EW / np.sqrt(2)
            M_R = Y_R * v_R  # Heavy scale!
            
            # Seesaw mechanism
            m_nu = seesaw_light_masses(M_D, M_R)
            if m_nu is None:
                return 1e10
            
            # Diagonalize neutrino mass
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
        
        # Take absolute values of neutrino masses
        nu_masses = np.abs(nu_masses)
        
        # Calculate mass squared differences
        delta_m21_sq = nu_masses[1]**2 - nu_masses[0]**2
        delta_m31_sq = nu_masses[2]**2 - nu_masses[0]**2
        
        if delta_m21_sq <= 0 or delta_m31_sq <= 0:
            return 1e10
        
        # Charged fermion mass errors (Theory #14 targets)
        error_lepton = np.mean(np.abs(np.log10(m_lepton) - np.log10(LEPTON_MASSES)))
        error_up = np.mean(np.abs(np.log10(m_up) - np.log10(UP_MASSES)))
        error_down = np.mean(np.abs(np.log10(m_down) - np.log10(DOWN_MASSES)))
        
        total_error = 0.5 * (error_lepton + error_up + error_down)  # Weight less (already good)
        
        # CKM mixing (Theory #14 target: 3/3)
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
        
        # Neutrino mass squared differences (NEW!)
        exp_dm21 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m21_sq']
        exp_dm31 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m31_sq']
        
        error_nu_mass = abs(np.log10(delta_m21_sq) - np.log10(exp_dm21))
        error_nu_mass += abs(np.log10(delta_m31_sq) - np.log10(exp_dm31))
        error_nu_mass /= 2
        
        total_error += 1.0 * error_nu_mass  # Weight neutrino masses
        
        # PMNS mixing angles (NEW! - the key test)
        try:
            pmns_calc = extract_mixing_angles(U_PMNS)
            
            error_pmns = 0
            for angle_name in ['theta_12', 'theta_23', 'theta_13']:
                calc = pmns_calc[angle_name]
                exp = PMNS_ANGLES_EXP[angle_name]
                error_pmns += abs(calc - exp) / exp
            
            error_pmns /= 3
            total_error += 1.5 * error_pmns  # Weight PMNS heavily (main goal!)
            
        except:
            total_error += 10.0
        
        return total_error
    
    # Bounds
    bounds = [
        (-1.0, 1.0),      # Re(τ)
        (0.5, 3.5),       # Im(τ)
        (2, 12),          # k_lepton
        (2, 12),          # k_up
        (2, 12),          # k_down
        (2, 12),          # k_D (Dirac)
        (2, 12),          # k_R (Majorana)
        (10, 16),         # log(v_R) in GeV
        # Charged lepton coeffs
        (-5.0, 5.0), (-5.0, 5.0),
        # Up coeffs
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
        # Down coeffs
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
        # Dirac neutrino coeffs
        (-5.0, 5.0), (-5.0, 5.0),
        # Majorana coeffs
        (-5.0, 5.0), (-5.0, 5.0),
    ]
    
    # Initial guess from Theory #14
    x0 = np.array([
        0.0, 2.69,           # τ from Theory #14
        8, 6, 4,             # k from Theory #14
        4, 6,                # k_D, k_R (guesses)
        14.0,                # log(v_R) ~ 10^14 GeV
        1.9, -1.9,           # lepton (from Theory #14)
        0.01, 4.8, -5.0,     # up
        -0.03, 0.7, -4.8,    # down
        3.0, 0.0,            # Dirac (democratic dominant)
        1.0, 1.0,            # Majorana
    ])
    
    print(f"\nStarting from Theory #14 solution:")
    print(f"  τ ≈ 2.69i")
    print(f"  k = (8, 6, 4)")
    print(f"  + k_D, k_R for neutrinos")
    print(f"  + v_R ~ 10^14 GeV (seesaw scale)")
    print(f"\nRunning optimization (10-15 minutes)...")
    print(f"Optimizing: 20 parameters")
    print(f"Targets: 9 charged + 2 neutrino masses + 3 CKM + 3 PMNS")
    
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
    
    # Extract and display results
    tau_opt = result.x[0] + 1j * result.x[1]
    k_lepton = 2 * max(1, round(result.x[2] / 2))
    k_up = 2 * max(1, round(result.x[3] / 2))
    k_down = 2 * max(1, round(result.x[4] / 2))
    k_D = 2 * max(1, round(result.x[5] / 2))
    k_R = 2 * max(1, round(result.x[6] / 2))
    v_R = 10**result.x[7]
    
    c_lepton = result.x[8:10]
    c_up = result.x[10:13]
    c_down = result.x[13:16]
    c_D = result.x[16:18]
    c_R = result.x[18:20]
    
    # Calculate final observables
    Y_lepton = yukawa_from_modular_forms(tau_opt, k_lepton, c_lepton, 'charged_lepton')
    Y_up = yukawa_from_modular_forms(tau_opt, k_up, c_up, 'up_quark')
    Y_down = yukawa_from_modular_forms(tau_opt, k_down, c_down, 'down_quark')
    Y_D = yukawa_from_modular_forms(tau_opt, k_D, c_D, 'dirac_neutrino')
    Y_R = yukawa_from_modular_forms(tau_opt, k_R, c_R, 'majorana')
    
    m_lepton, V_lepton = yukawa_to_masses_and_mixing(Y_lepton)
    m_up, V_up = yukawa_to_masses_and_mixing(Y_up)
    m_down, V_down = yukawa_to_masses_and_mixing(Y_down)
    
    M_D = Y_D * 246.0 / np.sqrt(2)
    M_R = Y_R * v_R
    m_nu = seesaw_light_masses(M_D, M_R)
    nu_masses, U_PMNS = diagonalize_neutrino_mass(m_nu)
    nu_masses = np.abs(nu_masses)
    
    ckm_calc = calculate_ckm_angles(V_up, V_down)
    pmns_calc = extract_mixing_angles(U_PMNS)
    
    # Print results
    print("\n" + "="*70)
    print("THEORY #14 + SEESAW RESULTS")
    print("="*70)
    
    print(f"\n*** UNIVERSAL MODULUS ***")
    print(f"τ = {tau_opt.real:.6f} + {tau_opt.imag:.6f}i")
    print(f"  Compare Theory #14: 0.000 + 2.687i")
    
    print(f"\n*** MODULAR WEIGHTS ***")
    print(f"  Charged leptons: k_e = {k_lepton}")
    print(f"  Up quarks:       k_u = {k_up}")
    print(f"  Down quarks:     k_d = {k_down}")
    print(f"  Dirac neutrino:  k_D = {k_D}")
    print(f"  Majorana:        k_R = {k_R}")
    
    print(f"\n*** SEESAW SCALE ***")
    print(f"  v_R = {v_R:.2e} GeV")
    print(f"  Typical GUT scale: ~10^14-10^15 GeV")
    
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
    
    print(f"\nCharged fermion fit: {total_match}/9")
    
    # CKM
    print(f"\n{'='*70}")
    print("CKM MIXING ANGLES")
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
    print(f"\n{'='*70}")
    print("NEUTRINO MASSES (from seesaw)")
    print("="*70)
    
    delta_m21_sq = nu_masses[1]**2 - nu_masses[0]**2
    delta_m31_sq = nu_masses[2]**2 - nu_masses[0]**2
    
    exp_dm21 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m21_sq']
    exp_dm31 = NEUTRINO_MASS_SQUARED_DIFFS['delta_m31_sq']
    
    print(f"\nLight neutrino masses:")
    print(f"  m1 = {nu_masses[0]*1000:.4f} meV")
    print(f"  m2 = {nu_masses[1]*1000:.4f} meV")
    print(f"  m3 = {nu_masses[2]*1000:.4f} meV")
    
    print(f"\nMass squared differences:")
    dm21_err = abs(np.log10(delta_m21_sq) - np.log10(exp_dm21))
    dm31_err = abs(np.log10(delta_m31_sq) - np.log10(exp_dm31))
    
    print(f"  Δm²₂₁ = {delta_m21_sq:.3e} eV² (exp: {exp_dm21:.3e}, log-err: {dm21_err:.3f}) {'✓' if dm21_err < 0.2 else '✗'}")
    print(f"  Δm²₃₁ = {delta_m31_sq:.3e} eV² (exp: {exp_dm31:.3e}, log-err: {dm31_err:.3f}) {'✓' if dm31_err < 0.2 else '✗'}")
    
    nu_mass_match = (dm21_err < 0.2) + (dm31_err < 0.2)
    
    # PMNS angles
    print(f"\n{'='*70}")
    print("PMNS MIXING ANGLES (from seesaw + modular forms)")
    print("="*70)
    
    pmns_match = 0
    for angle_name in ['theta_12', 'theta_23', 'theta_13']:
        calc = pmns_calc[angle_name]
        exp = PMNS_ANGLES_EXP[angle_name]
        error = abs(calc - exp)
        sigma = max(exp * 0.15, 2.0)  # At least 2° tolerance
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
        print("✓✓✓ SEESAW SUCCESS!")
        print("\nGeometric neutrino prediction works:")
        print("  • PMNS angles from modular forms + seesaw")
        print("  • Neutrino mass scale from v_R")
        print("  • Same universal τ as quark/lepton sector")
        print("\nThis extends Theory #14:")
        print("  • CKM from modular geometry ✓")
        print("  • PMNS from modular geometry + seesaw ✓")
        print("  • All mixing from symmetry!")
        
    elif pmns_match >= 1:
        print("✓ PARTIAL SUCCESS")
        print(f"\n  PMNS: {pmns_match}/3")
        print(f"  Neutrino masses: {nu_mass_match}/2")
        print("\nSeesaw framework works but needs refinement")
        
    else:
        print("✗ NEEDS WORK")
        print("\nSeesaw mechanism may need:")
        print("  • Different M_D structure")
        print("  • Complex phases for CP")
        print("  • Or separate neutrino sector τ")
    
    print(f"\nOptimization error: {result.fun:.6f}")
    print(f"Total parameters: 20")
    print("="*70)
    
    return {
        'tau': tau_opt,
        'weights': {'lepton': k_lepton, 'up': k_up, 'down': k_down, 'D': k_D, 'R': k_R},
        'v_R': v_R,
        'charged_masses': {'lepton': m_lepton, 'up': m_up, 'down': m_down},
        'neutrino_masses': nu_masses,
        'ckm': ckm_calc,
        'pmns': pmns_calc,
        'match_charged': total_match,
        'match_ckm': ckm_match,
        'match_pmns': pmns_match,
        'match_nu_mass': nu_mass_match,
        'error': result.fun,
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("THEORY #14 + SEESAW")
    print("="*70)
    print("\nTesting: Can modular symmetry explain neutrino sector?")
    print("\nStrategy:")
    print("  1. Keep Theory #14 framework (τ ≈ 2.7i)")
    print("  2. Add Dirac mass M_D (modular forms, democratic-like)")
    print("  3. Add Majorana mass M_R (modular forms, hierarchical)")
    print("  4. Seesaw: m_ν = -M_D^T M_R^{-1} M_D")
    print("\nKey test: Do we get PMNS mixing from geometry?")
    print("  (Like CKM came from geometry in Theory #14)")
    
    result = fit_theory14_with_seesaw()
    
    print("\n" + "="*70)
    print("COMPARISON: CKM vs PMNS")
    print("="*70)
    print("\nQuark sector (CKM):")
    print("  θ₁₂ = 13° (small)")
    print("  θ₂₃ = 2.4° (tiny)")
    print("  θ₁₃ = 0.2° (tiny)")
    print("  → Hierarchical, small mixing")
    print("\nLepton sector (PMNS):")
    print("  θ₁₂ = 33° (large)")
    print("  θ₂₃ = 49° (nearly maximal!)")
    print("  θ₁₃ = 8.6° (moderate)")
    print("  → Large mixing, near tribimaximal")
    print("\nThis difference suggests:")
    print("  • Different structure in neutrino sector")
    print("  • Seesaw mechanism crucial")
    print("  • Democratic M_D (Theory #11 insight)")
    print("="*70)
